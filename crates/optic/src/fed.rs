use optic_core::model::Model;
use optic_core::net::Net;
use anyhow::{anyhow, bail, Context, Result};
use optic_core::st::{self, Tensor, Tensors};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Serialize, Deserialize)]
pub struct PatchMeta {
    pub base_sha256: String,
    pub new_sha256: String,
    pub tensors: Vec<String>,
    pub n_params: usize,
    pub federated: serde_json::Value,
}

#[derive(Serialize, Deserialize)]
pub struct UpdateMeta {
    pub head: String,
    pub base_sha256: String,
    pub n_samples: usize,
    pub label_counts: HashMap<String, usize>,
    pub class_names: Vec<String>,
    pub loss: f32,
    pub grad_norm: f32,
}

pub fn sha256_file(path: &Path) -> Result<String> { crate::sha256::file_hex(path) }

/// Class mask for a label: one class, or every leaf under a node of the cell-of-origin tree.
fn label_mask(label: &str, names: &[String], model: &Model, head: &str) -> Result<Vec<f32>> {
    let mut m = vec![0.0f32; names.len()];
    if let Some(i) = names.iter().position(|n| n == label) { m[i] = 1.0; return Ok(m); }
    {
        let tree = &model.lineage;
        if tree.nodes.contains_key(label) {
            for l in tree.leaves_under(label) {
                if let Some(i) = names.iter().position(|n| *n == l) { m[i] = 1.0; }
            }
            if m.iter().any(|&v| v > 0.0) { return Ok(m); }
        }
    }
    bail!("label {label:?} is not a class of head {head} (or a node of the cell-of-origin tree)")
}

/// One backward pass over `proj` (n x 256, the frozen `first.0` projection) for `head`.
/// Returns summed gradients for every parameter above `first.0` (keyed `grad.<name>`), and the mean loss.
pub fn contribute(net: &Net, head: &str, proj: &[f32], masks: &[Vec<f32>]) -> Result<(Tensors, f32)> {
    let n = masks.len();
    let (logits, trace) = net.forward_train(proj, n, head)?;
    let (loss_sum, dlogits) = optic_core::net::set_nll(&logits, masks);
    let grads = net.backward(&trace, &dlogits)?;
    Ok((grads.into_iter().map(|(k, g)| (format!("grad.{k}"), g)).collect(), loss_sum / n as f32))
}

/// L2 norm over every gradient tensor in an update.
pub fn grad_norm(grads: &Tensors) -> f32 {
    grads.values().map(|g| g.data.iter().map(|v| v * v).sum::<f32>()).sum::<f32>().sqrt()
}

pub fn label_mask_pub(label: &str, names: &[String], model: &Model, head: &str) -> Result<Vec<f32>> { label_mask(label, names, model, head) }

pub fn write_update(path: &Path, grads: &Tensors, meta: &UpdateMeta) -> Result<()> {
    st::save(grads, path).with_context(|| format!("writing {}", path.display()))?;
    std::fs::write(path.with_extension("json"), serde_json::to_string_pretty(meta)?)?;
    Ok(())
}

pub fn merge(base: &Model, updates: &[std::path::PathBuf], lr: f32, max_grad_norm: f32,
             out_dir: &Path, out_patch: Option<&Path>) -> Result<()> {
    let base_st = base.dir.join("optic.safetensors");
    let base_sha = sha256_file(&base_st)?;
    let mut weights = st::load(&base_st)?;
    let mut grad_sum = Tensors::new();
    let mut n_total = 0usize; let mut heads = std::collections::BTreeSet::new(); let mut counts: HashMap<String, usize> = HashMap::new();
    let mut norms: Vec<(String, f32)> = Vec::new();
    for u in updates {
        let meta: UpdateMeta = serde_json::from_str(&std::fs::read_to_string(u.with_extension("json"))?)
            .with_context(|| format!("reading {}", u.with_extension("json").display()))?;
        if meta.base_sha256 != base_sha { bail!("{} was computed against base {} but merging into {}", u.display(), &meta.base_sha256[..12], &base_sha[..12]); }
        let names = base.class_names(&meta.head).ok_or_else(|| anyhow!("base has no head {}", meta.head))?;
        if names != meta.class_names.as_slice() { bail!("{} class names differ from base head {}", u.display(), meta.head); }
        let g = st::load(u)?;

        // trust but verify what the federated data say
        let observed = grad_norm(&g);
        if !observed.is_finite() { bail!("{}: gradient norm is not finite ({observed})", u.display()); }
        if (observed - meta.grad_norm).abs() > 1e-3 * observed.max(1.0) {
            bail!("{}: declares |grad| {:.4} but the payload is {:.4}", u.display(), meta.grad_norm, observed);
        }
        if meta.n_samples == 0 { bail!("{}: reports zero samples", u.display()); }
        let per_sample = observed / meta.n_samples as f32;
        if max_grad_norm > 0.0 && per_sample > max_grad_norm {
            bail!("{}: per-sample |grad| {:.3} exceeds --max-grad-norm {:.3}", u.display(), per_sample, max_grad_norm);
        }
        norms.push((u.display().to_string(), per_sample));

        for (k, t) in g {
            match grad_sum.get_mut(&k) {
                Some(s) if s.shape == t.shape => s.data.iter_mut().zip(&t.data).for_each(|(a, b)| *a += b),
                Some(s) => bail!("{}: {k} is {:?} but an earlier update has {:?}", u.display(), t.shape, s.shape),
                None => { grad_sum.insert(k, t); }
            }
        }
        n_total += meta.n_samples; heads.insert(meta.head.clone());
        for (l, c) in meta.label_counts { *counts.entry(l).or_default() += c; }
    }
    if n_total == 0 { bail!("no samples in updates"); }

    // One site pulling far harder than the rest is a red flag
    let mut sorted: Vec<f32> = norms.iter().map(|(_, v)| *v).collect();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let median = sorted[sorted.len() / 2];
    for (name, v) in &norms {
        let flag = if median > 0.0 && *v > 10.0 * median { "   <- 10x the median, check it out" } else { "" };
        eprintln!("  per-sample |grad| {:>10.4}  {}{}", v, name, flag);
    }
    let mut patch = Tensors::new();
    let scale = (lr as f64 / n_total as f64) as f32;
    for (gk, g) in &grad_sum {
        let wk = gk.strip_prefix("grad.").ok_or_else(|| anyhow!("bad key {gk}"))?;
        let w = weights.get_mut(wk).ok_or_else(|| anyhow!("base lacks {wk}"))?;
        if w.shape != g.shape { bail!("{gk} is {:?} but the base {wk} is {:?}", g.shape, w.shape); }
        w.data.iter_mut().zip(&g.data).for_each(|(w, g)| *w -= g * scale);
        patch.insert(wk.to_string(), w.clone());
    }
    let n_changed = patch.len();
    std::fs::create_dir_all(out_dir)?;
    for f in std::fs::read_dir(&base.dir)? {
        let f = f?; let name = f.file_name();
        if f.file_type()?.is_file() && name != "optic.safetensors" { std::fs::copy(f.path(), out_dir.join(&name))?; }
    }
    st::save(&weights, &out_dir.join("optic.safetensors"))?;
    let meta_name = "lineage.json";
    let mut manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(base.dir.join(meta_name))?)?;
    manifest["federated"] = serde_json::json!({
        "base_sha256": base_sha, "merged_updates": updates.len(), "n_samples": n_total, "lr": lr,
        "heads_updated": heads, "label_counts": counts, "tensors_changed": n_changed,
        "per_sample_grad_norm": norms.iter().map(|(_, v)| *v).collect::<Vec<_>>(), "max_grad_norm": max_grad_norm,
        "new_sha256": sha256_file(&out_dir.join("optic.safetensors"))?,
    });
    std::fs::write(out_dir.join(meta_name), serde_json::to_string_pretty(&manifest)?)?;
    eprintln!("merged {} updates ({} samples) into {} head(s), {} tensors -> {}", updates.len(), n_total, heads.len(), n_changed, out_dir.display());

    // The round trip only has to carry what moved. `first.0` is 96.8% of the weights and federation
    // never touches it, so a patch is ~12 MB against a 373 MB model.
    if let Some(pp) = out_patch {
        let n_params: usize = patch.values().map(Tensor::len).sum();
        let meta = PatchMeta {
            base_sha256: base_sha.clone(),
            new_sha256: sha256_file(&out_dir.join("optic.safetensors"))?,
            tensors: { let mut v: Vec<String> = patch.keys().cloned().collect(); v.sort(); v },
            n_params,
            federated: manifest["federated"].clone(),
        };
        st::save(&patch, pp).with_context(|| format!("writing {}", pp.display()))?;
        std::fs::write(pp.with_extension("json"), serde_json::to_string_pretty(&meta)?)?;
        eprintln!("patch: {} tensors, {} params -> {} (+ .json)", meta.tensors.len(), n_params, pp.display());
    }
    Ok(())
}

/// Apply a hub patch to a local model
pub fn apply(base: &Model, patch: &Path, out_dir: &Path) -> Result<()> {
    let meta: PatchMeta = serde_json::from_str(&std::fs::read_to_string(patch.with_extension("json"))?)
        .with_context(|| format!("reading {}", patch.with_extension("json").display()))?;
    let base_st = base.dir.join("optic.safetensors");
    let have = sha256_file(&base_st)?;
    if have != meta.base_sha256 {
        bail!("this model is {} but the patch applies to {} - update to that release first",
              &have[..12], &meta.base_sha256[..12]);
    }
    let mut weights = st::load(&base_st)?;
    let new = st::load(patch)?;
    for (k, t) in &new {
        let old = weights.get(k).ok_or_else(|| anyhow!("patch carries {k}, which this model does not have"))?;
        if old.shape != t.shape { bail!("{k}: patch is {:?} but this model has {:?}", t.shape, old.shape); }
    }
    weights.extend(new);
    std::fs::create_dir_all(out_dir)?;
    for f in std::fs::read_dir(&base.dir)? {
        let f = f?; let name = f.file_name();
        if f.file_type()?.is_file() && name != "optic.safetensors" { std::fs::copy(f.path(), out_dir.join(&name))?; }
    }
    st::save(&weights, &out_dir.join("optic.safetensors"))?;
    let got = sha256_file(&out_dir.join("optic.safetensors"))?;
    if got != meta.new_sha256 {
        bail!("applied cleanly but the result is {} and the hub published {} - do not use this model",
              &got[..12], &meta.new_sha256[..12]);
    }
    let meta_name = if base.dir.join("manifest.json").exists() { "manifest.json" } else { "lineage.json" };
    let mut doc: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(base.dir.join(meta_name))?)?;
    doc["federated"] = meta.federated.clone();
    std::fs::write(out_dir.join(meta_name), serde_json::to_string_pretty(&doc)?)?;
    eprintln!("applied {} tensors ({} params) -> {}  [sha {} matches the hub]",
              meta.tensors.len(), meta.n_params, out_dir.display(), &got[..12]);
    Ok(())
}
