use optic_core::model::Model;
use optic_core::net::Net;
use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, Tensor, Var};
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

pub fn sha256_file(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(std::fs::read(path)?);
    Ok(format!("{:x}", h.finalize()))
}

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
/// Returns summed gradients for every parameter above `first.0`, and the mean loss.
pub fn contribute(net: &Net, head: &str, proj: &Tensor, masks: &[Vec<f32>], device: &Device)
    -> Result<(HashMap<String, Tensor>, f32)> {
    let params = net.federated_params(head).ok_or_else(|| anyhow!("no head {head}"))?;
    let vars: Vec<(String, Var)> = params.iter()
        .map(|(k, t)| Ok((k.clone(), Var::from_tensor(t)?))).collect::<Result<_>>()?;
    let tensors: Vec<(String, Tensor)> = vars.iter().map(|(k, v)| (k.clone(), v.as_tensor().clone())).collect();
    let logits = optic_core::net::forward_from_proj(proj, net.n_blocks(), head, &tensors)?;
    let n = masks.len(); let k = masks[0].len();
    let mask = Tensor::from_vec(masks.concat(), (n, k), device)?;
    let lse = logits.exp()?.sum_keepdim(1)?.log()?;
    let masked = (&logits + ((mask - 1.0)? * 1e4)?)?;
    let lse_m = masked.exp()?.sum_keepdim(1)?.log()?;
    let loss_sum = (lse - lse_m)?.sum_all()?;
    let grads = loss_sum.backward()?;
    let mut out = HashMap::new();
    for (name, var) in &vars {
        let g = grads.get(var).ok_or_else(|| anyhow!("no gradient for {name}"))?;
        out.insert(format!("grad.{name}"), g.detach());
    }
    Ok((out, loss_sum.to_scalar::<f32>()? / n as f32))
}

/// L2 norm over every gradient tensor in an update.
pub fn grad_norm(grads: &HashMap<String, Tensor>) -> Result<f32> {
    let mut acc = 0.0f32;
    for g in grads.values() { acc += g.sqr()?.sum_all()?.to_scalar::<f32>()?; }
    Ok(acc.sqrt())
}

pub fn label_mask_pub(label: &str, names: &[String], model: &Model, head: &str) -> Result<Vec<f32>> { label_mask(label, names, model, head) }

pub fn write_update(path: &Path, grads: &HashMap<String, Tensor>, meta: &UpdateMeta) -> Result<()> {
    candle_core::safetensors::save(grads, path).with_context(|| format!("writing {}", path.display()))?;
    std::fs::write(path.with_extension("json"), serde_json::to_string_pretty(meta)?)?;
    Ok(())
}

pub fn merge(base: &Model, updates: &[std::path::PathBuf], lr: f32, max_grad_norm: f32,
             out_dir: &Path, out_patch: Option<&Path>, device: &Device) -> Result<()> {
    let base_st = base.dir.join("optic.safetensors");
    let base_sha = sha256_file(&base_st)?;
    let mut weights = candle_core::safetensors::load(&base_st, device)?;
    let mut grad_sum: HashMap<String, Tensor> = HashMap::new();
    let mut n_total = 0usize; let mut heads = std::collections::BTreeSet::new(); let mut counts: HashMap<String, usize> = HashMap::new();
    let mut norms: Vec<(String, f32)> = Vec::new();
    for u in updates {
        let meta: UpdateMeta = serde_json::from_str(&std::fs::read_to_string(u.with_extension("json"))?)
            .with_context(|| format!("reading {}", u.with_extension("json").display()))?;
        if meta.base_sha256 != base_sha { bail!("{} was computed against base {} but merging into {}", u.display(), &meta.base_sha256[..12], &base_sha[..12]); }
        let names = base.class_names(&meta.head).ok_or_else(|| anyhow!("base has no head {}", meta.head))?;
        if names != meta.class_names.as_slice() { bail!("{} class names differ from base head {}", u.display(), meta.head); }
        let g: HashMap<String, Tensor> = candle_core::safetensors::load(u, device)?
            .into_iter().map(|(k, t)| Ok((k, t.to_dtype(DType::F32)?))).collect::<Result<_>>()?;

        // trust but verify what the federated data say
        let observed = grad_norm(&g)?;
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

        for (k, t) in g { grad_sum.entry(k).and_modify(|s| { *s = (&*s + &t).unwrap(); }).or_insert(t); }
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
    let mut patch: HashMap<String, Tensor> = HashMap::new();
    for (gk, g) in &grad_sum {
        let wk = gk.strip_prefix("grad.").ok_or_else(|| anyhow!("bad key {gk}"))?;
        let w = weights.get(wk).ok_or_else(|| anyhow!("base lacks {wk}"))?.to_dtype(DType::F32)?;
        let step = (g * (lr as f64 / n_total as f64))?;
        let updated = (w - step)?;
        weights.insert(wk.to_string(), updated.clone());
        patch.insert(wk.to_string(), updated);
    }
    let n_changed = patch.len();
    std::fs::create_dir_all(out_dir)?;
    for f in std::fs::read_dir(&base.dir)? {
        let f = f?; let name = f.file_name();
        if f.file_type()?.is_file() && name != "optic.safetensors" { std::fs::copy(f.path(), out_dir.join(&name))?; }
    }
    candle_core::safetensors::save(&weights, out_dir.join("optic.safetensors"))?;
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
        let n_params: usize = patch.values().map(|t| t.elem_count()).sum();
        let meta = PatchMeta {
            base_sha256: base_sha.clone(),
            new_sha256: sha256_file(&out_dir.join("optic.safetensors"))?,
            tensors: { let mut v: Vec<String> = patch.keys().cloned().collect(); v.sort(); v },
            n_params,
            federated: manifest["federated"].clone(),
        };
        candle_core::safetensors::save(&patch, pp).with_context(|| format!("writing {}", pp.display()))?;
        std::fs::write(pp.with_extension("json"), serde_json::to_string_pretty(&meta)?)?;
        eprintln!("patch: {} tensors, {} params -> {} (+ .json)", meta.tensors.len(), n_params, pp.display());
    }
    Ok(())
}

/// Apply a hub patch to a local model
pub fn apply(base: &Model, patch: &Path, out_dir: &Path, device: &Device) -> Result<()> {
    let meta: PatchMeta = serde_json::from_str(&std::fs::read_to_string(patch.with_extension("json"))?)
        .with_context(|| format!("reading {}", patch.with_extension("json").display()))?;
    let base_st = base.dir.join("optic.safetensors");
    let have = sha256_file(&base_st)?;
    if have != meta.base_sha256 {
        bail!("this model is {} but the patch applies to {} - update to that release first",
              &have[..12], &meta.base_sha256[..12]);
    }
    let mut weights = candle_core::safetensors::load(&base_st, device)?;
    let new = candle_core::safetensors::load(patch, device)?;
    for (k, t) in &new {
        let old = weights.get(k).ok_or_else(|| anyhow!("patch carries {k}, which this model does not have"))?;
        if old.dims() != t.dims() { bail!("{k}: patch is {:?} but this model has {:?}", t.dims(), old.dims()); }
    }
    for (k, t) in new { weights.insert(k, t.to_dtype(DType::F32)?); }
    std::fs::create_dir_all(out_dir)?;
    for f in std::fs::read_dir(&base.dir)? {
        let f = f?; let name = f.file_name();
        if f.file_type()?.is_file() && name != "optic.safetensors" { std::fs::copy(f.path(), out_dir.join(&name))?; }
    }
    candle_core::safetensors::save(&weights, out_dir.join("optic.safetensors"))?;
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
