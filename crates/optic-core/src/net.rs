use crate::nn::{head_forward, layer_norm, silu};
use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor};
use std::path::Path;

pub struct Net {
    first0_w: Tensor,
    first1_w: Tensor,
    first1_b: Tensor,
    ln1_w: Tensor,
    ln1_b: Tensor,
    trunk: Vec<Block>,
    heads: Vec<(String, Head)>,
}

struct Block { lin_w: Tensor, lin_b: Tensor, ln_w: Tensor, ln_b: Tensor }

struct Head {
    h0_w: Tensor, h0_b: Tensor, h1_w: Tensor, h1_b: Tensor, h4_w: Tensor, h4_b: Tensor,
    calib: Option<(Vec<f32>, Vec<f32>)>,
}

pub struct Prediction {
    pub probs: Vec<(String, Vec<f32>)>,
    pub temperature: Vec<(String, f32)>,
    pub n_covered: usize,
}

impl Net {
    pub fn load(path: &Path, n_probes: usize, device: &Device) -> Result<Self> {
        let w = candle_core::safetensors::load(path, device)
            .with_context(|| format!("loading {}", path.display()))?;
        let get = |k: &str| -> Result<Tensor> {
            w.get(k).ok_or_else(|| anyhow!("missing weight {k}"))
                .and_then(|t| t.to_dtype(DType::F32).map_err(Into::into))
        };
        let mut trunk = Vec::new();
        for i in 0.. {
            let p = format!("trunk.{i}.");
            if !w.contains_key(&format!("{p}lin.weight")) { break; }
            trunk.push(Block {
                lin_w: get(&format!("{p}lin.weight"))?, lin_b: get(&format!("{p}lin.bias"))?,
                ln_w: get(&format!("{p}ln.weight"))?, ln_b: get(&format!("{p}ln.bias"))?,
            });
        }
        let first0_w = get("first.0.weight")?;
        if first0_w.dim(1)? != n_probes {
            return Err(anyhow!("model expects {} probes, probe_subset.bed has {}", first0_w.dim(1)?, n_probes));
        }
        let mut names: Vec<String> = w.keys()
            .filter_map(|k| k.strip_prefix("heads.").and_then(|s| s.strip_suffix(".4.weight")).map(str::to_string))
            .collect();
        names.sort();
        let mut heads = Vec::new();
        for name in names {
            let p = format!("heads.{name}.");
            let calib = match (w.get(&format!("{name}_bin_centers")), w.get(&format!("{name}_log_temps"))) {
                (Some(bc), Some(lt)) => Some((bc.flatten_all()?.to_vec1::<f32>()?, lt.flatten_all()?.to_vec1::<f32>()?)),
                _ => None,
            };
            heads.push((name.clone(), Head {
                h0_w: get(&format!("{p}0.weight"))?, h0_b: get(&format!("{p}0.bias"))?,
                h1_w: get(&format!("{p}1.weight"))?, h1_b: get(&format!("{p}1.bias"))?,
                h4_w: get(&format!("{p}4.weight"))?, h4_b: get(&format!("{p}4.bias"))?, calib,
            }));
        }
        if heads.is_empty() { return Err(anyhow!("no heads found in {}", path.display())); }
        Ok(Net {
            first0_w, first1_w: get("first.1.weight")?, first1_b: get("first.1.bias")?,
            ln1_w: get("ln1.weight")?, ln1_b: get("ln1.bias")?, trunk, heads,
        })
    }

    fn temperature(head: &Head, n_covered: usize) -> f32 {
        match &head.calib {
            Some((bc, lt)) => {
                let nc = n_covered as f32;
                let i = (0..bc.len()).min_by(|&a, &b| (bc[a] - nc).abs().total_cmp(&(bc[b] - nc).abs())).unwrap_or(0);
                lt[i].exp()
            }
            None => 1.0,
        }
    }

    /// Frozen-trunk features for a (n x probes) ternary batch.

    /// The frozen rank-256 projection. `first.0` is 96.8% of the parameters and its gradient is an
    /// outer product with the raw probe vector, so federation precomputes this and never updates it.
    pub fn first0_proj(&self, x: &Tensor) -> Result<Tensor> { Ok(x.matmul(&self.first0_w.t()?)?) }

    pub fn n_blocks(&self) -> usize { self.trunk.len() }

    /// Everything a site may update: the whole network above `first.0`, keyed by safetensors name.
    pub fn federated_params(&self, head: &str) -> Option<Vec<(String, Tensor)>> {
        let h = self.heads.iter().find(|(n, _)| n == head).map(|(_, h)| h)?;
        let mut v = vec![
            ("first.1.weight".to_string(), self.first1_w.clone()),
            ("first.1.bias".to_string(), self.first1_b.clone()),
            ("ln1.weight".to_string(), self.ln1_w.clone()),
            ("ln1.bias".to_string(), self.ln1_b.clone()),
        ];
        for (i, b) in self.trunk.iter().enumerate() {
            v.push((format!("trunk.{i}.lin.weight"), b.lin_w.clone()));
            v.push((format!("trunk.{i}.lin.bias"), b.lin_b.clone()));
            v.push((format!("trunk.{i}.ln.weight"), b.ln_w.clone()));
            v.push((format!("trunk.{i}.ln.bias"), b.ln_b.clone()));
        }
        for (k, t) in [("0.weight", &h.h0_w), ("0.bias", &h.h0_b), ("1.weight", &h.h1_w),
                       ("1.bias", &h.h1_b), ("4.weight", &h.h4_w), ("4.bias", &h.h4_b)] {
            v.push((format!("heads.{head}.{k}"), t.clone()));
        }
        Some(v)
    }

    pub fn trunk_features(&self, x: &Tensor) -> Result<Tensor> {
        let h = x.matmul(&self.first0_w.t()?)?.matmul(&self.first1_w.t()?)?.broadcast_add(&self.first1_b)?;
        let mut h = silu(&layer_norm(&h, &self.ln1_w, &self.ln1_b)?)?;
        for b in &self.trunk {
            let r = h.matmul(&b.lin_w.t()?)?.broadcast_add(&b.lin_b)?;
            h = (&h + silu(&layer_norm(&r, &b.ln_w, &b.ln_b)?)?)?;
        }
        Ok(h)
    }

    /// Head parameters keyed by their safetensors suffix ("0.weight", ...).
    pub fn head_params(&self, head: &str) -> Option<Vec<(String, Tensor)>> {
        self.heads.iter().find(|(n, _)| n == head).map(|(_, h)| vec![
            ("0.weight".into(), h.h0_w.clone()), ("0.bias".into(), h.h0_b.clone()), ("1.weight".into(), h.h1_w.clone()),
            ("1.bias".into(), h.h1_b.clone()), ("4.weight".into(), h.h4_w.clone()), ("4.bias".into(), h.h4_b.clone())])
    }

    pub fn infer(&self, ternary: &[f32], device: &Device) -> Result<Prediction> {
        let n_covered = ternary.iter().filter(|&&v| v != 0.0).count();
        let x = Tensor::from_vec(ternary.to_vec(), (1, ternary.len()), device)?;
        let h = self.trunk_features(&x)?;
        let mut probs = Vec::new();
        let mut temperature = Vec::new();
        for (name, hd) in &self.heads {
            let logits = head_forward(&h, &hd.h0_w, &hd.h0_b, &hd.h1_w, &hd.h1_b, &hd.h4_w, &hd.h4_b)?;
            let t = Self::temperature(hd, n_covered);
            let p = candle_nn::ops::softmax(&(&logits / t as f64)?, 1)?.flatten_all()?.to_vec1::<f32>()?;
            probs.push((name.clone(), p));
            temperature.push((name.clone(), t));
        }
        Ok(Prediction { probs, temperature, n_covered })
    }
}

/// Forward from the frozen `first.0` projection to head logits, reading every parameter out of `p`.
/// Federated training runs this over `Var`s so the gradients come from the same code path inference uses.
pub fn forward_from_proj(z: &Tensor, n_blocks: usize, head: &str, p: &[(String, Tensor)]) -> Result<Tensor> {
    let g = |k: &str| -> Result<&Tensor> {
        p.iter().find(|(n, _)| n == k).map(|(_, t)| t)
            .ok_or_else(|| anyhow!("federated forward wants {k}, which is not a federated parameter"))
    };
    let h = z.matmul(&g("first.1.weight")?.t()?)?.broadcast_add(g("first.1.bias")?)?;
    let mut h = silu(&layer_norm(&h, g("ln1.weight")?, g("ln1.bias")?)?)?;
    for i in 0..n_blocks {
        let r = h.matmul(&g(&format!("trunk.{i}.lin.weight"))?.t()?)?
            .broadcast_add(g(&format!("trunk.{i}.lin.bias"))?)?;
        h = (&h + silu(&layer_norm(&r, g(&format!("trunk.{i}.ln.weight"))?, g(&format!("trunk.{i}.ln.bias"))?)?)?)?;
    }
    head_forward(&h, g(&format!("heads.{head}.0.weight"))?, g(&format!("heads.{head}.0.bias"))?,
                 g(&format!("heads.{head}.1.weight"))?, g(&format!("heads.{head}.1.bias"))?,
                 g(&format!("heads.{head}.4.weight"))?, g(&format!("heads.{head}.4.bias"))?)
}
