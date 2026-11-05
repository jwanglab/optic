use crate::nn::{layer_norm, layer_norm_backward, linear, linear_backward, logsumexp, project_ternary, silu, silu_backward, softmax};
use crate::st::{Tensor, Tensors};
use anyhow::{anyhow, Context, Result};
use std::path::Path;

/// The network: `first.0` (probes -> 256, no bias), `first.1` (256 -> 1024) -> LayerNorm `ln1` -> SiLU, residual
/// blocks `trunk.i` (h + SiLU(LN(lin(h)))), and per-output heads `heads.<name>` (lin `0` -> LN `1` -> SiLU -> lin `4`).
pub struct Net {
    w: Tensors,
    n_blocks: usize,
    /// (head name, optional coverage-dependent temperature: bin centres, log temperatures)
    heads: Vec<(String, Option<(Vec<f32>, Vec<f32>)>)>,
}

pub struct Prediction {
    pub probs: Vec<(String, Vec<f32>)>,
    pub temperature: Vec<(String, f32)>,
    pub n_covered: usize,
}

/// Activations kept by `forward_train` for `backward`.
pub struct Trace {
    n: usize,
    head: String,
    z: Vec<f32>,
    first: Stage,
    blocks: Vec<(Vec<f32>, Stage)>, // block input h, and its stage
    head_in: Vec<f32>,
    head_stage: Stage,
    head_act: Vec<f32>, // SiLU output feeding heads.<name>.4
}

/// One lin -> LN -> SiLU stage: the LN's x̂ and 1/σ, and the LN output (the SiLU input).
struct Stage { xhat: Vec<f32>, inv: Vec<f32>, a: Vec<f32> }

impl Net {
    pub fn load(path: &Path, n_probes: usize) -> Result<Self> {
        let w = crate::st::load(path).with_context(|| format!("loading {}", path.display()))?;
        let first0 = w.get("first.0.weight").ok_or_else(|| anyhow!("missing weight first.0.weight"))?;
        if first0.dim(1) != n_probes {
            return Err(anyhow!("model expects {} probes, probes.T2T.bed has {}", first0.dim(1), n_probes));
        }
        let mut names: Vec<String> = w.keys()
            .filter_map(|k| k.strip_prefix("heads.").and_then(|s| s.strip_suffix(".4.weight")).map(str::to_string))
            .collect();
        names.sort();
        if names.is_empty() { return Err(anyhow!("no heads found in {}", path.display())); }
        let heads = names.into_iter().map(|name| {
            let calib = match (w.get(&format!("{name}_bin_centers")), w.get(&format!("{name}_log_temps"))) {
                (Some(bc), Some(lt)) => Some((bc.data.clone(), lt.data.clone())),
                _ => None,
            };
            (name, calib)
        }).collect();
        let n_blocks = (0..).take_while(|i| w.contains_key(&format!("trunk.{i}.lin.weight"))).count();
        let net = Net { w, n_blocks, heads };
        for k in net.federated_names(&net.heads[0].0) { net.get(&k)?; }
        Ok(net)
    }

    fn get(&self, k: &str) -> Result<&Tensor> { self.w.get(k).ok_or_else(|| anyhow!("missing weight {k}")) }

    fn temperature(calib: &Option<(Vec<f32>, Vec<f32>)>, n_covered: usize) -> f32 {
        match calib {
            Some((bc, lt)) => {
                let nc = n_covered as f32;
                let i = (0..bc.len()).min_by(|&a, &b| (bc[a] - nc).abs().total_cmp(&(bc[b] - nc).abs())).unwrap_or(0);
                lt[i].exp()
            }
            None => 1.0,
        }
    }

    pub fn n_blocks(&self) -> usize { self.n_blocks }

    /// The frozen rank-256 projection of a (n x probes) ternary batch. `first.0` is 96.8% of the parameters and its
    /// gradient is an outer product with the raw probe vector, so federation precomputes this and never updates it.
    pub fn first0_proj(&self, x: &[f32], n: usize) -> Result<Vec<f32>> { Ok(project_ternary(x, n, self.get("first.0.weight")?)) }

    /// Everything a site may update: the whole network above `first.0`, by safetensors name.
    pub fn federated_names(&self, head: &str) -> Vec<String> {
        let mut v: Vec<String> = ["first.1.weight", "first.1.bias", "ln1.weight", "ln1.bias"].map(String::from).to_vec();
        for i in 0..self.n_blocks {
            for k in ["lin.weight", "lin.bias", "ln.weight", "ln.bias"] { v.push(format!("trunk.{i}.{k}")); }
        }
        for k in ["0.weight", "0.bias", "1.weight", "1.bias", "4.weight", "4.bias"] { v.push(format!("heads.{head}.{k}")); }
        v
    }

    pub fn has_head(&self, head: &str) -> bool { self.heads.iter().any(|(h, _)| h == head) }

    /// lin -> LN -> SiLU on a batch; returns (SiLU output, stage).
    fn stage(&self, x: &[f32], n: usize, lin: &str, ln: &str) -> Result<(Vec<f32>, Stage)> {
        let r = linear(x, n, self.get(&format!("{lin}.weight"))?, Some(self.get(&format!("{lin}.bias"))?));
        let (a, xhat, inv) = layer_norm(&r, n, self.get(&format!("{ln}.weight"))?, self.get(&format!("{ln}.bias"))?);
        Ok((silu(&a), Stage { xhat, inv, a }))
    }

    /// Forward from the frozen `first.0` projection z (n x 256) to `head`'s logits, keeping what `backward` needs.
    pub fn forward_train(&self, z: &[f32], n: usize, head: &str) -> Result<(Vec<f32>, Trace)> {
        if !self.has_head(head) { return Err(anyhow!("no head {head}")); }
        let (mut h, first) = self.stage(z, n, "first.1", "ln1")?;
        let mut blocks = Vec::with_capacity(self.n_blocks);
        for i in 0..self.n_blocks {
            let (s, st) = self.stage(&h, n, &format!("trunk.{i}.lin"), &format!("trunk.{i}.ln"))?;
            let next: Vec<f32> = h.iter().zip(&s).map(|(a, b)| a + b).collect();
            blocks.push((std::mem::replace(&mut h, next), st));
        }
        let p = format!("heads.{head}");
        let (act, head_stage) = self.stage(&h, n, &format!("{p}.0"), &format!("{p}.1"))?;
        let logits = linear(&act, n, self.get(&format!("{p}.4.weight"))?, Some(self.get(&format!("{p}.4.bias"))?));
        Ok((logits, Trace { n, head: head.to_string(), z: z.to_vec(), first, blocks, head_in: h, head_stage, head_act: act }))
    }

    /// Gradients of the loss with respect to every federated parameter, given dL/dlogits.
    pub fn backward(&self, t: &Trace, dlogits: &[f32]) -> Result<Tensors> {
        let n = t.n;
        let mut g: Tensors = Tensors::new();
        for k in self.federated_names(&t.head) { g.insert(k.clone(), Tensor::zeros(self.get(&k)?.shape.clone())); }
        let mut grad = |k: &str| -> Result<Tensor> { g.remove(k).ok_or_else(|| anyhow!("no gradient slot {k}")) };
        let mut done: Vec<(String, Tensor)> = Vec::new();

        // back through lin -> LN -> SiLU: dL/d(SiLU out) -> dL/d(stage input)
        let stage_back = |x: &[f32], st: &Stage, ds: &[f32], lin: &str, ln: &str, want_dx: bool,
                          grad: &mut dyn FnMut(&str) -> Result<Tensor>, done: &mut Vec<(String, Tensor)>| -> Result<Option<Vec<f32>>> {
            let da = silu_backward(&st.a, ds);
            let (mut lw, mut lb) = (grad(&format!("{ln}.weight"))?, grad(&format!("{ln}.bias"))?);
            let dr = layer_norm_backward(&st.xhat, &st.inv, self.get(&format!("{ln}.weight"))?, &da, &mut lw.data, &mut lb.data);
            let (mut w, mut b) = (grad(&format!("{lin}.weight"))?, grad(&format!("{lin}.bias"))?);
            let dx = linear_backward(x, n, self.get(&format!("{lin}.weight"))?, &dr, &mut w.data, Some(&mut b.data), want_dx);
            for (k, v) in [(format!("{ln}.weight"), lw), (format!("{ln}.bias"), lb), (format!("{lin}.weight"), w), (format!("{lin}.bias"), b)] { done.push((k, v)); }
            Ok(dx)
        };

        let p = format!("heads.{}", t.head);
        let (mut w4, mut b4) = (grad(&format!("{p}.4.weight"))?, grad(&format!("{p}.4.bias"))?);
        let dact = linear_backward(&t.head_act, n, self.get(&format!("{p}.4.weight"))?, dlogits, &mut w4.data, Some(&mut b4.data), true).unwrap();
        done.push((format!("{p}.4.weight"), w4)); done.push((format!("{p}.4.bias"), b4));
        let mut dh = stage_back(&t.head_in, &t.head_stage, &dact, &format!("{p}.0"), &format!("{p}.1"), true, &mut grad, &mut done)?.unwrap();
        for (i, (hin, st)) in t.blocks.iter().enumerate().rev() {
            // h_out = h_in + stage(h_in): the residual passes dh through unchanged
            let dx = stage_back(hin, st, &dh, &format!("trunk.{i}.lin"), &format!("trunk.{i}.ln"), true, &mut grad, &mut done)?.unwrap();
            dh.iter_mut().zip(dx).for_each(|(a, b)| *a += b);
        }
        stage_back(&t.z, &t.first, &dh, "first.1", "ln1", false, &mut grad, &mut done)?;
        Ok(done.into_iter().collect())
    }

    pub fn infer(&self, ternary: &[f32]) -> Result<Prediction> {
        let n_covered = ternary.iter().filter(|&&v| v != 0.0).count();
        let z = self.first0_proj(ternary, 1)?;
        let mut probs = Vec::new();
        let mut temperature = Vec::new();
        for (name, calib) in &self.heads {
            let (logits, _) = self.forward_train(&z, 1, name)?;
            let t = Self::temperature(calib, n_covered);
            probs.push((name.clone(), softmax(&logits, 1, t)));
            temperature.push((name.clone(), t));
        }
        Ok(Prediction { probs, temperature, n_covered })
    }
}

/// Federated loss for one batch: -log P(true class set) per row, where `masks` (n x k) marks each row's allowed
/// classes (a leaf, or every leaf under a node). Returns (summed loss, dL/dlogits).
pub fn set_nll(logits: &[f32], masks: &[Vec<f32>]) -> (f32, Vec<f32>) {
    let n = masks.len();
    let k = logits.len() / n.max(1);
    let mut loss = 0f32;
    let mut d = vec![0f32; logits.len()];
    for r in 0..n {
        let row = &logits[r * k..(r + 1) * k];
        let masked: Vec<f32> = row.iter().zip(&masks[r]).map(|(&l, &m)| l + (m - 1.0) * 1e4).collect();
        loss += logsumexp(row) - logsumexp(&masked);
        let (p, q) = (softmax(row, 1, 1.0), softmax(&masked, 1, 1.0));
        for c in 0..k { d[r * k + c] = p[c] - q[c]; }
    }
    (loss, d)
}
