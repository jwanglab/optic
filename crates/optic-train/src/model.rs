//! The optic network on the GPU, forward and backward, weight-name compatible with optic's model files:
//!   first.0 (P -> rank, no bias) -> first.1 (rank -> hidden) -> ln1 -> dropout -> SiLU
//!   trunk.i: h + SiLU(dropout(LN(lin(h))))
//!   heads.<head>: .0 (hidden -> hidden/2) -> .1 (LN) -> dropout -> SiLU -> .4 (hidden/2 -> k)
//! Every lin -> LN -> dropout -> SiLU group is a `Stage`; the forward keeps what the backward needs.

use crate::gpu::{Buf, Gpu};
use anyhow::{anyhow, Result};
use optic_core::st::{Tensor, Tensors};
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub struct Cfg { pub probes: usize, pub rank: usize, pub hidden: usize, pub depth: usize, pub head: String, pub k: usize }

pub struct Net {
    pub cfg: Cfg,
    /// name -> (shape, values)
    pub p: BTreeMap<String, (Vec<usize>, Buf)>,
    /// gradients of the last backward, by name
    pub g: BTreeMap<String, Buf>,
    /// AdamW moments and step count
    m: BTreeMap<String, Buf>,
    v: BTreeMap<String, Buf>,
    t: usize,
}

/// What a stage keeps for its backward: its input, the LN input and statistics, and the SiLU input.
struct StageCache { hin: Buf, u: Buf, stats: Buf, d: Buf }

/// Activations of one forward pass.
pub struct Cache { rows: usize, z0: Buf, stages: Vec<StageCache>, act: Buf }

/// One stage's parameter names: (lin, LN).
fn stage_names(cfg: &Cfg) -> Vec<(String, String)> {
    let mut v = vec![("first.1".to_string(), "ln1".to_string())];
    for i in 0..cfg.depth { v.push((format!("trunk.{i}.lin"), format!("trunk.{i}.ln"))); }
    v.push((format!("heads.{}.0", cfg.head), format!("heads.{}.1", cfg.head)));
    v
}

impl Net {
    pub fn names(cfg: &Cfg) -> Vec<String> {
        let mut v = vec!["first.0.weight".to_string()];
        for (lin, ln) in stage_names(cfg) { for s in [format!("{lin}.weight"), format!("{lin}.bias"), format!("{ln}.weight"), format!("{ln}.bias")] { v.push(s); } }
        v.push(format!("heads.{}.4.weight", cfg.head)); v.push(format!("heads.{}.4.bias", cfg.head));
        v
    }

    pub fn from_tensors(gpu: &Gpu, cfg: Cfg, w: &Tensors) -> Result<Self> {
        let mut p = BTreeMap::new();
        let (mut m, mut v) = (BTreeMap::new(), BTreeMap::new());
        for n in Self::names(&cfg) {
            let t = w.get(&n).ok_or_else(|| anyhow!("missing weight {n}"))?;
            p.insert(n.clone(), (t.shape.clone(), gpu.upload(&t.data)?));
            m.insert(n.clone(), gpu.zeros::<f32>(t.len())?); v.insert(n.clone(), gpu.zeros::<f32>(t.len())?);
        }
        Ok(Net { cfg, p, g: BTreeMap::new(), m, v, t: 0 })
    }

    pub fn to_tensors(&self, gpu: &Gpu) -> Result<Tensors> {
        self.p.iter().map(|(n, (s, b))| Ok((n.clone(), Tensor::new(s.clone(), gpu.download(b)?)))).collect()
    }

    fn w(&self, n: &str) -> &Buf { &self.p[n].1 }
    fn dim(&self, n: &str, i: usize) -> usize { self.p[n].0[i] }

    /// lin -> LN -> (dropout) -> SiLU on `hin` (rows x din). Returns the SiLU output and the cache.
    fn stage_fwd(&self, gpu: &Gpu, hin: Buf, rows: usize, lin: &str, ln: &str, mask: Option<&Buf>) -> Result<(Buf, StageCache)> {
        let (dout, din) = (self.dim(&format!("{lin}.weight"), 0), self.dim(&format!("{lin}.weight"), 1));
        let mut u = gpu.zeros::<f32>(rows * dout)?;
        gpu.mm_xwt(&mut u, &hin, self.w(&format!("{lin}.weight")), rows, dout, din, 0.0)?;
        gpu.add_bias(&mut u, self.w(&format!("{lin}.bias")), rows, dout)?;
        let mut stats = gpu.zeros::<f32>(2 * rows)?;
        gpu.ln_stats(&mut stats, &u, rows, dout, 1e-5)?;
        let mut a = gpu.zeros::<f32>(rows * dout)?;
        gpu.ln_fwd(&mut a, &u, self.w(&format!("{ln}.weight")), self.w(&format!("{ln}.bias")), rows, dout, 1e-5)?;
        let d = match mask { Some(m) => { let mut d = gpu.zeros::<f32>(rows * dout)?; gpu.mul(&mut d, &a, m, rows * dout)?; d } None => a };
        let mut s = gpu.zeros::<f32>(rows * dout)?;
        gpu.silu(&mut s, &d, rows * dout)?;
        Ok((s, StageCache { hin, u, stats, d }))
    }

    /// Logits (rows x k) of a (rows x P) input; `masks` are the dropout masks in stage order (None: evaluation).
    pub fn forward(&self, gpu: &Gpu, x: &Buf, rows: usize, masks: Option<&[Buf]>) -> Result<(Buf, Cache)> {
        let c = &self.cfg;
        let mask = |i: usize| masks.map(|m| &m[i]);
        let mut z0 = gpu.zeros::<f32>(rows * c.rank)?;
        gpu.mm_xwt(&mut z0, x, self.w("first.0.weight"), rows, c.rank, c.probes, 0.0)?;
        let names = stage_names(c);
        let mut stages = Vec::with_capacity(names.len());
        let (mut h, st) = self.stage_fwd(gpu, z0.clone(), rows, &names[0].0, &names[0].1, mask(0))?;
        stages.push(st);
        for (i, (lin, ln)) in names.iter().enumerate().skip(1).take(c.depth) {
            let (s, st) = self.stage_fwd(gpu, h.clone(), rows, lin, ln, mask(i))?;
            gpu.add(&mut h, &s, rows * c.hidden)?;        // residual
            stages.push(st);
        }
        let (lin, ln) = &names[c.depth + 1];
        let (act, st) = self.stage_fwd(gpu, h, rows, lin, ln, mask(c.depth + 1))?;
        stages.push(st);
        let w4 = format!("heads.{}.4", c.head);
        let mut z = gpu.zeros::<f32>(rows * c.k)?;
        gpu.mm_xwt(&mut z, &act, self.w(&format!("{w4}.weight")), rows, c.k, c.hidden / 2, 0.0)?;
        gpu.add_bias(&mut z, self.w(&format!("{w4}.bias")), rows, c.k)?;
        Ok((z, Cache { rows, z0, stages, act }))
    }

    /// Backward of a stage from ds (rows x dout); stores its gradients; returns d(stage input) if asked.
    fn stage_bwd(&mut self, gpu: &Gpu, sc: &StageCache, ds: &Buf, rows: usize, lin: &str, ln: &str, mask: Option<&Buf>, want_dx: bool) -> Result<Option<Buf>> {
        let (dout, din) = (self.dim(&format!("{lin}.weight"), 0), self.dim(&format!("{lin}.weight"), 1));
        let n = rows * dout;
        let mut dd = gpu.zeros::<f32>(n)?;
        gpu.silu_bwd(&mut dd, &sc.d, ds, n)?;
        let da = match mask { Some(m) => { let mut da = gpu.zeros::<f32>(n)?; gpu.mul(&mut da, &dd, m, n)?; da } None => dd };
        let mut dxhat = gpu.zeros::<f32>(n)?;
        gpu.mul_cols(&mut dxhat, &da, self.w(&format!("{ln}.weight")), rows, dout)?;
        let mut du = gpu.zeros::<f32>(n)?;
        gpu.ln_dx(&mut du, &dxhat, &sc.u, &sc.stats, rows, dout)?;
        let mut dgdb = gpu.zeros::<f32>(2 * dout)?;
        gpu.ln_dwdb(&mut dgdb, &da, &sc.u, &sc.stats, rows, dout)?;
        let (mut dgam, mut dbet) = (gpu.zeros::<f32>(dout)?, gpu.zeros::<f32>(dout)?);
        gpu.stream.memcpy_dtod(&dgdb.slice(0..dout), &mut dgam)?;
        gpu.stream.memcpy_dtod(&dgdb.slice(dout..2 * dout), &mut dbet)?;
        self.g.insert(format!("{ln}.weight"), dgam); self.g.insert(format!("{ln}.bias"), dbet);
        let mut dw = gpu.zeros::<f32>(dout * din)?;
        gpu.mm_dytx(&mut dw, &du, &sc.hin, rows, dout, din)?;
        let mut db = gpu.zeros::<f32>(dout)?;
        gpu.col_sum(&mut db, &du, rows, dout)?;
        self.g.insert(format!("{lin}.weight"), dw); self.g.insert(format!("{lin}.bias"), db);
        if !want_dx { return Ok(None); }
        let mut dx = gpu.zeros::<f32>(rows * din)?;
        gpu.mm_dyw(&mut dx, &du, self.w(&format!("{lin}.weight")), rows, dout, din)?;
        Ok(Some(dx))
    }

    /// Gradients of every parameter (into `self.g`) given dL/dlogits (rows x k) and the forward's input and cache.
    pub fn backward(&mut self, gpu: &Gpu, x: &Buf, cache: &Cache, dz: &Buf, masks: Option<&[Buf]>) -> Result<()> {
        let c = self.cfg.clone();
        let rows = cache.rows;
        let mask = |i: usize| masks.map(|m| &m[i]);
        let names = stage_names(&c);
        let w4 = format!("heads.{}.4", c.head);
        let mut dw4 = gpu.zeros::<f32>(c.k * c.hidden / 2)?;
        gpu.mm_dytx(&mut dw4, dz, &cache.act, rows, c.k, c.hidden / 2)?;
        let mut db4 = gpu.zeros::<f32>(c.k)?;
        gpu.col_sum(&mut db4, dz, rows, c.k)?;
        self.g.insert(format!("{w4}.weight"), dw4); self.g.insert(format!("{w4}.bias"), db4);
        let mut dact = gpu.zeros::<f32>(rows * c.hidden / 2)?;
        gpu.mm_dyw(&mut dact, dz, self.w(&format!("{w4}.weight")), rows, c.k, c.hidden / 2)?;
        let last = c.depth + 1;
        let mut dh = self.stage_bwd(gpu, &cache.stages[last], &dact, rows, &names[last].0, &names[last].1, mask(last), true)?.unwrap();
        for i in (1..=c.depth).rev() {
            // h_out = h_in + stage(h_in): the residual passes dh straight through
            let dx = self.stage_bwd(gpu, &cache.stages[i], &dh, rows, &names[i].0, &names[i].1, mask(i), true)?.unwrap();
            gpu.add(&mut dh, &dx, rows * c.hidden)?;
        }
        let dz0 = self.stage_bwd(gpu, &cache.stages[0], &dh, rows, &names[0].0, &names[0].1, mask(0), true)?.unwrap();
        let mut dw0 = gpu.zeros::<f32>(c.rank * c.probes)?;
        gpu.mm_dytx(&mut dw0, &dz0, x, rows, c.rank, c.probes)?;
        self.g.insert("first.0.weight".into(), dw0);
        let _ = &cache.z0;
        Ok(())
    }

    /// One AdamW step over every parameter with a gradient.
    pub fn adamw(&mut self, gpu: &Gpu, lr: f64, b1: f64, b2: f64, eps: f64, wd: f64) -> Result<()> {
        self.t += 1;
        let scale_m = (1.0 - b1.powi(self.t as i32)).recip() as f32;
        let scale_v = (1.0 - b2.powi(self.t as i32)).recip() as f32;
        for (n, (_, p)) in self.p.iter_mut() {
            let Some(g) = self.g.get(n) else { continue };
            let len = p.len();
            gpu.adamw(p, g, self.m.get_mut(n).unwrap(), self.v.get_mut(n).unwrap(), len, lr as f32, b1 as f32, b2 as f32, eps as f32, wd as f32, scale_m, scale_v)?;
        }
        Ok(())
    }
}
