//! parse and build intermediate tables from training specification (spec.json)
//!
//! spec.json describes the training scheme and inputs
//!   beta_u8   rows x probes uint8 beta codes (0..250 -> (c+0.5)/250, 255 = missing)
//!   probes    probes.tsv, one line per beta column; the model's inputs are the beta columns in order
//!   labels    labels.tsv: row, target, in_training
//!   tree      tree.json: leaves (head order), nodes (child -> parent), leaf_domain
//! Everything else (targets, hierarchical levels, row weights, read-depth tables, purity bank) is computed here.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
pub use crate::balance::RowBalance;
pub use crate::purity::{Purity, PurityMixCfg};

#[derive(Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct ModelCfg { pub rank: usize, pub hidden: usize, pub depth: usize, pub dropout: f32 }

#[derive(Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct OptimCfg { pub lr: f64, pub wd: f64, pub epochs: usize, pub sched_tmax: usize, pub batch_size: usize, pub seed: u64, pub betas: [f64; 2], pub eps: f64 }

/// Cross-depth consistency: each row is also built as a shallow view (`lo`-`hi` covered probes, log-uniform, never
/// above its own rung) whose prediction is pulled toward the deep view's detached prediction (KL, T=1, training rows),
/// with weight `weight`. The shallow view draws its own row-level randomness.
#[derive(Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct Consistency { pub weight: f64, pub lo: f64, pub hi: f64 }

/// Read depth of a covered site on sequenced rows: a zero-truncated negative binomial (mean `mu` reads per CpG
/// before truncation, dispersion `r`; support 1..30), scaled by one of `scales` per row with probabilities `weights`.
#[derive(Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct ReadDepth { pub mu: f64, pub r: f64, pub scales: Vec<f64>, pub weights: Vec<f64> }

pub const MAX_DEPTH: usize = 30;

impl ReadDepth {
    /// P(depth = d | covered), d = 1..MAX_DEPTH
    pub fn base_pmf(&self) -> Vec<f64> {
        let p = self.r / (self.r + self.mu);
        let mut pm = p.powf(self.r);                       // P(0)
        let mut v = Vec::with_capacity(MAX_DEPTH);
        for d in 1..=MAX_DEPTH { pm *= (d as f64 - 1.0 + self.r) / d as f64 * (1.0 - p); v.push(pm); }
        let s: f64 = v.iter().sum(); v.iter().map(|x| x / s).collect()
    }
    /// one CDF per scale: depth d moves to min(round(d * scale), MAX_DEPTH)
    pub fn cdfs(&self) -> Vec<Vec<f32>> {
        let b = self.base_pmf();
        self.scales.iter().map(|&f| {
            let mut o = vec![0f64; MAX_DEPTH];
            for (k, &pk) in b.iter().enumerate() { o[(((k + 1) as f64 * f).round() as usize).clamp(1, MAX_DEPTH) - 1] += pk; }
            let s: f64 = o.iter().sum(); let mut acc = 0f64;
            o.iter().map(|x| { acc += x / s; acc as f32 }).collect()
        }).collect()
    }
}

#[derive(Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub corpus: String,
    pub beta_u8: PathBuf,
    pub probes: PathBuf,
    pub labels: PathBuf,
    pub tree: PathBuf,
    /// weight name of the classification head (heads.<head>.*), kept for the inference crate
    #[serde(default = "d_head")]
    pub head: String,
    pub model: ModelCfg,
    pub optim: OptimCfg,
    /// targets are softened at this temperature in the per-row KL
    pub kd_temperature: f64,
    /// weight of each tree level's marginal KL
    pub hier_level_weight: f64,
    /// every row is trained once per epoch at each of these covered-probe counts
    pub coverage_levels: Vec<u32>,
    /// extra probe dropout on top of the coverage subsample
    pub probe_mask_prob: f64,
    /// fraction of rows given the sequencing read model (the rest keep the clean array call)
    pub nano_prob: f64,
    /// per-read methylation call error
    pub read_error: f64,
    pub read_depth: ReadDepth,
    #[serde(default)]
    pub consistency: Option<Consistency>,
    /// row weights computed from the targets (balance.rs)
    pub row_balance: RowBalance,
    #[serde(default)]
    pub purity_mix: Option<PurityMixCfg>,
    /// further purity-mix blocks, each with its own domains and background bank (e.g. CNS tumours on normal brain);
    /// a row is mixed by the first block it is eligible in, so the blocks' domains must not overlap
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub purity_mix_extra: Vec<PurityMixCfg>,
}
fn d_head() -> String { "cell_origin".into() }

/// Everything resident for the run.
pub struct Tables {
    pub n: usize,                 // corpus rows
    pub k: usize,                 // leaves
    pub p: usize,                 // probes = beta columns = model inputs
    pub lut: Vec<f32>,            // (n, k) log-prob targets, row-major
    pub roww: Vec<f32>,           // (n,) row weights (0 = not trained)
    pub hier: Vec<(Vec<f32>, usize)>, // per tree level: (n_nodes x k 0/1 matrix, n_nodes)
    pub depth_lut: Vec<f32>,      // (n_scales * 4096) inverse-CDF read-depth tables for the fused kernel
    pub purity: Vec<Purity>,      // purity_mix then purity_mix_extra; set by train::attach_purity once the corpus is in RAM
    pub bank: Option<Vec<u8>>,    // all purity banks concatenated (n_bank_total x P); bank_offset[b] = first row of block b
    pub bank_offset: Vec<u32>,
    pub weights_note: String,
}

impl Spec {
    pub fn load(dir: &Path) -> Result<Self> {
        let txt = std::fs::read_to_string(dir.join("spec.json")).with_context(|| format!("reading {}/spec.json", dir.display()))?;
        let mut s: Spec = serde_json::from_str(&txt).context("spec.json")?;
        for p in [&mut s.beta_u8, &mut s.probes, &mut s.labels, &mut s.tree] { if p.is_relative() { *p = dir.join(&*p); } }
        let blocks: Vec<&PurityMixCfg> = s.purity_mix.iter().chain(&s.purity_mix_extra).collect();
        let mut doms: Vec<i64> = blocks.iter().flat_map(|p| p.domains.iter().cloned()).collect(); let nd = doms.len(); doms.sort(); doms.dedup();
        anyhow::ensure!(doms.len() == nd, "purity_mix blocks must not share domains");
        anyhow::ensure!(s.purity_mix.is_some() || s.purity_mix_extra.is_empty(), "purity_mix_extra needs purity_mix");
        anyhow::ensure!(s.read_depth.scales.len() == s.read_depth.weights.len() && !s.read_depth.scales.is_empty(), "read_depth: scales and weights must pair up");
        Ok(s)
    }

    pub fn load_tables(&self) -> Result<Tables> {
        let tree = crate::labels::Tree::load(&self.tree)?;
        let k = tree.leaves.len();
        let (lut, mask, n) = crate::labels::targets(&self.labels, &tree)?;
        let levels = tree.hier_levels();
        let (w, weights_note) = crate::balance::balance_weights(&lut, n, k, &mask, &self.row_balance);
        let p = crate::labels::probe_count(&self.probes)?;
        let mut lut_v = Vec::with_capacity(self.read_depth.scales.len() * 4096);
        for cdf in self.read_depth.cdfs() {
            for q in 0..4096 {
                let u = (q as f32 + 0.5) / 4096.0;
                lut_v.push((1 + cdf.iter().take(MAX_DEPTH - 1).filter(|&&c| u > c).count()) as f32);
            }
        }
        println!("  inputs: {n} rows, {} training; {k} leaves, tree levels {:?}; {p} probes",
                 w.iter().filter(|&&x| x > 0.0).count(), levels.iter().map(|x| x.1).collect::<Vec<_>>());
        println!("  {weights_note}");
        let rd = &self.read_depth; let means: Vec<String> = self.read_depth.cdfs().iter().map(|c| {
            let mut prev = 0f32; let m: f32 = c.iter().enumerate().map(|(i, &x)| { let d = (x - prev) * (i + 1) as f32; prev = x; d }).sum(); format!("{m:.2}") }).collect();
        println!("  read depth: ZT-NB(mu {}, r {}) x {:?} (p {:?}) -> mean reads/covered site {}", rd.mu, rd.r, rd.scales, rd.weights, means.join(" / "));
        Ok(Tables {
            n, k, p,
            lut, roww: w, hier: levels, depth_lut: lut_v,
            purity: Vec::new(), bank: None, bank_offset: Vec::new(), weights_note,
        })
    }
}
