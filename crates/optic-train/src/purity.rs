//! Tumour-purity mixing
//!
//! Background bank: real profiles of rows whose target commits to one of the named immune or normal-tissue leaves.
//!   `bank_size` rows, `immune_share` of them immune; inside each kind the budget is split evenly over leaves with at
//!   least `min_rows_per_leaf` rows and drawn without replacement (seeded). Sampling weight: each kind gets its share of
//!   the mass, split evenly over its rows.
//! Eligible rows: training rows whose target is a single tumour leaf (peak >= `min_target_peak`) in `domains`
//!   (tree leaf_domain), and whose own immune content is already low. The
//!   `marker_probes` probes that best separate the immune background from the candidate tumours, and each candidate's
//!   position between the two means along them (needs > 200 covered markers).
//! Mixing (aug.rs + fused kernel): with probability `prob`, beta <- pi*beta_row + (1-pi)*beta_background, pi ~ U(pi_lo, pi_hi).

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct PurityMixCfg {
    pub prob: f64,
    pub pi_lo: f64,
    pub pi_hi: f64,
    pub immune_leaves: Vec<String>,
    pub tissue_leaves: Vec<String>,
    /// tree leaf_domain codes whose tumour rows may be mixed (1 = solid)
    pub domains: Vec<i64>,
    #[serde(default = "d_bank")] pub bank_size: usize,
    #[serde(default = "d_share")] pub immune_share: f64,
    #[serde(default = "d_minrows")] pub min_rows_per_leaf: usize,
    #[serde(default = "d_maxi")] pub max_immune_fraction: f64,
    #[serde(default = "d_markers")] pub marker_probes: usize,
    #[serde(default = "d_peak")] pub min_target_peak: f64,
    /// leaves with these name prefixes are never tumours (never eligible)
    #[serde(default = "d_prefixes")] pub non_tumour_prefixes: Vec<String>,
    #[serde(default)] pub seed: u64,
}
fn d_bank() -> usize { 384 }
fn d_share() -> f64 { 0.70 }
fn d_minrows() -> usize { 8 }
fn d_maxi() -> f64 { 0.20 }
fn d_markers() -> usize { 4000 }
fn d_peak() -> f64 { 0.99 }
fn d_prefixes() -> Vec<String> { vec!["Normal ".into(), "Reactive ".into(), "Non-neoplastic".into()] }

pub use crate::labels::Tree;

/// Background bank (n_bank, P) U8 on the device, per-row eligibility, cumulative sampling weights.
pub struct Purity { pub cfg: PurityMixCfg, pub bank: Vec<u8>, pub n_bank: usize, pub eligible: Vec<bool>, pub cum_w: Vec<f64>, pub summary: serde_json::Value }

/// Normalised target distribution of one row from its stored log-prob LUT row.
pub fn row_probs(row: &[f32], p: &mut [f64]) {
    let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut s = 0.0;
    for (o, &v) in p.iter_mut().zip(row) { *o = (v as f64 - mx).exp(); s += *o; }
    for o in p.iter_mut() { *o /= s; }
}

/// (argmax, best, second best)
pub fn top2(p: &[f64]) -> (usize, f64, f64) {
    let (mut bj, mut bp, mut sp) = (0usize, 0.0f64, 0.0f64);
    for (j, &v) in p.iter().enumerate() { if v > bp { sp = bp; bp = v; bj = j; } else if v > sp { sp = v; } }
    (bj, bp, sp)
}

pub fn build(cfg: &PurityMixCfg, tree: &Tree, lut: &[f32], k: usize, mask: &[f32], corpus: &[u8], p: usize) -> Result<Purity> {
    let n = mask.len();
    anyhow::ensure!(tree.leaves.len() == k, "purity_mix: tree has {} leaves, head has {k}", tree.leaves.len());
    let idx_of: HashMap<&str, usize> = tree.leaves.iter().enumerate().map(|(i, l)| (l.as_str(), i)).collect();
    let mut kind_of = vec![0u8; k];                      // 0 none, 1 immune, 2 tissue
    for (names, kd) in [(&cfg.immune_leaves, 1u8), (&cfg.tissue_leaves, 2u8)] {
        for l in names { let i = *idx_of.get(l.as_str()).ok_or_else(|| anyhow::anyhow!("purity_mix: '{l}' is not a tree leaf"))?; kind_of[i] = kd; }
    }
    let tumour: Vec<bool> = tree.leaves.iter().map(|l| {
        !cfg.non_tumour_prefixes.iter().any(|pre| l.starts_with(pre.as_str())) && cfg.domains.contains(tree.leaf_domain.get(l).unwrap_or(&-1))
    }).collect();
    // committed leaf per training row (unique winner >= 0.5), and its peak
    let mut pool: Vec<Vec<usize>> = vec![Vec::new(); k];
    let mut cand: Vec<usize> = Vec::new();
    let mut pr = vec![0f64; k];
    for r in 0..n {
        if mask[r] <= 0.0 { continue; }
        row_probs(&lut[r * k..(r + 1) * k], &mut pr);
        let (j, b, s) = top2(&pr);
        if b < 0.5 - 1e-6 || b <= s + 1e-6 { continue; }
        if kind_of[j] > 0 { pool[j].push(r); }
        if tumour[j] && b >= cfg.min_target_peak - 1e-6 { cand.push(r); }
    }
    // ---- background bank ----
    let mut rng = crate::rng::StdRng::seed_from_u64(cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(270));
    let n_imm = (cfg.bank_size as f64 * cfg.immune_share).round() as usize;
    let mut picks: Vec<(usize, u8, usize)> = Vec::new();     // (row, kind, leaf)
    let mut leaf_rows: Vec<serde_json::Value> = Vec::new();
    for (kd, budget) in [(1u8, n_imm), (2u8, cfg.bank_size - n_imm)] {
        let ls: Vec<usize> = (0..k).filter(|&j| kind_of[j] == kd && pool[j].len() >= cfg.min_rows_per_leaf).collect();
        for j in (0..k).filter(|&j| kind_of[j] == kd && pool[j].len() < cfg.min_rows_per_leaf) {
            println!("  purity bank: {} has {} rows (< {}), skipped", tree.leaves[j], pool[j].len(), cfg.min_rows_per_leaf);
        }
        if ls.is_empty() { continue; }
        let per = (budget / ls.len()).max(1);
        for &j in &ls {
            let take = per.min(pool[j].len());
            let mut chosen: Vec<usize> = rng.sample_indices(pool[j].len(), take).into_iter().map(|i| pool[j][i]).collect();
            chosen.sort_unstable();
            leaf_rows.push(serde_json::json!({"leaf": tree.leaves[j], "kind": if kd == 1 { "immune" } else { "tissue" }, "available": pool[j].len(), "taken": take}));
            picks.extend(chosen.into_iter().map(|r| (r, kd, j)));
        }
    }
    picks.truncate(cfg.bank_size);
    anyhow::ensure!(!picks.is_empty(), "purity_mix: no background rows found for the named leaves");
    let n_k = |kd: u8| picks.iter().filter(|x| x.1 == kd).count();
    let (ni, nt) = (n_k(1), n_k(2));
    let mut cum_w = Vec::with_capacity(picks.len()); let mut acc = 0.0;
    for &(_, kd, _) in &picks {
        let share = if kd == 1 { cfg.immune_share } else { 1.0 - cfg.immune_share };
        acc += share / (if kd == 1 { ni } else { nt }).max(1) as f64; cum_w.push(acc);
    }
    let mut bank_cpu = vec![0u8; picks.len() * p];
    for (i, &(r, _, _)) in picks.iter().enumerate() { bank_cpu[i * p..(i + 1) * p].copy_from_slice(&corpus[r * p..(r + 1) * p]); }
    let bank = bank_cpu;
    // ---- eligibility: low existing immune content ----
    let imm_rows: Vec<usize> = picks.iter().filter(|x| x.1 == 1).map(|x| x.0).collect();
    let imm_all: Vec<usize> = (0..k).filter(|&j| kind_of[j] == 1 && pool[j].len() >= cfg.min_rows_per_leaf).flat_map(|j| pool[j].iter().cloned()).collect();
    let src_imm = if imm_all.is_empty() { imm_rows.clone() } else { { let mut v = imm_all; v.sort_unstable(); v } };
    let prof = |rows: &[usize]| -> (Vec<f64>, Vec<f64>) {
        let step = (rows.len() / 300).max(1);
        let sub: Vec<usize> = rows.iter().step_by(step).take(300).cloned().collect();
        let (mut s, mut s2, mut c) = (vec![0f64; p], vec![0f64; p], vec![0u32; p]);
        for &r in &sub { for (j, &v) in corpus[r * p..(r + 1) * p].iter().enumerate() { if v <= 250 { let b = (v as f64 + 0.5) / 250.0; s[j] += b; s2[j] += b * b; c[j] += 1; } } }
        let mean: Vec<f64> = (0..p).map(|j| if c[j] > 0 { s[j] / c[j] as f64 } else { f64::NAN }).collect();
        let sd: Vec<f64> = (0..p).map(|j| if c[j] > 0 { (s2[j] / c[j] as f64 - mean[j] * mean[j]).max(0.0).sqrt() } else { f64::NAN }).collect();
        (mean, sd)
    };
    let (mb, sb) = prof(&src_imm); let (ms, ss) = prof(&cand);
    let mut score: Vec<(f64, usize)> = (0..p).filter_map(|j| { let d = mb[j] - ms[j]; if d.is_finite() && sb[j].is_finite() && ss[j].is_finite() { Some((d.abs() / (sb[j] + ss[j] + 0.05), j)) } else { None } }).collect();
    score.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let mark: Vec<usize> = score.iter().take(cfg.marker_probes).map(|x| x.1).collect();
    let mut eligible = vec![false; n]; let mut kept = 0usize;
    for &r in &cand {
        let row = &corpus[r * p..(r + 1) * p];
        let (mut num, mut den, mut g) = (0f64, 0f64, 0usize);
        for &j in &mark { let v = row[j]; if v <= 250 { let d = mb[j] - ms[j]; num += ((v as f64 + 0.5) / 250.0 - ms[j]) * d; den += d * d; g += 1; } }
        let f = if g > 200 && den > 0.0 { (num / den).clamp(0.0, 1.0) } else { 1.0 };
        if f <= cfg.max_immune_fraction { eligible[r] = true; kept += 1; }
    }
    let summary = serde_json::json!({"bank_rows": picks.iter().map(|x| x.0).collect::<Vec<_>>(), "bank_leaves": leaf_rows,
        "n_immune": ni, "n_tissue": nt, "n_candidates": cand.len(), "n_eligible": kept});
    println!("purity mix: bank {} profiles ({ni} immune, {nt} tissue) from current targets; eligible {kept}/{} single-leaf tumour rows (immune fraction <= {}); pi ~ U({:.2},{:.2}) on {:.0}% of eligible rows",
             picks.len(), cand.len(), cfg.max_immune_fraction, cfg.pi_lo, cfg.pi_hi, 100.0 * cfg.prob);
    Ok(Purity { cfg: cfg.clone(), bank, n_bank: picks.len(), eligible, cum_w, summary })
}
