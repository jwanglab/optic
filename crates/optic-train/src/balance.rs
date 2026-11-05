//! Row weights computed from the targets (spec.json "row_balance")
//!
//!   leaf frequency  n_l = training rows COMMITTED to leaf l (it holds >= `commit` of the target mass and beats every
//!                   other leaf); vague targets (node / flat / even two-part) do not form classes of their own
//!   leaf weight     w_l = clip((median n / n_l)^power, lo, hi); a leaf with no committed rows gets `hi`
//!   row weight      w_r = (sum_l q_rl * w_l) * (max_l q_rl)^certainty_power     (q_r = target distribution)
//!   then rescaled so the mean over training rows is `mean`.
//! A leaf target counts fully; a target spread evenly over m leaves counts 1/m - leaf-balanced, scaled by how definite
//! the label is. (Leaf-balancing that let vague targets form their own small classes roughly doubled their share of
//! the weight and took it from the leaf rows.)

use serde::{Deserialize, Serialize};
use crate::purity::{row_probs, top2};

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct RowBalance {
    #[serde(default = "rb_power")] pub power: f64,
    #[serde(default = "rb_lo")] pub lo: f64,
    #[serde(default = "rb_hi")] pub hi: f64,
    #[serde(default = "rb_mean")] pub mean: f64,
    #[serde(default = "rb_commit")] pub commit: f64,
    #[serde(default = "rb_cert")] pub certainty_power: f64,
}
fn rb_power() -> f64 { 0.5 }
fn rb_lo() -> f64 { 0.3 }
fn rb_hi() -> f64 { 8.0 }
fn rb_mean() -> f64 { 1.0 }
fn rb_commit() -> f64 { 0.5 }
fn rb_cert() -> f64 { 1.0 }

/// lut: (n,k) log-prob targets; mask: rows that train (> 0). Returns the weights and a one-line summary.
pub fn balance_weights(lut: &[f32], n: usize, k: usize, mask: &[f32], rb: &RowBalance) -> (Vec<f32>, String) {
    let mut cnt = vec![0usize; k];
    let mut q = vec![0f64; k];
    let mut committed = 0usize;
    for r in 0..n {
        if mask[r] <= 0.0 { continue; }
        row_probs(&lut[r * k..(r + 1) * k], &mut q);
        let (j, b, s) = top2(&q);
        if b >= rb.commit - 1e-6 && b > s + 1e-6 { cnt[j] += 1; committed += 1; }
    }
    let mut nz: Vec<usize> = cnt.iter().cloned().filter(|&c| c > 0).collect();
    nz.sort_unstable();
    let med = if nz.is_empty() { 1.0 } else if nz.len() % 2 == 1 { nz[nz.len() / 2] as f64 } else { (nz[nz.len() / 2 - 1] + nz[nz.len() / 2]) as f64 / 2.0 };
    let lw: Vec<f64> = cnt.iter().map(|&c| if c > 0 { (med / c as f64).powf(rb.power).clamp(rb.lo, rb.hi) } else { rb.hi }).collect();
    let mut w = vec![0f32; n];
    let (mut sum, mut m) = (0f64, 0usize);
    for r in 0..n {
        if mask[r] <= 0.0 { continue; }
        row_probs(&lut[r * k..(r + 1) * k], &mut q);
        let x = q.iter().zip(&lw).map(|(a, b)| a * b).sum::<f64>() * top2(&q).1.powf(rb.certainty_power);
        w[r] = x as f32; sum += x; m += 1;
    }
    let scale = if m > 0 { rb.mean * m as f64 / sum } else { 1.0 };
    for x in w.iter_mut() { *x = (*x as f64 * scale) as f32; }
    let (lo, hi) = w.iter().filter(|&&x| x > 0.0).fold((f32::INFINITY, 0f32), |a, &x| (a.0.min(x), a.1.max(x)));
    (w, format!("row weights computed: {m} rows, {committed} committed to {} leaves (median {med:.0}), power {} clip [{}, {}], certainty^{}, mean {} -> w {lo:.3}..{hi:.3}",
               nz.len(), rb.power, rb.lo, rb.hi, rb.certainty_power, rb.mean))
}
