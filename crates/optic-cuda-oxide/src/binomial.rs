//! Binomial(n, p) sampling: inversion for n*p < 10, BTRS (Hormann 1993, as in PyTorch)
//! otherwise, in double precision for the acceptance tests.

use crate::philox::Philox;

/// Inversion through geometric skips over runs of failures.
#[inline(always)]
pub fn inversion(n: u32, p: f64, rng: &mut Philox) -> u32 {
    let log_q = (1.0 - p).ln();
    let mut x = 0u32;
    let mut sum = 0.0f64;
    loop {
        let u = rng.uniform() as f64;
        let geom = (u.ln() / log_q).ceil();
        sum += geom;
        if sum > n as f64 { break; }
        x += 1;
    }
    x
}

/// BTRS: transformed rejection with squeeze.
#[inline(always)]
pub fn btrs(n: u32, p: f64, rng: &mut Philox) -> u32 {
    let fnn = n as f64;
    let q = 1.0 - p;
    let npq = fnn * p * q;
    let stddev = npq.sqrt();
    let b = 1.15 + 2.53 * stddev;
    let a = -0.0873 + 0.0248 * b + 0.01 * p;
    let c = fnn * p + 0.5;
    let alpha = (2.83 + 5.1 / b) * stddev;
    let r = p / q;
    let m = ((fnn + 1.0) * p).floor() as i32;   // mode
    loop {
        let u = rng.uniform() as f64 - 0.5;
        let mut v = rng.uniform() as f64;
        let us = 0.5 - u.abs();
        let k = ((2.0 * a / us + b) * u + c).floor() as i32;
        if k < 0 || k > n as i32 { continue; }
        v = v * alpha / (a / (us * us) + b);
        let km = (k as f64 - m as f64).abs();
        if km <= 15.0 {
            // P(k)/P(m) as a product of PMF ratios
            let mut f = 1.0f64;
            if m < k {
                let mut i = m + 1;
                while i <= k { f *= (fnn - i as f64 + 1.0) * r / i as f64; i += 1; }
            } else if m > k {
                let mut i = k + 1;
                while i <= m { f /= (fnn - i as f64 + 1.0) * r / i as f64; i += 1; }
            }
            if v <= f { return k as u32; }
        } else {
            // log-space squeeze, then the exact log-PMF ratio
            v = v.ln();
            let rho = (km / npq) * (((km / 3.0 + 0.625) * km + 1.0 / 6.0) / npq + 0.5);
            let t = -km * km / (2.0 * npq);
            if v < t - rho { return k as u32; }
            if v > t + rho { continue; }
            let log_ratio = libm::lgamma((m + 1) as f64) - libm::lgamma((k + 1) as f64)
                + libm::lgamma(fnn - m as f64 + 1.0) - libm::lgamma(fnn - k as f64 + 1.0)
                + (k - m) as f64 * r.ln();
            if v <= log_ratio { return k as u32; }
        }
    }
}
