//! The network's arithmetic on row-major f32 batches (n rows x d columns), with the backward passes federated
//! training needs. Weights are `[out, in]`.

use crate::st::Tensor;

/// Dot product with 8 independent accumulators, so the compiler can vectorise it.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) { for k in 0..8 { acc[k] += x[k] * y[k]; } }
    let mut s = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    for (x, y) in ra.iter().zip(rb) { s += x * y; }
    s
}

/// y = x W^T + b for x (n x in), W (out x in).
pub fn linear(x: &[f32], n: usize, w: &Tensor, b: Option<&Tensor>) -> Vec<f32> {
    let (o, i) = (w.dim(0), w.dim(1));
    debug_assert_eq!(x.len(), n * i);
    let mut y = vec![0f32; n * o];
    for r in 0..n {
        let xr = &x[r * i..(r + 1) * i];
        for (c, yv) in y[r * o..(r + 1) * o].iter_mut().enumerate() {
            *yv = dot(xr, &w.data[c * i..(c + 1) * i]) + b.map_or(0.0, |b| b.data[c]);
        }
    }
    y
}

/// Backward of `linear`: returns dx (n x in) when asked, and accumulates dW (out x in) and db (out).
pub fn linear_backward(x: &[f32], n: usize, w: &Tensor, dy: &[f32], dw: &mut [f32], db: Option<&mut [f32]>, want_dx: bool) -> Option<Vec<f32>> {
    let (o, i) = (w.dim(0), w.dim(1));
    for r in 0..n {
        let (xr, dyr) = (&x[r * i..(r + 1) * i], &dy[r * o..(r + 1) * o]);
        for (c, &g) in dyr.iter().enumerate() {
            if g == 0.0 { continue; }
            for (d, &xv) in dw[c * i..(c + 1) * i].iter_mut().zip(xr) { *d += g * xv; }
        }
    }
    if let Some(db) = db { for r in 0..n { for (d, &g) in db.iter_mut().zip(&dy[r * o..(r + 1) * o]) { *d += g; } } }
    want_dx.then(|| {
        let mut dx = vec![0f32; n * i];
        for r in 0..n {
            let dxr = &mut dx[r * i..(r + 1) * i];
            for (c, &g) in dy[r * o..(r + 1) * o].iter().enumerate() {
                if g == 0.0 { continue; }
                for (d, &wv) in dxr.iter_mut().zip(&w.data[c * i..(c + 1) * i]) { *d += g * wv; }
            }
        }
        dx
    })
}

/// The first layer on a ternary input (n x probes, values -1 / 0 / +1): z = x W^T, summing only the covered probes.
pub fn project_ternary(x: &[f32], n: usize, w: &Tensor) -> Vec<f32> {
    let (o, p) = (w.dim(0), w.dim(1));
    let mut z = vec![0f32; n * o];
    for r in 0..n {
        let xr = &x[r * p..(r + 1) * p];
        let plus: Vec<usize> = (0..p).filter(|&j| xr[j] > 0.0).collect();
        let minus: Vec<usize> = (0..p).filter(|&j| xr[j] < 0.0).collect();
        for (c, zv) in z[r * o..(r + 1) * o].iter_mut().enumerate() {
            let row = &w.data[c * p..(c + 1) * p];
            let (mut a, mut b) = (0f64, 0f64);
            for &j in &plus { a += row[j] as f64; }
            for &j in &minus { b += row[j] as f64; }
            *zv = (a - b) as f32;
        }
    }
    z
}

/// LayerNorm over each row (eps 1e-5, biased variance). Returns y, and (x̂, 1/σ) for the backward pass.
pub fn layer_norm(x: &[f32], n: usize, w: &Tensor, b: &Tensor) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let d = w.len();
    let (mut y, mut xhat, mut inv) = (vec![0f32; n * d], vec![0f32; n * d], vec![0f32; n]);
    for r in 0..n {
        let xr = &x[r * d..(r + 1) * d];
        let mean = xr.iter().sum::<f32>() / d as f32;
        let var = xr.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let s = 1.0 / (var + 1e-5).sqrt();
        inv[r] = s;
        for k in 0..d {
            let h = (xr[k] - mean) * s;
            xhat[r * d + k] = h;
            y[r * d + k] = h * w.data[k] + b.data[k];
        }
    }
    (y, xhat, inv)
}

/// Backward of `layer_norm`: returns dx, accumulates dw and db.
pub fn layer_norm_backward(xhat: &[f32], inv: &[f32], w: &Tensor, dy: &[f32], dw: &mut [f32], db: &mut [f32]) -> Vec<f32> {
    let d = w.len();
    let n = inv.len();
    let mut dx = vec![0f32; n * d];
    for r in 0..n {
        let (xh, g) = (&xhat[r * d..(r + 1) * d], &dy[r * d..(r + 1) * d]);
        let (mut m1, mut m2) = (0f32, 0f32);
        for k in 0..d {
            dw[k] += g[k] * xh[k];
            db[k] += g[k];
            let gx = g[k] * w.data[k];
            m1 += gx;
            m2 += gx * xh[k];
        }
        let (m1, m2) = (m1 / d as f32, m2 / d as f32);
        for k in 0..d { dx[r * d + k] = inv[r] * (g[k] * w.data[k] - m1 - xh[k] * m2); }
    }
    dx
}

#[inline]
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

pub fn silu(x: &[f32]) -> Vec<f32> { x.iter().map(|&v| v * sigmoid(v)).collect() }

/// dL/dx of silu at x, given dL/dy.
pub fn silu_backward(x: &[f32], dy: &[f32]) -> Vec<f32> {
    x.iter().zip(dy).map(|(&v, &g)| { let s = sigmoid(v); g * s * (1.0 + v * (1.0 - s)) }).collect()
}

/// Row-wise softmax of x / t.
pub fn softmax(x: &[f32], n: usize, t: f32) -> Vec<f32> {
    let k = x.len() / n.max(1);
    let mut out = vec![0f32; x.len()];
    for r in 0..n {
        let row = &x[r * k..(r + 1) * k];
        let m = row.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v / t));
        let e: Vec<f32> = row.iter().map(|&v| (v / t - m).exp()).collect();
        let s: f32 = e.iter().sum();
        for (o, v) in out[r * k..(r + 1) * k].iter_mut().zip(e) { *o = v / s; }
    }
    out
}

/// log Σ exp of one row, stably.
pub fn logsumexp(row: &[f32]) -> f32 {
    let m = row.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
    m + row.iter().map(|&v| (v - m).exp()).sum::<f32>().ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(shape: Vec<usize>, seed: u32) -> Tensor {
        let n = shape.iter().product();
        let mut s = seed;
        let data = (0..n).map(|_| { s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223); (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5 }).collect();
        Tensor::new(shape, data)
    }

    /// Every backward pass against central finite differences of a scalar loss L = Σ y ⊙ g.
    #[test]
    fn gradients_match_finite_differences() {
        let (n, i, o) = (3, 7, 5);
        let (x, w, b, g) = (t(vec![n, i], 1), t(vec![o, i], 2), t(vec![o], 3), t(vec![n, o], 4));
        let lw = Tensor::new(vec![o], t(vec![o], 5).data.iter().map(|v| v + 1.0).collect());
        let lb = t(vec![o], 6);
        let loss = |x: &[f32], w: &Tensor, lw: &Tensor| -> f64 {
            let y = linear(x, n, w, Some(&b));
            let (a, _, _) = layer_norm(&y, n, lw, &lb);
            silu(&a).iter().zip(&g.data).map(|(p, q)| (*p as f64) * (*q as f64)).sum()
        };
        // analytic
        let y = linear(&x.data, n, &w, Some(&b));
        let (a, xh, inv) = layer_norm(&y, n, &lw, &lb);
        let da = silu_backward(&a, &g.data);
        let (mut dlw, mut dlb) = (vec![0f32; o], vec![0f32; o]);
        let dy = layer_norm_backward(&xh, &inv, &lw, &da, &mut dlw, &mut dlb);
        let (mut dw, mut db) = (vec![0f32; o * i], vec![0f32; o]);
        let dx = linear_backward(&x.data, n, &w, &dy, &mut dw, Some(&mut db), true).unwrap();
        let eps = 1e-3f32;
        let check = |an: f32, f: &dyn Fn(f32) -> f64, what: &str| {
            let num = ((f(eps) - f(-eps)) / (2.0 * eps as f64)) as f32;
            assert!((an - num).abs() < 2e-3 * (1.0 + num.abs()), "{what}: analytic {an} numeric {num}");
        };
        for k in [0, 4, 11, 20] {
            check(dx[k], &|e| { let mut xx = x.data.clone(); xx[k] += e; loss(&xx, &w, &lw) }, "dx");
            check(dw[k], &|e| { let mut ww = w.clone(); ww.data[k] += e; loss(&x.data, &ww, &lw) }, "dW");
        }
        for k in 0..o {
            check(dlw[k], &|e| { let mut l = lw.clone(); l.data[k] += e; loss(&x.data, &w, &l) }, "dLNw");
        }
        // db: the LayerNorm removes a constant shift, so check it through the full loss as well
        check(db[1], &|e| {
            let mut bb = b.clone(); bb.data[1] += e;
            let y = linear(&x.data, n, &w, Some(&bb));
            let (a, _, _) = layer_norm(&y, n, &lw, &lb);
            silu(&a).iter().zip(&g.data).map(|(p, q)| (*p as f64) * (*q as f64)).sum()
        }, "db");
    }

    #[test]
    fn ternary_projection_is_a_matmul() {
        let (n, p, o) = (2, 50, 4);
        let w = t(vec![o, p], 9);
        let x: Vec<f32> = (0..n * p).map(|k| [0.0, 1.0, -1.0][k % 3]).collect();
        let a = project_ternary(&x, n, &w);
        let b = linear(&x, n, &w, None);
        for (u, v) in a.iter().zip(&b) { assert!((u - v).abs() < 1e-5); }
    }
}
