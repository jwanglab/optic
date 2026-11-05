//! Element-wise, reduction, optimizer and loss kernels for the network's forward and backward. Matrices
//! are row-major; reductions over rows run in a fixed order, so every result is deterministic.

use crate::philox::Philox;
use cuda_device::{DynamicSharedArray, SharedArray, kernel, thread};

/// for i in the grid-stride range 0..n
macro_rules! grid_stride {
    ($i:ident, $n:expr, $body:block) => {{
        let n = $n;
        let step = thread::gridDim_x() as u64 * thread::blockDim_x() as u64;
        let mut $i = thread::blockIdx_x() as u64 * thread::blockDim_x() as u64 + thread::threadIdx_x() as u64;
        while $i < n { $body $i += step; }
    }};
}

/// y[r, c] += b[c]
#[kernel]
pub unsafe fn add_bias_kernel(y: *mut f32, b: *const f32, rows: u64, cols: u64) {
    grid_stride!(i, rows * cols, { unsafe { *y.add(i as usize) += *b.add((i % cols) as usize); } });
}

/// out[c] = sum_r dy[r, c]  (bias gradient), rows summed in order
#[kernel]
pub unsafe fn col_sum_kernel(out: *mut f32, dy: *const f32, rows: u64, cols: u64) {
    let c = thread::blockIdx_x() as u64 * thread::blockDim_x() as u64 + thread::threadIdx_x() as u64;
    if c >= cols { return; }
    unsafe {
        let mut s = 0.0f32;
        let mut r = 0u64;
        while r < rows { s += *dy.add((r * cols + c) as usize); r += 1; }
        *out.add(c as usize) = s;
    }
}

/// y = x * m  (dropout forward and backward: the mask is 0 or 1/(1-p))
#[kernel]
pub unsafe fn mul_kernel(y: *mut f32, x: *const f32, m: *const f32, n: u64) {
    grid_stride!(i, n, { unsafe { *y.add(i as usize) = *x.add(i as usize) * *m.add(i as usize); } });
}

/// y[r, c] = x[r, c] * g[c]
#[kernel]
pub unsafe fn mul_cols_kernel(y: *mut f32, x: *const f32, g: *const f32, rows: u64, cols: u64) {
    grid_stride!(i, rows * cols, { unsafe { *y.add(i as usize) = *x.add(i as usize) * *g.add((i % cols) as usize); } });
}

/// y += a
#[kernel]
pub unsafe fn add_kernel(y: *mut f32, a: *const f32, n: u64) {
    grid_stride!(i, n, { unsafe { *y.add(i as usize) += *a.add(i as usize); } });
}

/// y = x * sigmoid(x)
#[kernel]
pub unsafe fn silu_kernel(y: *mut f32, x: *const f32, n: u64) {
    grid_stride!(i, n, { unsafe { let v = *x.add(i as usize); *y.add(i as usize) = v / (1.0 + (-v).exp()); } });
}

/// dx = dy * s * (1 + x (1 - s)),  s = sigmoid(x)
#[kernel]
pub unsafe fn silu_bwd_kernel(dx: *mut f32, x: *const f32, dy: *const f32, n: u64) {
    grid_stride!(i, n, {
        unsafe {
            let v = *x.add(i as usize);
            let s = 1.0 / (1.0 + (-v).exp());
            *dx.add(i as usize) = *dy.add(i as usize) * s * (1.0 + v * (1.0 - s));
        }
    });
}

/// AdamW in place (decoupled weight decay; bias correction through scale_m = 1/(1-b1^t), scale_v = 1/(1-b2^t))
#[kernel]
pub unsafe fn adamw_inplace_kernel(p: *mut f32, g: *const f32, m: *mut f32, v: *mut f32, n: u64, lr: f32, beta1: f32, beta2: f32,
                                   eps: f32, wd: f32, scale_m: f32, scale_v: f32) {
    grid_stride!(i, n, {
        unsafe {
            let j = i as usize;
            let gi = *g.add(j);
            let mi = beta1 * *m.add(j) + (1.0 - beta1) * gi;
            let vi = beta2 * *v.add(j) + (1.0 - beta2) * gi * gi;
            let mh = mi * scale_m;
            let vh = vi * scale_v;
            *p.add(j) = *p.add(j) * (1.0 - lr * wd) - lr * mh / (vh.sqrt() + eps);
            *m.add(j) = mi;
            *v.add(j) = vi;
        }
    });
}

/// dropout mask: 1/(1-p) where a Philox uniform (seed, element) is >= p, else 0
#[kernel]
pub unsafe fn dropout_mask_kernel(mask: *mut f32, n: u64, p: f32, seed: u64) {
    let keep = 1.0 / (1.0 - p);
    grid_stride!(i, n, { unsafe { let mut st = Philox::new(seed, i); *mask.add(i as usize) = if st.uniform() >= p { keep } else { 0.0 }; } });
}

/// out[r, :] = src[idx[r], :]  (B rows of P bytes): the purity background profile drawn for each row
#[kernel]
pub unsafe fn gather_rows_u8_kernel(out: *mut u8, src: *const u8, idx: *const u32, b: u64, p: u64) {
    grid_stride!(i, b * p, { unsafe { let r = i / p; *out.add(i as usize) = *src.add((*idx.add(r as usize) as u64 * p + (i - r * p)) as usize); } });
}

// ---- loss ---------------------------------------------------------------------------------------------------------
// One block per training row i (B blocks; z holds 2B rows with consistency: deep view 0..B, shallow view B..2B; else B). For row i:
//   cls_i  = w_i * [ T^2 KL(softmax(t/T) || softmax(z/T)) - lam * sum_levels sum_n qm_n log(max(pm_n, 1e-8)) ]
//            q = exp(t) / max(sum exp(t), 1e-8), pl = softmax(z), qm = q M^T, pm = pl M^T  (M: all levels' 0/1 rows)
//   cons_i = cw * lab_i * KL(softmax(z_deep) || softmax(z_shallow))      (the deep view is a fixed target here)
// Batch losses are the means over i. dz gets d(cls_mean + cons_mean)/dz for both views.

const LB: u32 = 256;

#[inline(always)]
unsafe fn block_sum(v: f32, sh: *mut f32) -> f32 {
    let t = thread::threadIdx_x();
    unsafe {
        *sh.add(t as usize) = v; thread::sync_threads();
        let mut s = LB / 2;
        while s > 0 { if t < s { *sh.add(t as usize) += *sh.add((t + s) as usize); } thread::sync_threads(); s >>= 1; }
        let r = *sh; thread::sync_threads();
        r
    }
}

#[inline(always)]
unsafe fn block_max(v: f32, sh: *mut f32) -> f32 {
    let t = thread::threadIdx_x();
    unsafe {
        *sh.add(t as usize) = v; thread::sync_threads();
        let mut s = LB / 2;
        while s > 0 { if t < s { *sh.add(t as usize) = (*sh.add(t as usize)).max(*sh.add((t + s) as usize)); } thread::sync_threads(); s >>= 1; }
        let r = *sh; thread::sync_threads();
        r
    }
}

/// out[c] = softmax(x * scale)[c]; returns the log of the normaliser (logsumexp of x*scale)
#[inline(always)]
unsafe fn softmax_row(x: *const f32, scale: f32, out: *mut f32, k: u32, sh: *mut f32) -> f32 {
    unsafe {
        let mut mx = f32::NEG_INFINITY;
        let mut c = thread::threadIdx_x();
        while c < k { mx = mx.max(*x.add(c as usize) * scale); c += LB; }
        mx = block_max(mx, sh);
        let mut s = 0.0f32;
        let mut c = thread::threadIdx_x();
        while c < k { let e = (*x.add(c as usize) * scale - mx).exp(); *out.add(c as usize) = e; s += e; c += LB; }
        s = block_sum(s, sh);
        let mut c = thread::threadIdx_x();
        while c < k { *out.add(c as usize) /= s; c += LB; }
        thread::sync_threads();
        mx + s.ln()
    }
}

/// dynamic shared memory: 6*k + n_nodes floats
#[kernel]
pub unsafe fn loss_kernel(dz: *mut f32, cls_row: *mut f32, cons_row: *mut f32, z: *const f32, t: *const f32, w: *const f32,
                          lab: *const f32, mm: *const f32, n_nodes: u32, b: u32, k: u32, temp: f32, lam: f32, cw: f32, has_cons: u32) {
    static mut SH: SharedArray<f32, 256> = SharedArray::UNINIT;
    unsafe {
        let sh = SH.as_mut_ptr();
        let s: *mut f32 = DynamicSharedArray::<f32>::get();
        let ku = k as usize;
        let (st, tp, pl, q, g, ss, gp) = (s, s.add(ku), s.add(2 * ku), s.add(3 * ku), s.add(4 * ku), s.add(5 * ku), s.add(6 * ku));
        let tid = thread::threadIdx_x();
        let i = thread::blockIdx_x();
        let zr = z.add(i as usize * ku);
        let zsr = z.add((b + i) as usize * ku);
        let tr = t.add(i as usize * ku);

        // KD at temperature T
        let lse_t = softmax_row(zr, 1.0 / temp, st, k, sh);
        softmax_row(tr, 1.0 / temp, tp, k, sh);
        let (mut kd, mut tsum) = (0.0f32, 0.0f32);
        let mut c = tid;
        while c < k {
            let cu = c as usize;
            kd += *tp.add(cu) * ((*tp.add(cu) + 1e-12).ln() - (*zr.add(cu) / temp - lse_t));
            tsum += *tp.add(cu);
            c += LB;
        }
        kd = block_sum(kd, sh) * temp * temp; tsum = block_sum(tsum, sh);

        // hierarchy
        softmax_row(zr, 1.0, pl, k, sh);
        let mut qs = 0.0f32;
        let mut c = tid;
        while c < k { *q.add(c as usize) = (*tr.add(c as usize)).exp(); qs += *q.add(c as usize); c += LB; }
        qs = block_sum(qs, sh).max(1e-8);
        let mut c = tid;
        while c < k { *q.add(c as usize) /= qs; c += LB; }
        thread::sync_threads();
        let mut h = 0.0f32;
        let mut n = tid;
        while n < n_nodes {
            let mn = mm.add(n as usize * ku);
            let (mut qm, mut pm) = (0.0f32, 0.0f32);
            let mut c = 0u32;
            while c < k { let mv = *mn.add(c as usize); qm += mv * *q.add(c as usize); pm += mv * *pl.add(c as usize); c += 1; }
            h -= lam * qm * pm.max(1e-8).ln();
            *gp.add(n as usize) = if pm > 1e-8 { -lam * qm / pm } else { 0.0 };
            n += LB;
        }
        h = block_sum(h, sh);
        thread::sync_threads();
        // g_c = sum_n M[n, c] gp_n ; dz_h = pl * (g - sum_c pl_c g_c)
        let mut pg = 0.0f32;
        let mut c = tid;
        while c < k {
            let mut a = 0.0f32;
            let mut n = 0u32;
            while n < n_nodes { a += *mm.add(n as usize * ku + c as usize) * *gp.add(n as usize); n += 1; }
            *g.add(c as usize) = a; pg += *pl.add(c as usize) * a;
            c += LB;
        }
        pg = block_sum(pg, sh);
        let wi = *w.add(i as usize) / b as f32;
        let mut c = tid;
        while c < k {
            let cu = c as usize;
            *dz.add(i as usize * ku + cu) = wi * (temp * (*st.add(cu) * tsum - *tp.add(cu)) + *pl.add(cu) * (*g.add(cu) - pg));
            c += LB;
        }
        if tid == 0 { *cls_row.add(i as usize) = *w.add(i as usize) * (kd + h); }
        if has_cons == 0 { if tid == 0 { *cons_row.add(i as usize) = 0.0; } return; }

        // consistency: target softmax(z_deep) (pl), prediction softmax(z_shallow)
        let lse_s = softmax_row(zsr, 1.0, ss, k, sh);
        let (mut kc, mut psum) = (0.0f32, 0.0f32);
        let mut c = tid;
        while c < k {
            let cu = c as usize;
            kc += *pl.add(cu) * ((*pl.add(cu) + 1e-12).ln() - (*zsr.add(cu) - lse_s));
            psum += *pl.add(cu);
            c += LB;
        }
        kc = block_sum(kc, sh); psum = block_sum(psum, sh);
        let ci = cw * *lab.add(i as usize) / b as f32;
        let mut c = tid;
        while c < k { let cu = c as usize; *dz.add((b + i) as usize * ku + cu) = ci * (*ss.add(cu) * psum - *pl.add(cu)); c += LB; }
        if tid == 0 { *cons_row.add(i as usize) = cw * *lab.add(i as usize) * kc; }
    }
}

/// out[0] += sum(cls_row)/B, out[1] += sum(cons_row)/B  (single thread: fixed order; out accumulates over steps)
#[kernel]
pub unsafe fn loss_sum_kernel(out: *mut f32, cls_row: *const f32, cons_row: *const f32, b: u32) {
    if thread::blockIdx_x() != 0 || thread::threadIdx_x() != 0 { return; }
    unsafe {
        let (mut a, mut c) = (0.0f32, 0.0f32);
        let mut i = 0u32;
        while i < b { a += *cls_row.add(i as usize); c += *cons_row.add(i as usize); i += 1; }
        *out += a / b as f32;
        *out.add(1) += c / b as f32;
    }
}
