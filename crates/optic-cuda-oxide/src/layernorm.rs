//! Fused LayerNorm forward and backward over the last dimension: one 256-thread block per row, two-pass biased
//! variance with eps inside the sqrt. x is (B, H) row-major, gamma/beta are (H,), stats is (2, B) = [mean, rstd].

use cuda_device::{SharedArray, kernel, thread};

const LN_BLK: u32 = 256;

/// Tree sum over the block in fixed order; every thread gets the total.
#[inline(always)]
unsafe fn ln_block_sum(v: f32, sh: *mut f32) -> f32 {
    let t = thread::threadIdx_x();
    unsafe {
        *sh.add(t as usize) = v;
        thread::sync_threads();
        let mut s = LN_BLK / 2;
        while s > 0 {
            if t < s { *sh.add(t as usize) += *sh.add((t + s) as usize); }
            thread::sync_threads();
            s >>= 1;
        }
        let r = *sh;
        thread::sync_threads();
        r
    }
}

/// (mean, 1/sqrt(var + eps)) of one row, two-pass.
#[inline(always)]
unsafe fn ln_row_stats(xr: *const f32, h: u64, eps: f32, sh: *mut f32) -> (f32, f32) {
    unsafe {
        let mut s = 0.0f32;
        let mut i = thread::threadIdx_x() as u64;
        while i < h { s += *xr.add(i as usize); i += LN_BLK as u64; }
        let mean = ln_block_sum(s, sh) / h as f32;
        let mut q = 0.0f32;
        let mut i = thread::threadIdx_x() as u64;
        while i < h { let d = *xr.add(i as usize) - mean; q += d * d; i += LN_BLK as u64; }
        let var = ln_block_sum(q, sh) / h as f32;
        (mean, 1.0 / (var + eps).sqrt())
    }
}

#[kernel]
pub unsafe fn ln_fwd_kernel(out: *mut f32, x: *const f32, g: *const f32, b: *const f32, h: u64, eps: f32) {
    static mut SH: SharedArray<f32, 256> = SharedArray::UNINIT;
    unsafe {
        let row = thread::blockIdx_x() as u64;
        let xr = x.add((row * h) as usize);
        let orow = out.add((row * h) as usize);
        let (mean, rstd) = ln_row_stats(xr, h, eps, SH.as_mut_ptr());
        let mut i = thread::threadIdx_x() as u64;
        while i < h {
            *orow.add(i as usize) = (*xr.add(i as usize) - mean) * rstd * *g.add(i as usize) + *b.add(i as usize);
            i += LN_BLK as u64;
        }
    }
}

#[kernel]
pub unsafe fn ln_stats_kernel(stats: *mut f32, x: *const f32, b: u64, h: u64, eps: f32) {
    static mut SH: SharedArray<f32, 256> = SharedArray::UNINIT;
    unsafe {
        let row = thread::blockIdx_x() as u64;
        let (mean, rstd) = ln_row_stats(x.add((row * h) as usize), h, eps, SH.as_mut_ptr());
        if thread::threadIdx_x() == 0 { *stats.add(row as usize) = mean; *stats.add((b + row) as usize) = rstd; }
    }
}

/// dx = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat)),  dxhat = dy * gamma
#[kernel]
pub unsafe fn ln_dx_kernel(dx: *mut f32, dxhat: *const f32, x: *const f32, stats: *const f32, b: u64, h: u64) {
    static mut SH: SharedArray<f32, 256> = SharedArray::UNINIT;
    unsafe {
        let row = thread::blockIdx_x() as u64;
        let mean = *stats.add(row as usize);
        let rstd = *stats.add((b + row) as usize);
        let dr = dxhat.add((row * h) as usize);
        let xr = x.add((row * h) as usize);
        let orow = dx.add((row * h) as usize);
        let (mut s1, mut s2) = (0.0f32, 0.0f32);
        let mut i = thread::threadIdx_x() as u64;
        while i < h {
            let d = *dr.add(i as usize);
            s1 += d;
            s2 += d * (*xr.add(i as usize) - mean) * rstd;
            i += LN_BLK as u64;
        }
        let m1 = ln_block_sum(s1, SH.as_mut_ptr()) / h as f32;
        let m2 = ln_block_sum(s2, SH.as_mut_ptr()) / h as f32;
        let mut i = thread::threadIdx_x() as u64;
        while i < h {
            let xhat = (*xr.add(i as usize) - mean) * rstd;
            *orow.add(i as usize) = rstd * (*dr.add(i as usize) - m1 - xhat * m2);
            i += LN_BLK as u64;
        }
    }
}

/// out is (2, H): row 0 = d gamma = sum_b dy * xhat, row 1 = d beta = sum_b dy
#[kernel]
pub unsafe fn ln_dwdb_kernel(out: *mut f32, dy: *const f32, x: *const f32, stats: *const f32, b: u64, h: u64) {
    let col = thread::blockIdx_x() as u64 * thread::blockDim_x() as u64 + thread::threadIdx_x() as u64;
    if col >= h { return; }
    unsafe {
        let (mut dg, mut db) = (0.0f32, 0.0f32);
        let mut r = 0u64;
        while r < b {
            let d = *dy.add((r * h + col) as usize);
            dg += d * (*x.add((r * h + col) as usize) - *stats.add(r as usize)) * *stats.add((b + r) as usize);
            db += d;
            r += 1;
        }
        *out.add(col as usize) = dg;
        *out.add((h + col) as usize) = db;
    }
}
