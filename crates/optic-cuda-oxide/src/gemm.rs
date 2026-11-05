//! FP32 matrix products instead of cuBLAS - not quite identical output
//! Deterministic: every output is one thread's sequential sum over k, and split-k partial
//! sums are added in a fixed order by `gemm_reduce_kernel` (no atomics).
//!
//! One kernel serves every layout through strides: C (m x n, row-major) = sum_p A(i, p) B(p, j), with
//! A(i, p) = a[i*sai + p*sap] and B(p, j) = b[p*sbp + j*sbj]. Each operand is contiguous along either its k axis or
//! its other axis; the tile loads pick the coalesced order from which stride is 1. (Global loads stay scalar: the
//! first layer's rows are 353,233 floats, so they are not 16-byte aligned.)
//!
//! Blocks of 256 threads compute a 128 x 128 tile of C. The 8 warps split it 2 x 4 into 64 x 32 warp tiles; within a
//! warp, lanes form 8 x 4 and each lane owns two 4-row by two 4-column fragments (8 x 8 outputs), so every
//! shared-memory read is a 128-bit load that the lanes of a warp serve without bank conflicts. K advances 16 at a
//! time through double-buffered shared memory: the next tile is copied global -> shared by cp.async while the current
//! one is multiplied. With gridDim.z > 1 (split-k), block z sums only
//! k in [z*kchunk, (z+1)*kchunk) and writes its partial tile to c + z*m*n, a workspace that `gemm_reduce_kernel` folds
//! into the output.

use cuda_device::async_copy::{cp_async_ca_zfill_4, cp_async_commit_group, cp_async_wait_all, cp_async_wait_group};
use cuda_device::{SharedArray, kernel, thread};

const BM: usize = 128;
const BN: usize = 128;
const BK: usize = 16;
/// shared row stride (floats): 4 words of padding spread a k-contiguous load's stores over the banks and keep every
/// fragment 16-byte aligned
const LD: usize = BM + 4;
const BUF: usize = BK * LD;
/// values of each tile a thread stages: 128 x 16 / 256 threads
const Q: usize = 8;

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct F4([f32; 4]);

#[inline(always)]
unsafe fn ld4(p: *const f32) -> [f32; 4] { unsafe { (*(p as *const F4)).0 } }

/// Element q (0..Q) a thread stages from an operand sits at tile-relative (row, k) = (r0 + q*rq, p0 + q*pq): with the
/// operand contiguous along k, threads walk k fastest (r0 = t/16, rq = 16, pq = 0); otherwise they walk the rows
/// (r0 = t%128, rq = 0, pq = 2). So its global and shared offsets are a base plus q times a fixed step, in 32 bits.
#[derive(Clone, Copy)]
struct Plan { r0: u32, rq: u32, p0: u32, pq: u32, g0: u32, gq: u32, gp: u32, s0: u32, sq: u32 }

#[inline(always)]
fn plan(t: u32, k_contiguous: bool, s_row: u32, s_k: u32) -> Plan {
    let (r0, rq, p0, pq) = if k_contiguous { (t / BK as u32, 256 / BK as u32, t % BK as u32, 0) } else { (t % BM as u32, 0, t / BM as u32, 256 / BM as u32) };
    Plan { r0, rq, p0, pq, g0: r0 * s_row + p0 * s_k, gq: rq * s_row + pq * s_k, gp: s_k, s0: p0 * LD as u32 + r0, sq: pq * LD as u32 + rq }
}

#[kernel]
pub unsafe fn gemm_kernel(c: *mut f32, a: *const f32, b: *const f32, m: u64, n: u64, k: u64,
                          sai: u64, sap: u64, sbp: u64, sbj: u64, kchunk: u64, beta: f32) {
    static mut AS: SharedArray<f32, { 2 * BUF }, 16> = SharedArray::UNINIT;   // 2 x [p][i]
    static mut BS: SharedArray<f32, { 2 * BUF }, 16> = SharedArray::UNINIT;   // 2 x [p][j]
    // 32-bit offsets: optic-train (gpu.rs) checks every operand and the output fit
    let (m, n, k) = (m as u32, n as u32, k as u32);
    let t = thread::threadIdx_x();
    let (w, l) = (t / 32, t % 32);
    let rbase = ((w / 4) * 64 + (l / 4) * 4) as usize;   // this lane's rows: rbase + 0..4 and rbase + 32 + 0..4
    let cbase = ((w % 4) * 32 + (l % 4) * 4) as usize;   // its columns: cbase + 0..4 and cbase + 16 + 0..4
    let (bm0, bn0) = (thread::blockIdx_y() * BM as u32, thread::blockIdx_x() * BN as u32);
    let z = thread::blockIdx_z();
    let k0 = z * kchunk as u32;
    let k1 = if k0 + kchunk as u32 > k { k } else { k0 + kchunk as u32 };
    let ntiles = if k1 > k0 { (k1 - k0).div_ceil(BK as u32) } else { 0 };
    let pa_ = plan(t, sap == 1, sai as u32, sap as u32);
    let pb_ = plan(t, sbp == 1, sbj as u32, sbp as u32);
    // operand origins at this block's first row / column
    let (a, b) = unsafe { (a.add((bm0 * sai as u32) as usize), b.add((bn0 * sbj as u32) as usize)) };
    let (ma, nb) = (m - bm0, n - bn0);   // rows / columns left in the operands from this block's origin
    let mut acc = [0.0f32; 64];
    unsafe {
        let (wa, wb) = (AS.as_mut_ptr(), BS.as_mut_ptr());
        // queue the copies of the tile starting at k = p0 into shared buffer `buf` (cp.async: global -> shared without
        // passing through registers; an out-of-range value copies 0 bytes and is zero-filled)
        let issue = |p0: u32, buf: u32| {
            let (ga, gb) = (pa_.g0 + p0 * pa_.gp, pb_.g0 + p0 * pb_.gp);
            let mut q = 0u32;
            #[unroll]
            while q < Q as u32 {
                let ok = pa_.r0 + q * pa_.rq < ma && p0 + pa_.p0 + q * pa_.pq < k1;
                let src = if ok { a.add((ga + q * pa_.gq) as usize) } else { a };
                cp_async_ca_zfill_4(wa.add((buf + pa_.s0 + q * pa_.sq) as usize) as *mut u32, src as *const u8, if ok { 4 } else { 0 });
                let ok = pb_.r0 + q * pb_.rq < nb && p0 + pb_.p0 + q * pb_.pq < k1;
                let src = if ok { b.add((gb + q * pb_.gq) as usize) } else { b };
                cp_async_ca_zfill_4(wb.add((buf + pb_.s0 + q * pb_.sq) as usize) as *mut u32, src as *const u8, if ok { 4 } else { 0 });
                q += 1;
            }
            cp_async_commit_group();
        };
        if ntiles > 0 { issue(k0, 0); }
        let (pa, pb) = (wa as *const f32, wb as *const f32);
        let mut cur = 0u32;
        let mut tile = 0u32;
        while tile < ntiles {
            // the next tile's copies go out before this one is waited on, so they overlap its multiply
            if tile + 1 < ntiles { issue(k0 + (tile + 1) * BK as u32, (cur ^ 1) * BUF as u32); cp_async_wait_group(1); } else { cp_async_wait_all(); }
            thread::sync_threads();
            let base = (cur * BUF as u32) as usize;
            let mut p = 0usize;
            #[unroll]
            while p < BK {
                let off = base + p * LD;
                let (a0, a1) = (ld4(pa.add(off + rbase)), ld4(pa.add(off + rbase + 32)));
                let (b0, b1) = (ld4(pb.add(off + cbase)), ld4(pb.add(off + cbase + 16)));
                let ra = [a0[0], a0[1], a0[2], a0[3], a1[0], a1[1], a1[2], a1[3]];
                let rb = [b0[0], b0[1], b0[2], b0[3], b1[0], b1[1], b1[2], b1[3]];
                let mut r = 0usize;
                #[unroll]
                while r < 8 {
                    let mut s = 0usize;
                    #[unroll]
                    while s < 8 { acc[r * 8 + s] += ra[r] * rb[s]; s += 1; }
                    r += 1;
                }
                p += 1;
            }
            thread::sync_threads(); // every warp is done with this buffer before the copy after next overwrites it
            cur ^= 1;
            tile += 1;
        }
        let split = thread::gridDim_z() > 1;
        let out = if split { c.add(z as usize * (m * n) as usize) } else { c };
        let mut r = 0usize;
        #[unroll]
        while r < 8 {
            let i = bm0 as usize + rbase + (r % 4) + 32 * (r / 4);
            let mut s = 0usize;
            #[unroll]
            while s < 8 {
                let j = bn0 as usize + cbase + (s % 4) + 16 * (s / 4);
                if i < m as usize && j < n as usize {
                    let o = out.add(i * n as usize + j);
                    *o = if split || beta == 0.0 { acc[r * 8 + s] } else { acc[r * 8 + s] + beta * *o };
                }
                s += 1;
            }
            r += 1;
        }
    }
}

/// c[i] = sum_z part[z*mn + i] (z in order) + beta * c[i]
#[kernel]
pub unsafe fn gemm_reduce_kernel(c: *mut f32, part: *const f32, mn: u64, splits: u64, beta: f32) {
    let step = thread::gridDim_x() as u64 * thread::blockDim_x() as u64;
    let mut i = thread::blockIdx_x() as u64 * thread::blockDim_x() as u64 + thread::threadIdx_x() as u64;
    while i < mn {
        unsafe {
            let mut s = *part.add(i as usize);
            let mut z = 1u64;
            while z < splits { s += *part.add((z * mn + i) as usize); z += 1; }
            let o = c.add(i as usize);
            *o = if beta == 0.0 { s } else { s + beta * *o };
        }
        i += step;
    }
}
