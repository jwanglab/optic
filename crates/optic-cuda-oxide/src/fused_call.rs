//! Fused input construction for training (one launch per batch, one Philox stream per element):
//!   uint8 beta code -> coverage/probe-dropout keep -> tumour-purity mix -> (sequenced rows) read depth d from the
//!   row's inverse-CDF table -> k ~ Binomial(d, beta') with call error -> majority call in {-1, 0, +1}.
//! Calls match inference (optic probe.rs: fraction > 0.5 -> +1, < 0.5 -> -1, exactly 0.5 -> 0 = uncovered): a read tie
//! (2k == d) and an array beta of exactly 0.5 give 0. Codes 0..250 decode to (c+0.5)/250; anything above 250 is missing (0).
//! rowp layout (per row, 4 floats): [sequenced (0/1), depth table index, keep_prob, purity]
//! purity < 1: `codes` has 2*B rows and row B+i holds the background profile drawn for row i, so
//!   beta <- purity*beta_tumour + (1-purity)*beta_background before the read draw (each read comes from one cell).
//! lut: n_tables x 4096 inverse-CDF read-depth tables.

use crate::binomial;
use crate::philox::Philox;
use cuda_device::{kernel, thread};

#[kernel]
pub unsafe fn fused_call_kernel(out: *mut f32, codes: *const u8, rowp: *const f32, lut: *const f32, b: u32, p: u32, seed: u64, eps: f32) {
    let idx = thread::blockIdx_x() as u64 * thread::blockDim_x() as u64 + thread::threadIdx_x() as u64;
    let total = b as u64 * p as u64;
    if idx >= total { return; }
    unsafe {
        let o = out.add(idx as usize);
        let i = (idx / p as u64) as u32;
        let code = *codes.add(idx as usize);
        if code > 250 { *o = 0.0; return; }

        let rp = rowp.add(i as usize * 4);
        let sequenced = *rp > 0.5;
        let table = (*rp.add(1) + 0.5) as u32;
        let keep_p = *rp.add(2);
        let purity = *rp.add(3);

        let mut rng = Philox::new(seed, idx);
        // keep decided first so dropped probes cost nothing
        if rng.uniform() > keep_p { *o = 0.0; return; }

        let mut beta = (code as f32 + 0.5) / 250.0;
        if purity < 1.0 {
            let bg = *codes.add((b as u64 + i as u64) as usize * p as usize + (idx - i as u64 * p as u64) as usize);
            if bg <= 250 { beta = purity * beta + (1.0 - purity) * ((bg as f32 + 0.5) / 250.0); }   // background missing -> keep tumour beta
        }
        let x = if sequenced {
            let d = *lut.add(table as usize * 4096 + (((rng.uniform() * 4096.0) as u32) & 4095) as usize) as u32;
            let bp = beta * (1.0 - eps) + (1.0 - beta) * eps;
            let flipped = bp > 0.5;
            let q = if flipped { 1.0 - bp as f64 } else { bp as f64 };
            let mu = d as f64 * q;
            let mut k = if mu < 1e-10 { 0 } else if mu < 10.0 { binomial::inversion(d, q, &mut rng) } else { binomial::btrs(d, q, &mut rng) };
            if flipped { k = d - k; }
            if 2 * k > d { 1.0 } else if 2 * k < d { -1.0 } else { 0.0 }
        } else if beta > 0.5 { 1.0 } else if beta < 0.5 { -1.0 } else { 0.0 };
        *o = x;
    }
}
