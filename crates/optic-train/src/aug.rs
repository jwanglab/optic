//! Input construction on the GPU from uint8 beta codes
//!     Per row:
//!         nanopore:array profile fraction (`nano_prob`)
//!         the read-depth scale
//!         the keep probability (coverage rung x probe dropout)
//!         for purity-eligible rows (first purity block the row is eligible in):
//!             a tumour fraction
//!             background profile from that block's bank
//! Per probe (drawn in the fused kernel):
//!     keep, purity, mix, read depth, Binomial reads with call error, majority call
//! Output (B,P) in {-1, 0 (missing / dropped / read tie, as at inference), +1}.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use crate::gpu::{Buf, Gpu};
use std::sync::Mutex;
use crate::spec::{Spec, Tables};

pub struct Aug<'a> {
    pub spec: &'a Spec, pub tab: &'a Tables, pub gpu: &'a Gpu,
    /// per-batch seed of the fused kernel's Philox streams
    pub seed: std::sync::atomic::AtomicU64,
    /// row-level draws (sequenced, depth scale, purity, background)
    pub rng: Mutex<crate::rng::StdRng>,
    depth_lut: Buf,
    bank: Option<CudaSlice<u8>>,
}

impl<'a> Aug<'a> {
    pub fn new(spec: &'a Spec, tab: &'a Tables, gpu: &'a Gpu) -> Result<Self> {
        let s = spec.optim.seed;
        Ok(Aug { spec, tab, gpu, seed: std::sync::atomic::AtomicU64::new(s * 7_919 + 1),
                 rng: Mutex::new(crate::rng::StdRng::seed_from_u64(s * 15_485_863 + 3)),
                 depth_lut: gpu.upload(&tab.depth_lut)?,
                 bank: match &tab.bank { Some(b) => Some(gpu.upload(b)?), None => None } })
    }

    /// Host (B x k) log-prob targets, row-major: the stored rows.
    pub fn targets(&self, rows: &[u32]) -> Vec<f32> {
        let k = self.tab.k;
        rows.iter().flat_map(|&r| self.tab.lut[r as usize * k..(r as usize + 1) * k].iter().copied()).collect()
    }

    /// codes: (B, P) beta codes on the device; coverage: covered-probe count per row; rows: corpus row ids.
    /// Returns (B, P) inputs in {-1, 0, +1}.
    pub fn build(&self, codes: &CudaSlice<u8>, coverage: &[f32], rows: &[u32]) -> Result<Buf> {
        // row-level draws: sequenced or not, depth scale, purity and its background
        let b = rows.len();
        let (r_noise, r_scale, r_pm, r_pi, r_bg) = {
            let mut g = self.rng.lock().unwrap();
            let mut draw = || -> Vec<f32> { (0..b).map(|_| g.f32()).collect() };
            (draw(), draw(), draw(), draw(), draw())
        };
        let mut purity = vec![1f32; b]; let mut bg_idx = vec![0u32; b];
        for i in 0..b {
            // the first purity block this row is eligible in (blocks have disjoint domains, so at most one)
            let Some((pu, off)) = self.tab.purity.iter().zip(&self.tab.bank_offset).find(|(pu, _)| pu.eligible[rows[i] as usize]) else { continue };
            if (r_pm[i] as f64) < pu.cfg.prob {
                purity[i] = (pu.cfg.pi_lo + (pu.cfg.pi_hi - pu.cfg.pi_lo) * r_pi[i] as f64) as f32;
                let t = r_bg[i] as f64 * *pu.cum_w.last().unwrap();
                bg_idx[i] = off + pu.cum_w.partition_point(|&c| c < t).min(pu.n_bank - 1) as u32;
            }
        }
        let w = &self.spec.read_depth.weights; let wt: f64 = w.iter().sum();
        let cum: Vec<f32> = w.iter().scan(0.0, |a, &x| { *a += x / wt; Some(*a as f32) }).collect();
        let n_full = self.tab.p as f32; let keep_drop = 1.0 - self.spec.probe_mask_prob as f32;
        let mut rowp = vec![0f32; b * 4];
        for i in 0..b {
            rowp[i * 4] = (r_noise[i] < self.spec.nano_prob as f32) as u8 as f32;
            rowp[i * 4 + 1] = cum.iter().take(cum.len() - 1).filter(|&&c| r_scale[i] >= c).count() as f32;
            rowp[i * 4 + 2] = (coverage[i] / n_full) * keep_drop;
            rowp[i * 4 + 3] = purity[i];
        }
        let gpu = self.gpu;
        let rowp_t = gpu.upload(&rowp)?;
        let seed = self.seed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let p = self.tab.p;
        let mut out = gpu.zeros::<f32>(b * p)?;
        // when any row mixes, the drawn background profiles go in rows B..2B of the codes
        let codes2 = match &self.bank {
            Some(bank) if purity.iter().any(|&p| p < 1.0) => {
                let idx = gpu.upload(&bg_idx)?;
                let mut c2 = gpu.zeros::<u8>(2 * b * p)?;
                gpu.stream.memcpy_dtod(codes, &mut c2.slice_mut(0..b * p))?;
                let mut bgr = gpu.zeros::<u8>(b * p)?;
                gpu.gather_rows_u8(&mut bgr, bank, &idx, b, p)?;
                gpu.stream.memcpy_dtod(&bgr, &mut c2.slice_mut(b * p..2 * b * p))?;
                Some(c2)
            }
            _ => None,
        };
        gpu.fused_call(&mut out, codes2.as_ref().unwrap_or(codes), &rowp_t, &self.depth_lut, b, p, seed, self.spec.read_error as f32)?;
        Ok(out)
    }
}
