//! one CUDA context and stream (cudarc, on the CUDA driver API) and all kernels, matrix products
//! from one PTX module built from crates/optic-cuda-oxide by the cuda-oxide backend (see build.rs).

use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits};
use std::collections::HashMap;
use std::sync::Arc;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

pub type Buf = CudaSlice<f32>;

pub struct Gpu {
    pub stream: Arc<CudaStream>,
    sms: usize,
    funcs: HashMap<&'static str, CudaFunction>,
}

const FUNCS: &[&str] = &[
    "add_bias_kernel", "col_sum_kernel", "mul_kernel", "mul_cols_kernel", "add_kernel", "silu_kernel", "silu_bwd_kernel",
    "adamw_inplace_kernel", "loss_kernel", "loss_sum_kernel", "dropout_mask_kernel", "gather_rows_u8_kernel",
    "ln_fwd_kernel", "ln_stats_kernel", "ln_dx_kernel", "ln_dwdb_kernel", "fused_call_kernel", "gemm_kernel", "gemm_reduce_kernel",
];

impl Gpu {
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal).context("opening the CUDA device")?;
        // one stream does all the work, in order: no cross-stream event bookkeeping, and not the legacy default stream
        // (which synchronises with everything)
        unsafe { ctx.disable_event_tracking(); }
        let stream = ctx.new_stream().context("creating a CUDA stream")?;
        let m = ctx.load_module(cudarc::nvrtc::Ptx::from_src(PTX)).context("loading the kernels")?;
        let mut funcs = HashMap::new();
        for n in FUNCS { funcs.insert(*n, m.load_function(n).with_context(|| format!("kernel {n}"))?); }
        let sms = ctx.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as usize;
        Ok(Gpu { stream, sms, funcs })
    }

    pub fn zeros<T: DeviceRepr + ValidAsZeroBits>(&self, n: usize) -> Result<CudaSlice<T>> { Ok(self.stream.alloc_zeros::<T>(n)?) }
    pub fn upload<T: DeviceRepr>(&self, v: &[T]) -> Result<CudaSlice<T>> { Ok(self.stream.clone_htod(v)?) }
    pub fn download(&self, b: &Buf) -> Result<Vec<f32>> { Ok(self.stream.clone_dtoh(b)?) }
    pub fn sync(&self) -> Result<()> { Ok(self.stream.synchronize()?) }

    fn f(&self, name: &str) -> &CudaFunction { &self.funcs[name] }
    fn grid(n: usize) -> LaunchConfig {
        let blocks = n.div_ceil(256).clamp(1, 65_535 * 4) as u32;
        LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
    }

    // ---- matrix products (row-major) ----

    /// C (mm x nn, row-major) = sum_p A(i, p) B(p, j) + beta C, A(i, p) = a[i*sai + p*sap], B(p, j) = b[p*sbp + j*sbj]
    /// (crates/optic-cuda-oxide/src/gemm.rs). When the 128 x 128 output tiles cannot fill the GPU, k is split across
    /// blocks and the partial sums are added in a fixed order, so the result is deterministic.
    #[allow(clippy::too_many_arguments)]
    fn gemm(&self, c: &mut Buf, a: &Buf, b: &Buf, mm: usize, nn: usize, kk: usize, s: [usize; 4], beta: f32) -> Result<()> {
        let fits = |e: usize| e < u32::MAX as usize;   // the kernel indexes in 32 bits
        anyhow::ensure!(fits(a.len()) && fits(b.len()) && fits(mm * nn), "gemm: an operand of {mm} x {nn} x {kk} has 2^32 or more values");
        let tiles = mm.div_ceil(128) * nn.div_ceil(128);
        // enough blocks for two waves, at least 64 k-values each, and partial sums (re-read by the reduce) kept small
        // next to the product itself
        let by_reduce = ((1usize << 21).max(mm * nn * kk / 4096) / (mm * nn).max(1)).max(1);
        let splits = if tiles >= self.sms { 1 } else { (2 * self.sms).div_ceil(tiles).min(kk / 64).min(by_reduce).max(1) };
        let kchunk = kk.div_ceil(splits).div_ceil(16) * 16;
        let splits = kk.div_ceil(kchunk).max(1);
        let cfg = LaunchConfig { grid_dim: (nn.div_ceil(128) as u32, mm.div_ceil(128) as u32, splits as u32), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let (m64, n64, k64, kc) = (mm as u64, nn as u64, kk as u64, kchunk as u64);
        let [sai, sap, sbp, sbj] = s.map(|v| v as u64);
        let mut part = if splits > 1 { Some(unsafe { self.stream.alloc::<f32>(splits * mm * nn) }?) } else { None };
        let out = match part.as_mut() { Some(p) => p, None => &mut *c };
        let mut l = self.stream.launch_builder(self.f("gemm_kernel"));
        l.arg(out).arg(a).arg(b).arg(&m64).arg(&n64).arg(&k64).arg(&sai).arg(&sap).arg(&sbp).arg(&sbj).arg(&kc).arg(&beta);
        unsafe { l.launch(cfg) }?;
        if let Some(p) = &part {
            let (mn, sp) = ((mm * nn) as u64, splits as u64);
            let mut l = self.stream.launch_builder(self.f("gemm_reduce_kernel"));
            l.arg(c).arg(p).arg(&mn).arg(&sp).arg(&beta);
            unsafe { l.launch(Self::grid(mm * nn)) }?;
        }
        Ok(())
    }

    /// y (m x n) = x (m x k) · w (n x k)ᵀ, with `beta` times the old y added.
    pub fn mm_xwt(&self, y: &mut Buf, x: &Buf, w: &Buf, m: usize, n: usize, k: usize, beta: f32) -> Result<()> {
        self.gemm(y, x, w, m, n, k, [k, 1, 1, k], beta)
    }
    /// dw (n x k) = dy (m x n)ᵀ · x (m x k)
    pub fn mm_dytx(&self, dw: &mut Buf, dy: &Buf, x: &Buf, m: usize, n: usize, k: usize) -> Result<()> {
        self.gemm(dw, dy, x, n, k, m, [1, n, k, 1], 0.0)
    }
    /// dx (m x k) = dy (m x n) · w (n x k)
    pub fn mm_dyw(&self, dx: &mut Buf, dy: &Buf, w: &Buf, m: usize, n: usize, k: usize) -> Result<()> {
        self.gemm(dx, dy, w, m, k, n, [n, 1, k, 1], 0.0)
    }

    // ---- element-wise ----

    pub fn add_bias(&self, y: &mut Buf, b: &Buf, rows: usize, cols: usize) -> Result<()> {
        let (r, c) = (rows as u64, cols as u64);
        let mut l = self.stream.launch_builder(self.f("add_bias_kernel"));
        l.arg(y).arg(b).arg(&r).arg(&c);
        unsafe { l.launch(Self::grid(rows * cols)) }?; Ok(())
    }
    pub fn col_sum(&self, out: &mut Buf, dy: &Buf, rows: usize, cols: usize) -> Result<()> {
        let (r, c) = (rows as u64, cols as u64);
        let mut l = self.stream.launch_builder(self.f("col_sum_kernel"));
        l.arg(out).arg(dy).arg(&r).arg(&c);
        unsafe { l.launch(LaunchConfig { grid_dim: (cols.div_ceil(256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }?; Ok(())
    }
    pub fn mul(&self, y: &mut Buf, x: &Buf, m: &Buf, n: usize) -> Result<()> {
        let nn = n as u64;
        let mut l = self.stream.launch_builder(self.f("mul_kernel"));
        l.arg(y).arg(x).arg(m).arg(&nn);
        unsafe { l.launch(Self::grid(n)) }?; Ok(())
    }
    pub fn mul_cols(&self, y: &mut Buf, x: &Buf, g: &Buf, rows: usize, cols: usize) -> Result<()> {
        let (r, c) = (rows as u64, cols as u64);
        let mut l = self.stream.launch_builder(self.f("mul_cols_kernel"));
        l.arg(y).arg(x).arg(g).arg(&r).arg(&c);
        unsafe { l.launch(Self::grid(rows * cols)) }?; Ok(())
    }
    pub fn add(&self, y: &mut Buf, a: &Buf, n: usize) -> Result<()> {
        let nn = n as u64;
        let mut l = self.stream.launch_builder(self.f("add_kernel"));
        l.arg(y).arg(a).arg(&nn);
        unsafe { l.launch(Self::grid(n)) }?; Ok(())
    }
    pub fn silu(&self, y: &mut Buf, x: &Buf, n: usize) -> Result<()> {
        let nn = n as u64;
        let mut l = self.stream.launch_builder(self.f("silu_kernel"));
        l.arg(y).arg(x).arg(&nn);
        unsafe { l.launch(Self::grid(n)) }?; Ok(())
    }
    pub fn silu_bwd(&self, dx: &mut Buf, x: &Buf, dy: &Buf, n: usize) -> Result<()> {
        let nn = n as u64;
        let mut l = self.stream.launch_builder(self.f("silu_bwd_kernel"));
        l.arg(dx).arg(x).arg(dy).arg(&nn);
        unsafe { l.launch(Self::grid(n)) }?; Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn adamw(&self, p: &mut Buf, g: &Buf, m: &mut Buf, v: &mut Buf, n: usize, lr: f32, b1: f32, b2: f32, eps: f32, wd: f32, scale_m: f32, scale_v: f32) -> Result<()> {
        let nn = n as u64;
        let mut l = self.stream.launch_builder(self.f("adamw_inplace_kernel"));
        l.arg(p).arg(g).arg(m).arg(v).arg(&nn).arg(&lr).arg(&b1).arg(&b2).arg(&eps).arg(&wd).arg(&scale_m).arg(&scale_v);
        unsafe { l.launch(Self::grid(n)) }?; Ok(())
    }

    pub fn dropout_mask(&self, mask: &mut Buf, n: usize, p: f32, seed: u64) -> Result<()> {
        let nn = n as u64;
        let mut l = self.stream.launch_builder(self.f("dropout_mask_kernel"));
        l.arg(mask).arg(&nn).arg(&p).arg(&seed);
        unsafe { l.launch(Self::grid(n)) }?; Ok(())
    }
    pub fn gather_rows_u8(&self, out: &mut CudaSlice<u8>, src: &CudaSlice<u8>, idx: &CudaSlice<u32>, rows: usize, p: usize) -> Result<()> {
        let (bb, pp) = (rows as u64, p as u64);
        let mut l = self.stream.launch_builder(self.f("gather_rows_u8_kernel"));
        l.arg(out).arg(src).arg(idx).arg(&bb).arg(&pp);
        unsafe { l.launch(Self::grid(rows * p)) }?; Ok(())
    }
    /// Training inputs (out_rows x P, values -1/0/+1) from beta codes; see optic-cuda-oxide/src/fused_call.rs.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_call(&self, out: &mut Buf, codes: &CudaSlice<u8>, rowp: &Buf, lut: &Buf, out_rows: usize, p: usize, seed: u64, eps: f32) -> Result<()> {
        let (bb, pp) = (out_rows as u32, p as u32);
        let mut l = self.stream.launch_builder(self.f("fused_call_kernel"));
        l.arg(out).arg(codes).arg(rowp).arg(lut).arg(&bb).arg(&pp).arg(&seed).arg(&eps);
        let total = out_rows * p;
        unsafe { l.launch(LaunchConfig { grid_dim: (total.div_ceil(256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }?; Ok(())
    }

    // ---- LayerNorm (rows x h) ----

    pub fn ln_fwd(&self, y: &mut Buf, x: &Buf, g: &Buf, b: &Buf, rows: usize, h: usize, eps: f32) -> Result<()> {
        let hh = h as u64;
        let mut l = self.stream.launch_builder(self.f("ln_fwd_kernel"));
        l.arg(y).arg(x).arg(g).arg(b).arg(&hh).arg(&eps);
        unsafe { l.launch(LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }?; Ok(())
    }
    /// stats (2 x rows) = [mean, 1/std]
    pub fn ln_stats(&self, stats: &mut Buf, x: &Buf, rows: usize, h: usize, eps: f32) -> Result<()> {
        let (bb, hh) = (rows as u64, h as u64);
        let mut l = self.stream.launch_builder(self.f("ln_stats_kernel"));
        l.arg(stats).arg(x).arg(&bb).arg(&hh).arg(&eps);
        unsafe { l.launch(LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }?; Ok(())
    }
    pub fn ln_dx(&self, dx: &mut Buf, dxhat: &Buf, x: &Buf, stats: &Buf, rows: usize, h: usize) -> Result<()> {
        let (bb, hh) = (rows as u64, h as u64);
        let mut l = self.stream.launch_builder(self.f("ln_dx_kernel"));
        l.arg(dx).arg(dxhat).arg(x).arg(stats).arg(&bb).arg(&hh);
        unsafe { l.launch(LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }?; Ok(())
    }
    /// out (2 x h) = [d gamma, d beta]
    pub fn ln_dwdb(&self, out: &mut Buf, dy: &Buf, x: &Buf, stats: &Buf, rows: usize, h: usize) -> Result<()> {
        let (bb, hh) = (rows as u64, h as u64);
        let mut l = self.stream.launch_builder(self.f("ln_dwdb_kernel"));
        l.arg(out).arg(dy).arg(x).arg(stats).arg(&bb).arg(&hh);
        unsafe { l.launch(LaunchConfig { grid_dim: (h.div_ceil(256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }?; Ok(())
    }

    // ---- loss ----

    /// Per-row losses and dz (2B x k) for logits z (2B x k); see optic-cuda-oxide/src/ops.rs `loss_kernel`.
    #[allow(clippy::too_many_arguments)]
    pub fn loss(&self, dz: &mut Buf, cls_row: &mut Buf, cons_row: &mut Buf, z: &Buf, t: &Buf, w: &Buf, lab: &Buf, m: &Buf,
                n_nodes: usize, b: usize, k: usize, temp: f32, lam: f32, cw: Option<f32>) -> Result<()> {
        let (nn, bb, kk) = (n_nodes as u32, b as u32, k as u32);
        let (cwv, has) = (cw.unwrap_or(0.0), cw.is_some() as u32);
        let smem = ((6 * k + n_nodes) * 4) as u32;
        anyhow::ensure!(smem <= 48 * 1024, "loss: {k} classes and {n_nodes} tree nodes need more than 48 KB of shared memory");
        let mut l = self.stream.launch_builder(self.f("loss_kernel"));
        l.arg(dz).arg(cls_row).arg(cons_row).arg(z).arg(t).arg(w).arg(lab).arg(m).arg(&nn).arg(&bb).arg(&kk).arg(&temp).arg(&lam).arg(&cwv).arg(&has);
        unsafe { l.launch(LaunchConfig { grid_dim: (b as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem }) }?; Ok(())
    }
    /// out[0] += mean(cls_row), out[1] += mean(cons_row)
    pub fn loss_sum(&self, out: &mut Buf, cls_row: &Buf, cons_row: &Buf, b: usize) -> Result<()> {
        let bb = b as u32;
        let mut l = self.stream.launch_builder(self.f("loss_sum_kernel"));
        l.arg(out).arg(cls_row).arg(cons_row).arg(&bb);
        unsafe { l.launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 }) }?; Ok(())
    }
}
