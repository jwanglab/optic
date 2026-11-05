# optic-cuda-oxide

optic-train's CUDA kernels, written in Rust and compiled to PTX by
[cuda-oxide](https://github.com/NVIDIA/cuda-rust), NVIDIA's experimental rustc backend. `optic-train`'s `build.rs`
runs `cargo oxide build` here and embeds the module (`OPTIC_OXIDE_PTX=<file>` uses a prebuilt one instead). This crate
builds on its own pinned toolchain (1.99.1) and is not part of the optic workspace.

| file | kernels |
|---|---|
| `ops.rs` | element-wise ops, bias, AdamW, dropout mask, row gather, loss |
| `layernorm.rs` | LayerNorm forward, statistics and backward |
| `fused_call.rs` (+ `philox.rs`, `binomial.rs`) | training inputs from beta codes: read-depth draws, per-read call errors, purity mixing |
| `gemm.rs` | FP32 matrix products (replaces cuBLAS) |

**Arithmetic.** The build sets `CUDA_OXIDE_NO_FMA=1` and runs llc through `scripts/llc-nvcc-fma.sh`
(`-nvptx-fma-level=1`), so only `a*b + c` is fused.

**GEMM.** Blocks of 256 threads compute a 128×128 output tile; 8 warps take 64×32 warp tiles, each lane two 4×4
fragments read from shared memory as 128-bit loads. k advances 16 at a time through double-buffered shared memory filled
by `cp.async` (zero-fill at the edges). When the tiles can't fill the GPU, k is split across blocks and the partial
sums are added in a fixed order (`gemm_reduce_kernel`), so results are deterministic. Indexing is 32-bit. At the
trainer's shapes it takes 1.17× cuBLAS's time on the matrix products and trains at the same speed end to end.
