#!/usr/bin/env bash
# llc with nvcc's FMA contraction instead of cuda-oxide's: run it with CUDA_OXIDE_NO_FMA=1 (no IR `contract` flags,
# no -fp-contract=fast) and CUDA_OXIDE_LLC=<this script>. Only the NVPTX backend's own mul+add fusion then runs,
# as in nvcc's (LLVM-based) NVVM: fadd(fmul, x) becomes fma.rn, a*b - c stays mul.f32 + sub.f32 for ptxas.
set -eu
llc=$(dirname "$(rustup which --toolchain 1.99.1 rustc)")/../lib/rustlib/x86_64-unknown-linux-gnu/bin/llc
exec "$llc" -nvptx-fma-level="${OPTIC_OXIDE_FMA_LEVEL:-1}" "$@"
