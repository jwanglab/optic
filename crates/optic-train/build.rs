//! Build the CUDA kernels (crates/optic-cuda-oxide, Rust compiled to PTX by the cuda-oxide rustc backend) and embed
//! the module: `src/gpu.rs` includes `$OUT_DIR/kernels.ptx`. `$OPTIC_OXIDE_PTX` names a prebuilt module instead.
//!
//! The build runs `cargo oxide build` in that crate, on its own pinned toolchain, with conservative FMA contraction
//! (CUDA_OXIDE_NO_FMA=1 + scripts/llc-nvcc-fma.sh: only a*b+c is fused, the arithmetic optic-train was validated with).
//! GPU architecture: `$CUDA_COMPUTE_CAP` (e.g. 120, 90a, 8.6), else the first GPU nvidia-smi reports. A bare number gets
//! the architecture-specific suffix: 'a' for sm_90 to sm_120, 'f' above sm_120 (e.g. 120 -> sm_120a).

use std::path::Path;
use std::process::Command;

/// "120" -> "120a"; an explicit suffix ("90a", "121f") or a pre-sm_90 number is kept as given.
fn arch() -> String {
    let raw = std::env::var("CUDA_COMPUTE_CAP").ok().or_else(|| {
        let out = Command::new("nvidia-smi").args(["--query-gpu=compute_cap", "--format=csv,noheader"]).output().ok()?;
        String::from_utf8_lossy(&out.stdout).lines().next().map(str::to_string)
    }).unwrap_or_else(|| panic!("set CUDA_COMPUTE_CAP (e.g. 120 for Blackwell, 90 for Hopper): nvidia-smi could not report one"));
    let s = raw.trim().to_lowercase();
    let s = s.strip_prefix("sm_").unwrap_or(&s).replace('.', "");
    if s.ends_with('a') || s.ends_with('f') { return s; }
    let base: u32 = s.parse().unwrap_or_else(|_| panic!("CUDA_COMPUTE_CAP {raw:?} is not a compute capability"));
    let base = if base < 20 { base * 10 } else { base };
    match base { 120 => "120a".into(), b if b > 120 => format!("{b}f"), b if b >= 90 => format!("{b}a"), b => b.to_string() }
}

fn main() {
    let dst = Path::new(&std::env::var("OUT_DIR").unwrap()).join("kernels.ptx");
    for v in ["CUDA_COMPUTE_CAP", "OPTIC_OXIDE_PTX"] { println!("cargo:rerun-if-env-changed={v}"); }
    if let Some(p) = std::env::var_os("OPTIC_OXIDE_PTX") {
        std::fs::copy(&p, &dst).unwrap_or_else(|e| panic!("OPTIC_OXIDE_PTX {}: {e}", Path::new(&p).display()));
        return;
    }
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../optic-cuda-oxide").canonicalize().expect("crates/optic-cuda-oxide is missing");
    for f in ["src", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "scripts/llc-nvcc-fma.sh"] {
        println!("cargo:rerun-if-changed={}", dir.join(f).display());
    }
    let ptx = dir.join("optic_cuda_oxide.ptx");
    let started = std::time::SystemTime::now();
    // a clean environment: the outer cargo's toolchain, flags and target dir must not leak into the inner build
    let run = |args: &[&str]| {
        let mut cmd = Command::new("cargo");
        cmd.env_clear().current_dir(&dir).args(args);
        for k in ["PATH", "HOME", "USER", "CARGO_HOME", "RUSTUP_HOME", "CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH"] {
            if let Some(v) = std::env::var_os(k) { cmd.env(k, v); }
        }
        cmd.env("CUDA_OXIDE_NO_FMA", "1").env("CUDA_OXIDE_LLC", dir.join("scripts/llc-nvcc-fma.sh"));
        let o = cmd.output().unwrap_or_else(|e| panic!("could not run cargo in {} ({e})", dir.display()));
        if !o.status.success() {
            panic!("`cargo {}` failed in {} (is cargo-oxide installed? see the README):\n{}{}", args.join(" "), dir.display(),
                   String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
        }
    };
    let sm = format!("sm_{}", arch());
    run(&["oxide", "build", "--arch", &sm]);
    // cargo-oxide reuses a cached module when the sources are unchanged, even if it was built with other settings
    let fresh = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).map(|t| t >= started).unwrap_or(false);
    if !fresh(&ptx) {
        run(&["oxide", "clean"]);
        run(&["oxide", "build", "--arch", &sm]);
    }
    std::fs::copy(&ptx, &dst).unwrap_or_else(|e| panic!("{}: {e}", ptx.display()));
}
