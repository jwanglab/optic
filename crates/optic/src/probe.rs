use anyhow::{anyhow, Context, Result};
use std::collections::HashMap;
use std::path::Path;

pub struct ProbeSubset {
    cg_to_index: HashMap<String, usize>,
}

impl ProbeSubset {
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let mut cg_to_index = HashMap::new();
        for (i, line) in content.lines().enumerate() {
            if (i == 0 && line.starts_with("chrom")) || line.trim().is_empty() { continue; }
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() < 4 { return Err(anyhow!("malformed BED line {i}: {line}")); }
            cg_to_index.insert(f[3].to_string(), cg_to_index.len());
        }
        Ok(ProbeSubset { cg_to_index })
    }

    pub fn n_probes(&self) -> usize { self.cg_to_index.len() }

    /// beta > 0.5 -> +1, < 0.5 -> -1, else (or missing) 0. Returns (ternary, n_covered).
    pub fn cgs_to_ternary(&self, cgs: &[(String, f32)]) -> (Vec<f32>, usize) {
        let mut out = vec![0.0f32; self.n_probes()];
        let mut n = 0;
        for (cg, b) in cgs {
            if let Some(&i) = self.cg_to_index.get(cg) {
                if *b > 0.5 { out[i] = 1.0; n += 1; } else if *b < 0.5 { out[i] = -1.0; n += 1; }
            }
        }
        (out, n)
    }
}
