use anyhow::{anyhow, Context, Result};
use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// modkit pileup -> (cg_id, beta); rows are joined to probes on (chrom, chromStart).
pub fn parse_pileup(path: &Path, probe_bed: &Path) -> Result<Vec<(String, f32)>> {
    let coord_to_cg = load_coord_map(probe_bed)?;
    let f = std::fs::File::open(path).with_context(|| format!("opening pileup {}", path.display()))?;
    let mut counts: HashMap<String, (u64, u64)> = HashMap::new();
    let mut n_lines = 0u64;
    for line in BufReader::new(f).lines() {
        let line = line?;
        n_lines += 1;
        if line.starts_with('#') || line.trim().is_empty() { continue; }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 13 { continue; }
        let Ok(start) = f[1].parse::<u64>() else { continue };
        let Some(cg) = coord_to_cg.get(&(f[0].to_string(), start)) else { continue };
        let e = counts.entry(cg.clone()).or_insert((0, 0));
        e.0 += f[11].parse::<u64>().unwrap_or(0);
        e.1 += f[12].parse::<u64>().unwrap_or(0);
    }
    if n_lines == 0 { return Err(anyhow!("pileup file is empty: {}", path.display())); }
    let mut out: Vec<(String, f32)> = counts.into_iter()
        .filter(|(_, (m, u))| m + u > 0).map(|(cg, (m, u))| (cg, m as f32 / (m + u) as f32)).collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

/// Infinium CSV with `Name`/`ID_REF`/`probe_id` and `Beta`/`methylation_call` columns.
/// Values may be betas in [0,1] or ternary calls in {-1, 0, 1} (-1 -> 0.0).
pub fn parse_beta_csv(path: &Path) -> Result<Vec<(String, f32)>> {
    let mut rdr = csv::ReaderBuilder::new().has_headers(true).from_path(path)
        .with_context(|| format!("opening beta CSV {}", path.display()))?;
    let headers = rdr.headers()?.clone();
    let col = |names: &[&str], what: &str| headers.iter().position(|h| names.iter().any(|n| h.eq_ignore_ascii_case(n)))
        .ok_or_else(|| anyhow!("CSV missing {what} column"));
    let name_idx = col(&["Name", "ID_REF", "probe_id"], "Name")?;
    let beta_idx = col(&["Beta", "methylation_call"], "Beta")?;
    let mut out = Vec::new();
    for rec in rdr.records() {
        let r = rec?;
        let Some(cg) = r.get(name_idx).filter(|s| !s.is_empty()) else { continue };
        let Ok(b) = r.get(beta_idx).unwrap_or("").parse::<f32>() else { continue };
        out.push((cg.to_string(), if b == -1.0 { 0.0 } else { b }));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

/// T2T-CHM13v2.0 RefSeq accession <-> chr name; inputs may use either form.
const T2T_ALIASES: &[(&str, &str)] = &[
    ("NC_060925.1", "chr1"),  ("NC_060926.1", "chr2"),  ("NC_060927.1", "chr3"),  ("NC_060928.1", "chr4"),
    ("NC_060929.1", "chr5"),  ("NC_060930.1", "chr6"),  ("NC_060931.1", "chr7"),  ("NC_060932.1", "chr8"),
    ("NC_060933.1", "chr9"),  ("NC_060934.1", "chr10"), ("NC_060935.1", "chr11"), ("NC_060936.1", "chr12"),
    ("NC_060937.1", "chr13"), ("NC_060938.1", "chr14"), ("NC_060939.1", "chr15"), ("NC_060940.1", "chr16"),
    ("NC_060941.1", "chr17"), ("NC_060942.1", "chr18"), ("NC_060943.1", "chr19"), ("NC_060944.1", "chr20"),
    ("NC_060945.1", "chr21"), ("NC_060946.1", "chr22"), ("NC_060947.1", "chrX"),  ("NC_060948.1", "chrY"),
];

/// (chrom, start) -> cg_id from the probe BED, keyed under both naming forms.
pub fn load_coord_map(path: &Path) -> Result<HashMap<(String, u64), String>> {
    let alias: HashMap<&str, &str> = T2T_ALIASES.iter().flat_map(|(a, b)| [(*a, *b), (*b, *a)]).collect();
    let content = std::fs::read_to_string(path)?;
    let mut map = HashMap::new();
    for (i, line) in content.lines().enumerate() {
        if (i == 0 && line.starts_with("chrom")) || line.trim().is_empty() { continue; }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 4 { continue; }
        let Ok(start) = f[1].parse::<u64>() else { continue };
        if let Some(alt) = alias.get(f[0]) { map.insert((alt.to_string(), start), f[3].to_string()); }
        map.insert((f[0].to_string(), start), f[3].to_string());
    }
    Ok(map)
}
