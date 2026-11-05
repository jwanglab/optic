use anyhow::{anyhow, Context, Result};
use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// modkit pileup -> (cg_id, beta); rows are joined to probes on (chrom, chromStart).
pub fn parse_pileup(path: &Path, probe_bed: &Path) -> Result<Vec<(String, f32)>> { Ok(parse_pileup_counted(path, probe_bed)?.0) }

/// `parse_pileup`, also returning the malformed rows skipped: (count, first such line number).
fn parse_pileup_counted(path: &Path, probe_bed: &Path) -> Result<(Vec<(String, f32)>, (u64, u64))> {
    let coord_to_cg = load_coord_map(probe_bed)?;
    let f = std::fs::File::open(path).with_context(|| format!("opening pileup {}", path.display()))?;
    let mut counts: HashMap<String, (u64, u64)> = HashMap::new();
    let mut n_lines = 0u64;
    let mut bad = (0u64, 0u64);
    for line in BufReader::new(f).lines() {
        let line = line?;
        n_lines += 1;
        if line.starts_with('#') || line.trim().is_empty() { continue; }
        let f: Vec<&str> = line.split('\t').collect();
        let parsed = (f.len() >= 13).then(|| (f[1].parse::<u64>().ok(), f[11].parse::<u64>().ok(), f[12].parse::<u64>().ok()));
        let Some((Some(start), Some(m), Some(u))) = parsed else {
            if n_lines > 1 || f.len() < 2 || !f[1].chars().any(|c| c.is_ascii_alphabetic()) {   // a first-line column header is fine
                bad.0 += 1;
                if bad.1 == 0 { bad.1 = n_lines; }
            }
            continue;
        };
        let Some(cg) = coord_to_cg.get(&(f[0].to_string(), start)) else { continue };
        let e = counts.entry(cg.clone()).or_insert((0, 0));
        e.0 += m;
        e.1 += u;
    }
    if n_lines == 0 { return Err(anyhow!("pileup file is empty: {}", path.display())); }
    let mut out: Vec<(String, f32)> = counts.into_iter()
        .filter(|(_, (m, u))| m + u > 0).map(|(cg, (m, u))| (cg, m as f32 / (m + u) as f32)).collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok((out, bad))
}

/// One CSV record: comma-separated, fields optionally double-quoted ("" inside quotes is a literal quote).
/// Records do not span lines, which holds for every beta export optic reads.
fn csv_fields(line: &str) -> Vec<String> {
    let line = line.strip_suffix('\r').unwrap_or(line);
    let mut out = Vec::new();
    let mut cur = String::new();
    let (mut quoted, mut chars) = (false, line.chars().peekable());
    while let Some(c) = chars.next() {
        match (c, quoted) {
            ('"', true) if chars.peek() == Some(&'"') => { cur.push('"'); chars.next(); }
            ('"', true) => quoted = false,
            ('"', false) if cur.is_empty() => quoted = true,
            (',', false) => out.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    out.push(cur);
    out
}

/// Infinium CSV with `Name`/`ID_REF`/`probe_id` and `Beta`/`methylation_call` columns.
/// Values may be betas in [0,1] or ternary calls in {-1, 0, 1} (-1 -> 0.0).
pub fn parse_beta_csv(path: &Path) -> Result<Vec<(String, f32)>> {
    let f = std::fs::File::open(path).with_context(|| format!("opening beta CSV {}", path.display()))?;
    let mut lines = BufReader::new(f).lines();
    let headers = csv_fields(lines.next().transpose()?.as_deref().unwrap_or("").trim_start_matches('\u{feff}'));
    let col = |names: &[&str], what: &str| headers.iter().position(|h| names.iter().any(|n| h.trim().eq_ignore_ascii_case(n)))
        .ok_or_else(|| anyhow!("CSV missing {what} column"));
    let name_idx = col(&["Name", "ID_REF", "probe_id"], "Name")?;
    let beta_idx = col(&["Beta", "methylation_call"], "Beta")?;
    let mut out = Vec::new();
    let (mut bad, mut first_bad, mut out_of_range) = (0u64, String::new(), 0u64);
    for (i, line) in lines.enumerate() {
        let line = line?;
        if line.trim().is_empty() { continue; }
        let r = csv_fields(&line);
        let Some(cg) = r.get(name_idx).filter(|s| !s.is_empty()) else { continue };
        let v = r.get(beta_idx).map_or("", |s| s.trim());
        let b = match v.parse::<f32>() {
            Ok(b) if b.is_nan() => continue,
            Ok(b) if (0.0..=1.0).contains(&b) || b == -1.0 => b,
            Ok(b) if b.is_finite() => { out_of_range += 1; b }
            // a missing value is normal for arrays; anything else unreadable is reported
            _ if v.is_empty() || ["na", "nan", "null", "none", "."].contains(&v.to_ascii_lowercase().as_str()) => continue,
            _ => { bad += 1; if first_bad.is_empty() { first_bad = format!("line {}: {cg} = {v:?}", i + 2); } continue }
        };
        out.push((cg.to_string(), if b == -1.0 { 0.0 } else { b }));
    }
    if out_of_range > 0 {
        crate::warn::warn("CSV_RANGE", format!("{}: {out_of_range} betas lie outside [0, 1] (percentages or M-values?); they were used as given: > 0.5 methylated, < 0.5 unmethylated", path.display()));
    }
    if bad > 0 {
        crate::warn::warn("CSV_VALUES", format!("{}: {bad} rows have a beta that is not a number (first: {first_bad}); they were skipped", path.display()));
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
// fyi - GRCh38 can be used with chr# or #, but not accessions - ref is automatically detected in
// parse_pileup_any()

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

/// Write per-probe counts as TSV: probe_id, n_mod, n_unmod, fraction (sorted by probe id).
pub fn write_counts(path: &Path, counts: &crate::bam::Counts) -> Result<()> {
    use std::io::Write;
    let mut rows: Vec<(&String, &(u64, u64))> = counts.iter().filter(|(_, (m, u))| m + u > 0).collect();
    rows.sort_by(|a, b| a.0.cmp(b.0));
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?);
    writeln!(f, "probe_id\tn_mod\tn_unmod\tfraction")?;
    for (cg, (m, u)) in rows { writeln!(f, "{cg}\t{m}\t{u}\t{:.4}", *m as f64 / (m + u) as f64)?; }
    f.flush()?;
    Ok(())
}

/// Add a counts TSV (`optic counts`, or `rc pileup`: probe_id, n_mod, n_unmod, ...) into `counts`.
pub fn add_counts(path: &Path, counts: &mut crate::bam::Counts) -> Result<()> {
    let f = std::fs::File::open(path).with_context(|| format!("opening counts {}", path.display()))?;
    let (mut bad, mut first_bad) = (0u64, 0usize);
    for (i, line) in BufReader::new(f).lines().enumerate() {
        let line = line?;
        if i == 0 && line.starts_with("probe_id") || line.trim().is_empty() { continue; }
        let f: Vec<&str> = line.split('\t').collect();
        match (f.len() >= 3).then(|| (f[1].trim().parse::<u64>(), f[2].trim().parse::<u64>())) {
            Some((Ok(m), Ok(u))) => { let e = counts.entry(f[0].to_string()).or_insert((0, 0)); e.0 += m; e.1 += u; }
            _ => { bad += 1; if first_bad == 0 { first_bad = i + 1; } }
        }
    }
    if bad > 0 {
        crate::warn::warn("COUNTS_ROWS", format!("{}: {bad} rows are not probe_id<TAB>n_mod<TAB>n_unmod with whole-number counts (first at line {first_bad}); they were skipped",
                                                 path.display()));
    }
    Ok(())
}

/// (chrom, start) -> cg_id from a probe BED, with its contigs renamed to a BAM's own names (`rename`: BED contig ->
/// BAM contig, from `inspect::resolve`); contigs the BAM lacks are dropped.
pub fn load_coord_map_renamed(path: &Path, rename: &HashMap<String, String>) -> Result<HashMap<(String, u64), String>> {
    let content = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut map = HashMap::new();
    for (i, line) in content.lines().enumerate() {
        if (i == 0 && line.starts_with("chrom")) || line.trim().is_empty() { continue; }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 4 { continue; }
        let (Some(c), Ok(start)) = (rename.get(f[0]), f[1].parse::<u64>()) else { continue };
        map.insert((c.clone(), start), f[3].to_string());
    }
    Ok(map)
}

/// A modkit pileup against whichever reference it was made on: the T2T map (NC_/chr names) and, if the model has one,
/// the GRCh38 map (chr / bare names) are both tried, and the one covering more probes is used. Returns the reference too.
pub fn parse_pileup_any(path: &Path, model_dir: &Path) -> Result<(Vec<(String, f32)>, &'static str)> {
    let (t2t, bad) = parse_pileup_counted(path, &model_dir.join("probes.T2T.bed"))?;
    if bad.0 > 0 {
        crate::warn::warn("PILEUP_ROWS", format!("{}: {} rows could not be read as modkit pileup rows (13+ tab-separated columns, numeric start and counts; first at line {}); they were skipped",
                                                 path.display(), bad.0, bad.1));
    }
    let g = model_dir.join("probes.hg38.bed");
    if !g.exists() { return Ok((t2t, "T2T-CHM13v2.0")); }
    let tmp = std::env::temp_dir().join(format!("optic_grch38_{}.bed", std::process::id()));
    // GRCh38 map with bare (1, 2, ..) names added alongside chr names
    let mut text = String::new();
    for line in std::fs::read_to_string(&g)?.lines() {
        text.push_str(line); text.push('\n');
        if let Some(rest) = line.strip_prefix("chr") { text.push_str(rest); text.push('\n'); }
    }
    let hg = match std::fs::write(&tmp, &text) {
        Ok(_) => { let r = parse_pileup(path, &tmp); let _ = std::fs::remove_file(&tmp); r? }
        Err(_) => parse_pileup(path, &g)?,
    };
    Ok(if hg.len() > t2t.len() { (hg, "GRCh38") } else { (t2t, "T2T-CHM13v2.0") })
}

/// Byte-range shards (`optic counts --shard` on an unindexed BAM) leave `<counts>.range`: their first and stop record
/// virtual positions. When every input has one, they must chain into the whole BAM: no record missed or counted twice.
pub fn check_ranges(inputs: &[std::path::PathBuf]) -> Result<()> {
    let mut spans = Vec::new();
    for p in inputs {
        let Ok(s) = std::fs::read_to_string(format!("{}.range", p.display())) else { return Ok(()) };
        let v: Vec<u64> = s.split_whitespace().map(|x| x.parse()).collect::<std::result::Result<_, _>>()
            .with_context(|| format!("reading {}.range", p.display()))?;
        anyhow::ensure!(v.len() == 2, "{}.range: expected two virtual positions", p.display());
        spans.push((v[0], v[1]));
    }
    spans.sort_unstable();
    anyhow::ensure!(crate::bam::spans_chain(&spans),
        "these counts files are not the complete set of byte-range shards of one BAM (shard boundaries don't chain): \
         pass every shard's output, or rerun with one shard");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::csv_fields;
    #[test]
    fn csv_quoting() {
        assert_eq!(csv_fields("a,b,,c"), ["a", "b", "", "c"]);
        assert_eq!(csv_fields("\"cg1\",\"x, y\",0.5\r"), ["cg1", "x, y", "0.5"]);
        assert_eq!(csv_fields("\"say \"\"hi\"\"\",1"), ["say \"hi\"", "1"]);
    }
}
