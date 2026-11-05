//! Input precheck for BAMs: catch the common reasons a run would silently cover few or no probes, before piling up.
//!
//! Supported: nanopore modBAMs aligned to T2T-CHM13v2.0 or GRCh38
//! with 5mC (optionally + 5hmC) MM/ML tags. Contigs are recognised by length (unique per chromosome within an
//! assembly), so any naming works: RefSeq (NC_060925.1, NC_000001.11), UCSC (chr1), Ensembl (1), GenBank (CP068277.2).
//! The probes are on chr1-22.
//!
//! Checks (header, BAM index, and the first `n_records` reads):
//!   fatal  not a BAM / CRAM; no @SQ (unaligned); GRCh37 or an unknown assembly; GRCh38 without a GRCh38 probe map;
//!          index from a different BAM; no mapped reads on probe contigs; no MM/ML tags; tags without 5mC/5hmC; MM w/o ML
//!   warn   probe contigs missing; tags on only part of the reads; ML length or MN mismatches
//!   (no sort-order check: SO:unknown headers are common on sorted BAMs, and an index already implies sorting)
//!   info   assembly and naming, reads sampled, modification codes, mapped reads (index), basecaller/aligner (@PG)

use anyhow::Result;
use crate::bamio::{self, Record};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use std::io::Read;
use std::path::Path;

/// A reference assembly optic has a probe map for: (name in that map, UCSC name, length) per contig; chr1-22 first.
#[derive(Debug)]
pub struct Assembly { pub name: &'static str, pub probe_file: &'static str, pub contigs: &'static [(&'static str, &'static str, u64)] }

pub const T2T: Assembly = Assembly { name: "T2T-CHM13v2.0", probe_file: "probes.T2T.bed", contigs: &[
    ("NC_060925.1", "chr1", 248387328), ("NC_060926.1", "chr2", 242696752), ("NC_060927.1", "chr3", 201105948),
    ("NC_060928.1", "chr4", 193574945), ("NC_060929.1", "chr5", 182045439), ("NC_060930.1", "chr6", 172126628),
    ("NC_060931.1", "chr7", 160567428), ("NC_060932.1", "chr8", 146259331), ("NC_060933.1", "chr9", 150617247),
    ("NC_060934.1", "chr10", 134758134), ("NC_060935.1", "chr11", 135127769), ("NC_060936.1", "chr12", 133324548),
    ("NC_060937.1", "chr13", 113566686), ("NC_060938.1", "chr14", 101161492), ("NC_060939.1", "chr15", 99753195),
    ("NC_060940.1", "chr16", 96330374), ("NC_060941.1", "chr17", 84276897), ("NC_060942.1", "chr18", 80542538),
    ("NC_060943.1", "chr19", 61707364), ("NC_060944.1", "chr20", 66210255), ("NC_060945.1", "chr21", 45090682),
    ("NC_060946.1", "chr22", 51324926), ("NC_060947.1", "chrX", 154259566), ("NC_060948.1", "chrY", 62460029),
]};
pub const GRCH38: Assembly = Assembly { name: "GRCh38", probe_file: "probes.hg38.bed", contigs: &[
    ("chr1", "chr1", 248956422), ("chr2", "chr2", 242193529), ("chr3", "chr3", 198295559), ("chr4", "chr4", 190214555),
    ("chr5", "chr5", 181538259), ("chr6", "chr6", 170805979), ("chr7", "chr7", 159345973), ("chr8", "chr8", 145138636),
    ("chr9", "chr9", 138394717), ("chr10", "chr10", 133797422), ("chr11", "chr11", 135086622), ("chr12", "chr12", 133275309),
    ("chr13", "chr13", 114364328), ("chr14", "chr14", 107043718), ("chr15", "chr15", 101991189), ("chr16", "chr16", 90338345),
    ("chr17", "chr17", 83257441), ("chr18", "chr18", 80373285), ("chr19", "chr19", 58617616), ("chr20", "chr20", 64444167),
    ("chr21", "chr21", 46709983), ("chr22", "chr22", 50818468), ("chrX", "chrX", 156040895), ("chrY", "chrY", 57227415),
]};
pub const ASSEMBLIES: [&Assembly; 2] = [&T2T, &GRCH38];
/// chr1 lengths of assemblies optic does not support, to name them.
const UNSUPPORTED: &[(u64, &str)] = &[(249250621, "GRCh37/hg19"), (247249719, "NCBI36/hg18")];

/// The assemblies whose probe map is present in a model directory.
pub fn available(model_dir: &Path) -> Vec<&'static Assembly> {
    ASSEMBLIES.into_iter().filter(|a| model_dir.join(a.probe_file).exists()).collect()
}

/// A BAM header matched to an assembly: probe-map contig name -> the BAM's own name for it.
#[derive(Debug)]
pub struct Resolved { pub assembly: &'static Assembly, pub contigs: HashMap<String, String>, pub naming: String, pub missing: Vec<String> }

fn naming_of(names: &[&str]) -> String {
    let kind = |n: &str| if n.starts_with("NC_") { "RefSeq" } else if n.starts_with("chr") { "UCSC" } else if n.starts_with("CP") { "GenBank" }
                        else if n.chars().all(|c| c.is_ascii_digit()) { "Ensembl" } else { "other" };
    let mut k: Vec<&str> = names.iter().map(|n| kind(n)).collect(); k.sort(); k.dedup();
    let ex = names.first().copied().unwrap_or("?");
    format!("{} (e.g. {ex})", k.join(" / "))
}

/// Match a header's contigs to a supported assembly by length. Err = fatal issues.
pub fn resolve(refs: &[(String, u64)], available: &[&'static Assembly]) -> std::result::Result<Resolved, Vec<Issue>> {
    let fatal = |code, message: String, hint: &str| Err(vec![Issue { level: Level::Fatal, code, message, hint: hint.into() }]);
    if refs.is_empty() {
        return fatal("UNALIGNED", "the BAM has no reference sequences: it is unaligned (e.g. straight from dorado basecaller)".into(),
                     "align it keeping the MM/ML tags (dorado aligner chm13v2.0.fa in.bam, or samtools fastq -T MM,ML | minimap2 -y -ax map-ont chm13v2.0.fa -), then sort and index");
    }
    let mut by_len: HashMap<u64, Vec<&str>> = HashMap::new();
    for (n, l) in refs { by_len.entry(*l).or_default().push(n.as_str()); }
    let score = |a: &Assembly| a.contigs[..22].iter().filter(|(_, _, l)| by_len.contains_key(l)).count();
    let Some(asm) = ASSEMBLIES.into_iter().max_by_key(|a| score(a)).filter(|a| score(a) > 0) else {
        if let Some((_, name)) = UNSUPPORTED.iter().find(|(l, _)| by_len.contains_key(l)) {
            return fatal("WRONG_REFERENCE", format!("aligned to {name}: optic's probe coordinates are T2T-CHM13v2.0 (or GRCh38)"),
                         "realign the reads to chm13v2.0.fa (or GRCh38) keeping the MM/ML tags; liftover of a BAM is not supported");
        }
        return fatal("UNKNOWN_REFERENCE", format!("none of the {} contigs has the length of a T2T-CHM13v2.0 or GRCh38 autosome (first: {} {} bp)", refs.len(), refs[0].0, refs[0].1),
                     "align to T2T-CHM13v2.0 (or GRCh38); contig names do not matter, lengths identify the chromosomes");
    };
    if !available.iter().any(|a| std::ptr::eq(*a, asm)) {
        return fatal("NO_PROBE_MAP", format!("aligned to {}, but this model has no {} probe map ({})", asm.name, asm.name, asm.probe_file),
                     "use a model directory that includes it, or realign to T2T-CHM13v2.0");
    }
    let norm = |n: &str| n.trim_start_matches("chr").to_string();
    let (mut contigs, mut missing, mut used) = (HashMap::new(), Vec::new(), Vec::new());
    for (map_name, ucsc, len) in &asm.contigs[..22] {
        match by_len.get(len).map(|v| v.as_slice()) {
            Some([one]) => { contigs.insert(map_name.to_string(), one.to_string()); used.push(*one); }
            Some(many) => match many.iter().find(|n| norm(n) == norm(ucsc) || **n == *map_name) {   // equal lengths: prefer the matching name
                Some(n) => { contigs.insert(map_name.to_string(), n.to_string()); used.push(*n); }
                None => missing.push(ucsc.to_string()),
            },
            None => missing.push(ucsc.to_string()),
        }
    }
    Ok(Resolved { assembly: asm, contigs, naming: naming_of(&used), missing })
}

/// Header-only reference resolution for a BAM (no reads touched): what `counts` needs even with --no-checks.
pub fn resolve_bam(path: &Path, model_dir: &Path) -> Result<Resolved> {
    let (_, header) = bamio::open(path)?;
    let refs = header.refs;
    resolve(&refs, &available(model_dir)).map_err(|e| anyhow::anyhow!("{}: {}", e[0].code, e[0].message))
}

#[derive(Serialize, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Level { Info, Warn, Fatal }

#[derive(Serialize, Clone, Debug)]
pub struct Issue { pub level: Level, pub code: &'static str, pub message: String, pub hint: String }

#[derive(Serialize, Default, Debug)]
pub struct Inspection {
    pub reference: String,
    pub probe_map: String,
    pub naming: String,
    pub n_contigs: usize,
    pub probe_contigs_present: usize,
    pub sort_order: String,
    pub index: Option<IndexInfo>,
    pub sample: Sample,
    pub programs: Vec<String>,
    pub issues: Vec<Issue>,
}

#[derive(Serialize, Default, Debug)]
pub struct IndexInfo { pub n_references: usize, pub mapped_on_probe_contigs: Option<u64>, pub unmapped: Option<u64> }

#[derive(Serialize, Default, Debug)]
pub struct Sample {
    pub records: u64,
    pub primary_mapped: u64,
    pub with_mm: u64,
    pub with_ml: u64,
    pub ml_length_mismatch: u64,
    pub mn_mismatch: u64,
    pub mod_codes: BTreeMap<String, u64>,
}

impl Inspection {
    pub fn fatal(&self) -> Vec<&Issue> { self.issues.iter().filter(|i| i.level == Level::Fatal).collect() }
    fn push(&mut self, level: Level, code: &'static str, message: impl Into<String>, hint: impl Into<String>) {
        self.issues.push(Issue { level, code, message: message.into(), hint: hint.into() });
    }
}

/// Modification codes of an MM tag: "C+m", "C+h", "A+a", ... (base, strand, code; the ?/. mode dropped).
/// Returns (codes, expected ML length).
fn mm_codes(mm: &str) -> (Vec<String>, usize) {
    let (mut codes, mut n_ml) = (Vec::new(), 0usize);
    for block in mm.split(';').map(str::trim).filter(|b| !b.is_empty()) {
        let (head, rest) = block.split_once(',').unwrap_or((block, ""));
        if head.len() < 3 { continue; }
        let body = head[2..].trim_end_matches(['?', '.']);
        let n_codes = if body.chars().all(|c| c.is_ascii_digit()) { codes.push(format!("{}{}", &head[..2], body)); 1 }
                      else { for c in body.chars() { codes.push(format!("{}{c}", &head[..2])); } body.len().max(1) };
        n_ml += rest.split(',').filter(|d| !d.is_empty()).count() * n_codes;
    }
    (codes, n_ml)
}

pub fn inspect_bam(path: &Path, n_records: u64, available_maps: &[&'static Assembly]) -> Result<Inspection> {
    let mut r = Inspection::default();
    let mut magic = [0u8; 4];
    let mut f = std::fs::File::open(path)?;
    let got = f.read(&mut magic)?;
    if got >= 4 && &magic == b"CRAM" {
        r.push(Level::Fatal, "CRAM", "the input is a CRAM file", "convert it to BAM (samtools view -b -T chm13v2.0.fa in.cram > out.bam) and index it");
        return Ok(r);
    }
    let (mut reader, header) = match bamio::Records::open(path) {
        Ok(x) => x,
        Err(e) => { r.push(Level::Fatal, "NOT_BAM", format!("could not read a BAM header ({:#})", e), "the file must be a BAM (BGZF-compressed, BAM\\1 magic)"); return Ok(r); }
    };
    let refs = header.refs.clone();
    r.n_contigs = refs.len();
    for pg in header.lines("PG") {
        if let Some(pn) = pg.get("PN") {
            let p = format!("{pn}{}", pg.get("VN").map(|v| format!(" {v}")).unwrap_or_default());
            if !r.programs.contains(&p) { r.programs.push(p); }
        }
    }
    r.sort_order = header.field("HD", "SO").unwrap_or_else(|| "unknown".into());

    // --- reference and naming (by contig length)
    let resolved = match resolve(&refs, available_maps) {
        Ok(x) => x,
        Err(issues) => {
            r.reference = if refs.is_empty() { "none (unaligned)".into() } else { "unsupported".into() };
            r.issues.extend(issues); return Ok(r);
        }
    };
    r.reference = resolved.assembly.name.into(); r.probe_map = resolved.assembly.probe_file.into();
    r.naming = resolved.naming.clone(); r.probe_contigs_present = resolved.contigs.len();
    if !resolved.missing.is_empty() {
        r.push(Level::Warn, "MISSING_CONTIGS", format!("{} of 22 probe-bearing autosomes are absent from the header: {}", resolved.missing.len(), resolved.missing.join(", ")),
               "probes on those chromosomes cannot be covered; was the BAM subset to some chromosomes?");
    }
    r.push(Level::Info, "REFERENCE", format!("{}, contigs {}, {}/22 probe autosomes present", r.reference, r.naming, r.probe_contigs_present), "");
    let probe_names: std::collections::HashSet<String> = resolved.contigs.values().cloned().collect();

    // --- index
    if let Some(ip) = bamio::index_path(path) {
        {
            match bamio::read_index(&ip) {
                Ok(index) => {
                    let rs = &index.refs;
                    let mut info = IndexInfo { n_references: rs.len(), mapped_on_probe_contigs: None, unmapped: index.n_no_coor };
                    if rs.len() != refs.len() {
                        r.push(Level::Warn, "INDEX_MISMATCH", format!("the index lists {} references, the BAM header {}: the index belongs to a different BAM, so the BAM is read without it", rs.len(), refs.len()),
                               "rebuild it: samtools index in.bam");
                    } else {
                        let mut mapped = 0u64; let mut have_meta = false; let mut max_off = 0u64;
                        for (i, (n, _)) in refs.iter().enumerate() {
                            if let Some((_, end, n_mapped, _)) = rs[i].meta {
                                have_meta = true; max_off = max_off.max(end >> 16);
                                if probe_names.contains(n) { mapped += n_mapped; }
                            }
                        }
                        let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(u64::MAX);
                        if have_meta && max_off > size {
                            r.push(Level::Warn, "INDEX_MISMATCH", format!("the index points {} MB past the end of the BAM: it belongs to a different (or since-rewritten) BAM, so the BAM is read without it", (max_off - size) / 1_000_000),
                                   "rebuild it: samtools index in.bam");
                        } else if have_meta {
                            // only a consistent index's read counts are evidence
                            info.mapped_on_probe_contigs = Some(mapped);
                            if mapped == 0 { r.push(Level::Fatal, "NO_MAPPED_READS", "the index records no mapped reads on chr1-22", "check the alignment; the BAM may hold only unmapped reads"); }
                        }
                    }
                    r.index = Some(info);
                }
                Err(e) => r.push(Level::Warn, "INDEX_UNREADABLE", format!("could not read {} ({e:#}), so the BAM is read without it", ip.display()), "rebuild it: samtools index in.bam"),
            }
        }
    } else {
        r.push(Level::Info, "NO_INDEX", "no .bai next to the BAM: one linear pass instead of parallel shards", "samtools index in.bam");
    }

    // --- reads: modification tags
    let s = &mut r.sample;
    let mut buf = Vec::new();
    loop {
        match reader.next(&mut buf) {
            Ok(true) => {}
            Ok(false) => break,
            Err(e) => { crate::warn::warn("BAM_READ", format!("{}: reading stopped after {} records ({e:#})", path.display(), s.records)); break; }
        }
        let rec = Record(&buf);
        s.records += 1;
        if !(rec.is_unmapped() || rec.is_secondary() || rec.is_supplementary()) {
            s.primary_mapped += 1;
            if !bamio::aux_ok(rec.aux()) {
                crate::warn::warn("BAM_AUX", format!("{}: a sampled read ({} primary mapped so far) has malformed optional fields (tags); reads like it are skipped", path.display(), s.primary_mapped));
            }
            // the first field with each tag (MM before Mm, ML before Ml)
            let field = |a: &[u8; 2], b: &[u8; 2]| { let fs: Vec<_> = bamio::aux_fields(rec.aux()).collect(); fs.iter().position(|f| f.tag == *a).or_else(|| fs.iter().position(|f| f.tag == *b)).map(|i| (fs[i].ty, fs[i].value.to_vec())) };
            let mm = match field(b"MM", b"Mm") { Some((b'Z', v)) => Some(String::from_utf8_lossy(&v).into_owned()), _ => None };
            let ml_len = match field(b"ML", b"Ml") { Some((b'B', v)) if v.first() == Some(&b'C') => Some(v.len() - 5), Some((b'B', _)) => Some(usize::MAX), _ => None };
            if let Some(mm) = mm {
                s.with_mm += 1;
                let (codes, n_ml) = mm_codes(&mm);
                for c in codes { *s.mod_codes.entry(c).or_default() += 1; }
                if let Some(n) = ml_len { s.with_ml += 1; if n != n_ml { s.ml_length_mismatch += 1; } }
                if let Some(mn) = bamio::aux_fields(rec.aux()).find(|f| f.tag == *b"MN").and_then(|f| f.int()) { if mn as usize != rec.l_seq() { s.mn_mismatch += 1; } }
            }
        }
        if s.primary_mapped >= n_records { break; }
    }
    let (n, mm, ml) = (s.primary_mapped, s.with_mm, s.with_ml);
    let codes: Vec<String> = s.mod_codes.keys().cloned().collect();
    let has_5mc = s.mod_codes.keys().any(|k| k.starts_with("C+m") || k == "C+C");
    let has_5hmc = s.mod_codes.keys().any(|k| k.starts_with("C+h"));
    let (mlx, mnx) = (s.ml_length_mismatch, s.mn_mismatch);
    if n == 0 {
        r.push(Level::Fatal, "NO_MAPPED_READS", format!("no primary mapped reads among the first {} records", r.sample.records), "check the alignment");
    } else if mm == 0 {
        r.push(Level::Fatal, "NO_MOD_TAGS", format!("none of the first {n} mapped reads has MM/ML base-modification tags"),
               "basecall with a 5mC model (dorado basecaller ...,5mCG_5hmCG or 5mC_5hmC) and keep the tags through alignment: dorado aligner keeps them; with minimap2 use samtools fastq -T MM,ML and minimap2 -y");
    } else if !has_5mc && !has_5hmc {
        r.push(Level::Fatal, "NO_5MC", format!("MM tags have no cytosine methylation calls (codes: {})", codes.join(", ")), "basecall with a 5mC model (e.g. dorado ...,5mCG_5hmCG)");
    } else if ml == 0 {
        r.push(Level::Fatal, "NO_ML", "MM tags are present but ML (modification probabilities) is missing", "re-export keeping both MM and ML");
    } else {
        if !has_5mc { r.push(Level::Warn, "ONLY_5HMC", "cytosine calls are 5hmC only; optic was trained on 5mC (+5hmC) calls", "basecall with a 5mC model"); }
        if (mm as f64) < 0.9 * n as f64 {
            r.push(Level::Warn, "PARTIAL_MOD_TAGS", format!("only {mm} of {n} sampled mapped reads have MM/ML ({:.0}%)", 100.0 * mm as f64 / n as f64),
                   "a merge of tagged and untagged BAMs, or tags dropped for some reads; coverage will be lower than read depth suggests");
        }
        if mlx as f64 > 0.05 * mm as f64 { r.push(Level::Warn, "ML_MISMATCH", format!("{mlx} of {mm} reads have an ML length that does not match MM"), "tags may be corrupted; re-export from the basecaller output"); }
        if mnx as f64 > 0.05 * mm as f64 {
            r.push(Level::Warn, "MN_MISMATCH", format!("{mnx} of {mm} reads have MN != sequence length: MM/ML refer to a different (untrimmed/unclipped) sequence"),
                   "tags were not updated after trimming or hard clipping; realign from the basecaller BAM with tag-aware tools");
        }
        r.push(Level::Info, "MOD_TAGS", format!("{mm} of {n} sampled mapped reads tagged; codes {}", codes.join(", ")), "");
    }
    // damage met while sampling the reads (recovered from; see `warn`)
    for (code, msg) in crate::warn::take() {
        r.push(Level::Warn, code, msg, "the BAM is damaged: affected reads are skipped; re-copy or re-create it (samtools quickcheck -v in.bam)");
    }
    r.issues.sort_by(|a, b| b.level.cmp(&a.level));
    Ok(r)
}

pub fn coverage_tier(n_covered: usize) -> (&'static str, Level, String) {
    match n_covered {
        0 => ("none", Level::Fatal, "no probes covered".into()),
        1..=9999 => ("low", Level::Warn,
            format!("{n_covered} probes covered. >10,000 recommended")),
        _ => ("adequate", Level::Info, format!("{n_covered} probes covered")),
    }
}

pub fn render(r: &Inspection) -> String {
    let mut s = String::new();
    if !r.reference.is_empty() { s += &format!("reference : {}{}\n", r.reference, if r.naming.is_empty() { String::new() } else { format!("  (contigs {}, {}/22 probe autosomes; probe map {})", r.naming, r.probe_contigs_present, r.probe_map) }); }
    if let Some(i) = &r.index { s += &format!("index     : {} references, {} mapped reads on chr1-22\n", i.n_references, i.mapped_on_probe_contigs.map(|m| m.to_string()).unwrap_or("?".into())); }
    if r.sample.records > 0 { s += &format!("sample    : {} records, {} primary mapped, {} with MM, {} with ML\n", r.sample.records, r.sample.primary_mapped, r.sample.with_mm, r.sample.with_ml); }
    if !r.programs.is_empty() { s += &format!("programs  : {}\n", r.programs.join("; ")); }
    for i in &r.issues {
        s += &format!("  {:5} {:18} {}\n", match i.level { Level::Fatal => "FATAL", Level::Warn => "warn", Level::Info => "info" }, i.code, i.message);
        if !i.hint.is_empty() && i.level != Level::Info { s += &format!("        {:18} -> {}\n", "", i.hint); }
    }
    s
}
