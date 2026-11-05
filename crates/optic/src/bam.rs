use crate::bamio::{self, Record};
use anyhow::{Context, Result};
#[cfg(not(target_family = "wasm"))]
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

const BATCH_SIZE: usize = 10_000;

pub const DEFAULT_THRESHOLD_LOW: f32 = 0.3;
pub const DEFAULT_THRESHOLD_HIGH: f32 = 0.7;

struct ModGroup {
    canonical_base: u8,
    implicit: bool,
    deltas: Vec<usize>,
    num_mod_codes: usize,
}

fn parse_mm_tag(mm: &str) -> Vec<ModGroup> {
    let mut groups = Vec::new();
    for block in mm.split(';') {
        let block = block.trim();
        if block.is_empty() {
            continue;
        }
        let (header, delta_str) = match block.find(',') {
            Some(pos) => (&block[..pos], &block[pos + 1..]),
            None => (block, ""),
        };
        if header.len() < 3 {
            continue;
        }
        let canonical_base = header.as_bytes()[0];
        let mod_part = &header[2..];
        let implicit = !mod_part.contains('?');
        let num_mod_codes = mod_part
            .chars()
            .filter(|c| *c != '?' && *c != '.')
            .count()
            .max(1);
        let deltas: Vec<usize> = if delta_str.is_empty() {
            Vec::new()
        } else {
            delta_str
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect()
        };
        groups.push(ModGroup { canonical_base, implicit, deltas, num_mod_codes });
    }
    groups
}

/// relevant view into a mapped record with MM tag - borrows MM, ML, CIGAR without copying, lazy
/// decoding only when needed
struct RecView<'a> {
    chrom_idx: usize,
    ref_start: u64,
    is_reverse: bool,
    /// the MM string and ML values (raw aux bytes)
    mm: &'a [u8],
    ml: &'a [u8],
    /// 4-bit packed bases
    seq: &'a [u8],
    n_seq: usize,
    /// raw CIGAR (the CG-tag one for very long CIGARs)
    cigar: &'a [u8],
}

/// BAM 4-bit base codes, two per byte (high nibble first) -> ASCII.
const BASES: &[u8; 16] = b"=ACMGRSVTWYHKDBN";

fn decode_seq(packed: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n + 1);
    for &b in packed { out.push(BASES[(b >> 4) as usize]); out.push(BASES[(b & 0x0f) as usize]); }
    out.truncate(n);
    out
}

/// CIGAR operations, in BAM's op-code order (0..=8 = MIDNSHP=X).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind { Match, Insertion, Deletion, Skip, SoftClip, HardClip, Pad, SequenceMatch, SequenceMismatch }

/// Raw BAM CIGAR (u32 LE: len << 4 | op) -> ops; unknown op codes are dropped.
fn cigar_ops(raw: &[u8]) -> impl Iterator<Item = (Kind, usize)> + '_ {
    const KINDS: [Kind; 9] = [Kind::Match, Kind::Insertion, Kind::Deletion, Kind::Skip, Kind::SoftClip, Kind::HardClip,
                              Kind::Pad, Kind::SequenceMatch, Kind::SequenceMismatch];
    raw.chunks_exact(4).filter_map(|c| {
        let v = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        KINDS.get((v & 0x0f) as usize).map(|&k| (k, (v >> 4) as usize))
    })
}

/// Scan the raw aux fields for MM (else Mm) and ML (else Ml), the first occurrence of each tag:
/// Some((mm, ml)) when MM/Mm is a Z string (ml empty unless ML/Ml is a B:C array); None when there is no usable MM
/// (absent, another type, or malformed aux data before it).
fn aux_mm_ml(mut d: &[u8]) -> Option<(&[u8], &[u8])> {
    let (mut mm, mut mm_l, mut ml, mut ml_l): (Option<Option<&[u8]>>, Option<Option<&[u8]>>, Option<Option<&[u8]>>, Option<Option<&[u8]>>) = (None, None, None, None);
    fn fixed(t: u8) -> Option<usize> { match t { b'A' | b'c' | b'C' => Some(1), b's' | b'S' => Some(2), b'i' | b'I' | b'f' => Some(4), _ => None } }
    while d.len() >= 3 {
        let (tag, ty) = ([d[0], d[1]], d[2]); d = &d[3..];
        let (val, rest, is_z, is_bc): (&[u8], &[u8], bool, bool) = match ty {
            b'Z' | b'H' => { let e = d.iter().position(|&b| b == 0)?; (&d[..e], &d[e + 1..], ty == b'Z', false) }
            b'B' => {
                if d.len() < 5 { break; }
                let (sub, n) = (d[0], u32::from_le_bytes([d[1], d[2], d[3], d[4]]) as usize);
                let len = n.checked_mul(fixed(sub)?)?;
                if d.len() < 5 + len { break; }
                (&d[5..5 + len], &d[5 + len..], false, sub == b'C')
            }
            t => { let k = fixed(t)?; if d.len() < k { break; } (&d[..k], &d[k..], false, false) }
        };
        match &tag {
            b"MM" if mm.is_none() => mm = Some(is_z.then_some(val)),
            b"Mm" if mm_l.is_none() => mm_l = Some(is_z.then_some(val)),
            b"ML" if ml.is_none() => ml = Some(is_bc.then_some(val)),
            b"Ml" if ml_l.is_none() => ml_l = Some(is_bc.then_some(val)),
            _ => {}
        }
        d = rest;
    }
    let mm = mm.or(mm_l)??;
    Some((mm, ml.or(ml_l).flatten().unwrap_or(&[])))
}

/// Primary mapped reads whose aux fields do not parse (their MM/ML, if any, cannot be read); warned about per run.
static MALFORMED_AUX: AtomicU64 = AtomicU64::new(0);

/// The borrowed view of a record (None: unmapped / secondary / supplementary / unplaced, or no usable MM tag).
fn view(record: Record<'_>, n_refs: usize) -> Option<RecView<'_>> {
    if record.is_unmapped() || record.is_secondary() || record.is_supplementary() { return None; }
    let chrom_idx = match record.ref_id() { id if id >= 0 && (id as usize) < n_refs => id as usize, _ => return None };
    let ref_start = match record.pos() { p if p >= 0 => p as u64, _ => return None };
    let Some((mm, ml)) = aux_mm_ml(record.aux()) else {
        if !bamio::aux_ok(record.aux()) { MALFORMED_AUX.fetch_add(1, Ordering::Relaxed); }
        return None;
    };
    Some(RecView { chrom_idx, ref_start, is_reverse: record.is_reverse(), mm, ml, n_seq: record.l_seq(), seq: record.seq_packed(), cigar: record.cigar() })
}

/// Probe sites per reference contig: (position, probe index), sorted by position; probe names by index.
struct ProbeIndex { names: Vec<String>, by_ref: HashMap<String, Vec<(u64, u32)>> }

impl ProbeIndex {
    fn new(probe_map: &HashMap<(String, u64), String>) -> Self {
        let (mut ids, mut names, mut by_ref): (HashMap<&str, u32>, Vec<String>, HashMap<String, Vec<(u64, u32)>>) = Default::default();
        for ((chrom, pos), cg) in probe_map {
            let i = *ids.entry(cg.as_str()).or_insert_with(|| { names.push(cg.clone()); (names.len() - 1) as u32 });
            by_ref.entry(chrom.clone()).or_default().push((*pos, i));
        }
        for v in by_ref.values_mut() { v.sort_unstable(); }
        ProbeIndex { names, by_ref }
    }
    fn sites(&self, chrom: &str) -> &[(u64, u32)] { self.by_ref.get(chrom).map_or(&[], |v| v.as_slice()) }
}

/// Per-probe (modified, unmodified) tallies by probe index.
struct Tally(Vec<[u64; 2]>);

impl Tally {
    fn new(n: usize) -> Self { Tally(vec![[0, 0]; n]) }
    fn add(&mut self, calls: &[(u32, bool)]) { for &(i, m) in calls { self.0[i as usize][if m { 0 } else { 1 }] += 1; } }
    fn merge(&mut self, o: &Tally) { for (a, b) in self.0.iter_mut().zip(&o.0) { a[0] += b[0]; a[1] += b[1]; } }
    fn covered(&self) -> usize { self.0.iter().filter(|c| c[0] + c[1] > 0).count() }
    fn into_counts(self, names: &[String]) -> Counts {
        self.0.into_iter().enumerate().filter(|(_, c)| c[0] + c[1] > 0).map(|(i, c)| (names[i].clone(), (c[0], c[1]))).collect()
    }
}

/// Calls of one read at the probe sites `sites` (one contig's, sorted; optionally restricted to an interval).
/// Probe-side: only the probes inside the read's reference span are looked at (a few per long read), each mapped to its
/// read base through the CIGAR. Same calls as checking every listed cytosine against the probe set: an aligned reference
/// base maps to exactly one read base, and a reverse-strand read sees the CpG's C at reference position + 1.
fn process_record(
    rec: &RecView,
    sites: &[(u64, u32)],
    threshold_low: f32,
    threshold_high: f32,
) -> (Vec<(u32, bool)>, u64) {
    const NONE: u32 = u32::MAX;
    let n_seq = rec.n_seq;
    let ref_start = rec.ref_start;
    let cigar: &[u8] = rec.cigar;
    let span: u64 = cigar_ops(cigar).map(|(k, n)| match k {
        Kind::Match | Kind::SequenceMatch | Kind::SequenceMismatch | Kind::Deletion | Kind::Skip => n as u64,
        _ => 0,
    }).sum();
    let ref_end = ref_start + span;
    // probes whose CpG C can be read here: forward reads see P, reverse reads see P + 1
    let lo = sites.partition_point(|s| s.0 + 1 < ref_start);
    let hi = sites.partition_point(|s| s.0 < ref_end);
    if lo >= hi { return (Vec::new(), 0); }
    let window = &sites[lo..hi];

    // reference -> read position over the read's span (NONE = deleted / skipped)
    let mut r2q: Vec<u32> = vec![NONE; span as usize];
    let (mut q, mut r) = (0usize, 0usize);
    for (kind, len) in cigar_ops(cigar) {
        match kind {
            Kind::Match | Kind::SequenceMatch | Kind::SequenceMismatch => {
                for i in 0..len { if q + i < n_seq { r2q[r + i] = (q + i) as u32; } }
                q += len; r += len;
            }
            Kind::Insertion | Kind::SoftClip => q += len,
            Kind::Deletion | Kind::Skip => r += len,
            Kind::HardClip | Kind::Pad => {}
        }
    }

    let seq_bytes = decode_seq(rec.seq, n_seq);
    let ml = rec.ml;
    let mod_groups = parse_mm_tag(&String::from_utf8_lossy(rec.mm));
    let mut mod_probs: Vec<f32> = vec![0.0; n_seq];     // summed probability per read position
    let mut has_prob: Vec<bool> = vec![false; n_seq];   // position has an ML entry (the old map's keys)
    let mut eligible_c: Vec<bool> = vec![false; n_seq]; // position is an eligible C in some C group (implicit mode)
    let mut ml_offset: usize = 0;
    let mut any_implicit = false;

    for group in &mod_groups {
        if group.canonical_base != b'C' {
            ml_offset += group.deltas.len() * group.num_mod_codes;
            continue;
        }
        if group.implicit {
            any_implicit = true;
        }
        // MM/ML count canonical bases along the ORIGINAL basecalled read. A reverse alignment stores
        // the reverse complement, so the C's the tag is counting are not the C's of seq_bytes - they
        // are its G's, enumerated from the other end. Walk the original orientation, then map each hit
        // back to a stored index so everything downstream (q2r, the -1 CpG shift) is unchanged.
        let orig_base = |i: usize| -> u8 {
            if rec.is_reverse {
                match seq_bytes[n_seq - 1 - i] { b'A' => b'T', b'T' => b'A', b'C' => b'G', b'G' => b'C', o => o }
            } else { seq_bytes[i] }
        };
        let to_stored = |i: usize| -> usize { if rec.is_reverse { n_seq - 1 - i } else { i } };
        let eligible: Vec<usize> = (0..n_seq).filter(|&i| orig_base(i) == group.canonical_base).collect();
        for &pos in &eligible {
            eligible_c[to_stored(pos)] = true;
        }
        let mut eligible_idx: usize = 0;
        for (delta_i, &delta) in group.deltas.iter().enumerate() {
            eligible_idx += delta;
            if eligible_idx < eligible.len() {
                let seq_pos = to_stored(eligible[eligible_idx]);
                for mc in 0..group.num_mod_codes {
                    let ml_idx = ml_offset + delta_i * group.num_mod_codes + mc;
                    if ml_idx < ml.len() {
                        let prob = ml[ml_idx] as f32 / 256.0;
                        mod_probs[seq_pos] += prob; has_prob[seq_pos] = true;
                    }
                }
            }
            eligible_idx += 1;
        }
        ml_offset += group.deltas.len() * group.num_mod_codes;
    }

    for prob in mod_probs.iter_mut() {
        if *prob > 1.0 {
            *prob = 1.0;
        }
    }


    let mut calls: Vec<(u32, bool)> = Vec::new();
    let mut filtered: u64 = 0;
    for &(p, idx) in window {
        let rp = if rec.is_reverse { p + 1 } else { p };
        if rp < ref_start || rp >= ref_end { continue; }
        let sq = r2q[(rp - ref_start) as usize];
        if sq == NONE { continue; }
        let sq = sq as usize;
        if !(if any_implicit { eligible_c[sq] } else { has_prob[sq] }) { continue; }
        let combined_prob = mod_probs[sq];   // 0.0 where the tag gave no probability, as before
        if combined_prob >= threshold_high {
            calls.push((idx, true));
        } else if combined_prob < threshold_low {
            calls.push((idx, false));
        } else {
            filtered += 1;
        }
    }
    (calls, filtered)
}

/// Process a BAM into (cg_id, beta) pairs. Uses indexed region-querying if a `.bai`
/// exists alongside the BAM; otherwise falls back to a multi-threaded linear scan.
/// Per-probe (modified, unmodified) read counts. Shards of one BAM sum exactly to the whole.
pub type Counts = HashMap<String, (u64, u64)>;

/// One of `n` disjoint parts of a BAM (`index` in 0..n). With an index, a shard owns whole reference sequences,
/// balanced by probe count; without one, a compressed byte range. Either way each read is decoded by exactly one shard.
#[derive(Clone, Copy, Debug)]
pub struct Shard { pub index: usize, pub n: usize }

impl Shard {
    pub const WHOLE: Shard = Shard { index: 0, n: 1 };
}

impl std::str::FromStr for Shard {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        let (i, n) = s.split_once('/').ok_or_else(|| anyhow::anyhow!("shard is i/n, e.g. 0/4"))?;
        let (index, n): (usize, usize) = (i.trim().parse()?, n.trim().parse()?);
        anyhow::ensure!(n >= 1 && index < n, "shard {index}/{n}: need 0 <= i < n");
        Ok(Shard { index, n })
    }
}

pub fn process_bam(
    bam_path: &Path,
    probe_map: &HashMap<(String, u64), String>,
    threshold_low: f32,
    threshold_high: f32,
    threads: usize,
) -> Result<Vec<(String, f32)>> {
    finalize_counts(bam_counts(bam_path, probe_map, threshold_low, threshold_high, threads, Shard::WHOLE)?.0)
}

/// Counts of a BAM or one shard of it. An unindexed BAM's shards are compressed byte ranges and come with their
/// (first, stop) record virtual positions: the shards of one BAM are complete and disjoint iff these chain
/// (`spans_chain`).
pub fn bam_counts(
    bam_path: &Path,
    probe_map: &HashMap<(String, u64), String>,
    threshold_low: f32,
    threshold_high: f32,
    threads: usize,
    shard: Shard,
) -> Result<(Counts, Option<(u64, u64)>)> {
    #[cfg(not(target_family = "wasm"))]
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .ok();

    MALFORMED_AUX.store(0, Ordering::Relaxed);
    let out = bam_counts_inner(bam_path, probe_map, threshold_low, threshold_high, threads, shard);
    let bad = MALFORMED_AUX.swap(0, Ordering::Relaxed);
    if bad > 0 {
        crate::warn::warn("BAM_AUX", format!("{}: {bad} mapped reads have malformed optional fields (tags); their methylation calls could not be read and were skipped", bam_path.display()));
    }
    out
}

/// The BAM's index if it can be used: readable, listing the header's references, and not pointing past the end of the
/// file. Otherwise None, with a warning (the BAM is then read as unindexed: same result, no region queries).
fn usable_index(bam_path: &Path, header: &bamio::Header) -> Option<bamio::Index> {
    let ip = bamio::index_path(bam_path)?;
    let fall_back = "reading the BAM as unindexed instead (rebuild the index: samtools index)";
    let index = match bamio::read_index(&ip) {
        Ok(i) => i,
        Err(e) => { crate::warn::warn("INDEX_UNREADABLE", format!("{}: {e:#}; {fall_back}", ip.display())); return None; }
    };
    if index.refs.len() != header.refs.len() {
        crate::warn::warn("INDEX_MISMATCH", format!("{} lists {} references but the BAM header {}: it belongs to another BAM; {fall_back}",
                                                    ip.display(), index.refs.len(), header.refs.len()));
        return None;
    }
    let len = std::fs::metadata(bam_path).map(|m| m.len()).unwrap_or(u64::MAX);
    if let Some(end) = index.refs.iter().filter_map(|r| r.meta.map(|m| m.1 >> 16)).max().filter(|&e| e > len) {
        crate::warn::warn("INDEX_MISMATCH", format!("{} points {} bytes past the end of the BAM: it belongs to another (or since-rewritten) BAM; {fall_back}",
                                                    ip.display(), end - len));
        return None;
    }
    Some(index)
}

fn bam_counts_inner(
    bam_path: &Path,
    probe_map: &HashMap<(String, u64), String>,
    threshold_low: f32,
    threshold_high: f32,
    threads: usize,
    shard: Shard,
) -> Result<(Counts, Option<(u64, u64)>)> {
    let (_, header) = bamio::open(bam_path)?;
    if let Some(index) = usable_index(bam_path, &header) {
        eprintln!("[optic] BAM index detected — using parallel region-querying ({} workers)", threads);
        Ok((process_bam_indexed(bam_path, &header, index, probe_map, threshold_low, threshold_high, threads, shard)?, None))
    } else if shard.n > 1 {
        eprintln!("[optic] no BAM index: shard {}/{} is a compressed byte range", shard.index, shard.n);
        let pidx = ProbeIndex::new(probe_map);
        let mut tally = Tally::new(pidx.names.len());
        let span = range_shard(bam_path, &pidx, threshold_low, threshold_high, shard, &mut tally, true)?;
        Ok((tally.into_counts(&pidx.names), Some(span)))
    } else {
        #[cfg(not(target_family = "wasm"))]
        return Ok((process_bam_ranges(bam_path, probe_map, threshold_low, threshold_high, threads)?, None));
        #[cfg(target_family = "wasm")]
        {
            eprintln!("[optic] no BAM index found; one linear pass (shard it with --shard i/n, or index it)");
            Ok((process_bam_linear(bam_path, probe_map, threshold_low, threshold_high, threads)?, None))
        }
    }
}

/// True iff byte-range shard spans, in shard order, cover the BAM's records exactly once.
pub fn spans_chain(spans: &[(u64, u64)]) -> bool {
    spans.windows(2).all(|w| w[0].1 == w[1].0) && spans.last().is_some_and(|s| s.1 == SPAN_EOF)
}

/// Unindexed BAM, natively: ~4 byte-range shards per thread, each inflating and piling up its own range. Falls back
/// to the linear scan if the shard boundaries don't chain (never seen; it would take a mis-detected record boundary).
#[cfg(not(target_family = "wasm"))]
fn process_bam_ranges(
    bam_path: &Path,
    probe_map: &HashMap<(String, u64), String>,
    threshold_low: f32,
    threshold_high: f32,
    threads: usize,
) -> Result<Counts> {
    let pidx = ProbeIndex::new(probe_map);
    let n = 4 * threads.max(1);
    eprintln!("[optic] no BAM index: {n} byte-range shards on {threads} threads");
    let t0 = std::time::Instant::now();
    let new = || (Tally::new(pidx.names.len()), Vec::new());
    let (tally, mut spans) = (0..n).into_par_iter()
        .try_fold(new, |(mut t, mut s), i| -> Result<_> {
            let span = range_shard(bam_path, &pidx, threshold_low, threshold_high, Shard { index: i, n }, &mut t, false)?;
            s.push((i, span));
            Ok((t, s))
        })
        .try_reduce(new, |(mut a, mut sa), (b, sb)| { a.merge(&b); sa.extend(sb); Ok((a, sa)) })?;
    spans.sort_unstable();
    if !spans_chain(&spans.iter().map(|s| s.1).collect::<Vec<_>>()) {
        eprintln!("[optic] warning: byte-range shard boundaries do not chain; rerunning as one linear scan");
        return process_bam_linear(bam_path, probe_map, threshold_low, threshold_high, threads);
    }
    eprintln!("[optic] BAM done: {} unique CpGs ({:.1}s)", tally.covered(), t0.elapsed().as_secs_f64());
    Ok(tally.into_counts(&pidx.names))
}


/// One linear pass over the BAM (the wasm build's path for a whole unindexed BAM, and the native fallback).
fn process_bam_linear(
    bam_path: &Path,
    probe_map: &HashMap<(String, u64), String>,
    threshold_low: f32,
    threshold_high: f32,
    _threads: usize,
) -> Result<Counts> {
    let (mut reader, header) = bamio::Records::open(bam_path)?;
    let ref_names: Vec<String> = header.refs.iter().map(|r| r.0.clone()).collect();

    let pidx = ProbeIndex::new(probe_map);
    let ref_sites: Vec<&[(u64, u32)]> = ref_names.iter().map(|n| pidx.sites(n)).collect();
    let mut tally = Tally::new(pidx.names.len());
    let (mut n_records, mut n_with_mods, mut n_filtered) = (0u64, 0u64, 0u64);

    let start_time = std::time::Instant::now();
    let mut last_report = start_time;
    let mut batch: Vec<Vec<u8>> = vec![Vec::new(); BATCH_SIZE];
    loop {
        let mut k = 0;
        while k < BATCH_SIZE && reader.next(&mut batch[k])? { k += 1; }
        if k == 0 { break; }
        n_records += k as u64;

        #[cfg(not(target_family = "wasm"))]
        let batch_iter = batch[..k].par_iter();
        #[cfg(target_family = "wasm")]
        let batch_iter = batch[..k].iter();
        let batch_results: Vec<(Vec<(u32, bool)>, u64)> = batch_iter
            .filter_map(|b| view(Record(b), ref_names.len()))
            .map(|rec| process_record(&rec, ref_sites[rec.chrom_idx], threshold_low, threshold_high))
            .collect();
        n_with_mods += batch_results.len() as u64;
        for (calls, filt) in batch_results {
            n_filtered += filt;
            tally.add(&calls);
        }

        let now = std::time::Instant::now();
        if now.duration_since(last_report).as_secs() >= 10 {
            eprintln!("[optic] [{:>4}s] {:>10} records | {} with mods | {} CpGs | {} filtered",
                      now.duration_since(start_time).as_secs(), n_records, n_with_mods, tally.covered(), n_filtered);
            last_report = now;
        }
        if k < BATCH_SIZE { break; }
    }
    eprintln!("[optic] BAM done: {} records, {} with mod tags, {} unique CpGs, {} calls filtered ({}s)",
              n_records, n_with_mods, tally.covered(), n_filtered, start_time.elapsed().as_secs());
    Ok(tally.into_counts(&pidx.names))
}

/// Indexed-query implementation with parallel workers.
/// Each worker opens its own `IndexedReader` and processes a partition of intervals.
#[allow(clippy::too_many_arguments)]
fn process_bam_indexed(
    bam_path: &Path,
    header: &bamio::Header,
    index: bamio::Index,
    probe_map: &HashMap<(String, u64), String>,
    threshold_low: f32,
    threshold_high: f32,
    n_workers: usize,
    shard: Shard,
) -> Result<Counts> {
    let ref_names: Arc<Vec<String>> = Arc::new(header.refs.iter().map(|r| r.0.clone()).collect());
    let index = Arc::new(index);
    let ref_name_set: HashSet<String> = ref_names.iter().cloned().collect();

    let mut probe_positions: HashMap<String, HashSet<u64>> = HashMap::new();
    let mut unique_cgs: HashSet<&str> = HashSet::new();
    for ((chrom, pos), cg) in probe_map {
        if ref_name_set.contains(chrom) {
            probe_positions
                .entry(chrom.clone())
                .or_default()
                .insert(*pos);
            unique_cgs.insert(cg.as_str());
        }
    }
    if probe_positions.is_empty() {
        return Err(anyhow::anyhow!(
            "no probe chromosomes match BAM header - probes may be on a different reference"
        ));
    }

    let intervals = if shard.n == 1 {
        // contiguous chunks (~16 per worker, equal probe counts) pulled from a queue - a few reads straddle the boundaries and are decoded twice
        chunk_references(&probe_positions, 16 * n_workers.max(1))
    } else {
        shard_references(&probe_positions, shard)
    };
    let detected_form = if probe_positions.keys().any(|k| k.starts_with("chr")) {
        "chr-prefixed"
    } else if probe_positions.keys().any(|k| k.starts_with("NC_")) {
        "accession"
    } else {
        "mixed/unknown"
    };
    eprintln!(
        "[optic] {} probes -> {} {}; BAM chromosome naming: {}",
        unique_cgs.len(),
        intervals.len(),
        if shard.n == 1 { "chunks of whole chromosomes".to_string() } else { format!("whole references in shard {}/{}", shard.index, shard.n) },
        detected_form,
    );

    let probe_positions_arc = Arc::new(probe_positions);
    let pidx = Arc::new(ProbeIndex::new(probe_map));

    // A shared queue of intervals, largest (by probe count) first; each worker pulls the next one when it is free, so a
    // worker stuck on a deep region does not leave the others idle (static partitions balanced by probe count did).
    let n_workers = n_workers.max(1);
    let probes_in = |(c, a, b): &(String, u64, u64)| probe_positions_arc.get(c).map_or(0, |s| s.iter().filter(|&&p| p >= *a && p < *b).count());
    let mut by_size: Vec<(usize, (String, u64, u64))> = intervals.iter().map(|iv| (probes_in(iv), iv.clone())).collect();
    by_size.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let queue: Arc<Vec<(String, u64, u64)>> = Arc::new(by_size.into_iter().map(|(_, iv)| iv).collect());
    let next = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let n_records = Arc::new(AtomicU64::new(0));
    let n_with_mods = Arc::new(AtomicU64::new(0));
    let n_filtered = Arc::new(AtomicU64::new(0));
    let start_time = std::time::Instant::now();

    // Spawn workers
    let bam_path_owned = bam_path.to_path_buf();
    let mut handles = Vec::with_capacity(n_workers);
    for _ in 0..n_workers.min(queue.len()) {
        let (queue, next) = (queue.clone(), next.clone());
        let bam_path = bam_path_owned.clone();
        let ref_names = ref_names.clone();
        let pidx = pidx.clone();
        let n_records = n_records.clone();
        let n_with_mods = n_with_mods.clone();
        let n_filtered = n_filtered.clone();
        let index = index.clone();

        let handle = Worker::spawn(move || -> Result<(Tally, Vec<(String, u64, u64)>)> {
            let (mut reader, _) = bamio::Records::open(&bam_path)?;
            let mut local = Tally::new(pidx.names.len());
            let mut unqueryable = Vec::new();

            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some((chrom, start, end)) = queue.get(i) else { break };
                let Some(rid) = ref_names.iter().position(|n| n == chrom) else { continue };

                // Only this interval's probes (positions in [start, end)), so reads spanning two intervals count once
                let all = pidx.sites(chrom);
                let window = &all[all.partition_point(|x| x.0 < *start)..all.partition_point(|x| x.0 < *end)];
                if window.is_empty() {
                    continue;
                }

                // reads overlapping [start, end + 1): a reverse-strand read sees a CpG's C at its G (P + 1), so a read
                // starting exactly at the last probe's G still counts for it
                let chunks = match index.query(rid, *start, *end + 1) {
                    Ok(c) => c,
                    Err(e) => {
                        crate::warn::warn("INDEX_QUERY", format!("index query {chrom}:{}-{end} failed ({e:#}); this region is read without the index", start + 1));
                        unqueryable.push((chrom.clone(), *start, *end));
                        continue;
                    }
                };
                let n = bamio::query_records(&mut reader, &chunks, rid, *start, *end + 1, |record| {
                    let Some(parsed) = view(record, ref_names.len()) else { return };
                    n_with_mods.fetch_add(1, Ordering::Relaxed);
                    let (calls, filt) = process_record(&parsed, window, threshold_low, threshold_high);
                    n_filtered.fetch_add(filt, Ordering::Relaxed);
                    local.add(&calls);
                }).with_context(|| format!("reading {chrom}:{}-{end} of {}", start + 1, bam_path.display()))?;
                n_records.fetch_add(n, Ordering::Relaxed);
            }
            Ok((local, unqueryable))
        });
        handles.push(handle);
    }

    // Merge per-thread tallies
    let mut tally = Tally::new(pidx.names.len());
    let mut unqueryable = Vec::new();
    for h in handles {
        let (local, u) = h.join().map_err(|_| anyhow::anyhow!("worker panicked"))??;
        tally.merge(&local);
        unqueryable.extend(u);
    }
    if !unqueryable.is_empty() {
        // regions the index cannot address: one linear pass over the BAM for just those
        let (mut reader, _) = bamio::Records::open(bam_path)?;
        let mut buf = Vec::new();
        while reader.next(&mut buf)? {
            let Some(rec) = view(Record(&buf), ref_names.len()) else { continue };
            for (chrom, start, end) in unqueryable.iter().filter(|u| u.0 == ref_names[rec.chrom_idx]) {
                let all = pidx.sites(chrom);
                let window = &all[all.partition_point(|x| x.0 < *start)..all.partition_point(|x| x.0 < *end)];
                tally.add(&process_record(&rec, window, threshold_low, threshold_high).0);
            }
        }
    }
    let counts = tally.into_counts(&pidx.names);

    let elapsed = start_time.elapsed().as_secs();
    let nr = n_records.load(Ordering::Relaxed);
    let nm = n_with_mods.load(Ordering::Relaxed);
    let nf = n_filtered.load(Ordering::Relaxed);
    eprintln!(
        "[optic] BAM done (indexed, {} workers): {} records visited, {} with mods, {} CpGs, {} filtered ({}s)",
        n_workers, nr, nm, counts.len(), nf, elapsed
    );
    Ok(counts)
}

/// Shard `shard.index` of `shard.n`: whole references, greedily balanced by probe count (largest first), each as one
/// query spanning its probes, so a read is decoded once.
fn shard_references(probe_positions: &HashMap<String, HashSet<u64>>, shard: Shard) -> Vec<(String, u64, u64)> {
    let mut refs: Vec<(&String, &HashSet<u64>)> = probe_positions.iter().collect();
    refs.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.0.cmp(b.0)));
    let mut load = vec![0usize; shard.n];
    let mut out = Vec::new();
    for (chrom, pos) in refs {
        let bin = (0..shard.n).min_by_key(|&k| (load[k], k)).unwrap();
        load[bin] += pos.len();
        if bin == shard.index {
            out.push((chrom.clone(), *pos.iter().min().unwrap(), pos.iter().max().unwrap() + 1));
        }
    }
    out
}

/// Split each reference's probes into contiguous chunks so there are about `n_chunks` of roughly equal probe count;
/// a chunk spans [first probe, last probe + 1), and the next chunk starts at the next probe.
fn chunk_references(probe_positions: &HashMap<String, HashSet<u64>>, n_chunks: usize) -> Vec<(String, u64, u64)> {
    let total: usize = probe_positions.values().map(|s| s.len()).sum();
    let target = (total / n_chunks.max(1)).max(1);
    let mut out = Vec::new();
    let mut refs: Vec<(&String, &HashSet<u64>)> = probe_positions.iter().collect();
    refs.sort_by(|a, b| a.0.cmp(b.0));
    for (chrom, set) in refs {
        let mut pos: Vec<u64> = set.iter().copied().collect();
        pos.sort_unstable();
        let k = pos.len().div_ceil(target).max(1);
        let per = pos.len().div_ceil(k);
        for c in pos.chunks(per) { out.push((chrom.clone(), c[0], c[c.len() - 1] + 1)); }
    }
    out
}

/// A thread natively; on wasm (no threads) the work runs inline.
enum Worker<T> { Thread(std::thread::JoinHandle<T>), Done(T) }

impl<T: Send + 'static> Worker<T> {
    fn spawn<F: FnOnce() -> T + Send + 'static>(f: F) -> Self {
        if cfg!(target_family = "wasm") { Worker::Done(f()) } else { Worker::Thread(std::thread::spawn(f)) }
    }
    fn join(self) -> std::thread::Result<T> {
        match self { Worker::Thread(h) => h.join(), Worker::Done(t) => Ok(t) }
    }
}

pub fn finalize_counts(counts: Counts) -> Result<Vec<(String, f32)>> {
    let mut out: Vec<(String, f32)> = counts
        .into_iter()
        .filter_map(|(cg, (m, u))| {
            if m + u == 0 { None } else { Some((cg, m as f32 / (m + u) as f32)) }
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

// --- unindexed BAMs split by compressed byte range ----------------------------------------------------------------
// Shard i of n owns the records that start in [b_i, b_{i+1}): b_i = the first BGZF block starting at or after
// len * i / n. A shard finds its first record by checking candidate offsets as a chain of well-formed records, then
// reads on until a record starts at or past b_{i+1}. Each shard reports its (first, stop) virtual positions; adjacent
// shards must agree (stop of i == first of i + 1), which proves no record was dropped or read twice.

pub use crate::bamio::SPAN_EOF;
use crate::bamio::find_block;

/// Byte-range shard of an unindexed BAM: piles its records into `tally`, returns its (first, stop) virtual positions.
fn range_shard(
    bam_path: &Path,
    pidx: &ProbeIndex,
    threshold_low: f32,
    threshold_high: f32,
    shard: Shard,
    tally: &mut Tally,
    log: bool,
) -> Result<(u64, u64)> {
    let open = || std::fs::File::open(bam_path).with_context(|| format!("opening BAM {}", bam_path.display()));
    let len = open()?.metadata()?.len();
    let (mut reader, header) = bamio::Records::open(bam_path)?;
    let header_end = reader.virtual_position();
    let ref_names: Vec<String> = header.refs.iter().map(|r| r.0.clone()).collect();
    let ref_sites: Vec<&[(u64, u32)]> = ref_names.iter().map(|n| pidx.sites(n)).collect();
    let mut f = open()?;
    let mut bound = |i: usize| if i == 0 { Ok(0) } else if i == shard.n { Ok(len) } else { find_block(&mut f, len * i as u64 / shard.n as u64, len) };
    let (b0, b1) = (bound(shard.index)?, bound(shard.index + 1)?);
    let first = if b0 <= header_end >> 16 { header_end } else { reader.first_from(b0)? };
    let mut stop = SPAN_EOF;
    let mut n_records = 0u64;
    if first != SPAN_EOF {
        let mut record = Vec::new();
        if reader.seek(first)? {
            loop {
                let vp = reader.virtual_position();
                if vp >> 16 >= b1 {
                    if reader.has_more() { stop = vp; }
                    break;
                }
                if !reader.next(&mut record)? { break; }
            n_records += 1;
                if let Some(rec) = view(Record(&record), ref_names.len()) {
                    tally.add(&process_record(&rec, ref_sites[rec.chrom_idx], threshold_low, threshold_high).0);
                }
            }
        }
    }
    if log {
        eprintln!("[optic] unindexed shard {}/{}: compressed bytes {b0}..{b1}, {n_records} records", shard.index, shard.n);
    }
    Ok((first, stop))
}
