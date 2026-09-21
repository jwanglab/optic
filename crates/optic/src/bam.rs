use anyhow::{Context, Result};
use noodles::bam;
use noodles::bgzf;
use noodles::core::Region;
use noodles::sam::alignment::record::cigar::op::Kind;
use noodles::sam::alignment::record::data::field::Value;
use noodles::sam::alignment::record::data::field::value::Array;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
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

struct ParsedRecord {
    chrom_idx: usize,
    ref_start: u64,
    is_reverse: bool,
    mm_str: String,
    ml_values: Vec<u8>,
    seq_bytes: Vec<u8>,
    cigar_ops: Vec<(Kind, usize)>,
}

fn process_record(
    rec: &ParsedRecord,
    ref_names: &[String],
    probe_positions: &HashMap<&str, HashSet<u64>>,
    probe_map: &HashMap<(String, u64), String>,
    threshold_low: f32,
    threshold_high: f32,
) -> (Vec<(String, bool)>, u64) {
    let chrom = &ref_names[rec.chrom_idx];
    let chrom_probes = match probe_positions.get(chrom.as_str()) {
        Some(p) => p,
        None => return (Vec::new(), 0),
    };

    // Build query->ref mapping from CIGAR
    let mut q2r: HashMap<usize, u64> = HashMap::new();
    let mut q_pos: usize = 0;
    let mut r_pos = rec.ref_start;
    for &(kind, len) in &rec.cigar_ops {
        match kind {
            Kind::Match | Kind::SequenceMatch | Kind::SequenceMismatch => {
                for i in 0..len {
                    q2r.insert(q_pos + i, r_pos + i as u64);
                }
                q_pos += len;
                r_pos += len as u64;
            }
            Kind::Insertion | Kind::SoftClip => q_pos += len,
            Kind::Deletion | Kind::Skip => r_pos += len as u64,
            Kind::HardClip | Kind::Pad => {}
        }
    }

    let mod_groups = parse_mm_tag(&rec.mm_str);
    let mut mod_probs: HashMap<usize, f32> = HashMap::new();
    let mut all_eligible_c: HashSet<usize> = HashSet::new();
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
        let n_seq = rec.seq_bytes.len();
        let orig_base = |i: usize| -> u8 {
            if rec.is_reverse {
                match rec.seq_bytes[n_seq - 1 - i] { b'A' => b'T', b'T' => b'A', b'C' => b'G', b'G' => b'C', o => o }
            } else { rec.seq_bytes[i] }
        };
        let to_stored = |i: usize| -> usize { if rec.is_reverse { n_seq - 1 - i } else { i } };
        let eligible: Vec<usize> = (0..n_seq).filter(|&i| orig_base(i) == group.canonical_base).collect();
        for &pos in &eligible {
            all_eligible_c.insert(to_stored(pos));
        }
        let mut eligible_idx: usize = 0;
        for (delta_i, &delta) in group.deltas.iter().enumerate() {
            eligible_idx += delta;
            if eligible_idx < eligible.len() {
                let seq_pos = to_stored(eligible[eligible_idx]);
                for mc in 0..group.num_mod_codes {
                    let ml_idx = ml_offset + delta_i * group.num_mod_codes + mc;
                    if ml_idx < rec.ml_values.len() {
                        let prob = rec.ml_values[ml_idx] as f32 / 256.0;
                        *mod_probs.entry(seq_pos).or_insert(0.0) += prob;
                    }
                }
            }
            eligible_idx += 1;
        }
        ml_offset += group.deltas.len() * group.num_mod_codes;
    }

    for prob in mod_probs.values_mut() {
        if *prob > 1.0 {
            *prob = 1.0;
        }
    }

    let positions_to_check: Vec<usize> = if any_implicit {
        all_eligible_c.into_iter().collect()
    } else {
        mod_probs.keys().cloned().collect()
    };

    let mut calls: Vec<(String, bool)> = Vec::new();
    let mut filtered: u64 = 0;

    for seq_pos in positions_to_check {
        let ref_pos = match q2r.get(&seq_pos) {
            Some(&rp) => rp,
            None => continue,
        };
        let adjusted_pos = if rec.is_reverse { ref_pos.saturating_sub(1) } else { ref_pos };
        if !chrom_probes.contains(&adjusted_pos) {
            continue;
        }
        let cg_id = match probe_map.get(&(chrom.to_string(), adjusted_pos)) {
            Some(id) => id,
            None => continue,
        };
        let combined_prob = mod_probs.get(&seq_pos).copied().unwrap_or(0.0);
        if combined_prob >= threshold_high {
            calls.push((cg_id.clone(), true));
        } else if combined_prob < threshold_low {
            calls.push((cg_id.clone(), false));
        } else {
            filtered += 1;
        }
    }

    (calls, filtered)
}

/// Process a BAM into (cg_id, beta) pairs. Uses indexed region-querying if a `.bai`
/// exists alongside the BAM; otherwise falls back to a multi-threaded linear scan.
pub fn process_bam(
    bam_path: &Path,
    probe_map: &HashMap<(String, u64), String>,
    threshold_low: f32,
    threshold_high: f32,
    threads: usize,
) -> Result<Vec<(String, f32)>> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .ok();

    if has_index(bam_path) {
        eprintln!("[optic] BAM index detected — using parallel region-querying ({} workers)", threads);
        process_bam_indexed(bam_path, probe_map, threshold_low, threshold_high, threads)
    } else {
        eprintln!("[optic] no BAM index found; falling back to multi-threaded linear scan \
                   ({} BGZF workers; run `samtools index` for better speed)", threads);
        process_bam_linear(bam_path, probe_map, threshold_low, threshold_high, threads)
    }
}

fn has_index(bam_path: &Path) -> bool {
    let bai = PathBuf::from(format!("{}.bai", bam_path.display()));
    let csi = PathBuf::from(format!("{}.csi", bam_path.display()));
    bai.exists() || csi.exists()
}

/// Group probe positions into merged intervals for efficient region-querying.
/// Adjacent positions within `merge_dist` bp are coalesced into a single interval.
fn build_intervals_from_positions(
    by_chrom: &HashMap<String, HashSet<u64>>,
    merge_dist: u64,
) -> Vec<(String, u64, u64)> {
    // BTreeMap-iterate the chroms for deterministic interval order.
    let sorted: BTreeMap<&String, &HashSet<u64>> = by_chrom.iter().collect();
    let mut out = Vec::new();
    for (chrom, positions) in sorted {
        let mut positions: Vec<u64> = positions.iter().copied().collect();
        positions.sort();
        if positions.is_empty() {
            continue;
        }
        let mut start = positions[0];
        let mut end = positions[0] + 1;
        for &p in &positions[1..] {
            if p <= end + merge_dist {
                end = p + 1;
            } else {
                out.push((chrom.clone(), start, end));
                start = p;
                end = p + 1;
            }
        }
        out.push((chrom.clone(), start, end));
    }
    out
}

/// Linear-scan implementation with multi-threaded BGZF decompression.
fn process_bam_linear(
    bam_path: &Path,
    probe_map: &HashMap<(String, u64), String>,
    threshold_low: f32,
    threshold_high: f32,
    bgzf_workers: usize,
) -> Result<Vec<(String, f32)>> {
    let mut probe_positions: HashMap<&str, HashSet<u64>> = HashMap::new();
    for (chrom, pos) in probe_map.keys() {
        probe_positions
            .entry(chrom.as_str())
            .or_default()
            .insert(*pos);
    }

    let file = std::fs::File::open(bam_path)
        .with_context(|| format!("opening BAM {}", bam_path.display()))?;
    let workers = NonZeroUsize::new(bgzf_workers.max(1)).unwrap();
    let bgzf_reader = bgzf::io::MultithreadedReader::with_worker_count(workers, file);
    let mut reader = bam::io::Reader::from(bgzf_reader);
    let header = reader.read_header()?;

    let ref_names: Vec<String> = header
        .reference_sequences()
        .keys()
        .map(|name| name.to_string())
        .collect();

    eprintln!("[optic] BAM threads: {}", rayon::current_num_threads());

    let mut counts: HashMap<String, (u64, u64)> = HashMap::new();
    let n_records = AtomicU64::new(0);
    let n_with_mods = AtomicU64::new(0);
    let n_filtered = AtomicU64::new(0);

    let mm_tag_key: [u8; 2] = [b'M', b'M'];
    let mm_tag_lower: [u8; 2] = [b'M', b'm'];
    let ml_tag_key: [u8; 2] = [b'M', b'L'];
    let ml_tag_lower: [u8; 2] = [b'M', b'l'];

    let start_time = std::time::Instant::now();
    let mut last_report = start_time;
    let mut total_read: u64 = 0;

    loop {
        let mut batch: Vec<ParsedRecord> = Vec::with_capacity(BATCH_SIZE);

        for result in reader.records() {
            let record = result?;
            total_read += 1;

            let flags = record.flags();
            if flags.is_unmapped() || flags.is_secondary() || flags.is_supplementary() {
                continue;
            }
            let ref_id = match record.reference_sequence_id() {
                Some(Ok(id)) => id,
                _ => continue,
            };
            if ref_id >= ref_names.len() {
                continue;
            }
            let ref_start = match record.alignment_start() {
                Some(Ok(pos)) => pos.get() as u64 - 1,
                _ => continue,
            };
            let is_reverse = flags.is_reverse_complemented();

            let data = record.data();
            let mm_str: String = {
                let val = data.get(&mm_tag_key).or_else(|| data.get(&mm_tag_lower));
                match val {
                    Some(Ok(Value::String(s))) => String::from_utf8_lossy(s.as_ref()).into_owned(),
                    _ => continue,
                }
            };
            let ml_values: Vec<u8> = {
                let val = data.get(&ml_tag_key).or_else(|| data.get(&ml_tag_lower));
                match val {
                    Some(Ok(Value::Array(Array::UInt8(values)))) => {
                        values.iter().filter_map(|r| r.ok()).collect()
                    }
                    _ => Vec::new(),
                }
            };

            let seq = record.sequence();
            let seq_bytes: Vec<u8> = (0..seq.len()).map(|i| seq.get(i).unwrap_or(b'N')).collect();
            let cigar_ops: Vec<(Kind, usize)> = record
                .cigar()
                .iter()
                .filter_map(|r| r.ok())
                .map(|op| (op.kind(), op.len()))
                .collect();

            batch.push(ParsedRecord {
                chrom_idx: ref_id,
                ref_start,
                is_reverse,
                mm_str,
                ml_values,
                seq_bytes,
                cigar_ops,
            });

            if batch.len() >= BATCH_SIZE {
                break;
            }
        }

        if batch.is_empty() {
            break;
        }
        let batch_len = batch.len() as u64;

        let batch_results: Vec<(Vec<(String, bool)>, u64)> = batch
            .par_iter()
            .map(|rec| {
                process_record(rec, &ref_names, &probe_positions, probe_map,
                               threshold_low, threshold_high)
            })
            .collect();

        for (calls, filt) in batch_results {
            n_filtered.fetch_add(filt, Ordering::Relaxed);
            for (cg_id, is_modified) in calls {
                let entry = counts.entry(cg_id).or_insert((0, 0));
                if is_modified { entry.0 += 1; } else { entry.1 += 1; }
            }
        }

        n_records.fetch_add(total_read, Ordering::Relaxed);
        n_with_mods.fetch_add(batch_len, Ordering::Relaxed);

        let now = std::time::Instant::now();
        if now.duration_since(last_report).as_secs() >= 10 {
            let elapsed = now.duration_since(start_time).as_secs();
            let nr = n_records.load(Ordering::Relaxed);
            let nm = n_with_mods.load(Ordering::Relaxed);
            let nf = n_filtered.load(Ordering::Relaxed);
            eprintln!(
                "[optic] [{:>4}s] {:>10} records | {} with mods | {} CpGs | {} filtered",
                elapsed, nr, nm, counts.len(), nf
            );
            last_report = now;
        }
        total_read = 0;
    }

    let elapsed = start_time.elapsed().as_secs();
    let nr = n_records.load(Ordering::Relaxed);
    let nm = n_with_mods.load(Ordering::Relaxed);
    let nf = n_filtered.load(Ordering::Relaxed);
    eprintln!(
        "[optic] BAM done: {} records, {} with mod tags, {} unique CpGs, {} calls filtered ({}s)",
        nr, nm, counts.len(), nf, elapsed
    );

    finalize_counts(counts)
}

/// Indexed-query implementation with parallel workers.
/// Each worker opens its own `IndexedReader` and processes a partition of intervals.
fn process_bam_indexed(
    bam_path: &Path,
    probe_map: &HashMap<(String, u64), String>,
    threshold_low: f32,
    threshold_high: f32,
    n_workers: usize,
) -> Result<Vec<(String, f32)>> {
    let header_for_refs = bam::io::indexed_reader::Builder::default()
        .build_from_path(bam_path)
        .with_context(|| format!("opening indexed BAM {}", bam_path.display()))?
        .read_header()?;
    let ref_names: Arc<Vec<String>> = Arc::new(
        header_for_refs
            .reference_sequences()
            .keys()
            .map(|n| n.to_string())
            .collect(),
    );
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

    let intervals = build_intervals_from_positions(&probe_positions, 5_000);
    let detected_form = if probe_positions.keys().any(|k| k.starts_with("chr")) {
        "chr-prefixed"
    } else if probe_positions.keys().any(|k| k.starts_with("NC_")) {
        "accession"
    } else {
        "mixed/unknown"
    };
    eprintln!(
        "[optic] {} probes -> {} merged intervals (5 kb gap); BAM chromosome naming: {}",
        unique_cgs.len(),
        intervals.len(),
        detected_form,
    );

    let probe_map_arc = Arc::new(probe_map.clone());
    let probe_positions_arc = Arc::new(probe_positions);

    // Partition intervals across workers (round-robin keeps load balanced when
    // some chroms have many more probes than others)
    let n_workers = n_workers.max(1);
    let mut partitions: Vec<Vec<(String, u64, u64)>> = (0..n_workers).map(|_| Vec::new()).collect();
    for (i, iv) in intervals.iter().enumerate() {
        partitions[i % n_workers].push(iv.clone());
    }

    let n_records = Arc::new(AtomicU64::new(0));
    let n_with_mods = Arc::new(AtomicU64::new(0));
    let n_filtered = Arc::new(AtomicU64::new(0));
    let start_time = std::time::Instant::now();

    // Spawn workers
    let bam_path_owned = bam_path.to_path_buf();
    let mut handles = Vec::with_capacity(n_workers);
    for partition in partitions {
        if partition.is_empty() {
            continue;
        }
        let bam_path = bam_path_owned.clone();
        let ref_names = ref_names.clone();
        let probe_map = probe_map_arc.clone();
        let probe_positions = probe_positions_arc.clone();
        let n_records = n_records.clone();
        let n_with_mods = n_with_mods.clone();
        let n_filtered = n_filtered.clone();

        let handle = std::thread::spawn(move || -> Result<HashMap<String, (u64, u64)>> {
            let mut reader = bam::io::indexed_reader::Builder::default()
                .build_from_path(&bam_path)?;
            let header = reader.read_header()?;
            let mut local_counts: HashMap<String, (u64, u64)> = HashMap::new();

            let mm_tag_key: [u8; 2] = [b'M', b'M'];
            let mm_tag_lower: [u8; 2] = [b'M', b'm'];
            let ml_tag_key: [u8; 2] = [b'M', b'L'];
            let ml_tag_lower: [u8; 2] = [b'M', b'l'];

            for (chrom, start, end) in &partition {
                if !ref_names.iter().any(|n| n == chrom) {
                    continue;
                }
                let region_str = format!("{}:{}-{}", chrom, start + 1, end);
                let region: Region = match region_str.parse() {
                    Ok(r) => r,
                    Err(_) => continue,
                };
                let query = match reader.query(&header, &region) {
                    Ok(q) => q,
                    Err(_) => continue,
                };

                // Per-interval probe view (only positions in [start, end))
                let interval_probe_positions: HashSet<u64> = probe_positions
                    .get(chrom)
                    .map(|s| s.iter().filter(|&&p| p >= *start && p < *end).copied().collect())
                    .unwrap_or_default();
                if interval_probe_positions.is_empty() {
                    continue;
                }
                let mut interval_view: HashMap<&str, HashSet<u64>> = HashMap::new();
                interval_view.insert(chrom.as_str(), interval_probe_positions);

                for result in query.records() {
                    let record = match result {
                        Ok(r) => r,
                        Err(_) => continue,
                    };
                    n_records.fetch_add(1, Ordering::Relaxed);

                    let flags = record.flags();
                    if flags.is_unmapped() || flags.is_secondary() || flags.is_supplementary() {
                        continue;
                    }
                    let ref_id = match record.reference_sequence_id() {
                        Some(Ok(id)) => id,
                        _ => continue,
                    };
                    if ref_id >= ref_names.len() {
                        continue;
                    }
                    let ref_start = match record.alignment_start() {
                        Some(Ok(pos)) => pos.get() as u64 - 1,
                        _ => continue,
                    };
                    let is_reverse = flags.is_reverse_complemented();

                    let data = record.data();
                    let mm_str: String = {
                        let val = data.get(&mm_tag_key).or_else(|| data.get(&mm_tag_lower));
                        match val {
                            Some(Ok(Value::String(s))) => {
                                String::from_utf8_lossy(s.as_ref()).into_owned()
                            }
                            _ => continue,
                        }
                    };
                    let ml_values: Vec<u8> = {
                        let val = data.get(&ml_tag_key).or_else(|| data.get(&ml_tag_lower));
                        match val {
                            Some(Ok(Value::Array(Array::UInt8(values)))) => {
                                values.iter().filter_map(|r| r.ok()).collect()
                            }
                            _ => Vec::new(),
                        }
                    };
                    n_with_mods.fetch_add(1, Ordering::Relaxed);

                    let seq = record.sequence();
                    let seq_bytes: Vec<u8> = (0..seq.len())
                        .map(|i| seq.get(i).unwrap_or(b'N'))
                        .collect();
                    let cigar_ops: Vec<(Kind, usize)> = record
                        .cigar()
                        .iter()
                        .filter_map(|r| r.ok())
                        .map(|op| (op.kind(), op.len()))
                        .collect();

                    let parsed = ParsedRecord {
                        chrom_idx: ref_id,
                        ref_start,
                        is_reverse,
                        mm_str,
                        ml_values,
                        seq_bytes,
                        cigar_ops,
                    };

                    let (calls, filt) = process_record(
                        &parsed, &ref_names, &interval_view, &probe_map,
                        threshold_low, threshold_high,
                    );
                    n_filtered.fetch_add(filt, Ordering::Relaxed);
                    for (cg_id, is_modified) in calls {
                        let entry = local_counts.entry(cg_id).or_insert((0, 0));
                        if is_modified { entry.0 += 1; } else { entry.1 += 1; }
                    }
                }
            }
            Ok(local_counts)
        });
        handles.push(handle);
    }

    // Merge per-thread counts
    let mut counts: HashMap<String, (u64, u64)> = HashMap::new();
    for h in handles {
        let local = h.join().map_err(|_| anyhow::anyhow!("worker panicked"))??;
        for (cg, (m, u)) in local {
            let entry = counts.entry(cg).or_insert((0, 0));
            entry.0 += m;
            entry.1 += u;
        }
    }

    let elapsed = start_time.elapsed().as_secs();
    let nr = n_records.load(Ordering::Relaxed);
    let nm = n_with_mods.load(Ordering::Relaxed);
    let nf = n_filtered.load(Ordering::Relaxed);
    eprintln!(
        "[optic] BAM done (indexed, {} workers): {} records visited, {} with mods, {} CpGs, {} filtered ({}s)",
        n_workers, nr, nm, counts.len(), nf, elapsed
    );
    finalize_counts(counts)
}

fn finalize_counts(counts: HashMap<String, (u64, u64)>) -> Result<Vec<(String, f32)>> {
    let mut out: Vec<(String, f32)> = counts
        .into_iter()
        .filter_map(|(cg, (m, u))| {
            if m + u == 0 { None } else { Some((cg, m as f32 / (m + u) as f32)) }
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}
