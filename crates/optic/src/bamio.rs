//! BAM reading
//!
//! Region queries follow the spec's algorithm, as samtools and noodles do: the bins overlapping the region
//! (reg2bins), their chunks, minus those ending before the linear index (BAI) or the bin's loffset (CSI) for the region
//! start, merged; then every record in those chunks whose aligned span overlaps the region.

use crate::bgzf;
use anyhow::{anyhow, ensure, Context, Result};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

pub struct Header {
    /// SAM header text
    pub text: String,
    /// (name, length) of each reference, in id order
    pub refs: Vec<(String, u64)>,
}

impl Header {
    /// Value of `key` on the first `@<rec>` line that has it (e.g. ("HD", "SO")).
    pub fn field(&self, rec: &str, key: &str) -> Option<String> {
        self.lines(rec).find_map(|f| f.get(key).cloned())
    }
    /// Fields of every `@<rec>` line.
    pub fn lines<'a>(&'a self, rec: &'a str) -> impl Iterator<Item = HashMap<String, String>> + 'a {
        self.text.lines().filter(move |l| l.strip_prefix('@').is_some_and(|r| r.starts_with(rec) && r[rec.len()..].starts_with('\t')))
            .map(|l| l.split('\t').skip(1).filter_map(|f| f.split_once(':').map(|(k, v)| (k.to_string(), v.to_string()))).collect())
    }
}

pub fn read_header<R: Read>(r: &mut bgzf::Reader<R>) -> Result<Header> {
    let mut b4 = [0u8; 4];
    r.read_exact(&mut b4).context("reading the BAM magic")?;
    ensure!(&b4 == b"BAM\x01", "not a BAM file (no BAM\\1 magic)");
    let mut i32_ = |r: &mut bgzf::Reader<R>| -> Result<i32> { r.read_exact(&mut b4)?; Ok(i32::from_le_bytes(b4)) };
    let l_text = i32_(r)?;
    ensure!((0..1 << 30).contains(&l_text), "BAM header text length {l_text} is invalid");
    let mut text = vec![0u8; l_text as usize];
    r.read_exact(&mut text)?;
    let text = String::from_utf8_lossy(&text).trim_end_matches('\0').to_string();
    let n_ref = i32_(r)?;
    ensure!((0..1 << 24).contains(&n_ref), "BAM header reference count {n_ref} is invalid");
    let mut refs = Vec::with_capacity(n_ref as usize);
    for _ in 0..n_ref {
        let l_name = i32_(r)?;
        ensure!((1..1 << 16).contains(&l_name), "BAM reference name length {l_name} is invalid");
        let mut name = vec![0u8; l_name as usize];
        r.read_exact(&mut name)?;
        let name = String::from_utf8_lossy(&name[..name.len() - 1]).into_owned();
        let l_ref = i32_(r)?;
        ensure!(l_ref >= 0, "reference {name} has negative length");
        refs.push((name, l_ref as u64));
    }
    Ok(Header { text, refs })
}

pub fn open(path: &Path) -> Result<(bgzf::Reader<std::fs::File>, Header)> {
    let mut r = bgzf::Reader::new(std::fs::File::open(path).with_context(|| format!("opening BAM {}", path.display()))?);
    let h = read_header(&mut r).with_context(|| format!("reading the header of {}", path.display()))?;
    Ok((r, h))
}

/// Read the next record (without its block_size prefix) into `buf`. False at end of file.
pub fn read_record<R: Read>(r: &mut bgzf::Reader<R>, buf: &mut Vec<u8>) -> Result<bool> {
    let mut b4 = [0u8; 4];
    if !r.read_exact_or_eof(&mut b4)? { return Ok(false); }
    let n = u32::from_le_bytes(b4) as usize;
    if n == 0 { return Ok(false); }
    ensure!((32..1 << 30).contains(&n), "BAM record block_size {n} is invalid");
    buf.resize(n, 0);
    r.read_exact(buf).context("BAM ends inside a record")?;
    Ok(true)
}

/// A BAM's records in file order, recovering from damage instead of stopping: a malformed record is skipped, and after
/// a corrupt block (bad CRC, does not inflate) or broken record framing, reading resumes at the first record of a later
/// block that parses as a chain of well-formed records. Each recovery throws a warning
pub struct Records {
    r: bgzf::Reader<std::fs::File>,
    scan: std::fs::File,
    len: u64,
    ref_lens: Vec<u64>,
    name: String,
}

impl Records {
    pub fn open(path: &Path) -> Result<(Records, Header)> {
        let (r, h) = open(path)?;
        let scan = std::fs::File::open(path)?;
        let len = scan.metadata()?.len();
        let ref_lens = h.refs.iter().map(|x| x.1).collect();
        Ok((Records { r, scan, len, ref_lens, name: path.display().to_string() }, h))
    }

    pub fn virtual_position(&self) -> u64 { self.r.virtual_position() }

    /// Whether any data is left to read (unreadable data counts as data).
    pub fn has_more(&mut self) -> bool { self.r.fill_buf().map_or(true, |b| !b.is_empty()) }

    /// Seek to a record start; if the block there is unreadable, recover from it.
    pub fn seek(&mut self, vpos: u64) -> Result<bool> {
        match self.r.seek(vpos) {
            Ok(()) => Ok(true),
            Err(e) => self.recover(vpos, &anyhow::Error::from(e)),
        }
    }

    /// The next well-formed record into `buf`; false at the end of the readable data.
    pub fn next(&mut self, buf: &mut Vec<u8>) -> Result<bool> {
        loop {
            let at = self.r.virtual_position();
            match read_record(&mut self.r, buf) {
                Ok(false) => return Ok(false),
                Ok(true) if Record(buf).layout_ok() => return Ok(true),
                Ok(true) => crate::warn::warn("BAM_RECORD", format!("{}: malformed record at {}:{} (field lengths overrun it) skipped",
                                                                     self.name, at >> 16, at & 0xffff)),
                Err(e) => if !self.recover(at, &e)? { return Ok(false) },
            }
        }
    }

    /// First record at or after compressed offset `block` (SPAN_EOF if none), skipping unreadable blocks.
    pub fn first_from(&mut self, block: u64) -> Result<u64> {
        let mut b = block;
        loop {
            match first_record_from(&mut self.r, b, &self.ref_lens) {
                Ok(v) => return Ok(v),
                Err(e) => {
                    crate::warn::warn("BAM_CORRUPT", format!("{}: no readable record found from compressed offset {b} ({e:#}); trying the next block", self.name));
                    b = find_block(&mut self.scan, b + 1, self.len)?;
                    if b >= self.len { return Ok(SPAN_EOF); }
                }
            }
        }
    }

    /// After a read error at virtual position `at`: resume at the first readable record in a later block.
    fn recover(&mut self, at: u64, e: &anyhow::Error) -> Result<bool> {
        let mut from = (at >> 16) + 1;
        loop {
            let b = find_block(&mut self.scan, from, self.len)?;
            let found = if b >= self.len { Ok(SPAN_EOF) } else { first_record_from(&mut self.r, b, &self.ref_lens) };
            match found {
                Ok(SPAN_EOF) => {
                    crate::warn::warn("BAM_CORRUPT", format!("{}: {e:#} at compressed offset {}; no readable records after it, so the remaining {} compressed bytes are skipped",
                                                             self.name, at >> 16, self.len.saturating_sub(at >> 16)));
                    return Ok(false);
                }
                Ok(v) => {
                    self.r.seek(v)?;
                    crate::warn::warn("BAM_CORRUPT", format!("{}: {e:#} at compressed offset {}; reading resumed at offset {} ({} compressed bytes skipped)",
                                                             self.name, at >> 16, v >> 16, (v >> 16) - (at >> 16)));
                    return Ok(true);
                }
                Err(_) => from = b + 1,
            }
        }
    }
}

/// A raw BAM record (block_size excluded).
#[derive(Clone, Copy)]
pub struct Record<'a>(pub &'a [u8]);

impl<'a> Record<'a> {
    fn i32_at(&self, o: usize) -> i32 { i32::from_le_bytes(self.0[o..o + 4].try_into().unwrap()) }
    fn u16_at(&self, o: usize) -> u16 { u16::from_le_bytes([self.0[o], self.0[o + 1]]) }
    pub fn ref_id(&self) -> i32 { self.i32_at(0) }
    /// 0-based leftmost position (-1 if none)
    pub fn pos(&self) -> i32 { self.i32_at(4) }
    fn l_read_name(&self) -> usize { self.0[8] as usize }
    fn n_cigar_op(&self) -> usize { self.u16_at(12) as usize }
    pub fn flag(&self) -> u16 { self.u16_at(14) }
    pub fn l_seq(&self) -> usize { self.i32_at(16).max(0) as usize }
    fn cigar_at(&self) -> usize { 32 + self.l_read_name() }
    fn seq_at(&self) -> usize { self.cigar_at() + 4 * self.n_cigar_op() }
    fn aux_at(&self) -> usize { self.seq_at() + self.l_seq().div_ceil(2) + self.l_seq() }
    pub fn layout_ok(&self) -> bool { self.0.len() >= 32 && self.i32_at(16) >= 0 && self.aux_at() <= self.0.len() }
    pub fn is_unmapped(&self) -> bool { self.flag() & 0x4 != 0 }
    pub fn is_reverse(&self) -> bool { self.flag() & 0x10 != 0 }
    pub fn is_secondary(&self) -> bool { self.flag() & 0x100 != 0 }
    pub fn is_supplementary(&self) -> bool { self.flag() & 0x800 != 0 }
    /// 4-bit packed sequence, two bases per byte (high nibble first)
    pub fn seq_packed(&self) -> &'a [u8] { &self.0[self.seq_at()..self.seq_at() + self.l_seq().div_ceil(2)] }
    /// Raw aux fields
    pub fn aux(&self) -> &'a [u8] { &self.0[self.aux_at()..] }

    /// Raw CIGAR (u32 LE: len << 4 | op). A record whose CIGAR is too long for the 16-bit op count stores the
    /// placeholder `<l_seq>S<n>N` and the real CIGAR in its CG:B,I tag; that one is returned instead.
    pub fn cigar(&self) -> &'a [u8] {
        let raw = &self.0[self.cigar_at()..self.seq_at()];
        if raw.len() == 8 {
            let op = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().unwrap());
            let (a, b) = (op(0), op(4));
            if a & 0xf == 4 && (a >> 4) as usize == self.l_seq() && b & 0xf == 3 {
                if let Some(cg) = aux_fields(self.aux()).find_map(|f| (f.tag == *b"CG" && f.ty == b'B').then_some(f.value)) {
                    if cg.len() >= 5 && (cg[0] == b'I' || cg[0] == b'i') { return &cg[5..]; }
                }
            }
        }
        raw
    }

    /// Reference bases covered by the alignment (M, D, N, =, X).
    pub fn ref_span(&self) -> u64 {
        self.cigar().chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .filter(|v| matches!(v & 0xf, 0 | 2 | 3 | 7 | 8)).map(|v| (v >> 4) as u64).sum()
    }
}

/// One aux field: tag, type character, and the raw value (for B arrays: subtype, count and data).
pub struct AuxField<'a> { pub tag: [u8; 2], pub ty: u8, pub value: &'a [u8] }

impl AuxField<'_> {
    /// Integer value of an integer-typed field.
    pub fn int(&self) -> Option<i64> {
        let v = self.value;
        Some(match self.ty {
            b'c' => v[0] as i8 as i64, b'C' => v[0] as i64,
            b's' => i16::from_le_bytes([v[0], v[1]]) as i64, b'S' => u16::from_le_bytes([v[0], v[1]]) as i64,
            b'i' => i32::from_le_bytes(v[..4].try_into().ok()?) as i64, b'I' => u32::from_le_bytes(v[..4].try_into().ok()?) as i64,
            _ => return None,
        })
    }
}

fn fixed_width(t: u8) -> Option<usize> { match t { b'A' | b'c' | b'C' => Some(1), b's' | b'S' => Some(2), b'i' | b'I' | b'f' => Some(4), _ => None } }

/// Whether the aux fields parse exactly to the end.
pub fn aux_ok(mut d: &[u8]) -> bool {
    while !d.is_empty() {
        if d.len() < 3 { return false; }
        let rest = &d[3..];
        let n = match d[2] {
            b'Z' | b'H' => rest.iter().position(|&b| b == 0).map(|k| k + 1),
            b'B' => rest.first().and_then(|&t| fixed_width(t)).zip(rest.get(1..5))
                .and_then(|(w, c)| w.checked_mul(u32::from_le_bytes(c.try_into().unwrap()) as usize)).map(|k| k + 5),
            t => fixed_width(t),
        };
        match n { Some(n) if n <= rest.len() => d = &rest[n..], _ => return false }
    }
    true
}

/// The aux fields in order; stops at the first malformed one.
pub fn aux_fields(mut d: &[u8]) -> impl Iterator<Item = AuxField<'_>> {
    std::iter::from_fn(move || {
        if d.len() < 3 { return None; }
        let (tag, ty) = ([d[0], d[1]], d[2]);
        let rest = &d[3..];
        let n = match ty {
            b'Z' | b'H' => rest.iter().position(|&b| b == 0)? + 1,
            b'B' => {
                let w = fixed_width(*rest.first()?)?;
                5 + w.checked_mul(u32::from_le_bytes(rest.get(1..5)?.try_into().ok()?) as usize)?
            }
            t => fixed_width(t)?,
        };
        let value = rest.get(..n)?;
        d = &rest[n..];
        Some(AuxField { tag, ty, value: if ty == b'Z' || ty == b'H' { &value[..n - 1] } else { value } })
    })
}

// ---- index --------------------------------------------------------------------------------------------------------

/// Per-reference index content.
#[derive(Default)]
pub struct RefIndex {
    /// bin -> chunks (start, end) as virtual positions
    pub bins: HashMap<u32, Vec<(u64, u64)>>,
    /// CSI: bin -> loffset
    loffsets: HashMap<u32, u64>,
    /// BAI: 16 kbp-window linear index
    linear: Vec<u64>,
    /// the metadata pseudo-bin: (ref_beg, ref_end, n_mapped, n_unmapped)
    pub meta: Option<(u64, u64, u64, u64)>,
}

pub struct Index {
    pub min_shift: u32,
    pub depth: u32,
    pub csi: bool,
    pub refs: Vec<RefIndex>,
    pub n_no_coor: Option<u64>,
}

/// The index next to a BAM: `<bam>.bai`, else `<bam>.csi`.
pub fn index_path(bam: &Path) -> Option<PathBuf> {
    [format!("{}.bai", bam.display()), format!("{}.csi", bam.display())].into_iter().map(PathBuf::from).find(|p| p.exists())
}

struct Cursor<'a>(&'a [u8]);
impl Cursor<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        ensure!(self.0.len() >= n, "index ends early");
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        Ok(a)
    }
    fn u32(&mut self) -> Result<u32> { Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap())) }
    fn i32(&mut self) -> Result<i32> { Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap())) }
    fn u64(&mut self) -> Result<u64> { Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap())) }
    fn count(&mut self, what: &str) -> Result<usize> {
        let n = self.i32()?;
        ensure!(n >= 0 && (n as usize) <= self.0.len(), "index {what} count {n} is invalid");
        Ok(n as usize)
    }
}

/// Bin id of the metadata pseudo-bin for an index of this depth (37450 for BAI).
fn pseudo_bin(depth: u32) -> u32 { ((1u32 << ((depth + 1) * 3)) - 1) / 7 + 1 }

pub fn read_index(path: &Path) -> Result<Index> {
    let raw = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let csi = raw.starts_with(&[0x1f, 0x8b]);
    let bytes = if csi {
        let mut r = bgzf::Reader::new(std::io::Cursor::new(raw));
        let mut out = Vec::new();
        loop {
            let b = r.fill_buf()?;
            if b.is_empty() { break; }
            out.extend_from_slice(b);
            let n = b.len();
            r.consume(n);
        }
        out
    } else { raw };
    let mut c = Cursor(&bytes);
    let (min_shift, depth) = if csi {
        ensure!(c.take(4)? == b"CSI\x01", "{}: not a CSI index", path.display());
        let (m, d) = (c.i32()?, c.i32()?);
        ensure!((1..=30).contains(&m) && (1..=12).contains(&d) && m + 3 * d <= 62, "CSI min_shift {m} / depth {d} are invalid");
        let l_aux = c.count("aux length")?;
        c.take(l_aux)?;
        (m as u32, d as u32)
    } else {
        ensure!(c.take(4)? == b"BAI\x01", "{}: not a BAI or CSI index", path.display());
        (14, 5)
    };
    let pseudo = pseudo_bin(depth);
    let n_ref = c.count("reference")?;
    let mut refs = Vec::with_capacity(n_ref);
    for _ in 0..n_ref {
        let mut ri = RefIndex::default();
        for _ in 0..c.count("bin")? {
            let bin = c.u32()?;
            let loffset = if csi { Some(c.u64()?) } else { None };
            let n_chunk = c.count("chunk")?;
            let mut chunks = Vec::with_capacity(n_chunk);
            for _ in 0..n_chunk { chunks.push((c.u64()?, c.u64()?)); }
            if bin == pseudo {
                ensure!(chunks.len() == 2, "index metadata bin holds {} chunks, not 2", chunks.len());
                ri.meta = Some((chunks[0].0, chunks[0].1, chunks[1].0, chunks[1].1));
            } else {
                if let Some(l) = loffset { ri.loffsets.insert(bin, l); }
                ri.bins.entry(bin).or_default().extend(chunks);
            }
        }
        if !csi { for _ in 0..c.count("linear index")? { ri.linear.push(c.u64()?); } }
        refs.push(ri);
    }
    let n_no_coor = if c.0.len() >= 8 { Some(c.u64()?) } else { None };
    Ok(Index { min_shift, depth, csi, refs, n_no_coor })
}

impl Index {
    /// Chunks that can hold records overlapping 0-based [beg, end) on reference `rid`, sorted and merged.
    pub fn query(&self, rid: usize, beg: u64, end: u64) -> Result<Vec<(u64, u64)>> {
        let ri = self.refs.get(rid).ok_or_else(|| anyhow!("reference {rid} is not in the index"))?;
        let max = 1u64 << (self.min_shift + 3 * self.depth);
        ensure!(beg < end && end <= max, "region {beg}-{end} is outside what the index can address ({max})");
        let mut chunks: Vec<(u64, u64)> = Vec::new();
        let (mut t, mut s) = (0u64, self.min_shift + 3 * self.depth);
        for l in 0..=self.depth {
            for b in t + (beg >> s)..=t + ((end - 1) >> s) {
                if let Some(c) = ri.bins.get(&(b as u32)) { chunks.extend(c); }
            }
            s = s.saturating_sub(3);
            t += 1 << (3 * l);
        }
        let min_offset = if self.csi {
            // loffset of the smallest bin holding `beg`, else of its nearest ancestor that has one
            let mut bin = self.reg2bin(beg, beg + 1);
            loop {
                if let Some(&o) = ri.loffsets.get(&bin) { break o; }
                if bin == 0 { break 0; }
                bin = (bin - 1) / 8;
            }
        } else {
            ri.linear.get((beg >> 14) as usize).copied().unwrap_or(0)
        };
        chunks.retain(|c| c.1 > min_offset);
        chunks.sort_unstable_by_key(|c| c.0);
        let mut merged: Vec<(u64, u64)> = Vec::with_capacity(chunks.len());
        for c in chunks {
            match merged.last_mut() {
                Some(m) if c.0 <= m.1 => m.1 = m.1.max(c.1),
                _ => merged.push(c),
            }
        }
        Ok(merged)
    }

    fn reg2bin(&self, beg: u64, end: u64) -> u32 {
        let end = end - 1;
        let (mut l, mut s) = (self.depth, self.min_shift);
        let mut t = ((1u64 << (3 * self.depth)) - 1) / 7;
        while l > 0 {
            if beg >> s == end >> s { return (t + (beg >> s)) as u32; }
            l -= 1;
            s += 3;
            t -= 1 << (3 * l);
        }
        0
    }
}

/// Records on reference `rid` whose aligned span overlaps 0-based [beg, end), read from `chunks` of an indexed BAM.
/// Calls `f` with each; returns how many records were read.
pub fn query_records(r: &mut Records, chunks: &[(u64, u64)], rid: usize, beg: u64, end: u64,
                     mut f: impl FnMut(Record)) -> Result<u64> {
    let mut buf = Vec::new();
    let mut n = 0u64;
    for &(cs, ce) in chunks {
        if !r.seek(cs)? { break; }
        while r.virtual_position() < ce {
            if !r.next(&mut buf)? { break; }
            n += 1;
            let rec = Record(&buf);
            if rec.ref_id() != rid as i32 || rec.pos() < 0 { continue; }
            // 1-based closed [pos + 1, pos + span] against [beg + 1, end]
            let (s, e) = (rec.pos() as u64 + 1, rec.pos() as u64 + rec.ref_span());
            if s <= end && e >= beg + 1 { f(rec); }
        }
    }
    Ok(n)
}

// ---- finding record boundaries from an arbitrary block -------------------------------------------------------------

/// Compressed offset of the first BGZF block starting at or after `from`, however far (`len` if none): a header whose
/// BSIZE lands exactly on the next block header or on the end of the file, scanning on past damaged or garbage bytes.
pub fn find_block(f: &mut std::fs::File, from: u64, len: u64) -> Result<u64> {
    Ok(blitzgz::bgzf::find_block_start(f, from, len)?)
}

/// Virtual position reported for "no more records" (a range holding no record start, or reading to EOF).
pub const SPAN_EOF: u64 = u64::MAX;

/// Decompressed bytes from one virtual position on, with the virtual position of every byte.
struct Inflated<'a, R: std::io::Read> { r: &'a mut bgzf::Reader<R>, data: Vec<u8>, at: Vec<(usize, u64)>, eof: bool }

impl<R: std::io::Read> Inflated<'_, R> {
    const LIMIT: usize = 256 << 20;
    /// At least `n` bytes buffered; false at EOF (or past LIMIT, which no plausible record chain needs).
    fn fill(&mut self, n: usize) -> Result<bool> {
        while self.data.len() < n && !self.eof && self.data.len() < Self::LIMIT {
            let b = self.r.fill_buf()?;
            if b.is_empty() { self.eof = true; break; }
            let k = b.len();
            self.data.extend_from_slice(b);
            self.at.push((self.data.len() - k, self.r.virtual_position()));
            self.r.consume(k);
        }
        Ok(self.data.len() >= n)
    }
    fn vpos(&self, p: usize) -> u64 {
        let (s, v) = self.at[self.at.partition_point(|a| a.0 <= p) - 1];
        v + (p - s) as u64   // within one block's data, so only the low 16 bits (the uncompressed offset) move
    }
}

/// Fixed fields of a plausible BAM record at `b[0..]` (block_size first): its total length, if well formed.
/// Checks: sizes consistent, reference ids valid, positions within the reference, flag in range, read name printable
/// and NUL-terminated. `b` must hold the fixed fields and the read name.
pub fn record_len(b: &[u8], ref_lens: &[u64]) -> Option<usize> {
    let i32_at = |o: usize| i32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let u16_at = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]) as usize;
    let bs = i32_at(0);
    let (rid, pos, lrn, ncig, flag, lseq, nrid, npos) =
        (i32_at(4), i32_at(8), b[12] as usize, u16_at(16), u16_at(18), i32_at(20), i32_at(24), i32_at(28));
    let ref_ok = |id: i32, p: i32| p >= -1 && (id == -1 || id >= 0 && (id as usize) < ref_lens.len() && (p as i64) < ref_lens[id as usize] as i64);
    if bs < 32 || bs > (64 << 20) || lrn < 1 || flag > 0xfff || lseq < 0 || !ref_ok(rid, pos) || !ref_ok(nrid, npos) { return None; }
    let lseq = lseq as usize;
    if 32 + lrn + 4 * ncig + lseq.div_ceil(2) + lseq > bs as usize { return None; }
    let name = b.get(36..36 + lrn)?;
    if name[lrn - 1] != 0 || !name[..lrn - 1].iter().all(|&c| (b'!'..=b'~').contains(&c) && c != b'@') { return None; }
    Some(4 + bs as usize)
}

/// The variable part of a whole record `b` (as sized by `record_len`) parses exactly: CIGAR op codes valid, base
/// qualities all 0xFF or all <= 93, and aux fields (tag, type, value) ending exactly at the end of the record.
pub fn record_body_ok(b: &[u8]) -> bool {
    let (lrn, ncig, lseq) = (b[12] as usize, u16::from_le_bytes([b[16], b[17]]) as usize, i32::from_le_bytes(b[20..24].try_into().unwrap()) as usize);
    let mut o = 36 + lrn;
    if !b[o..o + 4 * ncig].chunks_exact(4).all(|c| c[0] & 0xf <= 8) { return false; }
    o += 4 * ncig + lseq.div_ceil(2);
    let q = &b[o..o + lseq];
    if !(q.iter().all(|&x| x == 0xff) || q.iter().all(|&x| x <= 93)) { return false; }
    o += lseq;
    let fixed = |t: u8| match t { b'A' | b'c' | b'C' => Some(1), b's' | b'S' => Some(2), b'i' | b'I' | b'f' => Some(4), _ => None };
    while o < b.len() {
        if o + 3 > b.len() || !b[o].is_ascii_alphabetic() || !b[o + 1].is_ascii_alphanumeric() { return false; }
        let t = b[o + 2];
        o += 3;
        o = o.saturating_add(match (t, fixed(t)) {
            (_, Some(n)) => n,
            (b'Z' | b'H', _) => match b[o..].iter().position(|&c| c == 0) { Some(k) => k + 1, None => return false },
            (b'B', _) => {
                let Some(n) = b.get(o).copied().and_then(fixed) else { return false };
                let Some(c) = b.get(o + 1..o + 5) else { return false };
                5usize.saturating_add(n.saturating_mul(u32::from_le_bytes(c.try_into().unwrap()) as usize))
            }
            _ => return false,
        });
    }
    o == b.len()
}

/// Virtual position of the first record starting at or after compressed offset `block` (SPAN_EOF if none).
pub fn first_record_from<R: std::io::Read + std::io::Seek>(r: &mut bgzf::Reader<R>, block: u64, ref_lens: &[u64]) -> Result<u64> {
    r.seek(block << 16)?;
    let mut buf = Inflated { r, data: Vec::new(), at: Vec::new(), eof: false };
    let mut u = 0usize;
    loop {
        if !buf.fill(u + 1)? {
            anyhow::ensure!(buf.eof, "no BAM record boundary within {} MB of compressed offset {block}", Inflated::<R>::LIMIT >> 20);
            return Ok(SPAN_EOF);
        }
        // a chain of 3 records, or fewer ending exactly at EOF
        let (mut p, mut k, mut ok) = (u, 0, false);
        while k < 3 {
            if !buf.fill(p + 36)? { ok = k > 0 && buf.eof && p == buf.data.len(); break; }
            let lrn = buf.data[p + 12] as usize;
            if !buf.fill(p + 36 + lrn)? { break; }
            let Some(n) = record_len(&buf.data[p..], ref_lens) else { break };
            if !buf.fill(p + n)? || !record_body_ok(&buf.data[p..p + n]) { break; }
            p += n; k += 1; ok = k == 3;
        }
        if ok { return Ok(buf.vpos(u)); }
        u += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bins_match_the_spec() {
        let bai = Index { min_shift: 14, depth: 5, csi: false, refs: vec![], n_no_coor: None };
        // SAM spec reg2bin examples: [0, 1) -> 4681 (16 kbp leaf), a 1 Mbp-spanning region -> a level-2 bin
        assert_eq!(bai.reg2bin(0, 1), 4681);
        assert_eq!(bai.reg2bin(16384, 16385), 4682);
        assert_eq!(bai.reg2bin(0, 1 << 26), 1);
        assert_eq!(bai.reg2bin(0, 1 << 29), 0);
        assert_eq!(pseudo_bin(5), 37450);
    }

    #[test]
    fn aux_parsing_and_long_cigar() {
        // record: ref 0, pos 99, name "r", 2 CIGAR ops (4S 10N placeholder), 4 bases, CG:B,I [4M]
        let l_seq = 4u32;
        let mut b = Vec::new();
        b.extend(0i32.to_le_bytes()); b.extend(99i32.to_le_bytes());
        b.push(2); b.push(60); b.extend(0u16.to_le_bytes()); b.extend(2u16.to_le_bytes()); b.extend(0u16.to_le_bytes());
        b.extend((l_seq as i32).to_le_bytes()); b.extend((-1i32).to_le_bytes()); b.extend((-1i32).to_le_bytes()); b.extend(0i32.to_le_bytes());
        b.extend(b"r\0");
        b.extend(((l_seq << 4) | 4).to_le_bytes()); b.extend(((10u32 << 4) | 3).to_le_bytes());
        b.extend([0x12, 0x48]); b.extend([30u8; 4]);
        b.extend(b"NMC\x05");
        b.extend(b"CGBI"); b.extend(1u32.to_le_bytes()); b.extend(((4u32 << 4) | 0).to_le_bytes());
        b.extend(b"MMZC+m,0;\0");
        let r = Record(&b);
        assert!(r.layout_ok());
        assert_eq!(r.cigar(), &((4u32 << 4) | 0).to_le_bytes());
        assert_eq!(r.ref_span(), 4);
        let f: Vec<_> = aux_fields(r.aux()).map(|f| (f.tag, f.ty)).collect();
        assert_eq!(f, [(*b"NM", b'C'), (*b"CG", b'B'), (*b"MM", b'Z')]);
        assert_eq!(aux_fields(r.aux()).next().unwrap().int(), Some(5));
        assert_eq!(aux_fields(r.aux()).nth(2).unwrap().value, b"C+m,0;");
    }
}
