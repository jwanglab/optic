//! BGZF: gzip members of at most 64 KiB, each recording its compressed size
//! in a `BC` header field. Every block's position and decompressed size are
//! known before decoding anything, so blocks decode independently, in
//! parallel, straight into their final place in one output buffer.

use std::io::{self, Read, Seek, SeekFrom};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::gzip::parse_header;
use crate::{Decompressor, Error};

/// Largest decompressed size of a BGZF block allowed by the spec.
pub const MAX_BLOCK_SIZE: usize = 65536;

/// Output bytes per parallel work unit.
const JOB_BYTES: usize = 1 << 20;

/// Location of one BGZF block within the compressed data.
#[derive(Debug, Clone, Copy)]
pub struct Block {
    /// Byte range of the raw DEFLATE payload.
    pub deflate_start: usize,
    pub deflate_end: usize,
    pub crc32: u32,
    pub isize: u32,
}

/// Returns true if `data` begins with a BGZF block header.
pub fn is_bgzf(data: &[u8]) -> bool {
    matches!(parse_header(data), Ok(h) if h.bgzf_bsize.is_some())
}

/// Total length (BSIZE + 1) of the BGZF block whose header starts `header`,
/// or `None` if it is not one. Needs the 12 fixed header bytes plus the
/// extra field (18 bytes for blocks written by bgzip/htslib).
pub fn block_size(header: &[u8]) -> Option<usize> {
    if header.len() < 12 || header[..3] != [0x1F, 0x8B, 8] || header[3] & 0x04 == 0 {
        return None;
    }
    let xlen = u16::from_le_bytes([header[10], header[11]]) as usize;
    let mut extra = header.get(12..12 + xlen)?;
    while extra.len() >= 4 {
        let slen = u16::from_le_bytes([extra[2], extra[3]]) as usize;
        if extra[0] == b'B' && extra[1] == b'C' && slen == 2 && extra.len() >= 6 {
            return Some(u16::from_le_bytes([extra[4], extra[5]]) as usize + 1);
        }
        extra = extra.get(4 + slen..)?;
    }
    None
}

/// Compressed offset of the first BGZF block starting at or after `from` in
/// a stream of `len` bytes, or `len` if there is none.
///
/// A candidate header counts as a block only if its size lands exactly on
/// another block header or on `len`, which rules out stray `1f 8b` bytes
/// inside compressed data. Scanning continues past damaged regions. Use this
/// to split a BGZF file into chunks without an index.
pub fn find_block_start<R: Read + Seek>(r: &mut R, mut from: u64, len: u64) -> io::Result<u64> {
    // A block is at most 64 KiB long, so each window of 64 KiB + 1 start
    // offsets contains a block start unless the data there is damaged. The
    // window also holds the successor header of most candidates.
    const STARTS: usize = MAX_BLOCK_SIZE + 1;
    const HEADER: usize = 64;
    let mut read_at = |pos: u64, n: usize| -> io::Result<Vec<u8>> {
        let n = n.min(len.saturating_sub(pos) as usize);
        let mut buf = vec![0u8; n];
        r.seek(SeekFrom::Start(pos))?;
        r.read_exact(&mut buf)?;
        Ok(buf)
    };
    while from < len {
        let w = read_at(from, STARTS + MAX_BLOCK_SIZE + HEADER)?;
        for p in 0..w.len().min(STARTS) {
            let Some(bsize) = block_size(&w[p..w.len().min(p + HEADER)]) else {
                continue;
            };
            let next = from + (p + bsize) as u64;
            if next == len {
                return Ok(from + p as u64);
            }
            if next > len {
                continue;
            }
            let q = p + bsize;
            let successor = if q + HEADER <= w.len() {
                block_size(&w[q..q + HEADER])
            } else {
                block_size(&read_at(next, HEADER)?)
            };
            if successor.is_some() {
                return Ok(from + p as u64);
            }
        }
        from += STARTS as u64;
    }
    Ok(len)
}

/// Parses the block at `pos`, returning it and the offset of the next block.
pub fn parse_block(data: &[u8], pos: usize) -> Result<(Block, usize), Error> {
    let header = parse_header(&data[pos..])?;
    let bsize = header.bgzf_bsize.ok_or(Error::NotBgzf)? as usize;
    let end = pos + bsize + 1;
    if end > data.len() {
        return Err(Error::Truncated);
    }
    if bsize + 1 < header.len + 8 {
        return Err(Error::BadHeader);
    }
    let t = &data[end - 8..end];
    let block = Block {
        deflate_start: pos + header.len,
        deflate_end: end - 8,
        crc32: u32::from_le_bytes([t[0], t[1], t[2], t[3]]),
        isize: u32::from_le_bytes([t[4], t[5], t[6], t[7]]),
    };
    if block.isize as usize > MAX_BLOCK_SIZE {
        return Err(Error::BadHeader);
    }
    Ok((block, end))
}

/// Walks the block headers of a complete BGZF file.
pub fn index(data: &[u8]) -> Result<Vec<Block>, Error> {
    let mut blocks = Vec::with_capacity(data.len() / 16384 + 1);
    let mut pos = 0;
    while pos < data.len() {
        let (block, next) = parse_block(data, pos)?;
        blocks.push(block);
        pos = next;
    }
    Ok(blocks)
}

/// Decompresses one block into `out`, which must be exactly `block.isize` long.
pub fn decompress_block(
    dec: &mut Decompressor,
    data: &[u8],
    block: &Block,
    out: &mut [u8],
) -> Result<(), Error> {
    debug_assert_eq!(out.len(), block.isize as usize);
    let (_, written) =
        dec.deflate_decompress(&data[block.deflate_start..block.deflate_end], out)?;
    if written != out.len() {
        return Err(Error::SizeMismatch);
    }
    if crate::crc32(out) != block.crc32 {
        return Err(Error::ChecksumMismatch);
    }
    Ok(())
}

/// Decompresses a complete in-memory BGZF file using `threads` threads.
pub fn decompress(data: &[u8], threads: usize) -> Result<Vec<u8>, Error> {
    let blocks = index(data)?;
    let total = blocks.iter().map(|b| b.isize as usize).sum();
    let mut out = vec![0u8; total];
    decompress_blocks_into(data, &blocks, &mut out, threads)?;
    Ok(out)
}

/// Decompresses `blocks` of `data` into `out`, whose length must equal the
/// sum of their sizes.
pub fn decompress_blocks_into(
    data: &[u8],
    blocks: &[Block],
    out: &mut [u8],
    threads: usize,
) -> Result<(), Error> {
    if out.len() != blocks.iter().map(|b| b.isize as usize).sum::<usize>() {
        return Err(Error::SizeMismatch);
    }
    let run = |dec: &mut Decompressor, blocks: &[Block], mut out: &mut [u8]| -> Result<(), Error> {
        for block in blocks {
            let (head, tail) = out.split_at_mut(block.isize as usize);
            decompress_block(dec, data, block, head)?;
            out = tail;
        }
        Ok(())
    };

    let threads = threads.max(1);
    if threads == 1 || blocks.len() <= 1 {
        return run(&mut Decompressor::new(), blocks, out);
    }

    // Carve the output into disjoint per-job slices up front.
    let mut jobs = Vec::new();
    let mut rest = out;
    let mut first = 0;
    let mut job_len = 0;
    for (i, block) in blocks.iter().enumerate() {
        job_len += block.isize as usize;
        if job_len >= JOB_BYTES || i + 1 == blocks.len() {
            let (head, tail) = rest.split_at_mut(job_len);
            jobs.push((&blocks[first..=i], head));
            rest = tail;
            first = i + 1;
            job_len = 0;
        }
    }

    let threads = threads.min(jobs.len());
    let queue = Mutex::new(jobs.into_iter());
    let failed = AtomicBool::new(false);
    let error = Mutex::new(None);
    let worker = || {
        let mut dec = Decompressor::new();
        while !failed.load(Ordering::Relaxed) {
            let Some((blocks, out)) = queue.lock().unwrap().next() else {
                break;
            };
            if let Err(e) = run(&mut dec, blocks, out) {
                failed.store(true, Ordering::Relaxed);
                error.lock().unwrap().get_or_insert(e);
            }
        }
    };
    std::thread::scope(|s| {
        // The calling thread is one of the workers. If spawning fails (e.g.
        // wasm32-wasip1 has no threads), it does the remaining work alone.
        for _ in 1..threads {
            if std::thread::Builder::new().spawn_scoped(s, worker).is_err() {
                break;
            }
        }
        worker();
    });
    match error.into_inner().unwrap() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}
