//! Streaming BGZF decompression, parallel or random-access.
//!
//! The reader splits the input into jobs of whole BGZF blocks (~256 KiB
//! compressed each) as it reads, hands them to a pool of worker threads, and
//! serves the decompressed jobs back in order. Reading ahead happens on the
//! caller's thread inside `read`, overlapping with the workers' decoding.
//!
//! With zero worker threads, it decodes one block at a time on the caller's
//! thread instead, which suits random access through
//! [`BgzfReader::seek_virtual`].

use std::collections::BTreeMap;
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::{Error, bgzf};

/// Compressed bytes per job; about 1 MiB once decompressed.
const JOB_BYTES: usize = 256 * 1024;
/// Jobs in flight per worker; bounds memory to roughly
/// `threads * JOBS_PER_THREAD * 1.3 MiB`.
const JOBS_PER_THREAD: usize = 3;

struct Job {
    seq: u64,
    epoch: u64,
    data: Vec<u8>,
    out: Vec<u8>,
}

struct Done {
    seq: u64,
    data: Vec<u8>,
    result: Result<Vec<u8>, Error>,
}

/// Where a block sits in the compressed stream and in its job's output.
#[derive(Clone, Copy)]
struct BlockPos {
    coffset: u64,
    ustart: usize,
}

/// The blocks of one job, and the compressed offset just past the last.
struct JobBlocks {
    blocks: Vec<BlockPos>,
    end: u64,
}

/// A [`Read`] + [`BufRead`] adapter that decompresses a BGZF stream (BAM,
/// bgzipped FASTQ/VCF, ...), optionally using multiple threads.
///
/// `fill_buf` never returns data from more than one BGZF block, so the
/// [`virtual_position`](Self::virtual_position) taken just before a
/// `fill_buf` locates every byte it returns.
///
/// Compressed offsets (and so virtual positions) count from where the
/// underlying reader was when the `BgzfReader` was created, normally the
/// start of the file.
///
/// ```no_run
/// use std::io::BufRead;
/// let file = std::fs::File::open("reads.fq.gz")?;
/// let reader = blitzgz::BgzfReader::new(file);
/// for line in reader.lines() {
///     let _line = line?;
/// }
/// # Ok::<(), std::io::Error>(())
/// ```
pub struct BgzfReader<R: Read> {
    inner: BufReader<R>,
    jobs: Option<Sender<Job>>,
    done: Receiver<Done>,
    /// Set when there are no worker threads (none requested, or none could
    /// be started, as on wasm32-wasip1): blocks are then decoded on the
    /// caller's thread, one at a time.
    inline: Option<Box<crate::Decompressor>>,
    workers: Vec<JoinHandle<()>>,
    /// Bumped on every seek; workers skip jobs from older epochs.
    epoch: Arc<AtomicU64>,
    max_in_flight: usize,
    next_seq: u64,
    next_out: u64,
    pending: BTreeMap<u64, Result<Vec<u8>, Error>>,
    /// Block positions of each job submitted but not yet current.
    job_blocks: BTreeMap<u64, JobBlocks>,
    input_done: bool,
    /// Compressed offset of the next block to read from `inner`.
    in_offset: u64,
    current: Vec<u8>,
    current_blocks: JobBlocks,
    pos: usize,
    free_data: Vec<Vec<u8>>,
    free_out: Vec<Vec<u8>>,
}

impl<R: Read> BgzfReader<R> {
    /// Uses one worker thread per available CPU.
    pub fn new(inner: R) -> Self {
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        Self::with_threads(inner, threads)
    }

    /// Uses `threads` worker threads. With 0, decodes one block at a time on
    /// the caller's thread with no read-ahead, which is best for random
    /// access.
    pub fn with_threads(inner: R, threads: usize) -> Self {
        let (job_tx, job_rx) = mpsc::channel::<Job>();
        let (done_tx, done_rx) = mpsc::channel::<Done>();
        let job_rx = Arc::new(Mutex::new(job_rx));
        let epoch = Arc::new(AtomicU64::new(0));
        let mut workers = Vec::with_capacity(threads);
        for _ in 0..threads {
            let job_rx = Arc::clone(&job_rx);
            let done_tx = done_tx.clone();
            let epoch = Arc::clone(&epoch);
            match std::thread::Builder::new().spawn(move || worker(&job_rx, &done_tx, &epoch)) {
                Ok(handle) => workers.push(handle),
                Err(_) => break,
            }
        }
        let inline = workers.is_empty();
        Self {
            // Inline mode reads a block at a time; a smaller buffer wastes
            // less after a seek.
            inner: BufReader::with_capacity(if inline { 1 << 17 } else { 1 << 20 }, inner),
            jobs: (!inline).then_some(job_tx),
            done: done_rx,
            inline: inline.then(|| Box::new(crate::Decompressor::new())),
            max_in_flight: workers.len().max(1) * JOBS_PER_THREAD,
            workers,
            epoch,
            next_seq: 0,
            next_out: 0,
            pending: BTreeMap::new(),
            job_blocks: BTreeMap::new(),
            input_done: false,
            in_offset: 0,
            current: Vec::new(),
            current_blocks: JobBlocks {
                blocks: Vec::new(),
                end: 0,
            },
            pos: 0,
            free_data: Vec::new(),
            free_out: Vec::new(),
        }
    }

    /// Virtual position (`compressed block offset << 16 | offset within the
    /// block's data`, as in BAM indexes) of the next byte to be read. At the
    /// end of a block this is the start of the next block, whether or not
    /// that block has been read yet.
    pub fn virtual_position(&self) -> u64 {
        let blocks = &self.current_blocks.blocks;
        if self.pos < self.current.len() {
            let b = blocks[self.block_index(self.pos)];
            (b.coffset << 16) | (self.pos - b.ustart) as u64
        } else {
            // The first trailing empty block (such as the EOF marker), so the
            // answer does not depend on how blocks were grouped into jobs.
            let i = blocks.partition_point(|b| b.ustart < self.current.len());
            blocks.get(i).map_or(self.current_blocks.end, |b| b.coffset) << 16
        }
    }

    /// Index of the last current block starting at or before `pos`, which
    /// for `pos < current.len()` is the non-empty block holding it.
    fn block_index(&self, pos: usize) -> usize {
        self.current_blocks
            .blocks
            .partition_point(|b| b.ustart <= pos)
            - 1
    }

    /// Reads ahead until `max_in_flight` jobs are queued or the input ends.
    fn submit_jobs(&mut self) -> io::Result<()> {
        // Inline mode decodes as it reads, so it reads one block per job.
        let job_bytes = if self.inline.is_some() { 1 } else { JOB_BYTES };
        while !self.input_done && (self.next_seq - self.next_out) < self.max_in_flight as u64 {
            let mut data = self.free_data.pop().unwrap_or_default();
            data.clear();
            let mut blocks = Vec::new();
            let mut ustart = 0;
            while data.len() < job_bytes {
                let start = data.len();
                if !read_block(&mut self.inner, &mut data)? {
                    self.input_done = true;
                    break;
                }
                blocks.push(BlockPos {
                    coffset: self.in_offset,
                    ustart,
                });
                self.in_offset += (data.len() - start) as u64;
                // ISIZE; decoding checks it against the real size.
                let t = &data[data.len() - 4..];
                ustart += u32::from_le_bytes([t[0], t[1], t[2], t[3]]) as usize;
            }
            if data.is_empty() {
                self.free_data.push(data);
                break;
            }
            let seq = self.next_seq;
            self.next_seq += 1;
            let end = self.in_offset;
            self.job_blocks.insert(seq, JobBlocks { blocks, end });
            let out = self.free_out.pop().unwrap_or_default();
            if let Some(dec) = &mut self.inline {
                let result = decode_job(dec, &data, out);
                self.free_data.push(data);
                self.pending.insert(seq, result);
                // Decoding ahead gains nothing without other threads.
                break;
            }
            let job = Job {
                seq,
                epoch: self.epoch.load(Ordering::Relaxed),
                data,
                out,
            };
            self.jobs
                .as_ref()
                .expect("job queue is open while the reader has workers")
                .send(job)
                .map_err(|_| io::Error::other("BGZF worker thread died"))?;
        }
        Ok(())
    }

    /// Makes the next decompressed job current. Returns false at the end.
    fn next_job(&mut self) -> io::Result<bool> {
        self.submit_jobs()?;
        if self.next_out == self.next_seq {
            return Ok(false);
        }
        let result = loop {
            if let Some(r) = self.pending.remove(&self.next_out) {
                break r;
            }
            let done = self
                .done
                .recv()
                .map_err(|_| io::Error::other("BGZF worker thread died"))?;
            self.free_data.push(done.data);
            if done.seq < self.next_out {
                // Submitted before a seek.
                if let Ok(out) = done.result {
                    self.free_out.push(out);
                }
                continue;
            }
            self.pending.insert(done.seq, done.result);
        };
        let blocks = self
            .job_blocks
            .remove(&self.next_out)
            .expect("every submitted job has its block list");
        self.next_out += 1;
        let out = result?;
        let old = std::mem::replace(&mut self.current, out);
        self.free_out.push(old);
        self.current_blocks = blocks;
        self.pos = 0;
        Ok(true)
    }
}

impl<R: Read + Seek> BgzfReader<R> {
    /// Moves to a virtual position, such as one from a BAM index or from
    /// [`virtual_position`](Self::virtual_position). Discards any decoded
    /// read-ahead.
    pub fn seek_virtual(&mut self, vpos: u64) -> io::Result<()> {
        let (coffset, within) = (vpos >> 16, (vpos & 0xFFFF) as usize);
        self.epoch.fetch_add(1, Ordering::Relaxed);
        self.next_out = self.next_seq;
        self.pending.clear();
        self.job_blocks.clear();
        self.inner.seek(SeekFrom::Start(coffset))?;
        self.in_offset = coffset;
        self.input_done = false;
        self.current.clear();
        self.current_blocks = JobBlocks {
            blocks: Vec::new(),
            end: coffset,
        };
        self.pos = 0;
        if within > 0 {
            let block_len = if self.next_job()? {
                self.current_blocks
                    .blocks
                    .get(1)
                    .map_or(self.current.len(), |b| b.ustart)
            } else {
                0
            };
            if within > block_len {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("virtual position {coffset}:{within} is past the end of its block"),
                ));
            }
            self.pos = within;
        }
        Ok(())
    }
}

impl<R: Read> Read for BgzfReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let avail = self.fill_buf()?;
        let n = avail.len().min(buf.len());
        buf[..n].copy_from_slice(&avail[..n]);
        self.consume(n);
        Ok(n)
    }
}

impl<R: Read> BufRead for BgzfReader<R> {
    /// The rest of the current BGZF block (skipping empty blocks); empty at
    /// the end of the stream.
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        // Loop: jobs made only of empty blocks (e.g. the EOF marker) decode
        // to nothing.
        while self.pos == self.current.len() {
            if !self.next_job()? {
                return Ok(&[]);
            }
        }
        let end = self
            .current_blocks
            .blocks
            .get(self.block_index(self.pos) + 1)
            .map_or(self.current.len(), |b| b.ustart);
        Ok(&self.current[self.pos..end])
    }

    fn consume(&mut self, amt: usize) {
        self.pos = (self.pos + amt).min(self.current.len());
    }
}

impl<R: Read> Drop for BgzfReader<R> {
    fn drop(&mut self) {
        // Closing the job queue stops the workers once they drain it.
        self.jobs = None;
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

fn worker(jobs: &Mutex<Receiver<Job>>, done: &Sender<Done>, epoch: &AtomicU64) {
    let mut dec = crate::Decompressor::new();
    loop {
        let job = match jobs.lock().unwrap().recv() {
            Ok(job) => job,
            Err(_) => return,
        };
        let Job {
            seq,
            epoch: job_epoch,
            data,
            out,
        } = job;
        // The reader has seeked since: hand the buffers back undecoded.
        let result = if job_epoch == epoch.load(Ordering::Relaxed) {
            decode_job(&mut dec, &data, out)
        } else {
            Ok(out)
        };
        if done.send(Done { seq, data, result }).is_err() {
            return;
        }
    }
}

/// Decodes the whole BGZF blocks in `data` into `out`.
fn decode_job(
    dec: &mut crate::Decompressor,
    data: &[u8],
    mut out: Vec<u8>,
) -> Result<Vec<u8>, Error> {
    let blocks = bgzf::index(data)?;
    let total = blocks.iter().map(|b| b.isize as usize).sum();
    // Reused buffers only zero the part that grows.
    out.resize(total, 0);
    for (block, chunk) in blocks.iter().zip(split_blocks(&mut out, &blocks)) {
        bgzf::decompress_block(dec, data, block, chunk)?;
    }
    Ok(out)
}

fn split_blocks<'a>(mut out: &'a mut [u8], blocks: &[bgzf::Block]) -> Vec<&'a mut [u8]> {
    blocks
        .iter()
        .map(|b| {
            let (head, tail) = std::mem::take(&mut out).split_at_mut(b.isize as usize);
            out = tail;
            head
        })
        .collect()
}

/// Appends one whole BGZF block from `r` to `data`. Returns false on a clean
/// end of input (no bytes before the block).
fn read_block<R: Read>(r: &mut R, data: &mut Vec<u8>) -> io::Result<bool> {
    let mut header = [0u8; 12];
    if !read_exact_or_eof(r, &mut header)? {
        return Ok(false);
    }
    if header[..3] != [0x1F, 0x8B, 8] || header[3] & 0x04 == 0 {
        return Err(Error::NotBgzf.into());
    }
    let xlen = u16::from_le_bytes([header[10], header[11]]) as usize;
    let start = data.len();
    data.extend_from_slice(&header);
    data.resize(start + 12 + xlen, 0);
    read_exact(r, &mut data[start + 12..])?;

    let block_len = bgzf::block_size(&data[start..]).ok_or(Error::NotBgzf)?;
    if block_len < 12 + xlen + 8 {
        return Err(Error::BadHeader.into());
    }
    data.resize(start + block_len, 0);
    read_exact(r, &mut data[start + 12 + xlen..])?;
    Ok(true)
}

/// Like `read_exact`, but a clean EOF before the first byte returns false.
fn read_exact_or_eof<R: Read>(r: &mut R, buf: &mut [u8]) -> io::Result<bool> {
    let mut filled = 0;
    while filled < buf.len() {
        match r.read(&mut buf[filled..]) {
            Ok(0) if filled == 0 => return Ok(false),
            Ok(0) => return Err(Error::Truncated.into()),
            Ok(n) => filled += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(true)
}

fn read_exact<R: Read>(r: &mut R, buf: &mut [u8]) -> io::Result<()> {
    if buf.is_empty() || read_exact_or_eof(r, buf)? {
        Ok(())
    } else {
        Err(Error::Truncated.into())
    }
}
