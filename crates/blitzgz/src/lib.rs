//! Fast pure-Rust DEFLATE, gzip and BGZF decompression.
//!
//! - [`Decompressor`]: whole-buffer raw DEFLATE decoder (libdeflate-style).
//! - [`gzip::decompress`]: sequential multi-member gzip.
//! - [`bgzf::decompress`]: parallel BGZF (BAM, bgzipped VCF/FASTQ, ...) held
//!   in memory.
//! - [`BgzfReader`]: parallel streaming BGZF for files of any size.
//! - [`decompress`]: picks between gzip and in-memory BGZF.
//! - [`crc32`] / [`Crc32`]: the gzip CRC-32, SIMD-accelerated.
//!
//! Build with the `safe` feature to replace the few unchecked fast-path
//! memory accesses with bounds-checked ones; the crate then forbids `unsafe`.

#![cfg_attr(feature = "safe", forbid(unsafe_code))]

pub mod bgzf;
mod crc32;
pub mod gzip;
pub mod inflate;
mod raw;
pub mod reader;

pub use crc32::{Crc32, crc32};
pub use inflate::Decompressor;
pub use reader::BgzfReader;

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Malformed DEFLATE data.
    InvalidData,
    /// Input ended mid-stream.
    Truncated,
    /// Decompressed data does not fit in the output buffer.
    OutputTooSmall,
    /// Malformed gzip/BGZF header.
    BadHeader,
    /// A gzip member lacks the BGZF `BC` field.
    NotBgzf,
    /// CRC32 in the trailer does not match the data.
    ChecksumMismatch,
    /// Decompressed size does not match the trailer.
    SizeMismatch,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Error::InvalidData => "invalid DEFLATE data",
            Error::Truncated => "truncated input",
            Error::OutputTooSmall => "output buffer too small",
            Error::BadHeader => "invalid gzip header",
            Error::NotBgzf => "not a BGZF file",
            Error::ChecksumMismatch => "CRC32 mismatch",
            Error::SizeMismatch => "decompressed size mismatch",
        })
    }
}

impl std::error::Error for Error {}

impl From<Error> for std::io::Error {
    fn from(e: Error) -> Self {
        std::io::Error::new(std::io::ErrorKind::InvalidData, e)
    }
}

/// Decompresses a gzip file held in memory: BGZF in parallel on all cores,
/// anything else sequentially.
pub fn decompress(data: &[u8]) -> Result<Vec<u8>, Error> {
    if bgzf::is_bgzf(data) {
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        bgzf::decompress(data, threads)
    } else {
        gzip::decompress(data)
    }
}
