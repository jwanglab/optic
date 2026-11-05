//! BGZF (blocked gzip, SAM spec 4.1) reading with virtual positions: (compressed offset of a block << 16) | offset
//! into its uncompressed data. A thin layer over the blitzgz reader, which inflates each
//! block whole and checks its CRC32. It runs with no worker threads, decoding one block
//! at a time on the caller's thread: optic parallelizes across regions and shards instead.

use std::io::{self, BufRead, Read, Seek};

/// blitzgz's errors reworded for optic's damaged-input warnings; other errors (I/O) pass through.
fn reword(e: io::Error) -> io::Error {
    use blitzgz::Error as E;
    let Some(b) = e.get_ref().and_then(|x| x.downcast_ref::<E>()) else { return e };
    let msg = match b {
        E::InvalidData | E::OutputTooSmall | E::SizeMismatch => "BGZF block does not inflate (corrupt data)",
        E::Truncated => "the file ends inside a BGZF block (truncated?)",
        E::BadHeader => "BGZF block header is invalid (bad block size)",
        E::NotBgzf => "not a BGZF block (bad gzip header)",
        E::ChecksumMismatch => "BGZF block fails its CRC32 check (corrupt file?)",
    };
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

pub struct Reader<R: Read>(blitzgz::BgzfReader<R>);

impl<R: Read> Reader<R> {
    pub fn new(inner: R) -> Self { Reader(blitzgz::BgzfReader::with_threads(inner, 0)) }

    /// The rest of the current block, reading on past empty blocks; empty at end of file.
    pub fn fill_buf(&mut self) -> io::Result<&[u8]> { self.0.fill_buf().map_err(reword) }

    pub fn consume(&mut self, n: usize) { self.0.consume(n) }

    /// Virtual position of the next byte; at the end of a block, the start of the next block.
    pub fn virtual_position(&self) -> u64 { self.0.virtual_position() }

    /// Fill `buf` completely. Ok(false) if the file ended before the first byte; an error if it ended partway.
    pub fn read_exact_or_eof(&mut self, buf: &mut [u8]) -> io::Result<bool> {
        let mut got = 0;
        while got < buf.len() {
            let src = self.fill_buf()?;
            if src.is_empty() {
                return if got == 0 { Ok(false) } else { Err(io::Error::new(io::ErrorKind::UnexpectedEof, "BAM ends inside a record")) };
            }
            let k = src.len().min(buf.len() - got);
            buf[got..got + k].copy_from_slice(&src[..k]);
            self.consume(k);
            got += k;
        }
        Ok(true)
    }

    pub fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        if self.read_exact_or_eof(buf)? || buf.is_empty() { Ok(()) } else { Err(io::Error::new(io::ErrorKind::UnexpectedEof, "unexpected end of BGZF data")) }
    }
}

impl<R: Read + Seek> Reader<R> {
    pub fn seek(&mut self, vpos: u64) -> io::Result<()> { self.0.seek_virtual(vpos).map_err(reword) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A BGZF block holding `data` stored uncompressed (deflate "stored" block), so the test needs no compressor.
    fn stored_block(data: &[u8]) -> Vec<u8> {
        let mut d = vec![1u8];   // BFINAL, stored
        d.extend((data.len() as u16).to_le_bytes());
        d.extend((!(data.len() as u16)).to_le_bytes());
        d.extend(data);
        let bsize = 12 + 6 + d.len() + 8;
        let mut b = vec![0x1f, 0x8b, 8, 4, 0, 0, 0, 0, 0, 0xff, 6, 0, b'B', b'C', 2, 0];
        b.extend(((bsize - 1) as u16).to_le_bytes());
        b.extend(d);
        b.extend(blitzgz::crc32(data).to_le_bytes());
        b.extend((data.len() as u32).to_le_bytes());
        b
    }

    #[test]
    fn reads_across_blocks_seeks_and_rewords_errors() {
        let (a, b) = (stored_block(b"hello "), stored_block(b"world"));
        let eof = stored_block(b"");
        let file: Vec<u8> = [a.clone(), b.clone(), eof].concat();
        let mut r = Reader::new(std::io::Cursor::new(file.clone()));
        let mut buf = [0u8; 11];
        r.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"hello world");
        assert_eq!(r.virtual_position(), ((a.len() + b.len()) as u64) << 16);
        assert!(r.fill_buf().unwrap().is_empty());
        r.seek(((a.len() as u64) << 16) | 2).unwrap();
        let mut two = [0u8; 3];
        r.read_exact(&mut two).unwrap();
        assert_eq!(&two, b"rld");
        // a corrupted payload byte fails the CRC
        let mut bad = file.clone();
        bad[a.len() - 9] ^= 1;
        let e = Reader::new(std::io::Cursor::new(bad)).read_exact(&mut buf).unwrap_err();
        assert!(e.to_string().contains("CRC32"), "{e}");
        // a cut inside the second block reads as truncation
        let e = Reader::new(std::io::Cursor::new(file[..a.len() + 10].to_vec())).read_exact(&mut buf).unwrap_err();
        assert!(e.to_string().contains("truncated?"), "{e}");
    }
}
