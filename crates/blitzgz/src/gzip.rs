//! gzip (RFC 1952) member framing.

use crate::{Decompressor, Error};

const FTEXT_RESERVED: u8 = 0xE0;
const FHCRC: u8 = 0x02;
const FEXTRA: u8 = 0x04;
const FNAME: u8 = 0x08;
const FCOMMENT: u8 = 0x10;

/// Parsed gzip member header.
#[derive(Debug, Clone, Copy)]
pub struct Header {
    /// Header length in bytes; the DEFLATE stream starts here.
    pub len: usize,
    /// BGZF `BC` extra subfield: total member size minus one.
    pub bgzf_bsize: Option<u16>,
}

pub fn parse_header(data: &[u8]) -> Result<Header, Error> {
    if data.len() < 10 {
        return Err(Error::Truncated);
    }
    if data[0] != 0x1F || data[1] != 0x8B || data[2] != 8 || data[3] & FTEXT_RESERVED != 0 {
        return Err(Error::BadHeader);
    }
    let flags = data[3];
    let mut pos = 10;
    let mut bgzf_bsize = None;

    if flags & FEXTRA != 0 {
        let xlen_bytes = data.get(pos..pos + 2).ok_or(Error::Truncated)?;
        let xlen = u16::from_le_bytes([xlen_bytes[0], xlen_bytes[1]]) as usize;
        pos += 2;
        let mut extra = data.get(pos..pos + xlen).ok_or(Error::Truncated)?;
        pos += xlen;
        while extra.len() >= 4 {
            let slen = u16::from_le_bytes([extra[2], extra[3]]) as usize;
            let field = extra.get(4..4 + slen).ok_or(Error::BadHeader)?;
            if extra[0] == b'B' && extra[1] == b'C' && slen == 2 {
                bgzf_bsize = Some(u16::from_le_bytes([field[0], field[1]]));
            }
            extra = &extra[4 + slen..];
        }
    }
    for flag in [FNAME, FCOMMENT] {
        if flags & flag != 0 {
            let nul = data
                .get(pos..)
                .and_then(|d| d.iter().position(|&b| b == 0))
                .ok_or(Error::Truncated)?;
            pos += nul + 1;
        }
    }
    if flags & FHCRC != 0 {
        pos += 2;
    }
    if pos > data.len() {
        return Err(Error::Truncated);
    }
    Ok(Header {
        len: pos,
        bgzf_bsize,
    })
}

fn read_trailer(data: &[u8], pos: usize) -> Result<(u32, u32), Error> {
    let t = data.get(pos..pos + 8).ok_or(Error::Truncated)?;
    Ok((
        u32::from_le_bytes([t[0], t[1], t[2], t[3]]),
        u32::from_le_bytes([t[4], t[5], t[6], t[7]]),
    ))
}

/// Decompresses a (possibly multi-member) gzip file sequentially.
///
/// Output sizes are not known up front, so each member is decoded into a
/// guessed buffer that is doubled on overflow. For BGZF prefer
/// [`crate::bgzf::decompress`], which knows every block's size and decodes in
/// parallel.
pub fn decompress(data: &[u8]) -> Result<Vec<u8>, Error> {
    let mut dec = Decompressor::new();
    let mut out = Vec::new();
    let mut pos = 0;
    // ISIZE of the final member is exact for the common single-member case.
    let last_isize = if data.len() >= 4 {
        u32::from_le_bytes(data[data.len() - 4..].try_into().unwrap()) as usize
    } else {
        0
    };
    loop {
        let header = parse_header(&data[pos..])?;
        let payload = &data[pos + header.len..];
        let start = out.len();
        let mut cap = last_isize.max(payload.len() * 4).max(1024);
        let (consumed, written) = loop {
            out.resize(start + cap, 0);
            match dec.deflate_decompress(payload, &mut out[start..]) {
                Ok(r) => break r,
                Err(Error::OutputTooSmall) => cap *= 2,
                Err(e) => return Err(e),
            }
        };
        out.truncate(start + written);
        let trailer_pos = pos + header.len + consumed;
        let (crc, isize) = read_trailer(data, trailer_pos)?;
        if isize != written as u32 {
            return Err(Error::SizeMismatch);
        }
        if crc != crate::crc32(&out[start..]) {
            return Err(Error::ChecksumMismatch);
        }
        pos = trailer_pos + 8;
        if pos == data.len() {
            return Ok(out);
        }
    }
}
