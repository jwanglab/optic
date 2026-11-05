//! Whole-buffer DEFLATE (RFC 1951) decoder.
//!
//! The design follows libdeflate: no streaming state machine, a 64-bit bit
//! buffer refilled a word at a time, packed decode-table entries that resolve a
//! symbol (and its extra-bit layout) in one lookup, and a "fast loop" that runs
//! while there is enough input and output slack to skip per-symbol bounds
//! checks. Near the buffer ends a careful generic loop takes over.

use crate::Error;
use crate::raw;

/// Evaluates a call to an `unsafe` fast-path helper from [`raw`]: inside an
/// `unsafe` block by default, plainly with the `safe` feature (where the
/// helpers are safe, bounds-checked functions).
macro_rules! fast {
    ($e:expr) => {{
        #[cfg(not(feature = "safe"))]
        // SAFETY: each use is guarded so the helper's documented preconditions
        // hold: in the fast loop by `has_room()` at the top of each iteration
        // (with FAST_IN_MARGIN / FAST_OUT_MARGIN of slack) and
        // `dist <= out.pos()`; in `refill_fast!` by `in_pos + 8 <= in_len`.
        let r = unsafe { $e };
        #[cfg(feature = "safe")]
        let r = $e;
        r
    }};
}

// Extra bits follow the codeword: mask off codeword + extra bits, then
// shift out the codeword. The shift is masked to 6 bits, covering
// bits 8..=13 of a length/distance entry (codeword length; bits 12-13
// are zero for such entries).
macro_rules! extra_bits {
    ($saved:expr, $e:expr) => {
        (($saved & ((1u64 << ($e & 0xFF)) - 1)).wrapping_shr($e >> 8) as usize)
    };
}

// ---------------------------------------------------------------------------
// Decode table entry layout (u32):
//
//   bits  0..=7   total bits to consume (codeword length + extra bits);
//                 for subtable pointers: the main table's index bits
//   bits  8..=11  codeword length (shift that skips the codeword when
//                 extracting extra bits); for subtable pointers: subtable bits
//   bit   13      end of block
//   bit   14      subtable pointer
//   bit   15      exceptional (end of block, subtable pointer, or invalid)
//   bits 16..=30  value: literal byte, length/distance base, or subtable start
//   bit   31      literal
// ---------------------------------------------------------------------------

const LITERAL: u32 = 1 << 31;
const EXCEPTIONAL: u32 = 1 << 15;
const SUBTABLE: u32 = 1 << 14;
const END_OF_BLOCK: u32 = 1 << 13;
const INVALID: u32 = EXCEPTIONAL;

const LITLEN_TABLEBITS: u32 = 11;
const DIST_TABLEBITS: u32 = 8;
const PRECODE_TABLEBITS: u32 = 7;
const LITLEN_MASK: u64 = (1 << LITLEN_TABLEBITS) - 1;
const DIST_MASK: u64 = (1 << DIST_TABLEBITS) - 1;
const PRECODE_MASK: u64 = (1 << PRECODE_TABLEBITS) - 1;

// Maximum table sizes including subtables (computed by zlib's `enough`).
const LITLEN_ENOUGH: usize = 2342;
const DIST_ENOUGH: usize = 402;
const DIST_OFFSET: usize = LITLEN_ENOUGH;
const TABLES_LEN: usize = LITLEN_ENOUGH + DIST_ENOUGH;

const NUM_LITLEN_SYMS: usize = 288;
const NUM_DIST_SYMS: usize = 32;
const NUM_PRECODE_SYMS: usize = 19;
const MAX_CODEWORD_LEN: u32 = 15;
const MAX_PRECODE_LEN: u32 = 7;

const PRECODE_ORDER: [u8; NUM_PRECODE_SYMS] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// Fast loop requires this much input left: up to three word refills per
/// iteration, each reading 8 bytes and advancing at most 7.
const FAST_IN_MARGIN: usize = 32;
/// Fast loop requires this much output left: two literals, a 258-byte match,
/// and the up-to-8-byte overwrite of the last word copied (see
/// `raw::copy_match`), with room to spare.
const FAST_OUT_MARGIN: usize = 2 + 258 + 8 + 20;

/// Bytes of zeros we may pretend to read past the end of the input. A valid
/// stream never consumes them, which is checked at the end.
const MAX_OVERREAD: usize = 8;

/// Per-symbol decode results, before the codeword length is merged in.
/// Low byte holds the number of extra bits.
const fn litlen_results() -> [u32; NUM_LITLEN_SYMS] {
    const LEN_BASE: [u16; 29] = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115,
        131, 163, 195, 227, 258,
    ];
    const LEN_EXTRA: [u8; 29] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
    ];
    let mut r = [0u32; NUM_LITLEN_SYMS];
    let mut i = 0;
    while i < 256 {
        r[i] = LITERAL | ((i as u32) << 16);
        i += 1;
    }
    r[256] = EXCEPTIONAL | END_OF_BLOCK;
    let mut s = 0;
    while s < 29 {
        r[257 + s] = ((LEN_BASE[s] as u32) << 16) | LEN_EXTRA[s] as u32;
        s += 1;
    }
    r[286] = INVALID;
    r[287] = INVALID;
    r
}

const fn dist_results() -> [u32; NUM_DIST_SYMS] {
    const DIST_BASE: [u16; 30] = [
        1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
        2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
    ];
    let mut r = [INVALID; NUM_DIST_SYMS];
    let mut i = 0;
    while i < 30 {
        let extra = if i < 4 { 0 } else { (i as u32 - 2) / 2 };
        r[i] = ((DIST_BASE[i] as u32) << 16) | extra;
        i += 1;
    }
    r
}

const fn precode_results() -> [u32; NUM_PRECODE_SYMS] {
    let mut r = [0u32; NUM_PRECODE_SYMS];
    let mut i = 0;
    while i < NUM_PRECODE_SYMS {
        r[i] = (i as u32) << 16;
        i += 1;
    }
    r
}

static LITLEN_RESULTS: [u32; NUM_LITLEN_SYMS] = litlen_results();
static DIST_RESULTS: [u32; NUM_DIST_SYMS] = dist_results();
static PRECODE_RESULTS: [u32; NUM_PRECODE_SYMS] = precode_results();

#[inline(always)]
fn make_entry(result: u32, len: u32) -> u32 {
    (result & !0xFF) | (len << 8) | (len + (result & 0xFF))
}

/// Builds a canonical-Huffman decode table from codeword lengths.
///
/// Codes longer than `table_bits` go to subtables appended after the main
/// table. Over-subscribed codes are rejected; incomplete codes are accepted and
/// their unused slots decode as invalid.
fn build_table(
    lens: &[u8],
    results: &[u32],
    table: &mut [u32],
    table_bits: u32,
    max_len: u32,
) -> Result<(), Error> {
    let mut count = [0u16; MAX_CODEWORD_LEN as usize + 1];
    for &l in lens {
        count[l as usize] += 1;
    }
    count[0] = 0;
    if count[max_len as usize + 1..].iter().any(|&c| c != 0) {
        return Err(Error::InvalidData);
    }

    let mut left: i32 = 1;
    for &c in &count[1..=max_len as usize] {
        left = (left << 1) - c as i32;
        if left < 0 {
            return Err(Error::InvalidData);
        }
    }
    let main_size = 1usize << table_bits;
    if left > 0 {
        table.fill(INVALID);
    }

    // Sort symbols by (length, symbol) and compute first canonical codes.
    let mut offs = [0u16; MAX_CODEWORD_LEN as usize + 2];
    let mut next_code = [0u32; MAX_CODEWORD_LEN as usize + 2];
    let mut code = 0u32;
    for len in 1..=max_len as usize {
        offs[len + 1] = offs[len] + count[len];
        code = (code + count[len - 1] as u32) << 1;
        next_code[len] = code;
    }
    let mut sorted = [0u16; NUM_LITLEN_SYMS];
    let mut pos = offs;
    for (sym, &l) in lens.iter().enumerate() {
        if l != 0 {
            sorted[pos[l as usize] as usize] = sym as u16;
            pos[l as usize] += 1;
        }
    }
    let num_codes = offs[max_len as usize + 1] as usize;

    // Main table. For a complete code, fill one entry per codeword at the
    // smallest table size that holds it and double the table (memcpy) for
    // each additional bit, as libdeflate does; this beats strided per-entry
    // stores by a wide margin. Incomplete codes take the simple path over the
    // INVALID-prefilled table.
    let num_short = offs[table_bits.min(max_len) as usize + 1] as usize;
    if left == 0 {
        let mut codeword = 0usize;
        let mut len = 1u32;
        let mut cur_end = 2usize;
        let mut i = 0;
        loop {
            for _ in 0..count[len as usize] {
                table[codeword] = make_entry(results[sorted[i] as usize], len);
                i += 1;
                // Increment the bit-reversed codeword (unless it is all ones,
                // which only the final codeword of a complete code can be).
                if codeword != cur_end - 1 {
                    let bit =
                        1usize << (usize::BITS - 1 - (codeword ^ (cur_end - 1)).leading_zeros());
                    codeword = (codeword & (bit - 1)) | bit;
                }
            }
            if len == table_bits {
                break;
            }
            table.copy_within(0..cur_end, cur_end);
            cur_end <<= 1;
            len += 1;
        }
    } else {
        for &sym in &sorted[..num_short] {
            let len = lens[sym as usize] as u32;
            let c = next_code[len as usize];
            next_code[len as usize] += 1;
            let rev = ((c as u16).reverse_bits() >> (16 - len)) as usize;
            let entry = make_entry(results[sym as usize], len);
            let mut i = rev;
            while i < main_size {
                table[i] = entry;
                i += 1 << len;
            }
        }
    }

    // Codes longer than the main table go to subtables.
    let mut remaining = count;
    let mut cur_prefix = usize::MAX;
    let mut sub_start = 0usize;
    let mut sub_bits = 0u32;
    let mut next_free = main_size;

    for &sym in &sorted[num_short..num_codes] {
        let len = lens[sym as usize] as u32;
        let c = next_code[len as usize];
        next_code[len as usize] += 1;
        let rev = ((c as u16).reverse_bits() >> (16 - len)) as usize;
        let result = results[sym as usize];
        let prefix = rev & (main_size - 1);
        if prefix != cur_prefix {
            // Size the subtable to hold every code sharing this prefix
            // (zlib's algorithm: grow until the remaining codes fill it).
            cur_prefix = prefix;
            sub_bits = len - table_bits;
            let mut sub_left: i32 = 1 << sub_bits;
            while sub_bits + table_bits < max_len {
                sub_left -= remaining[(sub_bits + table_bits) as usize] as i32;
                if sub_left <= 0 {
                    break;
                }
                sub_bits += 1;
                sub_left <<= 1;
            }
            sub_start = next_free;
            next_free += 1 << sub_bits;
            if next_free > table.len() {
                return Err(Error::InvalidData);
            }
            table[prefix] =
                EXCEPTIONAL | SUBTABLE | ((sub_start as u32) << 16) | (sub_bits << 8) | table_bits;
        }
        let r = len - table_bits;
        let entry = make_entry(result, r);
        let mut i = rev >> table_bits;
        while i < 1 << sub_bits {
            table[sub_start + i] = entry;
            i += 1 << r;
        }
        remaining[len as usize] -= 1;
    }
    Ok(())
}

/// Reusable DEFLATE decompressor. Holds the decode tables (~11 KiB), so keep
/// one per thread rather than creating one per block.
pub struct Decompressor {
    /// Litlen table followed by the distance table at `DIST_OFFSET`; one array
    /// so the decode loop addresses both from a single base register.
    tables: [u32; TABLES_LEN],
    precode: [u32; 1 << PRECODE_TABLEBITS],
    lens: [u8; NUM_LITLEN_SYMS + NUM_DIST_SYMS + 138],
    static_loaded: bool,
}

impl Default for Decompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Decompressor {
    pub fn new() -> Self {
        Self {
            tables: [INVALID; TABLES_LEN],
            precode: [INVALID; 1 << PRECODE_TABLEBITS],
            lens: [0; NUM_LITLEN_SYMS + NUM_DIST_SYMS + 138],
            static_loaded: false,
        }
    }

    fn load_static(&mut self) {
        if self.static_loaded {
            return;
        }
        let mut lens = [0u8; NUM_LITLEN_SYMS];
        lens[..144].fill(8);
        lens[144..256].fill(9);
        lens[256..280].fill(7);
        lens[280..].fill(8);
        build_table(
            &lens,
            &LITLEN_RESULTS,
            &mut self.tables[..DIST_OFFSET],
            LITLEN_TABLEBITS,
            MAX_CODEWORD_LEN,
        )
        .expect("static litlen code is valid");
        build_table(
            &[5; NUM_DIST_SYMS],
            &DIST_RESULTS,
            &mut self.tables[DIST_OFFSET..],
            DIST_TABLEBITS,
            MAX_CODEWORD_LEN,
        )
        .expect("static distance code is valid");
        self.static_loaded = true;
    }

    /// Decompresses a raw DEFLATE stream into `out`.
    ///
    /// Returns `(input bytes consumed, output bytes written)`. Fails with
    /// [`Error::OutputTooSmall`] if the data does not fit in `out`.
    pub fn deflate_decompress(
        &mut self,
        input: &[u8],
        out: &mut [u8],
    ) -> Result<(usize, usize), Error> {
        #[cfg(all(target_arch = "x86_64", not(feature = "safe")))]
        if std::is_x86_feature_detected!("bmi2") {
            return self.decompress_impl::<true>(input, out);
        }
        self.decompress_impl::<false>(input, out)
    }

    #[inline(always)]
    #[allow(unused_assignments)] // the final bit-buffer update is dead
    fn decompress_impl<const BMI2: bool>(
        &mut self,
        input: &[u8],
        out: &mut [u8],
    ) -> Result<(usize, usize), Error> {
        let in_len = input.len();
        let out_len = out.len();
        let in_fast_end = in_len.saturating_sub(FAST_IN_MARGIN);
        let out_fast_end = out_len.saturating_sub(FAST_OUT_MARGIN);

        let mut bitbuf: u64 = 0;
        let mut bitsleft: u32 = 0;
        let mut in_pos: usize = 0;
        let mut overread: usize = 0;
        let mut out_pos: usize = 0;

        // Invariant: bits of `bitbuf` above `bitsleft` are either zero or the
        // correct upcoming stream bits, so refills may OR the same byte twice.

        // Requires in_pos + 8 <= in_len. Leaves 56..=63 bits.
        // Only the low byte of `bitsleft` is used here, so this also works
        // inside the fast loop, where higher bits hold garbage.
        macro_rules! refill_fast {
            () => {{
                bitbuf |= fast!(raw::load_u64_le(input, in_pos)).wrapping_shl(bitsleft);
                in_pos += (7 - ((bitsleft >> 3) & 7)) as usize;
                bitsleft |= 56;
            }};
        }
        macro_rules! refill {
            () => {{
                if in_pos + 8 <= in_len {
                    refill_fast!();
                } else {
                    while bitsleft < 56 {
                        let byte = if in_pos < in_len {
                            in_pos += 1;
                            input[in_pos - 1]
                        } else {
                            overread += 1;
                            if overread > MAX_OVERREAD {
                                return Err(Error::Truncated);
                            }
                            0
                        };
                        bitbuf |= (byte as u64) << bitsleft;
                        bitsleft += 8;
                    }
                }
            }};
        }
        macro_rules! bits {
            ($n:expr) => {
                (bitbuf & ((1u64 << $n) - 1)) as u32
            };
        }
        macro_rules! consume {
            ($n:expr) => {{
                let n: u32 = $n;
                bitbuf >>= n;
                bitsleft -= n;
            }};
        }
        // Consumes an entry's bits; shift count is masked to 6 bits, which is
        // exact since no entry consumes more than 28.
        macro_rules! consume_entry {
            ($e:expr) => {{
                bitbuf = bitbuf.wrapping_shr($e);
                bitsleft -= $e & 0xFF;
            }};
        }

        loop {
            refill!();
            let is_final = bits!(1) != 0;
            let btype = (bitbuf >> 1) & 3;
            consume!(3);

            match btype {
                0 => {
                    // Stored block: rewind the bit buffer to a byte boundary.
                    consume!(bitsleft & 7);
                    let unconsumed = (bitsleft / 8) as usize;
                    if overread > unconsumed {
                        return Err(Error::Truncated);
                    }
                    in_pos -= unconsumed - overread;
                    overread = 0;
                    bitbuf = 0;
                    bitsleft = 0;
                    if in_len - in_pos < 4 {
                        return Err(Error::Truncated);
                    }
                    let len = u16::from_le_bytes([input[in_pos], input[in_pos + 1]]) as usize;
                    let nlen = u16::from_le_bytes([input[in_pos + 2], input[in_pos + 3]]) as usize;
                    in_pos += 4;
                    if len != !nlen & 0xFFFF {
                        return Err(Error::InvalidData);
                    }
                    if in_len - in_pos < len {
                        return Err(Error::Truncated);
                    }
                    if out_len - out_pos < len {
                        return Err(Error::OutputTooSmall);
                    }
                    out[out_pos..out_pos + len].copy_from_slice(&input[in_pos..in_pos + len]);
                    in_pos += len;
                    out_pos += len;
                    if is_final {
                        break;
                    }
                    continue;
                }
                1 => self.load_static(),
                2 => {
                    self.static_loaded = false;
                    let num_litlen = bits!(5) as usize + 257;
                    let num_dist = ((bitbuf >> 5) & 0x1F) as usize + 1;
                    let num_precode = ((bitbuf >> 10) & 0xF) as usize + 4;
                    consume!(14);

                    let mut precode_lens = [0u8; NUM_PRECODE_SYMS];
                    for &sym in &PRECODE_ORDER[..num_precode] {
                        refill!();
                        precode_lens[sym as usize] = bits!(3) as u8;
                        consume!(3);
                    }
                    build_table(
                        &precode_lens,
                        &PRECODE_RESULTS,
                        &mut self.precode,
                        PRECODE_TABLEBITS,
                        MAX_PRECODE_LEN,
                    )?;

                    let total = num_litlen + num_dist;
                    let mut i = 0;
                    while i < total {
                        refill!();
                        let entry = self.precode[(bitbuf & PRECODE_MASK) as usize];
                        if entry & EXCEPTIONAL != 0 {
                            return Err(Error::InvalidData);
                        }
                        consume_entry!(entry);
                        let sym = (entry >> 16) as u8;
                        if sym < 16 {
                            self.lens[i] = sym;
                            i += 1;
                            continue;
                        }
                        let (rep, val) = match sym {
                            16 => {
                                if i == 0 {
                                    return Err(Error::InvalidData);
                                }
                                let r = 3 + bits!(2) as usize;
                                consume!(2);
                                (r, self.lens[i - 1])
                            }
                            17 => {
                                let r = 3 + bits!(3) as usize;
                                consume!(3);
                                (r, 0)
                            }
                            _ => {
                                let r = 11 + bits!(7) as usize;
                                consume!(7);
                                (r, 0)
                            }
                        };
                        if i + rep > total {
                            return Err(Error::InvalidData);
                        }
                        self.lens[i..i + rep].fill(val);
                        i += rep;
                    }
                    let (ll, rest) = self.lens.split_at(num_litlen);
                    build_table(
                        ll,
                        &LITLEN_RESULTS,
                        &mut self.tables[..DIST_OFFSET],
                        LITLEN_TABLEBITS,
                        MAX_CODEWORD_LEN,
                    )?;
                    build_table(
                        &rest[..num_dist],
                        &DIST_RESULTS,
                        &mut self.tables[DIST_OFFSET..],
                        DIST_TABLEBITS,
                        MAX_CODEWORD_LEN,
                    )?;
                }
                _ => return Err(Error::InvalidData),
            }

            // ---------------- fast loop ----------------
            let mut block_done = false;
            if in_pos < in_fast_end && out_pos < out_fast_end {
                let mut st = BitState {
                    bitbuf,
                    bitsleft,
                    in_pos,
                    out_pos,
                };
                #[cfg(all(target_arch = "x86_64", not(feature = "safe")))]
                let done = if BMI2 {
                    // SAFETY: `BMI2` is only true when the CPU supports it.
                    // Only the fast loop is compiled for BMI2: being a
                    // target-feature function, it cannot be inlined into this
                    // one, which keeps its register allocation separate.
                    unsafe {
                        fast_loop_bmi2(&self.tables, input, out, in_fast_end, out_fast_end, &mut st)
                    }
                } else {
                    fast_loop(&self.tables, input, out, in_fast_end, out_fast_end, &mut st)
                };
                #[cfg(not(all(target_arch = "x86_64", not(feature = "safe"))))]
                let done = fast_loop(&self.tables, input, out, in_fast_end, out_fast_end, &mut st);
                block_done = done?;
                BitState {
                    bitbuf,
                    bitsleft,
                    in_pos,
                    out_pos,
                } = st;
            }
            let tables = &self.tables;

            // ---------------- generic loop ----------------
            // Runs to the end-of-block symbol, unless the fast loop found it.
            if !block_done {
                loop {
                    refill!();
                    let mut entry = tables[(bitbuf & LITLEN_MASK) as usize];
                    if entry & SUBTABLE != 0 {
                        consume_entry!(entry);
                        let idx = (entry >> 16) as usize + bits!((entry >> 8) & 0xF) as usize;
                        entry = tables[idx];
                    }
                    if entry & LITERAL != 0 {
                        if out_pos >= out_len {
                            return Err(Error::OutputTooSmall);
                        }
                        consume_entry!(entry);
                        out[out_pos] = (entry >> 16) as u8;
                        out_pos += 1;
                        continue;
                    }
                    if entry & EXCEPTIONAL != 0 {
                        if entry & END_OF_BLOCK != 0 {
                            consume_entry!(entry);
                            break;
                        }
                        return Err(Error::InvalidData);
                    }
                    let saved = bitbuf;
                    consume_entry!(entry);
                    let length = (entry >> 16) as usize + extra_bits!(saved, entry);

                    refill!();
                    entry = tables[DIST_OFFSET + (bitbuf & DIST_MASK) as usize];
                    if entry & SUBTABLE != 0 {
                        consume_entry!(entry);
                        let idx = (entry >> 16) as usize + bits!((entry >> 8) & 0xF) as usize;
                        entry = tables[DIST_OFFSET + idx];
                    }
                    if entry & EXCEPTIONAL != 0 {
                        return Err(Error::InvalidData);
                    }
                    let saved = bitbuf;
                    consume_entry!(entry);
                    let dist = (entry >> 16) as usize + extra_bits!(saved, entry);
                    if dist > out_pos {
                        return Err(Error::InvalidData);
                    }
                    if length > out_len - out_pos {
                        return Err(Error::OutputTooSmall);
                    }
                    for i in out_pos..out_pos + length {
                        out[i] = out[i - dist];
                    }
                    out_pos += length;
                }
            }

            if is_final {
                break;
            }
        }

        // Give back whole bytes still in the bit buffer; the stream must not
        // have consumed any of the virtual zero bytes.
        let unconsumed = (bitsleft / 8) as usize;
        if overread > unconsumed {
            return Err(Error::Truncated);
        }
        Ok((in_pos - (unconsumed - overread), out_pos))
    }
}

/// Decoder state handed between the generic code and [`fast_loop`].
struct BitState {
    bitbuf: u64,
    bitsleft: u32,
    in_pos: usize,
    out_pos: usize,
}

/// The hot decode loop, run while at least `FAST_IN_MARGIN` input and
/// `FAST_OUT_MARGIN` output bytes remain, so no per-symbol bounds checks are
/// needed. Returns whether the block's end was reached. Requires
/// `st.in_pos < in_fast_end` and `st.out_pos < out_fast_end`.
///
/// Scheduling follows libdeflate: each entry is consumed before its type is
/// examined, and the next entry is looked up *before* the refill and the
/// match copy, keeping both off the bitbuf -> lookup -> shift dependency
/// chain. Bit budget: a refill leaves >= 56 bits; a lookup needs 11 (litlen)
/// or 8 (distance) valid bits; a main-table literal uses <= 11 bits, any
/// litlen symbol <= 20 (15 code + 5 extra), a distance <= 28 (15 + 13).
///
/// This is deliberately a separate, non-inlined function (see [`fast_loop`]):
/// inlined into the decoder, it shares register allocation with header
/// parsing and the generic loop, and LLVM spills the table and buffer
/// pointers, costing ~5-8% overall.
#[inline(always)]
fn fast_loop_impl<const BMI2: bool>(
    tables: &[u32; TABLES_LEN],
    input: &[u8],
    out: &mut [u8],
    in_fast_end: usize,
    out_fast_end: usize,
    st: &mut BitState,
) -> Result<bool, Error> {
    let mut bitbuf = st.bitbuf;
    let mut bitsleft = st.bitsleft;
    let mut ip = raw::FastIn::new(input, st.in_pos, in_fast_end);
    let mut op = raw::FastOut::new(out, st.out_pos, out_fast_end);
    let mut block_done = false;

    macro_rules! bits {
        ($n:expr) => {
            (bitbuf & ((1u64 << $n) - 1)) as u32
        };
    }
    // Extra bits of `$e` from `$saved`. With BMI2, `bzhi` reads the bit
    // count straight from the entry's low byte; the portable form needs it
    // masked to 6 bits first.
    macro_rules! extra_bits_fast {
        ($saved:expr, $e:expr) => {
            if BMI2 {
                fast!(raw::bzhi($saved, $e)).wrapping_shr($e >> 8) as usize
            } else {
                extra_bits!($saved, $e)
            }
        };
    }
    // Subtracts the whole entry from `bitsleft` (libdeflate's trick): only
    // its low byte is meaningful in this function.
    macro_rules! consume_entry_fast {
        ($e:expr) => {{
            bitbuf = bitbuf.wrapping_shr($e);
            bitsleft = bitsleft.wrapping_sub($e);
        }};
    }
    // Only the low byte of `bitsleft` is meaningful in here.
    macro_rules! refill_ip {
        () => {{
            bitbuf |= fast!(ip.load_u64_le()).wrapping_shl(bitsleft);
            fast!(ip.advance((7 - ((bitsleft >> 3) & 7)) as usize));
            bitsleft |= 56;
        }};
    }
    refill_ip!();
    let mut entry = tables[(bitbuf & LITLEN_MASK) as usize];
    loop {
        let mut saved = bitbuf;
        consume_entry_fast!(entry);
        if entry & LITERAL != 0 {
            // Up to three literals per refill (33 bits, then an
            // 11-bit preload).
            let lit = (entry >> 16) as u8;
            entry = tables[(bitbuf & LITLEN_MASK) as usize];
            saved = bitbuf;
            consume_entry_fast!(entry);
            fast!(op.push(lit));
            if entry & LITERAL != 0 {
                let lit = (entry >> 16) as u8;
                entry = tables[(bitbuf & LITLEN_MASK) as usize];
                saved = bitbuf;
                consume_entry_fast!(entry);
                fast!(op.push(lit));
                if entry & LITERAL != 0 {
                    let lit = (entry >> 16) as u8;
                    entry = tables[(bitbuf & LITLEN_MASK) as usize];
                    refill_ip!();
                    fast!(op.push(lit));
                    if ip.has_room() && op.has_room() {
                        continue;
                    }
                    break;
                }
            }
        }
        // At most 22 bits used by literals plus this entry's 20.
        if entry & EXCEPTIONAL != 0 {
            if entry & SUBTABLE != 0 {
                // 22 + 11 used; the subtable entry needs <= 9 more.
                let idx = (entry >> 16) as usize + bits!((entry >> 8) & 0xF) as usize;
                entry = tables[idx];
                saved = bitbuf;
                consume_entry_fast!(entry);
                if entry & LITERAL != 0 {
                    fast!(op.push((entry >> 16) as u8));
                    entry = tables[(bitbuf & LITLEN_MASK) as usize];
                    refill_ip!();
                    if ip.has_room() && op.has_room() {
                        continue;
                    }
                    break;
                }
            }
            if entry & EXCEPTIONAL != 0 {
                if entry & END_OF_BLOCK != 0 {
                    block_done = true;
                    break;
                }
                return Err(Error::InvalidData);
            }
        }
        let length = (entry >> 16) as usize + extra_bits_fast!(saved, entry);

        // Chained literals come from the main table (<= 11 bits),
        // so at most 11 + 11 + 20 bits are used here and the 8-bit
        // distance lookup can go ahead of the refill (which only
        // adds bits above `bitsleft`). The distance (28 bits) plus
        // the next preload (11) may then need a refill.
        entry = tables[DIST_OFFSET + (bitbuf & DIST_MASK) as usize];
        if entry & EXCEPTIONAL != 0 {
            // Subtable: up to 28 bits, then the 11-bit preload.
            if (bitsleft as u8 as u32) < 28 + LITLEN_TABLEBITS {
                refill_ip!();
            }
            if entry & SUBTABLE != 0 {
                consume_entry_fast!(entry);
                let idx = (entry >> 16) as usize + bits!((entry >> 8) & 0xF) as usize;
                entry = tables[DIST_OFFSET + idx];
            }
            if entry & EXCEPTIONAL != 0 {
                return Err(Error::InvalidData);
            }
        } else if (bitsleft as u8 as u32) < 8 + 13 + LITLEN_TABLEBITS {
            // Main-table distance: up to 21 bits, then the preload.
            refill_ip!();
        }
        let saved = bitbuf;
        consume_entry_fast!(entry);
        let dist = (entry >> 16) as usize + extra_bits_fast!(saved, entry);
        if dist > op.pos() {
            return Err(Error::InvalidData);
        }
        entry = tables[(bitbuf & LITLEN_MASK) as usize];
        refill_ip!();
        // Keep the preload and refill ahead of the copy: LLVM
        // otherwise sinks them below it, where a mispredicted copy
        // branch would discard and restart the decode chain.
        raw::schedule_barrier(&mut entry, &mut bitbuf);
        fast!(op.copy_match(dist, length));
        if !(ip.has_room() && op.has_room()) {
            break;
        }
    }
    st.bitbuf = bitbuf;
    st.bitsleft = bitsleft & 0xFF;
    st.in_pos = ip.pos();
    st.out_pos = op.pos();
    Ok(block_done)
}

#[inline(never)]
fn fast_loop(
    tables: &[u32; TABLES_LEN],
    input: &[u8],
    out: &mut [u8],
    in_fast_end: usize,
    out_fast_end: usize,
    st: &mut BitState,
) -> Result<bool, Error> {
    fast_loop_impl::<false>(tables, input, out, in_fast_end, out_fast_end, st)
}

/// # Safety
/// The CPU must support BMI2.
#[cfg(all(target_arch = "x86_64", not(feature = "safe")))]
#[target_feature(enable = "bmi2")]
#[inline(never)]
unsafe fn fast_loop_bmi2(
    tables: &[u32; TABLES_LEN],
    input: &[u8],
    out: &mut [u8],
    in_fast_end: usize,
    out_fast_end: usize,
    st: &mut BitState,
) -> Result<bool, Error> {
    fast_loop_impl::<true>(tables, input, out, in_fast_end, out_fast_end, st)
}
