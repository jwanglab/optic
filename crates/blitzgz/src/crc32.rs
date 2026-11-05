//! CRC-32 with the gzip/zlib polynomial (reflected 0xEDB88320).
//!
//! Portable table-driven code everywhere. On x86_64 and aarch64, the
//! carry-less-multiply folding from Intel's "Fast CRC Computation for Generic
//! Polynomials Using PCLMULQDQ" white paper (as in zlib-ng, libdeflate and
//! crc32fast), selected at run time: 128 bits wide with PCLMULQDQ or 512 bits
//! wide with AVX-512 VPCLMULQDQ on x86_64, and 128 bits wide with PMULL on
//! aarch64. Wasm has no carry-less multiply, so it uses the tables.

/// CRC-32 of `data`.
///
/// ```
/// assert_eq!(blitzgz::crc32(b"123456789"), 0xcbf4_3926);
/// ```
pub fn crc32(data: &[u8]) -> u32 {
    update(0, data)
}

/// Incremental CRC-32.
#[derive(Clone, Copy, Default, Debug)]
pub struct Crc32 {
    crc: u32,
}

impl Crc32 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, data: &[u8]) {
        self.crc = update(self.crc, data);
    }

    /// CRC of all data passed to `update` so far.
    pub fn finalize(&self) -> u32 {
        self.crc
    }
}

/// Extends `crc` (the CRC of some prefix) over `data`.
fn update(crc: u32, data: &[u8]) -> u32 {
    #[cfg(all(target_arch = "x86_64", not(feature = "safe")))]
    if data.len() >= simd::MIN_LEN
        && is_x86_feature_detected!("pclmulqdq")
        && is_x86_feature_detected!("sse4.1")
    {
        if data.len() >= simd::MIN_LEN_512
            && is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("vpclmulqdq")
        {
            // SAFETY: the required CPU features were just detected.
            return unsafe { simd::update_512(crc, data) };
        }
        // SAFETY: the required CPU features were just detected.
        return unsafe { simd::update(crc, data) };
    }
    #[cfg(all(target_arch = "aarch64", not(feature = "safe")))]
    if data.len() >= pmull::MIN_LEN && std::arch::is_aarch64_feature_detected!("pmull") {
        // SAFETY: the required CPU feature was just detected.
        return unsafe { pmull::update(crc, data) };
    }
    !update_table(!crc, data)
}

const POLY: u32 = 0xEDB8_8320;

static TABLES: [[u32; 256]; 16] = make_tables();

const fn make_tables() -> [[u32; 256]; 16] {
    let mut t = [[0u32; 256]; 16];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { (c >> 1) ^ POLY } else { c >> 1 };
            k += 1;
        }
        t[0][i] = c;
        i += 1;
    }
    // t[j][i]: the CRC contribution of byte i followed by j zero bytes.
    let mut i = 0;
    while i < 256 {
        let mut j = 1;
        while j < 16 {
            let prev = t[j - 1][i];
            t[j][i] = (prev >> 8) ^ t[0][(prev & 0xFF) as usize];
            j += 1;
        }
        i += 1;
    }
    t
}

/// Inputs at least this long are split into `LANES` interleaved streams.
const MIN_LEN_LANES: usize = 2048;
const LANES: usize = 4;

/// Table-driven update of a raw (pre-inverted) CRC state.
fn update_table(c: u32, data: &[u8]) -> u32 {
    if data.len() < MIN_LEN_LANES {
        return update_table_serial(c, data);
    }
    // One table-lookup chain is latency-bound; four independent ones over
    // the four quarters of `data` overlap, then are joined algebraically.
    let lane_len = data.len() / LANES / 8 * 8;
    let (body, tail) = data.split_at(lane_len * LANES);
    let (l0, rest) = body.split_at(lane_len);
    let (l1, rest) = rest.split_at(lane_len);
    let (l2, l3) = rest.split_at(lane_len);
    let mut state = [c, !0, !0, !0];
    for (((w0, w1), w2), w3) in words(l0).zip(words(l1)).zip(words(l2)).zip(words(l3)) {
        state[0] = step8(state[0], &w0);
        state[1] = step8(state[1], &w1);
        state[2] = step8(state[2], &w2);
        state[3] = step8(state[3], &w3);
    }
    // Join the finished CRCs: crc(A ‖ B) = crc(A) * x^(8 |B|) + crc(B).
    let shift = x8n_mod_p(lane_len);
    let mut crc = !state[0];
    for &s in &state[1..] {
        crc = mul_mod_p(shift, crc) ^ !s;
    }
    update_table_serial(!crc, tail)
}

fn words(lane: &[u8]) -> impl Iterator<Item = [u8; 8]> + '_ {
    lane.chunks_exact(8).map(|w| w.try_into().unwrap())
}

/// Eight bytes of slice-by-8.
#[inline(always)]
fn step8(c: u32, w: &[u8; 8]) -> u32 {
    let t = &TABLES;
    let v = u64::from_le_bytes(*w) ^ c as u64;
    let b = |k: u32| ((v >> (8 * k)) & 0xFF) as usize;
    t[7][b(0)] ^ t[6][b(1)] ^ t[5][b(2)] ^ t[4][b(3)] ^ t[3][b(4)] ^ t[2][b(5)] ^ t[1][b(6)] ^ t[0][b(7)]
}

/// `a * b mod P` for polynomials in reflected form (bit 31 = x^0).
const fn mul_mod_p(a: u32, mut b: u32) -> u32 {
    let mut product = 0;
    let mut m = 1u32 << 31;
    while m != 0 {
        if a & m != 0 {
            product ^= b;
        }
        m >>= 1;
        b = if b & 1 != 0 { (b >> 1) ^ POLY } else { b >> 1 };
    }
    product
}

/// `X2K[k] = x^(2^k) mod P`, reflected.
static X2K: [u32; 64] = {
    let mut t = [0u32; 64];
    t[0] = 1 << 30;
    let mut k = 1;
    while k < 64 {
        t[k] = mul_mod_p(t[k - 1], t[k - 1]);
        k += 1;
    }
    t
};

/// `x^(8 n) mod P`, reflected: the shift past `n` bytes.
fn x8n_mod_p(n: usize) -> u32 {
    let mut p = 1u32 << 31;
    let (mut n, mut k) = (n, 3);
    while n != 0 {
        if n & 1 != 0 {
            p = mul_mod_p(X2K[k], p);
        }
        n >>= 1;
        k += 1;
    }
    p
}

/// Slice-by-16 update of a raw (pre-inverted) CRC state.
fn update_table_serial(mut c: u32, data: &[u8]) -> u32 {
    let t = &TABLES;
    let byte = |x: u32, shift: u32| ((x >> shift) & 0xFF) as usize;
    let mut chunks = data.chunks_exact(16);
    for chunk in &mut chunks {
        let chunk: &[u8; 16] = chunk.try_into().unwrap();
        let w = |i: usize| u32::from_le_bytes([chunk[i], chunk[i + 1], chunk[i + 2], chunk[i + 3]]);
        let (a, b, d, e) = (w(0) ^ c, w(4), w(8), w(12));
        c = t[15][byte(a, 0)]
            ^ t[14][byte(a, 8)]
            ^ t[13][byte(a, 16)]
            ^ t[12][byte(a, 24)]
            ^ t[11][byte(b, 0)]
            ^ t[10][byte(b, 8)]
            ^ t[9][byte(b, 16)]
            ^ t[8][byte(b, 24)]
            ^ t[7][byte(d, 0)]
            ^ t[6][byte(d, 8)]
            ^ t[5][byte(d, 16)]
            ^ t[4][byte(d, 24)]
            ^ t[3][byte(e, 0)]
            ^ t[2][byte(e, 8)]
            ^ t[1][byte(e, 16)]
            ^ t[0][byte(e, 24)];
    }
    for &b in chunks.remainder() {
        c = t[0][((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    c
}

/// `x^n mod P` in normal (unreflected) bit order, P = 0x1_04C1_1DB7.
#[cfg_attr(
    any(feature = "safe", not(any(target_arch = "x86_64", target_arch = "aarch64"))),
    allow(dead_code)
)]
const fn xpow_mod(n: u32) -> u32 {
    let mut r: u32 = 1;
    let mut i = 0;
    while i < n {
        let carry = r & 0x8000_0000 != 0;
        r <<= 1;
        if carry {
            r ^= 0x04C1_1DB7;
        }
        i += 1;
    }
    r
}

/// Folding constant for a distance of `n - 32` bits (or `n + 32`), in the
/// reflected, shifted-by-one form PCLMULQDQ folding uses.
#[cfg_attr(
    any(feature = "safe", not(any(target_arch = "x86_64", target_arch = "aarch64"))),
    allow(dead_code)
)]
const fn fold_key(n: u32) -> u64 {
    (xpow_mod(n).reverse_bits() as u64) << 1
}

#[cfg(all(target_arch = "x86_64", not(feature = "safe")))]
mod simd {
    use super::fold_key;
    use core::arch::x86_64::*;

    /// Below these, the setup and final reduction cost more than they save.
    pub const MIN_LEN: usize = 128;
    pub const MIN_LEN_512: usize = 1024;

    // Fold 4 x 128 bits across 512 bits, then 1 x 128 across 128.
    const K1: u64 = fold_key(512 + 32);
    const K2: u64 = fold_key(512 - 32);
    const K3: u64 = fold_key(128 + 32);
    const K4: u64 = fold_key(128 - 32);
    const K5: u64 = fold_key(64);
    // Barrett reduction: P (reflected, with its x^32 term) and
    // floor(x^64 / P) (reflected).
    const P_X: u64 = 0x1_DB71_0641;
    const U_PRIME: u64 = 0x1_F701_1641;

    #[inline(always)]
    unsafe fn load(data: &[u8], i: usize) -> __m128i {
        debug_assert!(i + 16 <= data.len());
        // SAFETY: in bounds per the caller; unaligned load.
        unsafe { _mm_loadu_si128(data.as_ptr().add(i).cast()) }
    }

    /// Folds `a` forward by the distance `keys` encode and adds `b`.
    #[inline(always)]
    unsafe fn fold(a: __m128i, b: __m128i, keys: __m128i) -> __m128i {
        // SAFETY: PCLMULQDQ support per the callers' target features.
        unsafe {
            let lo = _mm_clmulepi64_si128(a, keys, 0x00);
            let hi = _mm_clmulepi64_si128(a, keys, 0x11);
            _mm_xor_si128(_mm_xor_si128(lo, hi), b)
        }
    }

    /// # Safety
    /// The CPU must support PCLMULQDQ and SSE4.1; `data.len() >= MIN_LEN`.
    #[target_feature(enable = "pclmulqdq,sse4.1")]
    pub unsafe fn update(crc: u32, data: &[u8]) -> u32 {
        assert!(data.len() >= MIN_LEN);
        // SAFETY: every `load(data, i)` has `i + 16 <= data.len()`: the loop
        // conditions check it; the first four need `MIN_LEN >= 64`.
        unsafe {
            let mut x0 = _mm_xor_si128(load(data, 0), _mm_cvtsi32_si128(!crc as i32));
            let mut x1 = load(data, 16);
            let mut x2 = load(data, 32);
            let mut x3 = load(data, 48);
            let mut i = 64;
            let k1k2 = _mm_set_epi64x(K2 as i64, K1 as i64);
            while i + 64 <= data.len() {
                x0 = fold(x0, load(data, i), k1k2);
                x1 = fold(x1, load(data, i + 16), k1k2);
                x2 = fold(x2, load(data, i + 32), k1k2);
                x3 = fold(x3, load(data, i + 48), k1k2);
                i += 64;
            }
            let k3k4 = _mm_set_epi64x(K4 as i64, K3 as i64);
            let mut x = fold(x0, x1, k3k4);
            x = fold(x, x2, k3k4);
            x = fold(x, x3, k3k4);
            finish(x, data, i)
        }
    }

    /// Folds the 16-byte chunks of `data[i..]` into `x` (the folded state of
    /// `data[..i]`), reduces to the CRC, and runs the tail through the table.
    ///
    /// # Safety
    /// The CPU must support PCLMULQDQ and SSE4.1.
    #[target_feature(enable = "pclmulqdq,sse4.1")]
    unsafe fn finish(mut x: __m128i, data: &[u8], mut i: usize) -> u32 {
        // SAFETY: every `load(data, i)` has `i + 16 <= data.len()`.
        unsafe {
            let k3k4 = _mm_set_epi64x(K4 as i64, K3 as i64);
            while i + 16 <= data.len() {
                x = fold(x, load(data, i), k3k4);
                i += 16;
            }

            // 128 -> 64 bits, then Barrett reduction to 32.
            let low32 = _mm_set_epi32(0, 0, 0, -1);
            x = _mm_xor_si128(_mm_clmulepi64_si128(x, k3k4, 0x10), _mm_srli_si128(x, 8));
            x = _mm_xor_si128(
                _mm_clmulepi64_si128(_mm_and_si128(x, low32), _mm_set_epi64x(0, K5 as i64), 0x00),
                _mm_srli_si128(x, 4),
            );
            let pu = _mm_set_epi64x(U_PRIME as i64, P_X as i64);
            let t1 = _mm_clmulepi64_si128(_mm_and_si128(x, low32), pu, 0x10);
            let t2 = _mm_clmulepi64_si128(_mm_and_si128(t1, low32), pu, 0x00);
            let c = _mm_extract_epi32(_mm_xor_si128(x, t2), 1) as u32;
            !super::update_table(c, &data[i..])
        }
    }

    /// Key pair (for the low and high 64 bits) folding 128 bits forward
    /// across `bits`.
    const fn keys(bits: u32) -> (i64, i64) {
        (fold_key(bits + 32) as i64, fold_key(bits - 32) as i64)
    }
    const K_2048: (i64, i64) = keys(2048);
    const K_512: (i64, i64) = keys(512);
    const K_384: (i64, i64) = keys(384);
    const K_256: (i64, i64) = keys(256);
    const K_128: (i64, i64) = keys(128);

    /// `keys` in every 128-bit lane of a 512-bit vector.
    #[inline(always)]
    unsafe fn broadcast_512((lo, hi): (i64, i64)) -> __m512i {
        // SAFETY: AVX-512F support per the callers' target features.
        unsafe { _mm512_set_epi64(hi, lo, hi, lo, hi, lo, hi, lo) }
    }

    #[inline(always)]
    unsafe fn load_512(data: &[u8], i: usize) -> __m512i {
        debug_assert!(i + 64 <= data.len());
        // SAFETY: in bounds per the caller; unaligned load.
        unsafe { _mm512_loadu_si512(data.as_ptr().add(i).cast()) }
    }

    /// `fold` on four 128-bit lanes at once.
    #[inline(always)]
    unsafe fn fold_512(a: __m512i, b: __m512i, keys: __m512i) -> __m512i {
        // SAFETY: VPCLMULQDQ and AVX-512F support per the callers' target
        // features.
        unsafe {
            let lo = _mm512_clmulepi64_epi128(a, keys, 0x00);
            let hi = _mm512_clmulepi64_epi128(a, keys, 0x11);
            // Three-way XOR.
            _mm512_ternarylogic_epi64(lo, hi, b, 0x96)
        }
    }

    /// # Safety
    /// The CPU must support AVX-512F, VPCLMULQDQ, PCLMULQDQ and SSE4.1;
    /// `data.len() >= MIN_LEN_512`.
    #[target_feature(enable = "avx512f,vpclmulqdq,pclmulqdq,sse4.1")]
    pub unsafe fn update_512(crc: u32, data: &[u8]) -> u32 {
        assert!(data.len() >= MIN_LEN_512);
        // SAFETY: every `load_512(data, i)` has `i + 64 <= data.len()`: the
        // loop conditions check it; the first four need `MIN_LEN_512 >= 256`.
        unsafe {
            let mut z0 = _mm512_xor_si512(
                load_512(data, 0),
                _mm512_zextsi128_si512(_mm_cvtsi32_si128(!crc as i32)),
            );
            let mut z1 = load_512(data, 64);
            let mut z2 = load_512(data, 128);
            let mut z3 = load_512(data, 192);
            let mut i = 256;
            let k2048 = broadcast_512(K_2048);
            while i + 256 <= data.len() {
                z0 = fold_512(z0, load_512(data, i), k2048);
                z1 = fold_512(z1, load_512(data, i + 64), k2048);
                z2 = fold_512(z2, load_512(data, i + 128), k2048);
                z3 = fold_512(z3, load_512(data, i + 192), k2048);
                i += 256;
            }
            let k512 = broadcast_512(K_512);
            let mut z = fold_512(z0, z1, k512);
            z = fold_512(z, z2, k512);
            z = fold_512(z, z3, k512);
            while i + 64 <= data.len() {
                z = fold_512(z, load_512(data, i), k512);
                i += 64;
            }
            // Fold lanes 0-2 onto lane 3 (keys of 0 leave lane 3 out).
            let ((l384, h384), (l256, h256), (l128, h128)) = (K_384, K_256, K_128);
            let keys = _mm512_set_epi64(0, 0, h128, l128, h256, l256, h384, l384);
            let t = fold_512(z, _mm512_setzero_si512(), keys);
            let mut x = _mm512_extracti32x4_epi32::<3>(z);
            x = _mm_xor_si128(x, _mm512_castsi512_si128(t));
            x = _mm_xor_si128(x, _mm512_extracti32x4_epi32::<1>(t));
            x = _mm_xor_si128(x, _mm512_extracti32x4_epi32::<2>(t));
            finish(x, data, i)
        }
    }
}

#[cfg(all(target_arch = "aarch64", not(feature = "safe")))]
mod pmull {
    use super::fold_key;
    use core::arch::aarch64::vmull_p64;

    /// Below this, the setup costs more than it saves.
    pub const MIN_LEN: usize = 128;

    /// Key pairs (for the low and high 64 bits) folding 128 bits forward
    /// across 512 and 128 bits.
    const K_512: (u64, u64) = (fold_key(512 + 32), fold_key(512 - 32));
    const K_128: (u64, u64) = (fold_key(128 + 32), fold_key(128 - 32));

    #[inline(always)]
    fn load(data: &[u8], i: usize) -> u128 {
        u128::from_le_bytes(data[i..i + 16].try_into().unwrap())
    }

    /// Folds `a` forward by the distance `keys` encode and adds `b`.
    #[inline(always)]
    unsafe fn fold(a: u128, b: u128, (lo, hi): (u64, u64)) -> u128 {
        // SAFETY: PMULL support per the callers' target features.
        unsafe { vmull_p64(a as u64, lo) ^ vmull_p64((a >> 64) as u64, hi) ^ b }
    }

    /// # Safety
    /// The CPU must support PMULL (the `aes` feature).
    #[target_feature(enable = "neon,aes")]
    pub unsafe fn update(crc: u32, data: &[u8]) -> u32 {
        assert!(data.len() >= MIN_LEN);
        // SAFETY: PMULL support per this function's target features.
        unsafe {
            let mut x0 = load(data, 0) ^ (!crc) as u128;
            let mut x1 = load(data, 16);
            let mut x2 = load(data, 32);
            let mut x3 = load(data, 48);
            let mut i = 64;
            while i + 64 <= data.len() {
                x0 = fold(x0, load(data, i), K_512);
                x1 = fold(x1, load(data, i + 16), K_512);
                x2 = fold(x2, load(data, i + 32), K_512);
                x3 = fold(x3, load(data, i + 48), K_512);
                i += 64;
            }
            let mut x = fold(x0, x1, K_128);
            x = fold(x, x2, K_128);
            x = fold(x, x3, K_128);
            while i + 16 <= data.len() {
                x = fold(x, load(data, i), K_128);
                i += 16;
            }
            // `x` is now a 16-byte message with the same CRC as `data[..i]`
            // processed from a zero state, so the table finishes it.
            let c = super::update_table_serial(0, &x.to_le_bytes());
            !super::update_table_serial(c, &data[i..])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bitwise(data: &[u8]) -> u32 {
        let mut c = !0u32;
        for &b in data {
            c ^= b as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { (c >> 1) ^ POLY } else { c >> 1 };
            }
        }
        !c
    }

    /// Each SIMD path directly, so CPUs with AVX-512 still test the
    /// 128-bit one at long lengths.
    #[cfg(all(target_arch = "x86_64", not(feature = "safe")))]
    #[test]
    fn simd_paths_match_bitwise() {
        let data: Vec<u8> = (0..20_000u32).map(|i| (i.wrapping_mul(2_246_822_519) >> 11) as u8).collect();
        let has_128 = is_x86_feature_detected!("pclmulqdq") && is_x86_feature_detected!("sse4.1");
        let has_512 = has_128 && is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("vpclmulqdq");
        for start in 0..8 {
            for len in (simd::MIN_LEN..3000).step_by(5).chain([8191, 19_000]) {
                let d = &data[start..start + len];
                let want = bitwise(d);
                if has_128 {
                    // SAFETY: features detected above.
                    assert_eq!(unsafe { simd::update(0, d) }, want, "128 len={len}");
                    assert_eq!(unsafe { simd::update(0x1234_5678, d) }, crc_from(0x1234_5678, d));
                }
                if has_512 && len >= simd::MIN_LEN_512 {
                    // SAFETY: features detected above.
                    assert_eq!(unsafe { simd::update_512(0, d) }, want, "512 len={len}");
                    assert_eq!(unsafe { simd::update_512(0x1234_5678, d) }, crc_from(0x1234_5678, d));
                }
            }
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "safe")))]
    #[test]
    fn pmull_path_matches_bitwise() {
        if !std::arch::is_aarch64_feature_detected!("pmull") {
            return;
        }
        let data: Vec<u8> = (0..20_000u32).map(|i| (i.wrapping_mul(2_246_822_519) >> 11) as u8).collect();
        for start in 0..8 {
            for len in (pmull::MIN_LEN..3000).step_by(5).chain([8191, 19_000]) {
                let d = &data[start..start + len];
                // SAFETY: feature detected above.
                assert_eq!(unsafe { pmull::update(0, d) }, bitwise(d), "len={len}");
                assert_eq!(unsafe { pmull::update(0x1234_5678, d) }, crc_from(0x1234_5678, d));
            }
        }
    }

    /// Bitwise CRC continuing from the CRC `prev` of an earlier prefix.
    #[cfg(all(any(target_arch = "x86_64", target_arch = "aarch64"), not(feature = "safe")))]
    fn crc_from(prev: u32, data: &[u8]) -> u32 {
        let mut c = !prev;
        for &b in data {
            c ^= b as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { (c >> 1) ^ POLY } else { c >> 1 };
            }
        }
        !c
    }

    #[test]
    fn fold_keys_match_published_constants() {
        assert_eq!(fold_key(512 + 32), 0x1_5444_2BD4);
        assert_eq!(fold_key(512 - 32), 0x1_C6E4_1596);
        assert_eq!(fold_key(128 + 32), 0x1_7519_97D0);
        assert_eq!(fold_key(128 - 32), 0x0_CCAA_009E);
        assert_eq!(fold_key(64), 0x1_63CD_6124);
    }

    #[test]
    fn matches_bitwise_at_every_length_and_alignment() {
        let data: Vec<u8> = (0..20_000u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8).collect();
        for start in 0..16 {
            // Covers the table, 128-bit and 512-bit paths and their tails.
            let lens = (0..600).chain((1000..1400).step_by(7)).chain([2047, 4096, 4096 + 13, 8191, 19_000]);
            for len in lens {
                let d = &data[start..start + len];
                assert_eq!(crc32(d), bitwise(d), "start={start} len={len}");
                // Split anywhere: incremental equals one-shot.
                let mut h = Crc32::new();
                let cut = len / 3;
                h.update(&d[..cut]);
                h.update(&d[cut..]);
                assert_eq!(h.finalize(), bitwise(d));
            }
        }
    }
}
