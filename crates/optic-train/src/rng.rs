//! deterministic psuedorandom number generator matching StdRng output

pub struct StdRng {
    key: [u32; 8],
    /// block counter of the next refill (64-bit, words 12-13 of the state; the stream id, words 14-15, is 0)
    counter: u64,
    buf: [u32; 64],
    index: usize,
}

#[inline(always)]
fn quarter(x: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    x[a] = x[a].wrapping_add(x[b]); x[d] = (x[d] ^ x[a]).rotate_left(16);
    x[c] = x[c].wrapping_add(x[d]); x[b] = (x[b] ^ x[c]).rotate_left(12);
    x[a] = x[a].wrapping_add(x[b]); x[d] = (x[d] ^ x[a]).rotate_left(8);
    x[c] = x[c].wrapping_add(x[d]); x[b] = (x[b] ^ x[c]).rotate_left(7);
}

impl StdRng {
    /// `SeedableRng::seed_from_u64`: the 32-byte key from PCG32 outputs.
    pub fn seed_from_u64(mut state: u64) -> Self {
        let mut key = [0u32; 8];
        for k in &mut key {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(11634580027462260723);
            *k = ((((state >> 18) ^ state) >> 27) as u32).rotate_right((state >> 59) as u32);
        }
        StdRng { key, counter: 0, buf: [0; 64], index: 64 }
    }

    /// Four ChaCha12 blocks at once, as rand_chacha buffers them.
    fn refill(&mut self) {
        for blk in 0..4 {
            let ctr = self.counter.wrapping_add(blk as u64);
            let mut s = [0u32; 16];
            s[..4].copy_from_slice(&[0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574]);
            s[4..12].copy_from_slice(&self.key);
            s[12] = ctr as u32;
            s[13] = (ctr >> 32) as u32;
            let mut x = s;
            for _ in 0..6 {
                quarter(&mut x, 0, 4, 8, 12); quarter(&mut x, 1, 5, 9, 13); quarter(&mut x, 2, 6, 10, 14); quarter(&mut x, 3, 7, 11, 15);
                quarter(&mut x, 0, 5, 10, 15); quarter(&mut x, 1, 6, 11, 12); quarter(&mut x, 2, 7, 8, 13); quarter(&mut x, 3, 4, 9, 14);
            }
            for i in 0..16 { self.buf[blk * 16 + i] = x[i].wrapping_add(s[i]); }
        }
        self.counter = self.counter.wrapping_add(4);
    }

    /// `RngCore::next_u32` (rand_core BlockRng).
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= 64 { self.refill(); self.index = 0; }
        self.index += 1;
        self.buf[self.index - 1]
    }

    /// `RngCore::next_u64` (rand_core BlockRng: low word first, straddling a refill when one word is left).
    pub fn next_u64(&mut self) -> u64 {
        let i = self.index;
        if i < 63 {
            self.index += 2;
            (self.buf[i + 1] as u64) << 32 | self.buf[i] as u64
        } else if i >= 64 {
            self.refill();
            self.index = 2;
            (self.buf[1] as u64) << 32 | self.buf[0] as u64
        } else {
            let x = self.buf[63] as u64;
            self.refill();
            self.index = 1;
            (self.buf[0] as u64) << 32 | x
        }
    }

    /// `rng.gen::<f64>()`: 53 high bits, [0, 1).
    pub fn f64(&mut self) -> f64 { (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64) }

    /// `rng.gen::<f32>()`: 24 high bits, [0, 1).
    pub fn f32(&mut self) -> f32 { (self.next_u32() >> 8) as f32 * (1.0 / (1u32 << 24) as f32) }

    /// `rng.gen_range(low..=high)` for u32 (UniformInt::sample_single_inclusive, widening-multiply rejection).
    fn u32_incl(&mut self, low: u32, high: u32) -> u32 {
        let range = high.wrapping_sub(low).wrapping_add(1);
        if range == 0 { return self.next_u32(); }
        let zone = (range << range.leading_zeros()).wrapping_sub(1);
        loop {
            let m = self.next_u32() as u64 * range as u64;
            if (m as u32) <= zone { return low.wrapping_add((m >> 32) as u32); }
        }
    }

    /// `rng.gen_range(0..n)` for usize on a 64-bit target (u64 draws, 128-bit widening multiply).
    pub fn below_usize(&mut self, n: usize) -> usize {
        assert!(n > 0, "below_usize: empty range");
        let range = n as u64;
        let zone = (range << range.leading_zeros()).wrapping_sub(1);
        loop {
            let m = self.next_u64() as u128 * range as u128;
            if (m as u64) <= zone { return (m >> 64) as usize; }
        }
    }

    /// `SliceRandom::shuffle` (Durstenfeld from the end; u32 index draws while the bound fits).
    pub fn shuffle<T>(&mut self, s: &mut [T]) {
        for i in (1..s.len()).rev() {
            let j = if i < u32::MAX as usize { self.u32_incl(0, i as u32) as usize } else { self.below_usize(i + 1) };
            s.swap(i, j);
        }
    }

    /// `rand::seq::index::sample(rng, length, amount)`: the same algorithm choice (Floyd / in-place / rejection) and
    /// draws, so the same indices in the same order.
    pub fn sample_indices(&mut self, length: usize, amount: usize) -> Vec<usize> {
        assert!(amount <= length && length <= u32::MAX as usize, "sample_indices: amount {amount} of length {length}");
        let (len, amt) = (length as u32, amount as u32);
        let small = length < 500_000;
        let inplace = if amt < 163 {
            let (c0, c1) = if small { (1.6f32, 10.0f32) } else { (8.0 / 45.0, 70.0 / 9.0) };
            amt > 11 && (len as f32) < (c1 + c0 * amt as f32) * amt as f32
        } else {
            (len as f32) < (if small { 270.0f32 } else { 330.0 / 9.0 }) * amt as f32
        };
        let v: Vec<u32> = if inplace {
            let mut ix: Vec<u32> = (0..len).collect();
            for i in 0..amt { let j = self.u32_incl(i, len - 1); ix.swap(i as usize, j as usize); }
            ix.truncate(amount);
            ix
        } else if amt < 163 {
            // Floyd's algorithm
            let shuffle = amt < 50;
            let mut ix: Vec<u32> = Vec::with_capacity(amount);
            for j in len - amt..len {
                let t = self.u32_incl(0, j);
                if shuffle {
                    if let Some(p) = ix.iter().position(|&x| x == t) { ix.insert(p, j); continue; }
                } else if ix.contains(&t) { ix.push(j); continue; }
                ix.push(t);
            }
            if !shuffle { for i in (1..amt).rev() { let k = self.u32_incl(0, i); ix.swap(i as usize, k as usize); } }
            ix
        } else {
            // rejection with a set of the indices taken; Uniform::new(0, len) rejects by modulo, unlike gen_range
            let z = (u32::MAX - len + 1) % len;
            let zone = u32::MAX - z;
            let mut draw = || loop {
                let m = self.next_u32() as u64 * len as u64;
                if (m as u32) <= zone { return (m >> 32) as u32; }
            };
            let mut seen = std::collections::HashSet::with_capacity(amount);
            let mut ix = Vec::with_capacity(amount);
            for _ in 0..amount {
                let mut p = draw();
                while !seen.insert(p) { p = draw(); }
                ix.push(p);
            }
            ix
        };
        v.into_iter().map(|x| x as usize).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::StdRng;

    /// Golden values from rand 0.8.8 `StdRng::seed_from_u64(42)`.
    #[test]
    fn matches_rand_0_8() {
        let mut r = StdRng::seed_from_u64(42);
        let got: Vec<u32> = (0..4).map(|_| r.next_u32()).collect();
        assert_eq!(got, GOLD_U32);
        let mut r = StdRng::seed_from_u64(7);
        let mut v: Vec<u32> = (0..10).collect();
        r.shuffle(&mut v);
        assert_eq!(v, GOLD_SHUFFLE);
        assert_eq!(r.f64().to_bits(), GOLD_F64);
        let mut s = r.sample_indices(1000, 200);
        s.sort_unstable();
        assert_eq!(&s[..5], GOLD_SAMPLE);
    }
    const GOLD_U32: [u32; 4] = [572990626, 2261546851, 1068323197, 2330987027];
    const GOLD_SHUFFLE: [u32; 10] = [5, 7, 9, 6, 3, 8, 2, 1, 0, 4];
    const GOLD_F64: u64 = 0x3fe7cb5bd1459d61;
    const GOLD_SAMPLE: &[usize] = &[5, 18, 23, 26, 28];
}
