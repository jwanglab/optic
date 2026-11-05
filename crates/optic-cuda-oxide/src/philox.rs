//! Philox4x32-10 counter-based RNG: four u32 per round, one stream per (seed, element).

const M0: u32 = 0xD251_1F53;
const M1: u32 = 0xCD9E_8D57;
const W0: u32 = 0x9E37_79B9;
const W1: u32 = 0xBB67_AE85;

#[inline(always)]
fn umulhi(a: u32, b: u32) -> u32 { ((a as u64 * b as u64) >> 32) as u32 }

/// Per-thread state: key = seed, counter = (offset lo, offset hi, thread lo, thread hi), four cached outputs.
pub struct Philox {
    k0: u32, k1: u32,
    c0: u32, c1: u32, c2: u32, c3: u32,
    o0: u32, o1: u32, o2: u32, o3: u32,
    idx: u32,
}

impl Philox {
    #[inline(always)]
    pub fn new(seed: u64, thread_id: u64) -> Self {
        Philox { k0: seed as u32, k1: (seed >> 32) as u32, c0: 0, c1: 0, c2: thread_id as u32, c3: (thread_id >> 32) as u32,
                 o0: 0, o1: 0, o2: 0, o3: 0, idx: 4 }
    }

    /// Ten rounds with key bumping between them, then advance the 64-bit offset.
    #[inline(always)]
    fn next4(&mut self) {
        let (mut x0, mut x1, mut x2, mut x3) = (self.c0, self.c1, self.c2, self.c3);
        let (mut k0, mut k1) = (self.k0, self.k1);
        let mut r = 0;
        while r < 10 {
            if r > 0 { k0 = k0.wrapping_add(W0); k1 = k1.wrapping_add(W1); }
            let (hi0, lo0) = (umulhi(M0, x0), M0.wrapping_mul(x0));
            let (hi1, lo1) = (umulhi(M1, x2), M1.wrapping_mul(x2));
            x0 = hi1 ^ x1 ^ k0;
            x1 = lo1;
            x2 = hi0 ^ x3 ^ k1;
            x3 = lo0;
            r += 1;
        }
        (self.o0, self.o1, self.o2, self.o3) = (x0, x1, x2, x3);
        self.c0 = self.c0.wrapping_add(1);
        if self.c0 == 0 { self.c1 = self.c1.wrapping_add(1); }
        self.idx = 0;
    }

    /// A uniform float in (0, 1]: (v + 0.5) / 2^32, never exactly 0.
    #[inline(always)]
    pub fn uniform(&mut self) -> f32 {
        if self.idx >= 4 { self.next4(); }
        let v = match self.idx { 0 => self.o0, 1 => self.o1, 2 => self.o2, _ => self.o3 };
        self.idx += 1;
        (v as f32 + 0.5) * (1.0 / 4_294_967_296.0)
    }
}
