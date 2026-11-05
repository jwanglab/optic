//! Memory primitives for the inflate fast loop.
//!
//! By default these skip bounds checks and walk raw pointers (as libdeflate
//! does, which frees registers in the decode loop); the caller guarantees the
//! documented preconditions through its input/output margin checks. With the
//! `safe` feature the same API is implemented with bounds-checked slices and
//! indices, and the crate forbids `unsafe` entirely.
//!
//! Apart from the SIMD CRC-32 in `crc32.rs`, this module holds every
//! `unsafe` block in the crate.

#[cfg(not(feature = "safe"))]
mod imp {
    use std::marker::PhantomData;

    /// Optimization barrier: forces `entry` and `bitbuf` to be computed at this
    /// point in program order. Emits no instructions.
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "arm",
        target_arch = "aarch64",
        target_arch = "riscv32",
        target_arch = "riscv64",
        target_arch = "loongarch64"
    ))]
    #[inline(always)]
    pub fn schedule_barrier(entry: &mut u32, bitbuf: &mut u64) {
        // The template is only a comment; the 32-bit register modifier (which
        // differs per architecture) just silences the sub-register lint.
        macro_rules! barrier {
            ($template:literal) => {
                // SAFETY: the asm template is empty; it only claims to read
                // and write two registers, which it leaves unchanged.
                unsafe {
                    core::arch::asm!(
                        $template,
                        inout(reg) * entry,
                        inout(reg) * bitbuf,
                        options(nomem, nostack, preserves_flags)
                    )
                }
            };
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        barrier!("/* {0:e} {1} */");
        #[cfg(target_arch = "aarch64")]
        barrier!("/* {0:w} {1} */");
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        barrier!("/* {0} {1} */");
    }

    /// Fallback for targets without stable `asm!` or a `reg` register class
    /// (e.g. wasm): no barrier, so LLVM may schedule the preload later.
    #[cfg(not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "arm",
        target_arch = "aarch64",
        target_arch = "riscv32",
        target_arch = "riscv64",
        target_arch = "loongarch64"
    )))]
    #[inline(always)]
    pub fn schedule_barrier(_entry: &mut u32, _bitbuf: &mut u64) {}

    /// Hides a pointer from LLVM so it does not auto-vectorize the rarely-run
    /// long-match loops (the vectorized version needs overlap checks and extra
    /// registers that end up costing the hot path).
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "arm",
        target_arch = "aarch64",
        target_arch = "riscv32",
        target_arch = "riscv64",
        target_arch = "loongarch64"
    ))]
    #[inline(always)]
    fn opaque_ptr(p: *mut u8) -> *mut u8 {
        let mut addr = p.addr();
        // SAFETY: empty asm template; the register is left unchanged.
        unsafe {
            core::arch::asm!("/* {0} */", inout(reg) addr, options(nomem, nostack, preserves_flags))
        };
        p.with_addr(addr)
    }

    /// Fallback: identity, so the long-match loops may be auto-vectorized.
    #[cfg(not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "arm",
        target_arch = "aarch64",
        target_arch = "riscv32",
        target_arch = "riscv64",
        target_arch = "loongarch64"
    )))]
    #[inline(always)]
    fn opaque_ptr(p: *mut u8) -> *mut u8 {
        p
    }

    /// `x` with all bits at positions `>= n & 0xFF` cleared (BMI2 `bzhi`).
    ///
    /// # Safety
    /// The CPU must support BMI2.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    pub unsafe fn bzhi(x: u64, n: u32) -> u64 {
        // SAFETY: BMI2 support per the precondition.
        unsafe { core::arch::x86_64::_bzhi_u64(x, n) }
    }

    /// Portable stand-in so callers compile everywhere; never called with
    /// BMI2 off.
    #[cfg(not(target_arch = "x86_64"))]
    #[inline(always)]
    pub unsafe fn bzhi(x: u64, n: u32) -> u64 {
        x & ((1u64 << (n & 0xFF)) - 1)
    }

    /// # Safety
    /// `pos + 8 <= input.len()`.
    #[inline(always)]
    pub unsafe fn load_u64_le(input: &[u8], pos: usize) -> u64 {
        debug_assert!(pos + 8 <= input.len());
        // SAFETY: in bounds per the precondition; unaligned read.
        u64::from_le(unsafe { input.as_ptr().add(pos).cast::<u64>().read_unaligned() })
    }

    /// Input cursor for the fast loop.
    pub struct FastIn<'a> {
        base: *const u8,
        ptr: *const u8,
        fast_end: *const u8,
        #[cfg(debug_assertions)]
        end: *const u8,
        _input: PhantomData<&'a [u8]>,
    }

    impl<'a> FastIn<'a> {
        /// Requires `pos <= fast_end <= input.len()`.
        #[inline(always)]
        pub fn new(input: &'a [u8], pos: usize, fast_end: usize) -> Self {
            assert!(pos <= fast_end && fast_end <= input.len());
            let base = input.as_ptr();
            // SAFETY: both offsets are within the slice, checked above.
            unsafe {
                Self {
                    base,
                    ptr: base.add(pos),
                    fast_end: base.add(fast_end),
                    #[cfg(debug_assertions)]
                    end: base.add(input.len()),
                    _input: PhantomData,
                }
            }
        }

        /// True while the cursor is before `fast_end`.
        #[inline(always)]
        pub fn has_room(&self) -> bool {
            self.ptr < self.fast_end
        }

        #[inline(always)]
        pub fn pos(&self) -> usize {
            self.ptr as usize - self.base as usize
        }

        /// Loads 8 bytes at the cursor.
        ///
        /// # Safety
        /// The cursor must be at least 8 bytes before the end of the input.
        #[inline(always)]
        pub unsafe fn load_u64_le(&self) -> u64 {
            #[cfg(debug_assertions)]
            assert!(self.ptr as usize + 8 <= self.end as usize);
            // SAFETY: in bounds per the precondition; unaligned read.
            u64::from_le(unsafe { self.ptr.cast::<u64>().read_unaligned() })
        }

        /// # Safety
        /// The cursor must stay within the input.
        #[inline(always)]
        pub unsafe fn advance(&mut self, n: usize) {
            // SAFETY: per the precondition.
            self.ptr = unsafe { self.ptr.add(n) };
        }
    }

    /// Output cursor for the fast loop. Mutably borrows the output buffer
    /// until dropped.
    pub struct FastOut<'a> {
        base: *mut u8,
        ptr: *mut u8,
        fast_end: *mut u8,
        #[cfg(debug_assertions)]
        end: *mut u8,
        _out: PhantomData<&'a mut [u8]>,
    }

    impl<'a> FastOut<'a> {
        /// Requires `pos <= fast_end <= out.len()`.
        #[inline(always)]
        pub fn new(out: &'a mut [u8], pos: usize, fast_end: usize) -> Self {
            assert!(pos <= fast_end && fast_end <= out.len());
            let base = out.as_mut_ptr();
            // SAFETY: both offsets are within the slice, checked above.
            unsafe {
                Self {
                    base,
                    ptr: base.add(pos),
                    fast_end: base.add(fast_end),
                    #[cfg(debug_assertions)]
                    end: base.add(out.len()),
                    _out: PhantomData,
                }
            }
        }

        /// True while the cursor is before `fast_end`.
        #[inline(always)]
        pub fn has_room(&self) -> bool {
            self.ptr < self.fast_end
        }

        /// Bytes written so far (from the start of the buffer).
        #[inline(always)]
        pub fn pos(&self) -> usize {
            self.ptr as usize - self.base as usize
        }

        /// # Safety
        /// The cursor must be before the end of the buffer.
        #[inline(always)]
        pub unsafe fn push(&mut self, v: u8) {
            #[cfg(debug_assertions)]
            assert!(self.ptr < self.end);
            // SAFETY: in bounds per the precondition.
            unsafe {
                *self.ptr = v;
                self.ptr = self.ptr.add(1);
            }
        }

        /// Copies an LZ77 match of `len` bytes from `dist` back using 8-byte
        /// words (libdeflate's scheme), writing up to 39 bytes of garbage past
        /// the end of the match.
        ///
        /// # Safety
        /// `1 <= dist <= self.pos()`, and `max(len, 40) + 8` bytes must remain
        /// before the end of the buffer.
        #[inline(always)]
        pub unsafe fn copy_match(&mut self, dist: usize, len: usize) {
            #[cfg(debug_assertions)]
            assert!(
                dist >= 1
                    && dist <= self.pos()
                    && self.ptr as usize + len.max(40) + 8 <= self.end as usize
            );
            // SAFETY: every word written starts below `ptr + max(len, 40)`, so
            // ends in bounds; every word read starts at or after
            // `ptr - dist >= base` and below a word already written or the
            // current destination.
            unsafe {
                let mut dst = self.ptr;
                let end = dst.add(len);
                let mut src = dst.sub(dist).cast_const();
                self.ptr = end;
                if dist >= 8 {
                    // Words never overlap their own destination. Five cover
                    // most matches without a loop branch.
                    for _ in 0..5 {
                        copy8(src, dst);
                        src = src.add(8);
                        dst = dst.add(8);
                    }
                    while dst < end {
                        copy8(src, dst);
                        src = src.add(8);
                        dst = opaque_ptr(dst.add(8));
                    }
                } else if dist == 1 {
                    let v = 0x0101_0101_0101_0101u64 * *src as u64;
                    for _ in 0..4 {
                        dst.cast::<u64>().write_unaligned(v);
                        dst = dst.add(8);
                    }
                    while dst < end {
                        dst.cast::<u64>().write_unaligned(v);
                        dst = opaque_ptr(dst.add(8));
                    }
                } else {
                    // Each word copy gets the first `dist` bytes right (the
                    // rest may read not-yet-written bytes); advance by `dist`.
                    loop {
                        copy8(src, dst);
                        src = src.add(dist);
                        dst = dst.add(dist);
                        if dst >= end {
                            break;
                        }
                    }
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn copy8(src: *const u8, dst: *mut u8) {
        // SAFETY: caller guarantees both 8-byte ranges are valid.
        unsafe {
            dst.cast::<u64>()
                .write_unaligned(src.cast::<u64>().read_unaligned())
        }
    }
}

#[cfg(feature = "safe")]
mod imp {
    /// No-op: inline asm needs `unsafe`, and the safe alternative
    /// (`std::hint::black_box`) round-trips through the stack and is slower
    /// than no barrier at all.
    #[inline(always)]
    pub fn schedule_barrier(_entry: &mut u32, _bitbuf: &mut u64) {}

    /// Only reached through the BMI2 fast loop, which the safe build
    /// never uses.
    #[inline(always)]
    pub fn bzhi(x: u64, n: u32) -> u64 {
        x & ((1u64 << (n & 0xFF)) - 1)
    }

    #[inline(always)]
    pub fn load_u64_le(input: &[u8], pos: usize) -> u64 {
        u64::from_le_bytes(input[pos..pos + 8].try_into().unwrap())
    }

    pub struct FastIn<'a> {
        input: &'a [u8],
        pos: usize,
        fast_end: usize,
    }

    impl<'a> FastIn<'a> {
        #[inline(always)]
        pub fn new(input: &'a [u8], pos: usize, fast_end: usize) -> Self {
            assert!(pos <= fast_end && fast_end <= input.len());
            Self {
                input,
                pos,
                fast_end,
            }
        }

        #[inline(always)]
        pub fn has_room(&self) -> bool {
            self.pos < self.fast_end
        }

        #[inline(always)]
        pub fn pos(&self) -> usize {
            self.pos
        }

        #[inline(always)]
        pub fn load_u64_le(&self) -> u64 {
            load_u64_le(self.input, self.pos)
        }

        #[inline(always)]
        pub fn advance(&mut self, n: usize) {
            self.pos += n;
        }
    }

    pub struct FastOut<'a> {
        out: &'a mut [u8],
        pos: usize,
        fast_end: usize,
    }

    impl<'a> FastOut<'a> {
        #[inline(always)]
        pub fn new(out: &'a mut [u8], pos: usize, fast_end: usize) -> Self {
            assert!(pos <= fast_end && fast_end <= out.len());
            Self { out, pos, fast_end }
        }

        #[inline(always)]
        pub fn has_room(&self) -> bool {
            self.pos < self.fast_end
        }

        #[inline(always)]
        pub fn pos(&self) -> usize {
            self.pos
        }

        #[inline(always)]
        pub fn push(&mut self, v: u8) {
            self.out[self.pos] = v;
            self.pos += 1;
        }

        /// Same algorithm and preconditions as the unchecked version, with
        /// every access bounds-checked.
        #[inline(always)]
        pub fn copy_match(&mut self, dist: usize, len: usize) {
            let out = &mut *self.out;
            let mut dst = self.pos;
            let end = dst + len;
            let mut src = dst - dist;
            self.pos = end;
            if dist >= 8 {
                for _ in 0..5 {
                    copy8(out, src, dst);
                    src += 8;
                    dst += 8;
                }
                while dst < end {
                    copy8(out, src, dst);
                    src += 8;
                    dst += 8;
                }
            } else if dist == 1 {
                let v = (0x0101_0101_0101_0101u64 * out[src] as u64).to_ne_bytes();
                for _ in 0..4 {
                    out[dst..dst + 8].copy_from_slice(&v);
                    dst += 8;
                }
                while dst < end {
                    out[dst..dst + 8].copy_from_slice(&v);
                    dst += 8;
                }
            } else {
                loop {
                    copy8(out, src, dst);
                    src += dist;
                    dst += dist;
                    if dst >= end {
                        break;
                    }
                }
            }
        }
    }

    #[inline(always)]
    fn copy8(out: &mut [u8], src: usize, dst: usize) {
        let word: [u8; 8] = out[src..src + 8].try_into().unwrap();
        out[dst..dst + 8].copy_from_slice(&word);
    }
}

pub(crate) use imp::*;
