//! Dense f32 tensors and the safetensors file format (an 8-byte little-endian header length, a JSON header of
//! name -> {dtype, shape, data_offsets}, then the raw little-endian data).
//!
//! Loading streams the file tensor by tensor in 16 MB reads straight into each tensor's buffer, so peak memory is the
//! model plus one read chunk (wasm's linear memory is capped at 4 GB and has no mmap). BF16 and F16 are widened to f32
//! while streaming, which is exact. Saving writes f32 only, tensors in name order, so the same weights always give the
//! same bytes (federated `merge` and `apply` compare sha256 sums of the files they write).

use anyhow::{anyhow, bail, Context, Result};
use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        debug_assert_eq!(shape.iter().product::<usize>(), data.len());
        Tensor { shape, data }
    }
    pub fn zeros(shape: Vec<usize>) -> Self {
        let n = shape.iter().product();
        Tensor { shape, data: vec![0.0; n] }
    }
    /// Size of dimension `i` (1 past the last dimension).
    pub fn dim(&self, i: usize) -> usize { self.shape.get(i).copied().unwrap_or(1) }
    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
}

pub type Tensors = BTreeMap<String, Tensor>;

fn read_chunked(f: &mut std::fs::File, buf: &mut [u8]) -> std::io::Result<()> {
    for chunk in buf.chunks_mut(16 << 20) { f.read_exact(chunk)?; }
    Ok(())
}

/// IEEE half -> f32 (exact).
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, _) => {
            // subnormal: renormalise
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 { m <<= 1; e -= 1; }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, _) => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

pub fn load(path: &Path) -> Result<Tensors> {
    let mut f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let file_len = f.metadata()?.len();
    let mut n = [0u8; 8];
    f.read_exact(&mut n)?;
    let n = u64::from_le_bytes(n);
    if n > file_len.saturating_sub(8) { bail!("{}: not a safetensors file", path.display()); }
    let mut head = vec![0u8; n as usize];
    read_chunked(&mut f, &mut head)?;
    let base = 8 + n;
    let meta: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&head)
        .with_context(|| format!("{}: safetensors header", path.display()))?;
    let mut out = Tensors::new();
    for (name, v) in meta.iter().filter(|(k, _)| *k != "__metadata__") {
        let dtype = v["dtype"].as_str().ok_or_else(|| anyhow!("{name}: no dtype"))?;
        let shape: Vec<usize> = v["shape"].as_array().ok_or_else(|| anyhow!("{name}: no shape"))?
            .iter().map(|d| d.as_u64().map(|d| d as usize).ok_or_else(|| anyhow!("{name}: bad shape")))
            .collect::<Result<_>>()?;
        let off = v["data_offsets"].as_array().ok_or_else(|| anyhow!("{name}: no data_offsets"))?;
        let (a, b) = match (off.first().and_then(|x| x.as_u64()), off.get(1).and_then(|x| x.as_u64())) {
            (Some(a), Some(b)) if a <= b && base + b <= file_len => (a, b),
            _ => bail!("{name}: bad data_offsets"),
        };
        let numel: usize = shape.iter().product();
        let width = match dtype { "F32" => 4, "BF16" | "F16" => 2, d => bail!("{name}: unsupported dtype {d}") };
        if (b - a) as usize != numel * width { bail!("{name}: {} bytes for {numel} {dtype} values", b - a); }
        f.seek(SeekFrom::Start(base + a))?;
        let data = if width == 4 {
            let mut x = vec![0f32; numel];
            // SAFETY: f32 has no invalid bit patterns; the bytes are little-endian f32, fixed up below on big-endian
            let bytes = unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut u8, numel * 4) };
            read_chunked(&mut f, bytes)?;
            if cfg!(target_endian = "big") { x.iter_mut().for_each(|v| *v = f32::from_bits(u32::from_le(v.to_bits()))); }
            x
        } else {
            let mut x = Vec::with_capacity(numel);
            let mut chunk = vec![0u8; (16 << 20).min(numel * 2)];
            let mut left = numel * 2;
            while left > 0 {
                let k = left.min(chunk.len());
                f.read_exact(&mut chunk[..k])?;
                let halves = chunk[..k].chunks_exact(2).map(|h| u16::from_le_bytes([h[0], h[1]]));
                if dtype == "BF16" { x.extend(halves.map(|h| f32::from_bits((h as u32) << 16))); }   // bf16 = top half of an f32
                else { x.extend(halves.map(f16_to_f32)); }
                left -= k;
            }
            x
        };
        out.insert(name.clone(), Tensor { shape, data });
    }
    Ok(out)
}

pub fn save(tensors: &Tensors, path: &Path) -> Result<()> {
    // the header exactly as the reference `safetensors` writer lays it out (keys dtype, shape, data_offsets), so files
    // written here and by other tools are byte-identical for the same tensors
    let mut head = String::from("{");
    let mut off = 0usize;
    for (i, (name, t)) in tensors.iter().enumerate() {
        let n = t.data.len() * 4;
        let shape: Vec<String> = t.shape.iter().map(|d| d.to_string()).collect();
        head += &format!("{}{}:{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{off},{}]}}",
                         if i > 0 { "," } else { "" }, serde_json::to_string(name)?, shape.join(","), off + n);
        off += n;
    }
    head.push('}');
    let mut head = head.into_bytes();
    while head.len() % 8 != 0 { head.push(b' '); }   // keep the data 8-byte aligned, as the reference implementation does
    let mut w = std::io::BufWriter::with_capacity(16 << 20, std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?);
    w.write_all(&(head.len() as u64).to_le_bytes())?;
    w.write_all(&head)?;
    for t in tensors.values() {
        for c in t.data.chunks(1 << 20) {
            let bytes: Vec<u8> = c.iter().flat_map(|v| v.to_le_bytes()).collect();
            w.write_all(&bytes)?;
        }
    }
    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_widening() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));     // smallest subnormal
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert!(f16_to_f32(0x7c00).is_infinite());
    }

    #[test]
    fn round_trip() {
        let dir = std::env::temp_dir().join(format!("optic-st-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.safetensors");
        let mut t = Tensors::new();
        t.insert("b".into(), Tensor::new(vec![2, 3], vec![1.0, -2.0, 3.5, 0.0, f32::MIN_POSITIVE, 7.0]));
        t.insert("a".into(), Tensor::new(vec![1], vec![0.25]));
        save(&t, &p).unwrap();
        assert_eq!(load(&p).unwrap(), t);
        std::fs::remove_dir_all(&dir).ok();
    }
}
