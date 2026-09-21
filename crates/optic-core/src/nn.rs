use anyhow::Result;
use candle_core::Tensor;

pub fn layer_norm(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    let mean = x.mean_keepdim(1)?;
    let d = x.broadcast_sub(&mean)?;
    let inv = ((&d * &d)?.mean_keepdim(1)? + 1e-5)?.sqrt()?.recip()?;
    Ok(d.broadcast_mul(&inv)?.broadcast_mul(w)?.broadcast_add(b)?)
}

pub fn silu(x: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::ops::silu(x)?)
}

pub fn head_forward(h: &Tensor, w0: &Tensor, b0: &Tensor, w1: &Tensor, b1: &Tensor, w4: &Tensor, b4: &Tensor) -> Result<Tensor> {
    let z = h.matmul(&w0.t()?)?.broadcast_add(b0)?;
    let z = silu(&layer_norm(&z, w1, b1)?)?;
    Ok(z.matmul(&w4.t()?)?.broadcast_add(b4)?)
}
