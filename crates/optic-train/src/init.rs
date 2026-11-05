//! Starting weights: every weight matrix Kaiming normal for ReLU (N(0, 2 / fan_in)), every linear bias uniform on ±1/sqrt(fan_in), LayerNorm weights 1 and biases 0.
//! Drawn on the host from `seed`, so a run's starting point depends only on the spec.

use optic_core::st::{Tensor, Tensors};
use crate::model::{Cfg, Net};

pub fn init(cfg: &Cfg, seed: u64) -> Tensors {
    let mut rng = crate::rng::StdRng::seed_from_u64(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ 0x1417);
    let normal = |rng: &mut crate::rng::StdRng| -> f32 {
        // Box-Muller
        let (u1, u2): (f64, f64) = (rng.f64().max(f64::MIN_POSITIVE), rng.f64());
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    };
    let shapes: Vec<(String, Vec<usize>)> = {
        let mut v = vec![("first.0.weight".to_string(), vec![cfg.rank, cfg.probes]), ("first.1.weight".into(), vec![cfg.hidden, cfg.rank]),
                         ("first.1.bias".into(), vec![cfg.hidden]), ("ln1.weight".into(), vec![cfg.hidden]), ("ln1.bias".into(), vec![cfg.hidden])];
        for i in 0..cfg.depth {
            v.push((format!("trunk.{i}.lin.weight"), vec![cfg.hidden, cfg.hidden])); v.push((format!("trunk.{i}.lin.bias"), vec![cfg.hidden]));
            v.push((format!("trunk.{i}.ln.weight"), vec![cfg.hidden])); v.push((format!("trunk.{i}.ln.bias"), vec![cfg.hidden]));
        }
        let h = format!("heads.{}", cfg.head);
        v.push((format!("{h}.0.weight"), vec![cfg.hidden / 2, cfg.hidden])); v.push((format!("{h}.0.bias"), vec![cfg.hidden / 2]));
        v.push((format!("{h}.1.weight"), vec![cfg.hidden / 2])); v.push((format!("{h}.1.bias"), vec![cfg.hidden / 2]));
        v.push((format!("{h}.4.weight"), vec![cfg.k, cfg.hidden / 2])); v.push((format!("{h}.4.bias"), vec![cfg.k]));
        v
    };
    debug_assert_eq!(shapes.len(), Net::names(cfg).len());
    let mut out = Tensors::new();
    // linear weight name -> its fan-in, for the bias bound
    let fan_in: std::collections::HashMap<String, usize> = shapes.iter().filter(|(_, s)| s.len() == 2).map(|(n, s)| (n.trim_end_matches(".weight").to_string(), s[1])).collect();
    for (name, shape) in shapes {
        let n: usize = shape.iter().product();
        let base = name.rsplit_once('.').unwrap().0.to_string();
        let data: Vec<f32> = if shape.len() == 2 {
            let std = (2.0 / shape[1] as f64).sqrt() as f32;
            (0..n).map(|_| normal(&mut rng) * std).collect()
        } else if let Some(&fi) = fan_in.get(&base) {
            let b = 1.0 / (fi as f64).sqrt();
            (0..n).map(|_| (rng.f64() * 2.0 * b - b) as f32).collect() // linear bias
        } else if name.ends_with(".weight") { vec![1.0; n] } // LayerNorm gain
        else { vec![0.0; n] }; // LayerNorm bias
        out.insert(name, Tensor::new(shape, data));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn init_follows_the_rules() {
        let cfg = Cfg { probes: 4000, rank: 64, hidden: 128, depth: 2, head: "h".into(), k: 10 };
        let w = init(&cfg, 1);
        assert_eq!(w.len(), Net::names(&cfg).len());
        let sd = |v: &[f32]| { let m = v.iter().sum::<f32>() / v.len() as f32; (v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / v.len() as f32).sqrt() };
        assert!((sd(&w["first.0.weight"].data) / (2.0f32 / 4000.0).sqrt() - 1.0).abs() < 0.02);
        assert!(w["first.1.bias"].data.iter().all(|&b| b.abs() <= 1.0 / 8.0));       // fan-in 64
        assert!(w["ln1.weight"].data.iter().all(|&g| g == 1.0) && w["trunk.0.ln.bias"].data.iter().all(|&b| b == 0.0));
        assert_eq!(init(&cfg, 1)["heads.h.4.weight"], w["heads.h.4.weight"]);       // deterministic
    }
}
