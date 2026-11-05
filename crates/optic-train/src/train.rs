//! Training loop: every training row (weight > 0) x every coverage level, shuffled per epoch, batches of `batch_size` (drop last)
//! A producer thread gathers uint8 rows from the in-RAM corpus while the GPU works on the previous batch.
//! AdamW, per-epoch cosine LR (eta_min 0). With `consistency`, each batch is also built as a shallow view and its
//! prediction pulled toward the deep view's detached prediction on training rows.
//! The GPU side is gpu.rs (device, kernels) and model.rs (network forward/backward); every random draw is seeded from `optim.seed`, so a run is
//! reproducible.

use anyhow::{Context, Result};
use crate::gpu::{Buf, Gpu};
use crate::model::{Cfg, Net};
use std::path::Path;
use std::sync::mpsc;
use std::time::Instant;
use crate::aug::Aug;
use crate::spec::{Spec, Tables};

struct Batch { rows: Vec<u32>, cov: Vec<f32>, codes: Vec<u8> }

pub fn run(spec_dir: &Path, out_dir: &Path, epochs_override: Option<usize>, max_batches: Option<usize>) -> Result<()> {
    let gpu = Gpu::new(0)?;
    let spec = Spec::load(spec_dir)?;
    println!("optic-train: {} ({})", spec.corpus, spec_dir.display());
    let mut tab = spec.load_tables()?;
    let (n, k, p) = (tab.n, tab.k, tab.p);
    let epochs = epochs_override.unwrap_or(spec.optim.epochs);
    let bs = spec.optim.batch_size;
    std::fs::create_dir_all(out_dir)?;
    if let Some(c) = &spec.consistency { println!("  cross-depth consistency: weight {}, shallow view {:.0}-{:.0} covered probes", c.weight, c.lo, c.hi); }

    let cfg = Cfg { probes: p, rank: spec.model.rank, hidden: spec.model.hidden, depth: spec.model.depth, head: spec.head.clone(), k };
    let mut net = Net::from_tensors(&gpu, cfg.clone(), &crate::init::init(&cfg, spec.optim.seed))?;
    let n_params: usize = net.p.values().map(|(_, b)| b.len()).sum();
    println!("  params: {} ({:.1} MB)", n_params, n_params as f64 * 4.0 / 1e6);

    let corpus = load_corpus(&spec.beta_u8, &tab)?;
    attach_purity(&spec, &mut tab, &corpus)?;
    let tab = tab;
    let corpus = std::sync::Arc::new(corpus);
    let aug = Aug::new(&spec, &tab, &gpu)?;
    let hier: Vec<f32> = tab.hier.iter().flat_map(|(m, _)| m.iter().copied()).collect();
    let n_nodes: usize = tab.hier.iter().map(|(_, nn)| nn).sum();
    let hier_d = gpu.upload(&hier)?;
    // consistency coverage and dropout seeds: their own streams
    let mut cons_rng = crate::rng::StdRng::seed_from_u64(spec.optim.seed * 6_700_417 + 5);
    let mut mask_seed: u64 = spec.optim.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD15C;
    let mut log: Vec<serde_json::Value> = Vec::new();
    let t_max = if spec.optim.sched_tmax > 0 { spec.optim.sched_tmax } else { epochs };
    let train_rows: Vec<u32> = tab.roww.iter().enumerate().filter(|x| *x.1 > 0.0).map(|x| x.0 as u32).collect();
    println!("  batches draw from {} training rows x {} coverage levels ({} zero-weight rows left out)", train_rows.len(), spec.coverage_levels.len(), n - train_rows.len());
    let rows_x = if spec.consistency.is_some() { 2 * bs } else { bs };
    let mask_dims: Vec<usize> = std::iter::repeat(cfg.hidden).take(cfg.depth + 1).chain([cfg.hidden / 2]).collect();
    let save = |net: &Net, name: &str| -> Result<()> { optic_core::st::save(&net.to_tensors(&gpu)?, &out_dir.join(name)) };

    for ep in 0..epochs {
        let lr = spec.optim.lr * 0.5 * (1.0 + (std::f64::consts::PI * ep as f64 / t_max as f64).cos());
        let mut rng = crate::rng::StdRng::seed_from_u64(spec.optim.seed * 1_000_003 + ep as u64);
        let cov_levels = spec.coverage_levels.clone();
        let mut pairs: Vec<(u32, u32)> = train_rows.iter().flat_map(|&r| cov_levels.iter().map(move |&c| (r, c))).collect();
        rng.shuffle(&mut pairs);
        let n_batches = max_batches.map(|m| m.min(pairs.len() / bs)).unwrap_or(pairs.len() / bs);
        let (tx, rx) = mpsc::sync_channel::<Batch>(3);
        let (cp, pairs) = (corpus.clone(), std::sync::Arc::new(pairs));
        let prod = std::thread::spawn(move || {
            for bi in 0..n_batches {
                let sl = &pairs[bi * bs..(bi + 1) * bs];
                let mut codes = vec![0u8; bs * p];
                for (i, (r, _)) in sl.iter().enumerate() { codes[i * p..(i + 1) * p].copy_from_slice(&cp[(*r as usize) * p..(*r as usize + 1) * p]); }
                if tx.send(Batch { rows: sl.iter().map(|x| x.0).collect(), cov: sl.iter().map(|x| x.1 as f32).collect(), codes }).is_err() { break; }
            }
        });
        let t0 = Instant::now(); let mut nb = 0usize;
        let mut acc = gpu.zeros::<f32>(2)?;                    // summed (cls, cons) batch losses, read back for logging
        let (mut dz, mut cls_row, mut cons_row) = (gpu.zeros::<f32>(rows_x * k)?, gpu.zeros::<f32>(bs)?, gpu.zeros::<f32>(bs)?);
        let mut masks: Vec<Buf> = mask_dims.iter().map(|&d| gpu.zeros::<f32>(rows_x * d)).collect::<Result<_>>()?;
        while let Ok(b) = rx.recv() {
            let codes = gpu.upload(&b.codes)?;
            let t = gpu.upload(&aug.targets(&b.rows))?;
            let w: Vec<f32> = b.rows.iter().map(|&r| tab.roww[r as usize]).collect();
            let lab: Vec<f32> = w.iter().map(|&x| (x > 0.0) as u8 as f32).collect();
            let (w, lab) = (gpu.upload(&w)?, gpu.upload(&lab)?);
            let x = aug.build(&codes, &b.cov, &b.rows)?;
            let x = match &spec.consistency {
                Some(c) => {
                    let cs: Vec<f32> = b.cov.iter().map(|&cv| ((c.lo.ln() + (c.hi.ln() - c.lo.ln()) * cons_rng.f64()).exp() as f32).min(cv)).collect();
                    let xs = aug.build(&codes, &cs, &b.rows)?;
                    let mut both = gpu.zeros::<f32>(2 * bs * p)?;
                    gpu.stream.memcpy_dtod(&x, &mut both.slice_mut(0..bs * p))?;
                    gpu.stream.memcpy_dtod(&xs, &mut both.slice_mut(bs * p..2 * bs * p))?;
                    both
                }
                None => x,
            };
            for (m, &d) in masks.iter_mut().zip(&mask_dims) { mask_seed = mask_seed.wrapping_add(1); gpu.dropout_mask(m, rows_x * d, spec.model.dropout, mask_seed)?; }
            let (z, cache) = net.forward(&gpu, &x, rows_x, Some(&masks))?;
            gpu.loss(&mut dz, &mut cls_row, &mut cons_row, &z, &t, &w, &lab, &hier_d, n_nodes, bs, k,
                     spec.kd_temperature as f32, spec.hier_level_weight as f32, spec.consistency.as_ref().map(|c| c.weight as f32))?;
            gpu.loss_sum(&mut acc, &cls_row, &cons_row, bs)?;
            net.backward(&gpu, &x, &cache, &dz, Some(&masks))?;
            net.adamw(&gpu, lr, spec.optim.betas[0], spec.optim.betas[1], spec.optim.eps, spec.optim.wd)?;
            nb += 1;
            if nb % 200 == 0 {
                let a = gpu.download(&acc)?;
                println!("    ep {ep} batch {nb}/{n_batches}  loss avg {:.4}  {:.1}s", (a[0] + a[1]) as f64 / nb as f64, t0.elapsed().as_secs_f32());
            }
        }
        prod.join().ok();
        gpu.sync()?;
        let wall = t0.elapsed().as_secs_f32();
        let a = gpu.download(&acc)?;
        let (cls, cons) = (a[0] as f64 / nb.max(1) as f64, a[1] as f64 / nb.max(1) as f64);
        let tot = cls + cons;
        println!("   ep {ep:>2}  tot={tot:.4}  {}={cls:.4}{}  lr={lr:.2e}  t={wall:.1}s", spec.head, if spec.consistency.is_some() { format!("  cons={cons:.4}") } else { String::new() });
        let mut rec = serde_json::json!({"epoch": ep, "total": tot, "consistency": cons, "lr": lr, "wall_s": wall});
        rec[&spec.head] = serde_json::json!(cls);
        log.push(rec);
        save(&net, "optic.safetensors")?;
        std::fs::write(out_dir.join("log.json"), serde_json::to_string_pretty(&log)?)?;
        if (ep + 1) % 10 == 0 { save(&net, &format!("optic_ep{}.safetensors", ep + 1))?; }
    }
    let cfg = serde_json::json!({"arch": format!("optic {}", env!("CARGO_PKG_VERSION")), "trainer": "optic-train (cuda-oxide)", "corpus": spec.corpus,
        "probes_in": p, "rank": spec.model.rank, "hidden": spec.model.hidden, "depth": spec.model.depth, "dropout": spec.model.dropout,
        "n_classes": {spec.head.clone(): k}, "n_params": n_params, "epochs": epochs, "n_rows": n,
        "row_weights": tab.weights_note, "purity_mix": match tab.purity.len() { 0 => serde_json::Value::Null, 1 => tab.purity[0].summary.clone(), _ => serde_json::json!(tab.purity.iter().map(|pu| pu.summary.clone()).collect::<Vec<_>>()) }, "spec": spec});
    std::fs::write(out_dir.join("config.json"), serde_json::to_string_pretty(&cfg)?)?;
    println!("  written to {}", out_dir.display());
    Ok(())
}

/// Corpus uint8 codes in RAM (parallel copy of the file).
pub fn load_corpus(path: &Path, tab: &Tables) -> Result<Vec<u8>> {
    let (n, p) = (tab.n, tab.p);
    let t_load = Instant::now();
    let file = std::fs::File::open(&path).with_context(|| format!("opening {}", path.display()))?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    anyhow::ensure!(mmap.len() == n * p, "{}: {} bytes, expected {} rows x {} probes", path.display(), mmap.len(), n, p);
    let mut corpus = vec![0u8; n * p];
    {
        use rayon::prelude::*;
        corpus.par_chunks_mut(p).zip(mmap.par_chunks(p)).for_each(|(dst, src)| dst.copy_from_slice(src));
    }
    let bad = { use rayon::prelude::*; corpus.par_chunks(p).filter(|row| row.iter().any(|&c| c > 250 && c < 255)).count() };
    anyhow::ensure!(bad == 0, "{bad} rows with codes 251..254; beta codes must be 0..250 or 255 (missing)");
    println!("  corpus in RAM ({:.1} GB) in {:.0}s", (n * p) as f64 / 1e9, t_load.elapsed().as_secs_f32());
    Ok(corpus)
}

/// Build the purity-mixing bank and eligibility (purity.rs) from the targets, the tree and the corpus.
pub fn attach_purity(spec: &Spec, tab: &mut Tables, corpus: &[u8]) -> Result<()> {
    let blocks: Vec<&crate::purity::PurityMixCfg> = spec.purity_mix.iter().chain(&spec.purity_mix_extra).collect();
    if blocks.is_empty() { return Ok(()); }
    let tree = crate::labels::Tree::load(&spec.tree)?;
    let (lut, mask) = (tab.lut.clone(), tab.roww.clone());
    let mut off = 0u32;
    for pc in blocks {
        let pu = crate::purity::build(pc, &tree, &lut, tab.k, &mask, corpus, tab.p)?;
        tab.bank_offset.push(off); off += pu.n_bank as u32;
        tab.purity.push(pu);
    }
    tab.bank = Some(tab.purity.iter().flat_map(|pu| pu.bank.iter().copied()).collect());
    Ok(())
}

/// CPU-only dry run: build every table the run would use (targets, tree levels, row weights, read-depth tables, purity
/// bank, eligibility) and report them. `out`: JSON with row weights, bank rows and eligible rows. `dump`: the raw tables.
pub fn check(spec_dir: &Path, out: Option<&Path>, dump: Option<&Path>) -> Result<()> {
    let spec = Spec::load(spec_dir)?;
    let mut tab = spec.load_tables()?;
    if let Some(d) = dump {
        std::fs::create_dir_all(d)?;
        let wr = |name: &str, v: &[f32]| -> Result<()> { std::fs::write(d.join(name), v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?; Ok(()) };
        wr(&format!("lut_{}.f32", spec.head), &tab.lut)?;
        wr(&format!("roww_{}.f32", spec.head), &tab.roww)?;
        for (i, (m, _)) in tab.hier.iter().enumerate() { wr(&format!("hier_{}_L{}.f32", spec.head, i + 1), m)?; }
        wr("depth_lut.f32", &tab.depth_lut)?;
        println!("  tables written to {}", d.display());
    }
    let corpus = load_corpus(&spec.beta_u8, &tab)?;
    attach_purity(&spec, &mut tab, &corpus)?;
    if let Some(o) = out {
        let el: Vec<usize> = (0..tab.n).filter(|&r| tab.purity.iter().any(|pu| pu.eligible[r])).collect();
        std::fs::write(o, serde_json::to_string(&serde_json::json!({"row_weights": tab.roww, "purity": tab.purity.iter().map(|pu| pu.summary.clone()).collect::<Vec<_>>(), "eligible_rows": el}))?)?;
        println!("  wrote {}", o.display());
    }
    Ok(())
}
