mod bam;
mod data;
mod fed;
mod probe;

use anyhow::{Context, Result};
use optic_core::{model, net, report};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "optic", version, about = "Open Pediatric Tissue Classifier (heme / solid / CNS tumors)")]
struct Cli {
    /// Model directory (lineage.json, optic.safetensors, probe_subset.bed)
    #[arg(long, default_value = "model")]
    model: PathBuf,
    /// Emit JSON (full per-head probabilities + report) instead of text
    #[arg(long)]
    json: bool,
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// BAM/CRAM with MM/ML modification tags (T2T-CHM13v2.0 alignment)
    Bam {
        #[arg(short, long)] input: PathBuf,
        #[arg(short, long, default_value_t = 8)] threads: usize,
        /// Per-base mod probability >= high -> modified, < low -> unmodified
        #[arg(long, default_value_t = bam::DEFAULT_THRESHOLD_HIGH)] threshold_high: f32,
        #[arg(long, default_value_t = bam::DEFAULT_THRESHOLD_LOW)] threshold_low: f32,
    },
    /// modkit pileup bed (`modkit pileup --include-bed probe_subset.bed --cpg --combine-strands`)
    Pileup { #[arg(short, long)] input: PathBuf },
    /// Infinium-style CSV with `Name` and `Beta` columns
    Beta { #[arg(short, long)] input: PathBuf },
    /// Apply a hub weight patch to the local model, producing an updated model directory.
    Apply {
        #[arg(long)] patch: PathBuf,
        #[arg(long)] out: PathBuf,
    },
    /// Federated update: one backward pass over a labeled cohort.
    /// LABELS is a TSV: path <TAB> kind(bam|pileup|beta) <TAB> label
    Contribute {
        /// Only needed for a model that declares more than one output; defaults to the model's own.
        #[arg(long)] head: Option<String>,
        #[arg(long)] labels: PathBuf,
        #[arg(long, default_value = "update.safetensors")] out: PathBuf,
        #[arg(short, long, default_value_t = 8)] threads: usize,
    },
    Merge {
        #[arg(long, required = true, num_args = 1..)] updates: Vec<PathBuf>,
        #[arg(long, default_value_t = 0.01)] lr: f32,
        /// Reject an update whose per-sample gradient norm exceeds this (0 = no cap)
        #[arg(long, default_value_t = 0.0)] max_grad_norm: f32,
        /// Also write just the tensors that moved, for distribution to sites (~12 MB vs 373 MB)
        #[arg(long)] patch: Option<PathBuf>,
        #[arg(long)] out: PathBuf,
    },
}

fn load_sample(kind: &str, path: &std::path::Path, bed: &std::path::Path, threads: usize, lo: f32, hi: f32) -> Result<Vec<(String, f32)>> {
    match kind {
        "bam" => bam::process_bam(path, &data::load_coord_map(bed)?, lo, hi, threads).context("processing BAM"),
        "pileup" => data::parse_pileup(path, bed),
        "beta" => data::parse_beta_csv(path),
        k => anyhow::bail!("unknown input kind {k:?} (bam|pileup|beta)"),
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let device = candle_core::Device::Cpu;
    let model = model::Model::load(&cli.model)?;
    let bed = model.dir.join("probe_subset.bed");
    let probes = probe::ProbeSubset::load(&bed)?;
    let net = net::Net::load(&model.dir.join("optic.safetensors"), probes.n_probes(), &device)?;

    let (cgs, source) = match &cli.command {
        Cmd::Bam { input, threads, threshold_high, threshold_low } =>
            (load_sample("bam", input, &bed, *threads, *threshold_low, *threshold_high)?, format!("BAM {}", input.display())),
        Cmd::Pileup { input } => (load_sample("pileup", input, &bed, 1, 0.0, 0.0)?, format!("pileup {}", input.display())),
        Cmd::Beta { input } => (load_sample("beta", input, &bed, 1, 0.0, 0.0)?, format!("beta CSV {}", input.display())),
        Cmd::Contribute { head, labels, out, threads } => {
            let head = head.clone().unwrap_or_else(|| model.head.clone());
            let head = &head;
            let names = model.class_names(head).ok_or_else(|| anyhow::anyhow!("model has no output named {head}"))?.to_vec();
            let mut feats = Vec::new(); let mut masks = Vec::new(); let mut counts: std::collections::HashMap<String, usize> = Default::default();
            for (i, line) in std::fs::read_to_string(labels)?.lines().enumerate() {
                if line.trim().is_empty() || line.starts_with('#') { continue; }
                let f: Vec<&str> = line.split('\t').collect();
                if f.len() < 3 { anyhow::bail!("labels line {}: expected path<TAB>kind<TAB>label", i + 1); }
                let cgs = load_sample(f[1], std::path::Path::new(f[0]), &bed, *threads, bam::DEFAULT_THRESHOLD_LOW, bam::DEFAULT_THRESHOLD_HIGH)?;
                let (tern, _) = probes.cgs_to_ternary(&cgs);
                let x = candle_core::Tensor::from_vec(tern, (1, probes.n_probes()), &device)?;
                feats.push(net.first0_proj(&x)?.detach());
                masks.push(fed::label_mask_pub(f[2], &names, &model, head)?);
                *counts.entry(f[2].to_string()).or_default() += 1;
            }
            if feats.is_empty() { anyhow::bail!("no samples in {}", labels.display()); }
            let proj = candle_core::Tensor::cat(&feats, 0)?;
            let (grads, loss) = fed::contribute(&net, head, &proj, &masks, &device)?;
            let meta = fed::UpdateMeta { head: head.clone(), base_sha256: fed::sha256_file(&model.dir.join("optic.safetensors"))?,
                n_samples: masks.len(), label_counts: counts, class_names: names, loss,
                grad_norm: fed::grad_norm(&grads)? };
            fed::write_update(out, &grads, &meta)?;
            eprintln!("contribute: {} samples, mean loss {:.4}, |grad| {:.3} -> {} (+ .json)", meta.n_samples, loss, meta.grad_norm, out.display());
            return Ok(());
        }
        Cmd::Apply { patch, out } => { return fed::apply(&model, patch, out, &device); }
        Cmd::Merge { updates, lr, max_grad_norm, patch, out } => {
            return fed::merge(&model, updates, *lr, *max_grad_norm, out, patch.as_deref(), &device);
        }
    };
    let (ternary, _) = probes.cgs_to_ternary(&cgs);
    let pred = net.infer(&ternary, &device)?;
    let rep = report::build(&model, &pred);

    if cli.json {
        // One head, so the output names classes directly rather than nesting them under a head.
        let names = model.class_names(&model.head).unwrap_or(&[]);
        let probs = pred.probs.iter().find(|(h, _)| *h == model.head).map(|(_, p)| p.as_slice()).unwrap_or(&[]);
        let temperature = pred.temperature.iter().find(|(h, _)| *h == model.head).map(|x| x.1).unwrap_or(1.0);
        let out = serde_json::json!({
            "source": source, "model": {"name": model.info.name, "version": model.info.version},
            "n_covered": pred.n_covered, "n_probes_total": probes.n_probes(),
            "calibration_temperature": temperature,
            "report": rep,
            "classes": names.iter().zip(probs).map(|(n, &q)| serde_json::json!({"class": n, "probability": q})).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        print!("{}", report::render(&rep, &pred, &model, &source));
    }
    Ok(())
}
