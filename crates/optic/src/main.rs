mod bam;
mod bamio;
mod bgzf;
mod warn;
mod cli;
mod data;
mod fed;
mod inspect;
mod probe;
mod sha256;

use anyhow::{Context, Result};
use optic_core::{model, net, report};
use cli::Cmd;

/// The probe coordinate map for a BAM: the assembly its header matches (by contig length), keyed by its own names.
fn bam_coord_map(path: &std::path::Path, model_dir: &std::path::Path) -> Result<std::collections::HashMap<(String, u64), String>> {
    let res = inspect::resolve_bam(path, model_dir)?;
    if res.assembly.name != inspect::T2T.name { eprintln!("[optic] {}: {} ({}), probe map {}", path.display(), res.assembly.name, res.naming, res.assembly.probe_file); }
    data::load_coord_map_renamed(&model_dir.join(res.assembly.probe_file), &res.contigs)
}

fn load_sample(kind: &str, path: &std::path::Path, model_dir: &std::path::Path, threads: usize, lo: f32, hi: f32) -> Result<Vec<(String, f32)>> {
    match kind {
        "bam" => bam::process_bam(path, &bam_coord_map(path, model_dir)?, lo, hi, threads).context("processing BAM"),
        "pileup" => {
            let (cgs, reference) = data::parse_pileup_any(path, model_dir)?;
            if reference != inspect::T2T.name { eprintln!("[optic] {}: pileup matched the {reference} probe map", path.display()); }
            Ok(cgs)
        }
        "beta" => data::parse_beta_csv(path),
        k => anyhow::bail!("unknown input kind {k:?} (bam|pileup|beta)"),
    }
}

fn main() -> Result<()> {
    let cli = cli::parse()?;
    if let Cmd::Inspect { input, records } = &cli.command {
        // without a model directory, check the BAM against every assembly optic knows
        let maps = inspect::available(&cli.model);
        let r = inspect::inspect_bam(input, *records, if maps.is_empty() { &inspect::ASSEMBLIES } else { &maps })?;
        if cli.json { println!("{}", serde_json::to_string_pretty(&r)?); } else { print!("{}", inspect::render(&r)); }
        if !r.fatal().is_empty() { std::process::exit(2); }
        return Ok(());
    }
    let model = model::Model::load(&cli.model)?;
    let bed = model.dir.join("probes.T2T.bed");
    let probes = probe::ProbeSubset::load(&bed)?;
    // pileup only: no weights needed
    if let Cmd::Counts { input, out, shard, threads, threshold_high, threshold_low, no_checks } = &cli.command {
        if !no_checks && shard.index == 0 { preflight(input, &model.dir)?; }
        let (counts, span) = bam::bam_counts(input, &bam_coord_map(input, &model.dir)?, *threshold_low, *threshold_high, *threads, *shard)
            .context("processing BAM")?;
        data::write_counts(out, &counts)?;
        // recovered input problems travel with the counts, so `classify` reports them
        let w = warn::take();
        let wpath = format!("{}.warnings", out.display());
        if w.is_empty() { let _ = std::fs::remove_file(&wpath); }
        else { std::fs::write(&wpath, w.iter().map(|(c, m)| format!("{c}\t{}\n", m.replace(['\t', '\n'], " "))).collect::<String>())?; }
        if let Some((first, stop)) = span {
            // byte-range shard of an unindexed BAM: `classify` checks that the parts chain
            std::fs::write(format!("{}.range", out.display()), format!("{first}\t{stop}\n"))?;
        }
        eprintln!("[optic] counts: {} probes covered (shard {}/{}) -> {}", counts.len(), shard.index, shard.n, out.display());
        return Ok(());
    }
    let net = net::Net::load(&model.dir.join("optic.safetensors"), probes.n_probes())?;

    let mut checks: Vec<inspect::Issue> = Vec::new();
    let (cgs, source) = match &cli.command {
        Cmd::Bam { input, threads, threshold_high, threshold_low, no_checks } => {
            if !no_checks { checks = preflight(input, &model.dir)?; }
            (load_sample("bam", input, &model.dir, *threads, *threshold_low, *threshold_high)?, format!("BAM {}", input.display()))
        }
        Cmd::Pileup { input } => (load_sample("pileup", input, &model.dir, 1, 0.0, 0.0)?, format!("pileup {}", input.display())),
        Cmd::Beta { input } => (load_sample("beta", input, &model.dir, 1, 0.0, 0.0)?, format!("beta CSV {}", input.display())),
        Cmd::Classify { input } => {
            let mut counts = bam::Counts::new();
            for p in input {
                data::add_counts(p, &mut counts)?;
                if let Ok(w) = std::fs::read_to_string(format!("{}.warnings", p.display())) {
                    for (c, m) in w.lines().filter_map(|l| l.split_once('\t')) { warn::warn(Box::leak(c.to_string().into_boxed_str()), m); }
                }
            }
            data::check_ranges(input)?;
            (bam::finalize_counts(counts)?, format!("counts {}", input.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(" + ")))
        }
        Cmd::Counts { .. } | Cmd::Inspect { .. } => unreachable!(),
        Cmd::Contribute { head, labels, out, threads } => {
            let head = head.clone().unwrap_or_else(|| model.head.clone());
            let head = &head;
            let names = model.class_names(head).ok_or_else(|| anyhow::anyhow!("model has no output named {head}"))?.to_vec();
            let mut feats = Vec::new(); let mut masks = Vec::new(); let mut counts: std::collections::HashMap<String, usize> = Default::default();
            for (i, line) in std::fs::read_to_string(labels)?.lines().enumerate() {
                if line.trim().is_empty() || line.starts_with('#') { continue; }
                let f: Vec<&str> = line.split('\t').collect();
                if f.len() < 3 { anyhow::bail!("labels line {}: expected path<TAB>kind<TAB>label", i + 1); }
                let cgs = load_sample(f[1], std::path::Path::new(f[0]), &model.dir, *threads, bam::DEFAULT_THRESHOLD_LOW, bam::DEFAULT_THRESHOLD_HIGH)?;
                let (tern, _) = probes.cgs_to_ternary(&cgs);
                feats.extend(net.first0_proj(&tern, 1)?);
                masks.push(fed::label_mask_pub(f[2], &names, &model, head)?);
                *counts.entry(f[2].to_string()).or_default() += 1;
            }
            if masks.is_empty() { anyhow::bail!("no samples in {}", labels.display()); }
            let (grads, loss) = fed::contribute(&net, head, &feats, &masks)?;
            let meta = fed::UpdateMeta { head: head.clone(), base_sha256: fed::sha256_file(&model.dir.join("optic.safetensors"))?,
                n_samples: masks.len(), label_counts: counts, class_names: names, loss,
                grad_norm: fed::grad_norm(&grads) };
            fed::write_update(out, &grads, &meta)?;
            eprintln!("contribute: {} samples, mean loss {:.4}, |grad| {:.3} -> {} (+ .json)", meta.n_samples, loss, meta.grad_norm, out.display());
            return Ok(());
        }
        Cmd::Apply { patch, out } => { return fed::apply(&model, patch, out); }
        Cmd::Merge { updates, lr, max_grad_norm, patch, out } => {
            return fed::merge(&model, updates, *lr, *max_grad_norm, out, patch.as_deref());
        }
    };
    // what the run recovered from replaces the precheck's note of the same problem (it has the full count)
    let rec = recovered();
    checks.retain(|c| !rec.iter().any(|r| r.code == c.code));
    checks.extend(rec);
    let (ternary, _) = probes.cgs_to_ternary(&cgs);
    let n_covered = ternary.iter().filter(|&&v| v != 0.0).count();
    if n_covered == 0 {
        anyhow::bail!("no probes covered by {source}: {}", match &cli.command {
            Cmd::Bam { .. } | Cmd::Classify { .. } => "no MM/ML 5mC calls landed on probe CpGs (run `optic inspect -i <bam>`)",
            Cmd::Pileup { .. } => "no pileup rows matched the reference probe coordinates",
            Cmd::Beta { .. } => "no probe IDs (cg########) in the Name column matched the model's probes",
            _ => "empty input",
        });
    }
    let (tier, tier_level, tier_message) = inspect::coverage_tier(n_covered);
    if tier_level == inspect::Level::Warn { eprintln!("[optic] warning: {tier_message}"); }
    let pred = net.infer(&ternary)?;
    let rep = report::build(&model, &pred);

    if cli.json {
        let names = model.class_names(&model.head).unwrap_or(&[]);
        let probs = pred.probs.iter().find(|(h, _)| *h == model.head).map(|(_, p)| p.as_slice()).unwrap_or(&[]);
        let temperature = pred.temperature.iter().find(|(h, _)| *h == model.head).map(|x| x.1).unwrap_or(1.0);
        let out = serde_json::json!({
            "source": source, "model": {"name": model.info.name, "version": model.info.version},
            "n_covered": pred.n_covered, "n_probes_total": probes.n_probes(),
            "calibration_temperature": temperature,
            "report": rep,
            "qc": {"coverage_tier": tier, "level": tier_level, "message": tier_message, "input_checks": checks},
            "classes": names.iter().zip(probs).map(|(n, &q)| serde_json::json!({"class": n, "probability": q})).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        print!("{}", report::render(&rep, &pred, &model, &source));
        println!("  coverage : {tier} - {tier_message}");
        for c in checks.iter().filter(|c| c.level == inspect::Level::Warn) { println!("  warning  : {} - {}", c.code, c.message); }
    }
    Ok(())
}

/// Input problems the run recovered from (see `warn`), as report warnings.
fn recovered() -> Vec<inspect::Issue> {
    warn::take().into_iter().map(|(code, message)| inspect::Issue {
        level: inspect::Level::Warn, code, message,
        hint: match code {
            c if c.starts_with("BAM_") => "the BAM is damaged: the affected reads were skipped and the result uses the rest; re-copy or re-create the file (samtools quickcheck -v in.bam)",
            c if c.starts_with("INDEX_") => "rebuild the index: samtools index in.bam",
            _ => "the affected rows were skipped; check the file",
        }.into(),
    }).collect()
}

/// bam pre-check
fn preflight(input: &std::path::Path, model_dir: &std::path::Path) -> Result<Vec<inspect::Issue>> {
    let r = inspect::inspect_bam(input, 2000, &inspect::available(model_dir))?;
    let fatal = r.fatal();
    if !fatal.is_empty() {
        let msg: Vec<String> = fatal.iter().map(|i| format!("{}: {}\n    -> {}", i.code, i.message, i.hint)).collect();
        anyhow::bail!("{} is not usable:\n  {}\n(skip with --no-checks; details: optic inspect -i <bam>)", input.display(), msg.join("\n  "));
    }
    for i in r.issues.iter().filter(|i| i.level == inspect::Level::Warn) { eprintln!("[optic] warning: {} - {} ({})", i.code, i.message, i.hint); }
    Ok(r.issues.into_iter().filter(|i| i.level != inspect::Level::Info).collect())
}
