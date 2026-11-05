//! Command-line parsing. Global options (`--model`, `--json`) may come anywhere; each subcommand takes
//! `--flag value`, `--flag=value` or `-f value`, and flags marked "1 or more" take every value up to the next flag.

use crate::bam;
use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;
use std::path::PathBuf;

pub struct Cli {
    pub model: PathBuf,
    pub json: bool,
    pub command: Cmd,
}

pub enum Cmd {
    Bam { input: PathBuf, threads: usize, no_checks: bool, threshold_high: f32, threshold_low: f32 },
    Pileup { input: PathBuf },
    Beta { input: PathBuf },
    Counts { input: PathBuf, out: PathBuf, shard: bam::Shard, threads: usize, no_checks: bool, threshold_high: f32, threshold_low: f32 },
    Inspect { input: PathBuf, records: u64 },
    Classify { input: Vec<PathBuf> },
    Apply { patch: PathBuf, out: PathBuf },
    Contribute { head: Option<String>, labels: PathBuf, out: PathBuf, threads: usize },
    Merge { updates: Vec<PathBuf>, lr: f32, max_grad_norm: f32, patch: Option<PathBuf>, out: PathBuf },
}

/// (long name, short, takes a value, 1 or more values, help)
type Flag = (&'static str, Option<char>, bool, bool, &'static str);

const THRESH: [Flag; 2] = [
    ("threshold-high", None, true, false, "Per-base mod probability >= this -> modified [default: 0.7]"),
    ("threshold-low", None, true, false, "Per-base mod probability < this -> unmodified [default: 0.3]"),
];

struct Sub { name: &'static str, about: &'static str, flags: Vec<Flag> }

fn subcommands() -> Vec<Sub> {
    let input: Flag = ("input", Some('i'), true, false, "Input file (required)");
    let threads: Flag = ("threads", Some('t'), true, false, "Worker threads [default: 8]");
    vec![
        Sub { name: "bam", about: "BAM/CRAM with MM/ML modification tags",
              flags: [input, threads, ("no-checks", None, false, false, "Skip the input preflight (reference, contig names, index, MM/ML tags); see `optic inspect`")]
                  .into_iter().chain(THRESH).collect() },
        Sub { name: "pileup", about: "pileup bed file (ex. from `modkit pileup --include-bed probes.T2T.bed --cpg --combine-strands`)", flags: vec![input] },
        Sub { name: "beta", about: "Infinium-style CSV with `Name` and `Beta` columns", flags: vec![input] },
        Sub { name: "counts", about: "Pile up a BAM (or one shard of it) into per-probe counts: probe_id, n_mod, n_unmod, fraction",
              flags: [input, ("out", Some('o'), true, false, "Output TSV (required)"), ("shard", None, true, false, "Shard i/n [default: 0/1]"), threads,
                      ("no-checks", None, false, false, "Skip the input preflight (shard 0 runs it otherwise)")].into_iter().chain(THRESH).collect() },
        Sub { name: "inspect", about: "check BAM without running it: reference (T2T-CHM13v2.0, GRCh38), contig naming (NC_0609xx.1 or chr1..), \
index, and MM/ML base-modification tags in the first reads",
              flags: vec![input, ("records", None, true, false, "Primary mapped reads to sample for the tag checks [default: 2000]")] },
        Sub { name: "classify", about: "Classify from per-probe count files (`optic counts` output); several files are summed (shards of one BAM).",
              flags: vec![("input", Some('i'), true, true, "Count files, 1 or more (required)")] },
        Sub { name: "apply", about: "Apply a hub weight patch to the local model, producing an updated model directory.",
              flags: vec![("patch", None, true, false, "Patch .safetensors (required)"), ("out", None, true, false, "Output model directory (required)")] },
        Sub { name: "contribute", about: "Federated update: one backward pass over a labeled cohort. LABELS is a TSV: path <TAB> kind(bam|pileup|beta) <TAB> label",
              flags: vec![("head", None, true, false, "Only needed for a model that declares more than one output; defaults to the model's own"),
                          ("labels", None, true, false, "Cohort TSV (required)"),
                          ("out", None, true, false, "Update .safetensors [default: update.safetensors]"), threads] },
        Sub { name: "merge", about: "Merge multiple federated gradient updates",
              flags: vec![("updates", None, true, true, "Update .safetensors files, 1 or more (required)"), ("lr", None, true, false, "Learning rate [default: 0.01]"),
                          ("max-grad-norm", None, true, false, "Reject an update whose per-sample gradient norm exceeds this (0 = no cap) [default: 0]"),
                          ("patch", None, true, false, "Also write just the tensors that moved, for distribution to sites (~12 MB vs 373 MB)"),
                          ("out", None, true, false, "Output model directory (required)")] },
    ]
}

fn usage() -> String {
    let mut s = format!("optic {} - Open Pediatric Tissue Classifier\n\n\
Usage: optic [--model DIR] [--json] <COMMAND> [OPTIONS]\n\nCommands:\n", env!("CARGO_PKG_VERSION"));
    for c in subcommands() {
        // the first sentence: up to a ". " that starts a capitalised one ("ex. from" does not end it)
        let cut = c.about.match_indices(". ").find(|(i, _)| c.about[i + 2..].starts_with(|ch: char| ch.is_ascii_uppercase())).map_or(c.about.len(), |(i, _)| i);
        s += &format!("  {:<11} {}\n", c.name, c.about[..cut].trim_end_matches('.'));
    }
    s + "\nOptions:\n  --model DIR  Model directory (lineage.json, optic.safetensors, probes.T2T.bed) [default: model]\n  --json       Emit JSON (full probabilities + report) instead of text\n  -h, --help   Help (also `optic <COMMAND> --help`)\n  -V, --version\n"
}

fn sub_usage(c: &Sub) -> String {
    let mut s = format!("{}\n\nUsage: optic [--model DIR] [--json] {} [OPTIONS]\n\nOptions:\n", c.about, c.name);
    for (long, short, val, many, help) in &c.flags {
        let name = format!("{}--{long}{}", short.map_or("    ".to_string(), |c| format!("-{c}, ")),
                           if *many { " <VALUE>..." } else if *val { " <VALUE>" } else { "" });
        s += &format!("  {name:<32} {help}\n");
    }
    s
}

pub fn parse() -> Result<Cli> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match parse_from(&args) {
        Ok(c) => Ok(c),
        Err(e) => {
            let msg = e.to_string();
            if let Some(help) = msg.strip_prefix("HELP\n") { print!("{help}"); std::process::exit(0); }
            eprintln!("error: {msg}\n\nFor more information, try '--help'.");
            std::process::exit(2);
        }
    }
}

fn looks_like_flag(s: &str) -> bool { s.starts_with('-') && s.len() > 1 && s.parse::<f64>().is_err() }

pub fn parse_from(args: &[String]) -> Result<Cli> {
    let (mut model, mut json, mut sub, mut rest) = (PathBuf::from("model"), false, None::<String>, Vec::<String>::new());
    let mut it = args.iter().peekable();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" => model = it.next().ok_or_else(|| anyhow!("--model needs a directory"))?.into(),
            s if s.starts_with("--model=") => model = s["--model=".len()..].into(),
            "--json" => json = true,
            "-V" | "--version" => { println!("optic {}", env!("CARGO_PKG_VERSION")); std::process::exit(0); }
            "-h" | "--help" if sub.is_none() => bail!("HELP\n{}", usage()),
            "help" if sub.is_none() => match it.next() {
                Some(n) => match subcommands().into_iter().find(|c| c.name == n) {
                    Some(c) => bail!("HELP\n{}", sub_usage(&c)),
                    None => bail!("unrecognized subcommand '{n}'"),
                },
                None => bail!("HELP\n{}", usage()),
            },
            s if sub.is_none() && !s.starts_with('-') => sub = Some(s.to_string()),
            _ => rest.push(a.clone()),
        }
    }
    let name = sub.ok_or_else(|| anyhow!("a command is required\n\n{}", usage()))?;
    let spec = subcommands().into_iter().find(|c| c.name == name).ok_or_else(|| anyhow!("unrecognized subcommand '{name}'"))?;
    if rest.iter().any(|a| a == "-h" || a == "--help") { bail!("HELP\n{}", sub_usage(&spec)); }

    let mut got: HashMap<&'static str, Vec<String>> = HashMap::new();
    let mut it = rest.iter().peekable();
    while let Some(a) = it.next() {
        let (key, inline) = match a.strip_prefix("--") {
            Some(l) => match l.split_once('=') { Some((k, v)) => (k.to_string(), Some(v.to_string())), None => (l.to_string(), None) },
            None if looks_like_flag(a) && a.len() == 2 => {
                let c = a.chars().nth(1).unwrap();
                let f = spec.flags.iter().find(|f| f.1 == Some(c)).ok_or_else(|| anyhow!("unexpected argument '{a}' for '{name}'"))?;
                (f.0.to_string(), None)
            }
            _ => bail!("unexpected argument '{a}' for '{name}'"),
        };
        let key = key.replace('_', "-");
        let &(long, _, val, many, _) = spec.flags.iter().find(|f| f.0 == key).ok_or_else(|| anyhow!("unexpected argument '--{key}' for '{name}'"))?;
        let slot = got.entry(long).or_default();
        if !val {
            if inline.is_some() { bail!("--{long} takes no value"); }
            slot.push(String::new());
        } else if let Some(v) = inline {
            slot.push(v);
        } else {
            let first = it.next().filter(|v| !looks_like_flag(v)).ok_or_else(|| anyhow!("--{long} needs a value"))?;
            slot.push(first.clone());
            while many && it.peek().is_some_and(|v| !looks_like_flag(v)) { slot.push(it.next().unwrap().clone()); }
        }
        if !many && slot.len() > 1 { bail!("--{long} given more than once"); }
    }

    let one = |k: &str| got.get(k).and_then(|v| v.first().cloned());
    let req = |k: &str| one(k).ok_or_else(|| anyhow!("the following required argument was not provided: --{k}"));
    fn num<T: std::str::FromStr>(k: &str, v: Option<String>, d: T) -> Result<T> where T::Err: std::fmt::Display {
        v.map_or(Ok(d), |s| s.parse::<T>().map_err(|e| anyhow!("invalid value '{s}' for --{k}: {e}")))
    }
    let flag = |k: &str| got.contains_key(k);
    let all = |k: &str| -> Result<Vec<PathBuf>> {
        got.get(k).filter(|v| !v.is_empty()).map(|v| v.iter().map(PathBuf::from).collect())
            .ok_or_else(|| anyhow!("the following required argument was not provided: --{k}"))
    };
    let threads = num("threads", one("threads"), 8usize)?;
    let hi = num("threshold-high", one("threshold-high"), bam::DEFAULT_THRESHOLD_HIGH)?;
    let lo = num("threshold-low", one("threshold-low"), bam::DEFAULT_THRESHOLD_LOW)?;
    let command = match name.as_str() {
        "bam" => Cmd::Bam { input: req("input")?.into(), threads, no_checks: flag("no-checks"), threshold_high: hi, threshold_low: lo },
        "pileup" => Cmd::Pileup { input: req("input")?.into() },
        "beta" => Cmd::Beta { input: req("input")?.into() },
        "counts" => Cmd::Counts {
            input: req("input")?.into(), out: req("out")?.into(),
            shard: one("shard").map_or(Ok(bam::Shard::WHOLE), |s| s.parse().map_err(|e| anyhow!("invalid value '{s}' for --shard: {e}")))?,
            threads, no_checks: flag("no-checks"), threshold_high: hi, threshold_low: lo,
        },
        "inspect" => Cmd::Inspect { input: req("input")?.into(), records: num("records", one("records"), 2000u64)? },
        "classify" => Cmd::Classify { input: all("input")? },
        "apply" => Cmd::Apply { patch: req("patch")?.into(), out: req("out")?.into() },
        "contribute" => Cmd::Contribute { head: one("head"), labels: req("labels")?.into(),
                                          out: one("out").unwrap_or_else(|| "update.safetensors".into()).into(), threads },
        "merge" => Cmd::Merge { updates: all("updates")?, lr: num("lr", one("lr"), 0.01f32)?,
                                max_grad_norm: num("max-grad-norm", one("max-grad-norm"), 0f32)?, patch: one("patch").map(PathBuf::from), out: req("out")?.into() },
        _ => unreachable!(),
    };
    Ok(Cli { model, json, command })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn p(s: &str) -> Result<Cli> { parse_from(&s.split_whitespace().map(String::from).collect::<Vec<_>>()) }

    #[test]
    fn parses_the_documented_forms() {
        let c = p("--model m --json bam -i x.bam --threads 16 --no-checks").unwrap();
        assert!(c.json && c.model == PathBuf::from("m"));
        assert!(matches!(c.command, Cmd::Bam { threads: 16, no_checks: true, .. }));
        let c = p("counts -i x.bam --shard 2/4 -o p.tsv -t 3 --threshold-low=0.2").unwrap();
        assert!(matches!(c.command, Cmd::Counts { shard: bam::Shard { index: 2, n: 4 }, threads: 3, threshold_low, .. } if threshold_low == 0.2));
        let c = p("classify -i a.tsv b.tsv c.tsv --json").unwrap();
        assert!(c.json && matches!(&c.command, Cmd::Classify { input } if input.len() == 3));
        let c = p("merge --updates a b --lr 0.5 --out o --patch r1").unwrap();
        assert!(matches!(&c.command, Cmd::Merge { updates, lr, patch: Some(_), .. } if updates.len() == 2 && *lr == 0.5));
        assert!(matches!(p("contribute --labels l.tsv").unwrap().command, Cmd::Contribute { threads: 8, .. }));
    }

    #[test]
    fn rejects_bad_input() {
        for bad in ["bam", "bam -i", "bam -i x --bogus", "frobnicate", "counts -i x -o y --shard 5", "bam -i x -t many", "classify -i a -i"] {
            assert!(p(bad).is_err(), "{bad} should fail");
        }
    }
}
