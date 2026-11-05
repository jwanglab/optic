//! optic-train: GPU (CUDA) trainer for the optic hierarchical tissue classifier.
//! Input: a spec directory (spec.json: recipe + paths to the beta matrix, probes.tsv, labels.tsv, tree.json).
//! Output: optic.safetensors (optic weight layout) + config.json + log.json.

mod aug;
mod balance;
mod gpu;
mod init;
mod labels;
mod model;
mod purity;
mod rng;
mod spec;
mod train;

use anyhow::{anyhow, bail, Result};
use std::path::PathBuf;

const USAGE: &str = "optic-train - rust-oxide GPU trainer for optic

Usage:
  optic-train train --spec DIR --out DIR [--epochs N] [--max-batches N]
      Train from a spec directory. --epochs overrides the spec; --max-batches stops each epoch early
  optic-train check --spec DIR [--out FILE] [--dump DIR]
      CPU-only dry run: build every table training would use and report it. --out writes row weights, purity bank
      rows and eligible rows as JSON; --dump writes the raw tables (targets, tree levels, weights, read-depth tables)
      into DIR.
  optic-train --version
";

enum Cmd {
    Train { spec: PathBuf, out: PathBuf, epochs: Option<usize>, max_batches: Option<usize> },
    Check { spec: PathBuf, out: Option<PathBuf>, dump: Option<PathBuf> },
}

/// `--flag value` / `--flag=value` pairs after the subcommand; every flag takes a value.
fn parse(args: &[String]) -> Result<Cmd> {
    let sub = args.first().ok_or_else(|| anyhow!("a command is required"))?;
    let allowed: &[&str] = match sub.as_str() { "train" => &["spec", "out", "epochs", "max-batches"], "check" => &["spec", "out", "dump"],
                                                 s => bail!("unrecognized subcommand '{s}'") };
    let mut got = std::collections::HashMap::new();
    let mut it = args[1..].iter();
    while let Some(a) = it.next() {
        let flag = a.strip_prefix("--").ok_or_else(|| anyhow!("unexpected argument '{a}'"))?;
        let (k, v) = match flag.split_once('=') { Some((k, v)) => (k.replace('_', "-"), v.to_string()),
                                                   None => (flag.replace('_', "-"), it.next().ok_or_else(|| anyhow!("--{flag} needs a value"))?.clone()) };
        if !allowed.contains(&k.as_str()) { bail!("unexpected argument '--{k}' for '{sub}'"); }
        if got.insert(k.clone(), v).is_some() { bail!("--{k} given more than once"); }
    }
    let path = |k: &str| got.get(k).map(PathBuf::from);
    let req = |k: &str| path(k).ok_or_else(|| anyhow!("the following required argument was not provided: --{k}"));
    let num = |k: &str| got.get(k).map(|v| v.parse::<usize>().map_err(|e| anyhow!("invalid value '{v}' for --{k}: {e}"))).transpose();
    Ok(match sub.as_str() {
        "train" => Cmd::Train { spec: req("spec")?, out: req("out")?, epochs: num("epochs")?, max_batches: num("max-batches")? },
        _ => Cmd::Check { spec: req("spec")?, out: path("out"), dump: path("dump") },
    })
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help" || a == "help") { print!("{USAGE}"); return Ok(()); }
    if args.iter().any(|a| a == "-V" || a == "--version") { println!("optic-train {}", env!("CARGO_PKG_VERSION")); return Ok(()); }
    let cmd = match parse(&args) {
        Ok(c) => c,
        Err(e) => { eprintln!("error: {e}\n\n{USAGE}"); std::process::exit(2); }
    };
    match cmd {
        Cmd::Train { spec, out, epochs, max_batches } => train::run(&spec, &out, epochs, max_batches),
        Cmd::Check { spec, out, dump } => train::check(&spec, out.as_deref(), dump.as_deref()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn p(s: &str) -> Result<Cmd> { parse(&s.split_whitespace().map(String::from).collect::<Vec<_>>()) }
    #[test]
    fn cli() {
        assert!(matches!(p("train --spec s --out o --epochs 1 --max-batches 200").unwrap(), Cmd::Train { epochs: Some(1), max_batches: Some(200), .. }));
        assert!(matches!(p("train --spec=s --out=o").unwrap(), Cmd::Train { epochs: None, .. }));
        assert!(matches!(p("check --spec s --dump d").unwrap(), Cmd::Check { out: None, dump: Some(_), .. }));
        for bad in ["train --spec s", "train --spec s --out o --epochs x", "check --spec s --bogus 1", "frob", "train --spec"] { assert!(p(bad).is_err(), "{bad}"); }
    }
}
