//! Training inputs
//!   tree.json   -> leaf order, per-depth leaf->node membership (the hierarchical loss levels)
//!   labels.tsv  -> per-row target distribution over the leaves, and which rows train
//!   probes.T2T.tsv  -> number of probes (beta columns); the model's inputs are these probes in file order
//! Target grammar: leaf:<leaf> | node:<node> (uniform over the node's leaves) | mix:<node>=<m>;<node>=<m> (each part
//! uniform over its node's leaves, scaled by its mass). Stored targets are log-probabilities floored at -30.

use anyhow::{bail, ensure, Context, Result};
use serde::de::{Deserializer, MapAccess, Visitor};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// `nodes` (child -> parent, null for the root) in DOCUMENT order: the hierarchical levels list nodes in this order.
pub struct OrderedNodes(pub Vec<(String, Option<String>)>);
impl<'de> Deserialize<'de> for OrderedNodes {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = OrderedNodes;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("a map of node -> parent") }
            fn visit_map<A: MapAccess<'de>>(self, mut m: A) -> std::result::Result<OrderedNodes, A::Error> {
                let mut v = Vec::new();
                while let Some((k, p)) = m.next_entry::<String, Option<String>>()? { v.push((k, p.filter(|s| !s.is_empty()))); }
                Ok(OrderedNodes(v))
            }
        }
        d.deserialize_map(V)
    }
}

#[derive(Deserialize)]
pub struct Tree {
    pub leaves: Vec<String>,
    pub nodes: OrderedNodes,
    #[serde(default)]
    pub leaf_domain: HashMap<String, i64>,
}

impl Tree {
    pub fn load(path: &Path) -> Result<Tree> {
        serde_json::from_str(&std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?).with_context(|| format!("parsing {}", path.display()))
    }
    pub fn parent_map(&self) -> HashMap<&str, &str> {
        self.nodes.0.iter().filter_map(|(c, p)| p.as_deref().map(|p| (c.as_str(), p))).collect()
    }
    /// root .. x
    pub fn path<'a>(&'a self, x: &'a str, par: &HashMap<&'a str, &'a str>) -> Vec<&'a str> {
        let mut p = vec![x];
        while let Some(&q) = par.get(p.last().unwrap()) { p.push(q); }
        p.reverse(); p
    }
    /// leaf indices under every node (a leaf maps to itself)
    pub fn leaf_sets(&self) -> HashMap<&str, Vec<usize>> {
        let par = self.parent_map();
        let mut sets: HashMap<&str, Vec<usize>> = HashMap::new();
        for (i, l) in self.leaves.iter().enumerate() { for a in self.path(l, &par) { sets.entry(a).or_default().push(i); } }
        sets
    }
    /// Hierarchical levels: for depth d = 1.., a (n_nodes, k) 0/1 matrix whose rows are the nodes at depth d that hold
    /// at least one leaf (a leaf sitting at depth d is its own node), in tree.json `nodes` order.
    pub fn hier_levels(&self) -> Vec<(Vec<f32>, usize)> {
        let par = self.parent_map(); let k = self.leaves.len();
        let mut at: Vec<HashMap<&str, Vec<usize>>> = Vec::new();
        for (i, l) in self.leaves.iter().enumerate() {
            let p = self.path(l, &par);
            for d in 1..p.len() {
                if at.len() < d { at.resize_with(d, HashMap::new); }
                at[d - 1].entry(p[d]).or_default().push(i);
            }
        }
        at.iter().map(|g| {
            let order: Vec<&str> = self.nodes.0.iter().map(|(c, _)| c.as_str()).filter(|c| g.contains_key(c)).collect();
            let mut m = vec![0f32; order.len() * k];
            for (r, c) in order.iter().enumerate() { for &j in &g[c] { m[r * k + j] = 1.0; } }
            (m, order.len())
        }).collect()
    }
}

/// Parse labels.tsv into (log-prob targets (n,k) row-major, training mask: 1 for `in_training` rows, 0 elsewhere).
pub fn targets(path: &Path, tree: &Tree) -> Result<(Vec<f32>, Vec<f32>, usize)> {
    let k = tree.leaves.len();
    let sets = tree.leaf_sets();
    let txt = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut lines = txt.lines();
    let hdr: Vec<&str> = lines.next().context("labels.tsv is empty")?.split('\t').collect();
    let col = |name: &str| hdr.iter().position(|&h| h == name);
    let (c_row, c_t, c_in) = (col("row").context("labels.tsv: no `row` column")?, col("target").context("labels.tsv: no `target` column")?, col("in_training").context("labels.tsv: no `in_training` column")?);
    let body: Vec<&str> = lines.filter(|l| !l.is_empty()).collect();
    let n = body.len();
    let mut lut = vec![0f32; n * k]; let mut w = vec![0f32; n];
    let mut v = vec![0f64; k];
    for (i, line) in body.iter().enumerate() {
        let f: Vec<&str> = line.split('\t').collect();
        ensure!(f.len() == hdr.len(), "labels.tsv line {}: {} fields, header has {}", i + 2, f.len(), hdr.len());
        ensure!(f[c_row].parse::<usize>().ok() == Some(i), "labels.tsv must list rows 0..n-1 in order (line {} has row {})", i + 2, f[c_row]);
        let train = f[c_in].trim() == "1";
        w[i] = if train { 1.0 } else { 0.0 };
        let t = f[c_t].trim();
        if t.is_empty() { ensure!(w[i] == 0.0, "row {i}: in_training without a target"); continue; }
        v.iter_mut().for_each(|x| *x = 0.0);
        let leaves_of = |x: &str| -> Result<&Vec<usize>> { sets.get(x).with_context(|| format!("row {i}: '{x}' is not in the tree")) };
        if let Some(x) = t.strip_prefix("leaf:") {
            let s = leaves_of(x)?; ensure!(s.len() == 1 && tree.leaves[s[0]] == x, "row {i}: '{x}' is not a leaf"); v[s[0]] = 1.0;
        } else if let Some(x) = t.strip_prefix("node:") {
            let s = leaves_of(x)?; for &j in s { v[j] = 1.0 / s.len() as f64; }
        } else if let Some(m) = t.strip_prefix("mix:") {
            for part in m.split(';') {
                let (x, mass) = part.rsplit_once('=').with_context(|| format!("row {i}: bad mix part '{part}'"))?;
                let mass: f64 = mass.parse().with_context(|| format!("row {i}: bad mass in '{part}'"))?;
                let s = leaves_of(x)?; for &j in s { v[j] += mass / s.len() as f64; }
            }
        } else { bail!("row {i}: bad target '{t}'"); }
        let tot: f64 = v.iter().sum();
        ensure!((tot - 1.0).abs() < 1e-3, "row {i}: target masses sum to {tot}");
        for j in 0..k { lut[i * k + j] = v[j].max(1e-13).ln().max(-30.0) as f32; }
    }
    Ok((lut, w, n))
}

/// probes.*.tsv: a header and one line per beta column, in beta.u8 column order. Model input j is beta column j, so this
/// only counts the probes.
pub fn probe_count(path: &Path) -> Result<usize> {
    let txt = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let p = txt.lines().skip(1).filter(|l| !l.is_empty()).count();
    ensure!(p > 0, "{}: no probes", path.display());
    Ok(p)
}
