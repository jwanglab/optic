//! lineage.json, optic.safetensors, probe_subset.bed

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub struct ModelInfo {
    pub name: String,
    pub version: String,
    pub heads: HashMap<String, HeadInfo>,
}

pub struct HeadInfo {
    pub class_names: Vec<String>,
}

pub struct Tree {
    pub nodes: HashMap<String, Option<String>>,
    /// Per-depth descent thresholds ("1" -> tau_1, ...): the report steps down to depth d only when the best child's summed leaf mass is >= tau_d.
    pub descent_thresholds: Option<HashMap<String, f32>>,
    pub leaf_alerts: Vec<LeafAlert>,
}

pub struct LeafAlert {
    pub leaf: String,
    pub p_min: f32,
    pub note: Option<String>,
}

impl Tree {
    pub fn children(&self) -> HashMap<&str, Vec<&str>> {
        let mut c: HashMap<&str, Vec<&str>> = self.nodes.keys().map(|k| (k.as_str(), Vec::new())).collect();
        let mut kids: Vec<(&str, &str)> = self.nodes.iter().filter_map(|(n, p)| p.as_deref().map(|p| (p, n.as_str()))).collect();
        kids.sort();
        for (p, n) in kids { c.entry(p).or_default().push(n); }
        c
    }

    pub fn leaves_under(&self, node: &str) -> Vec<String> {
        let ch = self.children();
        fn walk(n: &str, ch: &HashMap<&str, Vec<&str>>, out: &mut Vec<String>) {
            match ch.get(n) {
                Some(k) if !k.is_empty() => k.iter().for_each(|c| walk(c, ch, out)),
                _ => out.push(n.to_string()),
            }
        }
        let mut out = Vec::new();
        walk(node, &ch, &mut out);
        out
    }
}

#[derive(Deserialize)]
struct RefDoc {
    name: String,
    version: String,
    #[serde(default = "cell_origin")]
    head: String,
    /// ORDERED output nodes: index i of the softmax is leaves[i].
    leaves: Vec<String>,
    #[serde(default)]
    descent_thresholds: Option<HashMap<String, f32>>,
    hierarchy: Vec<RefNode>,
}

#[derive(Deserialize)]
struct RefNode {
    name: String,
    #[serde(default)]
    children: Vec<RefNode>,
    #[serde(default)]
    alert: Option<RefAlert>,
}

#[derive(Deserialize)]
struct RefAlert {
    p_min: f32,
    #[serde(default)]
    note: Option<String>,
}

fn cell_origin() -> String { "cell_origin".to_string() }

impl RefDoc {
    fn info(&self) -> ModelInfo {
        ModelInfo { name: self.name.clone(), version: self.version.clone(),
                   heads: [(self.head.clone(), HeadInfo { class_names: self.leaves.clone() })].into_iter().collect() }
    }

    fn flatten(self) -> Result<Tree> {
        let mut t = Tree { nodes: HashMap::new(), descent_thresholds: self.descent_thresholds,
                           leaf_alerts: Vec::new() };
        t.nodes.insert("ROOT".to_string(), None);
        fn walk(n: RefNode, parent: &str, t: &mut Tree) -> Result<()> {
            if t.nodes.insert(n.name.clone(), Some(parent.to_string())).is_some() {
                return Err(anyhow!("lineage.json: duplicate node name {:?}", n.name));
            }
            if let Some(a) = n.alert {
                if !n.children.is_empty() { return Err(anyhow!("lineage.json: alert on non-leaf {:?}", n.name)); }
                t.leaf_alerts.push(LeafAlert { leaf: n.name.clone(), p_min: a.p_min, note: a.note });
            }
            for c in n.children { walk(c, &n.name, t)?; }
            Ok(())
        }
        for top in self.hierarchy { walk(top, "ROOT", &mut t)?; }
        Ok(t)
    }
}

pub struct Model {
    pub dir: PathBuf,
    pub head: String,
    pub info: ModelInfo,
    pub lineage: Tree,
}

impl Model {
    pub fn load(dir: &Path) -> Result<Self> {

        // A model is exactly three files: lineage.json, optic.safetensors, probe_subset.bed.
        let doc: RefDoc = serde_json::from_str(
            &std::fs::read_to_string(dir.join("lineage.json"))
                .with_context(|| format!("{} has no lineage.json", dir.display()))?)
            .context("lineage.json")?;
        if !dir.join("optic.safetensors").exists() { return Err(anyhow!("{} has no optic.safetensors", dir.display())); }
        let info = doc.info();
        let head = doc.head.clone();
        let model = Model { dir: dir.to_path_buf(), head: head.clone(), info, lineage: doc.flatten()? };
        {
            let (tree, names) = (&model.lineage, model.class_names(&head).unwrap_or(&[]));
            let named: std::collections::HashSet<&str> = names.iter().map(|s| s.as_str()).collect();
            let ch = tree.children();
            let missing: Vec<&str> = tree.nodes.keys().map(|k| k.as_str())
                .filter(|n| *n != "ROOT" && ch.get(n).map_or(true, |k| k.is_empty()) && !named.contains(n))
                .collect();
            if !missing.is_empty() {
                return Err(anyhow!("lineage leaves missing: {:?}", missing));
            }
        }
        Ok(model)
    }

    pub fn class_names(&self, head: &str) -> Option<&[String]> {
        self.info.heads.get(head).map(|h| h.class_names.as_slice())
    }
}
