use crate::model::Model;
use crate::net::Prediction;
use serde::Serialize;
use std::collections::HashMap;

const P_MIN: f32 = 0.5;

/// One step of the descent. `p` is the summed leaf mass under `node`; the step is accepted only
/// when `p >= tau`, and the first rejected step ends the call.
#[derive(Serialize)]
pub struct Step { pub depth: usize, pub node: String, pub p: f32, pub tau: f32, pub met: bool }

#[derive(Serialize)]
pub struct Alert { pub leaf: String, pub p: f32, pub p_min: f32, pub note: Option<String> }

#[derive(Serialize)]
pub struct Report {
    pub descent: Vec<Step>,
    pub call: Option<String>,
    pub call_depth: usize,
    pub call_p: f32,
    pub alerts: Vec<Alert>,
}

pub fn build(model: &Model, pred: &Prediction) -> Report {
    let head_probs = pred.probs.iter().find(|(h, _)| *h == model.head).map(|(_, p)| p);
    let (p, names) = match (head_probs, model.class_names(&model.head)) {
        (Some(p), Some(n)) => (p.as_slice(), n),
        _ => return Report { descent: Vec::new(), call: None, call_depth: 0, call_p: 0.0, alerts: Vec::new() },
    };
    let descent = descend(&model.lineage, p, names);
    let accepted = descent.iter().take_while(|s| s.met).count();
    let call = descent.get(accepted.wrapping_sub(1)).filter(|_| accepted > 0);
    let alerts = leaf_alerts(&model.lineage, p, names, call.map(|s| s.node.as_str()));
    Report {
        call: call.map(|s| s.node.clone()),
        call_depth: call.map_or(0, |s| s.depth),
        call_p: call.map_or(0.0, |s| s.p),
        descent, alerts,
    }
}

/// A leaf the argmax can lose while still holding mass far above anything that leaf carries on other
/// specimens - flag as suspicious.
fn leaf_alerts(tree: &crate::model::Tree, p: &[f32], leaf_names: &[String], called: Option<&str>) -> Vec<Alert> {
    let mut out: Vec<Alert> = tree.leaf_alerts.iter().filter_map(|a| {
        let i = leaf_names.iter().position(|n| *n == a.leaf)?;
        let pv = *p.get(i)?;
        if pv < a.p_min || called == Some(a.leaf.as_str()) { return None; }
        Some(Alert { leaf: a.leaf.clone(), p: pv, p_min: a.p_min, note: a.note.clone() })
    }).collect();
    out.sort_by(|x, y| y.p.total_cmp(&x.p));
    out
}

/// Greedy walk to a leaf, always taking the child with the highest total weight.
fn descend(tree: &crate::model::Tree, p: &[f32], leaf_names: &[String]) -> Vec<Step> {
    let leaf_p: HashMap<&str, f32> = leaf_names.iter().map(|n| n.as_str()).zip(p.iter().copied()).collect();
    let children = tree.children();
    fn mass(n: &str, ch: &HashMap<&str, Vec<&str>>, lp: &HashMap<&str, f32>) -> f32 {
        match ch.get(n) { Some(k) if !k.is_empty() => k.iter().map(|c| mass(c, ch, lp)).sum(), _ => *lp.get(n).unwrap_or(&0.0) }
    }
    let mut out: Vec<Step> = Vec::new();
    let mut node = "ROOT";
    while let Some(kids) = children.get(node).filter(|k| !k.is_empty()) {
        let (best, pb) = kids.iter().map(|k| (*k, mass(k, &children, &leaf_p)))
            .fold(("", f32::MIN), |b, x| if x.1 > b.1 { x } else { b });
        let depth = out.len() + 1;
        let tau = tree.descent_thresholds.as_ref()
            .and_then(|t| t.get(&depth.to_string()).copied()).unwrap_or(P_MIN);
        let met = pb >= tau && out.last().map_or(true, |s| s.met);
        out.push(Step { depth, node: best.to_string(), p: pb, tau, met });
        node = best;
    }
    out
}

pub fn render(r: &Report, pred: &Prediction, model: &Model, source: &str) -> String {
    let mut s = format!("optic {} v{}  |  {}\n  probes covered : {} / {}\n\n",
        model.info.name, model.info.version, source, pred.n_covered, n_probes(model));
    let w = r.descent.iter().map(|x| x.node.chars().count()).max().unwrap_or(0).max(20);
    let mut cut = false;
    for st in &r.descent {
        let mark = if st.met { "" } else if !cut { cut = true; "   <- below threshold, call stops above" } else { "" };
        s += &format!("  d{}  {:<w$}  {:.3}  {}  {:.2}{}\n",
            st.depth, st.node, st.p, if st.p >= st.tau { ">=" } else { " <" }, st.tau, mark, w = w);
    }
    for a in &r.alerts {
        s += &format!("  !   contender: {} ({:.3}, alert at {:.3}){}\n",
            a.leaf, a.p, a.p_min, a.note.as_ref().map(|n| format!(" - {n}")).unwrap_or_default());
    }
    s += &match &r.call {
        Some(c) => format!("\n  call : {}  (depth {}, p {:.3})\n", c, r.call_depth, r.call_p),
        None => "\n  call : none - nothing met the depth-1 threshold\n".to_string(),
    };
    s
}

fn n_probes(model: &Model) -> usize {
    std::fs::read_to_string(model.dir.join("probe_subset.bed"))
        .map(|s| s.lines().filter(|l| !l.starts_with("chrom") && !l.trim().is_empty()).count())
        .unwrap_or(0)
}
