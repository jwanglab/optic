fn main() -> anyhow::Result<()> {
    let dir = std::path::Path::new(&std::env::args().nth(1).unwrap()).to_path_buf();
    let a = optic_core::artifact::Artifact::load(&dir)?;
    let t = a.lineage.as_ref().unwrap();
    let ch = t.children();
    let leaves: Vec<&str> = t.nodes.keys().map(|k| k.as_str())
        .filter(|n| *n != "ROOT" && ch.get(n).map_or(true, |k| k.is_empty())).collect();
    println!("nodes {}  leaves {}  nonreportable {}  alerts {}  taus {:?}",
        t.nodes.len(), leaves.len(), t.nonreportable.len(), t.leaf_alerts.len(), t.descent_thresholds);
    println!("leaves_under(Vascular) = {:?}", t.leaves_under("Vascular"));
    for a in &t.leaf_alerts { println!("alert: {} p_min {}", a.leaf, a.p_min); }
    Ok(())
}
