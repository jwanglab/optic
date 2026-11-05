//! Warnings about the input that the run recovered from (corrupt BAM blocks, malformed records, an unusable index).
//! Each is printed to stderr as it happens and collected for the report

use std::collections::BTreeMap;
use std::sync::Mutex;

/// code -> (first message, how many times it occurred)
static SEEN: Mutex<BTreeMap<&'static str, (String, u64)>> = Mutex::new(BTreeMap::new());

/// Record a warning. The first few of each code are printed; all are counted.
pub fn warn(code: &'static str, msg: impl std::fmt::Display) {
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let e = seen.entry(code).or_insert_with(|| (msg.to_string(), 0));
    e.1 += 1;
    if e.1 <= 5 { eprintln!("[optic] WARNING {code}: {msg}"); }
    if e.1 == 6 { eprintln!("[optic] WARNING {code}: further {code} warnings are counted, not printed"); }
}

/// All warnings so far, one per code: (code, message with its count).
pub fn take() -> Vec<(&'static str, String)> {
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    std::mem::take(&mut *seen).into_iter()
        .map(|(c, (m, n))| (c, if n > 1 { format!("{m} (and {} more like it)", n - 1) } else { m })).collect()
}
