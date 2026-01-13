use std::fs;
use std::path::{Path, PathBuf};

pub struct CsvTable {
    pub header: Vec<String>,
    pub rows: Vec<Vec<f64>>,
}

pub fn data_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("openfoam_reference")
        .join("data")
        .join(rel)
}

pub fn load_csv(path: &Path) -> CsvTable {
    let text = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));

    let mut header: Option<Vec<String>> = None;
    let mut rows: Vec<Vec<f64>> = Vec::new();

    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if header.is_none() {
            header = Some(cols.into_iter().map(|s| s.to_string()).collect());
            continue;
        }

        let mut vals = Vec::with_capacity(cols.len());
        for c in cols {
            let v = c
                .trim_end_matches('\r')
                .parse::<f64>()
                .unwrap_or_else(|e| panic!("bad float '{c}' in {}: {e}", path.display()));
            vals.push(v);
        }
        rows.push(vals);
    }

    CsvTable {
        header: header.unwrap_or_default(),
        rows,
    }
}

pub fn column_idx(header: &[String], name: &str) -> usize {
    header
        .iter()
        .position(|h| h.trim_end_matches('\r') == name)
        .unwrap_or_else(|| panic!("missing column '{name}' in CSV header: {header:?}"))
}

pub fn rel_l2(a: &[f64], b: &[f64], floor: f64) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut num = 0.0;
    let mut den = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = x - y;
        num += d * d;
        den += y * y;
    }
    (num / a.len().max(1) as f64).sqrt() / (den / a.len().max(1) as f64).sqrt().max(floor)
}

