#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};

pub fn diag_enabled() -> bool {
    std::env::var("CFD2_OPENFOAM_DIAG").is_ok()
}

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

pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

pub fn max_abs(xs: &[f64]) -> f64 {
    xs.iter().map(|v| v.abs()).fold(0.0, f64::max)
}

pub fn max_abs_centered(xs: &[f64]) -> f64 {
    let m = mean(xs);
    xs.iter().map(|v| (v - m).abs()).fold(0.0, f64::max)
}

/// Relative L2 error after applying the best constant offset to `a` (least-squares fit):
///   a_shifted = a + shift, where shift = mean(b - a).
pub fn rel_l2_best_shift(a: &[f64], b: &[f64], floor: f64) -> (f64, f64) {
    assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return (0.0, 0.0);
    }
    let mut sum = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        sum += y - x;
    }
    let shift = sum / a.len() as f64;

    let mut num = 0.0;
    let mut den = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = (x + shift) - y;
        num += d * d;
        den += y * y;
    }
    let n = a.len() as f64;
    let err = (num / n).sqrt() / (den / n).sqrt().max(floor);
    (err, shift)
}

/// Relative L2 error after applying the best affine map to `a` (least-squares fit):
///   a_fit = scale * a + shift.
pub fn rel_l2_best_affine(a: &[f64], b: &[f64], floor: f64) -> (f64, f64, f64) {
    assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return (0.0, 1.0, 0.0);
    }

    let n = a.len() as f64;
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;

    let mut var_a = 0.0;
    let mut cov_ab = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let da = x - mean_a;
        var_a += da * da;
        cov_ab += da * (y - mean_b);
    }

    let scale = if var_a > 1e-30 { cov_ab / var_a } else { 0.0 };
    let shift = mean_b - scale * mean_a;

    let mut num = 0.0;
    let mut den = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = (scale * x + shift) - y;
        num += d * d;
        den += y * y;
    }

    let err = (num / n).sqrt() / (den / n).sqrt().max(floor);
    (err, scale, shift)
}

pub fn yx_key(x: f64, y: f64) -> (i64, i64) {
    let s = 1e12_f64;
    ((y * s).round() as i64, (x * s).round() as i64)
}

pub fn xy_key(x: f64, y: f64) -> (i64, i64) {
    let s = 1e12_f64;
    ((x * s).round() as i64, (y * s).round() as i64)
}
