#![allow(dead_code)]

use cfd2::solver::mesh::Mesh;
use image::{Rgb, RgbImage};
use std::collections::HashMap;
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

pub fn reference_fields_from_csv(
    mesh: &Mesh,
    table: &CsvTable,
    x_idx: usize,
    y_idx: usize,
    ux_idx: usize,
    uy_idx: Option<usize>,
    p_idx: usize,
) -> (Vec<(f64, f64)>, Vec<f64>) {
    assert_eq!(
        table.rows.len(),
        mesh.num_cells(),
        "reference rows must equal mesh.num_cells() for field plots"
    );

    let mut cell_map: HashMap<(i64, i64), usize> = HashMap::with_capacity(mesh.num_cells());
    for cell in 0..mesh.num_cells() {
        cell_map.insert(yx_key(mesh.cell_cx[cell], mesh.cell_cy[cell]), cell);
    }

    let mut u_ref = vec![(0.0f64, 0.0f64); mesh.num_cells()];
    let mut p_ref = vec![0.0f64; mesh.num_cells()];

    for row in &table.rows {
        let key = yx_key(row[x_idx], row[y_idx]);
        let cell = *cell_map.get(&key).unwrap_or_else(|| {
            panic!(
                "reference point ({:.12},{:.12}) not found in mesh cell centers",
                row[x_idx], row[y_idx]
            )
        });
        u_ref[cell] = (row[ux_idx], uy_idx.map(|idx| row[idx]).unwrap_or(0.0));
        p_ref[cell] = row[p_idx];
    }

    (u_ref, p_ref)
}

fn mesh_bounds(mesh: &Mesh) -> (f64, f64, f64, f64) {
    let min_x = mesh.vx.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_x = mesh.vx.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_y = mesh.vy.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_y = mesh.vy.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (min_x, max_x, min_y, max_y)
}

fn cell_polygon(mesh: &Mesh, cell: usize) -> Vec<[f64; 2]> {
    let start = mesh.cell_vertex_offsets[cell];
    let end = mesh.cell_vertex_offsets[cell + 1];
    let mut points = Vec::with_capacity(end - start);
    for idx in start..end {
        let v = mesh.cell_vertices[idx];
        points.push([mesh.vx[v], mesh.vy[v]]);
    }
    points
}

fn point_in_poly(px: f64, py: f64, poly: &[[f64; 2]]) -> bool {
    if poly.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let xi = poly[i][0];
        let yi = poly[i][1];
        let xj = poly[j][0];
        let yj = poly[j][1];
        let crosses = (yi > py) != (yj > py) && px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi;
        if crosses {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn colormap_sequential(t: f64) -> Rgb<u8> {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.5 {
        let s = t * 2.0;
        (0.0, s, 1.0 - s)
    } else {
        let s = (t - 0.5) * 2.0;
        (s, 1.0 - s, 0.0)
    };
    Rgb([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8])
}

fn colormap_diverging(t: f64) -> Rgb<u8> {
    let t = t.clamp(-1.0, 1.0);
    if t < 0.0 {
        let s = (-t).clamp(0.0, 1.0);
        let r = 255.0 * (1.0 - s);
        let g = 255.0 * (1.0 - s);
        Rgb([r as u8, g as u8, 255])
    } else {
        let s = t.clamp(0.0, 1.0);
        let g = 255.0 * (1.0 - s);
        let b = 255.0 * (1.0 - s);
        Rgb([255, g as u8, b as u8])
    }
}

fn save_scalar_image_with_scale(
    path: &Path,
    mesh: &Mesh,
    values: &[f64],
    width: usize,
    height: usize,
    min_val: f64,
    max_val: f64,
    diverging: bool,
) {
    if values.len() != mesh.num_cells() || width == 0 || height == 0 {
        return;
    }

    let (min_x, max_x, min_y, max_y) = mesh_bounds(mesh);
    let mesh_w = (max_x - min_x).max(1e-12);
    let mesh_h = (max_y - min_y).max(1e-12);
    let scale_x = (width as f64 - 1.0) / mesh_w;
    let scale_y = (height as f64 - 1.0) / mesh_h;
    let scale = scale_x.min(scale_y);
    let center_x = 0.5 * (min_x + max_x);
    let center_y = 0.5 * (min_y + max_y);
    let offset_x = (width as f64 - 1.0) * 0.5;
    let offset_y = (height as f64 - 1.0) * 0.5;

    let mut sum = vec![0.0; width * height];
    let mut count = vec![0u32; width * height];

    for cell in 0..mesh.num_cells() {
        let poly = cell_polygon(mesh, cell);
        if poly.len() < 3 {
            continue;
        }
        let mut poly_px = Vec::with_capacity(poly.len());
        let mut min_px = f64::INFINITY;
        let mut max_px = f64::NEG_INFINITY;
        let mut min_py = f64::INFINITY;
        let mut max_py = f64::NEG_INFINITY;
        for p in &poly {
            let x = (p[0] - center_x) * scale + offset_x;
            let y = (center_y - p[1]) * scale + offset_y;
            min_px = min_px.min(x);
            max_px = max_px.max(x);
            min_py = min_py.min(y);
            max_py = max_py.max(y);
            poly_px.push([x, y]);
        }
        let x0 = min_px.floor().max(0.0) as i32;
        let x1 = max_px.ceil().min(width as f64 - 1.0) as i32;
        let y0 = min_py.floor().max(0.0) as i32;
        let y1 = max_py.ceil().min(height as f64 - 1.0) as i32;
        for yi in y0..=y1 {
            for xi in x0..=x1 {
                let px = xi as f64 + 0.5;
                let py = yi as f64 + 0.5;
                if point_in_poly(px, py, &poly_px) {
                    let idx = yi as usize * width + xi as usize;
                    sum[idx] += values[cell];
                    count[idx] += 1;
                }
            }
        }
    }

    let mut img = RgbImage::new(width as u32, height as u32);
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let color = if count[idx] == 0 {
                Rgb([10, 10, 10])
            } else {
                let val = sum[idx] / count[idx] as f64;
                if diverging {
                    let denom = max_val.abs().max(min_val.abs()).max(1e-12);
                    colormap_diverging(val / denom)
                } else {
                    let denom = (max_val - min_val).max(1e-12);
                    let t = (val - min_val) / denom;
                    colormap_sequential(t)
                }
            };
            img.put_pixel(x as u32, y as u32, color);
        }
    }
    let _ = img.save(path);
}

fn range_min_max(values: &[f64]) -> (f64, f64) {
    values
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |acc, &v| {
            (acc.0.min(v), acc.1.max(v))
        })
}

fn range_min_max_two(a: &[f64], b: &[f64]) -> (f64, f64) {
    let (a_min, a_max) = range_min_max(a);
    let (b_min, b_max) = range_min_max(b);
    (a_min.min(b_min), a_max.max(b_max))
}

fn max_abs_two(a: &[f64], b: &[f64]) -> f64 {
    let a_max = a.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let b_max = b.iter().map(|v| v.abs()).fold(0.0, f64::max);
    a_max.max(b_max)
}

pub fn save_openfoam_field_plots(
    case_name: &str,
    mesh: &Mesh,
    u_ref: &[(f64, f64)],
    p_ref: &[f64],
    u_sol: &[(f64, f64)],
    p_sol: &[f64],
) {
    if u_ref.len() != mesh.num_cells()
        || p_ref.len() != mesh.num_cells()
        || u_sol.len() != mesh.num_cells()
        || p_sol.len() != mesh.num_cells()
    {
        return;
    }

    let base_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("test_plots")
        .join("openfoam")
        .join(case_name);
    let _ = fs::create_dir_all(&base_dir);

    let width = 520usize;
    let height = 260usize;

    let ux_ref: Vec<f64> = u_ref.iter().map(|v| v.0).collect();
    let uy_ref: Vec<f64> = u_ref.iter().map(|v| v.1).collect();
    let ux_sol: Vec<f64> = u_sol.iter().map(|v| v.0).collect();
    let uy_sol: Vec<f64> = u_sol.iter().map(|v| v.1).collect();
    let speed_ref: Vec<f64> = u_ref
        .iter()
        .map(|v| (v.0 * v.0 + v.1 * v.1).sqrt())
        .collect();
    let speed_sol: Vec<f64> = u_sol
        .iter()
        .map(|v| (v.0 * v.0 + v.1 * v.1).sqrt())
        .collect();

    let p_ref_mean = mean(p_ref);
    let p_sol_mean = mean(p_sol);
    let p_ref_c: Vec<f64> = p_ref.iter().map(|v| v - p_ref_mean).collect();
    let p_sol_c: Vec<f64> = p_sol.iter().map(|v| v - p_sol_mean).collect();

    let maxabs_ux = max_abs_two(&ux_ref, &ux_sol).max(1e-12);
    let maxabs_uy = max_abs_two(&uy_ref, &uy_sol).max(1e-12);
    let maxabs_p = max_abs_two(&p_ref_c, &p_sol_c).max(1e-12);
    let (min_speed, max_speed) = range_min_max_two(&speed_ref, &speed_sol);

    save_scalar_image_with_scale(
        &base_dir.join("u_x_ref.png"),
        mesh,
        &ux_ref,
        width,
        height,
        -maxabs_ux,
        maxabs_ux,
        true,
    );
    save_scalar_image_with_scale(
        &base_dir.join("u_x_sol.png"),
        mesh,
        &ux_sol,
        width,
        height,
        -maxabs_ux,
        maxabs_ux,
        true,
    );
    save_scalar_image_with_scale(
        &base_dir.join("u_y_ref.png"),
        mesh,
        &uy_ref,
        width,
        height,
        -maxabs_uy,
        maxabs_uy,
        true,
    );
    save_scalar_image_with_scale(
        &base_dir.join("u_y_sol.png"),
        mesh,
        &uy_sol,
        width,
        height,
        -maxabs_uy,
        maxabs_uy,
        true,
    );
    save_scalar_image_with_scale(
        &base_dir.join("u_mag_ref.png"),
        mesh,
        &speed_ref,
        width,
        height,
        min_speed,
        max_speed.max(min_speed + 1e-12),
        false,
    );
    save_scalar_image_with_scale(
        &base_dir.join("u_mag_sol.png"),
        mesh,
        &speed_sol,
        width,
        height,
        min_speed,
        max_speed.max(min_speed + 1e-12),
        false,
    );
    save_scalar_image_with_scale(
        &base_dir.join("p_ref.png"),
        mesh,
        &p_ref_c,
        width,
        height,
        -maxabs_p,
        maxabs_p,
        true,
    );
    save_scalar_image_with_scale(
        &base_dir.join("p_sol.png"),
        mesh,
        &p_sol_c,
        width,
        height,
        -maxabs_p,
        maxabs_p,
        true,
    );
}
