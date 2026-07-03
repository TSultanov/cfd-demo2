//! Mesh-generator validation matrix: runs every unstructured generator
//! (cut-cell, Delaunay, Voronoi) over every GUI geometry and checks the
//! solver-facing invariants a `Mesh` must satisfy. Run with:
//!
//! ```sh
//! cargo test --features meshgen --test meshgen_validation -- --nocapture
//! ```

#![cfg(feature = "meshgen")]

use cfd2::solver::mesh::{
    generate_cut_cell_mesh, generate_delaunay_mesh, generate_meshless_voronoi_mesh,
    generate_voronoi_mesh, BackwardsStep, ChannelWithObstacle, Geometry, Mesh, Nozzle,
    RectangularChannel,
};
use nalgebra::{Point2, Vector2};
use std::panic::AssertUnwindSafe;

/// Estimate the fluid area of a geometry by sampling `is_inside` on a fine grid.
fn fluid_area_estimate(geo: &(impl Geometry + Sync), domain: Vector2<f64>, n: usize) -> f64 {
    let mut hits = 0usize;
    for j in 0..n {
        for i in 0..n {
            let p = Point2::new(
                (i as f64 + 0.5) / n as f64 * domain.x,
                (j as f64 + 0.5) / n as f64 * domain.y,
            );
            if geo.is_inside(&p) {
                hits += 1;
            }
        }
    }
    hits as f64 / (n * n) as f64 * domain.x * domain.y
}

/// Validate solver-facing invariants. Returns a list of human-readable issues.
fn validate_mesh(mesh: &Mesh, expected_area: f64, min_cell_size: f64) -> Vec<String> {
    let mut issues = Vec::new();
    let nc = mesh.num_cells();
    let nf = mesh.num_faces();
    let nv = mesh.num_vertices();

    if nc == 0 {
        issues.push("mesh has 0 cells".to_string());
        return issues;
    }

    // 1. Finiteness
    let mut nan_count = 0usize;
    for arr in [
        &mesh.vx, &mesh.vy, &mesh.face_cx, &mesh.face_cy, &mesh.face_nx, &mesh.face_ny,
        &mesh.face_area, &mesh.cell_cx, &mesh.cell_cy, &mesh.cell_vol,
    ] {
        nan_count += arr.iter().filter(|v| !v.is_finite()).count();
    }
    if nan_count > 0 {
        issues.push(format!("{nan_count} non-finite geometric values"));
    }

    // 2. Cell volumes positive; count tiny cells
    let mut nonpos_vol = 0usize;
    let mut tiny_vol = 0usize;
    let mut min_vol = f64::MAX;
    let nominal_vol = min_cell_size * min_cell_size;
    for &v in &mesh.cell_vol {
        if !(v > 0.0) {
            nonpos_vol += 1;
        } else {
            min_vol = min_vol.min(v);
            if v < 1e-3 * nominal_vol {
                tiny_vol += 1;
            }
        }
    }
    if nonpos_vol > 0 {
        issues.push(format!("{nonpos_vol}/{nc} cells with volume <= 0"));
    }
    if tiny_vol > 0 {
        issues.push(format!(
            "{tiny_vol}/{nc} cells with volume < 1e-3*h^2 (min {min_vol:.3e} vs h^2 {nominal_vol:.3e})"
        ));
    }

    // 3. Face sanity: indices, areas, unit normals, owner!=neighbor
    let mut bad_idx = 0usize;
    let mut zero_area = 0usize;
    let mut bad_normal = 0usize;
    let mut self_neighbor = 0usize;
    let mut inward_normal = 0usize;
    for f in 0..nf {
        let o = mesh.face_owner[f];
        if o >= nc || mesh.face_v1[f] >= nv || mesh.face_v2[f] >= nv {
            bad_idx += 1;
            continue;
        }
        if let Some(nb) = mesh.face_neighbor[f] {
            if nb >= nc {
                bad_idx += 1;
                continue;
            }
            if nb == o {
                self_neighbor += 1;
            }
        }
        if !(mesh.face_area[f] > 1e-12 * min_cell_size) {
            zero_area += 1;
        }
        let nrm = (mesh.face_nx[f] * mesh.face_nx[f] + mesh.face_ny[f] * mesh.face_ny[f]).sqrt();
        if !((nrm - 1.0).abs() < 1e-6) {
            bad_normal += 1;
        } else {
            // Normal must point away from owner (toward neighbor).
            let target = if let Some(nb) = mesh.face_neighbor[f] {
                Point2::new(mesh.cell_cx[nb], mesh.cell_cy[nb])
            } else {
                Point2::new(mesh.face_cx[f], mesh.face_cy[f])
            };
            let c = Point2::new(mesh.cell_cx[o], mesh.cell_cy[o]);
            let d = target - c;
            if d.norm() > 1e-12 && d.x * mesh.face_nx[f] + d.y * mesh.face_ny[f] < 0.0 {
                inward_normal += 1;
            }
        }
    }
    if bad_idx > 0 {
        issues.push(format!("{bad_idx}/{nf} faces with out-of-range indices"));
    }
    if zero_area > 0 {
        issues.push(format!("{zero_area}/{nf} faces with ~zero area"));
    }
    if bad_normal > 0 {
        issues.push(format!("{bad_normal}/{nf} faces with non-unit normal"));
    }
    if self_neighbor > 0 {
        issues.push(format!("{self_neighbor}/{nf} faces with owner==neighbor"));
    }
    if inward_normal > 0 {
        issues.push(format!("{inward_normal}/{nf} faces with normal not pointing owner->neighbor"));
    }

    // 4. Per-cell closure: sum of outward A*n over the cell's faces must be ~0.
    let mut open_cells = 0usize;
    let mut worst_closure = 0.0f64;
    let mut few_faces = 0usize;
    for c in 0..nc {
        let start = mesh.cell_face_offsets[c];
        let end = mesh.cell_face_offsets[c + 1];
        if end - start < 3 {
            few_faces += 1;
        }
        let mut sx = 0.0;
        let mut sy = 0.0;
        let mut perim = 0.0;
        for &f in &mesh.cell_faces[start..end] {
            let sign = if mesh.face_owner[f] == c { 1.0 } else { -1.0 };
            sx += sign * mesh.face_area[f] * mesh.face_nx[f];
            sy += sign * mesh.face_area[f] * mesh.face_ny[f];
            perim += mesh.face_area[f];
        }
        if perim > 0.0 {
            let rel = (sx * sx + sy * sy).sqrt() / perim;
            worst_closure = worst_closure.max(rel);
            if rel > 1e-6 {
                open_cells += 1;
            }
        }
    }
    if few_faces > 0 {
        issues.push(format!("{few_faces}/{nc} cells with < 3 faces"));
    }
    if open_cells > 0 {
        issues.push(format!(
            "{open_cells}/{nc} cells NOT closed (worst rel closure {worst_closure:.3e})"
        ));
    }

    // 5. Total volume vs expected fluid area
    let total: f64 = mesh.cell_vol.iter().filter(|v| v.is_finite()).sum();
    let rel_err = (total - expected_area).abs() / expected_area;
    if rel_err > 0.02 {
        issues.push(format!(
            "total volume {total:.4} vs expected {expected_area:.4} (rel err {:.2}%)",
            rel_err * 100.0
        ));
    }

    // 6. Untagged boundary faces (no neighbor, no boundary type) — these
    // default to free-slip in the solver and hide missing walls.
    let untagged = (0..nf)
        .filter(|&f| mesh.face_neighbor[f].is_none() && mesh.face_boundary[f].is_none())
        .count();
    if untagged > 0 {
        issues.push(format!("{untagged}/{nf} boundary faces untagged (free-slip default)"));
    }

    issues
}

fn skew_stats(mesh: &Mesh) -> (f64, f64) {
    // (max skewness, mean skewness) over internal faces
    let mut max_s = 0.0f64;
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for f in 0..mesh.num_faces() {
        if let Some(nb) = mesh.face_neighbor[f] {
            let o = mesh.face_owner[f];
            let d = Vector2::new(
                mesh.cell_cx[nb] - mesh.cell_cx[o],
                mesh.cell_cy[nb] - mesh.cell_cy[o],
            );
            if d.norm() > 1e-14 {
                let dn = d.normalize();
                let s = 1.0 - (dn.x * mesh.face_nx[f] + dn.y * mesh.face_ny[f]).abs();
                max_s = max_s.max(s);
                sum += s;
                n += 1;
            }
        }
    }
    (max_s, if n > 0 { sum / n as f64 } else { 0.0 })
}

fn run_geometry(
    generator: &str,
    smooth: bool,
    name: &str,
    geo: &(impl Geometry + Sync),
    domain: Vector2<f64>,
) -> usize {
    // (min, max) cell-size pairs: uniform GUI-like sizes plus graded cases.
    let sizes = [
        (0.1, 0.1),
        (0.05, 0.05),
        (0.025, 0.025),
        (0.0125, 0.0125),
        (0.02, 0.08),
    ];
    let mut failures = 0usize;
    let expected = fluid_area_estimate(geo, domain, 2000);
    {
        for &(hmin, hmax) in &sizes {
            let label = format!("{generator}/{name}/h={hmin}/{hmax}");
            let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                let mut mesh = match generator {
                    "delaunay" => generate_delaunay_mesh(geo, hmin, hmax, 1.2, domain),
                    "voronoi" => generate_voronoi_mesh(geo, hmin, hmax, 1.2, domain),
                    "meshless" => generate_meshless_voronoi_mesh(geo, hmin, hmax, 1.2, domain),
                    "cutcell" => generate_cut_cell_mesh(geo, hmin, hmax, 1.2, domain),
                    _ => unreachable!(),
                };
                if smooth {
                    mesh.smooth(geo, 0.3, 50);
                }
                mesh
            }));
            match result {
                Err(e) => {
                    let msg = e
                        .downcast_ref::<&str>()
                        .map(|s| s.to_string())
                        .or_else(|| e.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "<non-string panic>".to_string());
                    println!("[{label}] PANIC: {msg}");
                    failures += 1;
                }
                Ok(mesh) => {
                    let issues = validate_mesh(&mesh, expected, hmin);
                    let (max_skew, mean_skew) = skew_stats(&mesh);
                    if issues.is_empty() {
                        println!(
                            "[{label}] OK cells={} skew max/mean {:.3}/{:.4}",
                            mesh.num_cells(),
                            max_skew,
                            mean_skew
                        );
                    } else {
                        failures += 1;
                        println!(
                            "[{label}] FAIL cells={} skew max/mean {:.3}/{:.4}",
                            mesh.num_cells(),
                            max_skew,
                            mean_skew
                        );
                        for issue in &issues {
                            println!("    - {issue}");
                        }
                    }
                }
            }
        }
    }
    failures
}

fn run_matrix(generator: &str, smooth: bool) -> usize {
    let mut failures = 0usize;
    failures += run_geometry(
        generator,
        smooth,
        "rect_channel",
        &RectangularChannel { length: 3.0, height: 1.0 },
        Vector2::new(3.0, 1.0),
    );
    failures += run_geometry(
        generator,
        smooth,
        "obstacle",
        &ChannelWithObstacle {
            length: 3.0,
            height: 1.0,
            obstacle_center: Point2::new(1.0, 0.51),
            obstacle_radius: 0.1,
        },
        Vector2::new(3.0, 1.0),
    );
    failures += run_geometry(
        generator,
        smooth,
        "backstep",
        &BackwardsStep {
            length: 3.5,
            height_inlet: 0.5,
            height_outlet: 1.0,
            step_x: 0.5,
        },
        Vector2::new(3.5, 1.0),
    );
    failures += run_geometry(
        generator,
        smooth,
        "nozzle",
        &Nozzle {
            length: 3.0,
            height: 1.0,
            throat_height: 0.40,
            throat_frac: 0.40,
            exit_height: 0.80,
        },
        Vector2::new(3.0, 1.0),
    );
    failures
}

#[test]
fn delaunay_all_geometries_valid() {
    let failures = run_matrix("delaunay", false);
    assert_eq!(failures, 0, "{failures} delaunay case(s) failed validation");
}

#[test]
fn voronoi_all_geometries_valid() {
    let failures = run_matrix("voronoi", false);
    assert_eq!(failures, 0, "{failures} voronoi case(s) failed validation");
}

/// Meshless-engine counterpart of `voronoi_all_geometries_valid` (M0.4):
/// the same invariant battery over the same geometry/size matrix. NOTE the
/// meshless path must never run `Mesh::smooth` afterwards (vertex smoothing
/// would move Voronoi vertices off the bisectors), so it is deliberately
/// absent from `smoothed_meshes_stay_valid`.
#[test]
fn meshless_all_geometries_valid() {
    let failures = run_matrix("meshless", false);
    assert_eq!(failures, 0, "{failures} meshless case(s) failed validation");
}

#[test]
fn cutcell_all_geometries_valid() {
    let failures = run_matrix("cutcell", false);
    assert_eq!(failures, 0, "{failures} cutcell case(s) failed validation");
}

#[test]
fn generators_are_deterministic() {
    let geo = ChannelWithObstacle {
        length: 3.0,
        height: 1.0,
        obstacle_center: Point2::new(1.0, 0.51),
        obstacle_radius: 0.1,
    };
    let domain = Vector2::new(3.0, 1.0);
    for generator in ["delaunay", "voronoi", "cutcell"] {
        let build = || match generator {
            "delaunay" => generate_delaunay_mesh(&geo, 0.05, 0.05, 1.2, domain),
            "voronoi" => generate_voronoi_mesh(&geo, 0.05, 0.05, 1.2, domain),
            "cutcell" => generate_cut_cell_mesh(&geo, 0.05, 0.05, 1.2, domain),
            _ => unreachable!(),
        };
        let a = build();
        let b = build();
        assert_eq!(a.num_cells(), b.num_cells(), "{generator}: cell count differs");
        assert_eq!(a.vx, b.vx, "{generator}: vertex x coords differ");
        assert_eq!(a.vy, b.vy, "{generator}: vertex y coords differ");
        assert_eq!(a.cell_vertices, b.cell_vertices, "{generator}: rings differ");
        assert_eq!(a.face_owner, b.face_owner, "{generator}: face owners differ");
    }
}

#[test]
fn smoothed_meshes_stay_valid() {
    let mut failures = 0;
    for generator in ["delaunay", "voronoi", "cutcell"] {
        failures += run_matrix(generator, true);
    }
    assert_eq!(failures, 0, "{failures} smoothed case(s) failed validation");
}
