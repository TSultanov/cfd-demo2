//! Repeated-run stress test for the randomized unstructured generators.
//! The Poisson sampling uses an unseeded RNG, so defects are probabilistic —
//! run each configuration many times and count panics / invariant violations.
//!
//! ```sh
//! cargo test --features meshgen --test meshgen_stress --release -- --nocapture
//! ```

#![cfg(feature = "meshgen")]

use cfd2::solver::mesh::{
    generate_delaunay_mesh, generate_voronoi_mesh, BackwardsStep, ChannelWithObstacle, Geometry,
    Nozzle,
};
use nalgebra::{Point2, Vector2};
use std::panic::AssertUnwindSafe;

fn stress_one(generator: &str, name: &str, geo: &(impl Geometry + Sync), domain: Vector2<f64>) {
    let runs = 8;
    for &(hmin, hmax) in &[(0.025, 0.025), (0.0125, 0.05)] {
        let mut panics = 0;
        let mut bad = 0;
        let mut msg = String::new();
        for _ in 0..runs {
            let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                let mut mesh = match generator {
                    "delaunay" => generate_delaunay_mesh(geo, hmin, hmax, 1.2, domain),
                    "voronoi" => generate_voronoi_mesh(geo, hmin, hmax, 1.2, domain),
                    _ => unreachable!(),
                };
                mesh.smooth(geo, 0.3, 50);
                mesh
            }));
            match result {
                Err(e) => {
                    panics += 1;
                    msg = e
                        .downcast_ref::<&str>()
                        .map(|s| s.to_string())
                        .or_else(|| e.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "<non-string>".to_string());
                }
                Ok(mesh) => {
                    // Cheap invariants: finite geometry + positive volumes + closure
                    let mut ok = true;
                    for arr in [&mesh.face_nx, &mesh.face_ny, &mesh.cell_vol] {
                        if arr.iter().any(|v| !v.is_finite()) {
                            ok = false;
                        }
                    }
                    if mesh.cell_vol.iter().any(|&v| !(v > 0.0)) {
                        ok = false;
                    }
                    for c in 0..mesh.num_cells() {
                        let s = mesh.cell_face_offsets[c];
                        let e = mesh.cell_face_offsets[c + 1];
                        let mut sx = 0.0;
                        let mut sy = 0.0;
                        let mut perim = 0.0;
                        for &f in &mesh.cell_faces[s..e] {
                            let sign = if mesh.face_owner[f] == c { 1.0 } else { -1.0 };
                            sx += sign * mesh.face_area[f] * mesh.face_nx[f];
                            sy += sign * mesh.face_area[f] * mesh.face_ny[f];
                            perim += mesh.face_area[f];
                        }
                        if perim > 0.0 && (sx * sx + sy * sy).sqrt() / perim > 1e-6 {
                            ok = false;
                            break;
                        }
                    }
                    if !ok {
                        bad += 1;
                    }
                }
            }
        }
        println!(
            "[{generator}/{name}/h={hmin}/{hmax}] runs={runs} panics={panics} invalid={bad}{}",
            if panics > 0 { format!(" last_panic={msg}") } else { String::new() }
        );
    }
}

#[test]
fn stress_voronoi_and_delaunay() {
    for generator in ["voronoi", "delaunay"] {
        stress_one(
            generator,
            "rect_channel",
            &RectangularChannelWrap { length: 3.0, height: 1.0 },
            Vector2::new(3.0, 1.0),
        );
        stress_one(
            generator,
            "obstacle",
            &ChannelWithObstacle {
                length: 3.0,
                height: 1.0,
                obstacle_center: Point2::new(1.0, 0.51),
                obstacle_radius: 0.1,
            },
            Vector2::new(3.0, 1.0),
        );
        stress_one(
            generator,
            "backstep",
            &BackwardsStep {
                length: 3.5,
                height_inlet: 0.5,
                height_outlet: 1.0,
                step_x: 0.5,
            },
            Vector2::new(3.5, 1.0),
        );
        stress_one(
            generator,
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
    }
}

// RectangularChannel is exported from meshgen; alias to keep imports tidy.
use cfd2::solver::mesh::RectangularChannel as RectangularChannelWrap;
