//! Solver-level gates that the solver produces correct physics (not just
//! valid geometry) on meshless/CVT meshes, both on the Re = 100 lid-driven
//! cavity (Ghia et al. 1982):
//!
//! - `meshless_cvt_lid_smoke` (always-run): coarse CVT mesh vs the
//!   incumbent Voronoi+smooth mesh, both marched to steady; the two steady
//!   centerline profiles must agree to a few % of lid speed and stay finite.
//! - `ghia_cvt_vs_incumbent_voronoi` (#[ignore]): Ghia centerline error on
//!   the CVT mesh ≤ incumbent-Voronoi error × 1.05.
//!
//! Both meshes are unstructured, so profiles use an UNSTRUCTURED centerline
//! sampler (`tensor_grid` breaks on Voronoi vertices): inverse-distance-
//! squared weighting over the k nearest cell centroids, identical for both
//! meshes so sampling bias cancels in the comparison.
//!
//! ```sh
//! cargo test --release --features "dev-tests meshgen" \
//!     --test meshless_solver_gate -- --nocapture
//! cargo test --release --features "dev-tests meshgen" \
//!     --test meshless_solver_gate -- --ignored --nocapture
//! ```

#![cfg(all(feature = "dev-tests", feature = "meshgen"))]

use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::mesh::{
    generate_cvt_mesh, generate_voronoi_mesh, BoundaryType, LloydConfig, Mesh,
    RectangularChannel,
};
use cfd2::solver::model::helpers::{SolverFieldAliasesExt, SolverRuntimeParamsExt};
use cfd2::solver::model::incompressible_momentum_model;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};
use nalgebra::Vector2;

const RE: f64 = 100.0;

/// Ghia, Ghia & Shin (1982), Re = 100: u_x through the vertical centerline.
const GHIA_UX: &[(f64, f64)] = &[
    (0.0547, -0.03717),
    (0.0625, -0.04192),
    (0.0703, -0.04775),
    (0.1016, -0.06434),
    (0.1719, -0.10150),
    (0.2813, -0.15662),
    (0.4531, -0.21090),
    (0.5000, -0.20581),
    (0.6172, -0.13641),
    (0.7344, 0.00332),
    (0.8516, 0.23151),
    (0.9531, 0.68717),
    (0.9609, 0.73722),
    (0.9688, 0.78871),
    (0.9766, 0.84123),
];

/// Ghia (1982), Re = 100: u_y through the horizontal centerline.
const GHIA_UY: &[(f64, f64)] = &[
    (0.0625, 0.09233),
    (0.0703, 0.10091),
    (0.0781, 0.10890),
    (0.0938, 0.12317),
    (0.1563, 0.16077),
    (0.2266, 0.17507),
    (0.2344, 0.17527),
    (0.5000, 0.05454),
    (0.8047, -0.24533),
    (0.8594, -0.22445),
    (0.9063, -0.16914),
    (0.9453, -0.10313),
    (0.9531, -0.08864),
    (0.9609, -0.07391),
    (0.9688, -0.05906),
];

/// Retag the unit-box boundary for the lid cavity: every boundary face is a
/// `Wall` except the top (y = 1), which drives as `MovingWall`. The meshless
/// and incumbent generators both tag the box via `classify_boundary`
/// (left = Inlet, right = Outlet), so retagging is required for BOTH.
fn retag_lid(mesh: &mut Mesh) {
    for f in 0..mesh.num_faces() {
        if mesh.face_boundary[f].is_some() {
            let bt = if mesh.face_cy[f] > 1.0 - 1e-6 {
                BoundaryType::MovingWall
            } else {
                BoundaryType::Wall
            };
            mesh.face_boundary[f] = Some(bt);
        }
    }
}

fn cvt_lid_mesh(h: f64) -> Mesh {
    let geo = RectangularChannel {
        length: 1.0,
        height: 1.0,
    };
    let mut mesh = generate_cvt_mesh(
        &geo,
        h,
        h,
        1.0,
        Vector2::new(1.0, 1.0),
        &LloydConfig::default(),
    );
    // NO Mesh::smooth — vertex smoothing would move Voronoi vertices off
    // the bisectors (the GUI's VoronoiCvt arm skips smoothing too).
    retag_lid(&mut mesh);
    mesh
}

fn incumbent_lid_mesh(h: f64) -> Mesh {
    let geo = RectangularChannel {
        length: 1.0,
        height: 1.0,
    };
    let mut mesh = generate_voronoi_mesh(&geo, h, h, 1.0, Vector2::new(1.0, 1.0));
    // GUI parity: unstructured meshes get smooth(0.3, 50).
    mesh.smooth(&geo, 0.3, 50);
    retag_lid(&mut mesh);
    mesh
}

/// March a lid cavity to steady on the given mesh; returns the steady cell
/// velocities. SOU + BDF2 + coupled — the production configuration.
fn run_lid_to_steady(mesh: &Mesh, label: &str, max_steps: usize) -> Vec<(f64, f64)> {
    let mut solver = pollster::block_on(UnifiedSolver::new(
        mesh,
        incompressible_momentum_model().expect("model"),
        SolverConfig {
            advection_scheme: Scheme::SecondOrderUpwind,
            time_scheme: TimeScheme::BDF2,
            preconditioner: PreconditionerType::Jacobi,
            stepping: SteppingMode::Coupled,
        },
        None,
        None,
    ))
    .expect("solver init");
    println!(
        "[meshless-solver][{label}] config: Re={RE} cells={} nu={}",
        mesh.num_cells(),
        1.0 / RE
    );
    solver.set_dt(0.02);
    solver.set_dtau(0.0).unwrap();
    solver.set_density(1.0).unwrap();
    solver.set_viscosity((1.0 / RE) as f32).unwrap();
    solver
        .set_boundary_vec2(GpuBoundaryType::MovingWall, "U", [1.0, 0.0])
        .unwrap();
    solver.set_alpha_u(0.7).unwrap();
    solver.set_alpha_p(0.3).unwrap();
    solver.set_outer_iters(5).unwrap();
    solver.set_u(&vec![(0.0, 0.0); mesh.num_cells()]);
    solver.set_p(&vec![0.0; mesh.num_cells()]);
    solver.initialize_history();

    const CHECK_EVERY: usize = 25;
    const STEADY_TOL: f64 = 1e-7;
    let mut prev = pollster::block_on(solver.get_u());
    let mut steps = 0usize;
    loop {
        for _ in 0..CHECK_EVERY {
            solver.step();
        }
        steps += CHECK_EVERY;
        let cur = pollster::block_on(solver.get_u());
        let delta = cur
            .iter()
            .zip(&prev)
            .map(|(a, b)| (a.0 - b.0).abs().max((a.1 - b.1).abs()))
            .fold(0.0f64, f64::max)
            / CHECK_EVERY as f64;
        prev = cur;
        assert!(
            delta.is_finite(),
            "[{label}] solver diverged (non-finite delta) after {steps} steps"
        );
        if delta < STEADY_TOL {
            println!("[meshless-solver][{label}] steady after {steps} steps (per-step delta {delta:.2e})");
            break;
        }
        assert!(
            steps < max_steps,
            "[{label}] no steady state within {max_steps} steps (delta {delta:.2e})"
        );
    }
    assert!(
        prev.iter().all(|v| v.0.is_finite() && v.1.is_finite()),
        "[{label}] non-finite steady velocity"
    );
    prev
}

/// Unstructured point sampler: inverse-distance-squared weighting over the
/// `k` nearest cell centroids (exact when a centroid coincides with the
/// sample point). O(n) per sample — fine at gate sizes.
fn sample_idw(mesh: &Mesh, vals: &[f64], x: f64, y: f64, k: usize) -> f64 {
    let mut near: Vec<(f64, usize)> = (0..mesh.num_cells())
        .map(|i| {
            let dx = mesh.cell_cx[i] - x;
            let dy = mesh.cell_cy[i] - y;
            (dx * dx + dy * dy, i)
        })
        .collect();
    near.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let eps2 = 1e-24;
    let mut wsum = 0.0;
    let mut vsum = 0.0;
    for &(d2, i) in near.iter().take(k) {
        let w = 1.0 / (d2 + eps2);
        wsum += w;
        vsum += w * vals[i];
    }
    vsum / wsum
}

/// Centerline profiles via the unstructured sampler: u_x at (0.5, y) for the
/// GHIA_UX stations, u_y at (x, 0.5) for the GHIA_UY stations.
fn centerline_profiles(mesh: &Mesh, u: &[(f64, f64)]) -> (Vec<f64>, Vec<f64>) {
    let ux: Vec<f64> = u.iter().map(|v| v.0).collect();
    let uy: Vec<f64> = u.iter().map(|v| v.1).collect();
    let prof_ux = GHIA_UX
        .iter()
        .map(|&(y, _)| sample_idw(mesh, &ux, 0.5, y, 4))
        .collect();
    let prof_uy = GHIA_UY
        .iter()
        .map(|&(x, _)| sample_idw(mesh, &uy, x, 0.5, 4))
        .collect();
    (prof_ux, prof_uy)
}

/// Max absolute deviation from the Ghia tables (lid speed = 1).
fn ghia_errors(label: &str, prof_ux: &[f64], prof_uy: &[f64]) -> (f64, f64) {
    let mut max_ux = 0.0f64;
    for (&(y, g), &s) in GHIA_UX.iter().zip(prof_ux) {
        println!("[meshless-solver][{label}] u_x  y={y:6.4} cfd2={s:8.5} ghia={g:8.5} diff={:+.5}", s - g);
        max_ux = max_ux.max((s - g).abs());
    }
    let mut max_uy = 0.0f64;
    for (&(x, g), &s) in GHIA_UY.iter().zip(prof_uy) {
        println!("[meshless-solver][{label}] u_y  x={x:6.4} cfd2={s:8.5} ghia={g:8.5} diff={:+.5}", s - g);
        max_uy = max_uy.max((s - g).abs());
    }
    println!("[meshless-solver][{label}] max|u_x diff|={max_ux:.5} max|u_y diff|={max_uy:.5}");
    (max_ux, max_uy)
}

/// Always-run smoke: the solver must run the lid case on a meshless CVT mesh
/// and land on the same steady flow as the incumbent-Voronoi mesh.
#[test]
fn meshless_cvt_lid_smoke() {
    std::env::set_var("CFD2_QUIET", "1");
    let h = 1.0 / 24.0;
    let cvt = cvt_lid_mesh(h);
    let inc = incumbent_lid_mesh(h);
    let u_cvt = run_lid_to_steady(&cvt, "smoke-cvt", 4000);
    let u_inc = run_lid_to_steady(&inc, "smoke-incumbent", 4000);
    let (px_cvt, py_cvt) = centerline_profiles(&cvt, &u_cvt);
    let (px_inc, py_inc) = centerline_profiles(&inc, &u_inc);
    let dx = px_cvt
        .iter()
        .zip(&px_inc)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    let dy = py_cvt
        .iter()
        .zip(&py_inc)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    println!("[meshless-solver][smoke] CVT-vs-incumbent centerline: max|du_x|={dx:.5} max|du_y|={dy:.5}");
    // Loose bands: this smoke catches divergence/garbage on meshless
    // topology, not accuracy (that gate is `ghia_cvt_vs_incumbent_voronoi`).
    assert!(dx < 0.08, "CVT steady u_x deviates {dx:.4} from incumbent-mesh run");
    assert!(dy < 0.045, "CVT steady u_y deviates {dy:.4} from incumbent-mesh run");
}

/// Solver gate: Ghia centerline error on the CVT mesh must not exceed the
/// incumbent-Voronoi mesh's error by more than 5%. Explicit-run like the
/// whole Ghia family (external-physics benchmark, minutes).
#[test]
#[ignore = "external-physics benchmark (~minutes); run explicitly like the Ghia suite"]
fn ghia_cvt_vs_incumbent_voronoi() {
    std::env::set_var("CFD2_QUIET", "1");
    let h = 1.0 / 64.0;
    let cvt = cvt_lid_mesh(h);
    let inc = incumbent_lid_mesh(h);
    let u_cvt = run_lid_to_steady(&cvt, "ghia-cvt", 8000);
    let u_inc = run_lid_to_steady(&inc, "ghia-incumbent", 8000);
    let (px_cvt, py_cvt) = centerline_profiles(&cvt, &u_cvt);
    let (px_inc, py_inc) = centerline_profiles(&inc, &u_inc);
    let (ex_cvt, ey_cvt) = ghia_errors("ghia-cvt", &px_cvt, &py_cvt);
    let (ex_inc, ey_inc) = ghia_errors("ghia-incumbent", &px_inc, &py_inc);
    println!(
        "[meshless-solver][gate] CVT err ({ex_cvt:.5}, {ey_cvt:.5}) vs incumbent ({ex_inc:.5}, {ey_inc:.5})"
    );
    // Gate on the combined metric max(u_x err, u_y err), not per-component:
    // the smaller component sits at the IDW sampler's O(h) noise floor.
    let err_cvt = ex_cvt.max(ey_cvt);
    let err_inc = ex_inc.max(ey_inc);
    assert!(
        err_cvt <= err_inc * 1.05 + 1e-4,
        "CVT Ghia error {err_cvt:.5} exceeds incumbent {err_inc:.5} x 1.05"
    );
}
