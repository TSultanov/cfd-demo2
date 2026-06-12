//! Ghia lid-driven cavity benchmark: external literature validation of the
//! incompressible model (the counterpart of the de Vahl Davis Nusselt
//! benchmark for the buoyant model).
//!
//! Steady lid-driven cavity at Re = 100 (64x64), Re = 400 and Re = 1000
//! (128x128), SOU convection (the production scheme), marched to steady;
//! centerline velocity profiles compared against Ghia, Ghia & Shin (1982),
//! Table I/II — u_x through the vertical centerline (x = 0.5) and u_y
//! through the horizontal centerline (y = 0.5). Measured June 2026:
//! Re=100 max diff 0.0041/0.0087, Re=400 0.0021/0.0045, Re=1000
//! 0.0054/0.0105 (of lid speed) — all stations within ~1% of the
//! published values. Table provenance: cross-checked against the
//! canonical reproduction (ivan-pi gists of the 1982 tables), whose
//! Re=100 column matches this file's original values exactly; the known
//! Re=400 misprint at x=0.9063 is excluded.
//!
//! Why this exists (June 2026): the OpenFOAM lid reference is a
//! matched-discretization code-to-code regression case (20x20, first-order
//! upwind, shared mesh/scheme); its steady end state still sits ~4.5% (of
//! lid speed) from Ghia from discretization diffusion alone. This test is
//! the accuracy anchor against published values, decoupled from any other
//! solver's discretization choices.

#![cfg(feature = "dev-tests")]

use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType};
use cfd2::solver::model::helpers::{SolverFieldAliasesExt, SolverRuntimeParamsExt};
use cfd2::solver::model::incompressible_momentum_model;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};

const N: usize = 64;
const RE: f64 = 100.0;

/// Ghia, Ghia & Shin (1982), Re = 100: u_x through the vertical centerline.
/// Boundary-trivial endpoints (y = 0, y = 1) excluded.
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

/// Ghia, Ghia & Shin (1982), Re = 100: u_y through the horizontal centerline.
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

/// Ghia, Ghia & Shin (1982), Re = 400 — u_x through the vertical
/// centerline. Values cross-checked June 2026 against the canonical
/// reproduction (gist.github.com/ivan-pi/3e9326d18a366ffe6a8e5bfda6353219),
/// whose Re=100 column matches GHIA_UX above exactly.
const GHIA_UX_400: &[(f64, f64)] = &[
    (0.0547, -0.08186),
    (0.0625, -0.09266),
    (0.0703, -0.10338),
    (0.1016, -0.14612),
    (0.1719, -0.24299),
    (0.2813, -0.32726),
    (0.4531, -0.17119),
    (0.5000, -0.11477),
    (0.6172, 0.02135),
    (0.7344, 0.16256),
    (0.8516, 0.29093),
    (0.9531, 0.55892),
    (0.9609, 0.61756),
    (0.9688, 0.68439),
    (0.9766, 0.75837),
];

/// Ghia (1982), Re = 400 — u_y through the horizontal centerline.
/// The published x = 0.9063 entry (−0.23827) is a KNOWN MISPRINT (it
/// breaks monotonicity between the neighboring stations; flagged in the
/// reproduction source) and is excluded.
const GHIA_UY_400: &[(f64, f64)] = &[
    (0.0625, 0.18360),
    (0.0703, 0.19713),
    (0.0781, 0.20920),
    (0.0938, 0.22965),
    (0.1563, 0.28124),
    (0.2266, 0.30203),
    (0.2344, 0.30174),
    (0.5000, 0.05186),
    (0.8047, -0.38598),
    (0.8594, -0.44993),
    (0.9453, -0.22847),
    (0.9531, -0.19254),
    (0.9609, -0.15663),
    (0.9688, -0.12146),
];

/// Ghia (1982), Re = 1000 — u_x through the vertical centerline.
const GHIA_UX_1000: &[(f64, f64)] = &[
    (0.0547, -0.18109),
    (0.0625, -0.20196),
    (0.0703, -0.22220),
    (0.1016, -0.29730),
    (0.1719, -0.38289),
    (0.2813, -0.27805),
    (0.4531, -0.10648),
    (0.5000, -0.06080),
    (0.6172, 0.05702),
    (0.7344, 0.18719),
    (0.8516, 0.33304),
    (0.9531, 0.46604),
    (0.9609, 0.51117),
    (0.9688, 0.57492),
    (0.9766, 0.65928),
];

/// Ghia (1982), Re = 1000 — u_y through the horizontal centerline.
const GHIA_UY_1000: &[(f64, f64)] = &[
    (0.0625, 0.27485),
    (0.0703, 0.29012),
    (0.0781, 0.30353),
    (0.0938, 0.32627),
    (0.1563, 0.37095),
    (0.2266, 0.33075),
    (0.2344, 0.32235),
    (0.5000, 0.02526),
    (0.8047, -0.31966),
    (0.8594, -0.42665),
    (0.9063, -0.51500),
    (0.9453, -0.39188),
    (0.9531, -0.33714),
    (0.9609, -0.27669),
    (0.9688, -0.21388),
];

/// Linearly interpolate a (coord, value) profile (sorted by coord) at `c`,
/// with the no-slip/lid boundary values pinned at the ends.
fn interp(profile: &[(f64, f64)], c: f64, end0: f64, end1: f64) -> f64 {
    let mut pts: Vec<(f64, f64)> = Vec::with_capacity(profile.len() + 2);
    pts.push((0.0, end0));
    pts.extend_from_slice(profile);
    pts.push((1.0, end1));
    for w in pts.windows(2) {
        let (c0, v0) = w[0];
        let (c1, v1) = w[1];
        if c0 <= c && c <= c1 {
            return v0 + (v1 - v0) * (c - c0) / (c1 - c0);
        }
    }
    pts.last().unwrap().1
}

/// March a steady lid-driven cavity at the given Re and return the
/// centerline profiles: (y, u_x at x=0.5) and (x, u_y at y=0.5).
fn run_cavity(
    re: f64,
    n: usize,
    dt: f32,
    max_steps: usize,
) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
    let mesh = generate_structured_rect_mesh(
        n,
        n,
        1.0,
        1.0,
        BoundarySides {
            left: BoundaryType::Wall,
            right: BoundaryType::Wall,
            bottom: BoundaryType::Wall,
            top: BoundaryType::MovingWall,
        },
    );
    let mut solver = pollster::block_on(UnifiedSolver::new(
        &mesh,
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

    // Echo the configuration so a parameter-threading no-op can never
    // silently masquerade as a result (a refactor once ran "Re=400" with
    // the Re=100 viscosity; the field data exposed it).
    println!("[ghia] config: Re={re} n={n} dt={dt} nu={}", 1.0 / re);
    solver.set_dt(dt);
    solver.set_dtau(0.0).unwrap();
    solver.set_density(1.0).unwrap();
    solver.set_viscosity((1.0 / re) as f32).unwrap();
    solver
        .set_boundary_vec2(GpuBoundaryType::MovingWall, "U", [1.0, 0.0])
        .unwrap();
    solver.set_alpha_u(0.7).unwrap();
    solver.set_alpha_p(0.3).unwrap();
    solver.set_outer_iters(5).unwrap();
    solver.set_u(&vec![(0.0, 0.0); mesh.num_cells()]);
    solver.set_p(&vec![0.0; mesh.num_cells()]);
    solver.initialize_history();

    // March to steady state (settling time grows with Re).
    const CHECK_EVERY: usize = 25;
    const STEADY_TOL: f64 = 1e-7;
    let max_steps = max_steps;
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
        if delta < STEADY_TOL {
            println!("[ghia] steady after {steps} steps (per-step delta {delta:.2e})");
            break;
        }
        assert!(
            steps < max_steps,
            "no steady state within {max_steps} steps (delta {delta:.2e})"
        );
    }
    let u = prev;

    // Centerline profiles from the two cell columns/rows adjacent to 0.5
    // (even n: centers at 0.5 ± h/2, averaged).
    let h = 1.0 / n as f64;
    let near = |c: f64| (c - 0.5).abs() < h * 0.75;
    let mut ux_prof: Vec<(f64, f64)> = Vec::new(); // (y, u_x at x=0.5)
    let mut uy_prof: Vec<(f64, f64)> = Vec::new(); // (x, u_y at y=0.5)
    {
        use std::collections::BTreeMap;
        let mut by_y: BTreeMap<i64, (f64, Vec<f64>)> = BTreeMap::new();
        let mut by_x: BTreeMap<i64, (f64, Vec<f64>)> = BTreeMap::new();
        for i in 0..mesh.num_cells() {
            let (x, y) = (mesh.cell_cx[i], mesh.cell_cy[i]);
            if near(x) {
                let e = by_y.entry((y / h).round() as i64).or_insert((y, vec![]));
                e.1.push(u[i].0);
            }
            if near(y) {
                let e = by_x.entry((x / h).round() as i64).or_insert((x, vec![]));
                e.1.push(u[i].1);
            }
        }
        for (_, (y, vs)) in by_y {
            ux_prof.push((y, vs.iter().sum::<f64>() / vs.len() as f64));
        }
        for (_, (x, vs)) in by_x {
            uy_prof.push((x, vs.iter().sum::<f64>() / vs.len() as f64));
        }
    }
    (ux_prof, uy_prof)
}

/// Compare measured centerline profiles against a Ghia table; returns the
/// max absolute deviations (lid speed = 1).
fn compare_to_ghia(
    label: &str,
    ux_prof: &[(f64, f64)],
    uy_prof: &[(f64, f64)],
    ghia_ux: &[(f64, f64)],
    ghia_uy: &[(f64, f64)],
) -> (f64, f64) {
    let mut max_ux = 0.0f64;
    for &(y, g) in ghia_ux {
        // u_x(x=0.5): 0 at the no-slip bottom, 1 at the lid.
        let s = interp(ux_prof, y, 0.0, 1.0);
        let d = (s - g).abs();
        max_ux = max_ux.max(d);
        println!("[ghia][{label}] u_x  y={y:6.4} cfd2={s:8.5} ghia={g:8.5} diff={:+.5}", s - g);
    }
    let mut max_uy = 0.0f64;
    for &(x, g) in ghia_uy {
        // u_y(y=0.5): 0 at both no-slip side walls.
        let s = interp(uy_prof, x, 0.0, 0.0);
        let d = (s - g).abs();
        max_uy = max_uy.max(d);
        println!("[ghia][{label}] u_y  x={x:6.4} cfd2={s:8.5} ghia={g:8.5} diff={:+.5}", s - g);
    }
    println!("[ghia][{label}] max|u_x diff|={max_ux:.5} max|u_y diff|={max_uy:.5} (lid speed = 1)");
    (max_ux, max_uy)
}

#[test]
#[ignore = "external-physics benchmark (~minutes); run explicitly like the OpenFOAM reference suite"]
fn ghia_re100_centerline_profiles() {
    std::env::set_var("CFD2_QUIET", "1");
    let (ux_prof, uy_prof) = run_cavity(RE, N, 0.02, 4000);
    let (max_ux, max_uy) = compare_to_ghia("Re100", &ux_prof, &uy_prof, GHIA_UX, GHIA_UY);
    // Measured June 2026 (first run, machine-steady at delta 3.6e-14 after
    // 975 steps): max|u_x diff| = 0.0041, max|u_y diff| = 0.0087 — every
    // station within 0.9% of lid speed, most under 0.4%. Bands at ~1.8x
    // measured; ratchet-only thereafter.
    assert!(
        max_ux < 0.008,
        "u_x centerline deviates {max_ux:.4} from Ghia Re=100"
    );
    assert!(
        max_uy < 0.015,
        "u_y centerline deviates {max_uy:.4} from Ghia Re=100"
    );
}
#[test]
#[ignore = "external-physics benchmark; 128^2, several minutes; run explicitly"]
fn ghia_re400_centerline_profiles() {
    std::env::set_var("CFD2_QUIET", "1");
    let (ux_prof, uy_prof) = run_cavity(400.0, 128, 0.02, 8000);
    let (max_ux, max_uy) = compare_to_ghia("Re400", &ux_prof, &uy_prof, GHIA_UX_400, GHIA_UY_400);
    // Measured June 2026 (first true-Re run, steady at 2100 steps/t=42,
    // ~22 min wall): max|u_x diff| = 0.0021, max|u_y diff| = 0.0045 —
    // vortex minimum within 0.06% of Ghia. Bands at ~2x measured;
    // ratchet-only thereafter.
    assert!(max_ux < 0.005, "u_x deviates {max_ux:.4} from Ghia Re=400");
    assert!(max_uy < 0.009, "u_y deviates {max_uy:.4} from Ghia Re=400");
}

#[test]
#[ignore = "external-physics benchmark; 128^2, settling ~t=60+; tens of minutes; run explicitly"]
fn ghia_re1000_centerline_profiles() {
    std::env::set_var("CFD2_QUIET", "1");
    let (ux_prof, uy_prof) = run_cavity(1000.0, 128, 0.02, 12000);
    let (max_ux, max_uy) =
        compare_to_ghia("Re1000", &ux_prof, &uy_prof, GHIA_UX_1000, GHIA_UY_1000);
    // Measured June 2026 (first true-Re run, steady at 4175 steps/t=83.5,
    // ~40 min wall): max|u_x diff| = 0.0054, max|u_y diff| = 0.0105 —
    // within ~1% of lid speed at every station including the
    // secondary-vortex region. Bands at ~2x measured; ratchet-only.
    assert!(max_ux < 0.011, "u_x deviates {max_ux:.4} from Ghia Re=1000");
    assert!(max_uy < 0.020, "u_y deviates {max_uy:.4} from Ghia Re=1000");
}
