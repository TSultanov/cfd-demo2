//! Momentum MMS: convergence-order tests for the coupled incompressible
//! saddle-point path (derived Rhie–Chow flux + Schur preconditioner).
//!
//! Manufactured solution: forced steady Taylor–Green.
//!
//!   U*(x, y) = ( sin(pi x) cos(pi y), -cos(pi x) sin(pi y) )   (div-free)
//!   p*(x, y) = (rho/4) (cos(2 pi x) + cos(2 pi y))
//!
//! With this pressure the convective term balances the pressure gradient
//! exactly: for this phase convention (U·grad)U = (pi/2)(sin 2pi x, sin 2pi y)
//! = -grad(p*)/rho, so the steady momentum source is purely viscous:
//!
//!   S = -mu lap(U*) = 2 mu pi^2 U*
//!
//! (The solver itself flagged an earlier sign error here: its converged
//! pressure disagreed with a wrongly-signed manufactured p* by exactly 2x
//! the field amplitude.)
//!
//! The source enters as one more declared equation term on the `_mms` model
//! variant (a Vector2 source field read per component — the first consumer
//! of per-component explicit sources). Continuity needs no source: U* is
//! divergence-free, so the pressure-row residual is pure discretization
//! error.
//!
//! Pressure is gauge-free on the all-wall mesh (zero-gradient everywhere),
//! so p is compared after demeaning both fields.
#![cfg(feature = "dev-tests")]

mod mms_support;

use std::f64::consts::PI;

use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::mesh::{
    generate_graded_rect_mesh, generate_structured_rect_mesh, AxisGrading, BoundarySides, Mesh,
};
use cfd2::solver::model::helpers::{SolverFieldAliasesExt, SolverRuntimeParamsExt};
use cfd2::solver::model::{incompressible_momentum_mms_model, INCOMPRESSIBLE_MMS_SOURCE_FIELD};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};

use mms_support::{
    assert_convergence_order, field_errors, field_errors_vec2, max_cell_extent,
    run_to_steady_vec2,
};

const MU: f64 = 1.0;
const RHO: f64 = 1.0;
const STEADY_TOL: f64 = 5e-6;
// The viscous-dominated transient settles in ~12 steps at every level
// (verified on the first run); the cap is headroom, not budget.
const STEADY_MAX_STEPS: usize = 60;

fn exact_u(x: f64, y: f64) -> (f64, f64) {
    ((PI * x).sin() * (PI * y).cos(), -(PI * x).cos() * (PI * y).sin())
}

fn exact_p(x: f64, y: f64) -> f64 {
    (RHO / 4.0) * ((2.0 * PI * x).cos() + (2.0 * PI * y).cos())
}

fn source(x: f64, y: f64) -> (f64, f64) {
    let (ux, uy) = exact_u(x, y);
    (2.0 * MU * PI * PI * ux, 2.0 * MU * PI * PI * uy)
}

fn solve_steady_taylor_green(n: usize, advection_scheme: Scheme) -> (Mesh, Vec<(f64, f64)>, Vec<f64>) {
    let mesh = generate_structured_rect_mesh(n, n, 1.0, 1.0, BoundarySides::wall());
    solve_steady_taylor_green_on_mesh(mesh, advection_scheme)
}

fn solve_steady_taylor_green_on_mesh(
    mesh: Mesh,
    advection_scheme: Scheme,
) -> (Mesh, Vec<(f64, f64)>, Vec<f64>) {
    let model = incompressible_momentum_mms_model().expect("model");
    let mut solver = pollster::block_on(UnifiedSolver::new(
        &mesh,
        model,
        SolverConfig {
            advection_scheme,
            time_scheme: TimeScheme::BDF2,
            preconditioner: PreconditionerType::Jacobi,
            stepping: SteppingMode::Coupled,
        },
        None,
        None,
    ))
    .expect("solver init");

    solver.set_dt(0.05);
    solver.set_dtau(0.0).expect("dtau");
    solver.set_density(RHO as f32).expect("density");
    solver.set_viscosity(MU as f32).expect("viscosity");
    solver.set_alpha_u(0.7).expect("alpha_u");
    solver.set_alpha_p(0.3).expect("alpha_p");
    // 25 outer correctors: the linear, viscous-dominated problem converges
    // well below discretization error; 50 (the lid-case setting) doubles the
    // runtime for no measurable error change.
    solver.set_outer_iters(25).expect("outer_iters");

    // Per-face Dirichlet U on all walls from the exact solution.
    let fx = mesh.face_cx.clone();
    let fy = mesh.face_cy.clone();
    let wall_u = move |c: usize| {
        let fx = fx.clone();
        let fy = fy.clone();
        move |face_idx: u32| {
            let i = face_idx as usize;
            let (ux, uy) = exact_u(fx[i], fy[i]);
            (if c == 0 { ux } else { uy }) as f32
        }
    };
    solver
        .set_boundary_values_per_face(GpuBoundaryType::Wall, "U", 0, &wall_u(0))
        .expect("wall u_x");
    solver
        .set_boundary_values_per_face(GpuBoundaryType::Wall, "U", 1, &wall_u(1))
        .expect("wall u_y");

    // Manufactured momentum source, per cell.
    let src: Vec<(f64, f64)> = (0..mesh.num_cells())
        .map(|i| source(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    solver
        .set_field_vec2(INCOMPRESSIBLE_MMS_SOURCE_FIELD, &src)
        .expect("upload mms source");

    solver.set_u(&vec![(0.0, 0.0); mesh.num_cells()]);
    solver.set_p(&vec![0.0; mesh.num_cells()]);
    solver.initialize_history();

    let u = run_to_steady_vec2(&mut solver, "U", STEADY_MAX_STEPS, STEADY_TOL);
    let p = pollster::block_on(solver.get_p());
    (mesh, u, p)
}

/// Volume-weighted mean of a scalar field (for pressure-gauge removal).
fn volume_mean(mesh: &Mesh, f: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut vol = 0.0;
    for i in 0..mesh.num_cells() {
        sum += mesh.cell_vol[i] * f[i];
        vol += mesh.cell_vol[i];
    }
    sum / vol
}

/// Run levels and collect (h, u_l2, demeaned p_l2) triples.
fn convergence_table(levels: &[usize], scheme: Scheme) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut hs = Vec::new();
    let mut u_errs = Vec::new();
    let mut p_errs = Vec::new();
    for &n in levels {
        let (mesh, u, p) = solve_steady_taylor_green(n, scheme);
        let u_err = field_errors_vec2(&mesh, &u, exact_u).l2;

        // Demean both pressures (all-wall mesh leaves the gauge free).
        let p_mean = volume_mean(&mesh, &p);
        let exact_mean = {
            let exact: Vec<f64> = (0..mesh.num_cells())
                .map(|i| exact_p(mesh.cell_cx[i], mesh.cell_cy[i]))
                .collect();
            volume_mean(&mesh, &exact)
        };
        let p_err = field_errors(&mesh, &p, |x, y| exact_p(x, y) - exact_mean + p_mean).l2;

        println!("[mms][taylor_green:{scheme:?}] n={n} u_l2={u_err:.4e} p_l2={p_err:.4e}");
        hs.push(1.0 / n as f64);
        u_errs.push(u_err);
        p_errs.push(p_err);
    }
    (hs, u_errs, p_errs)
}

/// Second-order-upwind convection: both the viscous and convective truncation
/// are O(h^2), so U converges at second order through the full coupled
/// saddle-point path (derived Rhie–Chow flux + Schur). First MMS coverage of
/// momentum+pressure. Pressure order is pinned at what the saddle point
/// currently delivers (>= ~1) so numerics changes are refereed.
#[test]
fn steady_taylor_green_sou_velocity_second_order() {
    let (hs, u_errs, p_errs) = convergence_table(&[8, 16, 32, 64], Scheme::SecondOrderUpwind);
    assert_convergence_order("taylor_green_sou_u", &hs, &u_errs, 2.0, 0.35, 1.0e-3);
    let p_order = mms_support::fit_order(&hs, &p_errs);
    println!("[mms][taylor_green_sou] pressure order {p_order:.3}");
    assert!(
        p_order > 0.9,
        "pressure order regressed: {p_order:.3} (errors {p_errs:?})"
    );
}

/// The SOU Taylor-Green study repeated on a two-sided geometrically graded
/// mesh (Arc M generality gate): smallest cells at every wall, center/wall
/// ratio 4 on both axes, orders fitted against the MAX cell extent (see
/// `max_cell_extent`). This drives the full coupled saddle-point path —
/// distance-weighted assembly coefficients, derived Rhie-Chow flux, and
/// gradient reconstruction — on non-uniform spacing; velocity must hold
/// second order and pressure must not regress below the uniform-mesh floor.
#[test]
fn steady_taylor_green_sou_graded_second_order() {
    let grading = AxisGrading::TwoSided { ratio: 4.0 };
    let mut hs = Vec::new();
    let mut u_errs = Vec::new();
    let mut p_errs = Vec::new();
    for n in [8usize, 16, 32, 64] {
        let mesh =
            generate_graded_rect_mesh(n, n, 1.0, 1.0, grading, grading, BoundarySides::wall());
        let h_eff = max_cell_extent(&mesh);
        let (mesh, u, p) = solve_steady_taylor_green_on_mesh(mesh, Scheme::SecondOrderUpwind);
        let u_err = field_errors_vec2(&mesh, &u, exact_u).l2;
        let p_mean = volume_mean(&mesh, &p);
        let exact_mean = {
            let exact: Vec<f64> = (0..mesh.num_cells())
                .map(|i| exact_p(mesh.cell_cx[i], mesh.cell_cy[i]))
                .collect();
            volume_mean(&mesh, &exact)
        };
        let p_err = field_errors(&mesh, &p, |x, y| exact_p(x, y) - exact_mean + p_mean).l2;
        println!("[mms][taylor_green_graded] n={n} h_eff={h_eff:.4e} u_l2={u_err:.4e} p_l2={p_err:.4e}");
        hs.push(h_eff);
        u_errs.push(u_err);
        p_errs.push(p_err);
    }
    // Measured June 2026 (first run): u order 1.978 (uniform study: 1.87),
    // finest u_l2 3.28e-4 at h_eff 0.0287; p order 1.678 (uniform: 1.69).
    // Graded errors sit BELOW the uniform line at matched h_eff. Cap ~2x.
    assert_convergence_order("taylor_green_sou_graded_u", &hs, &u_errs, 2.0, 0.35, 7.0e-4);
    let p_order = mms_support::fit_order(&hs, &p_errs);
    println!("[mms][taylor_green_sou_graded] pressure order {p_order:.3}");
    assert!(
        p_order > 0.9,
        "pressure order regressed: {p_order:.3} (errors {p_errs:?})"
    );
}

/// Upwind convection: the convective truncation is O(h) and overtakes the
/// O(h^2) viscous error under refinement, so the fitted order is a mix
/// trending toward 1 (observed ~1.3 over 8..32 on the first run). Assert
/// monotone convergence at first order or better with an absolute cap.
#[test]
fn steady_taylor_green_upwind_converges() {
    let (hs, u_errs, _p_errs) = convergence_table(&[8, 16, 32], Scheme::Upwind);
    assert_convergence_order("taylor_green_upwind_u", &hs, &u_errs, 1.0, 0.2, 3.0e-3);
}
