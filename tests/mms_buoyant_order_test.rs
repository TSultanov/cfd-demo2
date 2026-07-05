//! Coupled MMS for the buoyant (Boussinesq) model: manufactured
//! (U, p, T) with the buoyancy feedback ACTIVE, solved through the full
//! coupled path (derived Rhie–Chow flux, Schur preconditioner, bounded
//! convection, directional buoyancy source, T advected by the solved flux).
//!
//! Velocity/pressure: forced steady Taylor–Green as in
//! `mms_incompressible_order_test` (convection balanced by p*).
//! Temperature: T* = cos(pi x) cos(pi y) — satisfies the model's adiabatic
//! (zero-gradient) walls at y = 0,1 and varying isothermal Dirichlet values
//! on the left (hot, Inlet type) and right (cold, Outlet type) boundaries.
//!
//! Manufactured sources (equation terms sum to zero; explicit sources enter
//! the residual negatively):
//!
//!   S_U = -mu lap(U*) - f_buoy(T*)
//!       = 2 mu pi^2 U*  -  rho beta_g (T* - T0) (0, 1)
//!   S_T = rho (U* . grad T*) + (k/cp) 2 pi^2 T*
//!
//! The buoyancy force appears in S_U with opposite sign so the manufactured
//! Taylor–Green stays the exact solution WITH the coupling term exercised.
#![cfg(feature = "dev-tests")]

mod mms_support;

use std::f64::consts::PI;

use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType, Mesh};
use cfd2::solver::model::helpers::{SolverFieldAliasesExt, SolverRuntimeParamsExt};
use cfd2::solver::model::{
    buoyant_incompressible_mms_model, BUOYANT_BETA_G, BUOYANT_K_OVER_CP,
    BUOYANT_MMS_SOURCE_T_FIELD, BUOYANT_MMS_SOURCE_U_FIELD, BUOYANT_T0,
    BUOYANT_TEMPERATURE_FIELD,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};

use mms_support::{
    assert_convergence_order, field_errors, field_errors_vec2, run_to_steady_vec2_with_scalars,
};

const MU: f64 = 1.0;
const RHO: f64 = 1.0;
const STEADY_TOL: f64 = 5e-6;
const STEADY_MAX_STEPS: usize = 400;

fn exact_u(x: f64, y: f64) -> (f64, f64) {
    ((PI * x).sin() * (PI * y).cos(), -(PI * x).cos() * (PI * y).sin())
}

fn exact_p(x: f64, y: f64) -> f64 {
    (RHO / 4.0) * ((2.0 * PI * x).cos() + (2.0 * PI * y).cos())
}

fn exact_t(x: f64, y: f64) -> f64 {
    (PI * x).cos() * (PI * y).cos()
}

fn source_u(x: f64, y: f64) -> (f64, f64) {
    let (ux, uy) = exact_u(x, y);
    let visc = 2.0 * MU * PI * PI;
    // Buoyancy as assembled: f = rho*beta_g*(T - T0) along +y (gravity -y).
    let f_buoy_y = RHO * BUOYANT_BETA_G * (exact_t(x, y) - BUOYANT_T0);
    (visc * ux, visc * uy - f_buoy_y)
}

fn source_t(x: f64, y: f64) -> f64 {
    let (ux, uy) = exact_u(x, y);
    let dtdx = -PI * (PI * x).sin() * (PI * y).cos();
    let dtdy = -PI * (PI * x).cos() * (PI * y).sin();
    RHO * (ux * dtdx + uy * dtdy) + BUOYANT_K_OVER_CP * 2.0 * PI * PI * exact_t(x, y)
}

struct SteadySolution {
    mesh: Mesh,
    u: Vec<(f64, f64)>,
    p: Vec<f64>,
    t: Vec<f64>,
    grad_p: Vec<(f64, f64)>,
    d_p: Vec<f64>,
}

fn solve_steady(n: usize, scheme: Scheme) -> (Mesh, Vec<(f64, f64)>, Vec<f64>, Vec<f64>) {
    let s = solve_steady_with_right(n, scheme, BoundaryType::Outlet);
    (s.mesh, s.u, s.p, s.t)
}

fn solve_steady_with_right(n: usize, scheme: Scheme, right: BoundaryType) -> SteadySolution {
    // Left = hot isothermal (Inlet type), right = cold isothermal (Outlet
    // type), top/bottom = adiabatic walls; all four sides are no-slip for U.
    let mesh = generate_structured_rect_mesh(
        n,
        n,
        1.0,
        1.0,
        BoundarySides {
            left: BoundaryType::Inlet,
            right,
            bottom: BoundaryType::Wall,
            top: BoundaryType::Wall,
        },
    );
    let model = buoyant_incompressible_mms_model().expect("model");
    let mut solver = pollster::block_on(UnifiedSolver::new(
        &mesh,
        model,
        SolverConfig {
            advection_scheme: scheme,
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
    solver.set_outer_iters(25).expect("outer_iters");

    // Per-face exact Dirichlet U on every boundary type in play.
    let fx = mesh.face_cx.clone();
    let fy = mesh.face_cy.clone();
    let u_face = |c: usize, fx: &[f64], fy: &[f64]| {
        let fx = fx.to_vec();
        let fy = fy.to_vec();
        move |face_idx: u32| {
            let i = face_idx as usize;
            let (ux, uy) = exact_u(fx[i], fy[i]);
            (if c == 0 { ux } else { uy }) as f32
        }
    };
    for boundary in [
        GpuBoundaryType::Inlet,
        GpuBoundaryType::Outlet,
        GpuBoundaryType::Wall,
    ] {
        for c in 0..2usize {
            solver
                .set_boundary_values_per_face(boundary, "U", c as u32, &u_face(c, &fx, &fy))
                .expect("U bc");
        }
    }
    // Per-face exact Dirichlet T on the isothermal (left/right) boundaries.
    let t_face = {
        let fx = fx.clone();
        let fy = fy.clone();
        move |face_idx: u32| {
            let i = face_idx as usize;
            exact_t(fx[i], fy[i]) as f32
        }
    };
    for boundary in [GpuBoundaryType::Inlet, GpuBoundaryType::Outlet] {
        solver
            .set_boundary_values_per_face(boundary, BUOYANT_TEMPERATURE_FIELD, 0, &t_face)
            .expect("T bc");
    }

    // Outlet pressure pins the gauge: per-face exact Dirichlet p*. (With no
    // outlet faces in play the pressure system is pure-Neumann, the same
    // regime the all-wall momentum MMS exercises; the gauge is handled by
    // the de-meaned error metric.)
    if right == BoundaryType::Outlet {
        let p_face = {
            let fx = fx.clone();
            let fy = fy.clone();
            move |face_idx: u32| {
                let i = face_idx as usize;
                exact_p(fx[i], fy[i]) as f32
            }
        };
        solver
            .set_boundary_values_per_face(GpuBoundaryType::Outlet, "p", 0, &p_face)
            .expect("p bc");
    }

    // Manufactured sources.
    let src_u: Vec<(f64, f64)> = (0..mesh.num_cells())
        .map(|i| source_u(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    solver
        .set_field_vec2(BUOYANT_MMS_SOURCE_U_FIELD, &src_u)
        .expect("upload S_U");
    let src_t: Vec<f64> = (0..mesh.num_cells())
        .map(|i| source_t(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    solver
        .set_field_scalar(BUOYANT_MMS_SOURCE_T_FIELD, &src_t)
        .expect("upload S_T");

    solver.set_u(&vec![(0.0, 0.0); mesh.num_cells()]);
    solver.set_p(&vec![0.0; mesh.num_cells()]);
    solver
        .set_field_scalar(BUOYANT_TEMPERATURE_FIELD, &vec![0.0; mesh.num_cells()])
        .expect("init T");
    solver.initialize_history();

    // Watch BOTH U and T: T's transient is much slower than momentum's, and
    // a U-only criterion stops while T is still converging (the leftover
    // transient then reads as first-order error in the order study).
    let u = run_to_steady_vec2_with_scalars(
        &mut solver,
        "U",
        &[BUOYANT_TEMPERATURE_FIELD],
        STEADY_MAX_STEPS,
        STEADY_TOL,
    );
    let p = pollster::block_on(solver.get_p());
    let t = pollster::block_on(solver.get_field_scalar(BUOYANT_TEMPERATURE_FIELD)).expect("read T");
    let grad_p = pollster::block_on(solver.get_field_vec2("grad_p")).expect("read grad_p");
    let d_p = pollster::block_on(solver.get_field_scalar("d_p")).expect("read d_p");
    SteadySolution {
        mesh,
        u,
        p,
        t,
        grad_p,
        d_p,
    }
}

fn volume_mean(mesh: &Mesh, f: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut vol = 0.0;
    for i in 0..mesh.num_cells() {
        sum += mesh.cell_vol[i] * f[i];
        vol += mesh.cell_vol[i];
    }
    sum / vol
}

/// Acceptance test: manufactured (U, p, T) with active Boussinesq coupling
/// converges at second order (SOU) through the coupled path.
#[test]
fn steady_buoyant_coupled_second_order() {
    let mut hs = Vec::new();
    let mut u_errs = Vec::new();
    let mut t_errs = Vec::new();
    let mut p_errs = Vec::new();
    for n in [8usize, 16, 32, 64] {
        let (mesh, u, p, t) = solve_steady(n, Scheme::SecondOrderUpwind);
        let u_err = field_errors_vec2(&mesh, &u, exact_u).l2;
        let t_err = field_errors(&mesh, &t, exact_t).l2;

        let p_mean = volume_mean(&mesh, &p);
        let exact_mean = {
            let exact: Vec<f64> = (0..mesh.num_cells())
                .map(|i| exact_p(mesh.cell_cx[i], mesh.cell_cy[i]))
                .collect();
            volume_mean(&mesh, &exact)
        };
        let p_err = field_errors(&mesh, &p, |x, y| exact_p(x, y) - exact_mean + p_mean).l2;

        println!("[mms][buoyant] n={n} u_l2={u_err:.4e} t_l2={t_err:.4e} p_l2={p_err:.4e}");
        hs.push(1.0 / n as f64);
        u_errs.push(u_err);
        t_errs.push(t_err);
        p_errs.push(p_err);
    }
    assert_convergence_order("buoyant_u", &hs, &u_errs, 2.0, 0.35, 1.0e-3);
    // Regression guard for the grad_state slot-mapping contract: the gradients
    // kernel must key grad_state by STATE OFFSET, not unknown rank, or T (the
    // first solved unknown, placed behind aux fields) gets no gradient slot and
    // its SOU reconstruction silently degrades to first-order upwind.
    assert_convergence_order("buoyant_T", &hs, &t_errs, 2.0, 0.35, 3.0e-4);
    let p_order = mms_support::fit_order(&hs, &p_errs);
    println!("[mms][buoyant] pressure order {p_order:.3}");
    assert!(
        p_order > 0.9,
        "pressure order regressed: {p_order:.3} (errors {p_errs:?})"
    );
}

/// Diagnostic probe: same manufactured problem with NO outlet faces (right
/// boundary is Inlet type, pressure all-Neumann), plus ring-binned errors
/// and a host replication of the derived face flux split into its
/// face-averaged-velocity and Rhie-Chow bracket pieces. Documents that the
/// outlet-face Rhie-Chow closure costs pressure ~0.3 orders (p order 1.36
/// with an outlet vs 1.64 all-Neumann; the cell-centered grad_p vs one-sided
/// compact difference mismatch at outlet faces is O(h) where p'' is nonzero).
#[test]
#[ignore]
fn probe_buoyant_no_outlet_t_order() {
    let mut hs = Vec::new();
    let mut u_errs = Vec::new();
    let mut t_errs = Vec::new();
    let mut p_errs = Vec::new();
    for n in [8usize, 16, 32, 64] {
        let sol = solve_steady_with_right(n, Scheme::SecondOrderUpwind, BoundaryType::Inlet);
        let (mesh, u, p, t) = (&sol.mesh, &sol.u, &sol.p, &sol.t);
        let u_err = field_errors_vec2(mesh, u, exact_u).l2;
        let t_err = field_errors(mesh, t, exact_t).l2;

        // Host replication of the derived flux kernel on interior faces,
        // split into the face-averaged-velocity piece and the Rhie-Chow
        // pressure bracket, each compared against the exact mass flux.
        let mut sq_avg = 0.0f64;
        let mut sq_rc = 0.0f64;
        let mut sq_gp = 0.0f64;
        let mut n_int = 0usize;
        for f in 0..mesh.num_faces() {
            let Some(nb) = mesh.face_neighbor[f] else {
                continue;
            };
            let o = mesh.face_owner[f];
            let (mut nx, mut ny) = (mesh.face_nx[f], mesh.face_ny[f]);
            let (fcx, fcy) = (mesh.face_cx[f], mesh.face_cy[f]);
            let (ocx, ocy) = (mesh.cell_cx[o], mesh.cell_cy[o]);
            let (ncx, ncy) = (mesh.cell_cx[nb], mesh.cell_cy[nb]);
            if (fcx - ocx) * nx + (fcy - ocy) * ny < 0.0 {
                nx = -nx;
                ny = -ny;
            }
            let d_own = ((fcx - ocx) * nx + (fcy - ocy) * ny).abs();
            let d_nb = ((ncx - fcx) * nx + (ncy - fcy) * ny).abs();
            let lam = d_nb / (d_own + d_nb);
            let lam_o = 1.0 - lam;
            let dist = ((ncx - ocx) * nx + (ncy - ocy) * ny).abs();
            let area = mesh.face_area[f];
            let ubar_n = (lam * u[o].0 + lam_o * u[nb].0) * nx
                + (lam * u[o].1 + lam_o * u[nb].1) * ny;
            let gbar_n = (lam * sol.grad_p[o].0 + lam_o * sol.grad_p[nb].0) * nx
                + (lam * sol.grad_p[o].1 + lam_o * sol.grad_p[nb].1) * ny;
            let dbar = lam * sol.d_p[o] + lam_o * sol.d_p[nb];
            let phi_avg = RHO * ubar_n * area;
            let phi_rc = RHO * dbar * (gbar_n - (p[nb] - p[o]) / dist) * area;
            let (uex, uey) = exact_u(fcx, fcy);
            let phi_exact = RHO * (uex * nx + uey * ny) * area;
            sq_avg += (phi_avg - phi_exact).powi(2);
            sq_rc += phi_rc * phi_rc;
            // High-frequency content of the pressure ERROR: face-difference
            // of e_p over dist (a smooth e_p gives O(e_p); checkerboard
            // content gives O(e_p / h)).
            let e_diff = ((p[nb] - exact_p(ncx, ncy)) - (p[o] - exact_p(ocx, ocy))) / dist;
            sq_gp += e_diff * e_diff;
            n_int += 1;
        }
        let ni = n_int as f64;
        println!(
            "[mms][buoyant-no-outlet] n={n} flux_err_avg={:.3e} flux_rc={:.3e} ep_facediff={:.3e}",
            (sq_avg / ni).sqrt(),
            (sq_rc / ni).sqrt(),
            (sq_gp / ni).sqrt()
        );
        let p_mean = volume_mean(&mesh, &p);
        let exact_mean = {
            let exact: Vec<f64> = (0..mesh.num_cells())
                .map(|i| exact_p(mesh.cell_cx[i], mesh.cell_cy[i]))
                .collect();
            volume_mean(&mesh, &exact)
        };
        let p_err = field_errors(&mesh, &p, |x, y| exact_p(x, y) - exact_mean + p_mean).l2;
        println!("[mms][buoyant-no-outlet] n={n} u_l2={u_err:.4e} t_l2={t_err:.4e} p_l2={p_err:.4e}");
        let u_linf = field_errors_vec2(&mesh, &u, exact_u).linf;
        let t_linf = field_errors(&mesh, &t, exact_t).linf;
        println!("[mms][buoyant-no-outlet] n={n} u_linf={u_linf:.4e} t_linf={t_linf:.4e}");
        // Ring-binned L2: distance to the nearest boundary in cells.
        let h = 1.0 / n as f64;
        let mut ring_t = vec![(0.0f64, 0.0f64); 4];
        let mut ring_u = vec![(0.0f64, 0.0f64); 4];
        for i in 0..mesh.num_cells() {
            let (x, y) = (mesh.cell_cx[i], mesh.cell_cy[i]);
            let d = x.min(1.0 - x).min(y).min(1.0 - y);
            let ring = ((d / h - 0.5).round() as usize).min(3);
            let te = t[i] - exact_t(x, y);
            let (uex, uey) = exact_u(x, y);
            let ue2 = (u[i].0 - uex).powi(2) + (u[i].1 - uey).powi(2);
            ring_t[ring].0 += mesh.cell_vol[i] * te * te;
            ring_t[ring].1 += mesh.cell_vol[i];
            ring_u[ring].0 += mesh.cell_vol[i] * ue2;
            ring_u[ring].1 += mesh.cell_vol[i];
        }
        let fmt = |r: &[(f64, f64)]| {
            r.iter()
                .map(|(s, v)| format!("{:.3e}", (s / v.max(1e-300)).sqrt()))
                .collect::<Vec<_>>()
                .join(" ")
        };
        println!(
            "[mms][buoyant-no-outlet] n={n} t_rings=[{}] u_rings=[{}]",
            fmt(&ring_t),
            fmt(&ring_u)
        );
        hs.push(1.0 / n as f64);
        u_errs.push(u_err);
        t_errs.push(t_err);
        p_errs.push(p_err);
    }
    println!(
        "[mms][buoyant-no-outlet] orders: u={:.3} T={:.3} p={:.3}",
        mms_support::fit_order(&hs, &u_errs),
        mms_support::fit_order(&hs, &t_errs),
        mms_support::fit_order(&hs, &p_errs)
    );
}
