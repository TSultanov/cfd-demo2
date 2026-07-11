//! Method-of-Manufactured-Solutions convergence-order test for the **thermal**
//! all-Mach pressure model (`allmach_thermal`): the coupled momentum + pressure
//! + temperature system with a VARIABLE density recovered on-device from the EOS
//! `rho = rho_t_ref/T + psi*p`.
//!
//! Manufactured solution (steady, unit square, exact Dirichlet outlet contract):
//!   - divergence-free Taylor-Green velocity from
//!       Psi(x,y) = (A/pi) sin(pi x) sin(pi y)
//!       U*(x,y)  = ( dPsi/dy, -dPsi/dx )            ==> div(U*) = 0 exactly
//!   - temperature
//!       T*(x,y)  = T_ref (1.5 + 0.5 cos(pi x) cos(pi y))     in [T_ref, 2 T_ref]
//!   - pressure, with an exact right-outlet value
//!       p*(x,y)  = P_AMP (cos(2 pi x) + cos(2 pi y))
//!   - EOS density (psi = 0 here, isolating the thermal 1/T variation)
//!       rho*(x,y) = rho_t_ref / T*                            (the solver recovers this)
//!   - mass flux m* = rho* U*, which is not divergence-free for variable rho
//!
//! Therefore continuity uses S_p = div(m*). The bounded momentum and conservative
//! temperature sources match the assembled equations:
//!   S_U = div(m* U*) - U* div(m*) - mu lap(U*) + grad(p*)
//!   S_T = div(m* T*) - (k/cp) lap(T*)
//! Exact U/T data are imposed on every edge and the Outlet kinds are asserted.
//! At PSI=0 pressure remains a saddle-point gauge field, so its shape error is
//! demeaned; the compressible companion separately gates absolute pressure.
#![cfg(all(feature = "dev-tests", feature = "ui"))]

mod mms_support;

use std::f64::consts::PI;

use cfd2::solver::gpu::enums::{GpuBcKind, GpuBoundaryType};
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType, Mesh};
use cfd2::solver::model::helpers::{SolverFieldAliasesExt, SolverRuntimeParamsExt};
use cfd2::solver::model::{
    allmach_thermal_mms_model, ALLMACH_K_OVER_CP, ALLMACH_MMS_SOURCE_P_FIELD,
    ALLMACH_MMS_SOURCE_T_FIELD, ALLMACH_MMS_SOURCE_U_FIELD, ALLMACH_T_REF,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};

use mms_support::{
    assert_convergence_order, field_errors, field_errors_vec2, fit_order,
    run_to_steady_vec2_with_scalars,
};

const MU: f64 = 1.0;
const RHO_REF: f64 = 1.0;
const T_REF: f64 = ALLMACH_T_REF;
const K_OVER_CP: f64 = ALLMACH_K_OVER_CP;
const PSI: f64 = 0.0; // isolate the thermal (1/T) density variation
const U_AMP: f64 = 0.1; // Taylor-Green velocity amplitude -> moderate Peclet
const P_AMP: f64 = 0.25;
const RHO_T_REF: f64 = RHO_REF * T_REF;
const MMS_DT: f64 = 0.2;
const ALPHA_U: f64 = 0.7;

const STEADY_TOL: f64 = 5e-6;
const STEADY_MAX_STEPS: usize = 400;

// ---- exact closures ---------------------------------------------------------

fn exact_t(x: f64, y: f64) -> f64 {
    T_REF * (1.5 + 0.5 * (PI * x).cos() * (PI * y).cos())
}

fn exact_p(x: f64, y: f64) -> f64 {
    P_AMP * ((2.0 * PI * x).cos() + (2.0 * PI * y).cos())
}

fn exact_rho(x: f64, y: f64) -> f64 {
    RHO_T_REF / exact_t(x, y) + PSI * exact_p(x, y)
}

/// Divergence-free Taylor-Green velocity (so the dev2/grad(div U) term vanishes).
/// It is NOT mass-flux-free for variable density: div(rho*U*) = U*.grad(rho) != 0,
/// which is exactly the continuity source that FORCES the pressure.
fn exact_u(x: f64, y: f64) -> (f64, f64) {
    (
        U_AMP * (PI * x).sin() * (PI * y).cos(),
        -U_AMP * (PI * x).cos() * (PI * y).sin(),
    )
}

/// Rhie-Chow mass flux m* = rho* U* (variable density).
fn mass_flux(x: f64, y: f64) -> (f64, f64) {
    let (ux, uy) = exact_u(x, y);
    let r = exact_rho(x, y);
    (r * ux, r * uy)
}

// ---- 4th-order central finite differences over scalar closures --------------

const H: f64 = 1.0e-4;

fn d_dx(g: &impl Fn(f64, f64) -> f64, x: f64, y: f64) -> f64 {
    (g(x - 2.0 * H, y) - 8.0 * g(x - H, y) + 8.0 * g(x + H, y) - g(x + 2.0 * H, y)) / (12.0 * H)
}
fn d_dy(g: &impl Fn(f64, f64) -> f64, x: f64, y: f64) -> f64 {
    (g(x, y - 2.0 * H) - 8.0 * g(x, y - H) + 8.0 * g(x, y + H) - g(x, y + 2.0 * H)) / (12.0 * H)
}
fn d2_dx2(g: &impl Fn(f64, f64) -> f64, x: f64, y: f64) -> f64 {
    (-g(x - 2.0 * H, y) + 16.0 * g(x - H, y) - 30.0 * g(x, y) + 16.0 * g(x + H, y)
        - g(x + 2.0 * H, y))
        / (12.0 * H * H)
}
fn d2_dy2(g: &impl Fn(f64, f64) -> f64, x: f64, y: f64) -> f64 {
    (-g(x, y - 2.0 * H) + 16.0 * g(x, y - H) - 30.0 * g(x, y) + 16.0 * g(x, y + H)
        - g(x, y + 2.0 * H))
        / (12.0 * H * H)
}

fn u_x(x: f64, y: f64) -> f64 {
    exact_u(x, y).0
}
fn u_y(x: f64, y: f64) -> f64 {
    exact_u(x, y).1
}

/// Momentum source for the assembled bounded equation
/// `div(m U) - U div(m) - mu lap(U) + grad(p) = S_U`.
fn source_u(x: f64, y: f64) -> (f64, f64) {
    // convection: d/dx(m_x U_i) + d/dy(m_y U_i)
    let mx_ux = |x: f64, y: f64| mass_flux(x, y).0 * exact_u(x, y).0;
    let my_ux = |x: f64, y: f64| mass_flux(x, y).1 * exact_u(x, y).0;
    let mx_uy = |x: f64, y: f64| mass_flux(x, y).0 * exact_u(x, y).1;
    let my_uy = |x: f64, y: f64| mass_flux(x, y).1 * exact_u(x, y).1;
    let conv_x = d_dx(&mx_ux, x, y) + d_dy(&my_ux, x, y);
    let conv_y = d_dx(&mx_uy, x, y) + d_dy(&my_uy, x, y);
    let div_m = source_p(x, y);
    let (ux, uy) = exact_u(x, y);
    let bounded_conv_x = conv_x - ux * div_m;
    let bounded_conv_y = conv_y - uy * div_m;

    // viscous: mu lap(U)
    let visc_x = MU * (d2_dx2(&u_x, x, y) + d2_dy2(&u_x, x, y));
    let visc_y = MU * (d2_dx2(&u_y, x, y) + d2_dy2(&u_y, x, y));

    // pressure gradient
    let gp_x = d_dx(&exact_p, x, y);
    let gp_y = d_dy(&exact_p, x, y);

    (
        bounded_conv_x - visc_x + gp_x,
        bounded_conv_y - visc_y + gp_y,
    )
}

/// Continuity source S_p = +div(rho* U*) = +div(mass flux). NON-ZERO for variable
/// density (= U*.grad(rho)); this forces the pressure to p* (without it the
/// Rhie-Chow pressure equation drives p -> const). The +div sign matches the
/// assembled div_flux convention and recovers +p*; sign-invisible in the
/// incompressible limit (div(rho U*) = 0).
fn source_p(x: f64, y: f64) -> f64 {
    let mx = |x: f64, y: f64| mass_flux(x, y).0;
    let my = |x: f64, y: f64| mass_flux(x, y).1;
    d_dx(&mx, x, y) + d_dy(&my, x, y)
}

/// Temperature source for `div(m T) - (k/cp) lap(T) = S_T`.
fn source_t(x: f64, y: f64) -> f64 {
    let mx_t = |x: f64, y: f64| mass_flux(x, y).0 * exact_t(x, y);
    let my_t = |x: f64, y: f64| mass_flux(x, y).1 * exact_t(x, y);
    let conv = d_dx(&mx_t, x, y) + d_dy(&my_t, x, y);
    let conduction = K_OVER_CP * (d2_dx2(&exact_t, x, y) + d2_dy2(&exact_t, x, y));
    conv - conduction
}

fn volume_mean(mesh: &Mesh, field: &[f64]) -> f64 {
    let total_volume = mesh.cell_vol.iter().sum::<f64>();
    field
        .iter()
        .zip(&mesh.cell_vol)
        .map(|(&value, &volume)| value * volume)
        .sum::<f64>()
        / total_volume
}

fn assert_mms_outlet_boundary_contract() {
    let model = allmach_thermal_mms_model().expect("MMS model");
    for (field, components) in [("U", 2usize), ("p", 1), ("T", 1)] {
        let conditions = &model
            .boundaries
            .field(field)
            .unwrap_or_else(|| panic!("missing {field} boundary spec"))
            .by_boundary[&GpuBoundaryType::Outlet];
        assert_eq!(conditions.len(), components);
        assert!(
            conditions
                .iter()
                .all(|condition| condition.kind == GpuBcKind::Dirichlet),
            "MMS Outlet {field} must remain Dirichlet"
        );
    }
}

fn solve(n: usize) -> (Mesh, Vec<(f64, f64)>, Vec<f64>, Vec<f64>) {
    // Exact U/T Dirichlet on every edge. The right edge is Outlet-labelled so
    // its exact pressure value removes the pure-Neumann pressure null mode;
    // the MMS model deliberately keeps U/T Dirichlet on that patch.
    // (A zero-gradient T wall leaves T determined only up to a constant, which
    // would corrupt rho = rho_t_ref/T and hence the whole coupled solve.)
    let sides = BoundarySides {
        left: BoundaryType::MovingWall,
        right: BoundaryType::Outlet,
        bottom: BoundaryType::MovingWall,
        top: BoundaryType::MovingWall,
    };
    let mesh = generate_structured_rect_mesh(n, n, 1.0, 1.0, sides);
    let model = allmach_thermal_mms_model().expect("allmach_thermal_mms model");
    let mut solver = pollster::block_on(UnifiedSolver::new(
        &mesh,
        model,
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

    solver.set_dt(MMS_DT as f32);
    solver.set_dtau(0.0).expect("dtau");
    solver.set_density(RHO_REF as f32).expect("density");
    solver.set_viscosity(MU as f32).expect("viscosity");
    solver.set_alpha_u(ALPHA_U as f32).expect("alpha_u");
    solver.set_alpha_p(0.3).expect("alpha_p");
    solver.set_outer_iters(25).expect("outer_iters");

    let n_cells = mesh.num_cells();
    // EOS aux fields (see the on-device recovery rho = rho_t_ref/T + psi*p).
    solver
        .set_field_scalar("psi", &vec![PSI; n_cells])
        .expect("psi");
    // The pressure-row ddt reads the decoupled `psi_precond`; seed it equal to PSI
    // (=0 here) so the residual is byte-identical to the physical psi. The term
    // vanishes at steady state; this just removes buffer-init dependence.
    solver
        .set_field_scalar("psi_precond", &vec![PSI; n_cells])
        .expect("psi_precond");
    solver
        .set_field_scalar("rho_t_ref", &vec![RHO_T_REF; n_cells])
        .expect("rho_t_ref");

    // Per-face Dirichlet velocity AND temperature on all (MovingWall) walls,
    // from the exact solution — U Dirichlet imposes the boundary velocity, T
    // Dirichlet pins the temperature gauge.
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
        .set_boundary_values_per_face(GpuBoundaryType::MovingWall, "U", 0, &wall_u(0))
        .expect("wall u_x");
    solver
        .set_boundary_values_per_face(GpuBoundaryType::MovingWall, "U", 1, &wall_u(1))
        .expect("wall u_y");
    solver
        .set_boundary_values_per_face(GpuBoundaryType::Outlet, "U", 0, &wall_u(0))
        .expect("outlet u_x");
    solver
        .set_boundary_values_per_face(GpuBoundaryType::Outlet, "U", 1, &wall_u(1))
        .expect("outlet u_y");
    let fxt = mesh.face_cx.clone();
    let fyt = mesh.face_cy.clone();
    let wall_t = move |face_idx: u32| {
        let i = face_idx as usize;
        exact_t(fxt[i], fyt[i]) as f32
    };
    solver
        .set_boundary_values_per_face(GpuBoundaryType::MovingWall, "T", 0, &wall_t)
        .expect("wall T");
    solver
        .set_boundary_values_per_face(GpuBoundaryType::Outlet, "T", 0, &wall_t)
        .expect("outlet T");
    let fxp = mesh.face_cx.clone();
    let fyp = mesh.face_cy.clone();
    let outlet_p = move |face_idx: u32| {
        let i = face_idx as usize;
        exact_p(fxp[i], fyp[i]) as f32
    };
    solver
        .set_boundary_values_per_face(GpuBoundaryType::Outlet, "p", 0, &outlet_p)
        .expect("outlet p");

    // Manufactured sources, per cell.
    let src_u: Vec<(f64, f64)> = (0..n_cells)
        .map(|i| source_u(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    let src_t: Vec<f64> = (0..n_cells)
        .map(|i| source_t(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    let src_p: Vec<f64> = (0..n_cells)
        .map(|i| source_p(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    solver
        .set_field_vec2(ALLMACH_MMS_SOURCE_U_FIELD, &src_u)
        .expect("mms_src_U");
    solver
        .set_field_scalar(ALLMACH_MMS_SOURCE_T_FIELD, &src_t)
        .expect("mms_src_T");
    solver
        .set_field_scalar(ALLMACH_MMS_SOURCE_P_FIELD, &src_p)
        .expect("mms_src_p");

    // Initialise at the exact solution (and seed rho from the EOS).
    let u0: Vec<(f64, f64)> = (0..n_cells)
        .map(|i| exact_u(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    let t0: Vec<f64> = (0..n_cells)
        .map(|i| exact_t(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    let r0: Vec<f64> = (0..n_cells)
        .map(|i| exact_rho(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    let p0: Vec<f64> = (0..n_cells)
        .map(|i| exact_p(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    solver.set_u(&u0);
    solver.set_p(&p0);
    solver.set_field_scalar("T", &t0).expect("T init");
    solver.set_field_scalar("rho", &r0).expect("rho init");
    solver.initialize_history();

    let u = run_to_steady_vec2_with_scalars(&mut solver, "U", &["T"], STEADY_MAX_STEPS, STEADY_TOL);
    let p = pollster::block_on(solver.get_p());
    let t = pollster::block_on(solver.get_field_scalar("T")).expect("T");
    (mesh, u, p, t)
}

/// Second-order convergence of the coupled variable-density momentum + pressure
/// + temperature system. Velocity and temperature must hit ~2nd order; pressure
/// is pinned at the saddle-point floor (>= ~1, as in the incompressible MMS).
#[test]
fn allmach_thermal_coupled_second_order() {
    assert_mms_outlet_boundary_contract();
    let levels = [16usize, 32, 48];
    let mut hs = Vec::new();
    let mut u_errs = Vec::new();
    let mut t_errs = Vec::new();
    let mut p_errs = Vec::new();
    for &n in &levels {
        let (mesh, u, p, t) = solve(n);
        let u_err = field_errors_vec2(&mesh, &u, exact_u).l2;
        let t_err = field_errors(&mesh, &t, exact_t).l2;
        // PSI=0 deliberately leaves the pressure as a saddle-point gauge field.
        // The boundary contract is asserted above; compare its convergent shape
        // after removing the remaining global constant.
        let p_mean = volume_mean(&mesh, &p);
        let exact: Vec<_> = (0..mesh.num_cells())
            .map(|i| exact_p(mesh.cell_cx[i], mesh.cell_cy[i]))
            .collect();
        let exact_mean = volume_mean(&mesh, &exact);
        let p_err = field_errors(&mesh, &p, |x, y| exact_p(x, y) - exact_mean + p_mean).l2;
        println!("[mms][allmach_thermal] n={n} u_l2={u_err:.4e} t_l2={t_err:.4e} p_l2={p_err:.4e}");
        hs.push(1.0 / n as f64);
        u_errs.push(u_err);
        t_errs.push(t_err);
        p_errs.push(p_err);
    }
    // Velocity and temperature carry the thermal physics (energy equation,
    // variable-density momentum/flux, on-device EOS rho-from-T coupling) and both
    // converge at ~2nd order.
    assert!(
        u_errs.windows(2).all(|pair| pair[1] < pair[0]),
        "allmach_thermal_u did not decrease monotonically: {u_errs:?}"
    );
    assert_convergence_order(
        "allmach_thermal_u_fine",
        &hs[1..],
        &u_errs[1..],
        2.0,
        0.35,
        2.0e-3,
    );
    // T amplitude (~1.5) is ~15x the velocity amplitude, so its absolute L2 error
    // floor scales up accordingly.
    assert_convergence_order(
        "allmach_thermal_T_fine",
        &hs[1..],
        &t_errs[1..],
        2.0,
        0.35,
        5.0e-4,
    );

    // The gauge-free pressure shape must converge as well (its saddle-point
    // field order is lower than the transported variables, as in the
    // incompressible MMS).
    let p_order = fit_order(&hs, &p_errs);
    let p_finest = *p_errs.last().unwrap();
    println!("[mms][allmach_thermal] pressure field L2 order {p_order:.3} (errs {p_errs:?})");
    assert!(
        p_order >= 1.0 && p_finest < 3.0e-2,
        "pressure field failed anchored convergence: order {p_order:.3}, finest {p_finest:.3e}, errors {p_errs:?}"
    );
}

fn exact_pressure_row_residual_l2(n: usize) -> f64 {
    let h = 1.0 / n as f64;
    let kappa = ALPHA_U * MMS_DT;
    let cell = |i: usize, j: usize| i * n + j;
    let mut rho = vec![0.0; n * n];
    let mut ux = vec![0.0; n * n];
    let mut uy = vec![0.0; n * n];
    let mut p = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let x = (i as f64 + 0.5) * h;
            let y = (j as f64 + 0.5) * h;
            let c = cell(i, j);
            rho[c] = exact_rho(x, y);
            (ux[c], uy[c]) = exact_u(x, y);
            p[c] = exact_p(x, y);
        }
    }
    let mut gx = vec![0.0; n * n];
    let mut gy = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let c = cell(i, j);
            gx[c] = if i == 0 {
                (p[cell(1, j)] - p[c]) / (2.0 * h)
            } else if i + 1 == n {
                (p[c] - p[cell(i - 1, j)]) / (2.0 * h)
            } else {
                (p[cell(i + 1, j)] - p[cell(i - 1, j)]) / (2.0 * h)
            };
            gy[c] = if j == 0 {
                (p[cell(i, 1)] - p[c]) / (2.0 * h)
            } else if j + 1 == n {
                (p[c] - p[cell(i, j - 1)]) / (2.0 * h)
            } else {
                (p[cell(i, j + 1)] - p[cell(i, j - 1)]) / (2.0 * h)
            };
        }
    }
    let mut fx = vec![0.0; (n + 1) * n];
    let mut fy = vec![0.0; n * (n + 1)];
    for i in 1..n {
        for j in 0..n {
            let left = cell(i - 1, j);
            let right = cell(i, j);
            fx[i * n + j] = 0.5 * (rho[left] * ux[left] + rho[right] * ux[right])
                + kappa * (0.5 * (gx[left] + gx[right]) - (p[right] - p[left]) / h);
        }
    }
    for i in 0..n {
        for j in 1..n {
            let bottom = cell(i, j - 1);
            let top = cell(i, j);
            fy[i * (n + 1) + j] = 0.5 * (rho[bottom] * uy[bottom] + rho[top] * uy[top])
                + kappa * (0.5 * (gy[bottom] + gy[top]) - (p[top] - p[bottom]) / h);
        }
    }
    let mut sum_sq = 0.0;
    for i in 0..n {
        for j in 0..n {
            let div_h = (fx[(i + 1) * n + j] - fx[i * n + j]) / h
                + (fy[i * (n + 1) + j + 1] - fy[i * (n + 1) + j]) / h;
            let x = (i as f64 + 0.5) * h;
            let y = (j as f64 + 0.5) * h;
            let error = div_h - source_p(x, y);
            sum_sq += error * error;
        }
    }
    (sum_sq / (n * n) as f64).sqrt()
}

#[test]
fn allmach_thermal_pressure_row_exact_state_is_second_order() {
    let levels = [16usize, 32, 64];
    let hs: Vec<_> = levels.iter().map(|&n| 1.0 / n as f64).collect();
    let errors: Vec<_> = levels
        .iter()
        .map(|&n| exact_pressure_row_residual_l2(n))
        .collect();
    assert_convergence_order(
        "allmach_thermal_pressure_row_exact_state",
        &hs,
        &errors,
        2.0,
        0.05,
        4.0e-3,
    );
}
