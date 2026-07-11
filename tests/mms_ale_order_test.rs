//! Prescribed-motion MMS: convergence orders of the ALE incompressible solver
//! on a smoothly deforming structured mesh.
//!
//! **Spatial study** — the forced steady Taylor–Green of
//! tests/mms_incompressible_order_test.rs (same manufactured solution, source
//! and all-wall per-face Dirichlet BCs) on one smooth, fixed-amplitude deformed
//! mapping. Static solves on that same mapping provide the uncontaminated h
//! sweep. A separate n=32 run oscillates the vertices through the mapping (the
//! bump is zero on the boundary, so per-face Dirichlet values stay exact) and
//! must match its static same-final-geometry counterpart field-by-field. The
//! temporal study below independently gates the moving-volume BDF2 order.
//!
//! Per step (fixed dt): move vertices analytically → `recalculate_geometry`
//! → swept-quad fluxes + f32 SCL closure → `begin_ale_step` (rotates volume
//! history, uploads geometry + fluxes) → re-upload the manufactured source at
//! the MOVED centroids (`set_field_vec2_current`, history-preserving) →
//! `step()`.
//!
//! Marches 15 steps at dt=0.05 and samples at t=0.75 — the phase of MAXIMUM
//! deformation and momentarily zero mesh velocity (sin(2π·0.75)=−1), so the
//! error is measured on the deformed geometry.
//!
//! **Temporal study** — BDF2 on moving volumes: spatially uniform
//! U*(t) = U₀·e^{−t} (spatial operators are exact for uniform fields), source
//! S = ρ·dU*/dt uniform, on a mesh oscillating at FIXED amplitude while dt
//! refines, so the measured error is purely the temporal truncation.
#![cfg(feature = "dev-tests")]

mod mms_support;

use std::f64::consts::PI;

use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::mesh::{
    generate_structured_rect_mesh, swept_mesh_fluxes_closed, BoundarySides, BoundaryType, Mesh,
};
use cfd2::solver::model::helpers::{SolverFieldAliasesExt, SolverRuntimeParamsExt};
use cfd2::solver::model::{incompressible_momentum_ale_mms_model, INCOMPRESSIBLE_MMS_SOURCE_FIELD};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};

use mms_support::{assert_convergence_order, field_errors, field_errors_vec2};

const MU: f64 = 1.0;
const RHO: f64 = 1.0;
const DT: f64 = 0.05;
/// 0.75 motion periods, sampled at max deformation / zero mesh velocity.  With
/// mu=rho=1 the startup transient decays on the 1/(2 pi^2) time scale, so this
/// remains comfortably settled while cutting the spatial gate's runtime.
const STEPS: usize = 15;
const MOTION_PERIOD: f64 = 1.0;
/// Fixed physical bump amplitude for the spatial refinement study.  Keeping
/// this independent of h holds the mesh path and mapping Jacobian fixed as the
/// spatial grid refines; an h-scaled amplitude would make both skew and the ALE
/// terms vanish with refinement and contaminate the fitted spatial order.
// Equal to the former n=32 amplitude (0.2 h at h=1/32), retaining a clearly
// nonzero ALE path at the comparison level without pushing the coarsest grid
// outside the skew-correction regime.
const SPATIAL_AMPLITUDE: f64 = 0.00625;
/// The temporal study uses one fixed mesh, so retaining its original h-scaled
/// amplitude is harmless (and preserves the calibrated mesh path).
const TEMPORAL_AMP_FRAC: f64 = 0.2;

// ── manufactured solution (identical to the static incompressible suite) ──

fn exact_u(x: f64, y: f64) -> (f64, f64) {
    (
        (PI * x).sin() * (PI * y).cos(),
        -(PI * x).cos() * (PI * y).sin(),
    )
}

fn exact_p(x: f64, y: f64) -> f64 {
    (RHO / 4.0) * ((2.0 * PI * x).cos() + (2.0 * PI * y).cos())
}

fn source(x: f64, y: f64) -> (f64, f64) {
    let (ux, uy) = exact_u(x, y);
    (2.0 * MU * PI * PI * ux, 2.0 * MU * PI * PI * uy)
}

/// Prescribed vertex position at time `t` from the UNDEFORMED coordinates (no
/// incremental drift): smooth interior bump, zero on the boundary.
fn vertex_position(x0: f64, y0: f64, t: f64) -> (f64, f64) {
    let phase = (2.0 * PI * t / MOTION_PERIOD).sin();
    let amp = SPATIAL_AMPLITUDE * phase;
    let bump = (PI * x0).sin().powi(2) * (PI * y0).sin().powi(2);
    (x0 + amp * bump, y0 - 0.6 * amp * bump)
}

fn build_solver(mesh: &Mesh, model: cfd2::solver::model::ModelSpec) -> UnifiedSolver {
    pollster::block_on(UnifiedSolver::new(
        mesh,
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
    .expect("solver init")
}

/// Volume-weighted mean of a scalar field (pressure-gauge removal).
fn volume_mean(mesh: &Mesh, f: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut vol = 0.0;
    for i in 0..mesh.num_cells() {
        sum += mesh.cell_vol[i] * f[i];
        vol += mesh.cell_vol[i];
    }
    sum / vol
}

/// One spatial level: march the ALE protocol on an n×n all-wall unit square
/// with prescribed bump motion (or initialized directly on the final geometry
/// when `static_deform`); returns (deformed mesh at t_end, U, p).
fn solve_taylor_green_moving(n: usize, static_deform: bool) -> (Mesh, Vec<(f64, f64)>, Vec<f64>) {
    let mut mesh = generate_structured_rect_mesh(n, n, 1.0, 1.0, BoundarySides::wall());
    let x0 = mesh.vx.clone();
    let y0 = mesh.vy.clone();

    // The comparison solve is genuinely static: construct the solver directly
    // on the moving run's final geometry and seed all ALE volume history there.
    // This avoids the former deform-once first step and, crucially, uses the
    // actual final phase (sin(1.5π) = -1), not the opposite +1 deformation.
    if static_deform {
        let t_end = STEPS as f64 * DT;
        for v in 0..mesh.num_vertices() {
            (mesh.vx[v], mesh.vy[v]) = vertex_position(x0[v], y0[v], t_end);
        }
        mesh.recalculate_geometry();
    }

    let model = incompressible_momentum_ale_mms_model().expect("ale+mms model");
    let mut solver = build_solver(&mesh, model);

    solver.set_dt(DT as f32);
    solver.set_dtau(0.0).expect("dtau");
    solver.set_density(RHO as f32).expect("density");
    solver.set_viscosity(MU as f32).expect("viscosity");
    solver.set_alpha_u(0.7).expect("alpha_u");
    solver.set_alpha_p(0.3).expect("alpha_p");
    solver.set_outer_iters(25).expect("outer_iters");

    // Per-face Dirichlet U on all walls from the exact solution. Boundary
    // face centers are motion-invariant (bump ≡ 0 on the boundary).
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

    // Initial source at the current centroids (undeformed for the moving run,
    // final-deformed for the static comparison), zero initial U/p, seeded history.
    let src: Vec<(f64, f64)> = (0..mesh.num_cells())
        .map(|i| source(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    solver
        .set_field_vec2(INCOMPRESSIBLE_MMS_SOURCE_FIELD, &src)
        .expect("upload mms source");
    solver.set_u(&vec![(0.0, 0.0); mesh.num_cells()]);
    solver.set_p(&vec![0.0; mesh.num_cells()]);
    solver.initialize_history();

    let mut max_abs_mesh_flux = 0.0_f64;
    for step in 0..STEPS {
        let t_new = (step as f64 + 1.0) * DT;
        let old_vx = mesh.vx.clone();
        let old_vy = mesh.vy.clone();
        if !static_deform {
            for v in 0..mesh.num_vertices() {
                (mesh.vx[v], mesh.vy[v]) = vertex_position(x0[v], y0[v], t_new);
            }
        }
        mesh.recalculate_geometry();

        let swept =
            swept_mesh_fluxes_closed(&mesh, &old_vx, &old_vy, DT).expect("swept mesh fluxes");
        max_abs_mesh_flux = swept
            .fluxes
            .iter()
            .map(|&flux| (flux as f64).abs())
            .fold(max_abs_mesh_flux, f64::max);
        assert!(
            swept.max_identity_err_rel < 1e-12,
            "step {step}: f64 swept-quad identity violated: {:.3e}",
            swept.max_identity_err_rel
        );
        assert!(
            swept.max_defect_rel < 1e-8,
            "step {step}: f32 SCL closure defect above roundoff: {:.3e}",
            swept.max_defect_rel
        );
        solver
            .begin_ale_step(&mesh, &swept.fluxes)
            .expect("begin_ale_step");

        // Manufactured source re-evaluated at the MOVED centroids —
        // history-preserving upload (BDF2 history must not be clobbered).
        let src: Vec<(f64, f64)> = (0..mesh.num_cells())
            .map(|i| source(mesh.cell_cx[i], mesh.cell_cy[i]))
            .collect();
        solver
            .set_field_vec2_current(INCOMPRESSIBLE_MMS_SOURCE_FIELD, &src)
            .expect("re-upload mms source");

        solver.step();
    }

    if static_deform {
        assert!(
            max_abs_mesh_flux <= 1.0e-12,
            "static comparison generated mesh flux {max_abs_mesh_flux:.3e}"
        );
    } else {
        assert!(
            max_abs_mesh_flux >= 1.0e-5,
            "moving comparison did not exercise ALE: max mesh flux {max_abs_mesh_flux:.3e}"
        );
    }

    let u = pollster::block_on(solver.get_field_vec2("U")).expect("read U");
    let p = pollster::block_on(solver.get_p());
    (mesh, u, p)
}

/// SPATIAL order on a fixed, smoothly deformed mesh.
///
/// The fixed physical motion defines one smooth mesh mapping sampled at every
/// refinement level.  The order fit uses a static solve on that mapping so a
/// fixed-dt ALE time error cannot masquerade as spatial truncation.  A single
/// moving n=32 solve then isolates the ALE path by direct comparison with its
/// same-geometry static counterpart; temporal ALE order is gated separately
/// below.
///
/// The decisive ALE-correctness check is asserted in-test below: the
/// moving-mesh solve must match a STATIC solve on the same max-deformed
/// geometry, i.e. the ALE terms add no error on top of the skewed-cell band.
/// An order collapse below 1.35, a blown finest cap, or excessive direct
/// moving/static field deltas catches ALE-term regressions.
#[test]
fn ale_taylor_green_sou_velocity_second_order() {
    let mut hs = Vec::new();
    let mut u_errs = Vec::new();
    let mut p_errs = Vec::new();
    let levels: Vec<usize> = std::env::var("CFD2_ALE_SPATIAL_LEVELS")
        .ok()
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![8, 16, 32]);
    let mut static_n32 = None;
    for n in levels {
        let (mesh, u, p) = solve_taylor_green_moving(n, true);
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
        println!("[mms][ale_taylor_green/static] n={n} u_l2={u_err:.4e} p_l2={p_err:.4e}");
        hs.push(1.0 / n as f64);
        u_errs.push(u_err);
        p_errs.push(p_err);
        if n == 32 {
            static_n32 = Some((mesh, u, p, u_err, p_err));
        }
    }
    // Decisive ALE-correctness check: compare the fields themselves against a
    // solver initialized and marched statically on the identical final mesh.
    if let Some((mesh_s, u_s, p_s, u_err_static, p_err_static)) = static_n32 {
        let (mesh_m, u_m, p_m) = solve_taylor_green_moving(32, false);
        let u_err_moving = field_errors_vec2(&mesh_m, &u_m, exact_u).l2;
        let ratio = u_err_moving / u_err_static;

        let max_geometry_delta = mesh_m
            .vx
            .iter()
            .chain(mesh_m.vy.iter())
            .chain(mesh_m.cell_cx.iter())
            .chain(mesh_m.cell_cy.iter())
            .chain(mesh_m.cell_vol.iter())
            .zip(
                mesh_s
                    .vx
                    .iter()
                    .chain(mesh_s.vy.iter())
                    .chain(mesh_s.cell_cx.iter())
                    .chain(mesh_s.cell_cy.iter())
                    .chain(mesh_s.cell_vol.iter()),
            )
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_geometry_delta <= 1.0e-13,
            "moving/static final geometries differ: max delta {max_geometry_delta:.3e}"
        );

        let volume = mesh_m.cell_vol.iter().sum::<f64>();
        let u_delta = u_m
            .iter()
            .zip(&u_s)
            .zip(&mesh_m.cell_vol)
            .map(|((a, b), &v)| v * ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)))
            .sum::<f64>();
        let u_delta_l2 = (u_delta / volume).sqrt();
        let p_m_mean = volume_mean(&mesh_m, &p_m);
        let p_s_mean = volume_mean(&mesh_s, &p_s);
        let p_delta = p_m
            .iter()
            .zip(&p_s)
            .zip(&mesh_m.cell_vol)
            .map(|((a, b), &v)| v * ((a - p_m_mean) - (b - p_s_mean)).powi(2))
            .sum::<f64>();
        let p_delta_l2 = (p_delta / volume).sqrt();
        println!(
            "[mms][ale_taylor_green] n=32 moving/static-deformed u_l2 ratio {ratio:.4} \
             (moving {u_err_moving:.4e}, static {u_err_static:.4e}), \
             direct u_delta_l2={u_delta_l2:.4e} p_delta_l2={p_delta_l2:.4e}"
        );
        assert!(
            (ratio - 1.0).abs() <= 0.05,
            "ALE moving-mesh error diverged from the static solve on the same deformed \
             geometry: ratio {ratio:.4} (moving {u_err_moving:.4e}, static {u_err_static:.4e}) \
             — the ALE terms are injecting error beyond the spatial skew band"
        );
        assert!(
            u_delta_l2 <= 0.10 * u_err_static,
            "ALE/static direct U difference too large: {u_delta_l2:.3e} > 10% of \
             static error {u_err_static:.3e}"
        );
        assert!(
            p_delta_l2 <= 0.20 * p_err_static,
            "ALE/static direct demeaned-p difference too large: {p_delta_l2:.3e} > 20% of \
             static error {p_err_static:.3e}"
        );
    }

    assert_convergence_order("ale_taylor_green_u", &hs, &u_errs, 2.0, 0.65, 1.3e-3);
    let p_order = mms_support::fit_order(&hs, &p_errs);
    println!("[mms][ale_taylor_green] pressure order {p_order:.3}");
    assert!(
        p_errs.windows(2).all(|pair| pair[1] < pair[0]),
        "pressure error did not decrease monotonically: {p_errs:?}"
    );
    assert!(
        p_order > 0.6 && *p_errs.last().unwrap() < 3.0e-2,
        "pressure convergence regressed: order {p_order:.3}, errors {p_errs:?}"
    );
}

// ── temporal study ────────────────────────────────────────────────────────

const T_END: f64 = 1.0;
const U0: (f64, f64) = (1.0, 0.5);
/// Temporal-study mesh: fixed 24×16 channel (GCL BC layout: uniform flow
/// enters left+bottom, leaves right+top — every BC is exactly satisfied by a
/// spatially uniform U(t)).
const TNX: usize = 24;
const TNY: usize = 16;
const TLX: f64 = 1.5;
const TLY: f64 = 1.0;
/// FIXED motion amplitude/period across the dt sweep (the mesh path is the
/// same curve, sampled finer as dt shrinks).
const T_MOTION_PERIOD: f64 = 1.0;

fn exact_u_t(t: f64) -> (f64, f64) {
    ((-t).exp() * U0.0, (-t).exp() * U0.1)
}

/// One dt level of the temporal study: returns the U L2 error at t=1.
fn bdf2_moving_volume_error(steps: usize) -> f64 {
    let dt = T_END / steps as f64;
    let mut mesh = generate_structured_rect_mesh(
        TNX,
        TNY,
        TLX,
        TLY,
        BoundarySides {
            left: BoundaryType::Inlet,
            right: BoundaryType::Outlet,
            bottom: BoundaryType::Inlet,
            top: BoundaryType::Outlet,
        },
    );
    let x0 = mesh.vx.clone();
    let y0 = mesh.vy.clone();
    let h = TLX / TNX as f64;

    // Tighten the linear tolerance through the MODEL recipe field: works on
    // both backends (the CFD2_LIN_TOL env override is GPU-only).
    let mut model = incompressible_momentum_ale_mms_model().expect("ale+mms model");
    if let Some(ls) = model.linear_solver.as_mut() {
        ls.solver.tolerance = 1e-7;
    }
    let mut solver = build_solver(&mesh, model);

    solver.set_dt(dt as f32);
    solver.set_dtau(0.0).expect("dtau");
    solver.set_density(RHO as f32).expect("density");
    solver.set_viscosity(1e-2_f32).expect("viscosity");
    solver.set_alpha_u(0.7).expect("alpha_u");
    solver.set_alpha_p(0.3).expect("alpha_p");
    // 50 outers: a temporal-order instrument must drive each implicit step to
    // convergence well below the finest truncation error, or per-step
    // Picard/relaxation lag floors the sweep and collapses the order.
    solver.set_outer_iters(50).expect("outer_iters");

    // t=0 state: exactly U0 everywhere, p = 0.
    let n = mesh.num_cells();
    solver
        .set_field_vec2(INCOMPRESSIBLE_MMS_SOURCE_FIELD, &vec![(0.0, 0.0); n])
        .expect("zero source");
    solver.set_u(&vec![U0; n]);
    solver.set_p(&vec![0.0; n]);
    solver.initialize_history();
    solver
        .set_boundary_vec2(GpuBoundaryType::Inlet, "U", [U0.0 as f32, U0.1 as f32])
        .expect("inlet U");

    let bump = |x0: f64, y0: f64| (PI * x0 / TLX).sin().powi(2) * (PI * y0 / TLY).sin().powi(2);

    for step in 0..steps {
        let t_new = (step as f64 + 1.0) * dt;
        let old_vx = mesh.vx.clone();
        let old_vy = mesh.vy.clone();
        let amp = TEMPORAL_AMP_FRAC * h * (2.0 * PI * t_new / T_MOTION_PERIOD).sin();
        for v in 0..mesh.num_vertices() {
            let b = bump(x0[v], y0[v]);
            mesh.vx[v] = x0[v] + amp * b;
            mesh.vy[v] = y0[v] - 0.6 * amp * b;
        }
        mesh.recalculate_geometry();
        let swept =
            swept_mesh_fluxes_closed(&mesh, &old_vx, &old_vy, dt).expect("swept mesh fluxes");
        solver
            .begin_ale_step(&mesh, &swept.fluxes)
            .expect("begin_ale_step");

        // Time-varying manufactured source S = rho·dU*/dt and inlet Dirichlet
        // U*(t^{n+1}), both at the new time level (implicit consumption).
        let (ex, ey) = exact_u_t(t_new);
        solver
            .set_field_vec2_current(
                INCOMPRESSIBLE_MMS_SOURCE_FIELD,
                &vec![(-RHO * ex, -RHO * ey); n],
            )
            .expect("source upload");
        solver
            .set_boundary_vec2(GpuBoundaryType::Inlet, "U", [ex as f32, ey as f32])
            .expect("inlet U(t)");

        solver.step();
    }

    let u = pollster::block_on(solver.get_field_vec2("U")).expect("read U");
    let (ex, ey) = exact_u_t(T_END);
    field_errors_vec2(&mesh, &u, |_x, _y| (ex, ey)).l2
}

/// TEMPORAL order of BDF2 on moving volumes (dt sweep at fixed mesh + fixed
/// motion amplitude; spatially uniform manufactured solution so the error is
/// purely temporal). The moving-volume BDF2 (variable-dt Newton weights on
/// V·φ + the scheme-matched bounded rate `ale_dvdt_ddt`) holds SECOND order;
/// a naive swept-volume BDF2 can degrade to first order.
#[test]
fn ale_bdf2_moving_volume_temporal_order() {
    let mut dts = Vec::new();
    let mut errs = Vec::new();
    for steps in [10usize, 20, 40, 80] {
        let err = bdf2_moving_volume_error(steps);
        println!(
            "[mms][ale_bdf2_temporal] steps={steps} dt={:.4} u_l2={err:.4e}",
            T_END / steps as f64
        );
        dts.push(T_END / steps as f64);
        errs.push(err);
    }
    assert_convergence_order("ale_bdf2_temporal", &dts, &errs, 2.0, 0.3, 5.0e-5);
}
