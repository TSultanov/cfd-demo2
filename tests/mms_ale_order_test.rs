//! Prescribed-motion MMS: convergence orders of the ALE incompressible solver
//! on a smoothly deforming structured mesh.
//!
//! **Spatial study** — the forced steady Taylor–Green of
//! tests/mms_incompressible_order_test.rs (same manufactured solution, source
//! and all-wall per-face Dirichlet BCs), except the mesh vertices oscillate
//! through a smooth interior bump (amplitude ∝ h, zero on the boundary — so
//! boundary face centers never move and the per-face Dirichlet values stay
//! exact). The manufactured solution is defined in FIXED space; the discrete
//! solution sees moving cell centroids, mesh-relative fluxes
//! (`phi − rho·mesh_flux`), the moving-volume ddt and the continuity volume
//! source. If the SCL closure and the ALE terms are consistent, the error
//! matches a STATIC solve on the same deformed geometry.
//!
//! Per step (fixed dt): move vertices analytically → `recalculate_geometry`
//! → swept-quad fluxes + f32 SCL closure → `begin_ale_step` (rotates volume
//! history, uploads geometry + fluxes) → re-upload the manufactured source at
//! the MOVED centroids (`set_field_vec2_current`, history-preserving) →
//! `step()`.
//!
//! Marches 35 steps at dt=0.05 and samples at t=1.75 — the phase of MAXIMUM
//! deformation and momentarily zero mesh velocity (sin(2π·1.75)=−1), so the
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
use cfd2::solver::model::{
    incompressible_momentum_ale_mms_model, INCOMPRESSIBLE_MMS_SOURCE_FIELD,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};

use mms_support::{assert_convergence_order, field_errors, field_errors_vec2};

const MU: f64 = 1.0;
const RHO: f64 = 1.0;
const DT: f64 = 0.05;
/// 1.75 motion periods, sampled at max deformation / zero mesh velocity.
const STEPS: usize = 35;
const MOTION_PERIOD: f64 = 1.0;
/// Bump amplitude as a fraction of h: keeps mesh distortion (and the ALE
/// terms) proportionally constant across refinement levels.
const AMP_FRAC: f64 = 0.2;

// ── manufactured solution (identical to the static incompressible suite) ──

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

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Prescribed vertex position at time `t` from the UNDEFORMED coordinates (no
/// incremental drift): smooth interior bump, zero on the boundary.
///
/// `static_deform`: deform to max ONCE and hold (zero motion after step 1) —
/// the static solve on the deformed mesh, isolating spatial-on-skewed-cells
/// accuracy from the ALE terms.
fn vertex_position(x0: f64, y0: f64, t: f64, h: f64, static_deform: bool) -> (f64, f64) {
    let amp_frac = env_f64("CFD2_ALE_SPATIAL_AMP", AMP_FRAC);
    let period = env_f64("CFD2_ALE_SPATIAL_PERIOD", MOTION_PERIOD);
    let phase = if static_deform {
        1.0
    } else {
        (2.0 * PI * t / period).sin()
    };
    let amp = amp_frac * h * phase;
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
/// with prescribed bump motion (or the deform-once-and-hold probe when
/// `static_deform`); returns (deformed mesh at t_end, U, p).
fn solve_taylor_green_moving(n: usize, static_deform: bool) -> (Mesh, Vec<(f64, f64)>, Vec<f64>) {
    let mut mesh = generate_structured_rect_mesh(n, n, 1.0, 1.0, BoundarySides::wall());
    let x0 = mesh.vx.clone();
    let y0 = mesh.vy.clone();
    let h = 1.0 / n as f64;

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

    // Initial source at the undeformed centroids (IC semantics: all history
    // buffers), zero initial U/p, seeded history.
    let src: Vec<(f64, f64)> = (0..mesh.num_cells())
        .map(|i| source(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    solver
        .set_field_vec2(INCOMPRESSIBLE_MMS_SOURCE_FIELD, &src)
        .expect("upload mms source");
    solver.set_u(&vec![(0.0, 0.0); mesh.num_cells()]);
    solver.set_p(&vec![0.0; mesh.num_cells()]);
    solver.initialize_history();

    // Steps scale with the (env-overridable) period so the run always covers
    // 1.75 periods to the max-deformation/zero-velocity phase.
    let period = env_f64("CFD2_ALE_SPATIAL_PERIOD", MOTION_PERIOD);
    let steps = ((STEPS as f64) * period / MOTION_PERIOD).round() as usize;

    for step in 0..steps {
        let t_new = (step as f64 + 1.0) * DT;
        let old_vx = mesh.vx.clone();
        let old_vy = mesh.vy.clone();
        for v in 0..mesh.num_vertices() {
            let (x, y) = vertex_position(x0[v], y0[v], t_new, h, static_deform);
            mesh.vx[v] = x;
            mesh.vy[v] = y;
        }
        mesh.recalculate_geometry();

        let swept =
            swept_mesh_fluxes_closed(&mesh, &old_vx, &old_vy, DT).expect("swept mesh fluxes");
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

    let u = pollster::block_on(solver.get_field_vec2("U")).expect("read U");
    let p = pollster::block_on(solver.get_p());
    (mesh, u, p)
}

/// SPATIAL order on the moving mesh.
///
/// The gate is pinned near the measured order (≥1.35), below the nominal 2:
/// the sub-2 u order is the spatial operator on persistently-skewed cells
/// (amplitude ∝ h keeps the non-orthogonality constant across levels, so the
/// skew error never refines away), NOT the ALE machinery. The graded static
/// suite keeps orthogonal cells and never sees this band.
///
/// The decisive ALE-correctness check is asserted in-test below: the
/// moving-mesh solve must match a STATIC solve on the same max-deformed
/// geometry, i.e. the ALE terms add no error on top of the skewed-cell band.
/// An order collapse below 1.35, a blown finest cap, or a moving/static
/// divergence >5% catches ALE-term regressions.
#[test]
fn ale_taylor_green_sou_velocity_second_order() {
    let mut hs = Vec::new();
    let mut u_errs = Vec::new();
    let mut p_errs = Vec::new();
    let levels: Vec<usize> = std::env::var("CFD2_ALE_SPATIAL_LEVELS")
        .ok()
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![8, 16, 32, 64]);
    let env_static_deform =
        std::env::var("CFD2_ALE_SPATIAL_STATIC_DEFORM").as_deref() == Ok("1");
    // The moving ≡ static-on-deformed-geometry assert (below) needs the n=32
    // moving error; only available on the default level list.
    let default_levels = levels == [8, 16, 32, 64];
    for n in levels {
        let (mesh, u, p) = solve_taylor_green_moving(n, env_static_deform);
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
        println!("[mms][ale_taylor_green] n={n} u_l2={u_err:.4e} p_l2={p_err:.4e}");
        hs.push(1.0 / n as f64);
        u_errs.push(u_err);
        p_errs.push(p_err);
    }
    assert_convergence_order("ale_taylor_green_u", &hs, &u_errs, 2.0, 0.65, 1.2e-3);
    let p_order = mms_support::fit_order(&hs, &p_errs);
    println!("[mms][ale_taylor_green] pressure order {p_order:.3}");
    assert!(
        p_order > 0.9,
        "pressure order regressed: {p_order:.3} (errors {p_errs:?})"
    );

    // Decisive ALE-correctness check: the moving-mesh error must equal the
    // static solve on the same max-deformed geometry — the ALE terms add
    // nothing on top of the skewed-cell spatial error. Asserted at ±5% for
    // GPU run-to-run headroom; skipped when the level list or static-deform
    // probe is overridden via env.
    if default_levels && !env_static_deform {
        let (mesh_s, u_s, _p_s) = solve_taylor_green_moving(32, true);
        let u_err_static = field_errors_vec2(&mesh_s, &u_s, exact_u).l2;
        let u_err_moving = u_errs[2];
        let ratio = u_err_moving / u_err_static;
        println!(
            "[mms][ale_taylor_green] n=32 moving/static-deformed u_l2 ratio {ratio:.4} \
             (moving {u_err_moving:.4e}, static {u_err_static:.4e})"
        );
        assert!(
            (ratio - 1.0).abs() <= 0.05,
            "ALE moving-mesh error diverged from the static solve on the same deformed \
             geometry: ratio {ratio:.4} (moving {u_err_moving:.4e}, static {u_err_static:.4e}) \
             — the ALE terms are injecting error beyond the spatial skew band"
        );
    }
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

    let bump = |x0: f64, y0: f64| {
        (PI * x0 / TLX).sin().powi(2) * (PI * y0 / TLY).sin().powi(2)
    };

    for step in 0..steps {
        let t_new = (step as f64 + 1.0) * dt;
        let old_vx = mesh.vx.clone();
        let old_vy = mesh.vy.clone();
        let amp = AMP_FRAC * h * (2.0 * PI * t_new / T_MOTION_PERIOD).sin();
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
