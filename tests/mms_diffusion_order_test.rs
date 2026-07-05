//! MMS convergence-order verification for the diffusion operator (`generic_diffusion_demo_mms`
//! model variants): the manufactured source is part of the model's math declaration; tests
//! upload per-cell source values and verify the discrete solution converges to the exact
//! solution at design order under mesh refinement.
//!
//! Discrete equation solved:
//!   dphi/dt - kappa * lap(phi) = S       with kappa = 1
//! so a manufactured `phi*` requires `S = dphi*/dt - lap(phi*)`.
#![cfg(feature = "dev-tests")]

#[path = "mms_support/mod.rs"]
mod mms_support;

use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::mesh::{
    generate_graded_rect_mesh, generate_structured_rect_mesh, AxisGrading, BoundarySides,
    BoundaryType, Mesh,
};
use cfd2::solver::model::helpers::SolverRuntimeParamsExt;
use cfd2::solver::model::{
    generic_diffusion_demo_mms_dirichlet_model, generic_diffusion_demo_mms_model,
    generic_diffusion_demo_mms_neumann_model, ModelSpec, MMS_SOURCE_FIELD,
};
use cfd2::solver::{SolverConfig, TimeScheme, UnifiedSolver};
use mms_support::{assert_convergence_order, field_errors, max_cell_extent, run_to_steady};
use std::f64::consts::PI;

// Absolute steady-state threshold: f32 jitters a few ULPs of the O(1) solution
// amplitude, so this is the practical floor; still well below the finest-level
// discretization error (~1e-4).
const STEADY_TOL: f64 = 4e-6;
const STEADY_MAX_STEPS: usize = 200;

fn unit_square_mesh(n: usize, sides: BoundarySides) -> Mesh {
    generate_structured_rect_mesh(n, n, 1.0, 1.0, sides)
}

/// Left = Inlet, right = Outlet, bottom/top = Wall (matches the `_mms` BC variants).
fn inlet_outlet_wall_sides() -> BoundarySides {
    BoundarySides {
        left: BoundaryType::Inlet,
        right: BoundaryType::Outlet,
        bottom: BoundaryType::Wall,
        top: BoundaryType::Wall,
    }
}

fn build_solver(mesh: &Mesh, model: ModelSpec) -> UnifiedSolver {
    let mut solver =
        pollster::block_on(UnifiedSolver::new(mesh, model, SolverConfig::default(), None, None))
            .expect("create solver");
    // The problem is linear: a couple of outer iterations per step suffice, but the
    // linear solve itself must be driven well below the discretization error.
    solver.set_outer_iters(2).expect("outer_iters");
    solver
}

fn upload_source(solver: &mut UnifiedSolver, mesh: &Mesh, source: impl Fn(f64, f64) -> f64) {
    let values: Vec<f64> = (0..mesh.num_cells())
        .map(|i| source(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    solver
        .set_field_scalar(MMS_SOURCE_FIELD, &values)
        .expect("upload mms source");
}

fn zero_initial_phi(solver: &mut UnifiedSolver, mesh: &Mesh) {
    solver
        .set_field_scalar("phi", &vec![0.0; mesh.num_cells()])
        .expect("init phi");
    solver.initialize_history();
}

fn solve_steady_case(
    model_fn: fn() -> Result<ModelSpec, String>,
    n: usize,
    sides: BoundarySides,
    source: &dyn Fn(f64, f64) -> f64,
    setup_bcs: &dyn Fn(&mut UnifiedSolver, &Mesh),
) -> (Mesh, Vec<f64>) {
    let mesh = unit_square_mesh(n, sides);
    solve_steady_case_on_mesh(model_fn, mesh, source, setup_bcs)
}

fn solve_steady_case_on_mesh(
    model_fn: fn() -> Result<ModelSpec, String>,
    mesh: Mesh,
    source: &dyn Fn(f64, f64) -> f64,
    setup_bcs: &dyn Fn(&mut UnifiedSolver, &Mesh),
) -> (Mesh, Vec<f64>) {
    let model = model_fn().expect("model");
    let mut solver = build_solver(&mesh, model);
    // dt of the order of the diffusion time L^2/kappa: implicit Euler contracts the
    // transient by ~1/(1 + lambda*dt) per step, reaching steady state in a few steps.
    solver.set_dt(1.0);
    setup_bcs(&mut solver, &mesh);
    upload_source(&mut solver, &mesh, source);
    zero_initial_phi(&mut solver, &mesh);
    let phi = run_to_steady(&mut solver, "phi", STEADY_MAX_STEPS, STEADY_TOL);
    (mesh, phi)
}

/// A linear exact solution must be reproduced to solver/f32 tolerance on any mesh:
/// the classic finite-volume consistency check. Any boundary-condition inconsistency
/// shows up here as a non-vanishing error.
#[test]
fn steady_linear_solution_is_exact() {
    let exact = |x: f64, _y: f64| x;
    let (mesh, phi) = solve_steady_case(
        generic_diffusion_demo_mms_model,
        16,
        inlet_outlet_wall_sides(),
        &|_x, _y| 0.0,
        &|solver, _mesh| {
            solver
                .set_boundary_scalar(GpuBoundaryType::Inlet, "phi", 0.0)
                .expect("inlet bc");
            solver
                .set_boundary_scalar(GpuBoundaryType::Outlet, "phi", 1.0)
                .expect("outlet bc");
        },
    );
    let errors = field_errors(&mesh, &phi, exact);
    println!(
        "[mms][linear_exactness] l2={:.3e} linf={:.3e}",
        errors.l2, errors.linf
    );
    assert!(
        errors.linf < 5e-5,
        "linear solution not reproduced exactly: linf={:.3e}",
        errors.linf
    );
}

/// phi* = sin(pi x) cos(pi y): Dirichlet 0 at left/right (sin vanishes), natural
/// zero-gradient at bottom/top (cos has zero normal slope). Second-order interior +
/// boundary discretization expected.
#[test]
fn steady_dirichlet_zerograd_second_order() {
    let exact = |x: f64, y: f64| (PI * x).sin() * (PI * y).cos();
    let source = move |x: f64, y: f64| 2.0 * PI * PI * (PI * x).sin() * (PI * y).cos();

    let mut hs = Vec::new();
    let mut errs = Vec::new();
    for n in [8usize, 16, 32, 64] {
        let (mesh, phi) = solve_steady_case(
            generic_diffusion_demo_mms_model,
            n,
            inlet_outlet_wall_sides(),
            &source,
            &|_solver, _mesh| {},
        );
        hs.push(1.0 / n as f64);
        errs.push(field_errors(&mesh, &phi, exact).l2);
    }
    assert_convergence_order("steady_dirichlet_zerograd", &hs, &errs, 2.0, 0.25, 1.5e-4);
}

/// phi* = cos(pi x) cos(pi y): boundary values vary along every side, exercising the
/// per-face Dirichlet boundary value API on all four boundaries.
#[test]
fn steady_perface_dirichlet_second_order() {
    let exact = |x: f64, y: f64| (PI * x).cos() * (PI * y).cos();
    let source = move |x: f64, y: f64| 2.0 * PI * PI * (PI * x).cos() * (PI * y).cos();

    let mut hs = Vec::new();
    let mut errs = Vec::new();
    for n in [8usize, 16, 32, 64] {
        let (mesh, phi) = solve_steady_case(
            generic_diffusion_demo_mms_dirichlet_model,
            n,
            inlet_outlet_wall_sides(),
            &source,
            &|solver, mesh| {
                let face_value = |mesh: &Mesh| {
                    let fx = mesh.face_cx.clone();
                    let fy = mesh.face_cy.clone();
                    move |face_idx: u32| {
                        let i = face_idx as usize;
                        ((PI * fx[i]).cos() * (PI * fy[i]).cos()) as f32
                    }
                };
                for boundary in [
                    GpuBoundaryType::Inlet,
                    GpuBoundaryType::Outlet,
                    GpuBoundaryType::Wall,
                ] {
                    solver
                        .set_boundary_values_per_face(boundary, "phi", 0, &face_value(mesh))
                        .expect("per-face dirichlet");
                }
            },
        );
        hs.push(1.0 / n as f64);
        errs.push(field_errors(&mesh, &phi, exact).l2);
    }
    assert_convergence_order("steady_perface_dirichlet", &hs, &errs, 2.0, 0.25, 1.5e-4);
}

/// The per-face Dirichlet study repeated on a two-sided geometrically graded
/// mesh: smallest cells at every wall, center/wall ratio 4 on both axes. Orders
/// are fitted against the MAX cell extent — second order must hold, proving the
/// discretization (including the distance-weighted face-coefficient
/// interpolation) is not uniform-mesh-only.
#[test]
fn steady_perface_dirichlet_graded_second_order() {
    let exact = |x: f64, y: f64| (PI * x).cos() * (PI * y).cos();
    let source = move |x: f64, y: f64| 2.0 * PI * PI * (PI * x).cos() * (PI * y).cos();
    let grading = AxisGrading::TwoSided { ratio: 4.0 };

    let mut hs = Vec::new();
    let mut errs = Vec::new();
    for n in [8usize, 16, 32, 64] {
        let mesh = generate_graded_rect_mesh(
            n,
            n,
            1.0,
            1.0,
            grading,
            grading,
            inlet_outlet_wall_sides(),
        );
        let (mesh, phi) = solve_steady_case_on_mesh(
            generic_diffusion_demo_mms_dirichlet_model,
            mesh,
            &source,
            &|solver, mesh| {
                let face_value = |mesh: &Mesh| {
                    let fx = mesh.face_cx.clone();
                    let fy = mesh.face_cy.clone();
                    move |face_idx: u32| {
                        let i = face_idx as usize;
                        ((PI * fx[i]).cos() * (PI * fy[i]).cos()) as f32
                    }
                };
                for boundary in [
                    GpuBoundaryType::Inlet,
                    GpuBoundaryType::Outlet,
                    GpuBoundaryType::Wall,
                ] {
                    solver
                        .set_boundary_values_per_face(boundary, "phi", 0, &face_value(mesh))
                        .expect("per-face dirichlet");
                }
            },
        );
        hs.push(max_cell_extent(&mesh));
        errs.push(field_errors(&mesh, &phi, exact).l2);
    }
    // Graded beats the uniform line at matched h_eff (refinement sits where the
    // boundary-layer curvature is). Cap ~2x measured.
    assert_convergence_order("steady_perface_dirichlet_graded", &hs, &errs, 2.0, 0.25, 1.6e-4);
}

/// phi* = sin(pi x) sin(pi y): Dirichlet 0 at left/right, spatially varying *non-zero*
/// Neumann at bottom/top (dphi*/dn = -pi sin(pi x) on both walls). First test in the
/// repo to exercise a non-zero Neumann boundary value.
#[test]
fn steady_neumann_mix_second_order() {
    let exact = |x: f64, y: f64| (PI * x).sin() * (PI * y).sin();
    let source = move |x: f64, y: f64| 2.0 * PI * PI * (PI * x).sin() * (PI * y).sin();

    let mut hs = Vec::new();
    let mut errs = Vec::new();
    for n in [8usize, 16, 32, 64] {
        let (mesh, phi) = solve_steady_case(
            generic_diffusion_demo_mms_neumann_model,
            n,
            inlet_outlet_wall_sides(),
            &source,
            &|solver, mesh| {
                // Outward normal gradient on bottom (n = (0,-1)) and top (n = (0,1)):
                // both evaluate to -pi sin(pi x) for this solution.
                let fx = mesh.face_cx.clone();
                let neumann = move |face_idx: u32| {
                    (-PI * (PI * fx[face_idx as usize]).sin()) as f32
                };
                solver
                    .set_boundary_values_per_face(GpuBoundaryType::Wall, "phi", 0, &neumann)
                    .expect("per-face neumann");
            },
        );
        hs.push(1.0 / n as f64);
        errs.push(field_errors(&mesh, &phi, exact).l2);
    }
    assert_convergence_order("steady_neumann_mix", &hs, &errs, 2.0, 0.25, 5e-4);
}

/// Temporal order of BDF2: spatially uniform phi*(t) = exp(-t) (spatial operators are
/// exact for constants, so the measured error is purely temporal), S(t) = -exp(-t).
/// All-wall (zero-gradient) box; source re-uploaded at t^{n+1} before each step.
#[test]
fn transient_bdf2_temporal_second_order() {
    let t_end = 1.0f64;
    let n = 8usize;

    let mut dts = Vec::new();
    let mut errs = Vec::new();
    for steps in [5usize, 10, 20, 40] {
        let dt = t_end / steps as f64;
        let mesh = unit_square_mesh(n, BoundarySides::wall());
        let model = generic_diffusion_demo_mms_model().expect("model");
        let mut solver = build_solver(&mesh, model);
        solver.set_time_scheme(TimeScheme::BDF2);
        solver.set_dt(dt as f32);
        solver
            .set_field_scalar("phi", &vec![1.0; mesh.num_cells()])
            .expect("init phi");
        solver.initialize_history();

        for k in 0..steps {
            let t_next = (k as f64 + 1.0) * dt;
            let s = -(-t_next).exp();
            solver
                .set_field_scalar_current(MMS_SOURCE_FIELD, &vec![s; mesh.num_cells()])
                .expect("upload source");
            solver.step();
        }

        let phi = pollster::block_on(solver.get_field_scalar("phi")).expect("read phi");
        let exact = (-t_end).exp();
        let err = field_errors(&mesh, &phi, |_x, _y| exact);
        dts.push(dt);
        errs.push(err.l2);
    }
    assert_convergence_order("transient_bdf2", &dts, &errs, 2.0, 0.3, 5e-4);
}


/// Implicit Euler with a per-step re-uploaded source must reproduce the backward-Euler
/// quadrature exactly: phi(1) = 1 - sum dt*exp(-t_{n+1}). Guards the history-preserving
/// `set_field_scalar_current` path for single-step schemes.
#[test]
fn euler_time_varying_source_is_exact() {
    let mesh = unit_square_mesh(8, BoundarySides::wall());
    let model = generic_diffusion_demo_mms_model().expect("model");
    let mut solver = build_solver(&mesh, model);
    let dt = 0.2f64;
    solver.set_dt(dt as f32);
    solver
        .set_field_scalar("phi", &vec![1.0; mesh.num_cells()])
        .expect("init phi");
    solver.initialize_history();
    for k in 0..5 {
        let t_next = (k as f64 + 1.0) * dt;
        solver
            .set_field_scalar_current(MMS_SOURCE_FIELD, &vec![-(-t_next).exp(); mesh.num_cells()])
            .expect("source");
        solver.step();
    }
    let phi = pollster::block_on(solver.get_field_scalar("phi")).expect("read");
    let expected = 1.0 - dt * (1..=5).map(|k| (-(k as f64) * dt).exp()).sum::<f64>();
    let min = phi.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = phi.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("[mms][euler_varying] after 5 Euler steps: min={min:.6} max={max:.6} expected={expected:.6}");
    assert!((min - expected).abs() < 1e-4 && (max - expected).abs() < 1e-4,
        "expected uniform {expected}, got [{min}, {max}]");
}


/// BDF2 is exact for solutions linear in time: with S=-1, expect phi = 1 - t at every
/// step (first step falls back to Euler, which is also exact here).
#[test]
fn bdf2_constant_source_is_exact() {
    let mesh = unit_square_mesh(8, BoundarySides::wall());
    let model = generic_diffusion_demo_mms_model().expect("model");
    let mut solver = build_solver(&mesh, model);
    solver.set_time_scheme(TimeScheme::BDF2);
    solver.set_dt(0.25);
    solver
        .set_field_scalar("phi", &vec![1.0; mesh.num_cells()])
        .expect("init phi");
    solver.initialize_history();
    solver
        .set_field_scalar(MMS_SOURCE_FIELD, &vec![-1.0; mesh.num_cells()])
        .expect("source");
    let mut got = Vec::new();
    for _ in 0..4 {
        solver.step();
        got.push(pollster::block_on(solver.get_field_scalar("phi")).expect("read")[0]);
    }
    println!("[mms][bdf2_constant] phi after steps: {got:?} (expect [0.75, 0.5, 0.25, 0.0])");
    for (k, expect) in [0.75, 0.5, 0.25, 0.0].iter().enumerate() {
        assert!((got[k] - expect).abs() < 1e-5, "step {}: got {} expect {}", k + 1, got[k], expect);
    }
}

/// BDF2 with a source re-uploaded each step must match the exact BDF2 recurrence
/// computed on the host. Guards against mid-run state writes clobbering the time
/// history (the original `set_field_scalar` resets all ping-pong buffers; the
/// `_current` variant must not).
#[test]
fn bdf2_time_varying_source_matches_recurrence() {
    let mesh = unit_square_mesh(8, BoundarySides::wall());
    let model = generic_diffusion_demo_mms_model().expect("model");
    let mut solver = build_solver(&mesh, model);
    solver.set_time_scheme(TimeScheme::BDF2);
    let dt = 0.2f64;
    solver.set_dt(dt as f32);
    solver
        .set_field_scalar("phi", &vec![1.0; mesh.num_cells()])
        .expect("init phi");
    solver.initialize_history();

    let s_at = |t: f64| -(-t).exp();
    let mut expect_prev_prev = 1.0f64; // phi^{n-1}
    let mut expect_prev = 1.0f64; // phi^n
    let mut report = Vec::new();
    for k in 0..5 {
        let t_next = (k as f64 + 1.0) * dt;
        solver
            .set_field_scalar_current(MMS_SOURCE_FIELD, &vec![s_at(t_next); mesh.num_cells()])
            .expect("source");
        solver.step();
        let got = pollster::block_on(solver.get_field_scalar("phi")).expect("read")[0];
        let expect = if k == 0 {
            expect_prev + dt * s_at(t_next) // Euler startup
        } else {
            (4.0 * expect_prev - expect_prev_prev + 2.0 * dt * s_at(t_next)) / 3.0
        };
        report.push((k + 1, got, expect, got - expect));
        expect_prev_prev = expect_prev;
        expect_prev = expect;
    }
    for (step, got, expect, diff) in &report {
        println!("[mms][bdf2_varying] step {step}: got={got:.6} expect={expect:.6} diff={diff:+.3e}");
    }
    let max_diff = report.iter().map(|r| r.3.abs()).fold(0.0f64, f64::max);
    assert!(max_diff < 1e-5, "BDF2 varying-source deviates from recurrence: {report:?}");
}
