//! MMS convergence verification for the scalar transport vertical slice
//! (`scalar_transport` / `scalar_transport_sou` models): advection-diffusion of a
//! passive scalar with a declared (not hand-built) advecting flux.
//!
//! Discrete equation: dT/dt + div(phi_adv T) - kappa lap(T) = S with kappa = 1 and
//! phi_adv derived from the declared advecting velocity U_adv. Manufactured source:
//! S = dT*/dt + U.grad(T*) - kappa lap(T*) (for divergence-free U).
#![cfg(feature = "dev-tests")]

#[path = "mms_support/mod.rs"]
mod mms_support;

use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType, Mesh};
use cfd2::solver::model::helpers::SolverRuntimeParamsExt;
use cfd2::solver::model::{
    scalar_transport_model, scalar_transport_sou_model, ModelSpec, ADVECTING_VELOCITY_FIELD,
    SCALAR_TRANSPORT_FIELD, SCALAR_TRANSPORT_KAPPA, SCALAR_TRANSPORT_MMS_SOURCE_FIELD,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{SolverConfig, TimeScheme, UnifiedSolver};
use mms_support::{assert_convergence_order, field_errors, fit_order, run_to_steady};
use std::f64::consts::PI;

const STEADY_TOL: f64 = 4e-6;
const STEADY_MAX_STEPS: usize = 400;

fn unit_square_mesh(n: usize) -> Mesh {
    generate_structured_rect_mesh(
        n,
        n,
        1.0,
        1.0,
        BoundarySides {
            left: BoundaryType::Inlet,
            right: BoundaryType::Outlet,
            bottom: BoundaryType::Wall,
            top: BoundaryType::Wall,
        },
    )
}

struct SteadyCase<'a> {
    model_fn: fn() -> Result<ModelSpec, String>,
    advection_scheme: Scheme,
    /// Runtime knob override applied after creation (tests knob-independence of
    /// declared schemes).
    knob_override: Option<Scheme>,
    velocity: &'a dyn Fn(f64, f64) -> (f64, f64),
    exact: &'a dyn Fn(f64, f64) -> f64,
    source: &'a dyn Fn(f64, f64) -> f64,
}

fn solve_steady(case: &SteadyCase, n: usize) -> (Mesh, Vec<f64>) {
    let mesh = unit_square_mesh(n);
    let model = (case.model_fn)().expect("model");
    let config = SolverConfig {
        advection_scheme: case.advection_scheme,
        ..SolverConfig::default()
    };
    let mut solver =
        pollster::block_on(UnifiedSolver::new(&mesh, model, config, None, None))
            .expect("create solver");
    solver.set_outer_iters(2).expect("outer_iters");
    if let Some(knob) = case.knob_override {
        solver.set_advection_scheme(knob);
    }
    // Advective time scale ~ L/|U|; dt = 0.2 keeps implicit marching stable and
    // contracts to steady state in a few dozen steps.
    solver.set_dt(0.2);

    // Per-face Dirichlet values from the exact solution on every boundary.
    let exact = case.exact;
    let fx = mesh.face_cx.clone();
    let fy = mesh.face_cy.clone();
    let face_value = move |face_idx: u32| {
        let i = face_idx as usize;
        exact(fx[i], fy[i]) as f32
    };
    for boundary in [
        GpuBoundaryType::Inlet,
        GpuBoundaryType::Outlet,
        GpuBoundaryType::Wall,
    ] {
        solver
            .set_boundary_values_per_face(boundary, SCALAR_TRANSPORT_FIELD, 0, &face_value)
            .expect("per-face dirichlet");
    }

    let u_values: Vec<(f64, f64)> = (0..mesh.num_cells())
        .map(|i| (case.velocity)(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    solver
        .set_field_vec2(ADVECTING_VELOCITY_FIELD, &u_values)
        .expect("set advecting velocity");

    let src: Vec<f64> = (0..mesh.num_cells())
        .map(|i| (case.source)(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    solver
        .set_field_scalar(SCALAR_TRANSPORT_MMS_SOURCE_FIELD, &src)
        .expect("set source");

    solver
        .set_field_scalar(SCALAR_TRANSPORT_FIELD, &vec![0.0; mesh.num_cells()])
        .expect("init T");
    solver.initialize_history();

    let t = run_to_steady(&mut solver, SCALAR_TRANSPORT_FIELD, STEADY_MAX_STEPS, STEADY_TOL);
    (mesh, t)
}

fn convergence_study(case: &SteadyCase, levels: &[usize]) -> (Vec<f64>, Vec<f64>) {
    let mut hs = Vec::new();
    let mut errs = Vec::new();
    for &n in levels {
        let (mesh, t) = solve_steady(case, n);
        hs.push(1.0 / n as f64);
        errs.push(field_errors(&mesh, &t, case.exact).l2);
    }
    (hs, errs)
}

// Manufactured case: T* = sin(pi x) cos(pi y), constant U = (4, 2) (face-exact flux),
// S = U.grad(T*) - kappa lap(T*).
fn constant_velocity() -> (f64, f64) {
    (4.0, 2.0)
}
fn exact_sincos(x: f64, y: f64) -> f64 {
    (PI * x).sin() * (PI * y).cos()
}
fn source_sincos_constant_u(x: f64, y: f64) -> f64 {
    let (ux, uy) = constant_velocity();
    let k = SCALAR_TRANSPORT_KAPPA;
    let dtdx = PI * (PI * x).cos() * (PI * y).cos();
    let dtdy = -PI * (PI * x).sin() * (PI * y).sin();
    let lap = -2.0 * PI * PI * exact_sincos(x, y);
    ux * dtdx + uy * dtdy - k * lap
}

/// Upwind convection + second-order diffusion: the measured order sits between 1
/// and 2 depending on which error dominates; it must converge and stay within the
/// mixed-order window.
#[test]
fn steady_advection_diffusion_upwind_converges() {
    let case = SteadyCase {
        model_fn: scalar_transport_model,
        advection_scheme: Scheme::Upwind,
        knob_override: None,
        velocity: &|_x, _y| constant_velocity(),
        exact: &exact_sincos,
        source: &source_sincos_constant_u,
    };
    // n=8 is preasymptotic for this advection-dominated case; fit on 16..128.
    // The observed order approaches 1 from below as the first-order advection error
    // overtakes the second-order diffusion error (measured ~0.92 at the fine end).
    let (hs, errs) = convergence_study(&case, &[16, 32, 64, 128]);
    assert_convergence_order("scalar_upwind", &hs, &errs, 1.0, 0.2, 1e-2);
    // Mixed advection-diffusion: order must not exceed ~2.3 either (sanity).
    let order = fit_order(&hs, &errs);
    assert!(order < 2.4, "implausible order {order:.2} for upwind advection");
}

/// Second-order upwind via the runtime knob (set at solver creation so gradient
/// machinery is wired): order ~2 and strictly more accurate than upwind.
#[test]
fn steady_advection_diffusion_knob_sou_second_order() {
    let upwind_case = SteadyCase {
        model_fn: scalar_transport_model,
        advection_scheme: Scheme::Upwind,
        knob_override: None,
        velocity: &|_x, _y| constant_velocity(),
        exact: &exact_sincos,
        source: &source_sincos_constant_u,
    };
    let sou_case = SteadyCase {
        model_fn: scalar_transport_model,
        advection_scheme: Scheme::SecondOrderUpwind,
        knob_override: None,
        velocity: &|_x, _y| constant_velocity(),
        exact: &exact_sincos,
        source: &source_sincos_constant_u,
    };
    let (hs, sou_errs) = convergence_study(&sou_case, &[8, 16, 32, 64]);
    assert_convergence_order("scalar_knob_sou", &hs, &sou_errs, 2.0, 0.3, 1e-3);

    let (_, upwind_errs) = convergence_study(&upwind_case, &[64]);
    println!(
        "[mms][scalar_knob_sou] finest: sou={:.3e} upwind={:.3e}",
        sou_errs.last().unwrap(),
        upwind_errs[0]
    );
    assert!(
        sou_errs.last().unwrap() < &upwind_errs[0],
        "SOU should beat upwind at the finest level"
    );
}

/// The scheme *declared* on the convection term (scalar_transport_sou) is baked
/// into the kernel: results match knob-SOU and are independent of the runtime knob.
#[test]
fn declared_sou_matches_knob_sou_and_ignores_knob() {
    let n = 32;

    let knob_sou = SteadyCase {
        model_fn: scalar_transport_model,
        advection_scheme: Scheme::SecondOrderUpwind,
        knob_override: None,
        velocity: &|_x, _y| constant_velocity(),
        exact: &exact_sincos,
        source: &source_sincos_constant_u,
    };
    // Declared SOU; runtime knob deliberately pointed at Upwind. The declared
    // scheme must win (the knob literal is baked out of the kernel).
    let declared_sou_knob_upwind = SteadyCase {
        model_fn: scalar_transport_sou_model,
        advection_scheme: Scheme::SecondOrderUpwind, // creation-time gradient wiring
        knob_override: Some(Scheme::Upwind),
        velocity: &|_x, _y| constant_velocity(),
        exact: &exact_sincos,
        source: &source_sincos_constant_u,
    };

    let (_, t_knob) = solve_steady(&knob_sou, n);
    let (mesh, t_declared) = solve_steady(&declared_sou_knob_upwind, n);

    let max_diff = t_knob
        .iter()
        .zip(t_declared.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    println!("[mms][declared_sou] max |knob_sou - declared_sou(knob=upwind)| = {max_diff:.3e}");
    assert!(
        max_diff < 1e-5,
        "declared SOU must match knob SOU regardless of the runtime knob (diff {max_diff:.3e})"
    );

    let err = field_errors(&mesh, &t_declared, &exact_sincos).l2;
    println!("[mms][declared_sou] l2 error at n={n}: {err:.3e}");
    assert!(err < 2e-3, "declared-SOU error unexpectedly large: {err:.3e}");
}

/// Rotating advecting field U = (-(y-1/2), x-1/2): exercises a genuinely
/// space-varying coupled field through the derived flux. SOU via knob.
#[test]
fn steady_rotating_field_advection_second_order() {
    let velocity = |x: f64, y: f64| (-(y - 0.5), x - 0.5);
    let exact = |x: f64, y: f64| (PI * x).sin() * (PI * y).sin();
    let source = move |x: f64, y: f64| {
        let (ux, uy) = velocity(x, y);
        let k = SCALAR_TRANSPORT_KAPPA;
        let dtdx = PI * (PI * x).cos() * (PI * y).sin();
        let dtdy = PI * (PI * x).sin() * (PI * y).cos();
        let lap = -2.0 * PI * PI * exact(x, y);
        ux * dtdx + uy * dtdy - k * lap
    };
    let case = SteadyCase {
        model_fn: scalar_transport_model,
        advection_scheme: Scheme::SecondOrderUpwind,
        knob_override: None,
        velocity: &velocity,
        exact: &exact,
        source: &source,
    };
    let (hs, errs) = convergence_study(&case, &[8, 16, 32, 64]);
    assert_convergence_order("scalar_rotating_sou", &hs, &errs, 2.0, 0.3, 1e-3);
}

/// BDF2 temporal order with advection machinery active: spatially uniform
/// T*(t) = exp(-t) (advection and diffusion of a uniform field vanish), constant
/// advecting velocity, time-varying Dirichlet boundaries and source.
#[test]
fn transient_bdf2_with_advection_second_order() {
    let t_end = 1.0f64;
    let n = 8usize;

    let mut dts = Vec::new();
    let mut errs = Vec::new();
    for steps in [5usize, 10, 20, 40] {
        let dt = t_end / steps as f64;
        let mesh = unit_square_mesh(n);
        let model = scalar_transport_model().expect("model");
        let mut solver = pollster::block_on(UnifiedSolver::new(
            &mesh,
            model,
            SolverConfig::default(),
            None,
            None,
        ))
        .expect("create solver");
        solver.set_outer_iters(2).expect("outer_iters");
        solver.set_time_scheme(TimeScheme::BDF2);
        solver.set_dt(dt as f32);

        let u_values = vec![constant_velocity(); mesh.num_cells()];
        solver
            .set_field_vec2(ADVECTING_VELOCITY_FIELD, &u_values)
            .expect("set advecting velocity");
        solver
            .set_field_scalar(SCALAR_TRANSPORT_FIELD, &vec![1.0; mesh.num_cells()])
            .expect("init T");
        solver.initialize_history();

        for k in 0..steps {
            let t_next = (k as f64 + 1.0) * dt;
            let value = (-t_next).exp();
            for boundary in [
                GpuBoundaryType::Inlet,
                GpuBoundaryType::Outlet,
                GpuBoundaryType::Wall,
            ] {
                solver
                    .set_boundary_scalar(boundary, SCALAR_TRANSPORT_FIELD, value as f32)
                    .expect("bc");
            }
            solver
                .set_field_scalar_current(
                    SCALAR_TRANSPORT_MMS_SOURCE_FIELD,
                    &vec![-value; mesh.num_cells()],
                )
                .expect("source");
            solver.step();
        }

        let t = pollster::block_on(solver.get_field_scalar(SCALAR_TRANSPORT_FIELD)).expect("read");
        let exact = (-t_end).exp();
        errs.push(field_errors(&mesh, &t, |_x, _y| exact).l2);
        dts.push(dt);
    }
    assert_convergence_order("scalar_transient_bdf2", &dts, &errs, 2.0, 0.3, 5e-4);
}

