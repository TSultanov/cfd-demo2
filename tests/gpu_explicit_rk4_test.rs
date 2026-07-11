//! GPU integration gates for the matrix-free explicit RK4 path.
//!
//! The tests stay deliberately small: one unstructured pipeline compilation
//! covers both the exact constant-source seam and a temporal-order study, and
//! one structured compilation covers the corresponding direct GPU frontend.
//! A missing headless adapter is an allowed platform skip, matching the other
//! native GPU integration tests.
#![cfg(feature = "dev-tests")]

use cfd2::solver::gpu::context::GpuContext;
use cfd2::solver::gpu::structured::{BcComp, StructuredGpuSolver, StructuredGrid};
use cfd2::solver::gpu::unified_solver::PlanParamValue;
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides};
use cfd2::solver::model::helpers::SolverRuntimeParamsExt;
use cfd2::solver::model::{
    generic_diffusion_demo_mms_model, generic_diffusion_demo_structured_ibm_model,
    incompressible_momentum_model, MMS_SOURCE_FIELD,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};

fn gpu_context(gate: &str) -> Option<GpuContext> {
    match pollster::block_on(GpuContext::new(None, None)) {
        Ok(ctx) => Some(ctx),
        Err(error) => {
            eprintln!("[{gate}] no compatible GPU adapter ({error}); skipping GPU gate");
            None
        }
    }
}

fn observed_orders(dts: &[f64], errors: &[f64]) -> Vec<f64> {
    dts.windows(2)
        .zip(errors.windows(2))
        .map(|(h, e)| (e[0] / e[1]).ln() / (h[0] / h[1]).ln())
        .collect()
}

fn explicit_config() -> SolverConfig {
    SolverConfig {
        advection_scheme: Scheme::Upwind,
        time_scheme: TimeScheme::RK4,
        preconditioner: PreconditionerType::Jacobi,
        stepping: SteppingMode::Explicit,
    }
}

#[test]
fn gpu_unstructured_rk4_integrates_source_and_converges_at_order_four() {
    let Some(ctx) = gpu_context("gpu-rk4-unstructured") else {
        return;
    };
    let mesh = generate_structured_rect_mesh(4, 4, 1.0, 1.0, BoundarySides::wall());

    // The backend-independent capability gate must remain authoritative after
    // enabling the GPU runtime: a saddle-point pressure constraint is not an
    // explicit ODE and must still fail for its mathematical reason.
    let invalid = pollster::block_on(UnifiedSolver::new(
        &mesh,
        incompressible_momentum_model().expect("incompressible model"),
        explicit_config(),
        Some(ctx.device.clone()),
        Some(ctx.queue.clone()),
    ))
    .err()
    .expect("incompressible saddle-point model must reject explicit RK4");
    assert!(
        invalid.contains("d_p")
            || invalid.contains("Rhie")
            || invalid.contains("no own-variable ddt")
            || invalid.contains("not a method-of-lines row"),
        "unexpected explicit-capability rejection: {invalid}"
    );
    assert!(
        !invalid.contains("CPU matrix-free backend") && !invalid.contains("select the CPU backend"),
        "GPU RK4 must reach the model capability gate, not the retired backend gate: {invalid}"
    );

    let mut solver = pollster::block_on(UnifiedSolver::new(
        &mesh,
        generic_diffusion_demo_mms_model().expect("diffusion MMS model"),
        explicit_config(),
        Some(ctx.device.clone()),
        Some(ctx.queue.clone()),
    ))
    .expect("GPU explicit RK4 solver construction");
    assert!(
        !solver.is_cpu(),
        "the RK4 integration gate must exercise the GPU backend"
    );
    assert_eq!(solver.config().time_scheme, TimeScheme::RK4);
    assert_eq!(solver.config().stepping, SteppingMode::Explicit);
    assert!(
        pollster::block_on(solver.get_linear_rhs()).is_err(),
        "explicit GPU runtime must not expose an implicit linear system"
    );
    let dtau_error = solver
        .set_dtau(1.0e-3)
        .expect_err("explicit RK4 must reject pseudo-time stepping");
    assert!(dtau_error.contains("dtau") || dtau_error.contains("pseudo-time"));
    let switch_error = solver
        .try_set_time_scheme(TimeScheme::Euler)
        .expect_err("RK4-to-implicit switching requires a rebuilt GPU program");
    assert!(switch_error.contains("rebuild"));
    assert_eq!(solver.config().time_scheme, TimeScheme::RK4);
    let named_switch_error = solver
        .set_named_param("time_scheme", PlanParamValue::TimeScheme(TimeScheme::BDF2))
        .expect_err("the public named-param path must not bypass the RK4 family guard");
    assert!(named_switch_error.contains("rebuild"));
    assert_eq!(solver.config().time_scheme, TimeScheme::RK4);

    // A uniform field removes the spatial operator exactly. This pins the
    // residual sign, local ddt inversion, RK workspaces, and the public
    // step-with-stats path. Explicit stepping performs no linear solve.
    solver.set_dt(0.125);
    solver
        .set_field_scalar("phi", &vec![1.0; mesh.num_cells()])
        .expect("set phi");
    solver
        .set_field_scalar(MMS_SOURCE_FIELD, &vec![-2.0; mesh.num_cells()])
        .expect("set source");
    solver.initialize_history();
    let linear_stats = solver.step_with_stats().expect("explicit GPU step");
    assert!(
        linear_stats.is_empty(),
        "matrix-free RK4 must not report linear-solver work"
    );
    let phi = pollster::block_on(solver.get_field_scalar("phi")).expect("read phi");
    let max_error = phi
        .iter()
        .map(|&value| (value - 0.75).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_error < 2.0e-6,
        "uniform GPU RK4 source error = {max_error:e}"
    );

    // On this orthogonal 4x4 mesh, cos(pi*x)cos(pi*y) is an exact eigenvector
    // of the discrete zero-Neumann Laplacian. Comparing against that discrete
    // eigenvalue removes spatial truncation error. Crucially, this is a
    // state-dependent residual: evaluating it only once and then dispatching
    // all four stage updates degenerates to first order and fails this gate.
    let n = 4usize;
    let t_end = 0.2_f64;
    let lambda_h = 8.0 * (n * n) as f64 * (std::f64::consts::PI / (2.0 * n as f64)).sin().powi(2);
    let exact_scale = (-lambda_h * t_end).exp();
    let initial: Vec<f64> = (0..mesh.num_cells())
        .map(|cell| {
            (std::f64::consts::PI * mesh.cell_cx[cell]).cos()
                * (std::f64::consts::PI * mesh.cell_cy[cell]).cos()
        })
        .collect();
    let dts = [0.04_f64, 0.02, 0.01];
    let mut errors = Vec::with_capacity(dts.len());

    for &dt in &dts {
        solver.set_dt(dt as f32);
        solver.set_field_scalar("phi", &initial).expect("reset phi");
        solver
            .set_field_scalar(MMS_SOURCE_FIELD, &vec![0.0; mesh.num_cells()])
            .expect("clear source");
        solver.initialize_history();
        for _ in 0..(t_end / dt).round() as usize {
            solver.step();
        }
        let got =
            pollster::block_on(solver.get_field_scalar("phi")).expect("read temporal-order field");
        let l2 = (got
            .iter()
            .zip(initial.iter())
            .map(|(&value, &value0)| (value - exact_scale * value0).powi(2))
            .sum::<f64>()
            / mesh.num_cells() as f64)
            .sqrt();
        errors.push(l2);
    }

    let orders = observed_orders(&dts, &errors);
    eprintln!("[gpu-rk4-unstructured] dt={dts:?} error={errors:?} order={orders:?}");
    assert!(
        orders.iter().all(|&order| order > 3.5),
        "GPU RK4 temporal order below four: {orders:?}, errors={errors:?}"
    );

    // Matrix-free stepping has no linear solve whose residual could carry a
    // failure bit. The stats API must still surface a non-finite RK state on
    // the same step so the driver stops immediately.
    solver.set_dt(0.01);
    solver.set_field_scalar("phi", &initial).expect("reset phi");
    solver
        .set_field_scalar(MMS_SOURCE_FIELD, &vec![f64::NAN; mesh.num_cells()])
        .expect("set non-finite source");
    solver.initialize_history();
    let stats = solver
        .step_with_stats()
        .expect("non-finite explicit step must still return stats");
    assert!(
        stats.iter().any(|stat| stat.diverged),
        "non-finite GPU RK4 state was not reported as diverged"
    );
}

#[test]
fn gpu_structured_rk4_converges_at_order_four() {
    let Some(ctx) = gpu_context("gpu-rk4-structured") else {
        return;
    };
    let model =
        generic_diffusion_demo_structured_ibm_model().expect("structured reaction-diffusion model");
    let dts = [0.125_f64, 0.0625, 0.03125];
    let t_end = 0.5_f64;
    let exact = (-4.0 * t_end).exp();
    let mut solver = StructuredGpuSolver::with_config(
        ctx,
        StructuredGrid::new(4, 4, 1.0, 1.0),
        &model,
        dts[0],
        1,
        Scheme::Upwind,
        TimeScheme::RK4,
    )
    .expect("structured GPU explicit RK4 solver construction");
    let switch_error = solver
        .try_set_time_scheme(TimeScheme::BDF2)
        .expect_err("structured RK4-to-implicit switching requires reconstruction");
    assert!(switch_error.contains("rebuild"));
    solver.set_boundaries(|_edge, _x, _y| {
        (
            3,
            vec![BcComp {
                kind: 0,
                value: 0.0,
            }],
        )
    });

    // Uniform phi and zero-gradient faces remove diffusion, leaving the
    // declared autonomous ODE phi'=-4 phi. The reaction depends on the current
    // stage state, so this independently pins four residual evaluations in the
    // structured GPU frontend.
    let mut errors = Vec::with_capacity(dts.len());
    for &dt in &dts {
        solver.set_dt(dt);
        solver.set_named_field("phi", |_x, _y| 1.0);
        solver.set_named_field("ibm_penalty", |_x, _y| -4.0);
        for _ in 0..(t_end / dt).round() as usize {
            solver.step();
        }
        let phi = solver.state_field(solver.field_offset("phi").expect("phi offset"));
        let l2 = (phi
            .iter()
            .map(|&value| (value - exact).powi(2))
            .sum::<f64>()
            / phi.len() as f64)
            .sqrt();
        errors.push(l2);
    }

    let orders = observed_orders(&dts, &errors);
    eprintln!("[gpu-rk4-structured] dt={dts:?} error={errors:?} order={orders:?}");
    assert!(
        orders.iter().all(|&order| order > 3.5),
        "structured GPU RK4 temporal order below four: {orders:?}, errors={errors:?}"
    );
}
