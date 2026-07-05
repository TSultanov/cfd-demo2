/// Fusion Policy Wall-Clock Benchmark
///
/// Measures step wall-clock time for the three `KernelFusionPolicy` levels
/// (`Off`, `Safe`, `Aggressive`) and for the one-submission vs multi-submission
/// outer-loop paths on the `incompressible_momentum` model.
///
/// Run:
///   cargo bench --bench fusion_policy_benchmark --features meshgen
///
/// Criterion automatically generates HTML reports in `target/criterion/`.
/// Compare policies:
///   cargo bench --bench fusion_policy_benchmark --features meshgen -- fusion_policy
/// Compare submission paths:
///   cargo bench --bench fusion_policy_benchmark --features meshgen -- submission_path
use cfd2::solver::gpu::unified_solver::{GpuUnifiedSolver, SolverConfig};
use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep};
use cfd2::solver::model::helpers::{SolverInletVelocityExt, SolverRuntimeParamsExt};
use cfd2::solver::model::incompressible_momentum_model;
use cfd2::solver::model::kernel::KernelFusionPolicy;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SteppingMode, TimeScheme};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use nalgebra::Vector2;
use std::time::Duration;

fn setup_solver(policy: KernelFusionPolicy) -> GpuUnifiedSolver {
    let length = 3.5;
    let domain_size = Vector2::new(length, 1.0);
    let geo = BackwardsStep {
        length,
        height_inlet: 0.5,
        height_outlet: 1.0,
        step_x: 0.5,
    };

    let mut mesh = generate_cut_cell_mesh(&geo, 0.02, 0.02, 1.2, domain_size);
    mesh.smooth(&geo, 0.3, 50);

    let mut model = incompressible_momentum_model().expect("model");
    let mut linear_solver = model
        .linear_solver
        .expect("incompressible model missing linear solver");
    linear_solver.solver.kernel_fusion_policy = policy;
    model.linear_solver = Some(linear_solver);

    let config = SolverConfig {
        advection_scheme: Scheme::Upwind,
        time_scheme: TimeScheme::BDF2,
        preconditioner: PreconditionerType::Jacobi,
        stepping: SteppingMode::Coupled,
    };

    let mut solver = pollster::block_on(GpuUnifiedSolver::new(&mesh, model, config, None, None))
        .expect("solver init");

    solver.set_dt(0.02);
    solver.set_dtau(0.0).expect("set dtau");
    solver.set_density(1.0).expect("set density");
    solver.set_viscosity(0.01).expect("set viscosity");
    solver.set_inlet_velocity(1.0).expect("set inlet velocity");
    solver.set_alpha_u(0.7).expect("set alpha_u");
    solver.set_alpha_p(0.3).expect("set alpha_p");
    solver.set_outer_iters(8).expect("set outer iters");
    solver.initialize_history();

    solver
}

fn setup_solver_batched(policy: KernelFusionPolicy, outer_batched_mode: bool) -> GpuUnifiedSolver {
    let length = 3.5;
    let domain_size = Vector2::new(length, 1.0);
    let geo = BackwardsStep {
        length,
        height_inlet: 0.5,
        height_outlet: 1.0,
        step_x: 0.5,
    };

    let mut mesh = generate_cut_cell_mesh(&geo, 0.02, 0.02, 1.2, domain_size);
    mesh.smooth(&geo, 0.3, 50);

    let mut model = incompressible_momentum_model().expect("model");
    let mut linear_solver = model
        .linear_solver
        .expect("incompressible model missing linear solver");
    linear_solver.solver.kernel_fusion_policy = policy;
    model.linear_solver = Some(linear_solver);

    let config = SolverConfig {
        advection_scheme: Scheme::Upwind,
        time_scheme: TimeScheme::BDF2,
        preconditioner: PreconditionerType::Jacobi,
        stepping: SteppingMode::Coupled,
    };

    let mut solver = pollster::block_on(GpuUnifiedSolver::new(&mesh, model, config, None, None))
        .expect("solver init");

    solver.set_dt(0.02);
    solver.set_dtau(0.0).expect("set dtau");
    solver.set_density(1.0).expect("set density");
    solver.set_viscosity(0.01).expect("set viscosity");
    solver.set_inlet_velocity(1.0).expect("set inlet velocity");
    solver.set_alpha_u(0.7).expect("set alpha_u");
    solver.set_alpha_p(0.3).expect("set alpha_p");
    solver.set_outer_iters(8).expect("set outer iters");
    solver
        .set_outer_tolerance(0.0)
        .expect("set outer tolerance");
    solver
        .set_outer_tolerance_abs(0.0)
        .expect("set outer tolerance abs");
    solver
        .set_outer_fixed_iterations_mode(true)
        .expect("set outer fixed-iterations mode");
    solver
        .set_outer_batched_mode(outer_batched_mode)
        .expect("set outer batched mode");
    solver.initialize_history();

    solver
}

/// Benchmark step wall-clock time for Off, Safe, and Aggressive fusion policies.
fn bench_fusion_policy(c: &mut Criterion) {
    std::env::set_var("CFD2_QUIET", "1");

    let mut group = c.benchmark_group("fusion_policy");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    let policies = [
        ("Off", KernelFusionPolicy::Off),
        ("Safe", KernelFusionPolicy::Safe),
        ("Aggressive", KernelFusionPolicy::Aggressive),
    ];

    for (name, policy) in &policies {
        let mut solver = setup_solver(*policy);
        // Warm up: run 2 steps to stabilise GPU state.
        for _ in 0..2 {
            solver.step();
        }

        group.bench_function(BenchmarkId::new("5_steps", name), |b| {
            b.iter(|| {
                for _ in 0..5 {
                    solver.step();
                }
            });
        });
    }

    group.finish();
}

/// Benchmark one-submission vs multi-submission (host-driven) outer loop.
fn bench_submission_path(c: &mut Criterion) {
    std::env::set_var("CFD2_QUIET", "1");
    // Clear tuning knobs.
    std::env::remove_var("CFD2_ONE_SUBMISSION_SOLUTION_OMEGA");
    std::env::remove_var("CFD2_ONE_SUBMISSION_TAIL_OMEGA");
    std::env::remove_var("CFD2_ONE_SUBMISSION_CHUNKS");
    std::env::remove_var("CFD2_ONE_SUBMISSION_RESTART_BUDGET");
    std::env::remove_var("CFD2_ONE_SUBMISSION_TOTAL_ITERS");
    std::env::remove_var("CFD2_ONE_SUBMISSION_MIN_TAIL");

    let mut group = c.benchmark_group("submission_path");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    let paths = [("host_driven", false), ("one_submission", true)];

    for (name, batched) in &paths {
        let mut solver = setup_solver_batched(KernelFusionPolicy::Safe, *batched);
        // Warm up.
        for _ in 0..2 {
            solver.step();
        }

        group.bench_function(BenchmarkId::new("5_steps", name), |b| {
            b.iter(|| {
                for _ in 0..5 {
                    solver.step();
                }
            });
        });
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_fusion_policy, bench_submission_path
}

criterion_main!(benches);
