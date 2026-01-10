/// GPU Dispatch Performance Benchmark
///
/// This benchmark measures GPU solver performance with focus on dispatch operations.
/// It provides a baseline for tracking performance improvements when optimizing
/// the number of GPU dispatches.
use cfd2::solver::gpu::structs::PreconditionerType;
use cfd2::solver::gpu::unified_solver::{GpuUnifiedSolver, SolverConfig};
use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep};
use cfd2::solver::model::incompressible_momentum_model;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nalgebra::Vector2;

/// Setup a solver for benchmarking
fn setup_solver(cell_size: f64) -> (GpuUnifiedSolver, usize) {
    let length = 3.5;
    let domain_size = Vector2::new(length, 1.0);
    let geo = BackwardsStep {
        length,
        height_inlet: 0.5,
        height_outlet: 1.0,
        step_x: 0.5,
    };

    let mut mesh = generate_cut_cell_mesh(&geo, cell_size, cell_size, 1.2, domain_size);
    mesh.smooth(&geo, 0.3, 50);

    let num_cells = mesh.num_cells();

    let model = incompressible_momentum_model();
    let config = SolverConfig::default();
    let mut solver = pollster::block_on(GpuUnifiedSolver::new(&mesh, model, config, None, None))
        .expect("should create solver");
    solver.set_dt(0.001);
    solver.set_viscosity(0.001);
    solver.set_density(1.0);
    solver.set_alpha_p(0.3);
    solver.set_alpha_u(0.7);
    // Note: scheme is set via config in the new API

    solver.initialize_history();

    (solver, num_cells)
}

/// Benchmark single step performance with medium mesh
fn bench_gpu_step_medium(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_dispatch");
    group.sample_size(10);

    // Medium mesh - good balance for benchmarking
    let cell_size = 0.02;
    let (mut solver, num_cells) = setup_solver(cell_size);

    group.throughput(Throughput::Elements(num_cells as u64));
    group.bench_function(BenchmarkId::new("step_medium", num_cells), |b| {
        b.iter(|| {
            solver.step();
            if false {
                panic!("Solver stopped due to degenerate solution!");
            }
        });
    });

    group.finish();
}

/// Benchmark total runtime for multiple steps (tracks overall performance)
fn bench_gpu_total_runtime(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_total_runtime");
    group.sample_size(10);

    let cell_size = 0.02;
    let (mut solver, num_cells) = setup_solver(cell_size);

    // Benchmark 5 steps together to measure total runtime
    let num_steps = 5;
    group.throughput(Throughput::Elements((num_cells * num_steps) as u64));
    group.bench_function(BenchmarkId::new("5_steps", num_cells), |b| {
        b.iter(|| {
            for _ in 0..num_steps {
                solver.step();
                if false {
                    panic!("Solver stopped due to degenerate solution!");
                }
            }
        });
    });

    // Benchmark 10 steps for longer runs
    let num_steps = 10;
    group.throughput(Throughput::Elements((num_cells * num_steps) as u64));
    group.bench_function(BenchmarkId::new("10_steps", num_cells), |b| {
        b.iter(|| {
            for _ in 0..num_steps {
                solver.step();
                if false {
                    panic!("Solver stopped due to degenerate solution!");
                }
            }
        });
    });

    group.finish();
}

/// Benchmark single step performance across different mesh sizes
fn bench_gpu_step_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_dispatch_scaling");
    group.sample_size(10);

    // Test with different mesh sizes
    let cell_sizes = [0.05, 0.02, 0.01];

    for &cell_size in &cell_sizes {
        let (mut solver, num_cells) = setup_solver(cell_size);

        group.throughput(Throughput::Elements(num_cells as u64));
        group.bench_function(BenchmarkId::new("step", num_cells), |b| {
            b.iter(|| {
                solver.step();
            });
        });
    }

    group.finish();
}

/// Benchmark with profiling enabled to track dispatch statistics
fn bench_with_profiling(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_dispatch_profiled");
    group.sample_size(10);

    let cell_size = 0.02;
    let (mut solver, num_cells) = setup_solver(cell_size);
    solver
        .enable_detailed_profiling(true)
        .expect("profiling enable");

    group.throughput(Throughput::Elements(num_cells as u64));
    group.bench_function(BenchmarkId::new("step_profiled", num_cells), |b| {
        b.iter(|| {
            solver.start_profiling_session();
            solver.step();
            solver.end_profiling_session();
        });
    });

    // Print final profiling stats after benchmark
    solver.start_profiling_session();
    solver.step();
    solver.end_profiling_session();
    solver.print_profiling_report();

    group.finish();
}

/// Benchmark preconditioner performance (Jacobi vs AMG)
fn bench_preconditioner_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("preconditioner_comparison");
    group.sample_size(10);

    let cell_size = 0.01;

    // Setup solver for Jacobi
    let (mut solver_jacobi, num_cells) = setup_solver(cell_size);
    solver_jacobi.set_preconditioner(PreconditionerType::Jacobi);
    // Warm up solver
    solver_jacobi.step();

    // Setup solver for AMG
    let (mut solver_amg, _) = setup_solver(cell_size);
    solver_amg.set_preconditioner(PreconditionerType::Amg);
    // Warmup AMG (allocates resources)
    solver_amg.step();

    group.throughput(Throughput::Elements(num_cells as u64));

    group.bench_function(BenchmarkId::new("jacobi", num_cells), |b| {
        b.iter(|| {
            solver_jacobi.step();
        });
    });

    group.bench_function(BenchmarkId::new("amg", num_cells), |b| {
        b.iter(|| {
            solver_amg.step();
        });
    });

    group.finish();
}

/// Benchmark fine mesh
fn bench_fine_mesh(c: &mut Criterion) {
    let mut group = c.benchmark_group("fine_mesh");
    group.sample_size(10);

    let cell_size = 0.00175;

    let (mut solver_default_order, num_cells) = setup_solver(cell_size);
    solver_default_order.set_preconditioner(PreconditionerType::Amg);
    solver_default_order.step();

    // let (mut solver_reordered, _) = setup_solver(cell_size, true);
    // solver_reordered.set_preconditioner(PreconditionerType::Amg);
    // solver_reordered.step();

    group.throughput(Throughput::Elements(num_cells as u64));

    group.bench_function(BenchmarkId::new("default_order", num_cells), |b| {
        b.iter(|| {
            solver_default_order.step();
        });
    });

    // group.bench_function(BenchmarkId::new("reordered", num_cells), |b| {
    //     b.iter(|| {
    //         solver_reordered.step();
    //     });
    // });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().measurement_time(std::time::Duration::from_secs(10));
    targets = bench_gpu_step_medium, bench_gpu_total_runtime, bench_gpu_step_scaling, bench_preconditioner_comparison, bench_fine_mesh
}

criterion_group! {
    name = profiled_benches;
    config = Criterion::default().sample_size(10);
    targets = bench_with_profiling
}

criterion_main!(benches, profiled_benches);

// Standalone function for quick dispatch analysis
// Run with: cargo bench --bench gpu_dispatch_benchmark -- --nocapture dispatch_stats
#[test]
fn dispatch_stats() {
    report_dispatch_stats();
}
