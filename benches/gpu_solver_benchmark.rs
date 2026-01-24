use cfd2::solver::gpu::unified_solver::{GpuUnifiedSolver, SolverConfig};
use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep};
use cfd2::solver::model::incompressible_momentum_model;
use cfd2::solver::model::helpers::SolverRuntimeParamsExt;
use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::Vector2;

fn gpu_solver_step_benchmark(c: &mut Criterion) {
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

    let model = incompressible_momentum_model();
    let config = SolverConfig::default();
    let mut solver = pollster::block_on(GpuUnifiedSolver::new(&mesh, model, config, None, None))
        .expect("should create solver");
    solver.set_dt(0.01);
    solver.set_viscosity(0.001).unwrap();
    solver.set_density(1000.0).unwrap();
    solver.set_alpha_p(1.0).unwrap();

    // Note: Initial conditions are set via write_state_bytes in the new API
    // For simplicity in benchmarks, we skip custom initial conditions and use defaults
    solver.initialize_history();

    let mut group = c.benchmark_group("gpu_solver_step");
    group.sample_size(10);
    group.bench_function("step", |b| {
        b.iter(|| {
            solver.step();
        });
    });
    group.finish();
}

criterion_group!(benches, gpu_solver_step_benchmark);
criterion_main!(benches);
