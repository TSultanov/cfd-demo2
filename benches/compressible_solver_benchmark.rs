use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep};
use cfd2::solver::model::compressible_model_with_eos;
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::model::helpers::{
    SolverCompressibleIdealGasExt, SolverCompressibleInletExt, SolverRuntimeParamsExt,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nalgebra::Vector2;

fn setup_backstep_solver(cell_size: f64) -> UnifiedSolver {
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

    let density = 1.225f32;
    let viscosity = 1.81e-5f32;
    let inlet_u = 1.0f32;
    let eos = EosSpec::IdealGas {
        gamma: 1.4,
        gas_constant: 287.0,
        temperature: 300.0,
    };

    let mut solver = pollster::block_on(UnifiedSolver::new(
        &mesh,
        compressible_model_with_eos(eos),
        SolverConfig {
            advection_scheme: Scheme::Upwind,
            time_scheme: TimeScheme::Euler,
            preconditioner: PreconditionerType::Jacobi,
            stepping: SteppingMode::Implicit { outer_iters: 1 },
        },
        None,
        None,
    ))
    .expect("solver init");

    // Clear full state to avoid uninitialized auxiliary fields.
    let stride = solver.model().state_layout.stride() as usize;
    solver
        .write_state_f32(&vec![0.0f32; mesh.num_cells() * stride])
        .expect("clear state");

    solver.set_dt(0.001);
    solver.set_density(density).unwrap();
    solver.set_viscosity(viscosity).unwrap();
    solver.set_eos(&eos).unwrap();
    solver
        .set_compressible_inlet_isothermal_x(density, inlet_u, &eos)
        .unwrap();

    let p_ref = eos.pressure_for_density(density as f64) as f32;
    solver.set_uniform_state(density, [0.0, 0.0], p_ref);
    solver.initialize_history();

    solver
}

fn bench_compressible_backstep_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_compressible_step");
    group.sample_size(10);

    let cell_size = 0.05;
    let mut solver = setup_backstep_solver(cell_size);
    let num_cells = solver.num_cells();

    // Warm up.
    for _ in 0..3 {
        solver.step();
    }

    group.throughput(Throughput::Elements(num_cells as u64));
    group.bench_function(BenchmarkId::new("backstep", num_cells), |b| {
        b.iter(|| {
            solver.step();
        });
    });

    group.finish();
}

criterion_group!(benches, bench_compressible_backstep_step);
criterion_main!(benches);
