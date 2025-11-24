use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep};
use cfd2::solver::piso::PisoSolver;
use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::Vector2;

fn cpu_solver_step_benchmark(c: &mut Criterion) {
    let length = 3.5;
    let domain_size = Vector2::new(length, 1.0);
    let geo = BackwardsStep {
        length,
        height_inlet: 0.5,
        height_outlet: 1.0,
        step_x: 0.5,
    };

    let mut mesh = generate_cut_cell_mesh(&geo, 0.05, 0.05, domain_size);
    mesh.smooth(&geo, 0.3, 50);

    let mut solver = PisoSolver::new(mesh);
    solver.dt = 0.01;
    solver.viscosity = 0.001;
    solver.density = 1000.0;

    let n_cells = solver.mesh.num_cells();
    for i in 0..n_cells {
        let cx = solver.mesh.cell_cx[i];
        let cy = solver.mesh.cell_cy[i];
        if cx < 0.05 && cy > 0.5 {
            solver.u.vx[i] = 1.0;
            solver.u.vy[i] = 0.0;
        } else {
            solver.u.vx[i] = 0.0;
            solver.u.vy[i] = 0.0;
        }
    }

    let mut group = c.benchmark_group("cpu_solver_step");
    group.sample_size(10);
    group.bench_function("step", |b| {
        b.iter(|| {
            solver.step();
        });
    });
    group.finish();
}

criterion_group!(benches, cpu_solver_step_benchmark);
criterion_main!(benches);
