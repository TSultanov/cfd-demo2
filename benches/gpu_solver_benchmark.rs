use cfd2::solver::gpu::GpuSolver;
use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep};
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

    let mut solver = pollster::block_on(GpuSolver::new(&mesh));
    solver.set_dt(0.01);
    solver.set_viscosity(0.001);
    solver.set_density(1000.0);
    solver.set_alpha_p(1.0);

    let mut u_init = vec![(0.0f64, 0.0f64); mesh.num_cells()];
    for i in 0..mesh.num_cells() {
        let cx = mesh.cell_cx[i];
        let cy = mesh.cell_cy[i];
        if cx < 0.05 && cy > 0.5 {
            u_init[i] = (1.0, 0.0);
        }
    }
    solver.set_u(&u_init);

    let mut group = c.benchmark_group("gpu_solver_step");
    group.sample_size(10);
    group.bench_function("step", |b| {
        b.iter(|| {
            solver.step();
            if solver.should_stop && solver.degenerate_count > 10 {
                panic!("Solver stopped due to degenerate solution!");
            }
        });
    });
    group.finish();
}

criterion_group!(benches, gpu_solver_step_benchmark);
criterion_main!(benches);
