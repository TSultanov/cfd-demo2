use cfd2::solver::gpu::GpuSolver;
use cfd2::solver::mesh::{generate_cut_cell_mesh, ChannelWithObstacle};
use cfd2::solver::piso::PisoSolver;
use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::{Point2, Vector2};

fn solver_comparison_benchmark(c: &mut Criterion) {
    let geo = ChannelWithObstacle {
        length: 2.0,
        height: 1.0,
        obstacle_center: Point2::new(0.5, 0.5),
        obstacle_radius: 0.1,
    };
    let domain_size = Vector2::new(2.0, 1.0);

    // 0.005 => ~80,000 cells
    let cell_size = 0.005;

    println!("Generating mesh with cell size {}...", cell_size);
    let mut mesh = generate_cut_cell_mesh(&geo, cell_size, cell_size, domain_size);
    mesh.smooth(&geo, 0.3, 10);
    println!("Mesh generated with {} cells", mesh.num_cells());

    // GPU Solver Setup
    println!("Setting up GPU solver...");
    let mut gpu_solver = pollster::block_on(GpuSolver::new(&mesh));
    gpu_solver.set_dt(0.001);
    gpu_solver.set_viscosity(0.001);
    gpu_solver.set_density(1000.0);
    gpu_solver.set_alpha_p(1.0);

    let mut u_init = vec![(0.0f64, 0.0f64); mesh.num_cells()];
    for i in 0..mesh.num_cells() {
        u_init[i] = (1.0, 0.0);
    }
    gpu_solver.set_u(&u_init);

    // CPU Solver Setup
    println!("Setting up CPU solver...");
    let mut cpu_solver = PisoSolver::new(mesh.clone());
    cpu_solver.dt = 0.001;
    cpu_solver.viscosity = 0.001;
    cpu_solver.density = 1000.0;

    for i in 0..mesh.num_cells() {
        cpu_solver.u.vx[i] = 1.0;
        cpu_solver.u.vy[i] = 0.0;
    }

    let mut group = c.benchmark_group("solver_comparison");
    group.sample_size(10);

    group.bench_function("gpu_step", |b| {
        b.iter(|| {
            gpu_solver.step();
        });
    });

    group.bench_function("cpu_step", |b| {
        b.iter(|| {
            cpu_solver.step();
        });
    });

    group.finish();
}

criterion_group!(benches, solver_comparison_benchmark);
criterion_main!(benches);
