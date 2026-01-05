use cfd2::solver::gpu::GpuCompressibleSolver;
use cfd2::solver::mesh::{generate_cut_cell_mesh, ChannelWithObstacle, Mesh};
use nalgebra::{Point2, Vector2};

fn build_mesh() -> Mesh {
    let length = 1.0;
    let height = 1.0;
    let domain_size = Vector2::new(length, height);
    let geo = ChannelWithObstacle {
        length,
        height,
        obstacle_center: Point2::new(0.5, 0.5),
        obstacle_radius: 0.1,
    };

    generate_cut_cell_mesh(&geo, 0.2, 0.2, 1.0, domain_size)
}

#[test]
fn gpu_compressible_solver_preserves_uniform_state() {
    let mesh = build_mesh();
    let mut solver = pollster::block_on(GpuCompressibleSolver::new(&mesh, None, None));

    solver.set_dt(0.01);
    solver.set_inlet_velocity(0.0);
    solver.set_scheme(1);
    solver.set_uniform_state(1.0, [0.0, 0.0], 1.0);
    solver.initialize_history();

    for _ in 0..3 {
        solver.step();
    }

    let rho = pollster::block_on(solver.get_rho());
    let u = pollster::block_on(solver.get_u());
    let p = pollster::block_on(solver.get_p());

    let tol = 1e-4;
    for value in rho {
        assert!((value - 1.0).abs() < tol);
    }
    for (ux, uy) in u {
        assert!(ux.abs() < tol);
        assert!(uy.abs() < tol);
    }
    for value in p {
        assert!((value - 1.0).abs() < tol);
    }
}
