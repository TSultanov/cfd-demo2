use cfd2::solver::gpu::GpuCompressibleSolver;
use cfd2::solver::mesh::geometry::ChannelWithObstacle;
use cfd2::solver::mesh::generate_cut_cell_mesh;
use nalgebra::{Point2, Vector2};

#[test]
fn gpu_compressible_amg_smoke() {
    let length = 1.0;
    let height = 0.5;
    let domain_size = Vector2::new(length, height);
    let geo = ChannelWithObstacle {
        length,
        height,
        obstacle_center: Point2::new(0.4, 0.25),
        obstacle_radius: 0.05,
    };
    let mesh = generate_cut_cell_mesh(&geo, 0.2, 0.2, 1.0, domain_size);

    let mut solver = pollster::block_on(GpuCompressibleSolver::new(&mesh, None, None));
    solver.set_dt(0.01);
    solver.set_dtau(0.0);
    solver.set_time_scheme(1);
    solver.set_viscosity(0.01);
    solver.set_inlet_velocity(0.5);
    solver.set_scheme(2);
    solver.set_alpha_u(0.5);
    solver.set_precond_type(1);
    solver.set_outer_iters(2);

    let rho_init = vec![1.0f32; mesh.num_cells()];
    let p_init = vec![1.0f32; mesh.num_cells()];
    let u_init = vec![[0.0f32, 0.0f32]; mesh.num_cells()];
    solver.set_state_fields(&rho_init, &u_init, &p_init);
    solver.initialize_history();

    for _ in 0..2 {
        solver.step();
    }

    let p = pollster::block_on(solver.get_p());
    assert!(p.iter().all(|v| v.is_finite()));
}
