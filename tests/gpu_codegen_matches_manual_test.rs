use cfd2::solver::gpu::enums::TimeScheme;
use cfd2::solver::gpu::structs::PreconditionerType;
use cfd2::solver::gpu::{GpuUnifiedSolver, SolverConfig};
use cfd2::solver::mesh::{generate_cut_cell_mesh, ChannelWithObstacle, Mesh};
use cfd2::solver::model::incompressible_momentum_model;
use cfd2::solver::scheme::Scheme;
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

fn run_solver(mesh: &Mesh, steps: usize) -> (Vec<(f64, f64)>, Vec<f64>) {
    let mut solver = pollster::block_on(GpuUnifiedSolver::new(
        mesh,
        incompressible_momentum_model(),
        SolverConfig {
            advection_scheme: Scheme::Upwind,
            time_scheme: TimeScheme::Euler,
            preconditioner: PreconditionerType::Jacobi,
        },
        None,
        None,
    ))
    .expect("solver init");
    solver.set_dt(0.01);
    solver.set_viscosity(0.01);
    solver.set_density(1.0);
    solver.set_inlet_velocity(1.0);
    solver.set_ramp_time(0.001);

    let u_init = vec![(0.0, 0.0); mesh.num_cells()];
    solver.set_u(&u_init);
    solver.set_p(&vec![0.0; mesh.num_cells()]);
    solver.initialize_history();

    for _ in 0..steps {
        solver.step();
    }

    let u = pollster::block_on(solver.get_u());
    let p = pollster::block_on(solver.get_p());
    (u, p)
}

#[test]
fn gpu_codegen_kernels_run_nontrivial() {
    let mesh = build_mesh();

    let steps = 5;
    let (u, p) = run_solver(&mesh, steps);

    let mut max_u_mag = 0.0f64;
    for value in &u {
        assert!(value.0.is_finite());
        assert!(value.1.is_finite());
        max_u_mag = max_u_mag.max((value.0 * value.0 + value.1 * value.1).sqrt());
    }

    for value in &p {
        assert!(value.is_finite());
    }

    let nontrivial_tol = 1e-4;
    assert!(
        max_u_mag > nontrivial_tol,
        "max u magnitude {:.6e} is below nontrivial threshold {:.6e}",
        max_u_mag,
        nontrivial_tol
    );
}
