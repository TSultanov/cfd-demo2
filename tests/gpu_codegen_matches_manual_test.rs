use cfd2::solver::gpu::init::ShaderVariant;
use cfd2::solver::gpu::GpuSolver;
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

fn run_solver(mesh: &Mesh, variant: ShaderVariant, steps: usize) -> (Vec<(f64, f64)>, Vec<f64>) {
    let mut solver = pollster::block_on(GpuSolver::new_with_shader_variant(
        mesh, None, None, variant,
    ));
    solver.set_dt(0.01);
    solver.set_viscosity(0.01);
    solver.set_density(1.0);
    solver.set_scheme(0);
    solver.set_inlet_velocity(1.0);
    solver.set_ramp_time(0.001);
    solver.n_outer_correctors = 2;

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
fn gpu_generated_kernels_match_manual() {
    let mesh = build_mesh();

    let steps = 5;
    let (u_manual, p_manual) = run_solver(&mesh, ShaderVariant::Manual, steps);
    let (u_codegen, p_codegen) = run_solver(&mesh, ShaderVariant::Generated, steps);

    assert_eq!(u_manual.len(), u_codegen.len());
    assert_eq!(p_manual.len(), p_codegen.len());

    let mut max_u_diff = 0.0f64;
    let mut max_u_mag = 0.0f64;
    for (manual, codegen) in u_manual.iter().zip(u_codegen.iter()) {
        let dx = (manual.0 - codegen.0).abs();
        let dy = (manual.1 - codegen.1).abs();
        max_u_diff = max_u_diff.max(dx.max(dy));
        max_u_mag = max_u_mag.max((manual.0 * manual.0 + manual.1 * manual.1).sqrt());
    }

    let mut max_p_diff = 0.0f64;
    for (manual, codegen) in p_manual.iter().zip(p_codegen.iter()) {
        max_p_diff = max_p_diff.max((manual - codegen).abs());
    }

    let u_tol = 8e-6;
    let p_tol = 1.5e-4;
    let nontrivial_tol = 1e-4;
    assert!(
        max_u_diff < u_tol,
        "max u diff {:.6e} exceeds tolerance {:.6e}",
        max_u_diff,
        u_tol
    );
    assert!(
        max_u_mag > nontrivial_tol,
        "max u magnitude {:.6e} is below nontrivial threshold {:.6e}",
        max_u_mag,
        nontrivial_tol
    );
    assert!(
        max_p_diff < p_tol,
        "max p diff {:.6e} exceeds tolerance {:.6e}",
        max_p_diff,
        p_tol
    );
}
