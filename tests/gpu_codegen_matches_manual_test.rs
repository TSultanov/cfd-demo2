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

fn run_solver(mesh: &Mesh, variant: ShaderVariant) -> (Vec<(f64, f64)>, Vec<f64>) {
    let mut solver = pollster::block_on(GpuSolver::new_with_shader_variant(
        mesh, None, None, variant,
    ));
    solver.set_dt(0.01);
    solver.set_viscosity(0.01);
    solver.set_density(1.0);
    solver.set_scheme(0);
    solver.n_outer_correctors = 2;

    let mut u_init = vec![(0.0, 0.0); mesh.num_cells()];
    for (i, u_val) in u_init.iter_mut().enumerate() {
        if mesh.cell_cx[i] < 0.5 {
            *u_val = (1.0, 0.0);
        }
    }
    solver.set_u(&u_init);
    solver.set_p(&vec![0.0; mesh.num_cells()]);
    solver.initialize_history();

    solver.step();

    let u = pollster::block_on(solver.get_u());
    let p = pollster::block_on(solver.get_p());
    (u, p)
}

#[test]
fn gpu_generated_kernels_match_manual() {
    let mesh = build_mesh();

    let (u_manual, p_manual) = run_solver(&mesh, ShaderVariant::Manual);
    let (u_codegen, p_codegen) = run_solver(&mesh, ShaderVariant::Generated);

    assert_eq!(u_manual.len(), u_codegen.len());
    assert_eq!(p_manual.len(), p_codegen.len());

    let mut max_u_diff = 0.0f64;
    for (manual, codegen) in u_manual.iter().zip(u_codegen.iter()) {
        let dx = (manual.0 - codegen.0).abs();
        let dy = (manual.1 - codegen.1).abs();
        max_u_diff = max_u_diff.max(dx.max(dy));
    }

    let mut max_p_diff = 0.0f64;
    for (manual, codegen) in p_manual.iter().zip(p_codegen.iter()) {
        max_p_diff = max_p_diff.max((manual - codegen).abs());
    }

    let tol = 1e-5;
    assert!(
        max_u_diff < tol,
        "max u diff {:.6e} exceeds tolerance {:.6e}",
        max_u_diff,
        tol
    );
    assert!(
        max_p_diff < tol,
        "max p diff {:.6e} exceeds tolerance {:.6e}",
        max_p_diff,
        tol
    );
}
