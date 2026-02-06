#![cfg(all(feature = "meshgen", feature = "dev-tests"))]

use cfd2::solver::mesh::{generate_cut_cell_mesh, ChannelWithObstacle, Mesh};
use cfd2::solver::model::compressible_model;
use cfd2::solver::model::helpers::{
    SolverCompressibleIdealGasExt, SolverCompressibleInletExt, SolverFieldAliasesExt,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};
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
    let config = SolverConfig {
        advection_scheme: Scheme::SecondOrderUpwind,
        time_scheme: TimeScheme::BDF2,
        preconditioner: PreconditionerType::Jacobi,
        stepping: SteppingMode::Implicit { outer_iters: 1 },
    };
    let mut solver = pollster::block_on(UnifiedSolver::new(
        &mesh,
        compressible_model(),
        config,
        None,
        None,
    ))
    .expect("solver init");

    solver.set_dt(0.01);
    let eos = solver.model().eos();
    solver
        .set_compressible_inlet_isothermal_x(1.0, 0.0, &eos)
        .unwrap();
    solver.set_uniform_state(1.0f32, [0.0f32, 0.0f32], 1.0f32);
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
