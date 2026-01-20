#![cfg(all(feature = "meshgen", feature = "dev-tests"))]

use cfd2::solver::mesh::generate_cut_cell_mesh;
use cfd2::solver::mesh::ChannelWithObstacle;
use cfd2::solver::model::compressible_model;
use cfd2::solver::model::helpers::{
    SolverCompressibleIdealGasExt, SolverCompressibleInletExt, SolverFieldAliasesExt,
    SolverRuntimeParamsExt,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};
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

    let mut solver = pollster::block_on(UnifiedSolver::new(
        &mesh,
        compressible_model(),
        SolverConfig {
            advection_scheme: Scheme::QUICK,
            time_scheme: TimeScheme::BDF2,
            preconditioner: PreconditionerType::Amg,
            stepping: SteppingMode::Implicit { outer_iters: 1 },
        },
        None,
        None,
    ))
    .expect("solver init");
    solver.set_dt(0.01);
    solver.set_dtau(0.0).unwrap();
    solver.set_viscosity(0.01).unwrap();
    let eos = solver.model().eos();
    solver
        .set_compressible_inlet_isothermal_x(1.0, 0.5, &eos)
        .unwrap();
    solver.set_outer_iters(2).unwrap();

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
