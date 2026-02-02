use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType};
use cfd2::solver::model::helpers::{
    SolverFieldAliasesExt, SolverInletVelocityExt, SolverRuntimeParamsExt,
};
use cfd2::solver::model::incompressible_momentum_model;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};

#[test]
#[ignore]
fn diag_incompressible_channel_outer_iters_per_step() {
    std::env::set_var("CFD2_QUIET", "1");

    let nx = 40usize;
    let ny = 20usize;
    let length = 1.0;
    let height = 0.2;
    let mesh = generate_structured_rect_mesh(
        nx,
        ny,
        length,
        height,
        BoundarySides {
            left: BoundaryType::Inlet,
            right: BoundaryType::Outlet,
            bottom: BoundaryType::Wall,
            top: BoundaryType::Wall,
        },
    );

    let mut solver = pollster::block_on(UnifiedSolver::new(
        &mesh,
        incompressible_momentum_model(),
        SolverConfig {
            advection_scheme: Scheme::Upwind,
            time_scheme: TimeScheme::BDF2,
            preconditioner: PreconditionerType::Jacobi,
            stepping: SteppingMode::Coupled,
        },
        None,
        None,
    ))
    .expect("solver init");

    solver.set_dt(0.02);
    solver.set_dtau(0.0).unwrap();
    solver.set_density(1.0).unwrap();
    solver.set_viscosity(0.01).unwrap();
    solver.set_inlet_velocity(1.0).unwrap();
    solver.set_alpha_u(0.7).unwrap();
    solver.set_alpha_p(0.3).unwrap();
    solver.set_outer_iters(50).unwrap();
    solver.set_u(&vec![(0.0, 0.0); mesh.num_cells()]);
    solver.set_p(&vec![0.0; mesh.num_cells()]);
    solver.initialize_history();

    let stats = solver.step_with_stats().expect("step_with_stats");
    eprintln!(
        "[diag][incompressible_channel] linear solves this step: {}",
        stats.len()
    );
}

