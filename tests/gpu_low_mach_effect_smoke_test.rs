use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundaryType};
use cfd2::solver::model::compressible_model;
use cfd2::solver::model::helpers::{SolverCompressibleIdealGasExt, SolverRuntimeParamsExt};
use cfd2::solver::options::{
    GpuLowMachPrecondModel, PreconditionerType, SteppingMode, TimeScheme,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{SolverConfig, UnifiedSolver};

#[test]
fn low_mach_knob_changes_compressible_implicit_update() {
    std::env::set_var("CFD2_QUIET", "1");

    let mesh = generate_structured_rect_mesh(
        2,
        1,
        1.0,
        1.0,
        BoundaryType::Wall,
        BoundaryType::Wall,
        BoundaryType::Wall,
        BoundaryType::Wall,
    );

    let mut solver = pollster::block_on(UnifiedSolver::new(
        &mesh,
        compressible_model(),
        SolverConfig {
            advection_scheme: Scheme::Upwind,
            time_scheme: TimeScheme::Euler,
            preconditioner: PreconditionerType::Jacobi,
            stepping: SteppingMode::Implicit { outer_iters: 1 },
        },
        None,
        None,
    ))
    .expect("solver init");

    solver.set_dt(1e-4);
    solver.set_dtau(5e-5).expect("dtau");
    solver.set_viscosity(0.0).expect("viscosity");
    solver
        .set_precond_theta_floor(1e-6)
        .expect("theta floor");

    let rho = vec![1.0f32; mesh.num_cells()];
    let u = vec![[0.0f32, 0.0f32]; mesh.num_cells()];
    let p0 = 1.0e4f32;
    let dp = 100.0f32;
    let p = vec![p0 + dp, p0 - dp];

    solver.set_state_fields(&rho, &u, &p);
    solver.initialize_history();

    solver
        .set_precond_model(GpuLowMachPrecondModel::Off)
        .expect("precond model off");
    solver.step();
    let rho_e_off = pollster::block_on(solver.get_field_scalar("rho_e")).expect("rho_e off");

    solver
        .set_precond_model(GpuLowMachPrecondModel::Legacy)
        .expect("precond model legacy");
    solver.set_state_fields(&rho, &u, &p);
    solver.initialize_history();
    solver.step();
    let rho_e_on = pollster::block_on(solver.get_field_scalar("rho_e")).expect("rho_e on");

    let delta_off = (rho_e_off[0] - rho_e_off[1]).abs();
    let delta_on = (rho_e_on[0] - rho_e_on[1]).abs();

    assert!(
        delta_on > delta_off * 1.01,
        "expected low-mach preconditioning to reduce numerical diffusion (delta_on={delta_on:.6}, delta_off={delta_off:.6})"
    );
}
