use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides};
use cfd2::solver::model::compressible_model;
use cfd2::solver::model::helpers::{SolverCompressibleIdealGasExt, SolverRuntimeParamsExt};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{
    GpuLowMachPrecondModel, PreconditionerType, SolverConfig, SteppingMode, TimeScheme,
    UnifiedSolver,
};

#[test]
fn low_mach_knob_changes_compressible_implicit_update() {
    std::env::set_var("CFD2_QUIET", "1");

    let mesh = generate_structured_rect_mesh(2, 1, 1.0, 1.0, BoundarySides::wall());

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
    solver.set_precond_theta_floor(1e-2).expect("theta floor");

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
    let rho_e_legacy = pollster::block_on(solver.get_field_scalar("rho_e")).expect("rho_e legacy");

    solver
        .set_precond_model(GpuLowMachPrecondModel::WeissSmith)
        .expect("precond model weiss-smith");
    solver.set_state_fields(&rho, &u, &p);
    solver.initialize_history();
    solver.step();
    let rho_e_weiss =
        pollster::block_on(solver.get_field_scalar("rho_e")).expect("rho_e weiss-smith");

    let delta_off = (rho_e_off[0] - rho_e_off[1]).abs();
    let delta_legacy = (rho_e_legacy[0] - rho_e_legacy[1]).abs();
    let delta_weiss = (rho_e_weiss[0] - rho_e_weiss[1]).abs();

    eprintln!(
        "[low_mach_effect] delta_off={delta_off:.6} delta_weiss={delta_weiss:.6} delta_legacy={delta_legacy:.6}"
    );

    assert!(
        delta_legacy > delta_off * 1.0005,
        "expected low-mach preconditioning to reduce numerical diffusion (delta_legacy={delta_legacy:.6}, delta_off={delta_off:.6})"
    );

    assert!(
        delta_weiss > delta_off * 1.0001 && delta_legacy > delta_weiss * 1.0001,
        "expected low-mach model variants to differ (delta_off={delta_off:.6}, delta_weiss={delta_weiss:.6}, delta_legacy={delta_legacy:.6})"
    );
}
