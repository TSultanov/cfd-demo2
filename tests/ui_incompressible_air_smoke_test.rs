#![cfg(feature = "meshgen")]

use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep};
use cfd2::solver::model::helpers::{
    SolverFieldAliasesExt, SolverIncompressibleControlsExt, SolverRuntimeParamsExt,
    SolverInletVelocityExt,
};
use cfd2::solver::model::incompressible_momentum_model;
use cfd2::solver::options::TimeScheme;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{SolverConfig, UnifiedSolver};
use nalgebra::Vector2;

#[test]
fn ui_incompressible_air_smoke_does_not_blow_up_immediately() {
    // Match the UI defaults closely (BackwardsStep, CutCell, Air, dt=1e-3).
    let length = 3.5;
    let domain_size = Vector2::new(length, 1.0);
    let geo = BackwardsStep {
        length,
        height_inlet: 0.5,
        height_outlet: 1.0,
        step_x: 0.5,
    };

    let min_cell_size = 0.025;
    let max_cell_size = 0.025;
    let mut mesh = generate_cut_cell_mesh(&geo, min_cell_size, max_cell_size, 1.2, domain_size);
    mesh.smooth(&geo, 0.3, 50);

    let mut solver = pollster::block_on(UnifiedSolver::new(
        &mesh,
        incompressible_momentum_model(),
        SolverConfig {
            advection_scheme: Scheme::Upwind,
            time_scheme: TimeScheme::Euler,
            ..Default::default()
        },
        None,
        None,
    ))
    .expect("solver init");

    // Clear full state to avoid uninitialized auxiliary fields (d_p, grad_p, etc).
    let stride = solver.model().state_layout.stride() as usize;
    solver
        .write_state_f32(&vec![0.0f32; mesh.num_cells() * stride])
        .expect("clear state");

    // Air preset.
    solver.set_density(1.225).unwrap();
    solver.set_viscosity(1.81e-5).unwrap();

    // Conservative under-relaxation (the UI defaults should keep this stable).
    solver.set_alpha_u(0.7).unwrap();
    solver.set_alpha_p(0.3).unwrap();

    // Initial conditions.
    let n_cells = mesh.num_cells();
    solver.set_u(&vec![(0.0, 0.0); n_cells]);
    solver.set_p(&vec![0.0; n_cells]);
    solver.initialize_history();

    // Boundary conditions.
    solver.set_inlet_velocity(1.0).unwrap();
    solver.set_dt(0.001);
    solver.incompressible_set_should_stop(false);

    for step in 0..10 {
        let stats = solver.step_with_stats().expect("step with stats");
        if let Some(last) = stats.last() {
            assert!(
                last.residual.is_finite() && last.residual < 1e20,
                "step {step}: linear residual blew up: {:?}",
                last
            );
            assert!(!last.diverged, "step {step}: linear solver diverged: {last:?}");
        }

        let u = pollster::block_on(solver.get_u());
        let p = pollster::block_on(solver.get_p());

        let mut max_vel = 0.0f64;
        for (vx, vy) in &u {
            assert!(vx.is_finite() && vy.is_finite(), "step {step}: non-finite u");
            let v = (vx * vx + vy * vy).sqrt();
            max_vel = max_vel.max(v);
        }
        for pv in &p {
            assert!(pv.is_finite(), "step {step}: non-finite p");
        }
        assert!(
            max_vel < 1e6,
            "step {step}: velocity magnitude blew up: {max_vel:e}"
        );
    }
}
