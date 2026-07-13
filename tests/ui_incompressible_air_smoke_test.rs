#![cfg(all(feature = "meshgen", feature = "dev-tests"))]

//! Smoke test: the desktop GUI's incompressible-default construction + step loop,
//! built through the shared [`SolverDriver`] **without the `ui` feature** (primitives
//! only — no `Fluid`, no `model_defaults`). This is the load-bearing proof that the
//! driver is `ui`-independent: it lives under `meshgen` + `dev-tests` and never names
//! a `ui`-gated type. The richer convergence gate
//! (`tests/gui_default_convergence_test.rs`) tunes the real shipped defaults.

use cfd2::sim::{DriverBuild, RuntimeParams, SolverDriver};
use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep};
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::model::incompressible_momentum_model;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{GpuLowMachPrecondModel, PreconditionerType, TimeScheme};
use nalgebra::Vector2;
use std::ops::ControlFlow;

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
    let mut mesh = generate_cut_cell_mesh(&geo, 0.025, 0.025, 1.2, domain_size);
    mesh.smooth(&geo, 0.3, 50);
    let n = mesh.num_cells();

    // GUI-like incompressible (coupled SIMPLE) runtime knobs for Air, as primitives.
    let params = RuntimeParams {
        filter_sigma: 0.0,
        adaptive_dt: false,
        target_cfl: 0.9,
        requested_dt: 0.001,
        dtau: 0.0,
        log_convergence: false,
        log_every_steps: 50,
        advection_scheme: Scheme::Upwind,
        time_scheme: TimeScheme::BDF2,
        preconditioner: PreconditionerType::Jacobi,
        outer_iters: 8,
        outer_auto_converge: false,
        low_mach_model: GpuLowMachPrecondModel::Off,
        low_mach_theta_floor: 1e-6,
        low_mach_pressure_coupling_alpha: 1.0,
        alpha_u: 0.7,
        alpha_p: 0.3,
        inlet_velocity: 1.0,
        density: 1.225,
        viscosity: 1.81e-5,
        eos: EosSpec::Constant,
        compressibility_psi: 0.0,
        outlet_back_pressure: 0.0,
        allmach_precond_uref_min: 0.2,
        pressure_inlet: false,
        inlet_pressure: 0.0,
    };

    // Construct through the driver (config/stepping derivation, phase-1 setters, IC/BC)
    // + phase-2 `apply_params` — the same path the GUI worker runs.
    let DriverBuild { mut driver, .. } = pollster::block_on(SolverDriver::build(
        &mesh,
        incompressible_momentum_model().expect("model"),
        &params,
        &vec![(0.0, 0.0); n],
        &vec![0.0; n],
        None,
        None,
    ))
    .expect("driver build");
    driver.apply_params(&params);

    // Drive 10 steps via the shared loop, asserting no blow-up each step.
    let result = driver.run_steps(10, 1, |step, outcome| {
        if let Some(reason) = &outcome.diverged {
            panic!("step {step}: diverged: {reason:?}");
        }
        if let Some(rb) = &outcome.readback {
            let fs = &rb.stats;
            assert_eq!(fs.nonfinite_u, 0, "step {step}: non-finite u");
            assert!(fs.p_finite, "step {step}: non-finite p");
            assert!(
                fs.max_vel < 1e6,
                "step {step}: velocity magnitude blew up: {:e}",
                fs.max_vel
            );
        }
        ControlFlow::Continue(())
    });
    assert!(
        result.diverged.is_none(),
        "incompressible smoke diverged at step {:?}",
        result.stop_step
    );
}
