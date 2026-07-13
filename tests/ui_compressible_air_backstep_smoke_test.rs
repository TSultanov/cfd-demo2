#![cfg(all(feature = "meshgen", feature = "dev-tests"))]

//! Smoke test: the desktop GUI's compressible-default construction (uniform-freestream
//! IC + inlet BC + acoustic-aware adaptive timestep) + step loop, built through the
//! shared [`SolverDriver`] without the `ui` feature (primitives only). Proves the driver
//! is `ui`-independent for the compressible path.

use cfd2::sim::{DriverBuild, RuntimeParams, SolverDriver};
use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep};
use cfd2::solver::model::compressible_model_with_eos;
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{GpuLowMachPrecondModel, PreconditionerType, TimeScheme};
use nalgebra::Vector2;
use std::ops::ControlFlow;

#[test]
fn ui_compressible_air_backstep_smoke() {
    // Roughly match the UI defaults (BackwardsStep, CutCell, Air, inlet u=1).
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

    let eos = EosSpec::IdealGas {
        gamma: 1.4,
        gas_constant: 287.0,
        temperature: 300.0,
    };
    // GUI-like compressible runtime knobs for Air, as primitives. `adaptive_dt` with
    // `low_mach_model: Off` gives the full-sound-speed acoustic CFL update; the driver
    // applies the uniform-freestream IC internally.
    let params = RuntimeParams {
        filter_sigma: 0.0,
        adaptive_dt: true,
        target_cfl: 0.95,
        requested_dt: 0.001,
        dtau: 0.0,
        log_convergence: false,
        log_every_steps: 50,
        advection_scheme: Scheme::Upwind,
        time_scheme: TimeScheme::BDF2,
        preconditioner: PreconditionerType::Jacobi,
        outer_iters: 1,
        outer_auto_converge: false,
        low_mach_model: GpuLowMachPrecondModel::Off,
        low_mach_theta_floor: 1e-6,
        low_mach_pressure_coupling_alpha: 1.0,
        alpha_u: 1.0,
        alpha_p: 1.0,
        inlet_velocity: 1.0,
        density: 1.225,
        viscosity: 1.81e-5,
        eos,
        compressibility_psi: 0.0,
        outlet_back_pressure: 0.0,
        allmach_precond_uref_min: 0.2,
        pressure_inlet: false,
        inlet_pressure: 0.0,
    };

    let DriverBuild { mut driver, .. } = pollster::block_on(SolverDriver::build(
        &mesh,
        compressible_model_with_eos(eos).expect("model"),
        &params,
        &vec![(0.0, 0.0); n],
        &vec![0.0; n],
        None,
        None,
    ))
    .expect("driver build");
    driver.apply_params(&params);

    let result = driver.run_steps(10, 1, |step, outcome| {
        if let Some(reason) = &outcome.diverged {
            panic!("step {step}: diverged: {reason:?}");
        }
        if let Some(rb) = &outcome.readback {
            let fs = &rb.stats;
            assert_eq!(fs.nonfinite_u, 0, "step {step}: non-finite u");
            assert!(fs.p_finite, "step {step}: non-finite p");
        }
        ControlFlow::Continue(())
    });
    assert!(
        result.diverged.is_none(),
        "compressible smoke diverged at step {:?}",
        result.stop_step
    );
}
