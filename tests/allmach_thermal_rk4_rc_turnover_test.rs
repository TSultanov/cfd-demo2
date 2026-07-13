//! Regression for amplitude-sensitive Rhie--Chow transport in explicit thermal RK4.
#![cfg(all(feature = "dev-tests", feature = "cpu", feature = "meshgen"))]

use cfd2::sim::{RuntimeParams, SolverDriver};
use cfd2::solver::mesh::generate_structured_rect_mesh_periodic;
use cfd2::solver::model::{allmach_thermal_model, eos::EosSpec};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{GpuLowMachPrecondModel, PreconditionerType, TimeScheme};

fn rk4_params() -> RuntimeParams {
    RuntimeParams {
        filter_sigma: 0.0,
        adaptive_dt: true,
        target_cfl: 0.9,
        requested_dt: 0.02,
        dtau: 0.0,
        log_convergence: false,
        log_every_steps: 100,
        advection_scheme: Scheme::SecondOrderUpwindVanLeer,
        time_scheme: TimeScheme::RK4,
        preconditioner: PreconditionerType::Jacobi,
        outer_iters: 1,
        outer_auto_converge: false,
        low_mach_model: GpuLowMachPrecondModel::Off,
        low_mach_theta_floor: 1.0e-6,
        low_mach_pressure_coupling_alpha: 1.0,
        alpha_u: 0.7,
        alpha_p: 0.3,
        inlet_velocity: 0.0,
        density: 1.225,
        viscosity: 1.81e-5,
        eos: EosSpec::Constant,
        compressibility_psi: (1.0 / (347.0_f64 * 347.0)) as f32,
        outlet_back_pressure: 0.0,
        allmach_precond_uref_min: 1.0,
        pressure_inlet: false,
        inlet_pressure: 0.0,
    }
}

#[test]
fn adaptive_dt_resolves_seeded_rc_face_turnover() {
    const NX: usize = 32;
    const NY: usize = 16;
    const RK_CFL: f64 = 0.8 * 0.9;

    for amplitude in [200.0_f64, 500.0] {
        let mesh = generate_structured_rect_mesh_periodic(NX, NY, 1.0, 1.0);
        let pressure: Vec<f64> = (0..mesh.num_cells())
            .map(|cell| if cell % NX & 1 == 0 { amplitude } else { -amplitude })
            .collect();
        let params = rk4_params();
        let mut build = pollster::block_on(SolverDriver::build_forced_cpu_transpiled(
            &mesh,
            allmach_thermal_model().expect("all-Mach thermal model"),
            &params,
            &vec![(0.0, 0.0); mesh.num_cells()],
            &pressure,
        ))
        .expect("RK4 driver");
        build.driver.apply_params(&params);

        // For this Cartesian two-colour mode the reconstructed pressure gradients
        // vanish.  The two x-face compact RC jumps therefore give this exact
        // accepted-state row turnover; it must constrain the very first step.
        let tau = 0.25_f64.sqrt() / (NX + NY) as f64;
        let dx = 1.0 / NX as f64;
        let rc_turnover = 4.0 * tau * amplitude / (params.density as f64 * dx * dx);
        let rc_dt_limit = RK_CFL / rc_turnover;

        for step in 1..=16 {
            let outcome = build.driver.step(false);
            if step == 1 {
                assert!(
                    outcome.dt as f64 <= rc_dt_limit * 1.000_01,
                    "AMP={amplitude}: first dt={} exceeds accepted-state RC limit {rc_dt_limit}",
                    outcome.dt
                );
            }
            assert!(
                outcome.diverged.is_none(),
                "AMP={amplitude}: RK4 failed at step {step}, dt={}: {:?}",
                outcome.dt,
                outcome.diverged
            );
        }

        for field in ["T", "rho"] {
            let values = pollster::block_on(build.driver.solver().get_field_scalar(field))
                .unwrap_or_else(|error| panic!("read {field}: {error}"));
            assert!(
                values.iter().all(|value| value.is_finite() && *value > 0.0),
                "AMP={amplitude}: {field} lost positivity"
            );
        }
    }
}
