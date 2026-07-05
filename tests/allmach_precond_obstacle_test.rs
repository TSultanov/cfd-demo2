//! Low-Mach preconditioning gate: the all-Mach pressure model must be STABLE at the
//! REAL EOS compressibility (psi = 1/c^2, Air ≈ 8.3e-6), not just the artificial psi≈50.
//!
//! The pressure-row `ddt(psi,p)` couples the acoustic mode explicitly, and at real (tiny)
//! psi the sound speed c=1/√psi≈347 gives an acoustic CFL ≈ 667 with the convective dt.
//! The driver seeds `psi_precond = max(psi, 1/(k·U_inlet)²)` into a decoupled field that
//! ONLY the time term reads (density recovery keeps real psi), so the pseudo sound speed
//! tracks the velocity and a convective dt is acoustically stable.

#![cfg(all(feature = "dev-tests", feature = "ui"))]

use cfd2::sim::{DriverBuild, SolverDriver};
use cfd2::solver::mesh::{generate_cut_cell_mesh, ChannelWithObstacle, Mesh};
use cfd2::solver::model::allmach_pressure_model;
use cfd2::ui::fluid::Fluid;
use cfd2::ui::model_defaults::gui_defaults_for;
use nalgebra::{Point2, Vector2};

fn channel_obstacle_mesh() -> Mesh {
    let length = 3.0;
    let geo = ChannelWithObstacle {
        length,
        height: 1.0,
        obstacle_center: Point2::new(1.0, 0.51),
        obstacle_radius: 0.1,
    };
    let mut mesh = generate_cut_cell_mesh(&geo, 0.025, 0.025, 1.2, Vector2::new(length, 1.0));
    mesh.smooth(&geo, 0.3, 100);
    mesh
}

#[test]
fn allmach_real_psi_obstacle_is_bounded_with_preconditioning() {
    std::env::set_var("CFD2_QUIET", "1");
    let air = Fluid::presets()[1].clone(); // Air
    let d = gui_defaults_for("allmach_pressure"); // default exaggeration ×1 => real psi
    let mesh = channel_obstacle_mesh();

    let params = d.to_runtime_params(air.density as f32, air.viscosity as f32, air.eos);
    let psi = params.compressibility_psi as f64;
    let psi_phys = air.compressibility();
    // Confirm we are genuinely at REAL Air compressibility, not an exaggerated value.
    assert!(
        (psi - psi_phys).abs() / psi_phys < 1e-4 && psi < 1e-4,
        "expected real EOS psi≈{psi_phys:.3e}, got {psi:.3e}"
    );
    eprintln!(
        "[precond/obstacle] real psi={psi:.3e} (c≈{:.0} m/s), raw acoustic CFL would be ~667",
        air.sound_speed()
    );

    let n = mesh.num_cells();
    let DriverBuild { mut driver, .. } = pollster::block_on(SolverDriver::build(
        &mesh,
        allmach_pressure_model().expect("allmach_pressure model"),
        &params,
        &vec![(0.0, 0.0); n],
        &vec![0.0; n],
        None,
        None,
    ))
    .expect("driver build");
    driver.apply_params(&params);

    let u_in = d.inlet_velocity as f64;
    let mut max_seen = 0.0_f64;
    let mut first_step_max = 0.0_f64;
    let mut first_step_pmag = 0.0_f64;
    for step in 0..250 {
        let outcome = driver.step(true);
        let fs = &outcome.readback.as_ref().expect("readback").stats;
        let mv = fs.max_vel;
        let pmag = fs.p_min.abs().max(fs.p_max.abs());
        if step == 0 {
            first_step_max = mv;
            first_step_pmag = pmag;
        }
        assert!(
            fs.nonfinite_u == 0 && fs.p_finite && mv.is_finite(),
            "non-finite at step {step}: max|u|={mv:.3e}"
        );
        assert!(
            mv < 1.0,
            "DIVERGED at step {step}: max|u|={mv:.3e} (the 667-acoustic-CFL blow-up); \
             p=[{:.3e},{:.3e}]",
            fs.p_min, fs.p_max
        );
        max_seen = max_seen.max(mv);
    }

    // Step 0 must not blow up: the preconditioner holds it at the convective scale
    // O(inlet), with a gauge pressure O(rho·U²) « 1.
    eprintln!(
        "[precond/obstacle] step0 max|u|={first_step_max:.3e} (inlet {u_in:.3e}), |p|≈{first_step_pmag:.3e}; \
         run max|u|={max_seen:.3e}"
    );
    assert!(
        first_step_max < 0.1,
        "step 0 not preconditioned: max|u|={first_step_max:.3e} (blow-up was ≈36)"
    );
    // The wake still develops (the model reduces to the validated incompressible street).
    assert!(
        max_seen > 1.2 * u_in,
        "wake did not develop: max|u|={max_seen:.3e} vs inlet {u_in:.3e}"
    );
}
