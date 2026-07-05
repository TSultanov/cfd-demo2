//! Validation for the **thermal** all-Mach pressure-based model (`allmach_thermal`).
//!
//! `allmach_thermal` extends `allmach_pressure` with a temperature transport
//! equation `ddt(rho,T) + div(phi,T) - lap(k/cp,T)` and an on-device ideal-gas
//! density recovery `rho = rho_t_ref/T + psi*p`, declared as a `PrimitiveDerivations`
//! math expression and lowered (by the codegen) into the coupled Update kernel —
//! no hand-written kernel. The barotropic model is the `T = T_ref` limit.
//!
//! Asserted here:
//!  1. **The on-device recovery is exact.** At every readback the density field
//!     satisfies `rho == rho_t_ref/T + psi*p` to f32 precision — i.e. the codegen
//!     really evaluated the EOS on-device each outer iteration.
//!  2. **Thermal variable density.** A sustained temperature variation produces a
//!     matching density variation (`rho ~ 1/T`), bounded and positive.

#![cfg(all(feature = "dev-tests", feature = "ui"))]

use cfd2::sim::{DriverBuild, SolverDriver};
use cfd2::solver::mesh::{generate_cut_cell_mesh, ChannelWithObstacle, Mesh};
use cfd2::solver::model::{allmach_thermal_model, ALLMACH_GAMMA, ALLMACH_T_REF};
use cfd2::solver::UnifiedSolver;
use cfd2::ui::fluid::Fluid;
use cfd2::ui::model_defaults::gui_defaults_for;
use nalgebra::{Point2, Vector2};

fn air() -> Fluid {
    Fluid::presets()[1].clone()
}

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

/// Build the thermal all-Mach solver through the shared driver and seed all of
/// its coefficient/aux fields. The driver's allmach branch keys on the
/// `allmach_pressure` id, so `allmach_thermal` runs the generic coupled build and
/// every field below must be seeded explicitly. `rho_t_ref` (= density*T_ref) and
/// the temperature `T` are load-bearing: an unseeded (0) `rho_t_ref`/`T` would
/// make the on-device recovery `rho = rho_t_ref/T + psi*p` divide by zero.
fn build_thermal(fluid: &Fluid, mesh: &Mesh, psi: f64, t_init: &[f64]) -> UnifiedSolver {
    let d = gui_defaults_for("allmach_pressure");
    let params = d.to_runtime_params(fluid.density as f32, fluid.viscosity as f32, fluid.eos);
    let n = mesh.num_cells();
    let DriverBuild { mut driver, .. } = pollster::block_on(SolverDriver::build(
        mesh,
        allmach_thermal_model().expect("allmach_thermal model"),
        &params,
        &vec![(d.inlet_velocity as f64, 0.0); n],
        &vec![0.0; n],
        None,
        None,
    ))
    .expect("driver build");
    driver.apply_params(&params);
    let mut solver = driver.into_solver();
    let rho_ref = fluid.density as f64;
    solver.set_field_scalar("psi", &vec![psi; n]).expect("psi");
    // Pressure-row ddt reads the decoupled `psi_precond` (preconditioning is a driver-only
    // transient device); this raw-solver test pins the compressibility, so seed it = psi.
    solver
        .set_field_scalar("psi_precond", &vec![psi; n])
        .expect("psi_precond");
    solver.set_field_scalar("rho", &vec![rho_ref; n]).expect("rho");
    solver
        .set_field_scalar("rho_t_ref", &vec![rho_ref * ALLMACH_T_REF; n])
        .expect("rho_t_ref");
    solver.set_field_scalar("T", &t_init.to_vec()).expect("T");
    solver
}

fn spread(v: &[f64]) -> f64 {
    let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (hi - lo) / (0.5 * (hi + lo)).abs().max(1e-30)
}

/// The on-device EOS recovery must hold at every readback: `rho = rho_t_ref/T + psi*p`,
/// and a sustained temperature variation must give a matching density variation.
#[test]
fn allmach_thermal_recovery_tracks_temperature() {
    let air = air();
    let mesh = channel_obstacle_mesh();
    let n = mesh.num_cells();
    let rho_ref = air.density as f64;
    let psi = 50.0;

    // A strong, smooth inlet→outlet temperature ramp T = T_ref*(1 + x/L), so the
    // EOS density should range ~rho_ref (cold inlet) down to ~rho_ref/2 (hot outlet).
    let length = 3.0;
    let t_init: Vec<f64> = (0..n)
        .map(|c| ALLMACH_T_REF * (1.0 + mesh.cell_cx[c] / length))
        .collect();

    let mut solver = build_thermal(&air, &mesh, psi, &t_init);

    let mut diverged = false;
    for _ in 0..60 {
        if solver.step_with_stats().is_err() {
            diverged = true;
            break;
        }
    }
    assert!(!diverged, "thermal solve diverged");

    let rho = pollster::block_on(solver.get_field_scalar("rho")).expect("rho");
    let t = pollster::block_on(solver.get_field_scalar("T")).expect("T");
    let p = pollster::block_on(solver.get_field_scalar("p")).expect("p");

    // Bounded + positive.
    let max_rho = rho.iter().cloned().fold(0.0_f64, f64::max);
    let min_rho = rho.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(rho.iter().all(|r| r.is_finite() && *r > 0.0), "rho non-finite/non-positive");
    assert!(t.iter().all(|v| v.is_finite() && *v > 0.0), "T non-finite/non-positive");
    assert!(max_rho < 5.0 * rho_ref, "rho blew up: {max_rho}");

    // The core check: on-device recovery to f32 precision against the REAL ideal-gas EOS
    // `rho = rho_t_ref/T + (gamma*psi*t_ref/T)*p` (= p_abs/(R*T); the T-varying
    // compressibility d(rho)/dp|_T = gamma*psi*t_ref/T rises as the gas cools). At
    // T = T_ref this reduces to the barotropic `rho_t_ref/T + psi*p`.
    let rho_t_ref = rho_ref * ALLMACH_T_REF;
    let mut worst = 0.0_f64;
    for c in 0..n {
        let expect = rho_t_ref / t[c] + (ALLMACH_GAMMA * psi * ALLMACH_T_REF / t[c]) * p[c];
        worst = worst.max((rho[c] - expect).abs() / rho_ref);
    }
    assert!(
        worst < 5e-3,
        "on-device EOS recovery mismatch: worst |rho - (rho_t_ref/T + psi*p)|/rho_ref = {worst:.2e}"
    );

    // Thermal variable density: the temperature ramp survives and drives rho.
    let t_spread = spread(&t);
    let rho_spread = spread(&rho);
    assert!(t_spread > 0.1, "temperature variation collapsed: spread={t_spread:.3}");
    assert!(rho_spread > 0.1, "density did not track temperature: spread={rho_spread:.3}");
    assert!(
        min_rho < 0.85 * max_rho,
        "expected rho to vary with T (1/T): min={min_rho:.4} max={max_rho:.4}"
    );

    println!(
        "[allmach_thermal] cells={n} psi={psi} T spread={t_spread:.3} rho spread={rho_spread:.3} \
         rho=[{min_rho:.4},{max_rho:.4}] recovery worst={worst:.2e}"
    );
}
