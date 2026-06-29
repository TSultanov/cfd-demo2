//! GUI supersonic-nozzle demo: end-to-end validation of the *GUI build path*.
//!
//! The GUI's "Supersonic Nozzle" case is the composition
//!   model   = allmach_thermal           (the dropdown)
//!   mesh    = generate_structured_nozzle_mesh  (the geometry radio)
//!   params  = ALLMACH_THERMAL_NOZZLE     (the per-case defaults: choking inlet
//!             speed + sub-critical outlet back-pressure)
//! driven through the shared `SolverDriver` exactly as the GUI worker drives it.
//!
//! The whole point of this test is that it does NO manual field seeding: unlike the
//! lower-level `allmach_thermal_supersonic_test`, it relies ENTIRELY on the driver
//! to seed the thermal EOS fields (`psi`, `rho`, `rho_t_ref`, `T`) and to apply the
//! `outlet_back_pressure` from the runtime params — the seam the GUI uses. If this
//! passes, selecting "Supersonic Nozzle" in the GUI produces supersonic flow.

#![cfg(all(feature = "dev-tests", feature = "ui"))]

use cfd2::sim::{DriverBuild, SolverDriver};
use cfd2::solver::mesh::{generate_structured_nozzle_mesh, BoundarySides, BoundaryType, Mesh};
use cfd2::solver::model::allmach_thermal_model;
use cfd2::ui::fluid::Fluid;
use cfd2::ui::model_defaults::ALLMACH_THERMAL_NOZZLE;

// The GUI nozzle geometry (src/ui/app.rs `GeometryType::Nozzle`).
const LENGTH: f64 = 3.0;
const HEIGHT: f64 = 1.0;
const THROAT_H: f64 = 0.40;
const THROAT_FRAC: f64 = 0.40;
const EXIT_H: f64 = 0.80;

fn nozzle_mesh() -> Mesh {
    generate_structured_nozzle_mesh(
        96,
        32,
        LENGTH,
        HEIGHT,
        THROAT_H,
        THROAT_FRAC,
        EXIT_H,
        BoundarySides {
            left: BoundaryType::Inlet,
            right: BoundaryType::Outlet,
            bottom: BoundaryType::Wall,
            top: BoundaryType::Wall,
        },
    )
}

fn region_max(mesh: &Mesh, f: &[f64], x0: f64, x1: f64) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for c in 0..mesh.num_cells() {
        let x = mesh.cell_cx[c];
        if x >= x0 && x < x1 {
            m = m.max(f[c]);
        }
    }
    m
}

#[test]
fn gui_supersonic_nozzle_demo_reaches_mach_1() {
    // Air preset, exactly as the GUI fluid dropdown supplies it.
    let air = Fluid::presets()[1].clone();

    // The per-case GUI defaults → runtime params (carries inlet_velocity = 0.09,
    // compressibility_psi = 50, outlet_back_pressure = -0.045).
    let params = ALLMACH_THERMAL_NOZZLE.to_runtime_params(
        air.density as f32,
        air.viscosity as f32,
        air.eos,
    );
    assert!(
        params.outlet_back_pressure < 0.0,
        "the nozzle demo defaults must carry a sub-critical back-pressure"
    );

    let mesh = nozzle_mesh();
    let n = mesh.num_cells();

    // Uniform-freestream IC, as the GUI's `build_initial_velocity_with` produces for
    // the nozzle. NO field seeding here — the driver must do it.
    let initial_u = vec![(params.inlet_velocity as f64, 0.0); n];
    let initial_p = vec![0.0; n];

    let DriverBuild { mut driver, .. } = pollster::block_on(SolverDriver::build(
        &mesh,
        allmach_thermal_model().expect("allmach_thermal model"),
        &params,
        &initial_u,
        &initial_p,
        None,
        None,
    ))
    .expect("driver build");
    // Phase-2: applies the outlet back-pressure (and keeps psi live) — the GUI seam.
    driver.apply_params(&params);

    let mut solver = driver.into_solver();
    for _ in 0..300 {
        solver
            .step_with_stats()
            .expect("nozzle demo step diverged — the GUI default must stay bounded");
    }

    let u = pollster::block_on(solver.get_field_vec2("U")).expect("U");
    let t = pollster::block_on(solver.get_field_scalar("T")).expect("T");
    assert!(
        u.iter().all(|(a, b)| a.is_finite() && b.is_finite())
            && t.iter().all(|v| v.is_finite() && *v > 0.0),
        "nozzle demo produced non-finite state"
    );

    let c = 1.0 / (params.compressibility_psi as f64).sqrt();
    let mach: Vec<f64> = u.iter().map(|(a, b)| (a * a + b * b).sqrt() / c).collect();
    let m_throat = region_max(&mesh, &mach, 0.35 * LENGTH, 0.45 * LENGTH);
    let m_exit = region_max(&mesh, &mach, 0.85 * LENGTH, LENGTH);
    let t_min = t.iter().cloned().fold(f64::INFINITY, f64::min);

    println!(
        "[gui-nozzle] driver-seeded (no manual fields), back-pressure={:+.3}: \
         M_throat={m_throat:.3}  M_exit={m_exit:.3}  T_min={t_min:.4}",
        params.outlet_back_pressure
    );

    // The driver seeded the thermal EOS fields and applied the back-pressure purely
    // from the params: the GUI demo runs subsonic-in → choked throat → SUPERSONIC out.
    assert!(
        (0.85..=1.20).contains(&m_throat),
        "throat not ~choked: M_throat={m_throat:.3} (driver seeding may be wrong)"
    );
    assert!(
        m_exit > 1.0,
        "GUI nozzle demo did not reach supersonic exit: M_exit={m_exit:.3}"
    );
    assert!(
        t_min < 0.95,
        "expected expansion cooling below T_ref, got T_min={t_min:.4}"
    );
}
