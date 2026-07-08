//! GUI supersonic-nozzle demo: end-to-end validation of the GUI build path.
//!
//! The GUI's "Supersonic Nozzle" case composes model = allmach_thermal, mesh =
//! generate_structured_symmetric_nozzle_mesh (the iconic symmetric CD-nozzle bell,
//! area ratio exit/throat = 2), params = ALLMACH_THERMAL_NOZZLE, driven through the
//! shared `SolverDriver` exactly as the GUI worker drives it.
//!
//! Driven by a pressure inlet + supersonic (extrapolated) outlet
//! (`ALLMACH_THERMAL_NOZZLE.pressure_inlet = true`): the driver flips the Inlet/Outlet
//! boundary kinds, pins the inlet gauge pressure (the gauge anchor), and lets the outlet
//! float (no back-pressure). At the real Air sound speed (c≈347 m/s) the flow develops
//! FROM REST into a classic supersonic CD nozzle — a near-sonic (choked) throat
//! accelerating through the diverging section to a supersonic exit (M_exit≈2.2) with
//! strong expansion cooling (T_exit≈0.5·T_ref, the isentropic value). The extrapolated
//! outlet pressure row is kept well-posed by the pressure-flux Newton linearization (else
//! the exit runs to vacuum), and pseudo-transient continuation (dual time) damps the
//! from-rest acoustic transient so the local sound speed can't over-expand a cold spot.
//!
//! The point of this test: it does NO manual field seeding — the flow develops purely
//! from the pinned inlet pressure. It relies ENTIRELY on the driver to seed the thermal
//! EOS fields (`psi`, `rho`, `rho_t_ref`, `T`), flip the BCs, and pin the inlet pressure
//! (the seam the GUI uses).

#![cfg(all(feature = "dev-tests", feature = "ui"))]

use cfd2::sim::{DriverBuild, SolverDriver};
use cfd2::solver::mesh::{generate_structured_symmetric_nozzle_mesh, BoundarySides, BoundaryType, Mesh};
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
    generate_structured_symmetric_nozzle_mesh(
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

    // The per-case GUI defaults → runtime params (pressure-inlet nozzle at the REAL
    // Air sound speed: compressibility_psi ≈ 8.3e-6 (c≈347 m/s), pressure_inlet = true,
    // inlet_pressure = 6e4 Pa gauge — zero exaggeration).
    let params = ALLMACH_THERMAL_NOZZLE.to_runtime_params(
        air.density as f32,
        air.viscosity as f32,
        air.eos,
    );
    assert!(
        params.pressure_inlet && params.inlet_pressure > 0.0,
        "the nozzle demo defaults must carry a pressure inlet: pressure_inlet={}, inlet_pressure={}",
        params.pressure_inlet,
        params.inlet_pressure
    );

    let mesh = nozzle_mesh();
    let n = mesh.num_cells();

    // FROM REST, as the GUI's `build_initial_velocity_with` produces for the nozzle:
    // zero velocity + flat gauge p=0 (the driver seeds no ramp). NO field seeding here —
    // the driver must do the EOS fields + BC flip + inlet-pressure pin.
    let initial_u = vec![(0.0, 0.0); n];
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
    // apply_params pins the inlet pressure (and keeps psi live) — the GUI seam.
    driver.apply_params(&params);

    let mut solver = driver.into_solver();
    // From rest the flow accelerates from zero, chokes the throat, and settles toward the
    // supersonic CD profile. Pseudo-transient continuation (dual time, set by the preset)
    // damps the from-rest acoustic transient so the LOCAL sound speed cannot over-expand a
    // cold spot; 1200 steps clears the transient to the steady supersonic state.
    for _ in 0..1200 {
        solver
            .step_with_stats()
            .expect("nozzle demo step diverged — the GUI default must stay bounded");
    }

    let u = pollster::block_on(solver.get_field_vec2("U")).expect("U");
    let t = pollster::block_on(solver.get_field_scalar("T")).expect("T");
    // psi is recovered on-device as the LOCAL 1/c^2(T) = psi_ref*t_ref/T; read it
    // back for the true per-cell sound speed c = 1/sqrt(psi), so M = speed*sqrt(psi).
    let psi_local = pollster::block_on(solver.get_field_scalar("psi")).expect("psi");
    assert!(
        u.iter().all(|(a, b)| a.is_finite() && b.is_finite())
            && t.iter().all(|v| v.is_finite() && *v > 0.0)
            && psi_local.iter().all(|v| v.is_finite() && *v > 0.0),
        "nozzle demo produced non-finite state"
    );

    let mach: Vec<f64> = u
        .iter()
        .zip(psi_local.iter())
        .map(|((a, b), ps)| (a * a + b * b).sqrt() * ps.sqrt())
        .collect();
    let m_throat = region_max(&mesh, &mach, 0.35 * LENGTH, 0.45 * LENGTH);
    let m_exit = region_max(&mesh, &mach, 0.85 * LENGTH, LENGTH);
    let t_min = t.iter().cloned().fold(f64::INFINITY, f64::min);

    println!(
        "[gui-nozzle] driver-seeded (no manual fields), pressure-inlet p_in={:+.3}: \
         M_throat={m_throat:.3}  M_exit={m_exit:.3}  T_min={t_min:.4}",
        params.inlet_pressure
    );

    // Developing from rest, the inlet pressure drop chokes the throat and the flow
    // reaches a supersonic exit with expansion cooling — a classic CD nozzle,
    // M_exit > M_throat. Near-sonic (choked) throat: a 2D viscous throat with a smooth
    // cosine profile peaks just below M=1 at the geometric minimum (the sonic line curves
    // and sits partly downstream), so the throat window reads ~0.8 — transonic/choked,
    // NOT the isentropic-1D exactly-1. The band brackets that.
    assert!(
        (0.65..=1.35).contains(&m_throat),
        "throat should be transonic (near-sonic / choked): M_throat={m_throat:.3}"
    );
    // Clearly supersonic exit (area ratio 2 => isentropic M_exit ≈ 2.2).
    assert!(
        m_exit > 1.5,
        "GUI nozzle demo did not reach a supersonic exit: M_exit={m_exit:.3}"
    );
    assert!(
        m_exit > m_throat,
        "expected a proper CD-nozzle profile (supersonic diverging section): \
         M_exit={m_exit:.3} should exceed M_throat={m_throat:.3}"
    );
    // Strong expansion cooling: at M_exit ≈ 2.2 the isentropic T_exit ≈ 0.5·T_ref.
    assert!(
        t_min < 0.75,
        "expected strong expansion cooling below T_ref, got T_min={t_min:.4}"
    );
}
