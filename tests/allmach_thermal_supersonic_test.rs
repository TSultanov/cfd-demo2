//! SUPERSONIC validation for the all-Mach pressure-based thermal solver.
//!
//! A converging–diverging (CD) nozzle accelerates a subsonic inflow through a
//! sonic throat. The flow REGIME in the diverging section is set by the OUTLET
//! back-pressure — the standard way a supersonic nozzle is driven: lowering the
//! back-pressure below the critical value pulls the flow supersonic. Here the
//! pinned-pressure outlet lets us prescribe that back-pressure (a negative gauge
//! value) at runtime via `set_boundary_scalar`.
//!
//! Asserted (allmach_thermal, pinned outlet, CD nozzle, sound speed c=1/sqrt(psi)):
//!   1. CORRECT NOZZLE PHYSICS — the exit Mach rises MONOTONICALLY as the
//!      back-pressure is lowered, while the throat stays ~choked (M_throat~1).
//!   2. SUPERSONIC — at the lowest (most negative) stable back-pressure the exit
//!      is SUPERSONIC (M_exit > 1), having started SUBSONIC at zero back-pressure.
//!   3. EXPANSION COOLING — the accelerating gas cools (T_min < T_ref), the
//!      compression-heating term acting with U.grad(p) < 0.
//!
//! ENVELOPE NOTE: the achievable exit Mach (~1.07 here) is capped by the
//! near-vacuum limit of the gauge-pressure EOS rho = rho_t_ref/T + psi*p: the
//! gauge pressure can only fall ~P_REF = rho_ref/psi before the outlet density
//! crosses zero and the solve breaks. So this solver covers subsonic -> transonic
//! -> WEAK supersonic; strong supersonic (M >> 1, with shocks) is the domain of
//! the density-based `compressible` model.

#![cfg(all(feature = "dev-tests", feature = "ui"))]

use cfd2::sim::{DriverBuild, SolverDriver};
use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::mesh::{generate_structured_nozzle_mesh, BoundarySides, BoundaryType, Mesh};
use cfd2::solver::model::{allmach_thermal_model, ALLMACH_T_REF};
use cfd2::solver::UnifiedSolver;
use cfd2::ui::fluid::Fluid;
use cfd2::ui::model_defaults::gui_defaults_for;

fn air() -> Fluid {
    Fluid::presets()[1].clone()
}

const LENGTH: f64 = 3.0;
const HEIGHT: f64 = 1.0;
const THROAT_H: f64 = 0.40;
const THROAT_FRAC: f64 = 0.40;
const EXIT_H: f64 = 0.80; // area ratio exit/throat = 2.0

fn nozzle(nx: usize, ny: usize) -> Mesh {
    generate_structured_nozzle_mesh(
        nx,
        ny,
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

fn build_nozzle(fluid: &Fluid, mesh: &Mesh, psi: f64, inlet_v: f64) -> UnifiedSolver {
    let mut d = gui_defaults_for("allmach_pressure");
    d.inlet_velocity = inlet_v as f32;
    let params = d.to_runtime_params(fluid.density as f32, fluid.viscosity as f32, fluid.eos);
    let n = mesh.num_cells();
    let DriverBuild { mut driver, .. } = pollster::block_on(SolverDriver::build(
        mesh,
        allmach_thermal_model().expect("allmach_thermal model"),
        &params,
        &vec![(inlet_v, 0.0); n],
        &vec![0.0; n],
        None,
        None,
    ))
    .expect("driver build");
    driver.apply_params(&params);
    let mut solver = driver.into_solver();
    let rho_ref = fluid.density as f64;
    solver.set_field_scalar("psi", &vec![psi; n]).expect("psi");
    solver.set_field_scalar("rho", &vec![rho_ref; n]).expect("rho");
    solver
        .set_field_scalar("rho_t_ref", &vec![rho_ref * ALLMACH_T_REF; n])
        .expect("rho_t_ref");
    solver
        .set_field_scalar("T", &vec![ALLMACH_T_REF; n])
        .expect("T");
    solver
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

struct NozzleRun {
    m_throat: f64,
    m_exit: f64,
    t_min: f64,
}

fn run(air: &Fluid, psi: f64, inlet_v: f64, p_back: f64, steps: usize) -> Option<NozzleRun> {
    let c = 1.0 / psi.sqrt();
    let mesh = nozzle(96, 32);
    let mut solver = build_nozzle(air, &mesh, psi, inlet_v);
    solver
        .set_boundary_scalar(GpuBoundaryType::Outlet, "p", p_back as f32)
        .expect("set back-pressure");
    for _ in 0..steps {
        if solver.step_with_stats().is_err() {
            return None;
        }
    }
    let u = pollster::block_on(solver.get_field_vec2("U")).expect("U");
    let t = pollster::block_on(solver.get_field_scalar("T")).expect("T");
    if !u.iter().all(|(a, b)| a.is_finite() && b.is_finite())
        || !t.iter().all(|v| v.is_finite() && *v > 0.0)
    {
        return None;
    }
    let mach: Vec<f64> = u.iter().map(|(a, b)| (a * a + b * b).sqrt() / c).collect();
    Some(NozzleRun {
        m_throat: region_max(&mesh, &mach, 0.35 * LENGTH, 0.45 * LENGTH),
        m_exit: region_max(&mesh, &mach, 0.85 * LENGTH, LENGTH),
        t_min: t.iter().cloned().fold(f64::INFINITY, f64::min),
    })
}

#[test]
fn supersonic_cd_nozzle_backpressure_driven() {
    let air = air();
    let psi: f64 = 50.0; // c = 1/sqrt(50) ~ 0.1414
    let inlet_v = 0.09; // near the natural choking inlet speed
    let steps = 300;

    // Lower the outlet back-pressure in steps and watch the exit Mach climb.
    let backs = [0.0_f64, -0.015, -0.030, -0.045];
    println!(
        "[supersonic] CD nozzle (area exit/throat={:.1}), psi={psi} c={:.4}, inlet_v={inlet_v}",
        EXIT_H / THROAT_H,
        1.0 / psi.sqrt()
    );

    let mut exits = Vec::new();
    let mut throats = Vec::new();
    let mut tmins = Vec::new();
    for &pb in &backs {
        let r = run(&air, psi, inlet_v, pb, steps).unwrap_or_else(|| {
            panic!("nozzle diverged at back-pressure {pb:+.3}")
        });
        println!(
            "  p_back={pb:+.3}  M_throat={:.3}  M_exit={:.3}  T_min={:.4}",
            r.m_throat, r.m_exit, r.t_min
        );
        exits.push(r.m_exit);
        throats.push(r.m_throat);
        tmins.push(r.t_min);
    }

    // (1) Correct nozzle physics: exit Mach increases monotonically as the
    // back-pressure is lowered, and the throat stays ~choked throughout.
    for w in exits.windows(2) {
        assert!(
            w[1] > w[0] + 0.01,
            "exit Mach not monotonic in back-pressure: {:?}",
            exits
        );
    }
    for (i, &mt) in throats.iter().enumerate() {
        assert!(
            (0.85..=1.20).contains(&mt),
            "throat not ~choked at back-pressure {:+.3}: M_throat={mt:.3}",
            backs[i]
        );
    }

    // (2) Supersonic: starts subsonic at zero back-pressure, ends supersonic at the
    // lowest back-pressure.
    let first = *exits.first().unwrap();
    let last = *exits.last().unwrap();
    assert!(
        first < 1.0,
        "expected subsonic exit at zero back-pressure, got M_exit={first:.3}"
    );
    assert!(
        last > 1.0,
        "back-pressure-driven nozzle did not reach supersonic exit: M_exit={last:.3}"
    );

    // (3) Expansion cooling: the fastest case cooled well below T_ref.
    let coldest = tmins.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        coldest < ALLMACH_T_REF - 0.05,
        "expected clear expansion cooling, got coldest T_min={coldest:.4}"
    );

    println!(
        "[supersonic] subsonic({first:.3}) -> SUPERSONIC({last:.3}) exit, throat choked, \
         coldest T_min={coldest:.4}"
    );
}
