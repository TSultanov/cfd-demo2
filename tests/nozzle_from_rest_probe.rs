//! PROBE (`#[ignore]`): can the all-Mach CD nozzle develop a stable supersonic
//! flow FROM REST (no seeded velocity), driven purely by a physical pressure inlet
//! (rocket-engine-scale Pascals), optionally with a TIME soft-start (chamber-pressure
//! ramp, like ignition)?
//!
//! The shipped demo seeds a uniform freestream (inlet_velocity) because "from rest the
//! inlet-injected momentum piles up". This probe tests whether — with the hyperbolic
//! pressure-row Newton linearization + low-Mach preconditioning now in place — the
//! flow can instead develop from scratch.
//!
//! Env knobs (bar-ish; all Pascal):
//!   NOZZLE_P        inlet gauge pressure target (Pa)             default 6e4
//!   NOZZLE_STEPS    steps                                        default 800
//!   NOZZLE_RAMP     soft-start steps to reach full P (0=off)     default 0
//!   NOZZLE_REST     1 => rest IC + flat p=0; 0 => shipped seed   default 1
//!
//! Run: cargo test --release --features "dev-tests ui" --test nozzle_from_rest_probe -- --ignored --nocapture
#![cfg(all(feature = "dev-tests", feature = "ui"))]

use cfd2::sim::{DriverBuild, SolverDriver};
use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::mesh::{generate_structured_nozzle_mesh, BoundarySides, BoundaryType, Mesh};
use cfd2::solver::model::allmach_thermal_model;
use cfd2::ui::fluid::Fluid;
use cfd2::ui::model_defaults::ALLMACH_THERMAL_NOZZLE;

const LENGTH: f64 = 3.0;
const HEIGHT: f64 = 1.0;
const THROAT_H: f64 = 0.40;
const THROAT_FRAC: f64 = 0.40;
const EXIT_H: f64 = 0.80;

fn nozzle_mesh() -> Mesh {
    generate_structured_nozzle_mesh(
        96, 32, LENGTH, HEIGHT, THROAT_H, THROAT_FRAC, EXIT_H,
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

fn envf(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
}

#[test]
#[ignore]
fn nozzle_from_rest_probe() {
    let air = Fluid::presets()[1].clone();
    let mut params = ALLMACH_THERMAL_NOZZLE.to_runtime_params(air.density as f32, air.viscosity as f32, air.eos);

    let p_target = envf("NOZZLE_P", 6.0e4) as f32;
    let steps = envf("NOZZLE_STEPS", 800.0) as usize;
    let ramp = envf("NOZZLE_RAMP", 0.0) as usize;
    let from_rest = envf("NOZZLE_REST", 1.0) as u32 != 0;
    params.inlet_pressure = p_target;

    let mesh = nozzle_mesh();
    let n = mesh.num_cells();

    // From-rest: zero velocity IC. (Shipped path seeds a uniform freestream instead.)
    let initial_u = if from_rest {
        vec![(0.0, 0.0); n]
    } else {
        vec![(params.inlet_velocity as f64, 0.0); n]
    };
    let initial_p = vec![0.0; n];

    let DriverBuild { mut driver, .. } = pollster::block_on(SolverDriver::build(
        &mesh,
        allmach_thermal_model().expect("model"),
        &params,
        &initial_u,
        &initial_p,
        None,
        None,
    ))
    .expect("build");
    driver.apply_params(&params);
    let mut solver = driver.into_solver();

    // From-scratch: overwrite the driver's spatial pressure ramp with a flat p=0 gauge,
    // so the flow develops purely from the (soft-started) inlet BC.
    if from_rest {
        let _ = solver.set_field_scalar("p", &vec![0.0; n]);
        solver.initialize_history();
    }

    let c = 1.0 / (params.compressibility_psi as f64).sqrt();
    let mut diverged_at = None;
    for k in 0..steps {
        // Soft-start: ramp the inlet gauge pressure 0 -> target over `ramp` steps.
        if ramp > 0 {
            let frac = ((k + 1) as f32 / ramp as f32).min(1.0);
            let _ = solver.set_boundary_scalar(GpuBoundaryType::Inlet, "p", p_target * frac);
        }
        if solver.step_with_stats().is_err() {
            diverged_at = Some(k);
            break;
        }
        if k % 100 == 0 || k == steps - 1 {
            let u = pollster::block_on(solver.get_field_vec2("U")).unwrap();
            let maxu = u.iter().map(|(a, b)| (a * a + b * b).sqrt()).fold(0.0, f64::max);
            if !maxu.is_finite() || maxu > 5.0 * c {
                diverged_at = Some(k);
                eprintln!("[nozzle-rest] step {k}: max|u|={maxu:.1} (>{:.0}) -> diverged", 5.0 * c);
                break;
            }
            eprintln!("[nozzle-rest] step {k:>4}: max|u|={maxu:8.1} (M={:.2})", maxu / c);
        }
    }

    let u = pollster::block_on(solver.get_field_vec2("U")).unwrap();
    let t = pollster::block_on(solver.get_field_scalar("T")).unwrap();
    let mach: Vec<f64> = u.iter().map(|(a, b)| (a * a + b * b).sqrt() / c).collect();
    let m_throat = region_max(&mesh, &mach, 0.35 * LENGTH, 0.45 * LENGTH);
    let m_exit = region_max(&mesh, &mach, 0.85 * LENGTH, LENGTH);
    let t_min = t.iter().cloned().fold(f64::INFINITY, f64::min);
    let finite = u.iter().all(|(a, b)| a.is_finite() && b.is_finite()) && t.iter().all(|v| v.is_finite() && *v > 0.0);

    eprintln!(
        "[nozzle-rest] RESULT P={p_target:.3e} rest={from_rest} ramp={ramp} steps={steps}: \
         finite={finite} diverged_at={diverged_at:?} M_throat={m_throat:.3} M_exit={m_exit:.3} T_min={t_min:.2} (c={c:.0})"
    );
    assert!(finite && diverged_at.is_none(), "nozzle diverged / non-finite");
    assert!(m_exit > 1.0, "expected supersonic exit, got M_exit={m_exit:.3}");
}
