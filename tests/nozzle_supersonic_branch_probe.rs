//! PROBE (`#[ignore]`'d): sweeps `inlet_pressure` on the production nozzle driver
//! (ALLMACH_THERMAL_NOZZLE) and reports M_throat, M_exit, the gap, and T_min.
//! Success = some inlet pressure gives M_exit >= M_throat (diverging section
//! accelerates) while staying supersonic and vacuum-free.
//!
//! Run: cargo test --features "dev-tests ui" --test nozzle_supersonic_branch_probe -- --ignored --nocapture

#![cfg(all(feature = "dev-tests", feature = "ui"))]

use cfd2::sim::{DriverBuild, SolverDriver};
use cfd2::solver::mesh::{generate_structured_nozzle_mesh, BoundarySides, BoundaryType, Mesh};
use cfd2::solver::model::allmach_thermal_model;
use cfd2::ui::fluid::Fluid;
use cfd2::ui::model_defaults::ALLMACH_THERMAL_NOZZLE;

const LENGTH: f64 = 3.0;
const HEIGHT: f64 = 1.0;
const THROAT_H: f64 = 0.40;
const THROAT_FRAC: f64 = 0.40;
const EXIT_H: f64 = 0.80;

fn nozzle() -> Mesh {
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

fn run(air: &Fluid, mesh: &Mesh, p_in: f32) -> Option<(f64, f64, f64, f64)> {
    let mut d = ALLMACH_THERMAL_NOZZLE;
    d.inlet_pressure = p_in;
    let params = d.to_runtime_params(air.density as f32, air.viscosity as f32, air.eos);
    let n = mesh.num_cells();
    let initial_u = vec![(params.inlet_velocity as f64, 0.0); n];
    let initial_p = vec![0.0; n];
    let DriverBuild { mut driver, .. } = pollster::block_on(SolverDriver::build(
        mesh, allmach_thermal_model().unwrap(), &params, &initial_u, &initial_p, None, None,
    ))
    .ok()?;
    driver.apply_params(&params);
    let mut solver = driver.into_solver();
    for _ in 0..300 {
        if solver.step_with_stats().is_err() {
            return None;
        }
    }
    let u = pollster::block_on(solver.get_field_vec2("U")).ok()?;
    let t = pollster::block_on(solver.get_field_scalar("T")).ok()?;
    if !u.iter().all(|(a, b)| a.is_finite() && b.is_finite()) {
        return None;
    }
    let c = 1.0 / (params.compressibility_psi as f64).sqrt();
    let mach: Vec<f64> = u.iter().map(|(a, b)| (a * a + b * b).sqrt() / c).collect();
    let m_throat = region_max(mesh, &mach, 0.35 * LENGTH, 0.45 * LENGTH);
    let m_exit = region_max(mesh, &mach, 0.85 * LENGTH, LENGTH);
    let t_min = t.iter().cloned().fold(f64::INFINITY, f64::min);
    Some((m_throat, m_exit, m_exit - m_throat, t_min))
}

#[test]
#[ignore]
fn supersonic_branch_inlet_pressure_sweep() {
    let air = Fluid::presets()[1].clone();
    let mesh = nozzle();
    println!("=== B+C nozzle: is M_exit >= M_throat (true accelerating supersonic branch) reachable? ===");
    println!("target: M_exit > 1 AND M_exit - M_throat >= 0 (diverging section ACCELERATES)");
    for p_in in [0.030_f32, 0.040, 0.050, 0.060, 0.070] {
        match run(&air, &mesh, p_in) {
            None => println!("  p_in={p_in:+.3}: UNSTABLE / non-finite"),
            Some((mt, me, gap, tmin)) => {
                let flag = if me > 1.0 && gap >= 0.0 { "  <== TRUE SUPERSONIC BRANCH" } else { "" };
                println!(
                    "  p_in={p_in:+.3}: M_throat={mt:.3} M_exit={me:.3} gap={gap:+.3} T_min={tmin:.4}{flag}"
                );
            }
        }
    }
}
