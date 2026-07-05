//! PROBE (`#[ignore]`'d): does the all-Mach nozzle (real-compressibility +
//! density-upwind) converge to a bounded pseudo-laminar solution at the REAL
//! speed of sound (no compressibility exaggeration), driven by a pressure inlet?
//! The numerical questions at real c≈347: does preconditioning tame the acoustic
//! stiffness, and does gauge-pressure stay above the EOS vacuum floor (P_abs > 0)
//! while the flow goes transonic?
//!
//! Setup: exaggeration ×1 ⇒ psi = air's real 1/c² ≈ 8.3e-6, c ≈ 347, P_REF =
//! rho/psi ≈ 1.48e5. To reach M~1 the pressure drop is O(½ρc²) ≈ 7e4, so we sweep
//! inlet pressures of that magnitude. Velocity-scaled adaptive dt keeps step 1
//! in-bounds at the ~100s-of-m/s throughflow.
//!
//! Run: cargo test --features "dev-tests ui" --test nozzle_real_c_pseudolaminar_probe -- --ignored --nocapture

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

fn min_cell_h(mesh: &Mesh) -> f64 {
    mesh.cell_vol.iter().map(|&v| v.sqrt()).fold(f64::INFINITY, f64::min)
}

struct Out {
    m_throat: f64,
    m_exit: f64,
    min_pabs: f64,
    min_rho: f64,
    max_u: f64,
    finite: bool,
}

fn run(air: &Fluid, mesh: &Mesh, p_inlet: f32, steps: usize) -> Out {
    // psi = real air.compressibility() = 1/c² (no exaggeration).
    let mut d = ALLMACH_THERMAL_NOZZLE; // pressure_inlet = true
    d.inlet_pressure = p_inlet;
    // Throughflow scale for the preconditioner + CFL ~ Bernoulli sqrt(2*dP/rho).
    let u_scale = (2.0 * p_inlet as f64 / air.density).sqrt();
    d.inlet_velocity = u_scale as f32;
    // Velocity-scaled adaptive dt so step 1 is in-bounds at ~100s of m/s.
    let h = min_cell_h(mesh);
    d.adaptive_dt = true;
    d.target_cfl = 0.4;
    d.timestep = 0.2 * h / u_scale;

    let params = d.to_runtime_params(air.density as f32, air.viscosity as f32, air.eos);
    let psi = params.compressibility_psi.max(0.0) as f64; // real 1/c^2
    let c = 1.0 / psi.sqrt(); // ~347
    let p_ref = air.density / psi; // ~1.48e5
    let n = mesh.num_cells();
    let initial_u = vec![(u_scale, 0.0); n];
    let initial_p = vec![0.0; n];
    let DriverBuild { mut driver, .. } = pollster::block_on(SolverDriver::build(
        mesh, allmach_thermal_model().unwrap(), &params, &initial_u, &initial_p, None, None,
    ))
    .expect("build");
    driver.apply_params(&params);
    let mut solver = driver.into_solver();

    for s in 0..steps {
        if solver.step_with_stats().is_err() {
            println!("    [diverged at step {s}]");
            return Out { m_throat: f64::NAN, m_exit: f64::NAN, min_pabs: f64::NAN, min_rho: f64::NAN, max_u: f64::NAN, finite: false };
        }
    }
    let u = pollster::block_on(solver.get_field_vec2("U")).unwrap();
    let p = pollster::block_on(solver.get_field_scalar("p")).unwrap();
    let rho = pollster::block_on(solver.get_field_scalar("rho")).unwrap();
    let finite = u.iter().all(|(a, b)| a.is_finite() && b.is_finite())
        && p.iter().all(|v| v.is_finite())
        && rho.iter().all(|v| v.is_finite());
    let mach: Vec<f64> = u.iter().map(|(a, b)| (a * a + b * b).sqrt() / c).collect();
    Out {
        m_throat: region_max(mesh, &mach, 0.35 * LENGTH, 0.45 * LENGTH),
        m_exit: region_max(mesh, &mach, 0.85 * LENGTH, LENGTH),
        min_pabs: p.iter().map(|&pv| p_ref + pv).fold(f64::INFINITY, f64::min),
        min_rho: rho.iter().cloned().fold(f64::INFINITY, f64::min),
        max_u: u.iter().map(|(a, b)| (a * a + b * b).sqrt()).fold(0.0, f64::max),
        finite,
    }
}

#[test]
#[ignore]
fn real_c_pseudolaminar_nozzle_sweep() {
    let air = Fluid::presets()[1].clone();
    let mesh = nozzle();
    let c = air.sound_speed();
    let psi = air.compressibility();
    let p_ref = air.density / psi;
    println!("=== REAL-c nozzle (no exaggeration): c={c:.1} psi={psi:.3e} P_REF={p_ref:.3e} ===");
    println!("Q: does the laminar solver converge to a BOUNDED pseudo-laminar solution, or hit the vacuum floor?");
    println!("BOUNDED+TRANSONIC = finite + M_throat~1 + min_P_abs>0 (no vacuum)");
    let steps = 500;
    for p_inlet in [3.0e4_f32, 6.0e4, 9.0e4, 1.2e5] {
        let r = run(&air, &mesh, p_inlet, steps);
        if !r.finite {
            println!("  p_inlet={p_inlet:.2e}: DIVERGED / non-finite (max_u={:.2e})", r.max_u);
        } else {
            let vac = if r.min_pabs <= 0.0 { "  <== CROSSED VACUUM" } else { "" };
            println!(
                "  p_inlet={p_inlet:.2e}: M_throat={:.3} M_exit={:.3} max_u={:.1} | min_P_abs={:+.3e} min_rho={:.4}{vac}",
                r.m_throat, r.m_exit, r.max_u, r.min_pabs, r.min_rho
            );
        }
    }
}
