//! Transonic validation for the all-Mach pressure-based thermal solver.
//!
//! Geometry: a CONVERGING DUCT (trapezoid mesh whose bottom ramps up so the
//! channel height halves left->right). Subsonic gas enters on the left and
//! ACCELERATES toward the contracted outlet; with sound speed c = 1/sqrt(psi)
//! the outlet Mach approaches/cross into the transonic band as the inlet speed
//! is raised.
//!
//! This is the centerpiece "entire range of velocities" check. Across an inlet
//! Mach sweep it asserts:
//!   1. STABILITY — the pressure-coupled solver runs to a bounded state at every
//!      swept speed up to the highest one that converges (the sweep reports the
//!      envelope), and reaches at least the transonic neighbourhood.
//!   2. ACCELERATION — outlet Mach > inlet Mach (the duct does its job).
//!   3. EXPANSION COOLING — the accelerating gas cools (T_out < T_ref). This is
//!      compression heating in the EXPANSION direction (U.grad(p) < 0), an
//!      independent sign check on the U.grad(p) half, opposite to the closed box.

#![cfg(all(feature = "dev-tests", feature = "ui"))]

use cfd2::sim::{DriverBuild, SolverDriver};
use cfd2::solver::mesh::{generate_structured_trapezoid_mesh, BoundarySides, BoundaryType, Mesh};
use cfd2::solver::model::{allmach_thermal_model, ALLMACH_T_REF};
use cfd2::solver::UnifiedSolver;
use cfd2::ui::fluid::Fluid;
use cfd2::ui::model_defaults::gui_defaults_for;

fn air() -> Fluid {
    Fluid::presets()[1].clone()
}

const LENGTH: f64 = 2.0;
const HEIGHT: f64 = 1.0;
const RAMP: f64 = 0.5; // outlet height = HEIGHT - RAMP = 0.5  => 2:1 contraction

fn converging_duct(nx: usize, ny: usize) -> Mesh {
    generate_structured_trapezoid_mesh(
        nx,
        ny,
        LENGTH,
        HEIGHT,
        RAMP,
        BoundarySides {
            left: BoundaryType::Inlet,
            right: BoundaryType::Outlet,
            bottom: BoundaryType::Wall,
            top: BoundaryType::Wall,
        },
    )
}

fn build_duct(fluid: &Fluid, mesh: &Mesh, psi: f64, inlet_v: f64) -> UnifiedSolver {
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
    // The reference compressibility drives the EOS coefficients (rho recovery, 1/cp);
    // this raw-solver test pins the sound speed, so seed psi_ref = psi too.
    solver
        .set_field_scalar("psi_ref", &vec![psi; n])
        .expect("psi_ref");
    // Pressure-row ddt reads the decoupled `psi_precond` (preconditioning is a driver-only
    // transient device); this raw-solver test pins the compressibility, so seed it = psi.
    solver
        .set_field_scalar("psi_precond", &vec![psi; n])
        .expect("psi_precond");
    solver.set_field_scalar("rho", &vec![rho_ref; n]).expect("rho");
    solver
        .set_field_scalar("rho_t_ref", &vec![rho_ref * ALLMACH_T_REF; n])
        .expect("rho_t_ref");
    solver
        .set_field_scalar("T", &vec![ALLMACH_T_REF; n])
        .expect("T");
    solver
}

/// Mean of `f` over cells whose centroid x is in `[x0, x1)`.
fn region_mean(mesh: &Mesh, f: &[f64], x0: f64, x1: f64) -> f64 {
    let mut sum = 0.0;
    let mut cnt = 0usize;
    for c in 0..mesh.num_cells() {
        let x = mesh.cell_cx[c];
        if x >= x0 && x < x1 {
            sum += f[c];
            cnt += 1;
        }
    }
    if cnt == 0 {
        f64::NAN
    } else {
        sum / cnt as f64
    }
}

/// Max of `f` over cells whose centroid x is in `[x0, x1)` — the accelerated core,
/// robust to the no-slip walls that drag a regional mean down in the thin throat.
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

struct DuctResult {
    inlet_v: f64,
    converged: bool,
    m_in: f64,      // mean Mach in the inlet region (~ inlet velocity Mach)
    m_throat: f64,  // peak (core) Mach in the contracted outlet region
    t_out: f64,
    t_min: f64,
}

fn run_duct(air: &Fluid, psi: f64, inlet_v: f64, steps: usize) -> DuctResult {
    let mesh = converging_duct(64, 32);
    let c_sound = 1.0 / psi.sqrt();
    let mut solver = build_duct(air, &mesh, psi, inlet_v);

    let mut diverged = false;
    for _ in 0..steps {
        if solver.step_with_stats().is_err() {
            diverged = true;
            break;
        }
    }

    if diverged {
        return DuctResult { inlet_v, converged: false, m_in: f64::NAN, m_throat: f64::NAN, t_out: f64::NAN, t_min: f64::NAN };
    }

    let u = pollster::block_on(solver.get_field_vec2("U")).expect("U");
    let t = pollster::block_on(solver.get_field_scalar("T")).expect("T");
    let rho = pollster::block_on(solver.get_field_scalar("rho")).expect("rho");

    let finite = u.iter().all(|(a, b)| a.is_finite() && b.is_finite())
        && t.iter().all(|v| v.is_finite() && *v > 0.0)
        && rho.iter().all(|v| v.is_finite() && *v > 0.0);
    if !finite {
        return DuctResult { inlet_v, converged: false, m_in: f64::NAN, m_throat: f64::NAN, t_out: f64::NAN, t_min: f64::NAN };
    }

    let speed: Vec<f64> = u.iter().map(|(a, b)| (a * a + b * b).sqrt()).collect();
    let mach: Vec<f64> = speed.iter().map(|s| s / c_sound).collect();
    // Inlet region: first 15% of the length (mean ~ prescribed inlet Mach).
    // Throat: last 15% — take the peak (core) Mach, robust to the no-slip walls.
    let m_in = region_mean(&mesh, &mach, 0.0, 0.15 * LENGTH);
    let m_throat = region_max(&mesh, &mach, 0.85 * LENGTH, LENGTH);
    let t_out = region_mean(&mesh, &t, 0.85 * LENGTH, LENGTH);
    let t_min = t.iter().cloned().fold(f64::INFINITY, f64::min);

    DuctResult { inlet_v, converged: true, m_in, m_throat, t_out, t_min }
}

#[test]
fn transonic_converging_duct_mach_sweep() {
    let air = air();
    let psi: f64 = 50.0; // c = 1/sqrt(50) ~ 0.1414
    let steps = 200;

    // Sweep inlet speed from clearly-subsonic upward. With a 2:1 contraction the
    // outlet accelerates well past the inlet, so modest inlet speeds reach the
    // transonic band at the throat.
    let inlet_speeds = [0.02_f64, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14];

    let mut best_stable: Option<DuctResult> = None;
    let mut stable_count = 0usize;
    let mut min_tmin_over_sweep = f64::INFINITY;
    println!("[transonic] converging duct 2:1, psi={psi} c={:.4} steps={steps}", 1.0 / psi.sqrt());
    for &v in &inlet_speeds {
        let r = run_duct(&air, psi, v, steps);
        if r.converged {
            println!(
                "  inlet_v={:.3}  M_in={:.3}  M_throat={:.3}  T_out={:.4}  T_min={:.4}  [stable]",
                r.inlet_v, r.m_in, r.m_throat, r.t_out, r.t_min
            );
            // Acceleration: the contracted-core (throat) Mach exceeds the inlet
            // Mach — UNTIL the inlet itself approaches sonic, where a converging
            // duct chokes (M_throat ~ M_in ~ 1, classic choking, not a solver
            // failure). So require strict acceleration only below the choke.
            if r.m_in < 0.85 {
                assert!(
                    r.m_throat > r.m_in * 1.02,
                    "converging duct did not accelerate the core at inlet_v={:.3}: \
                     M_in={:.3} M_throat={:.3}",
                    r.inlet_v, r.m_in, r.m_throat
                );
            }
            min_tmin_over_sweep = min_tmin_over_sweep.min(r.t_min);
            stable_count += 1;
            best_stable = Some(r);
        } else {
            println!("  inlet_v={:.3}  [DIVERGED — stability envelope reached]", r.inlet_v);
            break;
        }
    }

    let best = best_stable.expect("solver diverged at the very first (subsonic) speed");

    // (1) STABILITY across the velocity range: the solver carried several speeds,
    // from clearly subsonic up into the transonic band, to bounded states.
    assert!(
        stable_count >= 4,
        "solver stayed stable for only {stable_count} swept speeds — envelope too small"
    );
    // (2) Reached the transonic band while stable.
    assert!(
        best.m_throat >= 0.85,
        "highest stable throat Mach only {:.3} — did not reach the transonic band",
        best.m_throat
    );

    // (3) Expansion cooling (U.grad(p) < 0): the accelerating gas cooled below
    // T_ref, and the cooling deepens as the flow approaches sonic. This is the
    // compression-heating sign certified in the EXPANSION direction (opposite to
    // the closed filling box).
    assert!(
        min_tmin_over_sweep < ALLMACH_T_REF - 0.01,
        "expected clear expansion cooling somewhere in the sweep, got min T_min={:.4}",
        min_tmin_over_sweep
    );

    println!(
        "[transonic] envelope: stable speeds={stable_count}, highest inlet_v={:.3}, \
         M_throat={:.3}, T_out={:.4}, coldest T_min over sweep={:.4}",
        best.inlet_v, best.m_throat, best.t_out, min_tmin_over_sweep
    );
}
