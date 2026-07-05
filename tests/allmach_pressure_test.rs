//! Validation for the all-Mach f32 pressure-based model (`allmach_pressure`):
//! `incompressible_momentum` plus a compressibility time term `ddt(psi,p)` in the
//! continuity/pressure equation (`psi = 1/c^2`). Asserts two properties: at `psi = 0`
//! the added term is inert and the model reproduces the incompressible solver (still
//! shedding a Kármán street); at `psi > 0` the implicit acoustic term sits on the
//! pressure diagonal and the solve stays bounded.

#![cfg(all(feature = "dev-tests", feature = "ui"))]

use cfd2::sim::{DriverBuild, SolverDriver};
use cfd2::solver::mesh::{generate_cut_cell_mesh, ChannelWithObstacle, Mesh};
use cfd2::solver::model::allmach_pressure_model;
use cfd2::solver::model::helpers::{SolverFieldAliasesExt, SolverRuntimeParamsExt};
use cfd2::solver::UnifiedSolver;
use cfd2::ui::fluid::Fluid;
use cfd2::ui::model_defaults::{gui_defaults_for, ModelGuiDefaults};
use nalgebra::{Point2, Vector2};

const VEL_CAP: f64 = 1.0;

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

fn actual_min_cell(mesh: &Mesh) -> f64 {
    mesh.cell_vol
        .iter()
        .map(|&v| v.sqrt())
        .fold(f64::INFINITY, f64::min)
}

/// Convective adaptive timestep (no sound speed — the model carries no EOS knob,
/// so the driver runs it in the incompressible coupled branch). Mirrors the GUI
/// worker's dt for the incompressible path.
fn convective_next_dt(d: &ModelGuiDefaults, prev_max_vel: f64, min_cell: f64, current_dt: f64) -> f64 {
    let adv_speed = prev_max_vel.max(d.inlet_velocity.abs() as f64);
    if min_cell > 1e-12 && adv_speed > 1e-12 {
        let mut next_dt = d.target_cfl as f64 * min_cell / adv_speed;
        if next_dt > current_dt * 1.2 {
            next_dt = current_dt * 1.2;
        }
        next_dt.clamp(1e-9, 100.0)
    } else {
        current_dt
    }
}

/// Build the all-Mach solver through the shared driver (incompressible branch:
/// `EosSpec::Constant` => no sound-speed knob => `SteppingMode::Coupled`), then
/// set the compressibility field `psi` uniformly.
fn build_allmach(d: &ModelGuiDefaults, fluid: &Fluid, mesh: &Mesh, psi: f64) -> UnifiedSolver {
    let params = d.to_runtime_params(fluid.density as f32, fluid.viscosity as f32, fluid.eos);
    let n = mesh.num_cells();
    let DriverBuild { mut driver, .. } = pollster::block_on(SolverDriver::build(
        mesh,
        allmach_pressure_model().expect("allmach model"),
        &params,
        &vec![(0.0, 0.0); n],
        &vec![0.0; n],
        None,
        None,
    ))
    .expect("driver build");
    driver.apply_params(&params);
    let mut solver = driver.into_solver();
    solver
        .set_field_scalar("psi", &vec![psi; n])
        .expect("set psi field");
    // The pressure-row ddt reads the decoupled `psi_precond` (low-Mach preconditioning is
    // a driver-only transient device). This raw-solver test pins compressibility directly,
    // so seed psi_precond = psi to keep the acoustic time term at the intended psi.
    solver
        .set_field_scalar("psi_precond", &vec![psi; n])
        .expect("set psi_precond field");
    // `rho` is a state-layout field (variable-density support); the driver's `set_density`
    // only sets the uniform constant, so initialise the per-cell field to rho_ref. With no
    // refresh it stays constant (incompressible); a barotropic refresh makes it compressible.
    solver
        .set_field_scalar("rho", &vec![fluid.density as f64; n])
        .expect("init rho field");
    solver
}

/// Refresh the per-cell density from the barotropic EOS `rho = rho_ref + psi*p`
/// (gauge pressure `p`), flooring it to stay positive. This is the host-side stand-in
/// for the production GPU density-refresh kernel; it makes the continuity genuinely
/// compressible (`div(rho*U)=0`).
fn refresh_rho(solver: &mut UnifiedSolver, rho_ref: f64, psi: f64) {
    let p = pollster::block_on(solver.get_p());
    let rho: Vec<f64> = p
        .iter()
        .map(|&pv| (rho_ref + psi * pv).max(0.05 * rho_ref))
        .collect();
    solver.set_field_scalar("rho", &rho).expect("refresh rho");
}

/// Drive the obstacle wake and sample `u_y` at the near-wake probe (1.5, 0.5).
/// Returns `(max|u| seen, u_y series, diverged)`.
fn drive_allmach_wake(
    d: &ModelGuiDefaults,
    air: &Fluid,
    mesh: &Mesh,
    steps: usize,
    sample_every: usize,
    psi: f64,
) -> (f64, Vec<f64>, bool) {
    let min_cell = actual_min_cell(mesh);
    let probe = (0..mesh.num_cells())
        .min_by(|&a, &b| {
            let da = (mesh.cell_cx[a] - 1.5).powi(2) + (mesh.cell_cy[a] - 0.5).powi(2);
            let db = (mesh.cell_cx[b] - 1.5).powi(2) + (mesh.cell_cy[b] - 0.5).powi(2);
            da.partial_cmp(&db).unwrap()
        })
        .unwrap();
    let mut solver = build_allmach(d, air, mesh, psi);
    let mut prev_max = 0.0;
    let mut max_seen = 0.0f64;
    let mut uy: Vec<f64> = Vec::new();
    let mut diverged = false;
    for step in 0..steps {
        let cur = solver.dt() as f64;
        let next = convective_next_dt(d, prev_max, min_cell, cur);
        solver.set_dt(next as f32);
        if solver.step_with_stats().is_err() {
            diverged = true;
            break;
        }
        if step % sample_every != 0 && step != steps - 1 {
            continue;
        }
        let u = pollster::block_on(solver.get_u());
        let mv = u
            .iter()
            .filter(|(x, y)| x.is_finite() && y.is_finite())
            .map(|(x, y)| (x * x + y * y).sqrt())
            .fold(0.0_f64, f64::max);
        prev_max = mv;
        max_seen = max_seen.max(mv);
        uy.push(u[probe].1);
        if !mv.is_finite() || mv > VEL_CAP {
            diverged = true;
            break;
        }
    }
    (max_seen, uy, diverged)
}

/// `(std, tail_min, tail_max)` of the post-transient tail of a wake `u_y` series.
fn wake_stats(uy: &[f64]) -> (f64, f64, f64) {
    let half = uy.len() / 2;
    let tail = &uy[half..];
    if tail.len() < 2 {
        return (0.0, 0.0, 0.0);
    }
    let mean = tail.iter().sum::<f64>() / tail.len() as f64;
    let var = tail.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / tail.len() as f64;
    let tmin = tail.iter().cloned().fold(f64::INFINITY, f64::min);
    let tmax = tail.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (var.sqrt(), tmin, tmax)
}

/// At `psi = 0` the all-Mach pressure-based model reduces exactly to the incompressible
/// solver and must shed a Kármán vortex street on the obstacle default — bounded, with a
/// self-sustained bidirectional `u_y` oscillation in the wake.
#[test]
fn allmach_psi_zero_sheds_vortex_street() {
    std::env::set_var("CFD2_QUIET", "1");
    let air = air();
    let d = gui_defaults_for("incompressible_momentum");
    let mesh = channel_obstacle_mesh();
    let u_in = d.inlet_velocity as f64;
    eprintln!(
        "[allmach/psi0] cells={} U={:.4e} scheme={:?}",
        mesh.num_cells(),
        u_in,
        d.advection_scheme
    );
    let (max_u, uy, diverged) = drive_allmach_wake(&d, &air, &mesh, 1000, 8, 0.0);
    let (std, uy_min, uy_max) = wake_stats(&uy);
    eprintln!(
        "[allmach/psi0] diverged={diverged} max|u|={max_u:.4e} ({:.2}xU) | wake u_y: std={std:.4e} ({:.3}xU) span=[{:+.3},{:+.3}]xU",
        max_u / u_in,
        std / u_in,
        uy_min / u_in,
        uy_max / u_in
    );
    assert!(!diverged, "allmach(psi=0) diverged (max|u|={max_u:.3e})");
    assert!(
        max_u < VEL_CAP && max_u < 10.0 * u_in,
        "allmach(psi=0) wake unbounded: max|u|={max_u:.3e} ({:.1}xU)",
        max_u / u_in
    );
    assert!(
        std > 0.05 * u_in,
        "allmach(psi=0) wake too steady (std={:.4}xU): not shedding — incompressible limit not recovered",
        std / u_in
    );
    assert!(
        uy_max > 0.05 * u_in && uy_min < -0.05 * u_in,
        "allmach(psi=0) wake does not reverse sign (span [{:+.3},{:+.3}]xU)",
        uy_min / u_in,
        uy_max / u_in
    );
}

/// Compressibility active + stable: with `psi > 0` the implicit acoustic term is
/// live on the pressure diagonal. Assert the solve stays bounded and the flow
/// still develops past uniform freestream (not frozen). `psi` is reported in terms
/// of the implied sound speed / Mach number.
#[test]
fn allmach_psi_positive_stays_bounded() {
    std::env::set_var("CFD2_QUIET", "1");
    let air = air();
    let d = gui_defaults_for("incompressible_momentum");
    let mesh = channel_obstacle_mesh();
    let u_in = d.inlet_velocity as f64;
    let psi: f64 = 10.0; // c = 1/sqrt(psi) ~ 0.316 m/s => Mach = U/c ~ 0.035
    let c = 1.0 / psi.sqrt();
    eprintln!(
        "[allmach/psi+] psi={psi} c={c:.4} Mach={:.4} U={:.4e}",
        u_in / c,
        u_in
    );
    let (max_u, uy, diverged) = drive_allmach_wake(&d, &air, &mesh, 600, 8, psi);
    let (std, uy_min, uy_max) = wake_stats(&uy);
    eprintln!(
        "[allmach/psi+] diverged={diverged} max|u|={max_u:.4e} ({:.2}xU) | wake u_y: std={:.3}xU span=[{:+.3},{:+.3}]xU",
        max_u / u_in,
        std / u_in,
        uy_min / u_in,
        uy_max / u_in
    );
    assert!(!diverged, "allmach(psi>0) diverged (max|u|={max_u:.3e})");
    assert!(
        max_u < VEL_CAP && max_u < 10.0 * u_in,
        "allmach(psi>0) unbounded: max|u|={max_u:.3e} ({:.1}xU)",
        max_u / u_in
    );
    assert!(
        max_u > 1.05 * u_in,
        "allmach(psi>0) frozen at freestream (max|u|={:.3}xU): compressibility over-damped the flow",
        max_u / u_in
    );
}

/// Drive a STEADY obstacle wake (low Re, no shedding) and return the wake-strength
/// development curve `max|u|` per iteration. When `lts` is true, fill the per-cell
/// `dt_local` field each step with the local stable step `cfl*sqrt(vol_c)/(|u_c|+floor)`
/// (Local Time Stepping); when false, `dt_local` stays 0 and the assembly uses the
/// global (sliver-throttled) `constants.dt` (time-accurate marching).
fn drive_allmach_steady(
    d: &ModelGuiDefaults,
    air: &Fluid,
    mesh: &Mesh,
    steps: usize,
    lts: bool,
    visc_override: Option<f64>,
    maxratio: f64,
) -> Vec<f64> {
    let min_cell = actual_min_cell(mesh);
    let n = mesh.num_cells();
    let mut solver = build_allmach(d, air, mesh, 0.0);
    if let Some(mu) = visc_override {
        let _ = solver.set_viscosity(mu as f32);
    }
    let mut prev_max = 0.0;
    let mut ke_curve = Vec::new();
    for _ in 0..steps {
        let u = pollster::block_on(solver.get_u());
        // Total kinetic energy: a GLOBAL convergence metric (the whole field, not the
        // peak |u| which saturates near the cylinder almost immediately).
        let ke: f64 = (0..n)
            .map(|c| (u[c].0 * u[c].0 + u[c].1 * u[c].1) * mesh.cell_vol[c])
            .sum();
        ke_curve.push(ke);
        prev_max = u
            .iter()
            .map(|(x, y)| (x * x + y * y).sqrt())
            .fold(0.0_f64, f64::max);
        // Global (time-accurate) dt = cfl*min_cell/adv_speed, throttled by the smallest sliver.
        let gdt = convective_next_dt(d, prev_max, min_cell, solver.dt() as f64);
        solver.set_dt(gdt as f32);
        if lts {
            // LOCAL pseudo-time: each cell steps at the global dt scaled UP by its
            // size relative to the throttling sliver (capped). Tying it to the ramped
            // global dt keeps the transient stable; the cap bounds the disparity so
            // the lagged Rhie-Chow coupling (sized by the global dt) stays consistent.
            let dtl: Vec<f64> = (0..n)
                .map(|c| {
                    let ratio = (mesh.cell_vol[c].sqrt() / min_cell).clamp(1.0, maxratio);
                    gdt * ratio
                })
                .collect();
            solver.set_field_scalar("dt_local", &dtl).expect("set dt_local");
        }
        if solver.step_with_stats().is_err() {
            break;
        }
    }
    ke_curve
}

/// Variable density: refresh `rho = rho_ref + psi*p` from the barotropic EOS each step,
/// so the Rhie–Chow mass flux carries a per-cell density and the continuity is genuinely
/// compressible (`div(rho*U)=0`, not `div(U)=0`). The feedback loop rho<->p<->U must stay
/// bounded and the density must vary across the domain in response to the pressure field.
#[test]
fn allmach_variable_density_compressible() {
    std::env::set_var("CFD2_QUIET", "1");
    let air = air();
    let mut d = gui_defaults_for("incompressible_momentum");
    d.inlet_velocity = 0.011;
    let mesh = channel_obstacle_mesh();
    let rho_ref = air.density;
    let psi: f64 = 50.0; // c = 1/sqrt(psi) ~ 0.141, Mach = U/c ~ 0.078
    let min_cell = actual_min_cell(&mesh);
    let u_in = d.inlet_velocity as f64;
    eprintln!(
        "[allmach/var-rho] psi={psi} c={:.4} Mach={:.4} rho_ref={rho_ref}",
        1.0 / psi.sqrt(),
        u_in / (1.0 / psi.sqrt())
    );
    let mut solver = build_allmach(&d, &air, &mesh, psi);
    let mut prev_max = 0.0;
    let mut diverged = false;
    for _ in 0..400 {
        refresh_rho(&mut solver, rho_ref, psi);
        let gdt = convective_next_dt(&d, prev_max, min_cell, solver.dt() as f64);
        solver.set_dt(gdt as f32);
        if solver.step_with_stats().is_err() {
            diverged = true;
            break;
        }
        let u = pollster::block_on(solver.get_u());
        prev_max = u
            .iter()
            .map(|(x, y)| (x * x + y * y).sqrt())
            .fold(0.0_f64, f64::max);
    }
    // Final density spread, reconstructed from the gauge pressure via the EOS.
    let p = pollster::block_on(solver.get_p());
    let (mut rmin, mut rmax) = (f64::INFINITY, f64::NEG_INFINITY);
    for &pv in &p {
        let r = (rho_ref + psi * pv).max(0.05 * rho_ref);
        rmin = rmin.min(r);
        rmax = rmax.max(r);
    }
    let spread = (rmax - rmin) / rho_ref;
    eprintln!(
        "[allmach/var-rho] diverged={diverged} max|u|={prev_max:.4e} ({:.2}xU) | rho in [{rmin:.5},{rmax:.5}] spread={:.3e} of rho_ref",
        prev_max / u_in,
        spread
    );
    assert!(!diverged, "variable-density run diverged (max|u|={prev_max:.3e})");
    assert!(
        prev_max < VEL_CAP && prev_max < 10.0 * u_in,
        "variable-density run unbounded: max|u|={prev_max:.3e}"
    );
    assert!(rmin > 0.0, "density went non-positive: rho_min={rmin:.4}");
    assert!(
        spread > 1e-4,
        "density did not vary (spread={spread:.2e}): the compressible continuity is inert"
    );
}

/// Local Time Stepping steady-state acceleration: on a sliver-throttled steady case
/// (Re~33, kept steady by higher viscosity but fast enough that the global time-accurate
/// dt is throttled to the tiny cut-cell sliver CFL), per-cell `dt_local` lets each bulk
/// cell march at the global dt scaled by its size (capped). Within a fixed iteration
/// budget LTS must develop the global flow (total kinetic energy) substantially more than
/// the throttled global march, and stay bounded.
#[test]
fn lts_accelerates_steady_obstacle() {
    std::env::set_var("CFD2_QUIET", "1");
    let air = air();
    let mut d = gui_defaults_for("incompressible_momentum");
    d.inlet_velocity = 0.05; // higher speed => global dt throttled hard by the slivers
    let mesh = channel_obstacle_mesh();
    let u_in = d.inlet_velocity as f64;
    // Keep it laminar/steady (Re<47) despite the higher speed via higher viscosity.
    let mu = 3.7e-4; // nu = mu/rho = 3.0e-4 => Re = U*D/nu = 0.05*0.2/3.0e-4 ~ 33
    let nu = mu / air.density;
    let min_cell = actual_min_cell(&mesh);
    let maxratio = 6.0;
    eprintln!(
        "[lts] Re={:.0} cells={} min_cell={:.3e} size-disparity={:.1}x maxratio={maxratio}",
        u_in * 0.2 / nu,
        mesh.num_cells(),
        min_cell,
        0.025 / min_cell
    );

    let steps = 300;
    let global = drive_allmach_steady(&d, &air, &mesh, steps, false, Some(mu), maxratio);
    let lts = drive_allmach_steady(&d, &air, &mesh, steps, true, Some(mu), maxratio);
    let g_final = *global.last().unwrap();
    let l_final = *lts.last().unwrap();
    let at = |c: &[f64], i: usize| c.get(i).copied().unwrap_or(f64::NAN);
    let norm = l_final.max(g_final).max(1e-30);
    eprintln!(
        "[lts] global KE/KEref: @50={:.3} @150={:.3} @final={:.3}",
        at(&global, 50) / norm, at(&global, 150) / norm, g_final / norm
    );
    eprintln!(
        "[lts] LTS    KE/KEref: @50={:.3} @150={:.3} @final={:.3}",
        at(&lts, 50) / norm, at(&lts, 150) / norm, l_final / norm
    );
    assert!(l_final.is_finite() && g_final.is_finite() && l_final > 0.0, "non-finite/zero KE");
    // Bounded: LTS must not blow up.
    let lmax = lts.iter().cloned().fold(0.0_f64, f64::max);
    assert!(
        lmax < 1e6 * g_final.max(1e-12),
        "LTS unbounded: max KE={lmax:.3e} vs global final {g_final:.3e}"
    );

    // Both runs must converge to the SAME steady state (LTS solves the correct steady
    // problem — at steady the per-cell ddt term vanishes regardless of dt_local).
    let reference = l_final.max(g_final);
    assert!(
        (l_final - g_final).abs() / reference < 0.05,
        "LTS converged to a different steady state: LTS KE={l_final:.4e} vs global {g_final:.4e}"
    );

    // ACCELERATION metric: the SETTLING iteration = the last step still outside a +/-2%
    // band around the (more-converged) steady KE, +1. This is robust to the impulsive-
    // start KE overshoot (which trips simple first-crossing thresholds for both runs).
    let band = 0.02 * reference;
    let settle_iter = |c: &[f64]| {
        (0..c.len())
            .rev()
            .find(|&i| (c[i] - reference).abs() > band)
            .map(|i| i + 2)
            .unwrap_or(1)
    };
    let il = settle_iter(&lts);
    let ig = settle_iter(&global);
    eprintln!(
        "[lts] settling iteration (last outside +/-2% of steady KE): global={ig} LTS={il}  =>  {:.2}x speedup",
        ig as f64 / il.max(1) as f64
    );
    // LTS must reach steady in meaningfully fewer iterations. The speedup is bounded by
    // the bulk/sliver size disparity (3.9x) and is modest because this IMPLICIT pressure-
    // coupled solver already converges fast (the elliptic pressure solve propagates info
    // globally each iteration) — LTS shines for explicit/convection-limited solvers.
    assert!(
        (il as f64) < 0.8 * (ig as f64),
        "LTS did not accelerate steady convergence: settled at iter {il} (LTS) vs {ig} (global)"
    );
}
