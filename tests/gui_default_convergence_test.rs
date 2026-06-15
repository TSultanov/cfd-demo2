//! Headless gate for the per-model default GUI solver parameters.
//!
//! The desktop GUI applies model-aware defaults (`cfd2::ui::model_defaults`).
//! This test is the arbiter for them: it builds the GUI's **actual default
//! geometries** (the cut-cell BackwardsStep and Channel-with-obstacle meshes, at
//! the default 0.025 cell size) with the real inlet boundary conditions, then
//! replicates the GUI worker loop (acoustic-aware adaptive-dt — or a fixed dt when
//! the model default disables it — + `set_*` parameter application +
//! `step_with_stats`) and asserts the shipped defaults do not diverge or produce
//! unphysical (checkerboard) fields.
//!
//! An earlier version of this gate used a closed lid cavity as a proxy and MISSED
//! two real failures: the compressible backstep checkerboards (a through-flow,
//! low-Mach instability the closed cavity never excited) and the incompressible
//! channel-with-obstacle diverges (high-Re Air past the cylinder on small cut-cell
//! slivers). Hence: test the real geometries.
//!
//! Tuning aids (run with `--ignored`): `sweep_compressible_backstep` and
//! `sweep_incompressible_obstacle` grid the stability knobs.

#![cfg(all(feature = "dev-tests", feature = "ui"))]

use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep, ChannelWithObstacle, Mesh};
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::model::helpers::{
    SolverCompressibleIdealGasExt, SolverCompressibleInletExt, SolverFieldAliasesExt,
    SolverIncompressibleControlsExt, SolverInletVelocityExt, SolverRuntimeParamsExt,
};
use cfd2::solver::model::{compressible_model_with_eos, incompressible_momentum_model};
use cfd2::solver::{GpuLowMachPrecondModel, SolverConfig, SteppingMode, UnifiedSolver};
use cfd2::ui::fluid::Fluid;
use cfd2::ui::model_defaults::{gui_defaults_for, ModelGuiDefaults};
use nalgebra::{Point2, Vector2};

const N_STEPS: usize = 150;
/// Compressible gate horizon: long enough to surface the slow low-Mach inlet
/// blow-up (unfixed: bounded for ~1000 steps, then max|u| rockets past the cap by
/// ~step 1300). 2000 steps clears that with margin; the fixed default holds at
/// the freestream (max|u| ~ inlet) the whole way.
const COMPRESSIBLE_STEPS: usize = 2000;
const READBACK_EVERY: usize = 5;
/// With the laminar low inlet speeds the physical max |u| is small (a few x the
/// inlet, ~10^-2). Any genuine instability — checkerboard or divergence — is a
/// numerical mode that amplifies far past the physical scale, so this absolute
/// cap (well above physical, far below an instability) flags both.
const VEL_CAP: f64 = 1.0;

fn air() -> Fluid {
    Fluid::presets()[1].clone()
}

fn backstep_mesh() -> Mesh {
    let length = 3.5;
    let geo = BackwardsStep {
        length,
        height_inlet: 0.5,
        height_outlet: 1.0,
        step_x: 0.5,
    };
    let mut mesh = generate_cut_cell_mesh(&geo, 0.025, 0.025, 1.2, Vector2::new(length, 1.0));
    mesh.smooth(&geo, 0.3, 50);
    mesh
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

/// `actual_min_cell_size` the GUI worker computes for adaptive-dt.
fn actual_min_cell(mesh: &Mesh) -> f64 {
    mesh.cell_vol
        .iter()
        .map(|&v| v.sqrt())
        .fold(f64::INFINITY, f64::min)
}

#[derive(Debug, Clone, Copy)]
struct Sample {
    step: usize,
    dt: f64,
    max_vel: f64,
    nonfinite_u: usize,
    p_finite: bool,
    p_min: f64,
    p_max: f64,
    rho_min: f64,
    rho_max: f64,
    outer_iters: u32,
}

struct DriveResult {
    samples: Vec<Sample>,
    diverged: bool,
    diverge_step: Option<usize>,
}

/// GUI worker adaptive timestep (true sound speed reduced by low-Mach), matching
/// `solver_worker_main` in `app.rs`.
fn adaptive_next_dt(
    d: &ModelGuiDefaults,
    prev_max_vel: f64,
    density: f64,
    eos: &EosSpec,
    supports_sound_speed: bool,
    min_cell: f64,
    current_dt: f64,
) -> f64 {
    let sound_speed = if supports_sound_speed {
        eos.sound_speed(density)
    } else {
        0.0
    };
    let adv_speed = prev_max_vel.max(d.inlet_velocity.abs() as f64);
    let effective_sound_speed = match d.low_mach_model {
        GpuLowMachPrecondModel::Off => sound_speed,
        GpuLowMachPrecondModel::Legacy => sound_speed.min(adv_speed),
        GpuLowMachPrecondModel::WeissSmith => {
            let theta = (d.low_mach_theta_floor as f64).max(0.0);
            let c_floor = sound_speed * theta.sqrt();
            sound_speed.min(adv_speed.max(c_floor))
        }
    };
    let wave_speed = adv_speed + effective_sound_speed;
    if min_cell > 1e-12 && wave_speed.is_finite() && wave_speed > 1e-12 {
        let mut next_dt = d.target_cfl * min_cell / wave_speed;
        if next_dt > current_dt * 1.2 {
            next_dt = current_dt * 1.2;
        }
        next_dt.clamp(1e-9, 100.0)
    } else {
        current_dt
    }
}

/// Drives `solver` for `N_STEPS`, mirroring the GUI worker. Records (never panics
/// on) divergence so the gate and the sweeps can react.
fn drive(
    solver: &mut UnifiedSolver,
    d: &ModelGuiDefaults,
    density: f64,
    eos: &EosSpec,
    supports_sound_speed: bool,
    read_rho: bool,
    min_cell: f64,
    n_steps: usize,
) -> DriveResult {
    let mut prev_max_vel = 0.0_f64;
    let mut samples = Vec::new();
    let mut diverged = false;
    let mut diverge_step = None;

    for step in 0..n_steps {
        if d.adaptive_dt {
            let current_dt = solver.dt() as f64;
            let next_dt =
                adaptive_next_dt(d, prev_max_vel, density, eos, supports_sound_speed, min_cell, current_dt);
            solver.set_dt(next_dt as f32);
        } else {
            solver.set_dt(d.timestep as f32);
        }

        if solver.step_with_stats().is_err() {
            diverged = true;
            diverge_step = Some(step);
            break;
        }
        let outer = solver.step_stats().outer_iterations.unwrap_or(0);

        let do_read = step % READBACK_EVERY == 0 || step == n_steps - 1;
        if !do_read {
            continue;
        }

        let u = pollster::block_on(solver.get_u());
        let p = pollster::block_on(solver.get_p());

        let mut max_vel = 0.0_f64;
        let mut nonfinite_u = 0usize;
        for (vx, vy) in &u {
            if !(vx.is_finite() && vy.is_finite()) {
                nonfinite_u += 1;
                continue;
            }
            let v = (vx * vx + vy * vy).sqrt();
            if v > max_vel {
                max_vel = v;
            }
        }
        prev_max_vel = if max_vel.is_finite() { max_vel } else { 0.0 };

        let p_finite = p.iter().all(|x| x.is_finite());
        let p_min = p.iter().cloned().fold(f64::INFINITY, f64::min);
        let p_max = p.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let (rho_min, rho_max) = if read_rho {
            let rho = pollster::block_on(solver.get_rho());
            (
                rho.iter().cloned().fold(f64::INFINITY, f64::min),
                rho.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            )
        } else {
            (1.0, 1.0)
        };

        samples.push(Sample {
            step,
            dt: solver.dt() as f64,
            max_vel,
            nonfinite_u,
            p_finite,
            p_min,
            p_max,
            rho_min,
            rho_max,
            outer_iters: outer,
        });

        if nonfinite_u > 0 || !p_finite || !max_vel.is_finite() || max_vel > VEL_CAP {
            diverged = true;
            diverge_step = Some(step);
            break;
        }
    }

    DriveResult {
        samples,
        diverged,
        diverge_step,
    }
}

fn print_trace(label: &str, res: &DriveResult) {
    eprintln!(
        "[{label}] {} samples, diverged={} (step {:?}):",
        res.samples.len(),
        res.diverged,
        res.diverge_step
    );
    for s in &res.samples {
        eprintln!(
            "  step {:>4} dt={:.3e} max|u|={:.4e} p=[{:.3e},{:.3e}] rho=[{:.3},{:.3}] outer={}",
            s.step, s.dt, s.max_vel, s.p_min, s.p_max, s.rho_min, s.rho_max, s.outer_iters
        );
    }
}

fn assert_bounded(label: &str, res: &DriveResult, p_lo: f64, p_hi: f64, rho_lo: f64, rho_hi: f64) {
    assert!(
        !res.diverged,
        "[{label}] diverged / went unphysical at step {:?} (max|u| exceeded {:.0} or non-finite)",
        res.diverge_step, VEL_CAP
    );
    for s in &res.samples {
        assert_eq!(s.nonfinite_u, 0, "[{label}] non-finite velocity at step {}", s.step);
        assert!(s.p_finite, "[{label}] non-finite pressure at step {}", s.step);
        assert!(
            s.max_vel.is_finite() && s.max_vel < VEL_CAP,
            "[{label}] velocity unphysical at step {}: max|u|={:.3e} (cap {:.0})",
            s.step, s.max_vel, VEL_CAP
        );
        assert!(
            s.p_min > p_lo && s.p_max < p_hi,
            "[{label}] pressure out of bounds at step {}: [{:.3e},{:.3e}]",
            s.step, s.p_min, s.p_max
        );
        assert!(
            s.rho_min > rho_lo && s.rho_max < rho_hi,
            "[{label}] density out of bounds at step {}: [{:.3e},{:.3e}]",
            s.step, s.rho_min, s.rho_max
        );
    }
}

fn build_incompressible(d: &ModelGuiDefaults, fluid: &Fluid, mesh: &Mesh) -> UnifiedSolver {
    let mut solver = pollster::block_on(UnifiedSolver::new(
        mesh,
        incompressible_momentum_model().expect("incompressible model"),
        SolverConfig {
            advection_scheme: d.advection_scheme,
            time_scheme: d.time_scheme,
            preconditioner: d.preconditioner,
            stepping: SteppingMode::Coupled,
        },
        None,
        None,
    ))
    .expect("solver init");

    let stride = solver.model().state_layout.stride() as usize;
    solver
        .write_state_f32(&vec![0.0f32; mesh.num_cells() * stride])
        .expect("clear state");

    solver.set_collect_convergence_stats(d.outer_auto_converge);
    solver.set_dt(d.timestep as f32);
    solver.set_dtau(0.0).ok();
    solver.set_density(fluid.density as f32).unwrap();
    solver
        .set_viscosity(fluid.viscosity as f32)
        .unwrap();
    solver.set_alpha_u(d.alpha_u as f32).unwrap();
    solver.set_alpha_p(d.alpha_p as f32).unwrap();
    solver.set_outer_iters(d.outer_iters as usize).unwrap();
    solver.set_inlet_velocity(d.inlet_velocity).unwrap();
    solver.set_u(&vec![(0.0, 0.0); mesh.num_cells()]);
    solver.set_p(&vec![0.0; mesh.num_cells()]);
    solver.incompressible_set_should_stop(false);
    solver.initialize_history();
    solver
}

fn build_compressible(d: &ModelGuiDefaults, fluid: &Fluid, mesh: &Mesh) -> UnifiedSolver {
    let eos = fluid.eos;
    let mut solver = pollster::block_on(UnifiedSolver::new(
        mesh,
        compressible_model_with_eos(eos).expect("compressible model"),
        SolverConfig {
            advection_scheme: d.advection_scheme,
            time_scheme: d.time_scheme,
            preconditioner: d.preconditioner,
            stepping: SteppingMode::Implicit {
                outer_iters: d.outer_iters as usize,
            },
        },
        None,
        None,
    ))
    .expect("solver init");

    let stride = solver.model().state_layout.stride() as usize;
    solver
        .write_state_f32(&vec![0.0f32; mesh.num_cells() * stride])
        .expect("clear state");

    let rho0 = fluid.density;
    let p0 = eos.pressure_for_density(rho0);

    solver.set_collect_convergence_stats(d.outer_auto_converge);
    solver.set_dt(d.timestep as f32);
    // Pseudo-transient continuation when the model default enables it (the
    // compressible low-Mach stabilizer); `0.0` (time-accurate) otherwise.
    solver.set_dtau(if d.dual_time { d.dtau as f32 } else { 0.0 }).ok();
    solver.set_eos(&eos).unwrap();
    solver
        .set_viscosity(fluid.viscosity as f32)
        .unwrap();
    solver.set_density(rho0 as f32).unwrap();
    solver.set_outer_iters(d.outer_iters as usize).unwrap();
    solver.set_precond_model(d.low_mach_model).unwrap();
    solver.set_precond_theta_floor(d.low_mach_theta_floor).unwrap();
    solver
        .set_precond_pressure_coupling_alpha(d.low_mach_pressure_coupling_alpha)
        .unwrap();
    solver
        .set_compressible_inlet_isothermal_x(rho0 as f32, d.inlet_velocity, &eos)
        .unwrap();
    // Initialize at the uniform freestream matching the inlet (NOT rest) — the
    // app does the same. From rest the inlet-injected momentum has no convective
    // transport and seeds the low-Mach inlet instability.
    solver.set_uniform_state(rho0 as f32, [d.inlet_velocity, 0.0], p0 as f32);
    solver.initialize_history();
    solver
}

fn run_incompressible(d: &ModelGuiDefaults, fluid: &Fluid, mesh: &Mesh) -> DriveResult {
    let min_cell = actual_min_cell(mesh);
    let mut solver = build_incompressible(d, fluid, mesh);
    drive(&mut solver, d, fluid.density, &fluid.eos, false, false, min_cell, N_STEPS)
}

/// Compressible runs need a LONG horizon: the low-Mach inlet instability this
/// gate guards against is slow — it stays under the cap for ~1000 steps before
/// blowing past it (the old 150-step gate passed straight through the blow-up).
/// `COMPRESSIBLE_STEPS` is well past where the unfixed default diverges.
fn run_compressible(d: &ModelGuiDefaults, fluid: &Fluid, mesh: &Mesh) -> DriveResult {
    let min_cell = actual_min_cell(mesh);
    let mut solver = build_compressible(d, fluid, mesh);
    drive(&mut solver, d, fluid.density, &fluid.eos, true, true, min_cell, COMPRESSIBLE_STEPS)
}

// --------------------------------------------------------------------------
// Gate
// --------------------------------------------------------------------------

#[test]
fn gui_default_incompressible_backstep_bounded() {
    std::env::set_var("CFD2_QUIET", "1");
    let air = air();
    let d = gui_defaults_for("incompressible_momentum");
    let mesh = backstep_mesh();
    eprintln!("[incompressible/backstep] cells={} min_cell={:.4e}", mesh.num_cells(), actual_min_cell(&mesh));
    let res = run_incompressible(&d, &air, &mesh);
    print_trace("incompressible/backstep", &res);
    assert_bounded("incompressible/backstep", &res, -1e6, 1e6, 0.5, 2.0);
}

// The cut-cell channel-obstacle has slivers (cells down to ~2% nominal). The
// root-cause fix is the immersed no-slip wall BC on the cylinder (see
// `generate_cut_cell_mesh`): it produces the physical boundary layer AND
// stabilizes the tiny cut cells (the no-slip wall-shear damping grows as cells
// shrink), so this stays bounded with SRD OFF (the default). The earlier
// mesh-level merge (geometry-distorting) and viscosity floor (unphysical) were
// both rejected/reverted; SRD is retained only as an opt-in stabilizer.
#[test]
fn gui_default_incompressible_obstacle_bounded() {
    std::env::set_var("CFD2_QUIET", "1");
    let air = air();
    let d = gui_defaults_for("incompressible_momentum");
    let mesh = channel_obstacle_mesh();
    eprintln!("[incompressible/obstacle] cells={} min_cell={:.4e}", mesh.num_cells(), actual_min_cell(&mesh));
    let res = run_incompressible(&d, &air, &mesh);
    print_trace("incompressible/obstacle", &res);
    assert_bounded("incompressible/obstacle", &res, -1e6, 1e6, 0.5, 2.0);
}

#[test]
fn gui_default_compressible_backstep_bounded_and_smooth() {
    std::env::set_var("CFD2_QUIET", "1");
    let air = air();
    let d = gui_defaults_for("compressible");
    let mesh = backstep_mesh();
    eprintln!("[compressible/backstep] cells={} min_cell={:.4e}", mesh.num_cells(), actual_min_cell(&mesh));
    let res = run_compressible(&d, &air, &mesh);
    print_trace("compressible/backstep", &res);
    let p0 = air.eos.pressure_for_density(air.density);
    assert_bounded("compressible/backstep", &res, 0.5 * p0, 4.0 * p0, 0.1 * air.density, 10.0 * air.density);
}

/// The GPU State-Redistribution pass must match the CPU operator exactly (to
/// f32 precision) on a known velocity field: applying `S` on-device through the
/// solver reproduces `SrdCsr::apply_cpu` on the same input.
#[test]
fn srd_gpu_matches_cpu_reference() {
    std::env::set_var("CFD2_QUIET", "1");
    let air = air();
    let d = gui_defaults_for("incompressible_momentum");
    let mesh = channel_obstacle_mesh();
    let mut solver = build_incompressible(&d, &air, &mesh);
    assert!(
        solver.srd_active(),
        "obstacle mesh must have an SRD operator built"
    );

    // A known, spatially varying velocity field (rounded to f32 to match the
    // device's storage precision, so the only remaining difference is f32
    // arithmetic ordering inside the operator).
    let n = mesh.num_cells();
    let u0: Vec<(f64, f64)> = (0..n)
        .map(|c| {
            let x = (0.5 * mesh.cell_cx[c]).sin() as f32 as f64;
            let y = (0.3 * mesh.cell_cy[c]).cos() as f32 as f64;
            (x, y)
        })
        .collect();
    solver.set_u(&u0);

    // GPU apply (through the solver), then read back.
    solver.apply_srd_pass();
    let gpu = pollster::block_on(solver.get_u());

    // CPU reference with the same precomputed operator.
    let csr = cfd2::solver::gpu::srd::build_srd_operator(&mesh).expect("operator");
    let mut cpu = u0.clone();
    csr.apply_cpu(&mut cpu);

    let mut max_diff = 0.0_f64;
    for c in 0..n {
        max_diff = max_diff
            .max((gpu[c].0 - cpu[c].0).abs())
            .max((gpu[c].1 - cpu[c].1).abs());
    }
    eprintln!("[srd-xcheck] cells={n} max|gpu-cpu|={max_diff:.3e}");
    assert!(
        max_diff < 1e-5,
        "GPU SRD diverges from CPU reference: {max_diff:e}"
    );
}

// --------------------------------------------------------------------------
// Tuning aids (run with --ignored)
// --------------------------------------------------------------------------

/// Build State-Redistribution (Berger & Giuliani 2021) neighborhoods for a mesh:
/// every cell has a neighborhood N_i (itself); cells below `threshold` grow N_i by
/// adding the largest face-adjacent cells until the neighborhood volume reaches
/// `target`. Returns (neighborhoods, theta) where theta_j = # neighborhoods
/// containing j (overlap count, >=1).
fn srd_neighborhoods(mesh: &Mesh, threshold: f64, target: f64) -> (Vec<Vec<usize>>, Vec<usize>) {
    use std::collections::HashSet;
    let n = mesh.num_cells();
    let other = |fi: usize, c: usize| -> Option<usize> {
        if mesh.face_owner[fi] == c { mesh.face_neighbor[fi] } else { Some(mesh.face_owner[fi]) }
    };
    let mut neigh: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    for i in 0..n {
        if mesh.cell_vol[i] >= threshold {
            continue;
        }
        let mut members = vec![i];
        let mut set: HashSet<usize> = HashSet::from([i]);
        let mut vol = mesh.cell_vol[i];
        while vol < target {
            let mut best: Option<(f64, usize)> = None;
            for &m in &members {
                for &fi in &mesh.cell_faces[mesh.cell_face_offsets[m]..mesh.cell_face_offsets[m + 1]] {
                    if let Some(o) = other(fi, m) {
                        if !set.contains(&o) && best.map_or(true, |(bv, _)| mesh.cell_vol[o] > bv) {
                            best = Some((mesh.cell_vol[o], o));
                        }
                    }
                }
            }
            match best {
                Some((v, nb)) => { members.push(nb); set.insert(nb); vol += v; }
                None => break,
            }
        }
        neigh[i] = members;
    }
    let mut theta = vec![0usize; n];
    for ni in &neigh {
        for &j in ni {
            theta[j] += 1;
        }
    }
    (neigh, theta)
}

/// Apply one SRD pass to a vector field `u` (conservative partition-of-unity
/// averaging). Modifies `u` in place.
fn srd_apply(u: &mut [(f64, f64)], mesh: &Mesh, neigh: &[Vec<usize>], theta: &[usize], inv: &[Vec<usize>]) {
    let n = mesh.num_cells();
    // gather: Q_hat_i = sum_{j in N_i} (V_j/theta_j) u_j / M_i
    let mut qhat = vec![(0.0f64, 0.0f64); n];
    for (i, ni) in neigh.iter().enumerate() {
        let (mut nx, mut ny, mut m) = (0.0, 0.0, 0.0);
        for &j in ni {
            let w = mesh.cell_vol[j] / theta[j] as f64;
            nx += w * u[j].0;
            ny += w * u[j].1;
            m += w;
        }
        qhat[i] = (nx / m, ny / m);
    }
    // scatter: u_new_j = (1/theta_j) sum_{i : j in N_i} Q_hat_i
    for j in 0..n {
        let (mut sx, mut sy) = (0.0, 0.0);
        for &i in &inv[j] {
            sx += qhat[i].0;
            sy += qhat[i].1;
        }
        u[j] = (sx / theta[j] as f64, sy / theta[j] as f64);
    }
}

/// CPU prototype: does post-step State Redistribution stabilize the true-geometry
/// obstacle? Validates the method before the GPU implementation.
#[test]
#[ignore]
fn prototype_srd_stabilizes_obstacle() {
    std::env::set_var("CFD2_QUIET", "1");
    let air = air();
    let d = gui_defaults_for("incompressible_momentum");
    let mesh = channel_obstacle_mesh();
    let nominal = 0.025 * 0.025;
    let (neigh, theta) = srd_neighborhoods(&mesh, 0.5 * nominal, nominal);
    let n = mesh.num_cells();
    let mut inv: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, ni) in neigh.iter().enumerate() {
        for &j in ni {
            inv[j].push(i);
        }
    }
    let stabilized = neigh.iter().filter(|ni| ni.len() > 1).count();
    eprintln!("[srd] {stabilized} small cells get neighborhoods; max |N_i|={}",
        neigh.iter().map(|ni| ni.len()).max().unwrap());

    let mut solver = build_incompressible(&d, &air, &mesh);
    let min_cell = actual_min_cell(&mesh);
    let mut prev_max = 0.0;
    let mut max_seen = 0.0f64;
    for step in 0..200 {
        let cur = solver.dt() as f64;
        let next = adaptive_next_dt(&d, prev_max, air.density, &air.eos, false, min_cell, cur);
        solver.set_dt(next as f32);
        solver.step_with_stats().expect("step");
        let mut u = pollster::block_on(solver.get_u());
        srd_apply(&mut u, &mesh, &neigh, &theta, &inv);
        solver.set_u(&u);
        let mv = u.iter().map(|(x, y)| (x * x + y * y).sqrt()).fold(0.0_f64, f64::max);
        prev_max = mv;
        max_seen = max_seen.max(mv);
        if step % 20 == 0 || mv > 5.0 {
            eprintln!("[srd] step {step} max|u|={mv:.4e}");
        }
        assert!(mv.is_finite() && mv < 5.0, "SRD failed: max|u|={mv:.3e} at step {step}");
    }
    eprintln!("[srd] SUCCESS: bounded over 200 steps, max|u|={max_seen:.4e}");
}

/// Diagnose WHERE/WHY the true-geometry (un-merged) obstacle blows up: report the
/// worst-|u| cell's volume, position relative to the cylinder, and face count each
/// step, to identify the small-cell failure mode for a proper solver fix.
#[test]
#[ignore]
fn diagnose_obstacle_blowup() {
    std::env::set_var("CFD2_QUIET", "1");
    let air = air();
    let d = gui_defaults_for("incompressible_momentum");
    let mesh = channel_obstacle_mesh();
    let nominal = 0.025 * 0.025;
    let min_vol = mesh.cell_vol.iter().cloned().fold(f64::INFINITY, f64::min);
    let n_sliver = mesh.cell_vol.iter().filter(|&&v| v < 0.5 * nominal).count();
    eprintln!(
        "[diag] cells={} nominal_vol={:.3e} min_vol={:.3e} ({:.1}% nom) slivers<50%={}",
        mesh.num_cells(), nominal, min_vol, 100.0 * min_vol / nominal, n_sliver
    );
    let cyl = (1.0_f64, 0.51_f64);
    let mut solver = build_incompressible(&d, &air, &mesh);
    let min_cell = actual_min_cell(&mesh);
    let mut prev_max = 0.0;
    for step in 0..50 {
        let cur = solver.dt() as f64;
        let next = adaptive_next_dt(&d, prev_max, air.density, &air.eos, false, min_cell, cur);
        solver.set_dt(next as f32);
        solver.step_with_stats().expect("step");
        let u = pollster::block_on(solver.get_u());
        let (mut mv, mut am) = (0.0_f64, 0usize);
        for (c, (x, y)) in u.iter().enumerate() {
            let v = (x * x + y * y).sqrt();
            if v.is_finite() && v > mv { mv = v; am = c; }
        }
        prev_max = mv;
        if step % 3 == 0 || mv > 5.0 {
            let vol = mesh.cell_vol[am];
            let (cx, cy) = (mesh.cell_cx[am], mesh.cell_cy[am]);
            let dcyl = ((cx - cyl.0).powi(2) + (cy - cyl.1).powi(2)).sqrt() - 0.1;
            let nf = mesh.cell_face_offsets[am + 1] - mesh.cell_face_offsets[am];
            eprintln!(
                "[diag] step {:>3} dt={:.2e} max|u|={:.3e} @cell {} vol={:.2e}({:.0}%nom) pos=({:.3},{:.3}) gap_to_cyl={:.3} faces={}",
                step, next, mv, am, vol, 100.0 * vol / nominal, cx, cy, dcyl, nf
            );
        }
        if mv > 100.0 { eprintln!("[diag] DIVERGED at step {step}"); break; }
    }
}

/// Diagnose the obstacle boundary layer: run to a developed state and report the
/// velocity profile binned by distance from the cylinder surface, plus whether
/// the run stayed bounded — separately for SRD on and off. A physical no-slip
/// wall shows |u| rising from ~0 at the surface to the free-stream value.
#[test]
#[ignore]
fn diagnose_obstacle_boundary_layer() {
    std::env::set_var("CFD2_QUIET", "1");
    let air = air();
    let d = gui_defaults_for("incompressible_momentum");
    let mesh = channel_obstacle_mesh();
    let cyl = (1.0_f64, 0.51_f64);
    let radius = 0.1_f64;
    let h = 0.025_f64;
    let n_wall_faces = mesh
        .face_boundary
        .iter()
        .zip(mesh.face_neighbor.iter())
        .filter(|(b, nb)| nb.is_none() && matches!(b, Some(cfd2::solver::mesh::BoundaryType::Wall)))
        .count();
    eprintln!("[bl] cells={} wall_faces(incl. immersed)={n_wall_faces}", mesh.num_cells());

    for srd_on in [true, false] {
        let mut solver = build_incompressible(&d, &air, &mesh);
        solver.set_srd_enabled(srd_on);
        let min_cell = actual_min_cell(&mesh);
        let mut prev_max = 0.0;
        let mut max_seen = 0.0f64;
        let mut diverged = false;
        for _ in 0..120 {
            let cur = solver.dt() as f64;
            let next = adaptive_next_dt(&d, prev_max, air.density, &air.eos, false, min_cell, cur);
            solver.set_dt(next as f32);
            solver.step_with_stats().expect("step");
            let u = pollster::block_on(solver.get_u());
            let mv = u.iter().map(|(x, y)| (x * x + y * y).sqrt()).fold(0.0_f64, f64::max);
            prev_max = mv;
            max_seen = max_seen.max(mv);
            if !mv.is_finite() || mv > 50.0 {
                diverged = true;
                break;
            }
        }
        // Radial bins (in units of h) from the cylinder surface.
        let u = pollster::block_on(solver.get_u());
        let edges = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0];
        let mut sum = [0.0f64; 5];
        let mut cnt = [0usize; 5];
        for (c, (x, y)) in u.iter().enumerate() {
            let dist = ((mesh.cell_cx[c] - cyl.0).powi(2) + (mesh.cell_cy[c] - cyl.1).powi(2)).sqrt()
                - radius;
            let band = dist / h;
            if !(0.0..edges[5]).contains(&band) {
                continue;
            }
            let speed = (x * x + y * y).sqrt();
            for b in 0..5 {
                if band >= edges[b] && band < edges[b + 1] {
                    sum[b] += speed;
                    cnt[b] += 1;
                    break;
                }
            }
        }
        eprintln!(
            "[bl] srd={srd_on} diverged={diverged} max|u|={max_seen:.3e} radial mean|u| by gap/h:"
        );
        for b in 0..5 {
            let mean = if cnt[b] > 0 { sum[b] / cnt[b] as f64 } else { f64::NAN };
            eprintln!("[bl]   gap [{:.0},{:.0})h: n={:>3} mean|u|={:.4e}", edges[b], edges[b + 1], cnt[b], mean);
        }
    }
}

/// Decisive check: does the runtime advection scheme actually change GPU output?
/// (Memory claims the GPU bakes the scheme; the generated assembly WGSL branches
/// on `constants.scheme`, so this verifies the value is plumbed end-to-end.)
#[test]
#[ignore]
fn scheme_actually_changes_gpu_output() {
    std::env::set_var("CFD2_QUIET", "1");
    let air = air();
    let mesh = backstep_mesh();
    let min_cell = actual_min_cell(&mesh);
    let base = gui_defaults_for("incompressible_momentum");

    let run = |scheme: cfd2::solver::scheme::Scheme| -> f64 {
        let mut d = base;
        d.advection_scheme = scheme;
        d.inlet_velocity = 1.0; // exercise advection strongly
        let mut solver = build_incompressible(&d, &air, &mesh);
        for _ in 0..8 {
            solver.set_dt(1e-3);
            solver.step_with_stats().expect("step");
        }
        let u = pollster::block_on(solver.get_u());
        u.iter()
            .map(|(x, y)| (x * x + y * y).sqrt())
            .fold(0.0_f64, f64::max)
    };

    let upwind = run(cfd2::solver::scheme::Scheme::Upwind);
    let sou = run(cfd2::solver::scheme::Scheme::SecondOrderUpwind);
    let vanleer = run(cfd2::solver::scheme::Scheme::SecondOrderUpwindVanLeer);
    eprintln!(
        "[scheme check / config] incompressible max|u| Upwind={:.6e} SOU={:.6e} VanLeer={:.6e}",
        upwind, sou, vanleer
    );

    // Runtime path (what the GUI radio uses): build Upwind, then set_advection_scheme.
    let run_runtime = |scheme: cfd2::solver::scheme::Scheme| -> f64 {
        let mut d = base;
        d.advection_scheme = cfd2::solver::scheme::Scheme::Upwind;
        d.inlet_velocity = 1.0;
        let mut solver = build_incompressible(&d, &air, &mesh);
        solver.set_advection_scheme(scheme);
        for _ in 0..8 {
            solver.set_dt(1e-3);
            solver.step_with_stats().expect("step");
        }
        let u = pollster::block_on(solver.get_u());
        u.iter().map(|(x, y)| (x * x + y * y).sqrt()).fold(0.0, f64::max)
    };
    let rt_upwind = run_runtime(cfd2::solver::scheme::Scheme::Upwind);
    let rt_vanleer = run_runtime(cfd2::solver::scheme::Scheme::SecondOrderUpwindVanLeer);
    eprintln!(
        "[scheme check / runtime set_advection_scheme] Upwind={:.6e} VanLeer={:.6e}",
        rt_upwind, rt_vanleer
    );

    // Compressible (the memory's specific case): config scheme.
    let run_comp = |scheme: cfd2::solver::scheme::Scheme| -> f64 {
        let mut d = gui_defaults_for("compressible");
        d.advection_scheme = scheme;
        d.inlet_velocity = 0.1;
        let mut solver = build_compressible(&d, &air, &mesh);
        for _ in 0..20 {
            solver.set_dt(1e-5);
            solver.step_with_stats().expect("step");
        }
        let u = pollster::block_on(solver.get_u());
        u.iter().map(|(x, y)| (x * x + y * y).sqrt()).fold(0.0, f64::max)
    };
    let c_upwind = run_comp(cfd2::solver::scheme::Scheme::Upwind);
    let c_vanleer = run_comp(cfd2::solver::scheme::Scheme::SecondOrderUpwindVanLeer);
    eprintln!(
        "[scheme check / compressible config] Upwind={:.6e} VanLeer={:.6e}",
        c_upwind, c_vanleer
    );

    assert!(
        (upwind - vanleer).abs() > 1e-9,
        "incompressible config scheme ignored"
    );
    assert!(
        (rt_upwind - rt_vanleer).abs() > 1e-9,
        "RUNTIME set_advection_scheme ignored (GUI radio is a no-op): Upwind={rt_upwind:.6e} VanLeer={rt_vanleer:.6e}"
    );
    assert!(
        (c_upwind - c_vanleer).abs() > 1e-9,
        "COMPRESSIBLE config scheme ignored: Upwind={c_upwind:.6e} VanLeer={c_vanleer:.6e}"
    );
    let _ = min_cell;
}

#[test]
#[ignore]
fn sweep_compressible_backstep() {
    std::env::set_var("CFD2_QUIET", "1");
    let air = air();
    let base = gui_defaults_for("compressible");
    let mesh = backstep_mesh();
    eprintln!("=== compressible backstep sweep (cells={}) ===", mesh.num_cells());

    // (label, inlet, theta, adaptive, target_cfl, timestep, pca) — real Air viscosity.
    let grid: &[(&str, f32, f32, bool, f64, f64, f32)] = &[
        ("U=1.0   theta=1    adaptive cfl=0.3 pca=0.01", 1.0, 1.0, true, 0.3, 1e-5, 0.01),
        ("U=0.002 theta=1    adaptive cfl=0.3 pca=0.01", 0.002, 1.0, true, 0.3, 1e-5, 0.01),
        ("U=0.002 theta=1    adaptive cfl=0.3 pca=0.1 ", 0.002, 1.0, true, 0.3, 1e-5, 0.1),
        ("U=0.002 theta=1e-8 fixed dt=1e-5 pca=0.01", 0.002, 1e-8, false, 0.3, 1e-5, 0.01),
        ("U=0.002 theta=1e-8 fixed dt=1e-5 pca=0.1 ", 0.002, 1e-8, false, 0.3, 1e-5, 0.1),
        ("U=0.01  theta=1    adaptive cfl=0.3 pca=0.01", 0.01, 1.0, true, 0.3, 1e-5, 0.01),
        ("U=0.002 theta=1    adaptive cfl=0.1 pca=0.01", 0.002, 1.0, true, 0.1, 1e-5, 0.01),
    ];

    for (label, inlet, theta, adaptive, cfl, ts, pca) in grid {
        let mut d = base;
        d.inlet_velocity = *inlet;
        d.low_mach_theta_floor = *theta;
        d.adaptive_dt = *adaptive;
        d.target_cfl = *cfl;
        d.timestep = *ts;
        d.low_mach_pressure_coupling_alpha = *pca;
        let res = run_compressible(&d, &air, &mesh);
        let last = res.samples.last();
        eprintln!(
            "  {:<46} -> diverged={} step={:?} final(max|u|={:.3e}, dt={:.2e})",
            label, res.diverged, res.diverge_step,
            last.map(|s| s.max_vel).unwrap_or(f64::NAN),
            last.map(|s| s.dt).unwrap_or(f64::NAN),
        );
    }
}

/// Long-run probe of the compressible GUI default on EITHER default geometry, on
/// whatever backend `CFD2_BACKEND` selects (gpu default; `cpu` routes to the CPU
/// solver). Prints the trajectory every 100 steps and the divergence step (if
/// any). This is the instrument that surfaced the low-Mach inlet blow-up the
/// 150-step gate missed (see [[cfd2-compressible-gui-divergence]]); with the
/// shipped defaults (uniform IC + `dtau`) it holds at the freestream.
///
/// Env knobs (each OVERRIDES the model default for one-recompile sweeps):
/// `CFD2_PROBE_STEPS` (default 3000), `CFD2_PROBE_GEO` (`backstep`|`obstacle`),
/// `CFD2_PROBE_SCHEME` (`upwind`|`sou`|else VanLeer), `CFD2_PROBE_VISC_MULT`,
/// `CFD2_PROBE_THETA`, `CFD2_PROBE_PCA`, `CFD2_PROBE_U` (inlet speed),
/// `CFD2_PROBE_OUTER`, `CFD2_PROBE_DT`, `CFD2_PROBE_DTAU` (pseudo-time; absent ⇒
/// model default), `CFD2_PROBE_INITU` (uniform-freestream IC u_x; absent ⇒ the
/// `build_compressible` default IC).
#[test]
#[ignore]
fn diag_compressible_default_long() {
    std::env::set_var("CFD2_QUIET", "1");
    let mut air = air();
    let mut d = gui_defaults_for("compressible");
    // Stabilizer-sweep knobs (one recompile, many experiments).
    if let Ok(s) = std::env::var("CFD2_PROBE_SCHEME") {
        d.advection_scheme = match s.as_str() {
            "upwind" => cfd2::solver::scheme::Scheme::Upwind,
            "sou" => cfd2::solver::scheme::Scheme::SecondOrderUpwind,
            _ => cfd2::solver::scheme::Scheme::SecondOrderUpwindVanLeer,
        };
    }
    let env_f64 = |k: &str| std::env::var(k).ok().and_then(|s| s.parse::<f64>().ok());
    let env_f32 = |k: &str| std::env::var(k).ok().and_then(|s| s.parse::<f32>().ok());
    if let Some(m) = env_f64("CFD2_PROBE_VISC_MULT") { air.viscosity *= m; }
    if let Some(t) = env_f32("CFD2_PROBE_THETA") { d.low_mach_theta_floor = t; }
    if let Some(p) = env_f32("CFD2_PROBE_PCA") { d.low_mach_pressure_coupling_alpha = p; }
    if let Some(u) = env_f32("CFD2_PROBE_U") { d.inlet_velocity = u; }
    if let Some(o) = std::env::var("CFD2_PROBE_OUTER").ok().and_then(|s| s.parse().ok()) { d.outer_iters = o; }
    if let Some(dtv) = env_f64("CFD2_PROBE_DT") { d.timestep = dtv; }
    let steps: usize = std::env::var("CFD2_PROBE_STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(3000);
    let geo = std::env::var("CFD2_PROBE_GEO").unwrap_or_else(|_| "backstep".into());
    // `CFD2_PROBE_DTAU` OVERRIDES the model default; absent, the model default
    // (compressible: dual_time + 1e-3) is what `build_compressible` applies.
    let dtau_override = env_f32("CFD2_PROBE_DTAU");
    let eff_dtau = dtau_override.unwrap_or(if d.dual_time { d.dtau as f32 } else { 0.0 });
    let mesh = if geo == "obstacle" { channel_obstacle_mesh() } else { backstep_mesh() };
    let min_cell = actual_min_cell(&mesh);
    eprintln!(
        "[compressible/{geo} long] cells={} min_cell={:.4e} steps={steps} dt={:.1e} adaptive={} scheme={:?} visc={:.3e} theta={:.1e} pca={:.3} dtau={:.1e}",
        mesh.num_cells(), min_cell, d.timestep, d.adaptive_dt, d.advection_scheme, air.viscosity, d.low_mach_theta_floor, d.low_mach_pressure_coupling_alpha, eff_dtau
    );
    let mut solver = build_compressible(&d, &air, &mesh);
    if let Some(dt) = dtau_override { solver.set_dtau(dt).ok(); }
    // Initialize at a UNIFORM FLOW matching the inlet (physical freestream IC, as the OpenFOAM
    // reference does) instead of rest — from rest the inlet-injected momentum has no convective
    // transport and piles up at the inlet. CFD2_PROBE_INITU sets the freestream u_x.
    if let Some(uinit) = std::env::var("CFD2_PROBE_INITU").ok().and_then(|s| s.parse::<f32>().ok()) {
        let rho0 = air.density as f32;
        let p0i = air.eos.pressure_for_density(air.density) as f32;
        solver.set_uniform_state(rho0, [uinit, 0.0], p0i);
        solver.initialize_history();
    }

    let p0 = air.eos.pressure_for_density(air.density);
    let mut prev_max_vel = 0.0_f64;
    let mut diverged_at: Option<usize> = None;
    for step in 0..steps {
        if d.adaptive_dt {
            let current_dt = solver.dt() as f64;
            let next_dt = adaptive_next_dt(&d, prev_max_vel, air.density, &air.eos, true, min_cell, current_dt);
            solver.set_dt(next_dt as f32);
        } else {
            solver.set_dt(d.timestep as f32);
        }
        if solver.step_with_stats().is_err() {
            eprintln!("  step {step}: step_with_stats ERROR");
            diverged_at = Some(step);
            break;
        }
        let report = step % 100 == 0 || step == steps - 1;
        if !report {
            continue;
        }
        let u = pollster::block_on(solver.get_u());
        let p = pollster::block_on(solver.get_p());
        let rho = pollster::block_on(solver.get_rho());
        let mut max_vel = 0.0_f64;
        let mut nonfinite = 0usize;
        for (vx, vy) in &u {
            if !(vx.is_finite() && vy.is_finite()) { nonfinite += 1; continue; }
            max_vel = max_vel.max((vx * vx + vy * vy).sqrt());
        }
        let p_min = p.iter().cloned().fold(f64::INFINITY, f64::min);
        let p_max = p.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let rho_min = rho.iter().cloned().fold(f64::INFINITY, f64::min);
        let rho_max = rho.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        prev_max_vel = if max_vel.is_finite() { max_vel } else { 0.0 };
        eprintln!(
            "  step {:>5} dt={:.2e} max|u|={:.4e} nf_u={} p=[{:.4e},{:.4e}] rho=[{:.4e},{:.4e}]",
            step, solver.dt(), max_vel, nonfinite, p_min, p_max, rho_min, rho_max
        );
        let bad = nonfinite > 0 || !max_vel.is_finite() || max_vel > 50.0
            || !p_min.is_finite() || !p_max.is_finite()
            || p_min < 0.1 * p0 || p_max > 10.0 * p0
            || rho_min < 0.05 * air.density || rho_max > 20.0 * air.density;
        if bad {
            eprintln!("  step {step}: DIVERGED / unphysical");
            diverged_at = Some(step);
            break;
        }
    }
    eprintln!("[compressible/{geo} long] diverged_at={diverged_at:?}");
}

/// Jet-like colormap for a normalized value in [0,1].
fn jet(t: f64) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    let r = (1.5 - (4.0 * t - 3.0).abs()).clamp(0.0, 1.0);
    let g = (1.5 - (4.0 * t - 2.0).abs()).clamp(0.0, 1.0);
    let b = (1.5 - (4.0 * t - 1.0).abs()).clamp(0.0, 1.0);
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}

/// Rasterize a per-cell scalar field over the mesh to a PNG by splatting each
/// cell centroid to a small pixel block. Auto-scales to [min,max] (printed) so the
/// spatial structure is visible regardless of magnitude.
fn save_field_png(path: &str, mesh: &Mesh, values: &[f64], px_per_unit: f64, label: &str) {
    let xmax = mesh.cell_cx.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ymax = mesh.cell_cy.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let w = (xmax * px_per_unit).ceil() as u32 + 8;
    let h = (ymax * px_per_unit).ceil() as u32 + 8;
    let finite: Vec<f64> = values.iter().cloned().filter(|v| v.is_finite()).collect();
    let vmin = finite.iter().cloned().fold(f64::INFINITY, f64::min);
    let vmax = finite.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let span = (vmax - vmin).max(1e-30);
    let mut img = image::RgbImage::from_pixel(w, h, image::Rgb([20, 20, 20]));
    let r = (0.6 * px_per_unit * 0.025).ceil().max(2.0) as i32;
    for c in 0..mesh.num_cells() {
        let px = (mesh.cell_cx[c] * px_per_unit) as i32 + 4;
        let py = h as i32 - 4 - (mesh.cell_cy[c] * px_per_unit) as i32;
        let color = if values[c].is_finite() {
            image::Rgb(jet((values[c] - vmin) / span))
        } else {
            image::Rgb([255, 0, 255]) // magenta = non-finite
        };
        for dy in -r..=r {
            for dx in -r..=r {
                let (x, y) = (px + dx, py + dy);
                if x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h {
                    img.put_pixel(x as u32, y as u32, color);
                }
            }
        }
    }
    img.save(path).expect("save png");
    eprintln!("[viz] {label}: {path}  range=[{vmin:.4e}, {vmax:.4e}]");
}

/// Render the compressible GUI-default flow fields at several checkpoints to PNGs
/// in /tmp/cfd_viz so the *nature* of the slow blow-up is visible (checkerboard vs
/// boundary-localized vs smooth large-scale). Honors `CFD2_BACKEND`. Env:
/// `CFD2_VIZ_GEO` (backstep|obstacle), `CFD2_VIZ_STEPS` (comma list of checkpoints).
/// Print the inlet-adjacent cell columns (smallest few x) over time, sorted by y,
/// to reveal whether the inlet blow-up is a checkerboard (sign alternation between
/// vertically-adjacent cells) or a smooth growing boundary layer, and which
/// variable leads. Honors `CFD2_BACKEND`.
#[test]
#[ignore]
fn diag_compressible_inlet_profile() {
    std::env::set_var("CFD2_QUIET", "1");
    let air = air();
    let d = gui_defaults_for("compressible");
    let mesh = backstep_mesh();
    // Inlet-adjacent cells: the two smallest distinct x-bands.
    let mut xs: Vec<f64> = mesh.cell_cx.clone();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let x_thresh = xs[0] + 0.03; // ~first column (cell ~0.025)
    let mut inlet_cells: Vec<usize> = (0..mesh.num_cells()).filter(|&c| mesh.cell_cx[c] < x_thresh).collect();
    inlet_cells.sort_by(|&a, &b| mesh.cell_cy[a].partial_cmp(&mesh.cell_cy[b]).unwrap());
    eprintln!("[inlet] {} inlet-column cells (x<{:.3})", inlet_cells.len(), x_thresh);

    let mut solver = build_compressible(&d, &air, &mesh);
    let p0 = air.eos.pressure_for_density(air.density);
    let checks = [0usize, 300, 1000, 2000];
    let mut done = 0usize;
    for step in 0..=*checks.last().unwrap() {
        if step == checks[done] {
            let u = pollster::block_on(solver.get_u());
            let p = pollster::block_on(solver.get_p());
            let rho = pollster::block_on(solver.get_rho());
            eprintln!("--- step {step} ---  (y, u_x, u_y, p-p0, rho)");
            for &c in &inlet_cells {
                eprintln!(
                    "  y={:.3}  u=({:+.4e},{:+.4e})  dp={:+.4e}  rho={:.5}",
                    mesh.cell_cy[c], u[c].0, u[c].1, p[c] - p0, rho[c]
                );
            }
            done += 1;
            if done >= checks.len() { break; }
        }
        solver.set_dt(d.timestep as f32);
        if solver.step_with_stats().is_err() { break; }
    }
}

#[test]
#[ignore]
fn viz_compressible_default_fields() {
    std::env::set_var("CFD2_QUIET", "1");
    let air = air();
    let mut d = gui_defaults_for("compressible");
    let geo = std::env::var("CFD2_VIZ_GEO").unwrap_or_else(|_| "backstep".into());
    let checkpoints: Vec<usize> = std::env::var("CFD2_VIZ_STEPS")
        .unwrap_or_else(|_| "200,800,1500,2500".into())
        .split(',').filter_map(|s| s.trim().parse().ok()).collect();
    let dtau: f32 = std::env::var("CFD2_VIZ_DTAU").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    if let Some(u) = std::env::var("CFD2_PROBE_U").ok().and_then(|s| s.parse::<f32>().ok()) {
        d.inlet_velocity = u;
    }
    let mesh = if geo == "obstacle" { channel_obstacle_mesh() } else { backstep_mesh() };
    std::fs::create_dir_all("/tmp/cfd_viz").ok();
    let mut solver = build_compressible(&d, &air, &mesh);
    if dtau > 0.0 { solver.set_dtau(dtau).ok(); }
    if let Some(uinit) = std::env::var("CFD2_PROBE_INITU").ok().and_then(|s| s.parse::<f32>().ok()) {
        let rho0 = air.density as f32;
        let p0i = air.eos.pressure_for_density(air.density) as f32;
        solver.set_uniform_state(rho0, [uinit, 0.0], p0i);
        solver.initialize_history();
    }
    let tag = if dtau > 0.0 { format!("{geo}_dtau") } else { geo.clone() };
    let last = *checkpoints.last().unwrap();
    let mut next = 0usize;
    for step in 0..=last {
        if step == checkpoints[next] {
            let u = pollster::block_on(solver.get_u());
            let p = pollster::block_on(solver.get_p());
            let umag: Vec<f64> = u.iter().map(|(x, y)| (x * x + y * y).sqrt()).collect();
            let max_u = umag.iter().cloned().fold(0.0, f64::max);
            eprintln!("[viz] {tag} step {step}: max|u|={max_u:.4e}");
            save_field_png(&format!("/tmp/cfd_viz/{tag}_umag_{step:05}.png"), &mesh, &umag, 220.0, &format!("|u| @ {step}"));
            save_field_png(&format!("/tmp/cfd_viz/{tag}_p_{step:05}.png"), &mesh, &p, 220.0, &format!("p @ {step}"));
            next += 1;
            if next >= checkpoints.len() { break; }
        }
        solver.set_dt(d.timestep as f32);
        if solver.step_with_stats().is_err() {
            eprintln!("[viz] step {step}: ERROR");
            break;
        }
    }
}

#[test]
#[ignore]
fn sweep_incompressible_obstacle() {
    std::env::set_var("CFD2_QUIET", "1");
    let air = air();
    let base = gui_defaults_for("incompressible_momentum");
    let mesh = channel_obstacle_mesh();
    eprintln!("=== incompressible obstacle sweep (cells={}, min_cell={:.4e}) ===", mesh.num_cells(), actual_min_cell(&mesh));

    // Real Air viscosity; low speed + vary dt (cfl) and outer iters. Is the
    // obstacle divergence the large adaptive dt (1/U) or the cut-cell slivers?
    let grid: &[(&str, f32, f64, u32)] = &[
        ("U=0.002 cfl=0.9  outer=8", 0.002, 0.9, 8),
        ("U=0.002 cfl=0.3  outer=8", 0.002, 0.3, 8),
        ("U=0.002 cfl=0.1  outer=8", 0.002, 0.1, 8),
        ("U=0.002 cfl=0.05 outer=8", 0.002, 0.05, 8),
        ("U=0.002 cfl=0.1  outer=20", 0.002, 0.1, 20),
        ("U=0.002 cfl=0.02 outer=20", 0.002, 0.02, 20),
        ("U=0.01  cfl=0.1  outer=8", 0.01, 0.1, 8),
    ];

    for (label, u, cfl, outer) in grid {
        let mut d = base;
        d.inlet_velocity = *u;
        d.target_cfl = *cfl;
        d.outer_iters = *outer;
        let res = run_incompressible(&d, &air, &mesh);
        let last = res.samples.last();
        eprintln!(
            "  {:<26} -> diverged={} step={:?} final(max|u|={:.3e}, dt={:.2e})",
            label, res.diverged, res.diverge_step,
            last.map(|s| s.max_vel).unwrap_or(f64::NAN),
            last.map(|s| s.dt).unwrap_or(f64::NAN),
        );
    }
}
