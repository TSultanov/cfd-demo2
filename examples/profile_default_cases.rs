//! Faithful GPU profiling harness for the **UI default cases**.
//!
//! The stock `profile_solver_performance` example drives a Jacobi / single-outer
//! configuration that does not match anything the GUI ships. This harness instead
//! reconstructs the *actual* default `RuntimeParams` (the incompressible coupled
//! SIMPLE solver: Van Leer + BDF2 + model-owned Schur + 8 outer iterations +
//! acoustic-aware adaptive dt, at the GUI's 0.025 cut-cell resolution) and drives
//! it through the shared `SolverDriver` — the same construction + step path the
//! desktop app runs. It reports:
//!
//!   * true per-step wall time (unprofiled), plus average outer/linear iterations,
//!   * the per-graph GPU-time breakdown from the profiler.
//!
//! Run (backstep default + channel-obstacle headline demo):
//!   cargo run --release --example profile_default_cases --features "meshgen profiling"
//!
//! Optional: pass a case name (`backstep` | `obstacle`) and a step count.
use std::io::Write;
use std::time::Instant;

use cfd2::sim::{DriverBuild, RuntimeParams, SolverDriver};
use cfd2::solver::mesh::{
    generate_cut_cell_mesh, BackwardsStep, ChannelWithObstacle, Mesh,
};
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::model::incompressible_momentum_model;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{GpuLowMachPrecondModel, PreconditionerType, TimeScheme};
use nalgebra::{Point2, Vector2};

/// Air, matching `Fluid::presets()[1]`.
fn air_eos() -> EosSpec {
    EosSpec::IdealGas {
        gamma: 1.4,
        gas_constant: 287.0,
        temperature: 300.0,
    }
}

/// The shipped incompressible GUI default (`ui::model_defaults::INCOMPRESSIBLE`),
/// reconstructed as a `RuntimeParams` so this harness needs no `ui` feature.
fn incompressible_default_params(outer_iters: u32) -> RuntimeParams {
    RuntimeParams {
        filter_sigma: 0.0,
        adaptive_dt: true,
        target_cfl: 0.9,
        requested_dt: 0.02,
        dtau: 0.0,
        log_convergence: false,
        log_every_steps: 50,
        advection_scheme: Scheme::SecondOrderUpwindVanLeer,
        time_scheme: TimeScheme::BDF2,
        // Model forces Schur internally; this is the GUI's default selection value.
        preconditioner: PreconditionerType::Jacobi,
        outer_iters,
        outer_auto_converge: true,
        low_mach_model: GpuLowMachPrecondModel::Off,
        low_mach_theta_floor: 1e-6,
        low_mach_pressure_coupling_alpha: 1.0,
        alpha_u: 0.7,
        alpha_p: 0.3,
        inlet_velocity: 0.011,
        density: 1.225,
        viscosity: 1.81e-5,
        eos: air_eos(),
        compressibility_psi: 0.0,
        outlet_back_pressure: 0.0,
        allmach_precond_uref_min: 0.2,
        pressure_inlet: false,
        inlet_pressure: 0.0,
    }
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

fn obstacle_mesh() -> Mesh {
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

fn build_driver(mesh: &Mesh, params: &RuntimeParams) -> SolverDriver {
    let n = mesh.num_cells();
    let DriverBuild { mut driver, .. } = pollster::block_on(SolverDriver::build(
        mesh,
        incompressible_momentum_model().expect("incompressible model"),
        params,
        &vec![(0.0, 0.0); n],
        &vec![0.0; n],
        None,
        None,
    ))
    .expect("driver build");
    driver.apply_params(params);
    driver
}

fn profile_case(name: &str, mesh: Mesh, steps: usize, outer_iters: u32) {
    let params = incompressible_default_params(outer_iters);
    let cells = mesh.num_cells();
    let mut driver = build_driver(&mesh, &params);

    println!("\n================================================================");
    println!("  CASE: {name}  ({cells} cells, outer_iters={outer_iters})");
    println!("================================================================");

    // Warm up (compiles pipelines, settles adaptive dt).
    for i in 0..20 {
        driver.step(i % 5 == 0);
    }

    // Shedding signature: max|u| at each readback (oscillates at the shedding freq).
    let mut max_vels: Vec<f64> = Vec::new();

    // ---- Phase A: true per-step timing (no profiling overhead) --------------
    let mut solver_ms = 0.0f64;
    let mut outer_sum = 0u64;
    let mut lin_iter_sum = 0u64;
    let mut lin_solves = 0u64;
    let mut last_readback = None;
    let wall = Instant::now();
    for i in 0..steps {
        // Force a readback on the final step so we can fingerprint the exact final state.
        let o = driver.step(i % 5 == 0 || i == steps - 1);
        solver_ms += o.step_time_ms as f64;
        outer_sum += o.outer_iters.unwrap_or(0) as u64;
        for s in &o.linear_stats {
            lin_iter_sum += s.iterations as u64;
            lin_solves += 1;
        }
        if let Some(rb) = &o.readback {
            max_vels.push(rb.stats.max_vel);
        }
        if o.readback.is_some() {
            last_readback = o.readback;
        }
    }
    let wall = wall.elapsed();

    // Shedding statistics over the SECOND HALF of the run (past the startup
    // transient): mean, std (oscillation amplitude), min/max, and a zero-crossing
    // count of (max_vel - mean) as a crude shedding-frequency proxy.
    if max_vels.len() >= 8 {
        let tail = &max_vels[max_vels.len() / 2..];
        let mean = tail.iter().sum::<f64>() / tail.len() as f64;
        let var = tail.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / tail.len() as f64;
        let std = var.sqrt();
        let mn = tail.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = tail.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let crossings = tail
            .windows(2)
            .filter(|w| (w[0] - mean).signum() != (w[1] - mean).signum())
            .count();
        println!(
            "     SHED max|u| (tail n={}): mean={:.5e} std={:.3e} ({:.2}%) min={:.5e} max={:.5e} crossings={}",
            tail.len(),
            mean,
            std,
            100.0 * std / mean.abs().max(1e-30),
            mn,
            mx,
            crossings
        );
    }
    let n = steps as f64;
    println!("\n  -- true timing ({steps} steps, unprofiled) --");
    println!("     wall/step        : {:.3} ms", wall.as_secs_f64() * 1e3 / n);
    println!("     solver/step      : {:.3} ms", solver_ms / n);
    println!("     outer iters/step : {:.2}", outer_sum as f64 / n);
    if lin_solves > 0 {
        println!(
            "     linear solves/step: {:.2}   iters/solve: {:.1}",
            lin_solves as f64 / n,
            lin_iter_sum as f64 / lin_solves as f64
        );
    }

    // Deterministic bit-level fingerprint of the final state — for byte-identical
    // regression checks across solver refactors (the GPU solve is deterministic, so
    // identical ops => identical bits). FNV-1a over the raw f64 bits of u and p.
    if let Some(rb) = &last_readback {
        let mut h: u64 = 0xcbf29ce484222325;
        let mut mix = |bits: u64| {
            for chunk in [bits as u32 as u64, bits >> 32] {
                h ^= chunk;
                h = h.wrapping_mul(0x100000001b3);
            }
        };
        for (vx, vy) in &rb.u {
            mix(vx.to_bits());
            mix(vy.to_bits());
        }
        for &pv in &rb.p {
            mix(pv.to_bits());
        }
        println!(
            "     STATE_FINGERPRINT: {:016x}  (cells={}, |u|={}, |p|={})",
            h,
            cells,
            rb.u.len(),
            rb.p.len()
        );
    }

    // ---- Phase B: per-graph GPU breakdown (profiling on) --------------------
    driver
        .solver_mut()
        .enable_detailed_profiling(true)
        .expect("enable profiling");
    driver
        .solver()
        .start_profiling_session()
        .expect("start session");
    let prof_steps = steps.min(30);
    let prof_wall = Instant::now();
    for i in 0..prof_steps {
        driver.step(i % 5 == 0);
    }
    let prof_wall = prof_wall.elapsed();
    driver.solver().end_profiling_session().expect("end session");
    println!(
        "\n  -- profiling overhead: {:.3} ms/step (vs {:.3} true) --",
        prof_wall.as_secs_f64() * 1e3 / prof_steps as f64,
        wall.as_secs_f64() * 1e3 / n
    );
    driver.solver().print_profiling_report().expect("report");

    let _ = std::io::stdout().flush();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let case = args.get(1).map(|s| s.as_str()).unwrap_or("all");
    let steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(60);
    // Optional arg 3: outer_iters override (default 8, the shipped value) for
    // sweeping the outer-iteration count.
    let outer_iters: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);

    if case == "backstep" || case == "all" {
        profile_case("backstep (startup default)", backstep_mesh(), steps, outer_iters);
    }
    if case == "obstacle" || case == "all" {
        profile_case(
            "channel-obstacle (vortex street)",
            obstacle_mesh(),
            steps,
            outer_iters,
        );
    }
    let _ = std::io::stdout().flush();
}
