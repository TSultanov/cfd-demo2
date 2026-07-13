//! PROBE (temporary): GPU vs CPU backend per-step wall time on the SAME GUI
//! default cases (incompressible VanLeer+BDF2+Schur, outer 8 auto-converge,
//! adaptive dt) — mirrors examples/profile_default_cases.rs configs exactly.
//! Run: CFD2_CPU_THREADS=6 cargo test --features "cpu meshgen" --test zz_gpu_vs_cpu_probe -- --test-threads=1 --nocapture
#![cfg(all(feature = "cpu", feature = "meshgen"))]

use cfd2::sim::{DriverBuild, RuntimeParams, SolverDriver};
use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep, ChannelWithObstacle, Mesh};
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::model::incompressible_momentum_model;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{GpuLowMachPrecondModel, PreconditionerType, TimeScheme};
use nalgebra::{Point2, Vector2};
use std::time::Instant;

fn params() -> RuntimeParams {
    RuntimeParams {
        filter_sigma: 0.0,
        adaptive_dt: true,
        target_cfl: 0.9,
        requested_dt: 0.02,
        dtau: 0.0,
        log_convergence: false,
        log_every_steps: 100000,
        advection_scheme: Scheme::SecondOrderUpwindVanLeer,
        time_scheme: TimeScheme::BDF2,
        preconditioner: PreconditionerType::Jacobi,
        outer_iters: 8,
        outer_auto_converge: true,
        low_mach_model: GpuLowMachPrecondModel::Off,
        low_mach_theta_floor: 1e-6,
        low_mach_pressure_coupling_alpha: 1.0,
        alpha_u: 0.7,
        alpha_p: 0.3,
        inlet_velocity: 0.011,
        density: 1.225,
        viscosity: 1.81e-5,
        eos: EosSpec::IdealGas {
            gamma: 1.4,
            gas_constant: 287.0,
            temperature: 300.0,
        },
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
    obstacle_mesh_at(0.025)
}

fn obstacle_mesh_at(cell: f64) -> Mesh {
    let length = 3.0;
    let geo = ChannelWithObstacle {
        length,
        height: 1.0,
        obstacle_center: Point2::new(1.0, 0.51),
        obstacle_radius: 0.1,
    };
    let mut mesh = generate_cut_cell_mesh(&geo, cell, cell, 1.2, Vector2::new(length, 1.0));
    mesh.smooth(&geo, 0.3, 100);
    mesh
}

fn run(name: &str, mesh: &Mesh, gpu: bool, steps: usize) {
    let p = params();
    let n = mesh.num_cells();
    let u0 = vec![(0.0, 0.0); n];
    let p0 = vec![0.0; n];
    let mut driver = if gpu {
        let DriverBuild { mut driver, .. } = pollster::block_on(SolverDriver::build(
            mesh,
            incompressible_momentum_model().expect("model"),
            &p,
            &u0,
            &p0,
            None,
            None,
        ))
        .expect("gpu driver build");
        driver.apply_params(&p);
        driver
    } else {
        let DriverBuild { mut driver, .. } = pollster::block_on(SolverDriver::build_forced_cpu(
            mesh,
            incompressible_momentum_model().expect("model"),
            &p,
            &u0,
            &p0,
        ))
        .expect("cpu driver build");
        driver.apply_params(&p);
        driver
    };

    // Warm up (pipelines, adaptive dt) — same cadence as profile_default_cases.
    for i in 0..20 {
        driver.step(i % 5 == 0);
    }

    let mut outer_sum = 0u64;
    let mut lin_iter_sum = 0u64;
    let mut lin_solves = 0u64;
    let mut max_vel = 0.0f64;
    let wall = Instant::now();
    for i in 0..steps {
        let o = driver.step(i % 5 == 0 || i == steps - 1);
        outer_sum += o.outer_iters.unwrap_or(0) as u64;
        for s in &o.linear_stats {
            lin_iter_sum += s.iterations as u64;
            lin_solves += 1;
        }
        if let Some(rb) = &o.readback {
            max_vel = rb.stats.max_vel;
        }
    }
    let wall = wall.elapsed().as_secs_f64();
    println!(
        "[{name}][{}] {n} cells: {:.2} ms/step | outers/step {:.1} | lin solves/step {:.1} iters/solve {:.1} | max|U| {:.7e}",
        if gpu { "GPU" } else { "CPU" },
        wall * 1e3 / steps as f64,
        outer_sum as f64 / steps as f64,
        lin_solves as f64 / steps as f64,
        if lin_solves > 0 { lin_iter_sum as f64 / lin_solves as f64 } else { 0.0 },
        max_vel
    );
}

/// `CFD2_PROBE_GPU_ONLY=1` skips the (slow) CPU reference lines so lever
/// iterations only pay for the GPU timing runs.
fn probe_gpu_only() -> bool {
    std::env::var("CFD2_PROBE_GPU_ONLY").is_ok_and(|v| v == "1")
}

#[test]
fn a_backstep_gpu_vs_cpu() {
    let mesh = backstep_mesh();
    run("backstep", &mesh, true, 60);
    if !probe_gpu_only() {
        run("backstep", &mesh, false, 60);
    }
}

#[test]
fn b_obstacle_gpu_vs_cpu() {
    let mesh = obstacle_mesh();
    run("obstacle", &mesh, true, 60);
    if !probe_gpu_only() {
        run("obstacle", &mesh, false, 60);
    }
}

/// Mesh-size sweep: where is the GPU/CPU crossover?
#[test]
fn d_obstacle_size_sweep() {
    for cell in [0.0125, 0.00625] {
        let mesh = obstacle_mesh_at(cell);
        run(&format!("obstacle-{cell}"), &mesh, true, 30);
        if !probe_gpu_only() {
            run(&format!("obstacle-{cell}"), &mesh, false, 30);
        }
    }
}

/// Count real submissions/dispatches per step for the GUI-default GPU config.
#[test]
fn c_obstacle_gpu_dispatch_census() {
    use cfd2::solver::gpu::dispatch_counter::global_dispatch_counter;
    let mesh = obstacle_mesh();
    let p = params();
    let n = mesh.num_cells();
    let DriverBuild { mut driver, .. } = pollster::block_on(SolverDriver::build(
        &mesh,
        incompressible_momentum_model().expect("model"),
        &p,
        &vec![(0.0, 0.0); n],
        &vec![0.0; n],
        None,
        None,
    ))
    .expect("gpu driver build");
    driver.apply_params(&p);
    for i in 0..20 {
        driver.step(i % 5 == 0);
    }
    let c = global_dispatch_counter();
    let s = cfd2::solver::gpu::submission_counter::global_submission_counter();
    c.reset();
    c.enable();
    s.reset();
    s.enable();
    let steps = 10usize;
    let wall = Instant::now();
    for _ in 0..steps {
        driver.step(false);
    }
    let no_readback_ms = wall.elapsed().as_secs_f64() * 1e3 / steps as f64;
    c.disable();
    s.disable();
    println!("=== no-readback steps: {no_readback_ms:.2} ms/step ===");
    let stats = c.get_stats();
    println!("=== dispatch census over {steps} steps (no readback) ===");
    cfd2::solver::gpu::dispatch_counter::DispatchCounter::print_stats_static(&stats);
    let sub = s.get_stats();
    println!(
        "=== submissions over {steps} steps: total {} ({:.1}/step) ===",
        sub.total_submissions,
        sub.total_submissions as f64 / steps as f64
    );
    let mut by: Vec<_> = sub.by_label.iter().collect();
    by.sort_by(|a, b| b.1.cmp(a.1));
    for (label, count) in by.iter().take(20) {
        println!("  {:<55} {}", label, count);
    }
}
