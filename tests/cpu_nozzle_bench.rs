//! CPU-backend performance benchmark on the CD nozzle (fine mesh).
//!
//! Drives the shipping GUI "Supersonic Nozzle" case (allmach_thermal +
//! structured nozzle mesh + ALLMACH_THERMAL_NOZZLE params) through the shared
//! `SolverDriver` — exactly the GUI seam — but on the CPU backend, for a handful
//! of initial steps. It exists to measure how well the transpiled CPU backend
//! saturates the cores, NOT to validate physics (a few steps from rest are not
//! converged).
//!
//! Everything is env-driven so a single test binary can be swept across thread
//! counts by re-invoking the process:
//!   CFD2_BACKEND=cpu CFD2_CPU_ENGINE=transpiled CFD2_CPU_THREADS=16 \
//!   CFD2_CPU_PROFILE=1 CFD2_BENCH_SIZE=0.002 CFD2_BENCH_STEPS=3 \
//!   cargo test --features "dev-tests ui" --release --test cpu_nozzle_bench -- --nocapture
//!
//! Env knobs:
//!   CFD2_BENCH_SIZE  target cell size (default 0.002)  → nx=round(3/size), ny=round(1/size)
//!   CFD2_BENCH_STEPS number of steps to time (default 3)

#![cfg(all(feature = "dev-tests", feature = "ui"))]

use cfd2::sim::{DriverBuild, SolverDriver};
use cfd2::solver::mesh::{generate_structured_nozzle_mesh, BoundarySides, BoundaryType, Mesh};
use cfd2::solver::model::allmach_thermal_model;
use cfd2::ui::fluid::Fluid;
use cfd2::ui::model_defaults::ALLMACH_THERMAL_NOZZLE;
use std::time::Instant;

// The GUI nozzle geometry (src/ui/app.rs `GeometryType::Nozzle`).
const LENGTH: f64 = 3.0;
const HEIGHT: f64 = 1.0;
const THROAT_H: f64 = 0.40;
const THROAT_FRAC: f64 = 0.40;
const EXIT_H: f64 = 0.80;

fn env_f64(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn nozzle_mesh(nx: usize, ny: usize) -> Mesh {
    let mut mesh = generate_structured_nozzle_mesh(
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
    );
    // Optional cell-renumbering A/B: CFD2_MESH_ORDER=rcm|hilbert|random.
    mesh.apply_env_cell_order();
    mesh
}

#[test]
fn cpu_nozzle_bench() {
    let size = env_f64("CFD2_BENCH_SIZE", 0.002);
    let steps = env_usize("CFD2_BENCH_STEPS", 3);
    let nx = (LENGTH / size).round() as usize;
    let ny = (HEIGHT / size).round() as usize;

    let threads = std::env::var("CFD2_CPU_THREADS").unwrap_or_else(|_| "1".into());
    let engine = std::env::var("CFD2_CPU_ENGINE").unwrap_or_else(|_| "interpreter".into());
    eprintln!(
        "[bench] size={size} nx={nx} ny={ny} steps={steps} engine={engine} threads={threads} \
         hw_par={}",
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0)
    );

    let air = Fluid::presets()[1].clone();
    let mut params = ALLMACH_THERMAL_NOZZLE.to_runtime_params(
        air.density as f32,
        air.viscosity as f32,
        air.eos,
    );
    // Optional preconditioner override for GPU A/B runs (e.g. amg vs jacobi).
    if let Ok(p) = std::env::var("CFD2_BENCH_PRECOND") {
        params.preconditioner = match p.as_str() {
            "amg" => cfd2::solver::PreconditionerType::Amg,
            "jacobi" => cfd2::solver::PreconditionerType::Jacobi,
            other => panic!("unknown CFD2_BENCH_PRECOND: {other}"),
        };
        eprintln!("[bench] preconditioner override: {p}");
    }

    let t_mesh = Instant::now();
    let mesh = nozzle_mesh(nx, ny);
    let n = mesh.num_cells();
    eprintln!("[bench] mesh built: {n} cells in {:.2}s", t_mesh.elapsed().as_secs_f64());

    let initial_u = vec![(0.0, 0.0); n];
    let initial_p = vec![0.0; n];

    let t_build = Instant::now();
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
    driver.apply_params(&params);
    let mut solver = driver.into_solver();
    eprintln!("[bench] driver+solver built in {:.2}s", t_build.elapsed().as_secs_f64());

    // Time each step. The per-phase split (assembly vs serial linear solve) is
    // printed by the CFD2_CPU_PROFILE instrument inside CpuSolver::step.
    let mut wall = Vec::new();
    for i in 0..steps {
        let t = Instant::now();
        solver.step_with_stats().expect("bench step diverged");
        let e = t.elapsed().as_secs_f64();
        wall.push(e);
        eprintln!("[bench] step {i} wall={:.3}s", e);
    }

    let warm: Vec<f64> = wall.iter().skip(1).cloned().collect();
    let avg = if warm.is_empty() {
        wall[0]
    } else {
        warm.iter().sum::<f64>() / warm.len() as f64
    };
    eprintln!(
        "[bench] RESULT engine={engine} threads={threads} cells={n} avg_warm_step={avg:.3}s",
    );
}
