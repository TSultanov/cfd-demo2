//! CPU-backend performance benchmark on the channel-with-obstacle (fine mesh).
//!
//! Drives the shipping GUI headline demo (incompressible momentum + cut-cell
//! channel-with-obstacle mesh + the GUI's INCOMPRESSIBLE defaults) through the
//! shared `SolverDriver` — exactly the GUI seam — but on the CPU backend, for a
//! handful of initial steps. It exists to measure CPU-backend performance on
//! the incompressible coupled solver, NOT to validate physics (a few steps from
//! rest are not converged).
//!
//! Everything is env-driven so a single test binary can be swept across thread
//! counts by re-invoking the process:
//!   CFD2_BACKEND=cpu CFD2_CPU_ENGINE=transpiled CFD2_CPU_THREADS=16 \
//!   CFD2_CPU_PROFILE=1 CFD2_BENCH_SIZE=0.005 CFD2_BENCH_STEPS=3 \
//!   cargo test --features "dev-tests ui" --release --test cpu_obstacle_bench -- --nocapture
//!
//! Env knobs:
//!   CFD2_BENCH_SIZE  target cut-cell size (default 0.005; GUI default is 0.025)
//!   CFD2_BENCH_STEPS number of steps to time (default 3)

#![cfg(all(feature = "dev-tests", feature = "ui"))]

use cfd2::sim::{DriverBuild, SolverDriver};
use cfd2::solver::mesh::{generate_cut_cell_mesh, ChannelWithObstacle, Mesh};
use cfd2::solver::model::incompressible_momentum_model;
use cfd2::ui::fluid::Fluid;
use cfd2::ui::model_defaults::gui_defaults_for;
use nalgebra::{Point2, Vector2};
use std::time::Instant;

// The GUI channel-with-obstacle geometry (src/ui/app.rs `GeometryType::ChannelWithObstacle`).
const LENGTH: f64 = 3.0;
const HEIGHT: f64 = 1.0;

fn env_f64(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn obstacle_mesh(size: f64) -> Mesh {
    let geo = ChannelWithObstacle {
        length: LENGTH,
        height: HEIGHT,
        obstacle_center: Point2::new(1.0, 0.51),
        obstacle_radius: 0.1,
    };
    let mut mesh = generate_cut_cell_mesh(&geo, size, size, 1.2, Vector2::new(LENGTH, HEIGHT));
    mesh.smooth(&geo, 0.3, 100);
    mesh
}

#[test]
fn cpu_obstacle_bench() {
    let size = env_f64("CFD2_BENCH_SIZE", 0.005);
    let steps = env_usize("CFD2_BENCH_STEPS", 3);

    let threads = std::env::var("CFD2_CPU_THREADS").unwrap_or_else(|_| "1".into());
    let engine = std::env::var("CFD2_CPU_ENGINE").unwrap_or_else(|_| "interpreter".into());
    eprintln!(
        "[bench] size={size} steps={steps} engine={engine} threads={threads} hw_par={}",
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0)
    );

    let air = Fluid::presets()[1].clone();
    let params = gui_defaults_for("incompressible_momentum").to_runtime_params(
        air.density as f32,
        air.viscosity as f32,
        air.eos,
    );

    let t_mesh = Instant::now();
    let mesh = obstacle_mesh(size);
    let n = mesh.num_cells();
    eprintln!("[bench] mesh built: {n} cells in {:.2}s", t_mesh.elapsed().as_secs_f64());

    let initial_u = vec![(0.0, 0.0); n];
    let initial_p = vec![0.0; n];

    let t_build = Instant::now();
    let DriverBuild { mut driver, .. } = pollster::block_on(SolverDriver::build(
        &mesh,
        incompressible_momentum_model().expect("incompressible model"),
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

    // Time each step. The per-phase split is printed by the CFD2_CPU_PROFILE
    // instrument inside CpuSolver::step.
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
