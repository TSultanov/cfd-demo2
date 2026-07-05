//! Regression test for the CPU-backend freeze on the all-Mach thermal
//! obstacle case (works on GPU, produced no sensible CPU solution). Root
//! cause: CPU FGMRES measured convergence against `||b||` alone while the GPU
//! clamps the relative scale to `min(||b||, ||r0||)`; with an RHS dominated by
//! BDF2/ddt terms every CPU solve "converged" at zero iterations under the
//! Eisenstat-Walker tolerance, so the state never moved.
//!
//! Drives the GUI ALLMACH defaults through `SolverDriver` (incl. Turkel
//! `psi_precond` seeding) on CPU and asserts the flow develops: the obstacle
//! carves a stagnation/wake region and builds a pressure field. Set
//! `CFD2_BACKEND=gpu` to compare; `CFD2_REPRO_SIZE`/`CFD2_REPRO_STEPS` override
//! the mesh size / step count.

#![cfg(all(feature = "dev-tests", feature = "ui"))]

use cfd2::sim::{DriverBuild, SolverDriver};
use cfd2::solver::mesh::{generate_cut_cell_mesh, ChannelWithObstacle, Mesh};
use cfd2::solver::model::allmach_thermal_model;
use cfd2::ui::fluid::Fluid;
use cfd2::ui::model_defaults::gui_defaults_for;
use nalgebra::{Point2, Vector2};

fn env_f64(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn obstacle_mesh(size: f64) -> Mesh {
    let geo = ChannelWithObstacle {
        length: 3.0,
        height: 1.0,
        obstacle_center: Point2::new(1.0, 0.51),
        obstacle_radius: 0.1,
    };
    let mut mesh = generate_cut_cell_mesh(&geo, size, size, 1.2, Vector2::new(3.0, 1.0));
    mesh.smooth(&geo, 0.3, 100);
    mesh
}

fn stats(name: &str, v: &[f64]) -> String {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut bad = 0usize;
    for &x in v {
        if !x.is_finite() {
            bad += 1;
            continue;
        }
        lo = lo.min(x);
        hi = hi.max(x);
    }
    format!("{name}=[{lo:.4e},{hi:.4e}]{}", if bad > 0 { format!(" NAN={bad}") } else { String::new() })
}

#[test]
fn allmach_thermal_obstacle_develops_on_cpu() {
    // CPU backend by default (the regression's subject); override via env for
    // a GPU comparison run.
    if std::env::var("CFD2_BACKEND").is_err() {
        std::env::set_var("CFD2_BACKEND", "cpu");
        std::env::set_var("CFD2_CPU_ENGINE", "transpiled");
        std::env::set_var("CFD2_CPU_THREADS", "4");
    }
    let size = env_f64("CFD2_REPRO_SIZE", 0.01);
    let steps = env_usize("CFD2_REPRO_STEPS", 10);
    let backend = std::env::var("CFD2_BACKEND").unwrap();

    let air = Fluid::presets()[1].clone();
    let d = gui_defaults_for("allmach_thermal");
    let params = d.to_runtime_params(air.density as f32, air.viscosity as f32, air.eos);

    let mesh = obstacle_mesh(size);
    let n = mesh.num_cells();
    eprintln!("[repro] backend={backend} size={size} cells={n} steps={steps}");

    let initial_u = vec![(d.inlet_velocity as f64, 0.0); n];
    let initial_p = vec![0.0; n];
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

    let mut min_umag = f64::INFINITY;
    let mut p_spread = 0.0f64;
    for i in 0..steps {
        solver.step_with_stats().expect("step diverged");
        let u = pollster::block_on(solver.get_field_vec2("U")).expect("U");
        let p = pollster::block_on(solver.get_field_scalar("p")).expect("p");
        let t = pollster::block_on(solver.get_field_scalar("T")).expect("T");
        let umag: Vec<f64> = u.iter().map(|&(x, y)| (x * x + y * y).sqrt()).collect();
        eprintln!(
            "[repro] step {i:>3} {} {} {}",
            stats("|U|", &umag),
            stats("p", &p),
            stats("T", &t),
        );
        assert!(
            umag.iter().chain(p.iter()).chain(t.iter()).all(|v| v.is_finite()),
            "step {i}: non-finite field values"
        );
        min_umag = umag.iter().cloned().fold(f64::INFINITY, f64::min);
        let plo = p.iter().cloned().fold(f64::INFINITY, f64::min);
        let phi = p.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        p_spread = phi - plo;
    }

    // The flow must DEVELOP: the no-slip obstacle carves a stagnation/wake
    // region (the frozen-state bug left min |U| at freestream and p at 0).
    let inlet = d.inlet_velocity as f64;
    assert!(
        min_umag < 0.5 * inlet,
        "no wake developed: min |U| = {min_umag:.3e} vs inlet {inlet:.3e} — state frozen?"
    );
    assert!(
        p_spread > 1e-6,
        "no pressure field developed: spread = {p_spread:.3e} — state frozen?"
    );
}
