//! Forced-stall test for the FGMRES stall-stop (roadmap Arc 3).
//!
//! With the production tolerance (1e-12, below the f32 floor) every solve
//! burns to the max_iters cap. With CFD2_FGMRES_STALL_REL=1.0 the level
//! criterion is always satisfied, so a solve must stop within two restart
//! checkpoints of reaching its f32 floor:
//! - host-driven path (outer_batched_mode=false): LinearSolverStats
//!   reports the real iteration count -> assert it breaks below the cap;
//! - encoded chunked path (default): the iteration count is the encoded
//!   budget regardless, so assert solution agreement with a stall-off run
//!   instead (the guard hands back the best iterate, so accuracy must not
//!   degrade).
#![cfg(feature = "dev-tests")]

use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType};
use cfd2::solver::model::helpers::{SolverFieldAliasesExt, SolverRuntimeParamsExt};
use cfd2::solver::model::incompressible_momentum_model;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};

const N: usize = 8;
const STEPS: usize = 3;

fn run(batched: bool) -> (Vec<(f64, f64)>, Vec<f64>, Vec<u32>) {
    let mesh = generate_structured_rect_mesh(
        N,
        N,
        1.0,
        1.0,
        BoundarySides {
            left: BoundaryType::Wall,
            right: BoundaryType::Wall,
            bottom: BoundaryType::Wall,
            top: BoundaryType::Wall,
        },
    );
    let model = incompressible_momentum_model().expect("model");
    let mut solver = pollster::block_on(UnifiedSolver::new(
        &mesh,
        model,
        SolverConfig {
            advection_scheme: Scheme::SecondOrderUpwind,
            time_scheme: TimeScheme::BDF2,
            preconditioner: PreconditionerType::Jacobi,
            stepping: SteppingMode::Coupled,
        },
        None,
        None,
    ))
    .expect("solver init");

    solver.set_dt(0.05);
    solver.set_dtau(0.0).unwrap();
    solver.set_density(1.0).unwrap();
    solver.set_viscosity(1.0).unwrap();
    solver.set_alpha_u(0.7).unwrap();
    solver.set_alpha_p(0.3).unwrap();
    solver.set_outer_iters(5).unwrap();
    solver.set_outer_batched_mode(batched).unwrap();
    solver.set_u(&vec![(0.1, 0.0); mesh.num_cells()]);
    solver.set_p(&vec![0.0; mesh.num_cells()]);
    solver.initialize_history();

    let mut iters = Vec::new();
    for _ in 0..STEPS {
        for s in solver.step_with_stats().expect("step") {
            assert!(!s.diverged, "solve diverged under stall-stop");
            iters.push(s.iterations);
        }
    }
    let u = pollster::block_on(solver.get_u());
    let p = pollster::block_on(solver.get_p());
    (u, p, iters)
}

fn rel_l2(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f64 = b.iter().map(|y| y * y).sum();
    (num / den.max(1e-30)).sqrt()
}

#[test]
fn stall_stop_breaks_early_and_preserves_solution() {
    std::env::set_var("CFD2_QUIET", "1");

    // Baseline: stall off, encoded path.
    std::env::remove_var("CFD2_FGMRES_STALL_REL");
    let (u_off, p_off, _) = run(true);

    std::env::set_var("CFD2_FGMRES_STALL_REL", "1.0");

    // Host-driven path: the reported iteration count reflects the break.
    let (_, _, host_iters) = run(false);
    let min_iters = host_iters.iter().copied().min().unwrap_or(u32::MAX);
    println!("[stall-stop] host-driven per-solve iterations: {host_iters:?}");
    assert!(
        host_iters.iter().any(|&it| it < 200),
        "stall-stop never fired on the host path: {host_iters:?}"
    );
    assert!(min_iters >= 60, "broke before a full first restart cycle?");

    // Encoded chunked path: assert agreement with the stall-off run.
    let (u_on, p_on, _) = run(true);
    std::env::remove_var("CFD2_FGMRES_STALL_REL");

    assert!(
        u_on.iter().all(|(x, y)| x.is_finite() && y.is_finite())
            && p_on.iter().all(|v| v.is_finite()),
        "non-finite fields under stall-stop"
    );
    let ux_on: Vec<f64> = u_on.iter().map(|v| v.0).collect();
    let uy_on: Vec<f64> = u_on.iter().map(|v| v.1).collect();
    let ux_off: Vec<f64> = u_off.iter().map(|v| v.0).collect();
    let uy_off: Vec<f64> = u_off.iter().map(|v| v.1).collect();
    let du = rel_l2(&ux_on, &ux_off).max(rel_l2(&uy_on, &uy_off));
    // All-wall box: pressure is pure-Neumann (gauge-free); compare de-meaned.
    let demean = |v: &[f64]| -> Vec<f64> {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        v.iter().map(|x| x - m).collect()
    };
    let dp = rel_l2(&demean(&p_on), &demean(&p_off));
    println!("[stall-stop] encoded path rel_l2 vs stall-off: u={du:.3e} p={dp:.3e}");
    assert!(
        du < 1e-3 && dp < 1e-3,
        "stall-stop changed the solution materially: u={du:.3e} p={dp:.3e}"
    );
}
