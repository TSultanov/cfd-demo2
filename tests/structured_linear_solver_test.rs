//! Regression gates for the structured banded linear solve's CONVERGENCE
//! SEMANTICS (see `src/solver/banded_schur.rs`):
//!
//! 1. `structured_channel_pressure_honesty` — the aggregate-norm 1e-4 exit used
//!    to under-resolve the PRESSURE block (momentum rows dominate the residual
//!    norm): the empty structured channel at dt=0.011 converged to a developed
//!    delta-p of ~0.254 when the honest (tight-tol 1e-8) answer is ~0.364
//!    (analytic Poiseuille 12·mu·L·u/H² = 0.36) — a ~30% pressure bias invisible
//!    to the Picard exit, which measures applied corrections (commit 9b3e241's
//!    "known-remaining" note). The per-block convergence test must recover the
//!    honest delta-p at the DEFAULT tolerance.
//!
//! 2. `banded_gmres_nan_*` — a non-finite operator/rhs used to run the full
//!    200-cycle x 60-iteration budget (every NaN comparison is false) and then
//!    write the NaN x into the state. The solve must bail immediately with a
//!    finite iterate (the tracked best) and a clear failure signal.
//!
//! 3. `outer_residuals_nan_reads_not_converged` — `f32::max` ignores NaN, so a
//!    diverged state used to read as delta 0 = "converged" in the shared Picard
//!    outer residual.
#![cfg(feature = "cpu")]

use cfd2::solver::banded_schur::{banded_gmres_t, BandedPrecond, CoupledPrecondKind};
use cfd2::solver::cpu::structured::StructuredModelSolver;
use cfd2::solver::gpu::structured::{BcComp, Edge, StructuredGrid};
use cfd2::solver::model::incompressible_momentum_structured_model;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::TimeScheme;

fn threads() -> usize {
    std::env::var("CFD2_CPU_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(3)
}

/// The empty structured incompressible channel (60x20, 3x1, rho=1, mu=0.02,
/// u_in=0.5 — `diag_structured_channel_development`'s configuration) at FIXED
/// dt=0.011: the small-dt regime where the OLD solve was dishonest about the
/// pressure block — the default-tol (1e-4) exit returned a converged claim at
/// 60 iterations/step while the honest (1e-8) solve needed ~711/step, and the
/// two disagreed on the developed delta-p (0.5028 vs 0.5093 measured at
/// baseline). The per-block true-residual exit must reproduce the HONEST
/// delta-p at the default tolerance. Returns the developed mean-inlet-column
/// pressure drop and mean linear iterations/step (last Picard outer).
fn channel_delta_p(precond: CoupledPrecondKind, steps: usize) -> (f64, f64) {
    let env = |k: &str, d: usize| -> usize {
        std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
    };
    let (nx, ny, lx, ly) = (env("DP_NX", 60), env("DP_NY", 20), 3.0f64, 1.0f64);
    let (rho, mu, u_in, dt) = (1.0f64, 0.02f64, 0.5f64, 0.011f64);
    let model = incompressible_momentum_structured_model().expect("model");
    let mut solver = StructuredModelSolver::with_config(
        StructuredGrid::new(nx, ny, lx, ly),
        &model,
        dt,
        8,
        Scheme::Upwind,
        TimeScheme::BDF2,
    )
    .expect("structured solver");
    solver.set_engine(cfd2::solver::cpu::CpuEngine::Transpiled, threads());
    solver.set_fluid(rho, mu);
    solver.set_alpha_u(0.7);
    solver.set_alpha_p(0.3);
    solver.set_preconditioner(precond);
    solver.set_outer_auto_converge(true);
    solver.set_outer_iters(8);
    solver.set_boundaries(move |edge, _x, _y| {
        let d = |v: f32| BcComp { kind: 1, value: v };
        let g = || BcComp { kind: 2, value: 0.0 };
        match edge {
            Edge::Left => (1, vec![d(u_in as f32), d(0.0), g()]),
            Edge::Right => (2, vec![g(), g(), d(0.0)]),
            _ => (3, vec![d(0.0), d(0.0), g()]),
        }
    });

    let p_off = solver.field_offset("p").expect("p");
    let col_dp = |solver: &StructuredModelSolver| -> f64 {
        let p = solver.get_scalar(p_off);
        let col = |i: usize| -> f64 { (0..ny).map(|j| p[j * nx + i]).sum::<f64>() / ny as f64 };
        col(0) - col(nx - 1)
    };

    let mut lin_iters_total: u64 = 0;
    for st in 1..=steps {
        solver.step();
        lin_iters_total += solver.last_stats().linear_iters as u64;
        if st % 500 == 0 {
            println!(
                "[dp-honesty]   step {st}: delta-p = {:.4}, mean iters/step = {:.1}",
                col_dp(&solver),
                lin_iters_total as f64 / st as f64
            );
        }
    }
    (col_dp(&solver), lin_iters_total as f64 / steps as f64)
}

/// FIX gate: at the DEFAULT linear tolerance the developed channel delta-p must
/// match the honest tight-tolerance reference (baseline 1e-8 measurement on
/// this exact protocol: 0.5093 at 711 iters/step), not drift with the exit
/// criterion — and it must get there without burning the old 12,000-iteration
/// restart budget.
#[test]
fn structured_channel_pressure_honesty() {
    // `DP_PRECOND=bj|schur|amg` and `DP_STEPS` for measurement sweeps; the gate
    // itself runs the GUI-default BlockJacobi.
    let precond = match std::env::var("DP_PRECOND").as_deref() {
        Ok("amg") => CoupledPrecondKind::SchurAmg,
        Ok("schur") => CoupledPrecondKind::Schur,
        _ => CoupledPrecondKind::BlockJacobi,
    };
    let steps = std::env::var("DP_STEPS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1000);
    let (dp, iters_per_step) = channel_delta_p(precond, steps);
    println!(
        "[dp-honesty] {precond:?} steps={steps}: delta-p = {dp:.4} (honest ref ~0.509), \
         mean linear iters/step = {iters_per_step:.1}"
    );
    assert!(
        (0.47..0.55).contains(&dp),
        "developed channel delta-p {dp:.4} outside the honest band [0.47, 0.55] \
         (tight-tol 1e-8 reference on this protocol: 0.5093)"
    );
}

/// THE mixed-norm false-convergence pin (FIX 1): with block diagonals >> 1
/// (small-dt `vol/dt` regime) the OLD left-preconditioned BlockJacobi head test
/// compared `||M^-1(b - A x)||` against `||b||` — at x = 0 that ratio is
/// ~1/diag, i.e. "converged" at tol=1e-4 WITHOUT A SINGLE ITERATION, returning
/// x = 0 with an O(1) true residual. The right-preconditioned per-block exit
/// must actually solve the system.
#[test]
fn banded_gmres_large_diagonal_no_false_convergence() {
    let (nx, ny, s) = (10usize, 8usize, 2usize);
    let ncells = nx * ny;
    let diag = 1.0e6f32;
    let mut a = vec![0.0f32; ncells * 5 * s * s];
    for p in 0..ncells {
        let (i, j) = (p % nx, p / nx);
        for r in 0..s {
            a[p * 5 * s * s + 5 * s * r + 2 * s + r] = diag; // diagonal band
            // In-grid neighbours: -1 Laplacian coupling (bands S=0,W=1,E=3,N=4).
            for (band, exists) in
                [(0usize, j > 0), (1, i > 0), (3, i + 1 < nx), (4, j + 1 < ny)]
            {
                if exists {
                    a[p * 5 * s * s + 5 * s * r + band * s + r] = -1.0;
                }
            }
        }
    }
    // Known solution; b = A x*.
    let xstar: Vec<f64> = (0..ncells * s)
        .map(|k| ((k * 29 % 13) as f64 - 6.0) * 0.1)
        .collect();
    let b64 = cfd2::solver::banded_schur::spmv(&a, nx, ny, s, &xstar);
    let b: Vec<f32> = b64.iter().map(|&v| v as f32).collect();

    let (x, rel, iters) = banded_gmres_t(
        &a,
        nx,
        ny,
        s,
        &b,
        &BandedPrecond::BlockJacobi,
        60,
        200,
        1e-4,
        1,
        None,
    );
    let max_err = (0..ncells * s)
        .map(|k| (x[k] as f64 - xstar[k]).abs())
        .fold(0.0f64, f64::max);
    println!("[large-diag] rel={rel:.3e} iters={iters} max_err={max_err:.3e}");
    assert!(iters > 0, "large-diagonal system must not be declared converged at x = 0");
    assert!(
        max_err < 1e-4,
        "solve did not recover x* (max_err {max_err:.3e}) — mixed-norm false convergence?"
    );
    assert!(rel <= 1e-4, "returned rel_res {rel:.3e} must reflect a truly converged solve");
}

/// A NaN rhs must make the banded solve bail out quickly with a finite iterate
/// (the tracked best — here the zero initial guess) and a non-converged signal,
/// instead of burning the full restart budget and returning NaN.
#[test]
fn banded_gmres_nan_rhs_bails_quickly() {
    let (nx, ny, s) = (8usize, 6usize, 3usize);
    let ncells = nx * ny;
    // Well-formed diagonally-dominant operator; poisoned rhs.
    let mut a = vec![0.0f32; ncells * 5 * s * s];
    for p in 0..ncells {
        for r in 0..s {
            a[p * 5 * s * s + 5 * s * r + 2 * s + r] = 4.0; // diagonal band
        }
    }
    let mut b = vec![1.0f32; ncells * s];
    b[7] = f32::NAN;

    let t0 = std::time::Instant::now();
    let (x, rel, iters) = banded_gmres_t(
        &a,
        nx,
        ny,
        s,
        &b,
        &BandedPrecond::BlockJacobi,
        60,
        200,
        1e-4,
        1,
        None,
    );
    let elapsed = t0.elapsed();
    println!("[nan-bail] rel={rel:.3e} iters={iters} elapsed={elapsed:?}");
    assert!(
        x.iter().all(|v| v.is_finite()),
        "NaN rhs leaked a non-finite iterate into x"
    );
    assert!(
        iters <= 60,
        "NaN rhs must bail within one restart cycle, ran {iters} iterations"
    );
    assert!(
        !(rel <= 1e-4),
        "NaN rhs must NOT report convergence (rel={rel})"
    );
}

/// Same for a NaN in the OPERATOR: the first Arnoldi product poisons the cycle;
/// the head must catch it and restore the best (finite) iterate.
#[test]
fn banded_gmres_nan_matrix_bails_quickly() {
    let (nx, ny, s) = (8usize, 6usize, 3usize);
    let ncells = nx * ny;
    let mut a = vec![0.0f32; ncells * 5 * s * s];
    for p in 0..ncells {
        for r in 0..s {
            a[p * 5 * s * s + 5 * s * r + 2 * s + r] = 4.0;
        }
    }
    a[5 * s * s + 2 * s] = f32::NAN; // poison one diagonal block entry
    let b = vec![1.0f32; ncells * s];

    let (x, rel, iters) = banded_gmres_t(
        &a,
        nx,
        ny,
        s,
        &b,
        &BandedPrecond::BlockJacobi,
        60,
        200,
        1e-4,
        1,
        None,
    );
    println!("[nan-mat-bail] rel={rel:.3e} iters={iters}");
    assert!(
        x.iter().all(|v| v.is_finite()),
        "NaN operator leaked a non-finite iterate into x"
    );
    assert!(
        iters <= 120,
        "NaN operator must bail within ~one cycle of detection, ran {iters} iterations"
    );
    assert!(!(rel <= 1e-4), "NaN operator must NOT report convergence");
}

/// A NaN state must read as NOT converged in the shared Picard outer residual
/// (f32::max ignores NaN, so an unguarded delta reads 0 = "converged").
#[test]
fn outer_residuals_nan_reads_not_converged() {
    use cfd2::solver::banded_schur::structured_outer_residuals;
    let stride = 3usize;
    let state = vec![f32::NAN; 4 * stride];
    let state_iter = vec![0.0f32; 4 * stride];
    let groups = vec![vec![0usize, 1], vec![2usize]];
    let res = structured_outer_residuals(&state, &state_iter, stride, &groups);
    assert!(
        res.iter().all(|r| r.is_infinite()),
        "NaN state must read as diverged (INFINITY), got {res:?}"
    );

    // Finite state still behaves normally.
    let state = vec![1.0f32; 4 * stride];
    let res = structured_outer_residuals(&state, &state_iter, stride, &groups);
    assert!(res.iter().all(|r| (*r - 1.0).abs() < 1e-6), "finite path changed: {res:?}");
}

/// FREEZE-ON-FAILURE semantics: a NaN-poisoned assembly (here: the thermal
/// model's psi_precond recovery with an UN-SEEDED u_ref emits non-finite values
/// at quiescent cells from step 2 on) must not let the solver either (a) write
/// NaN into the state, or (b) apply the bailed zero iterate and drive the flow
/// to zero. The step FREEZES (update skipped) and reports linear_res = inf.
/// The old NaN-blind solve "handled" this by grinding 12,000 iterations per
/// solve and freezing only by the accident of an update-kernel NaN guard.
#[test]
fn structured_nan_assembly_freezes_step_instead_of_zeroing() {
    use cfd2::solver::model::allmach_thermal_structured_model;
    let (nx, ny) = (16usize, 16usize);
    let model = allmach_thermal_structured_model().expect("model");
    let s = model.system.unknowns_per_cell() as usize;
    let mut solver = StructuredModelSolver::new(
        StructuredGrid::new(nx, ny, 1.0, 1.0),
        &model,
        0.02,
        3,
    )
    .expect("solver");
    solver.set_engine(cfd2::solver::cpu::CpuEngine::Transpiled, threads());
    solver.set_fluid(1.0, 0.02);
    let psi = 0.5f64;
    for (n, v) in [
        ("psi", psi),
        ("psi_precond", 1.0),
        ("rho", 1.0),
        ("rho_t_ref", 1.0),
        ("T", 1.0),
        ("t_ref", 1.0),
        ("rho_floor", psi * 1e-5),
        // deliberately NO u_ref / precond_mask -> psi_precond recovery goes
        // non-finite at rest cells after the first update.
    ] {
        if solver.field_offset(n).is_some() {
            solver.set_named_field(n, move |_, _| v);
        }
    }
    solver.set_boundaries(move |edge, _x, _y| {
        let u_wall = if matches!(edge, Edge::Top) { 1.0 } else { 0.0 };
        let btype = if matches!(edge, Edge::Top) { 5 } else { 3 };
        let mut v = vec![
            BcComp { kind: 1, value: u_wall },
            BcComp { kind: 1, value: 0.0 },
            BcComp { kind: 2, value: 0.0 },
        ];
        if s >= 4 {
            v.push(BcComp { kind: 2, value: 0.0 });
        }
        (btype, v)
    });

    for _ in 0..3 {
        solver.step();
    }
    let stats = solver.last_stats();
    let ux = solver.state_field(0);
    let uy = solver.state_field(1);
    let mut umax = 0.0f64;
    for (a, b) in ux.iter().zip(&uy) {
        assert!(a.is_finite() && b.is_finite(), "NaN leaked into the state");
        umax = umax.max(a.hypot(*b));
    }
    println!(
        "[freeze] umax={umax:.4} last linear_res={} linear_iters={}",
        stats.linear_res, stats.linear_iters
    );
    // The step-1 developed flow must be PRESERVED (frozen), not zeroed by an
    // applied bail iterate.
    assert!(
        umax > 0.02,
        "frozen state lost its developed flow (umax={umax:.2e}) — the bailed \
         zero iterate was applied instead of freezing"
    );
    // The failure must be SURFACED, not silent.
    assert!(
        !stats.linear_res.is_finite(),
        "a NaN-poisoned solve must report a non-finite linear_res, got {}",
        stats.linear_res
    );
    // And it must be CHEAP — the old NaN grind burned the full budget.
    assert!(
        stats.linear_iters <= 60,
        "NaN-poisoned solve should bail within one cycle, ran {}",
        stats.linear_iters
    );
}
