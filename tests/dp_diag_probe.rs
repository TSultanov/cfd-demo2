//! Diagnostic probes for the assembled-matrix d_p formulations: step the
//! incompressible model at momentum-MMS-like settings and dump the actual
//! d_p state values against the candidate scales, plus per-step |u| growth
//! and linear-solve stats over an (outer_iters, alpha_u, alpha_p,
//! lin_max_iters) sweep. Not a correctness gate.
//!
//! FINDINGS (June 2026), in discovery order:
//!
//! 1. FromAssembledDiagonal (OpenFOAM rAU, d_p = V/a_P): the kernel is
//!    CORRECT — d_p settles at ~3.5e-3, matching the physical interior
//!    estimate V/(rho*V/dt + 4*mu) (not the closed form 3.5e-2, not raw
//!    volume). The OUTER LOOP AMPLIFIES anyway: |u|max x3.5/step at
//!    outer=5 in an unforced closed box, NaN by step 2 at outer=25; all
//!    alpha pairings amplify; theta-damping irrelevant.
//!
//! 2. The amplification SURVIVES f32-floor linear solves (cap-200 abs
//!    residuals 1e-9..1e-6 while |u| still grows), so it is a property of
//!    the exact-solve outer fixed-point map at the Schur-consistent d_p
//!    scale — which REFUTES pressure-row equilibration as a fix (a pure
//!    row scaling cannot change exact-solve outer dynamics).
//!
//! 3. The lin_cap=2000 column exposed a SEPARATE bug: f32 FGMRES loses
//!    orthogonality over long runs and a restart cycle can APPLY an update
//!    that increases the true residual, compounding across restarts
//!    (2e-9 -> 5e15 -> NaN). The same breakdown caused the n=16 MMS
//!    late blow-up at cap=200 with the Schur preconditioner (probe
//!    probe_mms_n16_blowup: cap=100 stable, cap=400 explodes by step 5,
//!    default preconditioner solid — iteration count, not physics). FIXED
//!    by the restart-boundary monotonicity guard (gmres_logic/
//!    restart_guard + gmres_ops/guard_copy GPU-side; best-x snapshot loop
//!    host-side): with the guard all cap columns hold steady.
//!
//! 4. FromAssembledRowSum (SIMPLEC, d_p = V/Σ_row a) is STABLE by
//!    construction: interior row sum = ddt coefficient, so interior d_p
//!    stays at the closed-form scale ((2/3)dt/rho under BDF2) while
//!    Dirichlet boundaries shrink it locally. Probe decays at every
//!    configuration; all 5 MMS suites green. OpenFOAM: channel u/p
//!    -40%/-48%, backstep +3%, lid +11% — the lid corner error worsens
//!    with rAU-style spatial d_p (spatial-structure hypothesis refuted),
//!    so SIMPLEC stays default-off under the no-growth policy.
//!
//! To reproduce, flip the incompressible model's derive_rhie_chow call to
//! the formulation under test (see the comment there) and run these probes.
#![cfg(feature = "dev-tests")]

use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType};
use cfd2::solver::model::helpers::{SolverFieldAliasesExt, SolverRuntimeParamsExt};
use cfd2::solver::model::{
    incompressible_momentum_mms_model, incompressible_momentum_model,
    INCOMPRESSIBLE_MMS_SOURCE_FIELD,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};

#[test]
#[ignore]
fn probe_dp_diag_scale() {
    std::env::set_var("CFD2_QUIET", "1");
    let n = 8usize;
    // The lin_max_iters dimension discriminates fixed-point instability
    // (amplification survives near-exact solves) from linear-solve
    // inexactness (amplification disappears when the cap is lifted).
    for (outer_iters, alpha_u, alpha_p, lin_max_iters) in [
        (5usize, 0.7f32, 0.3f32, 200u32),
        (25, 0.7, 0.3, 200),
        (5, 1.0, 1.0, 200),
        (25, 1.0, 1.0, 200),
        (25, 1.0, 0.3, 200),
        (5, 0.7, 0.3, 2000),
        (25, 0.7, 0.3, 2000),
    ] {
        let mesh = generate_structured_rect_mesh(
            n,
            n,
            1.0,
            1.0,
            BoundarySides {
                left: BoundaryType::Wall,
                right: BoundaryType::Wall,
                bottom: BoundaryType::Wall,
                top: BoundaryType::Wall,
            },
        );
        let mut model = incompressible_momentum_model().expect("model");
        if let Some(ls) = model.linear_solver.as_mut() {
            ls.solver.max_iters = lin_max_iters;
        }
        let stride = model.state_layout.stride() as usize;
        let dp_off = model
            .state_layout
            .field("d_p")
            .expect("d_p in layout")
            .offset() as usize;
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

        let (mu, rho, dt) = (1.0f32, 1.0f32, 0.05f32);
        solver.set_dt(dt);
        solver.set_dtau(0.0).unwrap();
        solver.set_density(rho).unwrap();
        solver.set_viscosity(mu).unwrap();
        solver.set_alpha_u(alpha_u).unwrap();
        solver.set_alpha_p(alpha_p).unwrap();
        solver.set_outer_iters(outer_iters).unwrap();
        solver.set_u(&vec![(0.1, 0.0); mesh.num_cells()]);
        solver.set_p(&vec![0.0; mesh.num_cells()]);
        solver.initialize_history();

        for step in 0..4 {
            let stats = solver.step_with_stats().expect("stats");
            let last = stats.last().expect("solve stats");
            let worst_resid = stats.iter().map(|s| s.residual).fold(0.0f32, f32::max);
            let max_lin_iters = stats.iter().map(|s| s.iterations).max().unwrap_or(0);
            let state = pollster::block_on(solver.read_state_f32());
            let dp: Vec<f32> = (0..mesh.num_cells())
                .map(|i| state[i * stride + dp_off])
                .collect();
            let mut u_max = 0.0f32;
            for i in 0..mesh.num_cells() {
                u_max = u_max.max(state[i * stride].abs()).max(state[i * stride + 1].abs());
            }
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for &v in &dp {
                lo = lo.min(v);
                hi = hi.max(v);
            }
            let h = 1.0 / n as f32;
            let vol = h * h;
            let closed = 0.7 * dt / rho;
            let a_phys = rho * vol / dt + 4.0 * mu;
            println!(
                "[dp-probe] outer={outer_iters} a_u={alpha_u} a_p={alpha_p} lin_cap={lin_max_iters} step {step}: d_p [{lo:.4e},{hi:.4e}] |u|max={u_max:.4e} lin(last: iters={} resid={:.3e} conv={} div={}; step-worst: iters={max_lin_iters} resid={worst_resid:.3e}) | closed={closed:.4e} V/a_phys={:.4e}",
                last.iterations, last.residual, last.converged, last.diverged,
                vol / a_phys,
            );
        }
    }
}

/// Reproduces the n=16 Taylor-Green MMS late blow-up observed with the
/// SIMPLEC row-sum d_p (state converges to ~7e-4 by step 10, explodes by
/// step 20 at n=16 only; n=8/32/64 converge). Per-step d_p range, |u|max,
/// and per-outer linear stats around the explosion.
#[test]
#[ignore]
fn probe_mms_n16_blowup() {
    use std::f64::consts::PI;
    std::env::set_var("CFD2_QUIET", "1");
    let n = 16usize;
    // (lin_cap, use_default_precond): discriminate Krylov fragility (cap
    // changes the blow-up) from Schur-preconditioner pathology (default
    // preconditioner changes it).
    for (lin_cap, use_default_precond) in
        [(200u32, false), (100, false), (400, false), (200, true)]
    {
    let mesh = generate_structured_rect_mesh(n, n, 1.0, 1.0, BoundarySides::wall());
    let mut model = incompressible_momentum_mms_model().expect("model");
    if let Some(ls) = model.linear_solver.as_mut() {
        ls.solver.max_iters = lin_cap;
        if use_default_precond {
            ls.preconditioner =
                cfd2::solver::model::linear_solver::ModelPreconditionerSpec::Default;
        }
    }
    let stride = model.state_layout.stride() as usize;
    let dp_off = model
        .state_layout
        .field("d_p")
        .expect("d_p in layout")
        .offset() as usize;
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
    solver.set_outer_iters(25).unwrap();

    let exact_u = |x: f64, y: f64| {
        ((PI * x).sin() * (PI * y).cos(), -(PI * x).cos() * (PI * y).sin())
    };
    let fx = mesh.face_cx.clone();
    let fy = mesh.face_cy.clone();
    let wall_u = move |c: usize| {
        let fx = fx.clone();
        let fy = fy.clone();
        move |face_idx: u32| {
            let i = face_idx as usize;
            let (ux, uy) = exact_u(fx[i], fy[i]);
            (if c == 0 { ux } else { uy }) as f32
        }
    };
    solver
        .set_boundary_values_per_face(cfd2::solver::gpu::enums::GpuBoundaryType::Wall, "U", 0, &wall_u(0))
        .unwrap();
    solver
        .set_boundary_values_per_face(cfd2::solver::gpu::enums::GpuBoundaryType::Wall, "U", 1, &wall_u(1))
        .unwrap();
    let src: Vec<(f64, f64)> = (0..mesh.num_cells())
        .map(|i| {
            let (ux, uy) = exact_u(mesh.cell_cx[i], mesh.cell_cy[i]);
            (2.0 * PI * PI * ux, 2.0 * PI * PI * uy)
        })
        .collect();
    solver
        .set_field_vec2(INCOMPRESSIBLE_MMS_SOURCE_FIELD, &src)
        .unwrap();
    solver.set_u(&vec![(0.0, 0.0); mesh.num_cells()]);
    solver.set_p(&vec![0.0; mesh.num_cells()]);
    solver.initialize_history();

    for step in 0..24 {
        let stats = solver.step_with_stats().expect("stats");
        let state = pollster::block_on(solver.read_state_f32());
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        let mut u_max = 0.0f32;
        for i in 0..mesh.num_cells() {
            let d = state[i * stride + dp_off];
            lo = lo.min(d);
            hi = hi.max(d);
            u_max = u_max.max(state[i * stride].abs()).max(state[i * stride + 1].abs());
        }
        let worst_resid = stats.iter().map(|s| s.residual).fold(0.0f32, f32::max);
        let n_div = stats.iter().filter(|s| s.diverged).count();
        let last = stats.last().expect("solve stats");
        println!(
            "[n16-probe] cap={lin_cap} default_pc={use_default_precond} step {step}: d_p [{lo:.4e},{hi:.4e}] |u|max={u_max:.4e} lin(last resid={:.3e} worst resid={worst_resid:.3e} diverged_outers={n_div}/{})",
            last.residual,
            stats.len(),
        );
    }
    }
}
