//! Diagnostic probe for the FromAssembledDiagonal d_p formulation: step the
//! incompressible model at momentum-MMS-like settings and dump the actual
//! d_p state values against the candidate scales (closed form, V/a_P with
//! physical assembly scaling, raw cell volume = normalized-matrix
//! hypothesis), plus per-step |u| growth and linear-solve stats over an
//! (outer_iters, alpha_u, alpha_p) sweep. Not a correctness gate.
//!
//! FINDINGS (June 2026, with the incompressible model flipped to
//! FromAssembledDiagonal{include_relaxation: false, theta: 0.5}):
//! - The kernel is CORRECT: d_p settles at ~3.5e-3, matching the physical
//!   interior estimate V/(rho*V/dt + 4*mu) = 3.6e-3 (not the closed form
//!   3.5e-2, not raw volume 1.6e-2 — no in-place matrix normalization).
//! - The OUTER LOOP AMPLIFIES anyway: |u|max grows x3.5/step at
//!   outer_iters=5 in a closed box with no forcing (0.055 -> 0.21 -> 0.75
//!   -> 2.3), NaN by step 2 at outer_iters=25. alpha pairings 0.7/0.3,
//!   1.0/1.0, 1.0/0.3 all amplify; theta cannot help because the gain AT
//!   the settled d_p is unstable: the pressure response scales ~1/d_p while
//!   the update relaxation and the d_p-scaled velocity correction are
//!   calibrated for the closed-form scale. OpenFOAM tolerates small rAU
//!   because its segregated projection solves p exactly per corrector; the
//!   coupled path needs pressure-row equilibration (or an
//!   update/preconditioner redesign) before this formulation can ship.
//! To reproduce, flip the incompressible model to FromAssembledDiagonal
//! (see the comment at its derive_rhie_chow call) and run this probe.
#![cfg(feature = "dev-tests")]

use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType};
use cfd2::solver::model::helpers::{SolverFieldAliasesExt, SolverRuntimeParamsExt};
use cfd2::solver::model::incompressible_momentum_model;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};

#[test]
#[ignore]
fn probe_dp_diag_scale() {
    std::env::set_var("CFD2_QUIET", "1");
    let n = 8usize;
    for (outer_iters, alpha_u, alpha_p) in [
        (5usize, 0.7f32, 0.3f32),
        (25, 0.7, 0.3),
        (5, 1.0, 1.0),
        (25, 1.0, 1.0),
        (25, 1.0, 0.3),
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
        let model = incompressible_momentum_model().expect("model");
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
                "[dp-probe] outer={outer_iters} a_u={alpha_u} a_p={alpha_p} step {step}: d_p [{lo:.4e},{hi:.4e}] |u|max={u_max:.4e} lin(iters={} resid={:.3e} conv={} div={}) | closed={closed:.4e} V/a_phys={:.4e}",
                last.iterations, last.residual, last.converged, last.diverged,
                vol / a_phys,
            );
        }
    }
}
