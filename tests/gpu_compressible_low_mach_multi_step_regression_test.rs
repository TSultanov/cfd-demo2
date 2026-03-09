#![cfg(all(feature = "meshgen", feature = "dev-tests"))]

use cfd2::solver::gpu::enums::GpuLowMachPrecondModel;
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType};
use cfd2::solver::model::compressible_model_with_eos;
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::model::helpers::{
    SolverCompressibleIdealGasExt, SolverCompressibleInletExt, SolverFieldAliasesExt,
    SolverRuntimeParamsExt,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{
    OuterStepStatus, PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver,
};

/// Multi-step low-Mach physical regression: verifies that a compressible
/// dual-time solver with WeissSmith preconditioning progresses physically
/// over several steps on a structured channel, starting from a zero-velocity
/// interior field.
#[test]
fn compressible_low_mach_multi_step_progresses_physically() {
    std::env::set_var("CFD2_QUIET", "1");

    let mesh = generate_structured_rect_mesh(
        8,
        4,
        2.0,
        1.0,
        BoundarySides {
            left: BoundaryType::Inlet,
            right: BoundaryType::Outlet,
            bottom: BoundaryType::Wall,
            top: BoundaryType::Wall,
        },
    );

    let eos = EosSpec::IdealGas {
        gamma: 1.4,
        gas_constant: 287.0,
        temperature: 300.0,
    };
    let density = 1.225f32;
    let inlet_u = 0.1f32; // Mach ~ 0.0003
    let dt = 5e-4f32;
    let dtau = 1e-5f32;
    let viscosity = 1.81e-5f32;
    let p_ref = eos.pressure_for_density(density as f64) as f32;

    let mut solver = pollster::block_on(UnifiedSolver::new(
        &mesh,
        compressible_model_with_eos(eos).expect("model"),
        SolverConfig {
            advection_scheme: Scheme::SecondOrderUpwind,
            time_scheme: TimeScheme::BDF2,
            preconditioner: PreconditionerType::Jacobi,
            stepping: SteppingMode::Implicit { outer_iters: 1 },
        },
        None,
        None,
    ))
    .expect("solver init");

    solver.set_dt(dt);
    solver.set_dtau(dtau).expect("dtau");
    solver.set_density(density).expect("density");
    solver.set_viscosity(viscosity).expect("viscosity");
    solver.set_eos(&eos).expect("eos");
    solver
        .set_precond_model(GpuLowMachPrecondModel::WeissSmith)
        .expect("precond model");
    solver
        .set_precond_theta_floor(1e-6)
        .expect("theta floor");
    solver.set_outer_iters(8).expect("outer iters");
    solver.set_collect_convergence_stats(true);
    solver
        .set_compressible_inlet_isothermal_x(density, inlet_u, &eos)
        .expect("inlet");
    solver.set_uniform_state(density, [0.0, 0.0], p_ref);
    solver.initialize_history();

    let num_steps = 5;

    for step in 0..num_steps {
        let linear_stats = solver.step_with_stats().expect("step with stats");
        let ss = solver.step_stats();
        let last = linear_stats.last().copied().unwrap_or_default();

        eprintln!(
            "[low_mach_multi_step] step={step} attempts={:?} retries={:?} status={:?} dt={:?} dtau={:?} res={:.3e}",
            ss.step_attempt_count, ss.rejected_retry_count, ss.outer_step_status,
            ss.current_dt, ss.current_dtau, last.residual
        );

        // Per-step: outer loop must accept (converged or nonconverged).
        assert!(
            matches!(
                ss.outer_step_status,
                Some(OuterStepStatus::AcceptedConverged)
                    | Some(OuterStepStatus::AcceptedNonconverged)
            ),
            "step {step}: expected accepted status, got {:?}",
            ss.outer_step_status
        );

        // Per-step: linear residual must stay finite.
        assert!(
            last.residual.is_finite(),
            "step {step}: linear residual is not finite: {last:?}"
        );

        // Per-step: no positivity undershoots.
        assert!(
            matches!(ss.positivity_rho_undershoot_count, Some(0) | None),
            "step {step}: density undershoot count = {:?}",
            ss.positivity_rho_undershoot_count
        );
        assert!(
            matches!(ss.positivity_pressure_undershoot_count, Some(0) | None),
            "step {step}: pressure undershoot count = {:?}",
            ss.positivity_pressure_undershoot_count
        );

        // Per-step: velocity bounded at 2× inlet.
        let u = pollster::block_on(solver.get_u());
        let max_u = u
            .iter()
            .map(|(ux, uy)| (ux * ux + uy * uy).sqrt())
            .fold(0.0f64, f64::max);
        assert!(
            max_u.is_finite() && max_u <= inlet_u as f64 * 2.0,
            "step {step}: |u|max={max_u:.6e} exceeds 2× inlet={inlet_u:.6e}"
        );

        // Per-step: pressure stays positive.
        let p = pollster::block_on(solver.get_p());
        let p_min = p.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(
            p_min > 0.0,
            "step {step}: minimum pressure={p_min:.6e} is not positive"
        );

        // After step 2+: velocity field should be nonzero (solution is progressing).
        if step >= 2 {
            assert!(
                max_u > 1e-10,
                "step {step}: velocity field is still zero (max_u={max_u:.6e}), solution not progressing"
            );
        }
    }

    // Final check: verify physical progression — solution should have evolved
    // from the zero-velocity initial condition. The velocity field should show
    // nonzero flow driven by the inlet.
    let u_final = pollster::block_on(solver.get_u());
    let max_u_final = u_final
        .iter()
        .map(|(ux, uy)| (ux * ux + uy * uy).sqrt())
        .fold(0.0f64, f64::max);
    eprintln!(
        "[low_mach_multi_step] final: max|u|={max_u_final:.6e} inlet={inlet_u:.6e}"
    );
    assert!(
        max_u_final > 1e-6,
        "after {num_steps} steps, max velocity={max_u_final:.6e} is too small — solution not progressing"
    );
}
