use cfd2::solver::gpu::enums::GpuLowMachPrecondModel;
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType, Mesh};
use cfd2::solver::model::{compressible_model, compressible_model_with_eos};
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::model::helpers::{
    SolverCompressibleIdealGasExt, SolverCompressibleInletExt, SolverFieldAliasesExt,
    SolverRuntimeParamsExt,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{
    OuterStepStatus, PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver,
};

fn local_face_metric_scale(mesh: &Mesh, cell: usize) -> f64 {
    let start = mesh.cell_face_offsets[cell];
    let end = mesh.cell_face_offsets[cell + 1];
    let perimeter_sum = (start..end)
        .map(|idx| mesh.face_area[mesh.cell_faces[idx]])
        .sum::<f64>();
    ((perimeter_sum * perimeter_sum) / (16.0 * mesh.cell_vol[cell].max(1e-30))).max(1.0)
}

#[test]
fn compressible_dual_time_preserves_uniform_state() {
    std::env::set_var("CFD2_QUIET", "1");

    let mesh = generate_structured_rect_mesh(
        4,
        2,
        1.0,
        0.5,
        BoundarySides {
            left: BoundaryType::SlipWall,
            right: BoundaryType::SlipWall,
            bottom: BoundaryType::SlipWall,
            top: BoundaryType::SlipWall,
        },
    );

    let mut solver = pollster::block_on(UnifiedSolver::new(
        &mesh,
        compressible_model().expect("model"),
        SolverConfig {
            advection_scheme: Scheme::Upwind,
            time_scheme: TimeScheme::Euler,
            preconditioner: PreconditionerType::Jacobi,
            stepping: SteppingMode::Implicit { outer_iters: 1 },
        },
        None,
        None,
    ))
    .expect("solver init");

    solver.set_dt(0.01);
    solver.set_dtau(1e-3).expect("dtau");
    solver.set_viscosity(0.0).expect("viscosity");
    solver.set_uniform_state(1.0, [0.0, 0.0], 1.0);
    solver.initialize_history();

    let rho0 = pollster::block_on(solver.get_rho());
    let u0 = pollster::block_on(solver.get_u());
    let p0 = pollster::block_on(solver.get_p());

    for _ in 0..3 {
        solver.step();
    }

    let rho = pollster::block_on(solver.get_rho());
    let u = pollster::block_on(solver.get_u());
    let p = pollster::block_on(solver.get_p());

    let tol_rho = 1e-5;
    let tol_u = 1e-5;
    let tol_p = 1e-5;

    for (a, b) in rho.iter().zip(rho0.iter()) {
        assert!(
            (a - b).abs() < tol_rho,
            "rho drifted (a={a:.6e}, b={b:.6e})"
        );
    }
    for ((ax, ay), (bx, by)) in u.iter().zip(u0.iter()) {
        assert!(
            (ax - bx).abs() < tol_u && (ay - by).abs() < tol_u,
            "u drifted (a=({ax:.6e},{ay:.6e}), b=({bx:.6e},{by:.6e}))"
        );
    }
    for (a, b) in p.iter().zip(p0.iter()) {
        assert!((a - b).abs() < tol_p, "p drifted (a={a:.6e}, b={b:.6e})");
    }
}

#[test]
fn compressible_dual_time_structured_channel_keeps_unity_local_scale() {
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

    for cell in 0..mesh.num_cells() {
        let scale = local_face_metric_scale(&mesh, cell);
        assert!(
            (scale - 1.0).abs() <= 1e-9,
            "expected structured square cell {cell} to keep unit local dtau scale, got {scale:.12e}"
        );
    }

    let eos = EosSpec::IdealGas {
        gamma: 1.4,
        gas_constant: 287.0,
        temperature: 300.0,
    };
    let density = 1.225f32;
    let inlet_u = 0.2f32;
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

    let linear_stats = solver.step_with_stats().expect("step with stats");
    let step_stats = solver.step_stats();
    let last = linear_stats.last().copied().unwrap_or_default();

    eprintln!(
        "[structured_local_dtau] attempts={:?} retries={:?} status={:?} dt={:?} dtau={:?}",
        step_stats.step_attempt_count,
        step_stats.rejected_retry_count,
        step_stats.outer_step_status,
        step_stats.current_dt,
        step_stats.current_dtau
    );

    assert!(
        last.residual.is_finite() && last.residual < 1e12,
        "structured dual-time linear residual blew up: {last:?}"
    );
    assert!(
        !last.diverged,
        "structured dual-time linear solve diverged: {last:?}"
    );
    assert!(
        matches!(
            step_stats.outer_step_status,
            Some(OuterStepStatus::AcceptedConverged)
                | Some(OuterStepStatus::AcceptedNonconverged)
        ),
        "expected explicit pseudo-time acceptance status on structured mesh, got {:?}",
        step_stats.outer_step_status
    );
    assert!(
        matches!(step_stats.step_attempt_count, Some(1 | 2)),
        "expected structured-grid regression to finish in one accepted attempt or one rejected-retry plus acceptance, got {:?}",
        step_stats.step_attempt_count
    );
    assert!(
        matches!(step_stats.rejected_retry_count, None | Some(0 | 1)),
        "expected structured-grid regression to record at most one rejected retry, got {:?}",
        step_stats.rejected_retry_count
    );
    assert!(
        matches!(step_stats.current_dt, Some(current_dt) if current_dt.is_finite() && current_dt > 0.0 && current_dt <= dt + 1e-9),
        "expected structured-grid regression to keep dt finite and bounded, got {:?}",
        step_stats.current_dt
    );
    assert!(
        matches!(step_stats.current_dtau, Some(current_dtau) if current_dtau.is_finite() && current_dtau > 0.0 && current_dtau <= dtau + 1e-12),
        "expected structured-grid regression to keep dtau finite and bounded, got {:?}",
        step_stats.current_dtau
    );
    assert!(
        matches!(step_stats.positivity_rho_undershoot_count, Some(0)),
        "expected no density positivity undershoots on structured local-dtau regression, got {:?}",
        step_stats.positivity_rho_undershoot_count
    );
    assert!(
        matches!(step_stats.positivity_pressure_undershoot_count, Some(0)),
        "expected no pressure positivity undershoots on structured local-dtau regression, got {:?}",
        step_stats.positivity_pressure_undershoot_count
    );
    assert!(
        matches!(step_stats.positivity_min_rho, Some(value) if value.is_finite() && value > 0.0),
        "expected positive rho minimum on structured local-dtau regression, got {:?}",
        step_stats.positivity_min_rho
    );
    assert!(
        matches!(step_stats.positivity_min_p, Some(value) if value.is_finite() && value > 0.0),
        "expected positive pressure minimum on structured local-dtau regression, got {:?}",
        step_stats.positivity_min_p
    );

    let u = pollster::block_on(solver.get_u());
    let max_u = u
        .iter()
        .map(|(ux, uy)| (ux * ux + uy * uy).sqrt())
        .fold(0.0f64, f64::max);
    assert!(
        max_u.is_finite() && max_u <= inlet_u as f64 * 1.5,
        "expected structured-grid regression to remain bounded after one dual-time step, got |u|max={max_u:.6e} for inlet {inlet_u:.6e}"
    );
}
