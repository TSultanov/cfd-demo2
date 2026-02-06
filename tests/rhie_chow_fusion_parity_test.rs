use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType, Mesh};
use cfd2::solver::model::helpers::{
    SolverFieldAliasesExt, SolverInletVelocityExt, SolverRuntimeParamsExt,
};
use cfd2::solver::model::{incompressible_momentum_model, kernel::KernelFusionPolicy};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};

struct RhieChowSnapshot {
    u: Vec<(f64, f64)>,
    p: Vec<f64>,
    d_p: Vec<f64>,
    grad_p_old: Vec<(f64, f64)>,
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().copied().sum::<f64>() / values.len() as f64
}

fn run_with_policy(mesh: &Mesh, policy: KernelFusionPolicy) -> RhieChowSnapshot {
    let mut model = incompressible_momentum_model();
    let mut linear_solver = model
        .linear_solver
        .expect("incompressible model missing linear solver");
    linear_solver.solver.kernel_fusion_policy = policy;
    model.linear_solver = Some(linear_solver);

    let config = SolverConfig {
        advection_scheme: Scheme::Upwind,
        time_scheme: TimeScheme::BDF2,
        preconditioner: PreconditionerType::Jacobi,
        stepping: SteppingMode::Coupled,
    };

    let mut solver = pollster::block_on(UnifiedSolver::new(mesh, model, config, None, None))
        .expect("solver init");

    solver.set_dt(0.02);
    solver.set_dtau(0.0).expect("set dtau");
    solver.set_density(1.0).expect("set density");
    solver.set_viscosity(0.01).expect("set viscosity");
    solver.set_inlet_velocity(1.0).expect("set inlet velocity");
    solver.set_alpha_u(0.7).expect("set alpha_u");
    solver.set_alpha_p(0.3).expect("set alpha_p");
    solver.set_outer_iters(8).expect("set outer iters");

    solver.set_u(&vec![(0.0, 0.0); mesh.num_cells()]);
    solver.set_p(&vec![0.0; mesh.num_cells()]);
    solver.initialize_history();

    for _ in 0..8 {
        solver.step();
    }

    RhieChowSnapshot {
        u: pollster::block_on(solver.get_u()),
        p: pollster::block_on(solver.get_p()),
        d_p: pollster::block_on(solver.get_field_scalar("d_p"))
            .expect("read d_p from solver state"),
        grad_p_old: pollster::block_on(solver.get_field_vec2("grad_p_old"))
            .expect("read grad_p_old from solver state"),
    }
}

#[test]
fn rhie_chow_fused_safe_matches_unfused_off_within_tolerance() {
    std::env::set_var("CFD2_QUIET", "1");

    let mesh = generate_structured_rect_mesh(
        16,
        8,
        1.0,
        0.2,
        BoundarySides {
            left: BoundaryType::Inlet,
            right: BoundaryType::Outlet,
            bottom: BoundaryType::Wall,
            top: BoundaryType::Wall,
        },
    );

    let off = run_with_policy(&mesh, KernelFusionPolicy::Off);
    let safe = run_with_policy(&mesh, KernelFusionPolicy::Safe);

    assert_eq!(off.u.len(), safe.u.len());
    assert_eq!(off.p.len(), safe.p.len());
    assert_eq!(off.d_p.len(), safe.d_p.len());
    assert_eq!(off.grad_p_old.len(), safe.grad_p_old.len());

    let mut max_u_abs = 0.0f64;
    let mut max_u_scale = 0.0f64;
    for ((off_u_x, off_u_y), (safe_u_x, safe_u_y)) in off.u.iter().zip(safe.u.iter()) {
        max_u_abs = max_u_abs.max((safe_u_x - off_u_x).abs());
        max_u_abs = max_u_abs.max((safe_u_y - off_u_y).abs());

        max_u_scale = max_u_scale.max(off_u_x.abs());
        max_u_scale = max_u_scale.max(off_u_y.abs());
        max_u_scale = max_u_scale.max(safe_u_x.abs());
        max_u_scale = max_u_scale.max(safe_u_y.abs());
    }
    let max_u_rel = max_u_abs / max_u_scale.max(1e-9);

    let off_p_mean = mean(&off.p);
    let safe_p_mean = mean(&safe.p);
    let mut max_p_abs = 0.0f64;
    let mut max_p_scale = 0.0f64;
    for (off_p_raw, safe_p_raw) in off.p.iter().zip(safe.p.iter()) {
        let off_p = off_p_raw - off_p_mean;
        let safe_p = safe_p_raw - safe_p_mean;

        max_p_abs = max_p_abs.max((safe_p - off_p).abs());
        max_p_scale = max_p_scale.max(off_p.abs());
        max_p_scale = max_p_scale.max(safe_p.abs());
    }
    let max_p_rel = max_p_abs / max_p_scale.max(1e-9);

    let mut max_dp_abs = 0.0f64;
    let mut max_dp_scale = 0.0f64;
    for (off_d_p, safe_d_p) in off.d_p.iter().zip(safe.d_p.iter()) {
        max_dp_abs = max_dp_abs.max((safe_d_p - off_d_p).abs());
        max_dp_scale = max_dp_scale.max(off_d_p.abs());
        max_dp_scale = max_dp_scale.max(safe_d_p.abs());
    }
    let max_dp_rel = max_dp_abs / max_dp_scale.max(1e-9);

    let mut max_grad_old_abs = 0.0f64;
    let mut max_grad_old_scale = 0.0f64;
    for ((off_x, off_y), (safe_x, safe_y)) in off.grad_p_old.iter().zip(safe.grad_p_old.iter()) {
        max_grad_old_abs = max_grad_old_abs.max((safe_x - off_x).abs());
        max_grad_old_abs = max_grad_old_abs.max((safe_y - off_y).abs());

        max_grad_old_scale = max_grad_old_scale.max(off_x.abs());
        max_grad_old_scale = max_grad_old_scale.max(off_y.abs());
        max_grad_old_scale = max_grad_old_scale.max(safe_x.abs());
        max_grad_old_scale = max_grad_old_scale.max(safe_y.abs());
    }
    let max_grad_old_rel = max_grad_old_abs / max_grad_old_scale.max(1e-9);

    // Tolerance accounts for floating-point reordering in fused vs unfused kernels.
    let rel_tol = 1e-3f64;
    assert!(
        max_u_rel <= rel_tol,
        "u mismatch too large: max_abs={max_u_abs:.6e} max_rel={max_u_rel:.6e} (tol={rel_tol:.6e})"
    );
    assert!(
        max_p_rel <= rel_tol,
        "mean-free p mismatch too large: max_abs={max_p_abs:.6e} max_rel={max_p_rel:.6e} (tol={rel_tol:.6e})"
    );
    assert!(
        max_dp_rel <= rel_tol,
        "d_p mismatch too large: max_abs={max_dp_abs:.6e} max_rel={max_dp_rel:.6e} (tol={rel_tol:.6e})"
    );
    assert!(
        max_grad_old_rel <= rel_tol,
        "grad_p_old mismatch too large: max_abs={max_grad_old_abs:.6e} max_rel={max_grad_old_rel:.6e} (tol={rel_tol:.6e})"
    );
}
