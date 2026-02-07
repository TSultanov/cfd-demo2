use cfd2::solver::gpu::dispatch_counter::{get_dispatch_stats, DispatchScope};
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

fn run_with_policy_kernel_graph_dispatches(
    mesh: &Mesh,
    policy: KernelFusionPolicy,
    steps: usize,
    outer_iters: usize,
) -> u64 {
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
    solver
        .set_outer_iters(outer_iters)
        .expect("set outer iters");
    solver
        .set_outer_tolerance(0.0)
        .expect("set outer relative tolerance");
    solver
        .set_outer_tolerance_abs(0.0)
        .expect("set outer absolute tolerance");

    solver.set_u(&vec![(0.0, 0.0); mesh.num_cells()]);
    solver.set_p(&vec![0.0; mesh.num_cells()]);
    solver.initialize_history();

    let _dispatch_scope = DispatchScope::new();
    for _ in 0..steps {
        solver.step();
    }
    let stats = get_dispatch_stats();
    stats.by_category.get("Kernel Graph").copied().unwrap_or(0)
}

fn assert_snapshots_match(
    lhs_name: &str,
    lhs: &RhieChowSnapshot,
    rhs_name: &str,
    rhs: &RhieChowSnapshot,
    rel_tol: f64,
) {
    assert_eq!(lhs.u.len(), rhs.u.len());
    assert_eq!(lhs.p.len(), rhs.p.len());
    assert_eq!(lhs.d_p.len(), rhs.d_p.len());
    assert_eq!(lhs.grad_p_old.len(), rhs.grad_p_old.len());

    let mut max_u_abs = 0.0f64;
    let mut max_u_scale = 0.0f64;
    for ((lhs_u_x, lhs_u_y), (rhs_u_x, rhs_u_y)) in lhs.u.iter().zip(rhs.u.iter()) {
        max_u_abs = max_u_abs.max((rhs_u_x - lhs_u_x).abs());
        max_u_abs = max_u_abs.max((rhs_u_y - lhs_u_y).abs());

        max_u_scale = max_u_scale.max(lhs_u_x.abs());
        max_u_scale = max_u_scale.max(lhs_u_y.abs());
        max_u_scale = max_u_scale.max(rhs_u_x.abs());
        max_u_scale = max_u_scale.max(rhs_u_y.abs());
    }
    let max_u_rel = max_u_abs / max_u_scale.max(1e-9);

    let lhs_p_mean = mean(&lhs.p);
    let rhs_p_mean = mean(&rhs.p);
    let mut max_p_abs = 0.0f64;
    let mut max_p_scale = 0.0f64;
    for (lhs_p_raw, rhs_p_raw) in lhs.p.iter().zip(rhs.p.iter()) {
        let lhs_p = lhs_p_raw - lhs_p_mean;
        let rhs_p = rhs_p_raw - rhs_p_mean;

        max_p_abs = max_p_abs.max((rhs_p - lhs_p).abs());
        max_p_scale = max_p_scale.max(lhs_p.abs());
        max_p_scale = max_p_scale.max(rhs_p.abs());
    }
    let max_p_rel = max_p_abs / max_p_scale.max(1e-9);

    let mut max_dp_abs = 0.0f64;
    let mut max_dp_scale = 0.0f64;
    for (lhs_d_p, rhs_d_p) in lhs.d_p.iter().zip(rhs.d_p.iter()) {
        max_dp_abs = max_dp_abs.max((rhs_d_p - lhs_d_p).abs());
        max_dp_scale = max_dp_scale.max(lhs_d_p.abs());
        max_dp_scale = max_dp_scale.max(rhs_d_p.abs());
    }
    let max_dp_rel = max_dp_abs / max_dp_scale.max(1e-9);

    let mut max_grad_old_abs = 0.0f64;
    let mut max_grad_old_scale = 0.0f64;
    for ((lhs_x, lhs_y), (rhs_x, rhs_y)) in lhs.grad_p_old.iter().zip(rhs.grad_p_old.iter()) {
        max_grad_old_abs = max_grad_old_abs.max((rhs_x - lhs_x).abs());
        max_grad_old_abs = max_grad_old_abs.max((rhs_y - lhs_y).abs());

        max_grad_old_scale = max_grad_old_scale.max(lhs_x.abs());
        max_grad_old_scale = max_grad_old_scale.max(lhs_y.abs());
        max_grad_old_scale = max_grad_old_scale.max(rhs_x.abs());
        max_grad_old_scale = max_grad_old_scale.max(rhs_y.abs());
    }
    let max_grad_old_rel = max_grad_old_abs / max_grad_old_scale.max(1e-9);

    assert!(
        max_u_rel <= rel_tol,
        "{lhs_name} vs {rhs_name}: u mismatch too large: max_abs={max_u_abs:.6e} max_rel={max_u_rel:.6e} (tol={rel_tol:.6e})"
    );
    assert!(
        max_p_rel <= rel_tol,
        "{lhs_name} vs {rhs_name}: mean-free p mismatch too large: max_abs={max_p_abs:.6e} max_rel={max_p_rel:.6e} (tol={rel_tol:.6e})"
    );
    assert!(
        max_dp_rel <= rel_tol,
        "{lhs_name} vs {rhs_name}: d_p mismatch too large: max_abs={max_dp_abs:.6e} max_rel={max_dp_rel:.6e} (tol={rel_tol:.6e})"
    );
    assert!(
        max_grad_old_rel <= rel_tol,
        "{lhs_name} vs {rhs_name}: grad_p_old mismatch too large: max_abs={max_grad_old_abs:.6e} max_rel={max_grad_old_rel:.6e} (tol={rel_tol:.6e})"
    );
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
    // Tolerance accounts for floating-point reordering in fused vs unfused kernels.
    let rel_tol = 1e-3f64;
    assert_snapshots_match("off", &off, "safe", &safe, rel_tol);
}

#[test]
fn rhie_chow_aggressive_matches_safe_within_tolerance() {
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

    let safe = run_with_policy(&mesh, KernelFusionPolicy::Safe);
    let aggressive = run_with_policy(&mesh, KernelFusionPolicy::Aggressive);

    let rel_tol = 1e-3f64;
    assert_snapshots_match("safe", &safe, "aggressive", &aggressive, rel_tol);
}

#[test]
fn rhie_chow_aggressive_reduces_kernel_graph_dispatches_vs_safe() {
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

    let steps = 3usize;
    let outer_iters = 4usize;
    let safe = run_with_policy_kernel_graph_dispatches(
        &mesh,
        KernelFusionPolicy::Safe,
        steps,
        outer_iters,
    );
    let aggressive = run_with_policy_kernel_graph_dispatches(
        &mesh,
        KernelFusionPolicy::Aggressive,
        steps,
        outer_iters,
    );

    // Compile-time schedule tests enforce the exact per-iteration dispatch delta.
    // At runtime, adaptive solver behavior can vary graph execution counts, so
    // assert only that aggressive remains strictly lower-dispatch than safe.
    let actual_drop = safe.saturating_sub(aggressive);
    assert!(
        actual_drop > 0,
        "expected aggressive policy to reduce kernel-graph dispatches (safe={safe}, aggressive={aggressive})"
    );
}
