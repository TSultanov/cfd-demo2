//! Fusion parity tests for the `generic_diffusion_demo` and
//! `generic_diffusion_demo_neumann` models.
//!
//! Neither model has fusion rules declared (FUSION_COVERAGE.md).
//! The `assembly` / `assembly_grad_state` pair has a safe-fuseable opportunity
//! identified but no rule is declared, so all policies should produce the same
//! schedule and identical solver output.
//!
//! These tests verify:
//! - All three policies produce identical solver output for `phi`.
//! - Dispatch counts are unchanged across policies.

use cfd2::solver::gpu::dispatch_counter::{get_dispatch_stats, DispatchScope};
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType, Mesh};
use cfd2::solver::model::helpers::SolverRuntimeParamsExt;
use cfd2::solver::model::kernel::KernelFusionPolicy;
use cfd2::solver::model::{
    generic_diffusion_demo_model, generic_diffusion_demo_neumann_model, ModelLinearSolverSpec,
    ModelSpec,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};
use std::sync::{Mutex, OnceLock};

// ---------------------------------------------------------------------------
// Snapshot and helpers
// ---------------------------------------------------------------------------

struct DiffusionSnapshot {
    phi: Vec<f64>,
}

fn solver_test_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn make_mesh() -> Mesh {
    generate_structured_rect_mesh(
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
    )
}

fn run_diffusion_with_policy(
    mesh: &Mesh,
    model_fn: fn() -> Result<ModelSpec, String>,
    policy: KernelFusionPolicy,
    steps: usize,
) -> DiffusionSnapshot {
    let _lock = solver_test_lock()
        .lock()
        .expect("solver test lock poisoned");

    let mut model = model_fn().expect("model");
    // generic_diffusion_demo has linear_solver = None. Set it to control fusion policy.
    let mut spec = ModelLinearSolverSpec::default();
    spec.solver.kernel_fusion_policy = policy;
    model.linear_solver = Some(spec);

    let config = SolverConfig {
        advection_scheme: Scheme::Upwind,
        time_scheme: TimeScheme::BDF2,
        preconditioner: PreconditionerType::Jacobi,
        stepping: SteppingMode::Coupled,
    };

    let mut solver = pollster::block_on(UnifiedSolver::new(mesh, model, config, None, None))
        .expect("solver init");

    solver.set_dt(1e-2);
    solver.set_density(1.0).expect("set density");
    solver.set_viscosity(1e-3).expect("set viscosity");

    // Set initial condition: phi = sin(pi * x).
    let phi_init: Vec<f64> = mesh
        .cell_cx
        .iter()
        .map(|&x| (std::f64::consts::PI * x).sin())
        .collect();
    solver.set_field_scalar("phi", &phi_init).expect("set phi");
    solver.initialize_history();

    for _ in 0..steps {
        solver.step();
    }

    DiffusionSnapshot {
        phi: pollster::block_on(solver.get_field_scalar("phi")).expect("read phi"),
    }
}

fn run_diffusion_dispatch_count(
    mesh: &Mesh,
    model_fn: fn() -> Result<ModelSpec, String>,
    policy: KernelFusionPolicy,
    steps: usize,
) -> u64 {
    let _lock = solver_test_lock()
        .lock()
        .expect("solver test lock poisoned");

    let mut model = model_fn().expect("model");
    let mut spec = ModelLinearSolverSpec::default();
    spec.solver.kernel_fusion_policy = policy;
    model.linear_solver = Some(spec);

    let config = SolverConfig {
        advection_scheme: Scheme::Upwind,
        time_scheme: TimeScheme::BDF2,
        preconditioner: PreconditionerType::Jacobi,
        stepping: SteppingMode::Coupled,
    };

    let mut solver = pollster::block_on(UnifiedSolver::new(mesh, model, config, None, None))
        .expect("solver init");

    solver.set_dt(1e-2);
    solver.set_density(1.0).expect("set density");
    solver.set_viscosity(1e-3).expect("set viscosity");

    let phi_init: Vec<f64> = mesh
        .cell_cx
        .iter()
        .map(|&x| (std::f64::consts::PI * x).sin())
        .collect();
    solver.set_field_scalar("phi", &phi_init).expect("set phi");
    solver.initialize_history();

    let _dispatch_scope = DispatchScope::new();
    for _ in 0..steps {
        solver.step();
    }
    let stats = get_dispatch_stats();
    stats.by_category.get("Kernel Graph").copied().unwrap_or(0)
}

// ---------------------------------------------------------------------------
// generic_diffusion_demo tests
// ---------------------------------------------------------------------------

/// All three fusion policies should produce identical `phi` for
/// `generic_diffusion_demo` since no fusion rules are declared.
#[test]
fn generic_diffusion_demo_all_policies_identical() {
    std::env::set_var("CFD2_QUIET", "1");

    let mesh = make_mesh();
    let steps = 3;

    let off = run_diffusion_with_policy(
        &mesh,
        generic_diffusion_demo_model,
        KernelFusionPolicy::Off,
        steps,
    );
    let safe = run_diffusion_with_policy(
        &mesh,
        generic_diffusion_demo_model,
        KernelFusionPolicy::Safe,
        steps,
    );
    let aggressive = run_diffusion_with_policy(
        &mesh,
        generic_diffusion_demo_model,
        KernelFusionPolicy::Aggressive,
        steps,
    );

    assert_eq!(off.phi.len(), safe.phi.len());
    assert_eq!(off.phi.len(), aggressive.phi.len());

    for i in 0..off.phi.len() {
        assert_eq!(
            off.phi[i], safe.phi[i],
            "phi[{i}] Off vs Safe mismatch: {} vs {}",
            off.phi[i], safe.phi[i]
        );
        assert_eq!(
            off.phi[i], aggressive.phi[i],
            "phi[{i}] Off vs Aggressive mismatch: {} vs {}",
            off.phi[i], aggressive.phi[i]
        );
    }
}

/// Dispatch counts should be identical across all policies for
/// `generic_diffusion_demo` (no fusion rules => no dispatch reduction).
#[test]
fn generic_diffusion_demo_dispatch_count_unchanged() {
    std::env::set_var("CFD2_QUIET", "1");

    let mesh = make_mesh();
    let steps = 3;

    let off = run_diffusion_dispatch_count(
        &mesh,
        generic_diffusion_demo_model,
        KernelFusionPolicy::Off,
        steps,
    );
    let safe = run_diffusion_dispatch_count(
        &mesh,
        generic_diffusion_demo_model,
        KernelFusionPolicy::Safe,
        steps,
    );
    let aggressive = run_diffusion_dispatch_count(
        &mesh,
        generic_diffusion_demo_model,
        KernelFusionPolicy::Aggressive,
        steps,
    );

    assert_eq!(
        off, safe,
        "expected identical dispatch count for Off ({off}) and Safe ({safe}) policies"
    );
    assert_eq!(
        safe, aggressive,
        "expected identical dispatch count for Safe ({safe}) and Aggressive ({aggressive}) policies"
    );
}

// ---------------------------------------------------------------------------
// generic_diffusion_demo_neumann tests
// ---------------------------------------------------------------------------

/// All three fusion policies should produce identical `phi` for
/// `generic_diffusion_demo_neumann` since no fusion rules are declared.
#[test]
fn generic_diffusion_demo_neumann_all_policies_identical() {
    std::env::set_var("CFD2_QUIET", "1");

    let mesh = make_mesh();
    let steps = 3;

    let off = run_diffusion_with_policy(
        &mesh,
        generic_diffusion_demo_neumann_model,
        KernelFusionPolicy::Off,
        steps,
    );
    let safe = run_diffusion_with_policy(
        &mesh,
        generic_diffusion_demo_neumann_model,
        KernelFusionPolicy::Safe,
        steps,
    );
    let aggressive = run_diffusion_with_policy(
        &mesh,
        generic_diffusion_demo_neumann_model,
        KernelFusionPolicy::Aggressive,
        steps,
    );

    assert_eq!(off.phi.len(), safe.phi.len());
    assert_eq!(off.phi.len(), aggressive.phi.len());

    for i in 0..off.phi.len() {
        assert_eq!(
            off.phi[i], safe.phi[i],
            "phi[{i}] Off vs Safe (Neumann) mismatch: {} vs {}",
            off.phi[i], safe.phi[i]
        );
        assert_eq!(
            off.phi[i], aggressive.phi[i],
            "phi[{i}] Off vs Aggressive (Neumann) mismatch: {} vs {}",
            off.phi[i], aggressive.phi[i]
        );
    }
}

/// Dispatch counts should be identical across all policies for
/// `generic_diffusion_demo_neumann`.
#[test]
fn generic_diffusion_demo_neumann_dispatch_count_unchanged() {
    std::env::set_var("CFD2_QUIET", "1");

    let mesh = make_mesh();
    let steps = 3;

    let off = run_diffusion_dispatch_count(
        &mesh,
        generic_diffusion_demo_neumann_model,
        KernelFusionPolicy::Off,
        steps,
    );
    let safe = run_diffusion_dispatch_count(
        &mesh,
        generic_diffusion_demo_neumann_model,
        KernelFusionPolicy::Safe,
        steps,
    );
    let aggressive = run_diffusion_dispatch_count(
        &mesh,
        generic_diffusion_demo_neumann_model,
        KernelFusionPolicy::Aggressive,
        steps,
    );

    assert_eq!(
        off, safe,
        "expected identical dispatch count for Off ({off}) and Safe ({safe}) policies (Neumann)"
    );
    assert_eq!(
        safe, aggressive,
        "expected identical dispatch count for Safe ({safe}) and Aggressive ({aggressive}) policies (Neumann)"
    );
}
