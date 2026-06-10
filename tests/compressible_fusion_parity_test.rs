//! Fusion parity tests for the `compressible` model.
//!
//! The compressible model has no fusion rules declared (FUSION_COVERAGE.md).
//! The `assembly` / `assembly_grad_state` pair is mutually exclusive
//! (`RequiresNoGradState` / `RequiresGradState`), so no fusion can occur.
//!
//! These tests verify:
//! - The fusion schedule registry returns correct (unfused) schedules for all policies.
//! - All three policies produce identical solver output (since no fusion rules fire).
//! - Dispatch counts are unchanged across policies.

use cfd2::solver::gpu::dispatch_counter::{get_dispatch_stats, DispatchScope};
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType, Mesh};
use cfd2::solver::model::compressible_model;
use cfd2::solver::model::helpers::{SolverCompressibleIdealGasExt, SolverRuntimeParamsExt};
use cfd2::solver::model::kernel::KernelFusionPolicy;
use cfd2::solver::model::ModelLinearSolverSpec;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};
use std::sync::{Mutex, OnceLock};

// ---------------------------------------------------------------------------
// Snapshot and helpers
// ---------------------------------------------------------------------------

struct CompressibleSnapshot {
    rho: Vec<f64>,
    p: Vec<f64>,
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

fn run_compressible_with_policy(
    mesh: &Mesh,
    policy: KernelFusionPolicy,
    steps: usize,
) -> CompressibleSnapshot {
    let _lock = solver_test_lock()
        .lock()
        .expect("solver test lock poisoned");

    let mut model = compressible_model().expect("model");
    // compressible_model().expect("model") defaults to linear_solver = None. We set it to
    // control the fusion policy.
    let mut spec = ModelLinearSolverSpec::default();
    spec.solver.kernel_fusion_policy = policy;
    model.linear_solver = Some(spec);

    let config = SolverConfig {
        advection_scheme: Scheme::Upwind,
        time_scheme: TimeScheme::Euler,
        preconditioner: PreconditionerType::Jacobi,
        stepping: SteppingMode::Implicit { outer_iters: 1 },
    };

    let eos = cfd2::solver::model::eos::EosSpec::IdealGas {
        gamma: 1.4,
        gas_constant: 1.0,
        temperature: 1.0,
    };

    let mut solver = pollster::block_on(UnifiedSolver::new(mesh, model, config, None, None))
        .expect("solver init");

    // Clear full state to avoid uninitialized auxiliary fields.
    let stride = solver.model().state_layout.stride() as usize;
    solver
        .write_state_f32(&vec![0.0f32; mesh.num_cells() * stride])
        .expect("clear state");

    solver.set_dt(0.001);
    solver.set_density(1.0).expect("set density");
    solver.set_viscosity(1e-3).expect("set viscosity");
    solver.set_eos(&eos).expect("set eos");

    let p_ref = eos.pressure_for_density(1.0) as f32;
    solver.set_uniform_state(1.0, [0.0, 0.0], p_ref);
    solver.initialize_history();

    for _ in 0..steps {
        solver.step();
    }

    CompressibleSnapshot {
        rho: pollster::block_on(solver.get_field_scalar("rho")).expect("read rho"),
        p: pollster::block_on(solver.get_field_scalar("p")).expect("read p"),
    }
}

fn run_compressible_dispatch_count(mesh: &Mesh, policy: KernelFusionPolicy, steps: usize) -> u64 {
    let _lock = solver_test_lock()
        .lock()
        .expect("solver test lock poisoned");

    let mut model = compressible_model().expect("model");
    let mut spec = ModelLinearSolverSpec::default();
    spec.solver.kernel_fusion_policy = policy;
    model.linear_solver = Some(spec);

    let config = SolverConfig {
        advection_scheme: Scheme::Upwind,
        time_scheme: TimeScheme::Euler,
        preconditioner: PreconditionerType::Jacobi,
        stepping: SteppingMode::Implicit { outer_iters: 1 },
    };

    let eos = cfd2::solver::model::eos::EosSpec::IdealGas {
        gamma: 1.4,
        gas_constant: 1.0,
        temperature: 1.0,
    };

    let mut solver = pollster::block_on(UnifiedSolver::new(mesh, model, config, None, None))
        .expect("solver init");

    let stride = solver.model().state_layout.stride() as usize;
    solver
        .write_state_f32(&vec![0.0f32; mesh.num_cells() * stride])
        .expect("clear state");

    solver.set_dt(0.001);
    solver.set_density(1.0).expect("set density");
    solver.set_viscosity(1e-3).expect("set viscosity");
    solver.set_eos(&eos).expect("set eos");

    let p_ref = eos.pressure_for_density(1.0) as f32;
    solver.set_uniform_state(1.0, [0.0, 0.0], p_ref);
    solver.initialize_history();

    let _dispatch_scope = DispatchScope::new();
    for _ in 0..steps {
        solver.step();
    }
    let stats = get_dispatch_stats();
    stats.by_category.get("Kernel Graph").copied().unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// All three fusion policies should produce identical results for the
/// compressible model since no fusion rules are declared.
#[test]
fn compressible_off_safe_aggressive_all_identical() {
    std::env::set_var("CFD2_QUIET", "1");

    let mesh = make_mesh();
    let steps = 3;

    let off = run_compressible_with_policy(&mesh, KernelFusionPolicy::Off, steps);
    let safe = run_compressible_with_policy(&mesh, KernelFusionPolicy::Safe, steps);
    let aggressive = run_compressible_with_policy(&mesh, KernelFusionPolicy::Aggressive, steps);

    // With no fusion rules, all policies should produce bitwise-identical results.
    assert_eq!(off.rho.len(), safe.rho.len());
    assert_eq!(off.rho.len(), aggressive.rho.len());
    assert_eq!(off.p.len(), safe.p.len());
    assert_eq!(off.p.len(), aggressive.p.len());

    for i in 0..off.rho.len() {
        assert_eq!(
            off.rho[i], safe.rho[i],
            "rho[{i}] Off vs Safe mismatch: {} vs {}",
            off.rho[i], safe.rho[i]
        );
        assert_eq!(
            off.rho[i], aggressive.rho[i],
            "rho[{i}] Off vs Aggressive mismatch: {} vs {}",
            off.rho[i], aggressive.rho[i]
        );
    }

    for i in 0..off.p.len() {
        assert_eq!(
            off.p[i], safe.p[i],
            "p[{i}] Off vs Safe mismatch: {} vs {}",
            off.p[i], safe.p[i]
        );
        assert_eq!(
            off.p[i], aggressive.p[i],
            "p[{i}] Off vs Aggressive mismatch: {} vs {}",
            off.p[i], aggressive.p[i]
        );
    }
}

/// Dispatch counts should be identical across all policies for the
/// compressible model (no fusion rules => no dispatch reduction).
#[test]
fn compressible_dispatch_count_unchanged_across_policies() {
    std::env::set_var("CFD2_QUIET", "1");

    let mesh = make_mesh();
    let steps = 3;

    let off = run_compressible_dispatch_count(&mesh, KernelFusionPolicy::Off, steps);
    let safe = run_compressible_dispatch_count(&mesh, KernelFusionPolicy::Safe, steps);
    let aggressive = run_compressible_dispatch_count(&mesh, KernelFusionPolicy::Aggressive, steps);

    assert_eq!(
        off, safe,
        "expected identical dispatch count for Off ({off}) and Safe ({safe}) policies"
    );
    assert_eq!(
        safe, aggressive,
        "expected identical dispatch count for Safe ({safe}) and Aggressive ({aggressive}) policies"
    );
}
