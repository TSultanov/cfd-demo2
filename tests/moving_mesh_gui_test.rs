//! Headless GUI ↔ moving-mesh (ALE) worker seam test — the Stage-1 gate.
//!
//! There is no display in CI, so this exercises the *verifiable* half of the GUI
//! moving-mesh hookup, in the spirit of `cpu_gui_parity.rs` (which drives the
//! wrapped solver, not a window):
//!
//!  1. **Driver payload** — build a `MovingMeshDriver` exactly as the GUI init
//!     path does (`generate_cvt_mesh_with_seeds` → `MovingMeshDriver::build` on
//!     the incompressible ALE model, CPU backend) and step it N times, asserting
//!     it steps without error, the cell count stays fixed (fixed-seed v1), and
//!     the per-step `MovingMeshStats` are finite — the exact `(StepOutcome,
//!     MovingMeshStats)` the worker's `SolverMode::MovingMesh` arm consumes.
//!
//!  2. **Worker message path** — hand a fresh `MovingMeshDriver` to the *real*
//!     private solver worker via `moving_mesh_worker_smoke` (which sends
//!     `SetSolver { MovingMesh }` + `SetRunning(true)` and collects the
//!     `MeshRefreshed` events over the mpsc channel), asserting the worker steps,
//!     emits `MeshRefreshed` events with non-empty re-tessellation cells + finite
//!     stats, and the cell count never changes.
//!
//! No window is created; nothing here claims visual / interactive verification.
#![cfg(feature = "ui")]

use cfd2::solver::mesh::{BackwardsStep, LloydConfig};
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{GpuLowMachPrecondModel, PreconditionerType, TimeScheme};
use cfd2::sim::{MeshMotionSpec, MovingMeshDriver, RuntimeParams};
use cfd2::ui::app::moving_mesh_worker_smoke;
use nalgebra::Vector2;

/// Fixed-dt incompressible params for the ALE model (adaptive_dt MUST be off —
/// the swept mesh fluxes are SCL-closed against a fixed dt).
fn ale_params() -> RuntimeParams {
    RuntimeParams {
        adaptive_dt: false,
        target_cfl: 0.9,
        requested_dt: 0.01,
        dtau: 0.0,
        log_convergence: false,
        log_every_steps: 50,
        advection_scheme: Scheme::Upwind,
        time_scheme: TimeScheme::BDF2,
        preconditioner: PreconditionerType::Jacobi,
        outer_iters: 4,
        outer_auto_converge: false,
        low_mach_model: GpuLowMachPrecondModel::Off,
        low_mach_theta_floor: 1e-6,
        low_mach_pressure_coupling_alpha: 1.0,
        alpha_u: 0.7,
        alpha_p: 0.3,
        inlet_velocity: 1.0,
        density: 1.0,
        viscosity: 0.01,
        eos: EosSpec::Constant,
        compressibility_psi: 0.0,
        outlet_back_pressure: 0.0,
        pressure_inlet: false,
        inlet_pressure: 0.0,
    }
}

/// Build a moving-mesh driver on a coarse CVT backstep mesh (fast), the same way
/// the GUI's `build_moving_init` does.
fn build_driver(motion: MeshMotionSpec) -> (MovingMeshDriver, usize) {
    let domain = Vector2::new(3.5, 1.0);
    let geo = BackwardsStep {
        length: 3.5,
        height_inlet: 0.5,
        height_outlet: 1.0,
        step_x: 0.5,
    };
    let cvt = cfd2::meshgen::meshless::generate_cvt_mesh_with_seeds(
        &geo,
        0.09,
        0.09,
        1.0,
        domain,
        &LloydConfig::default(),
    );
    let n_cells = cvt.mesh.num_cells();
    assert!(n_cells > 20, "expected a non-trivial CVT mesh, got {n_cells} cells");
    let initial_u = vec![(0.0, 0.0); n_cells];
    let initial_p = vec![0.0; n_cells];
    let params = ale_params();
    let mut driver = pollster::block_on(MovingMeshDriver::build(
        cvt,
        &params,
        motion,
        &initial_u,
        &initial_p,
        None,
        None,
    ))
    .expect("MovingMeshDriver::build (CPU ALE) must succeed");
    // Phase-2 knobs (outer_iters / relaxation), exactly as the worker's SetSolver.
    driver.driver_mut().apply_params(&params);
    (driver, n_cells)
}

/// Part 1: step the driver directly and assert the per-step ALE telemetry the
/// worker publishes is well-formed and the cell count is fixed.
fn driver_steps_produce_finite_stats(motion: MeshMotionSpec, steps: usize) {
    let (mut driver, n_cells) = build_driver(motion);
    for s in 0..steps {
        let (_outcome, stats) = driver
            .step(true)
            .unwrap_or_else(|e| panic!("moving step {s} failed: {e}"));
        assert_eq!(
            stats.n_cells, n_cells,
            "cell count changed at step {s} (fixed-seed v1 must not add/remove cells)"
        );
        assert_eq!(driver.mesh().num_cells(), n_cells, "mesh cell count drifted at step {s}");
        assert!(
            stats.dt.is_finite()
                && stats.max_skew.is_finite()
                && stats.scl_defect.is_finite()
                && stats.identity_err.is_finite()
                && stats.plan_ms.is_finite()
                && stats.regen_ms.is_finite()
                && stats.swept_ms.is_finite()
                && stats.refresh_ms.is_finite(),
            "non-finite MovingMeshStats at step {s}: {stats:?}"
        );
        assert!(stats.dt > 0.0, "pinned dt must be positive at step {s}, got {}", stats.dt);
    }
    println!(
        "[moving-gui][driver] {steps} steps ok, {n_cells} cells fixed, stats finite"
    );
}

/// One env-using test (CFD2_BACKEND is process-global): the ALE driver payload
/// and the real worker message path, both on the CPU backend.
#[test]
fn moving_mesh_gui_worker_and_driver_smoke() {
    std::env::set_var("CFD2_BACKEND", "cpu");
    std::env::set_var("CFD2_CPU_ENGINE", "interpreter");

    // Part 1: the do-no-harm Frozen anchor + the flow-coupled path both step
    // cleanly and report finite, fixed-cell telemetry.
    driver_steps_produce_finite_stats(MeshMotionSpec::Frozen, 6);
    driver_steps_produce_finite_stats(MeshMotionSpec::FlowCoupled { regularization: 0.5 }, 6);

    // Part 2: drive the actual private solver worker through the moving-mesh
    // message path and observe the MeshRefreshed events it emits.
    let (driver, n_cells) = build_driver(MeshMotionSpec::Frozen);
    let smoke = moving_mesh_worker_smoke(driver, 400);
    println!(
        "[moving-gui][worker] refreshes={} cells=[{:?},{:?}] empty={} nonfinite={} err={:?}",
        smoke.mesh_refresh_events,
        smoke.min_cells,
        smoke.max_cells,
        smoke.saw_empty_cells,
        smoke.saw_nonfinite_stats,
        smoke.error,
    );
    assert!(smoke.error.is_none(), "worker reported an error: {:?}", smoke.error);
    assert!(
        smoke.mesh_refresh_events >= 3,
        "worker did not emit MeshRefreshed events via the message path (got {})",
        smoke.mesh_refresh_events
    );
    assert!(!smoke.saw_empty_cells, "a MeshRefreshed carried empty/degenerate cells");
    assert!(!smoke.saw_nonfinite_stats, "a MeshRefreshed carried non-finite stats");
    assert_eq!(
        (smoke.min_cells, smoke.max_cells),
        (Some(n_cells), Some(n_cells)),
        "worker mesh cell count must stay fixed at {n_cells}"
    );

    std::env::remove_var("CFD2_BACKEND");
    std::env::remove_var("CFD2_CPU_ENGINE");
}
