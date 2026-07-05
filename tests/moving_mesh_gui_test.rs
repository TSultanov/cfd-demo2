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
//!
//! # Manual visual smoke test (needs a display — NOT covered here)
//!
//! Headless CI cannot open the window, so the actual mesh-advecting animation
//! must be confirmed by a human. To do so, run the GUI and:
//!
//!  1. `cargo run --release --features "cpu ui"` (a CPU backend is required —
//!     GPU moving mesh is roadmap M5 and the toggle is disabled on GPU).
//!  2. In the left panel, under **Compute backend**, pick a CPU option
//!     (Interpreter / Transpiled).
//!  3. In the **Moving Mesh (ALE)** group, tick **Enable Moving Mesh (ALE)**.
//!     This auto-steers Mesh Type → *Voronoi (CVT)*, model →
//!     *incompressible_momentum_ale*, and forces a fixed timestep.
//!  4. Leave **Seed motion** on *Flow-coupled* (the "follows the flow" default);
//!     optionally drag **Regularization χ** (0 = pure flow advection, higher =
//!     more centroid steering to hold cell quality).
//!  5. Click **Initialize / Reset**, then **Run**.
//!
//! Expected: the Voronoi cells visibly advect / distort with the flow and the
//! wireframe re-tessellates every step (no flicker, no crash, no buffer-overflow
//! validation error even as per-cell vertex counts drift). The stats panel shows
//! a live **ALE mesh / ALE / ALE time** block (cells, faces, flip counts, dt,
//! skew, SCL defect, flip defect, and the plan/regen/swept/refresh millisecond
//! split) next to the usual step-time/residual labels. **Pause** (Run toggles
//! off) freezes it; **Initialize / Reset** rebuilds from scratch. Switching the
//! compute backend to **GPU** disables the toggle and reverts to the static path.
#![cfg(feature = "ui")]

use cfd2::solver::mesh::{BackwardsStep, LloydConfig};
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{GpuLowMachPrecondModel, PreconditionerType, TimeScheme};
use cfd2::sim::{MeshMotionSpec, MovingMeshDriver, RuntimeParams};
use cfd2::ui::app::moving_mesh_worker_smoke;
use cfd2::ui::cfd_renderer;
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
/// the GUI's `build_moving_init` does. `u0` is the uniform initial cell velocity
/// — `(0,0)` is the from-rest IC; a nonzero freestream makes the FlowCoupled
/// seeds actually advect from step one (so the moving-mesh path is exercised with
/// genuine motion, not a near-frozen mesh that never develops flow in 30 steps).
fn build_driver(motion: MeshMotionSpec, u0: (f64, f64)) -> (MovingMeshDriver, usize) {
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
    let initial_u = vec![u0; n_cells];
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
    let (mut driver, n_cells) = build_driver(motion, (0.0, 0.0));
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
    let (driver, n_cells) = build_driver(MeshMotionSpec::Frozen, (0.0, 0.0));
    let smoke = moving_mesh_worker_smoke(driver, 2000, 3, false);
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

    // Part 3: the FULL flow-coupled render loop, headless. Drive the real worker
    // in FlowCoupled moving mode for ~30 regens through the message API,
    // collecting each emitted `cached_cells`, then replay the UI's per-frame
    // re-tessellation + renderer capacity path (`build_mesh_vertices` ->
    // `update_mesh`) on the REAL emitted ALE meshes — the closest headless proxy
    // for the live GPU render loop (which needs a display we do not have).
    // Uniform freestream IC so the flow-coupled seeds advect from step one — the
    // mesh genuinely moves (nonzero swept fluxes, real per-step topology drift),
    // rather than sitting near-frozen while a from-rest flow slowly develops.
    let (fc_driver, fc_cells) =
        build_driver(MeshMotionSpec::FlowCoupled { regularization: 0.5 }, (1.0, 0.0));
    let fc = moving_mesh_worker_smoke(fc_driver, 120_000, 30, true);
    println!(
        "[moving-gui][flow-coupled] refreshes={} cells=[{:?},{:?}] max_scl={:.2e} \
         max_skew={:.3} meshes={} err={:?}",
        fc.mesh_refresh_events,
        fc.min_cells,
        fc.max_cells,
        fc.max_scl_defect,
        fc.max_skew,
        fc.meshes.len(),
        fc.error,
    );
    assert!(fc.error.is_none(), "flow-coupled worker reported an error: {:?}", fc.error);
    assert!(
        fc.mesh_refresh_events >= 30,
        "expected >=30 flow-coupled regens through the message API, got {}",
        fc.mesh_refresh_events
    );
    assert!(!fc.saw_empty_cells, "a flow-coupled refresh carried empty/degenerate cells");
    assert!(!fc.saw_nonfinite_stats, "a flow-coupled refresh carried non-finite stats");
    assert_eq!(
        (fc.min_cells, fc.max_cells),
        (Some(fc_cells), Some(fc_cells)),
        "flow-coupled cell count must stay fixed at {fc_cells} (no implied add/remove)"
    );
    // Physical: the post-closure per-cell SCL defect stays at f32-roundoff scale
    // over the whole run — a genuinely conservative moving mesh (no negative area
    // implied). ~1e-6 in practice; 1e-3 is a generous non-flaky ceiling.
    assert!(
        fc.max_scl_defect < 1e-3,
        "flow-coupled max SCL defect too large (non-conservative mesh): {:.3e}",
        fc.max_scl_defect
    );
    assert!(fc.max_skew.is_finite(), "flow-coupled max skew non-finite");
    assert_eq!(
        fc.meshes.len(),
        fc.mesh_refresh_events,
        "collect_meshes must retain every emitted refresh"
    );
    // Replay the real emitted meshes through the renderer's re-tessellation +
    // capacity-growth path.
    replay_through_renderer(&fc.meshes, fc_cells);

    std::env::remove_var("CFD2_BACKEND");
    std::env::remove_var("CFD2_CPU_ENGINE");
}

/// Replay the GUI's per-refresh re-tessellation + renderer upload on a SEQUENCE
/// of real emitted moving meshes, exactly as `poll_solver_worker` does at the
/// frame boundary: size the renderer ONCE from the first mesh (Initialize), then
/// for every emitted mesh call `build_mesh_vertices`/`build_line_vertices` +
/// `update_mesh`, asserting no overflow, that the buffers grow to fit, and that
/// `num_vertices` tracks the data exactly (no truncation). A real headless wgpu
/// device services every `write_buffer` (`poll(Wait)`), so an overrun would
/// validation-error — this exercises the ea2c421 fatal-crash class on the ACTUAL
/// per-step ALE topology, not synthetic polygons.
fn replay_through_renderer(meshes: &[Vec<Vec<[f64; 2]>>], n_cells: usize) {
    assert!(!meshes.is_empty(), "no meshes to replay through the renderer");
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(
        instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
    )
    .expect("adapter");
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("device");

    // Init sizing from the FIRST emitted mesh (what the GUI does at Initialize);
    // every later mesh must be absorbed by the growth path, never overflow.
    let v0 = cfd_renderer::build_mesh_vertices(&meshes[0]);
    let l0 = cfd_renderer::build_line_vertices(&meshes[0]);
    let mut renderer = cfd_renderer::CfdRenderResources::new(
        &device,
        wgpu::TextureFormat::Rgba8Unorm,
        v0.len().max(l0.len()).max(1),
    );

    for (step, cells) in meshes.iter().enumerate() {
        assert_eq!(cells.len(), n_cells, "cell count drifted at refresh {step}");
        let verts = cfd_renderer::build_mesh_vertices(cells);
        let lines = cfd_renderer::build_line_vertices(cells);
        assert!(
            !verts.is_empty() && !lines.is_empty(),
            "refresh {step} tessellated to nothing"
        );
        renderer.update_mesh(&device, &queue, &verts, &lines);
        assert_eq!(
            renderer.num_vertices as usize, verts.len(),
            "refresh {step}: vertex draw count truncated"
        );
        assert_eq!(
            renderer.num_line_vertices as usize, lines.len(),
            "refresh {step}: line draw count truncated"
        );
        assert!(
            renderer.capacity_vertices >= verts.len()
                && renderer.capacity_line_vertices >= lines.len(),
            "refresh {step}: renderer buffer did not grow to fit ({} tri / {} line vs cap {} / {})",
            verts.len(),
            lines.len(),
            renderer.capacity_vertices,
            renderer.capacity_line_vertices
        );
    }
    // Force the GPU to service every write_buffer; an overrun would have already
    // validation-errored.
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    println!(
        "[moving-gui][render] replayed {} real ALE meshes through the renderer \
         capacity path — no overflow, no truncation",
        meshes.len()
    );
}
