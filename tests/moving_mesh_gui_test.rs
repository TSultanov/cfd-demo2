//! Headless GUI ↔ moving-mesh (ALE) worker seam test.
//!
//! No display in CI, so this exercises the verifiable half of the GUI moving-mesh
//! hookup: build a `MovingMeshDriver` as the GUI init path does and step it, then
//! drive the real private solver worker through the `SolverMode::MovingMesh`
//! message path and observe the `MeshRefreshed` events. No window is created.
#![cfg(feature = "ui")]

use cfd2::meshgen::ChannelWithObstacle;
use cfd2::solver::mesh::{BackwardsStep, LloydConfig};
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{GpuLowMachPrecondModel, PreconditionerType, TimeScheme};
use cfd2::sim::{BoundaryMotionSpec, MeshMotionSpec, MovingMeshDriver, OscAxis, RuntimeParams};
use cfd2::ui::app::moving_mesh_worker_smoke;
use cfd2::ui::cfd_renderer;
use nalgebra::{Point2, Vector2};
use std::sync::Mutex;

/// `CFD2_BACKEND` is process-global; serialize the two backend-selecting worker
/// tests so one never flips the backend out from under the other's driver build.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Is a GPU adapter present? (The GPU worker gate skips cleanly without one.)
fn gpu_adapter_available() -> bool {
    let instance = wgpu::Instance::default();
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).is_ok()
}

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

/// Build a moving-mesh driver on a coarse CVT backstep mesh, as the GUI's
/// `build_moving_init` does. `u0` is the uniform initial cell velocity: `(0,0)`
/// is the from-rest IC; a nonzero freestream makes the FlowCoupled seeds advect
/// from step one.
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
    // Apply outer_iters / relaxation knobs, as the worker's SetSolver does.
    driver.driver_mut().apply_params(&params);
    (driver, n_cells)
}

/// Build a moving-mesh driver on a ChannelWithObstacle CVT with a cross-stream
/// oscillating obstacle + the MovingWall BC (the "Oscillating obstacle" GUI
/// toggle). Frozen interior seeds; only the obstacle loop (loop 1) moves.
fn build_oscillating_obstacle_driver() -> (MovingMeshDriver, usize) {
    let domain = Vector2::new(3.0, 1.0);
    let geo = ChannelWithObstacle {
        length: 3.0,
        height: 1.0,
        obstacle_center: Point2::new(1.0, 0.51),
        obstacle_radius: 0.1,
    };
    let cvt = cfd2::meshgen::meshless::generate_cvt_mesh_with_seeds(
        &geo,
        0.06,
        0.06,
        1.0,
        domain,
        &LloydConfig::default(),
    );
    let n_cells = cvt.mesh.num_cells();
    assert!(n_cells > 20, "expected a non-trivial CVT mesh, got {n_cells} cells");
    let params = ale_params();
    let mut driver = pollster::block_on(MovingMeshDriver::build(
        cvt,
        &params,
        MeshMotionSpec::Frozen, // interior frozen; only the obstacle oscillates
        &vec![(params.inlet_velocity as f64, 0.0); n_cells],
        &vec![0.0; n_cells],
        None,
        None,
    ))
    .expect("MovingMeshDriver::build (oscillating obstacle) must succeed");
    driver.driver_mut().apply_params(&params);
    driver.set_boundary_motion(BoundaryMotionSpec::Oscillation {
        loop_index: 1,
        amplitude: 0.03,
        omega: std::f64::consts::TAU * 0.5,
        axis: OscAxis::CrossStream,
    });
    driver.set_moving_wall_bc(true);
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
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::set_var("CFD2_BACKEND", "cpu");
    std::env::set_var("CFD2_CPU_ENGINE", "interpreter");

    // The Frozen anchor + flow-coupled path both step cleanly and report finite,
    // fixed-cell telemetry.
    driver_steps_produce_finite_stats(MeshMotionSpec::Frozen, 6);
    driver_steps_produce_finite_stats(MeshMotionSpec::FlowCoupled { regularization: 0.5 }, 6);

    // Drive the private solver worker through the moving-mesh message path and
    // observe the MeshRefreshed events it emits.
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

    // The full flow-coupled render loop, headless: drive the worker in FlowCoupled
    // moving mode for ~30 regens, collect each emitted mesh, then replay the UI's
    // per-frame re-tessellation + renderer capacity path on the real ALE meshes.
    // Uniform freestream IC so the seeds advect from step one (nonzero swept
    // fluxes, real per-step topology drift) rather than sitting near-frozen.
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
    // Post-closure per-cell SCL defect stays at f32-roundoff scale (a conservative
    // moving mesh): ~1e-6 in practice; 1e-3 is a generous non-flaky ceiling.
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
    replay_through_renderer(&fc.meshes, fc_cells);

    // The moving-boundary worker path: drive an oscillating-obstacle driver
    // through the worker, collect the emitted meshes, and assert the obstacle
    // actually moved (near-obstacle geometry changes across refreshes) while the
    // cell count stays fixed and the mesh stays conservative + renderable.
    let (osc_driver, osc_cells) = build_oscillating_obstacle_driver();
    let osc = moving_mesh_worker_smoke(osc_driver, 120_000, 30, true);
    println!(
        "[moving-gui][osc-obstacle] refreshes={} cells=[{:?},{:?}] max_scl={:.2e} \
         max_skew={:.3} meshes={} err={:?}",
        osc.mesh_refresh_events,
        osc.min_cells,
        osc.max_cells,
        osc.max_scl_defect,
        osc.max_skew,
        osc.meshes.len(),
        osc.error,
    );
    assert!(osc.error.is_none(), "oscillating-obstacle worker reported an error: {:?}", osc.error);
    assert!(
        osc.mesh_refresh_events >= 20,
        "expected >=20 oscillating-obstacle regens, got {}",
        osc.mesh_refresh_events
    );
    assert!(!osc.saw_empty_cells, "an oscillating-obstacle refresh carried empty/degenerate cells");
    assert!(!osc.saw_nonfinite_stats, "an oscillating-obstacle refresh carried non-finite stats");
    assert_eq!(
        (osc.min_cells, osc.max_cells),
        (Some(osc_cells), Some(osc_cells)),
        "oscillating-obstacle cell count must stay fixed at {osc_cells}"
    );
    assert!(osc.max_scl_defect < 1e-3, "oscillating-obstacle SCL defect too large: {:.3e}", osc.max_scl_defect);
    // The obstacle moved: some cell's polygon differs between the first refresh
    // and a later one (a rigid boundary displacement re-tessellates near-wall cells).
    let mesh_moved = osc.meshes.len() >= 2
        && osc.meshes.iter().skip(1).any(|m| moved_relative_to(&osc.meshes[0], m));
    assert!(mesh_moved, "oscillating-obstacle mesh never changed across refreshes (obstacle did not move)");
    replay_through_renderer(&osc.meshes, osc_cells);

    std::env::remove_var("CFD2_BACKEND");
    std::env::remove_var("CFD2_CPU_ENGINE");
}

/// Whether two emitted cell-polygon sets differ meaningfully (any vertex moved
/// by more than a roundoff epsilon) — the "the obstacle moved" discriminator.
fn moved_relative_to(a: &[Vec<[f64; 2]>], b: &[Vec<[f64; 2]>]) -> bool {
    if a.len() != b.len() {
        return true;
    }
    for (ca, cb) in a.iter().zip(b) {
        if ca.len() != cb.len() {
            return true;
        }
        for (pa, pb) in ca.iter().zip(cb) {
            if (pa[0] - pb[0]).abs() > 1e-9 || (pa[1] - pb[1]).abs() > 1e-9 {
                return true;
            }
        }
    }
    false
}

/// Replay the GUI's per-refresh re-tessellation + renderer upload on a SEQUENCE
/// of real emitted moving meshes, exactly as `poll_solver_worker` does at the
/// frame boundary: size the renderer ONCE from the first mesh (Initialize), then
/// for every emitted mesh call `build_mesh_vertices`/`build_line_vertices` +
/// `update_mesh`, asserting no overflow, that the buffers grow to fit, and that
/// `num_vertices` tracks the data exactly (no truncation). A real headless wgpu
/// device services every `write_buffer` (`poll(Wait)`), so an overrun would
/// validation-error — this exercises the buffer-overflow crash class on the
/// actual per-step ALE topology, not synthetic polygons.
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
        cfd_renderer::VERTEX_HEADROOM,
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

/// The moving-mesh worker on the **GPU backend** (surgical topology refresh +
/// carried BDF2 history). Drives the private solver worker
/// (`SetSolver { MovingMesh } + SetRunning`) with a flow-coupled CVT backstep on
/// the GPU backend (`CFD2_BACKEND` unset ⇒ GPU device), collects the
/// `MeshRefreshed` events, and replays the emitted meshes through the renderer
/// capacity path — the same assertions as the CPU worker smoke, but on the GPU.
/// Skips cleanly when no GPU adapter is present.
#[test]
fn moving_mesh_gui_worker_gpu_backend() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    if !gpu_adapter_available() {
        eprintln!("[moving-gui][gpu] no GPU adapter present; skipping GPU worker gate");
        return;
    }
    // GPU backend = CFD2_BACKEND not "cpu". Clear both so the driver build selects
    // the GPU device (and no stale CPU-engine hint leaks in).
    std::env::remove_var("CFD2_BACKEND");
    std::env::remove_var("CFD2_CPU_ENGINE");

    // Flow-coupled CVT backstep with a uniform freestream IC so the seeds advect
    // from step one — genuine per-step topology drift through the GPU surgical
    // refresh, not a near-frozen mesh.
    let (driver, n_cells) =
        build_driver(MeshMotionSpec::FlowCoupled { regularization: 0.5 }, (1.0, 0.0));
    let smoke = moving_mesh_worker_smoke(driver, 300_000, 10, true);
    println!(
        "[moving-gui][gpu] refreshes={} cells=[{:?},{:?}] max_scl={:.2e} max_skew={:.3} \
         meshes={} empty={} nonfinite={} err={:?}",
        smoke.mesh_refresh_events,
        smoke.min_cells,
        smoke.max_cells,
        smoke.max_scl_defect,
        smoke.max_skew,
        smoke.meshes.len(),
        smoke.saw_empty_cells,
        smoke.saw_nonfinite_stats,
        smoke.error,
    );
    assert!(smoke.error.is_none(), "GPU moving worker reported an error: {:?}", smoke.error);
    assert!(
        smoke.mesh_refresh_events >= 10,
        "GPU moving worker did not emit MeshRefreshed via the message path (got {})",
        smoke.mesh_refresh_events
    );
    assert!(!smoke.saw_empty_cells, "a GPU MeshRefreshed carried empty/degenerate cells");
    assert!(!smoke.saw_nonfinite_stats, "a GPU MeshRefreshed carried non-finite stats");
    assert_eq!(
        (smoke.min_cells, smoke.max_cells),
        (Some(n_cells), Some(n_cells)),
        "GPU worker mesh cell count must stay fixed at {n_cells}"
    );
    assert!(
        smoke.max_scl_defect < 1e-3,
        "GPU moving worker max SCL defect too large (non-conservative mesh): {:.3e}",
        smoke.max_scl_defect
    );
    // The emitted GPU meshes re-tessellate + upload without overflow, exactly as
    // the CPU path.
    replay_through_renderer(&smoke.meshes, n_cells);

    std::env::remove_var("CFD2_BACKEND");
    std::env::remove_var("CFD2_CPU_ENGINE");
}

/// The GPU-backend worker path with ON-DEVICE mesh reconstruction — the exact
/// configuration the GUI now enables by default on the GPU backend for a
/// moving-mesh run ("On-device mesh regen (GPU)" + oscillating obstacle):
/// drive the real worker message pump with `set_gpu_regen(true)` on a
/// polyline-boundary (obstacle) case with a MOVING boundary, and assert
/// (1) refreshes flow with no error, (2) at least one step ran fully on
/// device (the counter comes from `MovingMeshStats::regen_backend` — the same
/// field the GUI's "Mesh regen" status line shows), (3) the vertex-less
/// device meshes tessellate + render through the GUI's real capacity path
/// (`cache_cells` face-geometry reconstruction feeding `update_mesh`), and
/// (4) the cell count stays fixed and the run stays conservative.
#[test]
fn moving_mesh_gui_worker_gpu_regen_smoke() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    if !gpu_adapter_available() {
        eprintln!("SKIP: no GPU adapter available");
        return;
    }
    // GPU solver backend (the gpu_regen eligibility check reads it).
    std::env::remove_var("CFD2_BACKEND");
    std::env::remove_var("CFD2_CPU_ENGINE");

    let ctx = pollster::block_on(cfd2::solver::gpu::context::GpuContext::new(None, None))
        .expect("GPU context");
    let domain = Vector2::new(3.0, 1.0);
    let geo = ChannelWithObstacle {
        length: 3.0,
        height: 1.0,
        obstacle_center: Point2::new(1.0, 0.51),
        obstacle_radius: 0.1,
    };
    let cvt = cfd2::meshgen::meshless::generate_cvt_mesh_with_seeds(
        &geo,
        0.06,
        0.06,
        1.0,
        domain,
        &LloydConfig::default(),
    );
    let n_cells = cvt.mesh.num_cells();
    let params = ale_params();
    let mut driver = pollster::block_on(MovingMeshDriver::build(
        cvt,
        &params,
        MeshMotionSpec::Frozen, // interior frozen; only the obstacle oscillates
        &vec![(params.inlet_velocity as f64, 0.0); n_cells],
        &vec![0.0; n_cells],
        Some(ctx.device.clone()),
        Some(ctx.queue.clone()),
    ))
    .expect("MovingMeshDriver::build (GPU) must succeed");
    driver.driver_mut().apply_params(&params);
    driver.set_boundary_motion(BoundaryMotionSpec::Oscillation {
        loop_index: 1,
        amplitude: 0.03,
        omega: std::f64::consts::TAU * 0.5,
        axis: OscAxis::CrossStream,
    });
    driver.set_moving_wall_bc(true);
    driver.set_gpu_regen(true);
    assert!(driver.gpu_regen_active(), "gpu_regen must be active on the GPU backend");

    let smoke = moving_mesh_worker_smoke(driver, 120_000, 20, true);
    println!(
        "[moving-gui][gpu-regen] refreshes={} (on-device {}, cpu-fallback {}) \
         cells=[{:?},{:?}] max_scl={:.2e} empty={} nonfinite={} err={:?}",
        smoke.mesh_refresh_events,
        smoke.gpu_ondevice_refreshes,
        smoke.gpu_fallback_refreshes,
        smoke.min_cells,
        smoke.max_cells,
        smoke.max_scl_defect,
        smoke.saw_empty_cells,
        smoke.saw_nonfinite_stats,
        smoke.error,
    );
    assert!(smoke.error.is_none(), "gpu-regen worker reported an error: {:?}", smoke.error);
    assert!(
        smoke.mesh_refresh_events >= 20,
        "expected >=20 gpu-regen refreshes through the message API, got {}",
        smoke.mesh_refresh_events
    );
    assert!(
        smoke.gpu_ondevice_refreshes > 0,
        "no refresh ran on device — gpu_regen was requested but never took a step \
         (fallback {} / {} refreshes)",
        smoke.gpu_fallback_refreshes,
        smoke.mesh_refresh_events
    );
    assert!(
        !smoke.saw_empty_cells,
        "a gpu-regen refresh carried empty/degenerate cells — the vertex-less \
         face-geometry polygon reconstruction is broken"
    );
    assert!(!smoke.saw_nonfinite_stats, "a gpu-regen refresh carried non-finite stats");
    assert_eq!(
        (smoke.min_cells, smoke.max_cells),
        (Some(n_cells), Some(n_cells)),
        "gpu-regen cell count must stay fixed at {n_cells}"
    );
    assert!(
        smoke.max_scl_defect < 1e-3,
        "gpu-regen max SCL defect too large: {:.3e}",
        smoke.max_scl_defect
    );
    // The vertex-less meshes must survive the GUI's real tessellation +
    // renderer capacity path.
    replay_through_renderer(&smoke.meshes, n_cells);
}

/// The GUI's "All-Mach thermal + Moving Mesh (ALE)" combination, end-to-end
/// through the worker pump: the Model dropdown offers `allmach_thermal`, the
/// moving toggle maps it to `allmach_thermal_ale` (`ale_model_for`), and the
/// driver seeds the thermal state (T / rho_t_ref / psi / dt_local) via
/// `SolverDriver::build`. This gate mirrors that exact path — thermal ALE
/// model + FlowCoupled interior motion on the ChannelObstacle CVT — and
/// asserts refreshes flow with finite stats, a fixed cell count, and a
/// conservative mesh, so the UI enablement is backed by a running solver,
/// not just an enabled checkbox.
#[test]
fn moving_mesh_gui_worker_allmach_thermal_smoke() {
    use cfd2::solver::model::allmach_thermal_ale_model;
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::set_var("CFD2_BACKEND", "cpu");
    std::env::set_var("CFD2_CPU_ENGINE", "transpiled");

    let domain = Vector2::new(3.0, 1.0);
    let geo = ChannelWithObstacle {
        length: 3.0,
        height: 1.0,
        obstacle_center: Point2::new(1.0, 0.51),
        obstacle_radius: 0.1,
    };
    let cvt = cfd2::meshgen::meshless::generate_cvt_mesh_with_seeds(
        &geo,
        0.06,
        0.06,
        1.0,
        domain,
        &LloydConfig::default(),
    );
    let n_cells = cvt.mesh.num_cells();
    // The PROVEN thermal-ALE recipe (allmach_ale_test::run_obstacle_smoke):
    // Euler stepping, 6 outer sweeps, dt 5e-3, gauge-pressure psi = 1e-4.
    // The incompressible GUI defaults (BDF2, 4 sweeps, dt 1e-2) are NOT
    // stable for the thermal model on this coarse obstacle mesh.
    let mut params = ale_params();
    params.time_scheme = TimeScheme::Euler;
    params.outer_iters = 6;
    params.requested_dt = 5e-3;
    params.compressibility_psi = 1e-4;
    params.inlet_velocity = 0.4;
    let mut driver = pollster::block_on(MovingMeshDriver::build_with_model(
        cvt,
        allmach_thermal_ale_model().expect("allmach_thermal_ale model"),
        &params,
        MeshMotionSpec::FlowCoupled { regularization: 0.5 },
        &vec![(params.inlet_velocity as f64, 0.0); n_cells],
        &vec![0.0; n_cells],
        None,
        None,
    ))
    .expect("MovingMeshDriver::build_with_model(allmach_thermal_ale) must succeed");
    driver.driver_mut().apply_params(&params);

    let smoke = moving_mesh_worker_smoke(driver, 120_000, 15, true);
    println!(
        "[moving-gui][thermal-ale] refreshes={} cells=[{:?},{:?}] max_scl={:.2e} \
         empty={} nonfinite={} err={:?}",
        smoke.mesh_refresh_events,
        smoke.min_cells,
        smoke.max_cells,
        smoke.max_scl_defect,
        smoke.saw_empty_cells,
        smoke.saw_nonfinite_stats,
        smoke.error,
    );
    assert!(
        smoke.error.is_none(),
        "thermal-ALE worker reported an error: {:?}",
        smoke.error
    );
    assert!(
        smoke.mesh_refresh_events >= 15,
        "expected >=15 thermal-ALE refreshes, got {}",
        smoke.mesh_refresh_events
    );
    assert!(!smoke.saw_empty_cells, "a thermal-ALE refresh carried empty/degenerate cells");
    assert!(!smoke.saw_nonfinite_stats, "a thermal-ALE refresh carried non-finite stats");
    assert_eq!(
        (smoke.min_cells, smoke.max_cells),
        (Some(n_cells), Some(n_cells)),
        "thermal-ALE cell count must stay fixed at {n_cells}"
    );
    assert!(
        smoke.max_scl_defect < 1e-3,
        "thermal-ALE max SCL defect too large: {:.3e}",
        smoke.max_scl_defect
    );

    std::env::remove_var("CFD2_BACKEND");
    std::env::remove_var("CFD2_CPU_ENGINE");
}
