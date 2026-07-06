//! Variable cell count + flow-adaptive sizing (arc c+d).
//!
//! * `movingmesh_resize_free_stream_preserved` — the RESIZE seam
//!   ([`MovingMeshDriver::resize_cells`]): a uniform free stream survives a
//!   birth+kill event (solver rebuilt at the new count, full state rows
//!   transferred) bit-cleanly — any drift means the rebuild-and-gather
//!   transfer or the restarted volume history is wrong.
//! * `movingmesh_adaptive_sizing_obstacle` — the flow-adaptive planner on a
//!   GRADED obstacle channel: the |∇U|/|∇p|/strain indicator must trigger
//!   resize events (the boundary-distance grading mismatches the developing
//!   obstacle/wake gradient field) while the run stays stable and the count
//!   stays inside the adaptivity budget.
//! * `movingmesh_resize_gpu_regen_smoke` — after a resize, the opt-in
//!   ON-DEVICE regen path must resume at the NEW cell count (its fixed-size
//!   bundle is invalidated and lazily rebuilt).
//!
//! ```sh
//! cargo test --release --features "meshgen cpu" --test moving_mesh_adapt_test -- --nocapture
//! ```
#![cfg(all(feature = "meshgen", feature = "cpu"))]

use cfd2::meshgen::meshless::generate_cvt_mesh_with_seeds;
use cfd2::meshgen::{ChannelWithObstacle, LloydConfig, RectangularChannel};
use cfd2::sim::{MeshMotionSpec, MovingMeshDriver, RegenBackend, RuntimeParams};
use cfd2::solver::mesh::{BoundaryType, Mesh};
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{GpuLowMachPrecondModel, PreconditionerType, TimeScheme};
use nalgebra::{Point2, Vector2};
use std::sync::Mutex;

/// `CFD2_BACKEND` is process-global; serialize the tests.
static ENV_LOCK: Mutex<()> = Mutex::new(());

const LX: f64 = 1.0;
const LY: f64 = 1.0;
const H: f64 = 0.08;
const DT: f32 = 0.005;
const U0: f32 = 1.0;

fn test_params() -> RuntimeParams {
    RuntimeParams {
        adaptive_dt: false,
        target_cfl: 0.9,
        requested_dt: DT,
        dtau: 0.0,
        log_convergence: false,
        log_every_steps: 1000,
        advection_scheme: Scheme::SecondOrderUpwindVanLeer,
        time_scheme: TimeScheme::BDF2,
        preconditioner: PreconditionerType::Jacobi,
        outer_iters: 6,
        outer_auto_converge: false,
        low_mach_model: GpuLowMachPrecondModel::Off,
        low_mach_theta_floor: 1e-6,
        low_mach_pressure_coupling_alpha: 1.0,
        alpha_u: 0.7,
        alpha_p: 0.3,
        inlet_velocity: U0,
        density: 1.0,
        viscosity: 1e-2,
        eos: EosSpec::Constant,
        compressibility_psi: 0.0,
        outlet_back_pressure: 0.0,
        pressure_inlet: false,
        inlet_pressure: 0.0,
    }
}

/// Slip channel retag: Inlet left, Outlet right, SlipWall top/bottom — the
/// free-stream BC convention (the engine's no-slip walls would kill a uniform
/// flow).
fn tag_slip_channel(mesh: &mut Mesh) {
    let eps = 1e-6;
    for f in 0..mesh.num_faces() {
        if mesh.face_neighbor[f].is_some() {
            continue;
        }
        let (x, y) = (mesh.face_cx[f], mesh.face_cy[f]);
        let bt = if x < eps {
            BoundaryType::Inlet
        } else if x > LX - eps {
            BoundaryType::Outlet
        } else if y < eps || y > LY - eps {
            BoundaryType::SlipWall
        } else {
            continue;
        };
        mesh.face_boundary[f] = Some(bt);
    }
}

/// The seed nearest `target` that sits clear of every domain boundary (an
/// INTERIOR seed by construction — boundary seeds lie on the box/obstacle).
fn interior_seed_near(moving: &MovingMeshDriver, target: (f64, f64), margin: f64) -> usize {
    let seeds = moving.seeds();
    (0..seeds.len())
        .filter(|&i| {
            seeds[i].x > margin
                && seeds[i].x < LX - margin
                && seeds[i].y > margin
                && seeds[i].y < LY - margin
        })
        .min_by(|&a, &b| {
            let da = (seeds[a].x - target.0).powi(2) + (seeds[a].y - target.1).powi(2);
            let db = (seeds[b].x - target.0).powi(2) + (seeds[b].y - target.1).powi(2);
            da.total_cmp(&db)
        })
        .expect("no interior seed found")
}

fn max_freestream_err(moving: &MovingMeshDriver) -> f32 {
    let layout = moving.driver().solver().model().state_layout.clone();
    let stride = layout.stride() as usize;
    let u_off = layout.offset_for("U").expect("U") as usize;
    let state = pollster::block_on(moving.driver().solver().read_state_f32());
    let n = moving.mesh().num_cells();
    assert_eq!(state.len(), n * stride, "state length tracks the resized count");
    let mut e = 0.0f32;
    for c in 0..n {
        e = e
            .max((state[c * stride + u_off] - U0).abs())
            .max(state[c * stride + u_off + 1].abs());
    }
    e
}

/// Build the slip-channel free-stream driver (Frozen motion, uniform `(U0,0)`
/// IC — an exact fixed point of the scheme) and run the two resize events:
/// kill-1/birth-2 (net +1), then kill-2 (net −2). Returns the free-stream
/// errors after each event's settle window. Shared by the CPU gate and the
/// GPU on-device-regen smoke.
fn run_freestream_resize(
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    gpu_regen: bool,
) -> (f32, f32, usize, usize) {
    let geo = RectangularChannel {
        length: LX,
        height: LY,
    };
    let domain = Vector2::new(LX, LY);
    let mut cvt = generate_cvt_mesh_with_seeds(&geo, H, H, 1.0, domain, &LloydConfig::default());
    tag_slip_channel(&mut cvt.mesh);
    let n0 = cvt.mesh.num_cells();
    let params = test_params();
    let mut moving = pollster::block_on(MovingMeshDriver::build(
        cvt,
        &params,
        MeshMotionSpec::Frozen,
        &vec![(U0 as f64, 0.0); n0],
        &vec![0.0; n0],
        device,
        queue,
    ))
    .expect("driver build");
    moving.set_boundary_retag(Some(tag_slip_channel));
    moving.driver_mut().apply_params(&params);
    if gpu_regen {
        moving.set_gpu_regen(true);
        assert!(
            moving.gpu_regen_active(),
            "on-device regen requested but inactive (CPU backend or no device)"
        );
    }

    let mut device_steps_pre = 0usize;
    for step in 0..5 {
        let (outcome, stats) =
            moving.step(false).unwrap_or_else(|e| panic!("warm step {step}: {e}"));
        assert!(outcome.diverged.is_none(), "warm step {step} diverged");
        if stats.regen_backend == RegenBackend::GpuOnDevice {
            device_steps_pre += 1;
        }
    }

    // Event 1: kill one interior cell, birth two beside it (net +1).
    let k = interior_seed_near(&moving, (0.35 * LX, 0.5 * LY), 2.0 * H);
    let donor = interior_seed_near(&moving, (0.7 * LX, 0.5 * LY), 2.0 * H);
    assert_ne!(k, donor);
    let a = moving.seeds()[k];
    moving
        .resize_cells(
            &[k],
            &[
                (Point2::new(a.x - 0.3 * H, a.y), donor),
                (Point2::new(a.x + 0.3 * H, a.y), donor),
            ],
        )
        .expect("resize event 1 (kill 1, birth 2)");
    assert_eq!(moving.mesh().num_cells(), n0 + 1, "count after event 1");

    let mut device_steps_post = 0usize;
    for step in 0..10 {
        let (outcome, stats) =
            moving.step(false).unwrap_or_else(|e| panic!("post-birth step {step}: {e}"));
        assert!(outcome.diverged.is_none(), "post-birth step {step} diverged");
        assert_eq!(stats.n_cells, n0 + 1);
        if stats.regen_backend == RegenBackend::GpuOnDevice {
            device_steps_post += 1;
        }
    }
    let err1 = max_freestream_err(&moving);

    // Event 2: kill two well-separated interior cells (net −2 → n0 − 1).
    let k1 = interior_seed_near(&moving, (0.3 * LX, 0.3 * LY), 2.0 * H);
    let k2 = interior_seed_near(&moving, (0.7 * LX, 0.7 * LY), 2.0 * H);
    assert_ne!(k1, k2);
    moving
        .resize_cells(&[k1, k2], &[])
        .expect("resize event 2 (kill 2)");
    assert_eq!(moving.mesh().num_cells(), n0 - 1, "count after event 2");

    for step in 0..10 {
        let (outcome, stats) =
            moving.step(false).unwrap_or_else(|e| panic!("post-kill step {step}: {e}"));
        assert!(outcome.diverged.is_none(), "post-kill step {step} diverged");
        if stats.regen_backend == RegenBackend::GpuOnDevice {
            device_steps_post += 1;
        }
    }
    let err2 = max_freestream_err(&moving);
    (err1, err2, device_steps_pre, device_steps_post)
}

/// The RESIZE seam under a uniform free stream (an exact fixed point): birth
/// and kill events must leave the flow untouched — the solver is rebuilt at
/// the new count, every surviving/donor row transferred, and the volume
/// history restarted consistently, so the post-resize steps see the same
/// uniform state on the new mesh.
#[test]
fn movingmesh_resize_free_stream_preserved() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("CFD2_BACKEND", "cpu");
    let result = std::panic::catch_unwind(|| {
        let (err1, err2, _, _) = run_freestream_resize(None, None, false);
        eprintln!(
            "[resize-freestream] max|U-U0| after birth event = {err1:.3e}, \
             after kill event = {err2:.3e}"
        );
        assert!(
            err1 < 1e-4 && err2 < 1e-4,
            "free stream corrupted by resize: after birth {err1:.3e}, after kill {err2:.3e}"
        );
    });
    std::env::remove_var("CFD2_BACKEND");
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

/// GPU smoke: the on-device regen path across a resize. The device bundle is
/// sized at a fixed seed count; a resize must invalidate it and the next
/// device attempt must rebuild it at the NEW count — asserted as "GpuOnDevice
/// steps occur after the resize whenever they occurred before it".
#[test]
fn movingmesh_resize_gpu_regen_smoke() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("CFD2_BACKEND");
    let ctx = match pollster::block_on(cfd2::solver::gpu::context::GpuContext::new(None, None)) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("SKIP: no GPU adapter available ({e})");
            return;
        }
    };
    let (err1, err2, pre, post) = run_freestream_resize(
        Some(ctx.device.clone()),
        Some(ctx.queue.clone()),
        true,
    );
    eprintln!(
        "[resize-gpu] device steps: {pre}/5 before resize, {post}/20 after; \
         max|U-U0| = {err1:.3e} / {err2:.3e}"
    );
    assert!(
        err1 < 1e-3 && err2 < 1e-3,
        "free stream corrupted by resize on the GPU backend: {err1:.3e} / {err2:.3e}"
    );
    // A Frozen regen is byte-identical, so the device certifies it; if the
    // device path ran before the resize it must RESUME after it (the bundle
    // is rebuilt lazily at the new count — a stale-count bundle would either
    // error or fall back every step).
    if pre > 0 {
        assert!(
            post > 0,
            "on-device regen never resumed after the cell-count resize ({pre} device steps before)"
        );
    }
}

/// One adaptation run on the uniform slip channel with an EXPLICIT target
/// band ([`MovingMeshDriver::set_adaptive_sizing_band`], cell-size units):
/// returns cumulative (born, killed). Frozen motion + free stream — the
/// indicator is flow-agnostic here because the band places every eligible
/// cell decisively outside its hysteresis window in ONE direction.
fn band_run(band: (f64, f64), steps: usize) -> (usize, usize) {
    let geo = RectangularChannel {
        length: LX,
        height: LY,
    };
    let domain = Vector2::new(LX, LY);
    let mut cvt = generate_cvt_mesh_with_seeds(&geo, H, H, 1.0, domain, &LloydConfig::default());
    tag_slip_channel(&mut cvt.mesh);
    let n0 = cvt.mesh.num_cells();
    let params = test_params();
    let mut moving = pollster::block_on(MovingMeshDriver::build(
        cvt,
        &params,
        MeshMotionSpec::Frozen,
        &vec![(U0 as f64, 0.0); n0],
        &vec![0.0; n0],
        None,
        None,
    ))
    .expect("band driver build");
    moving.set_boundary_retag(Some(tag_slip_channel));
    moving.driver_mut().apply_params(&params);
    moving.set_adaptive_sizing(10);
    moving.set_adaptive_sizing_band(Some(band));
    let (mut born, mut killed) = (0usize, 0usize);
    for step in 0..steps {
        let (outcome, stats) = moving
            .step(false)
            .unwrap_or_else(|e| panic!("band step {step}: {e}"));
        assert!(outcome.diverged.is_none(), "band step {step} diverged");
        born += stats.cells_born;
        killed += stats.cells_killed;
    }
    (born, killed)
}

/// The explicit adaptation band steers the planner independently of the
/// built mesh: a band FINER than the mesh makes every bulk cell exceed 2×
/// its target (births must fire, kills must not — this also exercises the
/// target-scaled birth-clearance floor, which the global meshgen floor
/// would veto), and a band COARSER than the mesh puts every cell below
/// 0.45× its target (kills only).
#[test]
fn movingmesh_adaptive_band_steers_planner() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("CFD2_BACKEND", "cpu");
    let result = std::panic::catch_unwind(|| {
        // Fine band (half the mesh pitch): refine-only.
        let (born_f, killed_f) = band_run((0.5 * H, 0.6 * H), 15);
        // Coarse band (double the mesh pitch): coarsen-only.
        let (born_c, killed_c) = band_run((2.0 * H, 2.5 * H), 15);
        eprintln!(
            "[adapt-band] fine band: +{born_f}/−{killed_f}; coarse band: +{born_c}/−{killed_c}"
        );
        assert!(
            born_f > 0 && killed_f == 0,
            "fine band must refine only (+{born_f}/−{killed_f})"
        );
        assert!(
            killed_c > 0 && born_c == 0,
            "coarse band must coarsen only (+{born_c}/−{killed_c})"
        );
    });
    std::env::remove_var("CFD2_BACKEND");
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

/// The flow-adaptive planner on a GRADED obstacle channel (FlowCoupled): the
/// gradient indicator must fire resize events within the run (the
/// boundary-distance grading mismatches the developing obstacle/wake
/// gradient field), the count must respect the adaptivity budget, and the
/// run must stay stable through the events (each a full solver rebuild).
#[test]
fn movingmesh_adaptive_sizing_obstacle() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("CFD2_BACKEND", "cpu");
    let result = std::panic::catch_unwind(adaptive_obstacle_body);
    std::env::remove_var("CFD2_BACKEND");
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

fn adaptive_obstacle_body() {
    run_adaptive_obstacle(
        "adapt-obstacle",
        MeshMotionSpec::FlowCoupled { regularization: 0.5 },
        200,
        0,
    );
}

/// STATIONARY adaptive mesh: `Frozen` motion + flow-adaptive sizing +
/// scheduled smoothing — the mesh never advects, but cells birth/kill toward
/// the flow's gradient field and the Lloyd relaxation (which under Frozen
/// persists: the mesh advances from its current seeds) regularizes the
/// resized neighborhoods while PRESERVING the adapted spacing.
#[test]
fn movingmesh_adaptive_sizing_stationary_obstacle() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("CFD2_BACKEND", "cpu");
    let result = std::panic::catch_unwind(|| {
        run_adaptive_obstacle("adapt-stationary", MeshMotionSpec::Frozen, 100, 5);
    });
    std::env::remove_var("CFD2_BACKEND");
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

/// Shared graded-obstacle adaptive run: FlowCoupled or Frozen (stationary)
/// motion, adapt every 20 steps, optional scheduled smoothing. Asserts a
/// resize fired, the count stayed inside the budget, dt never collapsed and
/// the flow stayed bounded.
fn run_adaptive_obstacle(
    label: &str,
    motion: MeshMotionSpec,
    steps: usize,
    smooth_every: usize,
) {
    let (lx, ly) = (2.0, 1.0);
    let geo = ChannelWithObstacle {
        length: lx,
        height: ly,
        obstacle_center: Point2::new(0.6, 0.51),
        obstacle_radius: 0.1,
    };
    let domain = Vector2::new(lx, ly);
    let mut params = test_params();
    params.requested_dt = 0.01;
    params.viscosity = 1.33e-3; // Re ≈ 150 on the D=0.2 obstacle
    // GRADED mesh (fine 0.04 near boundaries → coarse 0.08 far): a wide
    // realized volume band gives the indicator room to both refine (coarse
    // cells reached by the wake's gradient field) and coarsen (fine cells in
    // smooth regions).
    let cvt = generate_cvt_mesh_with_seeds(&geo, 0.04, 0.08, 1.2, domain, &LloydConfig::default());
    let n0 = cvt.mesh.num_cells();
    let mut moving = pollster::block_on(MovingMeshDriver::build(
        cvt,
        &params,
        motion,
        &vec![(U0 as f64, 0.0); n0],
        &vec![0.0; n0],
        None,
        None,
    ))
    .expect("obstacle driver build");
    moving.driver_mut().apply_params(&params);
    moving.set_adaptive_sizing(20);
    if smooth_every > 0 {
        moving.set_smoothing(smooth_every, 1, 0.5);
    }

    let (mut born, mut killed) = (0usize, 0usize);
    for step in 0..steps {
        let (outcome, stats) = moving
            .step(false)
            .unwrap_or_else(|e| panic!("adapt step {step}: {e}"));
        assert!(outcome.diverged.is_none(), "adapt step {step} diverged");
        assert!(stats.dt > 1e-4, "dt collapsed at step {step}: {}", stats.dt);
        born += stats.cells_born;
        killed += stats.cells_killed;
        if stats.cells_born > 0 || stats.cells_killed > 0 {
            eprintln!(
                "[{label}] step {step}: +{} / −{} cells → {} (skew {:.3})",
                stats.cells_born, stats.cells_killed, stats.n_cells, stats.max_skew
            );
        }
    }

    let layout = moving.driver().solver().model().state_layout.clone();
    let stride = layout.stride() as usize;
    let u_off = layout.offset_for("U").expect("U") as usize;
    let state = pollster::block_on(moving.driver().solver().read_state_f32());
    let n_final = moving.mesh().num_cells();
    assert_eq!(state.len(), n_final * stride);
    let max_u = (0..n_final)
        .map(|c| {
            let u = state[c * stride + u_off] as f64;
            let v = state[c * stride + u_off + 1] as f64;
            (u * u + v * v).sqrt()
        })
        .fold(0.0f64, f64::max);

    eprintln!(
        "[{label}] n0={n0} → {n_final}: born={born} killed={killed} max|u|={max_u:.3}"
    );
    assert!(
        born + killed > 0,
        "adaptive sizing produced no resize event in {steps} steps"
    );
    assert!(
        n_final >= n0 / 2 && n_final <= 2 * n0,
        "cell count {n_final} left the adaptivity budget [{}, {}]",
        n0 / 2,
        2 * n0
    );
    assert!(
        max_u.is_finite() && max_u < 10.0 * U0 as f64,
        "flow not bounded under adaptive sizing: max|u| = {max_u:.3}"
    );
}
