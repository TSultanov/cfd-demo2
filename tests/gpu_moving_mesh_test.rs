//! GPU moving-mesh loop gates (`MovingMeshDriver` end-to-end).
//!
//! Gates:
//!   * `gpu_moving_mesh_gcl_{euler,bdf2}` — prescribed flip-free swirl keeps a
//!     uniform free stream uniform, non-compounding, over the run. BDF2 also
//!     proves surgical history preservation: `state_old_old` must survive the
//!     per-step topology refresh in place, or BDF2 cold-starts to Euler each step
//!     and the drift compounds.
//!   * `gpu_vs_cpu_moving_mesh_freestream_gcl` — the same prescribed-motion case
//!     on both backends must agree on the conservation observables (both preserve
//!     the free stream; final fields match to an f32 cross-backend RMS). The free
//!     stream is an exact discrete fixed point, so this proves cross-backend GCL
//!     preservation only, NOT a developed-field physics match.
//!   * `gpu_moving_perf` — per-step overhead split (plan/regen/swept/refresh vs
//!     total) at ~20k cells.
//!
//! Skips cleanly when no GPU adapter is present. Needs `feature = "meshgen"`; the
//! cross-backend gate additionally needs `cpu`.
#![cfg(feature = "meshgen")]

use cfd2::meshgen::meshless::generate_cvt_mesh_with_seeds;
use cfd2::meshgen::{LloydConfig, RectangularChannel};
use cfd2::sim::{MeshMotionSpec, MovingMeshDriver, RuntimeParams};
use cfd2::solver::gpu::context::GpuContext;
use cfd2::solver::mesh::{BoundaryType, Mesh};
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{GpuLowMachPrecondModel, PreconditionerType, TimeScheme};
use nalgebra::Vector2;
use std::sync::Mutex;

/// `CFD2_BACKEND` is process-global; serialize the backend-selecting tests.
static ENV_LOCK: Mutex<()> = Mutex::new(());

const LX: f64 = 1.0;
const LY: f64 = 1.0;
const H: f64 = 0.08;
const DT: f32 = 0.005;
const STEPS: usize = 120;
/// Motion period (~1.5 swirl cycles over the run).
const PERIOD: f64 = 80.0 * DT as f64;
/// Horizontal free stream — an exact discrete fixed point under inlet Dirichlet
/// `(U,0)` + outlet zero-gradient + slip top/bottom, so any drift is a GCL/solve
/// artifact, never BC physics.
const U0: (f32, f32) = (1.0, 0.0);

/// Flip-free interior swirl: rotate each seed about the domain centre by a
/// boundary-vanishing bump, small enough to never flip the Voronoi adjacency.
/// Evaluated from the t=0 label (no drift).
fn swirl(p: [f64; 2], t: f64) -> [f64; 2] {
    let (cx, cy) = (0.5 * LX, 0.5 * LY);
    let bump = (std::f64::consts::PI * p[0] / LX).sin().powi(2)
        * (std::f64::consts::PI * p[1] / LY).sin().powi(2);
    let theta = 0.03 * (2.0 * std::f64::consts::PI * t / PERIOD).sin() * bump;
    let (dx, dy) = (p[0] - cx, p[1] - cy);
    let (c, s) = (theta.cos(), theta.sin());
    [cx + c * dx - s * dy, cy + s * dx + c * dy]
}

fn test_params(time_scheme: TimeScheme) -> RuntimeParams {
    RuntimeParams {
        adaptive_dt: false,
        target_cfl: 0.9,
        requested_dt: DT,
        dtau: 0.0,
        log_convergence: false,
        log_every_steps: 1000,
        advection_scheme: Scheme::SecondOrderUpwindVanLeer,
        time_scheme,
        preconditioner: PreconditionerType::Jacobi,
        outer_iters: 6,
        outer_auto_converge: false,
        low_mach_model: GpuLowMachPrecondModel::Off,
        low_mach_theta_floor: 1e-6,
        low_mach_pressure_coupling_alpha: 1.0,
        alpha_u: 0.7,
        alpha_p: 0.3,
        inlet_velocity: U0.0,
        density: 1.0,
        viscosity: 1e-2,
        eos: EosSpec::Constant,
        compressibility_psi: 0.0,
        outlet_back_pressure: 0.0,
        pressure_inlet: false,
        inlet_pressure: 0.0,
    }
}

fn square() -> (RectangularChannel, Vector2<f64>) {
    (
        RectangularChannel {
            length: LX,
            height: LY,
        },
        Vector2::new(LX, LY),
    )
}

/// Slip channel: Inlet left, Outlet right, SlipWall top+bottom (the engine tags
/// bottom/top no-slip `Wall`, which would kill a uniform flow). A `fn` pointer so
/// it is the driver's per-regen retag hook.
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

/// Per-cell field snapshot + GCL/conservation observables of one run.
struct RunOut {
    /// Final per-cell (u, v, p) — the cross-backend comparison payload.
    final_u: Vec<f32>,
    final_v: Vec<f32>,
    final_p: Vec<f32>,
    n_cells: usize,
    max_du: f32,
    max_dp: f32,
    early_du: f32,
    late_du: f32,
    early_dp: f32,
    late_dp: f32,
    max_scl_defect: f64,
    max_identity_err: f64,
    max_skew: f64,
    rebuilds: usize,
}

/// Drive the uniform-flow swirl GCL protocol through the moving-mesh driver on
/// the given backend (`device`/`queue` = `Some` → GPU, `None` → CPU).
fn run_gcl(
    time_scheme: TimeScheme,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
) -> RunOut {
    let is_gpu = device.is_some();
    let (geo, domain) = square();
    let params = test_params(time_scheme);
    let mut cvt = generate_cvt_mesh_with_seeds(&geo, H, H, 1.0, domain, &LloydConfig::default());
    tag_slip_channel(&mut cvt.mesh);
    let n = cvt.mesh.num_cells();

    let mut moving = pollster::block_on(MovingMeshDriver::build(
        cvt,
        &params,
        MeshMotionSpec::Prescribed(swirl),
        &vec![(U0.0 as f64, U0.1 as f64); n],
        &vec![0.0; n],
        device,
        queue,
    ))
    .expect("moving driver build");
    assert_eq!(
        moving.driver().solver().is_cpu(),
        !is_gpu,
        "backend mismatch: expected {}",
        if is_gpu { "GPU" } else { "CPU" }
    );
    moving.set_boundary_retag(Some(tag_slip_channel));
    moving.driver_mut().apply_params(&params);

    let layout = moving.driver().solver().model().state_layout.clone();
    let stride = layout.stride() as usize;
    let u_off = layout.offset_for("U").expect("U offset") as usize;
    let p_off = layout.offset_for("p").expect("p offset") as usize;

    let mut out = RunOut {
        final_u: vec![0.0; n],
        final_v: vec![0.0; n],
        final_p: vec![0.0; n],
        n_cells: n,
        max_du: 0.0,
        max_dp: 0.0,
        early_du: 0.0,
        late_du: 0.0,
        early_dp: 0.0,
        late_dp: 0.0,
        max_scl_defect: 0.0,
        max_identity_err: 0.0,
        max_skew: 0.0,
        rebuilds: 0,
    };

    for step in 0..STEPS {
        let (outcome, stats) = moving.step(false).unwrap_or_else(|e| {
            panic!("step {step} failed (a flip means the amplitude is too large): {e}")
        });
        assert!(outcome.diverged.is_none(), "step {step} diverged");
        out.max_scl_defect = out.max_scl_defect.max(stats.scl_defect);
        out.max_identity_err = out.max_identity_err.max(stats.identity_err);
        out.max_skew = out.max_skew.max(stats.max_skew);
        if stats.topo_changed {
            out.rebuilds += 1;
        }
        assert!(
            stats.identity_err < 1e-10,
            "step {step}: f64 swept-quad identity {:.3e} above roundoff",
            stats.identity_err
        );
        assert!(
            stats.scl_defect < 1e-7,
            "step {step}: f32 SCL closure defect {:.3e} above roundoff",
            stats.scl_defect
        );

        let last = step + 1 == STEPS;
        if step % 10 != 9 && !last {
            continue;
        }
        let state = pollster::block_on(moving.driver().solver().read_state_f32());
        assert_eq!(state.len(), n * stride, "unexpected state layout");
        let (mut du, mut dp) = (0.0f32, 0.0f32);
        for c in 0..n {
            let (u, v, p) = (
                state[c * stride + u_off],
                state[c * stride + u_off + 1],
                state[c * stride + p_off],
            );
            du = du.max((u - U0.0).abs()).max((v - U0.1).abs());
            dp = dp.max(p.abs());
            if last {
                out.final_u[c] = u;
                out.final_v[c] = v;
                out.final_p[c] = p;
            }
        }
        out.max_du = out.max_du.max(du);
        out.max_dp = out.max_dp.max(dp);
        if (STEPS / 4..STEPS / 2).contains(&step) {
            out.early_du = out.early_du.max(du);
            out.early_dp = out.early_dp.max(dp);
        }
        if step >= 3 * STEPS / 4 {
            out.late_du = out.late_du.max(du);
            out.late_dp = out.late_dp.max(dp);
        }
    }
    out
}

fn print_gcl(label: &str, out: &RunOut) {
    println!(
        "[m5.2-gcl] {label}: n={} max|U-U0|={:.3e} (early {:.3e}, late {:.3e}), \
         max|p|={:.3e} (early {:.3e}, late {:.3e}), SCL={:.3e}, id={:.3e}, \
         skew={:.3e}, rebuilds={} ({STEPS} steps)",
        out.n_cells, out.max_du, out.early_du, out.late_du, out.max_dp, out.early_dp,
        out.late_dp, out.max_scl_defect, out.max_identity_err, out.max_skew, out.rebuilds
    );
}

/// GCL caps for the GPU moving loop (f32 solve). The whole-run caps are ~4× the
/// measured max (GPU f32 scale). A GCL violation or a lost-history BDF2
/// cold-restart COMPOUNDS step-over-step, so the non-compounding check is a
/// late-window (final quarter) absolute cap: a compounding drift rises toward the
/// whole-run cap instead of settling an order below it.
fn assert_gpu_gcl_caps(out: &RunOut) {
    assert!(out.max_identity_err < 1e-11, "f64 identity {:.3e}", out.max_identity_err);
    assert!(out.max_scl_defect < 1e-8, "SCL defect {:.3e}", out.max_scl_defect);
    assert!(out.max_du < 1e-4, "U drift {:.3e} above cap", out.max_du);
    assert!(out.max_dp < 3e-4, "p drift {:.3e} above cap", out.max_dp);
    // Late-window (final quarter) absolute caps — the live no-compounding floor.
    assert!(out.late_du < 1e-5, "late U drift {:.3e}: GCL error compounds", out.late_du);
    assert!(out.late_dp < 3e-5, "late p drift {:.3e}: GCL error compounds", out.late_dp);
    // ...and the late window must not exceed the settled early window (floors keep
    // this ratio bound live, not vacuous).
    assert!(
        out.late_du <= (out.early_du * 2.5).max(1e-5),
        "late U drift {:.3e} vs early {:.3e}: GCL error compounds",
        out.late_du, out.early_du
    );
    assert!(
        out.late_dp <= (out.early_dp * 2.5).max(2e-5),
        "late p drift {:.3e} vs early {:.3e}: GCL error compounds",
        out.late_dp, out.early_dp
    );
}

fn gpu_ctx() -> Option<GpuContext> {
    match pollster::block_on(GpuContext::new(None, None)) {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("[m5.2] no GPU adapter ({e}); skipping GPU moving-mesh gate");
            None
        }
    }
}

// ─── Gate 1: GPU moving-mesh GCL, Euler + BDF2 ───────────────────────────────

#[test]
fn gpu_moving_mesh_gcl_euler() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::remove_var("CFD2_BACKEND");
    let Some(ctx) = gpu_ctx() else { return };
    let out = run_gcl(TimeScheme::Euler, Some(ctx.device.clone()), Some(ctx.queue.clone()));
    print_gcl("gpu/euler", &out);
    assert_gpu_gcl_caps(&out);
}

/// BDF2 proves surgical history preservation: `state_old_old` must survive the
/// per-step topology refresh in place, or BDF2 cold-starts to Euler each step and
/// the drift compounds.
#[test]
fn gpu_moving_mesh_gcl_bdf2() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::remove_var("CFD2_BACKEND");
    let Some(ctx) = gpu_ctx() else { return };
    let out = run_gcl(TimeScheme::BDF2, Some(ctx.device.clone()), Some(ctx.queue.clone()));
    print_gcl("gpu/bdf2", &out);
    assert_gpu_gcl_caps(&out);
}

// ─── Gate 2: CPU vs GPU free-stream / GCL preservation (same moving case) ─────

/// Cross-backend conservation gate: the same prescribed-motion case on CPU (f64
/// solve) and GPU (f32) must agree on the conservation observables. The flow is a
/// uniform (1,0) free stream — an exact discrete fixed point on both backends —
/// so this proves the swept-flux + moving-volume GCL machinery preserves it
/// identically across backends (final-field RMS + both legs' free-stream drift),
/// within an f32 cross-backend tolerance (bits differ, never asserted). NOT a
/// developed-field physics match: any bug that vanishes on uniform flow is
/// invisible here.
#[test]
#[cfg(feature = "cpu")]
fn gpu_vs_cpu_moving_mesh_freestream_gcl() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::remove_var("CFD2_BACKEND");
    let Some(ctx) = gpu_ctx() else { return };
    let gpu = run_gcl(TimeScheme::BDF2, Some(ctx.device.clone()), Some(ctx.queue.clone()));

    // CPU leg: force the CPU backend via the env var (device None).
    std::env::set_var("CFD2_BACKEND", "cpu");
    let cpu = std::panic::catch_unwind(|| run_gcl(TimeScheme::BDF2, None, None));
    std::env::remove_var("CFD2_BACKEND");
    let cpu = cpu.unwrap_or_else(|e| std::panic::resume_unwind(e));

    assert_eq!(cpu.n_cells, gpu.n_cells, "both legs regen the same seed set");
    let n = cpu.n_cells;

    // Field L2 difference (per component) over the final state. Both legs hold a
    // near-uniform (1,0) free stream, so the RMS difference measures the genuine
    // cross-backend solve spread, not a physics disagreement.
    let mut sq = 0.0f64;
    let mut max_abs = 0.0f32;
    for c in 0..n {
        for (a, b) in [
            (cpu.final_u[c], gpu.final_u[c]),
            (cpu.final_v[c], gpu.final_v[c]),
            (cpu.final_p[c], gpu.final_p[c]),
        ] {
            let d = (a - b).abs();
            sq += (d as f64) * (d as f64);
            max_abs = max_abs.max(d);
        }
    }
    let l2 = (sq / (3 * n) as f64).sqrt();
    println!(
        "[m5.2-xbackend] field RMS diff = {l2:.3e}, max|diff| = {max_abs:.3e}; \
         GCL max|U-U0| cpu {:.3e} / gpu {:.3e}, max|p| cpu {:.3e} / gpu {:.3e} (n={n})",
        cpu.max_du, gpu.max_du, cpu.max_dp, gpu.max_dp
    );

    // Both backends must preserve the free stream (physics observable, not bits).
    assert!(cpu.max_du < 5e-3 && gpu.max_du < 5e-3, "a backend lost the free stream");
    // The two free-stream fields agree to f32 cross-backend scale (caps ~two
    // decades above the measured RMS). A free-stream/GCL agreement only: the fixed
    // point is trivial, so only conservation-breaking bugs show.
    assert!(l2 < 5e-4, "cross-backend free-stream RMS {l2:.3e} too large — GCL disagrees");
    assert!(max_abs < 5e-3, "cross-backend field max|diff| {max_abs:.3e} too large");
}

// ─── Gate 3: per-step overhead split at ~20k cells ───────────────────────────

/// Per-step overhead: at ~20k cells, report the moving-mesh overhead
/// (plan+regen+swept+refresh) against the total step time.
#[test]
fn gpu_moving_perf() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::remove_var("CFD2_BACKEND");
    let Some(ctx) = gpu_ctx() else { return };

    // ~20k cells: H ≈ 0.0057 on the unit square.
    let h = 0.0057;
    let (geo, domain) = square();
    let params = test_params(TimeScheme::BDF2);
    let mut cvt = generate_cvt_mesh_with_seeds(&geo, h, h, 1.0, domain, &LloydConfig::default());
    tag_slip_channel(&mut cvt.mesh);
    let n = cvt.mesh.num_cells();

    let mut moving = pollster::block_on(MovingMeshDriver::build(
        cvt,
        &params,
        MeshMotionSpec::Prescribed(swirl),
        &vec![(U0.0 as f64, U0.1 as f64); n],
        &vec![0.0; n],
        Some(ctx.device.clone()),
        Some(ctx.queue.clone()),
    ))
    .expect("moving driver build (perf)");
    assert!(!moving.driver().solver().is_cpu(), "expected the GPU backend");
    moving.set_boundary_retag(Some(tag_slip_channel));
    moving.driver_mut().apply_params(&params);

    // Warm up (first step compiles/populates caches; excluded from timing).
    let (o0, _) = moving.step(false).expect("perf warmup step");
    assert!(o0.diverged.is_none());

    let n_timed = 5usize;
    let (mut plan, mut regen, mut swept, mut refresh) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let mut total = std::time::Duration::ZERO;
    for _ in 0..n_timed {
        let t0 = std::time::Instant::now();
        let (o, s) = moving.step(false).expect("perf step");
        // Force GPU completion so the total reflects real device work.
        let _ = pollster::block_on(moving.driver().solver().read_state_f32());
        total += t0.elapsed();
        assert!(o.diverged.is_none());
        assert!(s.topo_changed, "any real motion drives the topology seam");
        plan += s.plan_ms;
        regen += s.regen_ms;
        swept += s.swept_ms;
        refresh += s.refresh_ms;
    }
    let inv = 1.0 / n_timed as f32;
    let total_ms = total.as_secs_f32() * 1000.0 * inv;
    let (plan, regen, swept, refresh) = (plan * inv, regen * inv, swept * inv, refresh * inv);
    let overhead = plan + regen + swept + refresh;
    println!(
        "[m5.2-perf] n={n}: total step {total_ms:.2} ms | plan {plan:.2} + regen {regen:.2} \
         + swept {swept:.2} + refresh {refresh:.2} = overhead {overhead:.2} ms \
         (refresh {:.0}% of step, {:.0}% of overhead)",
        100.0 * refresh / total_ms.max(1e-6),
        100.0 * refresh / overhead.max(1e-6),
    );
    // Do-no-harm regression gate. Pin refresh below the larger of the two
    // size-scaling mesh passes (regen / swept-flux): a reintroduced LA-stack
    // recompile is a size-independent ~10 ms added to `refresh` while regen/swept
    // are unchanged, pushing refresh above them. All three are GPU passes on the
    // same adapter, so the bound scales with the machine (portable, not an
    // absolute wall-clock cap); the 1.25× cushion absorbs run-to-run noise.
    let mesh_pass = regen.max(swept);
    assert!(
        refresh < mesh_pass * 1.25,
        "refresh {refresh:.2} ms exceeds 1.25× the larger mesh pass \
         (regen {regen:.2} / swept {swept:.2}) — the LA pipeline cache regressed \
         (recompile-per-refresh reintroduced?)"
    );
    assert!(
        refresh <= total_ms,
        "refresh {refresh:.2} ms exceeds the whole step {total_ms:.2} ms — stage-1 regressed"
    );
}
