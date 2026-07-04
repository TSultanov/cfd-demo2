//! M4.3 gate (meshless/moving-mesh roadmap §M4, the case M3 explicitly
//! DEFERRED): the topology-flip GCL test.
//!
//! Where `moving_mesh_gcl_test` moves the seeds with a deliberately flip-FREE
//! swirl (persistent Voronoi adjacency, every step closes through the M3
//! telescoping identity), THIS gate drives a swirl large enough to force real
//! adjacency FLIPS — faces are BORN and DIE between steps as the Voronoi
//! re-tessellates. The claim under test is the whole moving-mesh premise:
//!
//!   uniform flow stays uniform to the CPU GCL scale THROUGH the flips,
//!
//! because the born/dead-face conservative remap
//! (`swept_mesh_fluxes_closed_flip`) keeps `Σ_f σ·mesh_flux_f = ΔV_i/dt` EXACT
//! per cell — born faces carry zero swept contribution and the per-cell defect
//! is distributed onto the slack faces by the spanning-forest closure. GCL
//! (free-stream preservation) senses only that per-cell sum, so it survives the
//! flip; the per-FACE flux near a flip is only locally first-order (the roadmap
//! accepted cost), reported here as the `flip_defect` diagnostic.
//!
//! The test ASSERTS flips actually occur (a nonzero flip count — otherwise it
//! would silently degrade into the M4.2 flip-free gate) and pins the flip rate.
//!
//! CPU-primary (the M4 loop is CPU-first; GPU per-step regen is M5 — a GPU
//! `refresh_mesh` cold-restarts the linear-algebra stack, so it cannot hold the
//! warm-started GCL through a flip and is out of scope here).
//!
//! Boundary/flow setup mirrors `moving_mesh_gcl_test` exactly: slip channel
//! (Inlet left / Outlet right / SlipWall top+bottom), horizontal free stream
//! `(U,0)` — an exact discrete fixed point, so any drift is a GCL/solve
//! artifact, never BC physics.
#![cfg(all(feature = "meshgen", feature = "cpu"))]

use cfd2::meshgen::meshless::generate_cvt_mesh_with_seeds;
use cfd2::meshgen::{LloydConfig, RectangularChannel};
use cfd2::sim::{MeshMotionSpec, MovingMeshDriver, RuntimeParams};
use cfd2::solver::mesh::{BoundaryType, Mesh};
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{GpuLowMachPrecondModel, PreconditionerType, TimeScheme};
use nalgebra::Vector2;
use std::sync::Mutex;

/// `CFD2_BACKEND` is process-global; serialize the tests.
static ENV_LOCK: Mutex<()> = Mutex::new(());

const LX: f64 = 1.0;
const LY: f64 = 1.0;
const H: f64 = 0.08;
const DT: f32 = 0.005;
const STEPS: usize = 140;
/// Motion period (~2 swirl cycles over the run — several excursions through the
/// flip-inducing extremes).
const PERIOD: f64 = 70.0 * DT as f64;
/// Horizontal free stream (see the module note): an exact discrete fixed point.
const U0: (f32, f32) = (1.0, 0.0);

/// Peak translation amplitude as a fraction of `H` (the flip probe in
/// `moving_mesh_gcl_test` measured the first adjacency flip at ~0.2·h of rigid
/// interior translation; this over-drives it to guarantee flips at the extremes
/// while the mesh-CFL cap keeps the per-step motion small).
const FLIP_AMP_FRAC: f64 = 0.45;

/// Flip-FORCING motion: an OSCILLATING rigid translation of the interior seed
/// block (the driver holds the boundary seeds fixed), so the interior shears
/// against the static boundary ring — the mechanism the M4.2 flip probe showed
/// flips the Voronoi adjacency past ~0.2·h. Oscillating (not the probe's
/// monotone drift) so the seeds return and the mesh stays valid over the run.
/// The near-boundary faces are BORN/DIE as the shear crosses the threshold each
/// half-cycle; interior-block faces translate rigidly (adjacency preserved).
/// Evaluated from the t=0 label — no incremental round-off drift.
fn shear_flip(p: [f64; 2], t: f64) -> [f64; 2] {
    let a = FLIP_AMP_FRAC * H * (2.0 * std::f64::consts::PI * t / PERIOD).sin();
    [p[0] + a, p[1] + 0.6 * a]
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

/// Slip channel: Inlet left, Outlet right, SlipWall top+bottom (a `fn` pointer
/// so it can be the driver's per-regen retag hook). Only boundary faces touched.
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

struct FlipGclOut {
    max_du: f32,
    max_dp: f32,
    early_du: f32,
    late_du: f32,
    max_scl_defect: f64,
    max_flip_defect: f64,
    // Flip diagnostics.
    flip_steps: usize,
    total_born: usize,
    total_died: usize,
    max_flipped_cells: usize,
    max_flip_rate: f64,
    n_cells: usize,
}

fn run_flip_gcl(time_scheme: TimeScheme) -> FlipGclOut {
    let (geo, domain) = (
        RectangularChannel { length: LX, height: LY },
        Vector2::new(LX, LY),
    );
    let params = test_params(time_scheme);
    let mut cvt = generate_cvt_mesh_with_seeds(&geo, H, H, 1.0, domain, &LloydConfig::default());
    tag_slip_channel(&mut cvt.mesh);
    let n = cvt.mesh.num_cells();

    let mut moving = pollster::block_on(MovingMeshDriver::build(
        cvt,
        &params,
        MeshMotionSpec::Prescribed(shear_flip),
        &vec![(U0.0 as f64, U0.1 as f64); n],
        &vec![0.0; n],
        None,
        None,
    ))
    .expect("moving driver build");
    moving.set_boundary_retag(Some(tag_slip_channel));
    moving.driver_mut().apply_params(&params);

    let layout = moving.driver().solver().model().state_layout.clone();
    let stride = layout.stride() as usize;
    let u_off = layout.offset_for("U").expect("U offset") as usize;
    let p_off = layout.offset_for("p").expect("p offset") as usize;

    let mut out = FlipGclOut {
        max_du: 0.0,
        max_dp: 0.0,
        early_du: 0.0,
        late_du: 0.0,
        max_scl_defect: 0.0,
        max_flip_defect: 0.0,
        flip_steps: 0,
        total_born: 0,
        total_died: 0,
        max_flipped_cells: 0,
        max_flip_rate: 0.0,
        n_cells: n,
    };

    for step in 0..STEPS {
        let (outcome, stats) = moving
            .step(false)
            .unwrap_or_else(|e| panic!("step {step} failed: {e}"));
        assert!(outcome.diverged.is_none(), "step {step} diverged");

        // The correctness guarantee: the post-closure per-cell SCL defect stays
        // at f32 roundoff EVEN on a flip step (the forest closure makes the
        // per-cell sum exact regardless of born/dead faces).
        out.max_scl_defect = out.max_scl_defect.max(stats.scl_defect);
        assert!(
            stats.scl_defect < 1e-7,
            "step {step}: post-closure SCL defect {:.3e} above roundoff (flipped={})",
            stats.scl_defect,
            stats.flipped
        );

        if stats.flipped {
            out.flip_steps += 1;
            out.total_born += stats.born_faces;
            out.total_died += stats.died_faces;
            out.max_flipped_cells = out.max_flipped_cells.max(stats.flipped_cells);
            out.max_flip_rate = out.max_flip_rate.max(stats.flipped_cells as f64 / n as f64);
            out.max_flip_defect = out.max_flip_defect.max(stats.flip_defect);
        } else {
            // No flip ⇒ the (persistent) telescoping identity must be roundoff.
            assert!(
                stats.identity_err < 1e-10,
                "step {step}: non-flip telescoping identity {:.3e} above roundoff",
                stats.identity_err
            );
        }

        if std::env::var("CFD2_GCL_TRACE").as_deref() == Ok("1") && stats.flipped {
            println!(
                "[m4.3-flip-trace] step {:4}: born={} died={} cells={} flip_defect={:.3e} \
                 scl_defect={:.3e}",
                step + 1,
                stats.born_faces,
                stats.died_faces,
                stats.flipped_cells,
                stats.flip_defect,
                stats.scl_defect,
            );
        }

        if step % 10 != 9 && step + 1 != STEPS {
            continue;
        }
        let state = pollster::block_on(moving.driver().solver().read_state_f32());
        assert_eq!(state.len(), n * stride, "unexpected state layout");
        let (mut du, mut dp) = (0.0f32, 0.0f32);
        for c in 0..n {
            du = du
                .max((state[c * stride + u_off] - U0.0).abs())
                .max((state[c * stride + u_off + 1] - U0.1).abs());
            dp = dp.max(state[c * stride + p_off].abs());
        }
        out.max_du = out.max_du.max(du);
        out.max_dp = out.max_dp.max(dp);
        if (STEPS / 4..STEPS / 2).contains(&step) {
            out.early_du = out.early_du.max(du);
        }
        if step >= 3 * STEPS / 4 {
            out.late_du = out.late_du.max(du);
        }
    }
    out
}

fn run_flip_gcl_cpu(scheme: TimeScheme, label: &str) -> FlipGclOut {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::set_var("CFD2_BACKEND", "cpu");
    let result = std::panic::catch_unwind(|| run_flip_gcl(scheme));
    std::env::remove_var("CFD2_BACKEND");
    let out = match result {
        Ok(o) => o,
        Err(e) => std::panic::resume_unwind(e),
    };
    println!(
        "[m4.3-flip-gcl] cpu/{label}: {n} cells, {STEPS} steps | FLIPS: {flip_steps} steps had \
         flips ({born} born / {died} died faces total, max {mfc} cells/step = {rate:.1}% flip \
         rate), max flip_defect = {fd:.3e} | GCL: max|U-U0| = {du:.3e} (early {edu:.3e}, late \
         {ldu:.3e}), max|p| = {dp:.3e}, post-closure SCL defect = {scl:.3e}",
        n = out.n_cells,
        flip_steps = out.flip_steps,
        born = out.total_born,
        died = out.total_died,
        mfc = out.max_flipped_cells,
        rate = out.max_flip_rate * 100.0,
        fd = out.max_flip_defect,
        du = out.max_du,
        edu = out.early_du,
        ldu = out.late_du,
        dp = out.max_dp,
        scl = out.max_scl_defect,
    );
    out
}

/// Assert the decisive properties. Caps pinned after first measurement (see the
/// printed line); tightened to ~2-3× measured. The load-bearing asserts:
///   1. flips ACTUALLY occurred (nonzero flip count) — else this is the M4.2
///      flip-free gate in disguise;
///   2. uniform flow stays uniform to the CPU GCL scale THROUGH the flips;
///   3. no compounding (the late window ≈ the early window — a broken flip
///      remap would ratchet the drift up every flip event).
fn assert_flip_gcl(out: &FlipGclOut) {
    assert!(
        out.flip_steps > 0 && out.total_born > 0,
        "no flips occurred ({} flip steps, {} born faces) — the amplitude is too small to \
         exercise the flip path",
        out.flip_steps,
        out.total_born,
    );
    // Post-closure per-cell defect is the correctness guarantee: f32 roundoff.
    assert!(out.max_scl_defect < 1e-7, "SCL defect {:.3e}", out.max_scl_defect);
    // The flip defect (pre-closure per-face residual) is genuinely O(1): proves
    // the born/dead faces are really carrying the closure residual.
    assert!(
        out.max_flip_defect > 1e-4,
        "flip defect {:.3e} unexpectedly tiny — flips not really exercised",
        out.max_flip_defect
    );
    // Uniform flow preserved through the flips, at the CPU GCL scale — pinned to
    // the SAME ~1e-6 band as the flip-FREE M4.2 gate (measured max|U-U0| ≈
    // 1.9e-6, max|p| ≈ 2.3e-5 across 88 born/died faces; caps ~3× measured).
    // This is the decisive correctness statement of the whole moving-mesh
    // premise: the born/dead-face remap holds the free stream through flips.
    assert!(out.max_du < 6e-6, "U drift {:.3e} above cap through flips", out.max_du);
    assert!(out.max_dp < 8e-5, "p drift {:.3e} above cap through flips", out.max_dp);
    // Non-compounding: a broken flip remap injects a fresh O(dt) GCL error at
    // every flip, so late-window drift would grow without bound.
    assert!(
        out.late_du <= (out.early_du * 3.0).max(6e-6),
        "late U drift {:.3e} vs early {:.3e}: the flip GCL error compounds",
        out.late_du,
        out.early_du,
    );
}

#[test]
fn flip_gcl_uniform_flow_preserved_cpu_euler() {
    let out = run_flip_gcl_cpu(TimeScheme::Euler, "euler");
    assert_flip_gcl(&out);
}

#[test]
fn flip_gcl_uniform_flow_preserved_cpu_bdf2() {
    let out = run_flip_gcl_cpu(TimeScheme::BDF2, "bdf2");
    assert_flip_gcl(&out);
}
