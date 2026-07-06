//! FLOW-adaptive timestep gates for the moving-mesh (ALE) path.
//!
//! `MovingMeshDriver::set_adaptive_dt(Some(cfl))` re-computes the pinned dt
//! every step as `cfl · min_h / (max(|U|, |U_in|) + c_eos)` — INSIDE the GCL
//! dt handshake (pinned before the swept-flux closure), growth-limited to
//! 1.2× per committed step, still capped by the mesh-motion CFL. The BDF2
//! lowering carries variable-step coefficients (`r = dt/dt_old`), so a
//! per-step dt sequence is formally consistent; these gates prove it
//! numerically:
//!
//! 1. **Free-stream exactness under adaptive dt** — a uniform stream on a
//!    swirling (Prescribed) mesh must remain an exact discrete fixed point
//!    while the controller re-pins dt each step. Any GCL/handshake breakage
//!    under a varying dt shows up as stream drift.
//! 2. **Obstacle wake tracks the flow** — as the wake accelerates, the
//!    pinned dt must shrink accordingly, never grow faster than 1.2×/step,
//!    and compose with every-N flow adaptation (solver rebuilds).
//! 3. **All-Mach smoke** — the controller on the compressible family (its
//!    constant-EOS sound speed is 0, matching the static branch: the
//!    acoustic stiffness is handled by psi_precond, not the timestep).

#![cfg(all(feature = "meshgen", feature = "cpu"))]

use cfd2::meshgen::meshless::generate_cvt_mesh_with_seeds;
use cfd2::meshgen::{ChannelWithObstacle, LloydConfig, RectangularChannel};
use cfd2::sim::{MeshMotionSpec, MovingMeshDriver, RuntimeParams};
use cfd2::solver::mesh::{BoundaryType, Mesh};
use cfd2::solver::model::allmach_pressure_ale_model;
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
const DT0: f32 = 0.005;
const PERIOD: f64 = 80.0 * DT0 as f64;

fn base_params(dt: f32, time_scheme: TimeScheme, psi: f32) -> RuntimeParams {
    RuntimeParams {
        adaptive_dt: false,
        target_cfl: 0.9,
        requested_dt: dt,
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
        inlet_velocity: 1.0,
        density: 1.0,
        viscosity: 1e-2,
        eos: EosSpec::Constant,
        compressibility_psi: psi,
        outlet_back_pressure: 0.0,
        pressure_inlet: false,
        inlet_pressure: 0.0,
    }
}

/// Slip channel: Inlet left, Outlet right, SlipWall top+bottom (a uniform
/// horizontal flow satisfies all three). Re-stamped on every regen.
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

/// Flip-free interior swirl (the `moving_mesh_gcl_test` motion): rotate each
/// seed about the centre by a boundary-vanishing bump.
fn swirl(p: [f64; 2], t: f64) -> [f64; 2] {
    let (cx, cy) = (0.5 * LX, 0.5 * LY);
    let bump = (std::f64::consts::PI * p[0] / LX).sin().powi(2)
        * (std::f64::consts::PI * p[1] / LY).sin().powi(2);
    let theta = 0.03 * (2.0 * std::f64::consts::PI * t / PERIOD).sin() * bump;
    let (dx, dy) = (p[0] - cx, p[1] - cy);
    let (c, s) = (theta.cos(), theta.sin());
    [cx + c * dx - s * dy, cy + s * dx + c * dy]
}

/// Gate 1: adaptive dt + moving mesh must keep a uniform free stream an
/// EXACT discrete fixed point. The mesh swirls every step and the stream
/// must not drift. The configured dt is 1e-3 — ~6x BELOW the CFL law's
/// `0.15 · min_h / 1.0` (~5.8e-3 on this mesh), so an inert controller
/// (falling back to the configured dt) lands OUTSIDE the CFL-law window
/// asserted below and cannot pass silently.
#[test]
fn movingmesh_adaptive_dt_freestream_exact() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("CFD2_BACKEND", "cpu");
    let result = std::panic::catch_unwind(|| {
        let geo = RectangularChannel {
            length: LX,
            height: LY,
        };
        let domain = Vector2::new(LX, LY);
        let params = base_params(0.001, TimeScheme::BDF2, 0.0);
        let mut cvt =
            generate_cvt_mesh_with_seeds(&geo, H, H, 1.0, domain, &LloydConfig::default());
        tag_slip_channel(&mut cvt.mesh);
        let n0 = cvt.mesh.num_cells();
        let mut moving = pollster::block_on(MovingMeshDriver::build(
            cvt,
            &params,
            MeshMotionSpec::Prescribed(swirl),
            &vec![(1.0f64, 0.0); n0],
            &vec![0.0; n0],
            None,
            None,
        ))
        .expect("adaptive-dt freestream driver build");
        moving.driver_mut().apply_params(&params);
        moving.set_boundary_retag(Some(tag_slip_channel));
        moving.set_adaptive_dt(Some(0.15));

        let layout = moving.driver().solver().model().state_layout.clone();
        let stride = layout.stride() as usize;
        let u_off = layout.offset_for("U").expect("U") as usize;
        let mut worst = 0.0f64;
        let mut prev_dt: Option<f64> = None;
        for step in 0..30 {
            let (outcome, stats) = moving
                .step(false)
                .unwrap_or_else(|e| panic!("adaptive-dt freestream step {step}: {e}"));
            assert!(outcome.diverged.is_none(), "step {step} diverged");
            assert!(
                stats.scl_defect < 1e-5,
                "step {step}: SCL defect {:.3e} under adaptive dt",
                stats.scl_defect
            );
            // The controller engaged: dt obeys the CFL law
            // `0.15 * min_h / max|U|` with `min_h = sqrt(vol_min)` and
            // `max|U| = 1` on the free stream (factor-2 window: the pin
            // used the pre-step mesh's min cell, stats carry the post-step
            // one, and the mesh-CFL cap may shave the peak swirl steps).
            let expected = 0.15 * stats.vol_min.sqrt();
            assert!(
                stats.dt > 0.5 * expected && stats.dt < 2.0 * expected,
                "step {step}: dt {:.4e} vs CFL-law {expected:.4e} — the adaptive \
                 controller is not in charge of the pinned dt",
                stats.dt
            );
            // Growth-limit contract (shrink unlimited).
            if let Some(prev) = prev_dt {
                assert!(
                    stats.dt <= prev * 1.2 + 1e-12,
                    "step {step}: dt {:.4e} grew faster than 1.2x from {:.4e}",
                    stats.dt,
                    prev
                );
            }
            prev_dt = Some(stats.dt);
            let state = pollster::block_on(moving.driver().solver().read_state_f32());
            let n = moving.mesh().num_cells();
            for c in 0..n {
                let du = (state[c * stride + u_off] as f64 - 1.0).abs();
                let dv = (state[c * stride + u_off + 1] as f64).abs();
                worst = worst.max(du.max(dv));
            }
        }
        eprintln!(
            "[adaptive-dt freestream] 30 swirl steps, dt {:.4e}, worst stream drift {worst:.3e}",
            prev_dt.unwrap()
        );
        assert!(
            worst < 1e-3,
            "free stream drifted {worst:.3e} under adaptive dt — variable-dt GCL breakage"
        );
    });
    std::env::remove_var("CFD2_BACKEND");
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

/// Gate 2: on the shedding obstacle wake the pinned dt must TRACK the flow
/// (hold the realized CFL at the target as the wake accelerates and the
/// adaptation refines the mesh), respect the 1.2x/step growth limit, and
/// compose with flow adaptation (solver rebuilds at resize events).
///
/// `target_cfl = 0.1` is DISCRIMINATING under FlowCoupled motion: the
/// mesh-motion cap alone (`0.2 * min_h / w_max` with seed speed `w_max`
/// below the flow speed) can only hold the realized flow CFL at >= 0.2, and
/// the configured fallback (dt = 0.01) starts at CFL ~0.21 — both fail the
/// per-step `CFL <= 0.15` bound that the working controller satisfies by
/// construction.
#[test]
fn movingmesh_adaptive_dt_obstacle_tracks_flow() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("CFD2_BACKEND", "cpu");
    let result = std::panic::catch_unwind(|| {
        let geo = ChannelWithObstacle {
            length: 2.0,
            height: 1.0,
            obstacle_center: Point2::new(0.6, 0.51),
            obstacle_radius: 0.1,
        };
        let domain = Vector2::new(2.0, 1.0);
        let mut params = base_params(0.01, TimeScheme::BDF2, 0.0);
        params.viscosity = 1.33e-3;
        let cvt =
            generate_cvt_mesh_with_seeds(&geo, 0.05, 0.05, 1.0, domain, &LloydConfig::default());
        let n0 = cvt.mesh.num_cells();
        let mut moving = pollster::block_on(MovingMeshDriver::build(
            cvt,
            &params,
            MeshMotionSpec::FlowCoupled {
                regularization: 0.5,
            },
            &vec![(1.0f64, 0.0); n0],
            &vec![0.0; n0],
            None,
            None,
        ))
        .expect("adaptive-dt obstacle driver build");
        moving.driver_mut().apply_params(&params);
        moving.set_adaptive_dt(Some(0.1));
        // Compose with flow adaptation: rebuilds must not corrupt the
        // controller (configured/pinned dt survive the resize).
        moving.set_adaptive_sizing(5);
        moving.set_adaptive_sizing_band(Some((0.02, 0.06)));
        moving.set_smoothing(5, 1, 0.5);

        let mut dts: Vec<f64> = Vec::new();
        let mut umax_at: Vec<f64> = Vec::new();
        let layout = moving.driver().solver().model().state_layout.clone();
        let stride = layout.stride() as usize;
        let u_off = layout.offset_for("U").expect("U") as usize;
        let mut resizes = 0usize;
        for step in 0..80 {
            let (outcome, stats) = moving
                .step(false)
                .unwrap_or_else(|e| panic!("adaptive-dt obstacle step {step}: {e}"));
            assert!(outcome.diverged.is_none(), "step {step} diverged");
            if let Some(prev) = dts.last() {
                assert!(
                    stats.dt <= prev * 1.2 + 1e-12,
                    "step {step}: dt {:.4e} grew faster than 1.2x from {prev:.4e}",
                    stats.dt
                );
            }
            dts.push(stats.dt);
            resizes += usize::from(stats.cells_born > 0 || stats.cells_killed > 0);
            let state = pollster::block_on(moving.driver().solver().read_state_f32());
            let mesh = moving.mesh();
            let n = mesh.num_cells();
            let (mut umax, mut cell_cfl) = (0.0f64, 0.0f64);
            for c in 0..n {
                let w =
                    (state[c * stride + u_off] as f64).hypot(state[c * stride + u_off + 1] as f64);
                umax = umax.max(w);
                let h = mesh.cell_vol[c].max(1e-30).sqrt();
                cell_cfl = cell_cfl.max(stats.dt * w / h);
            }
            assert!(
                umax.is_finite() && umax < 10.0,
                "step {step}: |U| {umax:.3e}"
            );
            umax_at.push(umax);
            // THE discriminator: the realized PER-CELL flow CFL held at the
            // target — the controller law is `dt = cfl / max_i(|U_i|/h_i)`
            // (not the conservative min_h/max|U|, which under-runs the true
            // CFL when the finest cells sit in slow near-wall bands). Skip
            // the cold start; 1.5x margin absorbs the pin-time vs post-step
            // mesh/velocity drift within one step.
            if step >= 10 {
                assert!(
                    cell_cfl <= 0.15,
                    "step {step}: realized per-cell CFL {cell_cfl:.3} exceeds 1.5x the \
                     0.1 target — the flow-CFL controller is not in charge of the pinned dt"
                );
            }
        }
        let (dt_min, dt_max) = dts.iter().fold((f64::INFINITY, 0.0f64), |(lo, hi), &d| {
            (lo.min(d), hi.max(d))
        });
        eprintln!(
            "[adaptive-dt obstacle] 80 steps, dt range [{dt_min:.4e}, {dt_max:.4e}], \
             max|U| {:.3}, {resizes} resizes",
            umax_at.last().unwrap()
        );
        // The wake accelerates well past the inlet speed and the band
        // refines the mesh, so dt must have moved measurably over the run.
        assert!(
            dt_max / dt_min > 1.2,
            "dt never adapted: range [{dt_min:.4e}, {dt_max:.4e}]"
        );
        assert!(
            resizes > 0,
            "flow adaptation never fired alongside adaptive dt"
        );
    });
    std::env::remove_var("CFD2_BACKEND");
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

/// Gate 3: all-Mach family smoke under adaptive dt (Frozen mesh with
/// smoothing): stable, finite, controller engaged.
#[test]
fn movingmesh_adaptive_dt_allmach_smoke() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("CFD2_BACKEND", "cpu");
    let result = std::panic::catch_unwind(|| {
        let geo = ChannelWithObstacle {
            length: 2.0,
            height: 1.0,
            obstacle_center: Point2::new(0.6, 0.5),
            obstacle_radius: 0.12,
        };
        let domain = Vector2::new(2.0, 1.0);
        let mut params = base_params(DT0, TimeScheme::Euler, 1.0e-4);
        params.inlet_velocity = 0.4;
        let cvt = generate_cvt_mesh_with_seeds(&geo, H, H, 1.0, domain, &LloydConfig::default());
        let n = cvt.mesh.num_cells();
        let mut moving = pollster::block_on(MovingMeshDriver::build_with_model(
            cvt,
            allmach_pressure_ale_model().expect("allmach ale model"),
            &params,
            MeshMotionSpec::Frozen,
            &vec![(0.4f64, 0.0); n],
            &vec![0.0; n],
            None,
            None,
        ))
        .expect("adaptive-dt allmach driver build");
        moving.driver_mut().apply_params(&params);
        moving.set_adaptive_dt(Some(0.5));
        moving.set_smoothing(1, 1, 0.5);

        let layout = moving.driver().solver().model().state_layout.clone();
        let stride = layout.stride() as usize;
        let u_off = layout.offset_for("U").expect("U") as usize;
        let mut max_u = 0.0f64;
        let mut last_dt = 0.0f64;
        for step in 0..25 {
            let (outcome, stats) = moving
                .step(false)
                .unwrap_or_else(|e| panic!("adaptive-dt allmach step {step}: {e}"));
            assert!(outcome.diverged.is_none(), "step {step} diverged");
            assert!(
                stats.scl_defect < 1e-5,
                "step {step}: SCL {:.3e}",
                stats.scl_defect
            );
            last_dt = stats.dt;
            let state = pollster::block_on(moving.driver().solver().read_state_f32());
            let nc = moving.mesh().num_cells();
            for c in 0..nc {
                let (ux, uy) = (
                    state[c * stride + u_off] as f64,
                    state[c * stride + u_off + 1] as f64,
                );
                assert!(
                    ux.is_finite() && uy.is_finite(),
                    "step {step}: non-finite U"
                );
                max_u = max_u.max(ux.hypot(uy));
            }
        }
        eprintln!("[adaptive-dt allmach] 25 steps, final dt {last_dt:.4e}, max|U| {max_u:.3}");
        assert!(
            max_u < 10.0 * params.inlet_velocity as f64,
            "max|U| {max_u:.3e} unbounded"
        );
        // Constant-EOS sound speed is 0 => the controller is advective:
        // dt ~ 0.5*min_h/max(|U|,0.4) — well above the configured 5e-3.
        assert!(
            last_dt > 1.5 * DT0 as f64,
            "dt {last_dt:.4e}: the adaptive controller never engaged"
        );
    });
    std::env::remove_var("CFD2_BACKEND");
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}
