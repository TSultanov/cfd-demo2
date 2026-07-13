//! All-Mach compressible ALE (moving-mesh) gates.
//!
//! The `allmach_pressure_ale` model is the moving-mesh variant of the
//! pressure-based all-Mach compressible solver: `div(phi,U).bounded()` and
//! `div_flux(phi,p)` are `.with_mesh_relative()`, so assembly consumes
//! `phi_rel = phi - rho_f * mesh_fluxes[face]` with a variable-density face
//! density `rho_f = 0.5*(rho[idx]+rho[other])`. Two gates:
//!
//! 1. **Zero-flux byte-identity** — with `mesh_fluxes == 0` and equal volume
//!    history (a static mesh), `allmach_pressure_ale` reproduces the static
//!    `allmach_pressure` *bitwise*: the extra ALE terms are `x - rho_f*0.0` and
//!    `+rho_ref*0.0`, IEEE identities. This is the do-no-harm gate — the new
//!    variable-density `rho_f` path must not perturb the constant-flux solve.
//!
//! 2. **Compressible free-stream GCL** — a uniform compressible free stream
//!    (`U=(U0,0)`, `p=0` gauge, `rho=rho_ref`, `psi>0`) is an exact discrete
//!    fixed point on a *moving* Voronoi mesh (interior seeds swirl every step,
//!    regenerated through the full swept-flux + refresh loop). Any drift in `U`,
//!    `p` **or** `rho` is an ALE/GCL artifact. The barotropic volume source is
//!    `rho_ref*(V^{n+1}-V^n)/dt`, which cancels the mesh-flux part of `Σ φ_rel`
//!    at the reference density; at `p=0` the `ddt(psi_precond,p)` term vanishes,
//!    so the compressible free stream is preserved to the same scale as the
//!    incompressible one.
#![cfg(all(feature = "meshgen", feature = "cpu"))]

use cfd2::meshgen::meshless::generate_cvt_mesh_with_seeds;
use cfd2::meshgen::{ChannelWithObstacle, LloydConfig, RectangularChannel};
use nalgebra::Point2;
use cfd2::sim::{DriverBuild, MeshMotionSpec, MovingMeshDriver, RuntimeParams, SolverDriver};
use cfd2::solver::mesh::{BoundaryType, Mesh};
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::model::{
    allmach_pressure_ale_model, allmach_pressure_model, allmach_thermal_ale_model,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{GpuLowMachPrecondModel, PreconditionerType, TimeScheme};
use nalgebra::Vector2;
use std::sync::Mutex;

/// `CFD2_BACKEND` / `CFD2_CPU_ENGINE` are process-global; serialize the tests.
static ENV_LOCK: Mutex<()> = Mutex::new(());

const LX: f64 = 1.0;
const LY: f64 = 1.0;
const H: f64 = 0.08;
const DT: f32 = 0.005;
const STEPS: usize = 120;
const PERIOD: f64 = 80.0 * DT as f64;
/// Horizontal free stream (see `moving_mesh_gcl_test`): uniform `(U0,0)` is an
/// exact discrete fixed point on the slip channel.
const U0: (f32, f32) = (1.0, 0.0);
/// A genuinely compressible compressibility: `psi = 1/c^2 = 1e-4` ⇒ `c = 100`,
/// so at `|U| = 1` the free-stream Mach number is `0.01` (all-Mach regime). Big
/// enough that the `ddt(psi,p)` / variable-`rho_f` machinery is exercised, small
/// enough to stay well-conditioned.
const PSI: f32 = 1.0e-4;
const RHO_REF: f32 = 1.0;

/// Flip-free interior swirl (identical to `moving_mesh_gcl_test::swirl`): rotate
/// each seed about the centre by a boundary-vanishing bump, smooth and small
/// enough to never flip the Voronoi adjacency.
fn swirl(p: [f64; 2], t: f64) -> [f64; 2] {
    let (cx, cy) = (0.5 * LX, 0.5 * LY);
    let bump = (std::f64::consts::PI * p[0] / LX).sin().powi(2)
        * (std::f64::consts::PI * p[1] / LY).sin().powi(2);
    let theta = 0.03 * (2.0 * std::f64::consts::PI * t / PERIOD).sin() * bump;
    let (dx, dy) = (p[0] - cx, p[1] - cy);
    let (c, s) = (theta.cos(), theta.sin());
    [cx + c * dx - s * dy, cy + s * dx + c * dy]
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

fn test_params(time_scheme: TimeScheme, psi: f32) -> RuntimeParams {
    RuntimeParams {
        filter_sigma: 0.0,
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
        density: RHO_REF,
        viscosity: 1e-2,
        eos: EosSpec::Constant,
        compressibility_psi: psi,
        outlet_back_pressure: 0.0,
        allmach_precond_uref_min: 0.2,
        pressure_inlet: false,
        inlet_pressure: 0.0,
    }
}

// ---------------------------------------------------------------------------
// Gate 1: zero-flux byte-identity (allmach_pressure_ale == allmach_pressure).
// ---------------------------------------------------------------------------

fn state_bits(driver: &SolverDriver) -> Vec<u32> {
    pollster::block_on(driver.solver().read_state_f32())
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn build_static_driver(mesh: &Mesh, ale: bool, time_scheme: TimeScheme) -> SolverDriver {
    let params = test_params(time_scheme, PSI);
    let n = mesh.num_cells();
    let model = if ale {
        allmach_pressure_ale_model().expect("ale model")
    } else {
        allmach_pressure_model().expect("static model")
    };
    let DriverBuild { mut driver, .. } = pollster::block_on(SolverDriver::build(
        mesh,
        model,
        &params,
        &vec![(U0.0 as f64, 0.0); n],
        &vec![0.0; n],
        None,
        None,
    ))
    .expect("driver build");
    driver.apply_params(&params);
    driver
}

fn assert_allmach_zero_flux_equivalence(engine: &str, time_scheme: TimeScheme) {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::set_var("CFD2_BACKEND", "cpu");
    std::env::set_var("CFD2_CPU_ENGINE", engine);
    let result = std::panic::catch_unwind(|| {
        let (geo, domain) = square();
        let cvt = generate_cvt_mesh_with_seeds(&geo, H, H, 1.0, domain, &LloydConfig::default());
        let mesh = cvt.mesh;

        let mut base = build_static_driver(&mesh, false, time_scheme);
        let mut ale = build_static_driver(&mesh, true, time_scheme);
        assert!(base.solver().is_cpu() && ale.solver().is_cpu(), "expected CPU backend");

        const N: usize = 40;
        for step in 0..N {
            let out_base = base.step(false);
            let out_ale = ale.step(false);
            assert!(
                out_base.diverged.is_none() && out_ale.diverged.is_none(),
                "[{engine}] step {step} diverged (base {:?}, ale {:?})",
                out_base.diverged,
                out_ale.diverged
            );
            let (bb, ba) = (state_bits(&base), state_bits(&ale));
            assert_eq!(bb.len(), ba.len(), "state length mismatch");
            let ndiff = bb.iter().zip(&ba).filter(|(a, b)| a != b).count();
            assert_eq!(
                ndiff,
                0,
                "[{engine}] step {step}: allmach_pressure_ale with zero mesh_fluxes diverged \
                 bitwise from static allmach_pressure: {ndiff}/{} slots differ (max |diff| {:.3e})",
                bb.len(),
                bb.iter()
                    .zip(&ba)
                    .map(|(&a, &b)| (f32::from_bits(a) - f32::from_bits(b)).abs())
                    .fold(0.0f32, f32::max),
            );
        }
        println!("[allmach-ale-zero-flux] {engine}/{time_scheme:?}: ale == static bitwise over {N} steps");
    });
    std::env::remove_var("CFD2_BACKEND");
    std::env::remove_var("CFD2_CPU_ENGINE");
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

#[test]
fn allmach_ale_zero_flux_byte_identical_cpu_interpreter() {
    assert_allmach_zero_flux_equivalence("interpreter", TimeScheme::BDF2);
}

#[test]
fn allmach_ale_zero_flux_byte_identical_cpu_transpiled() {
    assert_allmach_zero_flux_equivalence("transpiled", TimeScheme::BDF2);
}

#[test]
fn allmach_ale_zero_flux_byte_identical_cpu_euler() {
    assert_allmach_zero_flux_equivalence("interpreter", TimeScheme::Euler);
}

// ---------------------------------------------------------------------------
// Gate 2: compressible free-stream GCL through the full moving-mesh loop.
// ---------------------------------------------------------------------------

struct FreestreamOut {
    max_du: f32,
    max_dp: f32,
    max_drho: f32,
    max_dt: f32,
    early_du: f32,
    late_du: f32,
    max_scl_defect: f64,
    max_identity_err: f64,
}

/// Reference temperature the thermal free stream sits at (= `ALLMACH_T_REF`).
const T_REF: f32 = 1.0;

fn run_freestream(time_scheme: TimeScheme, thermal: bool) -> FreestreamOut {
    let (geo, domain) = square();
    let params = test_params(time_scheme, PSI);
    let mut cvt = generate_cvt_mesh_with_seeds(&geo, H, H, 1.0, domain, &LloydConfig::default());
    tag_slip_channel(&mut cvt.mesh);
    let n = cvt.mesh.num_cells();

    let model = if thermal {
        allmach_thermal_ale_model().expect("allmach thermal ale model")
    } else {
        allmach_pressure_ale_model().expect("allmach ale model")
    };
    let mut moving = pollster::block_on(MovingMeshDriver::build_with_model(
        cvt,
        model,
        &params,
        MeshMotionSpec::Prescribed(swirl),
        &vec![(U0.0 as f64, U0.1 as f64); n],
        &vec![0.0; n],
        None,
        None,
    ))
    .expect("moving allmach driver build");
    moving.set_boundary_retag(Some(tag_slip_channel));
    moving.driver_mut().apply_params(&params);

    let layout = moving.driver().solver().model().state_layout.clone();
    let stride = layout.stride() as usize;
    let u_off = layout.offset_for("U").expect("U offset") as usize;
    let p_off = layout.offset_for("p").expect("p offset") as usize;
    let rho_off = layout.offset_for("rho").expect("rho offset") as usize;
    let t_off = if thermal {
        Some(layout.offset_for("T").expect("T offset") as usize)
    } else {
        None
    };

    let mut out = FreestreamOut {
        max_du: 0.0,
        max_dp: 0.0,
        max_drho: 0.0,
        max_dt: 0.0,
        early_du: 0.0,
        late_du: 0.0,
        max_scl_defect: 0.0,
        max_identity_err: 0.0,
    };

    for step in 0..STEPS {
        let (outcome, stats) = moving
            .step(false)
            .unwrap_or_else(|e| panic!("step {step} failed (flip?): {e}"));
        assert!(outcome.diverged.is_none(), "step {step} diverged");
        out.max_scl_defect = out.max_scl_defect.max(stats.scl_defect);
        out.max_identity_err = out.max_identity_err.max(stats.identity_err);
        assert!(
            stats.identity_err < 1e-10,
            "step {step}: f64 swept-quad identity {:.3e}",
            stats.identity_err
        );
        assert!(
            stats.scl_defect < 1e-7,
            "step {step}: f32 SCL defect {:.3e}",
            stats.scl_defect
        );

        if step % 10 != 9 && step + 1 != STEPS {
            continue;
        }
        let state = pollster::block_on(moving.driver().solver().read_state_f32());
        let (mut du, mut dp, mut drho, mut dt) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for c in 0..n {
            du = du
                .max((state[c * stride + u_off] - U0.0).abs())
                .max((state[c * stride + u_off + 1] - U0.1).abs());
            dp = dp.max(state[c * stride + p_off].abs());
            drho = drho.max((state[c * stride + rho_off] - RHO_REF).abs());
            if let Some(to) = t_off {
                dt = dt.max((state[c * stride + to] - T_REF).abs());
            }
        }
        out.max_du = out.max_du.max(du);
        out.max_dp = out.max_dp.max(dp);
        out.max_drho = out.max_drho.max(drho);
        out.max_dt = out.max_dt.max(dt);
        if (STEPS / 4..STEPS / 2).contains(&step) {
            out.early_du = out.early_du.max(du);
        }
        if step >= 3 * STEPS / 4 {
            out.late_du = out.late_du.max(du);
        }
    }
    out
}

fn run_freestream_cpu(scheme: TimeScheme, thermal: bool, label: &str) -> FreestreamOut {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::set_var("CFD2_BACKEND", "cpu");
    let result = std::panic::catch_unwind(|| run_freestream(scheme, thermal));
    std::env::remove_var("CFD2_BACKEND");
    let out = match result {
        Ok(o) => o,
        Err(e) => std::panic::resume_unwind(e),
    };
    println!(
        "[allmach-ale-freestream] cpu/{label}: max|U-U0| = {:.3e} (early {:.3e}, late {:.3e}), \
         max|p| = {:.3e}, max|rho-rho_ref| = {:.3e}, max|T-T_ref| = {:.3e}, SCL defect = {:.3e}, \
         f64 identity = {:.3e} ({STEPS} steps, psi={PSI:.1e})",
        out.max_du, out.early_du, out.late_du, out.max_dp, out.max_drho, out.max_dt,
        out.max_scl_defect, out.max_identity_err
    );
    out
}

/// Caps set to a few× the incompressible free-stream scale (the compressible
/// free stream at `p=0` reduces to the incompressible fixed point plus the tiny
/// `psi`-scaled acoustic coupling; the thermal cross-terms vanish at uniform T).
fn assert_freestream_caps(out: &FreestreamOut) {
    assert!(out.max_identity_err < 1e-10, "f64 identity {:.3e}", out.max_identity_err);
    assert!(out.max_scl_defect < 1e-7, "SCL defect {:.3e}", out.max_scl_defect);
    assert!(out.max_du < 5e-4, "U drift {:.3e} above cap", out.max_du);
    assert!(out.max_dp < 5e-3, "p drift {:.3e} above cap", out.max_dp);
    assert!(out.max_drho < 5e-4, "rho drift {:.3e} above cap", out.max_drho);
    assert!(out.max_dt < 5e-4, "T drift {:.3e} above cap", out.max_dt);
    // Non-compounding: a GCL violation compounds; solve noise saturates.
    assert!(
        out.late_du <= (out.early_du * 3.0).max(5e-4),
        "late U drift {:.3e} vs early {:.3e}: GCL error compounds",
        out.late_du,
        out.early_du
    );
}

#[test]
fn allmach_ale_freestream_preserved_cpu_euler() {
    let out = run_freestream_cpu(TimeScheme::Euler, false, "euler");
    assert_freestream_caps(&out);
}

#[test]
fn allmach_ale_freestream_preserved_cpu_bdf2() {
    let out = run_freestream_cpu(TimeScheme::BDF2, false, "bdf2");
    assert_freestream_caps(&out);
}

#[test]
fn allmach_thermal_ale_freestream_preserved_cpu_euler() {
    let out = run_freestream_cpu(TimeScheme::Euler, true, "thermal-euler");
    assert_freestream_caps(&out);
}

#[test]
fn allmach_thermal_ale_freestream_preserved_cpu_bdf2() {
    let out = run_freestream_cpu(TimeScheme::BDF2, true, "thermal-bdf2");
    assert_freestream_caps(&out);
}

// ---------------------------------------------------------------------------
// Gate 3 (Req 3): the compressible ALE models RUN with moving mesh on a real
// GUI geometry — the channel-with-obstacle (an embedded circular obstacle, the
// hardest meshless-Voronoi case). Flow-coupled seed motion drives the full
// dt-handshake → advect → regen → swept-flux → refresh → step loop; the gate
// asserts the run stays finite/bounded and never diverges or tangles (a flip
// that swallows a seed would Err out of `step`). This is the geometry-level
// counterpart to the free-stream physics gates above.
// ---------------------------------------------------------------------------

fn run_obstacle_smoke(thermal: bool) {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::set_var("CFD2_BACKEND", "cpu");
    let result = std::panic::catch_unwind(|| {
        let geo = ChannelWithObstacle {
            length: 2.0,
            height: 1.0,
            obstacle_center: Point2::new(0.6, 0.5),
            obstacle_radius: 0.12,
        };
        let domain = Vector2::new(2.0, 1.0);
        let mut params = test_params(TimeScheme::Euler, PSI);
        params.inlet_velocity = 0.4;
        let cvt = generate_cvt_mesh_with_seeds(&geo, H, H, 1.0, domain, &LloydConfig::default());
        let n = cvt.mesh.num_cells();

        let model = if thermal {
            allmach_thermal_ale_model().expect("thermal ale model")
        } else {
            allmach_pressure_ale_model().expect("ale model")
        };
        // Flow-coupled interior motion (χ=0.5 keeps mesh quality); boundary seeds
        // (incl. the obstacle contour) stay fixed.
        let mut moving = pollster::block_on(MovingMeshDriver::build_with_model(
            cvt,
            model,
            &params,
            MeshMotionSpec::FlowCoupled { regularization: 0.5 },
            &vec![(params.inlet_velocity as f64, 0.0); n],
            &vec![0.0; n],
            None,
            None,
        ))
        .expect("moving allmach driver build on channel-with-obstacle");
        moving.driver_mut().apply_params(&params);

        let layout = moving.driver().solver().model().state_layout.clone();
        let stride = layout.stride() as usize;
        let u_off = layout.offset_for("U").expect("U offset") as usize;

        let mut max_u = 0.0f32;
        for step in 0..25 {
            let (outcome, stats) = moving
                .step(false)
                .unwrap_or_else(|e| panic!("[{}] step {step} failed: {e}", model_label(thermal)));
            assert!(outcome.diverged.is_none(), "[{}] step {step} diverged", model_label(thermal));
            assert!(stats.scl_defect < 1e-6, "[{}] step {step} SCL defect {:.3e}", model_label(thermal), stats.scl_defect);
            let state = pollster::block_on(moving.driver().solver().read_state_f32());
            for c in 0..n {
                let (ux, uy) = (state[c * stride + u_off], state[c * stride + u_off + 1]);
                assert!(ux.is_finite() && uy.is_finite(), "[{}] step {step}: non-finite U", model_label(thermal));
                max_u = max_u.max(ux.hypot(uy));
            }
        }
        // Bounded: a healthy obstacle wake stays within a few× the inlet speed.
        assert!(
            max_u < 10.0 * params.inlet_velocity,
            "[{}] max|U| {max_u:.3e} unbounded (inlet {:.3e})",
            model_label(thermal),
            params.inlet_velocity
        );
        println!(
            "[allmach-ale-obstacle] {}: 25 steps on channel-with-obstacle, max|U| = {max_u:.3e}, cells = {n}",
            model_label(thermal)
        );
    });
    std::env::remove_var("CFD2_BACKEND");
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

fn model_label(thermal: bool) -> &'static str {
    if thermal {
        "allmach_thermal_ale"
    } else {
        "allmach_pressure_ale"
    }
}

#[test]
fn allmach_pressure_ale_runs_on_channel_obstacle_cpu() {
    run_obstacle_smoke(false);
}

#[test]
fn allmach_thermal_ale_runs_on_channel_obstacle_cpu() {
    run_obstacle_smoke(true);
}

// ---------------------------------------------------------------------------
// Gate 4 (regression): an OSCILLATING obstacle whose amplitude exceeds the
// near-wall cell spacing must NOT tangle the mesh / diverge.
//
// The GUI "moving obstacle" demo (allmach_thermal_ale + FlowCoupled interior +
// a cross-stream oscillating obstacle + MovingWall BC) shipped a default
// amplitude of 0.05 on a 0.025 mesh — 2× the cell spacing. With a FIXED seed
// count the frozen/flow interior seeds adjacent to the wall do not step aside,
// so a wall whose PEAK displacement exceeds the cell spacing sweeps THROUGH them
// and swallows them: the clipped near-wall cells collapse/invert and the
// swept-quad telescoping identity fails (`swept_mesh_fluxes` errors out of
// `step`) a few real-time seconds in — the reported "diverges + weird remeshing"
// bug. `MovingMeshDriver::set_boundary_motion` now clamps an `Oscillation`
// amplitude to `OSC_AMPLITUDE_CELL_FRACTION × min_cell_size`, enforcing the
// documented `amplitude < cell spacing` anti-swallow contract for every caller.
//
// Part A is a direct unit check of the clamp; Part B is the end-to-end smoke
// that a GUI-scale (2× cell) requested amplitude now runs bounded through the
// full moving loop where it used to tangle (~step 75 on this mesh).
// ---------------------------------------------------------------------------

use cfd2::sim::{BoundaryMotionSpec, OscAxis, OSC_AMPLITUDE_CELL_FRACTION};

/// The GUI ChannelObstacle geometry + a uniform CVT at `cell` spacing.
fn obstacle_cvt(cell: f64) -> cfd2::meshgen::meshless::CvtMeshSeeds {
    let geo = ChannelWithObstacle {
        length: 3.0,
        height: 1.0,
        obstacle_center: Point2::new(1.0, 0.51),
        obstacle_radius: 0.1,
    };
    let domain = Vector2::new(3.0, 1.0);
    generate_cvt_mesh_with_seeds(&geo, cell, cell, 1.0, domain, &LloydConfig::default())
}

#[test]
fn oscillating_obstacle_amplitude_is_clamped_to_cell_spacing() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::set_var("CFD2_BACKEND", "cpu");
    let result = std::panic::catch_unwind(|| {
        let cell = 0.05;
        let cvt = obstacle_cvt(cell);
        let n = cvt.mesh.num_cells();
        let params = test_params(TimeScheme::BDF2, PSI);
        let mut moving = pollster::block_on(MovingMeshDriver::build_with_model(
            cvt,
            allmach_thermal_ale_model().expect("thermal ale"),
            &params,
            MeshMotionSpec::FlowCoupled { regularization: 0.5 },
            &vec![(params.inlet_velocity as f64, 0.0); n],
            &vec![0.0; n],
            None,
            None,
        ))
        .expect("moving driver build");

        let cap = OSC_AMPLITUDE_CELL_FRACTION * cell;

        // A grossly-too-large amplitude (6× cell) is clamped down to the cap.
        moving.set_boundary_motion(BoundaryMotionSpec::Oscillation {
            loop_index: 1,
            amplitude: 6.0 * cell,
            omega: std::f64::consts::TAU * 0.5,
            axis: OscAxis::CrossStream,
        });
        match moving.boundary_motion() {
            BoundaryMotionSpec::Oscillation { amplitude, .. } => assert!(
                (amplitude - cap).abs() <= 1e-12,
                "too-large amplitude not clamped: {amplitude} vs cap {cap}"
            ),
            _ => panic!("expected Oscillation"),
        }

        // An in-envelope amplitude (0.3× cell, well below the cap) passes through
        // unchanged — the clamp never tightens a validated-scale request.
        let small = 0.3 * cell;
        moving.set_boundary_motion(BoundaryMotionSpec::Oscillation {
            loop_index: 1,
            amplitude: small,
            omega: std::f64::consts::TAU * 0.5,
            axis: OscAxis::CrossStream,
        });
        match moving.boundary_motion() {
            BoundaryMotionSpec::Oscillation { amplitude, .. } => assert!(
                (amplitude - small).abs() <= 1e-12,
                "in-envelope amplitude perturbed: {amplitude} vs {small}"
            ),
            _ => panic!("expected Oscillation"),
        }
        println!("[allmach-ale-osc-clamp] cap = {cap:.4} (= {OSC_AMPLITUDE_CELL_FRACTION}×{cell})");
    });
    std::env::remove_var("CFD2_BACKEND");
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

#[test]
fn oscillating_obstacle_gui_scale_stays_bounded_cpu() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::set_var("CFD2_BACKEND", "cpu");
    // Transpiled engine keeps this real-geometry moving run fast enough for CI
    // (the physics is engine-independent; the zero-flux gate covers interpreter).
    std::env::set_var("CFD2_CPU_ENGINE", "transpiled");
    let result = std::panic::catch_unwind(|| {
        // ChannelObstacle at a moderate cell spacing (kept coarse enough for CI;
        // the swallow trigger is the amplitude/cell RATIO, not the absolute size).
        let cell = 0.05;
        let cvt = obstacle_cvt(cell);
        let n = cvt.mesh.num_cells();
        let mut params = test_params(TimeScheme::BDF2, PSI);
        params.inlet_velocity = 0.011; // ALLMACH obstacle default (Re≈150 shed)
        params.viscosity = 1.81e-5;
        params.density = 1.225;

        let mut moving = pollster::block_on(MovingMeshDriver::build_with_model(
            cvt,
            allmach_thermal_ale_model().expect("thermal ale"),
            &params,
            MeshMotionSpec::FlowCoupled { regularization: 0.5 },
            &vec![(params.inlet_velocity as f64, 0.0); n],
            &vec![0.0; n],
            None,
            None,
        ))
        .expect("moving driver build");
        moving.driver_mut().apply_params(&params);
        // The shipped GUI-default amplitude: 2× the cell spacing. Pre-fix this
        // tangled the near-wall mesh at ~step 75; the clamp caps it to
        // OSC_AMPLITUDE_CELL_FRACTION×cell so the run stays bounded.
        moving.set_boundary_motion(BoundaryMotionSpec::Oscillation {
            loop_index: 1,
            amplitude: 2.0 * cell,
            omega: std::f64::consts::TAU * 0.5,
            axis: OscAxis::CrossStream,
        });
        moving.set_moving_wall_bc(true);

        let layout = moving.driver().solver().model().state_layout.clone();
        let stride = layout.stride() as usize;
        let u_off = layout.offset_for("U").expect("U offset") as usize;

        // Past the pre-fix tangle step (~75) with margin.
        const STEPS: usize = 120;
        let mut max_u = 0.0f32;
        for step in 0..STEPS {
            let (outcome, stats) = moving.step(false).unwrap_or_else(|e| {
                panic!("oscillating-obstacle step {step} tangled/failed: {e}")
            });
            assert!(outcome.diverged.is_none(), "step {step} diverged: {:?}", outcome.diverged);
            assert!(stats.scl_defect < 1e-6, "step {step} SCL defect {:.3e}", stats.scl_defect);
            let state = pollster::block_on(moving.driver().solver().read_state_f32());
            for c in 0..n {
                let (ux, uy) = (state[c * stride + u_off], state[c * stride + u_off + 1]);
                assert!(ux.is_finite() && uy.is_finite(), "step {step}: non-finite U");
                max_u = max_u.max(ux.hypot(uy));
            }
        }
        // A healthy forced wake stays within a few× the max wall speed
        // (amplitude_clamped × omega ≈ 0.03 × π ≈ 0.094).
        assert!(max_u < 2.0, "max|U| {max_u:.3e} unbounded over {STEPS} steps");
        println!(
            "[allmach-ale-osc-obstacle] {STEPS} steps bounded on channel-obstacle, \
             max|U| = {max_u:.3e}, cells = {n}"
        );
    });
    std::env::remove_var("CFD2_BACKEND");
    std::env::remove_var("CFD2_CPU_ENGINE");
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}
