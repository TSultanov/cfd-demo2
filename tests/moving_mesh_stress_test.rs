//! EVERY-STEP adaptation + smoothing stress gates for the moving-mesh ALE
//! path, incompressible AND all-Mach, Frozen AND FlowCoupled motion.
//!
//! Every step: Lloyd smoothing moves the interior seeds AND the flow-adaptive
//! sizing fires a resize event (birth/kill/wall-split → full rebuild-and-
//! gather state transfer → mass-row transfer projection). This is the
//! worst-case cadence for the transfer seam — the mesh never settles between
//! events, so any systematic error the projection misses (mass defects,
//! pressure dipoles, history corruption) compounds visibly within tens of
//! steps.
//!
//! Asserted per configuration:
//!  - no divergence, finite state, bounded velocity (wake < 10x inlet);
//!  - resize events actually fire on most steps (the stress is real);
//!  - the mass-row transfer projection measurably closes the solver's own
//!    continuity-row residual at every resize (post << pre);
//!  - the cell count respects the adaptivity budget.

#![cfg(all(feature = "meshgen", feature = "cpu"))]

use cfd2::meshgen::meshless::generate_cvt_mesh_with_seeds;
use cfd2::meshgen::{ChannelWithObstacle, LloydConfig};
use cfd2::sim::{MeshMotionSpec, MovingMeshDriver, RuntimeParams};
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::model::{allmach_pressure_ale_model, incompressible_momentum_ale_model};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{GpuLowMachPrecondModel, PreconditionerType, TimeScheme};
use nalgebra::{Point2, Vector2};
use std::sync::Mutex;

/// `CFD2_BACKEND` is process-global; serialize the tests.
static ENV_LOCK: Mutex<()> = Mutex::new(());

const STEPS: usize = 60;

/// Step count, env-overridable for profiling runs (`CFD2_STRESS_STEPS`,
/// default [`STEPS`]). Physics gates always pass at the default; a longer
/// profiling run keeps the identical per-step discipline.
fn steps() -> usize {
    std::env::var("CFD2_STRESS_STEPS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(STEPS)
}

/// Mesh resolution override for profiling runs (`CFD2_STRESS_H`): scales the
/// per-family default `h` (and its CELL-SIZE adaptation band with it, so the
/// flow-adaptive planner works the same relative sizing space). Unset = the
/// committed default — the gate configuration is unchanged.
fn h_override(default_h: f64, band: (f64, f64)) -> (f64, (f64, f64)) {
    match std::env::var("CFD2_STRESS_H")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
    {
        Some(h) if h > 0.0 => {
            let s = h / default_h;
            (h, (band.0 * s, band.1 * s))
        }
        _ => (default_h, band),
    }
}

/// Is a GPU adapter present? (The GPU stress gates skip cleanly without one.)
fn gpu_adapter_available() -> bool {
    let instance = wgpu::Instance::default();
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).is_ok()
}

#[derive(Clone, Copy, PartialEq)]
enum Family {
    Incompressible,
    AllMach,
}

fn params_for(family: Family) -> RuntimeParams {
    match family {
        // The validated incompressible moving-mesh recipe (the spike/hernia
        // configs): BDF2, VanLeer, 6 outers, Re ~ 150 on the D=0.2 obstacle.
        Family::Incompressible => RuntimeParams {
        filter_sigma: 0.0,
            adaptive_dt: false,
            target_cfl: 0.9,
            requested_dt: 0.01,
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
            inlet_velocity: 1.0,
            density: 1.0,
            viscosity: 1.33e-3,
            eos: EosSpec::Constant,
            compressibility_psi: 0.0,
            outlet_back_pressure: 0.0,
            allmach_precond_uref_min: 0.2,
            pressure_inlet: false,
            inlet_pressure: 0.0,
        },
        // The PROVEN all-Mach ALE obstacle recipe (allmach_ale_test /
        // moving_mesh_gui_test): Euler stepping, 6 outer sweeps, dt 5e-3,
        // gauge-pressure psi = 1e-4 (c = 100, M = 4e-3), inlet 0.4. The
        // incompressible defaults (BDF2, dt 1e-2) are NOT stable for the
        // all-Mach family on a coarse obstacle mesh.
        Family::AllMach => RuntimeParams {
        filter_sigma: 0.0,
            adaptive_dt: false,
            target_cfl: 0.9,
            requested_dt: 0.005,
            dtau: 0.0,
            log_convergence: false,
            log_every_steps: 1000,
            advection_scheme: Scheme::SecondOrderUpwindVanLeer,
            time_scheme: TimeScheme::Euler,
            preconditioner: PreconditionerType::Jacobi,
            outer_iters: 6,
            outer_auto_converge: false,
            low_mach_model: GpuLowMachPrecondModel::Off,
            low_mach_theta_floor: 1e-6,
            low_mach_pressure_coupling_alpha: 1.0,
            alpha_u: 0.7,
            alpha_p: 0.3,
            inlet_velocity: 0.4,
            density: 1.0,
            viscosity: 1e-2,
            eos: EosSpec::Constant,
            compressibility_psi: 1.0e-4,
            outlet_back_pressure: 0.0,
            allmach_precond_uref_min: 0.2,
            pressure_inlet: false,
            inlet_pressure: 0.0,
        },
    }
}

struct StressReport {
    adapt_steps: usize,
    max_u: f64,
    defect_pre_max: f64,
    /// Worst RESIDUAL ratio `post/pre` over resize events with a
    /// non-negligible defect — how well the projection closes the solver's
    /// own mass row.
    worst_close_ratio: f64,
}

/// One every-step-adaptation + every-step-smoothing run on the obstacle
/// channel; panics on divergence / non-finite state / budget violation.
/// `expect_gpu` asserts which backend actually built (a GPU gate that
/// silently fell back to CPU would pass vacuously).
fn stress_run(family: Family, motion: MeshMotionSpec, label: &str, expect_gpu: bool) -> StressReport {
    let (h, band) = match family {
        Family::Incompressible => h_override(0.05, (0.02, 0.06)),
        Family::AllMach => h_override(0.08, (0.04, 0.09)),
    };
    let geo = ChannelWithObstacle {
        length: 2.0,
        height: 1.0,
        obstacle_center: Point2::new(0.6, 0.51),
        obstacle_radius: 0.1,
    };
    let domain = Vector2::new(2.0, 1.0);
    let params = params_for(family);
    let cvt = generate_cvt_mesh_with_seeds(&geo, h, h, 1.0, domain, &LloydConfig::default());
    let n0 = cvt.mesh.num_cells();
    let model = match family {
        Family::Incompressible => {
            incompressible_momentum_ale_model().expect("incompressible ale model")
        }
        Family::AllMach => allmach_pressure_ale_model().expect("allmach ale model"),
    };
    let mut moving = pollster::block_on(MovingMeshDriver::build_with_model(
        cvt,
        model,
        &params,
        motion,
        &vec![(params.inlet_velocity as f64, 0.0); n0],
        &vec![0.0; n0],
        None,
        None,
    ))
    .unwrap_or_else(|e| panic!("[{label}] driver build: {e}"));
    assert_eq!(
        !moving.driver().solver().is_cpu(),
        expect_gpu,
        "[{label}] wrong backend (expect_gpu={expect_gpu})"
    );
    moving.driver_mut().apply_params(&params);
    // THE STRESS: adaptation and smoothing EVERY solution step (cadence 1 —
    // the planner runs each step; whether it finds work depends on the
    // flow). The incompressible Re~150 case sheds a Kármán street and
    // re-adapts continuously (~1 resize/step); the all-Mach recipe is
    // low-Re (steady wake), so its resize events concentrate in the
    // developing transient — a front-loaded storm of consecutive
    // rebuild+transfer+projection events, which is exactly the seam abuse
    // this gate exists for.
    moving.set_adaptive_sizing(1);
    moving.set_adaptive_sizing_band(Some(band));
    moving.set_smoothing(1, 1, 0.5);

    let layout = moving.driver().solver().model().state_layout.clone();
    let stride = layout.stride() as usize;
    let u_off = layout.offset_for("U").expect("U offset") as usize;
    let budget_cap = (n0 as f64 * cfd2::sim::ADAPT_BUDGET_MAX_FACTOR).ceil() as usize + 1;

    let mut report = StressReport {
        adapt_steps: 0,
        max_u: 0.0,
        defect_pre_max: 0.0,
        worst_close_ratio: 0.0,
    };
    let steps = steps();
    for step in 0..steps {
        let (outcome, stats) = moving
            .step(false)
            .unwrap_or_else(|e| panic!("[{label}] step {step}: {e}"));
        assert!(
            outcome.diverged.is_none(),
            "[{label}] step {step} diverged: {:?}",
            outcome.diverged
        );
        assert!(
            stats.n_cells <= budget_cap,
            "[{label}] step {step}: {} cells exceeds the adaptivity budget {budget_cap}",
            stats.n_cells
        );
        if stats.cells_born > 0 || stats.cells_killed > 0 {
            report.adapt_steps += 1;
        }
        // The projection must MEASURABLY close the mass row at every
        // transfer event with a substantial defect — resize AND recycle
        // (the recycle re-seed's zeroth-order copy is the same disease; the
        // stats fields carry the worst of both). The ratio is only
        // meaningful when the achieved `post` sits ABOVE the f32-assembly
        // noise floor (~5e-8 measured): a post below 1e-6 absolute is a
        // full close regardless of pre, and near-floor events would
        // otherwise read as spurious ratio failures (a rejected correction
        // reports post == pre).
        if stats.cells_born > 0 || stats.cells_killed > 0 || stats.recycled > 0 {
            report.defect_pre_max = report.defect_pre_max.max(stats.transfer_defect_pre);
            if stats.transfer_defect_pre > 1e-3 && stats.transfer_defect_post > 1e-6 {
                let ratio = stats.transfer_defect_post / stats.transfer_defect_pre;
                report.worst_close_ratio = report.worst_close_ratio.max(ratio);
            }
        }
        let state = pollster::block_on(moving.driver().solver().read_state_f32());
        let n = moving.mesh().num_cells();
        for c in 0..n {
            let (ux, uy) = (
                state[c * stride + u_off] as f64,
                state[c * stride + u_off + 1] as f64,
            );
            assert!(
                ux.is_finite() && uy.is_finite(),
                "[{label}] step {step}: non-finite U at cell {c}"
            );
            report.max_u = report.max_u.max(ux.hypot(uy));
        }
    }
    assert!(
        report.max_u < 10.0 * params.inlet_velocity as f64,
        "[{label}] max|U| {:.3e} unbounded (inlet {:.3e})",
        report.max_u,
        params.inlet_velocity
    );
    assert!(
        report.adapt_steps * 3 >= steps,
        "[{label}] only {}/{steps} steps carried a resize event — the stress \
         config no longer exercises the transfer seam (measured: 59 incompressible, \
         ~29-35 all-Mach whose steady low-Re wake front-loads the events)",
        report.adapt_steps
    );
    eprintln!(
        "[{label}] {steps} steps, {} resizes, max|U| {:.3}, defect pre_max {:.3e}, \
         worst close ratio {:.3e}",
        report.adapt_steps, report.max_u, report.defect_pre_max, report.worst_close_ratio
    );
    report
}

fn gate_asserts(label: &str, report: &StressReport, expect_close: f64) {
    // The projection actually RAN and measured real transfer defects
    // (2.7e-2..3.5e-2 across the four configs). Without this, a silently
    // disabled projection reports (0, 0) on every resize and the
    // close-ratio bound below passes vacuously.
    assert!(
        report.defect_pre_max > 1e-3,
        "[{label}] max transfer defect {:.3e} — the mass-row projection never \
         measured a real defect (disabled or not running?)",
        report.defect_pre_max
    );
    assert!(
        report.worst_close_ratio <= expect_close,
        "[{label}] mass-row projection close ratio {:.3e} exceeds {expect_close:.1e} \
         — the correction no longer closes the solver's own continuity row",
        report.worst_close_ratio
    );
}

fn run_gate(family: Family, motion: MeshMotionSpec, label: &str, expect_close: f64) {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("CFD2_BACKEND", "cpu");
    let result = std::panic::catch_unwind(|| {
        let report = stress_run(family, motion, label, false);
        gate_asserts(label, &report, expect_close);
    });
    std::env::remove_var("CFD2_BACKEND");
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

/// The SAME every-step adaptation + smoothing stress on the GPU backend —
/// the configuration whose missing resize discipline (flat-history reinit,
/// no BDF continuity, no re-solve) published every per-step rebuild as an
/// integrator restart and was the GUI's phantom pressure dipoles. The
/// mass-row projection runs through the params-faithful CPU companion; the
/// same close-ratio gates apply. Skips cleanly without a GPU adapter.
fn run_gate_gpu(family: Family, motion: MeshMotionSpec, label: &str, expect_close: f64) {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !gpu_adapter_available() {
        eprintln!("[{label}] no GPU adapter present; skipping GPU stress gate");
        return;
    }
    std::env::remove_var("CFD2_BACKEND");
    std::env::remove_var("CFD2_CPU_ENGINE");
    let result = std::panic::catch_unwind(|| {
        let report = stress_run(family, motion, label, true);
        gate_asserts(label, &report, expect_close);
    });
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

#[test]
fn movingmesh_stress_everystep_incompressible_frozen() {
    run_gate(
        Family::Incompressible,
        MeshMotionSpec::Frozen,
        "stress incompressible+frozen",
        1e-3,
    );
}

#[test]
fn movingmesh_stress_everystep_incompressible_flowcoupled() {
    run_gate(
        Family::Incompressible,
        MeshMotionSpec::FlowCoupled {
            regularization: 0.5,
        },
        "stress incompressible+flowcoupled",
        1e-3,
    );
}

#[test]
fn movingmesh_stress_everystep_allmach_frozen() {
    run_gate(
        Family::AllMach,
        MeshMotionSpec::Frozen,
        "stress allmach+frozen",
        1e-3,
    );
}

#[test]
fn movingmesh_stress_everystep_allmach_flowcoupled() {
    run_gate(
        Family::AllMach,
        MeshMotionSpec::FlowCoupled {
            regularization: 0.5,
        },
        "stress allmach+flowcoupled",
        1e-3,
    );
}

#[test]
fn movingmesh_stress_everystep_incompressible_frozen_gpu() {
    run_gate_gpu(
        Family::Incompressible,
        MeshMotionSpec::Frozen,
        "stress incompressible+frozen GPU",
        1e-3,
    );
}

#[test]
fn movingmesh_stress_everystep_incompressible_flowcoupled_gpu() {
    run_gate_gpu(
        Family::Incompressible,
        MeshMotionSpec::FlowCoupled {
            regularization: 0.5,
        },
        "stress incompressible+flowcoupled GPU",
        1e-3,
    );
}

#[test]
fn movingmesh_stress_everystep_allmach_frozen_gpu() {
    run_gate_gpu(
        Family::AllMach,
        MeshMotionSpec::Frozen,
        "stress allmach+frozen GPU",
        1e-3,
    );
}

#[test]
fn movingmesh_stress_everystep_allmach_flowcoupled_gpu() {
    run_gate_gpu(
        Family::AllMach,
        MeshMotionSpec::FlowCoupled {
            regularization: 0.5,
        },
        "stress allmach+flowcoupled GPU",
        1e-3,
    );
}
