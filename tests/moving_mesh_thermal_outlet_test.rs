//! Regression coverage for the all-Mach THERMAL moving-mesh model at the
//! OUTLET, under the flow-adaptive + recycling + reordering combination the
//! GUI drives but no prior test exercised.
//!
//! Background: the production thermal outlet used to pin `T = t_ref` (Dirichlet)
//! — an over-specified outflow BC that reflected the developed flow's state
//! back into the outlet band. Combined with every-step adaptation refining
//! tiny cells against the fixed gauge-pressure outlet ghost (and mesh
//! reordering, which had zero test coverage of any kind), the compressible
//! pressure/velocity field grew an unbounded outlet spike once the advection
//! wave reached the exit — the adaptive dt collapsing to chase it. The fix
//! switches the production thermal outlet `T` to ZeroGradient (convective
//! outflow), matching the supersonic-nozzle path.
//!
//! Two gates:
//!  1. `thermal_outlet_temperature_is_zero_gradient` — a fast model-spec
//!     assertion that the production thermal outlet `T` is ZeroGradient while
//!     the MMS variant keeps Dirichlet (its manufactured solution pins T at
//!     every boundary).
//!  2. `movingmesh_thermal_adapt_recycle_reorder_outlet_stable` — the
//!     behavioural gate: thermal ALE + adapt(1) + smooth(1) + recycle +
//!     reorder(20) on the obstacle channel, run until the wave reaches the
//!     outlet, asserting the exit stays bounded (no dt collapse / |U| blow-up).

use cfd2::meshgen::meshless::{generate_cvt_mesh_with_seeds, LloydConfig};
use cfd2::meshgen::ChannelWithObstacle;
use cfd2::sim::{MeshMotionSpec, MovingMeshDriver, RuntimeParams};
use cfd2::solver::gpu::enums::{GpuBcKind, GpuBoundaryType};
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::model::{allmach_thermal_ale_model, allmach_thermal_ale_mms_model};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{GpuLowMachPrecondModel, PreconditionerType, TimeScheme};
use nalgebra::{Point2, Vector2};

/// The production thermal ALE model must let temperature convect out of the
/// outlet (ZeroGradient); the MMS variant must keep Dirichlet so its
/// manufactured boundary values still pin the exit.
#[test]
fn thermal_outlet_temperature_is_zero_gradient() {
    let outlet_t_kind = |model: &cfd2::solver::model::ModelSpec| -> GpuBcKind {
        model
            .boundaries
            .field("T")
            .expect("thermal model has a T field boundary spec")
            .by_boundary
            .get(&GpuBoundaryType::Outlet)
            .expect("T has an Outlet condition")[0]
            .kind
    };

    let prod = allmach_thermal_ale_model().expect("thermal ale model");
    assert_eq!(
        outlet_t_kind(&prod),
        GpuBcKind::ZeroGradient,
        "production thermal outlet T must be ZeroGradient (convective outflow); a \
         Dirichlet(t_ref) over-specifies the outflow face and reflects the exit state, \
         feeding the moving-mesh outlet instability"
    );

    let mms = allmach_thermal_ale_mms_model().expect("thermal ale mms model");
    assert_eq!(
        outlet_t_kind(&mms),
        GpuBcKind::Dirichlet,
        "the MMS thermal variant must KEEP the Dirichlet outlet T: its manufactured \
         solution pins T at every boundary via per-face values, and a ZeroGradient \
         outlet would drop them and break the steady order test"
    );
}

/// The GUI thermal-obstacle config that used to diverge: `allmach_thermal_ale`,
/// FlowCoupled, adaptation + smoothing every step, mesh reorder every 20, growth
/// budget 8×, band (0.001, 0.03), a SLOW near-incompressible inlet (0.011) and
/// the driver flow-CFL adaptive dt. The divergence was a slow-inlet low-Mach
/// pseudo-acoustic mode standing at the fixed Dirichlet-p outlet, which grew
/// once the developed wake reached the exit (~step 1600) — with the velocity
/// spiking and the adaptive dt collapsing. It is cured by the preconditioner
/// reference-velocity floor (`allmach_psi_precond`) + the ZeroGradient outlet T.
///
/// OPT-IN (`CFD2_THERMAL_OUTLET_TEST=1`): the wave must physically cross the
/// domain before the failure onset, so it needs ~1900 steps — too slow for the
/// default suite. Step count overridable via `CFD2_THERMAL_OUTLET_STEPS`.
#[test]
fn movingmesh_thermal_adapt_recycle_reorder_outlet_stable() {
    if std::env::var("CFD2_THERMAL_OUTLET_TEST").as_deref() != Ok("1") {
        eprintln!(
            "[thermal-outlet] set CFD2_THERMAL_OUTLET_TEST=1 to run (long: wave must reach outlet)"
        );
        return;
    }
    // Process-global backend + engine (this binary owns the run).
    std::env::set_var("CFD2_BACKEND", "cpu");
    std::env::set_var("CFD2_CPU_ENGINE", "transpiled");

    let steps: usize = std::env::var("CFD2_THERMAL_OUTLET_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1900);

    let (lx, ly) = (3.0, 1.0);
    let domain = Vector2::new(lx, ly);
    let geo = ChannelWithObstacle {
        length: lx,
        height: ly,
        obstacle_center: Point2::new(1.0, 0.51),
        obstacle_radius: 0.1,
    };
    let cell = 0.035; // moderate: coarse enough to run, fine enough to shed + refine
    let cvt = generate_cvt_mesh_with_seeds(&geo, cell, cell, 1.2, domain, &LloydConfig::default());
    let n0 = cvt.mesh.num_cells();

    // The GUI ALLMACH thermal-obstacle params: BDF2, dt-seed 0.02, 8 outers
    // auto-converge, real Air (psi = 1/c² ≈ 8.3e-6), the reported inlet 0.011.
    let params = RuntimeParams {
        filter_sigma: 0.0,
        adaptive_dt: false, // driver flow-CFL adaptive dt is wired below instead
        target_cfl: 0.9,
        requested_dt: 0.02,
        dtau: 0.0,
        log_convergence: false,
        log_every_steps: 100_000,
        advection_scheme: Scheme::SecondOrderUpwindVanLeer,
        time_scheme: TimeScheme::BDF2,
        preconditioner: PreconditionerType::Jacobi,
        outer_iters: 8,
        outer_auto_converge: true,
        low_mach_model: GpuLowMachPrecondModel::Off,
        low_mach_theta_floor: 1e-6,
        low_mach_pressure_coupling_alpha: 1.0,
        alpha_u: 0.7,
        alpha_p: 0.3,
        inlet_velocity: 0.011,
        density: 1.225,
        viscosity: 1.81e-5,
        eos: EosSpec::Constant,
        compressibility_psi: 8.3e-6,
        outlet_back_pressure: 0.0,
        // The SHIPPED GUI default (cleans the standing pressure mode). At this
        // COARSE cell size (0.035) the raised floor's step-0 impulsive-start
        // transient would spike |U| to ~17x inlet WITHOUT the moving driver's
        // startup dt growth-cap — so this gates BOTH the divergence cure and the
        // dt-cap that makes the clean floor step-0-safe on a coarse mesh.
        allmach_precond_uref_min: 1.0,
        pressure_inlet: false,
        inlet_pressure: 0.0,
    };

    let mut moving = pollster::block_on(MovingMeshDriver::build_with_model(
        cvt,
        allmach_thermal_ale_model().expect("thermal ale model"),
        &params,
        MeshMotionSpec::FlowCoupled { regularization: 0.5 },
        &vec![(params.inlet_velocity as f64, 0.0); n0],
        &vec![0.0; n0],
        None,
        None,
    ))
    .expect("thermal moving driver build");
    moving.driver_mut().apply_params(&params);
    // The exact reported knobs.
    moving.set_adaptive_sizing(1);
    moving.set_adaptive_sizing_band(Some((0.001, 0.03)));
    moving.set_adaptive_budget_factor(8.0);
    moving.set_smoothing(1, 1, 0.5);
    moving.set_reorder_every_n(20);
    moving.set_adaptive_dt(Some(params.target_cfl)); // driver flow-CFL dt (GUI moving path)

    let layout = moving.driver().solver().model().state_layout.clone();
    let stride = layout.stride() as usize;
    let u_off = layout.offset_for("U").expect("U offset") as usize;
    let inlet = params.inlet_velocity as f64;

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut max_u = 0.0f64;
        for step in 0..steps {
            let (outcome, stats) = moving
                .step(false)
                .unwrap_or_else(|e| panic!("[thermal-outlet] step {step}: {e}"));
            assert!(
                outcome.diverged.is_none(),
                "[thermal-outlet] diverged at step {step}: {:?} — the compressible outlet \
                 instability regressed",
                outcome.diverged
            );
            assert!(
                stats.scl_defect < 1e-6,
                "[thermal-outlet] step {step}: SCL/GCL defect {:.3e}",
                stats.scl_defect
            );
            let state = pollster::block_on(moving.driver().solver().read_state_f32());
            let n = moving.mesh().num_cells();
            for c in 0..n {
                let (ux, uy) = (
                    state[c * stride + u_off] as f64,
                    state[c * stride + u_off + 1] as f64,
                );
                assert!(
                    ux.is_finite() && uy.is_finite(),
                    "[thermal-outlet] step {step}: non-finite U at cell {c}"
                );
                max_u = max_u.max(ux.hypot(uy));
            }
            // The healthy obstacle wake tops out ~1.6× inlet; the failure spikes
            // the outlet velocity by >20× (0.28 vs 0.017). 10× inlet cleanly
            // separates a bounded run from the divergence onset.
            assert!(
                max_u < 10.0 * inlet,
                "[thermal-outlet] step {step}: max|U| {max_u:.4e} exceeds 10× inlet \
                 ({:.4e}) — the outlet spike regressed",
                10.0 * inlet
            );
        }
        println!(
            "[thermal-outlet] {steps} steps bounded (max|U| = {max_u:.4e}, {:.1}× inlet); \
             cells {} -> {}",
            max_u / inlet,
            n0,
            moving.mesh().num_cells()
        );
    }));

    std::env::remove_var("CFD2_BACKEND");
    std::env::remove_var("CFD2_CPU_ENGINE");
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}
