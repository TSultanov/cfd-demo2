//! M4.1 gate (meshless/moving-mesh roadmap §M4): the `MovingMeshDriver`
//! frozen-seed static limit.
//!
//! The loop (dt handshake → advect seeds → regen via the M0 engine → swept-quad
//! mesh fluxes → topology refresh → step) is proven here in its EASY case:
//! `MeshMotionSpec::Frozen`, where the seeds never move. Because a regen from an
//! unchanged seed set reproduces the mesh byte-for-byte (deterministic M0
//! pipeline), the swept fluxes are exactly zero and the moving loop must
//! reproduce a plain static `incompressible_momentum_ale` run.
//!
//! Two do-no-harm statements:
//!   * `moving_loop_frozen_seeds_matches_static_cpu` — the FULL loop (regen +
//!     topology refresh + zero-flux ALE step) vs a static ALE solver on the same
//!     initial mesh. The test measures whether the match is byte-identical or
//!     only f32-exact and REPORTS which (it is the M4 do-no-harm claim). It also
//!     runs the skip-regen variant (a pure `SolverDriver::step` passthrough),
//!     which MUST be byte-identical.
//!   * `moving_loop_dt_pinned` — a regression guard for the F2 dt-ownership
//!     hazard: the swept-flux dt equals the step dt exactly, and adaptive dt
//!     stays off across the ALE step.
//!
//! The static-path gates (mesh_refresh_identity, ale_zero_flux_equivalence,
//! WGSL snapshots) are unchanged and run from their own files.
//!
//! Feature gate: `meshgen` (the M0 engine + the `sim` drivers) + `cpu` — the
//! CPU-first M4 loop, per the roadmap's validation program.
#![cfg(all(feature = "meshgen", feature = "cpu"))]

use cfd2::meshgen::meshless::{assemble_meshless_from_seeds, generate_cvt_mesh_with_seeds};
use cfd2::meshgen::RectangularChannel;
use cfd2::sim::{MeshMotionSpec, MovingMeshDriver, RuntimeParams, SolverDriver};
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::model::incompressible_momentum_ale_model;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{GpuLowMachPrecondModel, PreconditionerType, TimeScheme};
use nalgebra::Vector2;
use std::sync::Mutex;

/// `CFD2_BACKEND` is process-global; serialize the tests.
static ENV_LOCK: Mutex<()> = Mutex::new(());

const LX: f64 = 1.5;
const LY: f64 = 1.0;
const H: f64 = 0.08;
const STEPS: usize = 30;
const DT: f32 = 0.005;

fn channel() -> (RectangularChannel, Vector2<f64>) {
    (
        RectangularChannel {
            length: LX,
            height: LY,
        },
        Vector2::new(LX, LY),
    )
}

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
        inlet_velocity: 1.0,
        density: 1.0,
        viscosity: 1e-2,
        eos: EosSpec::Constant,
        compressibility_psi: 0.0,
        outlet_back_pressure: 0.0,
        pressure_inlet: false,
        inlet_pressure: 0.0,
    }
}

fn state_bits(driver: &SolverDriver) -> Vec<u32> {
    pollster::block_on(driver.solver().read_state_f32())
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn diff_stats(a: &[u32], b: &[u32]) -> (usize, f32) {
    assert_eq!(a.len(), b.len(), "state length mismatch");
    let ndiff = a.iter().zip(b).filter(|(x, y)| x != y).count();
    let maxd = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (f32::from_bits(x) - f32::from_bits(y)).abs())
        .fold(0.0f32, f32::max);
    (ndiff, maxd)
}

/// Build a plain static-mesh `incompressible_momentum_ale` driver (the
/// reference), matching the moving driver's build/apply_params sequence.
fn build_static(mesh: &cfd2::solver::mesh::Mesh, params: &RuntimeParams) -> SolverDriver {
    let n = mesh.num_cells();
    let build = pollster::block_on(SolverDriver::build(
        mesh,
        incompressible_momentum_ale_model().expect("ale model"),
        params,
        &vec![(0.0, 0.0); n],
        &vec![0.0; n],
        None,
        None,
    ))
    .expect("static driver build");
    let mut driver = build.driver;
    driver.apply_params(params);
    driver
}

/// THE M4.1 do-no-harm gate. Frozen-seed moving loop vs a static ALE run on the
/// identical initial mesh: measure and report byte-identity vs f32-exactness,
/// and assert the skip-regen variant is byte-identical.
#[test]
fn moving_loop_frozen_seeds_matches_static_cpu() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::set_var("CFD2_BACKEND", "cpu");
    let result = std::panic::catch_unwind(|| {
        let (geo, domain) = channel();
        let params = test_params();
        let cvt = generate_cvt_mesh_with_seeds(
            &geo,
            H,
            H,
            1.0,
            domain,
            &cfd2::meshgen::LloydConfig::default(),
        );
        let n = cvt.mesh.num_cells();
        println!("[m4.1] CVT channel: {n} cells, {} faces", cvt.mesh.num_faces());
        assert!(n > 100, "expected a non-trivial mesh, got {n} cells");

        // Determinism foundation: a regen from the authoritative seeds
        // reproduces the generation-time mesh byte-for-byte (vertices + faces).
        let regen = assemble_meshless_from_seeds(
            &cvt.seeds,
            &cvt.kinds,
            &cvt.spec,
            cvt.domain,
            cvt.min_cell_size,
        );
        let vx_bits: Vec<u64> = cvt.mesh.vx.iter().map(|v| v.to_bits()).collect();
        let rvx_bits: Vec<u64> = regen.vx.iter().map(|v| v.to_bits()).collect();
        let vy_bits: Vec<u64> = cvt.mesh.vy.iter().map(|v| v.to_bits()).collect();
        let rvy_bits: Vec<u64> = regen.vy.iter().map(|v| v.to_bits()).collect();
        assert_eq!(vx_bits, rvx_bits, "regen vx not byte-identical to build-time mesh");
        assert_eq!(vy_bits, rvy_bits, "regen vy not byte-identical to build-time mesh");
        assert_eq!(cvt.mesh.face_owner, regen.face_owner, "regen face_owner differs");
        assert_eq!(cvt.mesh.face_neighbor, regen.face_neighbor, "regen face_neighbor differs");
        assert_eq!(cvt.mesh.cell_faces, regen.cell_faces, "regen cell_faces differs");
        println!("[m4.1] regen from frozen seeds is BYTE-IDENTICAL to the build-time mesh");

        let init_mesh = cvt.mesh.clone();

        // ── Full regen loop vs static (default: geometry seam when unchanged) ──
        // A frozen regen is byte-identical, so the topology never changes and
        // the driver takes the surgical GEOMETRY seam every step — expected
        // byte-identical to a static ALE run.
        let mut static_ref = build_static(&init_mesh, &params);
        let mut moving = pollster::block_on(MovingMeshDriver::build(
            cvt,
            &params,
            MeshMotionSpec::Frozen,
            &vec![(0.0, 0.0); n],
            &vec![0.0; n],
            None,
            None,
        ))
        .expect("moving driver build");
        moving.driver_mut().apply_params(&params);

        let mut worst_ndiff = 0usize;
        let mut worst_maxd = 0.0f32;
        for step in 0..STEPS {
            let out_static = static_ref.step(false);
            assert!(out_static.diverged.is_none(), "static step {step} diverged");
            let (out_moving, stats) = moving.step(false).expect("moving step");
            assert!(out_moving.diverged.is_none(), "moving step {step} diverged");

            // Always-on ALE diagnostics for a frozen (zero-motion) step.
            assert_eq!(stats.identity_err, 0.0, "step {step}: nonzero f64 identity err");
            assert_eq!(stats.scl_defect, 0.0, "step {step}: nonzero SCL defect on frozen seeds");
            assert!(!stats.topo_changed, "step {step}: frozen regen reported a topology change");
            assert_eq!(stats.n_cells, n, "cell count changed");

            let (ndiff, maxd) = diff_stats(&state_bits(&static_ref), &state_bits(moving.driver()));
            worst_ndiff = worst_ndiff.max(ndiff);
            worst_maxd = worst_maxd.max(maxd);
        }

        if worst_ndiff == 0 {
            println!(
                "[m4.1] FROZEN REGEN LOOP (geometry seam) == STATIC ALE run BYTE-IDENTICALLY over \
                 {STEPS} steps ({n} cells): do-no-harm at the strongest level."
            );
        } else {
            println!(
                "[m4.1] frozen regen loop (geometry seam) matches static ALE to f32 (NOT \
                 byte-identical): worst {worst_ndiff} slots differ, max|diff| = {worst_maxd:.3e}."
            );
        }
        assert_eq!(
            worst_ndiff, 0,
            "frozen regen loop via the geometry seam is not byte-identical to static ALE: \
             {worst_ndiff} slots differ, max|diff| = {worst_maxd:.3e} — the geometry refresh is \
             supposed to be fully surgical for a zero-motion regen"
        );

        // ── Topology-seam variant: MEASURE the do-no-harm cost of the topology
        //    refresh (the finding). Routing every frozen step through the
        //    topology seam rebuilds the CSR stack, CLEARS the Schur-AMG
        //    hierarchy and re-scatters the bc tables; on a developing channel
        //    that AMG reset drives an O(1) drift from a static run even though
        //    the mesh is byte-identical. Reported, not required byte/f32-equal.
        let cvt_t = generate_cvt_mesh_with_seeds(
            &geo,
            H,
            H,
            1.0,
            domain,
            &cfd2::meshgen::LloydConfig::default(),
        );
        let mut static_ref_t = build_static(&init_mesh, &params);
        let mut moving_t = pollster::block_on(MovingMeshDriver::build(
            cvt_t,
            &params,
            MeshMotionSpec::Frozen,
            &vec![(0.0, 0.0); n],
            &vec![0.0; n],
            None,
            None,
        ))
        .expect("topology-seam driver build");
        moving_t.driver_mut().apply_params(&params);
        moving_t.set_force_topology_seam(true);
        let mut topo_maxd = 0.0f32;
        for step in 0..STEPS {
            let out_static = static_ref_t.step(false);
            assert!(out_static.diverged.is_none(), "static-t step {step} diverged");
            let (out_moving, _) = moving_t.step(false).expect("topology-seam step");
            assert!(out_moving.diverged.is_none(), "topology-seam step {step} diverged");
            let (_, maxd) = diff_stats(&state_bits(&static_ref_t), &state_bits(moving_t.driver()));
            topo_maxd = topo_maxd.max(maxd);
        }
        println!(
            "[m4.1] FINDING: forcing the TOPOLOGY seam every frozen step drifts from static ALE by \
             max|diff| = {topo_maxd:.3e} (the AMG-hierarchy reset + bc re-scatter of the topology \
             refresh; the geometry seam avoids it). Bounded (no divergence)."
        );

        // ── Skip-regen variant: MUST be byte-identical to static ──────────────
        let cvt2 = generate_cvt_mesh_with_seeds(
            &geo,
            H,
            H,
            1.0,
            domain,
            &cfd2::meshgen::LloydConfig::default(),
        );
        let mut static_ref2 = build_static(&init_mesh, &params);
        let mut skip = pollster::block_on(MovingMeshDriver::build(
            cvt2,
            &params,
            MeshMotionSpec::Frozen,
            &vec![(0.0, 0.0); n],
            &vec![0.0; n],
            None,
            None,
        ))
        .expect("skip-regen driver build");
        skip.driver_mut().apply_params(&params);
        skip.set_regen_each_step(false);

        for step in 0..STEPS {
            let out_static = static_ref2.step(false);
            assert!(out_static.diverged.is_none(), "static2 step {step} diverged");
            let (out_skip, _) = skip.step(false).expect("skip step");
            assert!(out_skip.diverged.is_none(), "skip step {step} diverged");
            let (ndiff, maxd) = diff_stats(&state_bits(&static_ref2), &state_bits(skip.driver()));
            assert_eq!(
                ndiff, 0,
                "step {step}: skip-regen variant diverged bitwise from static ALE \
                 ({ndiff} slots, max|diff| = {maxd:.3e})"
            );
        }
        println!("[m4.1] skip-regen variant == static ALE BYTE-IDENTICALLY over {STEPS} steps");
    });
    std::env::remove_var("CFD2_BACKEND");
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

/// F2 dt-handshake regression guard: the swept-flux dt equals the step dt
/// exactly, and adaptive dt stays off across the ALE step.
#[test]
fn moving_loop_dt_pinned() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::set_var("CFD2_BACKEND", "cpu");
    let result = std::panic::catch_unwind(|| {
        let (geo, domain) = channel();
        let params = test_params();
        let cvt = generate_cvt_mesh_with_seeds(
            &geo,
            H,
            H,
            1.0,
            domain,
            &cfd2::meshgen::LloydConfig::default(),
        );
        let n = cvt.mesh.num_cells();
        let mut moving = pollster::block_on(MovingMeshDriver::build(
            cvt,
            &params,
            MeshMotionSpec::Frozen,
            &vec![(0.0, 0.0); n],
            &vec![0.0; n],
            None,
            None,
        ))
        .expect("moving driver build");
        moving.driver_mut().apply_params(&params);

        for step in 0..5 {
            let (outcome, stats) = moving.step(false).expect("moving step");
            assert!(outcome.diverged.is_none(), "step {step} diverged");
            // The flux dt (stats.dt) is the exact f64-widened f32 the solver
            // stepped with (outcome.dt).
            assert_eq!(
                stats.dt as f32, outcome.dt,
                "step {step}: flux dt {} != step dt {}",
                stats.dt as f32, outcome.dt
            );
            assert_eq!(outcome.dt, DT, "step {step}: step dt drifted from the pinned value");
            // Adaptive dt must never turn on for an ALE run.
            assert!(
                !moving.driver().params().adaptive_dt,
                "step {step}: adaptive dt is on during an ALE step (F2 hazard)"
            );
        }
        println!("[m4.1] dt pinned: flux dt == step dt == {DT} across the ALE steps; adaptive off");
    });
    std::env::remove_var("CFD2_BACKEND");
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

/// A Frozen `MovingMeshDriver` must reject `adaptive_dt == true` at build
/// (the SCL closure needs a fixed dt; the mesh-motion cap is applied by the
/// driver, not the solver's adaptive path).
#[test]
fn moving_loop_rejects_adaptive_dt() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::set_var("CFD2_BACKEND", "cpu");
    let result = std::panic::catch_unwind(|| {
        let (geo, domain) = channel();
        let mut params = test_params();
        params.adaptive_dt = true;
        let cvt = generate_cvt_mesh_with_seeds(
            &geo,
            H,
            H,
            1.0,
            domain,
            &cfd2::meshgen::LloydConfig::default(),
        );
        let n = cvt.mesh.num_cells();
        let built = pollster::block_on(MovingMeshDriver::build(
            cvt,
            &params,
            MeshMotionSpec::Frozen,
            &vec![(0.0, 0.0); n],
            &vec![0.0; n],
            None,
            None,
        ));
        // MovingMeshDriver is not Debug (it owns a SolverDriver); match instead
        // of `expect_err`.
        let err = match built {
            Ok(_) => panic!("adaptive dt must be rejected at build"),
            Err(e) => e,
        };
        assert!(err.contains("adaptive_dt"), "unexpected error: {err}");
    });
    std::env::remove_var("CFD2_BACKEND");
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}
