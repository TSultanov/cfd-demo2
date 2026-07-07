//! Zero-flux equivalence gate: `incompressible_momentum_ale` with all-zero
//! `mesh_fluxes` and equal volume history over a STATIC mesh must reproduce
//! `incompressible_momentum` byte-identically. The ALE kernels differ by
//! exactly `phi_rel = phi - rho * mesh_fluxes[face]` at every convective
//! consumption point, and with `mesh_fluxes[face] == 0.0` the subtraction is
//! `x - rho*0.0 = x - 0.0`, an IEEE-754 bitwise identity for every finite x
//! (including -0.0: `-0.0 - 0.0 == -0.0` under round-to-nearest). Both backends
//! seed `cell_vols_old(_old) = cell_vols` so equal-volume history holds too.
//!
//! Comparison is on the CPU backend, where `read_state_f32` is a lossless view
//! of the f32-bit interpreter state. Both CPU engines are covered: the
//! interpreter (KernelProgram IR) and the transpiler (compiled Rust).
#![cfg(all(feature = "meshgen", feature = "cpu"))]

use cfd2::sim::{DriverBuild, RuntimeParams, SolverDriver};
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType, Mesh};
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::model::{
    incompressible_momentum_ale_model, incompressible_momentum_model, ModelSpec,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{GpuLowMachPrecondModel, PreconditionerType, TimeScheme};
use std::sync::Mutex;

/// `CFD2_BACKEND`/`CFD2_CPU_ENGINE` are process-global; serialize the tests.
static ENV_LOCK: Mutex<()> = Mutex::new(());

const NX: usize = 64;
const NY: usize = 32;
const LX: f64 = 2.0;
const LY: f64 = 1.0;
const STEPS: usize = 20;

/// ~2k-cell structured channel (inlet -> outlet with walls): the flow evolves
/// from rest, so byte-identity is not trivially true on a frozen zero state.
fn channel_mesh() -> Mesh {
    generate_structured_rect_mesh(
        NX,
        NY,
        LX,
        LY,
        BoundarySides {
            left: BoundaryType::Inlet,
            right: BoundaryType::Outlet,
            bottom: BoundaryType::Wall,
            top: BoundaryType::Wall,
        },
    )
}

/// Fixed-dt, fixed-outer-iteration knobs (adaptive dt / auto-converge off) so
/// both models execute the identical step sequence.
fn test_params_with(time_scheme: TimeScheme) -> RuntimeParams {
    RuntimeParams {
        adaptive_dt: false,
        target_cfl: 0.9,
        requested_dt: 0.005,
        dtau: 0.0,
        log_convergence: false,
        log_every_steps: 50,
        // A high-order limited scheme so the deferred-correction path (which
        // multiplies by phi) is exercised, not just plain upwind.
        advection_scheme: Scheme::SecondOrderUpwindVanLeer,
        time_scheme,
        preconditioner: PreconditionerType::Jacobi,
        outer_iters: 8,
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
        allmach_precond_uref_min: 0.2,
        pressure_inlet: false,
        inlet_pressure: 0.0,
    }
}

fn build_driver(mesh: &Mesh, model: ModelSpec, time_scheme: TimeScheme) -> SolverDriver {
    let params = test_params_with(time_scheme);
    let n = mesh.num_cells();
    let DriverBuild { mut driver, .. } = pollster::block_on(SolverDriver::build(
        mesh,
        model,
        &params,
        &vec![(0.0, 0.0); n],
        &vec![0.0; n],
        None,
        None,
    ))
    .expect("driver build");
    driver.apply_params(&params);
    driver
}

fn state_bits(driver: &SolverDriver) -> Vec<u32> {
    pollster::block_on(driver.solver().read_state_f32())
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// Run the static and ALE models side by side on the given CPU engine and
/// time scheme, asserting bit-equality of the full state every step. Euler
/// exercises the BDF1 moving-volume ddt branch directly; BDF2 exercises the
/// volume-ratio-weighted history chain (plus the step-0 Euler startup fallback
/// on the from-rest state).
fn assert_zero_flux_equivalence(engine: &str, time_scheme: TimeScheme) {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::set_var("CFD2_BACKEND", "cpu");
    std::env::set_var("CFD2_CPU_ENGINE", engine);
    let result = std::panic::catch_unwind(|| {
        let mesh = channel_mesh();

        let mut base = build_driver(
            &mesh,
            incompressible_momentum_model().expect("model"),
            time_scheme,
        );
        let mut ale = build_driver(
            &mesh,
            incompressible_momentum_ale_model().expect("model"),
            time_scheme,
        );
        assert!(base.solver().is_cpu(), "expected the CPU backend");
        assert!(ale.solver().is_cpu(), "expected the CPU backend");

        for step in 0..STEPS {
            let out_base = base.step(false);
            let out_ale = ale.step(false);
            assert!(
                out_base.diverged.is_none() && out_ale.diverged.is_none(),
                "[{engine}] step {step} diverged (base {:?}, ale {:?})",
                out_base.diverged,
                out_ale.diverged
            );
            let bits_base = state_bits(&base);
            let bits_ale = state_bits(&ale);
            assert_eq!(bits_base.len(), bits_ale.len(), "state length mismatch");
            let ndiff = bits_base
                .iter()
                .zip(&bits_ale)
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(
                ndiff,
                0,
                "[{engine}] step {step}: ALE model with zero mesh_fluxes diverged bitwise from \
                 the static model: {ndiff}/{} state slots differ (max |diff| = {:.3e})",
                bits_base.len(),
                bits_base
                    .iter()
                    .zip(&bits_ale)
                    .map(|(&a, &b)| (f32::from_bits(a) - f32::from_bits(b)).abs())
                    .fold(0.0f32, f32::max),
            );
        }
        println!(
            "[ale-zero-flux] {engine}/{time_scheme:?}: ALE == static bitwise over {STEPS} steps \
             ({} state slots)",
            (NX * NY) as f64
        );
    });
    std::env::remove_var("CFD2_BACKEND");
    std::env::remove_var("CFD2_CPU_ENGINE");
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

/// Interpreter engine, BDF2: executes the KernelProgram IR directly.
#[test]
fn ale_zero_flux_byte_identical_cpu_interpreter() {
    assert_zero_flux_equivalence("interpreter", TimeScheme::BDF2);
}

/// Transpiled engine, BDF2: the compiled-Rust kernels emitted at build time.
#[test]
fn ale_zero_flux_byte_identical_cpu_transpiled() {
    assert_zero_flux_equivalence("transpiled", TimeScheme::BDF2);
}

/// Interpreter engine, Euler: the BDF1 moving-volume branch as the DECLARED
/// scheme (the BDF2 legs only reach it through the step-0 startup fallback on a
/// from-rest state).
#[test]
fn ale_zero_flux_byte_identical_cpu_interpreter_euler() {
    assert_zero_flux_equivalence("interpreter", TimeScheme::Euler);
}

/// Transpiled engine, Euler (same rationale as the interpreter Euler leg).
#[test]
fn ale_zero_flux_byte_identical_cpu_transpiled_euler() {
    assert_zero_flux_equivalence("transpiled", TimeScheme::Euler);
}

/// ALE sequencing/handshake guards, pinned on the cheap CPU backend:
///   * `SolverDriver::begin_ale_step` under `adaptive_dt` must ERROR — the
///     fluxes are SCL-closed against one dt, and an adaptive recompute after
///     the closure silently injects mass;
///   * double-arming `begin_ale_step` without an intervening `step()` must
///     ERROR — it would rotate the volume history twice;
///   * `refresh_mesh` on an ALE model must ERROR — it updates volumes without
///     rotation/fluxes (the seam is `begin_ale_step`).
#[test]
fn ale_sequencing_guards_reject_misuse() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::set_var("CFD2_BACKEND", "cpu");
    std::env::set_var("CFD2_CPU_ENGINE", "interpreter");
    let result = std::panic::catch_unwind(|| {
        let mut mesh = channel_mesh();
        let mut driver = build_driver(
            &mesh,
            incompressible_momentum_ale_model().expect("model"),
            TimeScheme::BDF2,
        );

        // Static-mesh "motion": identical vertices, all-zero closed fluxes.
        let old_vx = mesh.vx.clone();
        let old_vy = mesh.vy.clone();
        mesh.recalculate_geometry();
        let swept = cfd2::solver::mesh::swept_mesh_fluxes_closed(
            &mesh,
            &old_vx,
            &old_vy,
            f64::from(test_params_with(TimeScheme::BDF2).requested_dt),
        )
        .expect("swept fluxes");

        // adaptive_dt guard.
        let mut adaptive = test_params_with(TimeScheme::BDF2);
        adaptive.adaptive_dt = true;
        driver.apply_params(&adaptive);
        let err = driver
            .begin_ale_step(&mesh, &swept.fluxes)
            .expect_err("begin_ale_step must reject adaptive dt");
        assert!(err.contains("adaptive dt"), "unexpected error: {err}");

        // Double-arm guard (fixed dt again).
        driver.apply_params(&test_params_with(TimeScheme::BDF2));
        driver
            .begin_ale_step(&mesh, &swept.fluxes)
            .expect("first begin_ale_step");
        let err = driver
            .begin_ale_step(&mesh, &swept.fluxes)
            .expect_err("double-arm must be rejected");
        assert!(err.contains("twice"), "unexpected error: {err}");
        // step() clears the arm; the next begin_ale_step is legal again.
        let out = driver.step(false);
        assert!(out.diverged.is_none(), "step diverged: {:?}", out.diverged);
        driver
            .begin_ale_step(&mesh, &swept.fluxes)
            .expect("re-arm after step");
        let out = driver.step(false);
        assert!(out.diverged.is_none(), "step diverged: {:?}", out.diverged);

        // refresh_mesh-on-ALE guard.
        let err = driver
            .refresh_mesh(&mesh, cfd2::solver::MeshRefreshLevel::Geometry)
            .expect_err("refresh_mesh on an ALE model must be rejected");
        assert!(err.contains("begin_ale_step"), "unexpected error: {err}");
        println!("[ale-guards] adaptive-dt, double-arm and refresh-on-ALE misuse all rejected");
    });
    std::env::remove_var("CFD2_BACKEND");
    std::env::remove_var("CFD2_CPU_ENGINE");
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

/// Shared GPU protocol: run static + ALE side by side, return
/// (differing-slot count, max abs diff) after `STEPS` steps.
fn run_gpu_pair(time_scheme: TimeScheme) -> Option<(usize, f32)> {
    let ctx = match pollster::block_on(cfd2::solver::gpu::context::GpuContext::new(None, None)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[ale-zero-flux] no GPU adapter ({e}); skipping GPU gate");
            return None;
        }
    };

    let mesh = channel_mesh();
    let params = test_params_with(time_scheme);
    let n = mesh.num_cells();
    let build = |model: ModelSpec| -> SolverDriver {
        let DriverBuild { mut driver, .. } = pollster::block_on(SolverDriver::build(
            &mesh,
            model,
            &params,
            &vec![(0.0, 0.0); n],
            &vec![0.0; n],
            Some(ctx.device.clone()),
            Some(ctx.queue.clone()),
        ))
        .expect("driver build");
        driver.apply_params(&params);
        driver
    };

    let mut base = build(incompressible_momentum_model().expect("model"));
    let mut ale = build(incompressible_momentum_ale_model().expect("model"));
    assert!(!base.solver().is_cpu() && !ale.solver().is_cpu(), "expected the GPU backend");

    for step in 0..STEPS {
        let out_base = base.step(false);
        let out_ale = ale.step(false);
        assert!(
            out_base.diverged.is_none() && out_ale.diverged.is_none(),
            "[gpu] step {step} diverged (base {:?}, ale {:?})",
            out_base.diverged,
            out_ale.diverged
        );
    }
    let bits_base = state_bits(&base);
    let bits_ale = state_bits(&ale);
    assert_eq!(bits_base.len(), bits_ale.len(), "state length mismatch");
    let maxd = bits_base
        .iter()
        .zip(&bits_ale)
        .map(|(&a, &b)| (f32::from_bits(a) - f32::from_bits(b)).abs())
        .fold(0.0f32, f32::max);
    let ndiff = bits_base.iter().zip(&bits_ale).filter(|(a, b)| a != b).count();
    Some((ndiff, maxd))
}

/// GPU leg, Euler: BITWISE. Also the binding-resolution gate — the ALE kernels
/// bind `mesh_fluxes`/`cell_vols_old{,_old}` (group 0 / bindings 8, 9, 15)
/// through `MeshResources::buffer_for_binding_name`, and a resolution gap would
/// fail pipeline/bind-group creation here, before real fluxes are uploaded.
/// Under Euler every ALE delta is an IEEE identity at zero fluxes / equal
/// volume history (`x - rho*0`, `x/x` via the select guard, `(v - v)/dt`), and
/// fp contraction cannot break any of them, so the exact-bit comparison holds
/// on the GPU too. `CFD2_ALLOW_GPU_BYTE_WAIVE=1` downgrades to <1e-6
/// (driver-update escape hatch). Skips without a GPU adapter.
#[test]
fn ale_zero_flux_byte_identical_gpu_euler() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::remove_var("CFD2_BACKEND");
    let Some((ndiff, maxd)) = run_gpu_pair(TimeScheme::Euler) else {
        return;
    };
    if std::env::var("CFD2_ALLOW_GPU_BYTE_WAIVE").as_deref() == Ok("1") {
        eprintln!(
            "[ale-zero-flux] *** GPU BYTE GATE WAIVED: comparing at <1e-6; max|diff| = {maxd:.3e} ***"
        );
        assert!(maxd < 1e-6, "waived GPU comparison exceeded 1e-6: {maxd:.3e}");
    } else {
        assert_eq!(
            ndiff, 0,
            "[gpu/euler] ALE model with zero mesh_fluxes diverged bitwise from the static \
             model: {ndiff} state slots differ (max |diff| = {maxd:.3e})",
        );
    }
    println!("[ale-zero-flux] gpu/euler: ALE == static bitwise over {STEPS} steps");
}

/// GPU leg, BDF2: TOLERANCE-GATED, deliberately NOT bitwise. The moving-volume
/// BDF2 ddt multiplies the history states by volume ratios that are an exact
/// 1.0 here, but Metal fast math reassociates the (textually different) static
/// and ALE rhs chains differently, giving ~1-ulp per-assembly differences even
/// at ratio == 1.0. The CPU legs stay bitwise under BDF2 (the
/// discretization-correctness statement); this leg only bounds the compiler
/// noise, which amplifies through the nonlinear solve to max|diff| ~6.15e-5
/// after 20 steps; cap ~2.5x.
#[test]
fn ale_zero_flux_equivalent_gpu_bdf2() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::remove_var("CFD2_BACKEND");
    let Some((ndiff, maxd)) = run_gpu_pair(TimeScheme::BDF2) else {
        return;
    };
    println!(
        "[ale-zero-flux] gpu/bdf2: ndiff = {ndiff}, max|diff| = {maxd:.3e} over {STEPS} steps \
         (tolerance gate; see tests/ale_metal_fastmath_evidence.rs)"
    );
    assert!(
        maxd < 1.5e-4,
        "[gpu/bdf2] ALE-vs-static difference {maxd:.3e} exceeds the pinned fast-math cap",
    );
}
