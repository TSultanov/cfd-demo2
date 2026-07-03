//! M2 Tier-A mesh-refresh seam gates (meshless/moving-mesh roadmap,
//! docs/meshless-moving-mesh-roadmap.md §M2).
//!
//! The load-bearing property: a no-op `refresh_mesh(same mesh, Geometry)` must
//! be **byte-invisible** — it rewrites the geometry buffers with identical
//! bytes, so stepping after the refresh is bit-identical to never refreshing.
//! This is the "stale mesh-derived caches" detector: any solver-side cache of
//! geometry that a refresh misses would show up here as a byte diff (and the
//! perturbed-refresh smoke exercises the genuinely-changed-geometry path).
//!
//! Precision note (verified against src/solver/cpu/interpreter.rs): CPU kernel
//! state is stored as **f32 bits in `AtomicU32`** (`Store::F32`); the linear
//! solve's f64 internals never persist beyond `x`. `read_state_f32` marshals
//! those bits 1:1 (`f32::from_bits`), so comparing its output by bit pattern
//! IS the strongest (exact, lossless) comparison the CPU backend exposes.
//! The GPU state buffer is f32; the staging readback in `read_state_f32` is
//! likewise exact.
//!
//! Feature gate: `meshgen` (the `sim::SolverDriver` seam lives under it);
//! CPU-backend tests additionally need `cpu`. These are Tier-1 always-on gates
//! under `--features meshgen,cpu` per the roadmap's validation program — not
//! `dev-tests`-gated.
#![cfg(feature = "meshgen")]

use cfd2::sim::{DriverBuild, RuntimeParams, SolverDriver};
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType, Mesh};
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::model::incompressible_momentum_model;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{
    GpuLowMachPrecondModel, MeshRefreshLevel, PreconditionerType, TimeScheme,
};
use std::sync::Mutex;

/// `CFD2_BACKEND` is process-global and the tests in this binary run on
/// concurrent threads: every test that builds a solver takes this lock, and
/// the CPU tests set/remove the env var strictly inside it.
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn lock_env() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

const NX: usize = 64;
const NY: usize = 32;
const LX: f64 = 2.0;
const LY: f64 = 1.0;

/// ~2k-cell structured channel: inlet -> outlet with walls, so the flow
/// actually evolves from rest (an all-wall cavity with no forcing would make
/// byte-identity trivially true on a frozen zero state).
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
/// the two runs being compared execute the identical step sequence.
fn test_params() -> RuntimeParams {
    RuntimeParams {
        adaptive_dt: false,
        target_cfl: 0.9,
        requested_dt: 0.005,
        dtau: 0.0,
        log_convergence: false,
        log_every_steps: 50,
        advection_scheme: Scheme::Upwind,
        time_scheme: TimeScheme::BDF2,
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
        pressure_inlet: false,
        inlet_pressure: 0.0,
    }
}

fn build_driver(
    mesh: &Mesh,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
) -> SolverDriver {
    let params = test_params();
    let n = mesh.num_cells();
    let DriverBuild { mut driver, .. } = pollster::block_on(SolverDriver::build(
        mesh,
        incompressible_momentum_model().expect("model"),
        &params,
        &vec![(0.0, 0.0); n],
        &vec![0.0; n],
        device,
        queue,
    ))
    .expect("driver build");
    driver.apply_params(&params);
    driver
}

fn run_steps(driver: &mut SolverDriver, n: usize, label: &str) {
    for i in 0..n {
        let out = driver.step(false);
        assert!(
            out.diverged.is_none(),
            "[{label}] step {i} diverged: {:?}",
            out.diverged
        );
    }
}

fn state_bits(driver: &SolverDriver) -> Vec<u32> {
    pollster::block_on(driver.solver().read_state_f32())
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn max_abs_diff(a: &[u32], b: &[u32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (f32::from_bits(x) - f32::from_bits(y)).abs())
        .fold(0.0f32, f32::max)
}

fn assert_bits_equal(a: &[u32], b: &[u32], label: &str) {
    assert_eq!(a.len(), b.len(), "[{label}] state length mismatch");
    let ndiff = a.iter().zip(b).filter(|(x, y)| x != y).count();
    assert_eq!(
        ndiff,
        0,
        "[{label}] no-op geometry refresh changed the stepped state: {ndiff}/{} state \
         slots differ (max |diff| = {:.3e})",
        a.len(),
        max_abs_diff(a, b)
    );
}

/// Deterministically perturb every interior vertex by `amp` × cell size and
/// recompute the geometry. Topology (faces, adjacency, boundary tags) is
/// untouched — exactly a Tier-A `Geometry` refresh input.
fn perturbed_mesh(mesh: &Mesh, amp: f64) -> Mesh {
    let mut m = mesh.clone();
    let h = LX / NX as f64;
    let eps = 1e-9;
    for v in 0..m.num_vertices() {
        let (x, y) = (m.vx[v], m.vy[v]);
        if x > eps && x < LX - eps && y > eps && y < LY - eps {
            // Deterministic pseudo-random direction from the vertex index.
            let a = (v as f64) * 12.9898;
            m.vx[v] += amp * h * a.sin();
            m.vy[v] += amp * h * (a * 1.7).cos();
        }
    }
    m.recalculate_geometry();
    m
}

// ─── CPU backend ────────────────────────────────────────────────────────────

/// Run `body` with the CPU backend selected (env-based, hence the lock).
#[cfg(feature = "cpu")]
fn with_cpu_backend<R>(body: impl FnOnce() -> R) -> R {
    let _guard = lock_env();
    std::env::set_var("CFD2_BACKEND", "cpu");
    std::env::set_var("CFD2_CPU_ENGINE", "interpreter");
    let out = body();
    std::env::remove_var("CFD2_BACKEND");
    std::env::remove_var("CFD2_CPU_ENGINE");
    out
}

/// No-op geometry refresh is byte-invisible on the CPU backend: 5 steps +
/// refresh(same mesh) + 5 steps == 10 straight steps, compared at the exact
/// bit level (see the header precision note: this comparison is lossless).
#[test]
#[cfg(feature = "cpu")]
fn noop_refresh_byte_identical_cpu() {
    with_cpu_backend(|| {
        let mesh = channel_mesh();

        let mut refreshed = build_driver(&mesh, None, None);
        assert!(refreshed.solver().is_cpu(), "expected the CPU backend");
        run_steps(&mut refreshed, 5, "cpu-refresh");
        refreshed
            .refresh_mesh(&mesh, MeshRefreshLevel::Geometry)
            .expect("no-op geometry refresh");
        run_steps(&mut refreshed, 5, "cpu-refresh");
        let bits_refreshed = state_bits(&refreshed);

        let mut straight = build_driver(&mesh, None, None);
        run_steps(&mut straight, 10, "cpu-straight");
        let bits_straight = state_bits(&straight);

        assert_bits_equal(&bits_refreshed, &bits_straight, "cpu");
        println!(
            "[mesh-refresh] CPU no-op refresh byte-identical over {} state slots",
            bits_straight.len()
        );
    });
}

/// Geometry refresh to a genuinely perturbed mesh (interior vertices moved by
/// ~2% of the cell size, `recalculate_geometry`): the solve keeps running and
/// stays finite. No byte gate — refresh≡fresh-build equivalence needs full
/// snapshot/restore (Tier B scope).
#[test]
#[cfg(feature = "cpu")]
fn geometry_refresh_perturbed_smoke() {
    with_cpu_backend(|| {
        let mesh = channel_mesh();
        let perturbed = perturbed_mesh(&mesh, 0.02);

        let mut driver = build_driver(&mesh, None, None);
        run_steps(&mut driver, 5, "perturb-pre");
        driver
            .refresh_mesh(&perturbed, MeshRefreshLevel::Geometry)
            .expect("perturbed geometry refresh");
        for i in 0..10 {
            let out = driver.step(true);
            assert!(
                out.diverged.is_none(),
                "[perturb] step {i} after refresh diverged: {:?}",
                out.diverged
            );
            let rb = out.readback.expect("readback requested");
            assert_eq!(rb.stats.nonfinite_u, 0, "[perturb] step {i}: non-finite u");
            assert!(rb.stats.p_finite, "[perturb] step {i}: non-finite p");
        }
        let state = pollster::block_on(driver.solver().read_state_f32());
        assert!(
            state.iter().all(|v| v.is_finite()),
            "[perturb] non-finite state after perturbed-geometry refresh"
        );
        println!("[mesh-refresh] perturbed-geometry refresh ran 10 finite steps (CPU)");
    });
}

/// `SolverDriver::refresh_mesh` recomputes the cached `min_cell_size` (the
/// adaptive-dt length scale): refreshing with a geometry whose smallest cell
/// shrank must be reflected by the driver's accessor, exactly (same reduction).
#[test]
#[cfg(feature = "cpu")]
fn refresh_recomputes_min_cell_size() {
    with_cpu_backend(|| {
        let mesh = channel_mesh();
        // A strong (30% of h) interior perturbation: cell areas redistribute, so
        // the minimum cell volume strictly shrinks below the uniform h*h.
        let shrunk = perturbed_mesh(&mesh, 0.3);
        let expected: f64 = shrunk
            .cell_vol
            .iter()
            .map(|&v| v.sqrt())
            .fold(f64::INFINITY, f64::min);

        let mut driver = build_driver(&mesh, None, None);
        let before = driver.min_cell();
        assert!(
            expected < before,
            "test setup: perturbation must shrink the min cell ({expected} !< {before})"
        );

        driver
            .refresh_mesh(&shrunk, MeshRefreshLevel::Geometry)
            .expect("geometry refresh");
        assert_eq!(
            driver.min_cell(),
            expected,
            "driver min_cell_size not recomputed on refresh"
        );
        println!(
            "[mesh-refresh] min_cell_size recomputed: {before:.6e} -> {:.6e}",
            driver.min_cell()
        );
    });
}

/// A `Geometry`-level refresh with mismatched topology must be rejected, and
/// `Topology` level is not yet implemented (Tier B).
#[test]
#[cfg(feature = "cpu")]
fn refresh_rejects_topology_mismatch() {
    with_cpu_backend(|| {
        let mesh = channel_mesh();
        let mut driver = build_driver(&mesh, None, None);

        // Different cell/face counts.
        let coarse = generate_structured_rect_mesh(
            NX / 2,
            NY / 2,
            LX,
            LY,
            BoundarySides {
                left: BoundaryType::Inlet,
                right: BoundaryType::Outlet,
                bottom: BoundaryType::Wall,
                top: BoundaryType::Wall,
            },
        );
        let err = driver
            .refresh_mesh(&coarse, MeshRefreshLevel::Geometry)
            .expect_err("refresh with a different-size mesh must fail");
        assert!(
            err.contains("num_cells") || err.contains("num_faces"),
            "unexpected error: {err}"
        );

        // Same dimensions, different boundary classification.
        let walled = generate_structured_rect_mesh(NX, NY, LX, LY, BoundarySides::wall());
        let err = driver
            .refresh_mesh(&walled, MeshRefreshLevel::Geometry)
            .expect_err("refresh with different boundary tags must fail");
        assert!(err.contains("face_boundary"), "unexpected error: {err}");

        // Topology level is Tier B.
        let err = driver
            .refresh_mesh(&mesh, MeshRefreshLevel::Topology)
            .expect_err("Topology refresh is not yet implemented");
        assert!(err.contains("not yet implemented"), "unexpected error: {err}");

        println!("[mesh-refresh] topology-mismatch and Topology-level rejections verified");
    });
}

// ─── GPU backend ────────────────────────────────────────────────────────────

/// No-op geometry refresh is byte-invisible on the GPU backend (the refresh
/// path re-uploads identical bytes into the same buffer objects, so the step
/// stream is unchanged). Exact per-device; `CFD2_ALLOW_GPU_BYTE_WAIVE=1`
/// downgrades the gate to max|diff| < 1e-6 (driver-update escape hatch) with a
/// loud notice. Skips when no GPU adapter is available.
#[test]
fn noop_refresh_byte_identical_gpu() {
    let _guard = lock_env();
    // This test must run the GPU backend even if the ambient environment
    // selects the CPU one.
    std::env::remove_var("CFD2_BACKEND");

    let ctx = match pollster::block_on(cfd2::solver::gpu::context::GpuContext::new(None, None)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[mesh-refresh] no GPU adapter ({e}); skipping GPU byte gate");
            return;
        }
    };

    let mesh = channel_mesh();

    // Both solvers share one device+queue: same adapter, same compiled
    // pipelines, deterministic per-device dispatch stream.
    let mut refreshed = build_driver(
        &mesh,
        Some(ctx.device.clone()),
        Some(ctx.queue.clone()),
    );
    assert!(!refreshed.solver().is_cpu(), "expected the GPU backend");
    run_steps(&mut refreshed, 5, "gpu-refresh");
    refreshed
        .refresh_mesh(&mesh, MeshRefreshLevel::Geometry)
        .expect("no-op geometry refresh");
    run_steps(&mut refreshed, 5, "gpu-refresh");
    let bits_refreshed = state_bits(&refreshed);

    let mut straight = build_driver(
        &mesh,
        Some(ctx.device.clone()),
        Some(ctx.queue.clone()),
    );
    run_steps(&mut straight, 10, "gpu-straight");
    let bits_straight = state_bits(&straight);

    let waive = std::env::var("CFD2_ALLOW_GPU_BYTE_WAIVE").as_deref() == Ok("1");
    if waive {
        let maxd = max_abs_diff(&bits_refreshed, &bits_straight);
        eprintln!(
            "[mesh-refresh] *** GPU BYTE GATE WAIVED (CFD2_ALLOW_GPU_BYTE_WAIVE=1): \
             comparing at <1e-6 instead of exact bits; max|diff| = {maxd:.3e} ***"
        );
        assert!(
            maxd < 1e-6,
            "no-op refresh state diff {maxd:.3e} exceeds even the waived 1e-6 tolerance"
        );
    } else {
        assert_bits_equal(&bits_refreshed, &bits_straight, "gpu");
        println!(
            "[mesh-refresh] GPU no-op refresh byte-identical over {} state slots",
            bits_straight.len()
        );
    }
}

/// GPU leg of the perturbed-geometry smoke: refresh to genuinely different
/// geometry through the COPY_DST `write_buffer` path and keep stepping finite.
/// Skips when no GPU adapter is available.
#[test]
fn geometry_refresh_perturbed_smoke_gpu() {
    let _guard = lock_env();
    std::env::remove_var("CFD2_BACKEND");

    let ctx = match pollster::block_on(cfd2::solver::gpu::context::GpuContext::new(None, None)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[mesh-refresh] no GPU adapter ({e}); skipping GPU perturbed smoke");
            return;
        }
    };

    let mesh = channel_mesh();
    let perturbed = perturbed_mesh(&mesh, 0.02);

    let mut driver = build_driver(&mesh, Some(ctx.device.clone()), Some(ctx.queue.clone()));
    assert!(!driver.solver().is_cpu(), "expected the GPU backend");
    run_steps(&mut driver, 5, "gpu-perturb-pre");
    driver
        .refresh_mesh(&perturbed, MeshRefreshLevel::Geometry)
        .expect("perturbed geometry refresh");
    for i in 0..10 {
        let out = driver.step(true);
        assert!(
            out.diverged.is_none(),
            "[gpu-perturb] step {i} after refresh diverged: {:?}",
            out.diverged
        );
        let rb = out.readback.expect("readback requested");
        assert_eq!(rb.stats.nonfinite_u, 0, "[gpu-perturb] step {i}: non-finite u");
        assert!(rb.stats.p_finite, "[gpu-perturb] step {i}: non-finite p");
    }
    println!("[mesh-refresh] perturbed-geometry refresh ran 10 finite steps (GPU)");
}
