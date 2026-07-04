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

use cfd2::meshgen::meshless::{
    assemble_mesh, build_diagram, meshless_seed_points, EngineConfig, MeshlessInput, SeedKind,
};
use cfd2::meshgen::MeshgenTolerances;
use cfd2::sim::{DriverBuild, RuntimeParams, SolverDriver};
use cfd2::solver::gpu::context::GpuContext;
use cfd2::solver::mesh::{
    generate_structured_rect_mesh, BoundarySides, BoundaryType, Mesh, RectangularChannel,
};
use nalgebra::{Point2, Vector2};
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

        // Topology level with a different cell count is rejected (the Tier-B
        // invariant is an unchanged cell count; faces/adjacency may change).
        let err = driver
            .refresh_mesh(&coarse, MeshRefreshLevel::Topology)
            .expect_err("CPU Topology refresh with a different cell count must fail");
        assert!(err.contains("cell count"), "unexpected error: {err}");

        // Topology level with the SAME mesh now succeeds (CPU Tier B stage 3).
        driver
            .refresh_mesh(&mesh, MeshRefreshLevel::Topology)
            .expect("CPU no-op Topology refresh succeeds");

        println!("[mesh-refresh] topology-mismatch and cell-count rejections verified (CPU)");
    });
}

/// A no-op CPU Topology refresh (refresh to the SAME mesh) is byte-invisible:
/// it rebuilds the CSR + reallocates every face/nnz buffer from a deterministic
/// builder and re-scatters the bc tables, leaving cell-indexed state untouched.
/// So refresh-then-step-N == step-N, compared at the exact f32-bit level. We
/// refresh at step 0 and re-apply params (mirroring the GPU topology byte gate):
/// a Topology refresh resets the bc tables to the model seeds (the documented
/// `bc_overrides_reset` contract), so the runtime inlet-velocity override must be
/// re-applied — exactly as a real caller (the GUI/driver) must. Re-applying
/// `apply_params` MID-run would itself re-seed inlet-adjacent state and confound
/// the gate; the mid-run survival of the cell-indexed state (history, warm-start
/// `x`, counters) is instead proven byte-exactly by the snapshot/restore gate
/// above. This gate isolates "the topology-refresh machinery corrupts nothing".
#[test]
#[cfg(feature = "cpu")]
fn noop_topology_refresh_byte_identical_cpu() {
    with_cpu_backend(|| {
        let mesh = channel_mesh();

        let mut refreshed = build_driver(&mesh, None, None);
        assert!(refreshed.solver().is_cpu(), "expected the CPU backend");
        refreshed
            .refresh_mesh(&mesh, MeshRefreshLevel::Topology)
            .expect("no-op topology refresh");
        refreshed.apply_params(&test_params());
        run_steps(&mut refreshed, 8, "cpu-topo-refresh");
        let bits_refreshed = state_bits(&refreshed);

        let mut straight = build_driver(&mesh, None, None);
        run_steps(&mut straight, 8, "cpu-topo-straight");
        let bits_straight = state_bits(&straight);

        assert_bits_equal(&bits_refreshed, &bits_straight, "cpu-topology");
        println!(
            "[mesh-refresh] CPU no-op topology refresh byte-identical over {} state slots",
            bits_straight.len()
        );
    });
}

/// Snapshot → fresh-build → restore reproduces the NEXT step byte-identically
/// on the CPU backend (M2 Tier B stage 3 deliverable). Reference: step 6, then a
/// 7th step. Restored: capture the snapshot after step 6, build a fresh solver,
/// restore, then take ONE step — the resulting state must match the reference's
/// step-7 bits exactly. This proves the snapshot captures the entire stepping
/// state the next step reads (history, warm-start `x`, `step_count` — so the
/// BDF2 path, not the Euler startup, is taken — and the scalar counters). The
/// 2k-cell channel keeps the CPU Schur inner solve on Jacobi (AMG never
/// activates), so the adaptivity flip does not perturb the byte comparison.
#[test]
#[cfg(feature = "cpu")]
fn snapshot_restore_next_step_byte_identical_cpu() {
    with_cpu_backend(|| {
        let mesh = channel_mesh();

        let mut reference = build_driver(&mesh, None, None);
        assert!(reference.solver().is_cpu(), "expected the CPU backend");
        run_steps(&mut reference, 6, "cpu-snap-ref");
        let snap = reference.snapshot();
        assert!(snap.has_history, "CPU snapshot must capture full history");
        run_steps(&mut reference, 1, "cpu-snap-ref");
        let bits_ref = state_bits(&reference);

        let mut restored = build_driver(&mesh, None, None);
        restored.restore(&snap).expect("restore snapshot");
        run_steps(&mut restored, 1, "cpu-snap-restored");
        let bits_restored = state_bits(&restored);

        assert_bits_equal(&bits_restored, &bits_ref, "cpu-snapshot");
        println!(
            "[snapshot] CPU fresh-build+restore reproduces the next step byte-identically \
             over {} state slots",
            bits_ref.len()
        );
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

/// GPU snapshot/restore roundtrip preserves the CURRENT state exactly (M2 Tier B
/// stage 3). The GPU capture is current-state-only (`has_history == false`); its
/// restore uses the exact `write_state`/`read_state` path, so a fresh solver
/// restored from a snapshot has bit-identical CURRENT state. (Full GPU history
/// capture — needed to reproduce the next *step* byte-identically on the GPU —
/// is a later stage; the byte-exact next-step proof lives on the CPU above.)
#[test]
fn snapshot_restore_current_state_roundtrip_gpu() {
    let _guard = lock_env();
    std::env::remove_var("CFD2_BACKEND");

    let ctx = match pollster::block_on(GpuContext::new(None, None)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[snapshot] no GPU adapter ({e}); skipping GPU snapshot roundtrip");
            return;
        }
    };

    let mesh = channel_mesh();
    let mut a = build_driver(&mesh, Some(ctx.device.clone()), Some(ctx.queue.clone()));
    assert!(!a.solver().is_cpu(), "expected the GPU backend");
    run_steps(&mut a, 5, "gpu-snap-a");
    let snap = a.snapshot();
    assert!(!snap.has_history, "GPU snapshot is current-state-only");
    let bits_a = state_bits(&a);

    let mut b = build_driver(&mesh, Some(ctx.device.clone()), Some(ctx.queue.clone()));
    b.restore(&snap).expect("restore snapshot");
    let bits_b = state_bits(&b);

    assert_bits_equal(&bits_a, &bits_b, "gpu-snapshot-state");
    println!(
        "[snapshot] GPU snapshot/restore preserves the current state exactly over {} slots",
        bits_a.len()
    );
}

// ─── GPU Topology refresh (M2 Tier B stage 2) ────────────────────────────────

/// Build a driver with a uniform initial velocity (an all-wall meshless cavity
/// needs a non-rest IC for the coupled solve to do real work each step).
fn build_driver_ic(
    mesh: &Mesh,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    u0: (f64, f64),
) -> SolverDriver {
    let params = test_params();
    let n = mesh.num_cells();
    let DriverBuild { mut driver, .. } = pollster::block_on(SolverDriver::build(
        mesh,
        incompressible_momentum_model().expect("model"),
        &params,
        &vec![u0; n],
        &vec![0.0; n],
        device,
        queue,
    ))
    .expect("driver build");
    driver.apply_params(&params);
    driver
}

/// A meshless-Voronoi channel with an optionally shifted interior seed. Same
/// seed set ⇒ same cell count (seed `i` = cell `i`); shifting one interior seed
/// flips the local Voronoi adjacency, so the face set / nnz genuinely differ —
/// exactly the "same cells, different topology" input a Tier B refresh must
/// absorb. Built via the low-level seed path so the seed set is controlled.
fn meshless_channel(shift_interior: Option<f64>) -> Mesh {
    let geo = RectangularChannel {
        length: 2.0,
        height: 1.0,
    };
    let min_cell = 0.08;
    let domain = Vector2::new(2.0, 1.0);
    let (mut seeds, kinds, spec) =
        meshless_seed_points(&geo, min_cell, min_cell, 1.2, domain);
    if let Some(amp) = shift_interior {
        // Shift the first interior seed nearest the domain centre by `amp` ×
        // min_cell in a fixed diagonal direction (deterministic).
        let target = seeds
            .iter()
            .enumerate()
            .filter(|(i, _)| matches!(kinds[*i], SeedKind::Interior))
            .min_by(|(_, a), (_, b)| {
                let da = (a.x - 1.0).hypot(a.y - 0.5);
                let db = (b.x - 1.0).hypot(b.y - 0.5);
                da.partial_cmp(&db).unwrap()
            })
            .map(|(i, _)| i)
            .expect("mesh has an interior seed");
        seeds[target] = Point2::new(seeds[target].x + amp * min_cell, seeds[target].y + amp * min_cell);
    }
    let tol = MeshgenTolerances::from_geometry(min_cell, domain);
    let input = MeshlessInput {
        seeds: &seeds,
        kinds: &kinds,
        boundary: &spec,
        domain,
        tol: &tol,
        cfg: EngineConfig::default(),
    };
    let diagram = build_diagram(&input);
    assemble_mesh(&input, &diagram)
}

/// A no-op Topology refresh (refresh to the SAME mesh, before stepping) must be
/// byte-invisible: it reallocates every face/nnz buffer + rebuilds all bind
/// groups from a *deterministic* CSR, so a solver that refreshes then runs N
/// steps is bit-identical to one that just runs N steps. This is the
/// "topology-refresh machinery corrupts nothing" gate (design §1.4 acceptance
/// (b)). We refresh at step 0 because the Tier B refresh reconstructs the
/// linear system (re-zeroing the warm-start `x`); at step 0 `x` is zero on both
/// legs, so the gate is unconditional. Exact per-device; the
/// `CFD2_ALLOW_GPU_BYTE_WAIVE=1` escape mirrors the geometry gate.
#[test]
fn noop_topology_refresh_byte_identical_gpu() {
    let _guard = lock_env();
    std::env::remove_var("CFD2_BACKEND");

    let ctx = match pollster::block_on(GpuContext::new(None, None)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[mesh-refresh] no GPU adapter ({e}); skipping GPU topology byte gate");
            return;
        }
    };

    let mesh = channel_mesh();

    let mut refreshed = build_driver(&mesh, Some(ctx.device.clone()), Some(ctx.queue.clone()));
    assert!(!refreshed.solver().is_cpu(), "expected the GPU backend");
    refreshed
        .refresh_mesh(&mesh, MeshRefreshLevel::Topology)
        .expect("no-op topology refresh");
    // A Topology refresh re-derives the bc tables from the model spec, so the
    // runtime inlet-velocity override (applied at build via `apply_params`) is
    // reset — this is the documented `bc_overrides_reset` caller contract. Re-
    // apply it, exactly as a real caller (the GUI/driver) must on a refresh.
    refreshed.apply_params(&test_params());
    run_steps(&mut refreshed, 8, "gpu-topo-refresh");
    let bits_refreshed = state_bits(&refreshed);

    let mut straight = build_driver(&mesh, Some(ctx.device.clone()), Some(ctx.queue.clone()));
    run_steps(&mut straight, 8, "gpu-topo-straight");
    let bits_straight = state_bits(&straight);

    let waive = std::env::var("CFD2_ALLOW_GPU_BYTE_WAIVE").as_deref() == Ok("1");
    if waive {
        let maxd = max_abs_diff(&bits_refreshed, &bits_straight);
        eprintln!(
            "[mesh-refresh] *** GPU TOPOLOGY BYTE GATE WAIVED: max|diff| = {maxd:.3e} ***"
        );
        assert!(maxd < 1e-6, "no-op topology refresh diff {maxd:.3e} exceeds waived 1e-6");
    } else {
        assert_bits_equal(&bits_refreshed, &bits_straight, "gpu-topology");
        println!(
            "[mesh-refresh] GPU no-op topology refresh byte-identical over {} state slots",
            bits_straight.len()
        );
    }
}

/// Refresh to a *genuinely different* topology (same cell count, one interior
/// Voronoi seed moved to flip adjacency ⇒ different faces/nnz): the solver
/// rebuilds its mesh/CSR/linear stack/bind groups and keeps stepping — finite,
/// non-divergent, with the coupled FGMRES/Schur solve converging on the new
/// sparsity (design §1.4 acceptance, Tier B). Skips when no GPU is available.
#[test]
fn topology_refresh_different_mesh_converges_gpu() {
    let _guard = lock_env();
    std::env::remove_var("CFD2_BACKEND");

    let ctx = match pollster::block_on(GpuContext::new(None, None)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[mesh-refresh] no GPU adapter ({e}); skipping GPU topology smoke");
            return;
        }
    };

    // Cell/face-adjacency signature: a Voronoi seed move swaps neighbours (the
    // face COUNT can stay fixed while the adjacency — owner/neighbour of each
    // face, hence the CSR column pattern — genuinely changes). That is exactly
    // the topology change a Tier B refresh must rebuild, so we discriminate on
    // the adjacency signature, not the face count.
    let adjacency = |m: &Mesh| -> Vec<(usize, i64)> {
        m.face_owner
            .iter()
            .zip(m.face_neighbor.iter())
            .map(|(&o, n)| (o, n.map(|x| x as i64).unwrap_or(-1)))
            .collect::<Vec<_>>()
    };
    let mesh_a = meshless_channel(None);
    let adj_a = adjacency(&mesh_a);
    let mesh_b = [0.4f64, 0.6, 0.8, 1.0]
        .into_iter()
        .map(|s| meshless_channel(Some(s)))
        .find(|m| m.num_cells() == mesh_a.num_cells() && adjacency(m) != adj_a)
        .expect("some interior-seed shift yields a same-cell-count, different-topology mesh");
    let adjdiff = adjacency(&mesh_b)
        .iter()
        .zip(&adj_a)
        .filter(|(x, y)| x != y)
        .count();

    let mut driver = build_driver_ic(
        &mesh_a,
        Some(ctx.device.clone()),
        Some(ctx.queue.clone()),
        (1.0, 0.0),
    );
    assert!(!driver.solver().is_cpu(), "expected the GPU backend");
    run_steps(&mut driver, 5, "topoB-pre");

    driver
        .refresh_mesh(&mesh_b, MeshRefreshLevel::Topology)
        .expect("topology refresh to a genuinely different mesh");

    for i in 0..12 {
        let out = driver.step(true);
        assert!(
            out.diverged.is_none(),
            "[topoB] step {i} after topology refresh diverged: {:?}",
            out.diverged
        );
        let rb = out.readback.expect("readback requested");
        assert_eq!(rb.stats.nonfinite_u, 0, "[topoB] step {i}: non-finite u");
        assert!(rb.stats.p_finite, "[topoB] step {i}: non-finite p");
    }
    let state = pollster::block_on(driver.solver().read_state_f32());
    assert!(
        state.iter().all(|v| v.is_finite()),
        "[topoB] non-finite state after topology refresh + stepping"
    );
    println!(
        "[mesh-refresh] GPU topology refresh {}c/{}f (adjacency changed in {adjdiff} faces) \
         ran 12 finite steps",
        mesh_a.num_cells(),
        mesh_a.num_faces()
    );
}
