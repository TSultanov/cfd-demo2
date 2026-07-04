//! M2 Tier B mesh-refresh TOPOLOGY gates (meshless/moving-mesh roadmap §M2).
//!
//! Tier A proved a `Geometry` refresh is byte-invisible (same faces, positions
//! moved). Tier B lets the FACE SET / adjacency / nnz change while the cell
//! count stays invariant (seed↔cell identity). These gates prove the topology
//! refresh:
//!   1. corrupts nothing on a no-op (byte-identical stepping, both backends);
//!   2. leaves the solver in a state that steps IDENTICALLY to a fresh build on
//!      the new mesh loaded with the same cell-state (catches any stale
//!      mesh-derived cache by construction);
//!   3. is stable + bounded under many alternating A↔B refresh cycles;
//!   4. rebuilds the byte-identical scalar CSR a fresh build produces.
//!
//! Precision note (identical to the Tier A gate): CPU kernel state is f32 bits
//! in `AtomicU32`; `read_state_f32` marshals them 1:1, so a bit-pattern compare
//! is the exact/lossless CPU comparison. The GPU state buffer is f32; the
//! staging readback is exact. GPU byte gates carry the `CFD2_ALLOW_GPU_BYTE_WAIVE`
//! driver-update escape hatch (max|diff| < 1e-6), mirroring Tier A.
#![cfg(feature = "meshgen")]

use cfd2::meshgen::meshless::{
    assemble_mesh, build_diagram, meshless_seed_points, EngineConfig, MeshlessInput, SeedKind,
};
use cfd2::meshgen::MeshgenTolerances;
use cfd2::sim::{DriverBuild, RuntimeParams, SolverDriver};
use cfd2::solver::gpu::context::GpuContext;
use cfd2::solver::mesh::{
    build_diag_first_scalar_csr, build_sorted_scalar_csr, generate_structured_rect_mesh,
    BoundarySides, BoundaryType, Mesh, RectangularChannel,
};
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::model::incompressible_momentum_model;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{GpuLowMachPrecondModel, MeshRefreshLevel, PreconditionerType, TimeScheme};
use nalgebra::{Point2, Vector2};
use std::sync::Mutex;

/// `CFD2_BACKEND` is process-global and this binary's tests run concurrently:
/// every solver-building test takes this lock; CPU tests set/clear the env var
/// strictly inside it.
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn lock_env() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

const NX: usize = 64;
const NY: usize = 32;
const LX: f64 = 2.0;
const LY: f64 = 1.0;

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

fn build_driver(mesh: &Mesh, device: Option<wgpu::Device>, queue: Option<wgpu::Queue>) -> SolverDriver {
    build_driver_ic(mesh, device, queue, (0.0, 0.0))
}

/// Build with a uniform initial velocity (a meshless cavity needs a non-rest IC
/// for the coupled solve to do real work each step).
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

fn run_steps(driver: &mut SolverDriver, n: usize, label: &str) {
    for i in 0..n {
        let out = driver.step(false);
        assert!(out.diverged.is_none(), "[{label}] step {i} diverged: {:?}", out.diverged);
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
        "[{label}] {ndiff}/{} state slots differ (max |diff| = {:.3e})",
        a.len(),
        max_abs_diff(a, b)
    );
}

/// GPU byte comparison with the shared driver-update escape hatch.
fn assert_gpu_state_match(a: &[u32], b: &[u32], label: &str) {
    if std::env::var("CFD2_ALLOW_GPU_BYTE_WAIVE").as_deref() == Ok("1") {
        let maxd = max_abs_diff(a, b);
        eprintln!("[mesh-refresh] *** GPU BYTE GATE WAIVED ({label}): max|diff| = {maxd:.3e} ***");
        assert!(maxd < 1e-6, "[{label}] diff {maxd:.3e} exceeds waived 1e-6");
    } else {
        assert_bits_equal(a, b, label);
    }
}

/// A meshless-Voronoi channel with an optionally shifted interior seed. Same
/// seed set ⇒ same cell count (seed `i` = cell `i`); shifting one interior seed
/// flips the local Voronoi adjacency, so the face set / nnz genuinely differ.
fn meshless_channel(shift_interior: Option<f64>) -> Mesh {
    let geo = RectangularChannel { length: 2.0, height: 1.0 };
    let min_cell = 0.08;
    let domain = Vector2::new(2.0, 1.0);
    let (mut seeds, kinds, spec) = meshless_seed_points(&geo, min_cell, min_cell, 1.2, domain);
    if let Some(amp) = shift_interior {
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

fn adjacency(m: &Mesh) -> Vec<(usize, i64)> {
    m.face_owner
        .iter()
        .zip(m.face_neighbor.iter())
        .map(|(&o, n)| (o, n.map(|x| x as i64).unwrap_or(-1)))
        .collect()
}

/// Mesh A and a same-cell-count, DIFFERENT-topology mesh B (one interior seed
/// shifted until the adjacency signature changes). `None` if no shift in the
/// probe set yields a valid B (caller skips).
fn meshless_ab() -> Option<(Mesh, Mesh)> {
    let a = meshless_channel(None);
    let adj_a = adjacency(&a);
    let b = [0.4f64, 0.6, 0.8, 1.0]
        .into_iter()
        .map(|s| meshless_channel(Some(s)))
        .find(|m| m.num_cells() == a.num_cells() && adjacency(m) != adj_a)?;
    Some((a, b))
}

// ─── 1. no-op topology refresh is byte-invisible ─────────────────────────────

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

/// No-op `Topology` refresh (refresh to the SAME mesh) is byte-invisible on the
/// CPU: rebuild the CSR + reallocate every face/nnz buffer + re-scatter the bc
/// tables, leaving cell-indexed state untouched ⇒ refresh-then-step-N == step-N.
/// Refresh at step 0 + re-apply params (a Topology refresh resets the bc tables
/// to the model seeds — the `bc_overrides_reset` contract — so the runtime
/// inlet override must be re-applied, exactly as a real caller does).
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
            "[mesh-refresh] CPU no-op topology refresh byte-identical over {} slots",
            bits_straight.len()
        );
    });
}

/// GPU leg of the no-op topology byte gate (refresh at step 0; `x` is zero on
/// both legs there, so the LA-stack reconstruction is unconditional).
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
    refreshed.apply_params(&test_params());
    run_steps(&mut refreshed, 8, "gpu-topo-refresh");
    let bits_refreshed = state_bits(&refreshed);

    let mut straight = build_driver(&mesh, Some(ctx.device.clone()), Some(ctx.queue.clone()));
    run_steps(&mut straight, 8, "gpu-topo-straight");
    let bits_straight = state_bits(&straight);

    assert_gpu_state_match(&bits_refreshed, &bits_straight, "gpu-topology");
    println!(
        "[mesh-refresh] GPU no-op topology refresh byte-identical over {} slots",
        bits_straight.len()
    );
}

// ─── 2. refresh-to-B matches a fresh build on B ──────────────────────────────

/// The load-bearing Tier B equivalence (design §1.4 acceptance): refreshing a
/// solver from mesh A onto a genuinely-different-topology mesh B leaves it in a
/// state that steps IDENTICALLY to a solver freshly built on B and loaded with
/// the same cell-state. Both legs are seeded from ONE snapshot (identical input
/// state — full history on CPU, IC-history on GPU), so the ONLY difference is
/// "reached B via refresh" vs "built fresh on B": any stale mesh-derived cache
/// the refresh missed shows up as a step divergence.
#[test]
#[cfg(feature = "cpu")]
fn topology_refresh_matches_fresh_build_cpu() {
    with_cpu_backend(|| {
        // Pin the CPU Schur inner solve to Jacobi (`CFD2_CPU_SCHUR_AMG=0`) for
        // this equivalence check. On the 243-cell meshless CUT-CELL mesh the
        // adaptive Jacobi→AMG flip fires within the first few steps, so the
        // snapshot carries `schur_amg_active=true`. The refresh correctly RESETS
        // that flag (F8 stale-aggregation: a topology change invalidates the AMG
        // aggregation), so a refreshed leg and a fresh-build+restore leg would
        // take DIFFERENT inner-solve modes (Jacobi vs a freshly-rebuilt AMG) —
        // an adaptive-solver-mode difference, ORTHOGONAL to the mesh-rebuild
        // correctness this gate targets. Pinning the mode removes the confound;
        // the flag then stays false on every leg. (Verified: without the pin the
        // legs diverge at exactly `schur_amg_active`, 5e-4; with it, byte-equal.)
        std::env::set_var("CFD2_CPU_SCHUR_AMG", "0");

        let Some((mesh_a, mesh_b)) = meshless_ab() else {
            eprintln!("[mesh-refresh] no A/B topology pair; skipping");
            std::env::remove_var("CFD2_CPU_SCHUR_AMG");
            return;
        };

        // Evolve a real (non-uniform) state on A, snapshot it (full CPU history).
        let mut src = build_driver_ic(&mesh_a, None, None, (1.0, 0.0));
        run_steps(&mut src, 5, "cpu-mfb-src");
        let snap = src.snapshot();
        assert!(snap.has_history, "CPU snapshot must carry full history");

        // Leg 1: fresh-on-A, restore, refresh A→B, apply_params, one step. The
        // refresh resets the bc tables to the model seeds (`bc_overrides_reset`),
        // so re-apply params to restore the inlet-velocity override — that only
        // rewrites the bc table (`set_inlet_velocity` → `set_boundary_vec2`),
        // never state, so the restored cell-state is untouched.
        let mut leg1 = build_driver_ic(&mesh_a, None, None, (1.0, 0.0));
        leg1.restore(&snap).expect("restore");
        leg1.refresh_mesh(&mesh_b, MeshRefreshLevel::Topology).expect("refresh A->B");
        leg1.apply_params(&test_params());
        run_steps(&mut leg1, 1, "cpu-mfb-leg1");
        let bits1 = state_bits(&leg1);

        // Leg 2: fresh-on-B, restore, one step.
        let mut leg2 = build_driver_ic(&mesh_b, None, None, (1.0, 0.0));
        leg2.restore(&snap).expect("restore");
        run_steps(&mut leg2, 1, "cpu-mfb-leg2");
        let bits2 = state_bits(&leg2);

        std::env::remove_var("CFD2_CPU_SCHUR_AMG");
        assert_bits_equal(&bits1, &bits2, "cpu-refresh-vs-fresh");
        println!(
            "[mesh-refresh] CPU refresh(A→B) steps byte-identically to fresh(B) \
             ({}c/{}f→{}f) over {} slots",
            mesh_a.num_cells(),
            mesh_a.num_faces(),
            mesh_b.num_faces(),
            bits1.len()
        );
    });
}

/// GPU leg of the refresh-vs-fresh equivalence. The GPU snapshot is
/// current-state-only, so both legs restore to the same current state with IC
/// history (identical inputs on B); a fresh AMG hierarchy is built on both legs
/// (refresh reconstructs the LA stack). f32-exact per device (waiver hatch).
#[test]
fn topology_refresh_matches_fresh_build_gpu() {
    let _guard = lock_env();
    std::env::remove_var("CFD2_BACKEND");
    let ctx = match pollster::block_on(GpuContext::new(None, None)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[mesh-refresh] no GPU adapter ({e}); skipping GPU refresh-vs-fresh");
            return;
        }
    };
    let Some((mesh_a, mesh_b)) = meshless_ab() else {
        eprintln!("[mesh-refresh] no A/B topology pair; skipping");
        return;
    };

    let mut src = build_driver_ic(&mesh_a, Some(ctx.device.clone()), Some(ctx.queue.clone()), (1.0, 0.0));
    assert!(!src.solver().is_cpu(), "expected the GPU backend");
    run_steps(&mut src, 5, "gpu-mfb-src");
    let snap = src.snapshot();

    let mut leg1 = build_driver_ic(&mesh_a, Some(ctx.device.clone()), Some(ctx.queue.clone()), (1.0, 0.0));
    leg1.restore(&snap).expect("restore");
    leg1.refresh_mesh(&mesh_b, MeshRefreshLevel::Topology).expect("refresh A->B");
    leg1.apply_params(&test_params()); // restore the bc override the refresh reset (bc-only, not state)
    run_steps(&mut leg1, 1, "gpu-mfb-leg1");
    let bits1 = state_bits(&leg1);

    let mut leg2 = build_driver_ic(&mesh_b, Some(ctx.device.clone()), Some(ctx.queue.clone()), (1.0, 0.0));
    leg2.restore(&snap).expect("restore");
    run_steps(&mut leg2, 1, "gpu-mfb-leg2");
    let bits2 = state_bits(&leg2);

    assert_gpu_state_match(&bits1, &bits2, "gpu-refresh-vs-fresh");
    println!(
        "[mesh-refresh] GPU refresh(A→B) steps == fresh(B) ({}c/{}f→{}f) over {} slots",
        mesh_a.num_cells(),
        mesh_a.num_faces(),
        mesh_b.num_faces(),
        bits1.len()
    );
}

// ─── 3. alternating A↔B refresh cycles are stable + bounded ───────────────────

/// Alternate topology refresh A↔B for 20 cycles, stepping between each: the
/// solve stays finite/non-divergent, and the LOGICAL face/nnz sizes cycle
/// between exactly the two mesh values (no monotonic growth ⇒ bounded memory,
/// old allocations dropped). Also proves the rebuild is deterministic: A's
/// rebuilt CSR is byte-identical on every A-cycle.
///
/// DEVIATION (honest, roadmap-aligned): with the default EXACT `CapacityPlan`
/// (headroom 1.0) each refresh reallocates at the new exact size, so "no
/// reallocation after first growth" (headroom capacity-reuse) is NOT asserted
/// here — that surgical in-place / reserve-to-max path is the M4 per-step-loop
/// optimization (see stage 2/3 deviation notes). Bounded logical sizes +
/// determinism are the properties deliverable at EXACT capacity.
#[test]
#[cfg(feature = "cpu")]
fn alternating_topology_refresh_stable() {
    with_cpu_backend(|| {
        let Some((mesh_a, mesh_b)) = meshless_ab() else {
            eprintln!("[mesh-refresh] no A/B topology pair; skipping");
            return;
        };
        let (fa, fb) = (mesh_a.num_faces(), mesh_b.num_faces());
        // Reference CSRs for the two topologies (the deterministic builder the
        // refresh path calls). Note fa==fb / nnz_a==nnz_b is expected for a
        // single-seed adjacency flip — the ADJACENCY differs, so the CSRs are
        // distinct even at equal size (that is what the byte-equal check tests).
        let csr_a0 = build_diag_first_scalar_csr(&mesh_a);
        let csr_b0 = build_diag_first_scalar_csr(&mesh_b);
        assert_ne!(csr_a0.col_indices, csr_b0.col_indices, "A/B must differ in adjacency");

        let mut driver = build_driver_ic(&mesh_a, None, None, (1.0, 0.0));
        run_steps(&mut driver, 2, "alt-warm");

        for cycle in 0..20 {
            // Even cycle → B, odd → A (keyed by parity, NOT face count: fa==fb).
            let is_a = cycle % 2 == 1;
            let (mesh, want) = if is_a { (&mesh_a, &csr_a0) } else { (&mesh_b, &csr_b0) };
            driver
                .refresh_mesh(mesh, MeshRefreshLevel::Topology)
                .unwrap_or_else(|e| panic!("cycle {cycle} refresh failed: {e}"));
            driver.apply_params(&test_params());

            // Logical CSR sizes track the current mesh exactly (no monotonic
            // growth ⇒ bounded memory), AND the rebuilt CSR is byte-identical to
            // the deterministic builder's output for the current topology.
            let (ro, ci, di, cfmi) = driver
                .solver()
                .debug_scalar_csr()
                .expect("cpu csr accessor");
            assert_eq!(ro.len(), mesh.num_cells() + 1, "cycle {cycle}: row_offsets len");
            assert_eq!(cfmi.len(), mesh.cell_faces.len(), "cycle {cycle}: cell_face_matrix len");
            assert_eq!(ro, want.row_offsets, "cycle {cycle}: row_offsets not deterministic");
            assert_eq!(ci, want.col_indices, "cycle {cycle}: col_indices not deterministic");
            assert_eq!(di, want.diagonal_indices, "cycle {cycle}: diagonal_indices not deterministic");
            assert_eq!(cfmi, want.cell_face_matrix_indices, "cycle {cycle}: cell_face_matrix not deterministic");

            let out = driver.step(true);
            assert!(out.diverged.is_none(), "cycle {cycle} step diverged: {:?}", out.diverged);
            let rb = out.readback.expect("readback");
            assert_eq!(rb.stats.nonfinite_u, 0, "cycle {cycle}: non-finite u");
            assert!(rb.stats.p_finite, "cycle {cycle}: non-finite p");
        }
        println!(
            "[mesh-refresh] 20 alternating A↔B topology refreshes stable + bounded \
             (faces {fa}↔{fb}, both CSRs byte-deterministic every cycle)"
        );
    });
}

// ─── 4. refreshed CSR byte-equals a fresh build on B ─────────────────────────

/// The scalar CSR a refresh produces is byte-identical to a fresh build's, for
/// both backend layouts. CPU is checked at the SOLVER level (refresh a solver
/// A→B, read its four CSR arrays, compare to a fresh solver on B). Both go
/// through the deterministic `build_csr_topology`, so they must agree bit-for-
/// bit. The GPU sorted-CSR layout has no readback plumbing; its refresh↔build
/// equality is covered by the deterministic-builder check below (both the GPU
/// refresh and build call `build_sorted_scalar_csr`) plus the no-op byte gate.
#[test]
#[cfg(feature = "cpu")]
fn csr_rebuild_correctness() {
    let Some((mesh_a, mesh_b)) = meshless_ab() else {
        eprintln!("[mesh-refresh] no A/B topology pair; skipping");
        return;
    };

    // Solver level (CPU): refresh A→B vs fresh build on B.
    with_cpu_backend(|| {
        let mut refreshed = build_driver_ic(&mesh_a, None, None, (1.0, 0.0));
        refreshed.refresh_mesh(&mesh_b, MeshRefreshLevel::Topology).expect("refresh A->B");
        let csr_refreshed = refreshed.solver().debug_scalar_csr().expect("cpu csr");

        let fresh = build_driver_ic(&mesh_b, None, None, (1.0, 0.0));
        let csr_fresh = fresh.solver().debug_scalar_csr().expect("cpu csr");

        assert_eq!(csr_refreshed.0, csr_fresh.0, "CPU row_offsets differ");
        assert_eq!(csr_refreshed.1, csr_fresh.1, "CPU col_indices differ");
        assert_eq!(csr_refreshed.2, csr_fresh.2, "CPU diagonal_indices differ");
        assert_eq!(csr_refreshed.3, csr_fresh.3, "CPU cell_face_matrix_indices differ");
        println!(
            "[mesh-refresh] CPU refreshed CSR byte-equals fresh build on B \
             ({} rows, {} nnz)",
            csr_fresh.0.len() - 1,
            csr_fresh.1.len()
        );
    });

    // Builder level (both layouts): the deterministic pure functions the
    // refresh AND build paths call — same mesh ⇒ byte-identical, both layouts.
    let cpu1 = build_diag_first_scalar_csr(&mesh_b);
    let cpu2 = build_diag_first_scalar_csr(&mesh_b);
    assert_eq!(cpu1.row_offsets, cpu2.row_offsets);
    assert_eq!(cpu1.col_indices, cpu2.col_indices);
    assert_eq!(cpu1.diagonal_indices, cpu2.diagonal_indices);
    assert_eq!(cpu1.cell_face_matrix_indices, cpu2.cell_face_matrix_indices);

    let g1 = build_sorted_scalar_csr(&mesh_b).expect("gpu csr");
    let g2 = build_sorted_scalar_csr(&mesh_b).expect("gpu csr");
    assert_eq!(g1.row_offsets, g2.row_offsets);
    assert_eq!(g1.col_indices, g2.col_indices);
    assert_eq!(g1.diagonal_indices, g2.diagonal_indices);
    assert_eq!(g1.cell_face_matrix_indices, g2.cell_face_matrix_indices);
    println!(
        "[mesh-refresh] CSR builders deterministic on B (cpu {} nnz / gpu {} nnz)",
        cpu1.col_indices.len(),
        g1.col_indices.len()
    );
}

// ─── 5. face-count / nnz CHANGE across the refresh (reallocation path) ────────

/// A structured mesh of `nx`×`ny` UNIT-square (h=0.1) cells. Two different
/// factorizations of the same cell count (e.g. 8×12 vs 4×24 = 96 cells) have
/// genuinely different face and nnz counts, while every cell stays h×h square
/// (well-conditioned — no thin-cell stiffness in the 1-step probe).
fn structured_square(nx: usize, ny: usize) -> Mesh {
    let h = 0.1;
    generate_structured_rect_mesh(
        nx,
        ny,
        nx as f64 * h,
        ny as f64 * h,
        BoundarySides {
            left: BoundaryType::Inlet,
            right: BoundaryType::Outlet,
            bottom: BoundaryType::Wall,
            top: BoundaryType::Wall,
        },
    )
}

/// The meshless A/B pair (`meshless_ab`) flips ADJACENCY at an EQUAL face count
/// (fa==fb for a single-seed shift), so it never resizes a buffer — it leaves
/// the buffer-REALLOCATION-at-a-different-length path (the core Tier B
/// capability: `refresh_face_count(new != old)`, `matrix_values`/`mesh_fluxes`/
/// block-CSR realloc at a new nnz, `init_matrix` capacity sizing on a size
/// delta) unexercised. Two structured meshes with the SAME cell count but a
/// different factorization (8×12 vs 4×24 = 96 square cells) have DIFFERENT face
/// and nnz counts, so refreshing between them resizes every topology-sized
/// buffer. Same equivalence contract as the meshless gate: refresh(A→B) must
/// step byte-identically to a fresh build on B — now across a size delta. Also
/// asserts the scalar CSR nnz genuinely changed (the reallocation happened).
#[test]
#[cfg(feature = "cpu")]
fn topology_refresh_face_count_change_matches_fresh_build_cpu() {
    with_cpu_backend(|| {
        // Same AMG pin rationale as `topology_refresh_matches_fresh_build_cpu`:
        // isolate the mesh-rebuild correctness from the adaptive Jacobi→AMG flip.
        std::env::set_var("CFD2_CPU_SCHUR_AMG", "0");

        let mesh_a = structured_square(8, 12);
        let mesh_b = structured_square(4, 24);
        assert_eq!(mesh_a.num_cells(), mesh_b.num_cells(), "cell count must be invariant");
        assert_ne!(
            mesh_a.num_faces(),
            mesh_b.num_faces(),
            "this gate REQUIRES a face-count change (got {}=={})",
            mesh_a.num_faces(),
            mesh_b.num_faces()
        );

        let mut src = build_driver_ic(&mesh_a, None, None, (1.0, 0.0));
        run_steps(&mut src, 5, "cpu-fcc-src");
        let snap = src.snapshot();
        assert!(snap.has_history, "CPU snapshot must carry full history");

        // Leg 1: fresh-on-A, restore, refresh A→B (resizes buffers), one step.
        let mut leg1 = build_driver_ic(&mesh_a, None, None, (1.0, 0.0));
        leg1.restore(&snap).expect("restore");
        let nnz_a = leg1.solver().debug_scalar_csr().expect("cpu csr").1.len();
        leg1.refresh_mesh(&mesh_b, MeshRefreshLevel::Topology).expect("refresh A->B");
        leg1.apply_params(&test_params());
        let nnz_b = leg1.solver().debug_scalar_csr().expect("cpu csr").1.len();
        assert_ne!(nnz_a, nnz_b, "refresh must resize the scalar CSR (nnz {nnz_a} unchanged)");
        run_steps(&mut leg1, 1, "cpu-fcc-leg1");
        let bits1 = state_bits(&leg1);

        // Leg 2: fresh-on-B, restore, one step.
        let mut leg2 = build_driver_ic(&mesh_b, None, None, (1.0, 0.0));
        leg2.restore(&snap).expect("restore");
        run_steps(&mut leg2, 1, "cpu-fcc-leg2");
        let bits2 = state_bits(&leg2);

        std::env::remove_var("CFD2_CPU_SCHUR_AMG");
        assert_bits_equal(&bits1, &bits2, "cpu-fcc-refresh-vs-fresh");
        println!(
            "[mesh-refresh] CPU refresh(A→B) matches fresh(B) across a FACE-COUNT change \
             ({}c, {}f→{}f, {}→{} nnz) over {} slots",
            mesh_a.num_cells(),
            mesh_a.num_faces(),
            mesh_b.num_faces(),
            nnz_a,
            nnz_b,
            bits1.len()
        );
    });
}

/// GPU leg of the face-count-change equivalence. This is the path that resizes
/// the block-expanded CSR (`matrix_values`/`col_indices`, ~S²× the scalar nnz —
/// review-F7) and the FGMRES/AMG/Schur bind groups over the reallocated buffers
/// (review-F10). Same current-state-snapshot contract as the equal-size GPU
/// gate; f32-exact per device (waiver hatch).
#[test]
fn topology_refresh_face_count_change_matches_fresh_build_gpu() {
    let _guard = lock_env();
    std::env::remove_var("CFD2_BACKEND");
    let ctx = match pollster::block_on(GpuContext::new(None, None)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[mesh-refresh] no GPU adapter ({e}); skipping GPU face-count-change gate");
            return;
        }
    };
    let mesh_a = structured_square(8, 12);
    let mesh_b = structured_square(4, 24);
    assert_eq!(mesh_a.num_cells(), mesh_b.num_cells(), "cell count must be invariant");
    assert_ne!(mesh_a.num_faces(), mesh_b.num_faces(), "this gate REQUIRES a face-count change");

    let mut src = build_driver_ic(&mesh_a, Some(ctx.device.clone()), Some(ctx.queue.clone()), (1.0, 0.0));
    assert!(!src.solver().is_cpu(), "expected the GPU backend");
    run_steps(&mut src, 5, "gpu-fcc-src");
    let snap = src.snapshot();

    let mut leg1 = build_driver_ic(&mesh_a, Some(ctx.device.clone()), Some(ctx.queue.clone()), (1.0, 0.0));
    leg1.restore(&snap).expect("restore");
    leg1.refresh_mesh(&mesh_b, MeshRefreshLevel::Topology).expect("refresh A->B");
    leg1.apply_params(&test_params());
    run_steps(&mut leg1, 1, "gpu-fcc-leg1");
    let bits1 = state_bits(&leg1);

    let mut leg2 = build_driver_ic(&mesh_b, Some(ctx.device.clone()), Some(ctx.queue.clone()), (1.0, 0.0));
    leg2.restore(&snap).expect("restore");
    run_steps(&mut leg2, 1, "gpu-fcc-leg2");
    let bits2 = state_bits(&leg2);

    assert_gpu_state_match(&bits1, &bits2, "gpu-fcc-refresh-vs-fresh");
    println!(
        "[mesh-refresh] GPU refresh(A→B) matches fresh(B) across a FACE-COUNT change \
         ({}c, {}f→{}f) over {} slots",
        mesh_a.num_cells(),
        mesh_a.num_faces(),
        mesh_b.num_faces(),
        bits1.len()
    );
}

// ─── refresh-cost benchmark (deliverable 4; `#[ignore]`d — run explicitly) ────

/// A structured grid sized to ~`target` cells (returns the actual mesh).
fn structured_n(target: usize) -> Mesh {
    let side = (target as f64).sqrt();
    let ny = side.round() as usize;
    let nx = (target / ny).max(1);
    generate_structured_rect_mesh(
        nx,
        ny,
        nx as f64 / ny as f64 * 1.0,
        1.0,
        BoundarySides {
            left: BoundaryType::Inlet,
            right: BoundaryType::Outlet,
            bottom: BoundaryType::Wall,
            top: BoundaryType::Wall,
        },
    )
}

/// Time a no-op `Topology` refresh (refresh to the SAME structured mesh) at ~20k
/// and ~300k cells, both backends. A no-op topology refresh does the FULL Tier B
/// rebuild work (both CSR layouts, reallocate every face/nnz buffer, block-CSR
/// re-expansion on GPU, re-scatter bc, rebuild bind groups / LA stack) — the
/// topology is unchanged, so the cost is faithful. Reports median-of-5 ms.
/// `#[ignore]` (heavy build at 300k); run with `--ignored --nocapture`.
#[test]
#[ignore]
fn bench_topology_refresh_cost() {
    fn median_refresh_ms(driver: &mut SolverDriver, mesh: &Mesh) -> f64 {
        let mut ts = Vec::new();
        for _ in 0..5 {
            let t = std::time::Instant::now();
            driver.refresh_mesh(mesh, MeshRefreshLevel::Topology).expect("refresh");
            // Force GPU completion by a state readback (a device sync point).
            let _ = pollster::block_on(driver.solver().read_state_f32());
            ts.push(t.elapsed().as_secs_f64() * 1e3);
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ts[ts.len() / 2]
    }

    for target in [20_000usize, 300_000] {
        let mesh = structured_n(target);
        let n = mesh.num_cells();

        // GPU.
        if let Ok(ctx) = pollster::block_on(GpuContext::new(None, None)) {
            let _g = lock_env();
            std::env::remove_var("CFD2_BACKEND");
            let mut d = build_driver(&mesh, Some(ctx.device.clone()), Some(ctx.queue.clone()));
            let ms = median_refresh_ms(&mut d, &mesh);
            println!("[refresh-cost] GPU {n} cells / {} faces: {ms:.2} ms/topology-refresh", mesh.num_faces());
        }

        // CPU.
        #[cfg(feature = "cpu")]
        with_cpu_backend(|| {
            let mut d = build_driver(&mesh, None, None);
            let ms = median_refresh_ms(&mut d, &mesh);
            println!("[refresh-cost] CPU {n} cells / {} faces: {ms:.2} ms/topology-refresh", mesh.num_faces());
        });
    }
}
