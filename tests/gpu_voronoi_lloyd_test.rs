//! M1 stage-4 gates for GPU Lloyd/CVT relaxation (`encode_lloyd_iterations`
//! + max-displacement reduce, src/solver/gpu/voronoi/lloyd.rs):
//!
//! 1. CPU/GPU relaxation equivalence: from the SAME f32-rounded Poisson
//!    seed set, N GPU Lloyd iterations vs N iterations of M0 `lloyd_relax`
//!    (same density exponent / omega / sizing) — the assembled meshes'
//!    interior skew statistics (mean, p99) agree within 10% (f32-drift
//!    tolerance: the trajectories decouple at f32 rounding per iteration,
//!    so per-seed positions are NOT compared). Uniform-density obstacle
//!    case + graded nozzle case (ρ = h⁻⁴ with the exact Poisson sizing
//!    rule — the GPU evaluates it through a bilerp node grid).
//! 2. Fixed seeds bit-unmoved: `SeedKind::Boundary` seeds (kind-keyed,
//!    like M0) and `SEED_FLAG_FIXED`-pinned interior seeds keep their f32
//!    bit patterns through the whole chained relaxation.
//! 3. Flag-rate budget after 30 Lloyd iterations ≤ 5e-3 (design gate 2:
//!    the near-hex/cocircular stress) on a 30k interior set, zero
//!    overflow statuses; the rate is printed.
//! 4. Max-displacement reduce validated against a CPU recompute of the
//!    seed movement; the standalone one-f32 convergence loop terminates.
//!
//! Run with:
//!
//! ```sh
//! cargo test --features meshgen --test gpu_voronoi_lloyd_test -- --nocapture
//! ```

#![cfg(feature = "meshgen")]

use cfd2::meshgen::meshless::{
    assemble_mesh, build_diagram, lloyd_relax, meshless_seed_points, BoundarySpec, EngineConfig,
    LloydConfig, MeshlessInput, SeedKind,
};
use cfd2::meshgen::MeshgenTolerances;
use cfd2::solver::gpu::context::GpuContext;
use cfd2::solver::gpu::readback::StagingBufferCache;
use cfd2::solver::gpu::voronoi::{status, GpuVoronoiEngine, SEED_FLAG_FIXED};
use cfd2::solver::mesh::{ChannelWithObstacle, Geometry, Mesh, Nozzle};
use nalgebra::{Point2, Vector2};
use rand::{Rng, SeedableRng};

fn gpu_context() -> Option<GpuContext> {
    match pollster::block_on(GpuContext::new(None, None)) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("SKIP: no GPU adapter available ({e})");
            None
        }
    }
}

fn poll(ctx: &GpuContext, idx: wgpu::SubmissionIndex) {
    let _ = ctx.device.poll(wgpu::PollType::Wait {
        submission_index: Some(idx),
        timeout: None,
    });
}

fn round_f32(seeds: &[Point2<f64>]) -> (Vec<f32>, Vec<Point2<f64>>) {
    let f: Vec<f32> = seeds.iter().flat_map(|p| [p.x as f32, p.y as f32]).collect();
    let r = (0..seeds.len())
        .map(|i| Point2::new(f[2 * i] as f64, f[2 * i + 1] as f64))
        .collect();
    (f, r)
}

/// Interior-face skew samples `1 − |d̂·n̂|` (the solver quality metric,
/// meshgen_ext.rs `calculate_max_skewness` restricted to interior faces).
fn interior_skew(mesh: &Mesh) -> Vec<f64> {
    let mut v = Vec::new();
    for f in 0..mesh.num_faces() {
        if let Some(nb) = mesh.face_neighbor[f] {
            let o = mesh.face_owner[f];
            let dx = mesh.cell_cx[nb] - mesh.cell_cx[o];
            let dy = mesh.cell_cy[nb] - mesh.cell_cy[o];
            let n = (dx * dx + dy * dy).sqrt();
            if n < 1e-14 {
                continue;
            }
            v.push(1.0 - ((dx * mesh.face_nx[f] + dy * mesh.face_ny[f]) / n).abs());
        }
    }
    v
}

/// (mean, p99, max) of a sample vector.
fn stats(mut v: Vec<f64>) -> (f64, f64, f64) {
    assert!(!v.is_empty());
    v.sort_by(f64::total_cmp);
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let p99 = v[((v.len() - 1) as f64 * 0.99) as usize];
    (mean, p99, *v.last().unwrap())
}

/// Two skew statistics agree within 10% (relative to the larger), with a
/// 1e-3 absolute floor guarding near-zero statistics against f32 drift.
fn assert_stat_close(name: &str, what: &str, cpu: f64, gpu: f64) {
    let tol = 0.10 * cpu.abs().max(gpu.abs()) + 1e-3;
    assert!(
        (cpu - gpu).abs() <= tol,
        "[{name}] interior skew {what} diverged: cpu {cpu:.5} vs gpu {gpu:.5} (tol {tol:.5})"
    );
}

/// Full N-vs-N relaxation equivalence protocol for one geometry + sizing.
#[allow(clippy::too_many_arguments)]
fn run_lloyd_case(
    name: &str,
    geo: &(impl Geometry + Sync),
    domain: Vector2<f64>,
    hmin: f64,
    hmax: f64,
    growth: f64,
    sizing_grid: (u32, u32),
    iters: u32,
) {
    let Some(ctx) = gpu_context() else { return };
    let (seeds, kinds, spec) = meshless_seed_points(geo, hmin, hmax, growth, domain);
    let n = seeds.len();
    let tol = MeshgenTolerances::from_geometry(hmin, domain);
    let (seeds_f32, rounded) = round_f32(&seeds);
    // The exact Poisson sizing rule (M0 generate_cvt_mesh); constant for
    // uniform cases (hmin == hmax).
    let sizing = |p: Point2<f64>| -> f64 {
        let dist = geo.sdf(&p).abs();
        (hmin + (growth - 1.0).max(0.0) * dist).min(hmax)
    };
    // FIXED flag mirrors the kind table (exercises the flag upload path;
    // the kernel keys fixedness off the kinds regardless, like M0).
    let flags: Vec<u32> = kinds
        .iter()
        .map(|k| match k {
            SeedKind::Boundary { .. } => SEED_FLAG_FIXED,
            SeedKind::Interior => 0,
        })
        .collect();

    // --- GPU relaxation -------------------------------------------------
    let mut engine = GpuVoronoiEngine::new(&ctx.device, n as u32, domain, &tol);
    engine.upload_case(&ctx.device, &ctx.queue, &seeds_f32, &flags, &kinds, &spec);
    let spec_r = engine.boundary().clone();
    engine.set_lloyd_density(
        &ctx.device,
        &ctx.queue,
        sizing_grid.0,
        sizing_grid.1,
        4.0,
        1.0,
        &sizing,
    );
    // Diagram 0, then the chained iterations (one encoder, no readback).
    poll(&ctx, engine.run_regen(&ctx.device, &ctx.queue));
    let t0 = std::time::Instant::now();
    poll(&ctx, engine.run_lloyd_iterations(&ctx.device, &ctx.queue, iters));
    let gpu_ms = t0.elapsed().as_secs_f64() * 1e3;
    let cache = StagingBufferCache::default();
    let (disp_rel, disp_abs) = engine.read_max_disp(&ctx, &cache);

    // Gate 2: fixed seeds bit-unmoved.
    let relaxed = engine.refresh_after_lloyd(&ctx, &cache);
    let mut n_fixed = 0usize;
    for i in 0..n {
        if matches!(kinds[i], SeedKind::Boundary { .. }) {
            assert_eq!(
                (relaxed[2 * i].to_bits(), relaxed[2 * i + 1].to_bits()),
                (seeds_f32[2 * i].to_bits(), seeds_f32[2 * i + 1].to_bits()),
                "[{name}] fixed boundary seed {i} moved"
            );
            n_fixed += 1;
        }
    }

    // Final canonical diagram: fresh grid, regen, f64 fallback.
    poll(&ctx, engine.run_regen(&ctx.device, &ctx.queue));
    let gpu_cells = engine.read_cells(&ctx, &cache);
    let report = engine.resolve_flagged(&ctx, &cache);
    assert!(report.unresolved.is_empty(), "[{name}] unresolved cells");
    let merged = engine.read_cells(&ctx, &cache);
    let diag = engine.cells_to_diagram(&merged);
    let relaxed_pts: Vec<Point2<f64>> = (0..n)
        .map(|i| Point2::new(relaxed[2 * i] as f64, relaxed[2 * i + 1] as f64))
        .collect();
    let gpu_input = MeshlessInput {
        seeds: &relaxed_pts,
        kinds: &kinds,
        boundary: &spec_r,
        domain,
        tol: &tol,
        cfg: EngineConfig::default(),
    };
    let gpu_mesh = assemble_mesh(&gpu_input, &diag);

    // --- CPU relaxation (M0 lloyd_relax, N forced iterations) ------------
    let mut seeds_cpu = rounded.clone();
    let lcfg = LloydConfig {
        max_iters: iters as usize,
        tol_disp: 0.0, // force exactly `iters` iterations
        omega: 1.0,
        density_exponent: 4.0,
    };
    let cstats = lloyd_relax(
        &mut seeds_cpu,
        &kinds,
        &spec_r,
        &sizing,
        domain,
        &tol,
        &EngineConfig::default(),
        &lcfg,
    );
    assert_eq!(cstats.iters, iters as usize);
    let cpu_input = MeshlessInput {
        seeds: &seeds_cpu,
        kinds: &kinds,
        boundary: &spec_r,
        domain,
        tol: &tol,
        cfg: EngineConfig::default(),
    };
    let cpu_diag = build_diagram(&cpu_input);
    let cpu_mesh = assemble_mesh(&cpu_input, &cpu_diag);

    // Pre-relaxation baseline for the improvement report.
    let base_input = MeshlessInput {
        seeds: &rounded,
        kinds: &kinds,
        boundary: &spec_r,
        domain,
        tol: &tol,
        cfg: EngineConfig::default(),
    };
    let base_mesh = assemble_mesh(&base_input, &build_diagram(&base_input));

    // --- Gate 1: interior skew statistics within 10% ----------------------
    let (bm, bp, bx) = stats(interior_skew(&base_mesh));
    let (cm, cp, cx) = stats(interior_skew(&cpu_mesh));
    let (gm, gp, gx) = stats(interior_skew(&gpu_mesh));
    println!(
        "[{name}] n={n} fixed={n_fixed} iters={iters} gpu_wall={gpu_ms:.1}ms \
         last_disp=(rel {disp_rel:.2e}, abs {disp_abs:.2e})"
    );
    println!("[{name}] skew initial: mean {bm:.5} p99 {bp:.5} max {bx:.5}");
    println!("[{name}] skew cpu:     mean {cm:.5} p99 {cp:.5} max {cx:.5}");
    println!("[{name}] skew gpu:     mean {gm:.5} p99 {gp:.5} max {gx:.5}");
    println!(
        "[{name}] final flag rate {:.2e} ({} of {n})",
        gpu_cells.flagged.len() as f64 / n as f64,
        gpu_cells.flagged.len()
    );
    assert_stat_close(name, "mean", cm, gm);
    assert_stat_close(name, "p99", cp, gp);
    assert!(
        gm < bm,
        "[{name}] GPU Lloyd failed to improve mean interior skew ({gm:.5} vs initial {bm:.5})"
    );

    // Both relaxed meshes must remain structurally sound.
    for (which, mesh) in [("cpu", &cpu_mesh), ("gpu", &gpu_mesh)] {
        assert_eq!(mesh.num_cells() > 0, true, "[{name}] {which} mesh empty");
        let neg = mesh.cell_vol.iter().filter(|&&v| !(v > 0.0)).count();
        assert_eq!(neg, 0, "[{name}] {which} mesh has non-positive cell volumes");
    }
}

#[test]
fn gpu_lloyd_uniform_obstacle_matches_cpu() {
    run_lloyd_case(
        "lloyd-obstacle/h=0.05",
        &ChannelWithObstacle {
            length: 3.0,
            height: 1.0,
            obstacle_center: Point2::new(1.0, 0.51),
            obstacle_radius: 0.1,
        },
        Vector2::new(3.0, 1.0),
        0.05,
        0.05,
        1.2,
        (1, 1), // constant sizing: a single bilerp cell suffices
        30,
    );
}

#[test]
fn gpu_lloyd_graded_nozzle_matches_cpu() {
    run_lloyd_case(
        "lloyd-nozzle-graded/h=0.02..0.08",
        &Nozzle {
            length: 3.0,
            height: 1.0,
            throat_height: 0.40,
            throat_frac: 0.40,
            exit_height: 0.80,
        },
        Vector2::new(3.0, 1.0),
        0.02,
        0.08,
        1.2,
        (300, 100), // pitch 0.01 = hmin/2 — resolves the sdf grading
        15,
    );
}

/// Design gate 2 stress: 30 Lloyd iterations drive a jittered-lattice 30k
/// interior set toward the near-hex (near-cocircular everywhere) CVT — the
/// epsilon filter's worst case. Budget: flag rate ≤ 5e-3 on the fresh-grid
/// final regen, zero overflows.
#[test]
fn gpu_lloyd_flag_rate_after_30_iters_30k() {
    let Some(ctx) = gpu_context() else { return };
    let domain = Vector2::new(2.0, 1.0);
    let (nx, ny) = (245usize, 123usize);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xCFD2_5EED);
    let (sx, sy) = (domain.x / nx as f64, domain.y / ny as f64);
    let mut seeds = Vec::with_capacity(nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            let jx = (rng.gen::<f64>() - 0.5) * 0.8 * sx;
            let jy = (rng.gen::<f64>() - 0.5) * 0.8 * sy;
            seeds.push(Point2::new(
                (i as f64 + 0.5) * sx + jx,
                (j as f64 + 0.5) * sy + jy,
            ));
        }
    }
    let n = seeds.len();
    let tol = MeshgenTolerances::from_geometry(0.5 * sy, domain);
    let (seeds_f32, _) = round_f32(&seeds);
    let flags = vec![0u32; n];

    let mut engine = GpuVoronoiEngine::new(&ctx.device, n as u32, domain, &tol);
    engine.upload_seeds(&ctx.device, &ctx.queue, &seeds_f32, &flags);
    engine.set_lloyd_density(&ctx.device, &ctx.queue, 1, 1, 4.0, 1.0, &|_| sy);
    poll(&ctx, engine.run_regen(&ctx.device, &ctx.queue));
    poll(&ctx, engine.run_lloyd_iterations(&ctx.device, &ctx.queue, 30));

    let cache = StagingBufferCache::default();
    // Flag rate of the LAST chained (stale-grid + slack) regen, reported
    // for context; the gate applies to the canonical fresh-grid regen.
    let stale = engine.read_cells(&ctx, &cache);
    let stale_rate = stale.flagged.len() as f64 / n as f64;

    // The stress is only meaningful if the chained iterations actually
    // relaxed the set: virtually every interior seed must have moved.
    let relaxed = engine.read_seeds(&ctx, &cache);
    let moved = (0..n)
        .filter(|&i| {
            relaxed[2 * i].to_bits() != seeds_f32[2 * i].to_bits()
                || relaxed[2 * i + 1].to_bits() != seeds_f32[2 * i + 1].to_bits()
        })
        .count();
    assert!(
        moved * 10 >= n * 9,
        "only {moved}/{n} seeds moved — chained Lloyd did not run"
    );

    engine.refresh_after_lloyd(&ctx, &cache);
    poll(&ctx, engine.run_regen(&ctx.device, &ctx.queue));
    let cells = engine.read_cells(&ctx, &cache);
    let rate = cells.flagged.len() as f64 / n as f64;
    let overflows = cells
        .status
        .iter()
        .filter(|&&s| s == status::VERT_OVERFLOW || s == status::FACE_OVERFLOW)
        .count();
    println!(
        "[lloyd-30k] n={n} post-30-iters flag rate: {rate:.3e} ({} cells; \
         last chained regen {stale_rate:.3e} / {}) overflows={overflows}",
        cells.flagged.len(),
        stale.flagged.len()
    );
    assert_eq!(overflows, 0, "overflow statuses after Lloyd");
    let budget = ((n as f64) * 5e-3).ceil() as usize;
    assert!(
        cells.flagged.len() <= budget,
        "post-Lloyd flag rate over the 5e-3 budget: {} > {budget}",
        cells.flagged.len()
    );
    // The relaxed diagram must still resolve cleanly end to end.
    let report = engine.resolve_flagged(&ctx, &cache);
    assert!(report.unresolved.is_empty());
}

/// Reduce correctness + the standalone convergence loop + FIXED-flag
/// pinning of interior seeds:
///  - the (abs, rel) max-displacement readback matches a CPU recompute
///    from before/after seed snapshots;
///  - one-f32 convergence polling reaches the M0 default tol (0.01 h) and
///    a converged CVT is a near-fixed-point;
///  - `SEED_FLAG_FIXED` interior seeds keep their bits.
#[test]
fn gpu_lloyd_reduce_convergence_and_pinning() {
    let Some(ctx) = gpu_context() else { return };
    let domain = Vector2::new(2.0, 1.0);
    let (nx, ny) = (50usize, 25usize);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x11_00D5);
    let (sx, sy) = (domain.x / nx as f64, domain.y / ny as f64);
    let mut seeds = Vec::with_capacity(nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            let jx = (rng.gen::<f64>() - 0.5) * 0.8 * sx;
            let jy = (rng.gen::<f64>() - 0.5) * 0.8 * sy;
            seeds.push(Point2::new(
                (i as f64 + 0.5) * sx + jx,
                (j as f64 + 0.5) * sy + jy,
            ));
        }
    }
    let n = seeds.len();
    let tol = MeshgenTolerances::from_geometry(0.5 * sy, domain);
    let (seeds_f32, _) = round_f32(&seeds);
    // Pin a scattering of interior seeds by flag alone.
    let pinned: Vec<usize> = vec![7, 123, 456, 789, 1111];
    let mut flags = vec![0u32; n];
    for &i in &pinned {
        flags[i] = SEED_FLAG_FIXED;
    }

    let mut engine = GpuVoronoiEngine::new(&ctx.device, n as u32, domain, &tol);
    engine.upload_case(
        &ctx.device,
        &ctx.queue,
        &seeds_f32,
        &flags,
        &[],
        &BoundarySpec::empty(),
    );
    engine.set_lloyd_density(&ctx.device, &ctx.queue, 1, 1, 4.0, 1.0, &|_| sy);
    poll(&ctx, engine.run_regen(&ctx.device, &ctx.queue));

    let cache = StagingBufferCache::default();
    // One iteration; validate the reduce against a CPU recompute.
    let before = engine.read_seeds(&ctx, &cache);
    poll(&ctx, engine.run_lloyd_iterations(&ctx.device, &ctx.queue, 1));
    let after = engine.read_seeds(&ctx, &cache);
    let (rel, abs) = engine.read_max_disp(&ctx, &cache);
    let mut max_abs = 0.0f64;
    for i in 0..n {
        let dx = after[2 * i] as f64 - before[2 * i] as f64;
        let dy = after[2 * i + 1] as f64 - before[2 * i + 1] as f64;
        max_abs = max_abs.max((dx * dx + dy * dy).sqrt());
    }
    println!(
        "[lloyd-reduce] max disp: gpu (abs {abs:.6e}, rel {rel:.6e}) cpu-recompute {max_abs:.6e}"
    );
    // The kernel's |d| is computed pre-rounding of p+d; allow f32 quanta.
    assert!(
        (abs as f64 - max_abs).abs() <= 1e-5 * max_abs.max(sy),
        "reduce abs max {abs:.6e} disagrees with recompute {max_abs:.6e}"
    );
    let expect_rel = max_abs / sy;
    assert!(
        (rel as f64 - expect_rel).abs() <= 1e-4 * expect_rel.max(1.0),
        "reduce rel max {rel:.6e} disagrees with recompute {expect_rel:.6e}"
    );

    // Standalone convergence loop: one-f32 polling to the M0 default tol.
    let mut iters_run = 1u32;
    let mut last_rel = rel;
    while last_rel >= 0.01 && iters_run < 300 {
        poll(&ctx, engine.run_lloyd_iterations(&ctx.device, &ctx.queue, 1));
        last_rel = engine.read_max_disp(&ctx, &cache).0;
        iters_run += 1;
    }
    println!("[lloyd-reduce] converged to rel {last_rel:.3e} after {iters_run} iterations");
    assert!(
        last_rel < 0.01,
        "Lloyd failed to converge to 0.01 h in {iters_run} iterations (last {last_rel:.3e})"
    );
    // Near-fixed-point: one more iteration stays small.
    poll(&ctx, engine.run_lloyd_iterations(&ctx.device, &ctx.queue, 1));
    let (post_rel, _) = engine.read_max_disp(&ctx, &cache);
    assert!(
        post_rel < 0.02,
        "converged CVT moved {post_rel:.3e} h on the next iteration"
    );

    // Pinned interior seeds bit-unmoved through everything.
    let final_seeds = engine.read_seeds(&ctx, &cache);
    for &i in &pinned {
        assert_eq!(
            (final_seeds[2 * i].to_bits(), final_seeds[2 * i + 1].to_bits()),
            (seeds_f32[2 * i].to_bits(), seeds_f32[2 * i + 1].to_bits()),
            "pinned interior seed {i} moved"
        );
    }
}
