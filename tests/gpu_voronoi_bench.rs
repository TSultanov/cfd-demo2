//! GPU Voronoi engine benchmark — REPORT-ONLY, no perf gates (integrated
//! GPUs run ~4-8x slower than the discrete-GPU numbers this targets).
//!
//! Per seed count (default 100k and 300k): per-phase wall times for
//! seed generation (CPU Poisson, context only), `SeedGrid::build` (the CPU
//! grid leg of upload), `upload_case` (grid + coalescing + write_buffers),
//! the regen kernel (submit→poll, best/mean of 5 after warmup), the full
//! validation readback (`read_cells`), the f64 fallback (`resolve_flagged`,
//! including its reciprocity readback), and chained Lloyd iterations
//! (per-iteration cost of `lloyd_update` + max-reduce + regen). Plus the
//! conservation sanity: Σ cell areas vs the boundary-loop shoelace area
//! (f64 sum over the f32 outputs).
//!
//! Env knobs:
//!   CFD2_BENCH_SEEDS        comma list of target seed counts (default
//!                           "100000,300000")
//!   CFD2_BENCH_LLOYD_ITERS  chained Lloyd iterations to time (default 10)
//!
//! Run with (release matters — several phases are CPU legs):
//!
//! ```sh
//! cargo test --release --features "meshgen dev-tests" \
//!   --test gpu_voronoi_bench -- --nocapture
//! ```

#![cfg(all(feature = "meshgen", feature = "dev-tests"))]

use std::time::Instant;

use cfd2::meshgen::meshless::{meshless_seed_points, SeedGrid, SeedKind};
use cfd2::meshgen::MeshgenTolerances;
use cfd2::solver::gpu::context::GpuContext;
use cfd2::solver::gpu::readback::StagingBufferCache;
use cfd2::solver::gpu::voronoi::{status, GpuVoronoiEngine, SEED_FLAG_FIXED};
use cfd2::solver::mesh::ChannelWithObstacle;
use nalgebra::{Point2, Vector2};

fn env_list(key: &str, default: &[usize]) -> Vec<usize> {
    std::env::var(key)
        .ok()
        .map(|v| {
            v.split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect::<Vec<usize>>()
        })
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| default.to_vec())
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

#[test]
fn gpu_voronoi_bench() {
    let Ok(ctx) = pollster::block_on(GpuContext::new(None, None)) else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };
    let targets = env_list("CFD2_BENCH_SEEDS", &[100_000, 300_000]);
    let lloyd_iters = env_u32("CFD2_BENCH_LLOYD_ITERS", 10);

    let domain = Vector2::new(3.0, 1.0);
    let geo = ChannelWithObstacle {
        length: 3.0,
        height: 1.0,
        obstacle_center: Point2::new(1.0, 0.51),
        obstacle_radius: 0.1,
    };
    let fluid_area = 3.0 - std::f64::consts::PI * 0.1 * 0.1;

    for &target in &targets {
        // Poisson-disk density on this geometry ≈ 0.716 seeds/h²; pick h to
        // land near the target count.
        let h = (0.716 * fluid_area / target as f64).sqrt();

        let t = Instant::now();
        let (seeds, kinds, spec) = meshless_seed_points(&geo, h, h, 1.2, domain);
        let t_gen = ms(t);
        let n = seeds.len();
        let n_boundary = kinds
            .iter()
            .filter(|k| matches!(k, SeedKind::Boundary { .. }))
            .count();
        let tol = MeshgenTolerances::from_geometry(h, domain);
        let seeds_f32: Vec<f32> = seeds
            .iter()
            .flat_map(|p| [p.x as f32, p.y as f32])
            .collect();
        let rounded: Vec<Point2<f64>> = (0..n)
            .map(|i| Point2::new(seeds_f32[2 * i] as f64, seeds_f32[2 * i + 1] as f64))
            .collect();
        let flags: Vec<u32> = kinds
            .iter()
            .map(|k| match k {
                SeedKind::Boundary { .. } => SEED_FLAG_FIXED,
                SeedKind::Interior => 0,
            })
            .collect();

        // CPU grid-build leg alone (it also runs inside upload_case).
        let t = Instant::now();
        let grid = SeedGrid::build(&rounded, domain);
        let t_grid = ms(t);
        let n_bins = grid.gw * grid.gh;
        drop(grid);

        let mut engine = GpuVoronoiEngine::new(&ctx.device, n as u32, domain, &tol);
        let t = Instant::now();
        engine.upload_case(&ctx.device, &ctx.queue, &seeds_f32, &flags, &kinds, &spec);
        let t_upload = ms(t);

        let poll = |idx: wgpu::SubmissionIndex| {
            let _ = ctx.device.poll(wgpu::PollType::Wait {
                submission_index: Some(idx),
                timeout: None,
            });
        };

        // Warmup (pipeline compile / first-touch), then timed regens.
        poll(engine.run_regen(&ctx.device, &ctx.queue));
        let mut best = f64::INFINITY;
        let mut total = 0.0;
        const REPS: usize = 5;
        for _ in 0..REPS {
            let t = Instant::now();
            poll(engine.run_regen(&ctx.device, &ctx.queue));
            let dt = ms(t);
            best = best.min(dt);
            total += dt;
        }

        let cache = StagingBufferCache::default();
        let t = Instant::now();
        let cells = engine.read_cells(&ctx, &cache);
        let t_readback = ms(t);

        let t = Instant::now();
        let report = engine.resolve_flagged(&ctx, &cache);
        let t_fallback = ms(t);

        // Conservation sanity: Σ areas of the RESOLVED pre-Lloyd diagram vs
        // the boundary-loop shoelace area — captured before Lloyd perturbs
        // the outputs.
        let merged = engine_area_sum(&ctx, &cache, &engine, n);
        let expected: f64 = spec.loops.iter().map(|lp| lp.signed_area()).sum();

        // Lloyd: constant density (uniform h), chained iterations.
        engine.set_lloyd_density(&ctx.device, &ctx.queue, 1, 1, 4.0, 1.0, &|_| h);
        let t = Instant::now();
        poll(engine.run_lloyd_iterations(&ctx.device, &ctx.queue, lloyd_iters));
        let t_lloyd = ms(t);
        let overflows = cells
            .status
            .iter()
            .filter(|&&s| s == status::VERT_OVERFLOW || s == status::FACE_OVERFLOW)
            .count();

        println!("=== gpu_voronoi_bench: n={n} (target {target}, boundary {n_boundary}, h={h:.5}, bins={n_bins}) ===");
        println!("  seed gen (CPU Poisson):      {t_gen:9.2} ms");
        println!("  SeedGrid::build (CPU):       {t_grid:9.2} ms");
        println!("  upload_case (grid+canon+wb): {t_upload:9.2} ms");
        println!(
            "  regen kernel (submit+poll):  best {best:8.2} ms   mean {:8.2} ms  ({REPS} reps)",
            total / REPS as f64
        );
        println!("  read_cells (full readback):  {t_readback:9.2} ms");
        println!(
            "  resolve_flagged (f64+recip): {t_fallback:9.2} ms   (flagged {}, patched {}, recip rounds {})",
            report.flagged.len(),
            report.patched.len(),
            report.reciprocity_rounds
        );
        println!(
            "  lloyd chained ({lloyd_iters:3} iters):    {t_lloyd:9.2} ms   ({:.2} ms/iter)",
            t_lloyd / lloyd_iters as f64
        );
        let cons_rel = (merged - expected).abs() / expected;
        println!(
            "  conservation: sum areas {merged:.6} vs loop shoelace {expected:.6} (rel {cons_rel:.2e}); overflows {overflows}"
        );
        // Asserted: measured ~1.85e-8 at 100k, so 1e-4 is generous.
        assert!(
            cons_rel <= 1e-4,
            "conservation broken: sum of cell areas off by rel {cons_rel:.3e}"
        );
        assert!(
            report.unresolved.is_empty(),
            "f64 fallback left unresolved cells"
        );
    }
}

/// Σ cell areas (f64 accumulation over the f32 per-cell outputs) of the
/// engine's current resolved outputs.
fn engine_area_sum(
    ctx: &GpuContext,
    cache: &StagingBufferCache,
    engine: &GpuVoronoiEngine,
    n: usize,
) -> f64 {
    let cells = engine.read_cells(ctx, cache);
    let mut sum = 0.0f64;
    for i in 0..n {
        if cells.status[i] == status::SUCCESS {
            sum += cells.area[i] as f64;
        }
    }
    sum
}
