//! GPU `derive_faces` count→scan parity (design-gpu §6.2 first half) — M5 stage 3.
//!
//! The M1 engine emits cell-major, padded per-cell face slots. The GPU-resident
//! regen turns those into face-major arrays via **count → scan → emit → gather**.
//! This gate pins the landed half — the `count_owned_faces` kernel + the new GPU
//! `scan` primitive producing per-cell face offsets + the total `num_faces`,
//! entirely on the GPU — bit-exact against a CPU reference computed from the
//! *same* cell-major outputs. Two seed sizes cover the single-block and the
//! multi-block (>1024 cells) scan paths.
//!
//! The `emit`/`gather` passes (face-major geometry + `cell_faces` CSR) and the
//! §6.3 CSR bridge are documented-deferred (see `voronoi/derive.rs`): the CPU
//! face-major geometry is a post-vertex-merge product of `assemble_mesh`, not a
//! direct projection of these cell-major slots.
//!
//! Skips cleanly when no GPU adapter is present.
#![cfg(feature = "meshgen")]

use cfd2::meshgen::MeshgenTolerances;
use cfd2::solver::gpu::context::GpuContext;
use cfd2::solver::gpu::readback::StagingBufferCache;
use cfd2::solver::gpu::voronoi::{
    status, DeriveFaces, GpuVoronoiCells, GpuVoronoiEngine, K_FACE_MAX, NBR_NONE,
};
use nalgebra::{Point2, Vector2};
use rand::{Rng, SeedableRng};

const DOMAIN: Vector2<f64> = Vector2::new(2.0, 1.0);

fn gpu_context() -> Option<GpuContext> {
    match pollster::block_on(GpuContext::new(None, None)) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("SKIP: no GPU adapter available ({e})");
            None
        }
    }
}

fn jittered_seeds(nx: usize, ny: usize, amp: f64, rng_seed: u64) -> Vec<Point2<f64>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(rng_seed);
    let (sx, sy) = (DOMAIN.x / nx as f64, DOMAIN.y / ny as f64);
    let mut pts = Vec::with_capacity(nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            let jx = (rng.gen::<f64>() - 0.5) * amp * sx;
            let jy = (rng.gen::<f64>() - 0.5) * amp * sy;
            pts.push(Point2::new((i as f64 + 0.5) * sx + jx, (j as f64 + 0.5) * sy + jy));
        }
    }
    pts
}

/// CPU reference: per-cell owned-face count under the SAME canonical
/// `owner = min(i, j)` rule the GPU kernel applies, on the same cell-major data.
fn cpu_owned_counts(cells: &GpuVoronoiCells) -> Vec<u32> {
    let n = cells.n;
    let k = K_FACE_MAX;
    let mut counts = vec![0u32; n];
    for i in 0..n {
        if cells.status[i] != status::SUCCESS {
            continue;
        }
        let nf = cells.nfaces[i] as usize;
        let mut c = 0u32;
        for e in 0..nf {
            let nbr = cells.nbr_ids[i * k + e];
            if nbr == NBR_NONE || (i as u32) < nbr {
                c += 1;
            }
        }
        counts[i] = c;
    }
    counts
}

fn cpu_exclusive_scan(counts: &[u32]) -> (Vec<u32>, u32) {
    let mut out = Vec::with_capacity(counts.len());
    let mut run = 0u32;
    for &v in counts {
        out.push(run);
        run += v;
    }
    (out, run)
}

fn run_case(nx: usize, ny: usize, seed: u64) {
    let Some(ctx) = gpu_context() else { return };
    let seeds = jittered_seeds(nx, ny, 0.5, seed);
    let n = seeds.len();
    let spacing = DOMAIN.x / nx as f64;
    let tol = MeshgenTolerances::from_geometry(0.5 * spacing, DOMAIN);

    let seeds_f32: Vec<f32> = seeds.iter().flat_map(|p| [p.x as f32, p.y as f32]).collect();
    let flags = vec![0u32; n];

    let mut engine = GpuVoronoiEngine::new(&ctx.device, n as u32, DOMAIN, &tol);
    engine.upload_seeds(&ctx.device, &ctx.queue, &seeds_f32, &flags);
    let idx = engine.run_regen(&ctx.device, &ctx.queue);
    let _ = ctx.device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: None });

    let cache = StagingBufferCache::default();
    let cells = engine.read_cells(&ctx, &cache);

    // CPU reference on the identical cell-major outputs.
    let want_counts = cpu_owned_counts(&cells);
    let (want_offsets, want_total) = cpu_exclusive_scan(&want_counts);

    // GPU count → scan.
    let derive = DeriveFaces::new(&ctx.device);
    let got = derive.derive_offsets(&ctx, &cache, &engine);

    assert_eq!(got.owned_counts, want_counts, "n={n}: per-cell owned counts mismatch");
    assert_eq!(got.offsets, want_offsets, "n={n}: per-cell offsets (exclusive scan) mismatch");
    assert_eq!(got.total, want_total, "n={n}: total num_faces mismatch");

    // Sanity: the total equals interior (i<j) + boundary faces of the raw
    // diagram — a positive, plausible count near the classic Euler estimate
    // (~3 faces/cell for a 2D Voronoi with a boundary).
    assert!(got.total >= n as u32, "n={n}: implausibly few faces ({})", got.total);
    println!(
        "[derive][n={n}] owned/offsets/total bit-exact; num_faces={} (~{:.2}/cell)",
        got.total,
        got.total as f64 / n as f64
    );
}

#[test]
fn gpu_derive_faces_count_scan_single_block() {
    // ~1000 cells < 1024 ⇒ single scan block.
    run_case(40, 25, 0xD1CE_0001);
}

#[test]
fn gpu_derive_faces_count_scan_multi_block() {
    // ~2400 cells > 1024 ⇒ exercises the block-sums scan level on real data.
    run_case(60, 40, 0xD1CE_0002);
}
