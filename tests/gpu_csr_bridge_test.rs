//! GPU CSR bridge parity ([`cfd2::solver::gpu::voronoi::GpuCsr`]) — Phase C
//! stage 2.
//!
//! The GPU builds the cell→face adjacency + the sorted-adjacency scalar CSR
//! (`col_indices` / `row_offsets` / `diagonal_indices` /
//! `cell_face_matrix_indices`) on device from the M1 clip slots. The gate checks
//! it reproduces the CPU `build_sorted_scalar_csr` (over `assemble_mesh`):
//!
//! - `row_offsets` / `col_indices` / `diagonal_indices` are ORDER-INDEPENDENT
//!   (sorted adjacency) and must match the CPU BIT-for-BIT.
//! - `cell_face_offsets` per-cell face counts match the CPU mesh.
//! - `cell_face_matrix_indices` is CONSISTENT: within each cell every interior
//!   face maps to exactly its neighbour's sorted column and every boundary face
//!   maps to the diagonal (the GPU cell-face ORDER differs from the CPU's, so
//!   this is a per-cell tally rather than an index-wise compare).
//!
//! ```sh
//! cargo test --features meshgen --test gpu_csr_bridge_test -- --nocapture
//! ```
#![cfg(feature = "meshgen")]

use std::collections::BTreeMap;

use cfd2::meshgen::meshless::{
    assemble_mesh, build_diagram, meshless_seed_points, EngineConfig, MeshlessInput,
};
use cfd2::meshgen::MeshgenTolerances;
use cfd2::solver::gpu::context::GpuContext;
use cfd2::solver::gpu::readback::StagingBufferCache;
use cfd2::solver::gpu::voronoi::{boundary_spec_f32, GpuCsr, GpuVoronoiEngine};
use cfd2::solver::mesh::{build_sorted_scalar_csr, Geometry, RectangularChannel};
use nalgebra::{Point2, Vector2};

fn gpu_context() -> Option<GpuContext> {
    match pollster::block_on(GpuContext::new(None, None)) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("SKIP: no GPU adapter available ({e})");
            None
        }
    }
}

fn run_case(name: &str, geo: &(impl Geometry + Sync), domain: Vector2<f64>, hmin: f64) {
    let Some(ctx) = gpu_context() else { return };
    let (seeds, kinds, spec) = meshless_seed_points(geo, hmin, hmin, 1.2, domain);
    let n = seeds.len();
    let tol = MeshgenTolerances::from_geometry(hmin, domain);
    let seeds_f32: Vec<f32> = seeds.iter().flat_map(|p| [p.x as f32, p.y as f32]).collect();
    let rounded: Vec<Point2<f64>> =
        (0..n).map(|i| Point2::new(seeds_f32[2 * i] as f64, seeds_f32[2 * i + 1] as f64)).collect();
    let spec_r = boundary_spec_f32(&spec);
    let flags = vec![0u32; n];

    let input = MeshlessInput {
        seeds: &rounded,
        kinds: &kinds,
        boundary: &spec_r,
        domain,
        tol: &tol,
        cfg: EngineConfig::default(),
    };
    let cpu = build_diagram(&input);
    let cpu_mesh = assemble_mesh(&input, &cpu);
    let cpu_csr = build_sorted_scalar_csr(&cpu_mesh).expect("cpu sorted csr");

    let mut engine = GpuVoronoiEngine::new(&ctx.device, n as u32, domain, &tol);
    engine.upload_case(&ctx.device, &ctx.queue, &seeds_f32, &flags, &kinds, &spec);
    let idx = engine.run_regen(&ctx.device, &ctx.queue);
    let _ = ctx.device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: None });
    let cache = StagingBufferCache::default();
    let _ = engine.resolve_flagged(&ctx, &cache);

    let csr = GpuCsr::new(&ctx.device);
    let gpu = csr.build_csr(&ctx, &cache, &engine);

    // --- Order-independent CSR: exact match. -------------------------------
    assert_eq!(gpu.row_offsets, cpu_csr.row_offsets, "[{name}] row_offsets mismatch");
    assert_eq!(gpu.col_indices, cpu_csr.col_indices, "[{name}] col_indices mismatch");
    assert_eq!(gpu.diagonal_indices, cpu_csr.diagonal_indices, "[{name}] diagonal_indices mismatch");

    // --- cell_face_offsets: per-cell face COUNT matches the CPU mesh. -------
    assert_eq!(
        gpu.cell_face_offsets.len(),
        cpu_mesh.cell_face_offsets.len(),
        "[{name}] cell_face_offsets length"
    );
    for i in 0..n {
        let gc = gpu.cell_face_offsets[i + 1] - gpu.cell_face_offsets[i];
        let cc = (cpu_mesh.cell_face_offsets[i + 1] - cpu_mesh.cell_face_offsets[i]) as u32;
        assert_eq!(gc, cc, "[{name}] cell {i}: face count {gc} != CPU {cc}");
    }

    // --- cell_face_matrix_indices consistency (per-cell tally). ------------
    // For each cell, tally the columns its faces map to: every neighbour column
    // (row minus diagonal) exactly once, the diagonal column `boundary` times.
    for i in 0..n {
        let rb = gpu.row_offsets[i] as usize;
        let re = gpu.row_offsets[i + 1] as usize;
        let diag_pos = gpu.diagonal_indices[i] as usize;
        let fb = gpu.cell_face_offsets[i] as usize;
        let fe = gpu.cell_face_offsets[i + 1] as usize;

        let mut hit: BTreeMap<u32, u32> = BTreeMap::new();
        for k in fb..fe {
            let mi = gpu.cell_face_matrix_indices[k] as usize;
            assert!(mi >= rb && mi < re, "[{name}] cell {i}: matrix index {mi} out of row [{rb},{re})");
            *hit.entry(gpu.col_indices[mi]).or_default() += 1;
        }
        // Expected: each non-diagonal column once, diagonal column `boundary` times.
        let n_faces = (fe - fb) as u32;
        let n_neighbors = (re - rb - 1) as u32; // row size minus the diagonal
        let n_boundary = n_faces - n_neighbors;
        for pos in rb..re {
            let col = gpu.col_indices[pos];
            let expected = if pos == diag_pos { n_boundary } else { 1 };
            let got = hit.get(&col).copied().unwrap_or(0);
            assert_eq!(
                got, expected,
                "[{name}] cell {i}: column {col} (diag={}) hit {got} times, expected {expected}",
                pos == diag_pos
            );
        }
    }

    println!(
        "[gpu-csr] {name}: n={n}, num_faces={}, nnz={}, cell_faces={} — sorted CSR bit-exact + \
         cfmi consistent vs CPU build_sorted_scalar_csr",
        gpu.num_faces,
        gpu.col_indices.len(),
        gpu.cell_faces.len(),
    );
}

#[test]
fn csr_bridge_parity_rect_channel_coarse() {
    run_case("rect_coarse", &RectangularChannel { length: 1.0, height: 1.0 }, Vector2::new(1.0, 1.0), 0.12);
}

#[test]
fn csr_bridge_parity_rect_channel_fine() {
    run_case("rect_fine", &RectangularChannel { length: 1.0, height: 1.0 }, Vector2::new(1.0, 1.0), 0.06);
}
