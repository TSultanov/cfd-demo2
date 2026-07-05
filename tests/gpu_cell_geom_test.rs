//! GPU cell-geometry parity ([`cfd2::solver::gpu::voronoi::CellGeometry`]) —
//! Phase C stage 2c. The pass turns the M1 engine's per-cell centroid (stored
//! seed-relative) + area into the cell-indexed `cell_centers` / `cell_vols` the
//! GPU solver binds, entirely on device. Two gates:
//!
//! 1. **Bit-exact vs the engine's own output:** `cell_centers[i]` ==
//!    `seed[i] + centroid_rel[i]` and `cell_vols[i]` == `area[i]` (the exact f32
//!    reconstruction `read_cells` performs). Isolates the kernel's indexing /
//!    seed-add from any f64 confound — it must be bitwise.
//! 2. **Tolerance vs the CPU f64 mesh:** the GPU (f32) cell volumes + centres
//!    match `assemble_mesh`'s (`cell_vol` / `cell_cx,cy`) within an f32·h band —
//!    the same f32-vs-f64 tolerance the meshless parity harness uses (the GPU is
//!    single-precision by construction, so this is a tolerance, not bitwise).
//!
//! ```sh
//! cargo test --features meshgen --test gpu_cell_geom_test -- --nocapture
//! ```
#![cfg(feature = "meshgen")]

use cfd2::meshgen::meshless::{
    assemble_mesh, build_diagram, meshless_seed_points, EngineConfig, MeshlessInput,
};
use cfd2::meshgen::MeshgenTolerances;
use cfd2::solver::gpu::context::GpuContext;
use cfd2::solver::gpu::readback::StagingBufferCache;
use cfd2::solver::gpu::voronoi::{boundary_spec_f32, CellGeometry, GpuVoronoiEngine};
use cfd2::solver::mesh::{Geometry, RectangularChannel};
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

    let mut engine = GpuVoronoiEngine::new(&ctx.device, n as u32, domain, &tol);
    engine.upload_case(&ctx.device, &ctx.queue, &seeds_f32, &flags, &kinds, &spec);
    let idx = engine.run_regen(&ctx.device, &ctx.queue);
    let _ = ctx.device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: None });
    let cache = StagingBufferCache::default();
    // Resolve f32-flagged cells so the centroid/area buffers hold the final
    // (f64-patched) per-cell geometry, matching the CPU oracle.
    let _ = engine.resolve_flagged(&ctx, &cache);
    let cells = engine.read_cells(&ctx, &cache);

    let geom = CellGeometry::new(&ctx.device);
    let out = geom.build(&ctx, &cache, &engine);
    assert_eq!(out.cell_centers.len(), n, "[{name}] cell_centers length");
    assert_eq!(out.cell_vols.len(), n, "[{name}] cell_vols length");

    // --- Gate 1: bit-exact vs the engine's own f32 reconstruction. -----------
    for i in 0..n {
        let seed = [seeds_f32[2 * i], seeds_f32[2 * i + 1]];
        let rel = cells.centroid_rel[i];
        let expect = [seed[0] + rel[0], seed[1] + rel[1]];
        assert_eq!(
            out.cell_centers[i][0].to_bits(),
            expect[0].to_bits(),
            "[{name}] cell {i} center.x not bit-exact (seed+centroid_rel)"
        );
        assert_eq!(
            out.cell_centers[i][1].to_bits(),
            expect[1].to_bits(),
            "[{name}] cell {i} center.y not bit-exact (seed+centroid_rel)"
        );
        assert_eq!(
            out.cell_vols[i].to_bits(),
            cells.area[i].to_bits(),
            "[{name}] cell {i} vol not bit-exact (area)"
        );
    }

    // --- Gate 2: tolerance vs the CPU f64 mesh geometry. ---------------------
    // f32 vs f64: an absolute band scaled by h (positions are O(1), areas O(h²)).
    let pos_tol = 1e-4 * hmin.max(1.0) as f32;
    let vol_tol = 1e-4 * (hmin * hmin) as f32;
    let mut max_pos_err = 0.0f32;
    let mut max_vol_err = 0.0f32;
    for i in 0..n {
        let dx = (out.cell_centers[i][0] - cpu_mesh.cell_cx[i] as f32).abs();
        let dy = (out.cell_centers[i][1] - cpu_mesh.cell_cy[i] as f32).abs();
        let dv = (out.cell_vols[i] - cpu_mesh.cell_vol[i] as f32).abs();
        max_pos_err = max_pos_err.max(dx).max(dy);
        max_vol_err = max_vol_err.max(dv);
        assert!(
            dx <= pos_tol && dy <= pos_tol,
            "[{name}] cell {i} center ({},{}) vs CPU ({},{}) exceeds {pos_tol}",
            out.cell_centers[i][0], out.cell_centers[i][1], cpu_mesh.cell_cx[i], cpu_mesh.cell_cy[i]
        );
        assert!(
            dv <= vol_tol,
            "[{name}] cell {i} vol {} vs CPU {} exceeds {vol_tol}",
            out.cell_vols[i], cpu_mesh.cell_vol[i]
        );
    }

    println!(
        "[gpu-cell-geom] {name}: n={n} — centers+vols bit-exact vs engine output, \
         within tol vs CPU f64 mesh (max pos err {max_pos_err:.2e} <= {pos_tol:.2e}, \
         max vol err {max_vol_err:.2e} <= {vol_tol:.2e})"
    );
}

#[test]
fn cell_geom_parity_rect_channel_coarse() {
    run_case("rect_coarse", &RectangularChannel { length: 1.0, height: 1.0 }, Vector2::new(1.0, 1.0), 0.12);
}

#[test]
fn cell_geom_parity_rect_channel_fine() {
    run_case("rect_fine", &RectangularChannel { length: 1.0, height: 1.0 }, Vector2::new(1.0, 1.0), 0.06);
}
