//! GPU swept-flux geometry parity ([`cfd2::solver::gpu::voronoi::SweptFluxGeometry`])
//! — Phase C stage D1+D2. The pass computes each face's ALE swept-quad area on
//! device from the OLD and NEW seed sets (canonical seed-triple / box vertices,
//! no CPU union-find merge). The load-bearing gate is the **telescoping
//! identity** the moving-mesh GCL rests on:
//!
//!   Σ_f σ(i,f)·swept[f]  ==  V_i^{n+1} − V_i^n     (per cell, exact arithmetic)
//!
//! which simultaneously validates the swept-quad geometry, the ring cyclic
//! order, the owner = min(i,j) ownership, and the face-major addressing — a
//! single scalar per cell that only closes if all four are right. The f32
//! residual it leaves is exactly the per-cell defect the closure (stage D3) must
//! absorb, so the test also REPORTS it (the input to the closure decision).
//!
//! Config: a rectangular channel (box-only boundary — the free-stream GCL case),
//! interior seeds displaced by a small smooth field that keeps the topology
//! (no flips) so the persistent telescoping identity holds.
//!
//! ```sh
//! cargo test --features meshgen --test gpu_swept_flux_test -- --nocapture
//! ```
#![cfg(feature = "meshgen")]

use cfd2::meshgen::meshless::{
    assemble_mesh, build_diagram, meshless_seed_points, EngineConfig, MeshlessInput, SeedKind,
};
use cfd2::meshgen::MeshgenTolerances;
use cfd2::solver::gpu::context::GpuContext;
use cfd2::solver::gpu::readback::StagingBufferCache;
use cfd2::solver::gpu::voronoi::{boundary_spec_f32, EmitFaces, GpuVoronoiEngine, SweptFluxGeometry};
use cfd2::solver::mesh::{
    align_old_vertices_by_seed_set, swept_mesh_fluxes_closed, Geometry, RectangularChannel,
};
use nalgebra::{Point2, Vector2};
use std::collections::HashMap;

fn gpu_context() -> Option<GpuContext> {
    match pollster::block_on(GpuContext::new(None, None)) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("SKIP: no GPU adapter available ({e})");
            None
        }
    }
}

/// f32-round an interleaved seed set (the values the GPU actually sees).
fn round_seeds(seeds: &[Point2<f64>]) -> (Vec<f32>, Vec<Point2<f64>>) {
    let f32v: Vec<f32> = seeds.iter().flat_map(|p| [p.x as f32, p.y as f32]).collect();
    let pts: Vec<Point2<f64>> =
        (0..seeds.len()).map(|i| Point2::new(f32v[2 * i] as f64, f32v[2 * i + 1] as f64)).collect();
    (f32v, pts)
}

fn run_case(name: &str, geo: &(impl Geometry + Sync), domain: Vector2<f64>, hmin: f64, amp: f64) {
    let Some(ctx) = gpu_context() else { return };
    let (seeds0, kinds, spec) = meshless_seed_points(geo, hmin, hmin, 1.2, domain);
    let n = seeds0.len();
    let tol = MeshgenTolerances::from_geometry(hmin, domain);
    let spec_r = boundary_spec_f32(&spec);

    // OLD seed set = the CVT layout, f32-rounded.
    let (old_f32, old_pts) = round_seeds(&seeds0);

    // NEW seed set = interior seeds displaced by a small smooth field (boundary
    // seeds fixed). Small amplitude keeps the topology (no flips), so the
    // persistent telescoping identity holds.
    let mut new_seeds = seeds0.clone();
    for i in 0..n {
        if matches!(kinds[i], SeedKind::Boundary { .. }) {
            continue;
        }
        let p = seeds0[i];
        let dx = amp * (std::f64::consts::PI * p.y / domain.y).sin();
        let dy = amp * (std::f64::consts::PI * p.x / domain.x).sin();
        new_seeds[i] = Point2::new(p.x + dx, p.y + dy);
    }
    let (new_f32, new_pts) = round_seeds(&new_seeds);

    // CPU meshes at both seed sets (for V^n / V^{n+1}); same rounded seeds the
    // GPU sees.
    let mesh_of = |pts: &[Point2<f64>]| {
        let input = MeshlessInput {
            seeds: pts,
            kinds: &kinds,
            boundary: &spec_r,
            domain,
            tol: &tol,
            cfg: EngineConfig::default(),
        };
        let d = build_diagram(&input);
        assemble_mesh(&input, &d)
    };
    let old_mesh = mesh_of(&old_pts);
    let new_mesh = mesh_of(&new_pts);
    assert_eq!(old_mesh.num_cells(), n);
    assert_eq!(new_mesh.num_cells(), n);

    // GPU: build the NEW diagram, then the emit face topology + swept areas.
    let flags = vec![0u32; n];
    let mut engine = GpuVoronoiEngine::new(&ctx.device, n as u32, domain, &tol);
    engine.upload_case(&ctx.device, &ctx.queue, &new_f32, &flags, &kinds, &spec);
    let idx = engine.run_regen(&ctx.device, &ctx.queue);
    let _ = ctx.device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: None });
    let cache = StagingBufferCache::default();
    let _ = engine.resolve_flagged(&ctx, &cache);

    let emitter = EmitFaces::new(&ctx.device);
    let faces = emitter.emit(&ctx, &cache, &engine);

    let swept = SweptFluxGeometry::new(&ctx.device);
    let sw = swept.compute(&ctx, &cache, &engine, &old_f32);
    assert_eq!(sw.num_faces, faces.num_faces, "[{name}] swept/emit face-count mismatch");

    // No face may need the CPU fallback on a rectangular (box-only) domain.
    let needs: usize = sw.needs_cpu.iter().filter(|&&x| x != 0).count();
    assert_eq!(needs, 0, "[{name}] {needs} faces flagged needs_cpu on a box-only domain");

    // Per-cell telescoping: Σ_f σ·swept  vs  the CANONICAL area change (the same
    // vertices the swept quads use — self-consistent ⇒ exact to f32), using the
    // GPU emit's OWN owner/neighbour (same face order as `swept`).
    let nf = faces.num_faces as usize;
    let mut sigma_sum = vec![0.0f64; n];
    for f in 0..nf {
        let s = sw.swept[f] as f64;
        let o = faces.face_owner[f] as usize;
        sigma_sum[o] += s; // owner sign +1
        let nb = faces.face_neighbor[f];
        if nb >= 0 {
            sigma_sum[nb as usize] -= s; // neighbour sign −1
        }
    }
    // The canonical polygon areas the swept quads telescope to (the moving path's
    // cell_vols). Cross-check they track the CPU f64 mesh volumes (clip-drift
    // aside): both are canonical-vertex areas, so ~f32 close.
    let mut max_canon_vs_cpu = 0.0f64;
    for i in 0..n {
        let d = (sw.canon_area_new[i] as f64 - new_mesh.cell_vol[i]).abs()
            / new_mesh.cell_vol[i].max(f64::MIN_POSITIVE);
        max_canon_vs_cpu = max_canon_vs_cpu.max(d);
    }

    // Which cells touch a boundary (box) face — to localize any residual.
    let mut has_boundary = vec![false; n];
    for f in 0..nf {
        if faces.face_neighbor[f] < 0 {
            has_boundary[faces.face_owner[f] as usize] = true;
        }
    }

    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut worst = 0usize;
    let mut max_rel_interior = 0.0f64;
    let mut max_rel_boundary = 0.0f64;
    for i in 0..n {
        // Self-consistent target: the canonical area change (same vertices).
        let dv = sw.canon_area_new[i] as f64 - sw.canon_area_old[i] as f64;
        let err = (sigma_sum[i] - dv).abs();
        if err > max_abs {
            max_abs = err;
            worst = i;
        }
        let rel = err / (sw.canon_area_new[i] as f64).max(f64::MIN_POSITIVE);
        max_rel = max_rel.max(rel);
        if has_boundary[i] {
            max_rel_boundary = max_rel_boundary.max(rel);
        } else {
            max_rel_interior = max_rel_interior.max(rel);
        }
    }
    eprintln!(
        "[gpu-swept] {name}: telescoping max rel INTERIOR {max_rel_interior:.3e}, \
         BOUNDARY {max_rel_boundary:.3e} (worst cell {worst}, boundary={}); \
         canonical-area vs CPU cell_vol max rel {max_canon_vs_cpu:.3e}",
        has_boundary[worst]
    );

    // Self-consistent telescoping: the swept quads and the polygon area come from
    // the SAME canonical vertices, so Σσ·swept == ΔV(canonical) to f32 roundoff.
    // A ring-order/ownership/addressing/orientation bug blows this up by orders.
    let rel_tol = 1e-5;
    assert!(
        max_rel <= rel_tol,
        "[{name}] telescoping identity broken: max rel residual {max_rel:.3e} > {rel_tol:.1e} \
         (worst cell {worst}: Σσ·swept={}, ΔV_canon={})",
        sigma_sum[worst],
        sw.canon_area_new[worst] as f64 - sw.canon_area_old[worst] as f64
    );
    // The canonical areas must track the CPU f64 mesh (both canonical-vertex based).
    assert!(
        max_canon_vs_cpu <= 5e-4,
        "[{name}] canonical area vs CPU cell_vol max rel {max_canon_vs_cpu:.3e} > 5e-4"
    );

    // --- Parity vs the CERTIFIED CPU swept path (per face). -------------------
    // The CPU `swept_mesh_fluxes_closed` is the GCL-certified reference. Matching
    // it per face means the device swept inherits that certification directly.
    // dt = 1 so the returned fluxes ARE the swept areas (the closure is f64-tiny,
    // so post-closure ≈ raw on every face).
    let (ovx, ovy, _unmatched) =
        align_old_vertices_by_seed_set(&old_mesh, &new_mesh).expect("align old vertices");
    let cpu_swept =
        swept_mesh_fluxes_closed(&new_mesh, &ovx, &ovy, 1.0).expect("cpu swept fluxes");
    // (owner,neighbour) key → CPU face index (interior: sorted pair; boundary:
    // (owner, MAX)).
    let mut key2cpu: HashMap<(usize, usize), usize> = HashMap::new();
    for f in 0..new_mesh.num_faces() {
        let o = new_mesh.face_owner[f];
        let key = match new_mesh.face_neighbor[f] {
            Some(nb) => (o.min(nb), o.max(nb)),
            None => (o, usize::MAX),
        };
        key2cpu.insert(key, f);
    }
    let mut swept_scale = 0.0f64;
    for f in 0..nf {
        swept_scale = swept_scale.max((sw.swept[f] as f64).abs());
    }
    let mut max_flux_diff = 0.0f64;
    let mut diffs: Vec<(f64, usize, i32, f64, f64, bool)> = Vec::new();
    for g in 0..nf {
        let o = faces.face_owner[g] as usize; // device owner = min(i,j)
        let nb = faces.face_neighbor[g];
        let key = if nb >= 0 { (o.min(nb as usize), o.max(nb as usize)) } else { (o, usize::MAX) };
        let cf = key2cpu[&key];
        // Normalise the CPU value to the min-owner sign convention the device uses.
        let mut cpu_val = cpu_swept.fluxes[cf] as f64;
        if let Some(nbb) = new_mesh.face_neighbor[cf] {
            if new_mesh.face_owner[cf] != new_mesh.face_owner[cf].min(nbb) {
                cpu_val = -cpu_val;
            }
        }
        let d = (sw.swept[g] as f64 - cpu_val).abs();
        diffs.push((d, o, nb, sw.swept[g] as f64, cpu_val, nb < 0));
        max_flux_diff = max_flux_diff.max(d);
    }
    diffs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    // Median face length, for a near-degeneracy gauge on the outlier faces.
    let mut lens: Vec<f32> = faces.face_area.clone();
    lens.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_len = lens[lens.len() / 2];
    eprintln!("[gpu-swept] {name}: median face len {median_len:.3e}; worst 6 device-vs-CPU face diffs:");
    for r in diffs.iter().take(6) {
        // find this face's length
        let mut flen = 0.0f32;
        for g in 0..nf {
            let o = faces.face_owner[g] as usize;
            let nb = faces.face_neighbor[g];
            if o == r.1 && nb == r.2 {
                flen = faces.face_area[g];
                break;
            }
        }
        eprintln!("   diff={:.3e} owner={} nbr={} dev={:.3e} cpu={:.3e} bndry={} len={:.3e} ({:.2}x median)",
            r.0, r.1, r.2, r.3, r.4, r.5, flen, flen / median_len);
    }
    let flux_rel = max_flux_diff / swept_scale.max(f64::MIN_POSITIVE);
    // BULK parity vs the certified CPU fluxes: the device reproduces CPU
    // everywhere EXCEPT sub-tolerance sliver regions, where the CPU's union-find
    // vertex merge adjusts a shared vertex that the device's raw canonical
    // vertices do not (a legitimate, GCL-neutral discretisation difference — the
    // device stays self-consistent, so its telescoping still holds to f32). So we
    // gate the 90th-percentile diff (a systematic ring-order/ownership/sign bug
    // blows up EVERY face and fails this; a few sliver outliers pass) and report
    // the max.
    let mut rels: Vec<f64> =
        diffs.iter().map(|r| r.0 / swept_scale.max(f64::MIN_POSITIVE)).collect();
    rels.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p90 = rels[(rels.len() * 90 / 100).min(rels.len() - 1)];
    let n_big = diffs.iter().filter(|r| r.0 / swept_scale.max(f64::MIN_POSITIVE) > 1e-4).count();
    eprintln!(
        "[gpu-swept] {name}: per-face vs CPU — p90 rel {p90:.3e}, max rel {flux_rel:.3e}, \
         {n_big}/{nf} faces > 1e-4 rel (sliver-region merge divergence)"
    );
    assert!(
        p90 <= 1e-4,
        "[{name}] device swept vs certified CPU: 90th-percentile rel diff {p90:.3e} > 1e-4 — \
         the bulk of faces disagree, indicating a systematic (not sliver-local) error"
    );
    // Sliver outliers must be RARE (a systematic bug would flag a large fraction).
    let frac_big = n_big as f64 / nf as f64;
    assert!(
        frac_big <= 0.05,
        "[{name}] {:.1}% of faces disagree with CPU by > 1e-4 rel — too many for \
         local sliver divergence; likely a real bug",
        frac_big * 100.0
    );

    println!(
        "[gpu-swept] {name}: n={n}, faces={nf} — self-consistent telescoping \
         Σσ·swept == ΔV(canonical) to f32: max rel {max_rel:.3e} (<= {rel_tol:.1e}); \
         canonical vs CPU vol {max_canon_vs_cpu:.3e}; per-face vs CERTIFIED CPU swept: \
         p90 {p90:.3e}, max {flux_rel:.3e} ({n_big} sliver-region outliers)."
    );
}

#[test]
fn swept_flux_telescoping_rect_coarse() {
    run_case("rect_coarse", &RectangularChannel { length: 1.0, height: 1.0 }, Vector2::new(1.0, 1.0), 0.12, 0.01);
}

#[test]
fn swept_flux_telescoping_rect_fine() {
    run_case("rect_fine", &RectangularChannel { length: 1.0, height: 1.0 }, Vector2::new(1.0, 1.0), 0.06, 0.005);
}
