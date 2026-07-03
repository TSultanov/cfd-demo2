//! M1 stage-3 full-geometry gates for the GPU meshless Voronoi engine:
//! boundary segments + seed kinds + graded sets + end-to-end mesh.
//!
//! For each of the four GUI geometries (rect channel / obstacle / backstep /
//! nozzle, loop-derived boundary seeding via `meshless_seed_points`) plus a
//! graded nozzle set (review F3):
//!
//! 1. Stage-1/2 parity protocol against the CPU f64 `build_diagram` oracle
//!    on the SAME f32-rounded seeds AND f32-rounded boundary spec (review
//!    F4): ZERO unflagged topology disagreements post `eps_face` (bisector
//!    ids AND boundary/box tags), flag-rate budget 2e-3 on the BULK
//!    interior + a defensive cap on the wall strip (see the in-code note:
//!    M0's same-segment guard pairs are knife-edge twins by construction),
//!    zero overflow statuses, run-to-run byte stability.
//! 2. `resolve_flagged` → full post-fallback parity + independent
//!    reciprocity check.
//! 3. End-to-end mesh: GPU outputs → `read_diagram` → `assemble_mesh` →
//!    `validate_mesh` battery green; mesh equivalent to the pure-CPU-engine
//!    mesh (interior neighbor-pair sets post `eps_face`, per-BoundaryType
//!    wall length sums rel < 1e-6, cell volumes/centroids within f32
//!    tolerances).
//! 4. Graded nozzle (h ratio 4×): visited-bin distribution measured (F3
//!    instrument) and reported; gates identical.
//!
//! Run with:
//!
//! ```sh
//! cargo test --features meshgen --test gpu_voronoi_geometry_test -- --nocapture
//! ```

#![cfg(feature = "meshgen")]

use std::collections::{BTreeMap, BTreeSet, HashMap};

use cfd2::meshgen::meshless::{
    assemble_mesh, build_diagram, CellStatus, EngineConfig, MeshlessDiagram, MeshlessInput,
    PlaneTag, SeedKind, MAX_CLIP_VERTS,
};
use cfd2::meshgen::MeshgenTolerances;
use cfd2::solver::gpu::context::GpuContext;
use cfd2::solver::gpu::readback::StagingBufferCache;
use cfd2::solver::gpu::voronoi::{
    boundary_spec_f32, status, GpuVoronoiCells, GpuVoronoiEngine, BC_SEG_FLAG, K_FACE_MAX,
    NBR_NONE,
};
use cfd2::solver::mesh::{
    BackwardsStep, ChannelWithObstacle, Geometry, Mesh, Nozzle, RectangularChannel,
};
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

/// Coalescing table over the f32-rounded seeds (same rule as engine/oracle).
fn canon_map(pts: &[Point2<f64>], tol: &MeshgenTolerances) -> Vec<u32> {
    let mut first: HashMap<(i64, i64), u32> = HashMap::with_capacity(pts.len());
    let mut canon = vec![0u32; pts.len()];
    for (i, p) in pts.iter().enumerate() {
        let key = tol.quantize_point(p.x, p.y);
        canon[i] = *first.entry(key).or_insert(i as u32);
    }
    canon
}

/// Local scale of cell `i`: nearest bisector-neighbor distance from the
/// (unfiltered) CPU ring; `hmin` fallback for rings without bisectors.
fn local_h(cpu: &MeshlessDiagram, rounded: &[Point2<f64>], i: usize, hmin: f64) -> f64 {
    let p = rounded[i];
    let len = cpu.ring_len[i] as usize;
    let mut h_i = f64::INFINITY;
    for e in 0..len {
        if let PlaneTag::Bisector(j) = cpu.ring_plane[i * MAX_CLIP_VERTS + e] {
            h_i = h_i.min((rounded[j as usize] - p).norm());
        }
    }
    if h_i.is_finite() {
        h_i
    } else {
        hmin
    }
}

/// Boundary/box tag in the GPU `b_face_bc` encoding (the comparison space).
fn bc_code(tag: PlaneTag) -> u32 {
    match tag {
        PlaneTag::Box(s) => s as u32,
        PlaneTag::Boundary(seg) => BC_SEG_FLAG | seg,
        PlaneTag::Bisector(_) => unreachable!("interior tags are not bc codes"),
    }
}

/// CPU eps_face-filtered topology of cell `i`: (canonical bisector ids,
/// boundary/box bc codes).
fn cpu_sets(
    cpu: &MeshlessDiagram,
    canon: &[u32],
    i: usize,
    eps_face: f64,
) -> (BTreeSet<u32>, BTreeSet<u32>) {
    let mut nbrs = BTreeSet::new();
    let mut bcs = BTreeSet::new();
    let len = cpu.ring_len[i] as usize;
    for e in 0..len {
        let v0 = cpu.ring_xy[i * MAX_CLIP_VERTS + e];
        let v1 = cpu.ring_xy[i * MAX_CLIP_VERTS + (e + 1) % len];
        let elen = ((v1[0] - v0[0]).powi(2) + (v1[1] - v0[1]).powi(2)).sqrt();
        if elen <= eps_face {
            continue;
        }
        match cpu.ring_plane[i * MAX_CLIP_VERTS + e] {
            PlaneTag::Bisector(j) => {
                nbrs.insert(canon[j as usize]);
            }
            tag => {
                bcs.insert(bc_code(tag));
            }
        }
    }
    (nbrs, bcs)
}

/// GPU eps_face-filtered topology of cell `i`.
fn gpu_sets(gpu: &GpuVoronoiCells, i: usize, eps_face: f64) -> (BTreeSet<u32>, BTreeSet<u32>) {
    let mut nbrs = BTreeSet::new();
    let mut bcs = BTreeSet::new();
    for s in 0..(gpu.nfaces[i] as usize).min(K_FACE_MAX) {
        let slot = i * K_FACE_MAX + s;
        if (gpu.face_geom[slot][2] as f64) <= eps_face {
            continue;
        }
        let nbr = gpu.nbr_ids[slot];
        if nbr == NBR_NONE {
            bcs.insert(gpu.face_bc[slot]);
        } else {
            nbrs.insert(nbr);
        }
    }
    (nbrs, bcs)
}

// ---------------------------------------------------------------------------
// validate_mesh battery (trimmed transplant of tests/meshgen_validation.rs —
// test crates cannot share modules; the checks are identical).
// ---------------------------------------------------------------------------

fn fluid_area_estimate(geo: &(impl Geometry + Sync), domain: Vector2<f64>, n: usize) -> f64 {
    let mut hits = 0usize;
    for j in 0..n {
        for i in 0..n {
            let p = Point2::new(
                (i as f64 + 0.5) / n as f64 * domain.x,
                (j as f64 + 0.5) / n as f64 * domain.y,
            );
            if geo.is_inside(&p) {
                hits += 1;
            }
        }
    }
    hits as f64 / (n * n) as f64 * domain.x * domain.y
}

fn validate_mesh(mesh: &Mesh, expected_area: f64, min_cell_size: f64) -> Vec<String> {
    let mut issues = Vec::new();
    let nc = mesh.num_cells();
    let nf = mesh.num_faces();
    let nv = mesh.num_vertices();
    if nc == 0 {
        issues.push("mesh has 0 cells".to_string());
        return issues;
    }

    let mut nan_count = 0usize;
    for arr in [
        &mesh.vx, &mesh.vy, &mesh.face_cx, &mesh.face_cy, &mesh.face_nx, &mesh.face_ny,
        &mesh.face_area, &mesh.cell_cx, &mesh.cell_cy, &mesh.cell_vol,
    ] {
        nan_count += arr.iter().filter(|v| !v.is_finite()).count();
    }
    if nan_count > 0 {
        issues.push(format!("{nan_count} non-finite geometric values"));
    }

    let mut nonpos_vol = 0usize;
    let mut tiny_vol = 0usize;
    let nominal_vol = min_cell_size * min_cell_size;
    for &v in &mesh.cell_vol {
        if !(v > 0.0) {
            nonpos_vol += 1;
        } else if v < 1e-3 * nominal_vol {
            tiny_vol += 1;
        }
    }
    if nonpos_vol > 0 {
        issues.push(format!("{nonpos_vol}/{nc} cells with volume <= 0"));
    }
    if tiny_vol > 0 {
        issues.push(format!("{tiny_vol}/{nc} cells with volume < 1e-3*h^2"));
    }

    let mut bad_idx = 0usize;
    let mut zero_area = 0usize;
    let mut bad_normal = 0usize;
    let mut self_neighbor = 0usize;
    let mut inward_normal = 0usize;
    for f in 0..nf {
        let o = mesh.face_owner[f];
        if o >= nc || mesh.face_v1[f] >= nv || mesh.face_v2[f] >= nv {
            bad_idx += 1;
            continue;
        }
        if let Some(nb) = mesh.face_neighbor[f] {
            if nb >= nc {
                bad_idx += 1;
                continue;
            }
            if nb == o {
                self_neighbor += 1;
            }
        }
        if !(mesh.face_area[f] > 1e-12 * min_cell_size) {
            zero_area += 1;
        }
        let nrm = (mesh.face_nx[f] * mesh.face_nx[f] + mesh.face_ny[f] * mesh.face_ny[f]).sqrt();
        if !((nrm - 1.0).abs() < 1e-6) {
            bad_normal += 1;
        } else {
            let target = if let Some(nb) = mesh.face_neighbor[f] {
                Point2::new(mesh.cell_cx[nb], mesh.cell_cy[nb])
            } else {
                Point2::new(mesh.face_cx[f], mesh.face_cy[f])
            };
            let c = Point2::new(mesh.cell_cx[o], mesh.cell_cy[o]);
            let d = target - c;
            if d.norm() > 1e-12 && d.x * mesh.face_nx[f] + d.y * mesh.face_ny[f] < 0.0 {
                inward_normal += 1;
            }
        }
    }
    if bad_idx > 0 {
        issues.push(format!("{bad_idx}/{nf} faces with out-of-range indices"));
    }
    if zero_area > 0 {
        issues.push(format!("{zero_area}/{nf} faces with ~zero area"));
    }
    if bad_normal > 0 {
        issues.push(format!("{bad_normal}/{nf} faces with non-unit normal"));
    }
    if self_neighbor > 0 {
        issues.push(format!("{self_neighbor}/{nf} faces with owner==neighbor"));
    }
    if inward_normal > 0 {
        issues.push(format!(
            "{inward_normal}/{nf} faces with normal not pointing owner->neighbor"
        ));
    }

    let mut open_cells = 0usize;
    let mut worst_closure = 0.0f64;
    let mut few_faces = 0usize;
    for c in 0..nc {
        let start = mesh.cell_face_offsets[c];
        let end = mesh.cell_face_offsets[c + 1];
        if end - start < 3 {
            few_faces += 1;
        }
        let mut sx = 0.0;
        let mut sy = 0.0;
        let mut perim = 0.0;
        for &f in &mesh.cell_faces[start..end] {
            let sign = if mesh.face_owner[f] == c { 1.0 } else { -1.0 };
            sx += sign * mesh.face_area[f] * mesh.face_nx[f];
            sy += sign * mesh.face_area[f] * mesh.face_ny[f];
            perim += mesh.face_area[f];
        }
        if perim > 0.0 {
            let rel = (sx * sx + sy * sy).sqrt() / perim;
            worst_closure = worst_closure.max(rel);
            if rel > 1e-6 {
                open_cells += 1;
            }
        }
    }
    if few_faces > 0 {
        issues.push(format!("{few_faces}/{nc} cells with < 3 faces"));
    }
    if open_cells > 0 {
        issues.push(format!(
            "{open_cells}/{nc} cells NOT closed (worst rel closure {worst_closure:.3e})"
        ));
    }

    let total: f64 = mesh.cell_vol.iter().filter(|v| v.is_finite()).sum();
    let rel_err = (total - expected_area).abs() / expected_area;
    if rel_err > 0.02 {
        issues.push(format!(
            "total volume {total:.4} vs expected {expected_area:.4} (rel err {:.2}%)",
            rel_err * 100.0
        ));
    }

    let untagged = (0..nf)
        .filter(|&f| mesh.face_neighbor[f].is_none() && mesh.face_boundary[f].is_none())
        .count();
    if untagged > 0 {
        issues.push(format!("{untagged}/{nf} boundary faces untagged"));
    }

    issues
}

// ---------------------------------------------------------------------------
// Mesh equivalence (GPU-diagram mesh vs pure-CPU-engine mesh)
// ---------------------------------------------------------------------------

/// Interior neighbor-pair set of a mesh, filtered at the reciprocity
/// sliver threshold (`face_area > 1e-6 · |p_j − p_i|`, the REAL_FACE_REL
/// contract) so sub-eps knife-edge faces cannot fail the comparison.
fn interior_pairs(mesh: &Mesh, rounded: &[Point2<f64>]) -> BTreeSet<(usize, usize)> {
    let mut pairs = BTreeSet::new();
    for f in 0..mesh.num_faces() {
        if let Some(nb) = mesh.face_neighbor[f] {
            let o = mesh.face_owner[f];
            let d = (rounded[nb] - rounded[o]).norm();
            if mesh.face_area[f] > 1e-6 * d {
                pairs.insert((o.min(nb), o.max(nb)));
            }
        }
    }
    pairs
}

/// Per-BoundaryType boundary-face length sums.
fn wall_length_sums(mesh: &Mesh) -> BTreeMap<u8, f64> {
    let mut sums: BTreeMap<u8, f64> = BTreeMap::new();
    for f in 0..mesh.num_faces() {
        if let Some(bt) = mesh.face_boundary[f] {
            *sums.entry(bt as u8).or_insert(0.0) += mesh.face_area[f];
        }
    }
    sums
}

fn assert_mesh_equivalent(
    name: &str,
    cpu_mesh: &Mesh,
    gpu_mesh: &Mesh,
    rounded: &[Point2<f64>],
    hmin: f64,
) {
    assert_eq!(
        cpu_mesh.num_cells(),
        gpu_mesh.num_cells(),
        "[{name}] cell count differs"
    );

    // Interior adjacency (post sliver filter) must be identical.
    let pc = interior_pairs(cpu_mesh, rounded);
    let pg = interior_pairs(gpu_mesh, rounded);
    let only_cpu: Vec<_> = pc.difference(&pg).take(8).collect();
    let only_gpu: Vec<_> = pg.difference(&pc).take(8).collect();
    assert!(
        pc == pg,
        "[{name}] interior neighbor pairs differ: cpu-only {only_cpu:?} gpu-only {only_gpu:?} \
         (|cpu|={}, |gpu|={})",
        pc.len(),
        pg.len()
    );

    // Wall length sums per BoundaryType, rel < 1e-6.
    let wc = wall_length_sums(cpu_mesh);
    let wg = wall_length_sums(gpu_mesh);
    assert_eq!(
        wc.keys().collect::<Vec<_>>(),
        wg.keys().collect::<Vec<_>>(),
        "[{name}] boundary-type sets differ"
    );
    for (bt, &lc) in &wc {
        let lg = wg[bt];
        let rel = (lc - lg).abs() / lc.max(1e-300);
        assert!(
            rel < 1e-6,
            "[{name}] wall length sum for type {bt} differs: cpu {lc:.9} gpu {lg:.9} (rel {rel:.2e})"
        );
    }

    // Cell geometry within f32 tolerances.
    for i in 0..cpu_mesh.num_cells() {
        let vc = cpu_mesh.cell_vol[i];
        let vg = gpu_mesh.cell_vol[i];
        let vol_rel = (vc - vg).abs() / vc;
        let dcx = cpu_mesh.cell_cx[i] - gpu_mesh.cell_cx[i];
        let dcy = cpu_mesh.cell_cy[i] - gpu_mesh.cell_cy[i];
        let cen_err = (dcx * dcx + dcy * dcy).sqrt();
        assert!(
            vol_rel < 1e-5 && cen_err < 1e-5 * hmin,
            "[{name}] cell {i}: vol_rel {vol_rel:.2e} cen_err {cen_err:.2e}"
        );
    }
}

// ---------------------------------------------------------------------------
// The full stage-3 protocol for one geometry
// ---------------------------------------------------------------------------

fn run_geometry_case(
    name: &str,
    geo: &(impl Geometry + Sync),
    domain: Vector2<f64>,
    hmin: f64,
    hmax: f64,
    report_bins: bool,
) {
    let Some(ctx) = gpu_context() else { return };
    let (seeds, kinds, spec) =
        cfd2::meshgen::meshless::meshless_seed_points(geo, hmin, hmax, 1.2, domain);
    let n = seeds.len();
    let n_boundary = kinds
        .iter()
        .filter(|k| matches!(k, SeedKind::Boundary { .. }))
        .count();
    let tol = MeshgenTolerances::from_geometry(hmin, domain);

    // f32-rounded seeds AND boundary spec — the identical inputs both
    // engines consume (review F4).
    let seeds_f32: Vec<f32> = seeds
        .iter()
        .flat_map(|p| [p.x as f32, p.y as f32])
        .collect();
    let rounded: Vec<Point2<f64>> = (0..n)
        .map(|i| Point2::new(seeds_f32[2 * i] as f64, seeds_f32[2 * i + 1] as f64))
        .collect();
    let spec_r = boundary_spec_f32(&spec);
    let flags = vec![0u32; n];
    let canon = canon_map(&rounded, &tol);

    // CPU f64 oracle diagram on the rounded inputs.
    let input = MeshlessInput {
        seeds: &rounded,
        kinds: &kinds,
        boundary: &spec_r,
        domain,
        tol: &tol,
        cfg: EngineConfig::default(),
    };
    let cpu = build_diagram(&input);
    let cpu_ok = |i: usize| matches!(cpu.status[i], CellStatus::Ok | CellStatus::OkEscalated(_));

    // GPU engine.
    let mut engine = GpuVoronoiEngine::new(&ctx.device, n as u32, domain, &tol);
    engine.upload_case(&ctx.device, &ctx.queue, &seeds_f32, &flags, &kinds, &spec);
    // The engine must have derived the exact same rounded spec.
    assert_eq!(engine.boundary().num_segments(), spec_r.num_segments());
    let idx = engine.run_regen(&ctx.device, &ctx.queue);
    let _ = ctx.device.poll(wgpu::PollType::Wait {
        submission_index: Some(idx),
        timeout: None,
    });
    let cache = StagingBufferCache::default();
    let gpu = engine.read_cells(&ctx, &cache);
    let flagged_set: BTreeSet<u32> = gpu.flagged.iter().copied().collect();

    // --- Pre-fallback zero-tolerance gates -----------------------------
    let mut zero_tol_failures: Vec<String> = Vec::new();
    let mut overflows = 0usize;
    for i in 0..n {
        match gpu.status[i] {
            status::SUCCESS => {
                assert!(
                    cpu_ok(i),
                    "[{name}] cell {i}: GPU SUCCESS but CPU status {:?}",
                    cpu.status[i]
                );
                let eps_face = 1e-6 * local_h(&cpu, &rounded, i, hmin);
                let (cn, cb) = cpu_sets(&cpu, &canon, i, eps_face);
                let (gn, gb) = gpu_sets(&gpu, i, eps_face);
                if cn != gn || cb != gb {
                    if zero_tol_failures.len() < 12 {
                        zero_tol_failures.push(format!(
                            "cell {i} (kind {:?}): nbrs cpu-only {:?} gpu-only {:?}; \
                             bc cpu {:?} gpu {:?}",
                            kinds[i],
                            cn.difference(&gn).collect::<Vec<_>>(),
                            gn.difference(&cn).collect::<Vec<_>>(),
                            cb,
                            gb
                        ));
                    } else {
                        zero_tol_failures.push("...".into());
                    }
                }
            }
            status::EMPTY_CELL => {
                if !flagged_set.contains(&(i as u32)) {
                    assert_eq!(
                        cpu.status[i],
                        CellStatus::EmptyCell,
                        "[{name}] cell {i}: GPU coalesced-empty but CPU status {:?}",
                        cpu.status[i]
                    );
                }
            }
            status::NEEDS_EXACT => {
                assert!(
                    flagged_set.contains(&(i as u32)),
                    "[{name}] cell {i}: NEEDS_EXACT but missing from the flag list"
                );
            }
            s @ (status::VERT_OVERFLOW | status::FACE_OVERFLOW) => {
                overflows += 1;
                assert!(
                    flagged_set.contains(&(i as u32)),
                    "[{name}] cell {i}: overflow status {s} but missing from the flag list"
                );
            }
            other => panic!("[{name}] cell {i}: unexpected GPU status {other}"),
        }
    }
    assert!(
        zero_tol_failures.is_empty(),
        "[{name}] ZERO-TOLERANCE violated: {} SUCCESS cells disagree with the f64 oracle:\n{}",
        zero_tol_failures.len(),
        zero_tol_failures.join("\n")
    );
    assert_eq!(
        overflows, 0,
        "[{name}] VERT/FACE_OVERFLOW must not fire on standard geometry sets"
    );

    // Flag-rate budget. The strict 2e-3 design budget applies to the BULK
    // interior (cells with no boundary-seed neighbor — the Poisson-like
    // class it was calibrated on; stage-2's interior suites keep asserting
    // it globally). The WALL STRIP (boundary-kind seeds + interior cells
    // adjacent to one) legitimately exceeds it on geometries with curved /
    // kinked walls: M0's `boundary_seeds` emits the two guards of adjacent
    // reflex vertices onto their shared segment at DIFFERENT `t` whenever
    // the flanking segment lengths differ, so they miss the midpoint
    // collapse and land 1e-3·h..1e-2·h apart — knife-edge twin seeds whose
    // surrounding vertices carry true f32 errors up to ~1e-4·h (measured:
    // every nozzle vert-err flag is such a same-segment guard pair).
    // Flag + f64 fallback is exactly the designed mechanism for them; a
    // defensive 35% strip cap catches regressions (measured strip rates:
    // rect 1.8%, obstacle 1.9%, backstep 2.5%, nozzle 27.7%, graded 28.7%).
    // (Root-cause lever, deferred to an M0 arc: collapse/equalize
    // same-segment guard pairs.)
    let mut wall_strip = vec![false; n];
    for i in 0..n {
        if matches!(kinds[i], SeedKind::Boundary { .. }) {
            wall_strip[i] = true;
            continue;
        }
        let len = cpu.ring_len[i] as usize;
        for e in 0..len {
            if let PlaneTag::Bisector(j) = cpu.ring_plane[i * MAX_CLIP_VERTS + e] {
                if matches!(kinds[j as usize], SeedKind::Boundary { .. }) {
                    wall_strip[i] = true;
                    break;
                }
            }
        }
    }
    let n_strip = wall_strip.iter().filter(|&&s| s).count();
    let n_bulk = n - n_strip;
    let flagged_strip = gpu
        .flagged
        .iter()
        .filter(|&&i| wall_strip[i as usize])
        .count();
    let flagged_bulk = gpu.flagged.len() - flagged_strip;
    let flag_rate = gpu.flagged.len() as f64 / n as f64;
    println!(
        "[{name}] n={n} (boundary {n_boundary}, wall-strip {n_strip}) flagged={} \
         (rate {:.2e}; strip {flagged_strip}/{n_strip} bulk {flagged_bulk}/{n_bulk}) overflow=0",
        gpu.flagged.len(),
        flag_rate,
    );
    // Filter-condition mask census of flagged cells (visited_bins high byte).
    let visited = engine.read_visited_bins(&ctx, &cache);
    if !gpu.flagged.is_empty() {
        let mut cond_counts = [0usize; 4];
        for &i in &gpu.flagged {
            let mask = visited[i as usize] >> 24;
            for (b, c) in cond_counts.iter_mut().enumerate() {
                if mask & (1 << b) != 0 {
                    *c += 1;
                }
            }
        }
        println!(
            "[{name}] flag conditions: band={} degenerate-det={} short-face={} vert-err={}",
            cond_counts[0], cond_counts[1], cond_counts[2], cond_counts[3]
        );
    }

    let bulk_budget = ((n_bulk as f64) * 2e-3).ceil() as usize;
    assert!(
        flagged_bulk <= bulk_budget,
        "[{name}] bulk-interior flag rate over the 2e-3 budget: {flagged_bulk} > {bulk_budget}"
    );
    let strip_cap = (n_strip as f64 * 0.35).ceil() as usize;
    assert!(
        flagged_strip <= strip_cap,
        "[{name}] wall-strip flag count {flagged_strip} exceeds the defensive 35% cap {strip_cap}"
    );

    // Visited-bin distribution (review F3 instrument for graded sets).
    if report_bins {
        let mut bins: Vec<u32> = visited.iter().map(|v| v & 0x00ff_ffff).collect();
        bins.sort_unstable();
        let sum: u64 = bins.iter().map(|&b| b as u64).sum();
        println!(
            "[{name}] visited bins: mean {:.1} p50 {} p99 {} max {}",
            sum as f64 / n as f64,
            bins[n / 2],
            bins[(n as f64 * 0.99) as usize],
            bins[n - 1]
        );
    }

    // --- Byte stability -------------------------------------------------
    let raw1 = engine.read_raw_outputs(&ctx, &cache);
    let idx = engine.run_regen(&ctx.device, &ctx.queue);
    let _ = ctx.device.poll(wgpu::PollType::Wait {
        submission_index: Some(idx),
        timeout: None,
    });
    let raw2 = engine.read_raw_outputs(&ctx, &cache);
    assert_eq!(raw1, raw2, "[{name}] regen outputs not byte-stable run-to-run");

    // --- Fallback + merged parity ----------------------------------------
    let report = engine.resolve_flagged(&ctx, &cache);
    println!(
        "[{name}] resolve: patched={} recip_rounds={} recip_flagged={} sub_eps_asym={} unresolved={}",
        report.patched.len(),
        report.reciprocity_rounds,
        report.reciprocity_flagged.len(),
        report.sub_eps_asymmetries,
        report.unresolved.len()
    );
    assert_eq!(report.flagged, gpu.flagged);
    assert!(
        report.unresolved.is_empty(),
        "[{name}] f64 fallback could not patch cells {:?}",
        report.unresolved
    );

    let merged = engine.read_cells(&ctx, &cache);
    for i in 0..n {
        match merged.status[i] {
            status::SUCCESS => {
                assert!(cpu_ok(i), "[{name}] merged cell {i}: CPU status {:?}", cpu.status[i]);
                let h_i = local_h(&cpu, &rounded, i, hmin);
                let eps_face = 1e-6 * h_i;
                let (cn, cb) = cpu_sets(&cpu, &canon, i, eps_face);
                let (gn, gb) = gpu_sets(&merged, i, eps_face);
                assert!(
                    cn == gn && cb == gb,
                    "[{name}] merged cell {i} (kind {:?}): topology mismatch: \
                     nbrs cpu-only {:?} gpu-only {:?}; bc cpu {:?} gpu {:?}",
                    kinds[i],
                    cn.difference(&gn).collect::<Vec<_>>(),
                    gn.difference(&cn).collect::<Vec<_>>(),
                    cb,
                    gb
                );
                let cpu_area = cpu.area[i];
                let area_rel = (merged.area[i] as f64 - cpu_area).abs() / cpu_area;
                let p = rounded[i];
                let c_rel = [cpu.centroid[i][0] - p.x, cpu.centroid[i][1] - p.y];
                let dcx = merged.centroid_rel[i][0] as f64 - c_rel[0];
                let dcy = merged.centroid_rel[i][1] as f64 - c_rel[1];
                let cen_err = (dcx * dcx + dcy * dcy).sqrt();
                let c_mag = (c_rel[0] * c_rel[0] + c_rel[1] * c_rel[1]).sqrt();
                let cen_tol = 1e-5 * h_i + 2.4e-7 * c_mag;
                assert!(
                    area_rel < 1e-5 && cen_err < cen_tol,
                    "[{name}] merged cell {i}: area_rel {area_rel:.2e} cen_err {cen_err:.2e} \
                     (h {h_i:.2e}, tol {cen_tol:.2e})"
                );
            }
            status::EMPTY_CELL => {
                assert_eq!(cpu.status[i], CellStatus::EmptyCell);
            }
            other => panic!("[{name}] merged cell {i}: status {other} survived resolve_flagged"),
        }
    }

    // --- Independent reciprocity check ------------------------------------
    let mut pairs: HashMap<(u32, u32), (bool, bool, f64)> = HashMap::new();
    for i in 0..n {
        for s in 0..(merged.nfaces[i] as usize).min(K_FACE_MAX) {
            let slot = i * K_FACE_MAX + s;
            let j = merged.nbr_ids[slot];
            if j == NBR_NONE {
                continue;
            }
            let iu = i as u32;
            let key = (iu.min(j), iu.max(j));
            let e = pairs.entry(key).or_insert((false, false, 0.0));
            if iu == key.0 {
                e.0 = true;
            } else {
                e.1 = true;
            }
            e.2 = e.2.max(merged.face_geom[slot][2] as f64);
        }
    }
    for (&(a, b), &(fwd, bwd, max_len)) in &pairs {
        if fwd != bwd {
            let d = (rounded[b as usize] - rounded[a as usize]).norm();
            assert!(
                max_len <= 1e-6 * d,
                "[{name}] merged diagram not reciprocal: real face ({a},{b}) one-sided \
                 (len {max_len:.3e}, |q| {d:.3e})"
            );
        }
    }

    // --- End-to-end mesh: GPU diagram -> assemble_mesh -> validate -------
    let diag = engine.cells_to_diagram(&merged);
    let gpu_mesh = assemble_mesh(&input, &diag);
    let expected = fluid_area_estimate(geo, domain, 1500);
    let issues = validate_mesh(&gpu_mesh, expected, hmin);
    println!(
        "[{name}] gpu mesh: cells={} faces={} verts={}",
        gpu_mesh.num_cells(),
        gpu_mesh.num_faces(),
        gpu_mesh.num_vertices()
    );
    assert!(
        issues.is_empty(),
        "[{name}] validate_mesh failed on the GPU-diagram mesh:\n  - {}",
        issues.join("\n  - ")
    );

    // --- Equivalence vs the pure-CPU-engine mesh --------------------------
    let cpu_mesh = assemble_mesh(&input, &cpu);
    let issues = validate_mesh(&cpu_mesh, expected, hmin);
    assert!(
        issues.is_empty(),
        "[{name}] validate_mesh failed on the CPU oracle mesh (harness bug?):\n  - {}",
        issues.join("\n  - ")
    );
    assert_mesh_equivalent(name, &cpu_mesh, &gpu_mesh, &rounded, hmin);
    println!("[{name}] mesh equivalence: OK");
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

#[test]
fn gpu_voronoi_geometry_rect_channel() {
    run_geometry_case(
        "rect/h=0.05",
        &RectangularChannel { length: 3.0, height: 1.0 },
        Vector2::new(3.0, 1.0),
        0.05,
        0.05,
        false,
    );
}

#[test]
fn gpu_voronoi_geometry_obstacle() {
    run_geometry_case(
        "obstacle/h=0.05",
        &ChannelWithObstacle {
            length: 3.0,
            height: 1.0,
            obstacle_center: Point2::new(1.0, 0.51),
            obstacle_radius: 0.1,
        },
        Vector2::new(3.0, 1.0),
        0.05,
        0.05,
        false,
    );
}

#[test]
fn gpu_voronoi_geometry_backstep() {
    run_geometry_case(
        "backstep/h=0.05",
        &BackwardsStep {
            length: 3.5,
            height_inlet: 0.5,
            height_outlet: 1.0,
            step_x: 0.5,
        },
        Vector2::new(3.5, 1.0),
        0.05,
        0.05,
        false,
    );
}

#[test]
fn gpu_voronoi_geometry_nozzle() {
    run_geometry_case(
        "nozzle/h=0.05",
        &Nozzle {
            length: 3.0,
            height: 1.0,
            throat_height: 0.40,
            throat_frac: 0.40,
            exit_height: 0.80,
        },
        Vector2::new(3.0, 1.0),
        0.05,
        0.05,
        // Uniform baseline for the visited-bin distribution.
        true,
    );
}

/// Graded nozzle seed set (review F3): min/max cell size 0.02/0.08 (h ratio
/// 4×, density ratio 16×). The CPU-built `SeedGrid` sizes bins by mean
/// occupancy, so coarse-region cells must sweep more rings before the
/// security-radius stop — the visited-bin distribution below is the
/// instrument; gates are identical to the uniform cases.
#[test]
fn gpu_voronoi_geometry_nozzle_graded() {
    run_geometry_case(
        "nozzle-graded/h=0.02..0.08",
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
        true,
    );
}
