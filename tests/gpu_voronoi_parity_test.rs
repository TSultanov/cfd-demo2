//! Parity gates for the GPU meshless Voronoi engine
//! (`src/solver/gpu/voronoi/`), interior-only bbox configuration:
//!
//! 1. ZERO-TOLERANCE topology: no `SUCCESS` cell may have its eps_face-filtered
//!    neighbor set (bisector ids canonicalized through the coalescing table, plus
//!    bbox sides) disagree with the CPU f64 `build_diagram` oracle on the same
//!    f32-rounded seeds. Any such cell is a kernel/filter bug, not a tolerance.
//! 2. Every disagreement must be flagged; `resolve_flagged` patches it to the
//!    exact f64 result; the merged diagram must be full-parity and reciprocal.
//! 3. Flag-rate budget ≤ 2e-3 on Poisson-like sets; zero VERT/FACE overflows.
//! 4. Geometry parity (`SUCCESS` + patched): area rel < 1e-5, centroid
//!    (seed-relative) < 1e-5 · local h.
//! 5. Run-to-run byte stability of all deterministic outputs.

#![cfg(feature = "meshgen")]

use std::collections::{BTreeSet, HashMap};

use cfd2::meshgen::meshless::{
    build_diagram, BoundarySpec, CellStatus, EngineConfig, MeshlessDiagram, MeshlessInput,
    PlaneTag, MAX_CLIP_VERTS,
};
use cfd2::meshgen::MeshgenTolerances;
use cfd2::solver::gpu::context::GpuContext;
use cfd2::solver::gpu::readback::StagingBufferCache;
use cfd2::solver::gpu::voronoi::{
    status, GpuVoronoiCells, GpuVoronoiEngine, K_FACE_MAX, NBR_NONE,
};
use nalgebra::{Point2, Vector2};
use rand::{Rng, SeedableRng};

const DOMAIN: Vector2<f64> = Vector2::new(2.0, 1.0);

/// Deterministic Poisson-like set: jittered lattice (`amp` is the jitter
/// amplitude relative to the lattice pitch; ≤ 0.8 keeps seeds pairwise
/// separated and strictly interior).
fn jittered_seeds(nx: usize, ny: usize, amp: f64, rng_seed: u64) -> Vec<Point2<f64>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(rng_seed);
    let (sx, sy) = (DOMAIN.x / nx as f64, DOMAIN.y / ny as f64);
    let mut pts = Vec::with_capacity(nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            let jx = (rng.gen::<f64>() - 0.5) * amp * sx;
            let jy = (rng.gen::<f64>() - 0.5) * amp * sy;
            pts.push(Point2::new(
                (i as f64 + 0.5) * sx + jx,
                (j as f64 + 0.5) * sy + jy,
            ));
        }
    }
    pts
}

fn gpu_context() -> Option<GpuContext> {
    match pollster::block_on(GpuContext::new(None, None)) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("SKIP: no GPU adapter available ({e})");
            None
        }
    }
}

/// Coalescing table over the f32-rounded seeds — the same rule the engine
/// and the CPU oracle apply (`canon[i]` = lowest index of i's quantize bin).
fn canon_map(pts: &[Point2<f64>], tol: &MeshgenTolerances) -> Vec<u32> {
    let mut first: HashMap<(i64, i64), u32> = HashMap::with_capacity(pts.len());
    let mut canon = vec![0u32; pts.len()];
    for (i, p) in pts.iter().enumerate() {
        let key = tol.quantize_point(p.x, p.y);
        canon[i] = *first.entry(key).or_insert(i as u32);
    }
    canon
}

/// Local scale of cell `i`: nearest-neighbor distance (the nearest neighbor
/// is always a Voronoi neighbor, so the unfiltered CPU ring suffices).
fn local_h(cpu: &MeshlessDiagram, rounded: &[Point2<f64>], i: usize) -> f64 {
    let p = rounded[i];
    let len = cpu.ring_len[i] as usize;
    let mut h_i = f64::INFINITY;
    for e in 0..len {
        if let PlaneTag::Bisector(j) = cpu.ring_plane[i * MAX_CLIP_VERTS + e] {
            h_i = h_i.min((rounded[j as usize] - p).norm());
        }
    }
    assert!(h_i.is_finite(), "cell {i} has no bisector neighbors");
    h_i
}

/// CPU eps_face-filtered (neighbor, side) topology of cell `i`, ids
/// canonicalized.
fn cpu_sets(
    cpu: &MeshlessDiagram,
    canon: &[u32],
    i: usize,
    eps_face: f64,
) -> (BTreeSet<u32>, BTreeSet<u32>) {
    let mut nbrs = BTreeSet::new();
    let mut sides = BTreeSet::new();
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
            PlaneTag::Box(s) => {
                sides.insert(s as u32);
            }
            PlaneTag::Boundary(_) => unreachable!("bbox-only configuration"),
        }
    }
    (nbrs, sides)
}

/// GPU eps_face-filtered topology of cell `i` (nbr ids are already
/// canonical — the kernel/patches emit `canon[tag]`).
fn gpu_sets(gpu: &GpuVoronoiCells, i: usize, eps_face: f64) -> (BTreeSet<u32>, BTreeSet<u32>) {
    let mut nbrs = BTreeSet::new();
    let mut sides = BTreeSet::new();
    for s in 0..(gpu.nfaces[i] as usize).min(K_FACE_MAX) {
        let slot = i * K_FACE_MAX + s;
        if (gpu.face_geom[slot][2] as f64) <= eps_face {
            continue;
        }
        let nbr = gpu.nbr_ids[slot];
        if nbr == NBR_NONE {
            sides.insert(gpu.face_bc[slot]);
        } else {
            nbrs.insert(nbr);
        }
    }
    (nbrs, sides)
}

struct CaseStats {
    n: usize,
    flagged: usize,
    patched: usize,
}

/// Full gate run for one seed set. `spacing` scales the meshgen
/// tolerances (as `min_cell_size = spacing / 2`); `poisson_budget` applies
/// the 2e-3 flag-rate budget (Poisson-like classes only).
fn run_case(
    name: &str,
    seeds: Vec<Point2<f64>>,
    spacing: f64,
    poisson_budget: bool,
    timing: bool,
) -> Option<CaseStats> {
    let ctx = gpu_context()?;
    let n = seeds.len();
    let tol = MeshgenTolerances::from_geometry(0.5 * spacing, DOMAIN);

    // f32-rounded seeds: what the kernel sees AND what the CPU oracle gets
    // (widened back to f64 — exact).
    let seeds_f32: Vec<f32> = seeds
        .iter()
        .flat_map(|p| [p.x as f32, p.y as f32])
        .collect();
    let rounded: Vec<Point2<f64>> = (0..n)
        .map(|i| Point2::new(seeds_f32[2 * i] as f64, seeds_f32[2 * i + 1] as f64))
        .collect();
    let flags = vec![0u32; n];
    let canon = canon_map(&rounded, &tol);

    // GPU diagram.
    let mut engine = GpuVoronoiEngine::new(&ctx.device, n as u32, DOMAIN, &tol);
    engine.upload_seeds(&ctx.device, &ctx.queue, &seeds_f32, &flags);
    let idx = engine.run_regen(&ctx.device, &ctx.queue);
    let _ = ctx.device.poll(wgpu::PollType::Wait {
        submission_index: Some(idx),
        timeout: None,
    });
    let cache = StagingBufferCache::default();
    let gpu = engine.read_cells(&ctx, &cache);
    let flagged_set: BTreeSet<u32> = gpu.flagged.iter().copied().collect();

    // CPU f64 oracle on the identical (f32-rounded) inputs.
    let boundary = BoundarySpec::empty();
    let input = MeshlessInput::interior_only(
        &rounded,
        &boundary,
        DOMAIN,
        &tol,
        EngineConfig::default(),
    );
    let cpu = build_diagram(&input);
    let cpu_ok = |i: usize| matches!(cpu.status[i], CellStatus::Ok | CellStatus::OkEscalated(_));

    // --- Pre-fallback gates -------------------------------------------
    // ZERO TOLERANCE: a SUCCESS cell disagreeing with f64 (post eps_face)
    // is a bug; disagreements are only legal on flagged cells.
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
                let eps_face = 1e-6 * local_h(&cpu, &rounded, i);
                let (cn, cs) = cpu_sets(&cpu, &canon, i, eps_face);
                let (gn, gs) = gpu_sets(&gpu, i, eps_face);
                if cn != gn || cs != gs {
                    if zero_tol_failures.len() < 12 {
                        zero_tol_failures.push(format!(
                            "cell {i}: nbrs cpu-only {:?} gpu-only {:?}; sides cpu {:?} gpu {:?}",
                            cn.difference(&gn).collect::<Vec<_>>(),
                            gn.difference(&cn).collect::<Vec<_>>(),
                            cs,
                            gs
                        ));
                    } else {
                        zero_tol_failures.push("...".into());
                    }
                }
            }
            status::EMPTY_CELL => {
                // Unflagged EMPTY = coalesced duplicate; must match the CPU
                // verdict. (Flagged EMPTY = f32 clipped the cell away; the
                // fallback resolves it.)
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
                    "[{name}] cell {i}: NEEDS_EXACT status but missing from the flag list"
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

    let flag_rate = gpu.flagged.len() as f64 / n as f64;
    println!(
        "[{name}] n={n} flagged={} (rate {:.2e}) overflow={overflows}",
        gpu.flagged.len(),
        flag_rate
    );
    assert_eq!(
        overflows, 0,
        "[{name}] VERT/FACE_OVERFLOW must not fire on these seed sets"
    );
    if poisson_budget {
        let budget = ((n as f64) * 2e-3).ceil() as usize;
        assert!(
            gpu.flagged.len() <= budget,
            "[{name}] flag rate over the 2e-3 Poisson budget: {} > {budget}",
            gpu.flagged.len()
        );
    }

    // --- Byte stability (before patches land) -------------------------
    let raw1 = engine.read_raw_outputs(&ctx, &cache);
    let idx = engine.run_regen(&ctx.device, &ctx.queue);
    let _ = ctx.device.poll(wgpu::PollType::Wait {
        submission_index: Some(idx),
        timeout: None,
    });
    let raw2 = engine.read_raw_outputs(&ctx, &cache);
    assert_eq!(raw1, raw2, "[{name}] regen outputs not byte-stable run-to-run");

    if timing {
        let mut best = f64::INFINITY;
        let mut total = 0.0;
        const REPS: usize = 10;
        for _ in 0..REPS {
            let t0 = std::time::Instant::now();
            let idx = engine.run_regen(&ctx.device, &ctx.queue);
            let _ = ctx.device.poll(wgpu::PollType::Wait {
                submission_index: Some(idx),
                timeout: None,
            });
            let dt = t0.elapsed().as_secs_f64() * 1e3;
            best = best.min(dt);
            total += dt;
        }
        println!(
            "[{name}] regen wall time over {REPS} reps: best {best:.3} ms, mean {:.3} ms",
            total / REPS as f64
        );
    }

    // --- Fallback + reciprocity enforcement ----------------------------
    let report = engine.resolve_flagged(&ctx, &cache);
    println!(
        "[{name}] resolve: patched={} recip_rounds={} recip_flagged={} sub_eps_asym={} unresolved={}",
        report.patched.len(),
        report.reciprocity_rounds,
        report.reciprocity_flagged.len(),
        report.sub_eps_asymmetries,
        report.unresolved.len()
    );
    assert_eq!(
        report.flagged, gpu.flagged,
        "[{name}] resolve_flagged bounded over-read disagrees with the full readback"
    );
    assert!(
        report.unresolved.is_empty(),
        "[{name}] f64 fallback could not patch cells {:?}",
        report.unresolved
    );

    // --- Merged-diagram gates: FULL parity ----------------------------
    let merged = engine.read_cells(&ctx, &cache);
    let patched_set: BTreeSet<u32> = report.patched.iter().copied().collect();
    for i in 0..n {
        match merged.status[i] {
            status::SUCCESS => {
                assert!(cpu_ok(i), "[{name}] merged cell {i}: CPU status {:?}", cpu.status[i]);
                let h_i = local_h(&cpu, &rounded, i);
                let eps_face = 1e-6 * h_i;
                let (cn, cs) = cpu_sets(&cpu, &canon, i, eps_face);
                let (gn, gs) = gpu_sets(&merged, i, eps_face);
                assert!(
                    cn == gn && cs == gs,
                    "[{name}] merged cell {i} (patched: {}): topology mismatch: \
                     nbrs cpu-only {:?} gpu-only {:?}; sides cpu {:?} gpu {:?}",
                    patched_set.contains(&(i as u32)),
                    cn.difference(&gn).collect::<Vec<_>>(),
                    gn.difference(&cn).collect::<Vec<_>>(),
                    cs,
                    gs
                );
                // Geometry parity. The centroid tolerance carries an
                // absolute f32-quantization term: patched cells store the
                // exact f64 centroid rounded to f32, and for knife-edge
                // twin cells (h = twin gap ≪ cell size) that quantum
                // (~ε·|c_rel|) exceeds 1e-5·h.
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
                assert_eq!(
                    cpu.status[i],
                    CellStatus::EmptyCell,
                    "[{name}] merged cell {i}: GPU empty but CPU status {:?}",
                    cpu.status[i]
                );
            }
            other => panic!("[{name}] merged cell {i}: status {other} survived resolve_flagged"),
        }
    }

    // --- Independent reciprocity check on the merged diagram ----------
    // (the engine enforces this internally; re-derive it from the raw
    // readback so the gate does not trust the enforcement code).
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
                "[{name}] merged diagram not reciprocal: real face ({a},{b}) is one-sided \
                 (len {max_len:.3e}, |q| {d:.3e})"
            );
        }
    }

    Some(CaseStats {
        n,
        flagged: gpu.flagged.len(),
        patched: report.patched.len(),
    })
}

#[test]
fn gpu_voronoi_interior_parity_5k() {
    let seeds = jittered_seeds(100, 50, 0.8, 0x5EED_CFD2);
    run_case("5k", seeds, DOMAIN.y / 50.0, true, false);
}

#[test]
fn gpu_voronoi_interior_parity_30k() {
    let seeds = jittered_seeds(245, 123, 0.8, 0xCFD2_5EED);
    run_case("30k", seeds, DOMAIN.y / 123.0, true, true);
}

/// Full protocol at ~300k seeds. dev-tests-gated: run in release.
#[cfg(feature = "dev-tests")]
#[test]
fn gpu_voronoi_interior_parity_300k() {
    let seeds = jittered_seeds(775, 388, 0.8, 0xCFD2_0300);
    run_case("300k", seeds, DOMAIN.y / 388.0, true, true);
}

/// Near-cocircular lattices: a 64×32 lattice on an exactly f32-representable
/// pitch (2/64 = 2⁻⁵), jitter swept from
/// Poisson-like down to EXACTLY cocircular (every interior Voronoi vertex
/// is a 4-seed tie; amp 1e-6 of the pitch is sub-f32-ulp, i.e. seeds land
/// 0-1 ulp off the exact lattice). Flag rates are expected to be large
/// here — the budget does not apply; the gates are zero-tolerance +
/// full post-fallback parity + reciprocity.
#[test]
fn gpu_voronoi_adversarial_near_cocircular() {
    let spacing = DOMAIN.x / 64.0;
    for (k, amp) in [0.0, 1e-6, 1e-4, 1e-2].into_iter().enumerate() {
        let name = format!("cocircular/amp{amp:.0e}");
        let seeds = jittered_seeds(64, 32, amp, 0xC0C1_0000 + k as u64);
        if let Some(s) = run_case(&name, seeds, spacing, false, false) {
            println!(
                "[{name}] class report: flag_rate={:.3e} patched={}",
                s.flagged as f64 / s.n as f64,
                s.patched
            );
        }
    }
}

/// Knife-edge twins + exact duplicates in a Poisson-like set: twin gaps
/// swept from 1e-2 down to 1e-5 of the pitch (the smallest is a handful of
/// f32 ulps — some twins collapse
/// to identical f32 values and must coalesce), plus exact duplicates that
/// must coalesce to `EMPTY_CELL` on both engines.
#[test]
fn gpu_voronoi_adversarial_twins_and_duplicates() {
    let spacing = DOMAIN.x / 64.0;
    let mut seeds = jittered_seeds(64, 32, 0.8, 0x7717_5EED);
    let n_base = seeds.len();
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x7717_0002);
    for gap_rel in [1e-2, 1e-3, 1e-4, 1e-5] {
        for _ in 0..10 {
            let b = rng.gen_range(0..n_base);
            let ang = rng.gen::<f64>() * std::f64::consts::TAU;
            let gap = gap_rel * spacing;
            let p = seeds[b];
            seeds.push(Point2::new(p.x + gap * ang.cos(), p.y + gap * ang.sin()));
        }
    }
    // Exact duplicates: identical f64 (hence f32) coordinates; the engines
    // must coalesce them (the merged-diagram gate cross-checks EMPTY_CELL
    // against the CPU EmptyCell verdict in both directions).
    for _ in 0..10 {
        let b = rng.gen_range(0..n_base);
        seeds.push(seeds[b]);
    }
    let stats = run_case("twins", seeds, spacing, false, false);
    if let Some(s) = stats {
        println!(
            "[twins] class report: flag_rate={:.3e} patched={}",
            s.flagged as f64 / s.n as f64,
            s.patched
        );
    }
}

/// The release reciprocity enforcement must actually repair a one-sided
/// diagram, not just observe clean ones (on healthy seed sets the epsilon
/// filter pre-empts every violation, so `reciprocity_rounds` stays 0).
/// Corrupt one cell's neighbor row after regen — its neighbors then hold
/// real one-sided faces — and assert `resolve_flagged` detects it, extends
/// the flagged set with the disagreeing cells, and converges to a
/// reciprocal, oracle-exact diagram.
#[test]
fn gpu_voronoi_reciprocity_enforcement_repairs_corruption() {
    let Some(ctx) = gpu_context() else { return };
    let seeds = jittered_seeds(100, 50, 0.8, 0x5EED_CFD2);
    let n = seeds.len();
    let spacing = DOMAIN.y / 50.0;
    let tol = MeshgenTolerances::from_geometry(0.5 * spacing, DOMAIN);
    let seeds_f32: Vec<f32> = seeds
        .iter()
        .flat_map(|p| [p.x as f32, p.y as f32])
        .collect();
    let rounded: Vec<Point2<f64>> = (0..n)
        .map(|i| Point2::new(seeds_f32[2 * i] as f64, seeds_f32[2 * i + 1] as f64))
        .collect();
    let flags = vec![0u32; n];
    let canon = canon_map(&rounded, &tol);

    let mut engine = GpuVoronoiEngine::new(&ctx.device, n as u32, DOMAIN, &tol);
    engine.upload_seeds(&ctx.device, &ctx.queue, &seeds_f32, &flags);
    let idx = engine.run_regen(&ctx.device, &ctx.queue);
    let _ = ctx.device.poll(wgpu::PollType::Wait {
        submission_index: Some(idx),
        timeout: None,
    });

    // Corrupt an interior cell: erase its whole neighbor row (its faces
    // toward every neighbor vanish; the neighbors still list it).
    const VICTIM: u32 = 2525;
    let empty_row = [NBR_NONE; K_FACE_MAX];
    ctx.queue.write_buffer(
        &engine.outputs.b_nbr_ids,
        (VICTIM as u64) * (K_FACE_MAX as u64) * 4,
        bytemuck::cast_slice(&empty_row),
    );

    let cache = StagingBufferCache::default();
    let report = engine.resolve_flagged(&ctx, &cache);
    println!(
        "[recip-repair] rounds={} recip_flagged={:?}",
        report.reciprocity_rounds, report.reciprocity_flagged
    );
    assert!(
        report.reciprocity_rounds >= 1,
        "corruption must trigger at least one enforcement round"
    );
    assert!(
        report.reciprocity_flagged.contains(&VICTIM),
        "the corrupted cell must be re-resolved"
    );
    assert!(report.unresolved.is_empty());

    // The repaired diagram must match the f64 oracle around the victim and
    // be reciprocal (resolve_flagged asserts convergence internally; verify
    // the victim's topology independently).
    let boundary = BoundarySpec::empty();
    let input = MeshlessInput::interior_only(
        &rounded,
        &boundary,
        DOMAIN,
        &tol,
        EngineConfig::default(),
    );
    let cpu = build_diagram(&input);
    let merged = engine.read_cells(&ctx, &cache);
    assert_eq!(merged.status[VICTIM as usize], status::SUCCESS);
    let eps_face = 1e-6 * local_h(&cpu, &rounded, VICTIM as usize);
    let (cn, cs) = cpu_sets(&cpu, &canon, VICTIM as usize, eps_face);
    let (gn, gs) = gpu_sets(&merged, VICTIM as usize, eps_face);
    assert!(
        cn == gn && cs == gs,
        "repaired victim cell topology must match the oracle: cpu {cn:?}/{cs:?} gpu {gn:?}/{gs:?}"
    );
}
