//! Meshless engine fuzz battery: adversarial seed sets must produce
//! *statuses*, never panics, and never NaNs. Classes:
//!
//! (a) min-separation-filtered white noise — the engine's design input;
//! (b) perturbed regular grid, perturbation swept 0 → 0.49h;
//! (c) EXACT cocircular lattice — every interior Voronoi vertex is exactly
//!     4-cocircular and every kNN distance ties (exercises the (d², id) total
//!     order + eps-inside policy);
//! (d) Gaussian density cluster with near-duplicate twins and boundary-hugging
//!     seeds — violates the smooth-density assumption; REPORT-ONLY on the
//!     success fraction (must still not panic; Ok cells must still be valid
//!     geometry).
//!
//! Contracts asserted per diagram: no panic (`catch_unwind`), status/slot
//! consistency for every cell, convexity + positive area + seed containment
//! for every Ok/OkEscalated cell, no NaN in any array (including padding and
//! overflow spills), success fraction ≥ 99.9% for classes a–c, and the
//! partition of unity (Σ areas == bbox) whenever no cell is Empty/Failed.
//!
//! Clean diagrams (no Empty/Failed) ALSO run `assemble_mesh` — the last
//! pipeline pass must share the "statuses, never panics" contract on legal
//! inputs; class (e) pins the knife-edge twin reproducer.
//!
//! RNG seeds are explicit per (class, run, sub-case). Scale with
//! CFD2_MESHLESS_FUZZ_RUNS (default 4):
//!
//! ```sh
//! CFD2_MESHLESS_FUZZ_RUNS=16 cargo test --features meshgen \
//!     --test meshless_fuzz_test -- --nocapture
//! ```

#![cfg(feature = "meshgen")]

use cfd2::meshgen::meshless::{
    assemble_mesh, build_diagram, BoundarySpec, CellStatus, EngineConfig, MeshlessDiagram,
    MeshlessInput, MAX_CLIP_VERTS,
};
use cfd2::meshgen::MeshgenTolerances;
use nalgebra::{Point2, Vector2};
use rand::{Rng, SeedableRng};
use std::panic::{catch_unwind, AssertUnwindSafe};

fn fuzz_runs() -> usize {
    std::env::var("CFD2_MESHLESS_FUZZ_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4)
        .max(1)
}

/// Explicit per-case RNG seed: class tag in the top byte, run and sub-case
/// below; independent of the crate's global fixed seed.
fn seed_for(class: u8, run: usize, sub: usize) -> u64 {
    0xF0_22_00_00_00_00_00_00u64
        | ((class as u64) << 48)
        | ((run as u64 & 0xFF_FF) << 16)
        | (sub as u64 & 0xFF_FF)
}

/// One fuzz case: a seed set with the nominal spacing its tolerances are
/// scaled by. All classes are pure-bbox inputs (no boundary loops) — the
/// boundary machinery has its own gate suite; fuzz targets the clip engine.
struct FuzzCase {
    name: String,
    seeds: Vec<Point2<f64>>,
    domain: Vector2<f64>,
    h: f64,
}

/// Per-cell geometric invariants + whole-diagram consistency. Returns the
/// status census so the caller can apply its class's success policy.
fn check_diagram(case: &FuzzCase) -> (MeshlessDiagram, (usize, usize, usize, usize, usize)) {
    let tol = MeshgenTolerances::from_geometry(case.h, case.domain);
    let boundary = BoundarySpec::empty();
    let input = MeshlessInput::interior_only(
        &case.seeds,
        &boundary,
        case.domain,
        &tol,
        EngineConfig::default(),
    );

    // 1. Never panics.
    let d = catch_unwind(AssertUnwindSafe(|| build_diagram(&input)))
        .unwrap_or_else(|_| panic!("{}: build_diagram panicked", case.name));

    // 2. No NaNs anywhere — including padded slots and overflow spills.
    assert!(
        d.ring_xy.iter().all(|v| v[0].is_finite() && v[1].is_finite()),
        "{}: non-finite ring vertex",
        case.name
    );
    assert!(
        d.centroid.iter().all(|c| c[0].is_finite() && c[1].is_finite()),
        "{}: non-finite centroid",
        case.name
    );
    assert!(d.area.iter().all(|a| a.is_finite()), "{}: non-finite area", case.name);
    for (i, ring) in &d.overflow {
        assert!(
            ring.iter().all(|(v, _)| v[0].is_finite() && v[1].is_finite()),
            "{}: non-finite overflow vertex (cell {i})",
            case.name
        );
    }

    // 3. Status/slot consistency + invariants on successful cells.
    let bbox = case.domain.x * case.domain.y;
    assert_eq!(d.status.len(), case.seeds.len(), "{}: status length", case.name);
    for i in 0..d.n {
        let ring: Option<Vec<[f64; 2]>> = match d.status[i] {
            CellStatus::Ok | CellStatus::OkEscalated(_) => {
                let len = d.ring_len[i] as usize;
                assert!(len >= 3, "{} cell {i}: Ok ring shorter than a triangle", case.name);
                Some(d.ring_xy[i * MAX_CLIP_VERTS..i * MAX_CLIP_VERTS + len].to_vec())
            }
            CellStatus::RingOverflow => {
                assert_eq!(d.ring_len[i], 0, "{} cell {i}: overflow slot not empty", case.name);
                let idx = d
                    .overflow
                    .binary_search_by_key(&(i as u32), |(c, _)| *c)
                    .unwrap_or_else(|_| panic!("{} cell {i}: overflow without spill", case.name));
                Some(d.overflow[idx].1.iter().map(|(v, _)| *v).collect())
            }
            CellStatus::EmptyCell | CellStatus::SecurityRadiusFailed => {
                assert_eq!(d.ring_len[i], 0, "{} cell {i}: failed slot not empty", case.name);
                None
            }
        };
        let Some(xy) = ring else { continue };
        let n = xy.len();

        // Positive, bounded area.
        assert!(
            d.area[i] > 0.0 && d.area[i] <= bbox * (1.0 + 1e-12),
            "{} cell {i}: area {} out of (0, bbox]",
            case.name,
            d.area[i]
        );

        // CCW convexity up to the cross tolerance (near-degenerate
        // cocircular edges legitimately produce ~0 crosses).
        for e in 0..n {
            let a = xy[e];
            let m = xy[(e + 1) % n];
            let c = xy[(e + 2) % n];
            let cross = (m[0] - a[0]) * (c[1] - m[1]) - (m[1] - a[1]) * (c[0] - m[0]);
            assert!(
                cross > -tol.cross_eps,
                "{} cell {i} vert {e}: concave turn (cross {cross:.3e})",
                case.name
            );
        }

        // Seed containment: the seed must not fall OUTSIDE its own cell.
        // (Strictly-inside is not assertable here: class (d) plants twins
        // 1e-12 apart, whose seeds sit 5e-13 from the shared bisector.)
        let s = case.seeds[i];
        for e in 0..n {
            let a = xy[e];
            let b = xy[(e + 1) % n];
            let (ex, ey) = (b[0] - a[0], b[1] - a[1]);
            let len = (ex * ex + ey * ey).sqrt();
            if len < 1e-14 {
                continue; // degenerate edge carries no constraint
            }
            let dist = (ex * (s.y - a[1]) - ey * (s.x - a[0])) / len;
            assert!(
                dist > -1e-9 * case.h.max(1.0),
                "{} cell {i}: seed outside its cell (signed dist {dist:.3e})",
                case.name
            );
        }
    }

    let counts = d.status_counts();

    // 4. Assembly must share the never-panics contract on clean diagrams.
    //    Sanity: seed-i == cell-i and cell volumes still partition the bbox.
    let (_, _, _, empty, failed) = counts;
    if empty + failed == 0 {
        let mesh = catch_unwind(AssertUnwindSafe(|| assemble_mesh(&input, &d)))
            .unwrap_or_else(|_| panic!("{}: assemble_mesh panicked", case.name));
        assert_eq!(mesh.num_cells(), case.seeds.len(), "{}: cell count", case.name);
        let total: f64 = mesh.cell_vol.iter().sum();
        let rel = ((total - bbox) / bbox).abs();
        assert!(
            rel < 1e-9,
            "{}: assembled volumes broke the partition (rel err {rel:.3e})",
            case.name
        );
    }

    (d, counts)
}

/// Class a–c policy: ≥ 99.9% of cells Ok/OkEscalated, and the partition of
/// unity holds whenever nothing was dropped (Empty/Failed).
fn assert_success_and_partition(case: &FuzzCase, d: &MeshlessDiagram) {
    let (ok, esc, ovf, empty, failed) = d.status_counts();
    let n = case.seeds.len();
    let frac = (ok + esc) as f64 / n as f64;
    assert!(
        frac >= 0.999,
        "{}: success fraction {frac:.5} < 99.9% (ok={ok} esc={esc} ovf={ovf} empty={empty} failed={failed})",
        case.name
    );
    if empty + failed == 0 {
        let bbox = case.domain.x * case.domain.y;
        let total: f64 = d.area.iter().sum();
        let rel = ((total - bbox) / bbox).abs();
        assert!(
            rel < 1e-9,
            "{}: partition of unity broken (Σ areas rel err {rel:.3e})",
            case.name
        );
    }
}

// ---------------------------------------------------------------------------
// Class (a): min-separation-filtered white noise
// ---------------------------------------------------------------------------

#[test]
fn fuzz_min_separation_white_noise() {
    let domain = Vector2::new(2.0, 1.0);
    for run in 0..fuzz_runs() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed_for(b'a', run, 0));
        // Vary density across runs; min separation = 0.55·h keeps the set a
        // legal Poisson-like input while packing tighter than the sampler.
        let h = 0.05 / (1.0 + 0.5 * (run % 4) as f64);
        let min_sep = 0.55 * h;
        let target = ((domain.x * domain.y) / (h * h) * 0.8) as usize;
        let mut seeds: Vec<Point2<f64>> = Vec::new();
        let mut attempts = 0usize;
        while seeds.len() < target && attempts < target * 200 {
            attempts += 1;
            let p = Point2::new(rng.gen::<f64>() * domain.x, rng.gen::<f64>() * domain.y);
            if seeds
                .iter()
                .all(|q| (q - p).norm_squared() >= min_sep * min_sep)
            {
                seeds.push(p);
            }
        }
        let case = FuzzCase {
            name: format!("white_noise/run{run}"),
            seeds,
            domain,
            h,
        };
        let (d, (ok, esc, ovf, empty, failed)) = check_diagram(&case);
        assert_success_and_partition(&case, &d);
        println!(
            "[{}] n={} ok={ok} esc={esc} ovf={ovf} empty={empty} failed={failed}",
            case.name,
            case.seeds.len()
        );
    }
}

// ---------------------------------------------------------------------------
// Class (b): perturbed regular grid, perturbation sweep 0 -> 0.49h
// ---------------------------------------------------------------------------

#[test]
fn fuzz_perturbed_grid_sweep() {
    let domain = Vector2::new(2.0, 1.0);
    let (nx, ny) = (40usize, 20usize);
    let (sx, sy) = (domain.x / nx as f64, domain.y / ny as f64);
    for run in 0..fuzz_runs() {
        for (sub, &amp) in [0.0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.49].iter().enumerate() {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed_for(b'b', run, sub));
            let mut seeds = Vec::with_capacity(nx * ny);
            for j in 0..ny {
                for i in 0..nx {
                    // Symmetric jitter of ±amp·spacing per axis; amp=0.49
                    // leaves a 0.02·spacing worst-case gap (still distinct).
                    let jx = (rng.gen::<f64>() * 2.0 - 1.0) * amp * sx;
                    let jy = (rng.gen::<f64>() * 2.0 - 1.0) * amp * sy;
                    seeds.push(Point2::new(
                        ((i as f64 + 0.5) * sx + jx).clamp(0.0, domain.x),
                        ((j as f64 + 0.5) * sy + jy).clamp(0.0, domain.y),
                    ));
                }
            }
            let case = FuzzCase {
                name: format!("perturbed_grid/run{run}/amp{amp}"),
                seeds,
                domain,
                h: sx.min(sy),
            };
            let (d, (ok, esc, ovf, empty, failed)) = check_diagram(&case);
            assert_success_and_partition(&case, &d);
            println!(
                "[{}] n={} ok={ok} esc={esc} ovf={ovf} empty={empty} failed={failed}",
                case.name,
                case.seeds.len()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Class (c): EXACT cocircular lattice
// ---------------------------------------------------------------------------

#[test]
fn fuzz_exact_cocircular_lattice() {
    // Power-of-two spacing: coordinates and their differences are exact in
    // f64, so every interior Voronoi vertex is exactly 4-cocircular and all
    // candidate distances tie exactly — the id tie-break carries the day.
    let s = 0.0625f64;
    for run in 0..fuzz_runs() {
        let (nx, ny) = (16 + 8 * run, 8 + 4 * run);
        let domain = Vector2::new(nx as f64 * s, ny as f64 * s);
        let half = s * 0.5;
        let mut seeds = Vec::with_capacity(nx * ny);
        for j in 0..ny {
            for i in 0..nx {
                seeds.push(Point2::new(half + i as f64 * s, half + j as f64 * s));
            }
        }
        let case = FuzzCase {
            name: format!("cocircular/{nx}x{ny}"),
            seeds,
            domain,
            h: s,
        };
        let (d, (ok, esc, ovf, empty, failed)) = check_diagram(&case);
        assert_success_and_partition(&case, &d);
        // On the exact lattice every cell must be a clean s×s square.
        let quads = d.ring_len.iter().filter(|&&l| l == 4).count();
        assert_eq!(
            quads,
            case.seeds.len(),
            "{}: every lattice cell must be a quad",
            case.name
        );
        println!(
            "[{}] n={} ok={ok} esc={esc} ovf={ovf} empty={empty} failed={failed} quads={quads}",
            case.name,
            case.seeds.len()
        );
    }
}

// ---------------------------------------------------------------------------
// Class (e): knife-edge twins in a Poisson-like set, THROUGH assembly.
//            A twin pair's sub-tolerance face can be Cut for one cell and
//            Redundant for the other; assembly must symmetrize, not panic
//            "non-reciprocal face".
// ---------------------------------------------------------------------------

#[test]
fn fuzz_knife_edge_twins_through_assembly() {
    let domain = Vector2::new(2.0, 1.0);
    let h = 0.04;
    for run in 0..fuzz_runs() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed_for(b'e', run, 0));
        // Poisson-like carrier set (min separation 0.04 — a legal input).
        let min_sep = h;
        let mut seeds: Vec<Point2<f64>> = Vec::new();
        let target = ((domain.x * domain.y) / (h * h) * 0.6) as usize;
        let mut attempts = 0usize;
        while seeds.len() < target && attempts < target * 200 {
            attempts += 1;
            let p = Point2::new(
                0.05 + rng.gen::<f64>() * (domain.x - 0.1),
                0.05 + rng.gen::<f64>() * (domain.y - 0.1),
            );
            if seeds
                .iter()
                .all(|q| (q - p).norm_squared() >= min_sep * min_sep)
            {
                seeds.push(p);
            }
        }
        // 20 knife-edge twins, 5 per gap, all ABOVE the coalescing pitch
        // (1e-6·h = 4e-8) so both twins keep real cells and the diagram
        // reaches assembly: the class where a shared neighbor swallows one
        // twin's plane into the other's (eps-Redundant) and face pairing
        // must recover the topology geometrically.
        // Smallest gap 1e-7: per-axis projection ≥ 1e-7/√2 ≈ 7.1e-8 > the
        // 4e-8 pitch, so the pair is guaranteed to stay un-coalesced at any
        // twin direction (a 5e-8 gap at ~45° coalesces).
        let carriers = seeds.len();
        for (g, gap) in [1e-5, 1e-6, 3e-7, 1e-7].iter().enumerate() {
            for t in 0..5 {
                let p = seeds[(g * 5 + t) % carriers];
                let th = rng.gen::<f64>() * 2.0 * std::f64::consts::PI;
                seeds.push(Point2::new(p.x + gap * th.cos(), p.y + gap * th.sin()));
            }
        }
        let case = FuzzCase {
            name: format!("knife_edge_twins/run{run}"),
            seeds,
            domain,
            h,
        };
        // check_diagram runs assemble_mesh on clean diagrams — the panic this
        // class targets lives there, not in build_diagram.
        let (_, (ok, esc, ovf, empty, failed)) = check_diagram(&case);
        assert_eq!(empty + failed, 0, "{}: twins must not drop cells", case.name);
        println!(
            "[{}] n={} ok={ok} esc={esc} ovf={ovf} empty={empty} failed={failed}",
            case.name,
            case.seeds.len()
        );
    }
}

// ---------------------------------------------------------------------------
// Class (f): exact + sub-pitch duplicates — the coalescing contract.
//            Duplicated seeds must produce STATUSES: lowest-index bin sibling
//            keeps the whole cell, the rest are EmptyCell, and the partition
//            of unity still holds.
// ---------------------------------------------------------------------------

#[test]
fn fuzz_duplicate_seeds_coalesce() {
    let domain = Vector2::new(2.0, 1.0);
    let h = 0.05;
    for run in 0..fuzz_runs() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed_for(b'f', run, 0));
        let min_sep = h;
        let mut seeds: Vec<Point2<f64>> = Vec::new();
        let target = ((domain.x * domain.y) / (h * h) * 0.6) as usize;
        let mut attempts = 0usize;
        while seeds.len() < target && attempts < target * 200 {
            attempts += 1;
            let p = Point2::new(rng.gen::<f64>() * domain.x, rng.gen::<f64>() * domain.y);
            if seeds
                .iter()
                .all(|q| (q - p).norm_squared() >= min_sep * min_sep)
            {
                seeds.push(p);
            }
        }
        let carriers = seeds.len();
        // 8 EXACT duplicates (bit-identical — guaranteed same bin) …
        for t in 0..8 {
            seeds.push(seeds[t % carriers]);
        }
        // … and 8 sub-pitch twins (1e-12 ≪ pitch 5e-8; same bin unless the
        // pair straddles a bin edge, which coalescing deliberately ignores
        // — straddlers stay distinct cells and go through assembly).
        for t in 8..16 {
            let p = seeds[t % carriers];
            seeds.push(Point2::new(p.x + 1e-12, p.y - 1e-12));
        }
        let case = FuzzCase {
            name: format!("duplicates/run{run}"),
            seeds,
            domain,
            h,
        };
        let (d, (ok, esc, ovf, empty, failed)) = check_diagram(&case);
        assert_eq!(failed, 0, "{}: no SecurityRadiusFailed", case.name);
        assert!(
            empty >= 8 && empty <= 16,
            "{}: expected the 8 exact dups (and up to 8 sub-pitch twins) to \
             coalesce, got empty={empty}",
            case.name
        );
        // Coalescing keeps the partition exact: the kept bin sibling's cell
        // absorbs the duplicate's region.
        let bbox = domain.x * domain.y;
        let total: f64 = d.area.iter().sum();
        let rel = ((total - bbox) / bbox).abs();
        assert!(rel < 1e-9, "{}: partition broken ({rel:.3e})", case.name);
        // Every duplicate's slot is a proper status, not garbage.
        for i in 0..d.n {
            if d.status[i] == CellStatus::EmptyCell {
                assert_eq!(d.ring_len[i], 0, "{} cell {i}: empty slot not empty", case.name);
                assert_eq!(d.area[i], 0.0, "{} cell {i}: empty cell with area", case.name);
            }
        }
        println!(
            "[{}] n={} ok={ok} esc={esc} ovf={ovf} empty={empty} failed={failed}",
            case.name,
            case.seeds.len()
        );
    }
}

// ---------------------------------------------------------------------------
// Class (d): Gaussian density cluster + twins + boundary-huggers
//            (violates the smooth-density assumption on purpose; report-only)
// ---------------------------------------------------------------------------

#[test]
fn fuzz_gaussian_cluster_report_only() {
    let domain = Vector2::new(2.0, 1.0);
    let center = Point2::new(1.0, 0.5);
    for run in 0..fuzz_runs() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed_for(b'd', run, 0));
        let sigma = 0.15 / (1.0 + run as f64);
        let n_target = 900usize;
        // Box–Muller; resample until strictly inside the domain.
        let mut gauss = || loop {
            let (u1, u2) = (rng.gen::<f64>().max(1e-300), rng.gen::<f64>());
            let r = (-2.0 * u1.ln()).sqrt();
            let th = 2.0 * std::f64::consts::PI * u2;
            let p = Point2::new(center.x + sigma * r * th.cos(), center.y + sigma * r * th.sin());
            if p.x > 0.0 && p.x < domain.x && p.y > 0.0 && p.y < domain.y {
                return p;
            }
        };
        let mut seeds: Vec<Point2<f64>> = (0..n_target).map(|_| gauss()).collect();
        // Near-duplicate twins at 1e-9 and knife-edge twins at 1e-12.
        for i in 0..12 {
            let p = seeds[i];
            let off = if i % 2 == 0 { 1e-9 } else { 1e-12 };
            seeds.push(Point2::new(
                (p.x + off).min(domain.x),
                (p.y + off).min(domain.y),
            ));
        }
        // Boundary-hugging seeds 1e-9 inside each wall.
        seeds.push(Point2::new(1e-9, 0.4));
        seeds.push(Point2::new(domain.x - 1e-9, 0.6));
        seeds.push(Point2::new(0.9, 1e-9));
        seeds.push(Point2::new(1.1, domain.y - 1e-9));

        let case = FuzzCase {
            name: format!("gaussian/run{run}/sigma{sigma:.3}"),
            seeds,
            domain,
            h: sigma / 3.0,
        };
        // No panic + per-cell invariants + no NaN; the success fraction is
        // reported, not gated — dense cores legitimately escalate hard.
        let (d, (ok, esc, ovf, empty, failed)) = check_diagram(&case);
        let n = case.seeds.len();
        println!(
            "[{}] n={n} ok={ok} esc={esc} ({:.2}%) ovf={ovf} empty={empty} failed={failed}",
            case.name,
            100.0 * esc as f64 / n as f64
        );
        // Even here the partition must hold: all seeds are strictly inside
        // the domain, so any EmptyCell in this class is a coalesced
        // sub-pitch twin — whose region the kept bin sibling absorbs.
        if failed == 0 {
            let total: f64 = d.area.iter().sum();
            let bbox = domain.x * domain.y;
            let rel = ((total - bbox) / bbox).abs();
            println!("[{}] partition rel err {rel:.3e}", case.name);
            assert!(rel < 1e-9, "{}: partition broken ({rel:.3e})", case.name);
        }
    }
}
