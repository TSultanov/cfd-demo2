//! Meshless engine core gates:
//! 1. `SeedGrid::knn` vs a brute-force O(n²) reference — exact id-set
//!    equality under the (d², id) total order, including tie cases;
//! 2. engine vs the clip-everything oracle — bit-identical rings (the
//!    kNN + security-radius machinery must be exact, not approximate);
//! 3. partition of unity — cell areas sum to the bbox area;
//! 4. byte-identical diagrams across rayon thread counts;
//! 5. adversarial inputs produce statuses, never panics.
//!
//! Run with:
//!
//! ```sh
//! cargo test --features meshgen --test meshless_core_test -- --nocapture
//! ```

#![cfg(feature = "meshgen")]

use cfd2::meshgen::meshless::{
    build_diagram, compute_cell, compute_cell_exhaustive, BoundarySpec, CellOut, CellStatus,
    EngineConfig, MeshlessDiagram, MeshlessInput, SeedGrid, MAX_CLIP_VERTS,
};
use cfd2::meshgen::MeshgenTolerances;
use nalgebra::{Point2, Vector2};
use rand::{Rng, SeedableRng};

const DOMAIN: Vector2<f64> = Vector2::new(2.0, 1.0);

// ---------------------------------------------------------------------------
// Deterministic seed-set generators (the crate's Poisson sampler is private, so
// use a small seeded sampler).
// ---------------------------------------------------------------------------

/// Poisson-like: uniform rejection sampling with a minimum separation.
fn poisson_like(rng_seed: u64, target: usize, min_sep: f64) -> Vec<Point2<f64>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(rng_seed);
    let mut pts: Vec<Point2<f64>> = Vec::new();
    let mut attempts = 0usize;
    while pts.len() < target && attempts < target * 400 {
        attempts += 1;
        let p = Point2::new(rng.gen::<f64>() * DOMAIN.x, rng.gen::<f64>() * DOMAIN.y);
        if pts
            .iter()
            .all(|q| (q - p).norm_squared() >= min_sep * min_sep)
        {
            pts.push(p);
        }
    }
    pts
}

/// Exact regular lattice with power-of-two spacing: coordinates and their
/// differences are exact in f64, so every interior Voronoi vertex is exactly
/// cocircular and kNN distances tie exactly.
fn cocircular_grid() -> Vec<Point2<f64>> {
    let s = 0.0625f64; // 1/16, exact
    let half = 0.03125f64;
    let (nx, ny) = (32usize, 16usize);
    let mut pts = Vec::with_capacity(nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            pts.push(Point2::new(half + i as f64 * s, half + j as f64 * s));
        }
    }
    pts
}

/// Lattice with deterministic sub-spacing jitter.
fn perturbed_grid(rng_seed: u64, amp: f64) -> Vec<Point2<f64>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(rng_seed);
    let (nx, ny) = (40usize, 20usize);
    let (sx, sy) = (DOMAIN.x / nx as f64, DOMAIN.y / ny as f64);
    let mut pts = Vec::with_capacity(nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            let jx = (rng.gen::<f64>() - 0.5) * amp * sx;
            let jy = (rng.gen::<f64>() - 0.5) * amp * sy;
            pts.push(Point2::new(
                ((i as f64 + 0.5) * sx + jx).clamp(0.0, DOMAIN.x),
                ((j as f64 + 0.5) * sy + jy).clamp(0.0, DOMAIN.y),
            ));
        }
    }
    pts
}

/// Center seed + `n_ring` seeds on an exact circle around it: the center
/// cell is a regular `n_ring`-gon, overflowing `MAX_CLIP_VERTS` when
/// `n_ring` exceeds it; the ring seeds are massively distance-tied.
fn star(n_ring: usize, radius: f64) -> Vec<Point2<f64>> {
    let c = Point2::new(1.0, 0.5);
    let mut pts = vec![c];
    for i in 0..n_ring {
        let th = 2.0 * std::f64::consts::PI * i as f64 / n_ring as f64;
        pts.push(Point2::new(c.x + radius * th.cos(), c.y + radius * th.sin()));
    }
    pts
}

fn named_sets() -> Vec<(&'static str, f64, Vec<Point2<f64>>)> {
    // (name, characteristic spacing h for tolerances, seeds)
    vec![
        ("poisson_a", 0.05, poisson_like(1, 500, 0.05)),
        ("poisson_b", 0.03, poisson_like(42, 900, 0.03)),
        ("perturbed_grid", 0.05, perturbed_grid(7, 0.4)),
        ("cocircular_grid", 0.0625, cocircular_grid()),
    ]
}

// ---------------------------------------------------------------------------
// 1. kNN vs brute force
// ---------------------------------------------------------------------------

/// Brute-force k smallest under the (d², id) total order. The d² formula
/// must match the engine's (`dist2`): dx = b.x − a.x with a = query point.
fn brute_knn(seeds: &[Point2<f64>], i: usize, k: usize) -> Vec<(f64, u32)> {
    let p = seeds[i];
    let mut all: Vec<(f64, u32)> = seeds
        .iter()
        .enumerate()
        .filter(|&(j, _)| j != i)
        .map(|(j, q)| {
            let dx = q.x - p.x;
            let dy = q.y - p.y;
            (dx * dx + dy * dy, j as u32)
        })
        .collect();
    all.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    all.truncate(k);
    all
}

#[test]
fn knn_matches_brute_force_exactly() {
    let mut sets = named_sets();
    // Duplicated-distance stress: exact cocircular ring around one seed.
    sets.push(("star_ties", 0.02, star(24, 0.2)));
    for (name, _h, seeds) in &sets {
        let grid = SeedGrid::build(seeds, DOMAIN);
        let n = seeds.len();
        for &k in &[1usize, 4, 16, 64, n - 1] {
            let mut out = Vec::new();
            for i in 0..n {
                grid.knn(seeds, i as u32, k, &mut out);
                let want = brute_knn(seeds, i, k);
                assert_eq!(
                    out.len(),
                    want.len(),
                    "{name} k={k} i={i}: neighbor count mismatch"
                );
                for (slot, (got, exp)) in out.iter().zip(&want).enumerate() {
                    assert!(
                        got.0.to_bits() == exp.0.to_bits() && got.1 == exp.1,
                        "{name} k={k} i={i} slot={slot}: got ({}, {}), want ({}, {})",
                        got.0,
                        got.1,
                        exp.0,
                        exp.1
                    );
                }
            }
        }
        println!("[knn/{name}] OK n={n}");
    }
}

// ---------------------------------------------------------------------------
// 2. Engine vs exhaustive oracle (bitwise)
// ---------------------------------------------------------------------------

fn assert_cells_bit_identical(name: &str, i: usize, got: &CellOut, want: &CellOut) {
    assert_eq!(got.len, want.len, "{name} cell {i}: ring length");
    for e in 0..got.len {
        assert_eq!(
            got.plane[e], want.plane[e],
            "{name} cell {i} edge {e}: plane tag"
        );
        assert_eq!(
            got.xy[e][0].to_bits(),
            want.xy[e][0].to_bits(),
            "{name} cell {i} vert {e}: x bits"
        );
        assert_eq!(
            got.xy[e][1].to_bits(),
            want.xy[e][1].to_bits(),
            "{name} cell {i} vert {e}: y bits"
        );
    }
    assert_eq!(got.area.to_bits(), want.area.to_bits(), "{name} cell {i}: area bits");
    assert_eq!(
        got.centroid[0].to_bits(),
        want.centroid[0].to_bits(),
        "{name} cell {i}: centroid x bits"
    );
    assert_eq!(
        got.centroid[1].to_bits(),
        want.centroid[1].to_bits(),
        "{name} cell {i}: centroid y bits"
    );
    match (&got.spill, &want.spill) {
        (None, None) => {}
        (Some(a), Some(b)) => {
            assert_eq!(a.len(), b.len(), "{name} cell {i}: spill length");
            for (e, (va, vb)) in a.iter().zip(b).enumerate() {
                assert_eq!(va.1, vb.1, "{name} cell {i} spill edge {e}: tag");
                assert_eq!(
                    va.0[0].to_bits(),
                    vb.0[0].to_bits(),
                    "{name} cell {i} spill vert {e}: x bits"
                );
                assert_eq!(
                    va.0[1].to_bits(),
                    vb.0[1].to_bits(),
                    "{name} cell {i} spill vert {e}: y bits"
                );
            }
        }
        _ => panic!("{name} cell {i}: spill presence mismatch"),
    }
}

#[test]
fn engine_matches_exhaustive_oracle_bitwise() {
    for (name, h, seeds) in &named_sets() {
        let tol = MeshgenTolerances::from_geometry(*h, DOMAIN);
        let boundary = BoundarySpec::empty();
        let input =
            MeshlessInput::interior_only(seeds, &boundary, DOMAIN, &tol, EngineConfig::default());
        let grid = SeedGrid::build(seeds, DOMAIN);
        let mut escalated = 0usize;
        for i in 0..seeds.len() {
            let got = compute_cell(&input, &grid, i);
            let want = compute_cell_exhaustive(&input, i);
            match got.status {
                CellStatus::Ok => {}
                CellStatus::OkEscalated(_) => escalated += 1,
                s => panic!("{name} cell {i}: unexpected status {s:?}"),
            }
            assert_cells_bit_identical(name, i, &got, &want);
        }
        println!(
            "[oracle/{name}] OK n={} escalated={escalated} ({:.2}%)",
            seeds.len(),
            100.0 * escalated as f64 / seeds.len() as f64
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Partition of unity
// ---------------------------------------------------------------------------

fn build(seeds: &[Point2<f64>], h: f64) -> MeshlessDiagram {
    let tol = MeshgenTolerances::from_geometry(h, DOMAIN);
    let boundary = BoundarySpec::empty();
    let input =
        MeshlessInput::interior_only(seeds, &boundary, DOMAIN, &tol, EngineConfig::default());
    build_diagram(&input)
}

#[test]
fn cell_areas_partition_the_bbox() {
    let bbox = DOMAIN.x * DOMAIN.y;
    let mut sets = named_sets();
    sets.push(("star_overflow", 0.02, star(48, 0.2)));
    for (name, h, seeds) in &sets {
        let d = build(seeds, *h);
        let total: f64 = d.area.iter().sum();
        let rel = ((total - bbox) / bbox).abs();
        println!("[partition/{name}] sum={total:.15} bbox={bbox} rel={rel:.3e}");
        assert!(
            rel < 1e-12,
            "{name}: cell areas sum to {total}, bbox {bbox} (rel {rel:.3e})"
        );
    }
}

// ---------------------------------------------------------------------------
// 4. Determinism across thread counts
// ---------------------------------------------------------------------------

fn assert_diagrams_byte_equal(a: &MeshlessDiagram, b: &MeshlessDiagram, label: &str) {
    assert_eq!(a.n, b.n, "{label}: n");
    assert_eq!(a.status, b.status, "{label}: status");
    assert_eq!(a.ring_len, b.ring_len, "{label}: ring_len");
    assert_eq!(a.ring_plane, b.ring_plane, "{label}: ring_plane");
    assert!(
        a.ring_xy
            .iter()
            .zip(&b.ring_xy)
            .all(|(x, y)| x[0].to_bits() == y[0].to_bits() && x[1].to_bits() == y[1].to_bits()),
        "{label}: ring_xy bits"
    );
    assert!(
        a.centroid
            .iter()
            .zip(&b.centroid)
            .all(|(x, y)| x[0].to_bits() == y[0].to_bits() && x[1].to_bits() == y[1].to_bits()),
        "{label}: centroid bits"
    );
    assert!(
        a.area
            .iter()
            .zip(&b.area)
            .all(|(x, y)| x.to_bits() == y.to_bits()),
        "{label}: area bits"
    );
    assert_eq!(a.overflow.len(), b.overflow.len(), "{label}: overflow count");
    for ((ia, ra), (ib, rb)) in a.overflow.iter().zip(&b.overflow) {
        assert_eq!(ia, ib, "{label}: overflow cell id");
        assert_eq!(ra.len(), rb.len(), "{label}: overflow ring len");
        for (va, vb) in ra.iter().zip(rb) {
            assert_eq!(va.1, vb.1, "{label}: overflow tag");
            assert!(
                va.0[0].to_bits() == vb.0[0].to_bits() && va.0[1].to_bits() == vb.0[1].to_bits(),
                "{label}: overflow vert bits"
            );
        }
    }
}

#[test]
fn diagram_is_byte_identical_across_thread_counts() {
    // Include an overflow cell so the serial spill pass is covered too.
    let mut seeds = poisson_like(11, 800, 0.03);
    seeds.retain(|p| (p - Point2::new(1.0, 0.5)).norm() > 0.35);
    seeds.extend(star(48, 0.2));
    let h = 0.02;

    let tol = MeshgenTolerances::from_geometry(h, DOMAIN);
    let boundary = BoundarySpec::empty();
    let input =
        MeshlessInput::interior_only(&seeds, &boundary, DOMAIN, &tol, EngineConfig::default());

    let reference = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| build_diagram(&input));
    for threads in [2usize, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let d = pool.install(|| build_diagram(&input));
        assert_diagrams_byte_equal(&reference, &d, &format!("threads={threads}"));
    }
    let (ok, esc, ovf, empty, failed) = reference.status_counts();
    println!("[determinism] n={} ok={ok} escalated={esc} overflow={ovf} empty={empty} failed={failed}", seeds.len());
}

// ---------------------------------------------------------------------------
// 5. Statuses on adversarial inputs; never panics
// ---------------------------------------------------------------------------

#[test]
fn poisson_sets_have_no_overflow_and_all_ok() {
    for (name, h, seeds) in &named_sets() {
        let d = build(seeds, *h);
        let (ok, esc, ovf, empty, failed) = d.status_counts();
        println!("[statuses/{name}] ok={ok} escalated={esc} overflow={ovf} empty={empty} failed={failed}");
        assert_eq!(ovf, 0, "{name}: RingOverflow on a Poisson-like set");
        assert_eq!(empty, 0, "{name}: EmptyCell on a Poisson-like set");
        assert_eq!(failed, 0, "{name}: SecurityRadiusFailed");
        assert!(d.overflow.is_empty(), "{name}: unexpected overflow spill");
    }
}

#[test]
fn near_coincident_seeds_never_panic() {
    let mut seeds = poisson_like(3, 300, 0.04);
    // Attach a near-twin (1e-9 apart) to 25 seeds — knife-edge bisectors.
    for i in 0..25 {
        let p = seeds[i];
        seeds.push(Point2::new(
            (p.x + 1e-9).min(DOMAIN.x),
            (p.y + 1e-9).min(DOMAIN.y),
        ));
    }
    let d = build(&seeds, 0.04);
    assert!(d.area.iter().all(|a| a.is_finite() && *a >= 0.0));
    assert!(d
        .centroid
        .iter()
        .all(|c| c[0].is_finite() && c[1].is_finite()));
    let total: f64 = d.area.iter().sum();
    let bbox = DOMAIN.x * DOMAIN.y;
    assert!(
        ((total - bbox) / bbox).abs() < 1e-9,
        "near-coincident set broke the partition: {total} vs {bbox}"
    );
}

#[test]
fn cocircular_lattice_yields_clean_statuses() {
    let seeds = cocircular_grid();
    let d = build(&seeds, 0.0625);
    let (_, _, ovf, empty, failed) = d.status_counts();
    assert_eq!(ovf + empty + failed, 0, "cocircular lattice must stay Ok");
    // Interior cells of an exact lattice are squares: 4 ring vertices.
    let four = d.ring_len.iter().filter(|&&l| l == 4).count();
    assert!(
        four > seeds.len() / 2,
        "expected mostly quad cells on the lattice, got {four}/{}",
        seeds.len()
    );
}

#[test]
fn oversized_cell_overflows_with_spill_matching_oracle() {
    let seeds = star(48, 0.2);
    let h = 0.02;
    let d = build(&seeds, h);

    assert_eq!(d.status[0], CellStatus::RingOverflow, "center cell status");
    assert_eq!(d.ring_len[0], 0, "overflow slot must stay empty");
    assert_eq!(d.overflow.len(), 1, "exactly one spill expected");
    let (id, ring) = &d.overflow[0];
    assert_eq!(*id, 0);
    assert_eq!(ring.len(), 48, "center cell is a regular 48-gon");
    assert!(ring.len() > MAX_CLIP_VERTS);

    // The spill must be bit-identical to the exhaustive oracle's ring.
    let tol = MeshgenTolerances::from_geometry(h, DOMAIN);
    let boundary = BoundarySpec::empty();
    let input =
        MeshlessInput::interior_only(&seeds, &boundary, DOMAIN, &tol, EngineConfig::default());
    let want = compute_cell_exhaustive(&input, 0);
    assert_eq!(want.status, CellStatus::RingOverflow);
    let want_ring = want.spill.as_ref().expect("oracle spill");
    assert_eq!(ring.len(), want_ring.len());
    for (got, exp) in ring.iter().zip(want_ring) {
        assert_eq!(got.1, exp.1);
        assert_eq!(got.0[0].to_bits(), exp.0[0].to_bits());
        assert_eq!(got.0[1].to_bits(), exp.0[1].to_bits());
    }
    assert_eq!(d.area[0].to_bits(), want.area.to_bits());

    // Ring seeds themselves stay ordinary cells.
    assert!(d.status[1..]
        .iter()
        .all(|s| matches!(s, CellStatus::Ok | CellStatus::OkEscalated(_))));
}
