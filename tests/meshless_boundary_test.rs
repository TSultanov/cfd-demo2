//! Meshless engine boundary gates (M0.3, review-corrected F1/F3/F6):
//! 1. loop integrity per geometry — closed, fluid-on-left orientation,
//!    tag census by TOTAL LENGTH per `BoundaryType` (review F3: face
//!    counts differ legitimately between seeding protocols);
//! 2. full-domain diagrams on all four GUI geometries — clean statuses,
//!    partition of unity against the *discrete* loop area (shoelace of the
//!    boundary loops, 1e-9) and against the analytic fluid area (the
//!    incumbent's 2% Monte-Carlo tolerance); plus engine == exhaustive
//!    oracle bitwise with boundary kinds, and thread-count determinism;
//! 3. watertight walls — the circle obstacle's Wall edges reconstruct the
//!    chord polygon (length within 1e-9), every polyline vertex is
//!    reproduced in some ring, the step's reflex corner exists, no
//!    phantom-wall edges off the polyline, shielding check green;
//! 4. boundary cells convex and containing/touching their seed; interior
//!    cells strictly containing theirs.
//!
//! Run with:
//!
//! ```sh
//! cargo test --features meshgen --test meshless_boundary_test -- --nocapture
//! ```

#![cfg(feature = "meshgen")]

use cfd2::meshgen::meshless::{
    build_diagram, compute_cell, compute_cell_exhaustive, meshless_seed_points,
    shielding_violations, tag_boundary_type, BoundaryLoop, BoundarySpec, CellStatus, EngineConfig,
    MeshlessDiagram, MeshlessInput, PlaneTag, SeedGrid, SeedKind, MAX_CLIP_VERTS,
};
use cfd2::meshgen::{
    BackwardsStep, ChannelWithObstacle, Geometry, MeshgenTolerances, Nozzle, RectangularChannel,
};
use cfd2::solver::mesh::BoundaryType;
use nalgebra::{Point2, Vector2};

const HMIN: f64 = 0.04;
const HMAX: f64 = 0.08;
const GROWTH: f64 = 1.2;

fn rect() -> (RectangularChannel, Vector2<f64>) {
    (RectangularChannel { length: 3.0, height: 1.0 }, Vector2::new(3.0, 1.0))
}

fn obstacle() -> (ChannelWithObstacle, Vector2<f64>) {
    (
        ChannelWithObstacle {
            length: 3.0,
            height: 1.0,
            obstacle_center: Point2::new(1.0, 0.51),
            obstacle_radius: 0.1,
        },
        Vector2::new(3.0, 1.0),
    )
}

fn backstep() -> (BackwardsStep, Vector2<f64>) {
    (
        BackwardsStep {
            length: 3.5,
            height_inlet: 0.5,
            height_outlet: 1.0,
            step_x: 0.5,
        },
        Vector2::new(3.5, 1.0),
    )
}

fn nozzle() -> (Nozzle, Vector2<f64>) {
    (
        Nozzle {
            length: 3.0,
            height: 1.0,
            throat_height: 0.40,
            throat_frac: 0.40,
            exit_height: 0.80,
        },
        Vector2::new(3.0, 1.0),
    )
}

/// Estimate the fluid area by sampling `is_inside` on a fine grid — the same
/// Monte-Carlo reference (and 2% tolerance) the incumbent validation matrix
/// uses.
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

/// Signed shoelace area of the discrete fluid region: outer (CCW) loops
/// count positive, holes (CW) negative.
fn loops_area(spec: &BoundarySpec) -> f64 {
    spec.loops.iter().map(|lp| lp.signed_area()).sum()
}

fn census_by_length(loops: &[BoundaryLoop]) -> (f64, f64, f64) {
    let (mut inlet, mut outlet, mut wall) = (0.0, 0.0, 0.0);
    for lp in loops {
        let n = lp.pts.len();
        for s in 0..n {
            let len = (lp.pts[(s + 1) % n] - lp.pts[s]).norm();
            match lp.tags[s] {
                BoundaryType::Inlet => inlet += len,
                BoundaryType::Outlet => outlet += len,
                BoundaryType::Wall => wall += len,
                other => panic!("unexpected tag {other:?}"),
            }
        }
    }
    (inlet, outlet, wall)
}

/// Every segment must have fluid on its left and solid on its right — probed
/// with the smooth SDF a quarter-spacing away from the segment midpoint
/// (far enough to clear the chord-vs-arc sagitta on curved walls).
fn assert_fluid_on_left(name: &str, geo: &impl Geometry, loops: &[BoundaryLoop], spacing: f64) {
    let eps = 0.25 * spacing;
    for (l, lp) in loops.iter().enumerate() {
        let n = lp.pts.len();
        for s in 0..n {
            let a = lp.pts[s];
            let b = lp.pts[(s + 1) % n];
            let d = (b - a).normalize();
            let left = Vector2::new(-d.y, d.x);
            let mid = Point2::new(0.5 * (a.x + b.x), 0.5 * (a.y + b.y));
            assert!(
                geo.is_inside(&(mid + left * eps)),
                "{name} loop {l} seg {s}: fluid not on the left at {mid:?}"
            );
            assert!(
                !geo.is_inside(&(mid - left * eps)),
                "{name} loop {l} seg {s}: solid not on the right at {mid:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 1. Loop integrity per geometry
// ---------------------------------------------------------------------------

#[test]
fn loops_are_closed_oriented_and_census_matches_analytic() {
    let spacing = 0.05;

    // Shared structural checks + census per geometry.
    fn structural(name: &str, geo: &(impl Geometry + Sync), domain: Vector2<f64>, spacing: f64) -> Vec<BoundaryLoop> {
        let tol = MeshgenTolerances::from_geometry(spacing, domain);
        let loops = geo.get_boundary_loops(spacing, domain, &tol);
        assert!(!loops.is_empty(), "{name}: no loops");
        for (l, lp) in loops.iter().enumerate() {
            assert_eq!(lp.pts.len(), lp.tags.len(), "{name} loop {l}: one tag per segment");
            assert!(lp.pts.len() >= 3, "{name} loop {l}: degenerate loop");
            let area = lp.signed_area();
            if l == 0 {
                assert!(area > 0.0, "{name}: outer loop must be CCW (area {area})");
            } else {
                assert!(area < 0.0, "{name} loop {l}: holes must be CW (area {area})");
            }
            // Consecutive points distinct, segments no longer than ~spacing.
            let n = lp.pts.len();
            for s in 0..n {
                let len = (lp.pts[(s + 1) % n] - lp.pts[s]).norm();
                assert!(len > 1e-12, "{name} loop {l} seg {s}: zero-length segment");
                assert!(len < 1.5 * spacing, "{name} loop {l} seg {s}: segment too long ({len})");
            }
        }
        assert_fluid_on_left(name, geo, &loops, spacing);
        loops
    }

    {
        let (geo, domain) = rect();
        let loops = structural("rect", &geo, domain, spacing);
        let (inlet, outlet, wall) = census_by_length(&loops);
        assert!((inlet - 1.0).abs() < 1e-9, "rect inlet {inlet}");
        assert!((outlet - 1.0).abs() < 1e-9, "rect outlet {outlet}");
        assert!((wall - 6.0).abs() < 1e-9, "rect wall {wall}");
        println!("[loops/rect] inlet={inlet:.6} outlet={outlet:.6} wall={wall:.6}");
    }
    {
        let (geo, domain) = obstacle();
        let loops = structural("obstacle", &geo, domain, spacing);
        assert_eq!(loops.len(), 2, "obstacle: box + circle");
        // The circle loop is all Wall and its length is the chord perimeter.
        let circle = &loops[1];
        assert!(circle.tags.iter().all(|t| *t == BoundaryType::Wall));
        let chord_perim: f64 = (0..circle.pts.len())
            .map(|s| (circle.pts[(s + 1) % circle.pts.len()] - circle.pts[s]).norm())
            .sum();
        let circ = 2.0 * std::f64::consts::PI * 0.1;
        assert!(
            chord_perim <= circ && chord_perim > 0.99 * circ,
            "chord perimeter {chord_perim} vs circumference {circ}"
        );
        let (inlet, outlet, wall) = census_by_length(&loops);
        assert!((inlet - 1.0).abs() < 1e-9, "obstacle inlet {inlet}");
        assert!((outlet - 1.0).abs() < 1e-9, "obstacle outlet {outlet}");
        assert!((wall - (6.0 + chord_perim)).abs() < 1e-9, "obstacle wall {wall}");
        println!("[loops/obstacle] inlet={inlet:.6} outlet={outlet:.6} wall={wall:.6} chords={chord_perim:.6}");
    }
    {
        let (geo, domain) = backstep();
        let loops = structural("backstep", &geo, domain, spacing);
        let (inlet, outlet, wall) = census_by_length(&loops);
        // Inlet = left edge above the step (height_inlet), Wall = top +
        // bottom-right + the two step faces.
        assert!((inlet - 0.5).abs() < 1e-9, "backstep inlet {inlet}");
        assert!((outlet - 1.0).abs() < 1e-9, "backstep outlet {outlet}");
        assert!((wall - 7.5).abs() < 1e-9, "backstep wall {wall}");
        println!("[loops/backstep] inlet={inlet:.6} outlet={outlet:.6} wall={wall:.6}");
    }
    {
        let (geo, domain) = nozzle();
        let loops = structural("nozzle", &geo, domain, spacing);
        let (inlet, outlet, wall) = census_by_length(&loops);
        // Inlet = full height at x=0, outlet = exit height, wall = flat
        // bottom + top-wall chord length (>= its straight span, < 1.3x the
        // bottom for this gentle profile).
        assert!((inlet - 1.0).abs() < 1e-9, "nozzle inlet {inlet}");
        assert!((outlet - 0.8).abs() < 1e-9, "nozzle outlet {outlet}");
        let top_chords = wall - 3.0;
        assert!(
            top_chords >= 3.0 && top_chords < 3.9,
            "nozzle top-wall chord length {top_chords} implausible"
        );
        println!("[loops/nozzle] inlet={inlet:.6} outlet={outlet:.6} wall={wall:.6}");
    }
}

// ---------------------------------------------------------------------------
// Shared full-domain build
// ---------------------------------------------------------------------------

struct Built {
    seeds: Vec<Point2<f64>>,
    kinds: Vec<SeedKind>,
    spec: BoundarySpec,
    tol: MeshgenTolerances,
    domain: Vector2<f64>,
    diagram: MeshlessDiagram,
}

fn build_case(geo: &(impl Geometry + Sync), domain: Vector2<f64>) -> Built {
    let (seeds, kinds, spec) = meshless_seed_points(geo, HMIN, HMAX, GROWTH, domain);
    let tol = MeshgenTolerances::from_geometry(HMIN, domain);
    let diagram = {
        let input = MeshlessInput {
            seeds: &seeds,
            kinds: &kinds,
            boundary: &spec,
            domain,
            tol: &tol,
            cfg: EngineConfig::default(),
        };
        build_diagram(&input)
    };
    Built { seeds, kinds, spec, tol, domain, diagram }
}

impl Built {
    fn input(&self) -> MeshlessInput<'_> {
        MeshlessInput {
            seeds: &self.seeds,
            kinds: &self.kinds,
            boundary: &self.spec,
            domain: self.domain,
            tol: &self.tol,
            cfg: EngineConfig::default(),
        }
    }

    /// Iterate `(cell, ring vertices, ring tags)` over packed slots.
    fn ring(&self, i: usize) -> (&[[f64; 2]], &[PlaneTag]) {
        let len = self.diagram.ring_len[i] as usize;
        (
            &self.diagram.ring_xy[i * MAX_CLIP_VERTS..i * MAX_CLIP_VERTS + len],
            &self.diagram.ring_plane[i * MAX_CLIP_VERTS..i * MAX_CLIP_VERTS + len],
        )
    }
}

fn assert_clean_statuses(name: &str, b: &Built) {
    let (ok, esc, ovf, empty, failed) = b.diagram.status_counts();
    println!(
        "[statuses/{name}] n={} ok={ok} escalated={esc} ({:.2}%) overflow={ovf} empty={empty} failed={failed}",
        b.diagram.n,
        100.0 * esc as f64 / b.diagram.n as f64
    );
    assert_eq!(ovf, 0, "{name}: RingOverflow cells");
    assert_eq!(empty, 0, "{name}: EmptyCell cells");
    assert_eq!(failed, 0, "{name}: SecurityRadiusFailed cells");
}

// ---------------------------------------------------------------------------
// 2. Full-domain diagrams: statuses + partition + oracle + determinism
// ---------------------------------------------------------------------------

#[test]
fn full_domain_diagrams_partition_the_fluid_area() {
    fn run(name: &str, geo: &(impl Geometry + Sync), domain: Vector2<f64>, tight_rel: f64) {
        let b = build_case(geo, domain);
        assert_clean_statuses(name, &b);

        let total: f64 = b.diagram.area.iter().sum();
        // Tight gate: the engine must tile the DISCRETE fluid region — the
        // shoelace area of the boundary loops — to fp accuracy. This is the
        // watertightness identity (no orphaned slivers, no solid overlap).
        let discrete = loops_area(&b.spec);
        let rel = ((total - discrete) / discrete).abs();
        // Loose gate: the discrete region matches the analytic fluid area
        // within the incumbent's 2% Monte-Carlo tolerance.
        let analytic = fluid_area_estimate(geo, domain, 2000);
        let rel_analytic = ((total - analytic) / analytic).abs();
        println!(
            "[partition/{name}] cells={n} sum={total:.12} discrete={discrete:.12} (rel {rel:.3e}) analytic~{analytic:.6} (rel {rel_analytic:.3e})",
            n = b.diagram.n
        );
        assert!(rel < tight_rel, "{name}: partition vs discrete loops rel {rel:.3e}");
        assert!(rel_analytic < 0.02, "{name}: partition vs analytic rel {rel_analytic:.3e}");
    }

    let (geo, domain) = rect();
    run("rect", &geo, domain, 1e-10);
    let (geo, domain) = obstacle();
    run("obstacle", &geo, domain, 1e-9);
    let (geo, domain) = backstep();
    run("backstep", &geo, domain, 1e-9);
    let (geo, domain) = nozzle();
    run("nozzle", &geo, domain, 1e-9);
}

#[test]
fn boundary_engine_matches_exhaustive_oracle_bitwise() {
    let (geo, domain) = obstacle();
    let b = build_case(&geo, domain);
    let input = b.input();
    let grid = SeedGrid::build(&b.seeds, domain);
    for i in 0..b.seeds.len() {
        let got = compute_cell(&input, &grid, i);
        let want = compute_cell_exhaustive(&input, i);
        assert_eq!(got.len, want.len, "cell {i}: ring length");
        for e in 0..got.len {
            assert_eq!(got.plane[e], want.plane[e], "cell {i} edge {e}: tag");
            assert_eq!(got.xy[e][0].to_bits(), want.xy[e][0].to_bits(), "cell {i} vert {e}: x");
            assert_eq!(got.xy[e][1].to_bits(), want.xy[e][1].to_bits(), "cell {i} vert {e}: y");
        }
        assert_eq!(got.area.to_bits(), want.area.to_bits(), "cell {i}: area");
    }
    println!("[oracle/obstacle-boundary] n={} bit-identical", b.seeds.len());
}

#[test]
fn boundary_diagram_is_byte_identical_across_thread_counts() {
    let (geo, domain) = obstacle();
    let (seeds, kinds, spec) = meshless_seed_points(&geo, HMIN, HMAX, GROWTH, domain);
    let tol = MeshgenTolerances::from_geometry(HMIN, domain);
    let input = MeshlessInput {
        seeds: &seeds,
        kinds: &kinds,
        boundary: &spec,
        domain,
        tol: &tol,
        cfg: EngineConfig::default(),
    };
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
        assert_eq!(reference.status, d.status, "threads={threads}: status");
        assert_eq!(reference.ring_len, d.ring_len, "threads={threads}: ring_len");
        assert_eq!(reference.ring_plane, d.ring_plane, "threads={threads}: ring_plane");
        assert!(
            reference
                .ring_xy
                .iter()
                .zip(&d.ring_xy)
                .all(|(a, b)| a[0].to_bits() == b[0].to_bits() && a[1].to_bits() == b[1].to_bits()),
            "threads={threads}: ring_xy bits"
        );
        assert!(
            reference
                .area
                .iter()
                .zip(&d.area)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "threads={threads}: area bits"
        );
    }
    println!("[determinism/obstacle-boundary] n={} pools 1/2/8 byte-identical", seeds.len());
}

// ---------------------------------------------------------------------------
// 3. Watertight walls + no phantoms + shielding
// ---------------------------------------------------------------------------

#[test]
fn obstacle_wall_edges_reconstruct_the_chord_polygon() {
    let (geo, domain) = obstacle();
    let b = build_case(&geo, domain);
    assert_clean_statuses("obstacle", &b);

    // Global segment id range of the circle loop (loop 1).
    let circle_range = b.spec.seg_offsets[1]..b.spec.seg_offsets[2];
    let circle = &b.spec.loops[1];
    let chord_perim: f64 = (0..circle.pts.len())
        .map(|s| (circle.pts[(s + 1) % circle.pts.len()] - circle.pts[s]).norm())
        .sum();

    // Union of circle-tagged ring edges == the chord polygon, by length.
    let mut wall_len = 0.0;
    for i in 0..b.diagram.n {
        let (xy, tags) = b.ring(i);
        for e in 0..xy.len() {
            if let PlaneTag::Boundary(seg) = tags[e] {
                if circle_range.contains(&(seg as usize)) {
                    let w = (e + 1) % xy.len();
                    let (dx, dy) = (xy[w][0] - xy[e][0], xy[w][1] - xy[e][1]);
                    wall_len += (dx * dx + dy * dy).sqrt();
                }
            }
        }
    }
    let rel = ((wall_len - chord_perim) / chord_perim).abs();
    println!("[watertight/obstacle] wall_len={wall_len:.12} chord_perim={chord_perim:.12} rel={rel:.3e}");
    assert!(rel < 1e-9, "circle wall length rel error {rel:.3e}");

    // Every chord-polygon vertex is reproduced in some ring (the flanking
    // midpoint seeds' mutual bisector passes through it — review F1).
    for (v, p) in circle.pts.iter().enumerate() {
        let mut best = f64::INFINITY;
        for i in 0..b.diagram.n {
            let (xy, _) = b.ring(i);
            for q in xy {
                let (dx, dy) = (q[0] - p.x, q[1] - p.y);
                best = best.min((dx * dx + dy * dy).sqrt());
            }
        }
        assert!(best < 1e-9, "circle vertex {v} not reproduced (nearest ring vertex {best:.3e})");
    }
}

#[test]
fn backstep_reflex_corner_is_reproduced_and_walls_clean() {
    let (geo, domain) = backstep();
    let b = build_case(&geo, domain);
    assert_clean_statuses("backstep", &b);

    let corner = Point2::new(0.5, 0.5); // (step_x, step_h): the reflex corner
    let mut best = f64::INFINITY;
    for i in 0..b.diagram.n {
        let (xy, _) = b.ring(i);
        for q in xy {
            let (dx, dy) = (q[0] - corner.x, q[1] - corner.y);
            best = best.min((dx * dx + dy * dy).sqrt());
        }
    }
    println!("[watertight/backstep] reflex corner nearest ring vertex {best:.3e}");
    assert!(best < 1e-9, "reflex corner not reproduced (nearest {best:.3e})");
}

#[test]
fn no_phantom_walls_and_shielding_holds_everywhere() {
    fn run(name: &str, geo: &(impl Geometry + Sync), domain: Vector2<f64>) {
        let b = build_case(geo, domain);
        let input = b.input();

        // Phantom-wall check: every boundary-tagged ring edge (longer than
        // the eps-sliver scale) must lie ON the boundary polyline — F1's
        // failure mode was wall edges on chord-line EXTENSIONS strictly
        // inside the fluid.
        let mut checked = 0usize;
        for i in 0..b.diagram.n {
            let (xy, tags) = b.ring(i);
            for e in 0..xy.len() {
                let is_boundary = matches!(tags[e], PlaneTag::Boundary(_) | PlaneTag::Box(_));
                if !is_boundary {
                    continue;
                }
                let w = (e + 1) % xy.len();
                let (dx, dy) = (xy[w][0] - xy[e][0], xy[w][1] - xy[e][1]);
                let len = (dx * dx + dy * dy).sqrt();
                if len <= b.tol.boundary_eps {
                    continue; // sub-eps degenerate sliver
                }
                let mid = Point2::new(0.5 * (xy[e][0] + xy[w][0]), 0.5 * (xy[e][1] + xy[w][1]));
                let dist = cfd2::meshgen::meshless::distance_to_loops(mid, &b.spec);
                assert!(
                    dist < b.tol.boundary_eps,
                    "{name} cell {i} edge {e} ({:?}): phantom wall — midpoint {mid:?} is {dist:.3e} off the polyline",
                    tags[e]
                );
                // Tag must resolve to a BoundaryType (parity plumbing).
                assert!(tag_boundary_type(tags[e], &b.spec).is_some());
                checked += 1;
            }
        }

        // Shielding (review F6): no Interior-cell ring vertex on the solid
        // side of the loops.
        let violations = shielding_violations(&input, &b.diagram);
        assert!(
            violations.is_empty(),
            "{name}: {} shielding violations, first: {:?}",
            violations.len(),
            violations.first()
        );
        println!("[phantom+shield/{name}] {checked} boundary edges on-polyline, shielding green");
    }

    let (geo, domain) = rect();
    run("rect", &geo, domain);
    let (geo, domain) = obstacle();
    run("obstacle", &geo, domain);
    let (geo, domain) = backstep();
    run("backstep", &geo, domain);
    let (geo, domain) = nozzle();
    run("nozzle", &geo, domain);
}

// ---------------------------------------------------------------------------
// 4. Convexity + seed containment
// ---------------------------------------------------------------------------

#[test]
fn cells_are_convex_and_contain_their_seeds() {
    fn run(name: &str, geo: &(impl Geometry + Sync), domain: Vector2<f64>) {
        let b = build_case(geo, domain);
        for i in 0..b.diagram.n {
            let (xy, _) = b.ring(i);
            let n = xy.len();
            assert!(n >= 3, "{name} cell {i}: ring too short");
            // CCW convexity: every consecutive edge pair turns left (up to
            // the cross tolerance for near-degenerate cocircular edges).
            for e in 0..n {
                let a = xy[e];
                let m = xy[(e + 1) % n];
                let c = xy[(e + 2) % n];
                let cross = (m[0] - a[0]) * (c[1] - m[1]) - (m[1] - a[1]) * (c[0] - m[0]);
                assert!(
                    cross > -b.tol.cross_eps,
                    "{name} cell {i} vert {e}: concave turn (cross {cross:.3e})"
                );
            }
            // Seed containment: signed distance to every edge line (interior
            // on the left of the CCW ring). Boundary seeds may TOUCH their
            // wall edges (distance 0); interior seeds are strictly inside.
            let s = b.seeds[i];
            let mut min_dist = f64::INFINITY;
            for e in 0..n {
                let a = xy[e];
                let bb = xy[(e + 1) % n];
                let (ex, ey) = (bb[0] - a[0], bb[1] - a[1]);
                let len = (ex * ex + ey * ey).sqrt();
                if len < 1e-14 {
                    continue; // degenerate edge carries no constraint
                }
                let cross = ex * (s.y - a[1]) - ey * (s.x - a[0]);
                min_dist = min_dist.min(cross / len);
            }
            match b.kinds[i] {
                SeedKind::Interior => assert!(
                    min_dist > 1e-9,
                    "{name} cell {i}: interior seed not strictly inside (min dist {min_dist:.3e})"
                ),
                SeedKind::Boundary { .. } => assert!(
                    min_dist > -1e-9,
                    "{name} cell {i}: boundary seed outside its cell (min dist {min_dist:.3e})"
                ),
            }
        }
        println!("[convex+contain/{name}] {} cells OK", b.diagram.n);
    }

    let (geo, domain) = rect();
    run("rect", &geo, domain);
    let (geo, domain) = obstacle();
    run("obstacle", &geo, domain);
    let (geo, domain) = backstep();
    run("backstep", &geo, domain);
    let (geo, domain) = nozzle();
    run("nozzle", &geo, domain);
}

// ---------------------------------------------------------------------------
// Sanity: statuses (referenced here so CellStatus stays imported even if
// gates evolve) — every built case must keep index identity seed i == cell i.
// ---------------------------------------------------------------------------

#[test]
fn seed_index_identity_holds() {
    let (geo, domain) = obstacle();
    let b = build_case(&geo, domain);
    assert_eq!(b.diagram.n, b.seeds.len());
    assert_eq!(b.diagram.status.len(), b.seeds.len());
    assert_eq!(b.kinds.len(), b.seeds.len());
    assert!(b
        .diagram
        .status
        .iter()
        .all(|s| matches!(s, CellStatus::Ok | CellStatus::OkEscalated(_))));
}
