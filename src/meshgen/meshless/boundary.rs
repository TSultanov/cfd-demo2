//! Boundary loops + boundary seeding for the meshless engine (M0.3, design
//! §1/§4 as amended by review F1/F3/F5/F6).
//!
//! The discrete boundary is a set of closed polyline loops walked with the
//! **fluid on the left** (outer loops CCW, embedded holes CW). Each segment
//! carries a `BoundaryType` — `classify_boundary` on the segment midpoint
//! with the `Wall` fallback for embedded contours, exact parity with
//! `close_untagged_boundary_faces`. Boundary-kind seeds clip the *lines* of
//! their own segments (distance 0, before any bisector), so adjacent
//! boundary cells cut the same global chord line and the walls assemble
//! watertight.
//!
//! Seeding protocol (review F1 — the vertex-seeded blocker fix): boundary
//! seeds are generated FROM the loops, not from `get_boundary_points`:
//!
//! - a polyline vertex whose fluid angle is ≤ π (straight-wall vertices, box
//!   corners, the step's outer corners) carries a **vertex seed** — its cell,
//!   clipped by the two own-segment lines through the seed, is exactly the
//!   true (convex) Voronoi cell, independent of neighbor spacing;
//! - a vertex whose fluid angle is > π (every obstacle-circle vertex, the
//!   step's reflex corner, concave nozzle-wall kinks) gets **no vertex
//!   seed** — its true cell would be non-convex (phantom walls + orphaned
//!   slivers, review F1). Instead two **guard seeds** go onto the two
//!   adjacent segments at the *same* distance `t = ½·min(len_prev,
//!   len_next)` from the vertex. Equidistance is load-bearing: the guards'
//!   mutual bisector then passes through the vertex, which both cells
//!   reproduce exactly as bisector ∩ own-line — watertight, both convex.
//!   On uniformly subdivided curved loops the guards emitted onto a shared
//!   segment coincide at its midpoint (review F1's "midpoint seeding");
//!   quantized dedup collapses them to one seed.
//!
//! Shielding (review F6): interior cells never need boundary planes because
//! the boundary-seed layer is closer to every wall-strip point than any
//! interior seed. `shielding_violations` checks that claim on a finished
//! diagram against the *loops* (distance to the chord polyline, not the
//! smooth SDF — the discrete boundary is the polyline).

use std::collections::HashMap;

use super::super::delaunay::{generate_poisson_points, morton_order};
use super::super::geometry::Geometry;
use super::super::tolerances::MeshgenTolerances;
use super::{CellStatus, MeshlessDiagram, MeshlessInput, PlaneTag, MAX_CLIP_VERTS};
use crate::solver::mesh::BoundaryType;
use nalgebra::{Point2, Vector2};

/// Global segment index into `BoundarySpec` (loop-major, see `seg_offsets`).
pub type SegId = u32;

/// Ordered, closed boundary polyline with per-segment BC tags.
/// Segment `s` runs `pts[s] -> pts[(s+1) % pts.len()]`; the fluid is on the
/// left of that direction.
#[derive(Clone, Debug)]
pub struct BoundaryLoop {
    pub pts: Vec<Point2<f64>>,
    /// `BoundaryType` per segment (parity with `classify_boundary`, `Wall`
    /// fallback for embedded contours).
    pub tags: Vec<BoundaryType>,
}

impl BoundaryLoop {
    /// Tag every segment of a closed point walk: `classify_boundary` on the
    /// segment midpoint, `Wall` for anything off the domain box — the same
    /// policy `close_untagged_boundary_faces` applies to open faces.
    pub fn from_points(
        pts: Vec<Point2<f64>>,
        domain: Vector2<f64>,
        tol: &MeshgenTolerances,
    ) -> Self {
        assert!(pts.len() >= 3, "a boundary loop needs at least 3 points");
        let n = pts.len();
        // Degenerate segments would produce NaN guard seeds downstream
        // (`t / l` with `l = 0` in `boundary_seeds`) — reject them here,
        // where the invariant belongs, instead of in a test (review F-2).
        for s in 0..n {
            let len = (pts[(s + 1) % n] - pts[s]).norm();
            assert!(
                len > tol.edge_len_eps,
                "boundary loop segment {s} is degenerate (length {len:.3e} <= edge_len_eps)"
            );
        }
        let tags = (0..n)
            .map(|s| {
                let a = pts[s];
                let b = pts[(s + 1) % n];
                tol.classify_boundary(0.5 * (a.x + b.x), 0.5 * (a.y + b.y), domain.x, domain.y)
                    .unwrap_or(BoundaryType::Wall)
            })
            .collect();
        Self { pts, tags }
    }

    /// Endpoints of segment `s` (local index).
    #[inline]
    pub fn segment(&self, s: usize) -> (Point2<f64>, Point2<f64>) {
        (self.pts[s], self.pts[(s + 1) % self.pts.len()])
    }

    /// Shoelace signed area: positive for CCW (outer) loops, negative for CW
    /// (hole) loops — with fluid-on-left, the sum over all loops is the area
    /// of the discrete fluid region.
    pub fn signed_area(&self) -> f64 {
        let n = self.pts.len();
        let mut acc = 0.0;
        for s in 0..n {
            let a = self.pts[s];
            let b = self.pts[(s + 1) % n];
            acc += a.x * b.y - b.x * a.y;
        }
        0.5 * acc
    }
}

/// All boundary loops of a geometry plus the flattened segment table that
/// gives every segment a stable global `SegId`.
#[derive(Clone, Debug)]
pub struct BoundarySpec {
    pub loops: Vec<BoundaryLoop>,
    /// `seg_offsets[l]` = global id of loop `l`'s first segment;
    /// `seg_offsets.last()` = total segment count. Always `len() + 1` entries.
    pub seg_offsets: Vec<usize>,
}

impl BoundarySpec {
    /// No embedded boundaries: cells clip against the domain bbox only
    /// (the pure-bbox configuration used by the engine-vs-oracle gates).
    pub fn empty() -> Self {
        Self {
            loops: Vec::new(),
            seg_offsets: vec![0],
        }
    }

    /// Build the flattened segment table (closed loop of `n` points = `n`
    /// segments).
    pub fn from_loops(loops: Vec<BoundaryLoop>) -> Self {
        let mut seg_offsets = Vec::with_capacity(loops.len() + 1);
        let mut total = 0usize;
        seg_offsets.push(0);
        for lp in &loops {
            assert_eq!(lp.pts.len(), lp.tags.len(), "one tag per segment");
            total += lp.pts.len();
            seg_offsets.push(total);
        }
        Self { loops, seg_offsets }
    }

    pub fn num_segments(&self) -> usize {
        *self.seg_offsets.last().unwrap()
    }

    /// `(loop index, local segment index)` of a global segment id.
    #[inline]
    pub fn locate(&self, seg: SegId) -> (usize, usize) {
        let l = self.seg_offsets.partition_point(|&o| o <= seg as usize) - 1;
        (l, seg as usize - self.seg_offsets[l])
    }

    /// Endpoints of a global segment.
    #[inline]
    pub fn segment_points(&self, seg: SegId) -> (Point2<f64>, Point2<f64>) {
        let (l, s) = self.locate(seg);
        self.loops[l].segment(s)
    }

    /// `BoundaryType` of a global segment.
    #[inline]
    pub fn segment_tag(&self, seg: SegId) -> BoundaryType {
        let (l, s) = self.locate(seg);
        self.loops[l].tags[s]
    }
}

/// Per-seed kind. Boundary seeds sit on the boundary polyline and know their
/// adjacent segments (`seg_prev` ends at the seed, `seg_next` starts at it;
/// the same id twice for a mid-segment — guard/midpoint — seed).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeedKind {
    Interior,
    Boundary { seg_prev: SegId, seg_next: SegId },
}

/// `BoundaryType` a ring-edge plane tag resolves to: segment tags come from
/// the spec, `Box` sides map exactly like `classify_boundary` on the domain
/// bbox (0=left Inlet, 1=right Outlet, 2=bottom/3=top Wall), bisectors are
/// interior faces.
pub fn tag_boundary_type(tag: PlaneTag, spec: &BoundarySpec) -> Option<BoundaryType> {
    match tag {
        PlaneTag::Bisector(_) => None,
        PlaneTag::Boundary(seg) => Some(spec.segment_tag(seg)),
        PlaneTag::Box(0) => Some(BoundaryType::Inlet),
        PlaneTag::Box(1) => Some(BoundaryType::Outlet),
        PlaneTag::Box(2) | PlaneTag::Box(3) => Some(BoundaryType::Wall),
        PlaneTag::Box(side) => unreachable!("invalid box side {side}"),
    }
}

// ---------------------------------------------------------------------------
// Loop construction helpers (used by the `Geometry::get_boundary_loops`
// implementations in geometry.rs).
// ---------------------------------------------------------------------------

/// Closed straight-sided loop through `corners` (walked in the given order —
/// the caller is responsible for fluid-on-left orientation), each side
/// subdivided into equal segments no longer than `spacing`.
pub fn polyline_loop(
    corners: &[Point2<f64>],
    spacing: f64,
    domain: Vector2<f64>,
    tol: &MeshgenTolerances,
) -> BoundaryLoop {
    let m = corners.len();
    let mut pts = Vec::new();
    for c in 0..m {
        let a = corners[c];
        let b = corners[(c + 1) % m];
        let n = (((b - a).norm() / spacing).ceil() as usize).max(1);
        for i in 0..n {
            let t = i as f64 / n as f64;
            pts.push(a + (b - a) * t);
        }
    }
    BoundaryLoop::from_points(pts, domain, tol)
}

/// Closed chord polygon of a circle, walked **clockwise** so the fluid
/// (outside the obstacle) is on the left. Uniform angles make every chord the
/// same length, which is what collapses the reflex-vertex guard seeds into
/// exact chord midpoints (see the module docs).
pub fn circle_loop(
    center: Point2<f64>,
    radius: f64,
    spacing: f64,
    domain: Vector2<f64>,
    tol: &MeshgenTolerances,
) -> BoundaryLoop {
    let circumference = 2.0 * std::f64::consts::PI * radius;
    let n = ((circumference / spacing).ceil() as usize).max(3);
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let theta = -2.0 * std::f64::consts::PI * i as f64 / n as f64;
        pts.push(Point2::new(
            center.x + radius * theta.cos(),
            center.y + radius * theta.sin(),
        ));
    }
    BoundaryLoop::from_points(pts, domain, tol)
}

// ---------------------------------------------------------------------------
// Boundary seeding (review F1)
// ---------------------------------------------------------------------------

/// Convexity threshold on the *normalized* cross product of consecutive
/// segment directions: |sin(turn)| below this is treated as straight. Far
/// above f64 noise on subdivided collinear sides (~1e-16), far below any
/// real polyline turn (a circle chord fan at h=r/3 turns by sin ≈ 0.33).
const TURN_SIN_EPS: f64 = 1e-12;

/// Generate the boundary seeds for a loop set: vertex seeds where the fluid
/// angle is ≤ π, equidistant guard-seed pairs around every reflex vertex
/// (fluid angle > π). Seeds are deduplicated on the `quantize_point` grid
/// (first occurrence wins; the map is lookup-only, so output order is the
/// deterministic loop-walk order).
///
/// Guard collapse (review): the two guards emitted onto a SEGMENT shared by
/// consecutive reflex vertices (the uniform-chord case: both land at the
/// chord midpoint) are computed from opposite ends and agree only to last
/// ulps — relying on the quantization bin to dedup them risks a bin-edge
/// straddle leaving twin seeds ~1 ulp apart. When the two flanking guards
/// of a segment land within `2·edge_len_eps` of each other, BOTH vertices
/// emit the midpoint of the pair instead — bit-identical from either side
/// (the two addends swap, fp `+` is commutative), so the dedup is exact by
/// construction.
pub fn boundary_seeds(
    spec: &BoundarySpec,
    tol: &MeshgenTolerances,
) -> (Vec<Point2<f64>>, Vec<SeedKind>) {
    let mut pts: Vec<Point2<f64>> = Vec::new();
    let mut kinds: Vec<SeedKind> = Vec::new();
    let mut seen: HashMap<(i64, i64), ()> = HashMap::new();
    let mut push = |p: Point2<f64>, kind: SeedKind| {
        let key = tol.quantize_point(p.x, p.y);
        if let std::collections::hash_map::Entry::Vacant(e) = seen.entry(key) {
            e.insert(());
            pts.push(p);
            kinds.push(kind);
        }
    };
    let collapse_eps = 2.0 * tol.edge_len_eps;

    for (l, lp) in spec.loops.iter().enumerate() {
        let n = lp.pts.len();
        let base = spec.seg_offsets[l];
        // Per-vertex classification pass: reflex flag + the equidistant
        // guard offset t (0 for convex vertices, unused).
        let mut reflex = vec![false; n];
        let mut t_of = vec![0.0f64; n];
        for v in 0..n {
            let a = lp.pts[(v + n - 1) % n];
            let b = lp.pts[v];
            let c = lp.pts[(v + 1) % n];
            let u1 = b - a;
            let u2 = c - b;
            let l1 = u1.norm();
            let l2 = u2.norm();
            // Fluid on the left ⇒ fluid angle = π − turn: a left turn
            // (cross > 0) is a convex fluid corner, a right turn is reflex.
            let cross = u1.x * u2.y - u1.y * u2.x;
            if cross < -TURN_SIN_EPS * l1 * l2 {
                reflex[v] = true;
                // Reflex: two guards at the SAME distance t along each wall.
                t_of[v] = 0.5 * l1.min(l2);
            }
        }
        for v in 0..n {
            let prev = (v + n - 1) % n;
            let next = (v + 1) % n;
            let a = lp.pts[prev];
            let b = lp.pts[v];
            let c = lp.pts[next];
            let u1 = b - a;
            let u2 = c - b;
            let l1 = u1.norm();
            let l2 = u2.norm();
            let seg_prev = (base + prev) as SegId;
            let seg_next = (base + v) as SegId;
            if !reflex[v] {
                push(b, SeedKind::Boundary { seg_prev, seg_next });
                continue;
            }
            // Guard on seg_prev; its potential partner is the previous
            // vertex's guard onto the same segment (emitted from `a`).
            let mut g1 = b - u1 * (t_of[v] / l1);
            if reflex[prev] {
                let partner = a + u1 * (t_of[prev] / l1);
                if (g1 - partner).norm() <= collapse_eps {
                    g1 = Point2::new(0.5 * (g1.x + partner.x), 0.5 * (g1.y + partner.y));
                }
            }
            push(
                g1,
                SeedKind::Boundary {
                    seg_prev,
                    seg_next: seg_prev,
                },
            );
            // Guard on seg_next; partner = the next vertex's guard onto it.
            let mut g2 = b + u2 * (t_of[v] / l2);
            if reflex[next] {
                let partner = c - u2 * (t_of[next] / l2);
                if (g2 - partner).norm() <= collapse_eps {
                    g2 = Point2::new(0.5 * (g2.x + partner.x), 0.5 * (g2.y + partner.y));
                }
            }
            push(
                g2,
                SeedKind::Boundary {
                    seg_prev: seg_next,
                    seg_next,
                },
            );
        }
    }
    (pts, kinds)
}

/// Loop-derived seeding entry for the meshless generators: boundary seeds
/// from the loops (fixed, review F1) + interior Poisson fill respecting the
/// min-separation from them (the same sampler and RNG seed `0x5EED_CFD2` as
/// the incumbent generators), Morton-sorted with `kinds` co-permuted
/// (review F4). Returns the `BoundarySpec` too — the `SegId`s inside `kinds`
/// index into it, so the pair must stay together.
pub fn meshless_seed_points(
    geo: &(impl Geometry + Sync),
    min_cell_size: f64,
    max_cell_size: f64,
    growth_rate: f64,
    domain: Vector2<f64>,
) -> (Vec<Point2<f64>>, Vec<SeedKind>, BoundarySpec) {
    let tol = MeshgenTolerances::from_geometry(min_cell_size, domain);
    let spec = BoundarySpec::from_loops(geo.get_boundary_loops(min_cell_size, domain, &tol));
    let (mut pts, mut kinds) = boundary_seeds(&spec, &tol);
    let interior = generate_poisson_points(
        &pts,
        geo,
        min_cell_size,
        max_cell_size,
        growth_rate,
        domain,
    );
    kinds.reserve(interior.len());
    for p in interior {
        pts.push(p);
        kinds.push(SeedKind::Interior);
    }
    let order = morton_order(&pts);
    let pts = order.iter().map(|&j| pts[j]).collect();
    let kinds = order.iter().map(|&j| kinds[j]).collect();
    (pts, kinds, spec)
}

// ---------------------------------------------------------------------------
// Shielding / watertightness instruments (review F6)
// ---------------------------------------------------------------------------

/// Unsigned distance from `p` to the nearest boundary-loop *segment* (the
/// chord polyline — NOT the smooth SDF; review F6).
pub fn distance_to_loops(p: Point2<f64>, spec: &BoundarySpec) -> f64 {
    let mut best = f64::INFINITY;
    for lp in &spec.loops {
        let n = lp.pts.len();
        for s in 0..n {
            let a = lp.pts[s];
            let b = lp.pts[(s + 1) % n];
            let d = b - a;
            let len2 = d.norm_squared();
            let t = if len2 > 0.0 {
                ((p - a).dot(&d) / len2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            best = best.min((p - (a + d * t)).norm());
        }
    }
    best
}

/// Even-odd point-in-fluid test against the discrete loops: inside the outer
/// (CCW) loop and outside every hole (CW) loop ⇔ an odd crossing count over
/// all loops together. Knife-edge points (on the polyline) are undefined —
/// callers must pair this with a distance threshold.
pub fn point_in_fluid(p: Point2<f64>, spec: &BoundarySpec) -> bool {
    let mut inside = false;
    for lp in &spec.loops {
        let n = lp.pts.len();
        for s in 0..n {
            let a = lp.pts[s];
            let b = lp.pts[(s + 1) % n];
            if (a.y > p.y) != (b.y > p.y) {
                let x_int = a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y);
                if p.x < x_int {
                    inside = !inside;
                }
            }
        }
    }
    inside
}

/// Shielding safety check (review F6): ring vertices of *Interior* cells
/// must never sit on the solid side of the boundary loops by more than
/// `tol.boundary_eps` — if one does, an interior cell reached a wall past
/// the boundary-seed layer and the shielding assumption is broken. Returns
/// `(cell, ring vertex, distance)` violations; empty = shielded.
pub fn shielding_violations(input: &MeshlessInput, d: &MeshlessDiagram) -> Vec<(usize, usize, f64)> {
    let mut bad = Vec::new();
    let check = |cell: usize, k: usize, xy: [f64; 2], bad: &mut Vec<(usize, usize, f64)>| {
        let p = Point2::new(xy[0], xy[1]);
        let dist = distance_to_loops(p, input.boundary);
        if dist > input.tol.boundary_eps && !point_in_fluid(p, input.boundary) {
            bad.push((cell, k, dist));
        }
    };
    for i in 0..d.n {
        if input.kind(i) != SeedKind::Interior {
            continue;
        }
        match d.status[i] {
            CellStatus::EmptyCell | CellStatus::SecurityRadiusFailed => continue,
            CellStatus::RingOverflow => {
                if let Some((_, ring)) = d.overflow.iter().find(|(c, _)| *c as usize == i) {
                    for (k, (xy, _)) in ring.iter().enumerate() {
                        check(i, k, *xy, &mut bad);
                    }
                }
            }
            CellStatus::Ok | CellStatus::OkEscalated(_) => {
                for k in 0..d.ring_len[i] as usize {
                    check(i, k, d.ring_xy[i * MAX_CLIP_VERTS + k], &mut bad);
                }
            }
        }
    }
    bad
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_box_spec(spacing: f64) -> (BoundarySpec, MeshgenTolerances) {
        let domain = Vector2::new(1.0, 1.0);
        let tol = MeshgenTolerances::from_geometry(spacing, domain);
        let corners = [
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let spec = BoundarySpec::from_loops(vec![polyline_loop(&corners, spacing, domain, &tol)]);
        (spec, tol)
    }

    #[test]
    fn seg_offsets_and_lookup_are_consistent() {
        let (spec, _) = unit_box_spec(0.25);
        assert_eq!(spec.seg_offsets.len(), 2);
        assert_eq!(spec.num_segments(), 16);
        // Segment 0 starts at the first loop point; the last segment closes
        // the walk back onto it.
        let (a, _) = spec.segment_points(0);
        let (_, b_last) = spec.segment_points(15);
        assert_eq!(a, b_last);
        assert_eq!(spec.locate(0), (0, 0));
        assert_eq!(spec.locate(15), (0, 15));
    }

    #[test]
    fn box_loop_is_ccw_with_classify_parity_tags() {
        let (spec, _) = unit_box_spec(0.5);
        let lp = &spec.loops[0];
        assert!(lp.signed_area() > 0.0, "outer loop must be CCW");
        assert!((lp.signed_area() - 1.0).abs() < 1e-12);
        // bottom Wall, right Outlet, top Wall, left Inlet.
        assert_eq!(lp.tags[0], BoundaryType::Wall);
        assert_eq!(lp.tags[2], BoundaryType::Outlet);
        assert_eq!(lp.tags[4], BoundaryType::Wall);
        assert_eq!(lp.tags[6], BoundaryType::Inlet);
    }

    #[test]
    fn box_vertices_all_get_vertex_seeds() {
        let (spec, tol) = unit_box_spec(0.25);
        let (pts, kinds) = boundary_seeds(&spec, &tol);
        // Straight/convex everywhere: one seed per polyline vertex.
        assert_eq!(pts.len(), 16);
        for (p, k) in pts.iter().zip(&kinds) {
            match *k {
                SeedKind::Boundary { seg_prev, seg_next } => {
                    assert_ne!(seg_prev, seg_next, "box vertices are not guards");
                }
                SeedKind::Interior => panic!("boundary seed marked interior at {p:?}"),
            }
        }
        // Corner (0,0) is one of them.
        assert!(pts.iter().any(|p| p.x == 0.0 && p.y == 0.0));
    }

    #[test]
    fn circle_guards_collapse_to_chord_midpoints() {
        let domain = Vector2::new(1.0, 1.0);
        let tol = MeshgenTolerances::from_geometry(0.05, domain);
        let lp = circle_loop(Point2::new(0.5, 0.5), 0.2, 0.05, domain, &tol);
        assert!(lp.signed_area() < 0.0, "hole loop must be CW");
        assert!(lp.tags.iter().all(|t| *t == BoundaryType::Wall));
        let n = lp.pts.len();
        let spec = BoundarySpec::from_loops(vec![lp]);
        let (pts, kinds) = boundary_seeds(&spec, &tol);
        // Every vertex is reflex ⇒ exactly one (deduped) midpoint per chord.
        assert_eq!(pts.len(), n);
        for (i, (p, k)) in pts.iter().zip(&kinds).enumerate() {
            match *k {
                SeedKind::Boundary { seg_prev, seg_next } => {
                    assert_eq!(seg_prev, seg_next, "circle seeds are mid-segment guards");
                    let (a, b) = spec.segment_points(seg_prev);
                    let mid = Point2::new(0.5 * (a.x + b.x), 0.5 * (a.y + b.y));
                    assert!(
                        (p - mid).norm() < 1e-12,
                        "seed {i} not at its chord midpoint"
                    );
                }
                SeedKind::Interior => panic!("circle seed marked interior"),
            }
        }
    }

    #[test]
    fn point_in_fluid_respects_holes() {
        let domain = Vector2::new(1.0, 1.0);
        let tol = MeshgenTolerances::from_geometry(0.05, domain);
        let outer = polyline_loop(
            &[
                Point2::new(0.0, 0.0),
                Point2::new(1.0, 0.0),
                Point2::new(1.0, 1.0),
                Point2::new(0.0, 1.0),
            ],
            0.25,
            domain,
            &tol,
        );
        let hole = circle_loop(Point2::new(0.5, 0.5), 0.2, 0.05, domain, &tol);
        let spec = BoundarySpec::from_loops(vec![outer, hole]);
        assert!(point_in_fluid(Point2::new(0.1, 0.1), &spec));
        assert!(!point_in_fluid(Point2::new(0.5, 0.5), &spec));
        assert!(!point_in_fluid(Point2::new(1.2, 0.5), &spec));
        assert!(distance_to_loops(Point2::new(0.5, 0.5), &spec) > 0.19);
    }
}
