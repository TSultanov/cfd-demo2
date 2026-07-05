//! Per-cell Voronoi clip kernel.
//!
//! A cell starts as the domain bounding box and is cut by one half-plane per
//! neighbor — the seed/neighbor bisector — in ascending neighbor distance.
//! All arithmetic is *seed-relative* (the seed sits at the origin) for
//! precision, the same idea as the relative-coordinate in-circle predicate
//! in `delaunay.rs`. The bisector is kept unnormalized,
//! `s(x) = x·q − ½|q|²` with `q = p_j − p_i` (no sqrt/normalize, robust for
//! near-coincident seeds), and intersections use the stable parametric form
//! `t = s_u/(s_u − s_v)`. Once the next neighbor satisfies `d² > 4·r2`
//! (`r2` = max squared vertex distance, the security radius), the cell is
//! provably final and clipping stops.
//!
//! Two ring representations share the exact same arithmetic (`sh_emit`):
//! `ClipPoly`, a fixed-capacity stack polygon (~1 KB, the GPU-shaped fast
//! path), and `ClipPolyVec`, the heap-backed slow path used when a ring
//! outgrows `MAX_CLIP_VERTS`. Bit-identical results between the two — and
//! between the kNN+security engine and the clip-everything oracle — are a
//! hard invariant, enforced by `tests/meshless_core_test.rs`.

use super::PlaneTag;
use nalgebra::{Point2, Vector2};

/// Fixed ring stride of the padded diagram SoA and capacity of the stack
/// polygon. Clipping a convex polygon changes the vertex count by at most +1
/// net, real Voronoi cells sit at 5–8 vertices, and k ≤ 64 neighbors keep the
/// worst case well below this; adversarial (cocircular) inputs that exceed it
/// fall back to `ClipPolyVec`.
pub const MAX_CLIP_VERTS: usize = 32;

/// Outcome of clipping a ring by one half-plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ClipOutcome {
    /// Every vertex is inside: the plane is redundant and the ring is
    /// bitwise untouched (no zero-length face is manufactured). This rule is
    /// what makes the security-radius early-out exact: skipped planes would
    /// all take this branch.
    Redundant,
    /// The ring was cut; at least 3 vertices remain.
    Cut,
    /// No vertex survived (or fewer than 3 did): the cell is empty.
    Empty,
    /// Fast path only: the result needs more than `MAX_CLIP_VERTS` slots.
    Overflow,
}

/// Half-plane in seed-relative coordinates: keep-set `{ x : s(x) ≤ eps }`
/// with `s(x) = x·q − off`. Degenerate vertices with `|s| ≤ eps` are
/// classified *inside* (kept), so cocircular seed quadruples produce
/// zero/near-zero-length ring edges rather than dropped topology.
pub(crate) struct HalfPlane {
    pub qx: f64,
    pub qy: f64,
    pub off: f64,
    pub eps: f64,
    pub tag: PlaneTag,
}

impl HalfPlane {
    /// Bisector of the seed (origin) and neighbor `j` at `q = p_j − p_i`:
    /// `s(x) = x·q − ½|q|²`. `eps = |q|·edge_len_eps` (a length × the
    /// existing sub-tolerance scale — dimensionally consistent with `s`).
    #[inline]
    pub fn bisector(q: Vector2<f64>, j: u32, edge_len_eps: f64) -> Self {
        let q2 = q.x * q.x + q.y * q.y;
        Self {
            qx: q.x,
            qy: q.y,
            off: 0.5 * q2,
            eps: q2.sqrt() * edge_len_eps,
            tag: PlaneTag::Bisector(j),
        }
    }

    /// Line through boundary segment `a -> b`, keep-set = the fluid side.
    /// Loops are walked with the fluid on the LEFT of the segment direction
    /// `d = b − a`, so the outward (solid-side) normal is `q = (d.y, −d.x)`
    /// and `s(x) = (x − a)·q ≤ 0` keeps the fluid. Everything is evaluated
    /// seed-relative (the seed lies ON the line, so `s(0) = 0` — kept by the
    /// ≤ rule). `eps = |d|·edge_len_eps`, dimensionally as for bisectors.
    #[inline]
    pub fn segment_line(
        a: Point2<f64>,
        b: Point2<f64>,
        seed: Point2<f64>,
        seg: u32,
        edge_len_eps: f64,
    ) -> Self {
        let ax = a.x - seed.x;
        let ay = a.y - seed.y;
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let qx = dy;
        let qy = -dx;
        Self {
            qx,
            qy,
            off: ax * qx + ay * qy,
            eps: (dx * dx + dy * dy).sqrt() * edge_len_eps,
            tag: PlaneTag::Boundary(seg),
        }
    }

    #[inline]
    fn eval(&self, x: f64, y: f64) -> f64 {
        x * self.qx + y * self.qy - self.off
    }
}

/// Shared Sutherland–Hodgman emit loop — the single copy of the clip
/// arithmetic, monomorphized over the output writer so the stack and Vec
/// paths produce bit-identical vertices. Edge `e` runs `v[e] -> v[e+1]` and
/// carries `tags[e]`; emitted vertices carry the tag of their *outgoing*
/// edge. Returns `false` if `push` refused a vertex (capacity overflow).
#[inline]
fn sh_emit<F: FnMut(f64, f64, PlaneTag) -> bool>(
    n: usize,
    x: &[f64],
    y: &[f64],
    tags: &[PlaneTag],
    s: &[f64],
    eps: f64,
    new_tag: PlaneTag,
    push: &mut F,
) -> bool {
    for e in 0..n {
        let w = if e + 1 == n { 0 } else { e + 1 };
        let u_in = s[e] <= eps;
        let w_in = s[w] <= eps;
        if u_in {
            // Vertex kept; its outgoing edge stays on its original plane
            // (cut edges keep their tag for the surviving sub-segment).
            if !push(x[e], y[e], tags[e]) {
                return false;
            }
        }
        if u_in != w_in {
            // Stable intersection: t = s_u/(s_u − s_w) ∈ [0,1] by sign
            // construction; the clamp absorbs the eps-inside fudge (a kept
            // vertex may have 0 < s ≤ eps, driving t marginally negative).
            let t = (s[e] / (s[e] - s[w])).clamp(0.0, 1.0);
            let xi = x[e] + t * (x[w] - x[e]);
            let yi = y[e] + t * (y[w] - y[e]);
            // Leaving the keep-set: the new vertex starts the clipping
            // plane's edge. Re-entering: it resumes the original edge.
            let tag = if u_in { new_tag } else { tags[e] };
            if !push(xi, yi, tag) {
                return false;
            }
        }
    }
    true
}

/// Common ring interface for the clip driver and geometry finalization.
pub(crate) trait CellRing {
    fn len(&self) -> usize;
    /// Seed-relative vertex coordinates.
    fn vert(&self, i: usize) -> (f64, f64);
    /// Tag of the edge `vert(i) -> vert(i+1)`.
    fn tag(&self, i: usize) -> PlaneTag;
    /// Max squared vertex distance from the seed (security radius squared).
    fn r2(&self) -> f64;
    fn clip(&mut self, hp: &HalfPlane) -> ClipOutcome;
}

/// Seed-relative domain-bbox ring (CCW) plus its r2: the initial state of
/// every cell. Edge tags: 0 bottom `Box(2)`, 1 right `Box(1)`, 2 top
/// `Box(3)`, 3 left `Box(0)` — side ids match `PlaneTag::Box` docs.
fn bbox_ring(seed: Point2<f64>, domain: Vector2<f64>) -> ([f64; 4], [f64; 4], [PlaneTag; 4], f64) {
    let x0 = -seed.x;
    let y0 = -seed.y;
    let x1 = domain.x - seed.x;
    let y1 = domain.y - seed.y;
    let xs = [x0, x1, x1, x0];
    let ys = [y0, y0, y1, y1];
    let tags = [
        PlaneTag::Box(2),
        PlaneTag::Box(1),
        PlaneTag::Box(3),
        PlaneTag::Box(0),
    ];
    let mut r2 = 0.0f64;
    for e in 0..4 {
        r2 = r2.max(xs[e] * xs[e] + ys[e] * ys[e]);
    }
    (xs, ys, tags, r2)
}

/// Fast path: fixed-capacity stack polygon (registers/L1, zero allocation).
pub(crate) struct ClipPoly {
    pub n: usize,
    pub x: [f64; MAX_CLIP_VERTS],
    pub y: [f64; MAX_CLIP_VERTS],
    pub plane: [PlaneTag; MAX_CLIP_VERTS],
    pub r2: f64,
}

impl ClipPoly {
    pub fn from_bbox(seed: Point2<f64>, domain: Vector2<f64>) -> Self {
        let (xs, ys, tags, r2) = bbox_ring(seed, domain);
        let mut ring = Self {
            n: 4,
            x: [0.0; MAX_CLIP_VERTS],
            y: [0.0; MAX_CLIP_VERTS],
            plane: [PlaneTag::PAD; MAX_CLIP_VERTS],
            r2,
        };
        ring.x[..4].copy_from_slice(&xs);
        ring.y[..4].copy_from_slice(&ys);
        ring.plane[..4].copy_from_slice(&tags);
        ring
    }
}

impl CellRing for ClipPoly {
    fn len(&self) -> usize {
        self.n
    }

    fn vert(&self, i: usize) -> (f64, f64) {
        (self.x[i], self.y[i])
    }

    fn tag(&self, i: usize) -> PlaneTag {
        self.plane[i]
    }

    fn r2(&self) -> f64 {
        self.r2
    }

    fn clip(&mut self, hp: &HalfPlane) -> ClipOutcome {
        let n = self.n;
        let mut s = [0.0f64; MAX_CLIP_VERTS];
        let mut any_out = false;
        let mut any_in = false;
        for e in 0..n {
            let sv = hp.eval(self.x[e], self.y[e]);
            s[e] = sv;
            if sv > hp.eps {
                any_out = true;
            } else {
                any_in = true;
            }
        }
        if !any_out {
            return ClipOutcome::Redundant;
        }
        if !any_in {
            self.n = 0;
            return ClipOutcome::Empty;
        }

        let mut nx = [0.0f64; MAX_CLIP_VERTS];
        let mut ny = [0.0f64; MAX_CLIP_VERTS];
        let mut nt = [PlaneTag::PAD; MAX_CLIP_VERTS];
        let mut m = 0usize;
        let mut r2 = 0.0f64;
        let ok = sh_emit(
            n,
            &self.x,
            &self.y,
            &self.plane,
            &s,
            hp.eps,
            hp.tag,
            &mut |vx, vy, tag| {
                if m == MAX_CLIP_VERTS {
                    return false;
                }
                nx[m] = vx;
                ny[m] = vy;
                nt[m] = tag;
                r2 = r2.max(vx * vx + vy * vy);
                m += 1;
                true
            },
        );
        if !ok {
            return ClipOutcome::Overflow;
        }
        if m < 3 {
            self.n = 0;
            return ClipOutcome::Empty;
        }
        self.x = nx;
        self.y = ny;
        self.plane = nt;
        self.n = m;
        self.r2 = r2;
        ClipOutcome::Cut
    }
}

/// Slow path: heap-backed ring with no capacity bound. Rare (adversarial
/// inputs only), so per-clip allocations are acceptable; the arithmetic is
/// the shared `sh_emit`, so results are bit-identical to the fast path.
pub(crate) struct ClipPolyVec {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub plane: Vec<PlaneTag>,
    pub r2: f64,
}

impl ClipPolyVec {
    pub fn from_bbox(seed: Point2<f64>, domain: Vector2<f64>) -> Self {
        let (xs, ys, tags, r2) = bbox_ring(seed, domain);
        Self {
            x: xs.to_vec(),
            y: ys.to_vec(),
            plane: tags.to_vec(),
            r2,
        }
    }
}

impl CellRing for ClipPolyVec {
    fn len(&self) -> usize {
        self.x.len()
    }

    fn vert(&self, i: usize) -> (f64, f64) {
        (self.x[i], self.y[i])
    }

    fn tag(&self, i: usize) -> PlaneTag {
        self.plane[i]
    }

    fn r2(&self) -> f64 {
        self.r2
    }

    fn clip(&mut self, hp: &HalfPlane) -> ClipOutcome {
        let n = self.x.len();
        let mut s = Vec::with_capacity(n);
        let mut any_out = false;
        let mut any_in = false;
        for e in 0..n {
            let sv = hp.eval(self.x[e], self.y[e]);
            s.push(sv);
            if sv > hp.eps {
                any_out = true;
            } else {
                any_in = true;
            }
        }
        if !any_out {
            return ClipOutcome::Redundant;
        }
        if !any_in {
            self.x.clear();
            self.y.clear();
            self.plane.clear();
            return ClipOutcome::Empty;
        }

        let mut nx = Vec::with_capacity(n + 2);
        let mut ny = Vec::with_capacity(n + 2);
        let mut nt = Vec::with_capacity(n + 2);
        let mut r2 = 0.0f64;
        sh_emit(
            n,
            &self.x,
            &self.y,
            &self.plane,
            &s,
            hp.eps,
            hp.tag,
            &mut |vx, vy, tag| {
                nx.push(vx);
                ny.push(vy);
                nt.push(tag);
                r2 = r2.max(vx * vx + vy * vy);
                true
            },
        );
        if nx.len() < 3 {
            self.x.clear();
            self.y.clear();
            self.plane.clear();
            return ClipOutcome::Empty;
        }
        self.x = nx;
        self.y = ny;
        self.plane = nt;
        self.r2 = r2;
        ClipOutcome::Cut
    }
}

/// Result of driving one cell's clip sequence over a neighbor list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Attempt {
    /// The security radius certified the cell final.
    Certified,
    /// The list was exhausted without certification (escalate, unless the
    /// list already contained every other seed — then the cell is exact).
    Uncertified,
    Empty,
    Overflow,
}

/// Clip the ring by each neighbor's bisector in ascending `(d², id)` order,
/// with the security-radius early-out `d² > 4·r2`. `nbrs` must be an exact
/// distance-ordered prefix of all other seeds (what `SeedGrid::knn`
/// produces), so every unvisited seed is at least as far as the last entry —
/// that makes the post-loop certification check sound.
pub(crate) fn drive<R: CellRing>(
    ring: &mut R,
    seed: Point2<f64>,
    seeds: &[Point2<f64>],
    nbrs: &[(f64, u32)],
    edge_len_eps: f64,
) -> Attempt {
    for &(d2, j) in nbrs {
        if d2 > 4.0 * ring.r2() {
            // Sorted ascending: this and every later plane misses the cell.
            return Attempt::Certified;
        }
        let q = seeds[j as usize] - seed;
        match ring.clip(&HalfPlane::bisector(q, j, edge_len_eps)) {
            ClipOutcome::Empty => return Attempt::Empty,
            ClipOutcome::Overflow => return Attempt::Overflow,
            ClipOutcome::Redundant | ClipOutcome::Cut => {}
        }
    }
    match nbrs.last() {
        // r2 shrank while clipping, so re-check against the farthest visited
        // neighbor: all unvisited seeds are at least that far.
        Some(&(d2, _)) if d2 > 4.0 * ring.r2() => Attempt::Certified,
        // No other seeds exist: the bbox ring is the exact cell.
        None => Attempt::Certified,
        _ => Attempt::Uncertified,
    }
}

/// Shoelace area + centroid on the absolute-translated ring — the same sums
/// (and the same `1e-12` degenerate-area fallback) as
/// `Mesh::recalculate_geometry`, so assembly/refresh stay bit-consistent.
pub(crate) fn ring_geometry<R: CellRing>(ring: &R, seed: Point2<f64>) -> (f64, [f64; 2]) {
    let n = ring.len();
    let mut signed_area = 0.0f64;
    let mut c_x = 0.0f64;
    let mut c_y = 0.0f64;
    for e in 0..n {
        let w = if e + 1 == n { 0 } else { e + 1 };
        let (x0, y0) = ring.vert(e);
        let (x1, y1) = ring.vert(w);
        let p0x = x0 + seed.x;
        let p0y = y0 + seed.y;
        let p1x = x1 + seed.x;
        let p1y = y1 + seed.y;
        let cross = p0x * p1y - p1x * p0y;
        signed_area += cross;
        c_x += (p0x + p1x) * cross;
        c_y += (p0y + p1y) * cross;
    }
    signed_area *= 0.5;
    let area = signed_area.abs();
    if area > 1e-12 {
        (area, [c_x / (6.0 * signed_area), c_y / (6.0 * signed_area)])
    } else {
        let mut ax = 0.0;
        let mut ay = 0.0;
        for e in 0..n {
            let (x, y) = ring.vert(e);
            ax += x + seed.x;
            ay += y + seed.y;
        }
        let inv = 1.0 / (n.max(1) as f64);
        (area, [ax * inv, ay * inv])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bbox_ring_is_ccw_and_covers_domain() {
        let seed = Point2::new(0.25, 0.5);
        let domain = Vector2::new(2.0, 1.0);
        let ring = ClipPoly::from_bbox(seed, domain);
        assert_eq!(ring.n, 4);
        let (area, centroid) = ring_geometry(&ring, seed);
        assert!((area - 2.0).abs() < 1e-14);
        assert!((centroid[0] - 1.0).abs() < 1e-14);
        assert!((centroid[1] - 0.5).abs() < 1e-14);
        // r2 = squared distance to the farthest corner (2, 1).
        let expect = (2.0 - 0.25f64).powi(2) + (1.0 - 0.5f64).powi(2);
        assert_eq!(ring.r2.to_bits(), expect.to_bits());
    }

    #[test]
    fn bisector_clip_halves_the_box() {
        let seed = Point2::new(0.5, 0.5);
        let domain = Vector2::new(2.0, 1.0);
        let mut ring = ClipPoly::from_bbox(seed, domain);
        // Neighbor at (1.5, 0.5): bisector is the vertical line x = 1.0.
        let q = Vector2::new(1.0, 0.0);
        let outcome = ring.clip(&HalfPlane::bisector(q, 7, 1e-9));
        assert_eq!(outcome, ClipOutcome::Cut);
        assert_eq!(ring.n, 4);
        let (area, _) = ring_geometry(&ring, seed);
        assert!((area - 1.0).abs() < 1e-14);
        // Exactly one edge carries the bisector tag.
        let bis = (0..ring.n)
            .filter(|&e| ring.plane[e] == PlaneTag::Bisector(7))
            .count();
        assert_eq!(bis, 1);
    }

    #[test]
    fn redundant_plane_leaves_ring_bitwise_untouched() {
        let seed = Point2::new(0.5, 0.5);
        let domain = Vector2::new(1.0, 1.0);
        let mut ring = ClipPoly::from_bbox(seed, domain);
        let before: Vec<u64> = (0..ring.n)
            .flat_map(|e| [ring.x[e].to_bits(), ring.y[e].to_bits()])
            .collect();
        // Far neighbor: the bisector misses the box entirely.
        let q = Vector2::new(10.0, 0.0);
        assert_eq!(
            ring.clip(&HalfPlane::bisector(q, 1, 1e-9)),
            ClipOutcome::Redundant
        );
        let after: Vec<u64> = (0..ring.n)
            .flat_map(|e| [ring.x[e].to_bits(), ring.y[e].to_bits()])
            .collect();
        assert_eq!(before, after);
    }

    #[test]
    fn fast_and_vec_paths_agree_bitwise() {
        let seed = Point2::new(0.31, 0.47);
        let domain = Vector2::new(2.0, 1.0);
        let mut fast = ClipPoly::from_bbox(seed, domain);
        let mut slow = ClipPolyVec::from_bbox(seed, domain);
        let nbrs = [
            Vector2::new(0.11, 0.02),
            Vector2::new(-0.09, 0.05),
            Vector2::new(0.01, -0.12),
            Vector2::new(-0.03, 0.10),
            Vector2::new(0.08, 0.09),
        ];
        for (j, q) in nbrs.iter().enumerate() {
            let hp = HalfPlane::bisector(*q, j as u32, 1e-9);
            fast.clip(&hp);
            let hp = HalfPlane::bisector(*q, j as u32, 1e-9);
            slow.clip(&hp);
        }
        assert_eq!(fast.n, slow.len());
        for e in 0..fast.n {
            let (fx, fy) = (fast.x[e], fast.y[e]);
            let (sx, sy) = slow.vert(e);
            assert_eq!(fx.to_bits(), sx.to_bits());
            assert_eq!(fy.to_bits(), sy.to_bits());
            assert_eq!(fast.plane[e], slow.tag(e));
        }
        assert_eq!(fast.r2.to_bits(), slow.r2.to_bits());
    }
}
