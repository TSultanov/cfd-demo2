//! Diagram -> `Mesh` assembly for the meshless engine.
//!
//! The per-cell rings of a `MeshlessDiagram` are stitched into one classic
//! static-pipeline `Mesh` in four passes:
//!
//! 1. **Tag-canonical vertex re-evaluation + quantized dedup.** Each ring
//!    vertex is re-evaluated *canonically* from the `PlaneTag` pair of its
//!    incident edges rather than from the clipped coordinates (which differ
//!    in last ulps between the 2–3 cells sharing it, each clipping in its own
//!    seed-relative frame): a `Bisector`/`Bisector` pair from cell `i` is the
//!    circumcenter of the sorted triple `{i, j, k}` relative to the smallest
//!    id, so all incident cells produce identical bits; `Bisector`/`Boundary`
//!    (or `Box`) is the canonical bisector ∩ line solve; `Boundary`/`Boundary`
//!    and `Box` corners come from polyline/domain constants. Quantized dedup
//!    then only absorbs the cocircular coincidences it is designed for.
//! 2. **Sub-tolerance edge merge.** Union-find over deduped vertices for ring
//!    edges shorter than `edge_len_eps` (smaller-root-wins), iterated with the
//!    ring rebuild until stable. Both cells incident to a collapsing face see
//!    the same merged ids, so faces disappear symmetrically.
//! 3. **Face resolution by canonical vertex pair.** Interior faces are paired
//!    GEOMETRICALLY — two ring edges sharing the same unordered deduped-vertex
//!    pair are one face — because tag reciprocity is not sound: the clip's
//!    Cut-vs-Redundant verdict is per cell, so near-coincident seed pairs make
//!    a shared neighbor swallow one twin's plane (a long real one-sided edge
//!    tagged with the eps-indistinguishable twin). Unpaired edges fall back
//!    to: endpoint-union for tiny (< 4·`edge_len_eps`) knife-edge stubs;
//!    CHAINING for coarse edges subdivided by finer twin cells; and a forced
//!    pairing with the tag's cell as total fallback (adversarial inputs only).
//!    Emission sweeps cells in index order: faces materialize at first
//!    reference (owner = smaller seed id, normal `normalize(p_b − p_a)`, kept
//!    by recalculate_geometry's sign-preservation). `Boundary`/`Box` edges
//!    emit boundary faces with outward normals and `face_boundary` from
//!    `tag_boundary_type`, so nothing is ever left untagged.
//! 4. **Cell arrays** — CCW `cell_vertices` rings, `cell_faces` in ring order,
//!    `v_fixed` on boundary-face vertices, and one defensive
//!    `recalculate_geometry()`.
//!
//! Neither `fix_concave_cells` (cells are convex by construction) nor
//! `Mesh::smooth` (would move Voronoi vertices off the bisectors) is called:
//! **cell `i` of the output is seed `i`**, always.
//!
//! Index-identity caveat: `Mesh::apply_env_cell_order` (`CFD2_MESH_ORDER`)
//! permutes cells post-generation and would void the seed-i == cell-i
//! invariant; the moving-mesh path must assert that hook is inactive before
//! trusting seed indices.

use ahash::AHashMap;
use nalgebra::{Point2, Vector2};

use super::super::geometry::Geometry;
use super::super::tolerances::MeshgenTolerances;
use super::boundary::{meshless_seed_points, tag_boundary_type, BoundarySpec};
use super::{
    build_diagram, CellStatus, EngineConfig, MeshlessDiagram, MeshlessInput, PlaneTag,
    MAX_CLIP_VERTS,
};
use crate::solver::mesh::Mesh;

/// Union-find over Voronoi vertices for the sub-tolerance face merge:
/// path-halving find, deterministic smaller-root-wins union.
struct DisjointSet {
    parent: Vec<usize>,
}

impl DisjointSet {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            // Deterministic: smaller root wins.
            let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
            self.parent[hi] = lo;
        }
    }
}

/// Intersection of two lines `n1·x = c1`, `n2·x = c2` (shared frame).
/// `None` when the determinant is degenerate (near-parallel lines — a
/// should-never-happen pairing for a real ring vertex).
#[inline]
fn line_intersect(
    n1: Vector2<f64>,
    c1: f64,
    n2: Vector2<f64>,
    c2: f64,
    det_eps: f64,
) -> Option<[f64; 2]> {
    let det = n1.x * n2.y - n1.y * n2.x;
    if det.abs() <= det_eps {
        return None;
    }
    Some([
        (c1 * n2.y - c2 * n1.y) / det,
        (n1.x * c2 - n2.x * c1) / det,
    ])
}

/// Segment line of global segment `seg` expressed as `n·x = c` in the frame
/// with `origin` subtracted (the canonical form all incident cells share).
#[inline]
fn segment_line(spec: &BoundarySpec, seg: u32, origin: Point2<f64>) -> (Vector2<f64>, f64) {
    let (a, b) = spec.segment_points(seg);
    let n = Vector2::new(b.y - a.y, -(b.x - a.x));
    let c = n.x * (a.x - origin.x) + n.y * (a.y - origin.y);
    (n, c)
}

/// Domain-box side line as `n·x = c` in the `origin`-subtracted frame.
#[inline]
fn box_line(side: u8, domain: Vector2<f64>, origin: Point2<f64>) -> (Vector2<f64>, f64) {
    match side {
        0 => (Vector2::new(1.0, 0.0), -origin.x),
        1 => (Vector2::new(1.0, 0.0), domain.x - origin.x),
        2 => (Vector2::new(0.0, 1.0), -origin.y),
        3 => (Vector2::new(0.0, 1.0), domain.y - origin.y),
        _ => unreachable!("invalid box side {side}"),
    }
}

/// Circumcenter of the seed triple `{i, j, k}`, computed with SORTED ids
/// relative to the smallest-id seed — a pure function of the id set, so the
/// 2–3 incident cells all produce bit-identical coordinates. Falls back to
/// the clipped coordinate for (near-)collinear triples.
fn circumcenter(
    seeds: &[Point2<f64>],
    ids: [usize; 3],
    det_eps: f64,
    clipped: [f64; 2],
) -> [f64; 2] {
    let mut t = ids;
    t.sort_unstable();
    let pa = seeds[t[0]];
    let qbx = seeds[t[1]].x - pa.x;
    let qby = seeds[t[1]].y - pa.y;
    let qcx = seeds[t[2]].x - pa.x;
    let qcy = seeds[t[2]].y - pa.y;
    let d = 2.0 * (qbx * qcy - qby * qcx);
    if d.abs() <= det_eps {
        return clipped;
    }
    let b2 = qbx * qbx + qby * qby;
    let c2 = qcx * qcx + qcy * qcy;
    let ux = (b2 * qcy - c2 * qby) / d;
    let uy = (c2 * qbx - b2 * qcx) / d;
    [pa.x + ux, pa.y + uy]
}

/// Canonically re-evaluate the ring vertex of cell `i` whose incident edges
/// were created by the plane pair `(t_in, t_out)`. Pure function of the tag
/// pair (plus `i` for bisectors, whose lines involve the owning seed), so
/// every cell incident to the vertex *with the same tag pair* computes the
/// exact same bits. Cells seeing one physical point through DIFFERENT tag
/// pairs (e.g. a reflex polyline corner) agree only to fp noise (~1e-16), a
/// coincidence class the quantized dedup absorbs like the cocircular one. The
/// `clipped` coordinate is the deterministic per-cell fallback for degenerate
/// pairings (same plane twice, parallel lines), which never arise from a valid
/// convex clip.
fn canonical_vertex(
    input: &MeshlessInput,
    i: usize,
    t_in: PlaneTag,
    t_out: PlaneTag,
    clipped: [f64; 2],
) -> [f64; 2] {
    let det_eps = input.tol.determinant_eps;
    let seeds = input.seeds;
    let spec = input.boundary;
    // Normalize the unordered pair (PlaneTag's derived Ord: Bisector <
    // Boundary < Box) so both edge orders hit the same formula.
    let (a, b) = if t_in <= t_out {
        (t_in, t_out)
    } else {
        (t_out, t_in)
    };
    match (a, b) {
        (PlaneTag::Bisector(j), PlaneTag::Bisector(k)) if j != k => {
            circumcenter(seeds, [i, j as usize, k as usize], det_eps, clipped)
        }
        (PlaneTag::Bisector(j), PlaneTag::Boundary(s)) => {
            let (lo, hi) = if (j as usize) < i {
                (j as usize, i)
            } else {
                (i, j as usize)
            };
            let p_lo = seeds[lo];
            let q = Vector2::new(seeds[hi].x - p_lo.x, seeds[hi].y - p_lo.y);
            let c1 = 0.5 * (q.x * q.x + q.y * q.y);
            let (n2, c2) = segment_line(spec, s, p_lo);
            match line_intersect(q, c1, n2, c2, det_eps) {
                Some(rel) => [p_lo.x + rel[0], p_lo.y + rel[1]],
                None => clipped,
            }
        }
        (PlaneTag::Bisector(j), PlaneTag::Box(side)) => {
            let (lo, hi) = if (j as usize) < i {
                (j as usize, i)
            } else {
                (i, j as usize)
            };
            let p_lo = seeds[lo];
            let q = Vector2::new(seeds[hi].x - p_lo.x, seeds[hi].y - p_lo.y);
            let c1 = 0.5 * (q.x * q.x + q.y * q.y);
            let (n2, c2) = box_line(side, input.domain, p_lo);
            match line_intersect(q, c1, n2, c2, det_eps) {
                Some(rel) => [p_lo.x + rel[0], p_lo.y + rel[1]],
                None => clipped,
            }
        }
        (PlaneTag::Boundary(s1), PlaneTag::Boundary(s2)) if s1 != s2 => {
            // Adjacent segments of one loop meet at an exact polyline point
            // (the common case: a vertex seed's own two segment lines).
            let (l1, i1) = spec.locate(s1);
            let (l2, i2) = spec.locate(s2);
            if l1 == l2 {
                let np = spec.loops[l1].pts.len();
                if (i1 + 1) % np == i2 {
                    let p = spec.loops[l1].pts[i2];
                    return [p.x, p.y];
                }
                if (i2 + 1) % np == i1 {
                    let p = spec.loops[l1].pts[i1];
                    return [p.x, p.y];
                }
            }
            // Non-adjacent (cannot arise from own-segment clipping, but stay
            // canonical): intersect the two lines in the absolute frame.
            let o = Point2::new(0.0, 0.0);
            let (n1, c1) = segment_line(spec, s1, o);
            let (n2, c2) = segment_line(spec, s2, o);
            line_intersect(n1, c1, n2, c2, det_eps).unwrap_or(clipped)
        }
        (PlaneTag::Boundary(s), PlaneTag::Box(side)) => {
            let o = Point2::new(0.0, 0.0);
            let (n1, c1) = segment_line(spec, s, o);
            let (n2, c2) = box_line(side, input.domain, o);
            line_intersect(n1, c1, n2, c2, det_eps).unwrap_or(clipped)
        }
        (PlaneTag::Box(s1), PlaneTag::Box(s2)) if s1 != s2 => {
            // One vertical side (0/1) meeting one horizontal side (2/3):
            // an exact domain corner. Parallel pairs fall through.
            let x = match (s1, s2) {
                (0, _) => Some(0.0),
                (1, _) => Some(input.domain.x),
                _ => None,
            };
            let y = match (s1, s2) {
                (_, 2) => Some(0.0),
                (_, 3) => Some(input.domain.y),
                _ => None,
            };
            match (x, y) {
                (Some(x), Some(y)) => [x, y],
                _ => clipped,
            }
        }
        _ => clipped,
    }
}

/// Ring view of cell `i`: the padded SoA slot for packed cells, the spill
/// ring for `RingOverflow` ones.
fn ring_of<'a>(
    d: &'a MeshlessDiagram,
    spill: &'a AHashMap<u32, (Vec<[f64; 2]>, Vec<PlaneTag>)>,
    i: usize,
) -> (&'a [[f64; 2]], &'a [PlaneTag]) {
    if d.status[i] == CellStatus::RingOverflow {
        let (xy, tags) = &spill[&(i as u32)];
        (xy.as_slice(), tags.as_slice())
    } else {
        let len = d.ring_len[i] as usize;
        (
            &d.ring_xy[i * MAX_CLIP_VERTS..i * MAX_CLIP_VERTS + len],
            &d.ring_plane[i * MAX_CLIP_VERTS..i * MAX_CLIP_VERTS + len],
        )
    }
}

/// Test instrument: number of mutual-orphan endpoint unions the pass below
/// performed since the last reset, and the largest endpoint gap it accepted
/// (stored as f64 bits — monotone for finite non-negative values). Assembly
/// is sequential, so `Relaxed` suffices.
pub static MUTUAL_ORPHAN_UNIONS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static MUTUAL_ORPHAN_MAX_GAP: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Reset both mutual-orphan instruments (test setup).
pub fn reset_mutual_orphan_stats() {
    MUTUAL_ORPHAN_UNIONS.store(0, std::sync::atomic::Ordering::Relaxed);
    MUTUAL_ORPHAN_MAX_GAP.store(0, std::sync::atomic::Ordering::Relaxed);
}

/// Assemble the diagram into a classic static-pipeline `Mesh` (see the
/// module docs for the pass structure). Sequential and deterministic: hash
/// maps are used for lookup only, never iterated. Panics on `EmptyCell` /
/// `SecurityRadiusFailed` statuses — the static `Mesh` format has no way to
/// carry an absent cell without breaking the seed-i == cell-i invariant, so
/// broken inputs are a hard error, not a silent compaction.
pub fn assemble_mesh(input: &MeshlessInput, d: &MeshlessDiagram) -> Mesh {
    let n = d.n;
    assert_eq!(input.seeds.len(), n, "diagram/seed count mismatch");
    for i in 0..n {
        assert!(
            !matches!(
                d.status[i],
                CellStatus::EmptyCell | CellStatus::SecurityRadiusFailed
            ),
            "cell {i} has status {:?}: broken input cannot be assembled",
            d.status[i]
        );
    }
    let spill: AHashMap<u32, (Vec<[f64; 2]>, Vec<PlaneTag>)> = d
        .overflow
        .iter()
        .map(|(c, ring)| {
            let xy: Vec<[f64; 2]> = ring.iter().map(|(p, _)| *p).collect();
            let tags: Vec<PlaneTag> = ring.iter().map(|(_, t)| *t).collect();
            (*c, (xy, tags))
        })
        .collect();

    // Pass 1: canonical vertices, deduplicated on the quantization grid.
    // The map is lookup-only; vertex ids follow the deterministic sweep
    // order (cells ascending, ring order within a cell).
    let mut vxy: Vec<[f64; 2]> = Vec::new();
    let mut vmap: AHashMap<(i64, i64), u32> = AHashMap::new();
    let mut cell_vert_ids: Vec<Vec<u32>> = Vec::with_capacity(n);
    for i in 0..n {
        let (xy, tags) = ring_of(d, &spill, i);
        let m = xy.len();
        assert!(m >= 3, "cell {i}: ring has {m} < 3 vertices");
        let mut ids = Vec::with_capacity(m);
        for e in 0..m {
            let t_in = tags[(e + m - 1) % m];
            let t_out = tags[e];
            let p = canonical_vertex(input, i, t_in, t_out, xy[e]);
            let key = input.tol.quantize_point(p[0], p[1]);
            let id = *vmap.entry(key).or_insert_with(|| {
                vxy.push(p);
                (vxy.len() - 1) as u32
            });
            ids.push(id);
        }
        cell_vert_ids.push(ids);
    }

    // Pass 2 + 3a, iterated: sub-tolerance edge merge and ring rebuild, then
    // face RESOLUTION by canonical vertex pair. Tag reciprocity is not a sound
    // pairing key: the Cut-vs-Redundant verdict is per cell (`sv > eps` against
    // the cell's OWN ring), so near-coincident seed pairs make a neighbor
    // swallow one twin's plane into the other's — the swallowed face is long
    // and real, merely tagged with the wrong (indistinguishable-within-eps)
    // twin. Faces are therefore paired GEOMETRICALLY: two ring edges sharing
    // the same unordered deduped-vertex-id pair are the same face (canonical
    // vertices make the ids bit-stable across cells). Leftover one-sided edges
    // are handled by, in order:
    //  - tiny orphans (< 4·edge_len_eps): the knife-edge class — union the
    //    endpoints and re-merge (the collapse both cells agree on);
    //  - long orphans, longest first: try to CHAIN them — a coarse edge
    //    [A,B] of cell x whose region is subdivided by finer cells pairs
    //    with a path of orphan edges tagged `x` from A to B (the twin-pair
    //    split case), yielding one face per sub-edge, all referenced by x;
    //  - anything still unmatched pairs with its own tag's cell, which
    //    references the face without owning a ring edge for it (total
    //    fallback: structurally valid, geometrically eps-approximate —
    //    reachable only from adversarial near-twin inputs).
    let mut dsu = DisjointSet::new(vxy.len());
    let merge_sq = input.tol.edge_len_eps * input.tol.edge_len_eps;
    // Tiny-orphan threshold: one-sided knife-edge stubs whose canonical
    // endpoints re-expanded past `edge_len_eps` still sit at that scale; 4x
    // gives headroom while staying far below real edges.
    let tiny_sq = 16.0 * merge_sq;

    // Pre-merge on the raw rings so the loop below runs its single-rebuild
    // fast path on clean inputs (iterating rebuild+merge from scratch costs
    // a full extra pass at 300k).
    for ids in &cell_vert_ids {
        let m = ids.len();
        for e in 0..m {
            let a = ids[e] as usize;
            let b = ids[(e + 1) % m] as usize;
            if a != b {
                let dx = vxy[b][0] - vxy[a][0];
                let dy = vxy[b][1] - vxy[a][1];
                if dx * dx + dy * dy < merge_sq {
                    dsu.union(a, b);
                }
            }
        }
    }

    let mut rings: Vec<(Vec<u32>, Vec<PlaneTag>)> = Vec::new();
    // Per-edge face resolution, parallel to `rings` (global edge index =
    // edge_off[cell] + ring position).
    #[derive(Clone, Debug)]
    enum EdgeRes {
        Boundary,
        /// Face descriptor index (interior face, this edge is one side).
        Face(u32),
        /// Coarse edge covered by several finer faces (chain case).
        Chain(Vec<u32>),
    }
    /// An interior face: the two incident cells (a < b) and the ring edge
    /// its geometry is taken from.
    struct FaceDesc {
        a: u32,
        b: u32,
        src_cell: u32,
        src_pos: u32,
    }
    let mut edge_off: Vec<usize> = Vec::new();
    let mut res: Vec<EdgeRes> = Vec::new();
    let mut descs: Vec<FaceDesc> = Vec::new();
    // Chain resolutions of the final pass: (coarse edge, its face descs).
    let mut res_chain: Vec<((u32, u32), Vec<u32>)> = Vec::new();
    // Faces force-referenced by a cell that has no ring edge for them
    // (fallback case), keyed by target cell; filled in ascending desc order.
    let mut forced_for: Vec<Vec<u32>> = vec![Vec::new(); n];

    loop {
        // Pass 3a: rebuild rings on merged vertex ids, dropping collapsed
        // edges (an edge survives iff its mapped endpoints differ; the
        // surviving ring chains exactly because collapsed runs share one
        // root).
        rings.clear();
        for i in 0..n {
            let ids = &cell_vert_ids[i];
            let (_, tags) = ring_of(d, &spill, i);
            let m = ids.len();
            let mapped: Vec<u32> = ids.iter().map(|&v| dsu.find(v as usize) as u32).collect();
            let mut rv: Vec<u32> = Vec::with_capacity(m);
            let mut rt: Vec<PlaneTag> = Vec::with_capacity(m);
            for e in 0..m {
                if mapped[e] != mapped[(e + 1) % m] {
                    rv.push(mapped[e]);
                    rt.push(tags[e]);
                }
            }
            assert!(
                rv.len() >= 3,
                "cell {i}: ring degenerated to {} vertices after sub-tolerance merge",
                rv.len()
            );
            rings.push((rv, rt));
        }

        // Length-based merge on the rebuilt rings (both incident cells see
        // the same roots, so collapses stay symmetric). Newly-merged edges
        // require another rebuild.
        let mut merged_any = false;
        for (rv, _) in &rings {
            let m = rv.len();
            for e in 0..m {
                let a = rv[e] as usize;
                let b = rv[(e + 1) % m] as usize;
                if a != b {
                    let dx = vxy[b][0] - vxy[a][0];
                    let dy = vxy[b][1] - vxy[a][1];
                    if dx * dx + dy * dy < merge_sq && dsu.find(a) != dsu.find(b) {
                        dsu.union(a, b);
                        merged_any = true;
                    }
                }
            }
        }
        if merged_any {
            continue;
        }

        // Pair interior edges by unordered vertex-id key: a single-slot map
        // holds the first unmatched edge per key, its partner (from another
        // cell) claims it — one entry per open face, no per-key Vecs.
        // Lookup/entry only, never iterated; pairing follows the
        // deterministic edge sweep (so on tag-consistent inputs the pair is
        // always claimed by the LARGER cell id, matching the incumbent
        // emission structure).
        edge_off.clear();
        let mut total = 0usize;
        for (rv, _) in &rings {
            edge_off.push(total);
            total += rv.len();
        }
        const UNRESOLVED: u32 = u32::MAX;
        let mut edge_face: Vec<u32> = vec![UNRESOLVED; total];
        descs.clear();
        for f in &mut forced_for {
            f.clear();
        }
        let mut open: AHashMap<(u32, u32), (u32, u32)> = AHashMap::with_capacity(total / 2 + 1);
        for (i, (rv, rt)) in rings.iter().enumerate() {
            let m = rv.len();
            for e in 0..m {
                if !matches!(rt[e], PlaneTag::Bisector(_)) {
                    continue;
                }
                let a = rv[e];
                let b = rv[(e + 1) % m];
                let key = (a.min(b), a.max(b));
                match open.get(&key).copied() {
                    Some((c2, e2)) if c2 as usize != i => {
                        open.remove(&key);
                        let id = descs.len() as u32;
                        descs.push(FaceDesc {
                            a: (i as u32).min(c2),
                            b: (i as u32).max(c2),
                            // Geometry from the FIRST (lower-id) side — the
                            // incumbent emission convention.
                            src_cell: c2,
                            src_pos: e2,
                        });
                        edge_face[edge_off[i] + e] = id;
                        edge_face[edge_off[c2 as usize] + e2 as usize] = id;
                    }
                    Some(_) => {
                        // Same-cell key reuse (degenerate ring): leave the
                        // stored edge open; this one becomes an orphan.
                    }
                    None => {
                        open.insert(key, (i as u32, e as u32));
                    }
                }
            }
        }
        // Unmatched edges (still open or shadowed) in deterministic order.
        let mut orphans: Vec<(u32, u32)> = Vec::new();
        for (i, (rv, _)) in rings.iter().enumerate() {
            for e in 0..rv.len() {
                if matches!(rings[i].1[e], PlaneTag::Bisector(_))
                    && edge_face[edge_off[i] + e] == UNRESOLVED
                {
                    orphans.push((i as u32, e as u32));
                }
            }
        }

        // Tiny orphans: knife-edge stubs the neighbor declared Redundant —
        // collapse them (endpoint union) and restart the merge loop.
        let mut any_tiny = false;
        for &(c, e) in &orphans {
            let (rv, _) = &rings[c as usize];
            let m = rv.len();
            let a = rv[e as usize] as usize;
            let b = rv[(e as usize + 1) % m] as usize;
            let dx = vxy[b][0] - vxy[a][0];
            let dy = vxy[b][1] - vxy[a][1];
            if dx * dx + dy * dy < tiny_sq {
                dsu.union(a, b);
                any_tiny = true;
            }
        }
        if any_tiny {
            continue;
        }

        // Long orphans, longest first (coarse edges before the finer edges
        // that subdivide them), ties on (cell, pos) for determinism.
        let mut order: Vec<usize> = (0..orphans.len()).collect();
        let edge_len_sq = |c: u32, e: u32| -> f64 {
            let (rv, _) = &rings[c as usize];
            let m = rv.len();
            let a = rv[e as usize] as usize;
            let b = rv[(e as usize + 1) % m] as usize;
            let dx = vxy[b][0] - vxy[a][0];
            let dy = vxy[b][1] - vxy[a][1];
            dx * dx + dy * dy
        };
        order.sort_by(|&x, &y| {
            let lx = edge_len_sq(orphans[x].0, orphans[x].1);
            let ly = edge_len_sq(orphans[y].0, orphans[y].1);
            ly.total_cmp(&lx).then(orphans[x].cmp(&orphans[y]))
        });
        // Orphans grouped by their tag's cell (chain candidates for a
        // coarse edge of cell x are orphan edges tagged x). Lookup-only.
        let mut by_tag: AHashMap<u32, Vec<(u32, u32)>> = AHashMap::new();
        for &(c, e) in &orphans {
            if let PlaneTag::Bisector(j) = rings[c as usize].1[e as usize] {
                by_tag.entry(j).or_default().push((c, e));
            }
        }

        // Mutual-orphan endpoint reconciliation: two cells that tag EACH
        // OTHER, whose edges share exactly one deduped vertex id and disagree
        // on the other by fp noise, are one face whose disagreeing endpoint was
        // canonicalized through DIFFERENT tag pairs. Two coincidence classes:
        //  - reflex/curved-wall guard pairs on f32-QUANTIZED seeds, where
        //    equidistance holds only to ~1 f32 position ulp (~1e-7·|x|), so one
        //    cell solves bisector ∩ seg_k and the other bisector ∩ seg_{k+1} to
        //    different bins;
        //  - sub-dedup-pitch BIN STRADDLES on either precision (gaps ~1.7e-8 <
        //    the 1e-6·h pitch — within the dedup's own tolerance, but the two
        //    solves round to adjacent quantize bins).
        // Union the disagreeing endpoints when they sit BOTH within 1e-3 of the
        // shorter edge's length AND within the f32 noise cap (64 ulps of domain
        // scale) — the relative condition alone could weld a genuine micro-face
        // of a third cell. Then restart the merge loop; both edges then share
        // both endpoint ids and pair geometrically.
        let noise_sq = {
            let s = 64.0 * 2f64.powi(-24) * input.domain.x.max(input.domain.y);
            s * s
        };
        let mut any_pair_union = false;
        for &(c, e) in &orphans {
            let PlaneTag::Bisector(j) = rings[c as usize].1[e as usize] else {
                unreachable!("orphans are bisector edges")
            };
            let (rv, _) = &rings[c as usize];
            let m = rv.len();
            let a1 = rv[e as usize] as usize;
            let b1 = rv[(e as usize + 1) % m] as usize;
            let Some(cands) = by_tag.get(&c) else { continue };
            for &(c2, e2) in cands {
                if c2 != j {
                    continue;
                }
                let (rv2, _) = &rings[c2 as usize];
                let m2 = rv2.len();
                let a2 = rv2[e2 as usize] as usize;
                let b2 = rv2[(e2 as usize + 1) % m2] as usize;
                // The partner edge runs the opposite way on a shared face:
                // (a1==b2, b1~a2) or (b1==a2, a1~b2). Same-orientation id
                // sharing is not the mutual-face pattern — skip it.
                let miss = if a1 == b2 && b1 != a2 {
                    Some((b1, a2))
                } else if b1 == a2 && a1 != b2 {
                    Some((a1, b2))
                } else {
                    None
                };
                let Some((x, y)) = miss else { continue };
                let dx = vxy[y][0] - vxy[x][0];
                let dy = vxy[y][1] - vxy[x][1];
                let d2 = dx * dx + dy * dy;
                let len2 = edge_len_sq(c, e).min(edge_len_sq(c2, e2));
                if d2 <= (1e-6 * len2).min(noise_sq) && dsu.find(x) != dsu.find(y) {
                    use std::sync::atomic::Ordering::Relaxed;
                    MUTUAL_ORPHAN_UNIONS.fetch_add(1, Relaxed);
                    MUTUAL_ORPHAN_MAX_GAP.fetch_max(d2.sqrt().to_bits(), Relaxed);
                    dsu.union(x, y);
                    any_pair_union = true;
                }
            }
        }
        if any_pair_union {
            continue;
        }
        for &oi in &order {
            let (c, e) = orphans[oi];
            if edge_face[edge_off[c as usize] + e as usize] != UNRESOLVED {
                continue; // consumed as a chain partner
            }
            let (rv, rt) = &rings[c as usize];
            let m = rv.len();
            let p_start = rv[e as usize];
            let p_end = rv[(e as usize + 1) % m];
            let PlaneTag::Bisector(j) = rt[e as usize] else {
                unreachable!("orphans are bisector edges")
            };
            debug_assert!((j as usize) < n && j as usize != c as usize);
            // Chain attempt: walk unresolved orphan edges tagged `c` from
            // p_start to p_end (each step picks the first candidate in
            // deterministic (cell, pos) order).
            let mut chain: Vec<(u32, u32)> = Vec::new();
            if let Some(cands) = by_tag.get(&c) {
                let mut cur = p_start;
                let mut guard = cands.len() + 1;
                while cur != p_end && guard > 0 {
                    guard -= 1;
                    let next = cands.iter().copied().find(|&(c2, e2)| {
                        if edge_face[edge_off[c2 as usize] + e2 as usize] != UNRESOLVED {
                            return false;
                        }
                        if chain.contains(&(c2, e2)) {
                            return false;
                        }
                        let (rv2, _) = &rings[c2 as usize];
                        let m2 = rv2.len();
                        let a2 = rv2[e2 as usize];
                        let b2 = rv2[(e2 as usize + 1) % m2];
                        a2 == cur || b2 == cur
                    });
                    match next {
                        Some((c2, e2)) => {
                            let (rv2, _) = &rings[c2 as usize];
                            let m2 = rv2.len();
                            let a2 = rv2[e2 as usize];
                            let b2 = rv2[(e2 as usize + 1) % m2];
                            cur = if a2 == cur { b2 } else { a2 };
                            chain.push((c2, e2));
                        }
                        None => break,
                    }
                }
                if cur != p_end {
                    chain.clear();
                }
            }
            if !chain.is_empty() {
                let mut ids = Vec::with_capacity(chain.len());
                for &(c2, e2) in &chain {
                    let id = descs.len() as u32;
                    descs.push(FaceDesc {
                        a: c.min(c2),
                        b: c.max(c2),
                        src_cell: c2,
                        src_pos: e2,
                    });
                    edge_face[edge_off[c2 as usize] + e2 as usize] = id;
                    ids.push(id);
                }
                // The coarse edge itself references the whole chain; encode
                // via a sentinel resolved below (Chain stored in `res`).
                edge_face[edge_off[c as usize] + e as usize] = u32::MAX - 1; // placeholder
                res_chain.push(((c, e), ids));
            } else {
                // Total fallback: pair with the tag's cell.
                let id = descs.len() as u32;
                descs.push(FaceDesc {
                    a: c.min(j),
                    b: c.max(j),
                    src_cell: c,
                    src_pos: e,
                });
                edge_face[edge_off[c as usize] + e as usize] = id;
                forced_for[j as usize].push(id);
            }
        }

        // Freeze the per-edge resolution for emission.
        res.clear();
        res.reserve(total);
        let mut chain_map: AHashMap<(u32, u32), Vec<u32>> = AHashMap::new();
        for (ce, ids) in res_chain.drain(..) {
            chain_map.insert(ce, ids);
        }
        for (i, (rv, rt)) in rings.iter().enumerate() {
            for e in 0..rv.len() {
                let r = match rt[e] {
                    PlaneTag::Bisector(_) => {
                        if let Some(ids) = chain_map.get(&(i as u32, e as u32)) {
                            EdgeRes::Chain(ids.clone())
                        } else {
                            EdgeRes::Face(edge_face[edge_off[i] + e])
                        }
                    }
                    _ => EdgeRes::Boundary,
                };
                res.push(r);
            }
        }
        break;
    }

    // Pass 4: emit faces sweeping cells in index order, filling the cell
    // arrays in the same sweep. A face is created the first time any of its
    // references is visited (for tag-consistent meshes that is the smaller
    // cell id's ring edge), so face ids are deterministic.
    let mut mesh = Mesh::new();
    mesh.vx = vxy.iter().map(|p| p[0]).collect();
    mesh.vy = vxy.iter().map(|p| p[1]).collect();
    mesh.v_fixed = vec![false; vxy.len()];
    mesh.cell_face_offsets.push(0);
    mesh.cell_vertex_offsets.push(0);

    const NOT_EMITTED: usize = usize::MAX;
    let mut desc_face: Vec<usize> = vec![NOT_EMITTED; descs.len()];
    let emit_desc = |mesh: &mut Mesh, desc_face: &mut Vec<usize>, id: u32| -> usize {
        if desc_face[id as usize] != NOT_EMITTED {
            return desc_face[id as usize];
        }
        let fd = &descs[id as usize];
        let (rv, _) = &rings[fd.src_cell as usize];
        let m = rv.len();
        let va = rv[fd.src_pos as usize] as usize;
        let vb = rv[(fd.src_pos as usize + 1) % m] as usize;
        let idx = mesh.face_v1.len();
        push_face_geometry(mesh, &vxy, va, vb);
        // Normal convention: normalize(p_b − p_a), owner = smaller seed id.
        let nrm = (input.seeds[fd.b as usize] - input.seeds[fd.a as usize]).normalize();
        mesh.face_nx.push(nrm.x);
        mesh.face_ny.push(nrm.y);
        mesh.face_owner.push(fd.a as usize);
        mesh.face_neighbor.push(Some(fd.b as usize));
        mesh.face_boundary.push(None);
        desc_face[id as usize] = idx;
        idx
    };

    for i in 0..n {
        let (rv, rt) = &rings[i];
        let m = rv.len();
        for e in 0..m {
            let va = rv[e] as usize;
            let vb = rv[(e + 1) % m] as usize;
            match &res[edge_off[i] + e] {
                EdgeRes::Face(id) => {
                    let f_idx = emit_desc(&mut mesh, &mut desc_face, *id);
                    mesh.cell_faces.push(f_idx);
                }
                EdgeRes::Chain(ids) => {
                    for id in ids {
                        let f_idx = emit_desc(&mut mesh, &mut desc_face, *id);
                        mesh.cell_faces.push(f_idx);
                    }
                }
                EdgeRes::Boundary => {
                    let bt = tag_boundary_type(rt[e], input.boundary)
                        .expect("boundary/box tags always resolve to a BoundaryType");
                    let idx = mesh.face_v1.len();
                    push_face_geometry(&mut mesh, &vxy, va, vb);
                    // Outward normal of a CCW ring edge: the right-hand
                    // perpendicular of the edge direction.
                    let dx = vxy[vb][0] - vxy[va][0];
                    let dy = vxy[vb][1] - vxy[va][1];
                    let len = (dx * dx + dy * dy).sqrt();
                    mesh.face_nx.push(dy / len);
                    mesh.face_ny.push(-dx / len);
                    mesh.face_owner.push(i);
                    mesh.face_neighbor.push(None);
                    mesh.face_boundary.push(Some(bt));
                    mesh.v_fixed[va] = true;
                    mesh.v_fixed[vb] = true;
                    mesh.cell_faces.push(idx);
                }
            }
        }
        // Faces this cell references without owning a ring edge for them
        // (fallback pairings targeting this cell), in ascending desc order.
        for id in &forced_for[i] {
            let f_idx = emit_desc(&mut mesh, &mut desc_face, *id);
            mesh.cell_faces.push(f_idx);
        }
        mesh.cell_face_offsets.push(mesh.cell_faces.len());
        mesh.cell_vertices.extend(rv.iter().map(|&v| v as usize));
        mesh.cell_vertex_offsets.push(mesh.cell_vertices.len());
        mesh.cell_cx.push(d.centroid[i][0]);
        mesh.cell_cy.push(d.centroid[i][1]);
        mesh.cell_vol.push(d.area[i]);
    }

    // Every boundary face was tagged at emission; `close_untagged_boundary_
    // faces` must never be needed on a meshless mesh.
    debug_assert_eq!(
        (0..mesh.num_faces())
            .filter(|&f| mesh.face_neighbor[f].is_none() && mesh.face_boundary[f].is_none())
            .count(),
        0,
        "meshless assembly left open untagged faces"
    );

    // Defensive recomputation from the final vertex set (preserves the
    // pre-filled normal orientation) so the stored geometry is
    // bit-consistent with any later `recalculate_geometry` refresh.
    mesh.recalculate_geometry();
    mesh
}

/// Push the vertex-derived face fields shared by interior and boundary
/// faces (`v1/v2/center/area`); the caller fills normal/owner/tags.
#[inline]
fn push_face_geometry(mesh: &mut Mesh, vxy: &[[f64; 2]], va: usize, vb: usize) {
    let pa = vxy[va];
    let pb = vxy[vb];
    mesh.face_v1.push(va);
    mesh.face_v2.push(vb);
    mesh.face_cx.push(0.5 * (pa[0] + pb[0]));
    mesh.face_cy.push(0.5 * (pa[1] + pb[1]));
    let dx = pb[0] - pa[0];
    let dy = pb[1] - pa[1];
    mesh.face_area.push((dx * dx + dy * dy).sqrt());
}

/// Meshless drop-in counterpart of `generate_voronoi_mesh`: loop-derived
/// boundary seeding + Poisson interior fill, per-cell clipped diagram,
/// canonical assembly. No triangulation, no generator smoothing, no concave
/// fixing — and cell `i` is seed `i`.
pub fn generate_meshless_voronoi_mesh(
    geo: &(impl Geometry + Sync),
    min_cell_size: f64,
    max_cell_size: f64,
    growth_rate: f64,
    domain_size: Vector2<f64>,
) -> Mesh {
    let (seeds, kinds, spec) =
        meshless_seed_points(geo, min_cell_size, max_cell_size, growth_rate, domain_size);
    let tol = MeshgenTolerances::from_geometry(min_cell_size, domain_size);
    let input = MeshlessInput {
        seeds: &seeds,
        kinds: &kinds,
        boundary: &spec,
        domain: domain_size,
        tol: &tol,
        cfg: EngineConfig::default(),
    };
    let diagram = build_diagram(&input);
    assemble_mesh(&input, &diagram)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The circumcenter must be a pure function of the id *set* — every
    /// incident cell computes the same bits regardless of which two
    /// bisectors it saw.
    #[test]
    fn circumcenter_is_order_invariant() {
        let seeds = vec![
            Point2::new(0.21, 0.34),
            Point2::new(0.55, 0.29),
            Point2::new(0.40, 0.61),
        ];
        let eps = 1e-18;
        let a = circumcenter(&seeds, [0, 1, 2], eps, [f64::NAN; 2]);
        let b = circumcenter(&seeds, [2, 0, 1], eps, [f64::NAN; 2]);
        let c = circumcenter(&seeds, [1, 2, 0], eps, [f64::NAN; 2]);
        assert_eq!(a[0].to_bits(), b[0].to_bits());
        assert_eq!(a[1].to_bits(), b[1].to_bits());
        assert_eq!(a[0].to_bits(), c[0].to_bits());
        assert_eq!(a[1].to_bits(), c[1].to_bits());
        // And it is equidistant from all three seeds.
        let p = Point2::new(a[0], a[1]);
        let r: Vec<f64> = seeds.iter().map(|s| (p - s).norm()).collect();
        assert!((r[0] - r[1]).abs() < 1e-12 && (r[0] - r[2]).abs() < 1e-12);
    }

    #[test]
    fn degenerate_circumcenter_falls_back_to_clipped() {
        let seeds = vec![
            Point2::new(0.0, 0.0),
            Point2::new(0.5, 0.0),
            Point2::new(1.0, 0.0), // collinear
        ];
        let out = circumcenter(&seeds, [0, 1, 2], 1e-12, [7.0, 8.0]);
        assert_eq!(out, [7.0, 8.0]);
    }
}
