//! Diagram -> `Mesh` assembly for the meshless engine (M0.4, design §5).
//!
//! The per-cell rings of a `MeshlessDiagram` are stitched into one classic
//! static-pipeline `Mesh` in four passes:
//!
//! 1. **Tag-canonical vertex re-evaluation + quantized dedup.** Every ring
//!    vertex is the intersection of the two planes that created its incident
//!    edges (`PlaneTag` pair). Instead of trusting the clipped coordinates —
//!    which differ in last ulps between the 2–3 cells sharing the vertex,
//!    each having clipped in its own seed-relative frame — the vertex is
//!    re-evaluated *canonically* from the tag pair: a `Bisector`/`Bisector`
//!    pair seen from cell `i` is the circumcenter of the sorted seed triple
//!    `{i, j, k}` computed relative to the smallest-id seed, so all incident
//!    cells produce the exact same bits; `Bisector`/`Boundary` (or `Box`) is
//!    the canonical `bisector(min,max)` ∩ line solve; `Boundary`/`Boundary`
//!    and `Box` corners come from global polyline/domain constants. Dedup on
//!    the `quantize_point` grid then only has to absorb the *cocircular*
//!    coincidences it was designed for (different triples of one degenerate
//!    vertex), exactly like the incumbent (`voronoi.rs:84-92`).
//! 2. **Sub-tolerance edge merge.** Union-find over the deduped vertices for
//!    ring edges shorter than `edge_len_eps` — a verbatim policy transplant
//!    of the incumbent's `DisjointSet` merge (`voronoi.rs:24-51,204-216`,
//!    smaller-root-wins). Both cells incident to a collapsing face see the
//!    same merged vertex ids, so faces disappear symmetrically and no ring
//!    gaps can open.
//! 3. **Face emission**, sweeping cells in index order: a `Bisector(j)` edge
//!    with `j > i` emits the interior face (owner `i`, neighbor `j`, normal
//!    `normalize(p_j − p_i)` — the incumbent convention, `voronoi.rs:139` —
//!    which recalculate_geometry's sign-preservation keeps); `j < i` looks up
//!    the face `j` already emitted. `Boundary`/`Box` edges emit boundary
//!    faces with outward normals and `face_boundary` from
//!    `tag_boundary_type` (so nothing is ever left untagged —
//!    `close_untagged_boundary_faces` is *not* needed and not called).
//! 4. **Cell arrays** — CCW `cell_vertices` rings, `cell_faces` in ring
//!    order (first-use-by-owner, Morton-friendly since the seeds arrive
//!    Morton-sorted), `v_fixed` on boundary-face vertices, `face_wrap_shift`
//!    empty — and one defensive `recalculate_geometry()` so the stored
//!    geometry is bit-consistent with any later refresh path.
//!
//! Neither `fix_concave_cells` (cells are convex by construction) nor
//! `Mesh::smooth` (vertex smoothing would move Voronoi vertices off the
//! bisectors) is called: **cell `i` of the output is seed `i`**, always.
//!
//! Index-identity caveat (review F9): `Mesh::apply_env_cell_order`
//! (`CFD2_MESH_ORDER`, `ordering.rs`) permutes cells *post-generation* and
//! would silently void the seed-i == cell-i invariant. Harmless today (the
//! default is a no-op and M0 consumers don't rely on the invariant yet), but
//! the moving-mesh path (M2+) must assert that hook is inactive before
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

/// Union-find over Voronoi vertices for the sub-tolerance face merge —
/// verbatim policy transplant of the incumbent's `DisjointSet`
/// (`voronoi.rs:24-51`): path-halving find, deterministic smaller-root-wins
/// union.
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
/// every cell incident to the vertex computes the exact same bits. The
/// `clipped` coordinate is the deterministic per-cell fallback for
/// degenerate pairings (same plane twice, parallel lines) — those never
/// arise from a valid convex clip, and if one ever does, quantized dedup
/// still absorbs sub-`vertex_merge` disagreement.
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

    // Pass 2: sub-tolerance edge merge (incumbent DSU policy). Interior
    // edges are visited from both incident cells — the union is idempotent
    // and both see the same merged roots, so collapses are symmetric.
    let mut dsu = DisjointSet::new(vxy.len());
    let merge_sq = input.tol.edge_len_eps * input.tol.edge_len_eps;
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

    // Pass 3a: rebuild rings on merged vertex ids, dropping collapsed edges
    // (an edge survives iff its mapped endpoints differ; the surviving ring
    // chains exactly because collapsed runs share one root).
    let mut rings: Vec<(Vec<u32>, Vec<PlaneTag>)> = Vec::with_capacity(n);
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

    // Pass 3b/4: emit faces sweeping cells in index order, filling the cell
    // arrays in the same sweep (a cell's face indices are fully resolved
    // when it is visited: `j > i` faces are emitted here, `j < i` ones were
    // emitted by cell j).
    let mut mesh = Mesh::new();
    mesh.vx = vxy.iter().map(|p| p[0]).collect();
    mesh.vy = vxy.iter().map(|p| p[1]).collect();
    mesh.v_fixed = vec![false; vxy.len()];
    mesh.cell_face_offsets.push(0);
    mesh.cell_vertex_offsets.push(0);

    let mut face_map: AHashMap<(u32, u32), usize> = AHashMap::new(); // (i, j), i < j; lookup only
    for i in 0..n {
        let (rv, rt) = &rings[i];
        let m = rv.len();
        for e in 0..m {
            let va = rv[e] as usize;
            let vb = rv[(e + 1) % m] as usize;
            let f_idx = match rt[e] {
                PlaneTag::Bisector(j) => {
                    let j = j as usize;
                    debug_assert!(j < n && j != i, "cell {i}: bad neighbor tag {j}");
                    if j > i {
                        let idx = mesh.face_v1.len();
                        push_face_geometry(&mut mesh, &vxy, va, vb);
                        // Incumbent normal convention: normalize(p_j − p_i),
                        // owner = smaller seed id (voronoi.rs:139-147).
                        let nrm = (input.seeds[j] - input.seeds[i]).normalize();
                        mesh.face_nx.push(nrm.x);
                        mesh.face_ny.push(nrm.y);
                        mesh.face_owner.push(i);
                        mesh.face_neighbor.push(Some(j));
                        mesh.face_boundary.push(None);
                        let prev = face_map.insert((i as u32, j as u32), idx);
                        assert!(prev.is_none(), "duplicate interior face ({i},{j})");
                        idx
                    } else {
                        *face_map.get(&(j as u32, i as u32)).unwrap_or_else(|| {
                            panic!("non-reciprocal face: cell {i} sees neighbor {j}, but {j} emitted no face to {i}")
                        })
                    }
                }
                tag @ (PlaneTag::Boundary(_) | PlaneTag::Box(_)) => {
                    let bt = tag_boundary_type(tag, input.boundary)
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
                    idx
                }
            };
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
/// boundary seeding (F1 protocol) + Poisson interior fill, per-cell clipped
/// diagram, canonical assembly. No triangulation, no generator smoothing, no
/// concave fixing — and cell `i` is seed `i`.
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
