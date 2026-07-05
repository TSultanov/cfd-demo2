use super::delaunay::{triangulate, Edge};
use super::geometry::Geometry;
use super::mesh_builder::{CellId, MeshBuilder, VertexId};
use super::tolerances::MeshgenTolerances;
use crate::solver::mesh::{BoundaryType, Mesh};
use ahash::AHashMap;
use nalgebra::{Point2, Vector2};
use rayon::prelude::*;

struct FaceResult {
    v1: usize,
    v2: usize,
    nx: f64,
    ny: f64,
    owner: usize,
    neighbor: Option<usize>,
    boundary: Option<BoundaryType>,
}

/// Union-find over Voronoi vertices, used to merge the endpoints of
/// sub-tolerance faces (cocircular generator quadruples produce coincident
/// circumcenters; without merging those faces have ~zero length and their
/// normals evaluate to NaN in `recalculate_geometry`).
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

pub fn generate_voronoi_mesh(
    geo: &(impl Geometry + Sync),
    min_cell_size: f64,
    max_cell_size: f64,
    growth_rate: f64,
    domain_size: Vector2<f64>,
) -> Mesh {
    let tol = MeshgenTolerances::from_geometry(min_cell_size, domain_size);

    let (points, triangles, _fixed_nodes) =
        triangulate(geo, min_cell_size, max_cell_size, growth_rate, domain_size);

    // 1. Edge -> triangle adjacency.
    let mut edge_to_triangles: AHashMap<Edge, Vec<usize>> = AHashMap::new();
    for (t_idx, t) in triangles.iter().enumerate() {
        for edge in [
            Edge::new(t.v1, t.v2),
            Edge::new(t.v2, t.v3),
            Edge::new(t.v3, t.v1),
        ] {
            edge_to_triangles.entry(edge).or_default().push(t_idx);
        }
    }

    // Deterministic edge order (hash-map iteration order is arbitrary).
    let mut edges: Vec<Edge> = edge_to_triangles.keys().cloned().collect();
    edges.sort_by(|a, b| a.v1.cmp(&b.v1).then(a.v2.cmp(&b.v2)));

    // 2. Voronoi vertices, deduplicated on the quantization grid. Coincident
    // circumcenters (cocircular points) collapse to a single vertex here.
    let mut voronoi_points: Vec<Point2<f64>> = Vec::new();
    let mut vor_map: AHashMap<(i64, i64), usize> = AHashMap::new();
    let mut add_vor_point = |p: Point2<f64>, voronoi_points: &mut Vec<Point2<f64>>| -> usize {
        *vor_map
            .entry(tol.quantize_point(p.x, p.y))
            .or_insert_with(|| {
                voronoi_points.push(p);
                voronoi_points.len() - 1
            })
    };

    let circumcenter_indices: Vec<usize> = triangles
        .iter()
        .map(|t| add_vor_point(t.circumcenter, &mut voronoi_points))
        .collect();

    // Hull-edge midpoints and hull-vertex positions (they close the boundary
    // cells: a boundary generator's cell runs ... -> midpoint -> generator ->
    // midpoint -> ... along the hull).
    let mut midpoint_indices: AHashMap<Edge, usize> = AHashMap::new();
    let mut vertex_indices: Vec<Option<usize>> = vec![None; points.len()];
    for edge in &edges {
        if edge_to_triangles[edge].len() == 1 {
            let p1 = points[edge.v1];
            let p2 = points[edge.v2];
            let mid = Point2::new((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0);
            midpoint_indices.insert(*edge, add_vor_point(mid, &mut voronoi_points));
            if vertex_indices[edge.v1].is_none() {
                vertex_indices[edge.v1] = Some(add_vor_point(p1, &mut voronoi_points));
            }
            if vertex_indices[edge.v2].is_none() {
                vertex_indices[edge.v2] = Some(add_vor_point(p2, &mut voronoi_points));
            }
        }
    }

    // 3. Face candidates, one batch per Delaunay edge (parallel over the
    // deterministic edge order).
    let process_edge = |edge: &Edge| -> Vec<FaceResult> {
        let tris = &edge_to_triangles[edge];
        let mut results = Vec::new();
        let v1 = edge.v1;
        let v2 = edge.v2;

        // Main face: the perpendicular-bisector segment between the two
        // circumcenters (or circumcenter -> hull-edge midpoint on the hull).
        let idx_a = circumcenter_indices[tris[0]];
        let idx_b = if tris.len() == 2 {
            circumcenter_indices[tris[1]]
        } else {
            midpoint_indices[edge]
        };

        if idx_a != idx_b {
            let p_v1 = points[v1];
            let p_v2 = points[v2];
            let normal = (p_v2 - p_v1).normalize();
            results.push(FaceResult {
                v1: idx_a,
                v2: idx_b,
                nx: normal.x,
                ny: normal.y,
                owner: v1,
                neighbor: Some(v2),
                boundary: None,
            });
        }

        // Hull edge: close the two boundary cells with the half-edges
        // [midpoint, v1] and [midpoint, v2] lying on the hull segment.
        if tris.len() == 1 {
            let idx_mid = midpoint_indices[edge];
            let idx_v1 = vertex_indices[v1].unwrap();
            let idx_v2 = vertex_indices[v2].unwrap();

            let p_v1 = points[v1];
            let p_v2 = points[v2];
            let tangent = p_v2 - p_v1;
            let mut normal = Vector2::new(tangent.y, -tangent.x).normalize();

            // Orient outward (away from the triangle's interior).
            let t = triangles[tris[0]];
            let t_center = (points[t.v1].coords + points[t.v2].coords + points[t.v3].coords) / 3.0;
            let edge_center = (p_v1.coords + p_v2.coords) / 2.0;
            if (edge_center - t_center).dot(&normal) < 0.0 {
                normal = -normal;
            }

            for (idx_end, own) in [(idx_v1, v1), (idx_v2, v2)] {
                if idx_mid == idx_end {
                    continue;
                }
                let p_mid = voronoi_points[idx_mid];
                let p_end = voronoi_points[idx_end];
                let center = Point2::from((p_mid.coords + p_end.coords) * 0.5);
                let boundary = tol.classify_boundary(center.x, center.y, domain_size.x, domain_size.y);
                results.push(FaceResult {
                    v1: idx_mid,
                    v2: idx_end,
                    nx: normal.x,
                    ny: normal.y,
                    owner: own,
                    neighbor: None,
                    boundary,
                });
            }
        }

        results
    };

    let mut all_faces: Vec<FaceResult> = if edges.len() < 5000 {
        edges.iter().flat_map(|edge| process_edge(edge)).collect()
    } else {
        edges
            .par_iter()
            .flat_map(|edge| process_edge(edge))
            .collect()
    };

    // 4. Merge the endpoints of sub-tolerance faces so no near-zero-length
    // faces (and no ring gaps) survive, then drop the collapsed faces.
    let mut dsu = DisjointSet::new(voronoi_points.len());
    for f in &all_faces {
        let d2 = (voronoi_points[f.v1] - voronoi_points[f.v2]).norm_squared();
        if d2 < tol.edge_len_eps * tol.edge_len_eps {
            dsu.union(f.v1, f.v2);
        }
    }
    for f in all_faces.iter_mut() {
        f.v1 = dsu.find(f.v1);
        f.v2 = dsu.find(f.v2);
    }
    all_faces.retain(|f| f.v1 != f.v2);

    // 5. Per-generator face lists.
    let n_gen = points.len();
    let mut cell_faces: Vec<Vec<usize>> = vec![Vec::new(); n_gen];
    for (f_idx, f) in all_faces.iter().enumerate() {
        cell_faces[f.owner].push(f_idx);
        if let Some(nb) = f.neighbor {
            cell_faces[nb].push(f_idx);
        }
    }

    // 6. Vertex rings per cell: chain the cell's faces end-to-end. Cells whose
    // face graph is not a single degree-2 cycle fall back to an angular sort
    // around the generator (star-shaped recovery) and are excluded from the
    // concave-split pass.
    let mut rings: Vec<Vec<usize>> = Vec::with_capacity(n_gen);
    let mut ring_consistent: Vec<bool> = Vec::with_capacity(n_gen);
    let mut alive: Vec<bool> = Vec::with_capacity(n_gen);
    for i in 0..n_gen {
        let (ring, consistent) = build_ring(&cell_faces[i], &all_faces, &voronoi_points, points[i]);
        alive.push(ring.len() >= 3);
        rings.push(ring);
        ring_consistent.push(consistent);
    }

    // 7. Remap dead generators (no resolvable polygon). Their faces either
    // vanish (both sides dead) or turn into hull faces of the survivor.
    let mut cell_id: Vec<usize> = vec![usize::MAX; n_gen];
    let mut n_cells = 0;
    for i in 0..n_gen {
        if alive[i] {
            cell_id[i] = n_cells;
            n_cells += 1;
        }
    }

    // 8. Assemble the mesh.
    let mut mesh = Mesh::new();
    mesh.vx = voronoi_points.iter().map(|p| p.x).collect();
    mesh.vy = voronoi_points.iter().map(|p| p.y).collect();

    // Fixed vertices: everything on the hull (midpoints and generator
    // positions) must not be moved by smoothing.
    mesh.v_fixed = vec![false; mesh.vx.len()];
    for idx in midpoint_indices.values() {
        mesh.v_fixed[dsu.find(*idx)] = true;
    }
    for idx in vertex_indices.iter().flatten() {
        mesh.v_fixed[dsu.find(*idx)] = true;
    }

    let mut kept_face_cells: Vec<(usize, Option<usize>)> = Vec::with_capacity(all_faces.len());
    for f in &all_faces {
        let owner_alive = alive[f.owner];
        let neighbor_alive = f.neighbor.map(|nb| alive[nb]).unwrap_or(false);
        let (owner, neighbor) = match (owner_alive, neighbor_alive) {
            (true, true) => (f.owner, f.neighbor),
            (true, false) => (f.owner, None),
            (false, true) => (f.neighbor.unwrap(), None),
            (false, false) => continue,
        };
        mesh.face_v1.push(f.v1);
        mesh.face_v2.push(f.v2);
        let pa = voronoi_points[f.v1];
        let pb = voronoi_points[f.v2];
        mesh.face_cx.push((pa.x + pb.x) * 0.5);
        mesh.face_cy.push((pa.y + pb.y) * 0.5);
        mesh.face_nx.push(f.nx);
        mesh.face_ny.push(f.ny);
        mesh.face_area.push((pa - pb).norm());
        mesh.face_owner.push(cell_id[owner]);
        mesh.face_neighbor.push(neighbor.map(|nb| cell_id[nb]));
        mesh.face_boundary
            .push(if neighbor.is_some() { None } else { f.boundary });
        kept_face_cells.push((owner, neighbor));
    }

    mesh.cell_face_offsets.push(0);
    mesh.cell_vertex_offsets.push(0);
    let mut mesh_cell_faces: Vec<Vec<usize>> = vec![Vec::new(); n_cells];
    for (f_idx, &(owner, neighbor)) in kept_face_cells.iter().enumerate() {
        mesh_cell_faces[cell_id[owner]].push(f_idx);
        if let Some(nb) = neighbor {
            mesh_cell_faces[cell_id[nb]].push(f_idx);
        }
    }
    let mut splittable = Vec::with_capacity(n_cells);
    let mut generators = Vec::with_capacity(n_cells);
    for i in 0..n_gen {
        if !alive[i] {
            continue;
        }
        mesh.cell_cx.push(points[i].x);
        mesh.cell_cy.push(points[i].y);
        mesh.cell_vol.push(0.0);
        mesh.cell_faces.extend(&mesh_cell_faces[cell_id[i]]);
        mesh.cell_face_offsets.push(mesh.cell_faces.len());
        mesh.cell_vertices.extend(&rings[i]);
        mesh.cell_vertex_offsets.push(mesh.cell_vertices.len());
        splittable.push(ring_consistent[i]);
        generators.push(points[i]);
    }

    mesh.recalculate_geometry();

    // 9. Fix Concave Cells
    mesh = fix_concave_cells(mesh, &generators, &splittable, &tol);

    // 10. Close embedded-geometry faces (obstacle/step/nozzle contours) that
    // are not on the domain box and so were left untyped — see the doc comment
    // on `close_untagged_boundary_faces`.
    super::delaunay::close_untagged_boundary_faces(&mut mesh);

    mesh
}

/// Order a cell's face endpoints into a closed CCW vertex ring.
///
/// Returns `(ring, consistent)`. `consistent` means the faces form a single
/// degree-2 cycle (each vertex appears in exactly two of the cell's faces), so
/// every face maps to a consecutive ring pair — the precondition for the
/// concave-split pass. Defective cells (degenerate topology) fall back to an
/// angular sort of the unique endpoints around the generator, which recovers a
/// valid star-shaped polygon but is excluded from splitting.
fn build_ring(
    face_indices: &[usize],
    faces: &[FaceResult],
    vor_points: &[Point2<f64>],
    generator: Point2<f64>,
) -> (Vec<usize>, bool) {
    let n = face_indices.len();
    if n < 3 {
        return (Vec::new(), false);
    }

    // Local adjacency: vertex -> up to 2 partner vertices.
    let mut verts: Vec<usize> = Vec::with_capacity(2 * n);
    for &fi in face_indices {
        verts.push(faces[fi].v1);
        verts.push(faces[fi].v2);
    }

    // Degree check: a chainable ring needs every vertex exactly twice.
    let mut sorted = verts.clone();
    sorted.sort_unstable();
    let mut degree_ok = true;
    let mut k = 0;
    while k < sorted.len() {
        let run = sorted[k..].iter().take_while(|&&v| v == sorted[k]).count();
        if run != 2 {
            degree_ok = false;
            break;
        }
        k += run;
    }

    let ring = if degree_ok {
        // Chain deterministically starting from the smallest vertex.
        let mut ring = Vec::with_capacity(n);
        let start = sorted[0];
        let mut used = vec![false; n];
        let mut curr = start;
        loop {
            ring.push(curr);
            let mut next = None;
            for (slot, &fi) in face_indices.iter().enumerate() {
                if used[slot] {
                    continue;
                }
                let (a, b) = (faces[fi].v1, faces[fi].v2);
                if a == curr {
                    used[slot] = true;
                    next = Some(b);
                    break;
                }
                if b == curr {
                    used[slot] = true;
                    next = Some(a);
                    break;
                }
            }
            match next {
                Some(v) if v != start => curr = v,
                _ => break,
            }
        }
        if ring.len() == n {
            ring
        } else {
            // Disconnected cycles — fall back below.
            Vec::new()
        }
    } else {
        Vec::new()
    };

    let (mut ring, consistent) = if ring.is_empty() {
        // Fallback: unique vertices sorted by angle around the generator.
        let mut unique = verts;
        unique.sort_unstable();
        unique.dedup();
        unique.sort_by(|&a, &b| {
            let pa = vor_points[a] - generator;
            let pb = vor_points[b] - generator;
            pa.y.atan2(pa.x)
                .partial_cmp(&pb.y.atan2(pb.x))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        (unique, false)
    } else {
        (ring, true)
    };

    // Enforce CCW ordering.
    if ring.len() >= 3 {
        let mut signed_area = 0.0;
        let m = ring.len();
        for k in 0..m {
            let p0 = vor_points[ring[k]];
            let p1 = vor_points[ring[(k + 1) % m]];
            signed_area += p0.x * p1.y - p1.x * p0.y;
        }
        if signed_area < 0.0 {
            ring.reverse();
        }
    }

    (ring, consistent)
}

struct SplitInfo {
    is_split: bool,
    /// For split cells: sub_cells[k] = CellId of the sub-cell that owns edge k.
    /// For non-split cells: single element with the CellId.
    new_cell_ids: Vec<CellId>,
    center_vert_id: VertexId,
}

fn fix_concave_cells(
    old_mesh: Mesh,
    generators: &[Point2<f64>],
    splittable: &[bool],
    tol: &MeshgenTolerances,
) -> Mesh {
    let mut builder = MeshBuilder::with_capacity(
        old_mesh.num_vertices() + old_mesh.num_cells(),
        old_mesh.num_cells() * 2,
        old_mesh.num_faces() * 2,
    );

    // 1. Copy all existing vertices into the builder
    let old_vert_ids: Vec<VertexId> = (0..old_mesh.num_vertices())
        .map(|i| builder.add_vertex(old_mesh.vx[i], old_mesh.vy[i], old_mesh.v_fixed[i]))
        .collect();

    // A cell may only be split if every one of its faces spans a consecutive
    // ring pair — otherwise `get_sub_cell` below cannot resolve which sub-cell
    // a face belongs to. (Fallback-ordered rings don't satisfy this.)
    let face_splittable = |cell_idx: usize| -> bool {
        if !splittable[cell_idx] {
            return false;
        }
        let start = old_mesh.cell_face_offsets[cell_idx];
        let end = old_mesh.cell_face_offsets[cell_idx + 1];
        let vstart = old_mesh.cell_vertex_offsets[cell_idx];
        let vend = old_mesh.cell_vertex_offsets[cell_idx + 1];
        let n = vend - vstart;
        old_mesh.cell_faces[start..end].iter().all(|&f| {
            let (a, b) = (old_mesh.face_v1[f], old_mesh.face_v2[f]);
            (0..n).any(|k| {
                let va = old_mesh.cell_vertices[vstart + k];
                let vb = old_mesh.cell_vertices[vstart + (k + 1) % n];
                (va == a && vb == b) || (va == b && vb == a)
            })
        })
    };

    let mut cell_info = Vec::with_capacity(old_mesh.num_cells());

    // 2. Process each cell: keep convex cells, split concave ones
    for i in 0..old_mesh.num_cells() {
        let start = old_mesh.cell_vertex_offsets[i];
        let end = old_mesh.cell_vertex_offsets[i + 1];
        let n = end - start;
        let cell_verts: Vec<VertexId> = (start..end)
            .map(|k| old_vert_ids[old_mesh.cell_vertices[k]])
            .collect();

        if !is_concave(&old_mesh, i, tol) || !face_splittable(i) {
            // Keep cell as-is
            let cell_id = builder.add_cell(&cell_verts);
            cell_info.push(SplitInfo {
                is_split: false,
                new_cell_ids: vec![cell_id],
                center_vert_id: VertexId(0), // unused
            });
            continue;
        }

        // Split concave cell
        let gen = generators[i];

        // Check if generator coincides with a vertex
        let gen_match_dist = tol.boundary_eps;
        let mut match_idx = None;
        for k in 0..n {
            let v_id = cell_verts[k];
            let (vx, vy) = builder.vertex_pos(v_id);
            let dist = ((vx - gen.x).powi(2) + (vy - gen.y).powi(2)).sqrt();
            if dist < gen_match_dist {
                match_idx = Some(k);
                break;
            }
        }

        let mut sub_cells = vec![CellId(0); n];
        let center_vert_id;

        if let Some(root_k) = match_idx {
            // Fan from existing vertex
            center_vert_id = cell_verts[root_k];

            let mut k_iter = 1;
            while k_iter <= n - 2 {
                let u0 = cell_verts[root_k];
                let uk = cell_verts[(root_k + k_iter) % n];
                let uk1 = cell_verts[(root_k + k_iter + 1) % n];

                let mut merged = false;
                if k_iter + 1 <= n - 2 {
                    let uk2 = cell_verts[(root_k + k_iter + 2) % n];

                    let p0 = builder.vertex_point(u0);
                    let pk = builder.vertex_point(uk);
                    let pk1 = builder.vertex_point(uk1);
                    let pk2 = builder.vertex_point(uk2);

                    if is_poly_convex(&[p0, pk, pk1, pk2], tol) {
                        let new_cell = builder.add_cell(&[u0, uk, uk1, uk2]);
                        sub_cells[(root_k + k_iter) % n] = new_cell;
                        sub_cells[(root_k + k_iter + 1) % n] = new_cell;
                        if k_iter == 1 {
                            sub_cells[root_k] = new_cell;
                        }
                        if k_iter + 1 == n - 2 {
                            sub_cells[(root_k + n - 1) % n] = new_cell;
                        }
                        k_iter += 2;
                        merged = true;
                    }
                }

                if !merged {
                    let new_cell = builder.add_cell(&[u0, uk, uk1]);
                    sub_cells[(root_k + k_iter) % n] = new_cell;
                    if k_iter == 1 {
                        sub_cells[root_k] = new_cell;
                    }
                    if k_iter == n - 2 {
                        sub_cells[(root_k + n - 1) % n] = new_cell;
                    }
                    k_iter += 1;
                }
            }
        } else {
            // Fan from new center vertex (generator point)
            center_vert_id = builder.add_vertex(gen.x, gen.y, false);

            let mut k_iter = 0;
            while k_iter < n {
                let v1 = cell_verts[k_iter];
                let v2 = cell_verts[(k_iter + 1) % n];

                let mut merged = false;
                if k_iter + 1 < n {
                    let v3 = cell_verts[(k_iter + 2) % n];

                    let p_c = builder.vertex_point(center_vert_id);
                    let p_v1 = builder.vertex_point(v1);
                    let p_v2 = builder.vertex_point(v2);
                    let p_v3 = builder.vertex_point(v3);

                    if is_poly_convex(&[p_c, p_v1, p_v2, p_v3], tol) {
                        let new_cell = builder.add_cell(&[center_vert_id, v1, v2, v3]);
                        sub_cells[k_iter] = new_cell;
                        sub_cells[k_iter + 1] = new_cell;
                        k_iter += 2;
                        merged = true;
                    }
                }

                if !merged {
                    let new_cell = builder.add_cell(&[center_vert_id, v1, v2]);
                    sub_cells[k_iter] = new_cell;
                    k_iter += 1;
                }
            }
        }

        cell_info.push(SplitInfo {
            is_split: true,
            new_cell_ids: sub_cells,
            center_vert_id,
        });
    }

    // 3. Process faces
    // Helper: find sub-cell that owns edge (v1, v2) in a split old cell.
    // Split cells passed `face_splittable`, so the lookup always succeeds.
    let get_sub_cell = |old_c_idx: usize, v1_raw: usize, v2_raw: usize| -> CellId {
        let info = &cell_info[old_c_idx];
        if !info.is_split {
            return info.new_cell_ids[0];
        }
        let start = old_mesh.cell_vertex_offsets[old_c_idx];
        let end = old_mesh.cell_vertex_offsets[old_c_idx + 1];
        let n = end - start;
        for k in 0..n {
            let va = old_mesh.cell_vertices[start + k];
            let vb = old_mesh.cell_vertices[start + (k + 1) % n];
            if (va == v1_raw && vb == v2_raw) || (va == v2_raw && vb == v1_raw) {
                return info.new_cell_ids[k];
            }
        }
        unreachable!("face_splittable guaranteed every face maps to a ring pair");
    };

    // A. Re-create old faces, redirecting to new sub-cells
    for f_idx in 0..old_mesh.num_faces() {
        let v1_raw = old_mesh.face_v1[f_idx];
        let v2_raw = old_mesh.face_v2[f_idx];
        let old_owner = old_mesh.face_owner[f_idx];
        let old_neighbor = old_mesh.face_neighbor[f_idx];

        let new_owner = get_sub_cell(old_owner, v1_raw, v2_raw);
        let new_neighbor = old_neighbor.map(|n_idx| get_sub_cell(n_idx, v1_raw, v2_raw));

        builder.add_face(
            old_vert_ids[v1_raw],
            old_vert_ids[v2_raw],
            new_owner,
            new_neighbor,
            old_mesh.face_boundary[f_idx],
        );
    }

    // B. Create new internal faces for split cells
    for i in 0..old_mesh.num_cells() {
        let info = &cell_info[i];
        if !info.is_split {
            continue;
        }

        let center = info.center_vert_id;
        let start = old_mesh.cell_vertex_offsets[i];
        let end = old_mesh.cell_vertex_offsets[i + 1];
        let n = end - start;

        for k in 0..n {
            let v_raw = old_mesh.cell_vertices[start + k];
            let v_curr = old_vert_ids[v_raw];

            // Skip if center is one of the vertices (degenerate)
            if v_curr == center {
                continue;
            }

            let idx_k = k;
            let idx_prev = (k + n - 1) % n;

            let cell_k = info.new_cell_ids[idx_k];
            let cell_prev = info.new_cell_ids[idx_prev];

            // Skip if same cell (no face needed)
            if cell_k == cell_prev {
                continue;
            }

            builder.add_face(center, v_curr, cell_k, Some(cell_prev), None);
        }
    }

    builder.build()
}

fn is_concave(mesh: &Mesh, cell_idx: usize, tol: &MeshgenTolerances) -> bool {
    let start = mesh.cell_vertex_offsets[cell_idx];
    let end = mesh.cell_vertex_offsets[cell_idx + 1];
    let n = end - start;
    if n < 4 {
        return false;
    }

    let mut target_sign = 0.0;

    for i in 0..n {
        let idx_prev = mesh.cell_vertices[start + (i + n - 1) % n];
        let idx_curr = mesh.cell_vertices[start + i];
        let idx_next = mesh.cell_vertices[start + (i + 1) % n];

        let p_prev = Point2::new(mesh.vx[idx_prev], mesh.vy[idx_prev]);
        let p_curr = Point2::new(mesh.vx[idx_curr], mesh.vy[idx_curr]);
        let p_next = Point2::new(mesh.vx[idx_next], mesh.vy[idx_next]);

        let v1 = p_curr - p_prev;
        let v2 = p_next - p_curr;

        let cross = v1.x * v2.y - v1.y * v2.x;

        if cross.abs() > tol.cross_eps {
            if target_sign == 0.0 {
                target_sign = cross.signum();
            } else if cross.signum() != target_sign {
                return true;
            }
        }
    }
    false
}

fn is_poly_convex(verts: &[Point2<f64>], tol: &MeshgenTolerances) -> bool {
    let n = verts.len();
    if n < 3 {
        return true;
    }
    let mut target_sign = 0.0;
    for i in 0..n {
        let p_prev = verts[(i + n - 1) % n];
        let p_curr = verts[i];
        let p_next = verts[(i + 1) % n];

        let v1 = p_curr - p_prev;
        let v2 = p_next - p_curr;
        let cross = v1.x * v2.y - v1.y * v2.x;

        if cross.abs() > tol.cross_eps {
            if target_sign == 0.0 {
                target_sign = cross.signum();
            } else if cross.signum() != target_sign {
                return false;
            }
        }
    }
    true
}
