use super::delaunay::{triangulate, Edge};
use super::geometry::Geometry;
use super::mesh_builder::{CellId, MeshBuilder, VertexId};
use super::tolerances::MeshgenTolerances;
use crate::solver::mesh::{BoundaryType, Mesh};
use nalgebra::{Point2, Vector2};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

struct FaceResult {
    v1: usize,
    v2: usize,
    cx: f64,
    cy: f64,
    nx: f64,
    ny: f64,
    area: f64,
    owner: usize,
    neighbor: Option<usize>,
    boundary: Option<BoundaryType>,
    cell_1: usize,
    cell_2: Option<usize>,
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

    let mut mesh = Mesh::new();
    mesh.cell_face_offsets.push(0);
    mesh.cell_vertex_offsets.push(0);

    // 1. Build Adjacency Maps
    // Edge -> Triangles
    let mut edge_to_triangles: HashMap<Edge, Vec<usize>> = HashMap::new();
    // Vertex -> Edges
    let mut vertex_to_edges: Vec<Vec<Edge>> = vec![Vec::new(); points.len()];

    for (t_idx, t) in triangles.iter().enumerate() {
        let edges = [
            Edge::new(t.v1, t.v2),
            Edge::new(t.v2, t.v3),
            Edge::new(t.v3, t.v1),
        ];

        for &edge in &edges {
            edge_to_triangles.entry(edge).or_default().push(t_idx);
        }
    }

    for (e, _) in &edge_to_triangles {
        vertex_to_edges[e.v1].push(*e);
        vertex_to_edges[e.v2].push(*e);
    }

    // 2. Construct Voronoi Cells (one per Delaunay vertex)
    for p in &points {
        mesh.cell_cx.push(p.x);
        mesh.cell_cy.push(p.y);
        mesh.cell_vol.push(0.0);
    }

    let mut cell_faces: Vec<Vec<usize>> = vec![Vec::new(); points.len()];

    // 3. Identify all unique Voronoi vertices.
    let mut voronoi_points: Vec<Point2<f64>> = Vec::new();

    // Direct mappings
    let mut circumcenter_indices: Vec<usize> = Vec::with_capacity(triangles.len());
    let mut vertex_indices: Vec<Option<usize>> = vec![None; points.len()];
    let mut midpoint_indices: HashMap<Edge, usize> = HashMap::new();

    // Add circumcenters
    for t in triangles.iter() {
        circumcenter_indices.push(voronoi_points.len());
        voronoi_points.push(t.circumcenter);
    }

    // Add midpoints and original vertices
    for (edge, tris) in &edge_to_triangles {
        if tris.len() == 1 {
            let p1 = points[edge.v1];
            let p2 = points[edge.v2];
            let mid = Point2::new((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0);

            midpoint_indices.insert(*edge, voronoi_points.len());
            voronoi_points.push(mid);

            if vertex_indices[edge.v1].is_none() {
                vertex_indices[edge.v1] = Some(voronoi_points.len());
                voronoi_points.push(p1);
            }
            if vertex_indices[edge.v2].is_none() {
                vertex_indices[edge.v2] = Some(voronoi_points.len());
                voronoi_points.push(p2);
            }
        }
    }

    // Sort edges for deterministic parallel execution
    let mut edges: Vec<Edge> = edge_to_triangles.keys().cloned().collect();
    edges.sort_by(|a, b| a.v1.cmp(&b.v1).then(a.v2.cmp(&b.v2)));

    // Parallel Face Generation
    let process_edge = |edge: &Edge| -> Vec<FaceResult> {
        let tris = &edge_to_triangles[edge];
        let mut results = Vec::new();
        let v1 = edge.v1;
        let v2 = edge.v2;

        // 1. Main Face
        let idx_a;
        let idx_b;

        if tris.len() == 2 {
            idx_a = circumcenter_indices[tris[0]];
            idx_b = circumcenter_indices[tris[1]];
        } else {
            idx_a = circumcenter_indices[tris[0]];
            idx_b = *midpoint_indices.get(edge).unwrap();
        }

        let pa = voronoi_points[idx_a];
        let pb = voronoi_points[idx_b];
        let f_center = Point2::new((pa.x + pb.x) / 2.0, (pa.y + pb.y) / 2.0);
        let f_len = (pa - pb).norm();

        let p_v1 = points[v1];
        let p_v2 = points[v2];
        let del_edge_vec = p_v2 - p_v1;
        let normal = del_edge_vec.normalize();

        results.push(FaceResult {
            v1: idx_a,
            v2: idx_b,
            cx: f_center.x,
            cy: f_center.y,
            nx: normal.x,
            ny: normal.y,
            area: f_len,
            owner: v1,
            neighbor: Some(v2),
            boundary: None,
            cell_1: v1,
            cell_2: Some(v2),
        });

        // 2. Boundary Faces
        if tris.len() == 1 {
            let idx_mid = *midpoint_indices.get(edge).unwrap();
            let idx_v1 = vertex_indices[v1].unwrap();
            let idx_v2 = vertex_indices[v2].unwrap();

            let p_mid = voronoi_points[idx_mid];
            let p_v1_vor = voronoi_points[idx_v1];
            let p_v2_vor = voronoi_points[idx_v2];

            // Face 1: Midpoint - V1
            let f1_center = Point2::new((p_mid.x + p_v1_vor.x) / 2.0, (p_mid.y + p_v1_vor.y) / 2.0);
            let tangent = p_v2 - p_v1;
            let mut normal = Vector2::new(tangent.y, -tangent.x).normalize();

            let t = triangles[tris[0]];
            let t_center = (points[t.v1].coords + points[t.v2].coords + points[t.v3].coords) / 3.0;
            let edge_center = (p_v1.coords + p_v2.coords) / 2.0;
            if (edge_center - t_center).dot(&normal) < 0.0 {
                normal = -normal;
            }

            let boundary_type =
                tol.classify_boundary(f1_center.x, f1_center.y, domain_size.x, domain_size.y);

            results.push(FaceResult {
                v1: idx_mid,
                v2: idx_v1,
                cx: f1_center.x,
                cy: f1_center.y,
                nx: normal.x,
                ny: normal.y,
                area: (p_mid - p_v1_vor).norm(),
                owner: v1,
                neighbor: None,
                boundary: boundary_type,
                cell_1: v1,
                cell_2: None,
            });

            // Face 2: Midpoint - V2
            let f2_center = Point2::new((p_mid.x + p_v2_vor.x) / 2.0, (p_mid.y + p_v2_vor.y) / 2.0);
            let boundary_type_2 =
                tol.classify_boundary(f2_center.x, f2_center.y, domain_size.x, domain_size.y);

            results.push(FaceResult {
                v1: idx_mid,
                v2: idx_v2,
                cx: f2_center.x,
                cy: f2_center.y,
                nx: normal.x,
                ny: normal.y,
                area: (p_mid - p_v2_vor).norm(),
                owner: v2,
                neighbor: None,
                boundary: boundary_type_2,
                cell_1: v2,
                cell_2: None,
            });
        }

        results
    };

    let all_faces: Vec<FaceResult>;
    if edges.len() < 5000 {
        all_faces = edges.iter().flat_map(|edge| process_edge(edge)).collect();
    } else {
        all_faces = edges
            .par_iter()
            .flat_map(|edge| process_edge(edge))
            .collect();
    }

    // Push faces to mesh
    for f in all_faces {
        let f_idx = mesh.face_cx.len();
        mesh.face_v1.push(f.v1);
        mesh.face_v2.push(f.v2);
        mesh.face_cx.push(f.cx);
        mesh.face_cy.push(f.cy);
        mesh.face_nx.push(f.nx);
        mesh.face_ny.push(f.ny);
        mesh.face_area.push(f.area);
        mesh.face_owner.push(f.owner);
        mesh.face_neighbor.push(f.neighbor);
        mesh.face_boundary.push(f.boundary);

        cell_faces[f.cell_1].push(f_idx);
        if let Some(c2) = f.cell_2 {
            cell_faces[c2].push(f_idx);
        }
    }

    // 3. Finalize Mesh
    mesh.vx = voronoi_points.iter().map(|p| p.x).collect();
    mesh.vy = voronoi_points.iter().map(|p| p.y).collect();

    // Mark boundary vertices as fixed
    mesh.v_fixed = vec![false; mesh.vx.len()];
    for idx in midpoint_indices.values() {
        mesh.v_fixed[*idx] = true;
    }
    for idx in vertex_indices.iter().flatten() {
        mesh.v_fixed[*idx] = true;
    }

    // Fill cell_faces and calculate volumes
    for i in 0..points.len() {
        mesh.cell_faces.extend(&cell_faces[i]);
        mesh.cell_face_offsets.push(mesh.cell_faces.len());

        // Reconstruct cell polygon by chaining faces
        let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
        for &f_idx in &cell_faces[i] {
            let v1 = mesh.face_v1[f_idx];
            let v2 = mesh.face_v2[f_idx];
            adj.entry(v1).or_default().push(v2);
            adj.entry(v2).or_default().push(v1);
        }

        let start_node = if let Some(idx) = vertex_indices[i] {
            idx
        } else {
            // Internal cell. Pick any vertex.
            if let Some(&first) = adj.keys().next() {
                first
            } else {
                mesh.cell_vertex_offsets.push(mesh.cell_vertices.len());
                continue;
            }
        };

        let mut c_verts = Vec::new();
        let mut curr = start_node;
        let mut visited = HashSet::new();

        if let Some(neighbors) = adj.get(&curr) {
            if neighbors.is_empty() {
                mesh.cell_vertex_offsets.push(mesh.cell_vertices.len());
                continue;
            }

            c_verts.push(curr);
            visited.insert(curr);

            let mut next = neighbors[0];

            while next != start_node {
                c_verts.push(next);
                visited.insert(next);

                if let Some(next_neighbors) = adj.get(&next) {
                    let mut found = false;
                    for &n in next_neighbors {
                        if n != curr {
                            if n == start_node {
                                found = true;
                                curr = next;
                                next = n;
                                break;
                            }
                            if !visited.contains(&n) {
                                found = true;
                                curr = next;
                                next = n;
                                break;
                            }
                        }
                    }
                    if !found {
                        break;
                    }
                } else {
                    break;
                }
            }
        }

        // Ensure CCW ordering
        if c_verts.len() >= 3 {
            let mut signed_area = 0.0;
            let n = c_verts.len();
            for k in 0..n {
                let v_idx0 = c_verts[k];
                let v_idx1 = c_verts[(k + 1) % n];
                let p0_x = voronoi_points[v_idx0].x;
                let p0_y = voronoi_points[v_idx0].y;
                let p1_x = voronoi_points[v_idx1].x;
                let p1_y = voronoi_points[v_idx1].y;
                signed_area += p0_x * p1_y - p1_x * p0_y;
            }
            if signed_area < 0.0 {
                c_verts.reverse();
            }
        }

        mesh.cell_vertices.extend(&c_verts);
        mesh.cell_vertex_offsets.push(mesh.cell_vertices.len());
    }

    // Recalculate geometry to ensure areas and centroids are correct
    mesh.recalculate_geometry();

    // 4. Fix Concave Cells
    mesh = fix_concave_cells(mesh, &points, &tol);

    mesh
}

struct SplitInfo {
    is_split: bool,
    /// For split cells: sub_cells[k] = CellId of the sub-cell that owns edge k.
    /// For non-split cells: single element with the CellId.
    new_cell_ids: Vec<CellId>,
    center_vert_id: VertexId,
}

fn fix_concave_cells(old_mesh: Mesh, generators: &[Point2<f64>], tol: &MeshgenTolerances) -> Mesh {
    let mut builder = MeshBuilder::with_capacity(
        old_mesh.num_vertices() + old_mesh.num_cells(),
        old_mesh.num_cells() * 2,
        old_mesh.num_faces() * 2,
    );

    // 1. Copy all existing vertices into the builder
    let old_vert_ids: Vec<VertexId> = (0..old_mesh.num_vertices())
        .map(|i| builder.add_vertex(old_mesh.vx[i], old_mesh.vy[i], old_mesh.v_fixed[i]))
        .collect();

    let mut cell_info = Vec::with_capacity(old_mesh.num_cells());

    // 2. Process each cell: keep convex cells, split concave ones
    for i in 0..old_mesh.num_cells() {
        let start = old_mesh.cell_vertex_offsets[i];
        let end = old_mesh.cell_vertex_offsets[i + 1];
        let n = end - start;
        let cell_verts: Vec<VertexId> = (start..end)
            .map(|k| old_vert_ids[old_mesh.cell_vertices[k]])
            .collect();

        if !is_concave(&old_mesh, i, tol) {
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
    // Helper: find sub-cell that owns edge (v1, v2) in a split old cell
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
        panic!("Edge not found in split cell");
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
