use super::delaunay::{triangulate, Edge};
use super::geometry::Geometry;
use super::structs::{BoundaryType, Mesh};
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

            let boundary_type = if f1_center.x < 1e-6 {
                Some(BoundaryType::Inlet)
            } else if (f1_center.x - domain_size.x).abs() < 1e-6 {
                Some(BoundaryType::Outlet)
            } else {
                Some(BoundaryType::Wall)
            };

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
            let boundary_type_2 = if f2_center.x < 1e-6 {
                Some(BoundaryType::Inlet)
            } else if (f2_center.x - domain_size.x).abs() < 1e-6 {
                Some(BoundaryType::Outlet)
            } else {
                Some(BoundaryType::Wall)
            };

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
        all_faces = edges.par_iter().flat_map(|edge| process_edge(edge)).collect();
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
        // We have a set of edges (faces). We need to order the vertices.
        // Build adjacency for this cell's vertices.
        let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
        for &f_idx in &cell_faces[i] {
            let v1 = mesh.face_v1[f_idx];
            let v2 = mesh.face_v2[f_idx];
            adj.entry(v1).or_default().push(v2);
            adj.entry(v2).or_default().push(v1);
        }

        // Find the start vertex.
        // If boundary cell, start at V (the cell center vertex).
        // If internal cell, pick any vertex.

        let start_node = if let Some(idx) = vertex_indices[i] {
            idx
        } else {
            // Internal cell. Pick any vertex.
            if let Some(&first) = adj.keys().next() {
                first
            } else {
                continue; // Should not happen
            }
        };

        let mut c_verts = Vec::new();
        let mut curr = start_node;
        let mut visited = HashSet::new();

        // Traverse the cycle
        // For boundary cells, V has degree 2 (connected to M1 and M2).
        // For internal cells, all vertices have degree 2 (simple polygon).
        // Just walk.

        // We need to handle the first step carefully to ensure we don't go back immediately.
        // But since it's a cycle, we just need to pick a neighbor and go.

        if let Some(neighbors) = adj.get(&curr) {
            if neighbors.is_empty() {
                continue;
            }

            c_verts.push(curr);
            visited.insert(curr);

            let mut next = neighbors[0];
            // If we are at V, we have 2 neighbors. Pick one.

            while next != start_node {
                c_verts.push(next);
                visited.insert(next);

                if let Some(next_neighbors) = adj.get(&next) {
                    // Find neighbor that is not the one we came from
                    // Note: 'curr' is where we came from.
                    let mut found = false;
                    for &n in next_neighbors {
                        if n != curr {
                            // Also check if we visited it already (unless it's start_node)
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
                        // Dead end or loop closed prematurely?
                        // If we reached start_node, loop terminates.
                        break;
                    }
                } else {
                    break;
                }
            }
        }

        // Ensure CCW (counter-clockwise) ordering for consistent polygon rendering.
        // The traversal above can produce either CW or CCW ordering depending on
        // which neighbor was arbitrarily chosen first. We fix this by calculating
        // the signed area (using the shoelace formula) and reversing if negative.
        // CCW polygons have positive signed area; CW polygons have negative.
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
            // If signed area is negative, vertices are CW - reverse to make CCW
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
    // Some boundary cells might be concave due to the way we handle boundary edges.
    // We split them into triangles to ensure convexity.
    mesh = fix_concave_cells(mesh, &points);

    mesh
}

struct SplitInfo {
    is_split: bool,
    new_cell_indices: Vec<usize>,
    center_vert_idx: usize,
}

fn fix_concave_cells(old_mesh: Mesh, generators: &[Point2<f64>]) -> Mesh {
    let mut new_mesh = Mesh::new();
    new_mesh.cell_face_offsets.push(0);
    new_mesh.cell_vertex_offsets.push(0);

    // 1. Copy Vertices
    new_mesh.vx = old_mesh.vx.clone();
    new_mesh.vy = old_mesh.vy.clone();
    new_mesh.v_fixed = old_mesh.v_fixed.clone();

    let mut cell_info = Vec::with_capacity(old_mesh.num_cells());

    // 2. Process Cells
    for i in 0..old_mesh.num_cells() {
        if is_concave(&old_mesh, i) {
            // Split
            let gen = generators[i];
            let start = old_mesh.cell_vertex_offsets[i];
            let end = old_mesh.cell_vertex_offsets[i + 1];
            let n = end - start;

            // Check if generator is a vertex
            let mut match_idx = None;
            for k in 0..n {
                let v_idx = old_mesh.cell_vertices[start + k];
                let p_v = Point2::new(new_mesh.vx[v_idx], new_mesh.vy[v_idx]);
                if (p_v - gen).norm() < 1e-6 {
                    match_idx = Some(k);
                    break;
                }
            }

            let mut sub_cells = vec![0; n];
            let center_idx;

            if let Some(root_k) = match_idx {
                // Fan from vertex logic
                center_idx = old_mesh.cell_vertices[start + root_k];
                
                let mut k_iter = 1;
                while k_iter <= n - 2 {
                    let u0 = old_mesh.cell_vertices[start + root_k];
                    let uk = old_mesh.cell_vertices[start + (root_k + k_iter) % n];
                    let uk1 = old_mesh.cell_vertices[start + (root_k + k_iter + 1) % n];
                    
                    let mut merged = false;
                    if k_iter + 1 <= n - 2 {
                        let uk2 = old_mesh.cell_vertices[start + (root_k + k_iter + 2) % n];
                        
                        // Check convexity of Quad (u0, uk, uk1, uk2)
                        let p0 = Point2::new(new_mesh.vx[u0], new_mesh.vy[u0]);
                        let pk = Point2::new(new_mesh.vx[uk], new_mesh.vy[uk]);
                        let pk1 = Point2::new(new_mesh.vx[uk1], new_mesh.vy[uk1]);
                        let pk2 = Point2::new(new_mesh.vx[uk2], new_mesh.vy[uk2]);
                        
                        if is_poly_convex(&[p0, pk, pk1, pk2]) {
                            // Create Quad
                            let new_cell_idx = new_mesh.num_cells();
                            new_mesh.cell_cx.push(0.0);
                            new_mesh.cell_cy.push(0.0);
                            new_mesh.cell_vol.push(0.0);
                            
                            new_mesh.cell_vertices.push(u0);
                            new_mesh.cell_vertices.push(uk);
                            new_mesh.cell_vertices.push(uk1);
                            new_mesh.cell_vertices.push(uk2);
                            new_mesh.cell_vertex_offsets.push(new_mesh.cell_vertices.len());
                            
                            // Assign edges
                            sub_cells[(root_k + k_iter) % n] = new_cell_idx;
                            sub_cells[(root_k + k_iter + 1) % n] = new_cell_idx;
                            
                            if k_iter == 1 {
                                sub_cells[root_k] = new_cell_idx;
                            }
                            if k_iter + 1 == n - 2 {
                                sub_cells[(root_k + n - 1) % n] = new_cell_idx;
                            }
                            
                            k_iter += 2;
                            merged = true;
                        }
                    }
                    
                    if !merged {
                        // Create Triangle
                        let new_cell_idx = new_mesh.num_cells();
                        new_mesh.cell_cx.push(0.0);
                        new_mesh.cell_cy.push(0.0);
                        new_mesh.cell_vol.push(0.0);
                        
                        new_mesh.cell_vertices.push(u0);
                        new_mesh.cell_vertices.push(uk);
                        new_mesh.cell_vertices.push(uk1);
                        new_mesh.cell_vertex_offsets.push(new_mesh.cell_vertices.len());
                        
                        sub_cells[(root_k + k_iter) % n] = new_cell_idx;
                        
                        if k_iter == 1 {
                            sub_cells[root_k] = new_cell_idx;
                        }
                        if k_iter == n - 2 {
                            sub_cells[(root_k + n - 1) % n] = new_cell_idx;
                        }
                        
                        k_iter += 1;
                    }
                }
            } else {
                // Fan from center logic
                center_idx = new_mesh.vx.len();
                new_mesh.vx.push(gen.x);
                new_mesh.vy.push(gen.y);
                new_mesh.v_fixed.push(false);

                let mut k_iter = 0; 
                while k_iter < n {
                    let v1 = old_mesh.cell_vertices[start + k_iter];
                    let v2 = old_mesh.cell_vertices[start + (k_iter + 1) % n];

                    let mut merged = false;
                    if k_iter + 1 < n {
                        let v3 = old_mesh.cell_vertices[start + (k_iter + 2) % n];

                        let p_c = Point2::new(new_mesh.vx[center_idx], new_mesh.vy[center_idx]);
                        let p_v1 = Point2::new(new_mesh.vx[v1], new_mesh.vy[v1]);
                        let p_v2 = Point2::new(new_mesh.vx[v2], new_mesh.vy[v2]);
                        let p_v3 = Point2::new(new_mesh.vx[v3], new_mesh.vy[v3]);

                        let quad_verts = vec![p_c, p_v1, p_v2, p_v3];
                        if is_poly_convex(&quad_verts) {
                            let new_cell_idx = new_mesh.num_cells();
                            new_mesh.cell_cx.push(0.0);
                            new_mesh.cell_cy.push(0.0);
                            new_mesh.cell_vol.push(0.0);

                            new_mesh.cell_vertices.push(center_idx);
                            new_mesh.cell_vertices.push(v1);
                            new_mesh.cell_vertices.push(v2);
                            new_mesh.cell_vertices.push(v3);
                            new_mesh
                                .cell_vertex_offsets
                                .push(new_mesh.cell_vertices.len());

                            sub_cells[k_iter] = new_cell_idx;
                            sub_cells[k_iter + 1] = new_cell_idx;

                            k_iter += 2;
                            merged = true;
                        }
                    }

                    if !merged {
                        let new_cell_idx = new_mesh.num_cells();
                        new_mesh.cell_cx.push(0.0);
                        new_mesh.cell_cy.push(0.0);
                        new_mesh.cell_vol.push(0.0);

                        new_mesh.cell_vertices.push(center_idx);
                        new_mesh.cell_vertices.push(v1);
                        new_mesh.cell_vertices.push(v2);
                        new_mesh
                            .cell_vertex_offsets
                            .push(new_mesh.cell_vertices.len());

                        sub_cells[k_iter] = new_cell_idx;
                        k_iter += 1;
                    }
                }
            }

            cell_info.push(SplitInfo {
                is_split: true,
                new_cell_indices: sub_cells,
                center_vert_idx: center_idx,
            });
        } else {
            // Keep
            let new_cell_idx = new_mesh.num_cells();
            new_mesh.cell_cx.push(old_mesh.cell_cx[i]);
            new_mesh.cell_cy.push(old_mesh.cell_cy[i]);
            new_mesh.cell_vol.push(old_mesh.cell_vol[i]);

            let start = old_mesh.cell_vertex_offsets[i];
            let end = old_mesh.cell_vertex_offsets[i + 1];
            for k in start..end {
                new_mesh.cell_vertices.push(old_mesh.cell_vertices[k]);
            }
            new_mesh
                .cell_vertex_offsets
                .push(new_mesh.cell_vertices.len());

            cell_info.push(SplitInfo {
                is_split: false,
                new_cell_indices: vec![new_cell_idx],
                center_vert_idx: 0,
            });
        }
    }

    // 3. Process Faces
    let mut new_cell_faces_temp = vec![Vec::new(); new_mesh.num_cells()];

    // Helper to find sub-cell
    let get_sub_cell = |old_c_idx: usize, v1: usize, v2: usize| -> usize {
        let info = &cell_info[old_c_idx];
        if !info.is_split {
            return info.new_cell_indices[0];
        }
        let start = old_mesh.cell_vertex_offsets[old_c_idx];
        let end = old_mesh.cell_vertex_offsets[old_c_idx + 1];
        let n = end - start;
        for k in 0..n {
            let va = old_mesh.cell_vertices[start + k];
            let vb = old_mesh.cell_vertices[start + (k + 1) % n];
            if (va == v1 && vb == v2) || (va == v2 && vb == v1) {
                return info.new_cell_indices[k];
            }
        }
        panic!("Edge not found in split cell");
    };

    // A. Old Faces
    for f_idx in 0..old_mesh.num_faces() {
        let v1 = old_mesh.face_v1[f_idx];
        let v2 = old_mesh.face_v2[f_idx];
        let old_owner = old_mesh.face_owner[f_idx];
        let old_neighbor = old_mesh.face_neighbor[f_idx];

        let new_owner = get_sub_cell(old_owner, v1, v2);
        let new_neighbor = if let Some(n_idx) = old_neighbor {
            Some(get_sub_cell(n_idx, v1, v2))
        } else {
            None
        };

        let new_f_idx = new_mesh.num_faces();
        new_mesh.face_v1.push(v1);
        new_mesh.face_v2.push(v2);
        new_mesh.face_cx.push(old_mesh.face_cx[f_idx]);
        new_mesh.face_cy.push(old_mesh.face_cy[f_idx]);
        new_mesh.face_nx.push(old_mesh.face_nx[f_idx]);
        new_mesh.face_ny.push(old_mesh.face_ny[f_idx]);
        new_mesh.face_area.push(old_mesh.face_area[f_idx]);
        new_mesh.face_boundary.push(old_mesh.face_boundary[f_idx]);
        new_mesh.face_owner.push(new_owner);
        new_mesh.face_neighbor.push(new_neighbor);

        new_cell_faces_temp[new_owner].push(new_f_idx);
        if let Some(n) = new_neighbor {
            new_cell_faces_temp[n].push(new_f_idx);
        }
    }

    // B. New Internal Faces
    for i in 0..old_mesh.num_cells() {
        if cell_info[i].is_split {
            let info = &cell_info[i];
            let center = info.center_vert_idx;
            let start = old_mesh.cell_vertex_offsets[i];
            let end = old_mesh.cell_vertex_offsets[i + 1];
            let n = end - start;

            for k in 0..n {
                let v_curr = old_mesh.cell_vertices[start + k];

                // Skip if degenerate (center is one of the vertices)
                if v_curr == center {
                    continue;
                }

                // Face between sub_cell[k] and sub_cell[prev]
                // Shared edge is (Center, v_curr)
                let idx_k = k;
                let idx_prev = (k + n - 1) % n;

                let cell_k = info.new_cell_indices[idx_k];
                let cell_prev = info.new_cell_indices[idx_prev];

                // Skip if same cell (e.g. internal edge of a Quad, or boundary edge in vertex fan)
                if cell_k == cell_prev {
                    continue;
                }

                let new_f_idx = new_mesh.num_faces();
                new_mesh.face_v1.push(center);
                new_mesh.face_v2.push(v_curr);
                // Placeholders
                new_mesh.face_cx.push(0.0);
                new_mesh.face_cy.push(0.0);
                new_mesh.face_nx.push(0.0);
                new_mesh.face_ny.push(0.0);
                new_mesh.face_area.push(0.0);
                new_mesh.face_boundary.push(None);

                new_mesh.face_owner.push(cell_k);
                new_mesh.face_neighbor.push(Some(cell_prev));

                new_cell_faces_temp[cell_k].push(new_f_idx);
                new_cell_faces_temp[cell_prev].push(new_f_idx);
            }
        }
    }

    // 4. Flatten cell_faces
    for faces in new_cell_faces_temp {
        new_mesh.cell_faces.extend(faces);
        new_mesh.cell_face_offsets.push(new_mesh.cell_faces.len());
    }

    new_mesh.recalculate_geometry();
    new_mesh
}

fn is_concave(mesh: &Mesh, cell_idx: usize) -> bool {
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

        if cross.abs() > 1e-12 {
            if target_sign == 0.0 {
                target_sign = cross.signum();
            } else if cross.signum() != target_sign {
                return true;
            }
        }
    }
    false
}

fn is_poly_convex(verts: &[Point2<f64>]) -> bool {
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

        if cross.abs() > 1e-12 {
            if target_sign == 0.0 {
                target_sign = cross.signum();
            } else if cross.signum() != target_sign {
                return false;
            }
        }
    }
    true
}
