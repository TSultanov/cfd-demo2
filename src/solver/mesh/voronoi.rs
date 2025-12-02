use super::delaunay::{triangulate, Edge};
use super::geometry::Geometry;
use super::structs::{BoundaryType, Mesh};
use nalgebra::{Point2, Vector2};
use std::collections::{HashMap, HashSet};

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
    // We use an edge-based approach to build faces first, then assemble cells.

    // We need to map Delaunay vertices to Voronoi cells.
    // cell_idx = delaunay_vertex_idx
    // So we can pre-allocate.
    for p in &points {
        mesh.cell_cx.push(p.x);
        mesh.cell_cy.push(p.y);
        // Vol will be calc later
        mesh.cell_vol.push(0.0);
    }

    // We need to store which faces belong to which cell.
    let mut cell_faces: Vec<Vec<usize>> = vec![Vec::new(); points.len()];

    // 3. Identify all unique Voronoi vertices.
    //    - Circumcenters of all triangles.
    //    - Midpoints of all boundary edges.
    //    - Original vertices (only for domain corners/boundary handling).

    let mut voronoi_points: Vec<Point2<f64>> = Vec::new();
    let mut voronoi_point_map: HashMap<String, usize> = HashMap::new(); // Key: "C_tidx" or "M_eidx" or "V_vidx"

    // Helper to add point
    let mut add_point = |p: Point2<f64>, key: String| -> usize {
        if let Some(&idx) = voronoi_point_map.get(&key) {
            idx
        } else {
            let idx = voronoi_points.len();
            voronoi_points.push(p);
            voronoi_point_map.insert(key, idx);
            idx
        }
    };

    // Add circumcenters
    for (i, t) in triangles.iter().enumerate() {
        add_point(t.circumcenter, format!("C_{}", i));
    }

    // Add midpoints for boundary edges
    // And add original vertices for boundary cells
    for (edge, tris) in &edge_to_triangles {
        if tris.len() == 1 {
            let p1 = points[edge.v1];
            let p2 = points[edge.v2];
            let mid = Point2::new((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0);
            add_point(mid, format!("M_{}_{}", edge.v1, edge.v2));

            // Also ensure the original vertices are added if they are part of the boundary cell
            add_point(p1, format!("V_{}", edge.v1));
            add_point(p2, format!("V_{}", edge.v2));
        }
    }

    // Now create faces
    for (edge, tris) in &edge_to_triangles {
        let v1 = edge.v1;
        let v2 = edge.v2;

        let idx_a;
        let idx_b;

        if tris.len() == 2 {
            idx_a = *voronoi_point_map.get(&format!("C_{}", tris[0])).unwrap();
            idx_b = *voronoi_point_map.get(&format!("C_{}", tris[1])).unwrap();
        } else {
            idx_a = *voronoi_point_map.get(&format!("C_{}", tris[0])).unwrap();
            idx_b = *voronoi_point_map.get(&format!("M_{}_{}", v1, v2)).unwrap();
        }

        let pa = voronoi_points[idx_a];
        let pb = voronoi_points[idx_b];

        let f_center = Point2::new((pa.x + pb.x) / 2.0, (pa.y + pb.y) / 2.0);
        let f_len = (pa - pb).norm();

        // Normal should point from v1 to v2 (or vice versa).
        // The Delaunay edge (v1 -> v2) is normal to the Voronoi face.
        let p_v1 = points[v1];
        let p_v2 = points[v2];
        let del_edge_vec = p_v2 - p_v1;
        let normal = del_edge_vec.normalize();

        let face_idx = mesh.face_cx.len();

        mesh.face_v1.push(idx_a);
        mesh.face_v2.push(idx_b);
        mesh.face_cx.push(f_center.x);
        mesh.face_cy.push(f_center.y);
        mesh.face_nx.push(normal.x);
        mesh.face_ny.push(normal.y);
        mesh.face_area.push(f_len);

        // Owner/Neighbor are the Delaunay vertices v1 and v2
        mesh.face_owner.push(v1);
        mesh.face_neighbor.push(Some(v2));

        mesh.face_boundary.push(None); // Internal face

        cell_faces[v1].push(face_idx);
        cell_faces[v2].push(face_idx);
    }

    // Add Boundary Faces (segments along the domain boundary)
    // Iterate over boundary edges of Delaunay
    for (edge, tris) in &edge_to_triangles {
        if tris.len() == 1 {
            let v1 = edge.v1;
            let v2 = edge.v2;

            let idx_mid = *voronoi_point_map.get(&format!("M_{}_{}", v1, v2)).unwrap();
            let idx_v1 = *voronoi_point_map.get(&format!("V_{}", v1)).unwrap();
            let idx_v2 = *voronoi_point_map.get(&format!("V_{}", v2)).unwrap();

            let p_mid = voronoi_points[idx_mid];
            let p_v1 = voronoi_points[idx_v1];
            let p_v2 = voronoi_points[idx_v2];

            // Face 1: Midpoint - V1
            let face_idx_1 = mesh.face_cx.len();
            mesh.face_v1.push(idx_mid);
            mesh.face_v2.push(idx_v1);
            let f1_center = Point2::new((p_mid.x + p_v1.x) / 2.0, (p_mid.y + p_v1.y) / 2.0);
            mesh.face_cx.push(f1_center.x);
            mesh.face_cy.push(f1_center.y);
            mesh.face_area.push((p_mid - p_v1).norm());

            let tangent = p_v2 - p_v1;
            let mut normal = Vector2::new(tangent.y, -tangent.x).normalize();

            // Check against triangle centroid to ensure it points out
            let t = triangles[tris[0]];
            let t_center = (points[t.v1].coords + points[t.v2].coords + points[t.v3].coords) / 3.0;
            let edge_center = (p_v1.coords + p_v2.coords) / 2.0;
            if (edge_center - t_center).dot(&normal) < 0.0 {
                normal = -normal;
            }

            mesh.face_nx.push(normal.x);
            mesh.face_ny.push(normal.y);

            mesh.face_owner.push(v1);
            mesh.face_neighbor.push(None); // Boundary

            // Determine Boundary Type
            let boundary_type = if f1_center.x < 1e-6 {
                Some(BoundaryType::Inlet)
            } else if (f1_center.x - domain_size.x).abs() < 1e-6 {
                Some(BoundaryType::Outlet)
            } else {
                Some(BoundaryType::Wall)
            };
            mesh.face_boundary.push(boundary_type);

            cell_faces[v1].push(face_idx_1);

            // Face 2: Midpoint - V2
            let face_idx_2 = mesh.face_cx.len();
            mesh.face_v1.push(idx_mid);
            mesh.face_v2.push(idx_v2);
            let f2_center = Point2::new((p_mid.x + p_v2.x) / 2.0, (p_mid.y + p_v2.y) / 2.0);
            mesh.face_cx.push(f2_center.x);
            mesh.face_cy.push(f2_center.y);
            mesh.face_area.push((p_mid - p_v2).norm());

            mesh.face_nx.push(normal.x);
            mesh.face_ny.push(normal.y);

            mesh.face_owner.push(v2);
            mesh.face_neighbor.push(None);

            let boundary_type_2 = if f2_center.x < 1e-6 {
                Some(BoundaryType::Inlet)
            } else if (f2_center.x - domain_size.x).abs() < 1e-6 {
                Some(BoundaryType::Outlet)
            } else {
                Some(BoundaryType::Wall)
            };
            mesh.face_boundary.push(boundary_type_2);

            cell_faces[v2].push(face_idx_2);
        }
    }

    // 3. Finalize Mesh
    mesh.vx = voronoi_points.iter().map(|p| p.x).collect();
    mesh.vy = voronoi_points.iter().map(|p| p.y).collect();

    // Mark boundary vertices as fixed
    mesh.v_fixed = vec![false; mesh.vx.len()];
    for (key, &idx) in &voronoi_point_map {
        if key.starts_with("M_") || key.starts_with("V_") {
            mesh.v_fixed[idx] = true;
        }
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

        let start_node = if let Some(&idx) = voronoi_point_map.get(&format!("V_{}", i)) {
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

        mesh.cell_vertices.extend(&c_verts);
        mesh.cell_vertex_offsets.push(mesh.cell_vertices.len());
    }

    // Recalculate geometry to ensure areas and centroids are correct
    mesh.recalculate_geometry();

    // 4. Fix Concave Cells
    // Some boundary cells might be concave due to the way we handle boundary edges.
    // We split them into triangles to ensure convexity.
    // mesh = fix_concave_cells(mesh, &points);

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
            let center_idx = new_mesh.vx.len();
            new_mesh.vx.push(gen.x);
            new_mesh.vy.push(gen.y);
            // Internal node for splitting, treat as not fixed
            new_mesh.v_fixed.push(false);

            let start = old_mesh.cell_vertex_offsets[i];
            let end = old_mesh.cell_vertex_offsets[i + 1];
            let n = end - start;

            let mut sub_cells = vec![0; n]; // Initialize with dummy values, will be filled
            let mut k_iter = 0; // This iterator tracks the original edges (0 to n-1)
            while k_iter < n {
                let v1 = old_mesh.cell_vertices[start + k_iter];
                let v2 = old_mesh.cell_vertices[start + (k_iter + 1) % n];

                // Potential Quad?
                // Try to merge with next triangle (Center, v2, v3)
                // Quad would be (Center, v1, v2, v3) -> vertices: Center, v1, v2, v3
                // Wait, if we merge (Center, v1, v2) and (Center, v2, v3)
                // The shared edge is (Center, v2).
                // The quad vertices are Center, v1, v2, v3.
                // Is this correct?
                // Triangle 1: C, v1, v2
                // Triangle 2: C, v2, v3
                // Merged: C, v1, v2, v3.
                // We need to check if this polygon is convex.

                let mut merged = false;
                // Check if there's a next edge to potentially merge with
                if k_iter + 1 < n {
                    let v3 = old_mesh.cell_vertices[start + (k_iter + 2) % n];

                    // Check convexity of Quad (Center, v1, v2, v3)
                    let p_c = Point2::new(new_mesh.vx[center_idx], new_mesh.vy[center_idx]);
                    let p_v1 = Point2::new(new_mesh.vx[v1], new_mesh.vy[v1]);
                    let p_v2 = Point2::new(new_mesh.vx[v2], new_mesh.vy[v2]);
                    let p_v3 = Point2::new(new_mesh.vx[v3], new_mesh.vy[v3]);

                    // Vertices must be in order for is_poly_convex
                    let quad_verts = vec![p_c, p_v1, p_v2, p_v3];
                    if is_poly_convex(&quad_verts) {
                        // Create Quad
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

                        // Assign this new cell to both original edges k_iter and k_iter+1
                        sub_cells[k_iter] = new_cell_idx;
                        sub_cells[k_iter + 1] = new_cell_idx;

                        k_iter += 2; // Advance iterator by 2 as two edges were consumed
                        merged = true;
                    }
                }

                if !merged {
                    // Create Triangle
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

                    // Assign this new cell to the original edge k_iter
                    sub_cells[k_iter] = new_cell_idx;
                    k_iter += 1; // Advance iterator by 1
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

                // Face between sub_cell[k] and sub_cell[prev]
                // Shared edge is (Center, v_curr)
                let idx_k = k;
                let idx_prev = (k + n - 1) % n;

                let cell_k = info.new_cell_indices[idx_k];
                let cell_prev = info.new_cell_indices[idx_prev];

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
