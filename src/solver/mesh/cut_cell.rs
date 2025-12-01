use nalgebra::{Point2, Vector2};
use rayon::prelude::*;
use std::collections::HashMap;
use wide::{f64x4, CmpGe, CmpGt, CmpLt};

use super::geometry::Geometry;
use super::quadtree::generate_base_polygons;
use super::structs::{BoundaryType, Mesh};

pub fn generate_cut_cell_mesh(
    geo: &(impl Geometry + Sync),
    min_cell_size: f64,
    max_cell_size: f64,
    growth_rate: f64,
    domain_size: Vector2<f64>,
) -> Mesh {
    let mut mesh = Mesh::new();

    // 1. Build Quadtree and Process Leaves
    let mut all_polys =
        generate_base_polygons(geo, min_cell_size, max_cell_size, growth_rate, domain_size);

    // 2. Imprint Hanging Nodes
    // Robustly merge vertices to fix holes
    let mut points: Vec<(Point2<f64>, bool)> = Vec::new();
    let mut poly_point_indices: Vec<Vec<usize>> = Vec::with_capacity(all_polys.len());

    for poly in &all_polys {
        let mut indices = Vec::with_capacity(poly.len());
        for (p, fixed) in poly {
            indices.push(points.len());
            points.push((*p, *fixed));
        }
        poly_point_indices.push(indices);
    }

    // Merge points using spatial hashing
    let mut unique_verts: Vec<(Point2<f64>, bool)> = Vec::new();
    let mut point_remap = vec![0; points.len()];
    let merge_tol = 1e-5;
    let grid_cell = merge_tol * 2.0;
    let mut merge_grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();

    for (i, (p, fixed)) in points.iter().enumerate() {
        let gx = (p.x / grid_cell).floor() as i64;
        let gy = (p.y / grid_cell).floor() as i64;

        let mut found = None;

        // Check 3x3 neighbors
        'search: for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(candidates) = merge_grid.get(&(gx + dx, gy + dy)) {
                    for &idx in candidates {
                        let (u_p, u_fixed) = unique_verts[idx];
                        let diff: Vector2<f64> = *p - u_p;
                        if diff.norm_squared() < merge_tol * merge_tol {
                            // Found match
                            if *fixed && !u_fixed {
                                unique_verts[idx].1 = true;
                            }
                            found = Some(idx);
                            break 'search;
                        }
                    }
                }
            }
        }

        if let Some(idx) = found {
            point_remap[i] = idx;
        } else {
            let idx = unique_verts.len();
            unique_verts.push((*p, *fixed));
            point_remap[i] = idx;
            merge_grid.entry((gx, gy)).or_default().push(idx);
        }
    }

    // Update polygons with merged vertices
    for (i, poly) in all_polys.iter_mut().enumerate() {
        for (j, item) in poly.iter_mut().enumerate() {
            let old_idx = poly_point_indices[i][j];
            let new_idx = point_remap[old_idx];
            *item = unique_verts[new_idx];
        }
    }

    let quantize = |v: f64| (v * 100000.0).round() as i64;

    // Map (v_min, v_max) -> face_index
    let mut face_map: HashMap<(usize, usize), usize> = HashMap::new();
    let mut vertex_map: HashMap<(i64, i64), usize> = HashMap::new();

    // Spatial Grid with SoA layout for SIMD
    let grid_size = max_cell_size;
    let grid_nx = (domain_size.x / grid_size).ceil() as usize + 1;
    let grid_ny = (domain_size.y / grid_size).ceil() as usize + 1;
    let grid_len = grid_nx * grid_ny;

    // 1. Count vertices per cell
    let mut grid_counts = vec![0; grid_len];
    let grid_indices: Vec<usize> = unique_verts
        .iter()
        .map(|(p, _)| {
            let gx = (p.x / grid_size).floor().max(0.0) as usize;
            let gy = (p.y / grid_size).floor().max(0.0) as usize;
            if gx < grid_nx && gy < grid_ny {
                let idx = gy * grid_nx + gx;
                grid_counts[idx] += 1;
                idx
            } else {
                grid_len
            }
        })
        .collect();

    // 2. Prefix sums for start indices
    let mut grid_starts = vec![0; grid_len + 1];
    let mut current = 0;
    for i in 0..grid_len {
        grid_starts[i] = current;
        current += grid_counts[i];
    }
    grid_starts[grid_len] = current;

    // 3. Fill SoA arrays
    let mut sorted_xs = vec![0.0; unique_verts.len()];
    let mut sorted_ys = vec![0.0; unique_verts.len()];
    let mut sorted_fixed = vec![false; unique_verts.len()];

    let mut current_starts = grid_starts.clone();
    for (i, (p, fixed)) in unique_verts.iter().enumerate() {
        let grid_idx = grid_indices[i];
        if grid_idx < grid_len {
            let pos = current_starts[grid_idx];
            sorted_xs[pos] = p.x;
            sorted_ys[pos] = p.y;
            sorted_fixed[pos] = *fixed;
            current_starts[grid_idx] += 1;
        }
    }

    // For each polygon, check if any unique vertex lies on its edges
    all_polys.par_iter_mut().for_each(|poly| {
        let mut new_poly = Vec::new();
        let n = poly.len();

        for k in 0..n {
            let (p_curr, fixed_curr) = poly[k];
            let (p_next, _) = poly[(k + 1) % n];

            new_poly.push((p_curr, fixed_curr));

            // Find vertices on this segment
            let mut on_segment = Vec::new();
            let seg_vec = p_next - p_curr;
            let seg_len_sq = seg_vec.norm_squared();

            if seg_len_sq < 1e-12 {
                continue;
            }

            // SIMD constants
            let p_curr_x = f64x4::splat(p_curr.x);
            let p_curr_y = f64x4::splat(p_curr.y);
            let p_next_x = f64x4::splat(p_next.x);
            let p_next_y = f64x4::splat(p_next.y);
            let seg_vec_x = f64x4::splat(seg_vec.x);
            let seg_vec_y = f64x4::splat(seg_vec.y);
            let seg_len_sq_simd = f64x4::splat(seg_len_sq);
            let epsilon = f64x4::splat(1e-10); // Increased tolerance
            let t_min = f64x4::splat(1e-6);
            let t_max = f64x4::splat(1.0 - 1e-6);

            // Bounding box of the segment
            let min_x = p_curr.x.min(p_next.x);
            let max_x = p_curr.x.max(p_next.x);
            let min_y = p_curr.y.min(p_next.y);
            let max_y = p_curr.y.max(p_next.y);

            let min_gx = (min_x / grid_size).floor().max(0.0) as usize;
            let max_gx = (max_x / grid_size).floor().max(0.0) as usize;
            let min_gy = (min_y / grid_size).floor().max(0.0) as usize;
            let max_gy = (max_y / grid_size).floor().max(0.0) as usize;

            for gy in min_gy..=max_gy.min(grid_ny - 1) {
                for gx in min_gx..=max_gx.min(grid_nx - 1) {
                    let cell_idx = gy * grid_nx + gx;
                    let start = grid_starts[cell_idx];
                    let end = grid_starts[cell_idx + 1];

                    if start == end {
                        continue;
                    }

                    let xs = &sorted_xs[start..end];
                    let ys = &sorted_ys[start..end];
                    let fixeds = &sorted_fixed[start..end];

                    let mut i = 0;
                    let chunks_count = xs.len() / 4;

                    // SIMD Loop
                    for _ in 0..chunks_count {
                        let v_x = f64x4::from(&xs[i..i + 4]);
                        let v_y = f64x4::from(&ys[i..i + 4]);

                        let dx_curr = v_x - p_curr_x;
                        let dy_curr = v_y - p_curr_y;
                        let d_curr = dx_curr * dx_curr + dy_curr * dy_curr;

                        let dx_next = v_x - p_next_x;
                        let dy_next = v_y - p_next_y;
                        let d_next = dx_next * dx_next + dy_next * dy_next;

                        // Check endpoints
                        let not_endpoint = d_curr.simd_ge(epsilon) & d_next.simd_ge(epsilon);

                        if not_endpoint.none() {
                            i += 4;
                            continue;
                        }

                        let dot = dx_curr * seg_vec_x + dy_curr * seg_vec_y;
                        let t = dot / seg_len_sq_simd;

                        let t_in_range = t.simd_gt(t_min) & t.simd_lt(t_max);
                        let mask = not_endpoint & t_in_range;

                        if mask.none() {
                            i += 4;
                            continue;
                        }

                        let proj_x = p_curr_x + seg_vec_x * t;
                        let proj_y = p_curr_y + seg_vec_y * t;

                        let d_proj_x = v_x - proj_x;
                        let d_proj_y = v_y - proj_y;
                        let dist_sq = d_proj_x * d_proj_x + d_proj_y * d_proj_y;

                        let is_close = dist_sq.simd_lt(epsilon);
                        let final_mask = mask & is_close;

                        if final_mask.any() {
                            let t_arr = t.to_array();
                            let mask_int = final_mask.to_bitmask();
                            for (lane, &t_val) in t_arr.iter().enumerate() {
                                if (mask_int & (1 << lane)) != 0 {
                                    let idx = i + lane;
                                    on_segment.push((
                                        t_val,
                                        Point2::new(xs[idx], ys[idx]),
                                        fixeds[idx],
                                    ));
                                }
                            }
                        }
                        i += 4;
                    }

                    // Remainder Loop
                    for j in i..xs.len() {
                        let v = Point2::new(xs[j], ys[j]);
                        let is_fixed = fixeds[j];

                        let d_curr = (v - p_curr).norm_squared();
                        let d_next = (v - p_next).norm_squared();

                        if d_curr < 1e-10 || d_next < 1e-10 {
                            continue;
                        }

                        let v_vec = v - p_curr;
                        let t = v_vec.dot(&seg_vec) / seg_len_sq;

                        if t > 1e-6 && t < 1.0 - 1e-6 {
                            let proj = p_curr + seg_vec * t;
                            if (v - proj).norm_squared() < 1e-10 {
                                on_segment.push((t, v, is_fixed));
                            }
                        }
                    }
                }
            }

            // Sort by t
            on_segment.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            for (_, v, is_fixed) in on_segment {
                new_poly.push((v, is_fixed));
            }
        }
        *poly = new_poly;
    });
    // 3. Create Mesh from Polygons
    // Populate mesh vertices from unique_verts
    for (i, (p, fixed)) in unique_verts.iter().enumerate() {
        mesh.vx.push(p.x);
        mesh.vy.push(p.y);
        mesh.v_fixed.push(*fixed);
        let key = (quantize(p.x), quantize(p.y));
        vertex_map.insert(key, i);
    }

    let n_polys = all_polys.len();
    mesh.cell_cx.reserve(n_polys);
    mesh.cell_cy.reserve(n_polys);
    mesh.cell_vol.reserve(n_polys);
    mesh.cell_face_offsets.push(0);
    mesh.cell_vertex_offsets.push(0);

    for poly_verts in all_polys {
        // Create Cell
        let mut center = Vector2::new(0.0, 0.0);
        let mut area = 0.0;
        let n = poly_verts.len();

        // Get vertex indices
        let mut cell_v_indices = Vec::new();
        for (p, _) in &poly_verts {
            let key = (quantize(p.x), quantize(p.y));
            // Vertices should already exist in the map
            if let Some(&idx) = vertex_map.get(&key) {
                cell_v_indices.push(idx);
            } else {
                // Fallback (should not happen if merging is correct)
                let idx = mesh.vx.len();
                mesh.vx.push(p.x);
                mesh.vy.push(p.y);
                mesh.v_fixed.push(false);
                vertex_map.insert(key, idx);
                cell_v_indices.push(idx);
            }
        }

        // Polygon area and centroid
        for k in 0..n {
            let (p_i, _) = poly_verts[k];
            let (p_j, _) = poly_verts[(k + 1) % n];
            let cross = p_i.x * p_j.y - p_j.x * p_i.y;
            area += cross;
            center += (p_i.coords + p_j.coords) * cross;
        }
        area *= 0.5;

        if area.abs() < 1e-9 {
            continue;
        }

        center /= 6.0 * area;

        let cell_idx = mesh.cell_cx.len();

        // Create Faces
        for k in 0..n {
            let v1 = cell_v_indices[k];
            let v2 = cell_v_indices[(k + 1) % n];

            if v1 == v2 {
                continue;
            }

            let p1 = Point2::new(mesh.vx[v1], mesh.vy[v1]);
            let p2 = Point2::new(mesh.vx[v2], mesh.vy[v2]);
            let edge_vec = p2 - p1;
            let edge_len = edge_vec.norm();

            if edge_len < 1e-9 {
                continue;
            }

            let (min_v, max_v) = if v1 < v2 { (v1, v2) } else { (v2, v1) };
            let key = (min_v, max_v);

            if let Some(&face_idx) = face_map.get(&key) {
                // Face exists, update neighbor
                mesh.face_neighbor[face_idx] = Some(cell_idx);
                // Internal face has no boundary type
                mesh.face_boundary[face_idx] = None;
                mesh.cell_faces.push(face_idx);
            } else {
                // New face
                let face_center = Point2::from((p1.coords + p2.coords) * 0.5);
                let normal = Vector2::new(edge_vec.y, -edge_vec.x).normalize();

                // Determine boundary type if boundary
                let boundary_type = if face_center.x < 1e-6 {
                    Some(BoundaryType::Inlet)
                } else if (face_center.x - domain_size.x).abs() < 1e-6 {
                    Some(BoundaryType::Outlet)
                } else {
                    Some(BoundaryType::Wall)
                };

                let face_idx = mesh.face_cx.len();

                mesh.face_v1.push(v1);
                mesh.face_v2.push(v2);
                mesh.face_owner.push(cell_idx);
                mesh.face_neighbor.push(None);
                mesh.face_boundary.push(boundary_type);
                mesh.face_nx.push(normal.x);
                mesh.face_ny.push(normal.y);
                mesh.face_area.push(edge_len);
                mesh.face_cx.push(face_center.x);
                mesh.face_cy.push(face_center.y);

                face_map.insert(key, face_idx);
                mesh.cell_faces.push(face_idx);
            }
        }

        mesh.cell_cx.push(center.x);
        mesh.cell_cy.push(center.y);
        mesh.cell_vol.push(area.abs());

        mesh.cell_face_offsets.push(mesh.cell_faces.len());

        mesh.cell_vertices.extend_from_slice(&cell_v_indices);
        mesh.cell_vertex_offsets.push(mesh.cell_vertices.len());
    }

    let mut min_vol = f64::MAX;
    let mut max_vol = f64::MIN;
    for &vol in &mesh.cell_vol {
        if vol < min_vol {
            min_vol = vol;
        }
        if vol > max_vol {
            max_vol = vol;
        }
    }
    println!(
        "Mesh generated. Cells: {}, Faces: {}. Min Vol: {:.6e}, Max Vol: {:.6e}",
        mesh.num_cells(),
        mesh.num_faces(),
        min_vol,
        max_vol
    );

    mesh
}
