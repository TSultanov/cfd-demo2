use super::geometry::Geometry;
use super::meshgen_utils::{compute_normal, intersect_lines};
use super::quadtree::{collect_leaves, refine_node, QuadNode};
use crate::solver::mesh::{BoundaryType, Mesh};
use ahash::AHashMap;
use nalgebra::{Point2, Vector2};
use std::time::Instant;
use wide::{f64x4, CmpGe, CmpGt, CmpLt};

pub fn generate_cut_cell_mesh(
    geo: &(impl Geometry + Sync),
    min_cell_size: f64,
    max_cell_size: f64,
    growth_rate: f64,
    domain_size: Vector2<f64>,
) -> Mesh {
    let start_total = Instant::now();

    // --- State for connected graph generation ---
    let mut vx: Vec<f64> = Vec::new();
    let mut vy: Vec<f64> = Vec::new();
    let mut v_fixed: Vec<bool> = Vec::new();
    let mut vertex_map: AHashMap<(i64, i64), usize> = AHashMap::new();
    let mut cells: Vec<Vec<usize>> = Vec::new();

    let quantize = |v: f64| (v * 100000.0).round() as i64;

    // Helper to add/find vertex
    let mut add_vertex = |p: Point2<f64>, fixed: bool| -> usize {
        let key = (quantize(p.x), quantize(p.y));
        if let Some(&idx) = vertex_map.get(&key) {
            if fixed && !v_fixed[idx] {
                v_fixed[idx] = true;
            }
            idx
        } else {
            let idx = vx.len();
            vx.push(p.x);
            vy.push(p.y);
            v_fixed.push(fixed);
            vertex_map.insert(key, idx);
            idx
        }
    };

    // 1. Generate Base Mesh (Serial, Connected)
    let t0 = Instant::now();
    let nx = (domain_size.x / max_cell_size).ceil() as usize;
    let ny = (domain_size.y / max_cell_size).ceil() as usize;

    for i in 0..nx {
        for j in 0..ny {
            let x0 = i as f64 * max_cell_size;
            let y0 = j as f64 * max_cell_size;
            let x1 = (x0 + max_cell_size).min(domain_size.x);
            let y1 = (y0 + max_cell_size).min(domain_size.y);

            let mut root = QuadNode::new(Point2::new(x0, y0), Point2::new(x1, y1));
            refine_node(&mut root, geo, min_cell_size, growth_rate);

            let mut leaves = Vec::new();
            collect_leaves(&root, &mut leaves);

            for leaf in leaves {
                let (min, max) = leaf.bounds;

                // Check 4 corners
                let p00 = min;
                let p10 = Point2::new(max.x, min.y);
                let p11 = max;
                let p01 = Point2::new(min.x, max.y);

                let d00 = geo.sdf(&p00);
                let d10 = geo.sdf(&p10);
                let d11 = geo.sdf(&p11);
                let d01 = geo.sdf(&p01);

                let sdf_tol = 1e-9;
                let all_outside =
                    d00 >= -sdf_tol && d10 >= -sdf_tol && d11 >= -sdf_tol && d01 >= -sdf_tol;

                if all_outside {
                    continue;
                }

                let mut poly_verts = Vec::new();
                let all_inside =
                    d00 < -sdf_tol && d10 < -sdf_tol && d11 < -sdf_tol && d01 < -sdf_tol;

                if all_inside {
                    // Rectangular cell
                    poly_verts.push((p00, false));
                    poly_verts.push((p10, false));
                    poly_verts.push((p11, false));
                    poly_verts.push((p01, false));
                } else {
                    // Cut cell
                    let corners = [p00, p10, p11, p01];
                    let dists = [d00, d10, d11, d01];

                    for k in 0..4 {
                        let p_curr = corners[k];
                        let p_next = corners[(k + 1) % 4];
                        let d_curr = dists[k];
                        let d_next = dists[(k + 1) % 4];

                        if d_curr < -sdf_tol {
                            poly_verts.push((p_curr, false));
                        }

                        if (d_curr < -sdf_tol && d_next >= -sdf_tol)
                            || (d_curr >= -sdf_tol && d_next < -sdf_tol)
                        {
                            // Intersection
                            let mut t_a = 0.0;
                            let mut t_b = 1.0;
                            let mut d_a = d_curr;
                            let mut d_b = d_next;

                            let mut t = t_a - d_a * (t_b - t_a) / (d_b - d_a);

                            for _ in 0..10 {
                                let p_inter = p_curr + (p_next - p_curr) * t;
                                let d_inter = geo.sdf(&p_inter);

                                if d_inter.abs() < 1e-12 {
                                    break;
                                }

                                if d_inter.signum() == d_a.signum() {
                                    t_a = t;
                                    d_a = d_inter;
                                } else {
                                    t_b = t;
                                    d_b = d_inter;
                                }

                                let denom = d_b - d_a;
                                if denom.abs() < 1e-20 {
                                    break;
                                }
                                t = t_a - d_a * (t_b - t_a) / denom;
                            }

                            let p_inter = p_curr + (p_next - p_curr) * t;
                            poly_verts.push((p_inter, true));
                        }
                    }
                }

                if poly_verts.len() >= 3 {
                    // Post-process for sharp corners
                    let mut reconstructed_poly = Vec::new();
                    let n = poly_verts.len();
                    for k in 0..n {
                        let (p_curr, is_inter_curr) = poly_verts[k];
                        let (p_next, is_inter_next) = poly_verts[(k + 1) % n];

                        reconstructed_poly.push((p_curr, is_inter_curr));

                        if is_inter_curr && is_inter_next {
                            let n1 = compute_normal(geo, p_curr);
                            let n2 = compute_normal(geo, p_next);

                            if n1.dot(&n2) < 0.7 {
                                if let Some(p_corner) = intersect_lines(p_curr, n1, p_next, n2) {
                                    let tol = 1e-5;
                                    if geo.sdf(&p_corner).abs() <= 1e-4 {
                                        if p_corner.x >= min.x - tol
                                            && p_corner.x <= max.x + tol
                                            && p_corner.y >= min.y - tol
                                            && p_corner.y <= max.y + tol
                                        {
                                            reconstructed_poly.push((p_corner, true));
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Convert to indices
                    let mut cell_indices = Vec::with_capacity(reconstructed_poly.len());
                    for (p, fixed) in reconstructed_poly {
                        cell_indices.push(add_vertex(p, fixed));
                    }
                    cells.push(cell_indices);
                }
            }
        }
    }
    println!("Base mesh generated in {:.2?}", t0.elapsed());

    // 2. Imprint Hanging Nodes (Using Indices)
    let t1 = Instant::now();
    let grid_size = max_cell_size;
    let grid_nx = (domain_size.x / grid_size).ceil() as usize + 1;
    let grid_ny = (domain_size.y / grid_size).ceil() as usize + 1;
    let grid_len = grid_nx * grid_ny;

    // 3. Count vertices per cell
    let mut grid_counts = vec![0; grid_len];
    let grid_indices: Vec<usize> = vx
        .iter()
        .zip(vy.iter())
        .map(|(x, y)| {
            let gx = (x / grid_size).floor().max(0.0) as usize;
            let gy = (y / grid_size).floor().max(0.0) as usize;
            if gx < grid_nx && gy < grid_ny {
                let idx = gy * grid_nx + gx;
                grid_counts[idx] += 1;
                idx
            } else {
                grid_len
            }
        })
        .collect();

    // 4. Prefix sums
    let mut grid_starts = vec![0; grid_len + 1];
    let mut current = 0;
    for i in 0..grid_len {
        grid_starts[i] = current;
        current += grid_counts[i];
    }
    grid_starts[grid_len] = current;

    // 5. Fill SoA arrays
    let mut sorted_xs = vec![0.0; vx.len()];
    let mut sorted_ys = vec![0.0; vy.len()];
    let mut sorted_indices = vec![0; vx.len()]; // Store original indices

    let mut current_starts = grid_starts.clone();
    for (i, idx) in grid_indices.iter().enumerate() {
        let grid_idx = *idx;
        if grid_idx < grid_len {
            let pos = current_starts[grid_idx];
            sorted_xs[pos] = vx[i];
            sorted_ys[pos] = vy[i];
            sorted_indices[pos] = i;
            current_starts[grid_idx] += 1;
        }
    }

    // 6. Process cells
    for cell in cells.iter_mut() {
        let mut new_cell = Vec::new();
        let n = cell.len();

        for k in 0..n {
            let idx_curr = cell[k];
            let idx_next = cell[(k + 1) % n];

            new_cell.push(idx_curr);

            let p_curr = Point2::new(vx[idx_curr], vy[idx_curr]);
            let p_next = Point2::new(vx[idx_next], vy[idx_next]);

            let seg_vec = p_next - p_curr;
            let seg_len_sq = seg_vec.norm_squared();

            if seg_len_sq < 1e-12 {
                continue;
            }

            let mut on_segment = Vec::new();

            // SIMD Setup
            let p_curr_x = f64x4::splat(p_curr.x);
            let p_curr_y = f64x4::splat(p_curr.y);
            let p_next_x = f64x4::splat(p_next.x);
            let p_next_y = f64x4::splat(p_next.y);
            let seg_vec_x = f64x4::splat(seg_vec.x);
            let seg_vec_y = f64x4::splat(seg_vec.y);
            let seg_len_sq_simd = f64x4::splat(seg_len_sq);
            let epsilon = f64x4::splat(1e-10);
            let t_min = f64x4::splat(1e-6);
            let t_max = f64x4::splat(1.0 - 1e-6);

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
                    let indices = &sorted_indices[start..end];

                    let mut i = 0;
                    let chunks_count = xs.len() / 4;

                    for _ in 0..chunks_count {
                        let v_x = f64x4::from(&xs[i..i + 4]);
                        let v_y = f64x4::from(&ys[i..i + 4]);

                        let dx_curr = v_x - p_curr_x;
                        let dy_curr = v_y - p_curr_y;
                        let d_curr = dx_curr * dx_curr + dy_curr * dy_curr;

                        let dx_next = v_x - p_next_x;
                        let dy_next = v_y - p_next_y;
                        let d_next = dx_next * dx_next + dy_next * dy_next;

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
                                    on_segment.push((t_val, indices[i + lane]));
                                }
                            }
                        }
                        i += 4;
                    }

                    for j in i..xs.len() {
                        let v = Point2::new(xs[j], ys[j]);
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
                                on_segment.push((t, indices[j]));
                            }
                        }
                    }
                }
            }

            on_segment.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            for (_, idx) in on_segment {
                new_cell.push(idx);
            }
        }
        *cell = new_cell;
    }
    println!("Hanging nodes imprinted in {:.2?}", t1.elapsed());

    // 7. Finalize Mesh
    let _t2 = Instant::now();
    let mut mesh = Mesh::new();
    mesh.vx = vx;
    mesh.vy = vy;
    mesh.v_fixed = v_fixed;

    let n_polys = cells.len();
    mesh.cell_cx.reserve(n_polys);
    mesh.cell_cy.reserve(n_polys);
    mesh.cell_vol.reserve(n_polys);
    mesh.cell_face_offsets.push(0);
    mesh.cell_vertex_offsets.push(0);

    let mut face_map: AHashMap<(usize, usize), usize> = AHashMap::new();

    for cell_v_indices in cells {
        let mut center = Vector2::new(0.0, 0.0);
        let mut area = 0.0;
        let n = cell_v_indices.len();

        for k in 0..n {
            let idx_i = cell_v_indices[k];
            let idx_j = cell_v_indices[(k + 1) % n];
            let p_i = Point2::new(mesh.vx[idx_i], mesh.vy[idx_i]);
            let p_j = Point2::new(mesh.vx[idx_j], mesh.vy[idx_j]);
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
                mesh.face_neighbor[face_idx] = Some(cell_idx);
                mesh.face_boundary[face_idx] = None;
                mesh.cell_faces.push(face_idx);
            } else {
                let face_center = Point2::from((p1.coords + p2.coords) * 0.5);
                let normal = Vector2::new(edge_vec.y, -edge_vec.x).normalize();

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
    println!("Total mesh generation time: {:.2?}", start_total.elapsed());

    mesh
}
