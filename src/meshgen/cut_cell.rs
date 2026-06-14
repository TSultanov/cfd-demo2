use super::geometry::Geometry;
use super::meshgen_utils::{compute_normal, intersect_lines};
use super::quadtree::{collect_leaves, refine_node, QuadNode};
use super::tolerances::MeshgenTolerances;
use crate::solver::mesh::Mesh;
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
    let tol = MeshgenTolerances::from_geometry(min_cell_size, domain_size);

    // --- State for connected graph generation ---
    let mut vx: Vec<f64> = Vec::new();
    let mut vy: Vec<f64> = Vec::new();
    let mut v_fixed: Vec<bool> = Vec::new();
    let mut vertex_map: AHashMap<(i64, i64), usize> = AHashMap::new();
    let mut cells: Vec<Vec<usize>> = Vec::new();

    // Helper to add/find vertex
    let mut add_vertex = |p: Point2<f64>, fixed: bool| -> usize {
        let key = tol.quantize_point(p.x, p.y);
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

                let sdf_tol = tol.sdf_root_eps;
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

                                if d_inter.abs() < tol.determinant_eps {
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
                            let n1 = compute_normal(geo, p_curr, &tol);
                            let n2 = compute_normal(geo, p_next, &tol);

                            if n1.dot(&n2) < 0.7 {
                                if let Some(p_corner) =
                                    intersect_lines(p_curr, n1, p_next, n2, &tol)
                                {
                                    let corner_tol = 1e-5;
                                    if geo.sdf(&p_corner).abs() <= 1e-4 {
                                        if p_corner.x >= min.x - corner_tol
                                            && p_corner.x <= max.x + corner_tol
                                            && p_corner.y >= min.y - corner_tol
                                            && p_corner.y <= max.y + corner_tol
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

    // Precompute SIMD tolerance values from MeshgenTolerances
    let point_coincidence_sq_val = tol.point_coincidence_sq;
    let t_eps_val = tol.t_eps;

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

            if seg_len_sq < tol.segment_degenerate_sq {
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
            let epsilon = f64x4::splat(point_coincidence_sq_val);
            let t_min = f64x4::splat(t_eps_val);
            let t_max = f64x4::splat(1.0 - t_eps_val);

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

                        if d_curr < point_coincidence_sq_val || d_next < point_coincidence_sq_val {
                            continue;
                        }

                        let v_vec = v - p_curr;
                        let t = v_vec.dot(&seg_vec) / seg_len_sq;

                        if t > t_eps_val && t < 1.0 - t_eps_val {
                            let proj = p_curr + seg_vec * t;
                            if (v - proj).norm_squared() < point_coincidence_sq_val {
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

    // 7. Finalize Mesh via MeshBuilder
    let _t2 = Instant::now();

    let mut builder = super::mesh_builder::MeshBuilder::with_capacity(
        vx.len(),
        cells.len(),
        cells.len() * 4, // rough estimate
    );

    // Add all vertices
    let vert_ids: Vec<super::mesh_builder::VertexId> = vx
        .iter()
        .zip(vy.iter())
        .zip(v_fixed.iter())
        .map(|((&x, &y), &fixed)| builder.add_vertex(x, y, fixed))
        .collect();

    let mut face_map: AHashMap<(usize, usize), super::mesh_builder::FaceId> = AHashMap::new();

    for cell_v_indices in cells {
        let mut center = Vector2::new(0.0, 0.0);
        let mut area = 0.0;
        let n = cell_v_indices.len();

        for k in 0..n {
            let idx_i = cell_v_indices[k];
            let idx_j = cell_v_indices[(k + 1) % n];
            let (pi_x, pi_y) = builder.vertex_pos(vert_ids[idx_i]);
            let (pj_x, pj_y) = builder.vertex_pos(vert_ids[idx_j]);
            let cross = pi_x * pj_y - pj_x * pi_y;
            area += cross;
            center += Vector2::new(pi_x + pj_x, pi_y + pj_y) * cross;
        }
        area *= 0.5;

        if area.abs() < tol.area_eps {
            continue;
        }

        let cell_verts: Vec<super::mesh_builder::VertexId> =
            cell_v_indices.iter().map(|&i| vert_ids[i]).collect();
        let cell_id = builder.add_cell(&cell_verts);

        for k in 0..n {
            let v1 = cell_v_indices[k];
            let v2 = cell_v_indices[(k + 1) % n];

            if v1 == v2 {
                continue;
            }

            let (p1_x, p1_y) = builder.vertex_pos(vert_ids[v1]);
            let (p2_x, p2_y) = builder.vertex_pos(vert_ids[v2]);
            let edge_len = ((p2_x - p1_x).powi(2) + (p2_y - p1_y).powi(2)).sqrt();

            if edge_len < tol.edge_len_eps {
                continue;
            }

            let (min_v, max_v) = if v1 < v2 { (v1, v2) } else { (v2, v1) };
            let key = (min_v, max_v);

            if let Some(&face_id) = face_map.get(&key) {
                builder.set_face_neighbor(face_id, cell_id);
            } else {
                let fc_x = (p1_x + p2_x) * 0.5;
                let fc_y = (p1_y + p2_y) * 0.5;

                let boundary_type =
                    tol.classify_boundary(fc_x, fc_y, domain_size.x, domain_size.y);

                let face_id = builder.add_face(
                    vert_ids[v1],
                    vert_ids[v2],
                    cell_id,
                    None,
                    boundary_type,
                );
                face_map.insert(key, face_id);
            }
        }
    }

    let mesh = merge_small_cells(builder.build(), min_cell_size);

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

/// Cut cells whose volume is below this fraction of the smallest intended cell
/// (`min_cell_size^2`) are "slivers" and get merged into a neighbor.
const SMALL_CELL_FRACTION: f64 = 0.5;

/// Merge cut-cell "slivers" — cells far smaller than the smallest intended cell —
/// into their largest-shared-face neighbor.
///
/// Tiny cut cells (a boundary cutting a background cell into a thin fragment)
/// otherwise produce ill-conditioned control volumes that destabilize the
/// finite-volume solver regardless of scheme, timestep, or (low) viscosity. This
/// is the standard cut-cell small-cell merging: faces keep their exact geometry
/// (only owner/neighbor cell indices are remapped, and faces internal to a merged
/// group are dropped), merged volumes sum, and centroids are volume-weighted — so
/// the discretization stays conservative. Cell vertex lists keep the largest
/// member's polygon (rendering-only; the solver uses faces + volumes).
fn merge_small_cells(mesh: Mesh, min_cell_size: f64) -> Mesh {
    let n = mesh.num_cells();
    if n == 0 {
        return mesh;
    }
    let threshold = SMALL_CELL_FRACTION * min_cell_size * min_cell_size;
    if !mesh.cell_vol.iter().any(|&v| v < threshold) {
        return mesh; // no slivers — nothing to do
    }

    // Union-find; the representative of each group is its largest-volume member so
    // the kept vertex polygon is the dominant one.
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }

    for c in 0..n {
        if mesh.cell_vol[c] >= threshold {
            continue;
        }
        // Largest interior face neighbor.
        let mut best: Option<(f64, usize)> = None;
        for &fi in &mesh.cell_faces[mesh.cell_face_offsets[c]..mesh.cell_face_offsets[c + 1]] {
            let other = if mesh.face_owner[fi] == c {
                mesh.face_neighbor[fi]
            } else {
                Some(mesh.face_owner[fi])
            };
            let Some(other) = other else { continue }; // boundary face
            if best.map_or(true, |(ba, _)| mesh.face_area[fi] > ba) {
                best = Some((mesh.face_area[fi], other));
            }
        }
        if let Some((_, other)) = best {
            let rc = find(&mut parent, c);
            let ro = find(&mut parent, other);
            if rc != ro {
                // Attach the smaller-volume root under the larger.
                if mesh.cell_vol[rc] >= mesh.cell_vol[ro] {
                    parent[ro] = rc;
                } else {
                    parent[rc] = ro;
                }
            }
        }
    }

    let root_of: Vec<usize> = {
        let mut r = vec![0usize; n];
        for (c, slot) in r.iter_mut().enumerate() {
            *slot = find(&mut parent, c);
        }
        r
    };

    // Compact roots -> new cell indices.
    let mut new_index = vec![usize::MAX; n];
    let mut roots: Vec<usize> = Vec::new();
    for &r in &root_of {
        if new_index[r] == usize::MAX {
            new_index[r] = roots.len();
            roots.push(r);
        }
    }
    let new_n = roots.len();
    if new_n == n {
        return mesh; // every "small" cell was isolated (no interior face) — unchanged
    }

    // Aggregate cell volume + volume-weighted centroid.
    let mut cell_vol = vec![0.0f64; new_n];
    let mut cell_cx = vec![0.0f64; new_n];
    let mut cell_cy = vec![0.0f64; new_n];
    for c in 0..n {
        let ni = new_index[root_of[c]];
        let v = mesh.cell_vol[c];
        cell_vol[ni] += v;
        cell_cx[ni] += mesh.cell_cx[c] * v;
        cell_cy[ni] += mesh.cell_cy[c] * v;
    }
    for i in 0..new_n {
        if cell_vol[i] > 0.0 {
            cell_cx[i] /= cell_vol[i];
            cell_cy[i] /= cell_vol[i];
        }
    }

    // Filter + remap faces. Drop faces internal to a merged group.
    let nf = mesh.num_faces();
    let has_wrap = !mesh.face_wrap_shift.is_empty();
    let mut face_v1 = Vec::with_capacity(nf);
    let mut face_v2 = Vec::with_capacity(nf);
    let mut face_owner = Vec::with_capacity(nf);
    let mut face_neighbor = Vec::with_capacity(nf);
    let mut face_boundary = Vec::with_capacity(nf);
    let mut face_nx = Vec::with_capacity(nf);
    let mut face_ny = Vec::with_capacity(nf);
    let mut face_area = Vec::with_capacity(nf);
    let mut face_cx = Vec::with_capacity(nf);
    let mut face_cy = Vec::with_capacity(nf);
    let mut face_wrap_shift = Vec::with_capacity(if has_wrap { nf } else { 0 });
    let mut cell_face_lists: Vec<Vec<usize>> = vec![Vec::new(); new_n];

    for fi in 0..nf {
        let owner_new = new_index[root_of[mesh.face_owner[fi]]];
        let neighbor_new = mesh.face_neighbor[fi].map(|nb| new_index[root_of[nb]]);
        if neighbor_new == Some(owner_new) {
            continue; // internal to a merged cell — remove
        }
        let new_fi = face_owner.len();
        face_v1.push(mesh.face_v1[fi]);
        face_v2.push(mesh.face_v2[fi]);
        face_owner.push(owner_new);
        face_neighbor.push(neighbor_new);
        face_boundary.push(mesh.face_boundary[fi]);
        face_nx.push(mesh.face_nx[fi]);
        face_ny.push(mesh.face_ny[fi]);
        face_area.push(mesh.face_area[fi]);
        face_cx.push(mesh.face_cx[fi]);
        face_cy.push(mesh.face_cy[fi]);
        if has_wrap {
            face_wrap_shift.push(mesh.face_wrap_shift[fi]);
        }
        cell_face_lists[owner_new].push(new_fi);
        if let Some(nn) = neighbor_new {
            cell_face_lists[nn].push(new_fi);
        }
    }

    // Rebuild CSR connectivity.
    let mut cell_faces = Vec::with_capacity(cell_face_lists.iter().map(|l| l.len()).sum());
    let mut cell_face_offsets = vec![0usize; new_n + 1];
    for i in 0..new_n {
        cell_faces.extend_from_slice(&cell_face_lists[i]);
        cell_face_offsets[i + 1] = cell_faces.len();
    }

    // Keep the representative (largest) cell's vertex polygon for rendering.
    let mut cell_vertices = Vec::new();
    let mut cell_vertex_offsets = vec![0usize; new_n + 1];
    for i in 0..new_n {
        let rc = roots[i];
        let v0 = mesh.cell_vertex_offsets[rc];
        let v1 = mesh.cell_vertex_offsets[rc + 1];
        cell_vertices.extend_from_slice(&mesh.cell_vertices[v0..v1]);
        cell_vertex_offsets[i + 1] = cell_vertices.len();
    }

    let merged = n - new_n;
    println!("Cut-cell sliver merge: {merged} small cells merged ({n} -> {new_n}).");

    Mesh {
        vx: mesh.vx,
        vy: mesh.vy,
        v_fixed: mesh.v_fixed,
        face_v1,
        face_v2,
        face_owner,
        face_neighbor,
        face_boundary,
        face_nx,
        face_ny,
        face_area,
        face_cx,
        face_cy,
        face_wrap_shift,
        cell_cx,
        cell_cy,
        cell_vol,
        cell_faces,
        cell_face_offsets,
        cell_vertices,
        cell_vertex_offsets,
    }
}
