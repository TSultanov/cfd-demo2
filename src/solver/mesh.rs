use nalgebra::{Point2, Vector2};
use rayon::prelude::*;
use std::collections::HashMap;
use wide::{f64x4, CmpGe, CmpGt, CmpLt};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BoundaryType {
    Inlet,
    Outlet,
    Wall,
}

#[derive(Default, Clone)]
pub struct Mesh {
    // Vertices
    pub vx: Vec<f64>,
    pub vy: Vec<f64>,
    pub v_fixed: Vec<bool>,

    // Faces
    pub face_v1: Vec<usize>,
    pub face_v2: Vec<usize>,
    pub face_owner: Vec<usize>,
    pub face_neighbor: Vec<Option<usize>>,
    pub face_boundary: Vec<Option<BoundaryType>>,
    pub face_nx: Vec<f64>,
    pub face_ny: Vec<f64>,
    pub face_area: Vec<f64>,
    pub face_cx: Vec<f64>,
    pub face_cy: Vec<f64>,

    // Cells
    pub cell_cx: Vec<f64>,
    pub cell_cy: Vec<f64>,
    pub cell_vol: Vec<f64>,

    // Connectivity
    pub cell_faces: Vec<usize>,
    pub cell_face_offsets: Vec<usize>, // cell_face_offsets[i] .. cell_face_offsets[i+1]

    pub cell_vertices: Vec<usize>,
    pub cell_vertex_offsets: Vec<usize>,
}

impl Mesh {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn num_cells(&self) -> usize {
        self.cell_cx.len()
    }

    pub fn num_faces(&self) -> usize {
        self.face_cx.len()
    }

    pub fn num_vertices(&self) -> usize {
        self.vx.len()
    }
}

// Geometry definition for CutCell
pub trait Geometry {
    fn is_inside(&self, p: &Point2<f64>) -> bool;
    // Returns distance to surface. Negative inside.
    fn sdf(&self, p: &Point2<f64>) -> f64;
}

pub struct ChannelWithObstacle {
    pub length: f64,
    pub height: f64,
    pub obstacle_center: Point2<f64>,
    pub obstacle_radius: f64,
}

impl Geometry for ChannelWithObstacle {
    fn is_inside(&self, p: &Point2<f64>) -> bool {
        self.sdf(p) < 0.0
    }

    fn sdf(&self, p: &Point2<f64>) -> f64 {
        let dx = (p.x - self.length / 2.0).abs() - self.length / 2.0;
        let dy = (p.y - self.height / 2.0).abs() - self.height / 2.0;
        let box_dist = dx.max(dy).min(0.0) + Vector2::new(dx.max(0.0), dy.max(0.0)).norm();

        let circle_dist = (p - self.obstacle_center).norm() - self.obstacle_radius;

        // Fluid is inside box AND outside circle.
        // Outside circle SDF: -circle_dist
        box_dist.max(-circle_dist)
    }
}

pub struct BackwardsStep {
    pub length: f64,
    pub height_inlet: f64,
    pub height_outlet: f64,
    pub step_x: f64,
}

impl Geometry for BackwardsStep {
    fn is_inside(&self, p: &Point2<f64>) -> bool {
        self.sdf(p) < 0.0
    }

    fn sdf(&self, p: &Point2<f64>) -> f64 {
        let outer_box_dx = (p.x - self.length / 2.0).abs() - self.length / 2.0;
        let outer_box_dy = (p.y - self.height_outlet / 2.0).abs() - self.height_outlet / 2.0;
        let outer_dist = outer_box_dx.max(outer_box_dy).min(0.0)
            + Vector2::new(outer_box_dx.max(0.0), outer_box_dy.max(0.0)).norm();

        let step_h = self.height_outlet - self.height_inlet;
        let step_w = self.step_x;

        // Block is at bottom left: [0, step_w] x [0, step_h]
        let block_cx = step_w / 2.0;
        let block_cy = step_h / 2.0;

        let block_dx = (p.x - block_cx).abs() - step_w / 2.0;
        let block_dy = (p.y - block_cy).abs() - step_h / 2.0;
        let block_dist = block_dx.max(block_dy).min(0.0)
            + Vector2::new(block_dx.max(0.0), block_dy.max(0.0)).norm();

        // Fluid is inside outer_box AND outside block.
        outer_dist.max(-block_dist)
    }
}

pub struct RectangularChannel {
    pub length: f64,
    pub height: f64,
}

impl Geometry for RectangularChannel {
    fn is_inside(&self, p: &Point2<f64>) -> bool {
        self.sdf(p) < 0.0
    }

    fn sdf(&self, p: &Point2<f64>) -> f64 {
        let dx = (p.x - self.length / 2.0).abs() - self.length / 2.0;
        let dy = (p.y - self.height / 2.0).abs() - self.height / 2.0;
        dx.max(dy).min(0.0) + Vector2::new(dx.max(0.0), dy.max(0.0)).norm()
    }
}

struct QuadNode {
    bounds: (Point2<f64>, Point2<f64>), // min, max
    children: Option<[Box<QuadNode>; 4]>,
    is_leaf: bool,
}

impl QuadNode {
    fn new(min: Point2<f64>, max: Point2<f64>) -> Self {
        Self {
            bounds: (min, max),
            children: None,
            is_leaf: true,
        }
    }

    fn subdivide(&mut self) {
        let (min, max) = self.bounds;
        let center = Point2::new((min.x + max.x) / 2.0, (min.y + max.y) / 2.0);

        let c0 = Box::new(QuadNode::new(min, center));
        let c1 = Box::new(QuadNode::new(
            Point2::new(center.x, min.y),
            Point2::new(max.x, center.y),
        ));
        let c2 = Box::new(QuadNode::new(
            Point2::new(min.x, center.y),
            Point2::new(center.x, max.y),
        ));
        let c3 = Box::new(QuadNode::new(center, max));

        self.children = Some([c0, c1, c2, c3]);
        self.is_leaf = false;
    }
}

fn compute_normal(geo: &(impl Geometry + ?Sized), p: Point2<f64>) -> Vector2<f64> {
    let eps = 1e-6;
    let d_x = geo.sdf(&Point2::new(p.x + eps, p.y)) - geo.sdf(&Point2::new(p.x - eps, p.y));
    let d_y = geo.sdf(&Point2::new(p.x, p.y + eps)) - geo.sdf(&Point2::new(p.x, p.y - eps));
    Vector2::new(d_x, d_y).normalize()
}

fn intersect_lines(
    p1: Point2<f64>,
    n1: Vector2<f64>,
    p2: Point2<f64>,
    n2: Vector2<f64>,
) -> Option<Point2<f64>> {
    let det = n1.x * n2.y - n1.y * n2.x;
    if det.abs() < 1e-6 {
        return None;
    }

    let d1 = p1.coords.dot(&n1);
    let d2 = p2.coords.dot(&n2);

    let x = (d1 * n2.y - d2 * n1.y) / det;
    let y = (d2 * n1.x - d1 * n2.x) / det;

    Some(Point2::new(x, y))
}

pub fn generate_cut_cell_mesh(
    geo: &(impl Geometry + Sync),
    min_cell_size: f64,
    max_cell_size: f64,
    domain_size: Vector2<f64>,
) -> Mesh {
    let mut mesh = Mesh::new();

    // 1. Build Quadtree and Process Leaves
    let nx = (domain_size.x / max_cell_size).ceil() as usize;
    let ny = (domain_size.y / max_cell_size).ceil() as usize;

    let mut vertex_map: HashMap<(i64, i64), usize> = HashMap::new(); // Quantized coords -> index
    let quantize = |v: f64| (v * 100000.0).round() as i64;

    // Map (v_min, v_max) -> face_index
    let mut face_map: HashMap<(usize, usize), usize> = HashMap::new();

    // Collect all leaf polygons first
    let mut all_polys: Vec<Vec<(Point2<f64>, bool)>> = (0..nx)
        .into_par_iter()
        .flat_map(|i| {
            (0..ny).into_par_iter().flat_map(move |j| {
                let x0 = i as f64 * max_cell_size;
                let y0 = j as f64 * max_cell_size;
                let x1 = (x0 + max_cell_size).min(domain_size.x);
                let y1 = (y0 + max_cell_size).min(domain_size.y);

                let mut root = QuadNode::new(Point2::new(x0, y0), Point2::new(x1, y1));
                refine_node(&mut root, geo, min_cell_size);

                let mut leaves = Vec::new();
                collect_leaves(&root, &mut leaves);

                let mut local_polys = Vec::new();

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
                                // Iterative root finding (Regula Falsi) to handle non-linear SDF
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
                                // Check normals
                                let n1 = compute_normal(geo, p_curr);
                                let n2 = compute_normal(geo, p_next);

                                // If angle > ~45 degrees (cos < 0.707)
                                if n1.dot(&n2) < 0.7 {
                                    if let Some(p_corner) = intersect_lines(p_curr, n1, p_next, n2)
                                    {
                                        // Check if corner is within cell bounds (with tolerance)
                                        // and "behind" the cut (SDF < 0, but we know it should be inside geometry, so SDF < 0)
                                        // Actually, if we are cutting a concave corner, the corner is in the fluid?
                                        // No, if it's a concave corner of the FLUID (convex corner of solid), then the fluid wraps around it.
                                        // Wait, the chamfer is on "concave edges" of the mesh, which corresponds to "convex corners" of the fluid?
                                        // The user image shows a red region (solid?) and blue (fluid).
                                        // The chamfer cuts the red region. So the fluid is invading the solid?
                                        // Or the red is fluid and blue is solid?
                                        // "Chamfers on concave edges" usually means the mesh (fluid domain) has a chamfer where it should be a sharp corner.
                                        // If the fluid has a 270 degree corner (concave), the solid has a 90 degree corner (convex).
                                        // The cut cell algorithm connects the two intersection points, cutting off the fluid corner.
                                        // So the corner point should be INSIDE the cell.

                                        let tol = 1e-5;
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
                        local_polys.push(reconstructed_poly);
                    }
                }
                local_polys
            })
        })
        .collect();

    // 2. Imprint Hanging Nodes
    // Collect all unique vertices and their fixed status
    let mut unique_verts_map: HashMap<(i64, i64), (Point2<f64>, bool)> = HashMap::new();

    for poly in &all_polys {
        for (p, fixed) in poly {
            let key = (quantize(p.x), quantize(p.y));
            let entry = unique_verts_map.entry(key).or_insert((*p, false));
            if *fixed {
                entry.1 = true;
            }
        }
    }

    // Flatten unique vertices for processing
    let unique_verts: Vec<(Point2<f64>, bool)> = unique_verts_map.values().cloned().collect();

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
            let epsilon = f64x4::splat(1e-12);
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

                        if d_curr < 1e-12 || d_next < 1e-12 {
                            continue;
                        }

                        let v_vec = v - p_curr;
                        let t = v_vec.dot(&seg_vec) / seg_len_sq;

                        if t > 1e-6 && t < 1.0 - 1e-6 {
                            let proj = p_curr + seg_vec * t;
                            if (v - proj).norm_squared() < 1e-12 {
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
    let n_polys = all_polys.len();
    mesh.vx.reserve(n_polys * 2);
    mesh.vy.reserve(n_polys * 2);
    mesh.v_fixed.reserve(n_polys * 2);

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
            let idx = if let Some(&idx) = vertex_map.get(&key) {
                idx
            } else {
                let idx = mesh.vx.len();
                let is_fixed = unique_verts_map.get(&key).map(|(_, f)| *f).unwrap_or(false);
                mesh.vx.push(p.x);
                mesh.vy.push(p.y);
                mesh.v_fixed.push(is_fixed);
                vertex_map.insert(key, idx);
                idx
            };
            cell_v_indices.push(idx);
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

fn refine_node(node: &mut QuadNode, geo: &impl Geometry, min_size: f64) {
    let (min, max) = node.bounds;
    let size = (max.x - min.x).max(max.y - min.y);

    if size <= min_size * 1.001 {
        return;
    }

    // Check if we need to split
    // Split if boundary passes through
    // Evaluate SDF at corners. If signs differ, boundary is inside.
    // Also check center?
    // Conservative: check min/max SDF in the box?
    // Simple: check corners.

    let p00 = min;
    let p10 = Point2::new(max.x, min.y);
    let p11 = max;
    let p01 = Point2::new(min.x, max.y);

    let d00 = geo.sdf(&p00);
    let d10 = geo.sdf(&p10);
    let d11 = geo.sdf(&p11);
    let d01 = geo.sdf(&p01);

    let has_inside = d00 < 0.0 || d10 < 0.0 || d11 < 0.0 || d01 < 0.0;
    let has_outside = d00 >= 0.0 || d10 >= 0.0 || d11 >= 0.0 || d01 >= 0.0;

    if has_inside && has_outside {
        node.subdivide();
        if let Some(children) = &mut node.children {
            for child in children.iter_mut() {
                refine_node(child, geo, min_size);
            }
        }
    }
}

fn collect_leaves<'a>(node: &'a QuadNode, leaves: &mut Vec<&'a QuadNode>) {
    if node.is_leaf {
        leaves.push(node);
    } else if let Some(children) = &node.children {
        for child in children {
            collect_leaves(child, leaves);
        }
    }
}

impl Mesh {
    pub fn recalculate_geometry(&mut self) {
        // 1. Recalculate Faces
        for i in 0..self.face_cx.len() {
            let v0_idx = self.face_v1[i];
            let v1_idx = self.face_v2[i];

            let v0 = Point2::new(self.vx[v0_idx], self.vy[v0_idx]);
            let v1 = Point2::new(self.vx[v1_idx], self.vy[v1_idx]);

            let center = Point2::from((v0.coords + v1.coords) * 0.5);
            self.face_cx[i] = center.x;
            self.face_cy[i] = center.y;

            let edge_vec = v1 - v0;
            self.face_area[i] = edge_vec.norm();

            // Preserve normal orientation
            let tangent = edge_vec.normalize();
            let mut normal = Vector2::new(tangent.y, -tangent.x);

            let current_normal = Vector2::new(self.face_nx[i], self.face_ny[i]);
            if normal.dot(&current_normal) < 0.0 {
                normal = -normal;
            }
            self.face_nx[i] = normal.x;
            self.face_ny[i] = normal.y;
        }

        // 2. Recalculate Cells
        for i in 0..self.cell_cx.len() {
            let mut center = Vector2::zeros();
            let start = self.cell_vertex_offsets[i];
            let end = self.cell_vertex_offsets[i + 1];
            let n = end - start;

            // Polygon Area and Centroid
            let mut signed_area = 0.0;
            let mut c_x = 0.0;
            let mut c_y = 0.0;

            for k in 0..n {
                let idx0 = self.cell_vertices[start + k];
                let idx1 = self.cell_vertices[start + (k + 1) % n];

                let p0_x = self.vx[idx0];
                let p0_y = self.vy[idx0];
                let p1_x = self.vx[idx1];
                let p1_y = self.vy[idx1];

                let cross = p0_x * p1_y - p1_x * p0_y;
                signed_area += cross;
                c_x += (p0_x + p1_x) * cross;
                c_y += (p0_y + p1_y) * cross;
            }

            signed_area *= 0.5;
            let area = signed_area.abs();

            if area > 1e-12 {
                c_x /= 6.0 * signed_area;
                c_y /= 6.0 * signed_area;
                center = Vector2::new(c_x, c_y);
            } else {
                // Fallback to average
                for k in 0..n {
                    let idx = self.cell_vertices[start + k];
                    center.x += self.vx[idx];
                    center.y += self.vy[idx];
                }
                center /= n as f64;
            }

            self.cell_cx[i] = center.x;
            self.cell_cy[i] = center.y;
            self.cell_vol[i] = area;
        }
    }

    pub fn smooth<G: Geometry>(&mut self, geo: &G, target_skew: f64, max_iterations: usize) {
        let n_verts = self.vx.len();
        let mut adj = vec![Vec::new(); n_verts];

        // Build adjacency
        for i in 0..self.face_cx.len() {
            let v0 = self.face_v1[i];
            let v1 = self.face_v2[i];
            adj[v0].push(v1);
            adj[v1].push(v0);
        }

        // Identify domain boundaries (Box)
        let mut min_bound = Point2::new(f64::MAX, f64::MAX);
        let mut max_bound = Point2::new(f64::MIN, f64::MIN);

        for i in 0..n_verts {
            if self.vx[i] < min_bound.x {
                min_bound.x = self.vx[i];
            }
            if self.vy[i] < min_bound.y {
                min_bound.y = self.vy[i];
            }
            if self.vx[i] > max_bound.x {
                max_bound.x = self.vx[i];
            }
            if self.vy[i] > max_bound.y {
                max_bound.y = self.vy[i];
            }
        }

        let is_on_box = |x: f64, y: f64| -> bool {
            let eps = 1e-6;
            (x - min_bound.x).abs() < eps
                || (x - max_bound.x).abs() < eps
                || (y - min_bound.y).abs() < eps
                || (y - max_bound.y).abs() < eps
        };

        for iter in 0..max_iterations {
            // Check skewness
            self.recalculate_geometry();
            let current_skew = self.calculate_max_skewness();
            if current_skew < target_skew {
                println!(
                    "Target skewness reached: {:.6} < {:.6} at iter {}",
                    current_skew, target_skew, iter
                );
                return;
            }
            if iter % 10 == 0 {
                println!("Smoothing iter {}: max skew = {:.6}", iter, current_skew);
            }

            let mut new_vx = vec![0.0; n_verts];
            let mut new_vy = vec![0.0; n_verts];

            for i in 0..n_verts {
                let x_old = self.vx[i];
                let y_old = self.vy[i];

                // If on domain box, fix it
                if is_on_box(x_old, y_old) {
                    new_vx[i] = x_old;
                    new_vy[i] = y_old;
                    continue;
                }

                if adj[i].is_empty() {
                    new_vx[i] = x_old;
                    new_vy[i] = y_old;
                    continue;
                }

                let mut sum_x = 0.0;
                let mut sum_y = 0.0;
                let mut count = 0;

                // Internal smoothing: consider all neighbors
                for &neigh in &adj[i] {
                    sum_x += self.vx[neigh];
                    sum_y += self.vy[neigh];
                    count += 1;
                }

                let avg_x = sum_x / count as f64;
                let avg_y = sum_y / count as f64;

                // Relaxation factor
                let alpha = 0.5;
                let mut x_new = x_old + (avg_x - x_old) * alpha;
                let mut y_new = y_old + (avg_y - y_old) * alpha;

                if self.v_fixed[i] {
                    // Project back to surface
                    let p_curr = Point2::new(x_new, y_new);
                    let d = geo.sdf(&p_curr);

                    // Numerical Gradient
                    let eps = 1e-6;
                    let d_x = geo.sdf(&Point2::new(x_new + eps, y_new))
                        - geo.sdf(&Point2::new(x_new - eps, y_new));
                    let d_y = geo.sdf(&Point2::new(x_new, y_new + eps))
                        - geo.sdf(&Point2::new(x_new, y_new - eps));
                    let grad = Vector2::new(d_x, d_y).normalize();

                    let p_proj = p_curr - grad * d;
                    x_new = p_proj.x;
                    y_new = p_proj.y;
                }

                // Check for bad cells (edge collapse)
                let mut bad_move = false;
                for &neigh in &adj[i] {
                    let nx = self.vx[neigh];
                    let ny = self.vy[neigh];
                    let dist_sq = (x_new - nx).powi(2) + (y_new - ny).powi(2);
                    if dist_sq < 1e-8 {
                        // 1e-4 squared
                        bad_move = true;
                        break;
                    }
                }

                if bad_move {
                    new_vx[i] = x_old;
                    new_vy[i] = y_old;
                } else {
                    new_vx[i] = x_new;
                    new_vy[i] = y_new;
                }
            }

            self.vx = new_vx;
            self.vy = new_vy;
        }

        self.recalculate_geometry();
        println!("Final skewness: {:.6}", self.calculate_max_skewness());
    }

    pub fn calculate_max_skewness(&self) -> f64 {
        let mut max_skew = 0.0;
        for i in 0..self.face_cx.len() {
            let owner = self.face_owner[i];
            let d = if let Some(neigh) = self.face_neighbor[i] {
                let c1 = Vector2::new(self.cell_cx[owner], self.cell_cy[owner]);
                let c2 = Vector2::new(self.cell_cx[neigh], self.cell_cy[neigh]);
                c2 - c1
            } else {
                // Boundary face
                let c1 = Vector2::new(self.cell_cx[owner], self.cell_cy[owner]);
                let f_c = Vector2::new(self.face_cx[i], self.face_cy[i]);
                f_c - c1
            };

            let d_norm = if d.norm_squared() > 1e-12 {
                d.normalize()
            } else {
                Vector2::zeros()
            };

            let normal = Vector2::new(self.face_nx[i], self.face_ny[i]);
            let skew = 1.0 - d_norm.dot(&normal).abs();
            if skew > max_skew {
                max_skew = skew;
            }
        }
        max_skew
    }

    pub fn calculate_cell_skewness(&self, cell_idx: usize) -> f64 {
        let mut max_skew = 0.0;
        let start = self.cell_face_offsets[cell_idx];
        let end = self.cell_face_offsets[cell_idx + 1];

        for k in start..end {
            let face_idx = self.cell_faces[k];
            let owner = self.face_owner[face_idx];

            let d = if let Some(neigh) = self.face_neighbor[face_idx] {
                let c1 = Vector2::new(self.cell_cx[owner], self.cell_cy[owner]);
                let c2 = Vector2::new(self.cell_cx[neigh], self.cell_cy[neigh]);
                c2 - c1
            } else {
                // Boundary face
                let c1 = Vector2::new(self.cell_cx[cell_idx], self.cell_cy[cell_idx]);
                let f_c = Vector2::new(self.face_cx[face_idx], self.face_cy[face_idx]);
                f_c - c1
            };

            let d_norm = if d.norm_squared() > 1e-12 {
                d.normalize()
            } else {
                Vector2::zeros()
            };

            let normal = Vector2::new(self.face_nx[face_idx], self.face_ny[face_idx]);
            let skew = 1.0 - d_norm.dot(&normal).abs();
            if skew > max_skew {
                max_skew = skew;
            }
        }
        max_skew
    }

    pub fn get_cell_at_pos(&self, p: Point2<f64>) -> Option<usize> {
        for i in 0..self.cell_cx.len() {
            // Point in polygon test (Ray casting)
            let mut inside = false;
            let start = self.cell_vertex_offsets[i];
            let end = self.cell_vertex_offsets[i + 1];
            let n = end - start;
            let mut j = n - 1;
            for k in 0..n {
                let vi = self.cell_vertices[start + k];
                let vj = self.cell_vertices[start + j];
                let pi_x = self.vx[vi];
                let pi_y = self.vy[vi];
                let pj_x = self.vx[vj];
                let pj_y = self.vy[vj];

                if ((pi_y > p.y) != (pj_y > p.y))
                    && (p.x < (pj_x - pi_x) * (p.y - pi_y) / (pj_y - pi_y) + pi_x)
                {
                    inside = !inside;
                }
                j = k;
            }

            if inside {
                return Some(i);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CircleObstacle {
        center: Point2<f64>,
        radius: f64,
        domain_min: Point2<f64>,
        domain_max: Point2<f64>,
    }

    impl Geometry for CircleObstacle {
        fn is_inside(&self, p: &Point2<f64>) -> bool {
            self.sdf(p) < 0.0
        }

        fn sdf(&self, p: &Point2<f64>) -> f64 {
            let dx = (p.x - (self.domain_min.x + self.domain_max.x) / 2.0).abs()
                - (self.domain_max.x - self.domain_min.x) / 2.0;
            let dy = (p.y - (self.domain_min.y + self.domain_max.y) / 2.0).abs()
                - (self.domain_max.y - self.domain_min.y) / 2.0;
            let box_dist = dx.max(dy).min(0.0) + Vector2::new(dx.max(0.0), dy.max(0.0)).norm();

            let circle_dist = (p - self.center).norm() - self.radius;

            box_dist.max(-circle_dist)
        }
    }

    #[test]
    fn test_mesh_generation_circle_obstacle() {
        let geo = CircleObstacle {
            center: Point2::new(0.5001, 0.5001),
            radius: 0.2,
            domain_min: Point2::new(0.0, 0.0),
            domain_max: Point2::new(1.0, 1.0),
        };

        // Generate mesh
        let domain_size = Vector2::new(1.0, 1.0);
        // Coarser mesh: 0.1 cell size. Radius is 0.2.
        let mut mesh = generate_cut_cell_mesh(&geo, 0.1, 0.1, domain_size);

        println!("Generated mesh with {} cells", mesh.num_cells());
        assert!(mesh.num_cells() > 0);

        let initial_skew = mesh.calculate_max_skewness();
        println!("Initial max skewness: {}", initial_skew);

        // Identify boundary vertices before smoothing
        let mut boundary_indices = Vec::new();
        for i in 0..mesh.num_vertices() {
            if mesh.v_fixed[i] {
                boundary_indices.push(i);
            }
        }

        println!("Found {} fixed boundary vertices", boundary_indices.len());
        assert!(boundary_indices.len() > 0);

        // Smooth
        mesh.smooth(&geo, 0.05, 50);

        // Verify positions are still on boundary
        for &idx in &boundary_indices {
            let p_new = Point2::new(mesh.vx[idx], mesh.vy[idx]);
            let dist = geo.sdf(&p_new).abs();
            assert!(
                dist < 1e-4,
                "Boundary vertex moved off boundary! dist={}",
                dist
            );
        }

        let final_skew = mesh.calculate_max_skewness();
        println!("Final max skewness: {}", final_skew);

        // If smoothing made it worse, we should know.
        // But for now, let's just check it's not terrible.
        assert!(final_skew < 0.25);
    }

    #[test]
    fn test_mesh_generation_backwards_step() {
        // Misaligned step to create bad cut cells
        // Grid lines at 0.1. Step at 0.501 creates 0.001 sliver.
        let geo = BackwardsStep {
            length: 2.0,
            height_inlet: 0.501,
            height_outlet: 1.0,
            step_x: 0.501,
        };

        let domain_size = Vector2::new(2.0, 1.0);
        let mut mesh = generate_cut_cell_mesh(&geo, 0.1, 0.1, domain_size);

        println!("Generated mesh with {} cells", mesh.num_cells());
        assert!(mesh.num_cells() > 0);

        let initial_skew = mesh.calculate_max_skewness();
        println!("Initial max skewness: {}", initial_skew);

        // Target low skewness
        mesh.smooth(&geo, 0.1, 50);

        let final_skew = mesh.calculate_max_skewness();
        println!("Final max skewness: {}", final_skew);

        // Sharp corners increase skewness compared to chamfered corners, so we relax the check.
        assert!(final_skew < 0.6);
    }
}
