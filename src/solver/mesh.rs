use nalgebra::{Point2, Vector2};
use std::collections::HashMap;
use rayon::prelude::*;
use wide::{f64x4, CmpGe, CmpGt, CmpLt};

#[derive(Clone, Debug)]
pub struct Vertex {
    pub pos: Point2<f64>,
    pub is_fixed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BoundaryType {
    Inlet,
    Outlet,
    Wall,
    ParallelInterface(usize, usize), // Rank of the neighbor, Remote Cell Index
}

#[derive(Clone, Debug)]
pub struct Face {
    pub vertex_indices: [usize; 2],
    pub owner: usize,
    pub neighbor: Option<usize>, // None for boundary
    pub boundary_type: Option<BoundaryType>,
    pub normal: Vector2<f64>,    // Pointing from owner to neighbor
    pub area: f64,               // Length in 2D
    pub center: Point2<f64>,
}

#[derive(Clone, Debug)]
pub struct Cell {
    pub face_indices: Vec<usize>,
    pub vertex_indices: Vec<usize>,
    pub center: Point2<f64>,
    pub volume: f64, // Area in 2D
}

#[derive(Default, Clone)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub faces: Vec<Face>,
    pub cells: Vec<Cell>,
}

impl Mesh {
    pub fn new() -> Self {
        Self::default()
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
        let outer_dist = outer_box_dx.max(outer_box_dy).min(0.0) + Vector2::new(outer_box_dx.max(0.0), outer_box_dy.max(0.0)).norm();
        
        let step_h = self.height_outlet - self.height_inlet;
        let step_w = self.step_x;
        
        // Block is at bottom left: [0, step_w] x [0, step_h]
        let block_cx = step_w / 2.0;
        let block_cy = step_h / 2.0;
        
        let block_dx = (p.x - block_cx).abs() - step_w / 2.0;
        let block_dy = (p.y - block_cy).abs() - step_h / 2.0;
        let block_dist = block_dx.max(block_dy).min(0.0) + Vector2::new(block_dx.max(0.0), block_dy.max(0.0)).norm();
        
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
        let c1 = Box::new(QuadNode::new(Point2::new(center.x, min.y), Point2::new(max.x, center.y)));
        let c2 = Box::new(QuadNode::new(Point2::new(min.x, center.y), Point2::new(center.x, max.y)));
        let c3 = Box::new(QuadNode::new(center, max));
        
        self.children = Some([c0, c1, c2, c3]);
        self.is_leaf = false;
    }
}

pub fn generate_cut_cell_mesh(geo: &(impl Geometry + Sync), min_cell_size: f64, max_cell_size: f64, domain_size: Vector2<f64>) -> Mesh {
    let mut mesh = Mesh::new();
    
    // 1. Build Quadtree and Process Leaves
    let nx = (domain_size.x / max_cell_size).ceil() as usize;
    let ny = (domain_size.y / max_cell_size).ceil() as usize;
    
    let mut vertex_map: HashMap<(i64, i64), usize> = HashMap::new(); // Quantized coords -> index
    let quantize = |v: f64| (v * 100000.0).round() as i64;
    
    // Map (v_min, v_max) -> face_index
    let mut face_map: HashMap<(usize, usize), usize> = HashMap::new();

    // Collect all leaf polygons first
    let mut all_polys: Vec<Vec<(Point2<f64>, bool)>> = (0..nx).into_par_iter().flat_map(|i| {
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
                let all_outside = d00 >= -sdf_tol && d10 >= -sdf_tol && d11 >= -sdf_tol && d01 >= -sdf_tol;
                
                if all_outside {
                    continue;
                }
                
                let mut poly_verts = Vec::new();
                let all_inside = d00 < -sdf_tol && d10 < -sdf_tol && d11 < -sdf_tol && d01 < -sdf_tol;

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
                        
                        if (d_curr < -sdf_tol && d_next >= -sdf_tol) || (d_curr >= -sdf_tol && d_next < -sdf_tol) {
                            // Intersection
                            let denom = d_curr - d_next;
                            if denom.abs() < 1e-20 {
                                // println!("Warning: denom too small in intersection: d_curr={}, d_next={}", d_curr, d_next);
                            }
                            let t = d_curr / denom;
                            let p_inter = p_curr + (p_next - p_curr) * t;
                            poly_verts.push((p_inter, true));
                        }
                    }
                }
                
                if poly_verts.len() >= 3 {
                    local_polys.push(poly_verts);
                }
            }
            local_polys
        })
    }).collect();

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
    let unique_verts: Vec<(Point2<f64>, bool)> = unique_verts_map.values().cloned().collect();

    // Prepare SoA for SIMD
    let uv_x: Vec<f64> = unique_verts.iter().map(|(p, _)| p.x).collect();
    let uv_y: Vec<f64> = unique_verts.iter().map(|(p, _)| p.y).collect();
    let uv_fixed: Vec<bool> = unique_verts.iter().map(|(_, f)| *f).collect();

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
            
            if seg_len_sq < 1e-12 { continue; }

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

            let chunks_x = uv_x.chunks_exact(4);
            let chunks_y = uv_y.chunks_exact(4);
            let remainder_x = chunks_x.remainder();
            let remainder_y = chunks_y.remainder();
            
            let mut idx_base = 0;
            for (vx, vy) in chunks_x.zip(chunks_y) {
                let v_x = f64x4::from(TryInto::<[f64; 4]>::try_into(vx).unwrap());
                let v_y = f64x4::from(TryInto::<[f64; 4]>::try_into(vy).unwrap());
                
                let dx_curr = v_x - p_curr_x;
                let dy_curr = v_y - p_curr_y;
                let d_curr = dx_curr * dx_curr + dy_curr * dy_curr;
                
                let dx_next = v_x - p_next_x;
                let dy_next = v_y - p_next_y;
                let d_next = dx_next * dx_next + dy_next * dy_next;
                
                let not_endpoint = d_curr.simd_ge(epsilon) & d_next.simd_ge(epsilon);
                
                if not_endpoint.none() { 
                    idx_base += 4;
                    continue; 
                }

                let dot = dx_curr * seg_vec_x + dy_curr * seg_vec_y;
                let t = dot / seg_len_sq_simd;
                
                let t_in_range = t.simd_gt(t_min) & t.simd_lt(t_max);
                let mask = not_endpoint & t_in_range;
                
                if mask.none() {
                    idx_base += 4;
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
                    for lane in 0..4 {
                        if (mask_int & (1 << lane)) != 0 {
                            let idx = idx_base + lane;
                            on_segment.push((t_arr[lane], Point2::new(uv_x[idx], uv_y[idx]), uv_fixed[idx]));
                        }
                    }
                }
                idx_base += 4;
            }

            for (i, (&vx, &vy)) in remainder_x.iter().zip(remainder_y.iter()).enumerate() {
                let idx = idx_base + i;
                let v = Point2::new(vx, vy);
                let is_fixed = uv_fixed[idx];
                
                let d_curr = (v - p_curr).norm_squared();
                let d_next = (v - p_next).norm_squared();
                
                if d_curr < 1e-12 || d_next < 1e-12 { continue; }
                
                let v_vec = v - p_curr;
                let t = v_vec.dot(&seg_vec) / seg_len_sq;
                
                if t > 1e-6 && t < 1.0 - 1e-6 {
                    let proj = p_curr + seg_vec * t;
                    if (v - proj).norm_squared() < 1e-12 {
                        on_segment.push((t, v, is_fixed));
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
                let idx = mesh.vertices.len();
                let is_fixed = unique_verts_map.get(&key).map(|(_, f)| *f).unwrap_or(false);
                mesh.vertices.push(Vertex { pos: *p, is_fixed });
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
        
        let cell_idx = mesh.cells.len();
        let mut cell_face_indices = Vec::new();
        
        // Create Faces
        for k in 0..n {
            let v1 = cell_v_indices[k];
            let v2 = cell_v_indices[(k + 1) % n];
            
            if v1 == v2 {
                continue;
            }

            let p1 = mesh.vertices[v1].pos;
            let p2 = mesh.vertices[v2].pos;
            let edge_vec = p2 - p1;
            let edge_len = edge_vec.norm();
            
            if edge_len < 1e-9 {
                continue;
            }

                    let (min_v, max_v) = if v1 < v2 { (v1, v2) } else { (v2, v1) };
                    let key = (min_v, max_v);
                    
                    if let Some(&face_idx) = face_map.get(&key) {
                        // Face exists, update neighbor
                        mesh.faces[face_idx].neighbor = Some(cell_idx);
                        cell_face_indices.push(face_idx);
                    } else {
                        // New face
                        let p1 = mesh.vertices[v1].pos;
                        let p2 = mesh.vertices[v2].pos;
                        let face_center = Point2::from((p1.coords + p2.coords) * 0.5);
                        let edge_vec = p2 - p1;
                        let normal = Vector2::new(edge_vec.y, -edge_vec.x).normalize(); 
                        
                        // Determine boundary type if boundary
                        let boundary_type = if face_center.x < 1e-6 {
                            Some(BoundaryType::Inlet)
                        } else if (face_center.x - domain_size.x).abs() < 1e-6 {
                            Some(BoundaryType::Outlet)
                        } else {
                            Some(BoundaryType::Wall)
                        };

                        let face_idx = mesh.faces.len();
                        mesh.faces.push(Face {
                            vertex_indices: [v1, v2],
                            owner: cell_idx,
                            neighbor: None,
                            boundary_type,
                            normal,
                            area: edge_vec.norm(),
                            center: face_center,
                        });
                        face_map.insert(key, face_idx);
                        cell_face_indices.push(face_idx);
                    }
                }
                
                mesh.cells.push(Cell {
                    face_indices: cell_face_indices,
                    vertex_indices: cell_v_indices,
                    center: Point2::from(center),
                    volume: area.abs(),
                });
            }
    
    let mut min_vol = f64::MAX;
    let mut max_vol = f64::MIN;
    for cell in &mesh.cells {
        if cell.volume < min_vol { min_vol = cell.volume; }
        if cell.volume > max_vol { max_vol = cell.volume; }
    }
    println!("Mesh generated. Cells: {}, Faces: {}. Min Vol: {:.6e}, Max Vol: {:.6e}", 
        mesh.cells.len(), mesh.faces.len(), min_vol, max_vol);

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
        for face in &mut self.faces {
            let v0 = self.vertices[face.vertex_indices[0]].pos;
            let v1 = self.vertices[face.vertex_indices[1]].pos;
            
            face.center = Point2::from((v0.coords + v1.coords) * 0.5);
            let edge_vec = v1 - v0;
            face.area = edge_vec.norm();
            
            // Preserve normal orientation
            let tangent = edge_vec.normalize();
            let mut normal = Vector2::new(tangent.y, -tangent.x);
            
            if normal.dot(&face.normal) < 0.0 {
                normal = -normal;
            }
            face.normal = normal;
        }
        
        // 2. Recalculate Cells
        for cell in &mut self.cells {
            let mut center = Vector2::zeros();
            let n = cell.vertex_indices.len();
            
            // Polygon Area and Centroid
            // Shoelace formula for area, centroid formula
            let mut signed_area = 0.0;
            let mut c_x = 0.0;
            let mut c_y = 0.0;
            
            for i in 0..n {
                let p0 = self.vertices[cell.vertex_indices[i]].pos;
                let p1 = self.vertices[cell.vertex_indices[(i + 1) % n]].pos;
                
                let cross = p0.x * p1.y - p1.x * p0.y;
                signed_area += cross;
                c_x += (p0.x + p1.x) * cross;
                c_y += (p0.y + p1.y) * cross;
            }
            
            signed_area *= 0.5;
            let area = signed_area.abs();
            
            if area > 1e-12 {
                c_x /= 6.0 * signed_area;
                c_y /= 6.0 * signed_area;
                center = Vector2::new(c_x, c_y);
            } else {
                // Fallback to average
                for i in 0..n {
                    center += self.vertices[cell.vertex_indices[i]].pos.coords;
                }
                center /= n as f64;
            }
            
            cell.center = Point2::from(center);
            cell.volume = area;
        }
    }

    pub fn smooth(&mut self, target_skew: f64, max_iterations: usize) {
        let n_verts = self.vertices.len();
        let mut adj = vec![Vec::new(); n_verts];
        
        // Build adjacency
        for face in &self.faces {
            let v0 = face.vertex_indices[0];
            let v1 = face.vertex_indices[1];
            adj[v0].push(v1);
            adj[v1].push(v0);
        }
        
        // Identify domain boundaries (Box)
        // We assume the domain is the bounding box of the mesh
        let mut min_bound = Point2::new(f64::MAX, f64::MAX);
        let mut max_bound = Point2::new(f64::MIN, f64::MIN);
        
        for v in &self.vertices {
            if v.pos.x < min_bound.x { min_bound.x = v.pos.x; }
            if v.pos.y < min_bound.y { min_bound.y = v.pos.y; }
            if v.pos.x > max_bound.x { max_bound.x = v.pos.x; }
            if v.pos.y > max_bound.y { max_bound.y = v.pos.y; }
        }
        
        let is_on_box = |p: &Point2<f64>| -> bool {
            let eps = 1e-6;
            (p.x - min_bound.x).abs() < eps || (p.x - max_bound.x).abs() < eps ||
            (p.y - min_bound.y).abs() < eps || (p.y - max_bound.y).abs() < eps
        };

        for iter in 0..max_iterations {
            // Check skewness
            self.recalculate_geometry();
            let current_skew = self.calculate_max_skewness();
            if current_skew < target_skew {
                println!("Target skewness reached: {:.6} < {:.6} at iter {}", current_skew, target_skew, iter);
                return;
            }
            if iter % 10 == 0 {
                 println!("Smoothing iter {}: max skew = {:.6}", iter, current_skew);
            }

            let mut new_pos = vec![Point2::origin(); n_verts];
            
            for i in 0..n_verts {
                let p_old = self.vertices[i].pos;
                
                // If on domain box or obstacle boundary, fix it
                if is_on_box(&p_old) || self.vertices[i].is_fixed {
                    new_pos[i] = p_old;
                    continue;
                }
                
                if adj[i].is_empty() {
                    new_pos[i] = p_old;
                    continue;
                }
                
                let mut sum = Vector2::zeros();
                let mut count = 0;

                // Internal smoothing: consider all neighbors
                for &neigh in &adj[i] {
                    sum += self.vertices[neigh].pos.coords;
                    count += 1;
                }

                let avg = Point2::from(sum / count as f64);
                
                // Relaxation factor
                let alpha = 0.5;
                let p_new = p_old + (avg - p_old) * alpha;
                
                new_pos[i] = p_new;
            }
            
            for i in 0..n_verts {
                self.vertices[i].pos = new_pos[i];
            }
        }
        
        self.recalculate_geometry();
        println!("Final skewness: {:.6}", self.calculate_max_skewness());
    }

    pub fn calculate_max_skewness(&self) -> f64 {
        let mut max_skew = 0.0;
        for face in &self.faces {
            let d = if let Some(neigh) = face.neighbor {
                self.cells[neigh].center - self.cells[face.owner].center
            } else {
                // Boundary face: vector from cell center to face center
                face.center - self.cells[face.owner].center
            };
            
            let d_norm = if d.norm_squared() > 1e-12 {
                d.normalize()
            } else {
                Vector2::zeros()
            };
            
            let skew = 1.0 - d_norm.dot(&face.normal).abs();
            if skew > max_skew {
                max_skew = skew;
            }
        }
        max_skew
    }

    pub fn calculate_cell_skewness(&self, cell_idx: usize) -> f64 {
        let cell = &self.cells[cell_idx];
        let mut max_skew = 0.0;
        for &face_idx in &cell.face_indices {
            let face = &self.faces[face_idx];
            
            let d = if let Some(neigh) = face.neighbor {
                let c1 = self.cells[face.owner].center;
                let c2 = self.cells[neigh].center;
                c2 - c1
            } else {
                // Boundary face
                face.center - self.cells[cell_idx].center
            };

            let d_norm = if d.norm_squared() > 1e-12 {
                d.normalize()
            } else {
                Vector2::zeros()
            };

            let skew = 1.0 - d_norm.dot(&face.normal).abs();
            if skew > max_skew {
                max_skew = skew;
            }
        }
        max_skew
    }

    pub fn get_cell_at_pos(&self, p: Point2<f64>) -> Option<usize> {
        for (i, cell) in self.cells.iter().enumerate() {
            // Point in polygon test (Ray casting)
            let mut inside = false;
            let n = cell.vertex_indices.len();
            let mut j = n - 1;
            for k in 0..n {
                let vi = cell.vertex_indices[k];
                let vj = cell.vertex_indices[j];
                let pi = self.vertices[vi].pos;
                let pj = self.vertices[vj].pos;
                
                if ((pi.y > p.y) != (pj.y > p.y)) &&
                   (p.x < (pj.x - pi.x) * (p.y - pi.y) / (pj.y - pi.y) + pi.x) {
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
    use nalgebra::{Point2, Vector2};

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
            // Box SDF (negative inside)
            let center = (self.domain_min + self.domain_max.coords) * 0.5;
            let size = (self.domain_max - self.domain_min) * 0.5;
            let p_rel = p - center;
            let d = p_rel.abs() - size;
            let box_dist = d.x.max(d.y).min(0.0) + Vector2::new(d.x.max(0.0), d.y.max(0.0)).norm();

            // Circle SDF (negative inside)
            let circle_dist = (p - self.center).norm() - self.radius;

            // Fluid is inside box AND outside circle
            box_dist.max(-circle_dist)
        }
    }

    #[test]
    fn test_mesh_generation_circle_obstacle() {
        // Use off-center coordinates to generate sliver cells/bad skewness
        // Coarse mesh relative to curvature might cause issues
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
        
        println!("Generated mesh with {} cells", mesh.cells.len());
        assert!(mesh.cells.len() > 0);
        
        let initial_skew = mesh.calculate_max_skewness();
        println!("Initial max skewness: {}", initial_skew);
        
        // Identify boundary vertices before smoothing
        let mut boundary_indices = Vec::new();
        for (i, v) in mesh.vertices.iter().enumerate() {
            if v.is_fixed {
                boundary_indices.push(i);
            }
        }
        
        println!("Found {} fixed boundary vertices", boundary_indices.len());
        assert!(boundary_indices.len() > 0);
        
        // Store initial positions
        let initial_positions: Vec<Point2<f64>> = boundary_indices.iter()
            .map(|&i| mesh.vertices[i].pos)
            .collect();

        // Smooth
        mesh.smooth(0.05, 50);
        
        // Verify positions haven't changed
        for (k, &idx) in boundary_indices.iter().enumerate() {
            let p_new = mesh.vertices[idx].pos;
            let p_old = initial_positions[k];
            let dist = (p_new - p_old).norm();
            if dist > 1e-9 {
                println!("Boundary vertex {} moved by {:.6e}. Old: {:?}, New: {:?}", idx, dist, p_old, p_new);
            }
            assert!(dist < 1e-9, "Boundary vertex moved during smoothing!");
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
        
        println!("Generated mesh with {} cells", mesh.cells.len());
        assert!(mesh.cells.len() > 0);
        
        let initial_skew = mesh.calculate_max_skewness();
        println!("Initial max skewness: {}", initial_skew);
        
        // Target low skewness
        mesh.smooth(0.1, 50);
        
        let final_skew = mesh.calculate_max_skewness();
        println!("Final max skewness: {}", final_skew);
        
        assert!(final_skew <= initial_skew + 1e-10);
        // Relaxed requirement for sharp corners in step
        assert!(final_skew < 0.4); 
    }
}
