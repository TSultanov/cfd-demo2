use nalgebra::{Point2, Vector2};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct Vertex {
    pub pos: Point2<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BoundaryType {
    Inlet,
    Outlet,
    Wall,
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

#[derive(Default)]
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

pub fn generate_cut_cell_mesh(geo: &impl Geometry, min_cell_size: f64, max_cell_size: f64, domain_size: Vector2<f64>) -> Mesh {
    let mut mesh = Mesh::new();
    
    // 1. Build Quadtree and Process Leaves
    let nx = (domain_size.x / max_cell_size).ceil() as usize;
    let ny = (domain_size.y / max_cell_size).ceil() as usize;
    
    let mut vertex_map: HashMap<(i64, i64), usize> = HashMap::new(); // Quantized coords -> index
    let quantize = |v: f64| (v * 100000.0).round() as i64;
    
    // Map (v_min, v_max) -> face_index
    let mut face_map: HashMap<(usize, usize), usize> = HashMap::new();

    // Collect all leaf polygons first
    let mut all_polys: Vec<Vec<Point2<f64>>> = Vec::new();

    for i in 0..nx {
        for j in 0..ny {
            let x0 = i as f64 * max_cell_size;
            let y0 = j as f64 * max_cell_size;
            let x1 = (x0 + max_cell_size).min(domain_size.x);
            let y1 = (y0 + max_cell_size).min(domain_size.y);
            
            let mut root = QuadNode::new(Point2::new(x0, y0), Point2::new(x1, y1));
            refine_node(&mut root, geo, min_cell_size);
            
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
                
                let all_outside = d00 >= 0.0 && d10 >= 0.0 && d11 >= 0.0 && d01 >= 0.0;
                
                if all_outside {
                    continue;
                }
                
                let mut poly_verts = Vec::new();
                let all_inside = d00 < 0.0 && d10 < 0.0 && d11 < 0.0 && d01 < 0.0;

                if all_inside {
                    // Rectangular cell
                    poly_verts.push(p00);
                    poly_verts.push(p10);
                    poly_verts.push(p11);
                    poly_verts.push(p01);
                } else {
                    // Cut cell
                    let corners = [p00, p10, p11, p01];
                    let dists = [d00, d10, d11, d01];
                    
                    for k in 0..4 {
                        let p_curr = corners[k];
                        let p_next = corners[(k + 1) % 4];
                        let d_curr = dists[k];
                        let d_next = dists[(k + 1) % 4];
                        
                        if d_curr < 0.0 {
                            poly_verts.push(p_curr);
                        }
                        
                        if (d_curr < 0.0 && d_next >= 0.0) || (d_curr >= 0.0 && d_next < 0.0) {
                            // Intersection
                            let denom = d_curr - d_next;
                            if denom.abs() < 1e-20 {
                                println!("Warning: denom too small in intersection: d_curr={}, d_next={}", d_curr, d_next);
                            }
                            let t = d_curr / denom;
                            if t.is_nan() {
                                println!("Warning: t is NaN. d_curr={}, d_next={}", d_curr, d_next);
                            }
                            let p_inter = p_curr + (p_next - p_curr) * t;
                            if p_inter.x.is_nan() || p_inter.y.is_nan() {
                                println!("Warning: p_inter is NaN. t={}, p_curr={:?}, p_next={:?}", t, p_curr, p_next);
                            }
                            poly_verts.push(p_inter);
                        }
                    }
                }
                
                if poly_verts.len() >= 3 {
                    all_polys.push(poly_verts);
                }
            }
        }
    }

    // 2. Imprint Hanging Nodes
    // Collect all unique vertices
    let mut unique_verts: Vec<Point2<f64>> = Vec::new();
    let mut vert_set: std::collections::HashSet<(i64, i64)> = std::collections::HashSet::new();
    
    for poly in &all_polys {
        for p in poly {
            let key = (quantize(p.x), quantize(p.y));
            if vert_set.insert(key) {
                unique_verts.push(*p);
            }
        }
    }

    // For each polygon, check if any unique vertex lies on its edges
    for poly in &mut all_polys {
        let mut new_poly = Vec::new();
        let n = poly.len();
        
        for k in 0..n {
            let p_curr = poly[k];
            let p_next = poly[(k + 1) % n];
            
            new_poly.push(p_curr);
            
            // Find vertices on this segment
            let mut on_segment = Vec::new();
            let seg_vec = p_next - p_curr;
            let seg_len_sq = seg_vec.norm_squared();
            
            if seg_len_sq < 1e-12 { continue; }

            for v in &unique_verts {
                // Check if v is p_curr or p_next
                let d_curr = (v - p_curr).norm_squared();
                let d_next = (v - p_next).norm_squared();
                
                if d_curr < 1e-12 || d_next < 1e-12 { continue; }
                
                // Check collinearity and bounds
                // Project v onto segment
                let v_vec = v - p_curr;
                let t = v_vec.dot(&seg_vec) / seg_len_sq;
                
                if t > 1e-6 && t < 1.0 - 1e-6 {
                    // Check distance to line
                    let proj = p_curr + seg_vec * t;
                    if (v - proj).norm_squared() < 1e-12 {
                        on_segment.push((t, *v));
                    }
                }
            }
            
            // Sort by t
            on_segment.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            
            for (_, v) in on_segment {
                new_poly.push(v);
            }
        }
        *poly = new_poly;
    }

    // 3. Create Mesh from Polygons
    for poly_verts in all_polys {
        // Create Cell
        let mut center = Vector2::new(0.0, 0.0);
        let mut area = 0.0;
        let n = poly_verts.len();
        
        // Get vertex indices
        let mut cell_v_indices = Vec::new();
        for p in &poly_verts {
            let key = (quantize(p.x), quantize(p.y));
            let idx = if let Some(&idx) = vertex_map.get(&key) {
                idx
            } else {
                let idx = mesh.vertices.len();
                mesh.vertices.push(Vertex { pos: *p });
                vertex_map.insert(key, idx);
                idx
            };
            cell_v_indices.push(idx);
        }

        // Polygon area and centroid
        for k in 0..n {
            let p_i = poly_verts[k];
            let p_j = poly_verts[(k + 1) % n];
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

