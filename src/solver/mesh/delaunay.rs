use nalgebra::{Point2, Vector2};
use std::collections::{HashMap, HashSet};

use super::geometry::Geometry;
use super::structs::{BoundaryType, Mesh};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Edge {
    pub v1: usize,
    pub v2: usize,
}

impl Edge {
    pub fn new(v1: usize, v2: usize) -> Self {
        if v1 < v2 {
            Self { v1, v2 }
        } else {
            Self { v1: v2, v2: v1 }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Triangle {
    pub v1: usize,
    pub v2: usize,
    pub v3: usize,
    pub circumcenter: Point2<f64>,
    pub r_sq: f64,
    // Neighbors opposite to edges:
    // 0: edge (v1, v2)
    // 1: edge (v2, v3)
    // 2: edge (v3, v1)
    pub neighbors: [Option<usize>; 3],
}

impl Triangle {
    pub fn new(
        v1: usize,
        v2: usize,
        v3: usize,
        p1: Point2<f64>,
        p2: Point2<f64>,
        p3: Point2<f64>,
    ) -> Self {
        // Enforce CCW
        let det = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
        let (v1, v2, v3, p1, p2, p3) = if det < 0.0 {
            (v1, v3, v2, p1, p3, p2)
        } else {
            (v1, v2, v3, p1, p2, p3)
        };

        let (circumcenter, r_sq) = Self::calculate_circumcircle(p1, p2, p3);
        Self {
            v1,
            v2,
            v3,
            circumcenter,
            r_sq,
            neighbors: [None; 3],
        }
    }

    pub fn calculate_circumcircle(
        a: Point2<f64>,
        b: Point2<f64>,
        c: Point2<f64>,
    ) -> (Point2<f64>, f64) {
        let d = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
        if d.abs() < 1e-12 {
            // Collinear or degenerate, return something safe but invalid
            return (Point2::origin(), f64::MAX);
        }
        let ux = ((a.x * a.x + a.y * a.y) * (b.y - c.y)
            + (b.x * b.x + b.y * b.y) * (c.y - a.y)
            + (c.x * c.x + c.y * c.y) * (a.y - b.y))
            / d;
        let uy = ((a.x * a.x + a.y * a.y) * (c.x - b.x)
            + (b.x * b.x + b.y * b.y) * (a.x - c.x)
            + (c.x * c.x + c.y * c.y) * (b.x - a.x))
            / d;
        let center = Point2::new(ux, uy);
        let r_sq = (center - a).norm_squared();
        (center, r_sq)
    }

    pub fn in_circumcircle(&self, p: Point2<f64>, points: &[Point2<f64>]) -> bool {
        // Robust check using determinant
        // We use the points array to get coordinates
        let a = points[self.v1];
        let b = points[self.v2];
        let c = points[self.v3];

        // Check orientation of a, b, c
        let det_abc = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
        
        // If det_abc is negative, points are clockwise. Swap b and c to make them CCW for the check.
        // Note: we don't change self.v2/v3, just local variables.
        let (b, c) = if det_abc < 0.0 { (c, b) } else { (b, c) };

        // Using relative coordinates to p improves precision
        let adx = a.x - p.x;
        let ady = a.y - p.y;
        let bdx = b.x - p.x;
        let bdy = b.y - p.y;
        let cdx = c.x - p.x;
        let cdy = c.y - p.y;

        let alift = adx * adx + ady * ady;
        let blift = bdx * bdx + bdy * bdy;
        let clift = cdx * cdx + cdy * cdy;

        let det = adx * (bdy * clift - cdy * blift)
                - ady * (bdx * clift - cdx * blift)
                + alift * (bdx * cdy - cdx * bdy);

        det > 1e-10 // Positive means inside
    }
}



use rand::Rng;
use rayon::prelude::*;
use wide::f64x4;

pub fn triangulate(
    geo: &(impl Geometry + Sync),
    min_cell_size: f64,
    max_cell_size: f64,
    growth_rate: f64,
    domain_size: Vector2<f64>,
) -> (Vec<Point2<f64>>, Vec<Triangle>, Vec<bool>) {
    // 1. Generate Points
    // Step 1a: Boundary Points
    let boundary_points = geo.get_boundary_points(min_cell_size);

    let mut points = Vec::new();
    let mut fixed_nodes = Vec::new();
    let mut unique_map = HashMap::new();
    let quantize = |x: f64, y: f64| ((x * 100000.0).round() as i64, (y * 100000.0).round() as i64);

    // Add boundary points
    for p in boundary_points {
        let key = quantize(p.x, p.y);
        if !unique_map.contains_key(&key) {
            unique_map.insert(key, points.len());
            points.push(p);
            fixed_nodes.push(true);
        }
    }

    // Step 1b: Interior Points using Poisson Disk Sampling
    let interior_points = generate_poisson_points(
        &points,
        geo,
        min_cell_size,
        max_cell_size,
        growth_rate,
        domain_size,
    );

    for p in interior_points {
        // Interior points are not fixed
        points.push(p);
        fixed_nodes.push(false);
    }

    // 2. Initial Triangulation
    let mut triangles = compute_triangulation(&points, domain_size, &fixed_nodes, geo);

    // 3. Smooth Generators (Laplacian Smoothing)
    // This helps to smooth out the sharp steps
    let smoothing_iters = 20;
    println!(
        "Starting generator smoothing for {} iterations...",
        smoothing_iters
    );
    for iter in 0..smoothing_iters {
        let (new_points, max_disp) = smooth_generators(
            &points,
            &triangles,
            &fixed_nodes,
            geo,
            min_cell_size,
            max_cell_size,
            growth_rate,
        );
        points = new_points;
        if iter % 10 == 0 {
            println!("  Gen Smooth iter {}: max disp = {:.6}", iter, max_disp);
        }
        triangles = compute_triangulation(&points, domain_size, &fixed_nodes, geo);
    }

    (points, triangles, fixed_nodes)
}

fn generate_poisson_points(
    boundary_points: &[Point2<f64>],
    geo: &(impl Geometry + Sync),
    min_cell_size: f64,
    max_cell_size: f64,
    growth_rate: f64,
    domain_size: Vector2<f64>,
) -> Vec<Point2<f64>> {
    let mut rng = rand::thread_rng();
    let r_min = min_cell_size;
    // Cell size for the background grid.
    // We use r_min / sqrt(2) so that each grid cell can contain at most one point.
    let cell_size = r_min / (2.0f64).sqrt();

    let grid_w = (domain_size.x / cell_size).ceil() as usize;
    let grid_h = (domain_size.y / cell_size).ceil() as usize;

    // Grid stores index of point in the `points` list (which includes boundary + generated)
    let mut grid: Vec<Option<usize>> = vec![None; grid_w * grid_h];

    let mut points = Vec::new(); // Local list of all points (boundary + new)
    let mut active_list = Vec::new(); // Indices into `points`

    // Initialize with boundary points
    for &p in boundary_points {
        let idx = points.len();
        points.push(p);
        active_list.push(idx);

        let gx = (p.x / cell_size).floor() as usize;
        let gy = (p.y / cell_size).floor() as usize;

        if gx < grid_w && gy < grid_h {
            grid[gy * grid_w + gx] = Some(idx);
        }
    }

    // Sizing function
    let get_radius = |p: Point2<f64>| -> f64 {
        let dist = geo.sdf(&p).abs();
        // Growth: r = min_size + (growth_rate - 1) * dist
        // But we want to cap at max_size
        let r = min_cell_size + (growth_rate - 1.0).max(0.0) * dist;
        r.min(max_cell_size)
    };

    let k = 30; // Max candidates per point

    while !active_list.is_empty() {
        // Pick a random point from active list
        let active_idx = rng.gen_range(0..active_list.len());
        let p_idx = active_list[active_idx];
        let p = points[p_idx];
        let r = get_radius(p);

        let mut found = false;
        for _ in 0..k {
            // Random point in annulus [r, 2r]
            let angle = rng.gen::<f64>() * 2.0 * std::f64::consts::PI;
            let dist = rng.gen_range(r..2.0 * r);
            let new_p = Point2::new(p.x + dist * angle.cos(), p.y + dist * angle.sin());

            // Check bounds
            if new_p.x < 0.0 || new_p.x > domain_size.x || new_p.y < 0.0 || new_p.y > domain_size.y
            {
                continue;
            }

            // Check geometry
            if !geo.is_inside(&new_p) {
                continue;
            }

            // Check neighbors
            let r_new = get_radius(new_p);

            let gx = (new_p.x / cell_size).floor() as isize;
            let gy = (new_p.y / cell_size).floor() as isize;

            let search_cells = (max_cell_size / cell_size).ceil() as isize;

            let mut conflict = false;

            'neighbor_check: for dy in -search_cells..=search_cells {
                for dx in -search_cells..=search_cells {
                    let nx = gx + dx;
                    let ny = gy + dy;

                    if nx >= 0 && nx < grid_w as isize && ny >= 0 && ny < grid_h as isize {
                        if let Some(n_idx) = grid[(ny as usize) * grid_w + (nx as usize)] {
                            let neighbor = points[n_idx];
                            let d2 = (neighbor - new_p).norm_squared();

                            // Check distance against radius of the NEW point (conservative)
                            // or max(r_new, r_neighbor)?
                            // Standard Bridson uses r (of the active point).
                            // Variable radius usually uses r_new.
                            let required_dist = r_new;

                            if d2 < required_dist * required_dist {
                                conflict = true;
                                break 'neighbor_check;
                            }
                        }
                    }
                }
            }

            if !conflict {
                // Add point
                let idx = points.len();
                points.push(new_p);
                active_list.push(idx);

                let gx = (new_p.x / cell_size).floor() as usize;
                let gy = (new_p.y / cell_size).floor() as usize;

                if gx < grid_w && gy < grid_h {
                    grid[gy * grid_w + gx] = Some(idx);
                }

                found = true;
                break;
            }
        }

        if !found {
            active_list.swap_remove(active_idx);
        }
    }

    // Return only the new points (exclude initial boundary points)
    // boundary_points.len() is the count of initial points.
    points.into_iter().skip(boundary_points.len()).collect()
}

fn smooth_generators(
    points: &[Point2<f64>],
    triangles: &[Triangle],
    fixed_nodes: &[bool],
    geo: &(impl Geometry + Sync),
    min_cell_size: f64,
    max_cell_size: f64,
    growth_rate: f64,
) -> (Vec<Point2<f64>>, f64) {
    let n = points.len();
    let mut new_points = points.to_vec();
    let mut adj = vec![Vec::new(); n];

    for t in triangles {
        adj[t.v1].push(t.v2);
        adj[t.v1].push(t.v3);
        adj[t.v2].push(t.v1);
        adj[t.v2].push(t.v3);
        adj[t.v3].push(t.v1);
        adj[t.v3].push(t.v2);
    }

    // Sizing function (same as in generate_poisson_points)
    let get_radius = |p: Point2<f64>| -> f64 {
        let dist = geo.sdf(&p).abs();
        let r = min_cell_size + (growth_rate - 1.0).max(0.0) * dist;
        r.min(max_cell_size)
    };

    let compute_new_pos = |i: usize| -> (Point2<f64>, f64) {
        if fixed_nodes[i] {
            return (points[i], 0.0);
        }
        let neighbors = &adj[i];
        if neighbors.is_empty() {
            return (points[i], 0.0);
        }

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_w = 0.0;

        let mut chunks = neighbors.chunks_exact(4);
        for chunk in chunks.by_ref() {
            let n0 = chunk[0];
            let n1 = chunk[1];
            let n2 = chunk[2];
            let n3 = chunk[3];

            let px = f64x4::new([points[n0].x, points[n1].x, points[n2].x, points[n3].x]);
            let py = f64x4::new([points[n0].y, points[n1].y, points[n2].y, points[n3].y]);

            let dist = geo.sdf_batch(px, py).abs();
            
            let min_sz = f64x4::splat(min_cell_size);
            let max_sz = f64x4::splat(max_cell_size);
            let gr = f64x4::splat((growth_rate - 1.0).max(0.0));
            
            let r = (min_sz + gr * dist).min(max_sz);
            let w = f64x4::splat(1.0) / r.max(f64x4::splat(1e-6));
            
            let wx = px * w;
            let wy = py * w;
            
            let arr_wx: [f64; 4] = wx.into();
            let arr_wy: [f64; 4] = wy.into();
            let arr_w: [f64; 4] = w.into();
            
            sum_x += arr_wx[0] + arr_wx[1] + arr_wx[2] + arr_wx[3];
            sum_y += arr_wy[0] + arr_wy[1] + arr_wy[2] + arr_wy[3];
            sum_w += arr_w[0] + arr_w[1] + arr_w[2] + arr_w[3];
        }

        for &neigh in chunks.remainder() {
            let r = get_radius(points[neigh]);
            let w = 1.0 / r.max(1e-6);

            sum_x += points[neigh].x * w;
            sum_y += points[neigh].y * w;
            sum_w += w;
        }
        let avg_x = sum_x / sum_w;
        let avg_y = sum_y / sum_w;

        let alpha = 0.1;
        let p_new = Point2::new(
            points[i].x + (avg_x - points[i].x) * alpha,
            points[i].y + (avg_y - points[i].y) * alpha,
        );

        if geo.is_inside(&p_new) {
            let disp = (p_new - points[i]).norm();
            (p_new, disp)
        } else {
            (points[i], 0.0)
        }
    };

    let max_disp;
    if n < 5000 {
        let mut m_disp = 0.0;
        for i in 0..n {
            let (p_new, disp) = compute_new_pos(i);
            if disp > m_disp {
                m_disp = disp;
            }
            new_points[i] = p_new;
        }
        max_disp = m_disp;
    } else {
        max_disp = new_points
            .par_iter_mut()
            .enumerate()
            .map(|(i, p_out)| {
                let (p_new, disp) = compute_new_pos(i);
                *p_out = p_new;
                disp
            })
            .reduce(|| 0.0, |a, b| a.max(b));
    }

    (new_points, max_disp)
}

struct DelaunayTriangulation {
    triangles: Vec<Triangle>,
    active: Vec<bool>,
}

impl DelaunayTriangulation {
    fn new(capacity: usize) -> Self {
        Self {
            triangles: Vec::with_capacity(capacity),
            active: Vec::with_capacity(capacity),
        }
    }

    fn add_triangle(&mut self, t: Triangle) -> usize {
        let idx = self.triangles.len();
        self.triangles.push(t);
        self.active.push(true);
        idx
    }

    fn deactivate(&mut self, idx: usize) {
        self.active[idx] = false;
    }
}

fn compute_triangulation(
    points: &[Point2<f64>],
    domain_size: Vector2<f64>,
    fixed_nodes: &[bool],
    geo: &(impl Geometry + Sync),
) -> Vec<Triangle> {
    let n_points = points.len();
    let mut dt = DelaunayTriangulation::new(n_points * 3);

    // Super-triangle
    let margin = 10.0 * domain_size.norm();
    let p1 = Point2::new(-margin, -margin);
    let p2 = Point2::new(2.0 * margin + domain_size.x, -margin);
    let p3 = Point2::new(-margin, 2.0 * margin + domain_size.y);

    let mut working_points = points.to_vec();
    working_points.push(p1);
    working_points.push(p2);
    working_points.push(p3);

    let st_v1 = n_points;
    let st_v2 = n_points + 1;
    let st_v3 = n_points + 2;

    let t_super = Triangle::new(st_v1, st_v2, st_v3, p1, p2, p3);
    dt.add_triangle(t_super);

    let mut last_tri_idx = 0;

    for (i, &p) in points.iter().enumerate() {
        let mut curr = last_tri_idx;
        
        // Walk to find triangle containing p
        let mut iter = 0;
        let max_iter = dt.triangles.len(); // Safety break
        loop {
            if iter > max_iter {
                // Fallback to linear search if walk fails
                if let Some(idx) = dt.active.iter().position(|&x| x) {
                    curr = idx;
                }
                break;
            }
            iter += 1;

            if !dt.active[curr] {
                 if let Some(idx) = dt.active.iter().position(|&x| x) {
                    curr = idx;
                 } else {
                     break; 
                 }
            }
            
            let t = dt.triangles[curr];
            let p_a = working_points[t.v1];
            let p_b = working_points[t.v2];
            let p_c = working_points[t.v3];

            let cross = |a: Point2<f64>, b: Point2<f64>, p: Point2<f64>| {
                (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)
            };

            if cross(p_a, p_b, p) < -1e-10 {
                if let Some(n) = t.neighbors[0] {
                    curr = n;
                    continue;
                }
            }
            if cross(p_b, p_c, p) < -1e-10 {
                if let Some(n) = t.neighbors[1] {
                    curr = n;
                    continue;
                }
            }
            if cross(p_c, p_a, p) < -1e-10 {
                if let Some(n) = t.neighbors[2] {
                    curr = n;
                    continue;
                }
            }
            break;
        }

        let start_bad = curr;
        
        let mut bad_triangles = Vec::new();
        let mut queue = Vec::new();
        let mut visited = HashSet::new();

        if dt.triangles[start_bad].in_circumcircle(p, &working_points) {
            queue.push(start_bad);
            visited.insert(start_bad);
        }

        while let Some(t_idx) = queue.pop() {
            let t = dt.triangles[t_idx];
            if t.in_circumcircle(p, &working_points) {
                bad_triangles.push(t_idx);
                for &n_opt in &t.neighbors {
                    if let Some(n) = n_opt {
                        if !visited.contains(&n) && dt.active[n] {
                            visited.insert(n);
                            queue.push(n);
                        }
                    }
                }
            }
        }

        if bad_triangles.is_empty() {
            continue;
        }

        let mut boundary_edges = Vec::new(); 

        for &t_idx in &bad_triangles {
            let t = dt.triangles[t_idx];
            let edges = [(t.v1, t.v2, 0), (t.v2, t.v3, 1), (t.v3, t.v1, 2)];
            
            for (v_start, v_end, neigh_idx) in edges {
                let neighbor = t.neighbors[neigh_idx];
                let is_boundary_edge = match neighbor {
                    Some(n) => !bad_triangles.contains(&n),
                    None => true,
                };

                if is_boundary_edge {
                    boundary_edges.push((v_start, v_end, neighbor));
                }
            }
        }

        for &t_idx in &bad_triangles {
            dt.deactivate(t_idx);
        }

        let mut new_tri_indices = Vec::new();
        
        for &(u, v, neighbor) in &boundary_edges {
            let new_t = Triangle::new(u, v, i, working_points[u], working_points[v], working_points[i]);
            let idx = dt.add_triangle(new_t);
            new_tri_indices.push(idx);
            
            // Assign neighbor for the boundary edge
            let t = &mut dt.triangles[idx];
            if t.v1 == u && t.v2 == v { t.neighbors[0] = neighbor; }
            else if t.v2 == u && t.v3 == v { t.neighbors[1] = neighbor; }
            else if t.v3 == u && t.v1 == v { t.neighbors[2] = neighbor; }
            // Handle swapped cases if any (though u,v should match edge direction)
            else if t.v1 == v && t.v2 == u { t.neighbors[0] = neighbor; }
            else if t.v2 == v && t.v3 == u { t.neighbors[1] = neighbor; }
            else if t.v3 == v && t.v1 == u { t.neighbors[2] = neighbor; }
            
            if let Some(n_idx) = neighbor {
                let n_tri = &mut dt.triangles[n_idx];
                if n_tri.v1 == v && n_tri.v2 == u { n_tri.neighbors[0] = Some(idx); }
                else if n_tri.v2 == v && n_tri.v3 == u { n_tri.neighbors[1] = Some(idx); }
                else if n_tri.v3 == v && n_tri.v1 == u { n_tri.neighbors[2] = Some(idx); }
            }
        }
        
        let n_new = new_tri_indices.len();
        for k in 0..n_new {
            let t_idx = new_tri_indices[k];
            let t = dt.triangles[t_idx];
            
            for edge_idx in 0..3 {
                if dt.triangles[t_idx].neighbors[edge_idx].is_some() {
                    continue;
                }
                
                let (ea, eb) = match edge_idx {
                    0 => (t.v1, t.v2),
                    1 => (t.v2, t.v3),
                    2 => (t.v3, t.v1),
                    _ => unreachable!(),
                };
                
                for m in 0..n_new {
                    if k == m { continue; }
                    let other_idx = new_tri_indices[m];
                    let other = dt.triangles[other_idx];
                    
                    if (other.v1 == eb && other.v2 == ea) || 
                       (other.v2 == eb && other.v3 == ea) || 
                       (other.v3 == eb && other.v1 == ea) {
                        dt.triangles[t_idx].neighbors[edge_idx] = Some(other_idx);
                        break;
                    }
                }
            }
        }
        
        if !new_tri_indices.is_empty() {
            last_tri_idx = new_tri_indices[0];
        }
    }

    dt.triangles.into_iter().enumerate().filter_map(|(i, t)| {
        if !dt.active[i] { return None; }
        if t.v1 >= n_points || t.v2 >= n_points || t.v3 >= n_points { return None; }
        
        if !fixed_nodes[t.v1] || !fixed_nodes[t.v2] || !fixed_nodes[t.v3] {
             return Some(t);
        }
        let p1 = points[t.v1];
        let p2 = points[t.v2];
        let p3 = points[t.v3];
        let centroid = Point2::new((p1.x + p2.x + p3.x) / 3.0, (p1.y + p2.y + p3.y) / 3.0);
        if geo.is_inside(&centroid) {
            Some(t)
        } else {
            None
        }
    }).collect()
}

pub fn generate_delaunay_mesh(
    geo: &(impl Geometry + Sync),
    min_cell_size: f64,
    max_cell_size: f64,
    growth_rate: f64,
    domain_size: Vector2<f64>,
) -> Mesh {
    let mut mesh = Mesh::new();
    mesh.cell_face_offsets.push(0);
    mesh.cell_vertex_offsets.push(0);

    let (points, triangles, fixed_nodes) =
        triangulate(geo, min_cell_size, max_cell_size, growth_rate, domain_size);

    // 3. Convert to Mesh
    mesh.vx = points.iter().map(|p| p.x).collect();
    mesh.vy = points.iter().map(|p| p.y).collect();
    mesh.v_fixed = fixed_nodes;

    // Build faces and cells
    // Map edge -> face_index
    let mut edge_face_map: HashMap<Edge, usize> = HashMap::new();

    for t in triangles {
        let cell_idx = mesh.cell_cx.len();

        // Calculate centroid and area
        let p1 = points[t.v1];
        let p2 = points[t.v2];
        let p3 = points[t.v3];

        let center = Point2::new((p1.x + p2.x + p3.x) / 3.0, (p1.y + p2.y + p3.y) / 3.0);
        let area = 0.5 * ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y)).abs();

        mesh.cell_cx.push(center.x);
        mesh.cell_cy.push(center.y);
        mesh.cell_vol.push(area);

        let edges = [
            Edge::new(t.v1, t.v2),
            Edge::new(t.v2, t.v3),
            Edge::new(t.v3, t.v1),
        ];

        for edge in edges {
            if let Some(&face_idx) = edge_face_map.get(&edge) {
                mesh.face_neighbor[face_idx] = Some(cell_idx);
                mesh.face_boundary[face_idx] = None; // Internal
                mesh.cell_faces.push(face_idx);
            } else {
                let face_idx = mesh.face_cx.len();
                let p_a = points[edge.v1];
                let p_b = points[edge.v2];
                let f_center = Point2::new((p_a.x + p_b.x) / 2.0, (p_a.y + p_b.y) / 2.0);
                let f_len = (p_a - p_b).norm();
                let normal = Vector2::new(p_b.y - p_a.y, p_a.x - p_b.x).normalize();

                mesh.face_v1.push(edge.v1);
                mesh.face_v2.push(edge.v2);
                mesh.face_owner.push(cell_idx);
                mesh.face_neighbor.push(None);

                // Determine boundary
                // If edge vertices are both fixed, it MIGHT be a boundary face.
                // But internal edges can also connect fixed vertices (e.g. corner to corner).
                // Better check: is the face center on boundary?
                let is_boundary = mesh.v_fixed[edge.v1] && mesh.v_fixed[edge.v2];
                let boundary_type = if is_boundary {
                    // Check position
                    if f_center.x < 1e-6 {
                        Some(BoundaryType::Inlet)
                    } else if (f_center.x - domain_size.x).abs() < 1e-6 {
                        Some(BoundaryType::Outlet)
                    } else {
                        Some(BoundaryType::Wall)
                    }
                } else {
                    None
                };

                mesh.face_boundary.push(boundary_type);
                mesh.face_nx.push(normal.x);
                mesh.face_ny.push(normal.y);
                mesh.face_area.push(f_len);
                mesh.face_cx.push(f_center.x);
                mesh.face_cy.push(f_center.y);

                edge_face_map.insert(edge, face_idx);
                mesh.cell_faces.push(face_idx);
            }
        }

        mesh.cell_face_offsets.push(mesh.cell_faces.len());

        // Vertices
        mesh.cell_vertices.push(t.v1);
        mesh.cell_vertices.push(t.v2);
        mesh.cell_vertices.push(t.v3);
        mesh.cell_vertex_offsets.push(mesh.cell_vertices.len());
    }

    // Fix normals (must point from owner to neighbor)
    // And handle single-neighbor faces (boundary)
    for i in 0..mesh.face_cx.len() {
        let owner = mesh.face_owner[i];
        let c_owner = Point2::new(mesh.cell_cx[owner], mesh.cell_cy[owner]);
        let f_center = Point2::new(mesh.face_cx[i], mesh.face_cy[i]);
        let normal = Vector2::new(mesh.face_nx[i], mesh.face_ny[i]);

        if (f_center - c_owner).dot(&normal) < 0.0 {
            mesh.face_nx[i] = -normal.x;
            mesh.face_ny[i] = -normal.y;
        }
    }

    mesh
}
