use super::geometry::Geometry;
use nalgebra::{Point2, Vector2};
use rayon::prelude::*;

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

    pub fn recalculate_geometry(&mut self) {
        // 1. Recalculate Faces
        let vx = &self.vx;
        let vy = &self.vy;
        let face_v1 = &self.face_v1;
        let face_v2 = &self.face_v2;

        self.face_cx
            .par_iter_mut()
            .zip(&mut self.face_cy)
            .zip(&mut self.face_area)
            .zip(&mut self.face_nx)
            .zip(&mut self.face_ny)
            .enumerate()
            .for_each(|(i, ((((cx, cy), area), nx), ny))| {
                let v0_idx = face_v1[i];
                let v1_idx = face_v2[i];

                let v0 = Point2::new(vx[v0_idx], vy[v0_idx]);
                let v1 = Point2::new(vx[v1_idx], vy[v1_idx]);

                let center = Point2::from((v0.coords + v1.coords) * 0.5);
                *cx = center.x;
                *cy = center.y;

                let edge_vec = v1 - v0;
                *area = edge_vec.norm();

                // Preserve normal orientation
                let tangent = edge_vec.normalize();
                let mut normal = Vector2::new(tangent.y, -tangent.x);

                let current_normal = Vector2::new(*nx, *ny);
                if normal.dot(&current_normal) < 0.0 {
                    normal = -normal;
                }
                *nx = normal.x;
                *ny = normal.y;
            });

        // 2. Recalculate Cells
        let cell_vertex_offsets = &self.cell_vertex_offsets;
        let cell_vertices = &self.cell_vertices;

        self.cell_cx
            .par_iter_mut()
            .zip(&mut self.cell_cy)
            .zip(&mut self.cell_vol)
            .enumerate()
            .for_each(|(i, ((cx_out, cy_out), vol_out))| {
                let mut center = Vector2::zeros();
                let start = cell_vertex_offsets[i];
                let end = cell_vertex_offsets[i + 1];
                let n = end - start;

                // Polygon Area and Centroid
                let mut signed_area = 0.0;
                let mut c_x = 0.0;
                let mut c_y = 0.0;

                for k in 0..n {
                    let idx0 = cell_vertices[start + k];
                    let idx1 = cell_vertices[start + (k + 1) % n];

                    let p0_x = vx[idx0];
                    let p0_y = vy[idx0];
                    let p1_x = vx[idx1];
                    let p1_y = vy[idx1];

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
                        let idx = cell_vertices[start + k];
                        center.x += vx[idx];
                        center.y += vy[idx];
                    }
                    center /= n as f64;
                }

                *cx_out = center.x;
                *cy_out = center.y;
                *vol_out = area;
            });
    }

    pub fn smooth<G: Geometry + Sync>(&mut self, geo: &G, target_skew: f64, max_iterations: usize) {
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

            let (new_vx, new_vy): (Vec<f64>, Vec<f64>) = (0..n_verts)
                .into_par_iter()
                .map(|i| {
                    let x_old = self.vx[i];
                    let y_old = self.vy[i];

                    // If on domain box, fix it
                    if is_on_box(x_old, y_old) {
                        return (x_old, y_old);
                    }

                    if adj[i].is_empty() {
                        return (x_old, y_old);
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
                        (x_old, y_old)
                    } else {
                        (x_new, y_new)
                    }
                })
                .unzip();

            self.vx = new_vx;
            self.vy = new_vy;
        }

        self.recalculate_geometry();
        println!("Final skewness: {:.6}", self.calculate_max_skewness());
    }

    pub fn calculate_max_skewness(&self) -> f64 {
        (0..self.face_cx.len())
            .into_par_iter()
            .map(|i| {
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
                1.0 - d_norm.dot(&normal).abs()
            })
            .reduce(|| 0.0, f64::max)
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
