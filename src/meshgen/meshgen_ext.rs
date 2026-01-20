use nalgebra::{Point2, Vector2};
use rayon::prelude::*;

use super::geometry::Geometry;
use crate::solver::mesh::Mesh;

impl Mesh {
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
