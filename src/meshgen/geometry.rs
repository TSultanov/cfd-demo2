use super::meshless::{circle_loop, polyline_loop, BoundaryLoop};
use super::tolerances::MeshgenTolerances;
use nalgebra::{Point2, Vector2};
use wide::f64x4;

// Geometry definition for CutCell
pub trait Geometry {
    fn is_inside(&self, p: &Point2<f64>) -> bool;
    // Returns distance to surface. Negative inside.
    fn sdf(&self, p: &Point2<f64>) -> f64;

    fn sdf_batch(&self, px: f64x4, py: f64x4) -> f64x4 {
        let arr_x: [f64; 4] = px.into();
        let arr_y: [f64; 4] = py.into();
        let mut res = [0.0; 4];
        for i in 0..4 {
            res[i] = self.sdf(&Point2::new(arr_x[i], arr_y[i]));
        }
        f64x4::from(res)
    }

    fn get_boundary_points(&self, spacing: f64) -> Vec<Point2<f64>>;

    /// Ordered, closed boundary polylines with per-segment BC tags, walked
    /// with the fluid on the LEFT (outer loop CCW, embedded holes CW) — the
    /// meshless engine's boundary input. Tags follow `classify_boundary` on
    /// segment midpoints with the `Wall` fallback.
    ///
    /// Default: the domain bounding box only — correct solely for geometries
    /// whose fluid region is the whole box; anything with embedded or curved
    /// boundaries must override. Note `get_boundary_points` implementations
    /// are deliberately NOT derived from the loops: their point order feeds
    /// the Poisson RNG of the generators and must stay byte-stable.
    fn get_boundary_loops(
        &self,
        spacing: f64,
        domain: Vector2<f64>,
        tol: &MeshgenTolerances,
    ) -> Vec<BoundaryLoop> {
        vec![polyline_loop(
            &[
                Point2::new(0.0, 0.0),
                Point2::new(domain.x, 0.0),
                Point2::new(domain.x, domain.y),
                Point2::new(0.0, domain.y),
            ],
            spacing,
            domain,
            tol,
        )]
    }
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

    fn sdf_batch(&self, px: f64x4, py: f64x4) -> f64x4 {
        let half_len = f64x4::splat(self.length / 2.0);
        let half_height = f64x4::splat(self.height / 2.0);

        let dx = (px - half_len).abs() - half_len;
        let dy = (py - half_height).abs() - half_height;

        let zero = f64x4::splat(0.0);
        let dx_max = dx.max(zero);
        let dy_max = dy.max(zero);

        let box_dist = dx.max(dy).min(zero) + (dx_max * dx_max + dy_max * dy_max).sqrt();

        let obs_cx = f64x4::splat(self.obstacle_center.x);
        let obs_cy = f64x4::splat(self.obstacle_center.y);
        let obs_r = f64x4::splat(self.obstacle_radius);

        let diff_x = px - obs_cx;
        let diff_y = py - obs_cy;
        let circle_dist = (diff_x * diff_x + diff_y * diff_y).sqrt() - obs_r;

        box_dist.max(-circle_dist)
    }

    fn get_boundary_points(&self, spacing: f64) -> Vec<Point2<f64>> {
        let mut points = Vec::new();

        // Outer box
        let nx = (self.length / spacing).ceil() as usize;
        let ny = (self.height / spacing).ceil() as usize;

        for i in 0..=nx {
            let x = (i as f64 * spacing).min(self.length);
            points.push(Point2::new(x, 0.0));
            points.push(Point2::new(x, self.height));
        }
        for i in 0..=ny {
            let y = (i as f64 * spacing).min(self.height);
            points.push(Point2::new(0.0, y));
            points.push(Point2::new(self.length, y));
        }

        // Obstacle
        let circumference = 2.0 * std::f64::consts::PI * self.obstacle_radius;
        let n_obs = (circumference / spacing).ceil() as usize;
        for i in 0..n_obs {
            let theta = 2.0 * std::f64::consts::PI * i as f64 / n_obs as f64;
            let x = self.obstacle_center.x + self.obstacle_radius * theta.cos();
            let y = self.obstacle_center.y + self.obstacle_radius * theta.sin();
            points.push(Point2::new(x, y));
        }

        points
    }

    fn get_boundary_loops(
        &self,
        spacing: f64,
        domain: Vector2<f64>,
        tol: &MeshgenTolerances,
    ) -> Vec<BoundaryLoop> {
        vec![
            // Outer channel box (CCW).
            polyline_loop(
                &[
                    Point2::new(0.0, 0.0),
                    Point2::new(self.length, 0.0),
                    Point2::new(self.length, self.height),
                    Point2::new(0.0, self.height),
                ],
                spacing,
                domain,
                tol,
            ),
            // Embedded obstacle (CW hole; all segments off-box ⇒ Wall).
            circle_loop(self.obstacle_center, self.obstacle_radius, spacing, domain, tol),
        ]
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

    fn sdf_batch(&self, px: f64x4, py: f64x4) -> f64x4 {
        let half_len = f64x4::splat(self.length / 2.0);
        let half_h_out = f64x4::splat(self.height_outlet / 2.0);

        let outer_box_dx = (px - half_len).abs() - half_len;
        let outer_box_dy = (py - half_h_out).abs() - half_h_out;

        let zero = f64x4::splat(0.0);
        let ob_dx_max = outer_box_dx.max(zero);
        let ob_dy_max = outer_box_dy.max(zero);

        let outer_dist = outer_box_dx.max(outer_box_dy).min(zero)
            + (ob_dx_max * ob_dx_max + ob_dy_max * ob_dy_max).sqrt();

        let step_h = self.height_outlet - self.height_inlet;
        let step_w = self.step_x;

        let block_cx = f64x4::splat(step_w / 2.0);
        let block_cy = f64x4::splat(step_h / 2.0);
        let half_step_w = f64x4::splat(step_w / 2.0);
        let half_step_h = f64x4::splat(step_h / 2.0);

        let block_dx = (px - block_cx).abs() - half_step_w;
        let block_dy = (py - block_cy).abs() - half_step_h;

        let b_dx_max = block_dx.max(zero);
        let b_dy_max = block_dy.max(zero);

        let block_dist =
            block_dx.max(block_dy).min(zero) + (b_dx_max * b_dx_max + b_dy_max * b_dy_max).sqrt();

        outer_dist.max(-block_dist)
    }

    fn get_boundary_points(&self, spacing: f64) -> Vec<Point2<f64>> {
        let mut points = Vec::new();
        // Walk the perimeter.
        // Vertices: (0, h_out), (L, h_out), (L, 0), (step_x, 0), (step_x, step_h), (0, step_h)

        let step_h = self.height_outlet - self.height_inlet;

        let segments = [
            (
                Point2::new(0.0, self.height_outlet),
                Point2::new(self.length, self.height_outlet),
            ), // Top
            (
                Point2::new(self.length, self.height_outlet),
                Point2::new(self.length, 0.0),
            ), // Right
            (Point2::new(self.length, 0.0), Point2::new(self.step_x, 0.0)), // Bottom Right
            (
                Point2::new(self.step_x, 0.0),
                Point2::new(self.step_x, step_h),
            ), // Step Vertical
            (Point2::new(self.step_x, step_h), Point2::new(0.0, step_h)), // Step Horizontal (Inlet bottom)
            (
                Point2::new(0.0, step_h),
                Point2::new(0.0, self.height_outlet),
            ), // Inlet Left
        ];

        for (p1, p2) in segments {
            let dist = (p1 - p2).norm();
            let n = (dist / spacing).ceil() as usize;
            for i in 0..n {
                let t = i as f64 / n as f64;
                points.push(p1 + (p2 - p1) * t);
            }
        }
        points
    }

    fn get_boundary_loops(
        &self,
        spacing: f64,
        domain: Vector2<f64>,
        tol: &MeshgenTolerances,
    ) -> Vec<BoundaryLoop> {
        let step_h = self.height_outlet - self.height_inlet;
        // Single CCW walk (fluid on the left); (step_x, step_h) is the one
        // reflex corner, handled by the guard-seed policy in `boundary_seeds`.
        vec![polyline_loop(
            &[
                Point2::new(self.step_x, 0.0),
                Point2::new(self.length, 0.0),
                Point2::new(self.length, self.height_outlet),
                Point2::new(0.0, self.height_outlet),
                Point2::new(0.0, step_h),
                Point2::new(self.step_x, step_h),
            ],
            spacing,
            domain,
            tol,
        )]
    }
}

/// Converging–diverging nozzle: a channel with a flat bottom wall at `y = 0` and
/// a shaped top wall at `y = nozzle_height(x)`. The channel narrows from `height`
/// at the inlet to `throat_height` at `throat_frac * length`, then widens to
/// `exit_height`. This is the same wall profile the body-fitted structured mesh
/// (`generate_structured_nozzle_mesh`) uses, so cut-cell / Delaunay / Voronoi
/// meshes conform to an identical geometry.
///
/// The bounding box is `[0, length] × [0, height]` (the inlet is the tallest
/// section), which is what `MeshgenTolerances::classify_boundary` expects: the
/// left edge is tagged `Inlet`, the right edge `Outlet`, and the flat bottom
/// `Wall`. The curved top wall sits below `y = height` everywhere except the
/// inlet, so it is left untagged by `classify_boundary` and must be closed as a
/// no-slip wall by the caller (cut-cell does this internally; see
/// `generate_cut_cell_mesh`).
pub struct Nozzle {
    pub length: f64,
    /// Inlet height, and the bounding-box height.
    pub height: f64,
    pub throat_height: f64,
    pub throat_frac: f64,
    pub exit_height: f64,
}

impl Nozzle {
    /// Top-wall height at absolute position `x` (delegates to the shared profile
    /// so the SDF matches the structured mesh exactly).
    fn top(&self, x: f64) -> f64 {
        let xi = (x / self.length).clamp(0.0, 1.0);
        crate::solver::mesh::structured::nozzle_height(
            xi,
            self.height,
            self.throat_height,
            self.throat_frac,
            self.exit_height,
        )
    }
}

impl Geometry for Nozzle {
    fn is_inside(&self, p: &Point2<f64>) -> bool {
        self.sdf(p) < 0.0
    }

    fn sdf(&self, p: &Point2<f64>) -> f64 {
        // Intersection of four half-spaces: x>=0, x<=length, y>=0, y<=top(x).
        // Each term is a signed distance (negative inside), and the intersection
        // SDF is their max. The top term uses the vertical gap to the wall — an
        // approximation for the (gently sloped) curved wall, exact elsewhere.
        let d_left = -p.x;
        let d_right = p.x - self.length;
        let d_bottom = -p.y;
        let d_top = p.y - self.top(p.x);
        d_left.max(d_right).max(d_bottom).max(d_top)
    }

    fn get_boundary_points(&self, spacing: f64) -> Vec<Point2<f64>> {
        let mut points = Vec::new();

        // Bottom (flat) and top (curved) walls, sampled along x.
        let nx = (self.length / spacing).ceil().max(1.0) as usize;
        for i in 0..=nx {
            let x = (i as f64 * spacing).min(self.length);
            points.push(Point2::new(x, 0.0));
            points.push(Point2::new(x, self.top(x)));
        }

        // Inlet (left, full height) and outlet (right, exit height) edges.
        let n_in = (self.height / spacing).ceil().max(1.0) as usize;
        for i in 0..=n_in {
            let y = (i as f64 * spacing).min(self.height);
            points.push(Point2::new(0.0, y));
        }
        let n_out = (self.exit_height / spacing).ceil().max(1.0) as usize;
        for i in 0..=n_out {
            let y = (i as f64 * spacing).min(self.exit_height);
            points.push(Point2::new(self.length, y));
        }

        points
    }

    fn get_boundary_loops(
        &self,
        spacing: f64,
        domain: Vector2<f64>,
        tol: &MeshgenTolerances,
    ) -> Vec<BoundaryLoop> {
        // Single CCW walk, fluid on the left. The curved top wall is sampled
        // at uniform x from the shared `nozzle_height` profile; its endpoint
        // heights `top(0)`/`top(length)` are used verbatim (instead of
        // `height`/`exit_height`) so the walk closes exactly in f64.
        let mut pts = Vec::new();
        // Bottom wall (0, 0) -> (length, 0).
        let n_b = ((self.length / spacing).ceil() as usize).max(1);
        for i in 0..n_b {
            pts.push(Point2::new(i as f64 / n_b as f64 * self.length, 0.0));
        }
        // Outlet (length, 0) -> (length, top(length)).
        let y_out = self.top(self.length);
        let n_r = ((y_out / spacing).ceil() as usize).max(1);
        for i in 0..n_r {
            pts.push(Point2::new(self.length, i as f64 / n_r as f64 * y_out));
        }
        // Top wall, walked right -> left (fluid below = on the left).
        let n_t = ((self.length / spacing).ceil() as usize).max(1);
        for k in 0..n_t {
            let x = (n_t - k) as f64 / n_t as f64 * self.length;
            pts.push(Point2::new(x, self.top(x)));
        }
        // Inlet (0, top(0)) -> (0, 0).
        let y_in = self.top(0.0);
        let n_l = ((y_in / spacing).ceil() as usize).max(1);
        for i in 0..n_l {
            pts.push(Point2::new(0.0, (n_l - i) as f64 / n_l as f64 * y_in));
        }
        vec![BoundaryLoop::from_points(pts, domain, tol)]
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

    fn sdf_batch(&self, px: f64x4, py: f64x4) -> f64x4 {
        let half_len = f64x4::splat(self.length / 2.0);
        let half_height = f64x4::splat(self.height / 2.0);

        let dx = (px - half_len).abs() - half_len;
        let dy = (py - half_height).abs() - half_height;

        let zero = f64x4::splat(0.0);
        let dx_max = dx.max(zero);
        let dy_max = dy.max(zero);

        dx.max(dy).min(zero) + (dx_max * dx_max + dy_max * dy_max).sqrt()
    }

    fn get_boundary_points(&self, spacing: f64) -> Vec<Point2<f64>> {
        let mut points = Vec::new();
        let nx = (self.length / spacing).ceil() as usize;
        let ny = (self.height / spacing).ceil() as usize;

        for i in 0..=nx {
            let x = (i as f64 * spacing).min(self.length);
            points.push(Point2::new(x, 0.0));
            points.push(Point2::new(x, self.height));
        }
        for i in 0..=ny {
            let y = (i as f64 * spacing).min(self.height);
            points.push(Point2::new(0.0, y));
            points.push(Point2::new(self.length, y));
        }
        points
    }

    fn get_boundary_loops(
        &self,
        spacing: f64,
        domain: Vector2<f64>,
        tol: &MeshgenTolerances,
    ) -> Vec<BoundaryLoop> {
        vec![polyline_loop(
            &[
                Point2::new(0.0, 0.0),
                Point2::new(self.length, 0.0),
                Point2::new(self.length, self.height),
                Point2::new(0.0, self.height),
            ],
            spacing,
            domain,
            tol,
        )]
    }
}
