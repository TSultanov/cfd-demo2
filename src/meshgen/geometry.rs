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

    // Returns boundary points with given spacing
    fn get_boundary_points(&self, spacing: f64) -> Vec<Point2<f64>>;
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
        // Bottom
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
        // Simplified boundary generation: just walk the perimeter
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
}
