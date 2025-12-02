use super::geometry::Geometry;
use super::utils::{compute_normal, intersect_lines};
use nalgebra::{Point2, Vector2};
use rayon::prelude::*;

pub struct QuadNode {
    pub bounds: (Point2<f64>, Point2<f64>), // min, max
    pub children: Option<[Box<QuadNode>; 4]>,
    pub is_leaf: bool,
}

impl QuadNode {
    pub fn new(min: Point2<f64>, max: Point2<f64>) -> Self {
        Self {
            bounds: (min, max),
            children: None,
            is_leaf: true,
        }
    }

    pub fn subdivide(&mut self) {
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

pub fn refine_node(node: &mut QuadNode, geo: &impl Geometry, min_size: f64, growth_rate: f64) {
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

    let mut should_split = has_inside && has_outside;

    // Growth rate restriction
    if !should_split {
        // Even if not crossing boundary, we might need to split if we are close to boundary
        // and the cell size is too large compared to distance.
        // Max allowed size = min_size + growth_rate * distance
        // Distance is approx min(|d|)
        let dist = d00.abs().min(d10.abs()).min(d11.abs()).min(d01.abs());

        // Interpret growth_rate as a ratio (e.g. 1.2), so slope is rate - 1.0
        let slope = (growth_rate - 1.0).max(0.0);
        let max_allowed = min_size + slope * dist;
        if size > max_allowed {
            should_split = true;
        }
    }

    if should_split {
        node.subdivide();
        if let Some(children) = &mut node.children {
            for child in children.iter_mut() {
                refine_node(child, geo, min_size, growth_rate);
            }
        }
    }
}

pub fn collect_leaves<'a>(node: &'a QuadNode, leaves: &mut Vec<&'a QuadNode>) {
    if node.is_leaf {
        leaves.push(node);
    } else if let Some(children) = &node.children {
        for child in children {
            collect_leaves(child, leaves);
        }
    }
}

pub fn generate_base_polygons(
    geo: &(impl Geometry + Sync),
    min_cell_size: f64,
    max_cell_size: f64,
    growth_rate: f64,
    domain_size: Vector2<f64>,
) -> Vec<Vec<(Point2<f64>, bool)>> {
    let nx = (domain_size.x / max_cell_size).ceil() as usize;
    let ny = (domain_size.y / max_cell_size).ceil() as usize;

    // Collect all leaf polygons first
    (0..nx)
        .into_par_iter()
        .flat_map(|i| {
            (0..ny).into_par_iter().flat_map(move |j| {
                let x0 = i as f64 * max_cell_size;
                let y0 = j as f64 * max_cell_size;
                let x1 = (x0 + max_cell_size).min(domain_size.x);
                let y1 = (y0 + max_cell_size).min(domain_size.y);

                let mut root = QuadNode::new(Point2::new(x0, y0), Point2::new(x1, y1));
                refine_node(&mut root, geo, min_cell_size, growth_rate);

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
                                        let tol = 1e-5;
                                        // Only accept corner if it's not inside an obstacle (SDF <= small epsilon)
                                        // For convex obstacles (like circle), the tangent intersection is inside the obstacle (SDF > 0).
                                        // For true corners (like step), it's on the boundary (SDF ~ 0).
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
                        local_polys.push(reconstructed_poly);
                    }
                }
                local_polys
            })
        })
        .collect()
}
