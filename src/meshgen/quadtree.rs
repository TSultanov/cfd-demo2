use super::geometry::Geometry;
use nalgebra::Point2;

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
