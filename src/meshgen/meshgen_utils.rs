use super::geometry::Geometry;
use nalgebra::{Point2, Vector2};

pub(super) fn compute_normal(geo: &(impl Geometry + ?Sized), p: Point2<f64>) -> Vector2<f64> {
    let eps = 1e-6;
    let d_x = geo.sdf(&Point2::new(p.x + eps, p.y)) - geo.sdf(&Point2::new(p.x - eps, p.y));
    let d_y = geo.sdf(&Point2::new(p.x, p.y + eps)) - geo.sdf(&Point2::new(p.x, p.y - eps));
    Vector2::new(d_x, d_y).normalize()
}

pub(super) fn intersect_lines(
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
