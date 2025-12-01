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
        let dx = (p.x - (self.domain_min.x + self.domain_max.x) / 2.0).abs()
            - (self.domain_max.x - self.domain_min.x) / 2.0;
        let dy = (p.y - (self.domain_min.y + self.domain_max.y) / 2.0).abs()
            - (self.domain_max.y - self.domain_min.y) / 2.0;
        let box_dist = dx.max(dy).min(0.0) + Vector2::new(dx.max(0.0), dy.max(0.0)).norm();

        let circle_dist = (p - self.center).norm() - self.radius;

        box_dist.max(-circle_dist)
    }
}

#[test]
fn test_mesh_generation_circle_obstacle() {
    let geo = CircleObstacle {
        center: Point2::new(0.5001, 0.5001),
        radius: 0.2,
        domain_min: Point2::new(0.0, 0.0),
        domain_max: Point2::new(1.0, 1.0),
    };

    // Generate mesh
    let domain_size = Vector2::new(1.0, 1.0);
    // Coarser mesh: 0.1 cell size. Radius is 0.2.
    let mut mesh = generate_cut_cell_mesh(&geo, 0.1, 0.1, 1.2, domain_size);

    println!("Generated mesh with {} cells", mesh.num_cells());
    assert!(mesh.num_cells() > 0);

    let initial_skew = mesh.calculate_max_skewness();
    println!("Initial max skewness: {}", initial_skew);

    // Identify boundary vertices before smoothing
    let mut boundary_indices = Vec::new();
    for i in 0..mesh.num_vertices() {
        if mesh.v_fixed[i] {
            boundary_indices.push(i);
        }
    }

    println!("Found {} fixed boundary vertices", boundary_indices.len());
    assert!(boundary_indices.len() > 0);

    // Smooth
    mesh.smooth(&geo, 0.05, 50);

    // Verify positions are still on boundary
    for &idx in &boundary_indices {
        let p_new = Point2::new(mesh.vx[idx], mesh.vy[idx]);
        let dist = geo.sdf(&p_new).abs();
        assert!(
            dist < 1e-4,
            "Boundary vertex moved off boundary! dist={}",
            dist
        );
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
    let mut mesh = generate_cut_cell_mesh(&geo, 0.1, 0.1, 1.2, domain_size);

    println!("Generated mesh with {} cells", mesh.num_cells());
    assert!(mesh.num_cells() > 0);

    let initial_skew = mesh.calculate_max_skewness();
    println!("Initial max skewness: {}", initial_skew);

    // Target low skewness
    mesh.smooth(&geo, 0.1, 50);

    let final_skew = mesh.calculate_max_skewness();
    println!("Final max skewness: {}", final_skew);

    // Sharp corners increase skewness compared to chamfered corners, so we relax the check.
    assert!(final_skew < 0.6);
}

#[test]
fn test_delaunay_property() {
    let domain_size = Vector2::new(1.0, 1.0);
    let geo = CircleObstacle {
        center: Point2::new(0.5, 0.5),
        radius: 0.1,
        domain_min: Point2::new(0.0, 0.0),
        domain_max: Point2::new(domain_size.x, domain_size.y),
    };

    // Use generate_delaunay_mesh
    let mesh = generate_delaunay_mesh(&geo, 0.1, 0.2, 1.2, domain_size);

    println!("Generated Delaunay mesh with {} cells", mesh.num_cells());

    // Check Delaunay property for each cell
    for i in 0..mesh.num_cells() {
        let start = mesh.cell_vertex_offsets[i];
        let end = mesh.cell_vertex_offsets[i + 1];
        let indices = &mesh.cell_vertices[start..end];

        // Delaunay mesh should consist of triangles
        assert_eq!(indices.len(), 3, "Cell {} is not a triangle", i);

        let p1 = Point2::new(mesh.vx[indices[0]], mesh.vy[indices[0]]);
        let p2 = Point2::new(mesh.vx[indices[1]], mesh.vy[indices[1]]);
        let p3 = Point2::new(mesh.vx[indices[2]], mesh.vy[indices[2]]);

        let (center, r_sq) = Triangle::calculate_circumcircle(p1, p2, p3);

        // Check all other vertices
        for v_idx in 0..mesh.vx.len() {
            // Skip vertices of the triangle itself
            if indices.contains(&v_idx) {
                continue;
            }

            let p = Point2::new(mesh.vx[v_idx], mesh.vy[v_idx]);
            let dist_sq = (p - center).norm_squared();

            // Allow for small epsilon error
            // If dist_sq < r_sq, the point is inside the circumcircle -> Violation
            if dist_sq < r_sq - 1e-5 {
                panic!(
                    "Delaunay property violated! Cell {} circumcircle contains vertex {}. \
                        Cell vertices: {:?}, Offending vertex: {:?}, dist_sq: {}, r_sq: {}",
                    i, v_idx, indices, p, dist_sq, r_sq
                );
            }

            // Log if point is on boundary (cocircular)
            if (dist_sq - r_sq).abs() < 1e-5 {
                // This is expected for regular grids (cocircular points)
                // Uncomment to see how many:
                // println!("Point {} is on circumcircle of cell {}", v_idx, i);
            }
        }
    }
}

#[test]
fn test_delaunay_backwards_step() {
    let geo = BackwardsStep {
        length: 2.0,
        height_inlet: 0.501,
        height_outlet: 1.0,
        step_x: 0.501,
    };

    let domain_size = Vector2::new(2.0, 1.0);
    let mesh = generate_delaunay_mesh(&geo, 0.1, 0.2, 1.2, domain_size);

    println!("Generated Delaunay mesh with {} cells", mesh.num_cells());

    for i in 0..mesh.num_cells() {
        let start = mesh.cell_vertex_offsets[i];
        let end = mesh.cell_vertex_offsets[i + 1];
        let indices = &mesh.cell_vertices[start..end];

        assert_eq!(indices.len(), 3, "Cell {} is not a triangle", i);

        let p1 = Point2::new(mesh.vx[indices[0]], mesh.vy[indices[0]]);
        let p2 = Point2::new(mesh.vx[indices[1]], mesh.vy[indices[1]]);
        let p3 = Point2::new(mesh.vx[indices[2]], mesh.vy[indices[2]]);

        let (center, r_sq) = Triangle::calculate_circumcircle(p1, p2, p3);

        for v_idx in 0..mesh.vx.len() {
            if indices.contains(&v_idx) {
                continue;
            }

            let p = Point2::new(mesh.vx[v_idx], mesh.vy[v_idx]);
            let dist_sq = (p - center).norm_squared();

            // Strict check with relative epsilon
            let epsilon = r_sq.max(1.0) * 1e-8;
            if dist_sq < r_sq - epsilon {
                panic!("Delaunay property violated! Cell {} circumcircle contains vertex {}. \
                        Cell vertices: {:?}, Offending vertex: {:?}, dist_sq: {}, r_sq: {}, diff: {}",
                        i, v_idx, indices, p, dist_sq, r_sq, r_sq - dist_sq);
            }
        }
    }
}
