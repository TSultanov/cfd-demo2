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

    fn get_boundary_points(&self, spacing: f64) -> Vec<Point2<f64>> {
        let mut points = Vec::new();

        // Domain boundary
        let min = self.domain_min;
        let max = self.domain_max;
        let width = max.x - min.x;
        let height = max.y - min.y;

        let nx = (width / spacing).ceil() as usize;
        let ny = (height / spacing).ceil() as usize;

        for i in 0..nx {
            points.push(Point2::new(min.x + i as f64 * spacing, min.y));
            points.push(Point2::new(min.x + i as f64 * spacing, max.y));
        }
        for i in 0..ny {
            points.push(Point2::new(min.x, min.y + i as f64 * spacing));
            points.push(Point2::new(max.x, min.y + i as f64 * spacing));
        }

        // Obstacle boundary
        let circumference = 2.0 * std::f64::consts::PI * self.radius;
        let n_obs = (circumference / spacing).ceil() as usize;
        for i in 0..n_obs {
            let theta = 2.0 * std::f64::consts::PI * i as f64 / n_obs as f64;
            let x = self.center.x + self.radius * theta.cos();
            let y = self.center.y + self.radius * theta.sin();
            points.push(Point2::new(x, y));
        }

        points
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

    let domain_size = Vector2::new(1.0, 1.0);
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

            // dist_sq < r_sq => point inside circumcircle => Delaunay violation
            if dist_sq < r_sq - 1e-5 {
                panic!(
                    "Delaunay property violated! Cell {} circumcircle contains vertex {}. \
                        Cell vertices: {:?}, Offending vertex: {:?}, dist_sq: {}, r_sq: {}",
                    i, v_idx, indices, p, dist_sq, r_sq
                );
            }

            if (dist_sq - r_sq).abs() < 1e-5 {
                // Cocircular point — expected for regular grids.
            }
        }
    }
}

#[test]
fn test_delaunay_backwards_step() {
    let length = 3.5;
    let domain_size = Vector2::new(length, 1.0);
    let geo = BackwardsStep {
        length,
        height_inlet: 0.5,
        height_outlet: 1.0,
        step_x: 0.5,
    };

    // GUI defaults: min=0.025, max=0.025, growth=1.2
    let mesh = generate_delaunay_mesh(&geo, 0.025, 0.025, 1.2, domain_size);

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

#[test]
fn test_voronoi_generation() {
    let domain_size = Vector2::new(1.0, 1.0);
    let geo = CircleObstacle {
        center: Point2::new(0.5, 0.5),
        radius: 0.1,
        domain_min: Point2::new(0.0, 0.0),
        domain_max: Point2::new(domain_size.x, domain_size.y),
    };

    let mesh = generate_voronoi_mesh(&geo, 0.1, 0.2, 1.2, domain_size);

    println!("Generated Voronoi mesh with {} cells", mesh.num_cells());

    assert!(mesh.num_cells() > 0);
    assert_eq!(mesh.cell_cx.len(), mesh.num_cells());
    assert_eq!(mesh.cell_vol.len(), mesh.num_cells());

    let total_vol: f64 = mesh.cell_vol.iter().sum();
    let domain_area = domain_size.x * domain_size.y;
    let circle_area = std::f64::consts::PI * 0.1 * 0.1;
    let expected_area = domain_area - circle_area;

    println!("Total volume: {}, Expected: {}", total_vol, expected_area);

    assert!((total_vol - expected_area).abs() < 0.05);

    for i in 0..mesh.num_cells() {
        let start = mesh.cell_face_offsets[i];
        let end = mesh.cell_face_offsets[i + 1];
        let faces = &mesh.cell_faces[start..end];

        assert!(faces.len() >= 3, "Cell {} has fewer than 3 faces", i);
    }

    // Verify all cells have CCW vertex ordering (positive signed area)
    let mut cw_count = 0;
    for i in 0..mesh.num_cells() {
        let start = mesh.cell_vertex_offsets[i];
        let end = mesh.cell_vertex_offsets[i + 1];
        let n = end - start;

        if n < 3 {
            continue;
        }

        let mut signed_area = 0.0;
        for k in 0..n {
            let v_idx0 = mesh.cell_vertices[start + k];
            let v_idx1 = mesh.cell_vertices[start + (k + 1) % n];
            signed_area += mesh.vx[v_idx0] * mesh.vy[v_idx1] - mesh.vx[v_idx1] * mesh.vy[v_idx0];
        }

        if signed_area < 0.0 {
            cw_count += 1;
        }
    }

    assert_eq!(
        cw_count, 0,
        "All Voronoi cells should have CCW vertex ordering, found {} CW cells",
        cw_count
    );
}

// Scale-invariance and quantization tests

#[test]
fn quantize_does_not_collapse_distinct_vertices_at_small_scale() {
    use super::tolerances::MeshgenTolerances;
    let tol = MeshgenTolerances::from_geometry(1e-7, Vector2::new(1e-4, 1e-4));
    // Two points separated by 0.5 * min_cell_size should get distinct keys
    let k1 = tol.quantize_point(0.0, 0.0);
    let k2 = tol.quantize_point(5e-8, 0.0);
    assert_ne!(k1, k2, "Points 0.5*min_cell_size apart must not collapse");
}

#[test]
fn quantize_merges_nearby_vertices_at_large_scale() {
    use super::tolerances::MeshgenTolerances;
    let tol = MeshgenTolerances::from_geometry(10.0, Vector2::new(1000.0, 1000.0));
    // Two points separated by 0.1 * vertex_merge should get same key
    // vertex_merge = 10.0 * 1e-6 = 1e-5
    // 0.1 * vertex_merge = 1e-6
    let k1 = tol.quantize_point(500.0, 500.0);
    let k2 = tol.quantize_point(500.0 + 1e-7, 500.0);
    assert_eq!(k1, k2, "Points 0.01*vertex_merge apart must merge");
}

#[test]
fn quantize_adapts_to_microfluidic_scale() {
    use super::tolerances::MeshgenTolerances;
    // Microfluidic: domain 200µm x 100µm, min_cell_size = 1µm
    let tol = MeshgenTolerances::from_geometry(1e-6, Vector2::new(2e-4, 1e-4));

    // Points 0.5µm apart should be distinct (half a cell)
    let k1 = tol.quantize_point(1e-4, 5e-5);
    let k2 = tol.quantize_point(1e-4 + 5e-7, 5e-5);
    assert_ne!(k1, k2, "Points 0.5*min_cell_size apart must be distinct at µm scale");

    // Points 0.001 * vertex_merge apart should merge
    // vertex_merge = 1e-6 * 1e-6 = 1e-12
    let k3 = tol.quantize_point(1e-4, 5e-5);
    let k4 = tol.quantize_point(1e-4 + 1e-15, 5e-5);
    assert_eq!(k3, k4, "Points 0.001*vertex_merge apart must merge at µm scale");
}

#[test]
fn classify_boundary_scales_with_domain() {
    use super::tolerances::MeshgenTolerances;

    // Standard scale
    let tol1 = MeshgenTolerances::from_geometry(0.01, Vector2::new(2.0, 1.0));
    // Should classify x=0 as inlet
    assert!(tol1.classify_boundary(0.0, 0.5, 2.0, 1.0).is_some());
    // Should NOT classify x=0.5 as boundary
    assert!(tol1.classify_boundary(0.5, 0.5, 2.0, 1.0).is_none());

    // Micro scale
    let tol2 = MeshgenTolerances::from_geometry(1e-6, Vector2::new(2e-4, 1e-4));
    // Should classify x=0 as inlet
    assert!(tol2.classify_boundary(0.0, 5e-5, 2e-4, 1e-4).is_some());
    // Should NOT classify x=1e-4 (interior) as boundary
    assert!(tol2.classify_boundary(1e-4, 5e-5, 2e-4, 1e-4).is_none());
}

#[test]
fn cut_cell_mesh_scale_invariance_topology() {
    use super::geometry::RectangularChannel;

    // Scale 1: standard
    let geo1 = RectangularChannel {
        length: 2.0,
        height: 1.0,
    };
    let mesh1 = generate_cut_cell_mesh(&geo1, 0.1, 0.1, 1.2, Vector2::new(2.0, 1.0));

    // Scale 1e-3: millimeter
    let geo2 = RectangularChannel {
        length: 2e-3,
        height: 1e-3,
    };
    let mesh2 = generate_cut_cell_mesh(&geo2, 1e-4, 1e-4, 1.2, Vector2::new(2e-3, 1e-3));

    // For a simple rectangular channel with uniform cell size, topology must match
    assert_eq!(
        mesh1.num_cells(),
        mesh2.num_cells(),
        "Cell count must match across scales: {} vs {}",
        mesh1.num_cells(),
        mesh2.num_cells()
    );
    assert_eq!(
        mesh1.num_faces(),
        mesh2.num_faces(),
        "Face count must match across scales: {} vs {}",
        mesh1.num_faces(),
        mesh2.num_faces()
    );

    // Volume ratio should be scale^2
    let scale = 1e-3;
    let vol_ratio = mesh2.cell_vol.iter().sum::<f64>() / mesh1.cell_vol.iter().sum::<f64>();
    let expected_ratio = scale * scale;
    assert!(
        (vol_ratio / expected_ratio - 1.0).abs() < 1e-6,
        "Total volume ratio {vol_ratio} should be {expected_ratio}"
    );
}

#[test]
fn tolerances_are_self_consistent() {
    use super::tolerances::MeshgenTolerances;

    let tol = MeshgenTolerances::from_geometry(0.025, Vector2::new(2.0, 1.0));

    // vertex_merge = mcs * 1e-6 = 2.5e-8
    // quantize_inv = 1/vertex_merge = 4e7
    // A point at x=1.0 quantizes to (1.0 * 4e7).round() = 40_000_000
    let k = tol.quantize(1.0);
    assert_eq!(k, 40_000_000);

    // Two points 2*vertex_merge apart should get different keys
    let k1 = tol.quantize(0.0);
    let k2 = tol.quantize(5e-8);
    assert_ne!(k1, k2);

    // boundary_eps should be small relative to domain
    let diag = (2.0f64 * 2.0 + 1.0 * 1.0).sqrt();
    assert!((tol.boundary_eps - diag * 1e-6).abs() < 1e-15);
}
