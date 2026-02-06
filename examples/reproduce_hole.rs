fn main() {
    use cfd2::solver::mesh::{generate_delaunay_mesh, ChannelWithObstacle, Geometry};
    use nalgebra::{Point2, Vector2};

    let length = 3.0;
    let domain_size = Vector2::new(length, 1.0);
    let geo = ChannelWithObstacle {
        length,
        height: 1.0,
        obstacle_center: Point2::new(1.0, 0.51),
        obstacle_radius: 0.1,
    };

    let min_cell_size = 0.005;
    let max_cell_size = 0.025;
    let growth_rate = 1.2;

    println!(
        "Generating Delaunay mesh with min_size={}, max_size={}, growth_rate={}",
        min_cell_size, max_cell_size, growth_rate
    );

    let mesh = generate_delaunay_mesh(&geo, min_cell_size, max_cell_size, growth_rate, domain_size);

    println!("Generated mesh with {} cells", mesh.num_cells());
    assert!(mesh.num_cells() > 0, "Mesh should not be empty");

    let expected_area =
        length * 1.0 - std::f64::consts::PI * geo.obstacle_radius * geo.obstacle_radius;
    let total_area: f64 = mesh.cell_vol.iter().copied().sum();

    println!(
        "Total mesh area: {}, Expected area: {}",
        total_area, expected_area
    );

    assert!(
        (total_area - expected_area).abs() < expected_area * 0.01,
        "Mesh area deviates significantly from expected area, possible holes!"
    );

    let mut hole_found = false;
    for i in 0..mesh.face_owner.len() {
        if mesh.face_neighbor[i].is_none() {
            let p = Point2::new(mesh.face_cx[i], mesh.face_cy[i]);
            let sdf = geo.sdf(&p);
            if sdf < -1e-3 {
                hole_found = true;
            }
        }
    }

    assert!(!hole_found, "Holes detected in the mesh!");
}
