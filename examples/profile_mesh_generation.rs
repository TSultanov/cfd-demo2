fn main() {
    use cfd2::solver::mesh::{generate_cut_cell_mesh, ChannelWithObstacle};
    use nalgebra::{Point2, Vector2};

    let geo = ChannelWithObstacle {
        length: 2.0,
        height: 1.0,
        obstacle_center: Point2::new(0.5, 0.5),
        obstacle_radius: 0.1,
    };
    let domain_size = Vector2::new(2.0, 1.0);
    let min_cell_size = 0.00175;
    let max_cell_size = 0.00175;

    let mesh = generate_cut_cell_mesh(&geo, min_cell_size, max_cell_size, 1.2, domain_size);

    println!("Generated mesh with {} cells", mesh.num_cells());
    assert!(mesh.num_cells() > 0);

    for (i, &vol) in mesh.cell_vol.iter().enumerate() {
        assert!(vol > 0.0, "Cell {} has non-positive volume: {}", i, vol);
    }

    let max_skew = mesh.calculate_max_skewness();
    println!("Max skewness: {}", max_skew);
    assert!(max_skew < 1.0, "Skewness too high: {}", max_skew);
}
