#![cfg(all(feature = "meshgen", feature = "dev-tests"))]

use cfd2::solver::mesh::{generate_delaunay_mesh, BackwardsStep};
use nalgebra::Vector2;

#[test]
fn test_delaunay_005_correctness() {
    let geo = BackwardsStep {
        length: 3.5,
        height_inlet: 0.5,
        height_outlet: 1.0,
        step_x: 0.5,
    };
    let domain_size = Vector2::new(3.5, 1.0);
    let min_cell_size = 0.005;
    let max_cell_size = 0.005;
    let growth_rate = 1.2;

    let mesh = generate_delaunay_mesh(&geo, min_cell_size, max_cell_size, growth_rate, domain_size);

    assert!(mesh.num_cells() > 0);
    println!("Generated mesh with {} cells", mesh.num_cells());

    // Check that we have a reasonable number of cells for this resolution
    // Area approx 3.5 * 1.0 - 0.5 * 0.5 = 3.25
    // Cell area approx 0.5 * 0.005^2 (very rough)
    // Actually, Delaunay triangles.
    // Point spacing 0.005.
    // Number of points approx Area / (0.005^2) ? No, spacing is edge length.
    // Area of equilateral triangle with side a is sqrt(3)/4 * a^2.
    // N_triangles approx Area / Area_tri.
    // 3.25 / (0.433 * 0.005^2) = 3.25 / 0.0000108 = ~300,000 cells.

    assert!(mesh.num_cells() > 100_000);
}
