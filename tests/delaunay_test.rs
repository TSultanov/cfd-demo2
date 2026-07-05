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

    // ~300k triangles expected: area 3.25 / equilateral-tri area (sqrt(3)/4 *
    // 0.005^2) = 3.25 / 1.08e-5. Assert a loose lower bound.
    assert!(mesh.num_cells() > 100_000);
}
