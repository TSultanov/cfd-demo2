use cfd2::solver::mesh::{generate_cut_cell_mesh, generate_delaunay_mesh, ChannelWithObstacle};
use nalgebra::{Point2, Vector2};

#[test]
fn test_mesh_comparison() {
    let domain_size = Vector2::new(2.0, 1.0);
    let obstacle = ChannelWithObstacle {
        length: 2.0,
        height: 1.0,
        obstacle_center: Point2::new(1.0, 0.5),
        obstacle_radius: 0.2,
    };

    let min_cell_size = 0.05;
    let max_cell_size = 0.1;
    let growth_rate = 1.2; // Added growth_rate parameter

    println!("Generating CutCell Mesh...");
    let cut_cell_mesh = generate_cut_cell_mesh(
        &obstacle,
        min_cell_size,
        max_cell_size,
        growth_rate,
        domain_size,
    );
    println!(
        "CutCell Mesh: {} cells, {} faces, {} vertices",
        cut_cell_mesh.num_cells(),
        cut_cell_mesh.num_faces(),
        cut_cell_mesh.num_vertices()
    );

    println!("Generating Delaunay Mesh...");
    let delaunay_mesh = generate_delaunay_mesh(
        &obstacle,
        min_cell_size,
        max_cell_size,
        growth_rate,
        domain_size,
    );
    println!(
        "Delaunay Mesh: {} cells, {} faces, {} vertices",
        delaunay_mesh.num_cells(),
        delaunay_mesh.num_faces(),
        delaunay_mesh.num_vertices()
    );

    // Basic checks
    assert!(cut_cell_mesh.num_cells() > 0);
    assert!(delaunay_mesh.num_cells() > 0);

    // Volume check
    let total_vol_cut = cut_cell_mesh.cell_vol.iter().sum::<f64>();
    let total_vol_del = delaunay_mesh.cell_vol.iter().sum::<f64>();

    let expected_vol = 2.0 * 1.0 - std::f64::consts::PI * 0.2 * 0.2;
    println!("Expected Volume: {:.6}", expected_vol);
    println!("CutCell Volume: {:.6}", total_vol_cut);
    println!("Delaunay Volume: {:.6}", total_vol_del);

    assert!(
        (total_vol_cut - expected_vol).abs() < 1e-2,
        "CutCell volume mismatch"
    );
    assert!(
        (total_vol_del - expected_vol).abs() < 1e-2,
        "Delaunay volume mismatch"
    );

    // Skewness check
    let skew_cut = cut_cell_mesh.calculate_max_skewness();
    let skew_del = delaunay_mesh.calculate_max_skewness();

    println!("CutCell Max Skewness: {:.6}", skew_cut);
    println!("Delaunay Max Skewness: {:.6}", skew_del);

    // Delaunay meshes can have slivers, but hopefully not too bad with good point distribution
    // We don't strictly assert skewness yet as we haven't implemented optimization/smoothing for Delaunay
    // But it shouldn't be 1.0 (degenerate)
    assert!(skew_del < 0.99, "Delaunay mesh has degenerate cells");
}
