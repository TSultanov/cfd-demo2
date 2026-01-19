#![cfg(feature = "meshgen")]

#[cfg(test)]
mod tests {
    use cfd2::solver::mesh::{generate_delaunay_mesh, ChannelWithObstacle, Geometry};
    use nalgebra::Vector2;

    #[test]
    fn test_reproduce_hole() {
        let length = 3.0;
        let domain_size = Vector2::new(length, 1.0);
        let geo = ChannelWithObstacle {
            length,
            height: 1.0,
            obstacle_center: nalgebra::Point2::new(1.0, 0.51),
            obstacle_radius: 0.1,
        };

        let min_cell_size = 0.005;
        let max_cell_size = 0.025;
        let growth_rate = 1.2;

        println!(
            "Generating Delaunay mesh with min_size={}, max_size={}, growth_rate={}",
            min_cell_size, max_cell_size, growth_rate
        );

        // The user reported holes with these parameters
        let mesh =
            generate_delaunay_mesh(&geo, min_cell_size, max_cell_size, growth_rate, domain_size);

        println!("Generated mesh with {} cells", mesh.num_cells());

        // Basic sanity checks
        assert!(mesh.num_cells() > 0, "Mesh should not be empty");

        // We could try to detect holes by checking total area vs expected area
        let expected_area = length * 1.0 - std::f64::consts::PI * 0.1 * 0.1;
        let mut total_area = 0.0;
        for i in 0..mesh.num_cells() {
            total_area += mesh.cell_vol[i];
        }

        println!(
            "Total mesh area: {}, Expected area: {}",
            total_area, expected_area
        );

        // If there's a significant hole, the area will be smaller
        // Let's say if it's off by more than 1%
        assert!(
            (total_area - expected_area).abs() < expected_area * 0.01,
            "Mesh area deviates significantly from expected area, possible holes!"
        );

        // Check for holes by inspecting boundary faces
        let mut hole_found = false;
        for i in 0..mesh.face_owner.len() {
            if mesh.face_neighbor[i].is_none() {
                // This is a boundary face
                let cx = mesh.face_cx[i];
                let cy = mesh.face_cy[i];
                let p = nalgebra::Point2::new(cx, cy);
                let sdf = geo.sdf(&p);

                // If it's a boundary face, it should be on the geometry boundary (SDF ~ 0)
                // If SDF is significantly negative, it means the face is inside the fluid,
                // which implies a hole.
                if sdf < -1e-3 {
                    hole_found = true;
                }
            }
        }

        assert!(!hole_found, "Holes detected in the mesh!");
    }
}
