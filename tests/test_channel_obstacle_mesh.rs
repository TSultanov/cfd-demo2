#![cfg(all(feature = "meshgen", feature = "dev-tests"))]

#[cfg(test)]
mod tests {
    use cfd2::solver::mesh::{ChannelWithObstacle, Geometry};
    use nalgebra::{Point2, Vector2};

    #[test]
    fn test_circle_geometry_fidelity() {
        let geo = ChannelWithObstacle {
            length: 3.0,
            height: 1.0,
            obstacle_center: Point2::new(1.0, 0.51), // Match GUI
            obstacle_radius: 0.1,
        };

        let domain_size = Vector2::new(3.0, 1.0);

        let mut mesh =
            cfd2::solver::mesh::generate_cut_cell_mesh(&geo, 0.05, 0.05, 1.2, domain_size);

        println!("Mesh generated with {} vertices", mesh.vx.len());

        // Check before smoothing
        let mut max_error_before = 0.0;
        for i in 0..mesh.vx.len() {
            if mesh.v_fixed[i] {
                let p = Point2::new(mesh.vx[i], mesh.vy[i]);
                let sdf = geo.sdf(&p);
                if sdf.abs() > max_error_before {
                    max_error_before = sdf.abs();
                }
            }
        }
        println!("Max boundary error before smoothing: {}", max_error_before);

        // Smooth
        mesh.smooth(&geo, 0.3, 50);

        // Identify boundary vertices from mesh topology
        let mut boundary_vertices = std::collections::HashSet::new();
        for f_idx in 0..mesh.face_cx.len() {
            if mesh.face_neighbor[f_idx].is_none() {
                boundary_vertices.insert(mesh.face_v1[f_idx]);
                boundary_vertices.insert(mesh.face_v2[f_idx]);
            }
        }

        let mut max_error = 0.0;
        let mut bad_vertices = 0;

        for &v_idx in &boundary_vertices {
            let p = Point2::new(mesh.vx[v_idx], mesh.vy[v_idx]);
            let sdf = geo.sdf(&p);

            // If sdf is negative, it's in the fluid.
            // If it's significantly negative, it's an ear (dent in mesh).
            // If sdf is positive, it's inside the obstacle (bump in mesh).

            if sdf.abs() > max_error {
                max_error = sdf.abs();
            }

            if sdf.abs() > 1e-3 {
                // println!(
                //     "Bad boundary vertex at {:?}, sdf: {}, fixed: {}",
                //     p, sdf, mesh.v_fixed[v_idx]
                // );
                bad_vertices += 1;
            }
        }

        println!("Max boundary error after smoothing: {}", max_error);

        assert!(
            max_error < 1e-3,
            "Boundary vertices deviate too much from true geometry"
        );
        assert_eq!(bad_vertices, 0, "Found bad boundary vertices");
    }
}
