#[cfg(test)]
mod tests {
    use cfd2::solver::mesh::{generate_voronoi_mesh, BackwardsStep, Geometry};
    use nalgebra::{Point2, Vector2};

    #[test]
    fn test_voronoi_boundary_fidelity() {
        let geo = BackwardsStep {
            length: 2.0,
            height_inlet: 0.5,
            height_outlet: 1.0,
            step_x: 0.5,
        };
        let domain_size = Vector2::new(2.0, 1.0);

        // Generate Voronoi mesh
        let mut mesh = generate_voronoi_mesh(&geo, 0.05, 0.05, 1.2, domain_size);

        // Apply smoothing, which is what causes the boundary deviation if vertices aren't fixed
        mesh.smooth(&geo, 0.3, 10);

        println!("Generated Voronoi mesh with {} faces", mesh.face_cx.len());

        let mut max_deviation: f64 = 0.0;
        let mut bad_vertices = 0;

        for i in 0..mesh.face_boundary.len() {
            if mesh.face_boundary[i].is_some() {
                let v1_idx = mesh.face_v1[i];
                let v2_idx = mesh.face_v2[i];

                for &v_idx in &[v1_idx, v2_idx] {
                    let p = Point2::new(mesh.vx[v_idx], mesh.vy[v_idx]);
                    let dist = geo.sdf(&p).abs();

                    if dist > 1e-3 {
                        max_deviation = max_deviation.max(dist);
                        bad_vertices += 1;
                        // println!("Boundary vertex {} at {:?} is off by {}", v_idx, p, dist);
                    }
                }
            }
        }

        println!(
            "Found {} boundary vertices with deviation > 1e-3. Max deviation: {}",
            bad_vertices, max_deviation
        );

        assert!(
            bad_vertices == 0,
            "Voronoi boundary vertices deviate from geometry! Max deviation: {}",
            max_deviation
        );
    }

    #[test]
    fn test_voronoi_cell_connectivity() {
        let geo = BackwardsStep {
            length: 2.0,
            height_inlet: 0.5,
            height_outlet: 1.0,
            step_x: 0.5,
        };
        let domain_size = Vector2::new(2.0, 1.0);

        let mesh = generate_voronoi_mesh(&geo, 0.05, 0.05, 1.2, domain_size);

        let mut bad_connections = 0;

        for i in 0..mesh.num_cells() {
            // Check if this is a boundary cell (has a fixed vertex as center)
            // Note: In our implementation, mesh.vx[i] corresponds to the cell center for cell i?
            // No, mesh.vx are the *corners* of the cells.
            // The cell centers are mesh.cell_cx, cell_cy.
            // But we know cell i corresponds to Delaunay vertex i.
            // And we know if Delaunay vertex i is on boundary.
            // How? We don't have the "is_boundary" flag for cells easily accessible here.
            // But we can check if the cell center is on the boundary.

            let center = Point2::new(mesh.cell_cx[i], mesh.cell_cy[i]);
            if geo.sdf(&center).abs() > 1e-4 {
                continue; // Internal cell
            }

            // This is a boundary cell.
            // Get its vertices.
            let start = mesh.cell_vertex_offsets[i];
            let end = mesh.cell_vertex_offsets[i + 1];
            let indices = &mesh.cell_vertices[start..end];

            // Find the vertex index that corresponds to the cell center (V)
            // It should be one of the vertices of the polygon.
            // Since V is the cell center, it is at (cell_cx, cell_cy).
            // We can find it by coordinate match.

            let mut v_idx_in_poly = None;
            for (k, &v_idx) in indices.iter().enumerate() {
                let p = Point2::new(mesh.vx[v_idx], mesh.vy[v_idx]);
                if (p - center).norm() < 1e-6 {
                    v_idx_in_poly = Some(k);
                    break;
                }
            }

            if let Some(k) = v_idx_in_poly {
                // Check neighbors of V in the polygon
                let prev = indices[(k + indices.len() - 1) % indices.len()];
                let next = indices[(k + 1) % indices.len()];

                for &neighbor_idx in &[prev, next] {
                    let p_neighbor = Point2::new(mesh.vx[neighbor_idx], mesh.vy[neighbor_idx]);
                    // Neighbor must be on boundary (Midpoint)
                    if geo.sdf(&p_neighbor).abs() > 1e-4 {
                        bad_connections += 1;
                        // println!("Cell {} (Boundary) has connection to internal vertex {} at {:?}", i, neighbor_idx, p_neighbor);
                    }
                }
            }
        }

        println!(
            "Found {} bad connections in boundary cells",
            bad_connections
        );
        assert!(
            bad_connections == 0,
            "Boundary cells have invalid connectivity (V connected to internal C)"
        );
    }
}
