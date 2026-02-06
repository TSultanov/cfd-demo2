fn main() {
    use cfd2::solver::mesh::{generate_voronoi_mesh, BackwardsStep, Geometry};
    use nalgebra::{Point2, Vector2};

    let geo = BackwardsStep {
        length: 2.0,
        height_inlet: 0.5,
        height_outlet: 1.0,
        step_x: 0.5,
    };
    let domain_size = Vector2::new(2.0, 1.0);

    {
        let mut mesh = generate_voronoi_mesh(&geo, 0.05, 0.05, 1.2, domain_size);
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

    {
        let mesh = generate_voronoi_mesh(&geo, 0.05, 0.05, 1.2, domain_size);

        let mut bad_connections = 0;
        for i in 0..mesh.num_cells() {
            let center = Point2::new(mesh.cell_cx[i], mesh.cell_cy[i]);
            if geo.sdf(&center).abs() > 1e-4 {
                continue;
            }

            let start = mesh.cell_vertex_offsets[i];
            let end = mesh.cell_vertex_offsets[i + 1];
            let indices = &mesh.cell_vertices[start..end];

            let mut v_idx_in_poly = None;
            for (k, &v_idx) in indices.iter().enumerate() {
                let p = Point2::new(mesh.vx[v_idx], mesh.vy[v_idx]);
                if (p - center).norm() < 1e-6 {
                    v_idx_in_poly = Some(k);
                    break;
                }
            }

            if let Some(k) = v_idx_in_poly {
                let prev = indices[(k + indices.len() - 1) % indices.len()];
                let next = indices[(k + 1) % indices.len()];

                for &neighbor_idx in &[prev, next] {
                    let p_neighbor = Point2::new(mesh.vx[neighbor_idx], mesh.vy[neighbor_idx]);
                    if geo.sdf(&p_neighbor).abs() > 1e-4 {
                        bad_connections += 1;
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
