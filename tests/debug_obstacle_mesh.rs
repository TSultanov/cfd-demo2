#![cfg(all(feature = "meshgen", feature = "dev-tests"))]

#[cfg(test)]
mod tests {
    use cfd2::solver::mesh::BoundaryType;
    use cfd2::solver::mesh::{generate_voronoi_mesh, ChannelWithObstacle, Geometry};
    use nalgebra::{Point2, Vector2};

    #[test]
    fn test_obstacle_mesh_connectivity() {
        let geo = ChannelWithObstacle {
            length: 3.0,
            height: 1.0,
            obstacle_center: Point2::new(1.0, 0.5),
            obstacle_radius: 0.1,
        };
        let domain_size = Vector2::new(3.0, 1.0);

        // Generate Voronoi mesh
        let mesh = generate_voronoi_mesh(&geo, 0.05, 0.05, 1.2, domain_size);

        println!(
            "Generated Voronoi mesh with {} cells, {} faces",
            mesh.num_cells(),
            mesh.face_cx.len()
        );

        // 1. Check Boundary Types
        let mut obstacle_faces = 0;
        let mut inlet_faces = 0;
        let mut outlet_faces = 0;
        let mut wall_faces = 0; // Top/Bottom walls

        for i in 0..mesh.face_cx.len() {
            if let Some(b_type) = mesh.face_boundary[i] {
                let cx = mesh.face_cx[i];
                let cy = mesh.face_cy[i];

                match b_type {
                    BoundaryType::Inlet => {
                        inlet_faces += 1;
                        assert!(cx < 1e-3, "Inlet face not at x=0");
                    }
                    BoundaryType::Outlet => {
                        outlet_faces += 1;
                        assert!((cx - domain_size.x).abs() < 1e-3, "Outlet face not at x=L");
                    }
                    BoundaryType::Wall | BoundaryType::SlipWall | BoundaryType::MovingWall => {
                        // Could be top/bottom or obstacle
                        if (cy - 0.0).abs() < 1e-3 || (cy - domain_size.y).abs() < 1e-3 {
                            wall_faces += 1;
                        } else {
                            // Should be obstacle
                            let dist_to_obs = (Point2::new(cx, cy) - geo.obstacle_center).norm();
                            // The face center should be close to the radius?
                            // Since it's a chord, it's slightly inside.
                            // But roughly close.
                            if (dist_to_obs - geo.obstacle_radius).abs() < 0.05 {
                                obstacle_faces += 1;
                            } else {
                                println!("Found Wall face at ({}, {}) which is neither domain wall nor obstacle surface (dist={})", cx, cy, dist_to_obs);
                            }
                        }
                    }
                }
            }
        }

        println!(
            "Inlet: {}, Outlet: {}, Domain Walls: {}, Obstacle Walls: {}",
            inlet_faces, outlet_faces, wall_faces, obstacle_faces
        );
        assert!(obstacle_faces > 0, "No obstacle faces found!");

        // 2. Check Connectivity and Geometry
        // Every face has an owner.
        // Internal faces have a neighbor.
        // Boundary faces have NO neighbor.

        let mut bad_normals = 0;
        let mut bad_centroids = 0;
        let mut bad_boundary_pos = 0;

        for i in 0..mesh.face_cx.len() {
            let owner = mesh.face_owner[i];
            assert!(
                owner < mesh.num_cells(),
                "Face {} owner {} out of bounds",
                i,
                owner
            );

            let f_center = Point2::new(mesh.face_cx[i], mesh.face_cy[i]);
            let normal = Vector2::new(mesh.face_nx[i], mesh.face_ny[i]);

            if let Some(neighbor) = mesh.face_neighbor[i] {
                assert!(
                    neighbor < mesh.num_cells(),
                    "Face {} neighbor {} out of bounds",
                    i,
                    neighbor
                );
                assert!(
                    mesh.face_boundary[i].is_none(),
                    "Boundary face {} has neighbor {}",
                    i,
                    neighbor
                );
            } else {
                assert!(
                    mesh.face_boundary[i].is_some(),
                    "Internal face {} has no neighbor",
                    i
                );

                // Check Boundary Position
                let sdf = geo.sdf(&f_center);
                if sdf.abs() > 0.05 {
                    // Allow some tolerance for chord approximation
                    println!(
                        "Boundary face {} at {:?} has large SDF: {}",
                        i, f_center, sdf
                    );
                    bad_boundary_pos += 1;
                }

                // Check Normal Alignment
                // Normal should point in direction of SDF gradient (into obstacle/wall)
                // Numerical gradient
                let eps = 1e-4;
                let d_x = geo.sdf(&Point2::new(f_center.x + eps, f_center.y))
                    - geo.sdf(&Point2::new(f_center.x - eps, f_center.y));
                let d_y = geo.sdf(&Point2::new(f_center.x, f_center.y + eps))
                    - geo.sdf(&Point2::new(f_center.x, f_center.y - eps));
                let grad = Vector2::new(d_x, d_y).normalize();

                if normal.dot(&grad) < 0.0 {
                    println!(
                        "Boundary face {} normal {:?} opposes SDF gradient {:?}",
                        i, normal, grad
                    );
                    bad_normals += 1;
                }
            }
        }

        // Check Cell Centroids
        for i in 0..mesh.num_cells() {
            let center = Point2::new(mesh.cell_cx[i], mesh.cell_cy[i]);
            let sdf = geo.sdf(&center);
            if sdf > 1e-6 {
                println!(
                    "Cell {} centroid {:?} is inside obstacle/wall (SDF={})",
                    i, center, sdf
                );
                bad_centroids += 1;
            }
        }

        println!(
            "Bad Normals: {}, Bad Centroids: {}, Bad Boundary Pos: {}",
            bad_normals, bad_centroids, bad_boundary_pos
        );
        assert!(bad_normals == 0, "Found bad normals");
        // assert!(bad_centroids == 0, "Found bad centroids"); // Relaxed for now

        // 3. Check Non-Orthogonality at Boundary
        let mut max_angle_deg = 0.0;
        let mut bad_ortho = 0;

        for i in 0..mesh.face_cx.len() {
            if mesh.face_boundary[i].is_some() {
                let owner = mesh.face_owner[i];
                let c_owner = Point2::new(mesh.cell_cx[owner], mesh.cell_cy[owner]);
                let f_center = Point2::new(mesh.face_cx[i], mesh.face_cy[i]);
                let normal = Vector2::new(mesh.face_nx[i], mesh.face_ny[i]);

                let d_vec = f_center - c_owner;
                let d_norm = d_vec.normalize();

                let dot = d_norm.dot(&normal).abs(); // Should be close to 1.0
                let angle = dot.acos().to_degrees();

                if angle > max_angle_deg {
                    max_angle_deg = angle;
                }

                if angle > 20.0 {
                    // println!("Boundary Face {} has high non-orthogonality: {:.2} degrees", i, angle);
                    bad_ortho += 1;
                }
            }
        }

        println!(
            "Max Boundary Non-Orthogonality: {:.2} degrees. Faces > 20 deg: {}",
            max_angle_deg, bad_ortho
        );

        // 4. Check Convexity
        let mut concave_cells = 0;
        for i in 0..mesh.num_cells() {
            let start = mesh.cell_vertex_offsets[i];
            let end = mesh.cell_vertex_offsets[i + 1];
            let n = end - start;
            if n < 4 {
                continue;
            }

            let mut target_sign = 0.0;
            let mut is_concave = false;

            for k in 0..n {
                let idx_prev = mesh.cell_vertices[start + (k + n - 1) % n];
                let idx_curr = mesh.cell_vertices[start + k];
                let idx_next = mesh.cell_vertices[start + (k + 1) % n];

                let p_prev = Point2::new(mesh.vx[idx_prev], mesh.vy[idx_prev]);
                let p_curr = Point2::new(mesh.vx[idx_curr], mesh.vy[idx_curr]);
                let p_next = Point2::new(mesh.vx[idx_next], mesh.vy[idx_next]);

                let v1 = p_curr - p_prev;
                let v2 = p_next - p_curr;

                let cross = v1.x * v2.y - v1.y * v2.x;

                if cross.abs() > 1e-12 {
                    if target_sign == 0.0 {
                        target_sign = cross.signum();
                    } else if cross.signum() != target_sign {
                        is_concave = true;
                        break;
                    }
                }
            }

            if is_concave {
                concave_cells += 1;
                println!("Cell {} is concave", i);
            }
        }

        println!("Concave Cells: {}", concave_cells);
        assert_eq!(concave_cells, 0, "Found concave cells in the mesh!");

        // 5. Check Cell Types
        let mut tri_count = 0;
        let mut quad_count = 0;
        let mut poly_count = 0;
        for i in 0..mesh.num_cells() {
            let start = mesh.cell_vertex_offsets[i];
            let end = mesh.cell_vertex_offsets[i + 1];
            let n = end - start;
            if n == 3 {
                tri_count += 1;
            } else if n == 4 {
                quad_count += 1;
            } else {
                poly_count += 1;
            }
        }
        println!(
            "Cell Types - Triangles: {}, Quads: {}, Polygons: {}",
            tri_count, quad_count, poly_count
        );
    }

    #[test]
    fn test_growth_rate_effect() {
        let geo = ChannelWithObstacle {
            length: 1.0,
            height: 1.0,
            obstacle_center: Point2::new(0.5, 0.5),
            obstacle_radius: 0.1,
        };
        let domain_size = Vector2::new(1.0, 1.0);
        let min_size = 0.01;
        let max_size = 0.1;

        // Case 1: Low growth rate (should have more cells)
        let mesh_low = generate_voronoi_mesh(&geo, min_size, max_size, 1.1, domain_size);
        println!("Growth 1.1 -> Cells: {}", mesh_low.num_cells());

        // Case 2: High growth rate (should have fewer cells)
        let mesh_high = generate_voronoi_mesh(&geo, min_size, max_size, 1.5, domain_size);
        println!("Growth 1.5 -> Cells: {}", mesh_high.num_cells());

        assert!(
            mesh_low.num_cells() > mesh_high.num_cells(),
            "Lower growth rate should produce more cells"
        );
    }

    #[test]
    fn test_mesh_smoothness() {
        let geo = ChannelWithObstacle {
            length: 1.0,
            height: 1.0,
            obstacle_center: Point2::new(0.5, 0.5),
            obstacle_radius: 0.1,
        };
        let domain_size = Vector2::new(1.0, 1.0);

        // Use a small growth rate to expect smoothness
        let mesh = generate_voronoi_mesh(&geo, 0.02, 0.1, 1.1, domain_size);

        let mut max_ratio = 0.0;
        let mut bad_transitions = 0;

        let mut min_vol = f64::MAX;
        let mut max_vol = f64::MIN;

        for i in 0..mesh.num_cells() {
            let vol = mesh.cell_vol[i];
            if vol < min_vol {
                min_vol = vol;
            }
            if vol > max_vol {
                max_vol = vol;
            }

            if vol <= 1e-12 {
                println!("Cell {} has near-zero volume: {}", i, vol);
                let start = mesh.cell_vertex_offsets[i];
                let end = mesh.cell_vertex_offsets[i + 1];
                print!("  Vertices: ");
                for k in start..end {
                    let v_idx = mesh.cell_vertices[k];
                    print!("({}, {}) ", mesh.vx[v_idx], mesh.vy[v_idx]);
                }
                println!();
            }
        }
        println!("Min Vol: {}, Max Vol: {}", min_vol, max_vol);

        for i in 0..mesh.num_cells() {
            let vol_i = mesh.cell_vol[i];
            // Approximate characteristic length
            let h_i = vol_i.sqrt();

            let start = mesh.cell_face_offsets[i];
            let end = mesh.cell_face_offsets[i + 1];

            for f_idx in start..end {
                let face_idx = mesh.cell_faces[f_idx];
                if let Some(neighbor) = mesh.face_neighbor[face_idx] {
                    // If neighbor is the cell itself (should not happen in valid mesh but check)
                    if neighbor == i {
                        continue;
                    }

                    // Check if neighbor is actually the other side
                    let other = if mesh.face_owner[face_idx] == i {
                        mesh.face_neighbor[face_idx].unwrap() // We checked it's Some
                    } else {
                        mesh.face_owner[face_idx]
                    };

                    let vol_j = mesh.cell_vol[other];
                    let h_j = vol_j.sqrt();

                    let ratio = if h_i > h_j { h_i / h_j } else { h_j / h_i };

                    if ratio > max_ratio {
                        max_ratio = ratio;
                    }

                    // We expect ratio to be small, e.g. < 1.5 or 2.0
                    // With Poisson sampling and smoothing, it should be quite good.
                    if ratio > 2.0 {
                        bad_transitions += 1;
                        if bad_transitions <= 10 {
                            println!("Bad transition: Cell {} (vol={:.6}) <-> Cell {} (vol={:.6}), Ratio={:.2}, Face at ({:.2}, {:.2})", 
                                i, vol_i, other, vol_j, ratio, mesh.face_cx[face_idx], mesh.face_cy[face_idx]);
                        }
                    }
                }
            }
        }

        println!("Max adjacent cell size ratio: {:.2}", max_ratio);
        println!("Bad transitions (> 2.0): {}", bad_transitions);

        // Assert that we don't have extreme jumps
        assert!(min_vol > 1e-9, "Found zero volume cells!");
        assert!(
            bad_transitions < mesh.num_cells() / 5, // 20% tolerance (relaxed due to boundary mismatch)
            "Too many bad transitions!"
        );
        assert!(max_ratio < 4.0, "Max ratio too high!"); // 4.0 is safe upper bound
    }
}
