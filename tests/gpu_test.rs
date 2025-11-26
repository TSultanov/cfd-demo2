use cfd2::solver::gpu::GpuSolver;
use cfd2::solver::mesh::{generate_cut_cell_mesh, Geometry};
use nalgebra::{Point2, Vector2};

struct BoxGeo {
    w: f64,
    h: f64,
}
impl Geometry for BoxGeo {
    fn is_inside(&self, p: &Point2<f64>) -> bool {
        p.x >= 0.0 && p.x <= self.w && p.y >= 0.0 && p.y <= self.h
    }
    fn sdf(&self, p: &Point2<f64>) -> f64 {
        let dx = (p.x - self.w / 2.0).abs() - self.w / 2.0;
        let dy = (p.y - self.h / 2.0).abs() - self.h / 2.0;
        dx.max(dy).min(0.0) + Vector2::new(dx.max(0.0), dy.max(0.0)).norm()
    }
}

#[test]
fn test_gpu_solver_init() {
    // Create a simple mesh
    let geo = BoxGeo { w: 1.0, h: 1.0 };
    let mesh = generate_cut_cell_mesh(&geo, 0.1, 0.1, Vector2::new(1.0, 1.0));

    println!("Face Neighbors: {:?}", mesh.face_neighbor);
    println!("Face Areas: {:?}", mesh.face_area);
    println!("Cell Vols: {:?}", mesh.cell_vol);

    pollster::block_on(async {
        let mut solver = GpuSolver::new(&mesh).await;

        // Set initial velocity
        let init_u: Vec<(f64, f64)> = (0..mesh.num_cells()).map(|_| (1.0, 0.5)).collect();
        solver.set_u(&init_u);

        // Set pressure field P = x + y
        let init_p: Vec<f64> = (0..mesh.num_cells())
            .map(|i| {
                let cx = mesh.cell_cx[i];
                let cy = mesh.cell_cy[i];
                cx + cy
            })
            .collect();
        solver.set_p(&init_p);

        // Run gradient calculation
        solver.compute_gradient();

        // Get result
        // let result_u = solver.get_u().await;
        let result_grad_p = solver.get_grad_p().await;

        // Verify U (should be unchanged)
        // assert_eq!(result_u.len(), mesh.num_cells());
        // for (i, (ux, uy)) in result_u.iter().enumerate() {
        //     assert!((ux - 1.0).abs() < 1e-5, "Cell {} ux mismatch: got {}, expected 1.0", i, ux);
        //     assert!((uy - 0.5).abs() < 1e-5, "Cell {} uy mismatch: got {}, expected 0.5", i, uy);
        // }

        // Debug Cell 12
        let cell_idx = 12;
        println!("Offsets: {:?}", &mesh.cell_face_offsets[0..20]);
        println!("Debug Cell {}:", cell_idx);
        let start = mesh.cell_face_offsets[cell_idx];
        let end = mesh.cell_face_offsets[cell_idx + 1];
        println!("Faces range: {}..{}", start, end);
        for k in start..end {
            let face_idx = mesh.cell_faces[k];
            let owner = mesh.face_owner[face_idx];
            let neighbor = mesh.face_neighbor[face_idx];
            let nx = mesh.face_nx[face_idx];
            let ny = mesh.face_ny[face_idx];
            let area = mesh.face_area[face_idx];
            println!(
                "  Face {}: Owner={}, Neighbor={:?}, Normal=({:.2}, {:.2}), Area={:.2}",
                face_idx, owner, neighbor, nx, ny, area
            );
        }

        // Verify Grad P (should be approx 1.0, 1.0)
        let mut max_err = 0.0;
        println!("Grad P results:");
        for (i, (gx, gy)) in result_grad_p.iter().enumerate() {
            let cx = mesh.cell_cx[i];
            let cy = mesh.cell_cy[i];
            println!(
                "Cell {}: pos=({:.2}, {:.2}), grad=({:.2}, {:.2})",
                i, cx, cy, gx, gy
            );
            // Check if internal
            if cx > 0.15 && cx < 0.85 && cy > 0.15 && cy < 0.85 {
                let err_x = (gx - 1.0).abs();
                let err_y = (gy - 1.0).abs();
                if err_x > max_err {
                    max_err = err_x;
                }
                if err_y > max_err {
                    max_err = err_y;
                }
            }
        }
        println!("Max Grad P error (internal): {}", max_err);
        assert!(
            max_err < 0.1,
            "Gradient calculation error too high: {}",
            max_err
        );

        // Run full step to assemble matrix
        solver.step();

        // Check matrix values
        let matrix_values = solver.get_matrix_values().await;

        let mut count_pos = 0;
        let mut count_neg = 0;
        for &val in &matrix_values {
            if val > 1e-6 {
                count_pos += 1;
            }
            if val < -1e-6 {
                count_neg += 1;
            }
        }

        assert!(
            count_pos >= 64,
            "Expected at least 64 positive diagonal entries"
        );
        assert!(
            count_neg >= 200,
            "Expected many negative off-diagonal entries"
        );

        // Test Linear Solver
        // Set U = (x, y) so div(U) = 2
        let u_div: Vec<(f64, f64)> = (0..mesh.num_cells())
            .map(|i| (mesh.cell_cx[i], mesh.cell_cy[i]))
            .collect();
        solver.set_u(&u_div);

        // Step to compute RHS
        solver.step();

        // Solve
        solver.solve().await;

        let x_sol = solver.get_x().await;
        let max_x = x_sol.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        println!("Max correction: {}", max_x);
        assert!(
            max_x > 0.0,
            "Solver should produce non-zero correction for non-zero divergence"
        );
    });
}
