use cfd2::solver::fvm::{Fvm, ScalarField, Scheme, VectorField};
use cfd2::solver::gpu::structs::GpuSolver;
use cfd2::solver::mesh::{
    generate_cut_cell_mesh, BoundaryType, Geometry, Mesh, RectangularChannel,
};
use nalgebra::{Point2, Vector2};
use std::collections::HashMap;

#[test]
fn test_quick_scheme_equivalence() {
    pollster::block_on(async {
        // 1. Setup Mesh (Small 10x10 grid)
        let width = 1.0;
        let height = 1.0;
        let nx = 10;
        let ny = 10;
        let dx = width / nx as f64;

        let channel = RectangularChannel {
            length: width,
            height: height,
        };
        let mesh = generate_cut_cell_mesh(&channel, dx, dx * 1.5, Vector2::new(width, height));

        // 2. Setup Fields
        let n_cells = mesh.num_cells();
        let mut u_cpu = VectorField::new(n_cells, Vector2::new(0.0, 0.0));
        let mut u_gpu_init = vec![(0.0, 0.0); n_cells];

        // Initialize with a vortex-like field to have non-trivial gradients
        for i in 0..n_cells {
            let cx = mesh.cell_cx[i];
            let cy = mesh.cell_cy[i];

            // Taylor-Green Vortex like field
            let u = (std::f64::consts::PI * cx).sin() * (std::f64::consts::PI * cy).cos();
            let v = -(std::f64::consts::PI * cx).cos() * (std::f64::consts::PI * cy).sin();

            u_cpu.vx[i] = u;
            u_cpu.vy[i] = v;
            u_gpu_init[i] = (u, v);
        }

        // 3. Setup CPU Solver
        let dt = 0.01;
        let density = 1.0;
        let viscosity = 0.01; // Low viscosity to make convection dominant

        // Compute Fluxes (Mass Fluxes)
        let mut fluxes = vec![0.0; mesh.num_faces()];
        for face_idx in 0..mesh.num_faces() {
            let owner = mesh.face_owner[face_idx];
            let neighbor = mesh.face_neighbor[face_idx];
            let nx = mesh.face_nx[face_idx];
            let ny = mesh.face_ny[face_idx];
            let area = mesh.face_area[face_idx];

            // Linear interpolation for flux
            let u_own = Vector2::new(u_cpu.vx[owner], u_cpu.vy[owner]);
            let u_face = if let Some(neigh) = neighbor {
                let u_neigh = Vector2::new(u_cpu.vx[neigh], u_cpu.vy[neigh]);
                0.5 * (u_own + u_neigh)
            } else {
                u_own // Simplified BC
            };

            let normal = Vector2::new(nx, ny);
            fluxes[face_idx] = density * u_face.dot(&normal) * area;
        }

        // Compute Gradients (CPU)
        let boundary_value = |bt: BoundaryType| -> Option<f64> {
            match bt {
                BoundaryType::Wall => Some(0.0),
                BoundaryType::Inlet => Some(1.0), // Match GPU hardcoded Inlet
                _ => None,                        // Neumann
            }
        };

        // We test X-momentum
        let phi_cpu = ScalarField {
            values: u_cpu.vx.clone(),
        };
        // let grads_cpu = Fvm::compute_gradients(&mesh, &phi_cpu, boundary_value, None, None, None);

        // Assemble CPU (QUICK)
        let (matrix_cpu, rhs_cpu) = Fvm::assemble_scalar_transport(
            &mesh,
            &phi_cpu,
            &fluxes,
            viscosity,
            dt,
            &Scheme::QUICK,
            boundary_value,
            None,
            None,
            None,
        );

        // 4. Setup GPU Solver
        let mut gpu_solver = GpuSolver::new(&mesh).await;
        gpu_solver.set_dt(dt as f32);
        gpu_solver.set_density(density as f32);
        gpu_solver.set_viscosity(viscosity as f32);
        gpu_solver.set_scheme(2); // 2 = QUICK

        gpu_solver.set_u(&u_gpu_init);

        // Let's expose `set_fluxes` in `GpuSolver` to be safe.
        gpu_solver.set_fluxes(&fluxes);

        // 5. Run GPU Assembly
        // Compute gradients first (lagged)
        gpu_solver.compute_velocity_gradient(0); // 0 = X component

        // Assemble Momentum (X component)
        gpu_solver.assemble_momentum(0);

        // 6. Download Results
        let matrix_gpu_vals = gpu_solver.get_matrix_values().await;
        let rhs_gpu = gpu_solver.get_rhs().await;
        let diag_indices = gpu_solver.get_diagonal_indices().await;

        // 7. Compare
        println!("Comparing CPU and GPU results...");

        // Compare RHS
        let mut max_diff_rhs = 0.0;
        for i in 0..n_cells {
            let diff = (rhs_cpu[i] - rhs_gpu[i]).abs();
            if diff > max_diff_rhs {
                max_diff_rhs = diff;
            }
            if diff > 1e-4 {
                println!(
                    "RHS Mismatch at {}: CPU={}, GPU={}, Diff={}",
                    i, rhs_cpu[i], rhs_gpu[i], diff
                );
            }
        }
        println!("Max RHS Difference: {}", max_diff_rhs);
        assert!(max_diff_rhs < 1e-2, "RHS difference too large");

        // Compare Matrix Diagonals
        let mut max_diff_diag = 0.0;

        // Extract diagonal from CPU matrix
        let mut diag_cpu = vec![0.0; n_cells];
        for i in 0..n_cells {
            let start = matrix_cpu.row_offsets[i];
            let end = matrix_cpu.row_offsets[i + 1];
            for k in start..end {
                if matrix_cpu.col_indices[k] == i {
                    diag_cpu[i] += matrix_cpu.values[k];
                }
            }
        }

        for i in 0..n_cells {
            let diag_gpu_idx = diag_indices[i] as usize;
            let diag_gpu = matrix_gpu_vals[diag_gpu_idx];

            let diff = (diag_cpu[i] - diag_gpu).abs();
            if diff > max_diff_diag {
                max_diff_diag = diff;
            }
        }
        assert!(max_diff_diag < 1e-2, "Diagonal difference too large");

        // Compare Matrix-Vector Product
        let x_test: Vec<f64> = (0..n_cells).map(|i| (i as f64 * 0.1).sin()).collect();

        // CPU MatVec
        let mut ax_cpu = vec![0.0; n_cells];
        matrix_cpu.mat_vec_mul(&x_test, &mut ax_cpu);

        // GPU MatVec (Simulated using downloaded values)
        let row_offsets = gpu_solver.get_row_offsets().await;
        let col_indices = gpu_solver.get_col_indices().await;

        let mut ax_gpu = vec![0.0; n_cells];
        for i in 0..n_cells {
            let start = row_offsets[i] as usize;
            let end = row_offsets[i + 1] as usize;
            let mut sum = 0.0;
            for k in start..end {
                let col = col_indices[k] as usize;
                let val = matrix_gpu_vals[k];
                sum += val * x_test[col];
            }
            ax_gpu[i] = sum;
        }

        let mut max_diff_mv = 0.0;
        for i in 0..n_cells {
            let diff = (ax_cpu[i] - ax_gpu[i]).abs();
            if diff > max_diff_mv {
                max_diff_mv = diff;
            }
        }
        println!("Max MatVec Difference: {}", max_diff_mv);
        assert!(
            max_diff_mv < 1e-2,
            "Matrix-Vector product difference too large"
        );
    });
}
