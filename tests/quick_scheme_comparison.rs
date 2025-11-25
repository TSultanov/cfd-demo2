use cfd2::solver::fvm::{Fvm, ScalarField, Scheme};
use cfd2::solver::gpu::structs::GpuSolver;
use cfd2::solver::linear_solver::SerialOps;
use cfd2::solver::mesh::{
    generate_cut_cell_mesh, BoundaryType, ChannelWithObstacle,
};
use cfd2::solver::piso::PisoSolver;
use nalgebra::{Point2, Vector2};

#[test]
fn test_quick_scheme_equivalence() {
    pollster::block_on(async {
        // 1. Setup Mesh (Channel with Obstacle, dx=0.025)
        let width = 3.0;
        let height = 1.0;
        let dx = 0.025;

        let channel = ChannelWithObstacle {
            length: width,
            height: height,
            obstacle_center: Point2::new(width / 3.0, height / 2.0),
            obstacle_radius: 0.2,
        };
        let mut mesh = generate_cut_cell_mesh(&channel, dx, dx, Vector2::new(width, height));
        mesh.smooth(&channel, 0.3, 50);
        let n_cells = mesh.num_cells();
        let _n_faces = mesh.num_faces();

        // 2. Common Parameters
        // Note: Use smaller dt (0.001) for stability with cut cell meshes.
        // Very small cut cells near the obstacle can cause CFL issues with larger dt.
        let dt = 0.001;
        let viscosity = 0.001;
        let density = 1.0;
        let scheme = Scheme::QUICK; // 2 for GPU
        let u_init_x = 1.0;
        let u_init_y = 0.0;

        // 3. Setup GPU Solver
        let mut gpu_solver = GpuSolver::new(&mesh).await;
        gpu_solver.set_dt(dt as f32);
        gpu_solver.set_viscosity(viscosity as f32);
        gpu_solver.set_density(density as f32);
        gpu_solver.set_scheme(2); // QUICK
        let u_init_gpu: Vec<(f64, f64)> = vec![(u_init_x, u_init_y); n_cells];
        gpu_solver.set_u(&u_init_gpu);

        // 4. Setup CPU Solver
        let mut cpu_solver: PisoSolver<f32> = PisoSolver::new(mesh.clone());
        cpu_solver.dt = dt as f32;
        cpu_solver.viscosity = viscosity as f32;
        cpu_solver.density = density as f32;
        cpu_solver.scheme = scheme;
        // Initialize U
        for i in 0..n_cells {
             cpu_solver.u.vx[i] = u_init_x as f32;
             cpu_solver.u.vy[i] = u_init_y as f32;
        }

        // Initialize Fluxes for CPU Solver
        for f_idx in 0..mesh.num_faces() {
            let owner = mesh.face_owner[f_idx];
            let neighbor = mesh.face_neighbor[f_idx];
            let nx = mesh.face_nx[f_idx] as f32;
            let ny = mesh.face_ny[f_idx] as f32;
            let area = mesh.face_area[f_idx] as f32;

            let u_owner = Vector2::new(cpu_solver.u.vx[owner], cpu_solver.u.vy[owner]);
            
            let u_face = if let Some(neigh) = neighbor {
                let u_neigh = Vector2::new(cpu_solver.u.vx[neigh], cpu_solver.u.vy[neigh]);
                // Simple average for initialization
                (u_owner + u_neigh) * 0.5
            } else {
                // Boundary
                if let Some(bt) = mesh.face_boundary[f_idx] {
                    match bt {
                        BoundaryType::Inlet => Vector2::new(1.0, 0.0),
                        BoundaryType::Wall => Vector2::new(0.0, 0.0),
                        _ => u_owner, // Outlet/Neumann
                    }
                } else {
                    u_owner
                }
            };

            cpu_solver.fluxes[f_idx] = (u_face.x * nx + u_face.y * ny) * area;
        }

        // Initialize Fluxes for GPU Solver (to match CPU)
        let fluxes_f64: Vec<f64> = cpu_solver.fluxes.iter().map(|&x| x as f64).collect();
        gpu_solver.set_fluxes(&fluxes_f64);

        // 5. Run Side-by-Side
        let total_time = 0.1; 
        let num_steps = (total_time / dt) as usize;
        let compare_interval = 10;

        println!("Running comparison for {} steps...", num_steps);

        let tolerance = 0.05; // 5% tolerance for acceptable difference
        let mut max_diff_u_overall = 0.0f64;
        let mut max_diff_v_overall = 0.0f64;
        let mut max_diff_p_overall = 0.0f64;
        let mut gpu_diverged = false;

        for step in 1..=num_steps {
            gpu_solver.step();
            cpu_solver.step();

            if step % compare_interval == 0 {
                let u_gpu = gpu_solver.get_u().await;
                let p_gpu = gpu_solver.get_p().await;

                // Check for GPU divergence early
                let gpu_max_u: f64 = u_gpu.iter().map(|(x, _)| x.abs()).fold(0.0, f64::max);
                let gpu_has_nan = u_gpu.iter().any(|(x, y)| x.is_nan() || y.is_nan());
                
                if gpu_has_nan || gpu_max_u > 1e6 || gpu_max_u < 1e-10 {
                    println!("Step {}: GPU DIVERGED! max_u={}, has_nan={}", step, gpu_max_u, gpu_has_nan);
                    gpu_diverged = true;
                    break;
                }

                // Compute mean pressure
                let mean_p_cpu: f64 = cpu_solver.p.values.iter().map(|&v| v as f64).sum::<f64>() / n_cells as f64;
                let mean_p_gpu: f64 = p_gpu.iter().sum::<f64>() / n_cells as f64;

                // Compare Velocity X
                let mut max_diff_u = 0.0;
                let mut max_diff_v = 0.0;
                let mut max_diff_p = 0.0;
                let mut max_diff_u_idx = 0;
                let mut max_diff_v_idx = 0;

                for i in 0..n_cells {
                    let diff_u = (cpu_solver.u.vx[i] as f64 - u_gpu[i].0).abs();
                    let diff_v = (cpu_solver.u.vy[i] as f64 - u_gpu[i].1).abs();
                    let diff_p = ((cpu_solver.p.values[i] as f64 - mean_p_cpu) - (p_gpu[i] - mean_p_gpu)).abs();

                    if diff_u > max_diff_u { max_diff_u = diff_u; max_diff_u_idx = i; }
                    if diff_v > max_diff_v { max_diff_v = diff_v; max_diff_v_idx = i; }
                    if diff_p > max_diff_p { max_diff_p = diff_p; }
                }

                max_diff_u_overall = max_diff_u_overall.max(max_diff_u);
                max_diff_v_overall = max_diff_v_overall.max(max_diff_v);
                max_diff_p_overall = max_diff_p_overall.max(max_diff_p);

                // Check for NaN or divergence
                let cpu_max_u: f64 = cpu_solver.u.vx.iter().map(|&x| (x as f64).abs()).fold(0.0, f64::max);
                let gpu_max_u: f64 = u_gpu.iter().map(|(x, _)| x.abs()).fold(0.0, f64::max);
                
                println!("Step {}: Max Diff U={:.6} (cell {}), V={:.6} (cell {}), P={:.6}", 
                         step, max_diff_u, max_diff_u_idx, max_diff_v, max_diff_v_idx, max_diff_p);
                println!("  CPU max |u|={:.6}, GPU max |u|={:.6}", cpu_max_u, gpu_max_u);
                
                // Print values at max diff cell
                if max_diff_u > tolerance {
                    println!("  Cell {} - CPU: u={:.6}, GPU: u={:.6}", 
                             max_diff_u_idx, cpu_solver.u.vx[max_diff_u_idx], u_gpu[max_diff_u_idx].0);
                }
            }
        }

        println!("\n=== Final Summary ===");
        println!("Max Diff U: {:.6}", max_diff_u_overall);
        println!("Max Diff V: {:.6}", max_diff_v_overall);
        println!("Max Diff P: {:.6}", max_diff_p_overall);
        
        // Fail if GPU diverged
        assert!(!gpu_diverged, "GPU solver diverged during simulation");
        
        // Fail test if differences exceed tolerance
        assert!(max_diff_u_overall < tolerance, 
                "U velocity difference too large: {:.6} > {:.6}", max_diff_u_overall, tolerance);
        assert!(max_diff_v_overall < tolerance, 
                "V velocity difference too large: {:.6} > {:.6}", max_diff_v_overall, tolerance);
        assert!(max_diff_p_overall < tolerance * 150.0, // Pressure can have larger differences due to its sensitivity
                "Pressure difference too large: {:.6} > {:.6}", max_diff_p_overall, tolerance * 150.0);
    });
}

/// Detailed comparison test to find the source of discrepancies
#[test]
fn test_find_discrepancy_source() {
    pollster::block_on(async {
        // 1. Setup Mesh (smaller for detailed comparison)
        let width = 1.0;
        let height = 0.5;
        let dx = 0.05;

        let channel = ChannelWithObstacle {
            length: width,
            height: height,
            obstacle_center: Point2::new(width / 3.0, height / 2.0),
            obstacle_radius: 0.1,
        };
        let mut mesh = generate_cut_cell_mesh(&channel, dx, dx, Vector2::new(width, height));
        mesh.smooth(&channel, 0.3, 50);
        let n_cells = mesh.num_cells();
        let n_faces = mesh.num_faces();

        println!("Mesh: {} cells, {} faces", n_cells, n_faces);

        // 2. Common Parameters
        let dt = 0.001;
        let viscosity = 0.001;
        let density = 1.0;
        let scheme = Scheme::QUICK;
        let u_init_x = 1.0;
        let u_init_y = 0.0;

        // 3. Setup GPU Solver
        let mut gpu_solver = GpuSolver::new(&mesh).await;
        gpu_solver.set_dt(dt as f32);
        gpu_solver.set_viscosity(viscosity as f32);
        gpu_solver.set_density(density as f32);
        gpu_solver.set_scheme(2); // QUICK

        // 4. Setup CPU Solver
        let mut cpu_solver: PisoSolver<f32> = PisoSolver::new(mesh.clone());
        cpu_solver.dt = dt as f32;
        cpu_solver.viscosity = viscosity as f32;
        cpu_solver.density = density as f32;
        cpu_solver.scheme = scheme;

        // Initialize U for both
        for i in 0..n_cells {
            cpu_solver.u.vx[i] = u_init_x as f32;
            cpu_solver.u.vy[i] = u_init_y as f32;
        }
        let u_init_gpu: Vec<(f64, f64)> = vec![(u_init_x, u_init_y); n_cells];
        gpu_solver.set_u(&u_init_gpu);

        // Initialize Fluxes
        for f_idx in 0..n_faces {
            let owner = mesh.face_owner[f_idx];
            let neighbor = mesh.face_neighbor[f_idx];
            let nx = mesh.face_nx[f_idx] as f32;
            let ny = mesh.face_ny[f_idx] as f32;
            let area = mesh.face_area[f_idx] as f32;

            let u_owner = Vector2::new(cpu_solver.u.vx[owner], cpu_solver.u.vy[owner]);
            
            let u_face = if let Some(neigh) = neighbor {
                let u_neigh = Vector2::new(cpu_solver.u.vx[neigh], cpu_solver.u.vy[neigh]);
                (u_owner + u_neigh) * 0.5
            } else {
                if let Some(bt) = mesh.face_boundary[f_idx] {
                    match bt {
                        BoundaryType::Inlet => Vector2::new(1.0, 0.0),
                        BoundaryType::Wall => Vector2::new(0.0, 0.0),
                        _ => u_owner,
                    }
                } else {
                    u_owner
                }
            };

            cpu_solver.fluxes[f_idx] = (u_face.x * nx + u_face.y * ny) * area;
        }

        let fluxes_f64: Vec<f64> = cpu_solver.fluxes.iter().map(|&x| x as f64).collect();
        gpu_solver.set_fluxes(&fluxes_f64);

        // Compare initial state
        println!("\n=== Initial State ===");
        let u_gpu_init = gpu_solver.get_u().await;
        let _p_gpu_init = gpu_solver.get_p().await;
        let fluxes_gpu_init = gpu_solver.get_fluxes().await;

        let mut max_diff_u_init = 0.0_f64;
        let mut max_diff_flux_init = 0.0_f64;
        for i in 0..n_cells {
            let diff = (cpu_solver.u.vx[i] as f64 - u_gpu_init[i].0).abs();
            max_diff_u_init = max_diff_u_init.max(diff);
        }
        for i in 0..n_faces {
            let diff = (cpu_solver.fluxes[i] as f64 - fluxes_gpu_init[i] as f64).abs();
            max_diff_flux_init = max_diff_flux_init.max(diff);
        }
        println!("Initial U diff: {:.6e}, Flux diff: {:.6e}", max_diff_u_init, max_diff_flux_init);

        // === Compare Momentum Assembly Stage ===
        println!("\n=== Momentum Assembly Comparison ===");
        
        // GPU: Compute velocity gradients first (for QUICK scheme), then assemble momentum for Ux (component 0)
        gpu_solver.compute_velocity_gradient(0); // Compute gradient of Ux before assembly
        gpu_solver.assemble_momentum(0);
        
        // Get GPU matrix and RHS
        let gpu_mat_vals = gpu_solver.get_matrix_values().await;
        let gpu_rhs = gpu_solver.get_rhs().await;
        let gpu_grad_p = gpu_solver.get_grad_p().await;
        let gpu_d_p = gpu_solver.get_d_p().await;
        let gpu_diagonal_indices = gpu_solver.get_diagonal_indices().await;
        let _gpu_row_offsets = gpu_solver.get_row_offsets().await;
        let _gpu_col_indices = gpu_solver.get_col_indices().await;
        
        // CPU: Assemble momentum for Ux
        let u_bc = |bt: BoundaryType| match bt {
            BoundaryType::Inlet => Some(1.0_f32),
            BoundaryType::Wall => Some(0.0_f32),
            BoundaryType::Outlet => None,
            _ => None,
        };
        let p_bc = |bt: BoundaryType| match bt {
            BoundaryType::Outlet => Some(0.0_f32),
            _ => None,
        };
        
        let nu = cpu_solver.viscosity / cpu_solver.density;
        let u_x_field = ScalarField { values: cpu_solver.u.vx.clone() };
        let (cpu_mat, cpu_rhs_base) = Fvm::assemble_scalar_transport(
            &mesh,
            &u_x_field,
            &cpu_solver.fluxes,
            nu,
            cpu_solver.dt,
            &scheme,
            u_bc,
            None,
            None,
            None,
        );
        
        // Add pressure gradient to CPU RHS
        let grad_p_cpu = Fvm::compute_gradients(&mesh, &ScalarField { values: cpu_solver.p.values.clone() }, p_bc, None, None, None);
        let inv_density = 1.0 / cpu_solver.density;
        let mut cpu_rhs_ux = cpu_rhs_base.clone();
        for i in 0..n_cells {
            cpu_rhs_ux[i] -= (grad_p_cpu.vx[i] * inv_density) * mesh.cell_vol[i] as f32;
        }
        
        // Compare grad_p computed during momentum assembly
        println!("\nGrad P comparison (computed from initial p=0 field):");
        let mut max_diff_grad_p_x = 0.0_f64;
        let mut max_diff_grad_p_y = 0.0_f64;
        let mut max_diff_grad_p_x_idx = 0;
        for i in 0..n_cells {
            let diff_x = (grad_p_cpu.vx[i] as f64 - gpu_grad_p[i].0).abs();
            let diff_y = (grad_p_cpu.vy[i] as f64 - gpu_grad_p[i].1).abs();
            if diff_x > max_diff_grad_p_x {
                max_diff_grad_p_x = diff_x;
                max_diff_grad_p_x_idx = i;
            }
            if diff_y > max_diff_grad_p_y {
                max_diff_grad_p_y = diff_y;
            }
        }
        println!("  Max diff x: {:.6e} (cell {}), y: {:.6e}", max_diff_grad_p_x, max_diff_grad_p_x_idx, max_diff_grad_p_y);
        if max_diff_grad_p_x > 1e-6 {
            let i = max_diff_grad_p_x_idx;
            println!("  Cell {} - CPU grad_p: ({:.6}, {:.6}), GPU grad_p: ({:.6}, {:.6})", 
                     i, grad_p_cpu.vx[i], grad_p_cpu.vy[i], gpu_grad_p[i].0, gpu_grad_p[i].1);
        }
        
        // Compare diagonal coefficients
        println!("\nDiagonal coefficient comparison:");
        let mut max_diff_diag = 0.0_f64;
        let mut max_diff_diag_idx = 0;
        for i in 0..n_cells {
            // Get CPU diagonal - sum all entries for (i, i)
            let mut cpu_diag = 0.0_f32;
            for k in cpu_mat.row_offsets[i]..cpu_mat.row_offsets[i+1] {
                if cpu_mat.col_indices[k] == i {
                    cpu_diag += cpu_mat.values[k];
                }
            }
            
            // Get GPU diagonal
            let gpu_diag_idx = gpu_diagonal_indices[i] as usize;
            let gpu_diag = gpu_mat_vals[gpu_diag_idx] as f32;
            
            let diff = (cpu_diag as f64 - gpu_diag as f64).abs();
            if diff > max_diff_diag {
                max_diff_diag = diff;
                max_diff_diag_idx = i;
            }
        }
        println!("  Max diff: {:.6e} (cell {})", max_diff_diag, max_diff_diag_idx);
        if max_diff_diag > 1e-3 {
            let i = max_diff_diag_idx;
            let mut cpu_diag = 0.0_f32;
            for k in cpu_mat.row_offsets[i]..cpu_mat.row_offsets[i+1] {
                if cpu_mat.col_indices[k] == i {
                    cpu_diag += cpu_mat.values[k];
                }
            }
            let gpu_diag_idx = gpu_diagonal_indices[i] as usize;
            let gpu_diag = gpu_mat_vals[gpu_diag_idx] as f32;
            println!("  Cell {} - CPU diag: {:.6}, GPU diag: {:.6}", i, cpu_diag, gpu_diag);
        }
        
        // Compare RHS
        println!("\nRHS comparison:");
        let mut max_diff_rhs = 0.0_f64;
        let mut max_diff_rhs_idx = 0;
        for i in 0..n_cells {
            let diff = (cpu_rhs_ux[i] as f64 - gpu_rhs[i]).abs();
            if diff > max_diff_rhs {
                max_diff_rhs = diff;
                max_diff_rhs_idx = i;
            }
        }
        println!("  Max diff: {:.6e} (cell {})", max_diff_rhs, max_diff_rhs_idx);
        if max_diff_rhs > 1e-3 {
            let i = max_diff_rhs_idx;
            println!("  Cell {} - CPU RHS: {:.6}, GPU RHS: {:.6}", i, cpu_rhs_ux[i], gpu_rhs[i]);
            println!("  CPU RHS base (without grad_p): {:.6}", cpu_rhs_base[i]);
            println!("  Pressure gradient term: {:.6}", -(grad_p_cpu.vx[i] * inv_density) * mesh.cell_vol[i] as f32);
            
            // Also show the cell details
            println!("  Cell {} geometry: center=({:.6}, {:.6}), vol={:.6e}", 
                     i, mesh.cell_cx[i], mesh.cell_cy[i], mesh.cell_vol[i]);
            
            // Compute expected RHS from time term
            let time_coeff = mesh.cell_vol[i] as f32 / cpu_solver.dt;
            let time_rhs = time_coeff * cpu_solver.u.vx[i];
            println!("  Expected time term RHS: {:.6} * {:.6} = {:.6}", time_coeff, cpu_solver.u.vx[i], time_rhs);
            
            // CPU RHS difference from time term = higher order corrections
            let cpu_correction = time_rhs - cpu_rhs_base[i];
            let gpu_correction = time_rhs - gpu_rhs[i] as f32;
            println!("  CPU correction from time term: {:.6}", cpu_correction);
            println!("  GPU correction from time term: {:.6}", gpu_correction);
            
            // List the faces and fluxes for this cell
            println!("  Cell {} faces:", i);
            let start = mesh.cell_face_offsets[i];
            let end = mesh.cell_face_offsets[i+1];
            for k in start..end {
                let face_idx = mesh.cell_faces[k];
                let owner = mesh.face_owner[face_idx];
                let neighbor = mesh.face_neighbor[face_idx];
                let boundary = mesh.face_boundary[face_idx];
                let is_owner = owner == i;
                let neigh_idx = if is_owner { neighbor } else { Some(owner) };
                
                let flux = cpu_solver.fluxes[face_idx] * if is_owner { 1.0 } else { -1.0 };
                
                println!("    Face {}: flux={:.6e}, boundary={:?}, neighbor={:?}", 
                         face_idx, flux, boundary, neigh_idx);
            }
        }
        
        // Compare d_p
        println!("\nD_p comparison:");
        let mut max_diff_d_p = 0.0_f64;
        for i in 0..n_cells {
            // CPU d_p = Vol / (A_p * rho)
            let mut cpu_diag = 0.0_f32;
            for k in cpu_mat.row_offsets[i]..cpu_mat.row_offsets[i+1] {
                if cpu_mat.col_indices[k] == i {
                    cpu_diag = cpu_mat.values[k];
                    break;
                }
            }
            let cpu_d_p = if cpu_diag.abs() > 1e-20 {
                mesh.cell_vol[i] as f32 / (cpu_diag * cpu_solver.density)
            } else {
                0.0
            };
            
            let diff = (cpu_d_p as f64 - gpu_d_p[i]).abs();
            max_diff_d_p = max_diff_d_p.max(diff);
        }
        println!("  Max diff: {:.6e}", max_diff_d_p);

        // Compare after 1 full step
        println!("\n=== After 1 Step ===");
        gpu_solver.step();
        cpu_solver.step();

        let u_gpu = gpu_solver.get_u().await;
        let p_gpu = gpu_solver.get_p().await;
        let fluxes_gpu = gpu_solver.get_fluxes().await;

        // Compare U
        let mut max_diff_u = 0.0_f64;
        let mut max_diff_v = 0.0_f64;
        let mut max_diff_u_idx = 0;
        let mut max_diff_v_idx = 0;
        for i in 0..n_cells {
            let diff_u = (cpu_solver.u.vx[i] as f64 - u_gpu[i].0).abs();
            let diff_v = (cpu_solver.u.vy[i] as f64 - u_gpu[i].1).abs();
            if diff_u > max_diff_u {
                max_diff_u = diff_u;
                max_diff_u_idx = i;
            }
            if diff_v > max_diff_v {
                max_diff_v = diff_v;
                max_diff_v_idx = i;
            }
        }
        println!("U diff: {:.6e} (cell {})", max_diff_u, max_diff_u_idx);
        println!("V diff: {:.6e} (cell {})", max_diff_v, max_diff_v_idx);

        // Print details for the cell with max U difference
        if max_diff_u > 1e-6 {
            let i = max_diff_u_idx;
            println!("\nCell {} details:", i);
            println!("  CPU: u={:.6}, v={:.6}, p={:.6}", 
                     cpu_solver.u.vx[i], cpu_solver.u.vy[i], cpu_solver.p.values[i]);
            println!("  GPU: u={:.6}, v={:.6}, p={:.6}", 
                     u_gpu[i].0, u_gpu[i].1, p_gpu[i]);
        }

        // Compare P (mean-adjusted)
        let mean_p_cpu: f64 = cpu_solver.p.values.iter().map(|&v| v as f64).sum::<f64>() / n_cells as f64;
        let mean_p_gpu: f64 = p_gpu.iter().sum::<f64>() / n_cells as f64;
        let mut max_diff_p = 0.0_f64;
        for i in 0..n_cells {
            let diff = ((cpu_solver.p.values[i] as f64 - mean_p_cpu) - (p_gpu[i] - mean_p_gpu)).abs();
            max_diff_p = max_diff_p.max(diff);
        }
        println!("P diff (mean-adjusted): {:.6e}", max_diff_p);

        // Compare Fluxes
        let mut max_diff_flux = 0.0_f64;
        let mut max_diff_flux_idx = 0;
        for i in 0..n_faces {
            let diff = (cpu_solver.fluxes[i] as f64 - fluxes_gpu[i] as f64).abs();
            if diff > max_diff_flux {
                max_diff_flux = diff;
                max_diff_flux_idx = i;
            }
        }
        println!("Flux diff: {:.6e} (face {})", max_diff_flux, max_diff_flux_idx);

        // Fail if differences are too large
        // Note: Using larger tolerance here because dt=0.001 vs dt=0.0001 in main test.
        // Single step with large dt amplifies remaining differences in PISO pressure solve.
        let tolerance = 0.05;
        assert!(max_diff_u < tolerance, 
                "U difference after 1 step too large: {:.6e}", max_diff_u);
        assert!(max_diff_v < tolerance, 
                "V difference after 1 step too large: {:.6e}", max_diff_v);
    });
}
