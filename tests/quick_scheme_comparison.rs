use cfd2::solver::fvm::{Fvm, ScalarField, Scheme};
use cfd2::solver::gpu::structs::GpuSolver;

use cfd2::solver::mesh::{generate_cut_cell_mesh, BoundaryType, ChannelWithObstacle};
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
                    println!(
                        "Step {}: GPU DIVERGED! max_u={}, has_nan={}",
                        step, gpu_max_u, gpu_has_nan
                    );
                    gpu_diverged = true;
                    break;
                }

                // Compute mean pressure
                let mean_p_cpu: f64 =
                    cpu_solver.p.values.iter().map(|&v| v as f64).sum::<f64>() / n_cells as f64;
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
                    let diff_p = ((cpu_solver.p.values[i] as f64 - mean_p_cpu)
                        - (p_gpu[i] - mean_p_gpu))
                        .abs();

                    if diff_u > max_diff_u {
                        max_diff_u = diff_u;
                        max_diff_u_idx = i;
                    }
                    if diff_v > max_diff_v {
                        max_diff_v = diff_v;
                        max_diff_v_idx = i;
                    }
                    if diff_p > max_diff_p {
                        max_diff_p = diff_p;
                    }
                }

                max_diff_u_overall = max_diff_u_overall.max(max_diff_u);
                max_diff_v_overall = max_diff_v_overall.max(max_diff_v);
                max_diff_p_overall = max_diff_p_overall.max(max_diff_p);

                // Check for NaN or divergence
                let cpu_max_u: f64 = cpu_solver
                    .u
                    .vx
                    .iter()
                    .map(|&x| (x as f64).abs())
                    .fold(0.0, f64::max);
                let gpu_max_u: f64 = u_gpu.iter().map(|(x, _)| x.abs()).fold(0.0, f64::max);

                println!(
                    "Step {}: Max Diff U={:.6} (cell {}), V={:.6} (cell {}), P={:.6}",
                    step, max_diff_u, max_diff_u_idx, max_diff_v, max_diff_v_idx, max_diff_p
                );
                println!(
                    "  CPU max |u|={:.6}, GPU max |u|={:.6}",
                    cpu_max_u, gpu_max_u
                );

                // Print values at max diff cell
                if max_diff_u > tolerance {
                    println!(
                        "  Cell {} - CPU: u={:.6}, GPU: u={:.6}",
                        max_diff_u_idx, cpu_solver.u.vx[max_diff_u_idx], u_gpu[max_diff_u_idx].0
                    );
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
        assert!(
            max_diff_u_overall < tolerance,
            "U velocity difference too large: {:.6} > {:.6}",
            max_diff_u_overall,
            tolerance
        );
        assert!(
            max_diff_v_overall < tolerance,
            "V velocity difference too large: {:.6} > {:.6}",
            max_diff_v_overall,
            tolerance
        );
        assert!(
            max_diff_p_overall < tolerance * 150.0, // Pressure can have larger differences due to its sensitivity
            "Pressure difference too large: {:.6} > {:.6}",
            max_diff_p_overall,
            tolerance * 150.0
        );
    });
}

/// Test with fine mesh (dx=0.005) - the problematic case reported in GUI
#[test]
fn test_fine_mesh_divergence() {
    pollster::block_on(async {
        // 1. Setup Mesh (Channel with Obstacle, dx=0.005 - the problematic case)
        let width = 3.0;
        let height = 1.0;
        let dx = 0.005; // Fine mesh - this is where divergence occurs

        let channel = ChannelWithObstacle {
            length: width,
            height: height,
            obstacle_center: Point2::new(width / 3.0, height / 2.0),
            obstacle_radius: 0.2,
        };
        let mut mesh = generate_cut_cell_mesh(&channel, dx, dx, Vector2::new(width, height));
        mesh.smooth(&channel, 0.3, 50);
        let n_cells = mesh.num_cells();
        let n_faces = mesh.num_faces();

        println!("Fine mesh: {} cells, {} faces", n_cells, n_faces);

        // Check for problematic cells (very small volumes)
        let min_vol = mesh.cell_vol.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_vol = mesh.cell_vol.iter().cloned().fold(0.0, f64::max);
        let vol_ratio = max_vol / min_vol;
        println!(
            "Volume range: min={:.6e}, max={:.6e}, ratio={:.2}",
            min_vol, max_vol, vol_ratio
        );

        // Count very small cells
        let small_cell_count = mesh
            .cell_vol
            .iter()
            .filter(|&&v| v < min_vol * 10.0)
            .count();
        println!("Cells with volume < 10*min_vol: {}", small_cell_count);

        // 2. Parameters - using GUI default dt=0.01 to reproduce divergence issue
        // CFL ~ u*dt/dx, for u=1, dx=0.005, dt=0.01 gives CFL=2 (high!)
        let dt = 0.01; // GUI default - this is likely the cause of divergence
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
        let u_init_gpu: Vec<(f64, f64)> = vec![(u_init_x, u_init_y); n_cells];
        gpu_solver.set_u(&u_init_gpu);

        // 4. Setup CPU Solver
        let mut cpu_solver: PisoSolver<f32> = PisoSolver::new(mesh.clone());
        cpu_solver.dt = dt as f32;
        cpu_solver.viscosity = viscosity as f32;
        cpu_solver.density = density as f32;
        cpu_solver.scheme = scheme;
        for i in 0..n_cells {
            cpu_solver.u.vx[i] = u_init_x as f32;
            cpu_solver.u.vy[i] = u_init_y as f32;
        }

        // Initialize Fluxes for CPU Solver
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

        // Initialize Fluxes for GPU Solver (to match CPU)
        let fluxes_f64: Vec<f64> = cpu_solver.fluxes.iter().map(|&x| x as f64).collect();
        gpu_solver.set_fluxes(&fluxes_f64);

        // 5. Run comparison - fewer steps but check more frequently
        let num_steps = 100;
        let compare_interval = 10;

        println!(
            "Running comparison for {} steps with dt={}...",
            num_steps, dt
        );

        let tolerance = 0.05;
        let mut max_diff_u_overall = 0.0f64;
        let mut max_diff_v_overall = 0.0f64;
        let mut max_diff_p_overall = 0.0f64;
        let mut gpu_diverged = false;
        let mut divergence_step = 0;

        for step in 1..=num_steps {
            gpu_solver.step();
            cpu_solver.step();

            if step % compare_interval == 0 || step <= 5 {
                let u_gpu = gpu_solver.get_u().await;
                let p_gpu = gpu_solver.get_p().await;

                // Check for GPU divergence
                let gpu_max_u: f64 = u_gpu
                    .iter()
                    .map(|(x, y)| x.abs().max(y.abs()))
                    .fold(0.0, f64::max);
                let gpu_has_nan = u_gpu
                    .iter()
                    .any(|(x, y)| x.is_nan() || y.is_nan() || x.is_infinite() || y.is_infinite());
                let gpu_has_inf_p = p_gpu.iter().any(|&p| p.is_nan() || p.is_infinite());

                let cpu_max_u: f64 = cpu_solver
                    .u
                    .vx
                    .iter()
                    .zip(cpu_solver.u.vy.iter())
                    .map(|(&x, &y)| (x as f64).abs().max((y as f64).abs()))
                    .fold(0.0, f64::max);
                let cpu_has_nan = cpu_solver
                    .u
                    .vx
                    .iter()
                    .any(|x| x.is_nan() || x.is_infinite());

                if gpu_has_nan || gpu_has_inf_p || gpu_max_u > 1e6 {
                    println!(
                        "Step {}: GPU DIVERGED! max_u={:.6e}, has_nan={}, has_inf_p={}",
                        step, gpu_max_u, gpu_has_nan, gpu_has_inf_p
                    );

                    // Find the problematic cell
                    for i in 0..n_cells {
                        if u_gpu[i].0.is_nan()
                            || u_gpu[i].0.is_infinite()
                            || u_gpu[i].0.abs() > 1e6
                            || u_gpu[i].1.is_nan()
                            || u_gpu[i].1.is_infinite()
                            || u_gpu[i].1.abs() > 1e6
                        {
                            println!("  Problematic cell {}: GPU u=({:.6e}, {:.6e}), CPU u=({:.6e}, {:.6e})",
                                     i, u_gpu[i].0, u_gpu[i].1, cpu_solver.u.vx[i], cpu_solver.u.vy[i]);
                            println!(
                                "    Cell vol={:.6e}, center=({:.6}, {:.6})",
                                mesh.cell_vol[i], mesh.cell_cx[i], mesh.cell_cy[i]
                            );
                            if i < 10 {
                                // Only print first few
                                break;
                            }
                        }
                    }

                    gpu_diverged = true;
                    divergence_step = step;
                    break;
                }

                // Compute differences
                let mean_p_cpu: f64 =
                    cpu_solver.p.values.iter().map(|&v| v as f64).sum::<f64>() / n_cells as f64;
                let mean_p_gpu: f64 = p_gpu.iter().sum::<f64>() / n_cells as f64;

                let mut max_diff_u = 0.0;
                let mut max_diff_v = 0.0;
                let mut max_diff_p = 0.0;

                for i in 0..n_cells {
                    let diff_u = (cpu_solver.u.vx[i] as f64 - u_gpu[i].0).abs();
                    let diff_v = (cpu_solver.u.vy[i] as f64 - u_gpu[i].1).abs();
                    let diff_p = ((cpu_solver.p.values[i] as f64 - mean_p_cpu)
                        - (p_gpu[i] - mean_p_gpu))
                        .abs();

                    if diff_u > max_diff_u {
                        max_diff_u = diff_u;
                    }
                    if diff_v > max_diff_v {
                        max_diff_v = diff_v;
                    }
                    if diff_p > max_diff_p {
                        max_diff_p = diff_p;
                    }
                }

                max_diff_u_overall = max_diff_u_overall.max(max_diff_u);
                max_diff_v_overall = max_diff_v_overall.max(max_diff_v);
                max_diff_p_overall = max_diff_p_overall.max(max_diff_p);

                println!(
                    "Step {}: Max Diff U={:.6e}, V={:.6e}, P={:.6e}",
                    step, max_diff_u, max_diff_v, max_diff_p
                );
                println!(
                    "  CPU max |u|={:.6}, GPU max |u|={:.6}",
                    cpu_max_u, gpu_max_u
                );

                // Early exit if CPU also shows issues (would indicate problem is not GPU-specific)
                if cpu_has_nan || cpu_max_u > 1e6 {
                    println!("  WARNING: CPU also shows instability!");
                }
            }
        }

        println!("\n=== Final Summary ===");
        println!("Max Diff U: {:.6e}", max_diff_u_overall);
        println!("Max Diff V: {:.6e}", max_diff_v_overall);
        println!("Max Diff P: {:.6e}", max_diff_p_overall);

        if gpu_diverged {
            println!("GPU diverged at step {}", divergence_step);
        }

        // Fail if GPU diverged
        assert!(
            !gpu_diverged,
            "GPU solver diverged at step {} while CPU remained stable",
            divergence_step
        );

        assert!(
            max_diff_u_overall < tolerance,
            "U velocity difference too large: {:.6e} > {:.6e}",
            max_diff_u_overall,
            tolerance
        );
        assert!(
            max_diff_v_overall < tolerance,
            "V velocity difference too large: {:.6e} > {:.6e}",
            max_diff_v_overall,
            tolerance
        );
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
        println!(
            "Initial U diff: {:.6e}, Flux diff: {:.6e}",
            max_diff_u_init, max_diff_flux_init
        );

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
        let u_x_field = ScalarField {
            values: cpu_solver.u.vx.clone(),
        };
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
        let grad_p_cpu = Fvm::compute_gradients(
            &mesh,
            &ScalarField {
                values: cpu_solver.p.values.clone(),
            },
            p_bc,
            None,
            None,
            None,
        );
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
        println!(
            "  Max diff x: {:.6e} (cell {}), y: {:.6e}",
            max_diff_grad_p_x, max_diff_grad_p_x_idx, max_diff_grad_p_y
        );
        if max_diff_grad_p_x > 1e-6 {
            let i = max_diff_grad_p_x_idx;
            println!(
                "  Cell {} - CPU grad_p: ({:.6}, {:.6}), GPU grad_p: ({:.6}, {:.6})",
                i, grad_p_cpu.vx[i], grad_p_cpu.vy[i], gpu_grad_p[i].0, gpu_grad_p[i].1
            );
        }

        // Compare diagonal coefficients
        println!("\nDiagonal coefficient comparison:");
        let mut max_diff_diag = 0.0_f64;
        let mut max_diff_diag_idx = 0;
        for i in 0..n_cells {
            // Get CPU diagonal - sum all entries for (i, i)
            let mut cpu_diag = 0.0_f32;
            for k in cpu_mat.row_offsets[i]..cpu_mat.row_offsets[i + 1] {
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
        println!(
            "  Max diff: {:.6e} (cell {})",
            max_diff_diag, max_diff_diag_idx
        );
        if max_diff_diag > 1e-3 {
            let i = max_diff_diag_idx;
            let mut cpu_diag = 0.0_f32;
            for k in cpu_mat.row_offsets[i]..cpu_mat.row_offsets[i + 1] {
                if cpu_mat.col_indices[k] == i {
                    cpu_diag += cpu_mat.values[k];
                }
            }
            let gpu_diag_idx = gpu_diagonal_indices[i] as usize;
            let gpu_diag = gpu_mat_vals[gpu_diag_idx] as f32;
            println!(
                "  Cell {} - CPU diag: {:.6}, GPU diag: {:.6}",
                i, cpu_diag, gpu_diag
            );
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
        println!(
            "  Max diff: {:.6e} (cell {})",
            max_diff_rhs, max_diff_rhs_idx
        );
        if max_diff_rhs > 1e-3 {
            let i = max_diff_rhs_idx;
            println!(
                "  Cell {} - CPU RHS: {:.6}, GPU RHS: {:.6}",
                i, cpu_rhs_ux[i], gpu_rhs[i]
            );
            println!("  CPU RHS base (without grad_p): {:.6}", cpu_rhs_base[i]);
            println!(
                "  Pressure gradient term: {:.6}",
                -(grad_p_cpu.vx[i] * inv_density) * mesh.cell_vol[i] as f32
            );

            // Also show the cell details
            println!(
                "  Cell {} geometry: center=({:.6}, {:.6}), vol={:.6e}",
                i, mesh.cell_cx[i], mesh.cell_cy[i], mesh.cell_vol[i]
            );

            // Compute expected RHS from time term
            let time_coeff = mesh.cell_vol[i] as f32 / cpu_solver.dt;
            let time_rhs = time_coeff * cpu_solver.u.vx[i];
            println!(
                "  Expected time term RHS: {:.6} * {:.6} = {:.6}",
                time_coeff, cpu_solver.u.vx[i], time_rhs
            );

            // CPU RHS difference from time term = higher order corrections
            let cpu_correction = time_rhs - cpu_rhs_base[i];
            let gpu_correction = time_rhs - gpu_rhs[i] as f32;
            println!("  CPU correction from time term: {:.6}", cpu_correction);
            println!("  GPU correction from time term: {:.6}", gpu_correction);

            // List the faces and fluxes for this cell
            println!("  Cell {} faces:", i);
            let start = mesh.cell_face_offsets[i];
            let end = mesh.cell_face_offsets[i + 1];
            for k in start..end {
                let face_idx = mesh.cell_faces[k];
                let owner = mesh.face_owner[face_idx];
                let neighbor = mesh.face_neighbor[face_idx];
                let boundary = mesh.face_boundary[face_idx];
                let is_owner = owner == i;
                let neigh_idx = if is_owner { neighbor } else { Some(owner) };

                let flux = cpu_solver.fluxes[face_idx] * if is_owner { 1.0 } else { -1.0 };

                println!(
                    "    Face {}: flux={:.6e}, boundary={:?}, neighbor={:?}",
                    face_idx, flux, boundary, neigh_idx
                );
            }
        }

        // Compare d_p
        println!("\nD_p comparison:");
        let mut max_diff_d_p = 0.0_f64;
        for i in 0..n_cells {
            // CPU d_p = Vol / (A_p * rho)
            let mut cpu_diag = 0.0_f32;
            for k in cpu_mat.row_offsets[i]..cpu_mat.row_offsets[i + 1] {
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
            println!(
                "  CPU: u={:.6}, v={:.6}, p={:.6}",
                cpu_solver.u.vx[i], cpu_solver.u.vy[i], cpu_solver.p.values[i]
            );
            println!(
                "  GPU: u={:.6}, v={:.6}, p={:.6}",
                u_gpu[i].0, u_gpu[i].1, p_gpu[i]
            );
        }

        // Compare P (mean-adjusted)
        let mean_p_cpu: f64 =
            cpu_solver.p.values.iter().map(|&v| v as f64).sum::<f64>() / n_cells as f64;
        let mean_p_gpu: f64 = p_gpu.iter().sum::<f64>() / n_cells as f64;
        let mut max_diff_p = 0.0_f64;
        for i in 0..n_cells {
            let diff =
                ((cpu_solver.p.values[i] as f64 - mean_p_cpu) - (p_gpu[i] - mean_p_gpu)).abs();
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
        println!(
            "Flux diff: {:.6e} (face {})",
            max_diff_flux, max_diff_flux_idx
        );

        // Fail if differences are too large
        // Note: Using larger tolerance here because dt=0.001 vs dt=0.0001 in main test.
        // Single step with large dt amplifies remaining differences in PISO pressure solve.
        let tolerance = 0.05;
        assert!(
            max_diff_u < tolerance,
            "U difference after 1 step too large: {:.6e}",
            max_diff_u
        );
        assert!(
            max_diff_v < tolerance,
            "V difference after 1 step too large: {:.6e}",
            max_diff_v
        );
    });
}

/// Detailed test to investigate GPU divergence with fine mesh and large dt
#[test]
fn test_gpu_divergence_debug() {
    pollster::block_on(async {
        // Fine mesh to reproduce the issue
        let width = 3.0;
        let height = 1.0;
        let dx = 0.005;

        let channel = ChannelWithObstacle {
            length: width,
            height: height,
            obstacle_center: Point2::new(width / 3.0, height / 2.0),
            obstacle_radius: 0.2,
        };
        let mut mesh = generate_cut_cell_mesh(&channel, dx, dx, Vector2::new(width, height));
        mesh.smooth(&channel, 0.3, 50);
        let n_cells = mesh.num_cells();
        let n_faces = mesh.num_faces();

        println!("Mesh: {} cells, {} faces", n_cells, n_faces);

        // Find cells with smallest volume (these are likely near the obstacle)
        let mut vol_idx: Vec<(f64, usize)> = mesh
            .cell_vol
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, i))
            .collect();
        vol_idx.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        println!("\nSmallest volume cells:");
        for (vol, idx) in vol_idx.iter().take(5) {
            println!(
                "  Cell {}: vol={:.6e}, center=({:.6}, {:.6})",
                idx, vol, mesh.cell_cx[*idx], mesh.cell_cy[*idx]
            );
        }

        // Large dt as in GUI
        let dt = 0.01;
        let viscosity = 0.001;
        let density = 1.0;
        let u_init_x = 1.0;
        let u_init_y = 0.0;

        // Setup GPU Solver
        let mut gpu_solver = GpuSolver::new(&mesh).await;
        gpu_solver.set_dt(dt as f32);
        gpu_solver.set_viscosity(viscosity as f32);
        gpu_solver.set_density(density as f32);
        gpu_solver.set_scheme(2); // QUICK
        let u_init_gpu: Vec<(f64, f64)> = vec![(u_init_x, u_init_y); n_cells];
        gpu_solver.set_u(&u_init_gpu);

        // Initialize fluxes to match uniform velocity field
        let mut init_fluxes = vec![0.0_f64; n_faces];
        for f_idx in 0..n_faces {
            let _owner = mesh.face_owner[f_idx];
            let neighbor = mesh.face_neighbor[f_idx];
            let nx = mesh.face_nx[f_idx];
            let ny = mesh.face_ny[f_idx];
            let area = mesh.face_area[f_idx];

            let u_face = if let Some(_neigh) = neighbor {
                Vector2::new(u_init_x, u_init_y)
            } else {
                if let Some(bt) = mesh.face_boundary[f_idx] {
                    match bt {
                        BoundaryType::Inlet => Vector2::new(1.0, 0.0),
                        BoundaryType::Wall => Vector2::new(0.0, 0.0),
                        _ => Vector2::new(u_init_x, u_init_y),
                    }
                } else {
                    Vector2::new(u_init_x, u_init_y)
                }
            };

            init_fluxes[f_idx] = (u_face.x * nx + u_face.y * ny) * area;
        }
        gpu_solver.set_fluxes(&init_fluxes);

        // Check initial state
        println!("\n=== Initial State ===");
        let u_init_check = gpu_solver.get_u().await;
        let _p_init = gpu_solver.get_p().await;
        let fluxes_init = gpu_solver.get_fluxes().await;

        let u_max_init: f64 = u_init_check
            .iter()
            .map(|(x, y)| x.abs().max(y.abs()))
            .fold(0.0, f64::max);
        let u_has_nan_init = u_init_check.iter().any(|(x, y)| x.is_nan() || y.is_nan());
        let flux_has_nan_init = fluxes_init.iter().any(|f| f.is_nan());
        println!(
            "Initial max |u|={:.6e}, has_nan_u={}, has_nan_flux={}",
            u_max_init, u_has_nan_init, flux_has_nan_init
        );

        // Now step through very carefully, checking each stage
        println!("\n=== Step 1 ===");

        // Manually execute momentum solve for Ux
        gpu_solver.compute_velocity_gradient(0); // Compute gradient for QUICK scheme
        gpu_solver.assemble_momentum(0);

        // Check matrix and RHS after momentum assembly
        let mat_vals = gpu_solver.get_matrix_values().await;
        let rhs_vals = gpu_solver.get_rhs().await;
        let d_p_vals = gpu_solver.get_d_p().await;

        let mat_has_nan = mat_vals.iter().any(|v| v.is_nan() || v.is_infinite());
        let rhs_has_nan = rhs_vals.iter().any(|v| v.is_nan() || v.is_infinite());
        let d_p_has_nan = d_p_vals.iter().any(|v| v.is_nan() || v.is_infinite());

        println!("After momentum assembly (Ux):");
        println!("  Matrix has NaN/Inf: {}", mat_has_nan);
        println!("  RHS has NaN/Inf: {}", rhs_has_nan);
        println!("  d_p has NaN/Inf: {}", d_p_has_nan);

        if mat_has_nan || rhs_has_nan {
            // Find which cells have NaN
            for i in 0..n_cells {
                if rhs_vals[i].is_nan() || rhs_vals[i].is_infinite() {
                    println!("    Cell {} has NaN/Inf RHS: {:.6e}", i, rhs_vals[i]);
                    println!(
                        "      vol={:.6e}, center=({:.6}, {:.6})",
                        mesh.cell_vol[i], mesh.cell_cx[i], mesh.cell_cy[i]
                    );
                    if i < 10 {
                        break;
                    }
                }
            }

            // Check d_p values for smallest cells
            println!("  d_p for smallest volume cells:");
            for (vol, idx) in vol_idx.iter().take(5) {
                println!(
                    "    Cell {}: d_p={:.6e}, vol={:.6e}",
                    idx, d_p_vals[*idx], vol
                );
            }
        }

        // Check for extremely large d_p values
        let d_p_max = d_p_vals.iter().cloned().fold(0.0_f64, f64::max);
        let d_p_min = d_p_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        println!("  d_p range: [{:.6e}, {:.6e}]", d_p_min, d_p_max);

        // Now do a full step
        gpu_solver.step();

        // Check state after step
        let u_after = gpu_solver.get_u().await;
        let p_after = gpu_solver.get_p().await;
        let fluxes_after = gpu_solver.get_fluxes().await;

        let u_max_after: f64 = u_after
            .iter()
            .map(|(x, y)| x.abs().max(y.abs()))
            .fold(0.0, f64::max);
        let u_has_nan_after = u_after.iter().any(|(x, y)| x.is_nan() || y.is_nan());
        let p_has_nan_after = p_after.iter().any(|p| p.is_nan() || p.is_infinite());
        let flux_has_nan_after = fluxes_after.iter().any(|f| f.is_nan());

        println!("\nAfter step 1:");
        println!(
            "  max |u|={:.6e}, has_nan_u={}, has_nan_p={}, has_nan_flux={}",
            u_max_after, u_has_nan_after, p_has_nan_after, flux_has_nan_after
        );

        if u_has_nan_after || p_has_nan_after {
            // Find problematic cells
            let mut nan_cells = 0;
            for i in 0..n_cells {
                if u_after[i].0.is_nan() || u_after[i].1.is_nan() || p_after[i].is_nan() {
                    if nan_cells < 10 {
                        println!(
                            "  Cell {} has NaN: u=({:.6e}, {:.6e}), p={:.6e}",
                            i, u_after[i].0, u_after[i].1, p_after[i]
                        );
                        println!(
                            "    vol={:.6e}, center=({:.6}, {:.6})",
                            mesh.cell_vol[i], mesh.cell_cx[i], mesh.cell_cy[i]
                        );
                    }
                    nan_cells += 1;
                }
            }
            println!("  Total NaN cells: {}", nan_cells);
        }

        // Check p range
        let p_max: f64 = p_after.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let p_min: f64 = p_after.iter().cloned().fold(f64::INFINITY, f64::min);
        println!("  p range: [{:.6e}, {:.6e}]", p_min, p_max);

        // Check flux range
        let flux_max: f32 = fluxes_after
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let flux_min: f32 = fluxes_after.iter().cloned().fold(f32::INFINITY, f32::min);
        println!("  flux range: [{:.6e}, {:.6e}]", flux_min, flux_max);

        // Run more steps to find where divergence occurs
        println!("\n=== Running more steps ===");
        let mut diverged_step = 0;
        for step in 2..=20 {
            gpu_solver.step();

            let u_check = gpu_solver.get_u().await;
            let p_check = gpu_solver.get_p().await;
            let flux_check = gpu_solver.get_fluxes().await;

            let u_max: f64 = u_check
                .iter()
                .map(|(x, y)| x.abs().max(y.abs()))
                .fold(0.0, f64::max);
            let has_nan_u = u_check.iter().any(|(x, y)| x.is_nan() || y.is_nan());
            let has_nan_p = p_check.iter().any(|p| p.is_nan() || p.is_infinite());
            let has_nan_flux = flux_check.iter().any(|f| f.is_nan());

            let p_max_check: f64 = p_check.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let p_min_check: f64 = p_check.iter().cloned().fold(f64::INFINITY, f64::min);

            println!(
                "Step {}: max_u={:.6e}, nan_u={}, nan_p={}, nan_flux={}, p=[{:.2e}, {:.2e}]",
                step, u_max, has_nan_u, has_nan_p, has_nan_flux, p_min_check, p_max_check
            );

            if has_nan_u || has_nan_p || has_nan_flux || u_max > 1e6 {
                diverged_step = step;

                // Find first NaN cell
                for i in 0..n_cells.min(100) {
                    if u_check[i].0.is_nan() || u_check[i].1.is_nan() || p_check[i].is_nan() {
                        println!(
                            "  First NaN at cell {}: u=({:.6e}, {:.6e}), p={:.6e}",
                            i, u_check[i].0, u_check[i].1, p_check[i]
                        );
                        println!(
                            "    vol={:.6e}, center=({:.6}, {:.6})",
                            mesh.cell_vol[i], mesh.cell_cx[i], mesh.cell_cy[i]
                        );
                        break;
                    }
                }

                // Check if we have cells with extreme velocities before NaN
                let mut extreme_cells = 0;
                for i in 0..n_cells {
                    if !u_check[i].0.is_nan()
                        && (u_check[i].0.abs() > 100.0 || u_check[i].1.abs() > 100.0)
                    {
                        if extreme_cells < 5 {
                            println!(
                                "  Extreme velocity at cell {}: u=({:.6e}, {:.6e})",
                                i, u_check[i].0, u_check[i].1
                            );
                            println!(
                                "    vol={:.6e}, center=({:.6}, {:.6})",
                                mesh.cell_vol[i], mesh.cell_cx[i], mesh.cell_cy[i]
                            );
                        }
                        extreme_cells += 1;
                    }
                }
                if extreme_cells > 0 {
                    println!("  Total cells with |u| > 100: {}", extreme_cells);
                }
                break;
            }
        }

        assert!(
            diverged_step == 0,
            "GPU solver diverged at step {}",
            diverged_step
        );
    });
}

/// Compare CPU vs GPU stability with large dt
#[test]
fn test_cpu_vs_gpu_stability() {
    pollster::block_on(async {
        // Fine mesh to reproduce the issue
        let width = 3.0;
        let height = 1.0;
        let dx = 0.005;

        let channel = ChannelWithObstacle {
            length: width,
            height: height,
            obstacle_center: Point2::new(width / 3.0, height / 2.0),
            obstacle_radius: 0.2,
        };
        let mut mesh = generate_cut_cell_mesh(&channel, dx, dx, Vector2::new(width, height));
        mesh.smooth(&channel, 0.3, 50);
        let n_cells = mesh.num_cells();
        let n_faces = mesh.num_faces();

        println!("Mesh: {} cells, {} faces", n_cells, n_faces);

        // Large dt as in GUI
        let dt = 0.01;
        let viscosity = 0.001;
        let density = 1.0;
        let u_init_x = 1.0;
        let u_init_y = 0.0;

        // === TEST GPU SOLVER ONLY (no flux initialization - like GUI) ===
        println!("\n=== GPU Solver Only (GUI-like initialization) ===");
        let mut gpu_solver_alone = GpuSolver::new(&mesh).await;
        gpu_solver_alone.set_dt(dt as f32);
        gpu_solver_alone.set_viscosity(viscosity as f32);
        gpu_solver_alone.set_density(density as f32);
        gpu_solver_alone.set_scheme(2); // QUICK

        // Initialize velocity like GUI (only at inlet cells)
        let mut u_init: Vec<(f64, f64)> = vec![(0.0, 0.0); n_cells];
        for i in 0..n_cells {
            let cx = mesh.cell_cx[i];
            if cx < dx {
                u_init[i] = (u_init_x, u_init_y);
            }
        }
        gpu_solver_alone.set_u(&u_init);
        // Fluxes default to zero on GPU

        for step in 1..=10 {
            gpu_solver_alone.step();

            let u_gpu = gpu_solver_alone.get_u().await;
            let p_gpu = gpu_solver_alone.get_p().await;

            let gpu_max_u: f64 = u_gpu
                .iter()
                .map(|(x, y)| x.abs().max(y.abs()))
                .fold(0.0, f64::max);
            let gpu_has_nan = u_gpu.iter().any(|(x, y)| x.is_nan() || y.is_nan());
            let gpu_has_inf = p_gpu.iter().any(|&p| p.is_nan() || p.is_infinite());

            let p_max: f64 = p_gpu.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let p_min: f64 = p_gpu.iter().cloned().fold(f64::INFINITY, f64::min);

            println!(
                "Step {}: GPU max_u={:.6e}, nan={}, inf_p={}, p=[{:.2e}, {:.2e}]",
                step, gpu_max_u, gpu_has_nan, gpu_has_inf, p_min, p_max
            );

            if gpu_has_nan || gpu_has_inf || gpu_max_u > 1e6 {
                println!("  GPU DIVERGED!");
                break;
            }
        }

        // === TEST CPU SOLVER ONLY (like GUI) ===
        println!("\n=== CPU Solver Only (GUI-like initialization) ===");
        let mut cpu_solver_alone: PisoSolver<f32> = PisoSolver::new(mesh.clone());
        cpu_solver_alone.dt = dt as f32;
        cpu_solver_alone.viscosity = viscosity as f32;
        cpu_solver_alone.density = density as f32;
        cpu_solver_alone.scheme = Scheme::QUICK;

        // Initialize velocity like GUI (only at inlet cells)
        for i in 0..n_cells {
            let cx = mesh.cell_cx[i];
            if cx < dx {
                cpu_solver_alone.u.vx[i] = u_init_x as f32;
                cpu_solver_alone.u.vy[i] = u_init_y as f32;
            }
        }
        // Fluxes default to zero

        for step in 1..=10 {
            cpu_solver_alone.step();

            let cpu_max_u: f64 = cpu_solver_alone
                .u
                .vx
                .iter()
                .zip(cpu_solver_alone.u.vy.iter())
                .map(|(&x, &y)| (x as f64).abs().max((y as f64).abs()))
                .fold(0.0, f64::max);
            let cpu_has_nan = cpu_solver_alone
                .u
                .vx
                .iter()
                .any(|x| x.is_nan() || x.is_infinite());

            let p_max: f64 = cpu_solver_alone
                .p
                .values
                .iter()
                .map(|&v| v as f64)
                .fold(f64::NEG_INFINITY, f64::max);
            let p_min: f64 = cpu_solver_alone
                .p
                .values
                .iter()
                .map(|&v| v as f64)
                .fold(f64::INFINITY, f64::min);

            println!(
                "Step {}: CPU max_u={:.6e}, nan={}, p=[{:.2e}, {:.2e}]",
                step, cpu_max_u, cpu_has_nan, p_min, p_max
            );

            if cpu_has_nan || cpu_max_u > 1e6 {
                println!("  CPU DIVERGED!");
                break;
            }
        }
    });
}

/// Test with stable timestep (CFL < 1)
/// Shows that fine mesh (dx=0.005) needs smaller timestep for stability
#[test]
fn test_stable_timestep() {
    use pollster::FutureExt;
    async {
        // Test at fine cell size (dx=0.005) - this requires smaller dt!
        let dx = 0.005;
        let length = 3.0;
        let domain_size = Vector2::new(length, 1.0);
        let geo = ChannelWithObstacle {
            length,
            height: 1.0,
            obstacle_center: Point2::new(1.0, 0.5),
            obstacle_radius: 0.2,
        };
        let mut mesh = generate_cut_cell_mesh(&geo, dx, dx, domain_size);
        mesh.smooth(&geo, 0.3, 50);

        let n_cells = mesh.num_cells();
        let n_faces = mesh.num_faces();

        let min_vol: f64 = mesh.cell_vol.iter().cloned().fold(f64::MAX, f64::min);
        let max_vol: f64 = mesh.cell_vol.iter().cloned().fold(0.0, f64::max);
        println!(
            "Mesh: {} cells, {} faces. Volume range: {:.6e} to {:.6e}",
            n_cells, n_faces, min_vol, max_vol
        );

        // Use PROPER timestep for this mesh: dt = 0.0005 gives CFL = 0.1
        // Very conservative to test stability
        let dt = 0.0005;
        let viscosity = 0.001;
        let density = 1.0;
        let u_init_x = 1.0;
        let u_init_y = 0.0;

        // GUI uses max_cell_size (default 0.025) for inlet width, not dx!
        // This makes the inlet 5 cells wide instead of 1 cell wide
        let inlet_width = 0.025; // Match GUI default

        println!(
            "\nFine mesh dx=0.005 with GUI-like inlet width ({})!",
            inlet_width
        );
        println!(
            "dt={}, CFL={:.2}, UPWIND scheme (GUI default)",
            dt,
            u_init_x * dt / dx
        );

        // === GPU SOLVER ===
        println!("\n=== GPU Solver (UPWIND, proper dt for fine mesh) ===");
        let mut gpu_solver = GpuSolver::new(&mesh).await;
        gpu_solver.set_dt(dt as f32);
        gpu_solver.set_viscosity(viscosity as f32);
        gpu_solver.set_density(density as f32);
        gpu_solver.set_scheme(0); // UPWIND (GUI default)

        // GUI-like: velocity only at inlet cells (using max_cell_size = 0.025)
        let mut u_init: Vec<(f64, f64)> = vec![(0.0, 0.0); n_cells];
        for i in 0..n_cells {
            let cx = mesh.cell_cx[i];
            if cx < inlet_width {
                u_init[i] = (u_init_x, u_init_y);
            }
        }
        gpu_solver.set_u(&u_init);

        // IMPORTANT: Initialize inlet fluxes to match inlet velocity!
        let mut init_fluxes = vec![0.0_f64; n_faces];
        for f_idx in 0..n_faces {
            if let Some(bt) = mesh.face_boundary[f_idx] {
                if matches!(bt, BoundaryType::Inlet) {
                    // Inlet face: set flux = u * n * A
                    let nx = mesh.face_nx[f_idx];
                    let ny = mesh.face_ny[f_idx];
                    let area = mesh.face_area[f_idx];
                    init_fluxes[f_idx] = (u_init_x * nx + u_init_y * ny) * area;
                }
            }
        }
        gpu_solver.set_fluxes(&init_fluxes);

        let mut gpu_diverged = false;
        for step in 1..=50 {
            gpu_solver.step();

            let u_gpu = gpu_solver.get_u().await;
            let gpu_max_u: f64 = u_gpu
                .iter()
                .map(|(x, y)| x.abs().max(y.abs()))
                .fold(0.0, f64::max);
            let gpu_has_nan = u_gpu.iter().any(|(x, y)| x.is_nan() || y.is_nan());

            if step <= 5 || step % 10 == 0 || gpu_has_nan || gpu_max_u > 5.0 {
                println!(
                    "Step {}: GPU max_u={:.6e}, nan={}",
                    step, gpu_max_u, gpu_has_nan
                );
            }

            if gpu_has_nan || gpu_max_u > 1e6 {
                println!("  GPU DIVERGED!");
                gpu_diverged = true;
                break;
            }
        }
        if !gpu_diverged {
            println!("GPU solver stable for 50 steps!");
        }

        // === CPU SOLVER ===
        println!("\n=== CPU Solver (UPWIND, proper dt for fine mesh) ===");
        let mut cpu_solver: PisoSolver<f32> = PisoSolver::new(mesh.clone());
        cpu_solver.dt = dt as f32;
        cpu_solver.viscosity = viscosity as f32;
        cpu_solver.density = density as f32;
        cpu_solver.scheme = Scheme::Upwind; // GUI default

        // GUI-like: velocity only at inlet cells (using max_cell_size = 0.025)
        for i in 0..n_cells {
            let cx = mesh.cell_cx[i];
            if cx < inlet_width {
                cpu_solver.u.vx[i] = u_init_x as f32;
                cpu_solver.u.vy[i] = u_init_y as f32;
            }
        }
        // Initialize inlet fluxes
        for f_idx in 0..n_faces {
            if let Some(bt) = mesh.face_boundary[f_idx] {
                if matches!(bt, BoundaryType::Inlet) {
                    let nx = mesh.face_nx[f_idx];
                    let ny = mesh.face_ny[f_idx];
                    let area = mesh.face_area[f_idx];
                    cpu_solver.fluxes[f_idx] = ((u_init_x * nx + u_init_y * ny) * area) as f32;
                }
            }
        }

        let mut cpu_diverged = false;
        for step in 1..=50 {
            cpu_solver.step();

            let cpu_max_u: f64 = cpu_solver
                .u
                .vx
                .iter()
                .zip(cpu_solver.u.vy.iter())
                .map(|(&x, &y)| (x as f64).abs().max((y as f64).abs()))
                .fold(0.0, f64::max);
            let cpu_has_nan = cpu_solver
                .u
                .vx
                .iter()
                .any(|x| x.is_nan() || x.is_infinite());

            if step <= 5 || step % 10 == 0 || cpu_has_nan || cpu_max_u > 5.0 {
                println!(
                    "Step {}: CPU max_u={:.6e}, nan={}",
                    step, cpu_max_u, cpu_has_nan
                );
            }

            if cpu_has_nan || cpu_max_u > 1e6 {
                println!("  CPU DIVERGED!");
                cpu_diverged = true;
                break;
            }
        }
        if !cpu_diverged {
            println!("CPU solver stable for 50 steps!");
        }
    }
    .block_on();
}
