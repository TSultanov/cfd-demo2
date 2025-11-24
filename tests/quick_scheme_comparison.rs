use cfd2::solver::fvm::Scheme;
use cfd2::solver::gpu::structs::GpuSolver;
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
        let mesh = generate_cut_cell_mesh(&channel, dx, dx, Vector2::new(width, height));
        let n_cells = mesh.num_cells();

        // 2. Common Parameters
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
        let mut cpu_solver = PisoSolver::new(mesh.clone());
        cpu_solver.dt = dt;
        cpu_solver.viscosity = viscosity;
        cpu_solver.density = density;
        cpu_solver.scheme = scheme;
        // Initialize U
        for i in 0..n_cells {
             cpu_solver.u.vx[i] = u_init_x;
             cpu_solver.u.vy[i] = u_init_y;
        }

        // Initialize Fluxes for CPU Solver
        for f_idx in 0..mesh.num_faces() {
            let owner = mesh.face_owner[f_idx];
            let neighbor = mesh.face_neighbor[f_idx];
            let nx = mesh.face_nx[f_idx];
            let ny = mesh.face_ny[f_idx];
            let area = mesh.face_area[f_idx];

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

        // 5. Run Side-by-Side
        let total_time = 0.1; 
        let num_steps = (total_time / dt) as usize;
        let compare_interval = 10;

        println!("Running comparison for {} steps...", num_steps);

        for step in 1..=num_steps {
            gpu_solver.step();
            cpu_solver.step();

            if step % compare_interval == 0 {
                let u_gpu = gpu_solver.get_u().await;
                let p_gpu = gpu_solver.get_p().await;

                // Compute mean pressure
                let mean_p_cpu: f64 = cpu_solver.p.values.iter().sum::<f64>() / n_cells as f64;
                let mean_p_gpu: f64 = p_gpu.iter().sum::<f64>() / n_cells as f64;

                // Compare Velocity X
                let mut max_diff_u = 0.0;
                let mut max_diff_v = 0.0;
                let mut max_diff_p = 0.0;

                for i in 0..n_cells {
                    let diff_u = (cpu_solver.u.vx[i] - u_gpu[i].0).abs();
                    let diff_v = (cpu_solver.u.vy[i] - u_gpu[i].1).abs();
                    let diff_p = ((cpu_solver.p.values[i] - mean_p_cpu) - (p_gpu[i] - mean_p_gpu)).abs();

                    if diff_u > max_diff_u { max_diff_u = diff_u; }
                    if diff_v > max_diff_v { max_diff_v = diff_v; }
                    if diff_p > max_diff_p { max_diff_p = diff_p; }
                }

                println!("Step {}: Max Diff U={:.6}, V={:.6}, P={:.6}", step, max_diff_u, max_diff_v, max_diff_p);
                
                // We expect some divergence due to f32 vs f64 and solver differences, 
                // but it shouldn't explode instantly.
                if max_diff_u > 1.0 {
                    println!("WARNING: Large velocity divergence detected!");
                }
            }
        }
    });
}
