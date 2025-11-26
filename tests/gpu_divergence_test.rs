use cfd2::solver::gpu::GpuSolver;
use cfd2::solver::mesh::{generate_cut_cell_mesh, ChannelWithObstacle};
use nalgebra::{Point2, Vector2};

#[test]
fn test_gpu_divergence_channel_obstacle() {
    // Setup similar to UI
    let length = 3.0;
    let domain_size = Vector2::new(length, 1.0);
    let geo = ChannelWithObstacle {
        length,
        height: 1.0,
        obstacle_center: Point2::new(1.0, 0.5),
        obstacle_radius: 0.2,
    };
    let min_cell_size = 0.025;
    let max_cell_size = 0.025;

    println!("Generating mesh...");
    let mut mesh = generate_cut_cell_mesh(&geo, min_cell_size, max_cell_size, domain_size);
    mesh.smooth(&geo, 0.3, 50);
    println!("Mesh generated: {} cells", mesh.num_cells());

    // Initialize Solver
    let timestep = 0.01;
    let density = 1.0;
    let viscosity = 0.01;

    // GPU Init
    println!("Initializing GPU solver...");
    let mut gpu_solver = pollster::block_on(GpuSolver::new(&mesh));
    gpu_solver.set_dt(timestep as f32);
    gpu_solver.set_viscosity(viscosity as f32);
    gpu_solver.set_density(density as f32);
    gpu_solver.set_scheme(0); // Upwind

    // Initial Conditions
    let mut u_init = Vec::new();
    for i in 0..mesh.num_cells() {
        let cx = mesh.cell_cx[i];
        if cx < max_cell_size {
            u_init.push((1.0, 0.0));
        } else {
            u_init.push((0.0, 0.0));
        }
    }
    gpu_solver.set_u(&u_init);

    // Run loop
    println!("Starting simulation loop...");
    let max_steps = 200;
    let target_cfl = 0.5;

    for step in 0..max_steps {
        // Adaptive Timestep Logic
        let u = pollster::block_on(gpu_solver.get_u());
        let mut max_vel = 0.0f64;
        for (vx, vy) in &u {
            let v = (vx.powi(2) + vy.powi(2)).sqrt();
            if v > max_vel {
                max_vel = v;
            }
        }

        if max_vel > 1e-6 {
            let next_dt = (target_cfl * min_cell_size / max_vel).clamp(1e-5, 0.1);
            gpu_solver.set_dt(next_dt as f32);
        }

        gpu_solver.step();

        if step % 10 == 0 {
            println!(
                "Step {}: Max Vel = {:.4}, dt = {:.6}",
                step, max_vel, gpu_solver.constants.dt
            );

            if max_vel > 20.0 || max_vel.is_nan() {
                panic!(
                    "Divergence detected at step {}! Max Vel = {}",
                    step, max_vel
                );
            }
        }
    }

    println!("Simulation completed without divergence.");
}
