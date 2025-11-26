use cfd2::solver::mesh::{generate_cut_cell_mesh, ChannelWithObstacle};
use cfd2::solver::parallel::ParallelPisoSolver;
use nalgebra::{Point2, Vector2};

#[test]
fn test_cpu_parallel_divergence() {
    // Setup similar to UI
    let length = 3.0;
    let domain_size = Vector2::new(length, 1.0);
    let geo = ChannelWithObstacle {
        length,
        height: 1.0,
        obstacle_center: Point2::new(1.0, 0.5),
        obstacle_radius: 0.2,
    };
    // User reported 0.005. This might be slow for a test, but let's try 0.01 first to see if it triggers.
    // If not, we might need 0.005.
    let min_cell_size = 0.01;
    let max_cell_size = 0.01;

    println!("Generating mesh...");
    let mut mesh = generate_cut_cell_mesh(&geo, min_cell_size, max_cell_size, domain_size);
    mesh.smooth(&geo, 0.3, 50);
    println!("Mesh generated: {} cells", mesh.num_cells());

    // Initialize Parallel Solver
    let n_threads = 4;
    let mut solver = ParallelPisoSolver::<f64>::new(mesh.clone(), n_threads);

    // Initial Conditions
    for part in &solver.partitions {
        let mut s = part.write().unwrap();
        s.dt = 0.001;
        s.density = 1.0;
        s.viscosity = 0.01;

        for i in 0..s.mesh.num_cells() {
            let cx = s.mesh.cell_cx[i];
            if cx < 0.1 {
                s.u.vx[i] = 1.0;
                s.u.vy[i] = 0.0;
            }
        }
    }

    // Run Serial Solver for comparison
    println!("Running Serial Solver...");
    let mut serial_solver = cfd2::solver::piso::PisoSolver::<f64>::new(mesh.clone());
    serial_solver.dt = 0.001;
    serial_solver.density = 1.0;
    serial_solver.viscosity = 0.01;
    for i in 0..serial_solver.mesh.num_cells() {
        let cx = serial_solver.mesh.cell_cx[i];
        if cx < 0.1 {
            serial_solver.u.vx[i] = 1.0;
            serial_solver.u.vy[i] = 0.0;
        }
    }

    for step in 0..100 {
        serial_solver.step();
    }
    println!("Serial solver step 100 done.");

    // Run loop with adaptive timestep
    println!("Starting Parallel simulation loop...");
    let max_steps = 1000;
    let target_cfl = 0.1; // Reduced CFL

    for step in 0..max_steps {
        // Adaptive Timestep Logic
        let mut max_vel = 0.0;
        for part in &solver.partitions {
            let s = part.read().unwrap();
            for i in 0..s.u.vx.len() {
                let v = (s.u.vx[i].powi(2) + s.u.vy[i].powi(2)).sqrt();
                if v > max_vel {
                    max_vel = v;
                }
            }
        }

        if max_vel > 1e-6 {
            let next_dt = (target_cfl * min_cell_size / max_vel).clamp(1e-5, 0.1);
            // Update dt in all partitions
            for part in &solver.partitions {
                let mut s = part.write().unwrap();
                s.dt = next_dt;
            }
        }

        solver.step();

        if step % 10 == 0 {
            // Check for divergence
            let mut max_vel = 0.0;
            for part in &solver.partitions {
                let s = part.read().unwrap();
                for i in 0..s.u.vx.len() {
                    let v = (s.u.vx[i].powi(2) + s.u.vy[i].powi(2)).sqrt();
                    if v > max_vel {
                        max_vel = v;
                    }
                }
            }

            println!("Step {}: Max Vel = {:.4}", step, max_vel);

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
