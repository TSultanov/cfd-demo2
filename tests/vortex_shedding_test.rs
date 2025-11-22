use cfd2::solver::mesh::{generate_cut_cell_mesh, ChannelWithObstacle, Mesh};
use cfd2::solver::piso::PisoSolver;
use cfd2::solver::parallel::ParallelPisoSolver;
use cfd2::solver::fvm::Scheme;
use nalgebra::{Vector2, Point2};

#[test]
fn test_vortex_shedding_discrepancy() {
    let length = 3.0;
    let height = 1.0;
    let domain_size = Vector2::new(length, height);
    let geo = ChannelWithObstacle {
        length,
        height,
        obstacle_center: Point2::new(1.0, 0.5),
        obstacle_radius: 0.2,
    };
    
    let min_cell_size = 0.025;
    let max_cell_size = 0.025;
    
    println!("Generating mesh...");
    let mut mesh = generate_cut_cell_mesh(&geo, min_cell_size, max_cell_size, domain_size);
    mesh.smooth(0.3, 50);
    println!("Mesh generated: {} cells", mesh.num_cells());
    
    let density = 1000.0;
    let viscosity = 0.001;
    let dt = 0.01;
    let scheme = Scheme::QUICK;
    
    // Run Serial
    println!("Running Serial Solver...");
    let mut serial_solver = PisoSolver::new(mesh.clone());
    serial_solver.dt = dt;
    serial_solver.density = density;
    serial_solver.viscosity = viscosity;
    serial_solver.scheme = scheme;
    
    // Init BC
    for i in 0..serial_solver.mesh.num_cells() {
        let cx = serial_solver.mesh.cell_cx[i];
        if cx < max_cell_size {
            serial_solver.u.vx[i] = 1.0;
            serial_solver.u.vy[i] = 0.0;
        }
    }
    
    // Run until vortex shedding. 
    let steps = 1000;
    
    for i in 0..steps {
        serial_solver.step();
        if i % 100 == 0 {
            println!("Serial Step {}", i);
        }
    }
    
    let serial_u_mag: Vec<f64> = (0..serial_solver.u.vx.len())
        .map(|i| (serial_solver.u.vx[i].powi(2) + serial_solver.u.vy[i].powi(2)).sqrt())
        .collect();
    
    // Run Parallel
    println!("Running Parallel Solver...");
    let mut parallel_solver = ParallelPisoSolver::new(mesh.clone(), 4);
    
    // Init Parallel Solvers
    for partition in &parallel_solver.partitions {
        let mut solver = partition.write().unwrap();
        solver.dt = dt;
        solver.density = density;
        solver.viscosity = viscosity;
        solver.scheme = scheme;
        
        for i in 0..solver.mesh.num_cells() {
            let cx = solver.mesh.cell_cx[i];
            if cx < max_cell_size {
                solver.u.vx[i] = 1.0;
                solver.u.vy[i] = 0.0;
            }
        }
    }
    
    for i in 0..steps {
        parallel_solver.step();
        if i % 100 == 0 {
            println!("Parallel Step {}", i);
        }
    }
    
    let mut total_parallel = 0.0;
    let mut count_parallel = 0;

    for partition in &parallel_solver.partitions {
        let solver = partition.read().unwrap();
        let u_mag: Vec<f64> = (0..solver.u.vx.len())
            .map(|i| (solver.u.vx[i].powi(2) + solver.u.vy[i].powi(2)).sqrt())
            .collect();
        
        total_parallel += u_mag.iter().sum::<f64>();
        count_parallel += u_mag.len();
    }

    let avg_serial: f64 = serial_u_mag.iter().sum::<f64>() / serial_u_mag.len() as f64;
    let avg_parallel = total_parallel / count_parallel as f64;
    
    println!("Average Velocity: Serial = {}, Parallel = {}", avg_serial, avg_parallel);
}
