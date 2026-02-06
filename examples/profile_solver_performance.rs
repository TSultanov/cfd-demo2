/// Solver Performance Profiling Benchmark
///
/// This benchmark uses the `profiling` feature to provide detailed breakdown
/// of GPU-CPU communication and identify performance bottlenecks.
use cfd2::solver::gpu::structs::PreconditionerType;
use cfd2::solver::gpu::unified_solver::{GpuUnifiedSolver, SolverConfig};
use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep};
use cfd2::solver::model::helpers::SolverRuntimeParamsExt;
use cfd2::solver::model::incompressible_momentum_model;
use nalgebra::Vector2;

/// Setup an incompressible solver for profiling
fn setup_incompressible_solver(cell_size: f64) -> (GpuUnifiedSolver, usize) {
    let length = 2.0;
    let domain_size = Vector2::new(length, 1.0);
    let geo = BackwardsStep {
        length,
        height_inlet: 0.5,
        height_outlet: 1.0,
        step_x: 0.5,
    };

    let mut mesh = generate_cut_cell_mesh(&geo, cell_size, cell_size, 1.2, domain_size);
    mesh.smooth(&geo, 0.3, 50);

    let num_cells = mesh.num_cells();

    let model = incompressible_momentum_model();
    let config = SolverConfig::default();
    let mut solver = pollster::block_on(GpuUnifiedSolver::new(&mesh, model, config, None, None))
        .expect("should create solver");
    solver.set_dt(0.001);
    solver.set_viscosity(0.01).unwrap();
    solver.set_density(1.0).unwrap();
    solver.set_alpha_p(0.3).unwrap();
    solver.set_alpha_u(0.7).unwrap();
    solver.set_preconditioner(PreconditionerType::Jacobi);

    solver.initialize_history();

    // Warm up
    for _ in 0..5 {
        solver.step();
    }

    (solver, num_cells)
}

/// Profile solver performance with detailed breakdown
fn profile_solver() {
    println!("\n========================================");
    println!("  SOLVER PERFORMANCE PROFILING REPORT");
    println!("========================================\n");

    // Test with different mesh sizes
    let cell_sizes = [0.04, 0.02];

    for &cell_size in &cell_sizes {
        println!("\n--- Mesh: cell_size = {:.3} ---", cell_size);

        let (mut solver, num_cells) = setup_incompressible_solver(cell_size);
        let num_faces = num_cells * 2; // Approximation for structured grid

        // Enable profiling
        solver.enable_detailed_profiling(true).unwrap();
        solver.start_profiling_session().unwrap();

        // Run multiple steps for profiling
        let num_steps = 30;
        let start = std::time::Instant::now();
        for _ in 0..num_steps {
            solver.step();
        }
        let total_time = start.elapsed();

        solver.end_profiling_session().unwrap();

        // Print profiling report
        let _ = solver.print_profiling_report();

        // Summary statistics
        let cells_per_sec = (num_cells * num_steps) as f64 / total_time.as_secs_f64();
        let faces_per_sec = (num_faces * num_steps) as f64 / total_time.as_secs_f64();
        let time_per_step = total_time / num_steps as u32;

        println!("\n--- Summary for {} cells ---", num_cells);
        println!("  Total time for {} steps: {:?}", num_steps, total_time);
        println!("  Time per step: {:?}", time_per_step);
        println!("  Throughput: {:.2} M cells/s", cells_per_sec / 1e6);
        println!("  Face throughput: {:.2} M faces/s", faces_per_sec / 1e6);
        println!();
    }
}

/// Profile different preconditioners
fn profile_preconditioners() {
    println!("\n========================================");
    println!("  PRECONDITIONER COMPARISON");
    println!("========================================\n");

    let cell_size = 0.02;
    let (mut solver_jacobi, num_cells) = setup_incompressible_solver(cell_size);

    // Create AMG solver
    let (mut solver_amg, _) = setup_incompressible_solver(cell_size);
    solver_amg.set_preconditioner(PreconditionerType::Amg);
    // Warm up AMG
    for _ in 0..5 {
        solver_amg.step();
    }

    let num_steps = 20;

    // Profile Jacobi
    println!("--- Jacobi Preconditioner ---");
    solver_jacobi.enable_detailed_profiling(true).unwrap();
    solver_jacobi.start_profiling_session().unwrap();

    let start = std::time::Instant::now();
    for _ in 0..num_steps {
        solver_jacobi.step();
    }
    let jacobi_time = start.elapsed();

    solver_jacobi.end_profiling_session().unwrap();
    let _ = solver_jacobi.print_profiling_report();

    // Profile AMG
    println!("\n--- AMG Preconditioner ---");
    solver_amg.enable_detailed_profiling(true).unwrap();
    solver_amg.start_profiling_session().unwrap();

    let start = std::time::Instant::now();
    for _ in 0..num_steps {
        solver_amg.step();
    }
    let amg_time = start.elapsed();

    solver_amg.end_profiling_session().unwrap();
    let _ = solver_amg.print_profiling_report();

    // Comparison
    println!("\n--- Preconditioner Comparison Summary ---");
    println!("  Mesh: {} cells", num_cells);
    println!("  Steps: {}", num_steps);
    println!(
        "  Jacobi: {:?} ({:?}/step)",
        jacobi_time,
        jacobi_time / num_steps as u32
    );
    println!(
        "  AMG:    {:?} ({:?}/step)",
        amg_time,
        amg_time / num_steps as u32
    );
    if amg_time < jacobi_time {
        println!(
            "  AMG is {:.2}x faster",
            jacobi_time.as_secs_f64() / amg_time.as_secs_f64()
        );
    } else {
        println!(
            "  Jacobi is {:.2}x faster",
            amg_time.as_secs_f64() / jacobi_time.as_secs_f64()
        );
    }
    println!();
}

fn main() {
    profile_solver();
    profile_preconditioners();

    println!("\n========================================");
    println!("  PROFILING COMPLETE");
    println!("========================================");
}
