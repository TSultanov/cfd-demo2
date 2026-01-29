/// GPU Dispatch Analysis
///
/// Analyzes solver performance by tracking:
/// - Number of dispatches per step
/// - Workgroup configuration
/// - GPU memory usage patterns
/// - Step timing breakdown
use cfd2::solver::gpu::structs::PreconditionerType;
use cfd2::solver::gpu::unified_solver::{GpuUnifiedSolver, SolverConfig};
use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep};
use cfd2::solver::model::incompressible_momentum_model;
use cfd2::solver::model::helpers::SolverRuntimeParamsExt;
use nalgebra::Vector2;
use std::time::{Duration, Instant};

/// Setup solver for analysis
fn setup_solver(cell_size: f64, preconditioner: PreconditionerType) -> (GpuUnifiedSolver, usize) {
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
    solver.set_preconditioner(preconditioner);

    solver.initialize_history();

    // Warm up
    for _ in 0..5 {
        solver.step();
    }

    (solver, num_cells)
}

/// Analyze step timing in detail
fn analyze_step_timing() {
    println!("\n========================================");
    println!("  STEP TIMING ANALYSIS");
    println!("========================================\n");

    let cell_sizes = [0.04, 0.02, 0.01];
    let steps_per_test = 100;

    for &cell_size in &cell_sizes {
        let (mut solver, num_cells) = setup_solver(cell_size, PreconditionerType::Jacobi);
        let num_faces = num_cells * 2; // Approximation for structured grid

        println!("--- Cell size: {:.3} ({} cells, {} faces) ---", cell_size, num_cells, num_faces);

        // Collect timing samples
        let mut samples: Vec<Duration> = Vec::with_capacity(steps_per_test);
        
        for _ in 0..steps_per_test {
            let start = Instant::now();
            solver.step();
            samples.push(start.elapsed());
        }

        // Calculate statistics
        samples.sort();
        let min = samples[0];
        let max = samples[samples.len() - 1];
        let median = samples[samples.len() / 2];
        let mean: Duration = samples.iter().sum::<Duration>() / samples.len() as u32;
        
        // Calculate standard deviation
        let mean_secs = mean.as_secs_f64();
        let variance: f64 = samples.iter()
            .map(|d| {
                let diff = d.as_secs_f64() - mean_secs;
                diff * diff
            })
            .sum::<f64>() / samples.len() as f64;
        let std_dev = Duration::from_secs_f64(variance.sqrt());

        // Throughput calculations
        let cells_per_sec = num_cells as f64 / mean.as_secs_f64();
        let faces_per_sec = num_faces as f64 / mean.as_secs_f64();

        println!("  Step timing statistics ({} samples):", steps_per_test);
        println!("    Min:    {:?}", min);
        println!("    Max:    {:?}", max);
        println!("    Median: {:?}", median);
        println!("    Mean:   {:?}", mean);
        println!("    StdDev: {:?}", std_dev);
        println!();
        println!("  Throughput:");
        println!("    Cells/s: {:.2} M", cells_per_sec / 1e6);
        println!("    Faces/s: {:.2} M", faces_per_sec / 1e6);
        println!();
    }
}

/// Compare different preconditioners
fn compare_preconditioners() {
    println!("\n========================================");
    println!("  PRECONDITIONER COMPARISON");
    println!("========================================\n");

    let cell_size = 0.02;
    let steps_per_test = 50;

    let preconditioners = [
        ("Jacobi", PreconditionerType::Jacobi),
        ("AMG", PreconditionerType::Amg),
    ];

    for (name, precond) in &preconditioners {
        let (mut solver, num_cells) = setup_solver(cell_size, *precond);

        // Collect timing samples
        let mut samples: Vec<Duration> = Vec::with_capacity(steps_per_test);
        
        for _ in 0..steps_per_test {
            let start = Instant::now();
            solver.step();
            samples.push(start.elapsed());
        }

        let total: Duration = samples.iter().sum();
        let mean = total / samples.len() as u32;
        let cells_per_sec = num_cells as f64 / mean.as_secs_f64();

        println!("--- {} Preconditioner ---", name);
        println!("  Mean step time: {:?}", mean);
        println!("  Total {} steps: {:?}", steps_per_test, total);
        println!("  Throughput: {:.2} M cells/s", cells_per_sec / 1e6);
        println!();
    }
}

/// Analyze scaling behavior
fn analyze_scaling() {
    println!("\n========================================");
    println!("  SCALING ANALYSIS");
    println!("========================================\n");

    let cell_sizes = [0.08, 0.04, 0.02, 0.01, 0.005];
    let steps_per_test = 30;

    println!("{:>10} {:>12} {:>15} {:>15} {:>15}", 
        "CellSize", "Cells", "Time/Step(ms)", "MCells/s", "Scaling");
    println!("{}", "-".repeat(75));

    let mut baseline_throughput: Option<f64> = None;

    for &cell_size in &cell_sizes {
        let (mut solver, num_cells) = setup_solver(cell_size, PreconditionerType::Jacobi);

        // Warmup
        for _ in 0..5 {
            solver.step();
        }

        // Collect timing samples
        let start = Instant::now();
        for _ in 0..steps_per_test {
            solver.step();
        }
        let total = start.elapsed();
        let mean = total / steps_per_test as u32;
        
        let cells_per_sec = num_cells as f64 / mean.as_secs_f64();
        let time_ms = mean.as_secs_f64() * 1000.0;

        // Calculate scaling efficiency vs smallest mesh
        let scaling = if let Some(base) = baseline_throughput {
            (cells_per_sec / base) * 100.0
        } else {
            baseline_throughput = Some(cells_per_sec);
            100.0
        };

        println!("{:>10.4} {:>12} {:>15.3} {:>15.2} {:>14.1}%", 
            cell_size, num_cells, time_ms, cells_per_sec / 1e6, scaling);
    }
    println!();
}

/// Memory analysis - rough estimate based on mesh size
fn analyze_memory_usage() {
    println!("\n========================================");
    println!("  MEMORY USAGE ESTIMATION");
    println!("========================================\n");

    let cell_sizes = [0.04, 0.02, 0.01];

    println!("{:>10} {:>12} {:>12} {:>15} {:>15}", 
        "CellSize", "Cells", "Faces", "State (MB)", "Total Est (MB)");
    println!("{}", "-".repeat(70));

    for &cell_size in &cell_sizes {
        let (solver, num_cells) = setup_solver(cell_size, PreconditionerType::Jacobi);
        let num_faces = num_cells * 2; // Approximation for structured grid

        // Rough estimates based on typical solver structure
        // State: 4 fields × 4 bytes × num_cells (U, V, P, and working buffers)
        let state_bytes = num_cells * 4 * 4;
        
        // Faces: fluxes, ap coefficients, etc (rough estimate)
        let face_bytes = num_faces * 4 * 4;
        
        // Linear solver working memory (estimate 10x state for FGMRES)
        let solver_bytes = state_bytes * 10;
        
        let total_bytes = state_bytes + face_bytes + solver_bytes;

        println!("{:>10.3} {:>12} {:>12} {:>15.2} {:>15.2}", 
            cell_size, 
            num_cells, 
            num_faces,
            state_bytes as f64 / 1e6,
            total_bytes as f64 / 1e6
        );
    }
    println!();
    println!("Note: These are rough estimates. Actual usage depends on:");
    println!("  - Preconditioner type (AMG uses more memory than Jacobi)");
    println!("  - Linear solver settings (GMRES iterations, basis size)");
    println!("  - Number of fields in the model");
    println!();
}

/// Main analysis
fn main() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║       GPU SOLVER PERFORMANCE ANALYSIS                    ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    analyze_step_timing();
    compare_preconditioners();
    analyze_scaling();
    analyze_memory_usage();

    println!("\n========================================");
    println!("  ANALYSIS COMPLETE");
    println!("========================================");
    println!("\nKey Metrics to Monitor:");
    println!("  1. Throughput (M cells/s) - should scale linearly with mesh size");
    println!("  2. Step time consistency - low std_dev indicates stable performance");
    println!("  3. Scaling efficiency - should stay near 100% as mesh grows");
    println!("  4. Memory usage - check against GPU limits");
    println!();
}
