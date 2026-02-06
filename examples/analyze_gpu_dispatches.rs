/// GPU Dispatch Analysis
///
/// Analyzes the number and types of GPU dispatches per solver step
/// to identify overhead and optimization opportunities.
use cfd2::solver::gpu::dispatch_counter::{get_dispatch_stats, global_dispatch_counter};
use cfd2::solver::gpu::structs::PreconditionerType;
use cfd2::solver::gpu::unified_solver::{GpuUnifiedSolver, SolverConfig};
use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep};
use cfd2::solver::model::helpers::SolverRuntimeParamsExt;
use cfd2::solver::model::incompressible_momentum_model;
use nalgebra::Vector2;
use std::time::Duration;

/// Setup solver
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

/// Analyze dispatches for different configurations
fn analyze_dispatches() {
    println!("\n========================================");
    println!("  GPU DISPATCH ANALYSIS");
    println!("========================================\n");

    // Test different preconditioners
    let configs = [
        ("Jacobi", PreconditionerType::Jacobi),
        ("AMG", PreconditionerType::Amg),
    ];

    let cell_size = 0.02;

    for (name, precond) in &configs {
        println!("\n--- {} Preconditioner ---", name);

        let (mut solver, num_cells) = setup_solver(cell_size, *precond);

        // Collect dispatch stats over multiple steps
        let mut all_stats = Vec::new();
        let num_steps = 10;

        for _ in 0..num_steps {
            global_dispatch_counter().reset();
            global_dispatch_counter().enable();

            solver.step();

            let stats = get_dispatch_stats();
            all_stats.push(stats);
        }

        // Aggregate statistics
        let total_dispatches: u64 = all_stats.iter().map(|s| s.total_dispatches).sum();
        let avg_dispatches = total_dispatches as f64 / num_steps as f64;
        let min_dispatches = all_stats
            .iter()
            .map(|s| s.total_dispatches)
            .min()
            .unwrap_or(0);
        let max_dispatches = all_stats
            .iter()
            .map(|s| s.total_dispatches)
            .max()
            .unwrap_or(0);

        println!("  Mesh: {} cells", num_cells);
        println!("  Steps analyzed: {}", num_steps);
        println!("  Avg dispatches/step: {:.1}", avg_dispatches);
        println!("  Min dispatches: {}", min_dispatches);
        println!("  Max dispatches: {}", max_dispatches);

        // Aggregate by category
        let mut category_totals: std::collections::HashMap<&str, u64> =
            std::collections::HashMap::new();
        for stats in &all_stats {
            for (cat, count) in &stats.by_category {
                *category_totals.entry(*cat).or_insert(0) += *count;
            }
        }

        println!("\n  Dispatches by category (avg/step):");
        let mut categories: Vec<_> = category_totals.iter().collect();
        categories.sort_by(|a, b| b.1.cmp(a.1));

        for (cat, total) in categories {
            let avg = *total as f64 / num_steps as f64;
            println!("    {:<25} {:.1}", cat, avg);
        }

        // Show sample from last step
        println!("\n  Detailed breakdown (last step):");
        global_dispatch_counter().print_stats();
    }
}

/// Analyze how dispatches scale with mesh size
fn analyze_scaling() {
    println!("\n========================================");
    println!("  DISPATCH COUNT SCALING");
    println!("========================================\n");

    let cell_sizes = [0.04, 0.02, 0.01];

    println!(
        "{:>10} {:>12} {:>15} {:>20}",
        "CellSize", "Cells", "Avg Dispatches", "Notes"
    );
    println!("{}", "-".repeat(65));

    for &cell_size in &cell_sizes {
        let (mut solver, num_cells) = setup_solver(cell_size, PreconditionerType::Jacobi);

        // Collect stats
        let mut total_dispatches = 0;
        let num_steps = 5;

        for _ in 0..num_steps {
            global_dispatch_counter().reset();
            global_dispatch_counter().enable();
            solver.step();
            total_dispatches += get_dispatch_stats().total_dispatches;
        }

        let avg_dispatches = total_dispatches as f64 / num_steps as f64;

        let notes = if avg_dispatches > 100.0 {
            "High overhead (consider fusion)"
        } else if avg_dispatches > 50.0 {
            "Moderate overhead"
        } else {
            "Efficient"
        };

        println!(
            "{:>10.3} {:>12} {:>15.1} {:>20}",
            cell_size, num_cells, avg_dispatches, notes
        );
    }
    println!();
}

/// Estimate overhead from dispatch count
fn estimate_overhead() {
    println!("\n========================================");
    println!("  DISPATCH OVERHEAD ESTIMATION");
    println!("========================================\n");

    // Typical GPU dispatch overhead (approximate)
    const DISPATCH_OVERHEAD_US: f64 = 10.0; // microseconds per dispatch
    const SYNC_OVERHEAD_US: f64 = 50.0; // microseconds per submit

    let (mut solver, num_cells) = setup_solver(0.02, PreconditionerType::Jacobi);

    // Time a step
    let num_steps = 20;
    let start = std::time::Instant::now();
    for _ in 0..num_steps {
        solver.step();
    }
    let total_time = start.elapsed();
    let time_per_step = total_time / num_steps as u32;

    // Count dispatches
    global_dispatch_counter().reset();
    global_dispatch_counter().enable();
    solver.step();
    let stats = get_dispatch_stats();

    let estimated_dispatch_overhead =
        Duration::from_micros((stats.total_dispatches as f64 * DISPATCH_OVERHEAD_US) as u64);

    // Estimate submits (rough approximation)
    let estimated_submits = stats.total_dispatches / 5 + 1;
    let estimated_sync_overhead =
        Duration::from_micros((estimated_submits as f64 * SYNC_OVERHEAD_US) as u64);

    let total_overhead = estimated_dispatch_overhead + estimated_sync_overhead;
    let overhead_pct = if time_per_step.as_nanos() > 0 {
        (total_overhead.as_nanos() as f64 / time_per_step.as_nanos() as f64) * 100.0
    } else {
        0.0
    };

    println!("Mesh: {} cells", num_cells);
    println!("Dispatches per step: {}", stats.total_dispatches);
    println!("Estimated submits: {}", estimated_submits);
    println!();
    println!("Time per step: {:?}", time_per_step);
    println!(
        "Estimated dispatch overhead: {:?}",
        estimated_dispatch_overhead
    );
    println!("Estimated sync overhead: {:?}", estimated_sync_overhead);
    println!(
        "Total estimated overhead: {:?} ({:.1}%)",
        total_overhead, overhead_pct
    );
    println!();

    if overhead_pct > 30.0 {
        println!("CONCLUSION: HIGH overhead - kernel fusion recommended");
    } else if overhead_pct > 15.0 {
        println!("CONCLUSION: MODERATE overhead - consider optimization");
    } else {
        println!("CONCLUSION: Low overhead - dispatch count is efficient");
    }
    println!();
}

fn main() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║       GPU DISPATCH ANALYSIS                              ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    analyze_dispatches();
    analyze_scaling();
    estimate_overhead();

    println!("\n========================================");
    println!("  ANALYSIS COMPLETE");
    println!("========================================");
    println!("\nKey Insights:");
    println!("  - Each dispatch has ~10µs CPU-side overhead");
    println!("  - Each submit has ~50µs GPU sync overhead");
    println!("  - Target: <50 dispatches/step for efficiency");
    println!("  - Consider kernel fusion if >100 dispatches/step");
    println!();
}
