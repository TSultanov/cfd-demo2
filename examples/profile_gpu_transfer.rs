#[cfg(not(all(feature = "meshgen", feature = "profiling")))]
fn main() {
    eprintln!("This example requires the `meshgen` + `profiling` features.");
    eprintln!("Try one of:");
    eprintln!("  cargo run --features profiling --example profile_gpu_transfer");
    eprintln!(
        "  cargo run --no-default-features --features \"meshgen profiling\" --example profile_gpu_transfer"
    );
}

#[cfg(all(feature = "meshgen", feature = "profiling"))]
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }

    let run_all = args.iter().any(|a| a == "--all");
    let run_scaling = args.iter().any(|a| a == "--scaling");

    if run_all {
        run_transfer_profile();
        run_scaling_profile();
    } else if run_scaling {
        run_scaling_profile();
    } else {
        run_transfer_profile();
    }
}

#[cfg(all(feature = "meshgen", feature = "profiling"))]
fn print_help() {
    println!("Usage: cargo run --features profiling --example profile_gpu_transfer [--all|--scaling]");
    println!();
    println!("  (default)  Run the detailed GPU<->CPU transfer profile");
    println!("  --scaling  Run scaling analysis across mesh sizes");
    println!("  --all      Run both");
}

/// Comprehensive GPU-CPU Communication Profiling
///
/// This measures where time is spent during GPU solver execution, specifically:
/// - GPU -> CPU data transfers (buffer reads)
/// - CPU -> GPU data transfers (buffer writes)
/// - GPU synchronization waits
/// - GPU compute dispatch overhead
/// - CPU-side computation that could be offloaded to GPU
#[cfg(all(feature = "meshgen", feature = "profiling"))]
fn run_transfer_profile() {
    use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep};
    use cfd2::solver::model::helpers::{SolverFieldAliasesExt, SolverRuntimeParamsExt};
    use cfd2::solver::model::incompressible_momentum_model;
    use cfd2::solver::options::{PreconditionerType, SteppingMode, TimeScheme};
    use cfd2::solver::profiling::ProfileCategory;
    use cfd2::solver::scheme::Scheme;
    use cfd2::solver::{SolverConfig, UnifiedSolver};
    use nalgebra::Vector2;
    use std::time::Instant;

    println!("\n");
    println!("{}", "=".repeat(80));
    println!("GPU-CPU Communication Profiling");
    println!("{}", "=".repeat(80));

    let length = 3.5;
    let domain_size = Vector2::new(length, 1.0);
    let geo = BackwardsStep {
        length,
        height_inlet: 0.5,
        height_outlet: 1.0,
        step_x: 0.5,
    };

    // Use a medium-sized mesh to get meaningful timings
    let cell_size = 0.005;
    let mut mesh = generate_cut_cell_mesh(&geo, cell_size, cell_size, 1.2, domain_size);
    mesh.smooth(&geo, 0.3, 50);

    println!("\nMesh Statistics:");
    println!("  Cells: {}", mesh.num_cells());
    println!("  Faces: {}", mesh.face_owner.len());
    println!();

    pollster::block_on(async {
        let mut solver = UnifiedSolver::new(
            &mesh,
            incompressible_momentum_model(),
            SolverConfig {
                advection_scheme: Scheme::Upwind,
                time_scheme: TimeScheme::Euler,
                preconditioner: PreconditionerType::Jacobi,
                stepping: SteppingMode::Coupled,
            },
            None,
            None,
        )
        .await
        .expect("solver init");
        solver.set_dt(0.001);
        solver.set_viscosity(0.001).unwrap();
        solver.set_density(1.0).unwrap();
        solver.set_alpha_p(0.3).unwrap();
        solver.set_alpha_u(0.7).unwrap();

        // Set initial conditions - inlet velocity
        let mut u_init = vec![(0.0, 0.0); mesh.num_cells()];
        for i in 0..mesh.num_cells() {
            let cx = mesh.cell_cx[i];
            let cy = mesh.cell_cy[i];
            if cx < 0.05 && cy > 0.5 {
                u_init[i] = (1.0, 0.0);
            }
        }
        solver.set_u(&u_init);
        solver.initialize_history();

        // Enable detailed profiling
        solver
            .enable_detailed_profiling(true)
            .expect("profiling enable");
        solver.start_profiling_session().expect("profiling start");

        println!("Starting profiled solver run...\n");

        let num_steps = 5;
        let mut step_times = Vec::with_capacity(num_steps);

        for step in 0..num_steps {
            let step_start = Instant::now();
            solver.step();
            let step_duration = step_start.elapsed();

            step_times.push(step_duration);
            println!("Step {} completed in {:?}", step + 1, step_duration);
        }

        solver.end_profiling_session().expect("profiling end");

        // Print step timing summary
        println!("\n");
        println!("Step Timing Summary:");
        println!("{}", "-".repeat(40));
        let total_step_time: std::time::Duration = step_times.iter().sum();
        let avg_step_time = total_step_time / num_steps as u32;
        println!(
            "  Total time for {} steps: {:?}",
            num_steps, total_step_time
        );
        println!("  Average time per step: {:?}", avg_step_time);

        // Print detailed profiling report
        solver.print_profiling_report().expect("profiling report");

        // Additional analysis
        println!("\n");
        println!("{}", "=".repeat(80));
        println!("Analysis and Recommendations");
        println!("{}", "=".repeat(80));

        let stats = solver.get_profiling_stats().expect("profiling stats");
        let location_stats = stats.get_location_stats();

        // Find top GPU read operations
        let mut gpu_reads: Vec<_> = location_stats
            .iter()
            .filter(|(k, _)| k.contains("GPU -> CPU"))
            .collect();
        gpu_reads.sort_by(|a, b| b.1.total_time.cmp(&a.1.total_time));

        if !gpu_reads.is_empty() {
            println!("\nTop GPU -> CPU Transfers (candidates for GPU offloading):");
            println!("{}", "-".repeat(70));
            for (location, stats) in gpu_reads.iter().take(10) {
                let name = location.replace("GPU -> CPU Transfer:", "");
                println!(
                    "  {:<40} {:>8?} ({} calls, {} bytes)",
                    name, stats.total_time, stats.call_count, stats.total_bytes
                );
            }
        }

        // Find CPU computations
        let mut cpu_compute: Vec<_> = location_stats
            .iter()
            .filter(|(k, _)| k.contains("CPU Compute"))
            .collect();
        cpu_compute.sort_by(|a, b| b.1.total_time.cmp(&a.1.total_time));

        if !cpu_compute.is_empty() {
            println!("\nCPU Computations (candidates for GPU shaders):");
            println!("{}", "-".repeat(70));
            for (location, stats) in cpu_compute.iter().take(10) {
                let name = location.replace("CPU Compute:", "");
                println!(
                    "  {:<40} {:>8?} ({} calls)",
                    name, stats.total_time, stats.call_count
                );
            }
        }

        // Specific recommendations
        println!("\n");
        println!("Specific Optimization Opportunities:");
        println!("{}", "-".repeat(70));

        // Check for norm computation reads
        let norm_reads: Vec<_> = location_stats
            .iter()
            .filter(|(k, _)| k.contains("gpu_norm") || k.contains("read_partial"))
            .collect();
        if !norm_reads.is_empty() {
            let total_norm_time: std::time::Duration =
                norm_reads.iter().map(|(_, s)| s.total_time).sum();
            let total_norm_calls: u64 = norm_reads.iter().map(|(_, s)| s.call_count).sum();
            println!(
                "\n1. NORM COMPUTATION: {:?} total ({} calls)",
                total_norm_time, total_norm_calls
            );
            println!("   The norm computation reads partial sums from GPU and reduces on CPU.");
            println!("   RECOMMENDATION: Add a GPU shader for final reduction to avoid readback.");
            println!(
                "   Expected improvement: Eliminate {} GPU->CPU transfers.",
                total_norm_calls
            );
        }

        // Check for convergence check reads
        let convergence_reads: Vec<_> = location_stats
            .iter()
            .filter(|(k, _)| k.contains("convergence"))
            .collect();
        if !convergence_reads.is_empty() {
            let total_conv_time: std::time::Duration =
                convergence_reads.iter().map(|(_, s)| s.total_time).sum();
            println!("\n2. CONVERGENCE CHECKS: {:?} total", total_conv_time);
            println!("   Full field reads (U, P) for convergence checking.");
            println!("   RECOMMENDATION: Compute max-diff on GPU and read single scalar.");
            println!("   This could reduce transfer size from O(n) to O(1).");
        }

        // Check for debug reads
        let debug_reads: Vec<_> = location_stats.iter().filter(|(k, _)| k.contains("debug")).collect();
        if !debug_reads.is_empty() {
            let total_debug_time: std::time::Duration =
                debug_reads.iter().map(|(_, s)| s.total_time).sum();
            let total_debug_bytes: u64 = debug_reads.iter().map(|(_, s)| s.total_bytes).sum();
            println!(
                "\n3. DEBUG READS: {:?} total ({} bytes)",
                total_debug_time, total_debug_bytes
            );
            println!("   Debug reads (matrix, RHS, d_p) on first iteration.");
            println!("   RECOMMENDATION: Disable in production or make conditional on debug flag.");
        }

        println!("\n");
        println!("{}", "=".repeat(80));
        println!("Summary");
        println!("{}", "=".repeat(80));

        let all_stats = stats.get_all_stats();
        let gpu_read_stats = all_stats
            .iter()
            .find(|(c, _)| *c == ProfileCategory::GpuRead);
        let cpu_compute_stats = all_stats
            .iter()
            .find(|(c, _)| *c == ProfileCategory::CpuCompute);

        if let Some((_, read_stats)) = gpu_read_stats {
            println!(
                "\nTotal GPU -> CPU transfer time: {:?} ({} calls, {} MB)",
                read_stats.total_time,
                read_stats.call_count,
                read_stats.total_bytes / 1_000_000
            );
        }

        if let Some((_, cpu_stats)) = cpu_compute_stats {
            println!(
                "Total CPU computation time: {:?} ({} calls)",
                cpu_stats.total_time, cpu_stats.call_count
            );
        }

        let session_total = stats.get_session_total();
        let gpu_read_time = gpu_read_stats.map(|(_, s)| s.total_time).unwrap_or_default();
        let cpu_time = cpu_compute_stats.map(|(_, s)| s.total_time).unwrap_or_default();
        let overhead = gpu_read_time + cpu_time;
        let overhead_pct = if session_total.as_nanos() > 0 {
            (overhead.as_nanos() as f64 / session_total.as_nanos() as f64) * 100.0
        } else {
            0.0
        };

        println!(
            "\nGPU-CPU overhead (transfers + CPU compute): {:?} ({:.1}% of total)",
            overhead, overhead_pct
        );
        println!(
            "\nOptimizing these areas could improve performance by up to {:.1}%",
            overhead_pct
        );
    });
}

/// Profile with different mesh sizes to understand scaling.
#[cfg(all(feature = "meshgen", feature = "profiling"))]
fn run_scaling_profile() {
    use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep};
    use cfd2::solver::model::helpers::{SolverFieldAliasesExt, SolverRuntimeParamsExt};
    use cfd2::solver::model::incompressible_momentum_model;
    use cfd2::solver::options::{PreconditionerType, SteppingMode, TimeScheme};
    use cfd2::solver::profiling::ProfileCategory;
    use cfd2::solver::scheme::Scheme;
    use cfd2::solver::{SolverConfig, UnifiedSolver};
    use nalgebra::Vector2;

    println!("\n");
    println!("{}", "=".repeat(80));
    println!("GPU-CPU Communication Scaling Analysis");
    println!("{}", "=".repeat(80));

    let length = 3.5;
    let domain_size = Vector2::new(length, 1.0);
    let geo = BackwardsStep {
        length,
        height_inlet: 0.5,
        height_outlet: 1.0,
        step_x: 0.5,
    };

    // Test with different mesh sizes
    let cell_sizes = [0.1, 0.05, 0.025];

    for cell_size in cell_sizes {
        let mut mesh = generate_cut_cell_mesh(&geo, cell_size, cell_size, 1.2, domain_size);
        mesh.smooth(&geo, 0.3, 50);

        let num_cells = mesh.num_cells();
        let num_faces = mesh.face_owner.len();

        println!("\n{}", "-".repeat(60));
        println!(
            "Mesh: {} cells, {} faces (cell_size = {})",
            num_cells, num_faces, cell_size
        );
        println!("{}", "-".repeat(60));

        pollster::block_on(async {
            let mut solver = UnifiedSolver::new(
                &mesh,
                incompressible_momentum_model(),
                SolverConfig {
                    advection_scheme: Scheme::Upwind,
                    time_scheme: TimeScheme::Euler,
                    preconditioner: PreconditionerType::Jacobi,
                    stepping: SteppingMode::Coupled,
                },
                None,
                None,
            )
            .await
            .expect("solver init");
            solver.set_dt(0.001);
            solver.set_viscosity(0.001).unwrap();
            solver.set_density(1.0).unwrap();
            solver.set_alpha_p(0.3).unwrap();
            solver.set_alpha_u(0.7).unwrap();

            let mut u_init = vec![(0.0, 0.0); mesh.num_cells()];
            for i in 0..mesh.num_cells() {
                let cx = mesh.cell_cx[i];
                let cy = mesh.cell_cy[i];
                if cx < 0.05 && cy > 0.5 {
                    u_init[i] = (1.0, 0.0);
                }
            }
            solver.set_u(&u_init);
            solver.initialize_history();

            solver
                .enable_detailed_profiling(true)
                .expect("profiling enable");
            solver.start_profiling_session().expect("profiling start");

            let num_steps = 3;
            for _ in 0..num_steps {
                solver.step();
            }

            solver.end_profiling_session().expect("profiling end");

            let stats = solver.get_profiling_stats().expect("profiling stats");
            let all_stats = stats.get_all_stats();

            println!("\nTiming Breakdown:");
            for (category, cat_stats) in &all_stats {
                if cat_stats.call_count > 0 {
                    let pct = if stats.get_session_total().as_nanos() > 0 {
                        (cat_stats.total_time.as_nanos() as f64
                            / stats.get_session_total().as_nanos() as f64)
                            * 100.0
                    } else {
                        0.0
                    };
                    println!(
                        "  {:<25} {:>10?} ({:>5.1}%)",
                        category.name(),
                        cat_stats.total_time,
                        pct
                    );
                }
            }

            // Calculate time per cell for GPU reads
            let gpu_read_stats = all_stats
                .iter()
                .find(|(c, _)| *c == ProfileCategory::GpuRead);
            if let Some((_, read_stats)) = gpu_read_stats {
                if read_stats.call_count > 0 {
                    let time_per_byte =
                        read_stats.total_time.as_nanos() as f64 / read_stats.total_bytes as f64;
                    println!("\n  GPU Read efficiency: {:.2} ns/byte", time_per_byte);
                    println!("  Throughput: {:.1} MB/s", read_stats.throughput_mb_per_sec());
                }
            }
        });
    }
}

