use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep};
use cfd2::solver::model::incompressible_momentum_model;
use cfd2::solver::options::{PreconditionerType, TimeScheme};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{SolverConfig, UnifiedSolver};
use nalgebra::Vector2;
use std::time::Instant;

#[test]
fn test_gpu_profile() {
    let length = 3.5;
    let domain_size = Vector2::new(length, 1.0);
    let geo = BackwardsStep {
        length,
        height_inlet: 0.5,
        height_outlet: 1.0,
        step_x: 0.5,
    };
    let mut mesh = generate_cut_cell_mesh(&geo, 0.02, 0.02, 1.2, domain_size);
    mesh.smooth(&geo, 0.3, 50);

    pollster::block_on(async {
        let mut solver = UnifiedSolver::new(
            &mesh,
            incompressible_momentum_model(),
            SolverConfig {
                advection_scheme: Scheme::Upwind,
                time_scheme: TimeScheme::Euler,
                preconditioner: PreconditionerType::Jacobi,
            },
            None,
            None,
        )
        .await
        .expect("solver init");
        solver.set_dt(0.01);
        solver.set_viscosity(0.001);
        solver.set_density(1000.0);
        solver.set_alpha_p(1.0);

        // Set initial BCs
        let mut u_init = vec![(0.0, 0.0); mesh.num_cells()];
        for i in 0..mesh.num_cells() {
            let cx = mesh.cell_cx[i];
            let cy = mesh.cell_cy[i];
            if cx < 0.05 && cy > 0.5 {
                u_init[i] = (1.0, 0.0);
            }
        }
        solver.set_u(&u_init);

        // Use the detailed ProfilingStats as the single source of truth
        solver
            .enable_detailed_profiling(true)
            .expect("profiling enable");
        solver.start_profiling_session().expect("profiling start");

        println!("Starting profiling...");
        let start_total = Instant::now();
        let steps = 50;

        for step in 0..steps {
            let start_step = Instant::now();
            solver.step();
            let duration = start_step.elapsed();

            if step % 10 == 0 {
                println!("Step {}: {:?}", step, duration);
            }
        }

        let total_duration = start_total.elapsed();
        solver.end_profiling_session().expect("profiling end");

        println!("Total time for {} steps: {:?}", steps, total_duration);
        println!("Average time per step: {:?}", total_duration / steps);

        // Print detailed profiling report from ProfilingStats
        let stats = solver.get_profiling_stats().expect("profiling stats");
        let session_total = stats.get_session_total();
        println!("\nProfiling Summary (ProfilingStats):");
        println!("  Session total: {:?}", session_total);
        for (category, cat_stats) in stats.get_all_stats() {
            if cat_stats.call_count > 0 {
                println!(
                    "  {:<22} total={:?} calls={} avg={:?}",
                    category.name(),
                    cat_stats.total_time,
                    cat_stats.call_count,
                    cat_stats.avg_time()
                );
            }
        }
    });
}
