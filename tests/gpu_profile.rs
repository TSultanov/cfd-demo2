use cfd2::solver::gpu::GpuSolver;
use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep};
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
    let mut mesh = generate_cut_cell_mesh(&geo, 0.05, 0.05, domain_size);
    mesh.smooth(&geo, 0.3, 50);

    pollster::block_on(async {
        let mut solver = GpuSolver::new(&mesh).await;
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

        solver.enable_profiling(true);

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
        println!("Total time for {} steps: {:?}", steps, total_duration);
        println!("Average time per step: {:?}", total_duration / steps);

        let (time_dot, time_compute, time_spmv, time_transfer) = solver.get_profiling_data();
        println!("Profiling Data:");
        println!("  Dot Product (CPU Wait): {:?}", time_dot);
        println!("  Compute Kernels (Submit): {:?}", time_compute);
        println!("  SpMV Kernels (Submit): {:?}", time_spmv);
        println!("  Transfer (Submit): {:?}", time_transfer);

        let total_profiled = time_dot + time_compute + time_spmv + time_transfer;
        println!("  Total Profiled: {:?}", total_profiled);
        // Note: total_duration includes initial setup time if not careful, but here we measure loop only?
        // No, start_total is before loop.
        // But profiling data is accumulated over the loop.
        // So it should be comparable.
        if total_duration > total_profiled {
            println!(
                "  Unaccounted (CPU Overhead): {:?}",
                total_duration - total_profiled
            );
        } else {
            println!("  Unaccounted (CPU Overhead): 0 (Profiled > Total?)");
        }
    });
}
