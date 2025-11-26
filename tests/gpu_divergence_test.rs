use cfd2::solver::gpu::GpuSolver;
use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep};
use nalgebra::Vector2;

#[test]
fn test_gpu_divergence() {
    let length = 3.5;
    let domain_size = Vector2::new(length, 1.0);
    let geo = BackwardsStep {
        length,
        height_inlet: 0.5,
        height_outlet: 1.0,
        step_x: 0.5,
    };
    let mut mesh = generate_cut_cell_mesh(&geo, 0.1, 0.1, domain_size);
    mesh.smooth(&geo, 0.3, 50);

    pollster::block_on(async {
        let mut solver = GpuSolver::new(&mesh).await;
        solver.set_dt(0.001);
        solver.set_viscosity(0.01);
        solver.set_density(1.0);
        solver.set_alpha_p(0.5);

        // Set initial BCs (Inlet velocity)
        let mut u_init = vec![(0.0, 0.0); mesh.num_cells()];
        for i in 0..mesh.num_cells() {
            let _cx = mesh.cell_cx[i];
            let cy = mesh.cell_cy[i];
            // Initialize flow in the upper channel
            if cy > 0.5 {
                u_init[i] = (1.0, 0.0);
            }
        }
        solver.set_u(&u_init);

        println!("Starting simulation...");
        for step in 0..100 {
            solver.step();

            let u = solver.get_u().await;
            let p = solver.get_p().await;

            let mut max_u = 0.0;
            let mut max_p = 0.0;
            let mut has_nan = false;

            for (ux, uy) in &u {
                let mag = (ux.powi(2) + uy.powi(2)).sqrt();
                if mag > max_u {
                    max_u = mag;
                }
                if ux.is_nan() || uy.is_nan() {
                    has_nan = true;
                }
            }

            for &val in &p {
                if val.abs() > max_p {
                    max_p = val.abs();
                }
                if val.is_nan() {
                    has_nan = true;
                }
            }

            println!("Step {}: Max U = {:.4}, Max P = {:.4}", step, max_u, max_p);

            let stats_p = solver.stats_p.lock().unwrap();
            println!(
                "  P Solver: iter={}, res={:.6e}, conv={}",
                stats_p.iterations, stats_p.residual, stats_p.converged
            );

            if has_nan {
                panic!("Divergence detected at step {}! NaN values found.", step);
            }
            if max_u > 100.0 {
                panic!(
                    "Divergence detected at step {}! Velocity too high: {}",
                    step, max_u
                );
            }
        }
    });
}
