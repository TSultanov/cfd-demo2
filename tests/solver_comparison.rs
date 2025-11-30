use cfd2::solver::gpu::multigrid_solver::CycleType;
use cfd2::solver::gpu::GpuSolver;
use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep};
use nalgebra::Vector2;

#[test]
fn test_coupled_solver_comparison() {
    // Use the default Backwards Step geometry from the UI
    let length = 3.5;
    let domain_size = Vector2::new(length, 1.0);
    let geo = BackwardsStep {
        length,
        height_inlet: 0.5,
        height_outlet: 1.0,
        step_x: 0.5,
    };
    let min_cell_size = 0.025;
    let max_cell_size = 0.025;

    let mut mesh = generate_cut_cell_mesh(&geo, min_cell_size, max_cell_size, domain_size);
    mesh.smooth(&geo, 0.3, 50);

    println!("Mesh created with {} cells", mesh.num_cells());

    pollster::block_on(async {
        // Baseline coupled solver (no AMG)
        let mut solver_baseline = GpuSolver::new(&mesh).await;

        // Coupled solver with AMG cycles enabled for reference comparison
        let mut solver_amg = GpuSolver::new(&mesh).await;
        solver_amg.set_amg_cycle("Ux", CycleType::VCycle);
        solver_amg.set_amg_cycle("Uy", CycleType::VCycle);
        solver_amg.set_amg_cycle("p", CycleType::WCycle);

        let init_u: Vec<(f64, f64)> = (0..mesh.num_cells()).map(|_| (0.1, 0.0)).collect();
        let init_p: Vec<f64> = vec![0.0; mesh.num_cells()];

        for solver in [&mut solver_baseline, &mut solver_amg] {
            solver.set_u(&init_u);
            solver.set_p(&init_p);
            solver.set_density(1.0);
            solver.set_viscosity(0.01);
            solver.set_alpha_u(0.9);
            solver.set_alpha_p(0.9);
            solver.set_dt(0.001);
            solver.update_constants();
        }

        const NUM_TIME_STEPS: usize = 5;
        for step_idx in 0..NUM_TIME_STEPS {
            println!(
                "Running time step {} / {} (baseline)",
                step_idx + 1,
                NUM_TIME_STEPS
            );
            solver_baseline.step();

            println!(
                "Running time step {} / {} (AMG)",
                step_idx + 1,
                NUM_TIME_STEPS
            );
            solver_amg.step();
        }

        let u_baseline = solver_baseline.get_u().await;
        let p_baseline = solver_baseline.get_p().await;
        let u_amg = solver_amg.get_u().await;
        let p_amg = solver_amg.get_p().await;

        let mut max_diff_u = 0.0f64;
        let mut max_diff_p = 0.0f64;

        for i in 0..mesh.num_cells() {
            let diff_ux = (u_baseline[i].0 - u_amg[i].0).abs();
            let diff_uy = (u_baseline[i].1 - u_amg[i].1).abs();
            let diff_p = (p_baseline[i] - p_amg[i]).abs();
            max_diff_u = max_diff_u.max(diff_ux).max(diff_uy);
            max_diff_p = max_diff_p.max(diff_p);

            assert!(u_baseline[i].0.is_finite());
            assert!(u_baseline[i].1.is_finite());
            assert!(u_amg[i].0.is_finite());
            assert!(u_amg[i].1.is_finite());
            assert!(p_baseline[i].is_finite());
            assert!(p_amg[i].is_finite());
        }

        println!("Max Coupled ΔU: {:.2e}", max_diff_u);
        println!("Max Coupled ΔP: {:.2e}", max_diff_p);

        assert!(max_diff_u.is_finite());
        assert!(max_diff_p.is_finite());
    });
}
