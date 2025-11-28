use cfd2::solver::gpu::multigrid_solver::CycleType;
use cfd2::solver::gpu::structs::SolverType;
use cfd2::solver::gpu::GpuSolver;
use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep};
use nalgebra::Vector2;

#[test]
fn test_solver_comparison() {
    // Create the Backwards Step mesh (same as UI default)
    let length = 3.5;
    let domain_size = Vector2::new(length, 1.0);
    let geo = BackwardsStep {
        length,
        height_inlet: 0.5,
        height_outlet: 1.0,
        step_x: 0.5,
    };
    // Use slightly coarser mesh for test speed, but keep geometry same
    // UI uses 0.025, we'll use 0.05 to keep test fast (approx 1400 cells)
    // Or should I use exact same? User said "same geometry", usually implies shape.
    // But "same geometry as the default ... in the UI" might imply resolution too.
    // Let's try 0.05 first, if it's too coarse for stability we can refine.
    // Actually, let's use 0.05. 0.025 might be too slow for a quick test.
    // Wait, user said "Change the comparison test to use the same geometry as the default backwards step in the UI".
    // I will use the exact parameters from UI to be safe.
    let min_cell_size = 0.025;
    let max_cell_size = 0.025;

    let mut mesh = generate_cut_cell_mesh(&geo, min_cell_size, max_cell_size, domain_size);
    mesh.smooth(&geo, 0.3, 50);

    println!("Mesh created with {} cells", mesh.num_cells());

    pollster::block_on(async {
        // 1. Initialize PISO Solver
        let mut solver_piso = GpuSolver::new(&mesh).await;
        solver_piso.solver_type = SolverType::Piso;
        solver_piso.set_amg_cycle("Ux", CycleType::VCycle);
        solver_piso.set_amg_cycle("Uy", CycleType::VCycle);
        solver_piso.set_amg_cycle("p", CycleType::WCycle);

        // 2. Initialize Coupled Solver
        let mut solver_coupled = GpuSolver::new(&mesh).await;
        solver_coupled.solver_type = SolverType::Coupled;

        // Set identical initial conditions
        let init_u: Vec<(f64, f64)> = (0..mesh.num_cells()).map(|_| (0.1, 0.0)).collect();
        let init_p: Vec<f64> = (0..mesh.num_cells()).map(|_| 0.0).collect();

        solver_piso.set_u(&init_u);
        solver_piso.set_p(&init_p);

        solver_coupled.set_u(&init_u);
        solver_coupled.set_p(&init_p);

        // Set identical constants
        // UI uses 0.001 timestep, let's match it for stability
        solver_piso.constants.dt = 0.001;
        solver_piso.update_constants();

        solver_coupled.constants.dt = 0.001;
        solver_coupled.update_constants();

        // March forward several full time steps to ensure stability comparisons stay meaningful
        const NUM_TIME_STEPS: usize = 5;
        for step_idx in 0..NUM_TIME_STEPS {
            println!(
                "Running time step {} / {} (PISO)",
                step_idx + 1,
                NUM_TIME_STEPS
            );
            solver_piso.step();

            println!(
                "Running time step {} / {} (Coupled)",
                step_idx + 1,
                NUM_TIME_STEPS
            );
            solver_coupled.step();
        }

        // Compare results
        let u_piso = solver_piso.get_u().await;
        let p_piso = solver_piso.get_p().await;

        let u_coupled = solver_coupled.get_u().await;
        let p_coupled = solver_coupled.get_p().await;

        // Calculate differences
        let mut max_diff_u: f64 = 0.0;
        let mut max_diff_p: f64 = 0.0;

        for i in 0..mesh.num_cells() {
            let diff_ux = (u_piso[i].0 - u_coupled[i].0).abs();
            let diff_uy = (u_piso[i].1 - u_coupled[i].1).abs();
            let diff_p = (p_piso[i] - p_coupled[i]).abs();

            max_diff_u = max_diff_u.max(diff_ux).max(diff_uy);
            max_diff_p = max_diff_p.max(diff_p);
        }

        println!("Max Diff U: {:.2e}", max_diff_u);
        println!("Max Diff P: {:.2e}", max_diff_p);

        // Note: We don't expect them to be exactly identical because they are different algorithms,
        // but they should be in the same ballpark for a single step if configured similarly.
        // However, Coupled solves the full system while PISO splits it.
        // For a single step, PISO does prediction + correction(s). Coupled does full solve.
        // If PISO converges fully (many correctors), it should match Coupled.
        // But with default settings, they will differ.
        // This test mainly ensures both run without crashing and produce finite results.

        assert!(max_diff_u.is_finite());
        assert!(max_diff_p.is_finite());

        // Check that fields are not NaN
        for i in 0..mesh.num_cells() {
            assert!(u_coupled[i].0.is_finite(), "Coupled Ux at {} is NaN", i);
            assert!(u_coupled[i].1.is_finite(), "Coupled Uy at {} is NaN", i);
            assert!(p_coupled[i].is_finite(), "Coupled P at {} is NaN", i);
        }
    });
}
