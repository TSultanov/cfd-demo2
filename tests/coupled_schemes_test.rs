use cfd2::solver::gpu::GpuSolver;
use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep};
use nalgebra::Vector2;

async fn run_coupled_solver(
    scheme: u32,
    time_scheme: u32,
    name: &str,
) -> (Vec<(f64, f64)>, Vec<f64>) {
    println!("Running Coupled Solver Test: {}", name);

    // Create Mesh
    let length = 3.5;
    let domain_size = Vector2::new(length, 1.0);
    let geo = BackwardsStep {
        length,
        height_inlet: 0.5,
        height_outlet: 1.0,
        step_x: 0.5,
    };
    let min_cell_size = 0.05; // Coarse mesh for speed
    let max_cell_size = 0.05;

    let mut mesh = generate_cut_cell_mesh(&geo, min_cell_size, max_cell_size, 1.2, domain_size);
    mesh.smooth(&geo, 0.3, 50);

    // Initialize Solver
    let mut solver = GpuSolver::new(&mesh).await;

    // Initial Conditions
    let init_u: Vec<(f64, f64)> = (0..mesh.num_cells()).map(|_| (0.1, 0.0)).collect();
    let init_p: Vec<f64> = (0..mesh.num_cells()).map(|_| 0.0).collect();
    solver.set_u(&init_u);
    solver.set_p(&init_p);

    // Constants
    solver.constants.dt = 0.001;
    solver.set_density(1.0);
    solver.set_viscosity(0.01);
    solver.set_alpha_u(0.9);
    solver.set_alpha_p(0.9);

    // Set Schemes
    solver.set_scheme(scheme);
    solver.set_time_scheme(time_scheme);

    solver.update_constants();

    // Run Steps
    const NUM_STEPS: usize = 2;
    for i in 0..NUM_STEPS {
        solver.step();

        if solver.should_stop {
            if solver.degenerate_count > 10 {
                panic!("Solver stopped due to degenerate solution!");
            }
            println!("Solver stopped early (steady state).");
            break;
        }

        // Check for NaN early
        let u = solver.get_u().await;
        let p = solver.get_p().await;

        let u_finite = u.iter().all(|(ux, uy)| ux.is_finite() && uy.is_finite());
        let p_finite = p.iter().all(|val| val.is_finite());

        if !u_finite || !p_finite {
            println!("Solver exploded at step {}", i + 1);
            return (u, p);
        }
    }

    (solver.get_u().await, solver.get_p().await)
}

#[test]
fn test_coupled_schemes() {
    pollster::block_on(async {
        // 1. Upwind + Euler (Baseline)
        // scheme: 0 (Upwind), time_scheme: 0 (Euler)
        let (u_upwind, p_upwind) = run_coupled_solver(0, 0, "Upwind + Euler").await;
        assert!(u_upwind.iter().all(|(x, y)| x.is_finite() && y.is_finite()));
        assert!(p_upwind.iter().all(|x| x.is_finite()));

        // 2. SOU + Euler
        // scheme: 1 (SOU), time_scheme: 0 (Euler)
        let (u_sou, p_sou) = run_coupled_solver(1, 0, "SOU + Euler").await;
        assert!(u_sou.iter().all(|(x, y)| x.is_finite() && y.is_finite()));
        assert!(p_sou.iter().all(|x| x.is_finite()));

        // 3. QUICK + Euler
        // scheme: 2 (QUICK), time_scheme: 0 (Euler)
        let (u_quick, p_quick) = run_coupled_solver(2, 0, "QUICK + Euler").await;
        assert!(u_quick.iter().all(|(x, y)| x.is_finite() && y.is_finite()));
        assert!(p_quick.iter().all(|x| x.is_finite()));

        // 4. Upwind + BDF2
        // scheme: 0 (Upwind), time_scheme: 1 (BDF2)
        let (u_bdf2, p_bdf2) = run_coupled_solver(0, 1, "Upwind + BDF2").await;
        assert!(u_bdf2.iter().all(|(x, y)| x.is_finite() && y.is_finite()));
        assert!(p_bdf2.iter().all(|x| x.is_finite()));

        println!("All coupled solver schemes passed stability check.");
    });
}
