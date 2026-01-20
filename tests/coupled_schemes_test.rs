#![cfg(all(feature = "meshgen", feature = "dev-tests"))]

use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep};
use cfd2::solver::model::helpers::{SolverFieldAliasesExt, SolverRuntimeParamsExt};
use cfd2::solver::model::incompressible_momentum_model;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};
use nalgebra::Vector2;

async fn run_coupled_solver(
    scheme: Scheme,
    time_scheme: TimeScheme,
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
    let mut solver = UnifiedSolver::new(
        &mesh,
        incompressible_momentum_model(),
        SolverConfig {
            advection_scheme: scheme,
            time_scheme,
            preconditioner: PreconditionerType::Jacobi,
            stepping: SteppingMode::Coupled,
        },
        None,
        None,
    )
    .await
    .expect("solver init");

    // Initial Conditions
    let init_u: Vec<(f64, f64)> = (0..mesh.num_cells()).map(|_| (0.1, 0.0)).collect();
    let init_p: Vec<f64> = (0..mesh.num_cells()).map(|_| 0.0).collect();
    solver.set_u(&init_u);
    solver.set_p(&init_p);

    // Constants
    solver.set_dt(0.001);
    solver.set_density(1.0).unwrap();
    solver.set_viscosity(0.01).unwrap();
    solver.set_alpha_u(0.9).unwrap();
    solver.set_alpha_p(0.9).unwrap();
    solver.set_advection_scheme(scheme);
    solver.set_time_scheme(time_scheme);

    // Run Steps
    const NUM_STEPS: usize = 2;
    for i in 0..NUM_STEPS {
        solver.step();

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
        let (u_upwind, p_upwind) =
            run_coupled_solver(Scheme::Upwind, TimeScheme::Euler, "Upwind + Euler").await;
        assert!(u_upwind.iter().all(|(x, y)| x.is_finite() && y.is_finite()));
        assert!(p_upwind.iter().all(|x| x.is_finite()));

        // 2. SOU + Euler
        let (u_sou, p_sou) =
            run_coupled_solver(Scheme::SecondOrderUpwind, TimeScheme::Euler, "SOU + Euler").await;
        assert!(u_sou.iter().all(|(x, y)| x.is_finite() && y.is_finite()));
        assert!(p_sou.iter().all(|x| x.is_finite()));

        // 3. QUICK + Euler
        let (u_quick, p_quick) =
            run_coupled_solver(Scheme::QUICK, TimeScheme::Euler, "QUICK + Euler").await;
        assert!(u_quick.iter().all(|(x, y)| x.is_finite() && y.is_finite()));
        assert!(p_quick.iter().all(|x| x.is_finite()));

        // 4. Upwind + BDF2
        let (u_bdf2, p_bdf2) =
            run_coupled_solver(Scheme::Upwind, TimeScheme::BDF2, "Upwind + BDF2").await;
        assert!(u_bdf2.iter().all(|(x, y)| x.is_finite() && y.is_finite()));
        assert!(p_bdf2.iter().all(|x| x.is_finite()));

        println!("All coupled solver schemes passed stability check.");
    });
}
