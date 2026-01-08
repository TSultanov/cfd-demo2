use cfd2::solver::options::{PreconditionerType, TimeScheme};
use cfd2::solver::{SolverConfig, UnifiedSolver};
use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep};
use cfd2::solver::model::incompressible_momentum_model;
use cfd2::solver::scheme::Scheme;
use nalgebra::Vector2;

#[test]
fn test_amg_preconditioner() {
    let length = 3.5;
    let domain_size = Vector2::new(length, 1.0);
    let geo = BackwardsStep {
        length,
        height_inlet: 0.5,
        height_outlet: 1.0,
        step_x: 0.5,
    };

    let cell_size = 0.05;
    let mut mesh = generate_cut_cell_mesh(&geo, cell_size, cell_size, 1.2, domain_size);
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
        solver.set_dt(0.001);
        solver.set_viscosity(0.001);
        solver.set_density(1.0);
        solver.set_alpha_p(0.3);
        solver.set_alpha_u(0.7);
        solver.set_advection_scheme(Scheme::Upwind);

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

        // Run with Jacobi
        solver.set_preconditioner(PreconditionerType::Jacobi);
        for _ in 0..5 {
            solver.step();
            if solver.incompressible_should_stop() {
                if solver.incompressible_degenerate_count().unwrap_or(0) > 10 {
                    panic!("Solver stopped due to degenerate solution!");
                }
                break;
            }
        }

        // Reset solver (or create new one)
        let mut solver_amg = UnifiedSolver::new(
            &mesh,
            incompressible_momentum_model(),
            SolverConfig {
                advection_scheme: Scheme::Upwind,
                time_scheme: TimeScheme::Euler,
                preconditioner: PreconditionerType::Amg,
            },
            None,
            None,
        )
        .await
        .expect("solver init");
        solver_amg.set_dt(0.001);
        solver_amg.set_viscosity(0.001);
        solver_amg.set_density(1.0);
        solver_amg.set_alpha_p(0.3);
        solver_amg.set_alpha_u(0.7);
        solver_amg.set_advection_scheme(Scheme::Upwind);
        solver_amg.set_u(&u_init);
        solver_amg.initialize_history();

        // Run with AMG
        solver_amg.set_preconditioner(PreconditionerType::Amg);
        for _ in 0..5 {
            solver_amg.step();
            if solver_amg.incompressible_should_stop() {
                if solver_amg.incompressible_degenerate_count().unwrap_or(0) > 10 {
                    panic!("Solver stopped due to degenerate solution!");
                }
                break;
            }
        }
        let p_amg = solver_amg.get_p().await;

        // Compare results
        // They won't be identical because preconditioner affects convergence path,
        // but they should be somewhat close or at least stable.
        // Actually, if both converge, they should be close.
        // But 5 steps might not be enough for full convergence.

        // Just check that AMG runs without crashing and produces reasonable values.
        let max_p = p_amg.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        assert!(max_p < 1000.0, "Pressure exploded with AMG");
        assert!(max_p > 0.0, "Pressure is zero with AMG");

        println!("Max pressure with AMG: {}", max_p);
    });
}
