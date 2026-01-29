#![cfg(all(feature = "meshgen", feature = "dev-tests"))]

use cfd2::solver::mesh::{generate_cut_cell_mesh, ChannelWithObstacle};
use cfd2::solver::model::helpers::{SolverFieldAliasesExt, SolverRuntimeParamsExt};
use cfd2::solver::model::incompressible_momentum_model;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};
use nalgebra::{Point2, Vector2};

#[test]
#[ignore]
fn test_gpu_fine_mesh_obstacle() {
    let length = 3.0;
    let height = 1.0;
    let domain_size = Vector2::new(length, height);
    let geo = ChannelWithObstacle {
        length,
        height,
        obstacle_center: Point2::new(1.0, 0.5),
        obstacle_radius: 0.2,
    };

    // Requested fine mesh size
    let min_cell_size = 0.001;
    let max_cell_size = 0.001;

    println!("Generating mesh with size {}...", min_cell_size);
    let mut mesh = generate_cut_cell_mesh(&geo, min_cell_size, max_cell_size, 1.2, domain_size);
    mesh.smooth(&geo, 0.3, 50);
    println!("Mesh generated: {} cells", mesh.num_cells());

    pollster::block_on(async {
        println!("Initializing GPU Solver...");
        let mut solver = UnifiedSolver::new(
            &mesh,
            incompressible_momentum_model(),
            SolverConfig {
                advection_scheme: Scheme::Upwind,
                time_scheme: TimeScheme::BDF2,
                preconditioner: PreconditionerType::Jacobi,
                stepping: SteppingMode::Coupled,
            },
            None,
            None,
        )
        .await
        .expect("solver init");

        let dt = 0.0001;
        let density = 1.0;
        let viscosity = 0.001;

        solver.set_dt(dt);
        solver.set_density(density).unwrap();
        solver.set_viscosity(viscosity).unwrap();

        // Init BC
        println!("Initializing BCs...");
        let mut u_init = vec![(0.0, 0.0); mesh.num_cells()];
        for i in 0..mesh.num_cells() {
            let cx = mesh.cell_cx[i];
            if cx < 0.01 {
                // Inlet
                u_init[i] = (1.0, 0.0);
            }
        }
        solver.set_u(&u_init);

        println!("Running solver steps...");
        for i in 0..10 {
            solver.step();

            if i % 1 == 0 {
                println!("Step {}", i);
                let u = solver.get_u().await;
                let max_u = u
                    .iter()
                    .fold(0.0f64, |acc, &(ux, uy)| acc.max((ux * ux + uy * uy).sqrt()));
                println!("  Max U: {}", max_u);

                if max_u.is_nan() {
                    panic!("Solver diverged: NaN detected at step {}", i);
                }
            }
        }
    });
}
