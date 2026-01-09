use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep};
use cfd2::solver::model::incompressible_momentum_model;
use cfd2::solver::options::{PreconditionerType, TimeScheme};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{SolverConfig, UnifiedSolver};
use nalgebra::Vector2;

#[test]
fn test_reproduce_divergence() {
    // Default UI Configuration
    // Geometry: BackwardsStep
    let length = 3.5;
    let domain_size = Vector2::new(length, 1.0);
    let geo = BackwardsStep {
        length,
        height_inlet: 0.5,
        height_outlet: 1.0,
        step_x: 0.5,
    };

    // Mesh: 0.025 min/max cell size
    let min_cell_size = 0.025;
    let max_cell_size = 0.025;
    let mut mesh = generate_cut_cell_mesh(&geo, min_cell_size, max_cell_size, 1.2, domain_size);
    mesh.smooth(&geo, 0.3, 50);

    println!("Mesh generated with {} cells", mesh.num_cells());

    // Solver Setup
    let mut solver = pollster::block_on(UnifiedSolver::new(
        &mesh,
        incompressible_momentum_model(),
        SolverConfig {
            advection_scheme: Scheme::Upwind,
            time_scheme: TimeScheme::Euler,
            preconditioner: PreconditionerType::Jacobi,
        },
        None,
        None,
    ))
    .expect("solver init");

    // Fluid: Water
    solver.set_density(1000.0);
    solver.set_viscosity(0.001);

    // Scheme: Upwind (ID 0)
    solver.set_advection_scheme(Scheme::Upwind);

    // Under-relaxation
    solver.set_alpha_u(0.7);
    solver.set_alpha_p(0.3);

    // Initial Conditions
    let n_cells = mesh.num_cells();
    let u = vec![(0.0, 0.0); n_cells];
    let p = vec![0.0; n_cells];
    solver.set_u(&u);
    solver.set_p(&p);

    // Time Stepping
    let mut dt: f32 = 0.001; // Reduced initial dt
    solver.set_dt(dt);

    let _target_cfl = 0.9;
    let actual_min_cell_size = mesh
        .cell_vol
        .iter()
        .map(|&v| v.sqrt())
        .fold(f64::INFINITY, f64::min);

    println!("Actual min cell size: {}", actual_min_cell_size);

    // Run loop
    let max_steps = 50;
    for step in 0..max_steps {
        solver.step();

        if solver.incompressible_should_stop() {
            if solver.incompressible_degenerate_count().unwrap_or(0) > 10 {
                panic!("Solver stopped due to degenerate solution!");
            }
            println!("Solver stopped early (steady state).");
            break;
        }

        {
            // Check for divergence
            let (outer_iters, outer_res_u, outer_res_p) =
                solver.incompressible_outer_stats().unwrap_or((0, 0.0, 0.0));
            let (stats_ux, _stats_uy, stats_p) = solver.incompressible_linear_stats().unwrap_or((
                Default::default(),
                Default::default(),
                Default::default(),
            ));

            println!(
                "Step {}: dt={:.2e}, ResU={:.2e}, ResP={:.2e}, LinU={} (res {:.2e}), LinP={} (res {:.2e}), OuterIters={}",
                step,
                dt,
                outer_res_u,
                outer_res_p,
                stats_ux.iterations,
                stats_ux.residual,
                stats_p.iterations,
                stats_p.residual,
                outer_iters
            );

            if outer_res_u.is_nan()
                || outer_res_p.is_nan()
                || outer_res_u > 1e10
                || outer_res_p > 1e10
            {
                panic!("Divergence detected at step {}", step);
            }
        }

        // Adaptive DT
        let u_curr = pollster::block_on(solver.get_u());
        let mut max_vel = 0.0f64;
        for (vx, vy) in &u_curr {
            let v = (vx.powi(2) + vy.powi(2)).sqrt();
            if v > max_vel {
                max_vel = v;
            }
        }

        if max_vel > 1e-6 {
            let target_cfl = 0.2;
            let max_cfl = max_vel * (dt as f64) / actual_min_cell_size;
            let mut new_dt = (dt as f64) * target_cfl / max_cfl;

            // Limit increase
            if new_dt > (dt as f64) * 1.2 {
                new_dt = (dt as f64) * 1.2;
            }

            if new_dt > 0.1 {
                new_dt = 0.1;
            }
            solver.set_dt(new_dt as f32);
            dt = new_dt as f32; // Update local dt for next iteration
        }
    }

    println!("Completed {} steps without divergence.", max_steps);
}
