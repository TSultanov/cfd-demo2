use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundaryType};
use cfd2::solver::model::{incompressible_momentum_generic_model, ModelPreconditionerSpec};
use cfd2::solver::options::{PreconditionerType, TimeScheme};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{SolverConfig, UnifiedSolver};

#[test]
fn gpu_generic_incompressible_schur_smoke() {
    // Keep this mesh tiny to ensure the test is fast and stable.
    // (wgpu device init dominates anyway.)
    let mesh = generate_structured_rect_mesh(
        2,
        2,
        1.0,
        1.0,
        BoundaryType::Inlet,
        BoundaryType::Outlet,
        BoundaryType::Wall,
        BoundaryType::Wall,
    );

    let model = incompressible_momentum_generic_model();
    assert!(matches!(
        model
            .linear_solver
            .expect("generic incompressible model must define linear solver")
            .preconditioner,
        ModelPreconditionerSpec::Schur { .. }
    ));

    // Model-owned Schur requires the config preconditioner to remain default/ignored.
    let config = SolverConfig {
        advection_scheme: Scheme::Upwind,
        time_scheme: TimeScheme::Euler,
        preconditioner: PreconditionerType::Jacobi,
    };

    let mut solver = pollster::block_on(UnifiedSolver::new(&mesh, model, config, None, None))
        .expect("solver init");

    // Ensure all user-tunable params are set to benign values.
    solver.set_dt(1e-2);
    solver.set_dtau(0.0);
    solver.set_density(1.0);
    solver.set_viscosity(1.0);
    solver.set_inlet_velocity(0.0);
    solver.set_alpha_u(0.7);
    solver.set_alpha_p(0.3);
    solver.set_outer_iters(1);

    let u0 = vec![(0.0f64, 0.0f64); mesh.num_cells()];
    let p0 = vec![0.0f64; mesh.num_cells()];
    solver.set_u(&u0);
    solver.set_p(&p0);
    solver.initialize_history();

    // Smoke: run a couple of steps and ensure the fields stay finite.
    for _ in 0..2 {
        solver.step();
    }

    let u = pollster::block_on(solver.get_u());
    let p = pollster::block_on(solver.get_p());
    assert_eq!(u.len(), mesh.num_cells());
    assert_eq!(p.len(), mesh.num_cells());
    assert!(u.iter().all(|(x, y)| x.is_finite() && y.is_finite()));
    assert!(p.iter().all(|v| v.is_finite()));

    // Deterministic invariant: with zero inlet velocity and zero initial state, the solver
    // should stay very close to rest. We avoid asserting absolute pressure (gauge freedom),
    // and instead assert that pressure variation across cells stays tiny.
    let max_speed = u
        .iter()
        .map(|(x, y)| (x * x + y * y).sqrt())
        .fold(0.0_f64, f64::max);
    let (min_p, max_p) = p.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |acc, v| {
        (acc.0.min(*v), acc.1.max(*v))
    });
    let p_range = max_p - min_p;

    assert!(
        max_speed < 1e-4,
        "rest-state drift: max speed {:.3e}",
        max_speed
    );
    assert!(
        p_range < 1e-4,
        "rest-state drift: pressure range {:.3e} (min {:.3e}, max {:.3e})",
        p_range,
        min_p,
        max_p
    );
}
