#[path = "openfoam_reference/common.rs"]
mod common;

use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundaryType};
use cfd2::solver::model::incompressible_momentum_model;
use cfd2::solver::options::{PreconditionerType, TimeScheme};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{SolverConfig, UnifiedSolver};

#[test]
fn openfoam_incompressible_channel_matches_reference_profile() {
    std::env::set_var("CFD2_QUIET", "1");

    let nx = 40usize;
    let ny = 20usize;
    let length = 1.0;
    let height = 0.2;

    let mesh = generate_structured_rect_mesh(
        nx,
        ny,
        length,
        height,
        BoundaryType::Inlet,
        BoundaryType::Outlet,
        BoundaryType::Wall,
        BoundaryType::Wall,
    );

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

    // Match the OpenFOAM case setup in `reference/openfoam/incompressible_channel`.
    solver.set_dt(0.02);
    solver.set_dtau(0.0);
    solver.set_density(1.0);
    solver.set_viscosity(0.01);
    solver.set_inlet_velocity(1.0);
    solver.set_ramp_time(0.0);
    solver.set_alpha_u(0.7);
    solver.set_alpha_p(0.3);
    solver
        .set_incompressible_outer_correctors(6)
        .expect("set outer correctors");
    solver.set_u(&vec![(0.0, 0.0); mesh.num_cells()]);
    solver.set_p(&vec![0.0; mesh.num_cells()]);
    solver.initialize_history();

    // Pseudo-time stepping towards the SIMPLE steady solution.
    for _ in 0..250 {
        solver.step();
    }

    let u = pollster::block_on(solver.get_u());

    // Sample at x = 0.4875 (cell center i=19 for Nx=40).
    let i_sample = 19usize;
    let mut u_x = Vec::with_capacity(ny);
    let mut u_y = Vec::with_capacity(ny);
    for j in 0..ny {
        let cell = j * nx + i_sample;
        u_x.push(u[cell].0);
        u_y.push(u[cell].1);
    }

    let table = common::load_csv(&common::data_path("incompressible_channel_centerline.csv"));
    let x_idx = common::column_idx(&table.header, "x");
    let y_idx = common::column_idx(&table.header, "y");
    let ux_idx = common::column_idx(&table.header, "u_x");
    let uy_idx = common::column_idx(&table.header, "u_y");

    assert_eq!(
        table.rows.len(),
        ny,
        "reference rows must equal ny for pointwise comparison"
    );
    for (k, row) in table.rows.iter().enumerate() {
        assert!(
            (row[x_idx] - 0.4875).abs() < 1e-12,
            "unexpected reference x at row {k}: {}",
            row[x_idx]
        );
        let y_expected = (k as f64 + 0.5) * (height / ny as f64);
        assert!(
            (row[y_idx] - y_expected).abs() < 1e-12,
            "unexpected reference y at row {k}: got {}, expected {}",
            row[y_idx],
            y_expected
        );
    }

    let u_x_ref: Vec<f64> = table.rows.iter().map(|r| r[ux_idx]).collect();
    let u_y_ref: Vec<f64> = table.rows.iter().map(|r| r[uy_idx]).collect();

    let u_x_err = common::rel_l2(&u_x, &u_x_ref, 1e-12);
    let u_y_err = common::rel_l2(&u_y, &u_y_ref, 1e-12);
    let (min_u_x, max_u_x) = u_x
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |acc, &v| {
            (acc.0.min(v), acc.1.max(v))
        });
    let (min_u_x_ref, max_u_x_ref) = u_x_ref
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |acc, &v| {
            (acc.0.min(v), acc.1.max(v))
        });

    // The code and OpenFOAM use different linear solver stacks and discretization details.
    // These tolerances are intended to catch gross regressions while allowing some drift.
    assert!(
        u_x_err < 0.25,
        "u_x profile mismatch vs OpenFOAM: rel_l2={u_x_err:.3} (solver [{min_u_x:.3},{max_u_x:.3}] ref [{min_u_x_ref:.3},{max_u_x_ref:.3}])"
    );
    assert!(
        u_y_err < 5.0,
        "u_y noise mismatch vs OpenFOAM: rel_l2={u_y_err:.3}"
    );
}
