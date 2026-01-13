#[path = "openfoam_reference/common.rs"]
mod common;

use cfd2::solver::gpu::helpers::SolverPlanParamsExt;
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundaryType};
use cfd2::solver::model::helpers::SolverFieldAliasesExt;
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
    solver.set_outer_iters(6);
    solver.set_u(&vec![(0.0, 0.0); mesh.num_cells()]);
    solver.set_p(&vec![0.0; mesh.num_cells()]);
    solver.initialize_history();

    // Pseudo-time stepping towards the SIMPLE steady solution.
    for _ in 0..80 {
        solver.step();
    }

    let u = pollster::block_on(solver.get_u());
    let p = pollster::block_on(solver.get_p());

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

    // Full-field comparison (all cell centers).
    let table = common::load_csv(&common::data_path("incompressible_channel_full_field.csv"));
    let x_idx = common::column_idx(&table.header, "x");
    let y_idx = common::column_idx(&table.header, "y");
    let ux_idx = common::column_idx(&table.header, "u_x");
    let uy_idx = common::column_idx(&table.header, "u_y");
    let p_idx = common::column_idx(&table.header, "p");

    assert_eq!(
        table.rows.len(),
        mesh.num_cells(),
        "reference rows must equal num_cells for full-field comparison"
    );

    let mut ref_rows: Vec<(f64, f64, f64, f64, f64)> = table
        .rows
        .iter()
        .map(|r| (r[x_idx], r[y_idx], r[ux_idx], r[uy_idx], r[p_idx]))
        .collect();
    ref_rows.sort_by_key(|r| common::yx_key(r.0, r.1));

    let mut sol_rows: Vec<(f64, f64, f64, f64, f64)> = (0..mesh.num_cells())
        .map(|i| (mesh.cell_cx[i], mesh.cell_cy[i], u[i].0, u[i].1, p[i]))
        .collect();
    sol_rows.sort_by_key(|r| common::yx_key(r.0, r.1));

    for (i, (sol, rf)) in sol_rows.iter().zip(ref_rows.iter()).enumerate() {
        let (sx, sy, _, _, _) = *sol;
        let (rx, ry, _, _, _) = *rf;
        assert!(
            (sx - rx).abs() < 1e-12,
            "x mismatch at sorted row {i}: solver={sx} ref={rx}"
        );
        assert!(
            (sy - ry).abs() < 1e-12,
            "y mismatch at sorted row {i}: solver={sy} ref={ry}"
        );
    }

    let u_x_sol: Vec<f64> = sol_rows.iter().map(|r| r.2).collect();
    let u_y_sol: Vec<f64> = sol_rows.iter().map(|r| r.3).collect();
    let p_sol: Vec<f64> = sol_rows.iter().map(|r| r.4).collect();
    let u_x_ref: Vec<f64> = ref_rows.iter().map(|r| r.2).collect();
    let u_y_ref: Vec<f64> = ref_rows.iter().map(|r| r.3).collect();
    let p_ref: Vec<f64> = ref_rows.iter().map(|r| r.4).collect();

    let u_x_err_full = common::rel_l2(&u_x_sol, &u_x_ref, 1e-12);
    let u_y_err_full = common::rel_l2(&u_y_sol, &u_y_ref, 1e-12);
    let (p_err_full, p_shift_full) = common::rel_l2_best_shift(&p_sol, &p_ref, 1e-12);

    // Diagnostics: global mass balance over inlet/outlet boundaries (using face geometry).
    let rho = 1.0f64;
    let u_inlet = (1.0f64, 0.0f64);
    let mut m_in = 0.0f64;
    let mut m_out = 0.0f64;
    for face in 0..mesh.num_faces() {
        let b = mesh.face_boundary[face];
        if b.is_none() {
            continue;
        }
        let owner = mesh.face_owner[face];
        let (mut nx, mut ny) = (mesh.face_nx[face], mesh.face_ny[face]);
        let (cx, cy) = (mesh.cell_cx[owner], mesh.cell_cy[owner]);
        let (fx, fy) = (mesh.face_cx[face], mesh.face_cy[face]);
        if (fx - cx) * nx + (fy - cy) * ny < 0.0 {
            nx = -nx;
            ny = -ny;
        }
        let area = mesh.face_area[face];

        let u_face = match b.unwrap() {
            BoundaryType::Inlet => u_inlet,
            BoundaryType::Outlet => u[owner],
            BoundaryType::Wall => (0.0, 0.0),
        };
        let flux = rho * (u_face.0 * nx + u_face.1 * ny) * area;
        match b.unwrap() {
            BoundaryType::Inlet => m_in += -flux,  // inflow magnitude
            BoundaryType::Outlet => m_out += flux, // outflow magnitude
            BoundaryType::Wall => {}
        }
    }

    assert!(
        u_x_err < 0.25
            && u_y_err < 5.0
            && u_x_err_full < 0.35
            && u_y_err_full < 10.0
            && p_err_full < 0.35,
        "mismatch vs OpenFOAM: centerline rel_l2(u_x)={u_x_err:.3} rel_l2(u_y)={u_y_err:.3} (solver [{min_u_x:.3},{max_u_x:.3}] ref [{min_u_x_ref:.3},{max_u_x_ref:.3}]); full rel_l2(u_x)={u_x_err_full:.3} rel_l2(u_y)={u_y_err_full:.3} rel_l2(p)={p_err_full:.3} (best shift {p_shift_full:.3e}); mass balance inflow={m_in:.6} outflow={m_out:.6}"
    );
}
