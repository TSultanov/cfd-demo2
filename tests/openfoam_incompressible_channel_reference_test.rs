#[path = "openfoam_reference/common.rs"]
mod common;

use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType};
use cfd2::solver::model::helpers::{
    SolverFieldAliasesExt, SolverInletVelocityExt, SolverRuntimeParamsExt,
};
use cfd2::solver::model::incompressible_momentum_model;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};

/// Test incompressible channel flow against OpenFOAM reference.
///
/// # Timeout
/// This test requires extended timeout (~60-120s) due to GPU compute.
/// Run with: `cargo test --test openfoam_incompressible_channel_reference_test -- --ignored --timeout 120`
#[test]
#[ignore]
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
        BoundarySides {
            left: BoundaryType::Inlet,
            right: BoundaryType::Outlet,
            bottom: BoundaryType::Wall,
            top: BoundaryType::Wall,
        },
    );

    let mut solver = pollster::block_on(UnifiedSolver::new(
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
    ))
    .expect("solver init");

    // Match the OpenFOAM case setup in `reference/openfoam/incompressible_channel`.
    solver.set_dt(0.02);
    solver.set_dtau(0.0).unwrap();
    solver.set_density(1.0).unwrap();
    solver.set_viscosity(0.01).unwrap();
    solver.set_inlet_velocity(1.0).unwrap();
    solver.set_alpha_u(0.7).unwrap();
    solver.set_alpha_p(0.3).unwrap();
    solver.set_outer_iters(50).unwrap();
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
    // NOTE: u_y on this centerline is ~0 in the OpenFOAM reference (O(1e-4)), so a *relative*
    // L2 error is ill-conditioned and will overreact to tiny absolute differences.
    let mut u_y_rms_err = 0.0f64;
    let mut u_y_max_abs_err = 0.0f64;
    for (&a, &b) in u_y.iter().zip(u_y_ref.iter()) {
        let d = a - b;
        u_y_rms_err += d * d;
        u_y_max_abs_err = u_y_max_abs_err.max(d.abs());
    }
    u_y_rms_err = (u_y_rms_err / (u_y.len().max(1) as f64)).sqrt();
    let (_min_u_x, max_u_x) = u_x
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |acc, &v| {
            (acc.0.min(v), acc.1.max(v))
        });
    let (_min_u_x_ref, max_u_x_ref) = u_x_ref
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |acc, &v| {
            (acc.0.min(v), acc.1.max(v))
        });

    // Non-triviality guards.
    assert!(max_u_x_ref > 0.9, "reference appears trivial: max_u_x_ref={max_u_x_ref:.3e}");
    assert!(max_u_x > 0.4, "solver appears trivial: max_u_x_sol={max_u_x:.3e}");

    // The code and OpenFOAM use different linear solver stacks and discretization details.
    // These tolerances are intended to catch gross regressions while allowing some drift.

    // Full-field comparison (all cell centers).
    let table = common::load_csv(&common::data_path("incompressible_channel_full_field.csv"));
    let x_idx = common::column_idx(&table.header, "x");
    let y_idx = common::column_idx(&table.header, "y");
    let ux_idx = common::column_idx(&table.header, "u_x");
    let uy_idx = common::column_idx(&table.header, "u_y");
    let p_idx = common::column_idx(&table.header, "p");

    let (u_ref_field, p_ref_field) =
        common::reference_fields_from_csv(&mesh, &table, x_idx, y_idx, ux_idx, Some(uy_idx), p_idx);
    common::save_openfoam_field_plots("incompressible_channel", &mesh, &u_ref_field, &p_ref_field, &u, &p);

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

    let u_sol_field: Vec<(f64, f64)> = u_x_sol
        .iter()
        .copied()
        .zip(u_y_sol.iter().copied())
        .collect();
    let u_ref_field: Vec<(f64, f64)> = u_x_ref
        .iter()
        .copied()
        .zip(u_y_ref.iter().copied())
        .collect();

    // Pressure gauge is arbitrary; compare mean-free pressure.
    let p_sol_mean = common::mean(&p_sol);
    let p_ref_mean = common::mean(&p_ref);
    let p_sol_c: Vec<f64> = p_sol.iter().map(|v| v - p_sol_mean).collect();
    let p_ref_c: Vec<f64> = p_ref.iter().map(|v| v - p_ref_mean).collect();

    let u_scale = common::rms_vec2_mag(&u_ref_field).max(1e-12);
    let p_scale = common::rms(&p_ref_c).max(1e-12);
    let u_max = common::max_cell_rel_error_vec2(&u_sol_field, &u_ref_field, u_scale);
    let p_max = common::max_cell_rel_error_scalar(&p_sol_c, &p_ref_c, p_scale);

    if common::diag_enabled() {
        let (x_u, y_u) = (sol_rows[u_max.idx].0, sol_rows[u_max.idx].1);
        let (x_p, y_p) = (sol_rows[p_max.idx].0, sol_rows[p_max.idx].1);
        eprintln!(
            "[openfoam][incompressible_channel] max_cell rel u={:.6} abs={:.6} at (x={:.4}, y={:.4}) | max_cell rel p(mean-free)={:.6} abs={:.3} at (x={:.4}, y={:.4}) | scales u_rms={:.3e} p_rms={:.3e} | centerline rel_l2 u_x={u_x_err:.6} u_y_abs_rms={u_y_rms_err:.6}",
            u_max.rel,
            u_max.abs,
            x_u,
            y_u,
            p_max.rel,
            p_max.abs,
            x_p,
            y_p,
            u_scale,
            p_scale,
        );
    }

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
            BoundaryType::Wall | BoundaryType::SlipWall | BoundaryType::MovingWall => (0.0, 0.0),
        };
        let flux = rho * (u_face.0 * nx + u_face.1 * ny) * area;
        match b.unwrap() {
            BoundaryType::Inlet => m_in += -flux,  // inflow magnitude
            BoundaryType::Outlet => m_out += flux, // outflow magnitude
            BoundaryType::Wall | BoundaryType::SlipWall | BoundaryType::MovingWall => {}
        }
    }

    assert!(
        u_max.rel < common::CELL_REL_TOL_U,
        "U mismatch vs OpenFOAM (per-cell): max_rel={:.6} (tol={:.6}) max_abs={:.6} at (x={:.6}, y={:.6}); mass balance inflow={m_in:.6} outflow={m_out:.6}",
        u_max.rel,
        common::CELL_REL_TOL_U,
        u_max.abs,
        sol_rows[u_max.idx].0,
        sol_rows[u_max.idx].1
    );
    assert!(
        p_max.rel < common::CELL_REL_TOL_P,
        "p mismatch vs OpenFOAM (per-cell, mean-free): max_rel={:.6} (tol={:.6}) max_abs={:.3} at (x={:.6}, y={:.6}); mass balance inflow={m_in:.6} outflow={m_out:.6}",
        p_max.rel,
        common::CELL_REL_TOL_P,
        p_max.abs,
        sol_rows[p_max.idx].0,
        sol_rows[p_max.idx].1
    );
}
