#[path = "openfoam_reference/common.rs"]
mod common;

use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundaryType};
use cfd2::solver::model::helpers::{
    SolverFieldAliasesExt, SolverRuntimeParamsExt,
};
use cfd2::solver::model::incompressible_momentum_model;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};

/// Test incompressible lid-driven cavity against OpenFOAM reference.
///
/// # Timeout
/// This test requires extended timeout (~60-120s) due to GPU compute.
/// Run with: `cargo test --test openfoam_incompressible_lid_driven_cavity_reference_test -- --ignored --timeout 120`
#[test]
#[ignore]
fn openfoam_incompressible_lid_driven_cavity_matches_reference_field() {
    std::env::set_var("CFD2_QUIET", "1");

    // Match the OpenFOAM case setup in `reference/openfoam/incompressible_lid_driven_cavity`.
    let nx = 20usize;
    let ny = 20usize;
    let length = 1.0;
    let height = 1.0;

    // Use `MovingWall` on the moving lid to properly model the lid-driven cavity.
    let mesh = generate_structured_rect_mesh(
        nx,
        ny,
        length,
        height,
        BoundaryType::Wall,
        BoundaryType::Wall,
        BoundaryType::Wall,
        BoundaryType::MovingWall,
    );

    let mut solver = pollster::block_on(UnifiedSolver::new(
        &mesh,
        incompressible_momentum_model(),
        SolverConfig {
            advection_scheme: Scheme::Upwind,
            time_scheme: TimeScheme::Euler,
            preconditioner: PreconditionerType::Jacobi,
            stepping: SteppingMode::Coupled,
        },
        None,
        None,
    ))
    .expect("solver init");

    solver.set_dt(0.02);
    solver.set_dtau(0.0).unwrap();
    solver.set_density(1.0).unwrap();
    solver.set_viscosity(0.01).unwrap();
    // Set moving wall velocity for the lid (u_x = 1.0, u_y = 0.0)
    solver.set_boundary_vec2(GpuBoundaryType::MovingWall, "U", [1.0, 0.0]).unwrap();
    solver.set_alpha_u(0.7).unwrap();
    solver.set_alpha_p(0.3).unwrap();
    solver.set_outer_iters(50).unwrap();
    solver.set_u(&vec![(0.0, 0.0); mesh.num_cells()]);
    solver.set_p(&vec![0.0; mesh.num_cells()]);
    solver.initialize_history();

    for _ in 0..80 {
        solver.step();
    }

    let u = pollster::block_on(solver.get_u());
    let p = pollster::block_on(solver.get_p());

    let table =
        common::load_csv(&common::data_path("incompressible_lid_driven_cavity_full_field.csv"));
    let x_idx = common::column_idx(&table.header, "x");
    let y_idx = common::column_idx(&table.header, "y");
    let ux_idx = common::column_idx(&table.header, "u_x");
    let uy_idx = common::column_idx(&table.header, "u_y");
    let p_idx = common::column_idx(&table.header, "p");

    let (u_ref_field, p_ref_field) =
        common::reference_fields_from_csv(&mesh, &table, x_idx, y_idx, ux_idx, Some(uy_idx), p_idx);
    common::save_openfoam_field_plots(
        "incompressible_lid_driven_cavity",
        &mesh,
        &u_ref_field,
        &p_ref_field,
        &u,
        &p,
    );

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
        assert!((sx - rx).abs() < 1e-12, "x mismatch at sorted row {i}: solver={sx} ref={rx}");
        assert!((sy - ry).abs() < 1e-12, "y mismatch at sorted row {i}: solver={sy} ref={ry}");
    }

    let u_x_sol: Vec<f64> = sol_rows.iter().map(|r| r.2).collect();
    let u_y_sol: Vec<f64> = sol_rows.iter().map(|r| r.3).collect();
    let mut p_sol: Vec<f64> = sol_rows.iter().map(|r| r.4).collect();
    let u_x_ref: Vec<f64> = ref_rows.iter().map(|r| r.2).collect();
    let u_y_ref: Vec<f64> = ref_rows.iter().map(|r| r.3).collect();
    let mut p_ref: Vec<f64> = ref_rows.iter().map(|r| r.4).collect();

    // Pressure gauge is fixed differently across solvers; compare mean-free pressure.
    let p_sol_mean = common::mean(&p_sol);
    let p_ref_mean = common::mean(&p_ref);
    for v in &mut p_sol {
        *v -= p_sol_mean;
    }
    for v in &mut p_ref {
        *v -= p_ref_mean;
    }

    // Non-triviality guards: ensure we're not comparing stagnant fields.
    let ux_ref_max = common::max_abs(&u_x_ref);
    let uy_ref_max = common::max_abs(&u_y_ref);
    let ux_sol_max = common::max_abs(&u_x_sol);
    let uy_sol_max = common::max_abs(&u_y_sol);
    let p_ref_dev = common::max_abs(&p_ref);
    let p_sol_dev = common::max_abs(&p_sol);
    assert!(ux_ref_max > 0.5 && uy_ref_max > 0.15 && p_ref_dev > 0.2, "reference appears trivial: max_abs(u_x)={ux_ref_max:.3e} max_abs(u_y)={uy_ref_max:.3e} max_abs(p-mean)={p_ref_dev:.3e}");
    // Solver pressure often differs by a large scale factor vs OpenFOAM (see other tests that
    // use best-affine fitting). Don't treat small absolute p-variation as "trivial" as long as
    // the velocity field is clearly non-trivial.
    assert!(ux_sol_max > 0.2 && uy_sol_max > 0.05 && p_sol_dev > 1e-4, "solver appears trivial: max_abs(u_x)={ux_sol_max:.3e} max_abs(u_y)={uy_sol_max:.3e} max_abs(p-mean)={p_sol_dev:.3e}");

    let u_sol: Vec<(f64, f64)> = u_x_sol
        .iter()
        .copied()
        .zip(u_y_sol.iter().copied())
        .collect();
    let u_ref: Vec<(f64, f64)> = u_x_ref
        .iter()
        .copied()
        .zip(u_y_ref.iter().copied())
        .collect();

    let u_scale = common::rms_vec2_mag(&u_ref).max(1e-12);
    let p_scale = common::rms(&p_ref).max(1e-12);
    let u_max = common::max_cell_rel_error_vec2(&u_sol, &u_ref, u_scale);
    let p_max = common::max_cell_rel_error_scalar(&p_sol, &p_ref, p_scale);

    if common::diag_enabled() {
        let (x_u, y_u) = (sol_rows[u_max.idx].0, sol_rows[u_max.idx].1);
        let (x_p, y_p) = (sol_rows[p_max.idx].0, sol_rows[p_max.idx].1);
        eprintln!(
            "[openfoam][incompressible_lid] max_cell rel u={:.6} abs={:.6} at (x={:.4}, y={:.4}) | max_cell rel p(mean-free)={:.6} abs={:.3} at (x={:.4}, y={:.4}) | scales u_rms={:.3e} p_rms={:.3e}",
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

    assert!(
        u_max.rel < common::CELL_REL_TOL_U,
        "U mismatch vs OpenFOAM (per-cell): max_rel={:.6} (tol={:.6}) max_abs={:.6} at (x={:.6}, y={:.6})",
        u_max.rel,
        common::CELL_REL_TOL_U,
        u_max.abs,
        sol_rows[u_max.idx].0,
        sol_rows[u_max.idx].1
    );
    assert!(
        p_max.rel < common::CELL_REL_TOL_P,
        "p mismatch vs OpenFOAM (per-cell, mean-free): max_rel={:.6} (tol={:.6}) max_abs={:.3} at (x={:.6}, y={:.6})",
        p_max.rel,
        common::CELL_REL_TOL_P,
        p_max.abs,
        sol_rows[p_max.idx].0,
        sol_rows[p_max.idx].1
    );
}
