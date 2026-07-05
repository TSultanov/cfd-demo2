#[path = "openfoam_reference/common.rs"]
mod common;

use cfd2::solver::mesh::generate_structured_backwards_step_mesh;
use cfd2::solver::model::helpers::{
    SolverFieldAliasesExt, SolverInletVelocityExt, SolverRuntimeParamsExt,
};
use cfd2::solver::model::incompressible_momentum_model;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};

/// Test incompressible backwards step against OpenFOAM reference.
///
/// The max-cell u error sits in the first cell at the re-entrant step corner
/// and the p max just downstream in the recirculation shear layer:
/// singular-corner-adjacent formulation differences vs pimpleFoam (RC d_p
/// coefficient, missing dev2 transpose term), not a solver defect.
///
/// # Timeout
/// This test requires extended timeout (~60-120s) due to GPU compute.
/// Run with: `cargo test --test openfoam_incompressible_backwards_step_reference_test -- --ignored --timeout 120`
#[test]
#[ignore]
fn openfoam_incompressible_backwards_step_matches_reference_field() {
    std::env::set_var("CFD2_QUIET", "1");

    // Match the OpenFOAM case setup in `reference/openfoam/incompressible_backwards_step`.
    let nx = 30usize;
    let ny = 10usize;
    let length = 3.0;
    let height_outlet = 1.0;
    let height_inlet = 0.5;
    let step_x = 1.0;

    let mesh = generate_structured_backwards_step_mesh(
        nx,
        ny,
        length,
        height_outlet,
        height_inlet,
        step_x,
    );

    let mut solver = pollster::block_on(UnifiedSolver::new(
        &mesh,
        incompressible_momentum_model().expect("model"),
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

    // d_p ∝ alpha_u*dt is part of cfd2's spatial discretization, so dt changes
    // the discrete steady state; dt=0.02 matches the reference's own dt.
    solver.set_dt(0.02);
    solver.set_dtau(0.0).unwrap();
    solver.set_density(1.0).unwrap();
    solver.set_viscosity(0.01).unwrap();
    solver.set_inlet_velocity(1.0).unwrap();
    solver.set_alpha_u(0.7).unwrap();
    solver.set_alpha_p(0.3).unwrap();
    solver.set_outer_iters(5).unwrap();
    solver.set_u(&vec![(0.0, 0.0); mesh.num_cells()]);
    solver.set_p(&vec![0.0; mesh.num_cells()]);
    solver.initialize_history();


    // March to steady state: the reference is machine-converged, so the
    // transient path (dt, time scheme, outer iterations) drops out of the
    // comparison; both codes compare converged states.
    const CHECK_EVERY: usize = 25;
    const STEADY_TOL: f64 = 1e-7;
    const MAX_STEPS: usize = 3000;
    let mut prev = pollster::block_on(solver.get_u());
    let mut steps = 0usize;
    loop {
        for _ in 0..CHECK_EVERY {
            solver.step();
        }
        steps += CHECK_EVERY;
        let cur = pollster::block_on(solver.get_u());
        let delta = cur
            .iter()
            .zip(&prev)
            .map(|(a, b)| (a.0 - b.0).abs().max((a.1 - b.1).abs()))
            .fold(0.0f64, f64::max)
            / CHECK_EVERY as f64;
        prev = cur;
        if delta < STEADY_TOL {
            println!("[openfoam][incompressible_backstep] steady after {steps} steps (per-step delta {delta:.2e})");
            break;
        }
        assert!(
            steps < MAX_STEPS,
            "no steady state within {MAX_STEPS} steps (delta {delta:.2e})"
        );
    }

    let u = pollster::block_on(solver.get_u());
    let p = pollster::block_on(solver.get_p());

    let table = common::load_csv(&common::data_path(
        "incompressible_backwards_step_full_field.csv",
    ));
    let x_idx = common::column_idx(&table.header, "x");
    let y_idx = common::column_idx(&table.header, "y");
    let ux_idx = common::column_idx(&table.header, "u_x");
    let uy_idx = common::column_idx(&table.header, "u_y");
    let p_idx = common::column_idx(&table.header, "p");

    let (u_ref_field, p_ref_field) =
        common::reference_fields_from_csv(&mesh, &table, x_idx, y_idx, ux_idx, Some(uy_idx), p_idx);
    common::save_openfoam_field_plots(
        "incompressible_backwards_step",
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

    // Pressure gauge is arbitrary; compare mean-free pressure.
    let p_sol_mean = common::mean(&p_sol);
    let p_ref_mean = common::mean(&p_ref);
    let p_sol_c: Vec<f64> = p_sol.iter().map(|v| v - p_sol_mean).collect();
    let p_ref_c: Vec<f64> = p_ref.iter().map(|v| v - p_ref_mean).collect();

    let u_scale = common::rms_vec2_mag(&u_ref).max(1e-12);
    let p_scale = common::rms(&p_ref_c).max(1e-12);
    let u_max = common::max_cell_rel_error_vec2(&u_sol, &u_ref, u_scale);
    let p_max = common::max_cell_rel_error_scalar(&p_sol_c, &p_ref_c, p_scale);

    // Non-triviality guards.
    let ux_ref_max = common::max_abs(&u_x_ref);
    let uy_ref_max = common::max_abs(&u_y_ref);
    let ux_sol_max = common::max_abs(&u_x_sol);
    let uy_sol_max = common::max_abs(&u_y_sol);
    assert!(
        ux_ref_max > 0.8 && uy_ref_max > 0.05,
        "reference appears trivial: max_abs(u_x)={ux_ref_max:.3e} max_abs(u_y)={uy_ref_max:.3e}"
    );
    assert!(
        ux_sol_max > 0.3 && uy_sol_max > 0.01,
        "solver appears trivial: max_abs(u_x)={ux_sol_max:.3e} max_abs(u_y)={uy_sol_max:.3e}"
    );

    if common::diag_enabled() {
        let (x_u, y_u) = (sol_rows[u_max.idx].0, sol_rows[u_max.idx].1);
        let (x_p, y_p) = (sol_rows[p_max.idx].0, sol_rows[p_max.idx].1);
        eprintln!(
            "[openfoam][incompressible_backstep] max_cell rel u={:.6} abs={:.6} at (x={:.4}, y={:.4}) | max_cell rel p(shifted)={:.6} abs={:.3} at (x={:.4}, y={:.4}) | scales u_rms={:.3e} p_rms={:.3e}",
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
        u_max.rel < common::reference_bands("incompressible_backstep").max_cell_u,
        "U mismatch vs OpenFOAM (per-cell): max_rel={:.6} (tol={:.6}) max_abs={:.6} at (x={:.6}, y={:.6})",
        u_max.rel,
        common::reference_bands("incompressible_backstep").max_cell_u,
        u_max.abs,
        sol_rows[u_max.idx].0,
        sol_rows[u_max.idx].1
    );
    assert!(
        p_max.rel < common::reference_bands("incompressible_backstep").max_cell_p,
        "p mismatch vs OpenFOAM (per-cell, mean-free): max_rel={:.6} (tol={:.6}) max_abs={:.3} at (x={:.6}, y={:.6})",
        p_max.rel,
        common::reference_bands("incompressible_backstep").max_cell_p,
        p_max.abs,
        sol_rows[p_max.idx].0,
        sol_rows[p_max.idx].1
    );
}
