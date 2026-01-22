#[path = "openfoam_reference/common.rs"]
mod common;

use cfd2::solver::mesh::generate_structured_backwards_step_mesh;
use cfd2::solver::model::helpers::{
    SolverFieldAliasesExt, SolverInletVelocityExt, SolverRuntimeParamsExt,
};
use cfd2::solver::model::incompressible_momentum_model;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};

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
    solver.set_inlet_velocity(1.0).unwrap();
    solver.set_alpha_u(0.7).unwrap();
    solver.set_alpha_p(0.3).unwrap();
    solver.set_outer_iters(3).unwrap();
    solver.set_u(&vec![(1.0, 0.0); mesh.num_cells()]);
    solver.set_p(&vec![0.0; mesh.num_cells()]);
    solver.initialize_history();

    for _ in 0..100 {
        solver.step();
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

    let u_x_err = common::rel_l2(&u_x_sol, &u_x_ref, 1e-12);
    let u_y_err = common::rel_l2(&u_y_sol, &u_y_ref, 1e-12);
    let (p_err, p_shift) = common::rel_l2_best_shift(&p_sol, &p_ref, 1e-12);
    let (p_err_affine, p_scale_affine, p_shift_affine) =
        common::rel_l2_best_affine(&p_sol, &p_ref, 1e-12);

    // Non-triviality guards.
    let ux_ref_max = common::max_abs(&u_x_ref);
    let uy_ref_max = common::max_abs(&u_y_ref);
    let ux_sol_max = common::max_abs(&u_x_sol);
    let uy_sol_max = common::max_abs(&u_y_sol);
    assert!(ux_ref_max > 0.8 && uy_ref_max > 0.05, "reference appears trivial: max_abs(u_x)={ux_ref_max:.3e} max_abs(u_y)={uy_ref_max:.3e}");
    assert!(ux_sol_max > 0.3 && uy_sol_max > 0.01, "solver appears trivial: max_abs(u_x)={ux_sol_max:.3e} max_abs(u_y)={uy_sol_max:.3e}");

    if common::diag_enabled() {
        eprintln!("[openfoam][incompressible_backstep] rel_l2 u_x={u_x_err:.6} u_y={u_y_err:.6} p_affine={p_err_affine:.6} | max_abs_ref u_x={ux_ref_max:.3e} u_y={uy_ref_max:.3e} | max_abs_sol u_x={ux_sol_max:.3e} u_y={uy_sol_max:.3e} | p_affine scale={p_scale_affine:.3e} shift={p_shift_affine:.3e} p_shift={p_shift:.3e} p_err={p_err:.6}");
    }

    assert!(
        u_x_err < 0.5 && u_y_err < 2.0 && p_err_affine < 0.21,
        "mismatch vs OpenFOAM: rel_l2(u_x)={u_x_err:.3} rel_l2(u_y)={u_y_err:.3} rel_l2(p)={p_err:.3} (best shift {p_shift:.3e}, best affine err={p_err_affine:.3} scale={p_scale_affine:.3e} shift={p_shift_affine:.3e})"
    );
}
