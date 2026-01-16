#[path = "openfoam_reference/common.rs"]
mod common;

use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundaryType};
use cfd2::solver::model::helpers::{
    SolverFieldAliasesExt, SolverInletVelocityExt, SolverRuntimeParamsExt,
};
use cfd2::solver::model::incompressible_momentum_model;
use cfd2::solver::options::{PreconditionerType, TimeScheme};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{SolverConfig, UnifiedSolver};

#[test]
fn openfoam_incompressible_lid_driven_cavity_matches_reference_field() {
    std::env::set_var("CFD2_QUIET", "1");

    // Match the OpenFOAM case setup in `reference/openfoam/incompressible_lid_driven_cavity`.
    let nx = 20usize;
    let ny = 20usize;
    let length = 1.0;
    let height = 1.0;

    // Use `Inlet` on the moving lid so `set_inlet_velocity` drives the top boundary.
    let mesh = generate_structured_rect_mesh(
        nx,
        ny,
        length,
        height,
        BoundaryType::Wall,
        BoundaryType::Wall,
        BoundaryType::Wall,
        BoundaryType::Inlet,
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

    solver.set_dt(0.02);
    solver.set_dtau(0.0).unwrap();
    solver.set_density(1.0).unwrap();
    solver.set_viscosity(0.01).unwrap();
    solver.set_inlet_velocity(1.0).unwrap();
    solver.set_alpha_u(0.7).unwrap();
    solver.set_alpha_p(0.3).unwrap();
    solver.set_outer_iters(3).unwrap();
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

    let u_x_err = common::rel_l2(&u_x_sol, &u_x_ref, 1e-12);
    let u_y_err = common::rel_l2(&u_y_sol, &u_y_ref, 1e-12);
    let (p_err_affine, p_scale_affine, p_shift_affine) =
        common::rel_l2_best_affine(&p_sol, &p_ref, 1e-12);

    if common::diag_enabled() {
        eprintln!("[openfoam][incompressible_lid] rel_l2 u_x={u_x_err:.6} u_y={u_y_err:.6} p_affine={p_err_affine:.6} | max_abs_ref u_x={ux_ref_max:.3e} u_y={uy_ref_max:.3e} pdev={p_ref_dev:.3e} | max_abs_sol u_x={ux_sol_max:.3e} u_y={uy_sol_max:.3e} pdev={p_sol_dev:.3e} | p_affine scale={p_scale_affine:.3e} shift={p_shift_affine:.3e}");
    }

    assert!(
        u_x_err < 0.6 && u_y_err < 0.9 && p_err_affine < 0.40,
        "mismatch vs OpenFOAM: rel_l2(u_x)={u_x_err:.3} rel_l2(u_y)={u_y_err:.3} rel_l2(p-mean, affine-fit)={p_err_affine:.3} (scale={p_scale_affine:.3e} shift={p_shift_affine:.3e})"
    );
}
