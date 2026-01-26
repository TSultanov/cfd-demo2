#[path = "openfoam_reference/common.rs"]
mod common;

use cfd2::solver::mesh::generate_structured_backwards_step_mesh;
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::model::helpers::{
    SolverCompressibleIdealGasExt, SolverCompressibleInletExt, SolverFieldAliasesExt,
    SolverRuntimeParamsExt,
};
use cfd2::solver::model::compressible_model_with_eos;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};

#[test]
#[ignore]
fn openfoam_compressible_backwards_step_matches_reference_field() {
    std::env::set_var("CFD2_QUIET", "1");

    // Match the OpenFOAM case setup in `reference/openfoam/compressible_backwards_step`.
    let nx = 30usize;
    let ny = 10usize;
    let length = 3.0;
    let height_outlet = 1.0;
    let height_inlet = 0.5;
    let step_x = 1.0;

    let mesh = generate_structured_backwards_step_mesh(nx, ny, length, height_outlet, height_inlet, step_x);

    let mut solver = pollster::block_on(UnifiedSolver::new(
        &mesh,
        compressible_model_with_eos(EosSpec::IdealGas {
            gamma: 1.4,
            gas_constant: 287.0,
            temperature: 300.0,
        }),
        SolverConfig {
            // OpenFOAM uses vanLeer reconstruction for rho/U/T with the Kurganov flux.
            advection_scheme: Scheme::SecondOrderUpwindVanLeer,
            time_scheme: TimeScheme::BDF2,
            preconditioner: PreconditionerType::Jacobi,
            stepping: SteppingMode::Implicit { outer_iters: 1 },
        },
        None,
        None,
    ))
    .expect("solver init");

    let eos = EosSpec::IdealGas {
        gamma: 1.4,
        gas_constant: 287.0,
        temperature: 300.0,
    };
    solver.set_eos(&eos).unwrap();

    // Match the OpenFOAM case setup in `reference/openfoam/compressible_backwards_step`.
    let p0 = 101325.0f32;
    let rho0 = (p0 as f64 / (287.0 * 300.0)) as f32;
    let u0 = 300.0f32;

    solver.set_dt(2e-6);
    solver.set_dtau(0.0).unwrap();
    solver.set_viscosity(1.81e-5).unwrap();
    solver.set_density(rho0).unwrap();
    solver
        .set_compressible_inlet_isothermal_x(rho0, u0, &eos)
        .unwrap();
    solver.set_outer_iters(20).unwrap();
    solver.set_uniform_state(rho0, [u0, 0.0], p0);
    solver.initialize_history();

    for _ in 0..200 {
        solver.step();
    }

    let u = pollster::block_on(solver.get_u());
    let p = pollster::block_on(solver.get_p());

    let table = common::load_csv(&common::data_path("compressible_backwards_step_full_field.csv"));
    let x_idx = common::column_idx(&table.header, "x");
    let y_idx = common::column_idx(&table.header, "y");
    let p_idx = common::column_idx(&table.header, "p");
    let ux_idx = common::column_idx(&table.header, "u_x");
    let uy_idx = common::column_idx(&table.header, "u_y");

    let (u_ref_field, p_ref_field) =
        common::reference_fields_from_csv(&mesh, &table, x_idx, y_idx, ux_idx, Some(uy_idx), p_idx);
    common::save_openfoam_field_plots(
        "compressible_backwards_step",
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
        .map(|r| (r[x_idx], r[y_idx], r[p_idx], r[ux_idx], r[uy_idx]))
        .collect();
    ref_rows.sort_by_key(|r| common::yx_key(r.0, r.1));

    let mut sol_rows: Vec<(f64, f64, f64, f64, f64)> = (0..mesh.num_cells())
        .map(|i| (mesh.cell_cx[i], mesh.cell_cy[i], p[i], u[i].0, u[i].1))
        .collect();
    sol_rows.sort_by_key(|r| common::yx_key(r.0, r.1));

    for (i, (sol, rf)) in sol_rows.iter().zip(ref_rows.iter()).enumerate() {
        let (sx, sy, _, _, _) = *sol;
        let (rx, ry, _, _, _) = *rf;
        assert!((sx - rx).abs() < 1e-12, "x mismatch at sorted row {i}: solver={sx} ref={rx}");
        assert!((sy - ry).abs() < 1e-12, "y mismatch at sorted row {i}: solver={sy} ref={ry}");
    }

    let p_sol: Vec<f64> = sol_rows.iter().map(|r| r.2).collect();
    let ux_sol: Vec<f64> = sol_rows.iter().map(|r| r.3).collect();
    let uy_sol: Vec<f64> = sol_rows.iter().map(|r| r.4).collect();
    let p_ref: Vec<f64> = ref_rows.iter().map(|r| r.2).collect();
    let ux_ref: Vec<f64> = ref_rows.iter().map(|r| r.3).collect();
    let uy_ref: Vec<f64> = ref_rows.iter().map(|r| r.4).collect();

    let u_sol: Vec<(f64, f64)> = ux_sol
        .iter()
        .copied()
        .zip(uy_sol.iter().copied())
        .collect();
    let u_ref: Vec<(f64, f64)> = ux_ref
        .iter()
        .copied()
        .zip(uy_ref.iter().copied())
        .collect();

    let u_scale = common::rms_vec2_mag(&u_ref).max(1e-12);
    let p_scale = common::rms(&p_ref).max(1e-12);
    let u_max = common::max_cell_rel_error_vec2(&u_sol, &u_ref, u_scale);
    let p_max = common::max_cell_rel_error_scalar(&p_sol, &p_ref, p_scale);

    // Non-triviality guards.
    let ux_ref_max = common::max_abs(&ux_ref);
    let uy_ref_max = common::max_abs(&uy_ref);
    let p_ref_dev = common::max_abs_centered(&p_ref);
    let ux_sol_max = common::max_abs(&ux_sol);
    let uy_sol_max = common::max_abs(&uy_sol);
    let p_sol_dev = common::max_abs_centered(&p_sol);
    assert!(ux_ref_max > 200.0 && uy_ref_max > 20.0 && p_ref_dev > 1e4, "reference appears trivial: max_abs(u_x)={ux_ref_max:.3e} max_abs(u_y)={uy_ref_max:.3e} max_abs_centered(p)={p_ref_dev:.3e}");
    assert!(ux_sol_max > 50.0 && uy_sol_max > 5.0 && p_sol_dev > 1e3, "solver appears trivial: max_abs(u_x)={ux_sol_max:.3e} max_abs(u_y)={uy_sol_max:.3e} max_abs_centered(p)={p_sol_dev:.3e}");

    if common::diag_enabled() {
        let (x_u, y_u) = (sol_rows[u_max.idx].0, sol_rows[u_max.idx].1);
        let (x_p, y_p) = (sol_rows[p_max.idx].0, sol_rows[p_max.idx].1);
        eprintln!(
            "[openfoam][compressible_backstep] max_cell rel u={:.6} abs={:.6} at (x={:.4}, y={:.4}) | max_cell rel p={:.6} abs={:.3} at (x={:.4}, y={:.4}) | scales u_rms={:.3e} p_rms={:.3e}",
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
        "p mismatch vs OpenFOAM (per-cell): max_rel={:.6} (tol={:.6}) max_abs={:.3} at (x={:.6}, y={:.6})",
        p_max.rel,
        common::CELL_REL_TOL_P,
        p_max.abs,
        sol_rows[p_max.idx].0,
        sol_rows[p_max.idx].1
    );
}
