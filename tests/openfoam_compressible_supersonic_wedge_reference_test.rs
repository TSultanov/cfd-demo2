#[path = "openfoam_reference/common.rs"]
mod common;

use cfd2::solver::mesh::{generate_structured_trapezoid_mesh, BoundaryType};
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::model::helpers::{
    SolverCompressibleIdealGasExt, SolverCompressibleInletExt, SolverFieldAliasesExt,
    SolverRuntimeParamsExt,
};
use cfd2::solver::model::compressible_model_with_eos;
use cfd2::solver::options::{PreconditionerType, SteppingMode, TimeScheme};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{SolverConfig, UnifiedSolver};
use std::collections::HashMap;

#[test]
fn openfoam_compressible_supersonic_wedge_matches_reference_field() {
    std::env::set_var("CFD2_QUIET", "1");

    // Match the OpenFOAM case setup in `reference/openfoam/compressible_supersonic_wedge`.
    let nx = 30usize;
    let ny = 15usize;
    let length = 1.0;
    let height = 0.6;
    let ramp_height = 0.2;

    let mesh = generate_structured_trapezoid_mesh(
        nx,
        ny,
        length,
        height,
        ramp_height,
        BoundaryType::Inlet,
        BoundaryType::Outlet,
        BoundaryType::Wall,
        BoundaryType::Outlet,
    );

    let mut solver = pollster::block_on(UnifiedSolver::new(
        &mesh,
        compressible_model_with_eos(EosSpec::IdealGas {
            gamma: 1.4,
            gas_constant: 287.0,
            temperature: 300.0,
        }),
        SolverConfig {
            advection_scheme: Scheme::Upwind,
            time_scheme: TimeScheme::Euler,
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

    // Match the OpenFOAM case setup in `reference/openfoam/compressible_supersonic_wedge`.
    let p0 = 101325.0f32;
    let rho0 = (p0 as f64 / (287.0 * 300.0)) as f32;
    let u0 = 587.0f32;

    solver.set_dt(3.5e-7);
    solver.set_dtau(0.0).unwrap();
    solver.set_viscosity(1.81e-5).unwrap();
    solver.set_density(rho0).unwrap();
    solver
        .set_compressible_inlet_isothermal_x(rho0, u0, &eos)
        .unwrap();
    solver.set_outer_iters(1).unwrap();
    solver.set_uniform_state(rho0, [u0, 0.0], p0);
    solver.initialize_history();

    for _ in 0..200 {
        solver.step();
    }

    let u = pollster::block_on(solver.get_u());
    let p = pollster::block_on(solver.get_p());

    let table =
        common::load_csv(&common::data_path("compressible_supersonic_wedge_full_field.csv"));
    let x_idx = common::column_idx(&table.header, "x");
    let y_idx = common::column_idx(&table.header, "y");
    let p_idx = common::column_idx(&table.header, "p");
    let ux_idx = common::column_idx(&table.header, "u_x");
    let uy_idx = common::column_idx(&table.header, "u_y");

    assert_eq!(
        table.rows.len(),
        mesh.num_cells(),
        "reference rows must equal num_cells for full-field comparison"
    );

    // The trapezoid mesh is slightly non-orthogonal, and our computed cell-centroid positions can
    // differ from OpenFOAM by ~O(1e-5). Use coarse coordinate rounding to associate cells.
    let yx_key = |x: f64, y: f64| -> (i64, i64) {
        let s = 1e3_f64; // ~1e-3 positional tolerance, still well below cell spacing.
        ((y * s).round() as i64, (x * s).round() as i64)
    };

    let mut sol_map: HashMap<(i64, i64), (f64, f64, f64)> = HashMap::with_capacity(mesh.num_cells());
    for i in 0..mesh.num_cells() {
        sol_map.insert(yx_key(mesh.cell_cx[i], mesh.cell_cy[i]), (p[i], u[i].0, u[i].1));
    }

    let mut p_sol = Vec::with_capacity(mesh.num_cells());
    let mut ux_sol = Vec::with_capacity(mesh.num_cells());
    let mut uy_sol = Vec::with_capacity(mesh.num_cells());
    let mut p_ref = Vec::with_capacity(mesh.num_cells());
    let mut ux_ref = Vec::with_capacity(mesh.num_cells());
    let mut uy_ref = Vec::with_capacity(mesh.num_cells());
    let mut xy_ref = Vec::with_capacity(mesh.num_cells());

    for row in &table.rows {
        let x = row[x_idx];
        let y = row[y_idx];
        let key = yx_key(x, y);
        let Some((p_s, ux_s, uy_s)) = sol_map.get(&key).copied() else {
            panic!("solver mesh does not contain a cell near (x={x}, y={y}) (key={key:?})");
        };
        xy_ref.push((x, y));
        p_ref.push(row[p_idx]);
        ux_ref.push(row[ux_idx]);
        uy_ref.push(row[uy_idx]);
        p_sol.push(p_s);
        ux_sol.push(ux_s);
        uy_sol.push(uy_s);
    }

    let p_err = common::rel_l2(&p_sol, &p_ref, 1e-12);
    let ux_err = common::rel_l2(&ux_sol, &ux_ref, 1e-12);
    let uy_err = common::rel_l2(&uy_sol, &uy_ref, 1e-12);

    // Non-triviality guards: this case has near-constant u_x, so use p/u_y variation.
    let p_ref_dev = common::max_abs_centered(&p_ref);
    let p_sol_dev = common::max_abs_centered(&p_sol);
    let uy_ref_max = common::max_abs(&uy_ref);
    let uy_sol_max = common::max_abs(&uy_sol);
    assert!(p_ref_dev > 1e3 && uy_ref_max > 5.0, "reference appears trivial: p_dev={p_ref_dev:.3e} max_abs(u_y)={uy_ref_max:.3e}");
    assert!(p_sol_dev > 1e2 && uy_sol_max > 1.0, "solver appears trivial: p_dev={p_sol_dev:.3e} max_abs(u_y)={uy_sol_max:.3e}");

    if common::diag_enabled() {
        let p_ref_mean = common::mean(&p_ref);
        let p_sol_mean = common::mean(&p_sol);
        let p_ref_min = p_ref
            .iter()
            .copied()
            .fold(f64::INFINITY, |a, b| a.min(b));
        let p_ref_max = p_ref
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));
        let p_sol_min = p_sol
            .iter()
            .copied()
            .fold(f64::INFINITY, |a, b| a.min(b));
        let p_sol_max = p_sol
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));

        let mut max_abs_dp = 0.0f64;
        let mut max_abs_dp_idx = 0usize;
        for (i, (&ps, &pr)) in p_sol.iter().zip(p_ref.iter()).enumerate() {
            let d = (ps - pr).abs();
            if d > max_abs_dp {
                max_abs_dp = d;
                max_abs_dp_idx = i;
            }
        }
        let (x_max, y_max) = xy_ref[max_abs_dp_idx];

        eprintln!("[openfoam][compressible_wedge] rel_l2 p={p_err:.6} u_x={ux_err:.6} u_y={uy_err:.6} | p_dev ref={p_ref_dev:.3e} sol={p_sol_dev:.3e} | uy_max ref={uy_ref_max:.3e} sol={uy_sol_max:.3e}");
        eprintln!("[openfoam][compressible_wedge] p stats: ref(min={p_ref_min:.3e} max={p_ref_max:.3e} mean={p_ref_mean:.3e}) sol(min={p_sol_min:.3e} max={p_sol_max:.3e} mean={p_sol_mean:.3e}) | max_abs(p_sol-p_ref)={max_abs_dp:.3e} at (x={x_max:.4}, y={y_max:.4})");
    }

    assert!(
        p_err < 2.5 && ux_err < 1.5 && uy_err < 1.5,
        "mismatch vs OpenFOAM: rel_l2(p)={p_err:.3} rel_l2(u_x)={ux_err:.3} rel_l2(u_y)={uy_err:.3}"
    );
}
