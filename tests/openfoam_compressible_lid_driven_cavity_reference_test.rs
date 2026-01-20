#[path = "openfoam_reference/common.rs"]
mod common;

use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundaryType};
use cfd2::solver::model::compressible_model_with_eos;
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::model::helpers::{
    SolverCompressibleInletExt,
    SolverCompressibleIdealGasExt, SolverFieldAliasesExt, SolverRuntimeParamsExt,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};

#[test]
fn openfoam_compressible_lid_driven_cavity_matches_reference_field() {
    std::env::set_var("CFD2_QUIET", "1");

    // Match the OpenFOAM case setup in `reference/openfoam/compressible_lid_driven_cavity`.
    let nx = 20usize;
    let ny = 20usize;
    let length = 1.0;
    let height = 1.0;

    let mesh = generate_structured_rect_mesh(
        nx,
        ny,
        length,
        height,
        BoundaryType::Wall,
        BoundaryType::Wall,
        BoundaryType::Wall,
        // Treat the moving lid as an `Inlet` so we can apply a Dirichlet tangential velocity
        // (with zero normal component) via the boundary table mechanism.
        BoundaryType::Inlet,
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

    let p0 = 101325.0f32;
    let rho0 = (p0 as f64 / (287.0 * 300.0)) as f32;

    // Match the OpenFOAM case (rhoCentralFoam) setup:
    // - Lid tangential speed is high (near Mach 1 at 300K), but normal velocity is 0 so there is
    //   no net mass flux through the lid.
    // - Use deliberately high viscosity so the flow becomes non-trivial quickly at this coarse
    //   resolution.
    let u_lid = 1.0f32;

    solver.set_dt(1e-5);
    solver.set_dtau(0.0).unwrap();
    solver.set_viscosity(1.0).unwrap();
    solver.set_density(rho0).unwrap();
    solver.set_outer_iters(1).unwrap();
    solver.set_uniform_state(rho0, [0.0, 0.0], p0);
    solver
        .set_compressible_inlet_isothermal_x(rho0, u_lid, &eos)
        .unwrap();
    solver.initialize_history();

    for _ in 0..300 {
        solver.step();
    }

    let u = pollster::block_on(solver.get_u());
    let p = pollster::block_on(solver.get_p());

    let table = common::load_csv(&common::data_path(
        "compressible_lid_driven_cavity_full_field.csv",
    ));
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
        assert!(
            (sx - rx).abs() < 1e-12,
            "x mismatch at sorted row {i}: solver={sx} ref={rx}"
        );
        assert!(
            (sy - ry).abs() < 1e-12,
            "y mismatch at sorted row {i}: solver={sy} ref={ry}"
        );
    }

    let p_sol: Vec<f64> = sol_rows.iter().map(|r| r.2).collect();
    let ux_sol: Vec<f64> = sol_rows.iter().map(|r| r.3).collect();
    let uy_sol: Vec<f64> = sol_rows.iter().map(|r| r.4).collect();
    let p_ref: Vec<f64> = ref_rows.iter().map(|r| r.2).collect();
    let ux_ref: Vec<f64> = ref_rows.iter().map(|r| r.3).collect();
    let uy_ref: Vec<f64> = ref_rows.iter().map(|r| r.4).collect();

    let p_err = common::rel_l2(&p_sol, &p_ref, 1e-12);
    let ux_err = common::rel_l2(&ux_sol, &ux_ref, 1e-12);
    let uy_err = common::rel_l2(&uy_sol, &uy_ref, 1e-12);

    let rms = |xs: &[f64]| -> f64 {
        if xs.is_empty() {
            return 0.0;
        }
        (xs.iter().map(|v| v * v).sum::<f64>() / xs.len() as f64).sqrt()
    };
    let rms_diff = |a: &[f64], b: &[f64]| -> f64 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }
        assert_eq!(a.len(), b.len());
        (a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            / a.len() as f64)
            .sqrt()
    };
    let max_abs = |xs: &[f64]| -> f64 { xs.iter().map(|v| v.abs()).fold(0.0, f64::max) };

    let ux_rms_ref = rms(&ux_ref);
    let uy_rms_ref = rms(&uy_ref);
    let ux_rms_sol = rms(&ux_sol);
    let uy_rms_sol = rms(&uy_sol);
    let ux_rms_diff = rms_diff(&ux_sol, &ux_ref);
    let uy_rms_diff = rms_diff(&uy_sol, &uy_ref);
    let ux_max_sol = max_abs(&ux_sol);
    let uy_max_sol = max_abs(&uy_sol);

    // Non-triviality guards: ensure we're not comparing stagnant fields.
    assert!(
        ux_max_sol > 1e-3 && uy_max_sol > 1e-3,
        "solver appears trivial: max_abs_sol(u_x)={ux_max_sol:.3e} max_abs_sol(u_y)={uy_max_sol:.3e}"
    );
    let ux_max_ref = max_abs(&ux_ref);
    let uy_max_ref = max_abs(&uy_ref);
    assert!(
        ux_max_ref > 1e-3 && uy_max_ref > 1e-3,
        "reference appears trivial: max_abs_ref(u_x)={ux_max_ref:.3e} max_abs_ref(u_y)={uy_max_ref:.3e}"
    );

    assert!(
        p_err < 2.0 && ux_err < 2.0 && uy_err < 2.0,
        "mismatch vs OpenFOAM: rel_l2(p)={p_err:.3} rel_l2(u_x)={ux_err:.3} rel_l2(u_y)={uy_err:.3} | rms_ref(u_x)={ux_rms_ref:.3e} rms_ref(u_y)={uy_rms_ref:.3e} | rms_sol(u_x)={ux_rms_sol:.3e} rms_sol(u_y)={uy_rms_sol:.3e} | rms_diff(u_x)={ux_rms_diff:.3e} rms_diff(u_y)={uy_rms_diff:.3e} | max_abs_sol(u_x)={ux_max_sol:.3e} max_abs_sol(u_y)={uy_max_sol:.3e}"
    );
}
