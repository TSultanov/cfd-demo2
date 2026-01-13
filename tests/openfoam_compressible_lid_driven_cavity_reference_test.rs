#[path = "openfoam_reference/common.rs"]
mod common;

use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundaryType};
use cfd2::solver::model::compressible_model;
use cfd2::solver::options::{PreconditionerType, TimeScheme};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{SolverConfig, UnifiedSolver};

#[test]
fn openfoam_compressible_lid_driven_cavity_matches_reference_field() {
    std::env::set_var("CFD2_QUIET", "1");

    // Match the OpenFOAM case setup in `reference/openfoam/compressible_lid_driven_cavity`.
    let nx = 20usize;
    let ny = 20usize;
    let length = 1.0;
    let height = 1.0;

    // Treat the moving lid as `Inlet` so `set_inlet_velocity` can drive it.
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
        compressible_model(),
        SolverConfig {
            advection_scheme: Scheme::Upwind,
            time_scheme: TimeScheme::Euler,
            preconditioner: PreconditionerType::Jacobi,
        },
        None,
        None,
    ))
    .expect("solver init");

    let rho0 = 1.0f32;
    let p0 = 1.0f32;
    let u_lid = 1.0f32;

    solver.set_dt(5e-4);
    solver.set_dtau(0.0);
    solver.set_viscosity(0.0);
    solver.set_density(rho0);
    solver.set_inlet_velocity(u_lid);
    solver.set_outer_iters(1);
    solver.set_uniform_state(rho0, [0.0, 0.0], p0);
    solver.initialize_history();

    for _ in 0..100 {
        solver.step();
    }

    let u = pollster::block_on(solver.get_u());
    let p = pollster::block_on(solver.get_p());

    let table =
        common::load_csv(&common::data_path("compressible_lid_driven_cavity_full_field.csv"));
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
        assert!((sx - rx).abs() < 1e-12, "x mismatch at sorted row {i}: solver={sx} ref={rx}");
        assert!((sy - ry).abs() < 1e-12, "y mismatch at sorted row {i}: solver={sy} ref={ry}");
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

    assert!(
        p_err < 2.0 && ux_err < 2.0 && uy_err < 2.0,
        "mismatch vs OpenFOAM: rel_l2(p)={p_err:.3} rel_l2(u_x)={ux_err:.3} rel_l2(u_y)={uy_err:.3}"
    );
}
