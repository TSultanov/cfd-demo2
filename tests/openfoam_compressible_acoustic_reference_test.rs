#[path = "openfoam_reference/common.rs"]
mod common;

use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundaryType};
use cfd2::solver::model::compressible_model;
use cfd2::solver::options::{PreconditionerType, TimeScheme};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{SolverConfig, UnifiedSolver};

#[test]
fn openfoam_compressible_acoustic_matches_reference_profile() {
    std::env::set_var("CFD2_QUIET", "1");

    let nx = 200usize;
    let ny = 1usize;
    let length = 1.0;
    let height = 0.05;

    // OpenFOAM case uses reflecting (slip) walls on all sides.
    let mesh = generate_structured_rect_mesh(
        nx,
        ny,
        length,
        height,
        BoundaryType::Wall,
        BoundaryType::Wall,
        BoundaryType::Wall,
        BoundaryType::Wall,
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

    let gamma = 1.4_f64;
    let p0 = 1.0_f64;
    let rho0 = 1.0_f64;
    let eps = 1e-3_f64;

    let mut rho = vec![0.0f32; mesh.num_cells()];
    let mut p = vec![0.0f32; mesh.num_cells()];
    let u = vec![[0.0f32, 0.0f32]; mesh.num_cells()];
    for i in 0..nx {
        let x = (i as f64 + 0.5) * (length / nx as f64);
        let c = (std::f64::consts::PI * x / length).cos();
        let p_i = p0 * (1.0 + eps * c);
        let rho_i = rho0 * (1.0 + (eps / gamma) * c);
        let cell = i;
        rho[cell] = rho_i as f32;
        p[cell] = p_i as f32;
    }

    solver.set_dt(5e-4);
    solver.set_dtau(0.0);
    solver.set_viscosity(0.0);
    solver.set_inlet_velocity(0.0);
    solver.set_outer_iters(1);
    solver.set_state_fields(&rho, &u, &p);
    solver.initialize_history();

    for _ in 0..100 {
        solver.step();
    }

    let p_out = pollster::block_on(solver.get_p());
    let u_out = pollster::block_on(solver.get_u());

    let table = common::load_csv(&common::data_path("compressible_acoustic_centerline.csv"));
    let x_idx = common::column_idx(&table.header, "x");
    let p_idx = common::column_idx(&table.header, "p");
    let ux_idx = common::column_idx(&table.header, "u_x");

    assert_eq!(
        table.rows.len(),
        nx,
        "reference rows must equal nx for pointwise comparison"
    );

    let mut p_ref = Vec::with_capacity(nx);
    let mut ux_ref = Vec::with_capacity(nx);
    for (i, row) in table.rows.iter().enumerate() {
        let x_expected = (i as f64 + 0.5) * (length / nx as f64);
        assert!(
            (row[x_idx] - x_expected).abs() < 1e-12,
            "unexpected reference x at row {i}: got {}, expected {}",
            row[x_idx],
            x_expected
        );
        p_ref.push(row[p_idx]);
        ux_ref.push(row[ux_idx]);
    }

    // Compare perturbations (avoids division by ~1 baseline).
    let p_out_dp: Vec<f64> = p_out.iter().map(|v| v - p0).collect();
    let p_ref_dp: Vec<f64> = p_ref.iter().map(|v| v - p0).collect();
    let ux_out: Vec<f64> = u_out.iter().take(nx).map(|(x, _)| *x).collect();

    let p_rel = common::rel_l2(&p_out_dp, &p_ref_dp, 1e-12);
    let ux_rel = common::rel_l2(&ux_out, &ux_ref, 1e-12);

    assert!(
        p_rel < 0.5,
        "pressure perturbation mismatch vs OpenFOAM: rel_l2={p_rel:.3}"
    );
    assert!(
        ux_rel < 2.0,
        "u_x mismatch vs OpenFOAM: rel_l2={ux_rel:.3}"
    );

    // Full-field comparison (for Ny=1 this is identical to the centerline, but we keep a
    // separate reference CSV to validate whole-field export/mapping).
    let table = common::load_csv(&common::data_path("compressible_acoustic_full_field.csv"));
    let x_idx = common::column_idx(&table.header, "x");
    let y_idx = common::column_idx(&table.header, "y");
    let p_idx = common::column_idx(&table.header, "p");
    let ux_idx = common::column_idx(&table.header, "u_x");

    assert_eq!(
        table.rows.len(),
        mesh.num_cells(),
        "reference rows must equal num_cells for full-field comparison"
    );

    let mut ref_rows: Vec<(f64, f64, f64, f64)> = table
        .rows
        .iter()
        .map(|r| (r[x_idx], r[y_idx], r[p_idx], r[ux_idx]))
        .collect();
    ref_rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.partial_cmp(&b.1).unwrap()));

    let mut sol_rows: Vec<(f64, f64, f64, f64)> = (0..mesh.num_cells())
        .map(|i| (mesh.cell_cx[i], mesh.cell_cy[i], p_out[i], u_out[i].0))
        .collect();
    sol_rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.partial_cmp(&b.1).unwrap()));

    for (i, (sol, rf)) in sol_rows.iter().zip(ref_rows.iter()).enumerate() {
        let (sx, sy, _, _) = *sol;
        let (rx, ry, _, _) = *rf;
        assert!((sx - rx).abs() < 1e-12, "x mismatch at sorted row {i}: solver={sx} ref={rx}");
        assert!((sy - ry).abs() < 1e-12, "y mismatch at sorted row {i}: solver={sy} ref={ry}");
    }

    let p_out_dp: Vec<f64> = sol_rows.iter().map(|r| r.2 - p0).collect();
    let p_ref_dp: Vec<f64> = ref_rows.iter().map(|r| r.2 - p0).collect();
    let ux_out: Vec<f64> = sol_rows.iter().map(|r| r.3).collect();
    let ux_ref: Vec<f64> = ref_rows.iter().map(|r| r.3).collect();

    let p_rel = common::rel_l2(&p_out_dp, &p_ref_dp, 1e-12);
    let ux_rel = common::rel_l2(&ux_out, &ux_ref, 1e-12);

    assert!(
        p_rel < 0.5,
        "pressure full-field perturbation mismatch vs OpenFOAM: rel_l2={p_rel:.3}"
    );
    assert!(
        ux_rel < 2.0,
        "u_x full-field mismatch vs OpenFOAM: rel_l2={ux_rel:.3}"
    );
}
