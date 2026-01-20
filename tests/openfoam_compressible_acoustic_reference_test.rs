#[path = "openfoam_reference/common.rs"]
mod common;

use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundaryType};
use cfd2::solver::model::compressible_model_with_eos;
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::model::helpers::{
    SolverCompressibleIdealGasExt, SolverFieldAliasesExt, SolverInletVelocityExt,
    SolverRuntimeParamsExt,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};

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
        BoundaryType::SlipWall,
        BoundaryType::SlipWall,
        BoundaryType::SlipWall,
        BoundaryType::SlipWall,
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

    let r_gas = 287.0_f64;
    let t0 = 300.0_f64;
    let p0 = 101325.0_f64;
    let eps = 1e-3_f64;

    let mut rho = vec![0.0f32; mesh.num_cells()];
    let mut p = vec![0.0f32; mesh.num_cells()];
    let u = vec![[0.0f32, 0.0f32]; mesh.num_cells()];
    for i in 0..nx {
        let x = (i as f64 + 0.5) * (length / nx as f64);
        let c = (std::f64::consts::PI * x / length).cos();
        let p_i = p0 * (1.0 + eps * c);
        // Match the OpenFOAM setup (uniform T with a pressure perturbation).
        let rho_i = p_i / (r_gas * t0);
        let cell = i;
        rho[cell] = rho_i as f32;
        p[cell] = p_i as f32;
    }

    solver.set_dt(1.7e-6);
    solver.set_dtau(0.0).unwrap();
    solver.set_viscosity(1.81e-5).unwrap();
    solver.set_inlet_velocity(0.0).unwrap();
    solver.set_outer_iters(1).unwrap();
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
    let ux_rel_pos = common::rel_l2(&ux_out, &ux_ref, 1e-12);
    let ux_out_neg: Vec<f64> = ux_out.iter().map(|v| -*v).collect();
    let ux_rel_neg = common::rel_l2(&ux_out_neg, &ux_ref, 1e-12);
    let (ux_rel, ux_sign) = if ux_rel_neg < ux_rel_pos {
        (ux_rel_neg, -1.0)
    } else {
        (ux_rel_pos, 1.0)
    };

    // Non-triviality guards (compare perturbations for pressure).
    let p_ref_dp_max = common::max_abs(&p_ref_dp);
    let p_out_dp_max = common::max_abs(&p_out_dp);
    let ux_ref_max = common::max_abs(&ux_ref);
    let ux_out_max = common::max_abs(&ux_out);
    assert!(p_ref_dp_max > 50.0 && ux_ref_max > 0.01, "reference appears trivial: max_abs(dp_ref)={p_ref_dp_max:.3e} max_abs(u_x_ref)={ux_ref_max:.3e}");
    assert!(p_out_dp_max > 5.0 && ux_out_max > 0.001, "solver appears trivial: max_abs(dp_out)={p_out_dp_max:.3e} max_abs(u_x_out)={ux_out_max:.3e}");

    if common::diag_enabled() {
        let ux_ref_min = ux_ref
            .iter()
            .copied()
            .fold(f64::INFINITY, |a, b| a.min(b));
        let ux_ref_max_v = ux_ref
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));
        let ux_out_min = ux_out
            .iter()
            .copied()
            .fold(f64::INFINITY, |a, b| a.min(b));
        let ux_out_max_v = ux_out
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));

        let ux_ref_mean = common::mean(&ux_ref);
        let ux_out_mean = common::mean(&ux_out);

        let mut dot = 0.0;
        for (&a, &b) in ux_out.iter().zip(ux_ref.iter()) {
            dot += a * b;
        }

        eprintln!("[openfoam][compressible_acoustic] rel_l2 dp={p_rel:.6} u_x(best)={ux_rel:.6} (pos={ux_rel_pos:.6} neg={ux_rel_neg:.6}, sign={ux_sign:+.0}) | max_abs dp ref={p_ref_dp_max:.3e} out={p_out_dp_max:.3e} | max_abs u_x ref={ux_ref_max:.3e} out={ux_out_max:.3e}");
        eprintln!("[openfoam][compressible_acoustic] u_x stats: ref(min={ux_ref_min:.3e} max={ux_ref_max_v:.3e} mean={ux_ref_mean:.3e}) out(min={ux_out_min:.3e} max={ux_out_max_v:.3e} mean={ux_out_mean:.3e}) dot(out,ref)={dot:.3e}");
    }

    assert!(
        p_rel < 0.35,
        "pressure perturbation mismatch vs OpenFOAM: rel_l2={p_rel:.3}"
    );
    assert!(
        ux_rel < 1.5,
        "u_x mismatch vs OpenFOAM (best sign {ux_sign:+.0}): rel_l2={ux_rel:.3} (pos={ux_rel_pos:.3} neg={ux_rel_neg:.3})"
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
    ref_rows.sort_by_key(|r| common::xy_key(r.0, r.1));

    let mut sol_rows: Vec<(f64, f64, f64, f64)> = (0..mesh.num_cells())
        .map(|i| (mesh.cell_cx[i], mesh.cell_cy[i], p_out[i], u_out[i].0))
        .collect();
    sol_rows.sort_by_key(|r| common::xy_key(r.0, r.1));

    for (i, (sol, rf)) in sol_rows.iter().zip(ref_rows.iter()).enumerate() {
        let (sx, sy, _, _) = *sol;
        let (rx, ry, _, _) = *rf;
        assert!(
            (sx - rx).abs() < 1e-12,
            "x mismatch at sorted row {i}: solver={sx} ref={rx}"
        );
        assert!(
            (sy - ry).abs() < 1e-12,
            "y mismatch at sorted row {i}: solver={sy} ref={ry}"
        );
    }

    let p_out_dp: Vec<f64> = sol_rows.iter().map(|r| r.2 - p0).collect();
    let p_ref_dp: Vec<f64> = ref_rows.iter().map(|r| r.2 - p0).collect();
    let ux_out: Vec<f64> = sol_rows.iter().map(|r| r.3).collect();
    let ux_ref: Vec<f64> = ref_rows.iter().map(|r| r.3).collect();

    let p_rel = common::rel_l2(&p_out_dp, &p_ref_dp, 1e-12);
    let ux_rel_pos = common::rel_l2(&ux_out, &ux_ref, 1e-12);
    let ux_out_neg: Vec<f64> = ux_out.iter().map(|v| -*v).collect();
    let ux_rel_neg = common::rel_l2(&ux_out_neg, &ux_ref, 1e-12);
    let (ux_rel, ux_sign) = if ux_rel_neg < ux_rel_pos {
        (ux_rel_neg, -1.0)
    } else {
        (ux_rel_pos, 1.0)
    };

    if common::diag_enabled() {
        eprintln!("[openfoam][compressible_acoustic_full] rel_l2 dp={p_rel:.6} u_x(best)={ux_rel:.6} (pos={ux_rel_pos:.6} neg={ux_rel_neg:.6}, sign={ux_sign:+.0})");
    }

    assert!(
        p_rel < 0.35,
        "pressure full-field perturbation mismatch vs OpenFOAM: rel_l2={p_rel:.3}"
    );
    assert!(
        ux_rel < 1.5,
        "u_x full-field mismatch vs OpenFOAM (best sign {ux_sign:+.0}): rel_l2={ux_rel:.3} (pos={ux_rel_pos:.3} neg={ux_rel_neg:.3})"
    );
}
