#[path = "openfoam_reference/common.rs"]
mod common;

use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundaryType};
use cfd2::solver::model::compressible_model_with_eos;
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::model::helpers::{
    SolverCompressibleIdealGasExt, SolverFieldAliasesExt, SolverRuntimeParamsExt,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};

#[test]
#[ignore]
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
        // Moving lid with Dirichlet velocity (tangential) while keeping scalars zeroGradient.
        BoundaryType::MovingWall,
    );

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
        .set_boundary_vec2(GpuBoundaryType::MovingWall, "u", [u_lid, 0.0])
        .unwrap();
    solver
        .set_boundary_vec2(GpuBoundaryType::MovingWall, "rho_u", [rho0 * u_lid, 0.0])
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

    let (u_ref_field, p_ref_field) =
        common::reference_fields_from_csv(&mesh, &table, x_idx, y_idx, ux_idx, Some(uy_idx), p_idx);
    common::save_openfoam_field_plots(
        "compressible_lid_driven_cavity",
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

    let u_sol: Vec<(f64, f64)> = ux_sol.iter().copied().zip(uy_sol.iter().copied()).collect();
    let u_ref: Vec<(f64, f64)> = ux_ref.iter().copied().zip(uy_ref.iter().copied()).collect();

    let u_scale = common::rms_vec2_mag(&u_ref).max(1e-12);
    let p_scale = common::rms(&p_ref).max(1e-12);
    let u_max = common::max_cell_rel_error_vec2(&u_sol, &u_ref, u_scale);
    let p_max = common::max_cell_rel_error_scalar(&p_sol, &p_ref, p_scale);

    // Non-triviality guards: ensure we're not comparing stagnant fields.
    assert!(
        common::max_abs(&ux_sol) > 1e-3 && common::max_abs(&uy_sol) > 1e-3,
        "solver appears trivial: max_abs_sol(u_x)={:.3e} max_abs_sol(u_y)={:.3e}",
        common::max_abs(&ux_sol),
        common::max_abs(&uy_sol)
    );
    let ux_max_ref = common::max_abs(&ux_ref);
    let uy_max_ref = common::max_abs(&uy_ref);
    assert!(
        ux_max_ref > 1e-3 && uy_max_ref > 1e-3,
        "reference appears trivial: max_abs_ref(u_x)={ux_max_ref:.3e} max_abs_ref(u_y)={uy_max_ref:.3e}"
    );

    if common::diag_enabled() {
        let (x_u, y_u) = (sol_rows[u_max.idx].0, sol_rows[u_max.idx].1);
        let (x_p, y_p) = (sol_rows[p_max.idx].0, sol_rows[p_max.idx].1);
        let (u_sol_x, u_sol_y) = u_sol[u_max.idx];
        let (u_ref_x, u_ref_y) = u_ref[u_max.idx];
        let p_sol_v = p_sol[p_max.idx];
        let p_ref_v = p_ref[p_max.idx];
        eprintln!(
            "[openfoam][compressible_lid] max_cell rel u={:.6} abs={:.6} at (x={:.4}, y={:.4}) | max_cell rel p={:.6} abs={:.3} at (x={:.4}, y={:.4}) | scales u_rms={:.3e} p_rms={:.3e}",
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
        eprintln!(
            "[openfoam][compressible_lid] u@max: sol=({u_sol_x:.6},{u_sol_y:.6}) ref=({u_ref_x:.6},{u_ref_y:.6})"
        );
        eprintln!("[openfoam][compressible_lid] p@max: sol={p_sol_v:.6} ref={p_ref_v:.6}");
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
