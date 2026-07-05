#[path = "openfoam_reference/common.rs"]
mod common;

use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType};
use cfd2::solver::model::helpers::{SolverFieldAliasesExt, SolverRuntimeParamsExt};
use cfd2::solver::model::incompressible_momentum_model;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};

/// Test incompressible lid-driven cavity against OpenFOAM reference.
///
/// The reference CSV is the machine-converged steady state (t = 32, final
/// pimpleFoam initial residuals ~5e-12), so the transient path drops out of
/// the comparison. Absolute accuracy vs literature is anchored separately by
/// tests/ghia_lid_cavity_test.rs.
///
/// Canonical config: dt = 0.02, alpha_u = 0.7. d_p ∝ alpha_u·dt/rho is part
/// of cfd2's SPATIAL discretization, so dt is NOT a free steady-marching knob
/// — the discrete steady state depends on it. dt=0.02 matches the reference's
/// own dt and hence its coupling-coefficient inputs.
///
/// Error structure is corner-localized at the two singular lid corners
/// ((0,1)/(1,1)); away from them the two codes agree to ~0.24%. The band is
/// the all-cells value; the corner-exclusion diag reports the smooth-field
/// agreement.
///
/// Requires extended timeout (~60-120s) due to GPU compute:
/// `cargo test --test openfoam_incompressible_lid_driven_cavity_reference_test -- --ignored --timeout 120`
#[test]
#[ignore]
fn openfoam_incompressible_lid_driven_cavity_matches_reference_field() {
    std::env::set_var("CFD2_QUIET", "1");

    // Match the OpenFOAM case setup in `reference/openfoam/incompressible_lid_driven_cavity`.
    let nx = 20usize;
    let ny = 20usize;
    let length = 1.0;
    let height = 1.0;

    let mesh = generate_structured_rect_mesh(
        nx,
        ny,
        length,
        height,
        BoundarySides {
            left: BoundaryType::Wall,
            right: BoundaryType::Wall,
            bottom: BoundaryType::Wall,
            top: BoundaryType::MovingWall,
        },
    );

    let mut solver = pollster::block_on(UnifiedSolver::new(
        &mesh,
        incompressible_momentum_model().expect("model"),
        SolverConfig {
            advection_scheme: Scheme::Upwind,
            // Reference is the machine-steady end state, so the transient
            // path (time scheme, dt, outer iters) drops out of the comparison.
            time_scheme: TimeScheme::BDF2,
            preconditioner: PreconditionerType::Jacobi,
            stepping: SteppingMode::Coupled,
        },
        None,
        None,
    ))
    .expect("solver init");

    // Steady marching: dt/outer chosen for wall time, not transient
    // fidelity (see the SolverConfig note).
    solver.set_dt(0.02);
    solver.set_dtau(0.0).unwrap();
    solver.set_density(1.0).unwrap();
    solver.set_viscosity(0.01).unwrap();
    solver
        .set_boundary_vec2(GpuBoundaryType::MovingWall, "U", [1.0, 0.0])
        .unwrap();
    // d_p sensitivity probe knob (diagnostic only).
    let alpha_u_probe: f32 = std::env::var("CFD2_LID_ALPHA_U")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.7);
    solver.set_alpha_u(alpha_u_probe).unwrap();
    solver.set_alpha_p(0.3).unwrap();
    solver.set_outer_iters(5).unwrap();
    solver.set_u(&vec![(0.0, 0.0); mesh.num_cells()]);
    solver.set_p(&vec![0.0; mesh.num_cells()]);
    solver.initialize_history();

    // March to steady state: stop when the per-step max velocity delta
    // drops below tolerance (lid speed = 1).
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
            println!("[openfoam][incompressible_lid] steady after {steps} steps (per-step delta {delta:.2e})");
            break;
        }
        assert!(
            steps < MAX_STEPS,
            "no steady state within {MAX_STEPS} steps (delta {delta:.2e})"
        );
    }

    let u = prev;
    let p = pollster::block_on(solver.get_p());

    let table = common::load_csv(&common::data_path(
        "incompressible_lid_driven_cavity_full_field.csv",
    ));
    let x_idx = common::column_idx(&table.header, "x");
    let y_idx = common::column_idx(&table.header, "y");
    let ux_idx = common::column_idx(&table.header, "u_x");
    let uy_idx = common::column_idx(&table.header, "u_y");
    let p_idx = common::column_idx(&table.header, "p");

    let (u_ref_field, p_ref_field) =
        common::reference_fields_from_csv(&mesh, &table, x_idx, y_idx, ux_idx, Some(uy_idx), p_idx);
    common::save_openfoam_field_plots(
        "incompressible_lid_driven_cavity",
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

    let u_scale = common::rms_vec2_mag(&u_ref).max(1e-12);
    let p_scale = common::rms(&p_ref).max(1e-12);
    let u_max = common::max_cell_rel_error_vec2(&u_sol, &u_ref, u_scale);
    let p_max = common::max_cell_rel_error_scalar(&p_sol, &p_ref, p_scale);

    if common::diag_enabled() {
        let (x_u, y_u) = (sol_rows[u_max.idx].0, sol_rows[u_max.idx].1);
        let (x_p, y_p) = (sol_rows[p_max.idx].0, sol_rows[p_max.idx].1);
        eprintln!(
            "[openfoam][incompressible_lid] max_cell rel u={:.6} abs={:.6} at (x={:.4}, y={:.4}) | max_cell rel p(mean-free)={:.6} abs={:.3} at (x={:.4}, y={:.4}) | scales u_rms={:.3e} p_rms={:.3e}",
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
        // Spatial structure: how much of the mismatch is the two singular
        // lid corners (velocity-discontinuous BC junctions at (0,1)/(1,1))?
        // Report the max over cells at least r cells away from both corners
        // (Chebyshev distance in cell units, h=0.05), plus the rel_l2 of
        // that exterior, for increasing exclusion radii.
        let h = length / nx as f64;
        for r in [1usize, 2, 3, 4] {
            let rad = r as f64 * h + 1e-9;
            let mut max_rel = 0.0f64;
            let mut max_xy = (0.0, 0.0);
            let mut sum_sq = 0.0f64;
            let mut count = 0usize;
            for (i, row) in sol_rows.iter().enumerate() {
                let (x, y) = (row.0, row.1);
                let d1 = (x - 0.0).abs().max((y - height).abs());
                let d2 = (x - length).abs().max((y - height).abs());
                if d1 < rad || d2 < rad {
                    continue;
                }
                let dx = u_sol[i].0 - u_ref[i].0;
                let dy = u_sol[i].1 - u_ref[i].1;
                let e = (dx * dx + dy * dy).sqrt() / u_scale;
                if e > max_rel {
                    max_rel = e;
                    max_xy = (x, y);
                }
                sum_sq += e * e;
                count += 1;
            }
            eprintln!(
                "[openfoam][incompressible_lid] corner-excl r={r}: max rel u={max_rel:.6} at (x={:.4}, y={:.4}) rel_l2={:.6} over {count} cells",
                max_xy.0,
                max_xy.1,
                (sum_sq / count as f64).sqrt(),
            );
        }
        // dev2-hypothesis probe: pimpleFoam's momentum equation carries
        // div(nu*dev2(T(grad U))) (see the reference fvSchemes); cfd2's
        // incompressible momentum has only the laplacian. The term vanishes
        // analytically for div-free u but not discretely. Compute it on the
        // 20x20 grid from the REFERENCE field (host Gauss: one-sided at
        // boundaries with no-slip/lid values) and compare its local size
        // against the viscous laplacian term and its pattern against the
        // observed error.
        let nu = 0.01f64;
        let n = nx;
        let h = length / nx as f64;
        let at = |i: isize, j: isize| -> (f64, f64) {
            // Ghost values from BCs: walls no-slip, lid (j==n) u=(1,0).
            if j >= n as isize {
                return (2.0 * 1.0 - u_ref[(n - 1) * n + i.clamp(0, n as isize - 1) as usize].0,
                        -u_ref[(n - 1) * n + i.clamp(0, n as isize - 1) as usize].1);
            }
            if i < 0 || i >= n as isize || j < 0 {
                let ii = i.clamp(0, n as isize - 1) as usize;
                let jj = j.clamp(0, n as isize - 1) as usize;
                let v = u_ref[jj * n + ii];
                return (-v.0, -v.1); // mirror for no-slip wall ghost
            }
            u_ref[j as usize * n + i as usize]
        };
        // Cell-centered gradients via central differences over ghosts.
        let grad = |i: usize, j: usize| -> [[f64; 2]; 2] {
            let (i, j) = (i as isize, j as isize);
            let (uxe, uye) = at(i + 1, j);
            let (uxw, uyw) = at(i - 1, j);
            let (uxn, uyn) = at(i, j + 1);
            let (uxs, uys) = at(i, j - 1);
            [
                [(uxe - uxw) / (2.0 * h), (uxn - uxs) / (2.0 * h)],
                [(uye - uyw) / (2.0 * h), (uyn - uys) / (2.0 * h)],
            ]
        };
        // dev2(T(grad U))_ij = dU_j/dx_i - 2/3 delta_ij div  (transpose part);
        // D = nu * div_h of that tensor; LAP = nu * lap(u) for scale.
        let mut top: Vec<(f64, usize, f64)> = Vec::new(); // (|D|, idx, |D|/|lap|)
        for j in 1..n - 1 {
            for i in 1..n - 1 {
                let gxp = grad(i + 1, j);
                let gxm = grad(i - 1, j);
                let gyp = grad(i, j + 1);
                let gym = grad(i, j - 1);
                let divv = |g: [[f64; 2]; 2]| g[0][0] + g[1][1];
                // T row k = (dU_k/dx - (2/3)div*delta.., ...) transpose form:
                // T[k][l] = g[l][k] - 2/3 div delta_kl
                let t = |g: [[f64; 2]; 2], k: usize, l: usize| {
                    g[l][k] - if k == l { 2.0 / 3.0 * divv(g) } else { 0.0 }
                };
                let dx_x = (t(gxp, 0, 0) - t(gxm, 0, 0)) / (2.0 * h);
                let dy_x = (t(gyp, 0, 1) - t(gym, 0, 1)) / (2.0 * h);
                let dx_y = (t(gxp, 1, 0) - t(gxm, 1, 0)) / (2.0 * h);
                let dy_y = (t(gyp, 1, 1) - t(gym, 1, 1)) / (2.0 * h);
                let dvec = (nu * (dx_x + dy_x), nu * (dx_y + dy_y));
                let dmag = (dvec.0 * dvec.0 + dvec.1 * dvec.1).sqrt();
                // laplacian via 5-point on u
                let (uc_x, uc_y) = at(i as isize, j as isize);
                let (ue, ve) = at(i as isize + 1, j as isize);
                let (uw, vw) = at(i as isize - 1, j as isize);
                let (un, vn) = at(i as isize, j as isize + 1);
                let (us, vs) = at(i as isize, j as isize - 1);
                let lap = (
                    nu * (ue + uw + un + us - 4.0 * uc_x) / (h * h),
                    nu * (ve + vw + vn + vs - 4.0 * uc_y) / (h * h),
                );
                let lmag = (lap.0 * lap.0 + lap.1 * lap.1).sqrt();
                top.push((dmag, j * n + i, dmag / lmag.max(1e-12)));
            }
        }
        top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for (dmag, idx, ratio) in top.iter().take(5) {
            let (x, y) = (sol_rows[*idx].0, sol_rows[*idx].1);
            let du = ((u_sol[*idx].0 - u_ref[*idx].0).powi(2)
                + (u_sol[*idx].1 - u_ref[*idx].1).powi(2))
            .sqrt()
                / u_scale;
            eprintln!(
                "[openfoam][incompressible_lid] dev2 probe: |D|={dmag:.4} at (x={x:.4}, y={y:.4}) |D|/|nu lap u|={ratio:.3} local rel err={du:.4}"
            );
        }
    }

    assert!(
        u_max.rel < common::reference_bands("incompressible_lid").max_cell_u,
        "U mismatch vs OpenFOAM (per-cell): max_rel={:.6} (tol={:.6}) max_abs={:.6} at (x={:.6}, y={:.6})",
        u_max.rel,
        common::reference_bands("incompressible_lid").max_cell_u,
        u_max.abs,
        sol_rows[u_max.idx].0,
        sol_rows[u_max.idx].1
    );
    assert!(
        p_max.rel < common::reference_bands("incompressible_lid").max_cell_p,
        "p mismatch vs OpenFOAM (per-cell, mean-free): max_rel={:.6} (tol={:.6}) max_abs={:.3} at (x={:.6}, y={:.6})",
        p_max.rel,
        common::reference_bands("incompressible_lid").max_cell_p,
        p_max.abs,
        sol_rows[p_max.idx].0,
        sol_rows[p_max.idx].1
    );
}
