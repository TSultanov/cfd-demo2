#[path = "openfoam_reference/common.rs"]
mod common;

use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType};
use cfd2::solver::model::helpers::{
    SolverFieldAliasesExt, SolverInletVelocityExt, SolverRuntimeParamsExt,
};
use cfd2::solver::model::incompressible_momentum_model;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};

#[test]
#[ignore]
fn diag_incompressible_channel_flux_balance() {
    std::env::set_var("CFD2_QUIET", "1");

    let nx = 40usize;
    let ny = 20usize;
    let length = 1.0;
    let height = 0.2;

    let mesh = generate_structured_rect_mesh(
        nx,
        ny,
        length,
        height,
        BoundarySides {
            left: BoundaryType::Inlet,
            right: BoundaryType::Outlet,
            bottom: BoundaryType::Wall,
            top: BoundaryType::Wall,
        },
    );

    let mut solver = pollster::block_on(UnifiedSolver::new(
        &mesh,
        incompressible_momentum_model(),
        SolverConfig {
            advection_scheme: Scheme::Upwind,
            time_scheme: TimeScheme::BDF2,
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
    solver.set_outer_iters(10).unwrap();
    solver.set_u(&vec![(0.0, 0.0); mesh.num_cells()]);
    solver.set_p(&vec![0.0; mesh.num_cells()]);
    solver.initialize_history();

    for _ in 0..10 {
        solver.step();
    }

    let u = pollster::block_on(solver.get_u());
    let p = pollster::block_on(solver.get_p());
    let d_p = pollster::block_on(solver.get_field_scalar("d_p")).expect("read d_p");
    let grad_p = pollster::block_on(solver.get_field_vec2("grad_p")).expect("read grad_p");

    let p_mean = common::mean(&p);
    let p_rms = common::rms(&p);
    let p_c: Vec<f64> = p.iter().map(|v| v - p_mean).collect();
    let p_rms_c = common::rms(&p_c);

    let dp_min = d_p
        .iter()
        .copied()
        .fold(f64::INFINITY, |acc, v| acc.min(v));
    let dp_max = d_p
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |acc, v| acc.max(v));

    eprintln!(
        "[diag][incompressible_channel] steps=10 outer=10 | p mean={p_mean:.3e} rms={p_rms:.3e} rms(mean-free)={p_rms_c:.3e} | d_p min={dp_min:.3e} max={dp_max:.3e}"
    );

    let rho = 1.0f64;
    let u_inlet = (1.0f64, 0.0f64);
    let mut phi_sum_corr = 0.0f64;
    let mut phi_sum_pred = 0.0f64;
    let mut m_in = 0.0f64;
    let mut m_out = 0.0f64;
    let mut m_in_u = 0.0f64;
    let mut m_out_u = 0.0f64;
    let mut m_out_u_minus = 0.0f64;
    let mut m_out_u_plus = 0.0f64;

    for face in 0..mesh.num_faces() {
        let b = mesh.face_boundary[face];
        if b.is_none() {
            continue;
        }

        let owner = mesh.face_owner[face];
        let (mut nx, mut ny) = (mesh.face_nx[face], mesh.face_ny[face]);
        let (cx, cy) = (mesh.cell_cx[owner], mesh.cell_cy[owner]);
        let (fx, fy) = (mesh.face_cx[face], mesh.face_cy[face]);
        if (fx - cx) * nx + (fy - cy) * ny < 0.0 {
            nx = -nx;
            ny = -ny;
        }
        let area = mesh.face_area[face];

        let (u_neigh, p_neigh, grad_p_neigh, d_p_neigh) = match b.unwrap() {
            BoundaryType::Inlet => (u_inlet, p[owner], (0.0, 0.0), d_p[owner]),
            // Match `flux_module_wgsl` special-casing: neighbor-side `grad_p` is *not* forced to
            // zero on outlet faces so the HbyA predictor sees the interior pressure gradient.
            BoundaryType::Outlet => (u[owner], 0.0, grad_p[owner], d_p[owner]),
            BoundaryType::Wall | BoundaryType::SlipWall | BoundaryType::MovingWall => {
                ((0.0, 0.0), p[owner], (0.0, 0.0), d_p[owner])
            }
        };

        let u_owner = u[owner];
        let p_owner = p[owner];
        let grad_p_owner = grad_p[owner];
        let d_p_owner = d_p[owner];

        // Match flux_module_incompressible_momentum.wgsl boundary interpolation behavior:
        // lambda=0 on boundary so face values use the "neighbor" (BC/ghost) value.
        let lambda = 0.0f64;
        let lambda_other = 1.0f64;

        let u_face = (
            u_owner.0 * lambda + u_neigh.0 * lambda_other,
            u_owner.1 * lambda + u_neigh.1 * lambda_other,
        );
        let d_p_face = d_p_owner * lambda + d_p_neigh * lambda_other;

        let grad_p_face = (
            grad_p_owner.0 * lambda + grad_p_neigh.0 * lambda_other,
            grad_p_owner.1 * lambda + grad_p_neigh.1 * lambda_other,
        );
        let hby_a_face = (
            u_face.0 + d_p_face * grad_p_face.0,
            u_face.1 + d_p_face * grad_p_face.1,
        );
        let u_n = hby_a_face.0 * nx + hby_a_face.1 * ny;
        let phi_pred = rho * u_n * area;

        // Use the projected normal distance (structured mesh is orthogonal).
        let dist = ((fx - cx) * nx + (fy - cy) * ny).abs().max(1e-12);
        let dp = p_neigh - p_owner;
        let phi_p = rho * d_p_face * dp / dist * area;
        let phi_corr = phi_pred - phi_p;

        phi_sum_pred += phi_pred;
        phi_sum_corr += phi_corr;

        let flux_u = rho * (u_face.0 * nx + u_face.1 * ny) * area;
        let flux_u_minus = match b.unwrap() {
            BoundaryType::Outlet => {
                let u_minus = (
                    u_owner.0 - d_p_owner * grad_p_owner.0,
                    u_owner.1 - d_p_owner * grad_p_owner.1,
                );
                rho * (u_minus.0 * nx + u_minus.1 * ny) * area
            }
            _ => flux_u,
        };
        let flux_u_plus = match b.unwrap() {
            BoundaryType::Outlet => {
                let u_plus = (
                    u_owner.0 + d_p_owner * grad_p_owner.0,
                    u_owner.1 + d_p_owner * grad_p_owner.1,
                );
                rho * (u_plus.0 * nx + u_plus.1 * ny) * area
            }
            _ => flux_u,
        };
        match b.unwrap() {
            BoundaryType::Inlet => m_in += -phi_corr,
            BoundaryType::Outlet => m_out += phi_corr,
            BoundaryType::Wall | BoundaryType::SlipWall | BoundaryType::MovingWall => {}
        }
        match b.unwrap() {
            BoundaryType::Inlet => m_in_u += -flux_u,
            BoundaryType::Outlet => m_out_u += flux_u,
            BoundaryType::Wall | BoundaryType::SlipWall | BoundaryType::MovingWall => {}
        }
        match b.unwrap() {
            BoundaryType::Outlet => {
                m_out_u_minus += flux_u_minus;
                m_out_u_plus += flux_u_plus;
            }
            _ => {}
        }
    }

    eprintln!(
        "[diag][incompressible_channel] boundary flux sums (outward): phi_pred_sum={phi_sum_pred:.6e} phi_corr_sum={phi_sum_corr:.6e} | mass(phi_corr) inflow={m_in:.6} outflow={m_out:.6} | mass(U) inflow={m_in_u:.6} outflow={m_out_u:.6} | out(U-d_p*grad_p)={m_out_u_minus:.6} out(U+d_p*grad_p)={m_out_u_plus:.6}"
    );
}
