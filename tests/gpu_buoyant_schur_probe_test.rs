//! Probe: does the generic Schur preconditioner work for the buoyant model
//! (extra scalar unknown T in the u-block), or does the corruption recorded
//! during the capstone ("p -> 2.7e5 on step 0, NaN cascade") still
//! reproduce?
//!
//! Background (plan Arc 1, June 2026): the capstone experiment predates two
//! fixes that each explain that exact symptom — (a) the Schur validator
//! originally compared STATE offsets (T = offset 8) instead of coupled
//! FluxLayout ranks (T = rank 3) and was fixed within the same commit, and
//! (b) the FGMRES restart-corruption guard (commit 1d3dde9), whose
//! motivating failure has the identical signature. The Schur kernels
//! themselves are N-generic (u_index tables, u_len-sized buffers,
//! rank-correct col decoding), so the "Schur bridge assumes the velocity
//! pair" conclusion was never re-established after either fix.
//!
//! OUTCOME (June 2026): the corruption did NOT reproduce (rel_l2 vs the
//! default preconditioner ~1e-7..1e-6, equal residual floors), so the
//! buoyant model now declares Schur{u=[U_x,U_y,T], p} by default and this
//! probe is promoted to a smoke test: it twin-runs the buoyant MMS setup
//! (8x8, 5 steps) with the model's declared Schur preconditioner vs a
//! block-Jacobi override (linear_solver: None) and asserts agreement.
//! Only the preconditioner differs, so the outer fixed point is identical
//! and the fields must agree closely. It also asserts the model's default
//! spec IS Schur (guards against silent fallback regressions).
#![cfg(feature = "dev-tests")]

use std::f64::consts::PI;

use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType};
use cfd2::solver::model::helpers::{SolverFieldAliasesExt, SolverRuntimeParamsExt};
use cfd2::solver::model::{
    buoyant_incompressible_mms_model, FluxLayout, ModelLinearSolverSpec, ModelPreconditionerSpec,
    SchurBlockLayout, BUOYANT_BETA_G, BUOYANT_K_OVER_CP, BUOYANT_MMS_SOURCE_T_FIELD,
    BUOYANT_MMS_SOURCE_U_FIELD, BUOYANT_T0, BUOYANT_TEMPERATURE_FIELD,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{
    LinearSolverStats, PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver,
};

const MU: f64 = 1.0;
const RHO: f64 = 1.0;
const N: usize = 8;
const STEPS: usize = 5;

fn exact_u(x: f64, y: f64) -> (f64, f64) {
    ((PI * x).sin() * (PI * y).cos(), -(PI * x).cos() * (PI * y).sin())
}

fn exact_p(x: f64, y: f64) -> f64 {
    (RHO / 4.0) * ((2.0 * PI * x).cos() + (2.0 * PI * y).cos())
}

fn exact_t(x: f64, y: f64) -> f64 {
    (PI * x).cos() * (PI * y).cos()
}

fn source_u(x: f64, y: f64) -> (f64, f64) {
    let (ux, uy) = exact_u(x, y);
    let visc = 2.0 * MU * PI * PI;
    let f_buoy_y = RHO * BUOYANT_BETA_G * (exact_t(x, y) - BUOYANT_T0);
    (visc * ux, visc * uy - f_buoy_y)
}

fn source_t(x: f64, y: f64) -> f64 {
    let (ux, uy) = exact_u(x, y);
    let dtdx = -PI * (PI * x).sin() * (PI * y).cos();
    let dtdy = -PI * (PI * x).cos() * (PI * y).sin();
    RHO * (ux * dtdx + uy * dtdy) + BUOYANT_K_OVER_CP * 2.0 * PI * PI * exact_t(x, y)
}

#[allow(clippy::type_complexity)]
fn run(
    schur: bool,
) -> (
    Vec<(f64, f64)>,
    Vec<f64>,
    Vec<f64>,
    Vec<LinearSolverStats>,
) {
    let mesh = generate_structured_rect_mesh(
        N,
        N,
        1.0,
        1.0,
        BoundarySides {
            left: BoundaryType::Inlet,
            right: BoundaryType::Outlet,
            bottom: BoundaryType::Wall,
            top: BoundaryType::Wall,
        },
    );
    let mut model = buoyant_incompressible_mms_model().expect("model");
    if schur {
        // The model's shipped default must BE the Schur spec with ranks
        // from the coupled FluxLayout (equation order), NOT the state
        // layout: buoyant T sits at state offset 8 but coupled rank 3.
        let fl = FluxLayout::from_system(&model.system);
        let expected = SchurBlockLayout::from_u_p(
            &[
                fl.offset_for("U_x").expect("U_x rank"),
                fl.offset_for("U_y").expect("U_y rank"),
                fl.offset_for("T").expect("T rank"),
            ],
            fl.offset_for("p").expect("p rank"),
        )
        .expect("layout");
        match model.linear_solver {
            Some(ModelLinearSolverSpec {
                preconditioner: ModelPreconditionerSpec::Schur { layout, .. },
                ..
            }) => assert_eq!(
                layout, expected,
                "buoyant model's declared Schur layout drifted from FluxLayout ranks"
            ),
            other => panic!("buoyant model no longer declares Schur by default: {other:?}"),
        }
    } else {
        model.linear_solver = None;
    }
    let mut solver = pollster::block_on(UnifiedSolver::new(
        &mesh,
        model,
        SolverConfig {
            advection_scheme: Scheme::SecondOrderUpwind,
            time_scheme: TimeScheme::BDF2,
            preconditioner: PreconditionerType::Jacobi,
            stepping: SteppingMode::Coupled,
        },
        None,
        None,
    ))
    .expect("solver init");

    solver.set_dt(0.05);
    solver.set_dtau(0.0).expect("dtau");
    solver.set_density(RHO as f32).expect("density");
    solver.set_viscosity(MU as f32).expect("viscosity");
    solver.set_alpha_u(0.7).expect("alpha_u");
    solver.set_alpha_p(0.3).expect("alpha_p");
    solver.set_outer_iters(25).expect("outer_iters");

    let fx = mesh.face_cx.clone();
    let fy = mesh.face_cy.clone();
    let u_face = |c: usize, fx: &[f64], fy: &[f64]| {
        let fx = fx.to_vec();
        let fy = fy.to_vec();
        move |face_idx: u32| {
            let i = face_idx as usize;
            let (ux, uy) = exact_u(fx[i], fy[i]);
            (if c == 0 { ux } else { uy }) as f32
        }
    };
    for boundary in [
        GpuBoundaryType::Inlet,
        GpuBoundaryType::Outlet,
        GpuBoundaryType::Wall,
    ] {
        for c in 0..2usize {
            solver
                .set_boundary_values_per_face(boundary, "U", c as u32, &u_face(c, &fx, &fy))
                .expect("U bc");
        }
    }
    let t_face = {
        let fx = fx.clone();
        let fy = fy.clone();
        move |face_idx: u32| {
            let i = face_idx as usize;
            exact_t(fx[i], fy[i]) as f32
        }
    };
    for boundary in [GpuBoundaryType::Inlet, GpuBoundaryType::Outlet] {
        solver
            .set_boundary_values_per_face(boundary, BUOYANT_TEMPERATURE_FIELD, 0, &t_face)
            .expect("T bc");
    }
    let p_face = {
        let fx = fx.clone();
        let fy = fy.clone();
        move |face_idx: u32| {
            let i = face_idx as usize;
            exact_p(fx[i], fy[i]) as f32
        }
    };
    solver
        .set_boundary_values_per_face(GpuBoundaryType::Outlet, "p", 0, &p_face)
        .expect("p bc");

    let src_u: Vec<(f64, f64)> = (0..mesh.num_cells())
        .map(|i| source_u(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    solver
        .set_field_vec2(BUOYANT_MMS_SOURCE_U_FIELD, &src_u)
        .expect("upload S_U");
    let src_t: Vec<f64> = (0..mesh.num_cells())
        .map(|i| source_t(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    solver
        .set_field_scalar(BUOYANT_MMS_SOURCE_T_FIELD, &src_t)
        .expect("upload S_T");

    solver.set_u(&vec![(0.0, 0.0); mesh.num_cells()]);
    solver.set_p(&vec![0.0; mesh.num_cells()]);
    solver
        .set_field_scalar(BUOYANT_TEMPERATURE_FIELD, &vec![0.0; mesh.num_cells()])
        .expect("init T");
    solver.initialize_history();

    let mut stats = Vec::new();
    for _ in 0..STEPS {
        stats.extend(solver.step_with_stats().expect("step"));
    }
    let u = pollster::block_on(solver.get_field_vec2("U")).expect("read U");
    let p = pollster::block_on(solver.get_p());
    let t =
        pollster::block_on(solver.get_field_scalar(BUOYANT_TEMPERATURE_FIELD)).expect("read T");
    (u, p, t, stats)
}

fn rel_l2(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f64 = b.iter().map(|y| y * y).sum();
    (num / den.max(1e-30)).sqrt()
}

fn stats_summary(label: &str, stats: &[LinearSolverStats]) {
    let total_iters: u32 = stats.iter().map(|s| s.iterations).sum();
    let worst = stats.iter().map(|s| s.residual).fold(0.0f32, f32::max);
    let diverged = stats.iter().filter(|s| s.diverged).count();
    println!(
        "[buoyant-schur-probe] {label}: solves={} total_iters={total_iters} worst_resid={worst:.3e} diverged={diverged}",
        stats.len()
    );
}

#[test]
fn buoyant_schur_matches_block_jacobi_smoke() {
    std::env::set_var("CFD2_QUIET", "1");
    let (u_def, p_def, t_def, stats_def) = run(false);
    let (u_schur, p_schur, t_schur, stats_schur) = run(true);

    stats_summary("default", &stats_def);
    stats_summary("schur  ", &stats_schur);

    for (name, v) in [("p", &p_schur), ("T", &t_schur)] {
        assert!(
            v.iter().all(|x| x.is_finite()),
            "{name} contains non-finite values under Schur"
        );
    }
    assert!(
        u_schur.iter().all(|(x, y)| x.is_finite() && y.is_finite()),
        "U contains non-finite values under Schur"
    );

    let ux_def: Vec<f64> = u_def.iter().map(|v| v.0).collect();
    let uy_def: Vec<f64> = u_def.iter().map(|v| v.1).collect();
    let ux_schur: Vec<f64> = u_schur.iter().map(|v| v.0).collect();
    let uy_schur: Vec<f64> = u_schur.iter().map(|v| v.1).collect();
    let du = rel_l2(&ux_schur, &ux_def).max(rel_l2(&uy_schur, &uy_def));
    let dp = rel_l2(&p_schur, &p_def);
    let dt = rel_l2(&t_schur, &t_def);
    println!("[buoyant-schur-probe] rel_l2 vs default: u={du:.3e} p={dp:.3e} T={dt:.3e}");

    // Only the preconditioner differs; the outer fixed point is identical.
    // Loose bound: linear solves at different floors leave small wakes.
    let tol = 5e-2;
    assert!(
        du < tol && dp < tol && dt < tol,
        "Schur run diverged from default-preconditioner run: u={du:.3e} p={dp:.3e} T={dt:.3e}"
    );
}
