//! Heated-cavity (de Vahl Davis) Nusselt benchmark for the buoyant model.
//!
//! External-physics check against published values: square cavity, hot left
//! wall (T=1), cold right wall (T=0), adiabatic top/bottom, no-slip
//! everywhere, Boussinesq buoyancy. Benchmark: de Vahl Davis (1983),
//! Pr = 0.71: Nu_avg = 1.118 at Ra = 1e3, Nu_avg = 2.243 at Ra = 1e4.
//!
//! Setup notes:
//! - Hot/cold walls are MovingWall-typed (the buoyant model's thermally-held
//!   wall: Dirichlet T with per-face values, no-slip U); adiabatic walls are
//!   Wall-typed (zero-gradient T). No Outlet => pure-Neumann pressure
//!   (gauge-free).
//! - Ra = beta_g * dT * L^3 / (nu * alpha) is dialed in entirely through the
//!   runtime params (buoyant.beta_g, buoyant.k_over_cp) plus set_viscosity —
//!   no recompiles.
//! - Nu on each held wall from a second-order one-sided gradient using the
//!   two interior cell layers: dT/dx|_w ≈ (9*T1 - T2 - 8*T_w) / (3h).
//!   Steady-state energy balance requires Nu_hot ≈ Nu_cold; both are
//!   asserted against each other and against the reference.

#![cfg(feature = "dev-tests")]

#[path = "mms_support/mod.rs"]
mod mms_support;

use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::gpu::unified_solver::PlanParamValue;
use cfd2::solver::mesh::{
    generate_graded_rect_mesh, generate_structured_rect_mesh, AxisGrading, BoundarySides,
    BoundaryType, Mesh,
};
use cfd2::solver::model::buoyant_incompressible_model;
use cfd2::solver::model::helpers::{SolverFieldAliasesExt, SolverRuntimeParamsExt};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};
use mms_support::tensor_grid;

const N: usize = 40;
const PRANDTL: f64 = 0.71;
/// Thermal diffusivity alpha = k_over_cp / rho (rho = 1). Free scale choice;
/// sets the velocity scale u_c ~ alpha*sqrt(Ra) and the settling time
/// ~ L^2/alpha.
const ALPHA: f64 = 0.05;

struct CavityResult {
    nu_hot: f64,
    nu_cold: f64,
    steps: usize,
    final_delta: f64,
}

fn cavity_sides() -> BoundarySides {
    BoundarySides {
        left: BoundaryType::MovingWall,
        right: BoundaryType::MovingWall,
        bottom: BoundaryType::Wall,
        top: BoundaryType::Wall,
    }
}

fn run_cavity(ra: f64, dt: f64, max_steps: usize) -> CavityResult {
    let mesh = generate_structured_rect_mesh(N, N, 1.0, 1.0, cavity_sides());
    run_cavity_on_mesh(&mesh, ra, dt, max_steps)
}

fn run_cavity_on_mesh(mesh: &Mesh, ra: f64, dt: f64, max_steps: usize) -> CavityResult {
    let model = buoyant_incompressible_model().expect("model");
    let mut solver = pollster::block_on(UnifiedSolver::new(
        mesh,
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

    let nu = PRANDTL * ALPHA;
    let beta_g = ra * nu * ALPHA; // dT = 1, L = 1

    solver.set_dt(dt as f32);
    solver.set_dtau(0.0).expect("dtau");
    solver.set_density(1.0).expect("density");
    solver.set_viscosity(nu as f32).expect("viscosity");
    solver.set_alpha_u(0.7).expect("alpha_u");
    solver.set_alpha_p(0.3).expect("alpha_p");
    solver.set_outer_iters(5).expect("outer_iters");
    solver
        .set_named_param("buoyant.beta_g", PlanParamValue::F32(beta_g as f32))
        .expect("set beta_g");
    solver
        .set_named_param("buoyant.t0", PlanParamValue::F32(0.5))
        .expect("set t0");
    solver
        .set_named_param("buoyant.k_over_cp", PlanParamValue::F32(ALPHA as f32))
        .expect("set k_over_cp");

    // Hot wall x=0 (T=1), cold wall x=1 (T=0), discriminated per face.
    let fx = mesh.face_cx.clone();
    let t_face = move |face_idx: u32| {
        if fx[face_idx as usize] < 0.5 {
            1.0f32
        } else {
            0.0f32
        }
    };
    solver
        .set_boundary_values_per_face(GpuBoundaryType::MovingWall, "T", 0, &t_face)
        .expect("T bc");

    // Initial condition: conduction profile T = 1 - x, fluid at rest.
    let t_init: Vec<f64> = (0..mesh.num_cells())
        .map(|i| 1.0 - mesh.cell_cx[i])
        .collect();
    solver.set_field_scalar("T", &t_init).expect("init T");
    solver.set_u(&vec![(0.0, 0.0); mesh.num_cells()]);
    solver.set_p(&vec![0.0; mesh.num_cells()]);
    solver.initialize_history();

    // March to steady state: stop when the T field's per-interval max delta
    // drops below tolerance (dT = 1 scale).
    const CHECK_EVERY: usize = 25;
    const STEADY_TOL: f64 = 2e-6;
    let mut t_prev = t_init.clone();
    let mut steps = 0usize;
    let mut final_delta = f64::INFINITY;
    while steps < max_steps {
        for _ in 0..CHECK_EVERY {
            solver.step();
        }
        steps += CHECK_EVERY;
        let t_now = pollster::block_on(solver.get_field_scalar("T")).expect("read T");
        let delta = t_now
            .iter()
            .zip(&t_prev)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max)
            / (CHECK_EVERY as f64); // per-step delta
        t_prev = t_now;
        final_delta = delta;
        if delta < STEADY_TOL {
            break;
        }
    }

    // Wall-average Nu via second-order one-sided gradients on the two
    // interior cell layers adjacent to each held wall. Coordinate-based
    // (tensor_grid + non-uniform quadratic stencil), valid on graded
    // meshes; at uniform spacing this reduces exactly to the classic
    // (9 T1 - T2 - 8 T_w) / (3h) form.
    let t = t_prev;
    let grid = tensor_grid(mesh);
    let (nx, ny) = (grid.nx(), grid.ny());
    let lx = *grid.x_bounds.last().unwrap();
    // dT/dn at the wall (n = distance into the fluid) from the quadratic
    // through (0, t_w), (d1, t1), (d2, t2).
    let one_sided_grad = |t_w: f64, t1: f64, d1: f64, t2: f64, d2: f64| -> f64 {
        -t_w * (d1 + d2) / (d1 * d2) + t1 * d2 / (d1 * (d2 - d1)) - t2 * d1 / (d2 * (d2 - d1))
    };
    let mut grid_idx = vec![usize::MAX; nx * ny];
    for c in 0..mesh.num_cells() {
        grid_idx[grid.cell_row[c] * nx + grid.cell_col[c]] = c;
    }
    let mut nu_hot = 0.0f64;
    let mut nu_cold = 0.0f64;
    for j in 0..ny {
        let dy = grid.y_bounds[j + 1] - grid.y_bounds[j];
        // Hot wall x=0, T_w = 1; q = -dT/dx = -dT/dn.
        let c1 = grid_idx[j * nx];
        let c2 = grid_idx[j * nx + 1];
        let g_hot = one_sided_grad(1.0, t[c1], mesh.cell_cx[c1], t[c2], mesh.cell_cx[c2]);
        nu_hot += -g_hot * dy;
        // Cold wall x=lx, T_w = 0; n points in -x, so q = -dT/dx = +dT/dn.
        let c1 = grid_idx[j * nx + (nx - 1)];
        let c2 = grid_idx[j * nx + (nx - 2)];
        let g_cold = one_sided_grad(
            0.0,
            t[c1],
            lx - mesh.cell_cx[c1],
            t[c2],
            lx - mesh.cell_cx[c2],
        );
        nu_cold += g_cold * dy;
    }

    CavityResult {
        nu_hot,
        nu_cold,
        steps,
        final_delta,
    }
}

/// High-Ra extension on a wall-refined graded mesh: the
/// thermal boundary layers at the held walls thin as Ra^(-1/4) (~0.056 at
/// Ra = 1e5), so the x axis gets two-sided geometric refinement while y
/// stays uniform. de Vahl Davis (1983): Nu_avg = 4.519 at Ra = 1e5.
#[test]
#[ignore = "external-physics benchmark, tens of minutes; run explicitly like the OpenFOAM reference suite"]
fn heated_cavity_high_ra_nusselt_graded() {
    std::env::set_var("CFD2_QUIET", "1");

    // (Ra, reference Nu, n, x-grading ratio, dt, step cap). Wall-cell CFL
    // sizing: u_max ~ alpha * v_max(Ra) (de Vahl Davis v_max ≈ 68.6 at 1e5
    // in alpha/L units), dt ≈ 0.5 * h_wall / u_max.
    for &(ra, nu_ref, n, ratio, dt, cap) in &[(1e5, 4.519, 64, 4.0, 1e-3, 20000)] {
        let mesh = generate_graded_rect_mesh(
            n,
            n,
            1.0,
            1.0,
            AxisGrading::TwoSided { ratio },
            AxisGrading::Uniform,
            cavity_sides(),
        );
        let r = run_cavity_on_mesh(&mesh, ra, dt, cap);
        let nu_avg = 0.5 * (r.nu_hot + r.nu_cold);
        let balance = (r.nu_hot - r.nu_cold).abs() / nu_ref;
        let rel_err = (nu_avg - nu_ref).abs() / nu_ref;
        println!(
            "[cavity-graded] Ra={ra:.0e} n={n} ratio={ratio} dt={dt}: Nu_hot={:.4} \
             Nu_cold={:.4} avg={nu_avg:.4} ref={nu_ref} rel_err={rel_err:.4} \
             balance={balance:.4} steps={} delta={:.2e}",
            r.nu_hot, r.nu_cold, r.steps, r.final_delta
        );
        assert!(
            r.final_delta < 1e-5,
            "Ra={ra:.0e}: did not reach steady state (delta {:.2e} after {} steps)",
            r.final_delta,
            r.steps
        );
        // Bands ~2.5x measured; ratchet-only thereafter.
        assert!(
            balance < 0.005,
            "Ra={ra:.0e}: hot/cold wall Nusselt imbalance {balance:.4}"
        );
        assert!(
            rel_err < 0.005,
            "Ra={ra:.0e}: Nu_avg {nu_avg:.4} deviates {rel_err:.4} from de Vahl Davis {nu_ref}"
        );
    }
}

#[test]
#[ignore = "external-physics benchmark, ~10-20 min; run explicitly like the OpenFOAM reference suite"]
fn heated_cavity_nusselt_matches_de_vahl_davis() {
    std::env::set_var("CFD2_QUIET", "1");

    // (Ra, reference Nu, dt, step cap)
    for &(ra, nu_ref, dt, cap) in &[(1e3, 1.118, 0.01, 4000), (1e4, 2.243, 0.004, 8000)] {
        let r = run_cavity(ra, dt, cap);
        let nu_avg = 0.5 * (r.nu_hot + r.nu_cold);
        let balance = (r.nu_hot - r.nu_cold).abs() / nu_ref;
        let rel_err = (nu_avg - nu_ref).abs() / nu_ref;
        println!(
            "[cavity] Ra={ra:.0e}: Nu_hot={:.4} Nu_cold={:.4} avg={nu_avg:.4} ref={nu_ref} \
             rel_err={rel_err:.4} balance={balance:.4} steps={} delta={:.2e}",
            r.nu_hot, r.nu_cold, r.steps, r.final_delta
        );
        assert!(
            r.final_delta < 1e-5,
            "Ra={ra:.0e}: did not reach steady state (delta {:.2e} after {} steps)",
            r.final_delta,
            r.steps
        );
        // Steady-state energy balance: heat in = heat out.
        assert!(
            balance < 0.005,
            "Ra={ra:.0e}: hot/cold wall Nusselt imbalance {balance:.4}"
        );
        // Published-value band.
        assert!(
            rel_err < 0.02,
            "Ra={ra:.0e}: Nu_avg {nu_avg:.4} deviates {rel_err:.4} from de Vahl Davis {nu_ref}"
        );
    }
}
