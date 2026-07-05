//! ALE conservation audit: closed box, prescribed motion,
//! `incompressible_momentum_ale`.
//!
//! Two conservation statements, asserted every step:
//!
//! 1. **Mesh-side**: `Σ_i V_i` must track the analytic deformed-domain area to
//!    f64 geometry precision. The prescribed motion slides boundary vertices
//!    TANGENTIALLY along their own wall (each wall maps to itself, corners
//!    fixed), so the analytic area is exactly `LX·LY` at every instant while
//!    boundary faces still sweep degenerate (zero-area) quads and interior
//!    cells deform arbitrarily. Pins the swept-quad/shoelace geometry chain.
//!
//! 2. **Global mass audit**: `M = Σ_i rho·V_i` drift per step, pinned to what
//!    is measured. For the incompressible model `rho` is a solver constant, so
//!    `M = rho·Σ V` and this audit is mesh-side too — its solver content is
//!    only that the run stays finite/bounded.
//!
//! The CPU tight variant sets the linear tolerance through the MODEL's recipe
//! field (`model.linear_solver.solver.tolerance`); `CFD2_LIN_TOL` is GPU-only,
//! so an env override would silently not tighten the CPU solve.
#![cfg(feature = "cpu")]

use cfd2::solver::mesh::{
    generate_structured_rect_mesh, swept_mesh_fluxes_closed, BoundarySides, Mesh,
};
use cfd2::solver::model::helpers::{SolverFieldAliasesExt, SolverRuntimeParamsExt};
use cfd2::solver::model::incompressible_momentum_ale_model;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{
    PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver,
};
use std::f64::consts::PI;
use std::sync::Mutex;

/// `CFD2_BACKEND` is process-global; serialize the tests.
static ENV_LOCK: Mutex<()> = Mutex::new(());

const NX: usize = 24;
const NY: usize = 24;
const LX: f64 = 1.0;
const LY: f64 = 1.0;
const DT: f64 = 0.005;
const STEPS: usize = 120;
const PERIOD: f64 = 40.0 * DT;
const RHO: f64 = 1.0;
/// Amplitude ~0.15·h: strong per-step deformation (~0.5% volume swing).
const AMP_FRAC: f64 = 0.15;

/// Prescribed vertex position: interior deformation + TANGENTIAL slide of
/// boundary vertices along their own wall (x-motion depends only on x,
/// y-motion only on y ⇒ every wall maps to itself, corners fixed, domain
/// area analytically constant LX·LY).
fn vertex_position(x0: f64, y0: f64, t: f64) -> (f64, f64) {
    let h = LX / NX as f64;
    let s = AMP_FRAC * h * (2.0 * PI * t / PERIOD).sin();
    (
        x0 + s * (PI * x0 / LX).sin(),
        y0 - 0.7 * s * (PI * y0 / LY).sin(),
    )
}

struct AuditRun {
    /// max_n |M_n − M_{n−1}| / M_0 (per-step relative mass drift).
    max_step_drift: f64,
    /// max_n |M_n − M_0| / M_0 (accumulated relative mass drift).
    max_total_drift: f64,
    /// max_n |Σ V − LX·LY| / (LX·LY) (mesh-side area tracking).
    max_area_err: f64,
    /// max over run of max_i |U_i| (boundedness sanity).
    max_u: f64,
}

fn run_audit(
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    tighten_cpu_tol: Option<f32>,
) -> AuditRun {
    let mut mesh =
        generate_structured_rect_mesh(NX, NY, LX, LY, BoundarySides::wall());
    let x0 = mesh.vx.clone();
    let y0 = mesh.vy.clone();
    let n = mesh.num_cells();

    let mut model = incompressible_momentum_ale_model().expect("ale model");
    if let Some(tol) = tighten_cpu_tol {
        // CPU linear tolerance is the RECIPE field, not CFD2_LIN_TOL (GPU-only).
        let ls = model
            .linear_solver
            .as_mut()
            .expect("model declares a linear solver");
        ls.solver.tolerance = tol;
    }

    let mut solver = pollster::block_on(UnifiedSolver::new(
        &mesh,
        model,
        SolverConfig {
            advection_scheme: Scheme::SecondOrderUpwindVanLeer,
            time_scheme: TimeScheme::BDF2,
            preconditioner: PreconditionerType::Jacobi,
            stepping: SteppingMode::Coupled,
        },
        device,
        queue,
    ))
    .expect("solver init");

    solver.set_dt(DT as f32);
    solver.set_dtau(0.0).expect("dtau");
    solver.set_density(RHO as f32).expect("density");
    solver.set_viscosity(1e-2).expect("viscosity");
    solver.set_alpha_u(0.7).expect("alpha_u");
    solver.set_alpha_p(0.3).expect("alpha_p");
    solver.set_outer_iters(6).expect("outer_iters");

    // From rest in a closed box. The PHYSICAL domain never deforms (walls map
    // to themselves), so the exact solution stays U ≡ 0, p ≡ const: the
    // interior mesh deformation is a pure discretization change, and any
    // velocity the run develops is SPURIOUS ALE-injected motion. max|U| is
    // therefore a zero-flow free-stream-preservation statement, pinned at
    // f32-noise scale below.
    solver.set_u(&vec![(0.0, 0.0); n]);
    solver.set_p(&vec![0.0; n]);
    solver.initialize_history();

    let area = LX * LY;
    let mass = |mesh: &Mesh| -> f64 { RHO * mesh.cell_vol.iter().sum::<f64>() };
    let m0 = mass(&mesh);
    let mut m_prev = m0;

    let mut out = AuditRun {
        max_step_drift: 0.0,
        max_total_drift: 0.0,
        max_area_err: 0.0,
        max_u: 0.0,
    };

    for step in 0..STEPS {
        let t_new = (step as f64 + 1.0) * DT;
        let old_vx = mesh.vx.clone();
        let old_vy = mesh.vy.clone();
        for v in 0..mesh.num_vertices() {
            let (x, y) = vertex_position(x0[v], y0[v], t_new);
            mesh.vx[v] = x;
            mesh.vy[v] = y;
        }
        mesh.recalculate_geometry();
        let swept =
            swept_mesh_fluxes_closed(&mesh, &old_vx, &old_vy, DT).expect("swept mesh fluxes");
        assert!(
            swept.max_identity_err_rel < 1e-12,
            "step {step}: swept-quad identity violated: {:.3e}",
            swept.max_identity_err_rel
        );
        solver
            .begin_ale_step(&mesh, &swept.fluxes)
            .expect("begin_ale_step");
        solver.step();

        let m_n = mass(&mesh);
        out.max_step_drift = out.max_step_drift.max((m_n - m_prev).abs() / m0);
        out.max_total_drift = out.max_total_drift.max((m_n - m0).abs() / m0);
        out.max_area_err = out
            .max_area_err
            .max((mesh.cell_vol.iter().sum::<f64>() - area).abs() / area);
        m_prev = m_n;

        // Boundedness sanity every 10 steps + at the end (readbacks are GPU
        // sync points).
        if step % 10 == 9 || step + 1 == STEPS {
            let u = pollster::block_on(solver.get_field_vec2("U")).expect("read U");
            for (ux, uy) in u {
                assert!(
                    ux.is_finite() && uy.is_finite(),
                    "step {step}: non-finite velocity"
                );
                out.max_u = out.max_u.max(ux.abs()).max(uy.abs());
            }
        }
    }
    out
}

fn print_audit(label: &str, out: &AuditRun) {
    println!(
        "[ale-conservation] {label}: per-step mass drift = {:.3e}, total = {:.3e}, \
         area err = {:.3e}, max|U| = {:.3e} ({STEPS} steps)",
        out.max_step_drift, out.max_total_drift, out.max_area_err, out.max_u
    );
}

/// Caps pinned from measurement (24×24, 120 steps): the area/mass identities
/// are pure f64 shoelace-sum telescopes over a fixed-boundary domain (~5e-15
/// relative; caps carry ~20× headroom against platform accumulation). max|U|
/// is f32-noise-scale spurious motion; cap ~15× the worst backend.
fn assert_conservation_caps(out: &AuditRun) {
    assert!(
        out.max_area_err < 1e-13,
        "Σ V does not track the analytic domain area: {:.3e}",
        out.max_area_err
    );
    assert!(
        out.max_step_drift < 1e-13,
        "per-step Σ rho·V drift above f64 geometry precision: {:.3e}",
        out.max_step_drift
    );
    assert!(
        out.max_total_drift < 1e-13,
        "accumulated Σ rho·V drift above f64 geometry precision: {:.3e}",
        out.max_total_drift
    );
    // Zero-flow free-stream preservation: the exact solution is a still
    // fluid, so ANY velocity is ALE-injected noise. f32 scale, no
    // compounding over 120 steps of ~0.5%-volume-swing deformation.
    assert!(
        out.max_u < 5e-7,
        "spurious ALE-injected velocity above f32-noise scale: max|U| = {:.3e}",
        out.max_u
    );
}

/// CPU, default (inexact-Picard 1e-4) linear tolerance.
#[test]
fn ale_mass_conservation_cpu() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::set_var("CFD2_BACKEND", "cpu");
    let result = std::panic::catch_unwind(|| run_audit(None, None, None));
    std::env::remove_var("CFD2_BACKEND");
    let out = match result {
        Ok(o) => o,
        Err(e) => std::panic::resume_unwind(e),
    };
    print_audit("cpu/default-tol", &out);
    assert_conservation_caps(&out);
}

/// CPU, tight linear tolerance (1e-8) via the model recipe field. The
/// mesh-side caps are tolerance-independent (they must hold identically); this
/// variant pins that the recipe path accepts a tolerance override and the
/// audit is not silently solver-tolerance-shaped.
#[test]
fn ale_mass_conservation_cpu_tight_tol() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::set_var("CFD2_BACKEND", "cpu");
    let result = std::panic::catch_unwind(|| run_audit(None, None, Some(1e-8)));
    std::env::remove_var("CFD2_BACKEND");
    let out = match result {
        Ok(o) => o,
        Err(e) => std::panic::resume_unwind(e),
    };
    print_audit("cpu/tight-tol", &out);
    assert_conservation_caps(&out);
}

/// GPU (skips without an adapter): identical audit, f32 kernels — the
/// mesh-side caps are host-f64 statements and hold regardless of backend.
#[test]
fn ale_mass_conservation_gpu() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::remove_var("CFD2_BACKEND");
    let ctx = match pollster::block_on(cfd2::solver::gpu::context::GpuContext::new(None, None)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[ale-conservation] no GPU adapter ({e}); skipping GPU audit");
            return;
        }
    };
    let out = run_audit(Some(ctx.device.clone()), Some(ctx.queue.clone()), None);
    print_audit("gpu", &out);
    assert_conservation_caps(&out);
}
