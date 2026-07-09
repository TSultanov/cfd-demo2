//! TRANSIENT **ALE (moving-mesh)** Method-of-Manufactured-Solutions temporal-order test for
//! the COMPRESSIBLE thermal all-Mach model (`allmach_thermal_compressible_mms_ale`).
//!
//! This is the MOVING-MESH companion to `mms_allmach_thermal_compressible_transient_test`
//! (static mesh, order ~2). It proves the compressible-thermal ALE transient is fully
//! SECOND-ORDER in time on a genuinely moving mesh — the cross-variable BDF2 ddt terms
//! (thermal expansion `rho_dT*dT/dt` in the pressure row, T1 `inv_cp*dp/dt` compression
//! heating in the energy row; landed `f103ddf`, emitted UNWEIGHTED so they carry the
//! referential `dphi/dt` at V^{n+1} to O(dt^2)) AND the surrounding own-variable ALE terms.
//!
//! WHY IT DID NOT USED TO REACH 2 — and the fix it now guards. A moving mesh caps this at
//! FIRST order unless every row's geometric `phi*dV/dt` cancels at CONSISTENT weights. The
//! momentum row always did (conservative ddt `rho*phi*ale_dvdt_ddt` at BDF2 + `bounded`
//! correction that removes it at the SAME BDF2 rate, cancelling the SCL-rate mesh flux). Two
//! other rows did not, and the fix makes them:
//!  * CONTINUITY: the acoustic own-var `ddt(psi_precond,p)` is now `non_conservative_ale`
//!    (unweighted, so it contributes NO geometric part), and the volume source carries the
//!    FULL per-cell `rho*dV/dt` at the SCL rate (`ale_bounded_density`, not `rho_ref`) — so it
//!    cancels the per-cell mesh flux EXACTLY for a real (thermal-EOS) density. Previously the
//!    conservative acoustic ddt's `psi*p*dV/dt` (BDF2) mismatched the SCL volume source/flux
//!    by O(dt), and the `rho_ref` source mismatched the real `rho` by O(1).
//!  * ENERGY (MMS variant): now uses `bounded` convection on ALE (like production / momentum),
//!    so its geometric `rho*T*dV/dt` cancels at BDF2; the former conservative MMS form left a
//!    first-order `rho*T*(ale_dvdt_ddt - ale_dvdt_scl)` that poisoned rho -> p -> U via the
//!    EOS. Bounded is exact here because the manufactured flow is spatially UNIFORM (div = 0).
//!
//! DESIGN — a merge of two existing instruments:
//!  * PHYSICS from the static compressible transient MMS: a spatially-UNIFORM, time-dependent
//!    manufactured solution (U*(t), T*(t) > t_ref, p*(t) — INDEPENDENT, real thermal EOS
//!    `rho(p,T)`) drives every transient term; every spatial operator is exact for a uniform
//!    field on ANY mesh, so the L2 error at t=1 is PURELY temporal truncation. `precond_mask=1`
//!    enables the acoustic `ddt(psi_precond,p)` (which also pins the otherwise-Neumann all-wall
//!    pressure gauge). Sources = the sum of each row's ddt terms at the exact solution.
//!  * MOTION from the incompressible ALE temporal template (`mms_ale_order_test`): the mesh
//!    vertices oscillate through a smooth bump at FIXED amplitude (amp ∝ h; the bump is zero on
//!    the boundary, so the domain walls stay fixed and only the interior deforms) while dt
//!    refines, so the mesh PATH is the same curve sampled finer. Per step: move vertices ->
//!    `recalculate_geometry` -> swept-quad SCL fluxes -> `begin_ale_step` -> re-upload the
//!    source at t^{n+1} (history-preserving `set_field_*_current`) -> `step`.
//!
//! All four fields gate at ~2 (U inherits 2nd order through the `rho(p,T)` coefficient of its
//! own BDF2 ddt; rho is a nonlinear recovery of the two 2nd-order primitives). Before the ALE
//! discretization fix the SAME manufactured solution collapsed to order ~0.1 on a moving mesh.
#![cfg(all(feature = "dev-tests", feature = "ui"))]

mod mms_support;

use std::f64::consts::PI;

use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::mesh::{
    generate_structured_rect_mesh, swept_mesh_fluxes_closed, BoundarySides, BoundaryType,
};
use cfd2::solver::model::helpers::{SolverFieldAliasesExt, SolverRuntimeParamsExt};
use cfd2::solver::model::{
    allmach_thermal_compressible_mms_ale_model, ALLMACH_GAMMA, ALLMACH_MMS_SOURCE_P_FIELD,
    ALLMACH_MMS_SOURCE_T_FIELD, ALLMACH_MMS_SOURCE_U_FIELD, ALLMACH_T_REF,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};

use mms_support::{assert_convergence_order, field_errors, field_errors_vec2, fit_order};

// ── physics constants + manufactured solution (verbatim from the static transient test) ──
const GAMMA: f64 = ALLMACH_GAMMA;
const T_REF: f64 = ALLMACH_T_REF; // = 1.0
const RHO_REF: f64 = 1.0;
const RHO_T_REF: f64 = RHO_REF * T_REF; // = 1.0
const PSI: f64 = 0.1;
const T_END: f64 = 1.0;

const UX0: f64 = 0.5;
const UY0: f64 = -0.3;
const U_DECAY: f64 = 0.7;
// Independent, O(1)-varying T*(t) > t_ref (so the on-device local psi clamps to psi_ref) and
// p*(t) — the SAME manufactured solution as the static transient MMS. It exercises every
// transient term (variable-density inertia, dp/dt heating, thermal expansion, acoustic ddt)
// well above the f32 floor and — unlike a barotropic-constrained T(p) — keeps dT/dt and dp/dt
// INDEPENDENT, so the thermal-expansion and compression-heating cross terms are both driven
// hard. With the 2nd-order moving-mesh ALE discretization this converges cleanly at order 2
// for a REAL (thermal-EOS) density; before the fix the same solution collapsed to order ~0.
const T_A: f64 = 1.4;
const T_B: f64 = 0.35;
const T_W: f64 = 1.5;
const P_A: f64 = 2.0;
const P_W: f64 = 1.2;
const P_PH: f64 = 0.5;
const U_REF: f64 = 10.0;

// ── mesh-motion constants (from the incompressible ALE temporal template) ──
/// Bump amplitude as a fraction of h — keeps mesh distortion (and the ALE terms)
/// proportionally constant across dt levels, so the sampled mesh path is the same curve.
const AMP_FRAC: f64 = 0.2;
const MOTION_PERIOD: f64 = 1.0;
const NX: usize = 8;
const NY: usize = 8;
const LX: f64 = 1.0;
const LY: f64 = 1.0;

// Finest-dt (128-step) error caps — ~2x the observed values (measured U=4.6e-6, T=5.4e-6,
// p=6.3e-5, rho=6.1e-6), a coefficient-regression guard on top of the order gate.
const CAP_U: f64 = 1.0e-5;
const CAP_T: f64 = 1.1e-5;
const CAP_P: f64 = 1.3e-4;
const CAP_RHO: f64 = 1.3e-5;

/// Diagnostic env override for the motion amplitude fraction (default `AMP_FRAC`).
/// `CFD2_ALE_MMS_AMP=0` freezes the mesh → isolates the moving-mesh error from the
/// static-solve baseline (which should match `mms_allmach_thermal_compressible_transient`).
fn amp_frac() -> f64 {
    std::env::var("CFD2_ALE_MMS_AMP")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(AMP_FRAC)
}
/// Diagnostic env override for the dt-sweep step counts (default `[16,32,64,128]`).
fn step_counts() -> Vec<usize> {
    std::env::var("CFD2_ALE_MMS_STEPS")
        .ok()
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![16, 32, 64, 128])
}

fn exact_u(t: f64) -> (f64, f64) {
    let a = (-U_DECAY * t).exp();
    (UX0 * a, UY0 * a)
}
fn dudt(t: f64) -> (f64, f64) {
    let da = -U_DECAY * (-U_DECAY * t).exp();
    (UX0 * da, UY0 * da)
}
fn exact_t(t: f64) -> f64 {
    T_A + T_B * (T_W * t).cos()
}
fn dtdt(t: f64) -> f64 {
    -T_B * T_W * (T_W * t).sin()
}
fn exact_p(t: f64) -> f64 {
    P_A * (P_W * t + P_PH).sin()
}
fn dpdt(t: f64) -> f64 {
    P_A * P_W * (P_W * t + P_PH).cos()
}
/// Real-EOS density, matching the on-device recovery
/// `rho = rho_t_ref/T + gamma*psi_ref*t_ref*p/T` (psi_ref = PSI, t_ref = T_REF).
fn exact_rho(t: f64) -> f64 {
    let tt = exact_t(t);
    RHO_T_REF / tt + GAMMA * PSI * T_REF * exact_p(t) / tt
}
/// Thermal-expansion coefficient `rho_dT = d(rho)/dT = -(rho_t_ref + gamma*psi_ref*t_ref*p)/T^2`,
/// matching the device recovery.
fn exact_rho_dt(t: f64) -> f64 {
    let tt = exact_t(t);
    -(RHO_T_REF + GAMMA * PSI * T_REF * exact_p(t)) / (tt * tt)
}

/// Momentum: `ddt(rho, U)` => `S_U = rho(t) * dU*/dt(t)`.
fn source_u(t: f64) -> (f64, f64) {
    let r = exact_rho(t);
    let (dx, dy) = dudt(t);
    (r * dx, r * dy)
}
/// Continuity: `ddt(psi_precond, p)` [acoustic, BDF2] + `ddt(rho_dT, T)` [thermal expansion,
/// cross] => `S_p = psi_precond*dp/dt + rho_dT*dT/dt`, with psi_precond = PSI.
fn source_p(t: f64) -> f64 {
    PSI * dpdt(t) + exact_rho_dt(t) * dtdt(t)
}
/// Energy: `ddt(rho, T)` [BDF2] + T1 `ddt(inv_cp, p)` [cross] with
/// `inv_cp = -(gamma-1)*T_ref*psi_ref` => `S_T = rho*dT/dt - (gamma-1)*T_ref*PSI*dp/dt`.
fn source_t(t: f64) -> f64 {
    let inv_cp = -(GAMMA - 1.0) * T_REF * PSI;
    exact_rho(t) * dtdt(t) + inv_cp * dpdt(t)
}

/// One dt level of the moving-mesh temporal study: march to T_END with `steps` uniform
/// steps on an OSCILLATING mesh, return the (U, T, p, rho) L2 errors at t = T_END.
fn solve_moving(steps: usize) -> (f64, f64, f64, f64) {
    let dt = T_END / steps as f64;
    let mut mesh = generate_structured_rect_mesh(
        NX,
        NY,
        LX,
        LY,
        BoundarySides {
            left: BoundaryType::MovingWall,
            right: BoundaryType::MovingWall,
            bottom: BoundaryType::MovingWall,
            top: BoundaryType::MovingWall,
        },
    );
    let x0 = mesh.vx.clone();
    let y0 = mesh.vy.clone();
    let h = LX / NX as f64;
    let n = mesh.num_cells();

    let mut model =
        allmach_thermal_compressible_mms_ale_model().expect("compressible ale mms model");
    // Drive each implicit step below the finest truncation floor (Picard/linear lag would
    // otherwise cap the sweep). Backend-portable via the model recipe.
    if let Some(ls) = model.linear_solver.as_mut() {
        ls.solver.tolerance = 1e-7;
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

    solver.set_dt(dt as f32);
    solver.set_dtau(0.0).expect("dtau");
    solver.set_density(RHO_REF as f32).expect("density");
    solver.set_viscosity(1.0).expect("viscosity");
    solver.set_alpha_u(0.7).expect("alpha_u");
    solver.set_alpha_p(0.3).expect("alpha_p");
    solver.set_outer_iters(50).expect("outer_iters");

    // Constant EOS / preconditioner fields (seeded once as the step-0 guess; recovered
    // on-device each outer). precond_mask = 1 ENABLES the acoustic ddt(psi_precond,p).
    for (name, v) in [
        ("psi_ref", PSI),
        ("psi", PSI),
        ("psi_precond", PSI),
        ("t_ref", T_REF),
        ("rho_t_ref", RHO_T_REF),
        ("rho_floor", PSI * 1.0e-5),
        ("u_ref", U_REF),
        ("precond_mask", 1.0),
        ("rho_dT", exact_rho_dt(0.0)),
        ("u_dot_grad_p", 0.0),
    ] {
        solver
            .set_field_scalar(name, &vec![v; n])
            .unwrap_or_else(|_| panic!("seed {name}"));
    }

    // Initialise at the exact solution at t = 0 (all BDF history levels).
    solver.set_u(&vec![exact_u(0.0); n]);
    solver.set_p(&vec![exact_p(0.0); n]);
    solver
        .set_field_scalar("T", &vec![exact_t(0.0); n])
        .expect("T init");
    solver
        .set_field_scalar("rho", &vec![exact_rho(0.0); n])
        .expect("rho init");
    solver.initialize_history();

    let bump = |xv: f64, yv: f64| (PI * xv / LX).sin().powi(2) * (PI * yv / LY).sin().powi(2);

    for step in 0..steps {
        let t = (step as f64 + 1.0) * dt; // implicit: sources / BCs / motion at the new level
        let old_vx = mesh.vx.clone();
        let old_vy = mesh.vy.clone();
        let amp = amp_frac() * h * (2.0 * PI * t / MOTION_PERIOD).sin();
        for v in 0..mesh.num_vertices() {
            let b = bump(x0[v], y0[v]);
            mesh.vx[v] = x0[v] + amp * b;
            mesh.vy[v] = y0[v] - 0.6 * amp * b;
        }
        mesh.recalculate_geometry();
        let swept =
            swept_mesh_fluxes_closed(&mesh, &old_vx, &old_vy, dt).expect("swept mesh fluxes");
        assert!(
            swept.max_identity_err_rel < 1e-10 && swept.max_defect_rel < 1e-7,
            "[ale-compressible] SCL closure broke at step {step}: identity {:.3e} defect {:.3e}",
            swept.max_identity_err_rel,
            swept.max_defect_rel
        );
        solver
            .begin_ale_step(&mesh, &swept.fluxes)
            .expect("begin_ale_step");

        // Time-varying manufactured sources at the new (implicit) time level — history-
        // preserving upload (the plain setters have IC semantics and would clobber BDF2 history).
        solver
            .set_field_vec2_current(ALLMACH_MMS_SOURCE_U_FIELD, &vec![source_u(t); n])
            .expect("src U");
        solver
            .set_field_scalar_current(ALLMACH_MMS_SOURCE_P_FIELD, &vec![source_p(t); n])
            .expect("src p");
        solver
            .set_field_scalar_current(ALLMACH_MMS_SOURCE_T_FIELD, &vec![source_t(t); n])
            .expect("src T");

        // Uniform MovingWall Dirichlet tracks the exact solution at the new time level.
        let (ux, uy) = exact_u(t);
        solver
            .set_boundary_vec2(GpuBoundaryType::MovingWall, "U", [ux as f32, uy as f32])
            .expect("wall U");
        solver
            .set_boundary_scalar(GpuBoundaryType::MovingWall, "T", exact_t(t) as f32)
            .expect("wall T");

        solver.step();
    }

    let u = pollster::block_on(solver.get_field_vec2("U")).expect("U");
    let p = pollster::block_on(solver.get_p());
    let tf = pollster::block_on(solver.get_field_scalar("T")).expect("T");
    let rho = pollster::block_on(solver.get_field_scalar("rho")).expect("rho");

    let (uex, uey) = exact_u(T_END);
    let ue = field_errors_vec2(&mesh, &u, |_, _| (uex, uey));
    let te = field_errors(&mesh, &tf, |_, _| exact_t(T_END));
    let pe = field_errors(&mesh, &p, |_, _| exact_p(T_END));
    let re = field_errors(&mesh, &rho, |_, _| exact_rho(T_END));
    (ue.l2, te.l2, pe.l2, re.l2)
}

/// TEMPORAL order of the compressible thermal ALE transient on a MOVING mesh — the whole
/// coupled row set (cross-variable BDF2 ddt + the 2nd-order ALE geometric terms) must stay
/// 2nd-order-accurate under genuine mesh motion. Gates ALL FOUR fields at ~2. A regression of
/// any of the moving-mesh fixes (`non_conservative_ale` acoustic ddt, per-cell continuity
/// volume source, ALE-MMS bounded energy, or the cross-var BDF2 itself) drops the order back
/// toward 1 and trips these floors.
#[test]
fn allmach_thermal_compressible_ale_transient_order() {
    let step_counts = step_counts();
    let mut dts = Vec::new();
    let mut u_err = Vec::new();
    let mut t_err = Vec::new();
    let mut p_err = Vec::new();
    let mut r_err = Vec::new();

    for &steps in &step_counts {
        let (ue, te, pe, re) = solve_moving(steps);
        let dt = T_END / steps as f64;
        println!(
            "[ale-compressible-transient] steps={steps:>3} dt={dt:.4}  \
             L2(U)={ue:.4e}  L2(T)={te:.4e}  L2(p)={pe:.4e}  L2(rho)={re:.4e}"
        );
        dts.push(dt);
        u_err.push(ue);
        t_err.push(te);
        p_err.push(pe);
        r_err.push(re);
    }

    let u_order = fit_order(&dts, &u_err);
    let t_order = fit_order(&dts, &t_err);
    let p_order = fit_order(&dts, &p_err);
    let r_order = fit_order(&dts, &r_err);
    println!(
        "[ale-compressible-transient] temporal order: U={u_order:.3} T={t_order:.3} \
         p={p_order:.3} rho={r_order:.3}  (all gated ~2)"
    );

    // Monotone convergence pins every transient term's sign/magnitude (a wrong sign diverges,
    // a dropped/frozen term flattens the sweep).
    for (name, errs) in [("U", &u_err), ("T", &t_err), ("p", &p_err), ("rho", &r_err)] {
        for w in errs.windows(2) {
            assert!(
                w[1] < w[0],
                "[ale-compressible] {name} did not converge under dt-refinement: {errs:?}"
            );
        }
    }

    // ALL FOUR fields converge at ~2 now that every moving-mesh row is 2nd-order-consistent:
    // continuity (`non_conservative_ale` acoustic ddt + per-cell SCL volume source), energy
    // (ALE-MMS bounded convection), momentum (always bounded), and the cross-variable BDF2
    // ddt. Measured U~2.02, T~2.05, p~2.02, rho~1.94 (rho slightly under 2 — a nonlinear
    // recovery `rho = rho_t_ref/T + gamma*psi_ref*t_ref*p/T` of the two 2nd-order primitives).
    // Floors sit well below the observed values to absorb GPU/backend fit jitter while cleanly
    // rejecting the pre-fix first-order (~1) or free-stream-broken (~0) regressions.
    assert_convergence_order("ale_compressible_U", &dts, &u_err, 2.0, 0.4, CAP_U);
    assert_convergence_order("ale_compressible_T", &dts, &t_err, 2.0, 0.4, CAP_T);
    assert_convergence_order("ale_compressible_p", &dts, &p_err, 2.0, 0.4, CAP_P);
    assert_convergence_order("ale_compressible_rho", &dts, &r_err, 2.0, 0.4, CAP_RHO);
}
