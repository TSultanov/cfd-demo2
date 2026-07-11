//! TRANSIENT Method-of-Manufactured-Solutions temporal-order test for the
//! COMPRESSIBLE thermal all-Mach model (`allmach_thermal_compressible_mms`).
//!
//! The steady companion (`mms_allmach_thermal_compressible_order_test`) verifies the
//! SPATIAL compressible operator (real-EOS rho(p,T), viscous dissipation Phi, the
//! U.grad(p) heating). It CANNOT touch the terms that are identically zero at steady
//! state — the genuinely TRANSIENT couplings:
//!   * momentum   ddt(rho, U)                 — variable-density inertia          (BDF2, own-var)
//!   * energy     ddt(rho, T)                 — variable-density thermal inertia  (BDF2, own-var)
//!   * energy     ddt(inv_cp, p) = T1         — dp/dt compression heating         (BDF2, cross-var)
//!   * continuity ddt(psi_precond, p)         — the acoustic pseudo-compressibility(BDF2, own-var)
//!   * continuity ddt(rho_dT, T)              — rho_dT*dT/dt thermal expansion     (BDF2, cross-var)
//! This test drives all of them with a SPATIALLY-UNIFORM, time-dependent manufactured
//! solution and refines dt, measuring the temporal convergence order.
//!
//! Why spatially uniform:
//!   * Every SPATIAL operator (convection, diffusion, grad p, div, Phi, U.grad p) is
//!     identically zero for a uniform field on ANY mesh, so the residual is PURELY
//!     temporal — a clean temporal-order instrument (the same design as the incompressible
//!     `ale_bdf2_moving_volume_temporal_order` template) — AND the spatial error is ~0,
//!     so a tiny mesh suffices (fast).
//!   * It SIDESTEPS the steady test's pressure-Lagrange-multiplier limitation: with the
//!     acoustic `ddt(psi_precond,p)` term ENABLED (`precond_mask = 1`), the all-wall
//!     pressure row is NON-singular (a `psi_precond*V/dt` diagonal on every cell), so the
//!     uniform p(t) is uniquely pinned by its own ddt — no free gauge, no p<->rho shape
//!     defect.
//!
//! Expected order: both own-variable and cross-variable ddts use the variable-step BDF2
//! stencil. Cross terms write the current coefficient to the off-diagonal block and use
//! both history levels, so the fully coupled transient should retain second order. The
//! gate asserts monotone convergence, near-second-order fits, and tight finest-step caps.
#![cfg(all(feature = "dev-tests", feature = "ui"))]

mod mms_support;

use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType, Mesh};
use cfd2::solver::model::helpers::{SolverFieldAliasesExt, SolverRuntimeParamsExt};
use cfd2::solver::model::{
    allmach_thermal_compressible_mms_model, ALLMACH_GAMMA, ALLMACH_MMS_SOURCE_P_FIELD,
    ALLMACH_MMS_SOURCE_T_FIELD, ALLMACH_MMS_SOURCE_U_FIELD, ALLMACH_T_REF,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};

use mms_support::{field_errors, field_errors_vec2, fit_order};

const GAMMA: f64 = ALLMACH_GAMMA;
const T_REF: f64 = ALLMACH_T_REF; // = 1.0
const RHO_REF: f64 = 1.0;
const RHO_T_REF: f64 = RHO_REF * T_REF; // = 1.0
/// Compressibility ON: a healthy pressure<->density coupling so the acoustic ddt, the
/// T1 dp/dt heating and the thermal-expansion cross-term are all exercised well above
/// the f32 noise floor. psi_ref = 1/c_ref^2; the local psi clamps to psi_ref because
/// T > t_ref everywhere here (see the manufactured T below), so the target Schur
/// compressibility is PSI. The raw pressure-row coefficient also includes the
/// thermal cross contribution `(gamma-1)*PSI*T_ref/T`.
const PSI: f64 = 0.1;
const T_END: f64 = 1.0;

// ── manufactured solution (spatially UNIFORM, time-dependent) ────────────────
// Chosen smooth with nonzero 1st/2nd time-derivatives (so BDF2 truncation is exercised)
// and T(t) > t_ref = 1 everywhere (so the on-device local psi clamps to psi_ref and the
// post-temperature-elimination acoustic coefficient is the constant PSI).

const UX0: f64 = 0.5;
const UY0: f64 = -0.3;
const U_DECAY: f64 = 0.7;
const T_A: f64 = 1.4;
const T_B: f64 = 0.35;
// Gentle temporal frequencies (~1/4 period over [0,1]): the cross-variable BDF1
// truncation of `rho_dT*dT/dt` scales with d2T/dt2 ~ T_B*T_W^2, so a large T_W throws the
// coarse-dt points out of the asymptotic regime and depresses the fitted order. Keeping
// T_W/P_W moderate makes every dt level asymptotic for the BDF2 fit.
const T_W: f64 = 1.5;
const P_A: f64 = 2.0;
const P_W: f64 = 1.2;
const P_PH: f64 = 0.5;
/// Large velocity floor for the preconditioner so `1/max(|U|^2,u_ref^2) = 1/u_ref^2`
/// is well below PSI: the target Schur coefficient is max(psi=PSI, 1/u_ref^2) = PSI.
const U_REF: f64 = 10.0;

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
/// rho = rho_t_ref/T + gamma*psi_ref*t_ref*p/T   (psi_ref = PSI, t_ref = T_REF).
fn exact_rho(t: f64) -> f64 {
    let tt = exact_t(t);
    RHO_T_REF / tt + GAMMA * PSI * T_REF * exact_p(t) / tt
}
/// Thermal-expansion coefficient rho_dT = d(rho)/dT = -(rho_t_ref + gamma*psi_ref*t_ref*p)/T^2,
/// matching the device recovery.
fn exact_rho_dt(t: f64) -> f64 {
    let tt = exact_t(t);
    -(RHO_T_REF + GAMMA * PSI * T_REF * exact_p(t)) / (tt * tt)
}

// ── manufactured sources ──────────────────────────────────────────────────────
// All spatial operators vanish for a uniform field, so each equation's source is the
// SUM of its ddt-term values at the exact solution (the +c*dphi/dt convention the
// incompressible ale temporal template establishes: it uploads rho*dU/dt = -RHO*U*).

/// Momentum: ddt(rho, U)  =>  S_U = rho*(t) * dU*/dt(t).
fn source_u(t: f64) -> (f64, f64) {
    let r = exact_rho(t);
    let (dx, dy) = dudt(t);
    (r * dx, r * dy)
}
/// Raw pressure row: ddt(psi_precond, p) [acoustic, BDF2] + ddt(rho_dT, T)
/// [thermal expansion, BDF2 cross]. The raw coefficient includes the
/// temperature-row Schur contribution so elimination leaves target `PSI`.
fn source_p(t: f64) -> f64 {
    let psi_precond = PSI + (GAMMA - 1.0) * PSI * T_REF / exact_t(t);
    psi_precond * dpdt(t) + exact_rho_dt(t) * dtdt(t)
}
/// Energy: ddt(rho, T) [BDF2] + T1 ddt(inv_cp, p) [BDF2 cross] with
/// inv_cp = -(gamma-1)*T_ref*psi_ref  =>  S_T = rho*dT/dt - (gamma-1)*T_ref*PSI*dp/dt.
fn source_t(t: f64) -> f64 {
    let inv_cp = -(GAMMA - 1.0) * T_REF * PSI;
    exact_rho(t) * dtdt(t) + inv_cp * dpdt(t)
}

fn mesh() -> Mesh {
    // A uniform field carries zero spatial error on any mesh, so a small box is enough
    // (the error we measure is purely temporal). All-MovingWall: no in/outflow, and the
    // acoustic ddt pins the otherwise-Neumann pressure gauge.
    generate_structured_rect_mesh(
        8,
        8,
        1.0,
        1.0,
        BoundarySides {
            left: BoundaryType::MovingWall,
            right: BoundaryType::MovingWall,
            bottom: BoundaryType::MovingWall,
            top: BoundaryType::MovingWall,
        },
    )
}

/// One dt level: march to T_END with `steps` uniform steps, return the L2 field errors.
fn solve(steps: usize) -> (f64, f64, f64, f64) {
    let dt = T_END / steps as f64;
    let mesh = mesh();
    let n = mesh.num_cells();
    let model = allmach_thermal_compressible_mms_model().expect("compressible mms model");
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
    // A temporal-order instrument must drive each implicit step BELOW the finest
    // truncation error, or per-step Picard/relaxation lag floors the sweep. The
    // compressible rho(p,T) recovery is a per-outer Picard coupling, so it needs more
    // outers than the incompressible template to reach the truncation floor at fine dt.
    solver.set_outer_iters(50).expect("outer_iters");

    // Constant EOS / preconditioner fields (seeded once). precond_mask = 1 ENABLES the
    // acoustic ddt(psi_precond,p) — the term this test exists to exercise (and what pins
    // the all-wall pressure gauge). The recovered fields (rho, rho_dT, psi, psi_precond,
    // u_dot_grad_p) are overwritten on-device each outer iteration; seed them only as
    // the step-0 initial guess.
    for (name, v) in [
        ("psi_ref", PSI),
        ("psi", PSI),
        (
            "psi_precond",
            PSI + (GAMMA - 1.0) * PSI * T_REF / exact_t(0.0),
        ),
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

    // Initialise at the exact solution at t = 0.
    solver.set_u(&vec![exact_u(0.0); n]);
    solver.set_p(&vec![exact_p(0.0); n]);
    solver
        .set_field_scalar("T", &vec![exact_t(0.0); n])
        .expect("T init");
    solver
        .set_field_scalar("rho", &vec![exact_rho(0.0); n])
        .expect("rho init");
    solver.initialize_history();

    for step in 0..steps {
        let t = (step as f64 + 1.0) * dt; // implicit: sources & BCs at the NEW time level
        solver
            .set_field_vec2_current(ALLMACH_MMS_SOURCE_U_FIELD, &vec![source_u(t); n])
            .expect("src U");
        solver
            .set_field_scalar_current(ALLMACH_MMS_SOURCE_P_FIELD, &vec![source_p(t); n])
            .expect("src p");
        solver
            .set_field_scalar_current(ALLMACH_MMS_SOURCE_T_FIELD, &vec![source_t(t); n])
            .expect("src T");

        // Uniform Dirichlet walls track the exact solution at the new time level.
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

    let (uex, _) = exact_u(T_END);
    let (_, uey) = exact_u(T_END);
    let ue = field_errors_vec2(&mesh, &u, |_, _| (uex, uey));
    let te = field_errors(&mesh, &tf, |_, _| exact_t(T_END));
    let pe = field_errors(&mesh, &p, |_, _| exact_p(T_END));
    let re = field_errors(&mesh, &rho, |_, _| exact_rho(T_END));
    (ue.l2, te.l2, pe.l2, re.l2)
}

/// Temporal convergence of the coupled COMPRESSIBLE transient operator. Verifies the
/// transient terms (variable-density inertia, dp/dt heating, thermal expansion, acoustic
/// pseudo-compressibility) are correctly signed and scaled by driving them to the
/// manufactured solution and refining dt.
///
/// The observed order is SECOND (~2) on every field: `time_integration.rs` now emits BOTH
/// the own-variable ddt (rho*U, rho*T, psi_precond*p) AND the two CROSS-variable ddt
/// couplings (thermal expansion `rho_dT*dT/dt` in the pressure row, T1 `inv_cp*dp/dt` in the
/// energy row) as BDF2 — the cross-variable branch writes directly to the off-diagonal
/// `matrix_values` entry and carries the same variable-dt BDF2 stencil (2r+1)/(r+1) plus a
/// two-level `state_old_old` history, WITHOUT the diagonal's conservative `ale_vol_ratio`
/// weighting (which would break free-stream on a moving mesh). This lifted the coupled
/// transient from the former BDF1-cross-coupling cap (~0.95) to full second order. The point
/// of THIS test is transient-term CORRECTNESS (monotone convergence at the designed 2nd-order
/// rate — a BDF1 regression of either cross term would drop the order back to ~1 and trip the
/// floors below), complementing the steady test which covers the spatial operator.
#[test]
fn allmach_thermal_compressible_transient_order() {
    let step_counts = [16usize, 32, 64, 128];
    let mut dts = Vec::new();
    let mut u_err = Vec::new();
    let mut t_err = Vec::new();
    let mut p_err = Vec::new();
    let mut r_err = Vec::new();

    for &steps in &step_counts {
        let (ue, te, pe, re) = solve(steps);
        let dt = T_END / steps as f64;
        println!(
            "[compressible-transient] steps={steps:>3} dt={dt:.4}  \
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
        "[compressible-transient] temporal order: U={u_order:.3} T={t_order:.3} \
         p={p_order:.3} rho={r_order:.3}"
    );

    // PRIMARY correctness gate — monotone temporal convergence. Every field falls under
    // dt-refinement across the full 8x range. This is what pins each transient term's SIGN
    // and coefficient: a wrong sign diverges, and a wrong magnitude converges to the wrong
    // field (which the finest-error caps below then catch).
    for (name, errs) in [("U", &u_err), ("T", &t_err), ("p", &p_err), ("rho", &r_err)] {
        for w in errs.windows(2) {
            assert!(
                w[1] < w[0],
                "[compressible-transient] {name} did not converge under dt-refinement: {errs:?}"
            );
        }
    }

    // Observed temporal order is SECOND now that BOTH cross-variable ddt couplings (thermal
    // expansion `rho_dT*dT/dt` in the pressure row, T1 `inv_cp*dp/dt` in the energy row) are
    // emitted as BDF2 off-diagonal terms alongside the own-variable BDF2 ddt. Measured on
    // this instrument: U~2.09, T~2.25, p~2.05, rho~1.91. The pressure/temperature — produced
    // DIRECTLY by the transient terms — sit cleanly above 2; density is a nonlinear recovery
    // `rho = rho_t_ref/T + gamma*psi_ref*t_ref*p/T` of two now-2nd-order primitives, so its
    // fit sits just under 2 (~1.91) at these step counts; velocity inherits 2nd order through
    // the `rho(p,T)` coefficient of its own BDF2 ddt. A BDF1 regression of EITHER cross term
    // (dropping the `state_old_old` history / the (2r+1)/(r+1) matrix scale) collapses the
    // order back to ~0.95 and trips this floor. Floor set well below the observed values to
    // absorb GPU/backend fit jitter while still rejecting the first-order regression.
    for (name, order) in [
        ("U", u_order),
        ("T", t_order),
        ("p", p_order),
        ("rho", r_order),
    ] {
        assert!(
            order >= 1.6,
            "[compressible-transient] {name} temporal order {order:.3} below 1.6 \
             (a cross-variable ddt regressed from BDF2 to BDF1 — expected ~2)"
        );
    }
    // Pressure and temperature are produced DIRECTLY by the transient terms (acoustic ddt, T1,
    // thermal expansion, variable-density thermal inertia) with no nonlinear-recovery softening,
    // so their fits are asymptotically clean; hold them to a tighter near-2 bound.
    for (name, order) in [("T", t_order), ("p", p_order)] {
        assert!(
            order >= 1.8,
            "[compressible-transient] {name} temporal order {order:.3} below 1.8 \
             (a transient ddt term regressed from clean 2nd order)"
        );
    }

    // Coefficient-regression guard: a transient term with the right sign but a wrong
    // magnitude still converges monotonically (to the wrong field), inflating the finest-dt
    // error even at 2nd order. Cap it at ~2x the observed finest error (128 steps, measured
    // U=6.4e-7 T=5.0e-6 p=6.5e-5 rho=6.2e-6 — an order of magnitude tighter than the former
    // BDF1-era caps, now that the cross-variable ddt is BDF2).
    let finest = (
        *u_err.last().unwrap(),
        *t_err.last().unwrap(),
        *p_err.last().unwrap(),
        *r_err.last().unwrap(),
    );
    assert!(
        finest.0 < 1.3e-6 && finest.1 < 1.1e-5 && finest.2 < 1.4e-4 && finest.3 < 1.3e-5,
        "[compressible-transient] finest-dt errors too large: U={:.3e} T={:.3e} p={:.3e} rho={:.3e}",
        finest.0, finest.1, finest.2, finest.3
    );
}
