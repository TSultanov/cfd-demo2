//! Compressible MMS: manufactured (rho, u, p) for the coupled compressible
//! Navier–Stokes path (KT central-upwind flux with vanLeer reconstruction,
//! implicit mu/kappa laplacians, EOS recovery rows) — the first MMS oracle
//! for the conservative-flux operator family.
//!
//! Setup: unit box, all four sides Inlet (subsonic inflow everywhere: the
//! manufactured velocity points inward on every boundary, which the
//! continuity source balances). Inlet BCs prescribe rho and u per face from
//! the exact solution; the model's declared inlet expressions keep
//! rho_u/rho_e/p/T consistent with the interior pressure. The manufactured
//! pressure has zero normal derivative on all boundaries, so the inlet
//! "p follows interior" closure is second-order consistent.
//!
//! THE OPERATOR CONTRACT THIS TEST PINS: physical Navier-Stokes. The
//! momentum viscous operator is `div(tau)` with
//! `tau = mu (grad u + grad u^T - 2/3 I div u)`, split rhoCentralFoam-style
//! between the implicit `laplacian(mu, u)` and the explicit transpose-only
//! `tauMC` traction in the flux (`tau_mc_dot_n_components` in
//! flux_schemes.rs); the energy viscous work flux is `tau . u`. The
//! conduction coefficient lowers to `mu cp / 0.71` (hardcoded Prandtl).
//! Sources are derived for that operator (`EXTRA_SHEAR = 0`).
//!
//! HISTORY (June 2026): tauMC was originally built as the FULL stress (its
//! comment misread OpenFOAM's `dev2(T(grad U))`, which has only ONE
//! gradient term), so combined with the assembled laplacian the effective
//! operator was `div(tau) + mu lap(u)` - shear viscosity DOUBLED - with an
//! extra `(mu/2) grad(|u|^2)` in the energy work flux. This oracle proved
//! it in both directions at MU = 0.05, Re ~ 8: doubled-shear sources
//! converged at design order (rho 1.94 / u 2.19 / p 1.86 / T 1.79) while
//! physical-NS sources saturated h-independently at 30-70x (rho 4.6e-2,
//! u 1.5e-2 at both n=16 and n=32). After the tauMC fix the roles swap:
//! this test asserts order-2 convergence with physical-NS sources, and the
//! `#[ignore]` probe (doubled-shear sources, `EXTRA_SHEAR = 1`) saturates -
//! it guards against reintroducing the double-counted laplacian. See the FD
//! cross-check test for source verification independent of the
//! hand-derived partials.

#![cfg(feature = "dev-tests")]

mod mms_support;

use std::f64::consts::PI;

use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType, Mesh};
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::model::helpers::SolverRuntimeParamsExt;
use cfd2::solver::model::{
    compressible_mms_model, COMPRESSIBLE_MMS_SOURCE_RHO_E_FIELD,
    COMPRESSIBLE_MMS_SOURCE_RHO_FIELD, COMPRESSIBLE_MMS_SOURCE_RHO_U_FIELD,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};

use mms_support::{assert_convergence_order, field_errors, field_errors_vec2};

// Nondimensional gas: c = sqrt(gamma R T) ~ 1.2, |u| <= ~0.7 (subsonic).
const GAMMA: f64 = 1.4;
const R_GAS: f64 = 1.0;
/// The assembly lowers the declared `laplacian(kappa, T)` coefficient to
/// `mu * gamma * R / (gamma - 1) / 0.71` (= mu cp / Pr with Pr = 0.71).
const PRANDTL: f64 = 0.71;
/// Viscosity of the Navier-Stokes study (Re ~ 8). Conduction k = mu cp / Pr
/// tracks mu in the sources for both studies.
const MU: f64 = 0.05;
/// Euler-dominated study: smallest STABLE viscous floor (Re ~ 112; the
/// convective error dominates the budget ~6x). Probed June 2026: mu = 0
/// diverges at n=48/dt=0.01 and grows secularly at every dt; the growth
/// rate rises with n and is bounded only by physical damping (see the
/// inviscid-margin instability record in the test docs below).
const MU_EULER: f64 = 5.0e-3;
/// CFL ~ 0.5 at n=32 for the Euler study (the n=48/dt=0.01 blow-up is the
/// fast branch of the inviscid-margin instability).
const DT_EULER: f64 = 0.005;

/// 0.0 = sources for physical NS (the operator since the tauMC fix).
/// 1.0 = sources for the pre-fix doubled-shear operator (full-stress tauMC
/// + assembly laplacian) - used by the regression probe.
const EXTRA_SHEAR: f64 = 0.0;

const U0: f64 = 0.4;
const V0: f64 = 0.3;
const UA: f64 = 0.15;
const UB: f64 = 0.1;
const RHO0: f64 = 1.0;
const RHOA: f64 = 0.15;
const PHX: f64 = 0.4;
const PHY: f64 = 0.3;
const P0: f64 = 1.0;
const PA: f64 = 0.2;

// Steady-marching policy: BDF2 pseudo-marching. The vanLeer-limited flux
// sustains a small steady-state limit cycle (the classic TVD-limiter
// convergence stall): the per-step delta decays to a floor whose RATE is
// dt-independent (~3.6e-3 per time unit measured at dt = 0.005 and 0.02,
// unchanged by outer_iters 1 vs 2, persisting to t = 16 ~ 10 physical time
// constants). The runner therefore stops on a delta PLATEAU (no improvement
// over a window) rather than an absolute tolerance; the cycle's state
// amplitude is far below discretization error (validated by the error-drift
// check inside the order study).
const DT: f64 = 0.01;
const OUTER_ITERS: usize = 1;
const STEADY_TOL: f64 = 1e-5;
const STEADY_MAX_STEPS: usize = 1600;
/// Plateau detection: stop when the best (smallest) max-delta seen has not
/// improved by >2% within this many steps.
const PLATEAU_WINDOW: usize = 80;
/// No stop criterion may fire before this many steps: starting from the
/// exact solution the state still has to travel to the discrete fixed point,
/// and the slowest (thermal, tau ~ 1.4 time units) mode needs ~4 tau to
/// settle. Measured: at n=48 the absolute tol alone fired at t=0.73 with the
/// rho error still drifting 22% per +1 time unit.
const MIN_STEPS: usize = 600;
/// Accept the state unconditionally after this many steps (t = 12, ~8x the
/// slowest physical mode). Needed by the mismatched-sources probe: at an
/// O(1)-displaced solution the limiter wander occasionally sets a new best
/// delta and starves the plateau detector forever; the wander amplitude
/// (~2e-5/step) is irrelevant against the saturated error level it
/// measures.
const LONG_MARCH_ACCEPT_STEPS: usize = 1200;

// ---------------------------------------------------------------------------
// Exact solution.
// ---------------------------------------------------------------------------

fn exact_rho(x: f64, y: f64) -> f64 {
    RHO0 * (1.0 + RHOA * (PI * x + PHX).sin() * (PI * y + PHY).sin())
}

/// Inflow on all four unit-box boundaries: u_x(0,·)=U0>0, u_x(1,·)=-U0,
/// u_y(·,0)=V0>0, u_y(·,1)=-V0 (the UA/UB terms vanish on the boundary).
fn exact_u(x: f64, y: f64) -> (f64, f64) {
    (
        U0 * (PI * x).cos() + UA * (PI * x).sin() * (PI * y).sin(),
        V0 * (PI * y).cos() + UB * (PI * x).sin() * (PI * y).sin(),
    )
}

/// Zero normal derivative on all four boundaries (sin(2 pi {0,1}) = 0):
/// required so the inlet "p follows interior" closure is O(h^2) consistent.
fn exact_p(x: f64, y: f64) -> f64 {
    P0 * (1.0 + PA * (2.0 * PI * x).cos() * (2.0 * PI * y).cos())
}

fn exact_t(x: f64, y: f64) -> f64 {
    exact_p(x, y) / (R_GAS * exact_rho(x, y))
}

fn exact_rho_e(x: f64, y: f64) -> f64 {
    let (u, v) = exact_u(x, y);
    exact_p(x, y) / (GAMMA - 1.0) + 0.5 * exact_rho(x, y) * (u * u + v * v)
}

// ---------------------------------------------------------------------------
// Analytic partial derivatives (cross-checked against finite differences of
// the closures above by `analytic_sources_match_finite_differences`).
// ---------------------------------------------------------------------------

/// (rho, rho_x, rho_y, rho_xx, rho_yy)
fn rho_partials(x: f64, y: f64) -> (f64, f64, f64, f64, f64) {
    let (sx, cx) = (PI * x + PHX).sin_cos();
    let (sy, cy) = (PI * y + PHY).sin_cos();
    let a = RHO0 * RHOA;
    (
        RHO0 + a * sx * sy,
        a * PI * cx * sy,
        a * PI * sx * cy,
        -a * PI * PI * sx * sy,
        -a * PI * PI * sx * sy,
    )
}

/// (u, u_x, u_y, u_xx, u_yy, u_xy)
fn u_partials(x: f64, y: f64) -> (f64, f64, f64, f64, f64, f64) {
    let (sx, cx) = (PI * x).sin_cos();
    let (sy, cy) = (PI * y).sin_cos();
    (
        U0 * cx + UA * sx * sy,
        -U0 * PI * sx + UA * PI * cx * sy,
        UA * PI * sx * cy,
        -U0 * PI * PI * cx - UA * PI * PI * sx * sy,
        -UA * PI * PI * sx * sy,
        UA * PI * PI * cx * cy,
    )
}

/// (v, v_x, v_y, v_xx, v_yy, v_xy)
fn v_partials(x: f64, y: f64) -> (f64, f64, f64, f64, f64, f64) {
    let (sx, cx) = (PI * x).sin_cos();
    let (sy, cy) = (PI * y).sin_cos();
    (
        V0 * cy + UB * sx * sy,
        UB * PI * cx * sy,
        -V0 * PI * sy + UB * PI * sx * cy,
        -UB * PI * PI * sx * sy,
        -V0 * PI * PI * cy - UB * PI * PI * sx * sy,
        UB * PI * PI * cx * cy,
    )
}

/// (p, p_x, p_y, p_xx, p_yy)
fn p_partials(x: f64, y: f64) -> (f64, f64, f64, f64, f64) {
    let (s2x, c2x) = (2.0 * PI * x).sin_cos();
    let (s2y, c2y) = (2.0 * PI * y).sin_cos();
    let a = P0 * PA;
    (
        P0 + a * c2x * c2y,
        -2.0 * PI * a * s2x * c2y,
        -2.0 * PI * a * c2x * s2y,
        -4.0 * PI * PI * a * c2x * c2y,
        -4.0 * PI * PI * a * c2x * c2y,
    )
}

// ---------------------------------------------------------------------------
// Manufactured sources: S = sum of the equation's non-source terms at the
// exact solution (explicit sources enter the assembled RHS positively, so
// the exact solution then satisfies the discrete equations).
// ---------------------------------------------------------------------------

/// S_rho = div(rho u).
fn source_rho(x: f64, y: f64) -> f64 {
    let (rho, rho_x, rho_y, _, _) = rho_partials(x, y);
    let (u, ux, _, _, _, _) = u_partials(x, y);
    let (v, _, vy, _, _, _) = v_partials(x, y);
    rho_x * u + rho * ux + rho_y * v + rho * vy
}

/// S_rho_u = div(rho u x u) + grad p - div(tau_full) - extra * mu lap(u).
/// `mu` parameterizes the viscosity so the Euler-dominated study (mu = 0)
/// shares the derivation; the viscous study passes the canonical `MU`.
fn source_rho_u(x: f64, y: f64, extra: f64, mu: f64) -> (f64, f64) {
    let (rho, rho_x, rho_y, _, _) = rho_partials(x, y);
    let (u, ux, uy, uxx, uyy, uxy) = u_partials(x, y);
    let (v, vx, vy, vxx, vyy, vxy) = v_partials(x, y);
    let (_, px, py, _, _) = p_partials(x, y);

    let conv_x = rho_x * u * u + 2.0 * rho * u * ux + px + rho_y * u * v + rho * (uy * v + u * vy);
    let conv_y = rho_x * u * v + rho * (ux * v + u * vx) + rho_y * v * v + 2.0 * rho * v * vy + py;

    // tau_full = mu (grad u + grad u^T - 2/3 I div u)
    let div_x = uxx + vxy; // d/dx (div u)
    let div_y = uxy + vyy; // d/dy (div u)
    let div_tau_x = mu * (2.0 * uxx - (2.0 / 3.0) * div_x + uyy + vxy);
    let div_tau_y = mu * (uxy + vxx + 2.0 * vyy - (2.0 / 3.0) * div_y);

    (
        conv_x - div_tau_x - extra * mu * (uxx + uyy),
        conv_y - div_tau_y - extra * mu * (vxx + vyy),
    )
}

/// S_rho_e = div((rho_e + p) u) - div(W) - k lap(T), where the work flux is
/// W = tau_full . u + extra * (mu/2) grad(|u|^2) (the as-coded sigmaDotU
/// traction is `mu (grad u) n + tau_full n`).
fn source_rho_e(x: f64, y: f64, extra: f64, mu: f64) -> f64 {
    // Conduction tracks the viscosity (kappa = mu cp / Pr in the assembly).
    let k_cond = mu * GAMMA * R_GAS / (GAMMA - 1.0) / PRANDTL;
    let (rho, rho_x, rho_y, rho_xx, rho_yy) = rho_partials(x, y);
    let (u, ux, uy, uxx, uyy, uxy) = u_partials(x, y);
    let (v, vx, vy, vxx, vyy, vxy) = v_partials(x, y);
    let (p, px, py, pxx, pyy) = p_partials(x, y);

    // Convection of total enthalpy: H = rho_e + p = gamma p/(gamma-1) + rho q^2/2.
    let q2 = u * u + v * v;
    let h = GAMMA * p / (GAMMA - 1.0) + 0.5 * rho * q2;
    let hx = GAMMA * px / (GAMMA - 1.0) + 0.5 * rho_x * q2 + rho * (u * ux + v * vx);
    let hy = GAMMA * py / (GAMMA - 1.0) + 0.5 * rho_y * q2 + rho * (u * uy + v * vy);
    let conv = hx * u + h * ux + hy * v + h * vy;

    // Viscous work divergence.
    let div_u = ux + vy;
    let tau_xx = mu * (2.0 * ux - (2.0 / 3.0) * div_u);
    let tau_yy = mu * (2.0 * vy - (2.0 / 3.0) * div_u);
    let tau_xy = mu * (uy + vx);
    let div_x = uxx + vxy;
    let div_y = uxy + vyy;
    let tau_xx_x = mu * (2.0 * uxx - (2.0 / 3.0) * div_x);
    let tau_xy_x = mu * (uxy + vxx);
    let tau_xy_y = mu * (uyy + vxy);
    let tau_yy_y = mu * (2.0 * vyy - (2.0 / 3.0) * div_y);
    let q2_xx = 2.0 * (ux * ux + u * uxx + vx * vx + v * vxx);
    let q2_yy = 2.0 * (uy * uy + u * uyy + vy * vy + v * vyy);
    let div_w = tau_xx_x * u + tau_xx * ux + tau_xy_x * v + tau_xy * vx
        + tau_xy_y * u
        + tau_xy * uy
        + tau_yy_y * v
        + tau_yy * vy
        + extra * 0.5 * mu * (q2_xx + q2_yy);

    // Conduction: k lap(T), T = p/(R rho).
    let w = 1.0 / rho;
    let wx = -rho_x / (rho * rho);
    let wy = -rho_y / (rho * rho);
    let wxx = (2.0 * rho_x * rho_x - rho * rho_xx) / (rho * rho * rho);
    let wyy = (2.0 * rho_y * rho_y - rho * rho_yy) / (rho * rho * rho);
    let t_xx = (pxx * w + 2.0 * px * wx + p * wxx) / R_GAS;
    let t_yy = (pyy * w + 2.0 * py * wy + p * wyy) / R_GAS;

    conv - div_w - k_cond * (t_xx + t_yy)
}

// ---------------------------------------------------------------------------
// FD cross-check of the analytic sources: rebuild every source as a (nested)
// 4th-order finite-difference divergence of closed-form flux functions that
// use ONLY the exact closures — fully independent of the hand-derived
// partials and the source assembly above.
// ---------------------------------------------------------------------------

fn d4x(f: &dyn Fn(f64, f64) -> f64, x: f64, y: f64, h: f64) -> f64 {
    (-f(x + 2.0 * h, y) + 8.0 * f(x + h, y) - 8.0 * f(x - h, y) + f(x - 2.0 * h, y)) / (12.0 * h)
}

fn d4y(f: &dyn Fn(f64, f64) -> f64, x: f64, y: f64, h: f64) -> f64 {
    (-f(x, y + 2.0 * h) + 8.0 * f(x, y + h) - 8.0 * f(x, y - h) + f(x, y - 2.0 * h)) / (12.0 * h)
}

#[test]
fn analytic_sources_match_finite_differences() {
    let h = 1e-3;
    let ux_f = |x: f64, y: f64| exact_u(x, y).0;
    let uy_f = |x: f64, y: f64| exact_u(x, y).1;
    // First derivatives of velocity via FD (independent of u_partials).
    let dudx = move |x: f64, y: f64| d4x(&ux_f, x, y, h);
    let dudy = move |x: f64, y: f64| d4y(&ux_f, x, y, h);
    let dvdx = move |x: f64, y: f64| d4x(&uy_f, x, y, h);
    let dvdy = move |x: f64, y: f64| d4y(&uy_f, x, y, h);
    let tau = move |x: f64, y: f64, mu: f64| {
        let (ux, uy, vx, vy) = (dudx(x, y), dudy(x, y), dvdx(x, y), dvdy(x, y));
        let div = ux + vy;
        (
            mu * (2.0 * ux - (2.0 / 3.0) * div),
            mu * (uy + vx),
            mu * (2.0 * vy - (2.0 / 3.0) * div),
        )
    };
    let t_f = |x: f64, y: f64| exact_t(x, y);

    let mut max_rel = 0.0f64;
    for i in 1..=5 {
        for j in 1..=5 {
            let x = 0.1 + 0.16 * i as f64;
            let y = 0.08 + 0.16 * j as f64;
            for &(extra, mu) in &[(0.0, MU), (1.0, MU), (0.0, 0.0)] {
                let k_cond = mu * GAMMA * R_GAS / (GAMMA - 1.0) / PRANDTL;
                // Continuity.
                let f1 = |x: f64, y: f64| exact_rho(x, y) * exact_u(x, y).0;
                let f2 = |x: f64, y: f64| exact_rho(x, y) * exact_u(x, y).1;
                let fd_rho = d4x(&f1, x, y, h) + d4y(&f2, x, y, h);
                let an_rho = source_rho(x, y);

                // Momentum x / y.
                let gx = move |x: f64, y: f64| {
                    let r = exact_rho(x, y);
                    let (u, _v) = exact_u(x, y);
                    let (txx, _txy, _tyy) = tau(x, y, mu);
                    r * u * u + exact_p(x, y) - txx - extra * mu * dudx(x, y)
                };
                let gy = move |x: f64, y: f64| {
                    let r = exact_rho(x, y);
                    let (u, v) = exact_u(x, y);
                    let (_txx, txy, _tyy) = tau(x, y, mu);
                    r * u * v - txy - extra * mu * dudy(x, y)
                };
                let fd_mx = d4x(&gx, x, y, h) + d4y(&gy, x, y, h);
                let hx_ = move |x: f64, y: f64| {
                    let r = exact_rho(x, y);
                    let (u, v) = exact_u(x, y);
                    let (_txx, txy, _tyy) = tau(x, y, mu);
                    r * u * v - txy - extra * mu * dvdx(x, y)
                };
                let hy_ = move |x: f64, y: f64| {
                    let r = exact_rho(x, y);
                    let (_u, v) = exact_u(x, y);
                    let (_txx, _txy, tyy) = tau(x, y, mu);
                    r * v * v + exact_p(x, y) - tyy - extra * mu * dvdy(x, y)
                };
                let fd_my = d4x(&hx_, x, y, h) + d4y(&hy_, x, y, h);
                let (an_mx, an_my) = source_rho_u(x, y, extra, mu);

                // Energy.
                let q2x = move |x: f64, y: f64| {
                    let (u, v) = exact_u(x, y);
                    2.0 * (u * dudx(x, y) + v * dvdx(x, y))
                };
                let q2y = move |x: f64, y: f64| {
                    let (u, v) = exact_u(x, y);
                    2.0 * (u * dudy(x, y) + v * dvdy(x, y))
                };
                let ex = move |x: f64, y: f64| {
                    let (u, v) = exact_u(x, y);
                    let hgas = exact_rho_e(x, y) + exact_p(x, y);
                    let (txx, txy, _tyy) = tau(x, y, mu);
                    let wx = txx * u + txy * v + extra * 0.5 * mu * q2x(x, y);
                    hgas * u - wx - k_cond * d4x(&t_f, x, y, h)
                };
                let ey = move |x: f64, y: f64| {
                    let (u, v) = exact_u(x, y);
                    let hgas = exact_rho_e(x, y) + exact_p(x, y);
                    let (_txx, txy, tyy) = tau(x, y, mu);
                    let wy = txy * u + tyy * v + extra * 0.5 * mu * q2y(x, y);
                    hgas * v - wy - k_cond * d4y(&t_f, x, y, h)
                };
                let fd_e = d4x(&ex, x, y, h) + d4y(&ey, x, y, h);
                let an_e = source_rho_e(x, y, extra, mu);

                for (name, an, fd) in [
                    ("rho", an_rho, fd_rho),
                    ("rho_u_x", an_mx, fd_mx),
                    ("rho_u_y", an_my, fd_my),
                    ("rho_e", an_e, fd_e),
                ] {
                    let denom = 1.0_f64.max(an.abs());
                    let rel = (an - fd).abs() / denom;
                    max_rel = max_rel.max(rel);
                    assert!(
                        rel < 1e-6,
                        "source_{name} mismatch at ({x:.3},{y:.3}) extra={extra}: \
                         analytic={an:.10e} fd={fd:.10e} rel={rel:.3e}"
                    );
                }
            }
        }
    }
    println!("[mms][compressible] FD cross-check max_rel={max_rel:.3e}");
}

// ---------------------------------------------------------------------------
// GPU solve.
// ---------------------------------------------------------------------------

struct SteadyRun {
    mesh: Mesh,
    solver: UnifiedSolver,
}

/// (rho_l2, u_l2, p_l2, t_l2) against the exact solution.
fn read_errors(run: &SteadyRun) -> (f64, f64, f64, f64) {
    let u = pollster::block_on(run.solver.get_field_vec2("u")).expect("read u");
    let rho = pollster::block_on(run.solver.get_field_scalar("rho")).expect("read rho");
    let p = pollster::block_on(run.solver.get_field_scalar("p")).expect("read p");
    let t = pollster::block_on(run.solver.get_field_scalar("T")).expect("read T");
    (
        field_errors(&run.mesh, &rho, exact_rho).l2,
        field_errors_vec2(&run.mesh, &u, exact_u).l2,
        field_errors(&run.mesh, &p, exact_p).l2,
        field_errors(&run.mesh, &t, exact_t).l2,
    )
}

fn solve_steady(n: usize, extra: f64, mu: f64, dt: f64) -> SteadyRun {
    let mut run = build_run(n, extra, mu, dt, TimeScheme::BDF2, OUTER_ITERS);
    march_to_plateau(&mut run.solver);
    run
}

/// Construct mesh + solver + BCs + sources + exact-solution init WITHOUT
/// marching (shared by the order studies and the stability probes).
fn build_run(
    n: usize,
    extra: f64,
    mu: f64,
    dt: f64,
    time_scheme: TimeScheme,
    outer_iters: usize,
) -> SteadyRun {
    let mesh = generate_structured_rect_mesh(
        n,
        n,
        1.0,
        1.0,
        BoundarySides {
            left: BoundaryType::Inlet,
            right: BoundaryType::Inlet,
            bottom: BoundaryType::Inlet,
            top: BoundaryType::Inlet,
        },
    );
    let model = compressible_mms_model().expect("model");
    let eos = EosSpec::IdealGas {
        gamma: GAMMA,
        gas_constant: R_GAS,
        temperature: 1.0,
    };
    let mut solver = pollster::block_on(UnifiedSolver::new(
        &mesh,
        model,
        SolverConfig {
            advection_scheme: Scheme::SecondOrderUpwindVanLeer,
            time_scheme,
            preconditioner: PreconditionerType::Jacobi,
            stepping: SteppingMode::Implicit { outer_iters },
        },
        None,
        None,
    ))
    .expect("solver init");

    solver.set_eos(&eos).expect("eos");
    solver.set_dt(dt as f32);
    solver.set_dtau(0.0).expect("dtau");
    solver.set_viscosity(mu as f32).expect("viscosity");
    solver.set_density(RHO0 as f32).expect("density");
    solver.set_outer_iters(OUTER_ITERS).expect("outer_iters");

    // Per-face exact Dirichlet rho and u on the (all-Inlet) boundary; seed
    // the expression-valued entries (rho_u/rho_e/p/T) with exact face values
    // too — the bc_expr kernel refreshes them every outer iteration.
    let fx = mesh.face_cx.clone();
    let fy = mesh.face_cy.clone();
    let scalar_face = |f: &'static dyn Fn(f64, f64) -> f64, fx: &[f64], fy: &[f64]| {
        let fx = fx.to_vec();
        let fy = fy.to_vec();
        move |face_idx: u32| f(fx[face_idx as usize], fy[face_idx as usize]) as f32
    };
    solver
        .set_boundary_values_per_face(
            GpuBoundaryType::Inlet,
            "rho",
            0,
            &scalar_face(&exact_rho, &fx, &fy),
        )
        .expect("rho bc");
    solver
        .set_boundary_values_per_face(
            GpuBoundaryType::Inlet,
            "p",
            0,
            &scalar_face(&exact_p, &fx, &fy),
        )
        .expect("p bc");
    solver
        .set_boundary_values_per_face(
            GpuBoundaryType::Inlet,
            "T",
            0,
            &scalar_face(&exact_t, &fx, &fy),
        )
        .expect("T bc");
    solver
        .set_boundary_values_per_face(
            GpuBoundaryType::Inlet,
            "rho_e",
            0,
            &scalar_face(&exact_rho_e, &fx, &fy),
        )
        .expect("rho_e bc");
    for c in 0..2usize {
        let u_face = {
            let fx = fx.clone();
            let fy = fy.clone();
            move |face_idx: u32| {
                let i = face_idx as usize;
                let (ux, uy) = exact_u(fx[i], fy[i]);
                (if c == 0 { ux } else { uy }) as f32
            }
        };
        let rho_u_face = {
            let fx = fx.clone();
            let fy = fy.clone();
            move |face_idx: u32| {
                let i = face_idx as usize;
                let (ux, uy) = exact_u(fx[i], fy[i]);
                (exact_rho(fx[i], fy[i]) * if c == 0 { ux } else { uy }) as f32
            }
        };
        solver
            .set_boundary_values_per_face(GpuBoundaryType::Inlet, "u", c as u32, &u_face)
            .expect("u bc");
        solver
            .set_boundary_values_per_face(GpuBoundaryType::Inlet, "rho_u", c as u32, &rho_u_face)
            .expect("rho_u bc");
    }

    // Manufactured sources at cell centers (per-volume PDE residuals).
    let cells = mesh.num_cells();
    let mut src_rho: Vec<f64> = (0..cells)
        .map(|i| source_rho(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    // Discrete mass-compatibility projection. At Dirichlet (inlet) faces the
    // KT boundary mass flux is exactly rho_bc (u_bc . n) A — both
    // reconstruction sides take the ghost state — so it has ZERO sensitivity
    // to the interior. Total mass then obeys
    //   dM/dt = sum(S_rho V) - sum_boundary(rho* u* . n A) = const,
    // an O(h^2) quadrature mismatch with no restoring mode: without this
    // projection the solution drifts secularly (~2e-4/time-unit at n=48,
    // error growing linearly long past every physical relaxation time) and
    // no discrete steady state exists. Subtracting the uniform constant
    // eps = imbalance / V_total enforces dM/dt = 0 exactly (flux divergence
    // telescopes); eps is O(h^2) and vanishes under refinement, so it does
    // not affect the convergence order. Momentum/energy need no projection:
    // their boundary fluxes track the interior pressure through the declared
    // inlet expressions (p_blend, rho_e ghost), which restores equilibrium.
    let vol_total: f64 = mesh.cell_vol.iter().sum();
    let vol_int: f64 = (0..cells).map(|i| src_rho[i] * mesh.cell_vol[i]).sum();
    let bflux: f64 = (0..mesh.face_owner.len())
        .filter(|&f| mesh.face_neighbor[f].is_none())
        .map(|f| {
            let (ux, uy) = exact_u(mesh.face_cx[f], mesh.face_cy[f]);
            exact_rho(mesh.face_cx[f], mesh.face_cy[f])
                * (ux * mesh.face_nx[f] + uy * mesh.face_ny[f])
                * mesh.face_area[f]
        })
        .sum();
    let eps = (vol_int - bflux) / vol_total;
    println!("[mms][compressible] n={n} mass-compatibility eps={eps:.3e}");
    for s in src_rho.iter_mut() {
        *s -= eps;
    }
    let src_rho_u: Vec<(f64, f64)> = (0..cells)
        .map(|i| source_rho_u(mesh.cell_cx[i], mesh.cell_cy[i], extra, mu))
        .collect();
    let src_rho_e: Vec<f64> = (0..cells)
        .map(|i| source_rho_e(mesh.cell_cx[i], mesh.cell_cy[i], extra, mu))
        .collect();
    solver
        .set_field_scalar(COMPRESSIBLE_MMS_SOURCE_RHO_FIELD, &src_rho)
        .expect("upload S_rho");
    solver
        .set_field_vec2(COMPRESSIBLE_MMS_SOURCE_RHO_U_FIELD, &src_rho_u)
        .expect("upload S_rho_u");
    solver
        .set_field_scalar(COMPRESSIBLE_MMS_SOURCE_RHO_E_FIELD, &src_rho_e)
        .expect("upload S_rho_e");

    // Initialize at the exact solution (consistent conserved + primitive set);
    // the run then settles to the nearby discrete fixed point.
    let rho0: Vec<f64> = (0..cells)
        .map(|i| exact_rho(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    let u0: Vec<(f64, f64)> = (0..cells)
        .map(|i| exact_u(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    let rho_u0: Vec<(f64, f64)> = (0..cells)
        .map(|i| {
            let r = rho0[i];
            (r * u0[i].0, r * u0[i].1)
        })
        .collect();
    let p0: Vec<f64> = (0..cells)
        .map(|i| exact_p(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    let t0: Vec<f64> = (0..cells)
        .map(|i| exact_t(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    let rho_e0: Vec<f64> = (0..cells)
        .map(|i| exact_rho_e(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    solver.set_field_scalar("rho", &rho0).expect("init rho");
    solver.set_field_vec2("rho_u", &rho_u0).expect("init rho_u");
    solver
        .set_field_scalar("rho_e", &rho_e0)
        .expect("init rho_e");
    solver.set_field_scalar("p", &p0).expect("init p");
    solver.set_field_scalar("T", &t0).expect("init T");
    solver.set_field_vec2("u", &u0).expect("init u");
    solver.initialize_history();
    SteadyRun { mesh, solver }
}

/// March until the watched per-step max delta (over u, rho, T) either drops
/// below `STEADY_TOL` or plateaus: the smallest delta seen stops improving
/// by more than 2% for `PLATEAU_WINDOW` consecutive steps. See the policy
/// comment at the constants for why an absolute tolerance alone stalls.
fn march_to_plateau(solver: &mut UnifiedSolver) {
    let read = |solver: &UnifiedSolver| -> (Vec<(f64, f64)>, Vec<f64>, Vec<f64>) {
        (
            pollster::block_on(solver.get_field_vec2("u")).expect("read u"),
            pollster::block_on(solver.get_field_scalar("rho")).expect("read rho"),
            pollster::block_on(solver.get_field_scalar("T")).expect("read T"),
        )
    };
    let mut prev = read(solver);
    let mut best = f64::INFINITY;
    let mut best_step = 0usize;
    for step in 0..STEADY_MAX_STEPS {
        solver.step();
        let cur = read(solver);
        let mut max_delta = 0.0f64;
        for (a, b) in cur.0.iter().zip(prev.0.iter()) {
            max_delta = max_delta.max((a.0 - b.0).abs()).max((a.1 - b.1).abs());
        }
        for (c, p) in [(&cur.1, &prev.1), (&cur.2, &prev.2)] {
            for (a, b) in c.iter().zip(p.iter()) {
                max_delta = max_delta.max((a - b).abs());
            }
        }
        if max_delta < STEADY_TOL && step >= MIN_STEPS {
            println!("[mms] steady after {} steps (max_delta={max_delta:.3e})", step + 1);
            return;
        }
        if max_delta < best * 0.98 {
            best = max_delta;
            best_step = step;
        } else if step > best_step + PLATEAU_WINDOW && step >= MIN_STEPS {
            println!(
                "[mms] delta plateau after {} steps (max_delta={max_delta:.3e}, best={best:.3e} at step {best_step})",
                step + 1
            );
            return;
        }
        if step >= LONG_MARCH_ACCEPT_STEPS {
            println!(
                "[mms] long-march acceptance after {} steps (max_delta={max_delta:.3e}, best={best:.3e})",
                step + 1
            );
            return;
        }
        if step % 20 == 0 {
            println!("[mms] step {step}: max_delta={max_delta:.3e}");
        }
        prev = cur;
    }
    panic!("no steady tolerance or plateau within {STEADY_MAX_STEPS} steps (best={best:.3e})");
}

fn order_study(extra: f64, mu: f64, dt: f64, levels: &[usize], label: &str) -> (Vec<f64>, [Vec<f64>; 4]) {
    let mut hs = Vec::new();
    let mut rho_errs = Vec::new();
    let mut u_errs = Vec::new();
    let mut p_errs = Vec::new();
    let mut t_errs = Vec::new();
    for (li, &n) in levels.iter().enumerate() {
        let mut run = solve_steady(n, extra, mu, dt);
        let (rho_err, u_err, p_err, t_err) = read_errors(&run);
        println!(
            "[mms][{label}] n={n} rho_l2={rho_err:.4e} u_l2={u_err:.4e} p_l2={p_err:.4e} t_l2={t_err:.4e}"
        );
        // Limit-cycle amplitude guard at the finest level (where the
        // discretization error is smallest): the plateau-state error metric
        // must be stable under further marching, otherwise the "steady"
        // measurement is an artifact of when the plateau detector fired.
        if li + 1 == levels.len() {
            for _ in 0..100 {
                run.solver.step();
            }
            let (rho2, u2, p2, t2) = read_errors(&run);
            println!(
                "[mms][{label}] n={n} after +100 steps: rho_l2={rho2:.4e} u_l2={u2:.4e} p_l2={p2:.4e} t_l2={t2:.4e}"
            );
            for (name, a, b) in [("rho", rho_err, rho2), ("u", u_err, u2), ("T", t_err, t2)] {
                let drift = (a - b).abs() / a.max(1e-300);
                assert!(
                    drift < 0.10,
                    "[mms][{label}] {name} error drifts {:.1}% over +100 steps at n={n} \
                     (limit cycle not negligible: {a:.4e} -> {b:.4e})",
                    drift * 100.0
                );
            }
        }
        hs.push(1.0 / n as f64);
        rho_errs.push(rho_err);
        u_errs.push(u_err);
        p_errs.push(p_err);
        t_errs.push(t_err);
    }
    (hs, [rho_errs, u_errs, p_errs, t_errs])
}

/// The compressible MMS oracle: manufactured subsonic Navier–Stokes solution
/// through the full coupled path (KT/vanLeer flux, EOS recovery, implicit
/// viscous/conduction laplacians, expression-valued inlet BCs).
#[test]
fn steady_compressible_vanleer_order() {
    let (hs, [rho_errs, u_errs, p_errs, t_errs]) =
        order_study(EXTRA_SHEAR, MU, DT, &[16, 24, 32, 48], "compressible");
    // Measured at the ratchet (June 2026, post-tauMC-fix, physical-NS
    // sources): rho 1.956 / u 2.153 / p 1.831 / T 1.858, finest errors
    // 5.6e-4 / 2.7e-4 / 5.2e-4 / 5.8e-4. (Pre-fix, with doubled-shear
    // sources matching the pre-fix operator: 1.937 / 2.186 / 1.859 / 1.791.)
    assert_convergence_order("compressible_rho", &hs, &rho_errs, 2.0, 0.35, 1.5e-3);
    assert_convergence_order("compressible_u", &hs, &u_errs, 2.0, 0.35, 7.0e-4);
    assert_convergence_order("compressible_p", &hs, &p_errs, 2.0, 0.35, 1.2e-3);
    assert_convergence_order("compressible_T", &hs, &t_errs, 2.0, 0.40, 1.2e-3);
}

/// Euler-dominated variant of the oracle: the same manufactured solution
/// and harness with mu = 0 — no viscous stress, no conduction (k tracks mu),
/// pure KT/vanLeer convection + EOS recovery + pressure work. This orders
/// the convective operator in isolation: the NS study at Re ~ 8 is
/// viscosity-dominated, so a convective-flux defect could hide under the
/// laplacians there.
///
/// INVISCID-MARGIN INSTABILITY (June 2026, probed during this test's
/// derivation — the reason mu is 5e-3 and not 0): the coupled-implicit
/// compressible path develops a slow secular instability as mu -> 0,
/// strongest at fine grids (rate rises with n, bounded only by physical
/// damping mu k^2):
/// - mu = 0,    dt = 0.01:  n=48 diverges outright (state -> inf).
/// - mu = 0,    dt = 0.005: no blow-up, but secular growth at every level
///   (u error +52% per 100 steps at n=48) — not CFL, a genuine
///   marginal-mode instability.
/// - mu = 1e-3, dt = 0.01:  n=48 still diverges (fast branch at CFL ~ 1).
/// - mu = 1e-3, dt = 0.005: n=32 clean; n=48 drifts +49%/100 steps.
/// - mu = 5e-3, dt = 0.005: n=16/24/32 clean (this test); n=48 still
///   marginal (errors grow ~80% per 3 time units under extended march).
/// Recorded as an engine-robustness backlog item (rhoCentralFoam runs
/// inviscid fine, so the coupled-implicit path's inviscid stability is a
/// real gap). Within the stable envelope the conservative viscous-era
/// march policy holds (plateaus at steps 215-839, drift guard <10%).
#[test]
fn steady_euler_dominated_vanleer_order() {
    let (hs, [rho_errs, u_errs, p_errs, t_errs]) =
        order_study(EXTRA_SHEAR, MU_EULER, DT_EULER, &[16, 24, 32], "euler");
    // Measured June 2026 (first run, mu = 5e-3, dt = 5e-3, n = 16/24/32):
    // orders rho 2.430 / u ~2.05 / p ~3.2 / T ~2.45; finest errors
    // 3.28e-3 / 6.36e-3 / 6.17e-4 / 3.21e-3 (larger than the NS study's:
    // no conduction smoothing). n=32 drift guard measured +8.4% (rho) —
    // close to the 10% bound; if it ever flakes, the march is the lever,
    // not the band. Caps at ~1.3x measured; ratchet-only thereafter.
    assert_convergence_order("euler_rho", &hs, &rho_errs, 2.0, 0.35, 4.5e-3);
    assert_convergence_order("euler_u", &hs, &u_errs, 2.0, 0.35, 8.5e-3);
    assert_convergence_order("euler_p", &hs, &p_errs, 2.0, 0.35, 9.0e-4);
    assert_convergence_order("euler_T", &hs, &t_errs, 2.0, 0.40, 4.5e-3);
}

// ---------------------------------------------------------------------------
// Inviscid-margin instability probes (Arc I, June 2026).
// ---------------------------------------------------------------------------

/// March `steps`, sampling the u-vs-exact L2 error every `sample_every`
/// steps; return (growth rate in %/time-unit fitted on the second half of
/// the series, final p checkerboard fraction, final u_x checkerboard
/// fraction). A diverged run reports f64::INFINITY.
fn measure_growth(
    run: &mut SteadyRun,
    dt: f64,
    steps: usize,
    sample_every: usize,
) -> (f64, f64, f64, f64, f64) {
    let mut series: Vec<(f64, f64)> = Vec::new(); // (t, u_l2)
    let mut t = 0.0;
    for k in 0..(steps / sample_every) {
        for _ in 0..sample_every {
            run.solver.step();
        }
        t = ((k + 1) * sample_every) as f64 * dt;
        let u = pollster::block_on(run.solver.get_field_vec2("u")).expect("read u");
        let e = field_errors_vec2(&run.mesh, &u, exact_u).l2;
        if !e.is_finite() {
            return (f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY);
        }
        series.push((t, e));
    }
    // Log-slope fit over the second half (skips the settle-in transient).
    let half = series.len() / 2;
    let (t0, e0) = series[half];
    let (t1, e1) = *series.last().unwrap();
    let rate = if e0 > 0.0 && e1 > 0.0 && t1 > t0 {
        ((e1 / e0).ln() / (t1 - t0)) * 100.0
    } else {
        f64::NAN
    };

    // Grid-Nyquist (checkerboard) fraction of the de-meaned field:
    // |sum f' * (-1)^(i+j)| / (N * rms(f')).
    let n = (run.mesh.num_cells() as f64).sqrt().round() as usize;
    let h = 1.0 / n as f64;
    let nyq = |f: &[f64]| -> f64 {
        let mean = f.iter().sum::<f64>() / f.len() as f64;
        let mut alt = 0.0;
        let mut ss = 0.0;
        for c in 0..run.mesh.num_cells() {
            let i = (run.mesh.cell_cx[c] / h - 0.5).round() as i64;
            let j = (run.mesh.cell_cy[c] / h - 0.5).round() as i64;
            let v = f[c] - mean;
            alt += v * if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
            ss += v * v;
        }
        let rms = (ss / f.len() as f64).sqrt();
        (alt / f.len() as f64).abs() / rms.max(1e-300)
    };
    let p = pollster::block_on(run.solver.get_field_scalar("p")).expect("read p");
    let u = pollster::block_on(run.solver.get_field_vec2("u")).expect("read u");
    let ux: Vec<f64> = u.iter().map(|v| v.0).collect();

    // Spatial localization of the (final) u error: fraction of the squared
    // error mass within 2 cells of the boundary.
    let band = 2.5 * h;
    let mut e_band = 0.0;
    let mut e_tot = 0.0;
    for c in 0..run.mesh.num_cells() {
        let (x, y) = (run.mesh.cell_cx[c], run.mesh.cell_cy[c]);
        let (uex, uey) = exact_u(x, y);
        let e2 = (u[c].0 - uex).powi(2) + (u[c].1 - uey).powi(2);
        e_tot += e2;
        if x < band || x > 1.0 - band || y < band || y > 1.0 - band {
            e_band += e2;
        }
    }
    let bfrac = e_band / e_tot.max(1e-300);

    // Final error LEVELS (u and rho): the growth rate alone is blind to a
    // fast blow-up that saturates before the fit window — and to
    // thermo-field blow-up hidden by the bounded u = rho_u/rho recovery.
    let u_final = series.last().unwrap().1;
    let rho = pollster::block_on(run.solver.get_field_scalar("rho")).expect("read rho");
    let rho_final = field_errors(&run.mesh, &rho, exact_rho).l2;
    (rate, nyq(&p), bfrac, u_final, rho_final)
}

/// Probe matrix for the inviscid-margin instability (Arc I). All runs at
/// mu = 0, dt = 5e-3 (the slow-growth regime; the fast CFL~1 branch at
/// dt = 0.01/n=48 is probed separately by the last row).
///
/// MEASURED (June 12, 2026) — growth %/tu, p-nyquist, boundary fraction,
/// FINAL u/rho error levels (the levels matter: a rate fitted on the
/// second half is blind to a fast blow-up that saturates early, and a
/// u-only metric is blind to thermo-field blow-up hidden by the bounded
/// u = rho_u/rho recovery — both blindnesses produced a wrong
/// "preconditioning stabilizes" reading on the first pass):
///   BDF2  o1 n32                 103.5  nyq 1e-3  bf 0.08  u 4.6e-2  rho 1.5e-2
///   BDF2  o1 n48                  60.6  nyq 1e-4  bf 0.07  u 3.5e-2  rho 1.8e-2
///   Euler o1 n32/n48          104/116  (same character)
///   BDF2  o2/o4 n48               60.6  bit-identical to o1
///   BDF2  o1 n48 dtau=dt lm=Off   77.2  bf 0.51   u 5.1e-3  rho 3.0e-3
///   BDF2  o1 n48 dtau=dt lm=WS    "0"   u 0.18    rho 9.0e10  (BLOWN)
///   BDF2  o1 n48 dtau lm=WS a=0   "0"   u 1.7e7   (BLOWN)
///   BDF2  o1 n48 dtau lm=Legacy   "0"   rho 9.0e10  (BLOWN)
///   BDF2  o1 n48 dt=1e-2 (fast)   "0"   u 1.5e12  (BLOWN)
///
/// VERDICT — every knob-level hypothesis REFUTED; the instability is
/// intrinsic to the spatial discretization at mu = 0, moderate Mach:
/// - NOT time integration (Euler grows like BDF2).
/// - NOT outer-iteration lag (o2/o4 bit-identical — which also proves the
///   flux module is evaluated once per STEP, frozen across outer iters).
/// - NOT a checkerboard (nyquist ~1e-4 on the growing runs) and NOT
///   boundary-fed (boundary band holds LESS error mass than uniform).
/// - Pseudo-time damping (dtau = dt, preconditioning Off) DAMPS the mode
///   ~7x in final error but does not stabilize it.
/// - Low-Mach preconditioning (either model, with or without the
///   pressure-coupling term) makes it catastrophically WORSE: at M ~ 0.5
///   it rescales the dissipation wave speed c -> ~|u|, halving the
///   acoustic dissipation — a low-Mach tool misapplied at moderate Mach.
/// The growing object is a smooth INTERIOR mode of the coupled
/// KT-flux + inv_dt-scaled-EOS-recovery system, damped only by physical
/// viscosity (mu k^2 must beat the mode's growth; mu = 5e-3 holds through
/// n = 32 at this problem's scales — the Euler study's operating point).
/// STABILITY ENVELOPE (model contract, see the compressible model docs):
/// time-accurate compressible marching requires nonzero physical
/// viscosity; the inviscid limit is out of envelope on this
/// discretization at moderate Mach.
///
/// Run:
/// `cargo test --features dev-tests --test mms_compressible_order_test -- --ignored probe_inviscid --nocapture`
#[test]
#[ignore]
fn probe_inviscid_margin_matrix() {
    use cfd2::solver::gpu::enums::GpuLowMachPrecondModel;
    use cfd2::solver::gpu::unified_solver::PlanParamValue;

    let dt = 5.0e-3;
    let steps = 600;
    println!(
        "[inviscid-probe] {:34} {:>12} {:>8} {:>8} {:>9} {:>9}",
        "config", "growth %/tu", "nyq(p)", "bfrac", "u_l2", "rho_l2"
    );
    let mut report = |label: &str, mut run: SteadyRun, dt: f64, steps: usize| {
        let (rate, nyq_p, bfrac, u_l2, rho_l2) = measure_growth(&mut run, dt, steps, 25);
        println!("[inviscid-probe] {label:34} {rate:12.2} {nyq_p:8.4} {bfrac:8.4} {u_l2:9.2e} {rho_l2:9.2e}");
    };

    // Baselines (BDF2, outer 1, dtau 0).
    report("BDF2 o1 n32", build_run(32, EXTRA_SHEAR, 0.0, dt, TimeScheme::BDF2, 1), dt, steps);
    report("BDF2 o1 n48", build_run(48, EXTRA_SHEAR, 0.0, dt, TimeScheme::BDF2, 1), dt, steps);

    // H1: Euler time scheme.
    report("Euler o1 n32", build_run(32, EXTRA_SHEAR, 0.0, dt, TimeScheme::Euler, 1), dt, steps);
    report("Euler o1 n48", build_run(48, EXTRA_SHEAR, 0.0, dt, TimeScheme::Euler, 1), dt, steps);

    // H2: outer iterations.
    report("BDF2 o2 n48", build_run(48, EXTRA_SHEAR, 0.0, dt, TimeScheme::BDF2, 2), dt, steps);
    report("BDF2 o4 n48", build_run(48, EXTRA_SHEAR, 0.0, dt, TimeScheme::BDF2, 4), dt, steps);

    // H3: pseudo-time damping (dtau also un-gates low-mach paths; vary the
    // model to separate the two effects).
    {
        let mut run = build_run(48, EXTRA_SHEAR, 0.0, dt, TimeScheme::BDF2, 1);
        run.solver.set_dtau(dt as f32).expect("dtau");
        report("BDF2 o1 n48 dtau=dt lm=Off", run, dt, steps);
    }
    {
        let mut run = build_run(48, EXTRA_SHEAR, 0.0, dt, TimeScheme::BDF2, 1);
        run.solver.set_dtau(dt as f32).expect("dtau");
        run.solver
            .set_named_param(
                "low_mach.model",
                PlanParamValue::LowMachModel(GpuLowMachPrecondModel::WeissSmith),
            )
            .expect("low_mach.model");
        report("BDF2 o1 n48 dtau=dt lm=WS", run, dt, steps);
    }

    // WS-row decomposition: WeissSmith turned on BOTH the wave-speed
    // rescale and the pressure-coupling dissipation (rho' = alpha*p'/c^2 in
    // the KT dissipation state). Separate them.
    {
        let mut run = build_run(48, EXTRA_SHEAR, 0.0, dt, TimeScheme::BDF2, 1);
        run.solver.set_dtau(dt as f32).expect("dtau");
        run.solver
            .set_named_param(
                "low_mach.model",
                PlanParamValue::LowMachModel(GpuLowMachPrecondModel::WeissSmith),
            )
            .expect("low_mach.model");
        run.solver
            .set_named_param("low_mach.pressure_coupling_alpha", PlanParamValue::F32(0.0))
            .expect("alpha");
        report("BDF2 o1 n48 dtau lm=WS a=0", run, dt, steps);
    }
    {
        let mut run = build_run(48, EXTRA_SHEAR, 0.0, dt, TimeScheme::BDF2, 1);
        run.solver.set_dtau(dt as f32).expect("dtau");
        run.solver
            .set_named_param(
                "low_mach.model",
                PlanParamValue::LowMachModel(GpuLowMachPrecondModel::Legacy),
            )
            .expect("low_mach.model");
        report("BDF2 o1 n48 dtau lm=Legacy", run, dt, steps);
    }

    // Fast branch (CFL ~ 1): the n=48/dt=0.01 outright divergence.
    report("BDF2 o1 n48 dt=1e-2 (fast)", build_run(48, EXTRA_SHEAR, 0.0, 1.0e-2, TimeScheme::BDF2, 1), 1.0e-2, steps);
}

/// Order study of the PRECONDITIONED inviscid operator: mu = 0 with
/// dual-time WeissSmith preconditioning. MEASURED June 12, 2026: BLOWS UP
/// (rho/p -> 1e10..1e12 by n=24; u stays bounded through the rho_u/rho
/// recovery, which is how the first probe pass mislabeled this
/// configuration "stable"). Kept as the falsification record for the
/// "just use preconditioning for inviscid" idea — at moderate Mach the
/// preconditioned wave-speed rescale REMOVES dissipation and worsens the
/// instability. See the probe matrix verdict above.
#[test]
#[ignore]
fn probe_euler_preconditioned_order() {
    use cfd2::solver::gpu::enums::GpuLowMachPrecondModel;
    use cfd2::solver::gpu::unified_solver::PlanParamValue;

    let dt = 5.0e-3;
    let mut hs = Vec::new();
    let mut u_errs = Vec::new();
    let mut rho_errs = Vec::new();
    let mut p_errs = Vec::new();
    for &n in &[16usize, 24, 32, 48] {
        let mut run = build_run(n, EXTRA_SHEAR, 0.0, dt, TimeScheme::BDF2, 1);
        run.solver.set_dtau(dt as f32).expect("dtau");
        run.solver
            .set_named_param(
                "low_mach.model",
                PlanParamValue::LowMachModel(GpuLowMachPrecondModel::WeissSmith),
            )
            .expect("low_mach.model");
        march_to_plateau(&mut run.solver);
        let (rho_err, u_err, p_err, t_err) = read_errors(&run);
        println!("[euler-precond] n={n} rho_l2={rho_err:.4e} u_l2={u_err:.4e} p_l2={p_err:.4e} t_l2={t_err:.4e}");
        hs.push(1.0 / n as f64);
        u_errs.push(u_err);
        rho_errs.push(rho_err);
        p_errs.push(p_err);
    }
    for (name, errs) in [("rho", &rho_errs), ("u", &u_errs), ("p", &p_errs)] {
        let o = ((errs[0] / errs[errs.len() - 1]).ln()) / ((hs[0] / hs[hs.len() - 1]).ln());
        println!("[euler-precond] {name} order(first-last) = {o:.3}");
    }
}

/// Probe: the same study with sources for the PRE-FIX doubled-shear
/// operator (`extra = 1`). After the tauMC fix these must SATURATE
/// h-independently instead of converging — if this probe ever shows order-2
/// convergence again, the double-counted laplacian has been reintroduced.
/// Run manually:
/// `cargo test --features dev-tests --test mms_compressible_order_test -- --ignored probe_ --nocapture`
///
/// Measured post-fix (June 2026): rho 5.60e-2 -> 5.28e-2, u 2.49e-2 ->
/// 2.31e-2, T 1.39e-2 -> 1.20e-2 from n=16 to n=32 — saturated, ~100x the
/// converging study's n=32 errors. Pre-fix the roles were reversed
/// (doubled-shear operator in the solver, physical-NS sources saturating at
/// rho 4.6e-2 / u 1.5e-2), completing the two-direction operator
/// identification. (The drift guard passes on a saturated run too: it is a
/// genuine steady state — of the wrong continuous problem.)
#[test]
#[ignore]
fn probe_doubled_shear_sources_saturate() {
    let (hs, [rho_errs, u_errs, p_errs, t_errs]) =
        order_study(1.0, MU, DT, &[16, 32], "compressible-doubled-shear");
    let _ = (&hs, &p_errs);
    println!(
        "[mms][compressible-doubled-shear] rho={rho_errs:?} u={u_errs:?} T={t_errs:?} (must NOT converge; compare against the physical-NS study)"
    );
}
