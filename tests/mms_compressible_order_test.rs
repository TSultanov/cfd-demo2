//! Compressible MMS: manufactured (rho, u, p) for the coupled compressible
//! Navier–Stokes path (KT central-upwind flux with vanLeer reconstruction,
//! implicit mu/kappa laplacians, EOS recovery rows).
//!
//! Setup: unit box, all four sides Inlet (manufactured velocity points inward
//! on every boundary; the continuity source balances it). Inlet BCs prescribe
//! rho and u per face from the exact solution; the model's inlet expressions
//! keep rho_u/rho_e/p/T consistent with interior pressure. The manufactured
//! pressure has zero normal derivative on all boundaries, so the inlet
//! "p follows interior" closure is second-order consistent.
//!
//! Operator contract: physical Navier-Stokes. The momentum viscous operator is
//! `div(tau)` with `tau = mu (grad u + grad u^T - 2/3 I div u)`, split
//! rhoCentralFoam-style between the implicit `laplacian(mu, u)` and the explicit
//! transpose-only `tauMC` traction in the flux; the energy viscous work flux is
//! `tau . u`. Conduction lowers to `mu cp / 0.71` (hardcoded Prandtl). Sources
//! are derived for that operator (`EXTRA_SHEAR = 0`).

#![cfg(feature = "dev-tests")]

mod mms_support;

use std::f64::consts::PI;

use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::mesh::{
    generate_structured_rect_mesh, generate_structured_rect_mesh_periodic, BoundarySides,
    BoundaryType, Mesh,
};
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::model::helpers::SolverRuntimeParamsExt;
use cfd2::solver::model::{
    compressible_mms_biharmonic_model, compressible_mms_model, COMPRESSIBLE_MMS_SOURCE_RHO_E_FIELD,
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
/// Euler-dominated study: smallest STABLE viscous floor (Re ~ 112). mu = 0 is
/// unstable (secular growth, rate rising with n, bounded only by physical
/// damping); see the inviscid-margin note on the Euler order test.
const MU_EULER: f64 = 5.0e-3;
/// CFL ~ 0.5 at n=32 for the Euler study.
const DT_EULER: f64 = 0.005;

/// 0.0 = sources for physical NS. 1.0 = sources for the doubled-shear operator
/// (full-stress tauMC + assembly laplacian), used by the regression probe.
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
// sustains a small steady-state limit cycle (TVD-limiter convergence stall):
// the per-step delta decays to a dt-independent floor rather than to zero, so
// the runner stops on a delta PLATEAU (no improvement over a window) rather
// than an absolute tolerance. The cycle amplitude is far below discretization
// error (checked by the error-drift guard in the order study).
const DT: f64 = 0.01;
const OUTER_ITERS: usize = 1;
const STEADY_TOL: f64 = 1e-5;
const STEADY_MAX_STEPS: usize = 1600;
/// Plateau detection: stop when the best (smallest) max-delta seen has not
/// improved by >2% within this many steps.
const PLATEAU_WINDOW: usize = 80;
/// No stop criterion may fire before this many steps: from the exact-solution
/// start the state must still travel to the discrete fixed point, and the
/// slowest (thermal, tau ~ 1.4 time units) mode needs ~4 tau to settle.
const MIN_STEPS: usize = 600;
/// Accept the state unconditionally after this many steps (t = 12, ~8x the
/// slowest physical mode). The mismatched-sources probe needs it: at an
/// O(1)-displaced solution the limiter wander occasionally sets a new best
/// delta and starves the plateau detector, though its amplitude is irrelevant
/// against the saturated error it measures.
const LONG_MARCH_ACCEPT_STEPS: usize = 1200;

// ---------------------------------------------------------------------------
// Exact solution.
// ---------------------------------------------------------------------------

fn exact_rho(x: f64, y: f64) -> f64 {
    RHO0 * (1.0 + amp_scale() * RHOA * (PI * x + PHX).sin() * (PI * y + PHY).sin())
}

/// Strain-free manufactured-field family toggle: uniform velocity (U0, V0)
/// with rho/p waves advecting through it (zero velocity gradients). Isolates
/// the interior instability from base-flow strain.
/// NOTE: uniform flow exits through the right/top faces — runs must use
/// left/bottom Inlet + right/top Outlet sides, not the all-Inlet box.
static UNIFORM_FLOW_FAMILY: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

fn uniform_family() -> bool {
    UNIFORM_FLOW_FAMILY.load(std::sync::atomic::Ordering::Relaxed)
}

/// Global amplitude scale on the manufactured PERTURBATIONS — the velocity
/// scales (U0/V0/UA/UB) and the rho/p wave amplitudes (RHOA/PA); the RHO0/P0
/// backgrounds are NOT scaled. Default 1.0 (bits 0 = unset = 1.0). The sources
/// derive from the same `amp_scale()`, so the scaled family stays a consistent
/// MMS.
static AMP_SCALE_BITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn amp_scale() -> f64 {
    let b = AMP_SCALE_BITS.load(std::sync::atomic::Ordering::Relaxed);
    if b == 0 {
        1.0
    } else {
        f64::from_bits(b)
    }
}

fn set_amp_scale(s: f64) {
    AMP_SCALE_BITS.store(s.to_bits(), std::sync::atomic::Ordering::Relaxed);
}

/// Mechanism toggles. RECON_UPWIND forces first-order Upwind (no gradient
/// reconstruction); MASS_COMPAT_OFF skips the source mass-compatibility
/// projection. Both default OFF.
static RECON_UPWIND: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static MASS_COMPAT_OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Select the KNP central-upwind MINMOD reconstruction (psi capped at 1)
/// instead of vanLeer. Default OFF (vanLeer). Ignored if RECON_UPWIND is set.
static RECON_MINMOD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn recon_upwind() -> bool {
    RECON_UPWIND.load(std::sync::atomic::Ordering::Relaxed)
}

fn recon_minmod() -> bool {
    RECON_MINMOD.load(std::sync::atomic::Ordering::Relaxed)
}

fn mass_compat_off() -> bool {
    MASS_COMPAT_OFF.load(std::sync::atomic::Ordering::Relaxed)
}

/// Run on a fully-periodic box (zero boundary faces) instead of the Dirichlet
/// rectangle, isolating the interior operator from the boundary stencil.
/// Default OFF. The boundary seeding in `build_run_box` no-ops and the
/// mass-compatibility `bflux` term is zero, so `eps` becomes the mean source —
/// the closed-system constraint a periodic domain requires.
static PERIODIC: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn periodic() -> bool {
    PERIODIC.load(std::sync::atomic::Ordering::Relaxed)
}

/// Toggle + coefficient for the IMPLICIT biharmonic-dissipation compressible
/// MMS model. When set, `build_run_box` builds
/// `compressible_mms_biharmonic_model()` (stride-12: the `lap_X = laplacian(X)`
/// constraint unknowns + `laplacian(-bih_eps4, lap_X)` on each conserved row)
/// and fills the per-cell `bih_eps4` field with `eps4_knob()`; otherwise the
/// plain MMS model. eps4 is a runtime field (no recompile to sweep). The
/// implicit form uses `PreconditionerType::BlockJacobi` (per-cell 12x12 inverse)
/// because point-Jacobi stalls on the grad^4 conditioning at fine mesh.
static BIHARMONIC: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static EPS4_BITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn biharmonic() -> bool {
    BIHARMONIC.load(std::sync::atomic::Ordering::Relaxed)
}

fn eps4_knob() -> f32 {
    f64::from_bits(EPS4_BITS.load(std::sync::atomic::Ordering::Relaxed)) as f32
}

fn set_eps4(v: f64) {
    EPS4_BITS.store(v.to_bits(), std::sync::atomic::Ordering::Relaxed);
}

/// Inflow on all four unit-box boundaries: u_x(0,·)=U0>0, u_x(1,·)=-U0,
/// u_y(·,0)=V0>0, u_y(·,1)=-V0 (the UA/UB terms vanish on the boundary).
fn exact_u(x: f64, y: f64) -> (f64, f64) {
    let s = amp_scale();
    if uniform_family() {
        return (U0 * s, V0 * s);
    }
    (
        s * (U0 * (PI * x).cos() + UA * (PI * x).sin() * (PI * y).sin()),
        s * (V0 * (PI * y).cos() + UB * (PI * x).sin() * (PI * y).sin()),
    )
}

/// Zero normal derivative on all four boundaries (sin(2 pi {0,1}) = 0):
/// required so the inlet "p follows interior" closure is O(h^2) consistent.
fn exact_p(x: f64, y: f64) -> f64 {
    P0 * (1.0 + amp_scale() * PA * (2.0 * PI * x).cos() * (2.0 * PI * y).cos())
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
    let a = RHO0 * RHOA * amp_scale();
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
    let s = amp_scale();
    if uniform_family() {
        return (U0 * s, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let (sx, cx) = (PI * x).sin_cos();
    let (sy, cy) = (PI * y).sin_cos();
    (
        s * (U0 * cx + UA * sx * sy),
        s * (-U0 * PI * sx + UA * PI * cx * sy),
        s * (UA * PI * sx * cy),
        s * (-U0 * PI * PI * cx - UA * PI * PI * sx * sy),
        s * (-UA * PI * PI * sx * sy),
        s * (UA * PI * PI * cx * cy),
    )
}

/// (v, v_x, v_y, v_xx, v_yy, v_xy)
fn v_partials(x: f64, y: f64) -> (f64, f64, f64, f64, f64, f64) {
    let s = amp_scale();
    if uniform_family() {
        return (V0 * s, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let (sx, cx) = (PI * x).sin_cos();
    let (sy, cy) = (PI * y).sin_cos();
    (
        s * (V0 * cy + UB * sx * sy),
        s * (UB * PI * cx * sy),
        s * (-V0 * PI * sy + UB * PI * sx * cy),
        s * (-UB * PI * PI * sx * sy),
        s * (-V0 * PI * PI * cy - UB * PI * PI * sx * sy),
        s * (UB * PI * PI * cx * cy),
    )
}

/// (p, p_x, p_y, p_xx, p_yy)
fn p_partials(x: f64, y: f64) -> (f64, f64, f64, f64, f64) {
    let (s2x, c2x) = (2.0 * PI * x).sin_cos();
    let (s2y, c2y) = (2.0 * PI * y).sin_cos();
    let a = P0 * PA * amp_scale();
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
    build_run_box(
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
        extra,
        mu,
        dt,
        time_scheme,
        outer_iters,
    )
}

/// Generalization of `build_run`: arbitrary rectangular box and per-side
/// boundary types. The manufactured fields/sources/BC closures are evaluated
/// at mesh coordinates, so any (lx, ly) works; when a side is Outlet, the
/// model's outlet closure needs its per-face Dirichlet p, seeded from the
/// exact solution like the inlet entries.
#[allow(clippy::too_many_arguments)]
fn build_run_box(
    nx: usize,
    ny: usize,
    lx: f64,
    ly: f64,
    sides: BoundarySides,
    extra: f64,
    mu: f64,
    dt: f64,
    time_scheme: TimeScheme,
    outer_iters: usize,
) -> SteadyRun {
    let has_outlet = [sides.left, sides.right, sides.bottom, sides.top]
        .iter()
        .any(|s| *s == BoundaryType::Outlet);
    let mesh = if periodic() {
        generate_structured_rect_mesh_periodic(nx, ny, lx, ly)
    } else {
        generate_structured_rect_mesh(nx, ny, lx, ly, sides)
    };
    let model = if biharmonic() {
        compressible_mms_biharmonic_model().expect("biharmonic model")
    } else {
        compressible_mms_model().expect("model")
    };
    let eos = EosSpec::IdealGas {
        gamma: GAMMA,
        gas_constant: R_GAS,
        temperature: 1.0,
    };
    let mut solver = pollster::block_on(UnifiedSolver::new(
        &mesh,
        model,
        SolverConfig {
            advection_scheme: if recon_upwind() {
                Scheme::Upwind
            } else if recon_minmod() {
                Scheme::SecondOrderUpwindMinMod
            } else {
                Scheme::SecondOrderUpwindVanLeer
            },
            time_scheme,
            // The implicit biharmonic couples a 4th-order (condition ~h^-4)
            // block whose stiffness is dominated by intra-cell coupling
            // (inv_dt-scaled recovery rows + the -I/+4 auxiliary-Laplacian
            // block, diagonal entries spanning ~2000x); point-Jacobi stalls at
            // fine mesh, the per-cell 12x12 block inverse resolves it in ~80
            // iters/step. Non-biharmonic runs keep point-Jacobi.
            preconditioner: if biharmonic() {
                PreconditionerType::BlockJacobi
            } else {
                PreconditionerType::Jacobi
            },
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
    // Thread the requested outer count and pin the adaptive outer-convergence
    // break OPEN (tol 0) so requested iterations actually run: the break's
    // relative tolerance otherwise fires after iteration 1 on these smooth
    // marches, collapsing the step to a single Picard iteration.
    solver.set_outer_iters(outer_iters).expect("outer_iters");
    solver.set_outer_tolerance(0.0).expect("outer_tol");
    solver.set_outer_tolerance_abs(0.0).expect("outer_tol_abs");
    println!("[probe] config: nx={nx} ny={ny} lx={lx} ly={ly} mu={mu} dt={dt} outer_iters={outer_iters} (break pinned open)");

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
    // Outlet sides: the model's outlet closure takes a per-face Dirichlet p;
    // everything else extrapolates via bc_expr.
    if has_outlet {
        solver
            .set_boundary_values_per_face(
                GpuBoundaryType::Outlet,
                "p",
                0,
                &scalar_face(&exact_p, &fx, &fy),
            )
            .expect("outlet p bc");
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
    // projection the solution drifts secularly and no discrete steady state
    // exists. Subtracting the uniform constant
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
    println!("[mms][compressible] nx={nx} mass-compatibility eps={eps:.3e}");
    if !mass_compat_off() {
        for s in src_rho.iter_mut() {
            *s -= eps;
        }
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
    if biharmonic() {
        // `bih_eps4` (= eps4 * acoustic-speed scale) is a uniform-valued storage
        // field (like mu); one registered model serves any eps4 with no shader
        // recompile. Default 0 keeps the dissipation inert.
        solver
            .set_field_scalar("bih_eps4", &vec![eps4_knob() as f64; cells])
            .expect("init bih_eps4");
    }
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
    assert_convergence_order("compressible_rho", &hs, &rho_errs, 2.0, 0.35, 1.5e-3);
    assert_convergence_order("compressible_u", &hs, &u_errs, 2.0, 0.35, 7.0e-4);
    assert_convergence_order("compressible_p", &hs, &p_errs, 2.0, 0.35, 1.2e-3);
    assert_convergence_order("compressible_T", &hs, &t_errs, 2.0, 0.40, 1.2e-3);
}

/// Euler-dominated variant of the oracle: the same manufactured solution and
/// harness with mu = 0 — no viscous stress, no conduction (k tracks mu), pure
/// KT/vanLeer convection + EOS recovery + pressure work. Orders the convective
/// operator in isolation (the Re ~ 8 NS study is viscosity-dominated, so a
/// convective defect could hide under the laplacians).
///
/// mu is 5e-3 not 0 because the coupled-implicit path has a marginal-mode
/// instability as mu -> 0, strongest at fine grids (damped only by physical
/// mu k^2): at mu = 0 it grows secularly at every dt and n=48/dt=0.01 diverges
/// outright. mu = 5e-3, dt = 5e-3 is clean through n = 32; n = 48 is still
/// marginal.
#[test]
fn steady_euler_dominated_vanleer_order() {
    let (hs, [rho_errs, u_errs, p_errs, t_errs]) =
        order_study(EXTRA_SHEAR, MU_EULER, DT_EULER, &[16, 24, 32], "euler");
    assert_convergence_order("euler_rho", &hs, &rho_errs, 2.0, 0.35, 4.5e-3);
    assert_convergence_order("euler_u", &hs, &u_errs, 2.0, 0.35, 8.5e-3);
    assert_convergence_order("euler_p", &hs, &p_errs, 2.0, 0.35, 9.0e-4);
    assert_convergence_order("euler_T", &hs, &t_errs, 2.0, 0.40, 4.5e-3);
}

// ---------------------------------------------------------------------------
// Inviscid-margin instability probes.
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
    // Domain extents derived from the mesh (probes run on boxes other than the
    // unit square); assumes uniform square cells.
    let lx = run.mesh.vx.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ly = run.mesh.vy.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let h = (lx * ly / run.mesh.num_cells() as f64).sqrt();
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
        if x < band || x > lx - band || y < band || y > ly - band {
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

/// Probe matrix for the inviscid-margin instability. All runs at mu = 0,
/// dt = 5e-3 (the slow-growth regime; the fast CFL~1 branch at dt = 0.01/n=48
/// is the last row). Reports growth %/tu, p-nyquist, boundary fraction, and
/// final u/rho error levels for each time-scheme / outer-iter / preconditioning
/// configuration; the levels catch fast blow-ups that a second-half rate fit
/// misses. Stability-envelope contract: time-accurate compressible marching
/// requires nonzero physical viscosity — the inviscid limit is out of envelope
/// on this discretization at moderate Mach.
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

    // H2: outer iterations (converged Picard = implicit-in-flux).
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

/// Strain-free (uniform-velocity) family at mu = 0: with zero base-flow
/// velocity gradients there is no production mechanism, so a faithful
/// discretization should march it stably and converge at design order. Uniform
/// flow needs right/top Outlets, so this also exercises the Outlet closure.
#[test]
#[ignore]
fn probe_arcn_uniform_mu0() {
    use std::sync::atomic::Ordering;
    UNIFORM_FLOW_FAMILY.store(true, Ordering::Relaxed);
    let dt = 5.0e-3;
    let steps = 600;
    let sides = BoundarySides {
        left: BoundaryType::Inlet,
        right: BoundaryType::Outlet,
        bottom: BoundaryType::Inlet,
        top: BoundaryType::Outlet,
    };
    println!(
        "[arcn-uniform] {:34} {:>12} {:>8} {:>8} {:>9} {:>9}",
        "config", "growth %/tu", "nyq(p)", "bfrac", "u_l2", "rho_l2"
    );
    let mut report = |label: &str, mut run: SteadyRun| {
        let (rate, nyq_p, bfrac, u_l2, rho_l2) = measure_growth(&mut run, dt, steps, 25);
        println!("[arcn-uniform] {label:34} {rate:12.2} {nyq_p:8.4} {bfrac:8.4} {u_l2:9.2e} {rho_l2:9.2e}");
    };
    report(
        "uniform mu0 BDF2 o1 n32",
        build_run_box(32, 32, 1.0, 1.0, sides, EXTRA_SHEAR, 0.0, dt, TimeScheme::BDF2, 1),
    );
    report(
        "uniform mu0 BDF2 o1 n48",
        build_run_box(48, 48, 1.0, 1.0, sides, EXTRA_SHEAR, 0.0, dt, TimeScheme::BDF2, 1),
    );
    report(
        "uniform mu0 BDF2 o2 n48 (conv)",
        build_run_box(48, 48, 1.0, 1.0, sides, EXTRA_SHEAR, 0.0, dt, TimeScheme::BDF2, 2),
    );

    // Order study at mu = 0: march to plateau, read errors, fit first-last.
    let mut hs = Vec::new();
    let mut errs: Vec<[f64; 4]> = Vec::new();
    for &n in &[16usize, 24, 32] {
        let mut run =
            build_run_box(n, n, 1.0, 1.0, sides, EXTRA_SHEAR, 0.0, dt, TimeScheme::BDF2, 1);
        march_to_plateau(&mut run.solver);
        let (rho_err, u_err, p_err, t_err) = read_errors(&run);
        println!("[arcn-uniform] n={n} rho_l2={rho_err:.4e} u_l2={u_err:.4e} p_l2={p_err:.4e} t_l2={t_err:.4e}");
        hs.push(1.0 / n as f64);
        errs.push([rho_err, u_err, p_err, t_err]);
    }
    for (k, name) in ["rho", "u", "p", "T"].iter().enumerate() {
        let o = ((errs[0][k] / errs[errs.len() - 1][k]).ln())
            / ((hs[0] / hs[hs.len() - 1]).ln());
        println!("[arcn-uniform] {name} order(first-last) = {o:.3}");
    }
    UNIFORM_FLOW_FAMILY.store(false, Ordering::Relaxed);
}

/// Boundary-vs-interior discriminators at MATCHED h.
/// - [0,3]² all-Inlet (odd L keeps the all-inflow boundary structure:
///   cos(3π) = −1, sin(3π) = 0): an interior-driven mode must reproduce the
///   unit-box rate; a boundary-driven one scales with perimeter/area (×1/3).
/// - [0,2]² left/bottom Inlet + right/top Outlet (genuine outflow, u_x(2,y) =
///   +U0): changes the boundary reflection structure; a strongly different rate
///   convicts the boundary closure.
#[test]
#[ignore]
fn probe_arcn_domain_matrix() {
    let dt = 5.0e-3;
    let steps = 600;
    let all_inlet = BoundarySides {
        left: BoundaryType::Inlet,
        right: BoundaryType::Inlet,
        bottom: BoundaryType::Inlet,
        top: BoundaryType::Inlet,
    };
    let inlet_outlet = BoundarySides {
        left: BoundaryType::Inlet,
        right: BoundaryType::Outlet,
        bottom: BoundaryType::Inlet,
        top: BoundaryType::Outlet,
    };
    println!(
        "[arcn-domain] {:34} {:>12} {:>8} {:>8} {:>9} {:>9}",
        "config", "growth %/tu", "nyq(p)", "bfrac", "u_l2", "rho_l2"
    );
    let mut report = |label: &str, mut run: SteadyRun| {
        let (rate, nyq_p, bfrac, u_l2, rho_l2) = measure_growth(&mut run, dt, steps, 25);
        println!("[arcn-domain] {label:34} {rate:12.2} {nyq_p:8.4} {bfrac:8.4} {u_l2:9.2e} {rho_l2:9.2e}");
    };
    report(
        "L1 allIn n32 (base)",
        build_run_box(32, 32, 1.0, 1.0, all_inlet, EXTRA_SHEAR, 0.0, dt, TimeScheme::BDF2, 1),
    );
    report(
        "L1 allIn n48 (base)",
        build_run_box(48, 48, 1.0, 1.0, all_inlet, EXTRA_SHEAR, 0.0, dt, TimeScheme::BDF2, 1),
    );
    report(
        "L3 allIn n96  (h=1/32)",
        build_run_box(96, 96, 3.0, 3.0, all_inlet, EXTRA_SHEAR, 0.0, dt, TimeScheme::BDF2, 1),
    );
    report(
        "L3 allIn n144 (h=1/48)",
        build_run_box(144, 144, 3.0, 3.0, all_inlet, EXTRA_SHEAR, 0.0, dt, TimeScheme::BDF2, 1),
    );
    report(
        "L2 in/out n64 (h=1/32)",
        build_run_box(64, 64, 2.0, 2.0, inlet_outlet, EXTRA_SHEAR, 0.0, dt, TimeScheme::BDF2, 1),
    );
    report(
        "L2 in/out n96 (h=1/48)",
        build_run_box(96, 96, 2.0, 2.0, inlet_outlet, EXTRA_SHEAR, 0.0, dt, TimeScheme::BDF2, 1),
    );
}

/// Is the boundary-band μ=0 instability genuine? Converged-Picard (outer=2),
/// small amplitude (perturbations ×AMP so the amplitude-independent boundary
/// mode dominates the amplitude-linear interior KH physics), all-Inlet box,
/// n = 32/48/64/96. The growth-rate trend under refinement is the verdict:
/// rate → 0 with h ⇒ benign O(h) boundary inconsistency; rate constant/↑ ⇒
/// genuine discrete instability.
#[test]
#[ignore]
fn probe_arcn_refinement_trend() {
    let dt = 5.0e-3;
    let steps = 600;
    let amp = 0.25;
    set_amp_scale(amp);
    println!(
        "[arcn-refine] amp={amp}  {:14} {:>12} {:>8} {:>8} {:>9} {:>9}",
        "config", "growth %/tu", "nyq(p)", "bfrac", "u_l2", "rho_l2"
    );
    for &n in &[32usize, 48, 64, 96] {
        // outer=2 = converged Picard (build_run pins the outer break open).
        let mut run = build_run(n, EXTRA_SHEAR, 0.0, dt, TimeScheme::BDF2, 2);
        let (rate, nyq_p, bfrac, u_l2, rho_l2) = measure_growth(&mut run, dt, steps, 25);
        println!("[arcn-refine] o2 n={n:<11} {rate:12.2} {nyq_p:8.4} {bfrac:8.4} {u_l2:9.2e} {rho_l2:9.2e}");
    }
    set_amp_scale(1.0);
}

/// Mechanism isolation for the grid-scale boundary instability (small
/// amplitude, converged Picard, all-Inlet). Two discriminators:
/// - Upwind vs vanLeer: first-order Upwind disables the whole reconstruction
///   path; if the mode vanishes under Upwind, the 2nd-order vanLeer
///   reconstruction is the source.
/// - mass-compat on/off: rules out the source projection (expected null, O(h²)).
#[test]
#[ignore]
fn probe_arcn_mechanism() {
    use std::sync::atomic::Ordering;
    let dt = 5.0e-3;
    let steps = 600;
    set_amp_scale(0.25);
    println!(
        "[arcn-mech] amp=0.25 o2  {:20} {:>12} {:>8} {:>8} {:>9} {:>9}",
        "config", "growth %/tu", "nyq(p)", "bfrac", "u_l2", "rho_l2"
    );
    let mut report = |label: &str, mut run: SteadyRun| {
        let (rate, nyq_p, bfrac, u_l2, rho_l2) = measure_growth(&mut run, dt, steps, 25);
        println!("[arcn-mech] {label:20} {rate:12.2} {nyq_p:8.4} {bfrac:8.4} {u_l2:9.2e} {rho_l2:9.2e}");
    };
    // Baseline vanLeer.
    report("vanLeer n32", build_run(32, EXTRA_SHEAR, 0.0, dt, TimeScheme::BDF2, 2));
    report("vanLeer n48", build_run(48, EXTRA_SHEAR, 0.0, dt, TimeScheme::BDF2, 2));
    // First-order Upwind (no gradient reconstruction).
    RECON_UPWIND.store(true, Ordering::Relaxed);
    report("upwind n32", build_run(32, EXTRA_SHEAR, 0.0, dt, TimeScheme::BDF2, 2));
    report("upwind n48", build_run(48, EXTRA_SHEAR, 0.0, dt, TimeScheme::BDF2, 2));
    report("upwind n64", build_run(64, EXTRA_SHEAR, 0.0, dt, TimeScheme::BDF2, 2));
    RECON_UPWIND.store(false, Ordering::Relaxed);
    // Mass-compat projection off.
    MASS_COMPAT_OFF.store(true, Ordering::Relaxed);
    report("no-masscompat n48", build_run(48, EXTRA_SHEAR, 0.0, dt, TimeScheme::BDF2, 2));
    MASS_COMPAT_OFF.store(false, Ordering::Relaxed);
    set_amp_scale(1.0);
}

/// Periodic-domain instrument: the interior-vs-boundary separator. A
/// fully-periodic box has ZERO boundary faces, so any growth is the interior
/// operator alone. The manufactured fields are 2-periodic, so the box is
/// [0,2]^2 at 2n cells/side (h = 1/n), matched to the unit-box rows. Small
/// amplitude (0.25), mu = 0, converged Picard (outer=2, break pinned open).
#[test]
#[ignore]
fn probe_arcn_periodic() {
    use std::sync::atomic::Ordering;
    let dt = 5.0e-3;
    let steps = 600;
    let amp = 0.25;
    set_amp_scale(amp);
    PERIODIC.store(true, Ordering::Relaxed);
    // `sides` is ignored on a periodic mesh (it has no boundary faces).
    let inlet = BoundarySides {
        left: BoundaryType::Inlet,
        right: BoundaryType::Inlet,
        bottom: BoundaryType::Inlet,
        top: BoundaryType::Inlet,
    };
    println!(
        "[arcn-periodic] amp={amp} mu=0 [0,2]^2  {:12} {:>12} {:>9} {:>9}",
        "config", "growth %/tu", "u_l2", "rho_l2"
    );
    // 2n cells over [0,2] => h = 1/n, matched to the unit-box rows.
    for &n in &[32usize, 48, 64, 96] {
        let cells = 2 * n;
        // outer=2 = converged Picard (build_run_box pins the outer break open).
        let mut run = build_run_box(
            cells, cells, 2.0, 2.0, inlet, EXTRA_SHEAR, 0.0, dt, TimeScheme::BDF2, 2,
        );
        let (rate, _nyq_p, _bfrac, u_l2, rho_l2) = measure_growth(&mut run, dt, steps, 25);
        println!("[arcn-periodic] h=1/{n:<7} {rate:12.2} {u_l2:9.2e} {rho_l2:9.2e}");
    }
    PERIODIC.store(false, Ordering::Relaxed);
    set_amp_scale(1.0);
}

/// Does the sharper MINMOD limiter (KNP central-upwind) suppress the interior
/// grid-scale instability that vanLeer under-dissipates? A/B on the
/// boundary-free [0,2]^2 periodic box, small amplitude, mu=0, converged Picard.
/// SUCCESS = minmod growth bounded / <= 0 where vanLeer diverges, at fine h.
/// (The mu>0 MMS-order gate is separate.)
#[test]
#[ignore]
fn probe_arcn_minmod() {
    use std::sync::atomic::Ordering;
    let dt = 5.0e-3;
    let steps = 600;
    set_amp_scale(0.25);
    PERIODIC.store(true, Ordering::Relaxed);
    let inlet = BoundarySides {
        left: BoundaryType::Inlet,
        right: BoundaryType::Inlet,
        bottom: BoundaryType::Inlet,
        top: BoundaryType::Inlet,
    };
    println!(
        "[arcn-minmod] amp=0.25 mu=0 [0,2]^2  {:16} {:>12} {:>9} {:>9}",
        "config", "growth %/tu", "u_l2", "rho_l2"
    );
    for &n in &[32usize, 48, 64, 96] {
        let cells = 2 * n;
        for (label, minmod) in [("vanLeer", false), ("minmod", true)] {
            RECON_MINMOD.store(minmod, Ordering::Relaxed);
            let mut run = build_run_box(
                cells, cells, 2.0, 2.0, inlet, EXTRA_SHEAR, 0.0, dt, TimeScheme::BDF2, 2,
            );
            let (rate, _nyq, _bfrac, u_l2, rho_l2) = measure_growth(&mut run, dt, steps, 25);
            println!("[arcn-minmod] {label:>7} h=1/{n:<7} {rate:12.2} {u_l2:9.2e} {rho_l2:9.2e}");
        }
    }
    RECON_MINMOD.store(false, Ordering::Relaxed);
    PERIODIC.store(false, Ordering::Relaxed);
    set_amp_scale(1.0);
}

/// k-selective BIHARMONIC dissipation. Sweeps eps4 on the boundary-free
/// [0,2]^2 periodic box, small amplitude, mu=0, converged Picard, at the meshes
/// where vanLeer and minmod diverge. eps4=0 is the control (term x0, must
/// reproduce the vanLeer divergence). SUCCESS = an eps4>0 that bounds the n>=64
/// growth without blowing up coarse h; a wrong dissipation sign diverges HARDER
/// than eps4=0. The mu>0 order gate is `probe_arcn_biharmonic_order`.
#[test]
#[ignore]
fn probe_arcn_biharmonic() {
    use std::sync::atomic::Ordering;
    let dt = 5.0e-3;
    let steps = 600;
    set_amp_scale(0.25);
    PERIODIC.store(true, Ordering::Relaxed);
    BIHARMONIC.store(true, Ordering::Relaxed);
    let inlet = BoundarySides {
        left: BoundaryType::Inlet,
        right: BoundaryType::Inlet,
        bottom: BoundaryType::Inlet,
        top: BoundaryType::Inlet,
    };
    println!(
        "[arcn-bih] amp=0.25 mu=0 [0,2]^2  {:>6} {:>6} {:>12} {:>9} {:>9}",
        "h=1/n", "eps4", "growth %/tu", "u_l2", "rho_l2"
    );
    for &n in &[32usize, 48, 64, 96] {
        let cells = 2 * n;
        for &eps4 in &[0.0_f64, 0.1, 0.25, 0.5] {
            set_eps4(eps4);
            let mut run = build_run_box(
                cells, cells, 2.0, 2.0, inlet, EXTRA_SHEAR, 0.0, dt, TimeScheme::BDF2, 2,
            );
            let (rate, _nyq, _bfrac, u_l2, rho_l2) = measure_growth(&mut run, dt, steps, 25);
            println!(
                "[arcn-bih] {:>6} {eps4:>6.2} {rate:12.2} {u_l2:9.2e} {rho_l2:9.2e}",
                format!("1/{n}")
            );
        }
    }
    set_eps4(0.0);
    BIHARMONIC.store(false, Ordering::Relaxed);
    PERIODIC.store(false, Ordering::Relaxed);
    set_amp_scale(1.0);
}

/// Minmod's mu>0 convergence order — the accuracy gate. Minmod can clip to
/// first order at smooth extrema, so this measures whether KNP+minmod still
/// reaches order ~2. Reports observed orders without asserting.
#[test]
#[ignore]
fn probe_arcn_minmod_order() {
    use std::sync::atomic::Ordering;
    RECON_MINMOD.store(true, Ordering::Relaxed);
    let levels = [16usize, 24, 32, 48];
    let (hs, [rho_errs, u_errs, p_errs, t_errs]) =
        order_study(EXTRA_SHEAR, MU, DT, &levels, "minmod");
    RECON_MINMOD.store(false, Ordering::Relaxed);
    let fit = |errs: &[f64]| -> f64 {
        let k = errs.len();
        (errs[k - 2] / errs[k - 1]).ln() / (hs[k - 2] / hs[k - 1]).ln()
    };
    for (name, errs) in [
        ("rho", &rho_errs),
        ("u", &u_errs),
        ("p", &p_errs),
        ("T", &t_errs),
    ] {
        println!(
            "[arcn-minmod-order] {name}: finest-pair order={:.3} finest_err={:.3e}",
            fit(errs),
            errs[errs.len() - 1]
        );
    }
}

/// IMPLICIT biharmonic mu>0 convergence order. eps4=0 (control) reproduces
/// plain `compressible_mms` to order ~2, proving the stride-12 mixed formulation
/// is correct. For eps4>0 the sub-2 orders are the biharmonic hyperviscosity
/// consistency error (the -eps4*grad^4 term assembles as ~eps4*h^2*grad^4 U, an
/// O(h^2) inconsistency with the non-biharmonic MMS sources), trending to 2
/// under refinement.
#[test]
#[ignore]
fn probe_arcn_biharmonic_order() {
    use std::sync::atomic::Ordering;
    BIHARMONIC.store(true, Ordering::Relaxed);
    let levels = [16usize, 24, 32, 48];
    let fit = |hs: &[f64], errs: &[f64]| -> f64 {
        let k = errs.len();
        (errs[k - 2] / errs[k - 1]).ln() / (hs[k - 2] / hs[k - 1]).ln()
    };
    for &eps4 in &[0.0_f64, 0.1, 0.25] {
        set_eps4(eps4);
        let (hs, [rho_errs, u_errs, p_errs, t_errs]) =
            order_study(EXTRA_SHEAR, MU, DT, &levels, "bih");
        for (name, errs) in [
            ("rho", &rho_errs),
            ("u", &u_errs),
            ("p", &p_errs),
            ("T", &t_errs),
        ] {
            println!(
                "[arcn-bih-order] eps4={eps4:.2} {name}: order={:.3} finest_err={:.3e}",
                fit(&hs, errs),
                errs[errs.len() - 1]
            );
        }
    }
    BIHARMONIC.store(false, Ordering::Relaxed);
    set_eps4(0.0);
}

/// Discriminator: biharmonic mu>0 order on the boundary-free PERIODIC box. The
/// Dirichlet mu>0 order gate blows up for eps4>0; this isolates whether the
/// cause is the boundary closure (periodic then order ~2) or a mu>0+biharmonic
/// interaction (periodic also blows up). Full amplitude, MU, march-to-plateau,
/// on [0,2]^2 periodic.
#[test]
#[ignore]
fn probe_arcn_biharmonic_periodic_order() {
    use std::sync::atomic::Ordering;
    PERIODIC.store(true, Ordering::Relaxed);
    BIHARMONIC.store(true, Ordering::Relaxed);
    let inlet = BoundarySides {
        left: BoundaryType::Inlet,
        right: BoundaryType::Inlet,
        bottom: BoundaryType::Inlet,
        top: BoundaryType::Inlet,
    };
    let levels = [16usize, 24, 32, 48];
    let fit = |hs: &[f64], errs: &[f64]| -> f64 {
        let k = errs.len();
        (errs[k - 2] / errs[k - 1]).ln() / (hs[k - 2] / hs[k - 1]).ln()
    };
    for &eps4 in &[0.0_f64, 0.1] {
        set_eps4(eps4);
        let mut hs = Vec::new();
        let mut rho_errs = Vec::new();
        let mut u_errs = Vec::new();
        for &n in &levels {
            let cells = 2 * n; // h = 1/n on [0,2]^2
            let mut run = build_run_box(
                cells, cells, 2.0, 2.0, inlet, EXTRA_SHEAR, MU, DT, TimeScheme::BDF2, OUTER_ITERS,
            );
            march_to_plateau(&mut run.solver);
            let (rho_err, u_err, _p, _t) = read_errors(&run);
            hs.push(1.0 / n as f64);
            rho_errs.push(rho_err);
            u_errs.push(u_err);
            println!("[arcn-bih-perord] eps4={eps4:.2} n={n} rho_l2={rho_err:.4e} u_l2={u_err:.4e}");
        }
        println!(
            "[arcn-bih-perord] eps4={eps4:.2} ORDER rho={:.3} u={:.3}",
            fit(&hs, &rho_errs),
            fit(&hs, &u_errs)
        );
    }
    BIHARMONIC.store(false, Ordering::Relaxed);
    set_eps4(0.0);
    PERIODIC.store(false, Ordering::Relaxed);
}

/// Is the Dirichlet mu>0 biharmonic blow-up an EXPLICIT-stiffness problem? The
/// term reads a frozen (lagged) `lap`, so it acts like an explicit 4th-difference
/// (stability limit dt ~ h^4), which OUTER_ITERS=1 + dt=0.01 violates. Varies
/// outer-iterations and dt at eps4=0.1 on the Dirichlet box; if order recovers to
/// ~2, the fix is iteration/relaxation, not a boundary treatment.
#[test]
#[ignore]
fn probe_arcn_biharmonic_stab() {
    use std::sync::atomic::Ordering;
    BIHARMONIC.store(true, Ordering::Relaxed);
    set_eps4(0.1);
    let levels = [16usize, 24, 32, 48];
    let fit = |hs: &[f64], errs: &[f64]| -> f64 {
        let k = errs.len();
        (errs[k - 2] / errs[k - 1]).ln() / (hs[k - 2] / hs[k - 1]).ln()
    };
    for &(outer, dt) in &[(1usize, 0.01_f64), (8, 0.01), (1, 0.002)] {
        let mut hs = Vec::new();
        let mut u_errs = Vec::new();
        for &n in &levels {
            let mut run = build_run(n, EXTRA_SHEAR, MU, dt, TimeScheme::BDF2, outer);
            march_to_plateau(&mut run.solver);
            let (_rho, u_err, _p, _t) = read_errors(&run);
            hs.push(1.0 / n as f64);
            u_errs.push(u_err);
            println!("[arcn-bih-stab] outer={outer} dt={dt} n={n} u_l2={u_err:.4e}");
        }
        println!(
            "[arcn-bih-stab] outer={outer} dt={dt} ORDER u={:.3}",
            fit(&hs, &u_errs)
        );
    }
    BIHARMONIC.store(false, Ordering::Relaxed);
    set_eps4(0.0);
}

/// Per-step FGMRES telemetry for the implicit biharmonic solve: classifies the
/// linear-solve failure mode and compares preconditioners. Prints, for each
/// (eps4, n), per-step (iterations, residual, converged/diverged) plus a SUMMARY
/// with total iterations, wall time, and the short-march errors. Env knobs (one
/// compile, many runs):
///   BIH_DIAG_PRECOND  = jacobi|block|amg     (default jacobi)
///   BIH_DIAG_RESTART  = <usize>              (<= build-time capacity 60; default model)
///   BIH_DIAG_MAXITERS = <u32>                (default model 4000)
///   BIH_DIAG_TOL      = <f32>                (default model 1e-4)
///   BIH_DIAG_EPS4     = "0,0.25"             (comma list)
///   BIH_DIAG_N        = "16,32"              (comma list)
///   BIH_DIAG_STEPS    = <usize>              (default 8)
#[test]
#[ignore]
fn probe_arcn_biharmonic_diag() {
    use cfd2::solver::gpu::unified_solver::PlanParamValue;
    use std::sync::atomic::Ordering;
    BIHARMONIC.store(true, Ordering::Relaxed);

    let precond_name = std::env::var("BIH_DIAG_PRECOND").unwrap_or_else(|_| "jacobi".into());
    let precond = match precond_name.as_str() {
        "block" => PreconditionerType::BlockJacobi,
        "amg" => PreconditionerType::Amg,
        _ => PreconditionerType::Jacobi,
    };
    let restart: Option<usize> = std::env::var("BIH_DIAG_RESTART").ok().and_then(|s| s.parse().ok());
    let max_iters: Option<u32> = std::env::var("BIH_DIAG_MAXITERS").ok().and_then(|s| s.parse().ok());
    let tol: Option<f32> = std::env::var("BIH_DIAG_TOL").ok().and_then(|s| s.parse().ok());
    let parse_list_f64 = |s: String| -> Vec<f64> {
        s.split(',').filter_map(|x| x.trim().parse().ok()).collect()
    };
    let parse_list_usize = |s: String| -> Vec<usize> {
        s.split(',').filter_map(|x| x.trim().parse().ok()).collect()
    };
    let eps4s = std::env::var("BIH_DIAG_EPS4").ok().map(parse_list_f64).unwrap_or_else(|| vec![0.0, 0.25]);
    let ns = std::env::var("BIH_DIAG_N").ok().map(parse_list_usize).unwrap_or_else(|| vec![16, 32]);
    let steps: usize = std::env::var("BIH_DIAG_STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(8);

    println!(
        "[bih-diag] precond={precond_name} restart={restart:?} max_iters={max_iters:?} tol={tol:?} eps4s={eps4s:?} ns={ns:?} steps={steps}"
    );
    for &eps4 in &eps4s {
        set_eps4(eps4);
        for &n in &ns {
            let mut run = build_run(n, EXTRA_SHEAR, MU, DT, TimeScheme::BDF2, OUTER_ITERS);
            run.solver.set_preconditioner(precond);
            if let Some(r) = restart {
                run.solver
                    .set_named_param("linear_solver.max_restart", PlanParamValue::Usize(r))
                    .expect("set max_restart");
            }
            if let Some(mi) = max_iters {
                run.solver
                    .set_named_param("linear_solver.max_iters", PlanParamValue::U32(mi))
                    .expect("set max_iters");
            }
            if let Some(t) = tol {
                run.solver
                    .set_named_param("linear_solver.tolerance", PlanParamValue::F32(t))
                    .expect("set tolerance");
            }
            let mut total_iters = 0u64;
            let mut any_diverged = false;
            let t0 = std::time::Instant::now();
            for s in 0..steps {
                // step_with_stats returns (first, best, last) across the step's
                // outer iterations; OUTER_ITERS=1 makes them identical, so the
                // representative single solve is the first entry.
                let stats = run.solver.step_with_stats().expect("step_with_stats");
                let st = stats.first().copied().unwrap_or_default();
                total_iters += st.iterations as u64;
                any_diverged |= st.diverged;
                println!(
                    "[bih-diag] eps4={eps4:.2} n={n} step={s} iters={} resid={:.3e} conv={} div={} t={:.1}ms",
                    st.iterations,
                    st.residual,
                    st.converged,
                    st.diverged,
                    st.time.as_secs_f64() * 1e3
                );
            }
            let (rho_err, u_err, p_err, t_err) = read_errors(&run);
            println!(
                "[bih-diag] SUMMARY eps4={eps4:.2} n={n} precond={precond_name} steps={steps} total_iters={total_iters} avg_iters={:.0} wall={:.2}s diverged={any_diverged} rho_l2={rho_err:.3e} u_l2={u_err:.3e} p_l2={p_err:.3e} t_l2={t_err:.3e}",
                total_iters as f64 / steps as f64,
                t0.elapsed().as_secs_f64()
            );
        }
    }
    BIHARMONIC.store(false, Ordering::Relaxed);
    set_eps4(0.0);
}

/// Eigenmode dump: capture the growing inviscid mode's spatial structure.
/// Marches n=48 / mu=0 / BDF2 / o1, snapshots per-cell deltas vs the exact
/// solution at steps 400 and 500, and writes CSV to
/// target/arcn_probes/eigenmode_n48.csv with columns x,y,d1_<f>,d2_<f>,g_<f>
/// for f in rho,ux,uy,p,T — d1/d2 are the two snapshots' deltas and g = d2 - d1
/// is the growing-mode component (the difference cancels the steady O(h^2)
/// discretization error).
#[test]
#[ignore]
fn probe_arcn_eigenmode_dump() {
    let dt = 5.0e-3;
    let n = 48;
    let mut run = build_run(n, EXTRA_SHEAR, 0.0, dt, TimeScheme::BDF2, 1);

    let read_deltas = |run: &SteadyRun| -> Vec<[f64; 5]> {
        let u = pollster::block_on(run.solver.get_field_vec2("u")).expect("read u");
        let rho = pollster::block_on(run.solver.get_field_scalar("rho")).expect("read rho");
        let p = pollster::block_on(run.solver.get_field_scalar("p")).expect("read p");
        let t = pollster::block_on(run.solver.get_field_scalar("T")).expect("read T");
        (0..run.mesh.num_cells())
            .map(|i| {
                let (x, y) = (run.mesh.cell_cx[i], run.mesh.cell_cy[i]);
                let (eux, euy) = exact_u(x, y);
                [
                    rho[i] - exact_rho(x, y),
                    u[i].0 - eux,
                    u[i].1 - euy,
                    p[i] - exact_p(x, y),
                    t[i] - exact_t(x, y),
                ]
            })
            .collect()
    };

    for _ in 0..400 {
        run.solver.step();
    }
    let d1 = read_deltas(&run);
    for _ in 0..100 {
        run.solver.step();
    }
    let d2 = read_deltas(&run);

    std::fs::create_dir_all("target/arcn_probes").expect("mkdir");
    let mut csv = String::from("x,y,d1_rho,d1_ux,d1_uy,d1_p,d1_T,d2_rho,d2_ux,d2_uy,d2_p,d2_T,g_rho,g_ux,g_uy,g_p,g_T\n");
    for i in 0..run.mesh.num_cells() {
        let (x, y) = (run.mesh.cell_cx[i], run.mesh.cell_cy[i]);
        csv.push_str(&format!("{x},{y}"));
        for v in &d1[i] {
            csv.push_str(&format!(",{v:.6e}"));
        }
        for v in &d2[i] {
            csv.push_str(&format!(",{v:.6e}"));
        }
        for k in 0..5 {
            csv.push_str(&format!(",{:.6e}", d2[i][k] - d1[i][k]));
        }
        csv.push('\n');
    }
    std::fs::write("target/arcn_probes/eigenmode_n48.csv", csv).expect("write csv");

    // Quick in-test summary: RMS of the growth component per field.
    let mut rms = [0.0f64; 5];
    for i in 0..d1.len() {
        for k in 0..5 {
            let g = d2[i][k] - d1[i][k];
            rms[k] += g * g;
        }
    }
    let nn = d1.len() as f64;
    println!(
        "[arcn-mode] growth-component rms: rho={:.3e} ux={:.3e} uy={:.3e} p={:.3e} T={:.3e}",
        (rms[0] / nn).sqrt(),
        (rms[1] / nn).sqrt(),
        (rms[2] / nn).sqrt(),
        (rms[3] / nn).sqrt(),
        (rms[4] / nn).sqrt()
    );
}

/// Diagnostic: does the outer Picard loop actually refresh the KT flux/assembly
/// inputs, or is the step effectively explicit-in-flux? Run with
/// CFD2_DEBUG_FGMRES=1 CFD2_LIN_TOL=1e-7 and read the per-solve trace within
/// each step.
#[test]
#[ignore]
fn probe_arcr_outer_refresh() {
    let dt = 5.0e-3;
    let mut run = build_run(32, EXTRA_SHEAR, 0.0, dt, TimeScheme::BDF2, 4);
    // The batched one-submission outer loop encodes all solves up front with
    // no readback — the FGMRES trace never fires there. Force the
    // per-iteration path so each solve is visible.
    run.solver
        .set_outer_batched_mode(false)
        .expect("outer_batched_mode");
    for k in 0..3 {
        run.solver.step();
        println!("[arcr-refresh] completed step {}", k + 1);
    }
}

/// Order study of the UNPRECONDITIONED operator at mu = 0, for measuring
/// candidate flux-dissipation designs against the success criterion (stable at
/// mu = 0 AND orders ~2). Run with a candidate dissipation active in
/// flux_schemes.rs.
#[test]
#[ignore]
fn probe_arck_mu0_order() {
    let dt = 5.0e-3;
    let mut hs = Vec::new();
    let mut rho_errs = Vec::new();
    let mut u_errs = Vec::new();
    let mut p_errs = Vec::new();
    let mut t_errs = Vec::new();
    for &n in &[16usize, 24, 32, 48] {
        let mut run = build_run(n, EXTRA_SHEAR, 0.0, dt, TimeScheme::BDF2, 1);
        march_to_plateau(&mut run.solver);
        let (rho_err, u_err, p_err, t_err) = read_errors(&run);
        println!("[arck-mu0] n={n} rho_l2={rho_err:.4e} u_l2={u_err:.4e} p_l2={p_err:.4e} t_l2={t_err:.4e}");
        hs.push(1.0 / n as f64);
        rho_errs.push(rho_err);
        u_errs.push(u_err);
        p_errs.push(p_err);
        t_errs.push(t_err);
    }
    for (name, errs) in [
        ("rho", &rho_errs),
        ("u", &u_errs),
        ("p", &p_errs),
        ("T", &t_errs),
    ] {
        let o = ((errs[0] / errs[errs.len() - 1]).ln()) / ((hs[0] / hs[hs.len() - 1]).ln());
        println!("[arck-mu0] {name} order(first-last) = {o:.3}");
    }
}

/// Order study of the PRECONDITIONED inviscid operator: mu = 0 with dual-time
/// WeissSmith preconditioning. Kept as a falsification record for "just use
/// preconditioning for inviscid": at moderate Mach the preconditioned
/// wave-speed rescale removes dissipation and the run blows up (rho/p ->
/// 1e10..1e12 by n=24, while u stays bounded through the rho_u/rho recovery).
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

/// Sources for the doubled-shear operator (`extra = 1`). With physical-NS in
/// the solver these must SATURATE h-independently instead of converging — if
/// this probe ever shows order-2 convergence again, the double-counted
/// laplacian has been reintroduced.
/// Run manually:
/// `cargo test --features dev-tests --test mms_compressible_order_test -- --ignored probe_ --nocapture`
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
