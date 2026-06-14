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
use cfd2::solver::gpu::unified_solver::PlanParamValue;
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
    RHO0 * (1.0 + amp_scale() * RHOA * (PI * x + PHX).sin() * (PI * y + PHY).sin())
}

/// ARC N: strain-free manufactured-field family toggle (uniform velocity
/// (U0, V0); rho/p waves advect through it). The default (strained) family's
/// base flow is inviscidly UNSTABLE as physics — O(1) strain with
/// inflection-point shear; measured growth is linear in the velocity
/// amplitude (116/76/56.6 %/tu at amp 1/0.5/0.25 on the converged-Picard
/// rows, h-independent) and matches the strain-rate scale. The uniform
/// family has zero velocity gradients (no production mechanism), so a
/// faithful discretization must march it stably at mu = 0: the diagnosis's
/// falsifiable counterpart, exercised by `probe_arcn_uniform_mu0`.
/// NOTE: uniform flow exits through the right/top faces — runs must use
/// left/bottom Inlet + right/top Outlet sides, not the all-Inlet box.
static UNIFORM_FLOW_FAMILY: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

fn uniform_family() -> bool {
    UNIFORM_FLOW_FAMILY.load(std::sync::atomic::Ordering::Relaxed)
}

/// ARC N: global amplitude scale on the manufactured PERTURBATIONS — the
/// velocity scales (U0/V0/UA/UB) and the rho/p wave amplitudes (RHOA/PA);
/// the RHO0/P0 backgrounds are NOT scaled. Shrinking the whole solution
/// deviation isolates the amplitude-INDEPENDENT boundary mode from the
/// amplitude-LINEAR interior KH physics, repeatably, without editing
/// consts. Default 1.0 (bits 0 = unset = 1.0). The sources derive from the
/// same `amp_scale()`, so the scaled family is a consistent MMS.
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

/// ARC N′ N2 mechanism toggles. RECON_UPWIND forces first-order Upwind (no
/// gradient reconstruction) to test whether the 2nd-order vanLeer
/// reconstruction is the source of the grid-scale boundary instability;
/// MASS_COMPAT_OFF skips the source mass-compatibility projection to rule it
/// out as the boundary-mode seed. Both default OFF.
static RECON_UPWIND: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static MASS_COMPAT_OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// ARC N N4: select the KNP central-upwind MINMOD reconstruction (sharper
/// limiter) instead of vanLeer. Minmod caps psi at 1 (no compressive overshoot
/// for r>1) — the candidate fix for the interior grid-scale under-dissipation.
/// Default OFF (vanLeer). Ignored if RECON_UPWIND is set.
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

/// ARC N′ N3 toggle: run on a fully-periodic box (zero boundary faces) instead
/// of the Dirichlet rectangle, so the interior operator is isolated from the
/// boundary stencil. Default OFF. The boundary seeding in `build_run_box`
/// no-ops (no Inlet/Outlet faces) and the mass-compatibility `bflux` term is
/// zero, so `eps` becomes the mean source — exactly the closed-system
/// constraint a periodic domain requires.
static PERIODIC: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn periodic() -> bool {
    PERIODIC.load(std::sync::atomic::Ordering::Relaxed)
}

/// ARC N N4b toggle + coefficient: select the biharmonic-dissipation compressible
/// MMS model and set the eps4 coefficient. When `BIHARMONIC` is set, `build_run_box`
/// builds `compressible_mms_biharmonic_model(eps4_knob())` (which appends the
/// `lap_<conserved>` fields and emits `+ eps4*c*(lap_neigh-lap_own)*area` on each
/// conserved flux); otherwise the plain MMS model. Default OFF / eps4=0 (the model
/// is rebuilt per run, so sweeping eps4 needs no recompile — the shader is
/// regenerated at `UnifiedSolver::new`). The candidate cure for the interior
/// grid-scale under-dissipation that no order-preserving limiter could fix.
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

/// Arc N generalization of `build_run`: arbitrary rectangular box and
/// per-side boundary types. The manufactured fields/sources/BC closures
/// are all evaluated at mesh coordinates, so any (lx, ly) works; when a
/// side is Outlet, the model's outlet closure needs its per-face Dirichlet
/// p, seeded from the exact solution like the inlet entries.
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
    // Thread the REQUESTED outer count (a previous version set the
    // OUTER_ITERS=1 const here, silently running the o2/o4 probe rows at
    // one iteration — the parameter-threading no-op bug class), and pin
    // the adaptive outer-convergence break OPEN so requested iterations
    // actually run: the break's relative tolerance fires after iteration
    // 1 on these smooth marches, collapsing the step to a single Picard
    // iteration (= forward-Euler-in-flux).
    solver.set_outer_iters(outer_iters).expect("outer_iters");
    solver.set_outer_tolerance(0.0).expect("outer_tol");
    solver.set_outer_tolerance_abs(0.0).expect("outer_tol_abs");
    if biharmonic() {
        // Arc N4b: set the runtime biharmonic coefficient (one registered model serves
        // every eps4). Default 0 keeps the term inert.
        solver
            .set_named_param("low_mach.eps4", PlanParamValue::F32(eps4_knob()))
            .expect("eps4");
    }
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
    // Outlet sides (Arc N BC-flip rows): the model's outlet closure takes a
    // per-face Dirichlet p; everything else extrapolates via bc_expr.
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
    // Domain extents derived from the mesh (Arc N: probes run on boxes
    // other than the unit square); assumes uniform square cells.
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
///   BDF2  o2/o4 n48            116.1/116.5  (genuine iterations; see below)
///   BDF2  o1 n48 dtau=dt lm=Off   77.2  bf 0.51   u 5.1e-3  rho 3.0e-3
///   BDF2  o1 n48 dtau=dt lm=WS    "0"   u 0.18    rho 9.0e10  (BLOWN)
///   BDF2  o1 n48 dtau lm=WS a=0   "0"   u 1.7e7   (BLOWN)
///   BDF2  o1 n48 dtau lm=Legacy   "0"   rho 9.0e10  (BLOWN)
///   BDF2  o1 n48 dt=1e-2 (fast)   "0"   u 1.5e12  (BLOWN)
///
/// CORRECTION (Arc R, June 12 2026): the original o2/o4 rows read
/// "bit-identical to o1" and were taken as proof the flux is frozen per
/// step. BOTH were instrument artifacts: (1) build_run overrode the
/// outer_iters parameter with the OUTER_ITERS=1 const (the
/// parameter-threading no-op bug class), and (2) the adaptive
/// outer-convergence break fires after one iteration on these smooth
/// marches anyway. With the threading fixed and the break pinned open
/// (outer_tol = 0), genuine o2/o4 Picard iterations CONVERGE (o2 = o4 to
/// 0.4%) and the converged implicit-in-flux step grows at ~116 %/tu —
/// matching Euler-o1 and WORSE than the unconverged o1-BDF2 map (60.6,
/// which partially damps the mode by accident). Time integration is
/// thereby exonerated BY MEASUREMENT: the spatial semi-discretization
/// itself has an eigenvalue with positive real part at this
/// configuration.
///
/// VERDICT — every knob-level hypothesis REFUTED; the instability is
/// intrinsic to the SPATIAL discretization at mu = 0, moderate Mach:
/// - NOT time integration (Euler ~ BDF2 ~ converged Picard, all ~116).
/// - NOT outer-iteration lag (converged o2/o4 grow at the spatial rate).
/// - NOT the implicit EOS-recovery coupling (Arc R P3a/P3b: with the
///   recovery rows decoupled to trivial holds and primitives recovered
///   explicitly from conserved state — rhoCentralFoam semantics — the
///   instability persists at comparable magnitude: n32 66, n48 diverges).
/// - NOT a checkerboard (nyquist ~1e-4 on the growing runs) and NOT
///   boundary-fed (boundary band holds LESS error mass than uniform);
///   the Inlet closure is characteristic-correct by design (rho/u
///   prescribed, p follows the interior via bc_expr).
/// - Pseudo-time damping (dtau = dt, preconditioning Off) DAMPS the mode
///   ~7x in final error but does not stabilize it.
/// - Low-Mach preconditioning (either model, with or without the
///   pressure-coupling term) makes it catastrophically WORSE: at M ~ 0.5
///   it rescales the dissipation wave speed c -> ~|u|, halving the
///   acoustic dissipation — a low-Mach tool misapplied at moderate Mach.
/// - Flux-dissipation redesign cannot fix it (Arc K): jump-proportional
///   dissipation is O(h^3) on the smooth mode; non-vanishing raw-jump
///   dissipation damps it linearly in dose but collapses orders first.
/// The growing object is a smooth INTERIOR eigenmode of the spatial
/// KT-flux discretization (EOS coupling exonerated), damped only by
/// physical viscosity (mu k^2 must beat the mode's growth; mu = 5e-3
/// holds through n = 32 at this problem's scales — the Euler study's
/// operating point). Remaining suspects for a future arc: the discrete
/// interplay of the bc_expr boundary refresh with the face flux
/// (Kreiss-type discrete well-posedness — needs periodic-domain support
/// to discriminate), and the vanLeer-reconstructed acoustic-speed field
/// feeding the wave bounds.
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

    // H2: outer iterations (genuine since the Arc R threading fix +
    // break pinning in build_run; converged Picard = implicit-in-flux).
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

/// ARC N verification: the strain-free (uniform-velocity) family at mu = 0.
/// The physical-instability diagnosis predicts: with zero base-flow
/// velocity gradients there is no production mechanism, so the same
/// discretization that grows at ~116 %/tu on the strained family must
/// march this family STABLY at mu = 0 — and converge at design order.
///
/// MEASURED (June 13, 2026): CONFOUNDED by the outlet closure, not a
/// clean interior test — uniform flow must exit somewhere, the run needs
/// right/top Outlets, and the Outlet-bearing MMS configuration has its
/// own boundary-band instability/inconsistency (bfrac 0.93-1.00, n48
/// blown, plateau errors 0.1-0.2 with orders ~0.35 — the SAME boundary
/// problem the small-amplitude all-Inlet rows unmask at bfrac 0.90; see
/// the domain-matrix verdict). The interior half of the diagnosis is
/// instead confirmed by the amplitude-scaling law; THIS probe becomes the
/// acceptance test for the boundary-closure fix family: it must turn
/// stable (growth <= 0, orders ~2) when the closure is fixed — or when
/// run on a periodic domain.
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

/// ARC N S2+S3: boundary-vs-interior discriminators at MATCHED h.
/// - [0,3]² all-Inlet (odd L keeps the unit box's all-inflow boundary
///   structure: cos(3π) = −1, sin(3π) = 0): an interior-driven mode must
///   reproduce the unit-box rate; boundary-driven growth scales down with
///   the perimeter/area ratio (×1/3) or changes character.
/// - [0,2]² left/bottom Inlet + right/top Outlet (genuine outflow there:
///   u_x(2,y) = +U0): changes the boundary reflection structure entirely;
///   a strongly different rate convicts the boundary closure.
///
/// MEASURED (June 13, 2026): L3 reproduces the unit-box rate at matched h
/// (61.6 vs 60.6 at h=1/48; 83.8 vs 103.5 at h=1/32) — interior-driven at
/// FULL amplitude. The in/out rows grow in the same band but with large
/// boundary-band error levels (bfrac 0.58-0.66, levels 0.2-0.3): the
/// Outlet-bearing MMS configuration has its own boundary problem (never
/// previously validated — the MMS suite is all-Inlet).
///
/// AMPLITUDE-SCALING VERDICT (the arc's decisive measurement, run by
/// scaling the manufactured-field consts in the working tree): on the
/// converged-Picard rows, growth vs amplitude is
///   velocity amps x1.0/0.5/0.25 (rho,p fixed):  116 / 76 / 56.6 %/tu
///   (two-point linear fit 80*amp + 36 predicted 56 at 0.25 — confirmed)
///   ALL amps x0.25:                              63.7 %/tu, bfrac 0.90
/// Two superposed mechanisms:
/// 1. INTERIOR, amplitude-LINEAR (dominates at full amplitude, bfrac
///    0.09): the manufactured base flow itself — O(1) strain with
///    inflection-point shear — is linearly UNSTABLE as inviscid PHYSICS
///    (strain scale 0.8-1.3/tu matches the measured 1.04-1.16/tu;
///    h-independent on the converged rows; immune to every numerics
///    change probed across Arcs I/K/R/N; quenched by physical mu*k^2).
///    NOT a discretization defect; do not try to "fix" it.
/// 2. BOUNDARY-BAND, amplitude-independent ~60 %/tu at n48 (unmasked at
///    small amplitude: bfrac flips 0.09 -> 0.90): a genuine DISCRETE
///    boundary-closure instability — the remaining numerics target
///    (characteristic/LODI inlet closure family; the periodic-domain
///    instrument separates it cleanly if needed).
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

/// ARC N′ N1 — IS THE BOUNDARY-BAND μ=0 INSTABILITY GENUINE? Converged-Picard
/// (outer=2), small amplitude (perturbations ×AMP so the amplitude-INDEPENDENT
/// boundary mode dominates the amplitude-LINEAR interior KH physics), all-Inlet
/// box, n = 32/48/64/96. The growth-rate trend under refinement is the verdict:
///   rate → 0 with h  ⇒ BENIGN O(h) boundary inconsistency (the inviscid limit
///                      is reachable at resolution — close the arc);
///   rate constant/↑  ⇒ GENUINE discrete instability (continue to N2/N3).
/// Bit-reproduces the committed full-amplitude probe at AMP=1 (the amp_scale
/// refactor is transparent there).
///
/// MEASURED (June 13, 2026; amp=0.25, o2, μ=0, 600 steps): n=32 → 61.2,
/// n=48 → 63.7 (rate CONSTANT, not falling), n=64 → DIVERGES (u_l2 9.96e8),
/// n=96 → DIVERGES (u_l2 2.03e13). **VERDICT: GENUINE and
/// REFINEMENT-AMPLIFIED** — finer grids blow up harder (the "0.00" rate at
/// n≥64 is the saturated-blowup metric trap; the levels show divergence).
/// This is a grid-scale numerical instability (consistent with the
/// high-k eigenmode card), NOT a benign O(h) inconsistency. NOTE for
/// re-measuring the n≥64 *rate*: shorten `steps` so the fit window precedes
/// saturation.
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

/// ARC N′ N2 — mechanism isolation for the grid-scale boundary instability
/// (small amplitude, converged Picard, all-Inlet). Two cheap discriminators:
/// - Upwind vs vanLeer: first-order Upwind disables the whole gradient/
///   reconstruction path. If the boundary mode VANISHES under Upwind, the
///   2nd-order vanLeer reconstruction (the owner-side reconstruct-to-face vs
///   the 0th-order face ghost asymmetry) is the source.
/// - mass-compat on/off: rules out the source projection (expected null —
///   O(h²)).
///
/// MEASURED (June 13, 2026; amp=0.25, o2): vanLeer n32/n48 = 61.2/63.7;
/// **Upwind n32/n48 = 0.28/2.74 — a 20–200× reduction, NEARLY STABLE**;
/// no-masscompat n48 = 63.1 (≈ baseline ⇒ mass-compat RULED OUT). Verdict:
/// the 2nd-order vanLeer reconstruction is the DOMINANT driver — its
/// near-central limiter (ψ≈1 on the smooth background) is under-dissipative
/// at grid scale (the KNP jump dissipation is O(h³) on smooth reconstructed
/// states), exactly matching the high-k eigenmode card; Upwind's numerical
/// dissipation suppresses it. CAVEAT: Upwind n64 still DIVERGES (u_l2 6e8) —
/// its O(h) dissipation VANISHES under refinement, so Upwind only DELAYS the
/// refinement-amplified mode. The instability is fundamentally
/// under-dissipation at high k. N3 (periodic) separates whether this is the
/// INTERIOR reconstruction or specifically the BOUNDARY.
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
    // Baseline vanLeer (boundary mode present: ~61/64 at n32/48).
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

/// ARC N′ N3 — the periodic-domain instrument: the decisive interior-vs-boundary
/// separator. A fully-periodic box has ZERO boundary faces, so any growth is the
/// INTERIOR operator alone — the all-Inlet box's boundary band is structurally
/// absent. The manufactured fields are 2-periodic, so the box is [0,2]^2 at 2n
/// cells/side (h = 1/n), matched to the N1/N2 unit-box rows. Small amplitude
/// (0.25), mu = 0, converged Picard (outer=2, break pinned open) — identical to
/// `probe_arcn_refinement_trend` except for the periodic mesh.
///
/// MEASURED (June 13, 2026) — VERDICT: ENTIRELY INTERIOR.
///   h=1/32 (64 cells):  +43.94 %/tu, u_l2 1.26e-2  (all-Inlet N1: 61.2)
///   h=1/48 (96 cells):  +41.06 %/tu, u_l2 9.72e-3  (all-Inlet N1: 63.7)
///   h=1/64 (128 cells): DIVERGES, u_l2 5.77e5      (all-Inlet N1: 9.96e8)
///   h=1/96 (192 cells): DIVERGES, u_l2 2.86e8      (all-Inlet N1: 2.03e13)
/// The boundary-free box reproduces the WHOLE N1 signature — finite growth at
/// coarse h AND the refinement-amplified blow-up at fine h (the "0.00" rate at
/// n>=64 is the saturated-blowup metric trap; the LEVELS show divergence). So
/// the instability needs NO boundary faces: it is the interior vanLeer
/// reconstruction's high-k under-dissipation (N2), full stop. The all-Inlet
/// box's bfrac~0.90 "boundary band" was a metric artifact (the interior high-k
/// mode's amplitude concentrating near edges); the boundary geometry AMPLIFIES
/// the mode (~3 orders harder at n=64) but does not cause it. => N4 targets a
/// k-selective INTERIOR dissipation (must hold the mu>0 MMS orders).
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
    // 2n cells over [0,2] => h = 1/n, matched to the N1 unit-box n rows
    // (where all-Inlet diverged at n=64/96 — does the boundary-free box too?).
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

/// ARC N N4: does the sharper MINMOD limiter (KNP central-upwind) suppress the
/// interior grid-scale instability that vanLeer under-dissipates? A/B on the
/// boundary-free [0,2]^2 periodic box (the N3 instrument), small amplitude,
/// mu=0, converged Picard. vanLeer is the baseline (the N3 divergence at fine
/// h); minmod is the candidate fix — psi capped at 1, no compressive overshoot.
/// SUCCESS = minmod growth bounded / <= 0 where vanLeer diverges, at fine h.
/// (The mu>0 MMS-order gate — minmod must still reach order ~2 — is separate.)
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

/// ARC N N4b: k-selective BIHARMONIC dissipation — the actual cure. Sweeps eps4
/// on the boundary-free [0,2]^2 periodic box (the N3 instrument), small amplitude,
/// mu=0, converged Picard, at the meshes where vanLeer AND minmod diverge.
/// eps4=0 is the control (the lap machinery is on but the term is x0, so it must
/// reproduce the vanLeer divergence). SUCCESS = an eps4>0 that drives the n>=64
/// growth to bounded (rate <= 0 / u_l2 at the coarse-h scale, not 1e5..1e12)
/// WITHOUT blowing up coarse h. A WRONG dissipation sign makes eps4>0 diverge
/// HARDER than eps4=0 — the instrument catches that on the first row. The mu>0
/// order gate (term must stay O(h^3) => order ~2) is `probe_arcn_biharmonic_order`.
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

/// ARC N N4: minmod's mu>0 convergence order — the accuracy gate. Minmod can
/// clip to first order at smooth extrema, so this measures whether KNP+minmod
/// still reaches order ~2 (the bar vanLeer clears at 1.96/2.15/1.83/1.86).
/// Reports observed orders without asserting (characterization probe).
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

/// ARC N N4b: biharmonic's mu>0 convergence order — the ACCURACY gate. The
/// undivided-Laplacian 4th-difference term is O(h^3) in the residual, so it must
/// stay subdominant to the O(h^2) scheme and preserve order ~2 (vanLeer clears
/// ~1.97/2.12/1.91/2.20). Sweeps a few eps4 so an over-large coefficient that
/// pollutes coarse-h accuracy is visible. eps4=0 = the vanLeer control.
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

/// ARC N N4b: DISCRIMINATOR — biharmonic mu>0 order on the boundary-free PERIODIC
/// box. The Dirichlet mu>0 order gate blows up for eps4>0; this isolates whether
/// the cause is the boundary closure (then periodic should be order ~2 and a
/// smooth boundary treatment is the fix) or a mu>0+biharmonic interaction (then
/// periodic also blows up and a boundary fix won't help). Full amplitude, MU,
/// march-to-plateau, same as the Dirichlet order study but on [0,2]^2 periodic.
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

/// ARC N N4b: is the Dirichlet mu>0 biharmonic blow-up an EXPLICIT-stiffness
/// problem? The term reads a frozen (lagged) `lap`, so it acts like an explicit
/// 4th-difference (stability limit dt ~ h^4) — which OUTER_ITERS=1 + dt=0.01
/// violates, worsening under refinement (the observed negative order). This
/// varies outer-iterations and dt at eps4=0.1 on the Dirichlet box; if order
/// recovers to ~2 with more outer / smaller dt, the fix is iteration/relaxation,
/// not a boundary treatment.
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

/// ARC N S1: eigenmode dump — capture the growing inviscid mode's spatial
/// structure. Marches n=48 / mu=0 / BDF2 / o1, snapshots per-cell deltas
/// vs the exact solution at steps 400 and 500, and writes CSV to
/// target/arcn_probes/eigenmode_n48.csv with columns
/// x,y,d1_<f>,d2_<f>,g_<f> for f in rho,ux,uy,p,T — where d1/d2 are the
/// two snapshots' deltas and g = d2 - d1 is the GROWING-MODE component
/// (the snapshot difference cancels the steady O(h^2) discretization
/// error).
///
/// MODE CARD (measured June 12, 2026; n=48, mu=0, BDF2 o1, steps 400-500):
/// - growth-component rms: rho 1.98e-2, ux 1.65e-2, uy 1.48e-2,
///   p 2.43e-2, T 1.10e-2 — ALL fields participate; p is largest.
/// - spectral content: 86-93% of energy at |freq| > 8 (NEAR GRID SCALE;
///   peaks at (3,15), (21,-19) etc.) — the mode is HIGH-FREQUENCY, not
///   the smooth k~11 object inferred earlier. THIRD METRIC TRAP: the
///   nyquist-checkerboard fraction (strict alternating-sign measure) is
///   blind to broadband high-k content; it read ~1e-4 while 90% of the
///   mode energy sat above kh ~ 1.
/// - corr(p, rho) = +0.978, rms p/rho = 1.23 (between isothermal 1.0 and
///   acoustic c^2 = 1.4): an acoustic-LIKE correlated pattern, not an
///   entropy mode. corr(p, T) = +0.64.
/// - envelope: interior rms slightly ABOVE edge rms — interior-
///   distributed, consistent with the bfrac findings.
/// Mechanism candidate consistent with all of this: small-amplitude
/// high-k perturbations riding a smooth background see near-central
/// (psi ~ 1) limited reconstruction, whose linearized face jumps nearly
/// cancel — the KNP jump dissipation has a near-null direction there,
/// leaving non-dissipative central transport that the coupled system
/// tips unstable. The k-content REOPENS k-selective (JST-style compact
/// 4th-difference) dissipation as the fix family: O(h^3) on smooth
/// fields (order-preserving) but O(amplitude) at grid scale — Arc K
/// had excluded it under the (wrong) smooth-mode belief.
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

/// ARC R diagnostic: does the outer Picard loop actually refresh the KT
/// flux/assembly inputs, or is the step effectively explicit-in-flux?
/// Run with CFD2_DEBUG_FGMRES=1 CFD2_LIN_TOL=1e-7 and read the per-solve
/// trace within each step.
///
/// FINDINGS (June 12, 2026):
/// - The batched one-submission outer loop emits NO per-solve trace
///   (encoded up front, no readback) — disable it first, as below.
/// - Even un-batched, only ONE solve fired per step at outer_iters=4:
///   the adaptive outer-convergence break (outer_tol relative) fires
///   after iteration 1 on smooth marches. Production compressible
///   stepping is therefore one-Picard-iteration (explicit-in-flux) by
///   default — ironically MORE stable here than the converged implicit
///   step (60.6 vs 116 %/tu at n48; see the matrix verdict), so this is
///   recorded as a characterization, not a defect to fix.
/// - With the break pinned open (outer_tol = 0) the assembly does
///   re-read the updated state each iteration and Picard converges
///   (o2 = o4 to 0.4%): the per-outer refresh architecture works.
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

/// ARC K probe: order study of the UNPRECONDITIONED operator at mu = 0,
/// for measuring candidate flux-dissipation designs against the success
/// criterion (stable at mu = 0 AND orders ~2). Run with a candidate
/// dissipation active in flux_schemes.rs.
///
/// ARC K VERDICT (June 12, 2026) — two candidate families probed and
/// REFUTED; the timebox closed the arc:
/// - Family 1, Rusanov symmetric wave-speed split (ap = a_max = -am,
///   dissipation (|u|+c)/2 instead of the signed KT ~(1-M^2)c/2, ~2x at
///   M=0.5): growth moved only 103.5->98.3 / 115.9->109.9 %/tu. Jump-
///   proportional dissipation vanishes at O(h^3) on smooth reconstructed
///   fields and cannot reach the smooth thermo-mode AT ANY coefficient
///   scale. (Side finding: it does rescue the PRECONDITIONED rows from
///   catastrophic blow-up, rho 9e10 -> bounded, and stabilizes the
///   dt=1e-2 row — relevant if preconditioning is ever revisited.)
/// - Family 2, raw-cell-jump dissipation on the rho/rhoE rows
///   (mu_art ~ k2*c*h, non-vanishing on smooth fields): clean monotone
///   dose-response — n32 growth 103.5 (k2=0) -> 72.7 (0.05) -> 34.4 (0.2)
///   -> 5.6 (0.4) %/tu — the mode IS reachable by smooth-field
///   dissipation. But the joint criterion fails: at k2=0.4 this probe
///   measured orders rho 0.458 / u 0.235 / p 0.709 / T 0.673 (the O(h)
///   dissipation error dominates), errors RISE with k2 at fixed n, and a
///   pressure checkerboard emerges (nyq(p) 0.136 on the Euler n48 row).
///   There is no sweet spot: full stabilization needs k2 >~ 0.5 and
///   order 2 dies well before k2 = 0.4.
/// Conclusion: face-local dissipation design cannot deliver "stable at
/// mu=0 with orders ~2"; the gap between the O(h^3) jump dissipation and
/// the O(h) damping the mode needs is structural. A real fix must change
/// the structure of the inv_dt-scaled EOS-recovery coupling (e.g.
/// entropy-consistent coupling rows) — backlog, own plan. The stability
/// envelope contract in compressible.rs stands.
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
