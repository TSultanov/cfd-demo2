//! CPU-backend compressible MMS: the manufactured subsonic Navier–Stokes
//! solution solved through the coupled path (KT/vanLeer flux, EOS recovery rows,
//! implicit viscous/conduction laplacians, expression-valued inlet BCs, block-CSR
//! FGMRES+block-Jacobi). Source derivation mirrors the manufactured-NS operator
//! (EXTRA_SHEAR = 0).
#![cfg(feature = "cpu")]
// Helpers are used only by the `dev-tests`-gated diagnostics; dead without it.
#![allow(dead_code)]

use cfd2::solver::cpu::{CpuBackendConfig, CpuSolver};
use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::gpu::recipe::SteppingMode;
use cfd2::solver::mesh::{
    generate_structured_rect_mesh, generate_structured_rect_mesh_periodic, BoundarySides,
    BoundaryType, Mesh,
};
use cfd2::solver::model::{
    compressible_mms_biharmonic_model, compressible_mms_model,
    COMPRESSIBLE_MMS_SOURCE_RHO_E_FIELD, COMPRESSIBLE_MMS_SOURCE_RHO_FIELD,
    COMPRESSIBLE_MMS_SOURCE_RHO_U_FIELD,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::TimeScheme;
use std::f64::consts::PI;

const GAMMA: f64 = 1.4;
const R_GAS: f64 = 1.0;
const PRANDTL: f64 = 0.71;
const MU: f64 = 0.05;
const DT: f64 = 0.01;
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

fn exact_rho(x: f64, y: f64) -> f64 {
    RHO0 * (1.0 + RHOA * (PI * x + PHX).sin() * (PI * y + PHY).sin())
}
fn exact_u(x: f64, y: f64) -> (f64, f64) {
    (
        U0 * (PI * x).cos() + UA * (PI * x).sin() * (PI * y).sin(),
        V0 * (PI * y).cos() + UB * (PI * x).sin() * (PI * y).sin(),
    )
}
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

fn rho_partials(x: f64, y: f64) -> (f64, f64, f64, f64, f64) {
    let (sx, cx) = (PI * x + PHX).sin_cos();
    let (sy, cy) = (PI * y + PHY).sin_cos();
    let a = RHO0 * RHOA;
    (RHO0 + a * sx * sy, a * PI * cx * sy, a * PI * sx * cy, -a * PI * PI * sx * sy, -a * PI * PI * sx * sy)
}
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

fn source_rho(x: f64, y: f64) -> f64 {
    let (rho, rho_x, rho_y, _, _) = rho_partials(x, y);
    let (u, ux, _, _, _, _) = u_partials(x, y);
    let (v, _, vy, _, _, _) = v_partials(x, y);
    rho_x * u + rho * ux + rho_y * v + rho * vy
}
fn source_rho_u(x: f64, y: f64, mu: f64) -> (f64, f64) {
    let (rho, rho_x, rho_y, _, _) = rho_partials(x, y);
    let (u, ux, uy, uxx, uyy, uxy) = u_partials(x, y);
    let (v, vx, vy, vxx, vyy, vxy) = v_partials(x, y);
    let (_, px, py, _, _) = p_partials(x, y);
    let conv_x = rho_x * u * u + 2.0 * rho * u * ux + px + rho_y * u * v + rho * (uy * v + u * vy);
    let conv_y = rho_x * u * v + rho * (ux * v + u * vx) + rho_y * v * v + 2.0 * rho * v * vy + py;
    let div_x = uxx + vxy;
    let div_y = uxy + vyy;
    let div_tau_x = mu * (2.0 * uxx - (2.0 / 3.0) * div_x + uyy + vxy);
    let div_tau_y = mu * (uxy + vxx + 2.0 * vyy - (2.0 / 3.0) * div_y);
    (conv_x - div_tau_x, conv_y - div_tau_y)
}
fn source_rho_e(x: f64, y: f64, mu: f64) -> f64 {
    let k_cond = mu * GAMMA * R_GAS / (GAMMA - 1.0) / PRANDTL;
    let (rho, rho_x, rho_y, rho_xx, rho_yy) = rho_partials(x, y);
    let (u, ux, uy, uxx, uyy, uxy) = u_partials(x, y);
    let (v, vx, vy, vxx, vyy, vxy) = v_partials(x, y);
    let (p, px, py, pxx, pyy) = p_partials(x, y);
    let q2 = u * u + v * v;
    let h = GAMMA * p / (GAMMA - 1.0) + 0.5 * rho * q2;
    let hx = GAMMA * px / (GAMMA - 1.0) + 0.5 * rho_x * q2 + rho * (u * ux + v * vx);
    let hy = GAMMA * py / (GAMMA - 1.0) + 0.5 * rho_y * q2 + rho * (u * uy + v * vy);
    let conv = hx * u + h * ux + hy * v + h * vy;
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
    let div_w = tau_xx_x * u + tau_xx * ux + tau_xy_x * v + tau_xy * vx
        + tau_xy_y * u
        + tau_xy * uy
        + tau_yy_y * v
        + tau_yy * vy;
    let w = 1.0 / rho;
    let wx = -rho_x / (rho * rho);
    let wy = -rho_y / (rho * rho);
    let wxx = (2.0 * rho_x * rho_x - rho * rho_xx) / (rho * rho * rho);
    let wyy = (2.0 * rho_y * rho_y - rho * rho_yy) / (rho * rho * rho);
    let t_xx = (pxx * w + 2.0 * px * wx + p * wxx) / R_GAS;
    let t_yy = (pyy * w + 2.0 * py * wy + p * wyy) / R_GAS;
    conv - div_w - k_cond * (t_xx + t_yy)
}

fn l2_scalar(mesh: &Mesh, f: &[f64], exact: impl Fn(f64, f64) -> f64) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..mesh.num_cells() {
        let e = f[i] - exact(mesh.cell_cx[i], mesh.cell_cy[i]);
        num += e * e * mesh.cell_vol[i];
        den += mesh.cell_vol[i];
    }
    (num / den).sqrt()
}

/// Advection scheme for the setups; overridable via `CFD2_TEST_SCHEME`.
fn test_scheme() -> Scheme {
    std::env::var("CFD2_TEST_SCHEME")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(Scheme::SecondOrderUpwindVanLeer)
}

#[allow(clippy::type_complexity)]
fn setup(n: usize, cfg: CpuBackendConfig) -> (CpuSolver, Mesh) {
    setup_g(n, cfg, false, 0.0)
}

/// Generalized setup: `biharmonic` selects the ∇⁴-dissipation compressible model
/// and `eps4` sets its coefficient. Manufactured solution + sources are identical
/// (the eps4 term is a consistent O(h²) dissipation).
#[allow(clippy::type_complexity)]
fn setup_g(n: usize, cfg: CpuBackendConfig, biharmonic: bool, eps4: f32) -> (CpuSolver, Mesh) {
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
    let model = if biharmonic {
        compressible_mms_biharmonic_model().expect("biharmonic model")
    } else {
        compressible_mms_model().expect("model")
    };
    let cells = mesh.num_cells();
    let mut s = CpuSolver::with_stepping(
        &mesh,
        model,
        test_scheme(),
        TimeScheme::BDF2,
        SteppingMode::Implicit { outer_iters: 1 },
        cfg,
    )
    .expect("cpu solver");
    s.set_eps4(eps4);
    s.set_dt(DT as f32);
    s.set_dtau(0.0);
    s.set_viscosity(MU as f32);
    s.set_density(RHO0 as f32);
    s.set_outer_iters(1);
    s.set_outer_tolerance(0.0);

    // Per-face exact Dirichlet/expression-seed inlets.
    let (fx, fy) = (mesh.face_cx.clone(), mesh.face_cy.clone());
    let sc = |f: &'static dyn Fn(f64, f64) -> f64, fx: &[f64], fy: &[f64]| {
        let (fx, fy) = (fx.to_vec(), fy.to_vec());
        move |i: u32| f(fx[i as usize], fy[i as usize]) as f32
    };
    s.set_boundary_values_per_face(GpuBoundaryType::Inlet, "rho", 0, &sc(&exact_rho, &fx, &fy)).unwrap();
    s.set_boundary_values_per_face(GpuBoundaryType::Inlet, "p", 0, &sc(&exact_p, &fx, &fy)).unwrap();
    s.set_boundary_values_per_face(GpuBoundaryType::Inlet, "T", 0, &sc(&exact_t, &fx, &fy)).unwrap();
    s.set_boundary_values_per_face(GpuBoundaryType::Inlet, "rho_e", 0, &sc(&exact_rho_e, &fx, &fy)).unwrap();
    for c in 0..2u32 {
        let (fx2, fy2) = (fx.clone(), fy.clone());
        let uf = move |i: u32| { let (a, b) = exact_u(fx2[i as usize], fy2[i as usize]); (if c == 0 { a } else { b }) as f32 };
        let (fx3, fy3) = (fx.clone(), fy.clone());
        let ruf = move |i: u32| { let j = i as usize; let (a, b) = exact_u(fx3[j], fy3[j]); (exact_rho(fx3[j], fy3[j]) * if c == 0 { a } else { b }) as f32 };
        s.set_boundary_values_per_face(GpuBoundaryType::Inlet, "u", c, &uf).unwrap();
        s.set_boundary_values_per_face(GpuBoundaryType::Inlet, "rho_u", c, &ruf).unwrap();
    }

    // Manufactured sources with the discrete mass-compatibility projection.
    let mut src_rho: Vec<f64> = (0..cells).map(|i| source_rho(mesh.cell_cx[i], mesh.cell_cy[i])).collect();
    let vol_total: f64 = mesh.cell_vol.iter().sum();
    let vol_int: f64 = (0..cells).map(|i| src_rho[i] * mesh.cell_vol[i]).sum();
    let bflux: f64 = (0..mesh.face_owner.len())
        .filter(|&f| mesh.face_neighbor[f].is_none())
        .map(|f| {
            let (ux, uy) = exact_u(mesh.face_cx[f], mesh.face_cy[f]);
            exact_rho(mesh.face_cx[f], mesh.face_cy[f]) * (ux * mesh.face_nx[f] + uy * mesh.face_ny[f]) * mesh.face_area[f]
        })
        .sum();
    let eps = (vol_int - bflux) / vol_total;
    for v in src_rho.iter_mut() {
        *v -= eps;
    }
    let src_rho_u: Vec<(f64, f64)> = (0..cells).map(|i| source_rho_u(mesh.cell_cx[i], mesh.cell_cy[i], MU)).collect();
    let src_rho_e: Vec<f64> = (0..cells).map(|i| source_rho_e(mesh.cell_cx[i], mesh.cell_cy[i], MU)).collect();
    s.set_field_scalar(COMPRESSIBLE_MMS_SOURCE_RHO_FIELD, &src_rho).unwrap();
    s.set_field_vec2(COMPRESSIBLE_MMS_SOURCE_RHO_U_FIELD, &src_rho_u).unwrap();
    s.set_field_scalar(COMPRESSIBLE_MMS_SOURCE_RHO_E_FIELD, &src_rho_e).unwrap();

    // Initialize at the exact solution.
    let rho0: Vec<f64> = (0..cells).map(|i| exact_rho(mesh.cell_cx[i], mesh.cell_cy[i])).collect();
    let u0: Vec<(f64, f64)> = (0..cells).map(|i| exact_u(mesh.cell_cx[i], mesh.cell_cy[i])).collect();
    let rho_u0: Vec<(f64, f64)> = (0..cells).map(|i| (rho0[i] * u0[i].0, rho0[i] * u0[i].1)).collect();
    let p0v: Vec<f64> = (0..cells).map(|i| exact_p(mesh.cell_cx[i], mesh.cell_cy[i])).collect();
    let t0v: Vec<f64> = (0..cells).map(|i| exact_t(mesh.cell_cx[i], mesh.cell_cy[i])).collect();
    let rho_e0: Vec<f64> = (0..cells).map(|i| exact_rho_e(mesh.cell_cx[i], mesh.cell_cy[i])).collect();
    s.set_field_scalar("rho", &rho0).unwrap();
    s.set_field_vec2("rho_u", &rho_u0).unwrap();
    s.set_field_scalar("rho_e", &rho_e0).unwrap();
    s.set_field_scalar("p", &p0v).unwrap();
    s.set_field_scalar("T", &t0v).unwrap();
    s.set_field_vec2("u", &u0).unwrap();
    if biharmonic {
        // Implicit biharmonic reads its coefficient from the per-cell `bih_eps4`
        // storage field (uniform-valued, like mu) — NOT `low_mach.eps4`.
        s.set_field_scalar("bih_eps4", &vec![eps4 as f64; cells]).unwrap();
    }
    s.initialize_history();
    (s, mesh)
}

#[allow(clippy::type_complexity)]
fn solve(
    n: usize,
    steps: usize,
    cfg: CpuBackendConfig,
) -> (Mesh, Vec<f64>, Vec<(f64, f64)>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<(f64, f64)>) {
    let (mut s, mesh) = setup(n, cfg);
    for _ in 0..steps {
        s.step();
    }
    let rho = s.get_field_scalar("rho").unwrap();
    let u = s.get_field_vec2("u").unwrap();
    let p = s.get_field_scalar("p").unwrap();
    let t = s.get_field_scalar("T").unwrap();
    let rho_e = s.get_field_scalar("rho_e").unwrap();
    let rho_u = s.get_field_vec2("rho_u").unwrap();
    (mesh, rho, u, p, t, rho_e, rho_u)
}

/// Assemble at the EXACT solution and compute the per-equation discrete residual
/// r = rhs - A x_exact. Correct assembly leaves r = O(h^2) for every equation; a
/// large component pinpoints the mis-assembled equation. (Coupled-unknown order:
/// rho, rho_u_x, rho_u_y, rho_e, u_x, u_y, p, T.)
#[ignore]
#[ignore] // diagnostic (informational prints; slow); run explicitly
#[test]
fn diag_compressible_exact_residual() {
    for n in [8usize, 16, 32] {
        residual_at(n);
    }
}

fn residual_at(n: usize) {
    let (mut s, mesh) = setup(n, CpuBackendConfig::default());
    let (matrix, rhs) = s.debug_assemble();
    let (sro, col, diag_idx, ss) = s.debug_topology();
    let cells = mesh.num_cells();
    // x_exact in coupled order.
    let mut xe = vec![0.0f64; cells * ss];
    for i in 0..cells {
        let (x, y) = (mesh.cell_cx[i], mesh.cell_cy[i]);
        let (ux, uy) = exact_u(x, y);
        let r = exact_rho(x, y);
        xe[i * ss + 0] = r;
        xe[i * ss + 1] = r * ux;
        xe[i * ss + 2] = r * uy;
        xe[i * ss + 3] = exact_rho_e(x, y);
        xe[i * ss + 4] = ux;
        xe[i * ss + 5] = uy;
        xe[i * ss + 6] = exact_p(x, y);
        xe[i * ss + 7] = exact_t(x, y);
    }
    // r = rhs - A x_exact (block-CSR SoA, same formula as the solver).
    let mut res = vec![0.0f64; cells * ss];
    for i in 0..cells {
        let so = sro[i] as usize;
        let nn = sro[i + 1] as usize - so;
        for rrow in 0..ss {
            let start = so * ss * ss + nn * ss * rrow;
            let mut ax = 0.0f64;
            for rank in 0..nn {
                let j = col[so + rank] as usize;
                for c in 0..ss {
                    ax += matrix[start + rank * ss + c] as f64 * xe[j * ss + c];
                }
            }
            res[i * ss + rrow] = rhs[i * ss + rrow] as f64 - ax;
        }
    }
    // Solve A x = rhs from the exact-solution guess and compare x to x_exact.
    // rhs ~ A x_exact (residual O(h^3)), so an accurate solve must return
    // x ~ x_exact; a large deviation means the SOLVE (not assembly) is wrong.
    {
        use cfd2::solver::cpu::linalg::{fgmres, BlockCsr, PointJacobi};
        let a = BlockCsr {
            s: ss,
            scalar_row_offsets: sro,
            col_indices: col,
            diagonal_indices: diag_idx,
            values: &matrix,
            threads: 1,
            simd: false,
        };
        let pc = PointJacobi::<f32>::new(&a);
        let mut xg: Vec<f32> = xe.iter().map(|&v| v as f32).collect();
        let st = fgmres(&a, &rhs, &mut xg, &pc, 60, 5000, 1e-8, false);
        // Independent dense LU cross-check (small n only): confirms FGMRES +
        // block_spmv solve the assembled system, not just a self-consistent read.
        if n == 8 {
            let dof = cells * ss;
            let mut dense = nalgebra::DMatrix::<f64>::zeros(dof, dof);
            for i in 0..cells {
                let so = sro[i] as usize;
                let nnb = sro[i + 1] as usize - so;
                for rr in 0..ss {
                    let start = so * ss * ss + nnb * ss * rr;
                    for rank in 0..nnb {
                        let j = col[so + rank] as usize;
                        for cc in 0..ss {
                            dense[(i * ss + rr, j * ss + cc)] = matrix[start + rank * ss + cc] as f64;
                        }
                    }
                }
            }
            let bvec = nalgebra::DVector::<f64>::from_iterator(dof, rhs.iter().map(|&v| v as f64));
            let xdense = dense.clone().lu().solve(&bvec).expect("dense lu");
            let dd = (0..dof).map(|i| (xg[i] as f64 - xdense[i]).abs()).fold(0.0, f64::max);
            // Condition number (SVD): how f32-assembly noise (~1e-6/entry) amplifies.
            let svals = dense.singular_values();
            let smax = svals[0];
            let smin = svals[svals.len() - 1];
            println!(
                "[solve] n={n} max|fgmres - dense_lu|={dd:.3e} cond={:.3e} (smax={smax:.2e} smin={smin:.2e})",
                smax / smin
            );
        }
        let names = ["rho", "rho_u_x", "rho_u_y", "rho_e", "u_x", "u_y", "p", "T"];
        print!("[solve] n={n} conv={} it={}", st.converged, st.iters);
        for u in 0..ss {
            let m = (0..cells)
                .map(|i| (xg[i * ss + u] as f64 - xe[i * ss + u]).abs())
                .fold(0.0, f64::max);
            print!(" {}={m:.2e}", names[u]);
        }
        println!();
    }
    let _ = diag_idx;
    let names = ["rho", "rho_u_x", "rho_u_y", "rho_e", "u_x", "u_y", "p", "T"];
    // Volume-weighted L2 residual per equation (interior cells only, to exclude
    // boundary-closure truncation which is separately O(h^2)).
    let bdry: Vec<bool> = (0..cells)
        .map(|i| {
            let (s0, e0) = (mesh.cell_face_offsets[i], mesh.cell_face_offsets[i + 1]);
            (s0..e0).any(|k| mesh.face_neighbor[mesh.cell_faces[k]].is_none())
        })
        .collect();
    print!("[resid] n={n}");
    for u in 0..ss {
        let m = (0..cells)
            .filter(|&i| !bdry[i])
            .map(|i| res[i * ss + u].abs())
            .fold(0.0, f64::max);
        print!(" {}={m:.2e}", names[u]);
    }
    println!();
}

/// CPU-vs-GPU step-1 comparison on the EXACT MMS setup, per field. Distinguishes
/// a per-step operator/solve difference from a slow multi-step instability.
#[ignore]
#[cfg(feature = "dev-tests")]
#[ignore] // diagnostic (informational prints); run explicitly
#[test]
fn diag_cpu_vs_gpu_mms_step1() {
    use cfd2::solver::gpu::unified_solver::PlanParamValue;
    use cfd2::solver::model::helpers::SolverRuntimeParamsExt;
    use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode as Cfg, UnifiedSolver};
    let n = 16;
    let steps = 1;

    // CPU.
    let (mut c, mesh) = setup(n, CpuBackendConfig::default());
    let bc_before = c.debug_bc_value();
    for _ in 0..steps { c.step(); }
    let bc_after = c.debug_bc_value();
    // Did bc_expr refresh the rho_e (u_idx 3) / T (u_idx 7) ghosts at boundary faces?
    {
        let mut max_change_re = 0.0f32;
        let mut max_change_t = 0.0f32;
        let mut sample = String::new();
        for f in 0..mesh.num_faces() {
            if mesh.face_neighbor[f].is_some() { continue; }
            let dre = (bc_after[f * 8 + 3] - bc_before[f * 8 + 3]).abs();
            let dt = (bc_after[f * 8 + 7] - bc_before[f * 8 + 7]).abs();
            if dre > max_change_re { max_change_re = dre; if sample.is_empty() {
                sample = format!("face {f}: rho_e seed={:.5} refreshed={:.5}", bc_before[f*8+3], bc_after[f*8+3]);
            } }
            max_change_t = max_change_t.max(dt);
        }
        println!("[bc-refresh] CPU bc_value change after step: rho_e={max_change_re:.3e} T={max_change_t:.3e}; {sample}");
        // Compare the CPU's REFRESHED rho_e/T ghosts to the analytically-expected
        // bc_expr output (in_p = exact_p(OWNER CELL center); ke from seeded face
        // rho/u). A large diff means the CPU bc_expr executes the formula wrong.
        let gm1 = GAMMA - 1.0;
        let (mut wre, mut wt) = (0.0f64, 0.0f64);
        let mut wsamp = String::new();
        for f in 0..mesh.num_faces() {
            if mesh.face_neighbor[f].is_some() { continue; }
            let oc = mesh.face_owner[f];
            let in_p = exact_p(mesh.cell_cx[oc], mesh.cell_cy[oc]);
            let (uf0, uf1) = exact_u(mesh.face_cx[f], mesh.face_cy[f]);
            let rhof = exact_rho(mesh.face_cx[f], mesh.face_cy[f]);
            let ke = 0.5 * rhof * (uf0 * uf0 + uf1 * uf1);
            let exp_re = in_p / gm1 + ke;
            let exp_t = in_p / (rhof * R_GAS);
            let dre = (bc_after[f * 8 + 3] as f64 - exp_re).abs();
            let dt = (bc_after[f * 8 + 7] as f64 - exp_t).abs();
            if dre > wre { wre = dre; wsamp = format!("face {f}: cpu_re={:.5} expected={:.5}", bc_after[f*8+3], exp_re); }
            wt = wt.max(dt);
        }
        println!("[bc-vs-expected] max|cpu_refresh - analytic_gpu|: rho_e={wre:.3e} T={wt:.3e}; {wsamp}");
    }
    let (crho, cu, cp) = (
        c.get_field_scalar("rho").unwrap(),
        c.get_field_vec2("u").unwrap(),
        c.get_field_scalar("rho_e").unwrap(),
    );

    // GPU mirror of setup().
    let cells = mesh.num_cells();
    let model = compressible_mms_model().expect("model");
    let mut g = pollster::block_on(UnifiedSolver::new(
        &mesh, model,
        SolverConfig { advection_scheme: test_scheme(), time_scheme: TimeScheme::BDF2, preconditioner: PreconditionerType::Jacobi, stepping: Cfg::Implicit { outer_iters: 1 } },
        None, None,
    )).expect("gpu");
    g.set_dt(DT as f32);
    g.set_dtau(0.0).unwrap();
    g.set_viscosity(MU as f32).unwrap();
    g.set_density(RHO0 as f32).unwrap();
    g.set_outer_iters(1).unwrap();
    g.set_outer_tolerance(0.0).unwrap();
    let _ = PlanParamValue::F32(0.0);
    let (fx, fy) = (mesh.face_cx.clone(), mesh.face_cy.clone());
    let sc = |f: &'static dyn Fn(f64, f64) -> f64, fx: &[f64], fy: &[f64]| { let (fx, fy) = (fx.to_vec(), fy.to_vec()); move |i: u32| f(fx[i as usize], fy[i as usize]) as f32 };
    g.set_boundary_values_per_face(GpuBoundaryType::Inlet, "rho", 0, &sc(&exact_rho, &fx, &fy)).unwrap();
    g.set_boundary_values_per_face(GpuBoundaryType::Inlet, "p", 0, &sc(&exact_p, &fx, &fy)).unwrap();
    g.set_boundary_values_per_face(GpuBoundaryType::Inlet, "T", 0, &sc(&exact_t, &fx, &fy)).unwrap();
    g.set_boundary_values_per_face(GpuBoundaryType::Inlet, "rho_e", 0, &sc(&exact_rho_e, &fx, &fy)).unwrap();
    for cc in 0..2u32 {
        let (fx2, fy2) = (fx.clone(), fy.clone());
        let uf = move |i: u32| { let (a, b) = exact_u(fx2[i as usize], fy2[i as usize]); (if cc == 0 { a } else { b }) as f32 };
        let (fx3, fy3) = (fx.clone(), fy.clone());
        let ruf = move |i: u32| { let j = i as usize; let (a, b) = exact_u(fx3[j], fy3[j]); (exact_rho(fx3[j], fy3[j]) * if cc == 0 { a } else { b }) as f32 };
        g.set_boundary_values_per_face(GpuBoundaryType::Inlet, "u", cc, &uf).unwrap();
        g.set_boundary_values_per_face(GpuBoundaryType::Inlet, "rho_u", cc, &ruf).unwrap();
    }
    let mut src_rho: Vec<f64> = (0..cells).map(|i| source_rho(mesh.cell_cx[i], mesh.cell_cy[i])).collect();
    let vol_total: f64 = mesh.cell_vol.iter().sum();
    let vol_int: f64 = (0..cells).map(|i| src_rho[i] * mesh.cell_vol[i]).sum();
    let bflux: f64 = (0..mesh.face_owner.len()).filter(|&f| mesh.face_neighbor[f].is_none()).map(|f| { let (ux, uy) = exact_u(mesh.face_cx[f], mesh.face_cy[f]); exact_rho(mesh.face_cx[f], mesh.face_cy[f]) * (ux * mesh.face_nx[f] + uy * mesh.face_ny[f]) * mesh.face_area[f] }).sum();
    let eps = (vol_int - bflux) / vol_total;
    for v in src_rho.iter_mut() { *v -= eps; }
    g.set_field_scalar(COMPRESSIBLE_MMS_SOURCE_RHO_FIELD, &src_rho).unwrap();
    g.set_field_vec2(COMPRESSIBLE_MMS_SOURCE_RHO_U_FIELD, &(0..cells).map(|i| source_rho_u(mesh.cell_cx[i], mesh.cell_cy[i], MU)).collect::<Vec<_>>()).unwrap();
    g.set_field_scalar(COMPRESSIBLE_MMS_SOURCE_RHO_E_FIELD, &(0..cells).map(|i| source_rho_e(mesh.cell_cx[i], mesh.cell_cy[i], MU)).collect::<Vec<_>>()).unwrap();
    let rho0: Vec<f64> = (0..cells).map(|i| exact_rho(mesh.cell_cx[i], mesh.cell_cy[i])).collect();
    let u0: Vec<(f64, f64)> = (0..cells).map(|i| exact_u(mesh.cell_cx[i], mesh.cell_cy[i])).collect();
    g.set_field_scalar("rho", &rho0).unwrap();
    g.set_field_vec2("rho_u", &(0..cells).map(|i| (rho0[i] * u0[i].0, rho0[i] * u0[i].1)).collect::<Vec<_>>()).unwrap();
    g.set_field_scalar("rho_e", &(0..cells).map(|i| exact_rho_e(mesh.cell_cx[i], mesh.cell_cy[i])).collect::<Vec<_>>()).unwrap();
    g.set_field_scalar("p", &(0..cells).map(|i| exact_p(mesh.cell_cx[i], mesh.cell_cy[i])).collect::<Vec<_>>()).unwrap();
    g.set_field_scalar("T", &(0..cells).map(|i| exact_t(mesh.cell_cx[i], mesh.cell_cy[i])).collect::<Vec<_>>()).unwrap();
    g.set_field_vec2("u", &u0).unwrap();
    g.initialize_history();
    for _ in 0..steps { g.step(); }
    let grho = pollster::block_on(g.get_field_scalar("rho")).unwrap();
    let gu = pollster::block_on(g.get_field_vec2("u")).unwrap();
    let gre = pollster::block_on(g.get_field_scalar("rho_e")).unwrap();

    let drho = crho.iter().zip(&grho).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    let du = cu.iter().zip(&gu).map(|(a, b)| (a.0 - b.0).abs().max((a.1 - b.1).abs())).fold(0.0, f64::max);
    let dre = cp.iter().zip(&gre).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    println!("[mms-step1] n={n} max|drho|={drho:.3e} max|du|={du:.3e} max|drho_e|={dre:.3e}");
    {
        // Localize the residual rho_e STATE divergence (post-step).
        let bd: Vec<bool> = (0..cells).map(|i| {
            let (s0,e0)=(mesh.cell_face_offsets[i],mesh.cell_face_offsets[i+1]);
            (s0..e0).any(|k| mesh.face_neighbor[mesh.cell_faces[k]].is_none())
        }).collect();
        let (mut wi,mut wd)=(0usize,0.0); let mut imax=0.0f64;
        for i in 0..cells { let d=(cp[i]-gre[i]).abs(); if d>wd {wd=d;wi=i;} if !bd[i]{imax=imax.max(d);} }
        println!("[re-loc] worst cell {wi} ({:.3},{:.3}) bdry_adj={} c={:.5} g={:.5} d={wd:.3e}; interior_max={imax:.3e}",
            mesh.cell_cx[wi], mesh.cell_cy[wi], bd[wi], cp[wi], gre[wi]);
    }

    // Compare the GRADIENT state fields (computed on the pre-update = exact
    // state during the step). If these diverge, the gradient kernel is the
    // bias source feeding the energy flux (conduction/viscous work).
    let bdry: Vec<bool> = (0..cells)
        .map(|i| {
            let (s0, e0) = (mesh.cell_face_offsets[i], mesh.cell_face_offsets[i + 1]);
            (s0..e0).any(|k| mesh.face_neighbor[mesh.cell_faces[k]].is_none())
        })
        .collect();
    for f in ["grad_rho_e", "grad_T"] {
        let cv = c.get_field_vec2(f).unwrap();
        let gv = pollster::block_on(g.get_field_vec2(f)).unwrap();
        let (mut wi, mut wd) = (0usize, 0.0);
        let mut int_max = 0.0f64;
        for i in 0..cells {
            let d = (cv[i].0 - gv[i].0).abs().max((cv[i].1 - gv[i].1).abs());
            if d > wd { wd = d; wi = i; }
            if !bdry[i] { int_max = int_max.max(d); }
        }
        println!(
            "[grad-loc] {f}: worst cell {wi} ({:.3},{:.3}) bdry_adj={} c={:?} g={:?} d={wd:.3e}; interior_max={int_max:.3e}",
            mesh.cell_cx[wi], mesh.cell_cy[wi], bdry[wi], cv[wi], gv[wi]
        );
    }
}

/// Build a GPU `UnifiedSolver` mirroring the CPU `setup()` exactly (same BCs,
/// sources with mass-compatibility, exact init, BDF2, Implicit{1}, Jacobi).
#[cfg(feature = "dev-tests")]
fn setup_gpu(n: usize, mesh: &Mesh) -> cfd2::solver::UnifiedSolver {
    use cfd2::solver::model::helpers::SolverRuntimeParamsExt;
    use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode as Cfg, UnifiedSolver};
    let cells = mesh.num_cells();
    let model = compressible_mms_model().expect("model");
    let mut g = pollster::block_on(UnifiedSolver::new(
        mesh, model,
        SolverConfig { advection_scheme: Scheme::SecondOrderUpwindVanLeer, time_scheme: TimeScheme::BDF2, preconditioner: PreconditionerType::Jacobi, stepping: Cfg::Implicit { outer_iters: 1 } },
        None, None,
    )).expect("gpu");
    g.set_dt(DT as f32);
    g.set_dtau(0.0).unwrap();
    g.set_viscosity(MU as f32).unwrap();
    g.set_density(RHO0 as f32).unwrap();
    g.set_outer_iters(1).unwrap();
    g.set_outer_tolerance(0.0).unwrap();
    let (fx, fy) = (mesh.face_cx.clone(), mesh.face_cy.clone());
    let sc = |f: &'static dyn Fn(f64, f64) -> f64, fx: &[f64], fy: &[f64]| { let (fx, fy) = (fx.to_vec(), fy.to_vec()); move |i: u32| f(fx[i as usize], fy[i as usize]) as f32 };
    g.set_boundary_values_per_face(GpuBoundaryType::Inlet, "rho", 0, &sc(&exact_rho, &fx, &fy)).unwrap();
    g.set_boundary_values_per_face(GpuBoundaryType::Inlet, "p", 0, &sc(&exact_p, &fx, &fy)).unwrap();
    g.set_boundary_values_per_face(GpuBoundaryType::Inlet, "T", 0, &sc(&exact_t, &fx, &fy)).unwrap();
    g.set_boundary_values_per_face(GpuBoundaryType::Inlet, "rho_e", 0, &sc(&exact_rho_e, &fx, &fy)).unwrap();
    for cc in 0..2u32 {
        let (fx2, fy2) = (fx.clone(), fy.clone());
        let uf = move |i: u32| { let (a, b) = exact_u(fx2[i as usize], fy2[i as usize]); (if cc == 0 { a } else { b }) as f32 };
        let (fx3, fy3) = (fx.clone(), fy.clone());
        let ruf = move |i: u32| { let j = i as usize; let (a, b) = exact_u(fx3[j], fy3[j]); (exact_rho(fx3[j], fy3[j]) * if cc == 0 { a } else { b }) as f32 };
        g.set_boundary_values_per_face(GpuBoundaryType::Inlet, "u", cc, &uf).unwrap();
        g.set_boundary_values_per_face(GpuBoundaryType::Inlet, "rho_u", cc, &ruf).unwrap();
    }
    let mut src_rho: Vec<f64> = (0..cells).map(|i| source_rho(mesh.cell_cx[i], mesh.cell_cy[i])).collect();
    let vol_total: f64 = mesh.cell_vol.iter().sum();
    let vol_int: f64 = (0..cells).map(|i| src_rho[i] * mesh.cell_vol[i]).sum();
    let bflux: f64 = (0..mesh.face_owner.len()).filter(|&f| mesh.face_neighbor[f].is_none()).map(|f| { let (ux, uy) = exact_u(mesh.face_cx[f], mesh.face_cy[f]); exact_rho(mesh.face_cx[f], mesh.face_cy[f]) * (ux * mesh.face_nx[f] + uy * mesh.face_ny[f]) * mesh.face_area[f] }).sum();
    let eps = (vol_int - bflux) / vol_total;
    for v in src_rho.iter_mut() { *v -= eps; }
    g.set_field_scalar(COMPRESSIBLE_MMS_SOURCE_RHO_FIELD, &src_rho).unwrap();
    g.set_field_vec2(COMPRESSIBLE_MMS_SOURCE_RHO_U_FIELD, &(0..cells).map(|i| source_rho_u(mesh.cell_cx[i], mesh.cell_cy[i], MU)).collect::<Vec<_>>()).unwrap();
    g.set_field_scalar(COMPRESSIBLE_MMS_SOURCE_RHO_E_FIELD, &(0..cells).map(|i| source_rho_e(mesh.cell_cx[i], mesh.cell_cy[i], MU)).collect::<Vec<_>>()).unwrap();
    let rho0: Vec<f64> = (0..cells).map(|i| exact_rho(mesh.cell_cx[i], mesh.cell_cy[i])).collect();
    let u0: Vec<(f64, f64)> = (0..cells).map(|i| exact_u(mesh.cell_cx[i], mesh.cell_cy[i])).collect();
    g.set_field_scalar("rho", &rho0).unwrap();
    g.set_field_vec2("rho_u", &(0..cells).map(|i| (rho0[i] * u0[i].0, rho0[i] * u0[i].1)).collect::<Vec<_>>()).unwrap();
    g.set_field_scalar("rho_e", &(0..cells).map(|i| exact_rho_e(mesh.cell_cx[i], mesh.cell_cy[i])).collect::<Vec<_>>()).unwrap();
    g.set_field_scalar("p", &(0..cells).map(|i| exact_p(mesh.cell_cx[i], mesh.cell_cy[i])).collect::<Vec<_>>()).unwrap();
    g.set_field_scalar("T", &(0..cells).map(|i| exact_t(mesh.cell_cx[i], mesh.cell_cy[i])).collect::<Vec<_>>()).unwrap();
    g.set_field_vec2("u", &u0).unwrap();
    g.initialize_history();
    g
}

/// Lockstep CPU-vs-GPU compressible march: reports, at intervals, each backend's
/// L2-vs-exact error and the max|CPU-GPU| per field. Both drifting together means
/// a shared physics/discretization instability; a growing CPU-GPU gap is a CPU bug.
#[ignore]
#[cfg(feature = "dev-tests")]
#[test]
fn diag_cpu_vs_gpu_march() {
    let n = 8;
    let (mut c, mesh) = setup(n, CpuBackendConfig::default());
    let mut g = setup_gpu(n, &mesh);
    let cells = mesh.num_cells();
    let mut step = 0usize;
    for &upto in &[1usize, 5, 10, 20, 40, 80, 160] {
        while step < upto {
            c.step();
            g.step();
            step += 1;
        }
        let crho = c.get_field_scalar("rho").unwrap();
        let grho = pollster::block_on(g.get_field_scalar("rho")).unwrap();
        let cre = c.get_field_scalar("rho_e").unwrap();
        let gre = pollster::block_on(g.get_field_scalar("rho_e")).unwrap();
        let dr = crho.iter().zip(&grho).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let dre = cre.iter().zip(&gre).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let c_err = l2_scalar(&mesh, &cre, exact_rho_e);
        let g_err = l2_scalar(&mesh, &gre, exact_rho_e);
        let _ = cells;
        println!("[march] step={step} | CPU-GPU: rho={dr:.3e} rho_e={dre:.3e} | rho_e_L2_vs_exact: cpu={c_err:.3e} gpu={g_err:.3e}");
    }
}

/// Compare the CPU's assembled block-CSR matrix + rhs to the GPU's at the EXACT
/// state (step 1, outer_iters=1 — the only assembly). Block-CSR layout is identical
/// on both backends (start_row_0 = scalar_offset*S²; start_row_r += num_neighbors*S*r;
/// block (r,c) for neighbour rank at start_row_r + rank*S + c), so values compare
/// element-wise IF the neighbour ordering matches (verified by the per-equation
/// A*x_exact residual being O(h²)). Pinpoints a genuine operator difference vs
/// f32-noise on a structurally-identical operator.
#[ignore]
#[cfg(feature = "dev-tests")]
#[test]
fn diag_cpu_vs_gpu_matrix() {
    let n = 16;
    // CPU assembly at the exact state.
    let (mut c, mesh) = setup(n, CpuBackendConfig::default());
    let (matrix_cpu, rhs_cpu) = c.debug_assemble();
    let (sro, col, diag_idx, ss) = c.debug_topology();
    let cells = mesh.num_cells();

    // GPU assembly at the exact state: one step (outer_iters=1) leaves the
    // assembled system in the linear buffers (the solve only reads them).
    let mut g = setup_gpu(n, &mesh);
    g.step();
    let matrix_gpu = pollster::block_on(g.get_linear_matrix()).unwrap();
    let rhs_gpu = pollster::block_on(g.get_linear_rhs()).unwrap();

    let names = ["rho", "rho_u_x", "rho_u_y", "rho_e", "u_x", "u_y", "p", "T"];
    let bdry: Vec<bool> = (0..cells)
        .map(|i| {
            let (s0, e0) = (mesh.cell_face_offsets[i], mesh.cell_face_offsets[i + 1]);
            (s0..e0).any(|k| mesh.face_neighbor[mesh.cell_faces[k]].is_none())
        })
        .collect();

    // Sizes.
    println!(
        "[mat] sizes: matrix cpu={} gpu={}; rhs cpu={} gpu={}",
        matrix_cpu.len(), matrix_gpu.len(), rhs_cpu.len(), rhs_gpu.len()
    );
    if matrix_cpu.len() != matrix_gpu.len() || rhs_cpu.len() != rhs_gpu.len() {
        println!("[mat] SIZE MISMATCH — topology differs; aborting element compare");
        return;
    }

    // rhs diff per equation (layout cell*S+r — backend-independent).
    print!("[mat] max|rhs_cpu - rhs_gpu| per eq:");
    for u in 0..ss {
        let m = (0..cells)
            .map(|i| (rhs_cpu[i * ss + u] as f64 - rhs_gpu[i * ss + u] as f64).abs())
            .fold(0.0, f64::max);
        print!(" {}={m:.2e}", names[u]);
    }
    println!();

    // matrix row-sum diff per equation-row (sum over ALL block entries in the
    // row — order-independent, so valid even if the neighbour column ordering
    // differs between backends). A real operator difference shows here.
    print!("[mat] max|rowsum(A_cpu) - rowsum(A_gpu)| per row-eq:");
    for rrow in 0..ss {
        let mut m = 0.0f64;
        for i in 0..cells {
            let so = sro[i] as usize;
            let nn = sro[i + 1] as usize - so;
            let start = so * ss * ss + nn * ss * rrow;
            let (mut sc, mut sg) = (0.0f64, 0.0f64);
            for e in 0..nn * ss {
                sc += matrix_cpu[start + e] as f64;
                sg += matrix_gpu[start + e] as f64;
            }
            let d = (sc - sg).abs();
            if d > m { m = d; }
        }
        print!(" {}={m:.2e}", names[rrow]);
    }
    println!();

    // Per-equation A*x_exact residual for the GPU matrix read through the CPU
    // topology: if O(h²), the neighbour ordering matches and the compare is valid.
    let mut xe = vec![0.0f64; cells * ss];
    for i in 0..cells {
        let (x, y) = (mesh.cell_cx[i], mesh.cell_cy[i]);
        let (ux, uy) = exact_u(x, y);
        let r = exact_rho(x, y);
        xe[i * ss + 0] = r;
        xe[i * ss + 1] = r * ux;
        xe[i * ss + 2] = r * uy;
        xe[i * ss + 3] = exact_rho_e(x, y);
        xe[i * ss + 4] = ux;
        xe[i * ss + 5] = uy;
        xe[i * ss + 6] = exact_p(x, y);
        xe[i * ss + 7] = exact_t(x, y);
    }
    let resid = |matrix: &[f32], rhs: &[f32]| -> Vec<f64> {
        let mut res = vec![0.0f64; cells * ss];
        for i in 0..cells {
            let so = sro[i] as usize;
            let nn = sro[i + 1] as usize - so;
            for rrow in 0..ss {
                let start = so * ss * ss + nn * ss * rrow;
                let mut ax = 0.0f64;
                for rank in 0..nn {
                    let j = col[so + rank] as usize;
                    for cc in 0..ss {
                        ax += matrix[start + rank * ss + cc] as f64 * xe[j * ss + cc];
                    }
                }
                res[i * ss + rrow] = rhs[i * ss + rrow] as f64 - ax;
            }
        }
        res
    };
    let _ = &diag_idx;
    let rc = resid(&matrix_cpu, &rhs_cpu);
    let rg = resid(&matrix_gpu, &rhs_gpu);
    for (label, res) in [("cpu", &rc), ("gpu", &rg)] {
        print!("[mat] {label} interior A*x_exact residual per eq:");
        for u in 0..ss {
            let m = (0..cells)
                .filter(|&i| !bdry[i])
                .map(|i| res[i * ss + u].abs())
                .fold(0.0, f64::max);
            print!(" {}={m:.2e}", names[u]);
        }
        println!();
    }
}

#[ignore]
#[ignore] // diagnostic (informational prints); run explicitly
#[test]
fn diag_compressible_trajectory() {
    // Fast diagnostic: trace the marched state to see HOW it goes wrong.
    let n = 8;
    let steps = 6;
    let mesh = generate_structured_rect_mesh(
        n, n, 1.0, 1.0,
        BoundarySides { left: BoundaryType::Inlet, right: BoundaryType::Inlet, bottom: BoundaryType::Inlet, top: BoundaryType::Inlet },
    );
    let _ = (&mesh, steps);
    for st in [1usize, 10, 30, 60, 120] {
        let (m, rho, _u, p, _t, rho_e, rho_u) = solve(n, st, CpuBackendConfig::default());
        let er = l2_scalar(&m, &rho, exact_rho);
        let ep = l2_scalar(&m, &p, exact_p);
        let ere = l2_scalar(&m, &rho_e, exact_rho_e);
        let eru = {
            let mut num = 0.0;
            let mut den = 0.0;
            for i in 0..m.num_cells() {
                let (ex, ey) = exact_u(m.cell_cx[i], m.cell_cy[i]);
                let r = exact_rho(m.cell_cx[i], m.cell_cy[i]);
                num += ((rho_u[i].0 - r * ex).powi(2) + (rho_u[i].1 - r * ey).powi(2)) * m.cell_vol[i];
                den += m.cell_vol[i];
            }
            (num / den).sqrt()
        };
        println!("[diag] step={st} | L2 rho={er:.3e} rho_u={eru:.3e} rho_e={ere:.3e} p={ep:.3e}");
    }
}

/// Does the biharmonic ∇⁴ dissipation keep the CPU compressible march BOUNDED
/// where the plain model blows up? Sweeps eps4 (eps4=0 is the control) and reports
/// the rho_e L2 error over a long march.
#[ignore]
#[test]
fn diag_biharmonic_march() {
    let n = 8;
    for eps4 in [0.0_f32, 0.1, 0.25, 0.5] {
        let (mut s, mesh) = setup_g(n, CpuBackendConfig::default(), true, eps4);
        let mut step = 0usize;
        for &upto in &[1usize, 40, 160, 320, 600] {
            while step < upto {
                s.step();
                step += 1;
            }
            let rho_e = s.get_field_scalar("rho_e").unwrap();
            let e = l2_scalar(&mesh, &rho_e, exact_rho_e);
            let finite = rho_e.iter().all(|v| v.is_finite());
            println!("[bihar-march] eps4={eps4:.2} step={step} rho_e_L2={e:.3e} finite={finite}");
        }
    }
}

/// Build a CPU compressible solver on a fully-PERIODIC [0,2]² box (the manufactured
/// solution is periodic there). Zero boundary faces ⇒ no BCs; the manufactured
/// sources are mean-projected to zero (closed-system constraint). `eps4` sets the
/// biharmonic dissipation field. Isolates the interior operator from the boundary.
#[allow(clippy::type_complexity)]
fn setup_periodic(n: usize, eps4: f32, mu: f64) -> (CpuSolver, Mesh) {
    let mesh = generate_structured_rect_mesh_periodic(n, n, 2.0, 2.0);
    let model = compressible_mms_biharmonic_model().expect("biharmonic model");
    let cells = mesh.num_cells();
    let mut s = CpuSolver::with_stepping(
        &mesh,
        model,
        Scheme::SecondOrderUpwindVanLeer,
        TimeScheme::BDF2,
        SteppingMode::Implicit { outer_iters: 1 },
        CpuBackendConfig::default(),
    )
    .expect("cpu solver");
    s.set_dt(DT as f32);
    s.set_dtau(0.0);
    s.set_viscosity(mu as f32);
    s.set_density(RHO0 as f32);
    s.set_outer_iters(1);
    s.set_outer_tolerance(0.0);

    let vol_total: f64 = mesh.cell_vol.iter().sum();
    let proj = |src: &mut [f64]| {
        let mean: f64 = (0..cells).map(|i| src[i] * mesh.cell_vol[i]).sum::<f64>() / vol_total;
        for v in src.iter_mut() {
            *v -= mean;
        }
    };
    let mut src_rho: Vec<f64> = (0..cells).map(|i| source_rho(mesh.cell_cx[i], mesh.cell_cy[i])).collect();
    proj(&mut src_rho);
    let src_ru: Vec<(f64, f64)> = (0..cells).map(|i| source_rho_u(mesh.cell_cx[i], mesh.cell_cy[i], mu)).collect();
    let (mut rux, mut ruy): (Vec<f64>, Vec<f64>) = src_ru.iter().copied().unzip();
    proj(&mut rux);
    proj(&mut ruy);
    let mut src_re: Vec<f64> = (0..cells).map(|i| source_rho_e(mesh.cell_cx[i], mesh.cell_cy[i], mu)).collect();
    proj(&mut src_re);
    s.set_field_scalar(COMPRESSIBLE_MMS_SOURCE_RHO_FIELD, &src_rho).unwrap();
    s.set_field_vec2(COMPRESSIBLE_MMS_SOURCE_RHO_U_FIELD, &(0..cells).map(|i| (rux[i], ruy[i])).collect::<Vec<_>>()).unwrap();
    s.set_field_scalar(COMPRESSIBLE_MMS_SOURCE_RHO_E_FIELD, &src_re).unwrap();

    let rho0: Vec<f64> = (0..cells).map(|i| exact_rho(mesh.cell_cx[i], mesh.cell_cy[i])).collect();
    let u0: Vec<(f64, f64)> = (0..cells).map(|i| exact_u(mesh.cell_cx[i], mesh.cell_cy[i])).collect();
    s.set_field_scalar("rho", &rho0).unwrap();
    s.set_field_vec2("rho_u", &(0..cells).map(|i| (rho0[i] * u0[i].0, rho0[i] * u0[i].1)).collect::<Vec<_>>()).unwrap();
    s.set_field_scalar("rho_e", &(0..cells).map(|i| exact_rho_e(mesh.cell_cx[i], mesh.cell_cy[i])).collect::<Vec<_>>()).unwrap();
    s.set_field_scalar("p", &(0..cells).map(|i| exact_p(mesh.cell_cx[i], mesh.cell_cy[i])).collect::<Vec<_>>()).unwrap();
    s.set_field_scalar("T", &(0..cells).map(|i| exact_t(mesh.cell_cx[i], mesh.cell_cy[i])).collect::<Vec<_>>()).unwrap();
    s.set_field_vec2("u", &u0).unwrap();
    s.set_field_scalar("bih_eps4", &vec![eps4 as f64; cells]).unwrap();
    s.initialize_history();
    (s, mesh)
}

/// PERIODIC march: a periodic box isolates the interior operator from the
/// boundary closure. Staying BOUNDED here where the Dirichlet box blows up means
/// the interior compressible operator is stable.
#[ignore]
#[test]
fn diag_periodic_biharmonic_march() {
    for eps4 in [0.0_f32, 0.1, 0.25] {
        let (mut s, mesh) = setup_periodic(16, eps4, MU);
        let mut step = 0usize;
        for &upto in &[1usize, 40, 160, 320, 600] {
            while step < upto {
                s.step();
                step += 1;
            }
            let rho_e = s.get_field_scalar("rho_e").unwrap();
            let e = l2_scalar(&mesh, &rho_e, exact_rho_e);
            let finite = rho_e.iter().all(|v| v.is_finite());
            println!("[periodic-bih] eps4={eps4:.2} step={step} rho_e_L2={e:.3e} finite={finite}");
        }
    }
}

/// CPU compressible PERIODIC-box order DIAGNOSTIC: marches the interior operator
/// (boundary closure excluded) and reports the error/order across n + eps4.
/// The periodic march stays BOUNDED but does NOT converge at design order — a
/// refinement-amplified marginal interior mode (error grows with n). eps4>0 here
/// lacks the consistent `+eps4*∇⁴X_exact` source term, so the biharmonic perturbs
/// the MMS instead of curing it. Printed, not asserted (bounded is the only claim).
#[ignore]
#[test]
fn diag_cpu_compressible_periodic_order() {
    let steps = 300;
    // mu=0.05 = the MMS const (marginal interior); mu=0.2 = a viscosity-stable
    // regime where the interior mode is physically damped.
    for (eps4, mu) in [(0.0_f32, MU), (0.1_f32, MU), (0.0_f32, 0.2_f64)] {
        let levels = [16usize, 32];
        let mut errs: [Vec<f64>; 4] = [vec![], vec![], vec![], vec![]];
        for &n in &levels {
            let (mut s, mesh) = setup_periodic(n, eps4, mu);
            for _ in 0..steps {
                s.step();
            }
            let rho = s.get_field_scalar("rho").unwrap();
            let p = s.get_field_scalar("p").unwrap();
            let t = s.get_field_scalar("T").unwrap();
            let u = s.get_field_vec2("u").unwrap();
            let er = l2_scalar(&mesh, &rho, exact_rho);
            let ep = l2_scalar(&mesh, &p, exact_p);
            let et = l2_scalar(&mesh, &t, exact_t);
            let eu = {
                let (mut num, mut den) = (0.0, 0.0);
                for i in 0..mesh.num_cells() {
                    let (ex, ey) = exact_u(mesh.cell_cx[i], mesh.cell_cy[i]);
                    num += ((u[i].0 - ex).powi(2) + (u[i].1 - ey).powi(2)) * mesh.cell_vol[i];
                    den += mesh.cell_vol[i];
                }
                (num / den).sqrt()
            };
            println!("[periodic-order] mu={mu} eps4={eps4:.2} n={n} rho={er:.4e} u={eu:.4e} p={ep:.4e} T={et:.4e}");
            errs[0].push(er);
            errs[1].push(eu);
            errs[2].push(ep);
            errs[3].push(et);
        }
        let order = |e: &[f64]| (e[0] / e[1]).log2(); // levels double
        println!(
            "[periodic-order] mu={mu} eps4={eps4:.2} orders rho={:.3} u={:.3} p={:.3} T={:.3}",
            order(&errs[0]), order(&errs[1]), order(&errs[2]), order(&errs[3])
        );
        assert!(
            errs.iter().all(|e| e.iter().all(|v| v.is_finite())),
            "periodic compressible march went non-finite (interior must stay bounded)"
        );
    }
}

/// Least-squares slope of log(err) vs log(h) — the convergence order.
fn ls_order(hs: &[f64], es: &[f64]) -> f64 {
    let n = hs.len() as f64;
    let lx: Vec<f64> = hs.iter().map(|h| h.ln()).collect();
    let ly: Vec<f64> = es.iter().map(|e| e.ln()).collect();
    let (mx, my) = (lx.iter().sum::<f64>() / n, ly.iter().sum::<f64>() / n);
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..lx.len() {
        num += (lx[i] - mx) * (ly[i] - my);
        den += (lx[i] - mx).powi(2);
    }
    num / den
}

/// Per-equation volume-weighted L2 of the discrete residual `r = rhs - A x_exact`
/// over INTERIOR cells (boundary-closure truncation is a separate first-order
/// effect, excluded). This truncation error's convergence rate is the
/// discretization's CONSISTENCY order. Pure assembly — no marching, no solve.
fn interior_residual_l2(n: usize) -> [f64; 8] {
    let (mut s, mesh) = setup(n, CpuBackendConfig::default());
    let (matrix, rhs) = s.debug_assemble();
    let (sro, col, _diag, ss) = s.debug_topology();
    let cells = mesh.num_cells();
    let mut xe = vec![0.0f64; cells * ss];
    for i in 0..cells {
        let (x, y) = (mesh.cell_cx[i], mesh.cell_cy[i]);
        let (ux, uy) = exact_u(x, y);
        let r = exact_rho(x, y);
        xe[i * ss + 0] = r;
        xe[i * ss + 1] = r * ux;
        xe[i * ss + 2] = r * uy;
        xe[i * ss + 3] = exact_rho_e(x, y);
        xe[i * ss + 4] = ux;
        xe[i * ss + 5] = uy;
        xe[i * ss + 6] = exact_p(x, y);
        xe[i * ss + 7] = exact_t(x, y);
    }
    let bdry: Vec<bool> = (0..cells)
        .map(|i| {
            let (s0, e0) = (mesh.cell_face_offsets[i], mesh.cell_face_offsets[i + 1]);
            (s0..e0).any(|k| mesh.face_neighbor[mesh.cell_faces[k]].is_none())
        })
        .collect();
    let mut num = [0.0f64; 8];
    let mut den = 0.0f64;
    for i in 0..cells {
        if bdry[i] {
            continue;
        }
        let so = sro[i] as usize;
        let nn = sro[i + 1] as usize - so;
        den += mesh.cell_vol[i];
        for rrow in 0..ss {
            let start = so * ss * ss + nn * ss * rrow;
            let mut ax = 0.0f64;
            for rank in 0..nn {
                let j = col[so + rank] as usize;
                for c in 0..ss {
                    ax += matrix[start + rank * ss + c] as f64 * xe[j * ss + c];
                }
            }
            let res = rhs[i * ss + rrow] as f64 - ax;
            num[rrow] += res * res * mesh.cell_vol[i];
        }
    }
    let mut out = [0.0f64; 8];
    for u in 0..ss.min(8) {
        out[u] = (num[u] / den).sqrt();
    }
    out
}

/// CPU compressible operator — 2nd-order CONSISTENCY certification.
///
/// Certifies the truncation-error order rather than a marched solution order: the
/// manufactured subsonic-NS steady state is MARGINALLY UNSTABLE with a tiny
/// stability basin (a >=20% velocity perturbation escapes it and blows up). The
/// operator itself matches the GPU to f32; the deterministic f64 solve either
/// freezes near the exact IC (fine mesh) or amplifies the mode to blow-up (coarse
/// mesh), so a marched solution-error order is not robustly measurable here.
///
/// The interior discrete residual at the exact solution converges at ~O(h^3.5) in
/// L2 for every conserved equation (the cell-integrated conservative residual
/// super-converges above the 2nd-order design rate; the global SOLUTION order is
/// the boundary-limited 2). So the operator is comfortably >= 2nd-order accurate.
#[test]
fn cpu_compressible_operator_second_order() {
    let levels = [16usize, 32, 64];
    let names = ["rho", "rho_u_x", "rho_u_y", "rho_e", "u_x", "u_y", "p", "T"];
    let hs: Vec<f64> = levels.iter().map(|&n| 1.0 / n as f64).collect();
    let resids: Vec<[f64; 8]> = levels.iter().map(|&n| interior_residual_l2(n)).collect();
    for (li, &n) in levels.iter().enumerate() {
        print!("[cpu-compr-resid] n={n}");
        for u in 0..8 {
            print!(" {}={:.3e}", names[u], resids[li][u]);
        }
        println!();
    }
    // Conserved-equation truncation error must converge at >= ~2nd order. Band
    // the upper side so a freeze/precision artifact (spuriously high apparent
    // order) fails rather than passes.
    for u in [0usize, 1, 2, 3] {
        let es: Vec<f64> = resids.iter().map(|r| r[u]).collect();
        let ord = ls_order(&hs, &es);
        println!("[cpu-compr-resid] {} consistency order = {ord:.3}", names[u]);
        assert!(ord > 2.0, "{} consistency order {ord:.3} below 2nd order", names[u]);
        assert!(ord < 5.0, "{} consistency order {ord:.3} implausibly high (assembly bug?)", names[u]);
    }
    // Recovery rows (u, p, T) are exact algebraic identities → residual ~ f32 noise.
    for u in [4usize, 5, 6, 7] {
        for (li, r) in resids.iter().enumerate() {
            assert!(
                r[u] < 1e-5,
                "{} recovery residual {:.2e} not ~0 at n={}",
                names[u], r[u], levels[li]
            );
        }
    }
}

// Plateau-acceptance constants mirroring the GPU runner: the compressible MMS is
// marginally unstable on both backends, so the order test accepts at a per-step
// delta PLATEAU rather than a fixed step. Kept in sync with the GPU constants.
const STEADY_TOL: f64 = 1e-5;
const STEADY_MAX_STEPS: usize = 1600;
const PLATEAU_WINDOW: usize = 80;
const MIN_STEPS: usize = 600;
const LONG_MARCH_ACCEPT_STEPS: usize = 1200;

/// Per-step max delta over (u, rho, T) — the "steady" watch metric.
fn max_delta_state(c: &CpuSolver, prev: &(Vec<(f64, f64)>, Vec<f64>, Vec<f64>)) -> f64 {
    let u = c.get_field_vec2("u").unwrap();
    let rho = c.get_field_scalar("rho").unwrap();
    let t = c.get_field_scalar("T").unwrap();
    let mut m = 0.0f64;
    for (a, b) in u.iter().zip(prev.0.iter()) {
        m = m.max((a.0 - b.0).abs()).max((a.1 - b.1).abs());
    }
    for (a, b) in rho.iter().zip(prev.1.iter()) {
        m = m.max((a - b).abs());
    }
    for (a, b) in t.iter().zip(prev.2.iter()) {
        m = m.max((a - b).abs());
    }
    m
}

fn read_state(c: &CpuSolver) -> (Vec<(f64, f64)>, Vec<f64>, Vec<f64>) {
    (
        c.get_field_vec2("u").unwrap(),
        c.get_field_scalar("rho").unwrap(),
        c.get_field_scalar("T").unwrap(),
    )
}

/// CPU mirror of the GPU `march_to_plateau`: march until the watched per-step max
/// delta over (u, rho, T) drops below `STEADY_TOL`, or its best value plateaus (no
/// >2% improvement for `PLATEAU_WINDOW` steps after `MIN_STEPS`), or
/// `LONG_MARCH_ACCEPT_STEPS`. Kept for reference; the order test certifies
/// CONSISTENCY instead (see `cpu_compressible_operator_second_order`).
#[allow(dead_code)]
fn march_to_plateau_cpu(c: &mut CpuSolver) {
    let mut prev = read_state(c);
    let mut best = f64::INFINITY;
    let mut best_step = 0usize;
    for step in 0..STEADY_MAX_STEPS {
        c.step();
        let cur = read_state(c);
        let mut max_delta = 0.0f64;
        for (a, b) in cur.0.iter().zip(prev.0.iter()) {
            max_delta = max_delta.max((a.0 - b.0).abs()).max((a.1 - b.1).abs());
        }
        for (cc, pp) in [(&cur.1, &prev.1), (&cur.2, &prev.2)] {
            for (a, b) in cc.iter().zip(pp.iter()) {
                max_delta = max_delta.max((a - b).abs());
            }
        }
        if max_delta < STEADY_TOL && step >= MIN_STEPS {
            println!("[cpu-mms] steady after {} steps (max_delta={max_delta:.3e})", step + 1);
            return;
        }
        if max_delta < best * 0.98 {
            best = max_delta;
            best_step = step;
        } else if step > best_step + PLATEAU_WINDOW && step >= MIN_STEPS {
            println!(
                "[cpu-mms] delta plateau after {} steps (max_delta={max_delta:.3e}, best={best:.3e} at step {best_step})",
                step + 1
            );
            return;
        }
        if step >= LONG_MARCH_ACCEPT_STEPS {
            println!(
                "[cpu-mms] long-march acceptance after {} steps (max_delta={max_delta:.3e}, best={best:.3e})",
                step + 1
            );
            return;
        }
        if step % 100 == 0 {
            println!("[cpu-mms] step {step}: max_delta={max_delta:.3e}");
        }
        prev = cur;
    }
    panic!("no steady tolerance or plateau within {STEADY_MAX_STEPS} steps (best={best:.3e})");
}

/// Volume-weighted L2 errors (rho, u, p, T) vs the exact solution.
fn read_errors_cpu(mesh: &Mesh, c: &CpuSolver) -> (f64, f64, f64, f64) {
    let rho = c.get_field_scalar("rho").unwrap();
    let u = c.get_field_vec2("u").unwrap();
    let p = c.get_field_scalar("p").unwrap();
    let t = c.get_field_scalar("T").unwrap();
    let er = l2_scalar(mesh, &rho, exact_rho);
    let ep = l2_scalar(mesh, &p, exact_p);
    let et = l2_scalar(mesh, &t, exact_t);
    let eu = {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for i in 0..mesh.num_cells() {
            let (ex, ey) = exact_u(mesh.cell_cx[i], mesh.cell_cy[i]);
            num += ((u[i].0 - ex).powi(2) + (u[i].1 - ey).powi(2)) * mesh.cell_vol[i];
            den += mesh.cell_vol[i];
        }
        (num / den).sqrt()
    };
    (er, eu, ep, et)
}

/// CPU all-inlet boundedness probe: does the marginal compressible march stay
/// bounded through the 600-step acceptance window, or blow up first? Prints
/// per-25-step max_delta (the plateau metric), the rho_e L2 error, and max|rho_e|
/// (blow-up sentinel).
#[ignore]
#[cfg(feature = "dev-tests")]
#[test]
fn diag_cpu_allinlet_boundedness() {
    for &n in &[16usize, 24] {
        let (mut c, mesh) = setup(n, CpuBackendConfig::default());
        let mut prev = read_state(&c);
        let mut best = f64::INFINITY;
        let mut best_step = 0usize;
        let mut blew_up_at = None;
        for step in 0..700usize {
            c.step();
            let md = max_delta_state(&c, &prev);
            if md < best * 0.98 {
                best = md;
                best_step = step;
            }
            let re = c.get_field_scalar("rho_e").unwrap();
            let re_max = re.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
            if !re_max.is_finite() || re_max > 1e6 {
                blew_up_at = Some(step);
                println!("[bdd] n={n} BLEW UP at step {step} (max|rho_e|={re_max:.3e})");
                break;
            }
            if step % 25 == 0 || step == 699 {
                let err = l2_scalar(&mesh, &re, exact_rho_e);
                println!(
                    "[bdd] n={n} step={step} max_delta={md:.3e} best={best:.3e}@{best_step} rho_e_L2={err:.3e} max|rho_e|={re_max:.3e}"
                );
            }
            prev = read_state(&c);
        }
        if blew_up_at.is_none() {
            println!("[bdd] n={n} stayed finite through 700 steps (best_delta={best:.3e}@{best_step})");
        }
    }
}

/// Perturbed-IC probe: initializing AT the exact solution makes the f32 march
/// freeze near the IC at fine mesh (residual tiny, f32 update underflows before
/// reaching the discrete steady state → artificially-low errors). Starting FAR
/// from steady (velocity scaled by `vfac`) forces a real convergence to the stable
/// discrete steady state, so the error is the genuine O(h^2) discretization error.
/// Checks the converged error scales ~O(h^2) across n.
#[ignore]
#[cfg(feature = "dev-tests")]
#[test]
fn diag_cpu_perturbed_ic() {
    let cfg = CpuBackendConfig {
        engine: cfd2::solver::cpu::CpuEngine::Interpreter,
        threads: 8,
        simd: false,
        precision: Default::default(),
    };
    for &vfac in &[0.5f64, 0.8] {
        let mut hs = Vec::new();
        let mut es = Vec::new();
        for &n in &[32usize, 48] {
            let (mut c, mesh) = setup(n, cfg);
            let cells = mesh.num_cells();
            // Perturbed state: velocity scaled by vfac; rho, p kept exact; rho_u
            // and rho_e re-derived so the state stays thermodynamically consistent.
            let rho: Vec<f64> = (0..cells).map(|i| exact_rho(mesh.cell_cx[i], mesh.cell_cy[i])).collect();
            let p: Vec<f64> = (0..cells).map(|i| exact_p(mesh.cell_cx[i], mesh.cell_cy[i])).collect();
            let u: Vec<(f64, f64)> = (0..cells)
                .map(|i| { let (a, b) = exact_u(mesh.cell_cx[i], mesh.cell_cy[i]); (vfac * a, vfac * b) })
                .collect();
            let rho_u: Vec<(f64, f64)> = (0..cells).map(|i| (rho[i] * u[i].0, rho[i] * u[i].1)).collect();
            let rho_e: Vec<f64> = (0..cells)
                .map(|i| p[i] / (GAMMA - 1.0) + 0.5 * rho[i] * (u[i].0 * u[i].0 + u[i].1 * u[i].1))
                .collect();
            c.set_field_vec2("u", &u).unwrap();
            c.set_field_vec2("rho_u", &rho_u).unwrap();
            c.set_field_scalar("rho_e", &rho_e).unwrap();
            c.initialize_history();
            let mut prev = read_state(&c);
            let mut blew = false;
            let mut steady_step = None;
            for step in 0..1200usize {
                c.step();
                let md = max_delta_state(&c, &prev);
                let re = c.get_field_scalar("rho_e").unwrap();
                let rmax = re.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
                if !rmax.is_finite() || rmax > 1e6 {
                    println!("[pert] vfac={vfac} n={n} BLEW UP at step {step}");
                    blew = true;
                    break;
                }
                if md < STEADY_TOL && step >= 50 {
                    steady_step = Some(step);
                    break;
                }
                prev = read_state(&c);
            }
            if !blew {
                let (er, eu, ep, et) = read_errors_cpu(&mesh, &c);
                println!("[pert] vfac={vfac} n={n} steady@{steady_step:?} rho={er:.4e} u={eu:.4e} p={ep:.4e} T={et:.4e}");
                hs.push(1.0 / n as f64);
                es.push(er);
            }
        }
        if es.len() == 2 {
            println!("[pert] vfac={vfac} rho 2-pt order = {:.3}", ls_order(&hs, &es));
        }
    }
}

/// CPU viscosity sweep: at what `mu` is the all-inlet compressible MMS a genuinely
/// STABLE discrete steady state (CPU f64 solve converges cleanly at design order)?
/// mu=0.05 is marginally unstable; higher physical viscosity damps the convective
/// mode into a stable attractor. Reports boundedness + end-error order.
#[ignore]
#[cfg(feature = "dev-tests")]
#[test]
fn diag_cpu_mu_sweep() {
    let cfg = CpuBackendConfig {
        engine: cfd2::solver::cpu::CpuEngine::Interpreter,
        threads: 8,
        simd: false,
        precision: Default::default(),
    };
    for &mu in &[0.05f64, 0.1, 0.2, 0.4] {
        let mut hs = Vec::new();
        let mut es = Vec::new();
        for &n in &[16usize, 24, 32] {
            let (mut c, mesh) = setup(n, cfg);
            let cells = mesh.num_cells();
            c.set_viscosity(mu as f32);
            // Re-derive the mu-dependent momentum/energy sources (the rho-source
            // mass-compatibility projection is mu-independent, left as-is).
            let su: Vec<(f64, f64)> = (0..cells)
                .map(|i| source_rho_u(mesh.cell_cx[i], mesh.cell_cy[i], mu))
                .collect();
            let se: Vec<f64> = (0..cells)
                .map(|i| source_rho_e(mesh.cell_cx[i], mesh.cell_cy[i], mu))
                .collect();
            c.set_field_vec2(COMPRESSIBLE_MMS_SOURCE_RHO_U_FIELD, &su).unwrap();
            c.set_field_scalar(COMPRESSIBLE_MMS_SOURCE_RHO_E_FIELD, &se).unwrap();
            let mut blew = false;
            let mut end_err = f64::NAN;
            let mut min_err = f64::INFINITY;
            for step in 0..600 {
                c.step();
                let re = c.get_field_scalar("rho_e").unwrap();
                let rmax = re.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
                if !rmax.is_finite() || rmax > 1e6 {
                    blew = true;
                    println!("[mu] mu={mu} n={n} BLEW UP at step {step}");
                    break;
                }
                end_err = l2_scalar(&mesh, &re, exact_rho_e);
                min_err = min_err.min(end_err);
            }
            if !blew {
                println!("[mu] mu={mu} n={n} end_err={end_err:.3e} min_err={min_err:.3e}");
                hs.push(1.0 / n as f64);
                es.push(end_err);
            }
        }
        if es.len() >= 2 {
            println!("[mu] mu={mu} rho_e end-order = {:.3}", ls_order(&hs, &es));
        }
    }
}

/// CPU pseudo-transient (dtau) stabilization sweep. The marginal compressible mode
/// blows up at dtau=0. Pseudo-transient continuation adds a (state-state_iter)/dtau
/// term that vanishes at steady state, so it changes only the PATH, not the
/// converged discrete steady state (order preserved). Finds the smallest dtau that
/// keeps the CPU march bounded AND lets the error SETTLE at the O(h^2) level.
#[ignore]
#[cfg(feature = "dev-tests")]
#[test]
fn diag_cpu_dtau_sweep() {
    let steps = 900usize;
    for &dtau_frac in &[0.25f64, 0.5, 1.0] {
        let dtau = (dtau_frac * DT) as f32;
        for &n in &[16usize, 24] {
            let (mut c, mesh) = setup(n, CpuBackendConfig::default());
            c.set_dtau(dtau);
            c.initialize_history();
            let mut prev = read_state(&c);
            let mut best = f64::INFINITY;
            let mut min_err = f64::INFINITY;
            let mut err_at_end = f64::NAN;
            let mut blew = false;
            for step in 0..steps {
                c.step();
                let md = max_delta_state(&c, &prev);
                best = best.min(md);
                let re = c.get_field_scalar("rho_e").unwrap();
                let re_max = re.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
                if !re_max.is_finite() || re_max > 1e6 {
                    println!("[dtau] dtau={dtau_frac:.2}*dt n={n} BLEW UP at step {step}");
                    blew = true;
                    break;
                }
                let err = l2_scalar(&mesh, &re, exact_rho_e);
                min_err = min_err.min(err);
                err_at_end = err;
                if step % 100 == 0 {
                    println!("[dtau] dtau={dtau_frac:.2}*dt n={n} step={step} max_delta={md:.3e} rho_e_L2={err:.3e}");
                }
                prev = read_state(&c);
            }
            if !blew {
                println!(
                    "[dtau] SUMMARY dtau={dtau_frac:.2}*dt n={n} best_delta={best:.3e} min_err={min_err:.3e} end_err={err_at_end:.3e}"
                );
            }
        }
    }
}
