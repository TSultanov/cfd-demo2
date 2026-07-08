//! Shared host-f64 banded block linear solve for the structured (dense-Cartesian)
//! solver.
//!
//! The `TopologyMode::Structured2D` coupled U–p system is a matrix-free **banded
//! block** operator: per cell `s` unknowns and a 5-point stencil, stored as
//! `matrix_values[p*5*s*s + 5*s*r + band*s + c]` with bands `[S, W, diag, E, N]`.
//! This module owns the restarted GMRES over that operator plus its two
//! preconditioners:
//!
//! - **BlockJacobi** — invert each cell's `s×s` diagonal block (the robust
//!   default; the only stable choice for the singular-pressure-diagonal saddle
//!   point at rest).
//! - **Schur** — the model-owned SIMPLE block-lower-triangular preconditioner the
//!   *unstructured* path uses (`cpu::linalg::SchurPrecond`): a velocity-block
//!   diagonal-inverse predict, a fixed-sweep heavy-ball inner solve on the
//!   *already-assembled* pressure block `A_pp` (= `Ŝ`, a ready-made 5-point
//!   pressure-Poisson operator — no need to form `−B A⁻¹ Bᵀ`), and a velocity
//!   correction. Fixed inner sweeps keep the preconditioner a FIXED linear
//!   operator, so plain (non-flexible) GMRES stays valid.
//!
//! Both `StructuredModelSolver` (CPU) and `StructuredGpuSolver` (GPU host solve)
//! delegate here, so the two backends are bit-identical by construction.

const BAND_DIAG: usize = 2;

/// Preconditioner selection for the banded coupled solve.
#[derive(Clone, Debug)]
pub enum BandedPrecond {
    /// Per-cell `s×s` diagonal-block inverse.
    BlockJacobi,
    /// SIMPLE Schur complement. `u_idx`/`p` are the velocity-like and pressure
    /// unknown ranks within the per-cell block (coupled-system block indices, as
    /// the model's `SchurBlockLayout` declares them). `omega`==1.0 auto-selects
    /// the heavy-ball weight 1.95 (symmetric pressure block); other values are
    /// used verbatim. `sweeps_cap` caps the inner pressure sweeps.
    Schur { u_idx: Vec<usize>, p: usize, omega: f32, sweeps_cap: u32 },
}

/// `matrix_values[p*5*s*s + 5*s*r + band*s + c]` as f64.
#[inline]
fn block(a: &[f32], s: usize, p: usize, band: usize, r: usize, c: usize) -> f64 {
    a[p * 5 * s * s + 5 * s * r + band * s + c] as f64
}

/// The `(band, neighbour cell)` pairs present at cell `(i, j)` — the diagonal
/// plus whichever of S/W/E/N are in-grid. Edge bands are zero (closed via RHS).
#[inline]
fn neighbors(i: usize, j: usize, nx: usize, ny: usize) -> Stencil5 {
    let p = j * nx + i;
    let mut nb = Stencil5::new();
    nb.push((BAND_DIAG, p));
    if j > 0 {
        nb.push((0, p - nx)); // S
    }
    if i > 0 {
        nb.push((1, p - 1)); // W
    }
    if i + 1 < nx {
        nb.push((3, p + 1)); // E
    }
    if j + 1 < ny {
        nb.push((4, p + nx)); // N
    }
    nb
}

/// Tiny fixed-capacity (≤5) stack vector to avoid a heap alloc per cell in the
/// hot SpMV / preconditioner loops.
struct Stencil5 {
    data: [(usize, usize); 5],
    len: usize,
}
impl Stencil5 {
    #[inline]
    fn new() -> Self {
        Self { data: [(0, 0); 5], len: 0 }
    }
    #[inline]
    fn push(&mut self, v: (usize, usize)) {
        self.data[self.len] = v;
        self.len = self.len + 1;
    }
    #[inline]
    fn iter(&self) -> impl Iterator<Item = &(usize, usize)> {
        self.data[..self.len].iter()
    }
}

/// `y = A x` over the 5-point block stencil (public for tests that build a
/// known operator and need `b = A x*`).
pub fn spmv(a: &[f32], nx: usize, ny: usize, s: usize, x: &[f64]) -> Vec<f64> {
    let n = nx * ny;
    let mut y = vec![0.0f64; n * s];
    for j in 0..ny {
        for i in 0..nx {
            let p = j * nx + i;
            let nbrs = neighbors(i, j, nx, ny);
            for r in 0..s {
                let mut acc = 0.0;
                for &(band, q) in nbrs.iter() {
                    for c in 0..s {
                        acc += block(a, s, p, band, r, c) * x[q * s + c];
                    }
                }
                y[p * s + r] = acc;
            }
        }
    }
    y
}

/// Invert a general `s×s` matrix (row-major) via Gauss–Jordan with partial
/// pivoting; singular blocks fall back to the (pseudo-)diagonal inverse so the
/// preconditioner stays well-defined on weak saddle rows.
fn invert_block(m: &[f64], s: usize) -> Vec<f64> {
    let mut a = vec![0.0f64; s * 2 * s];
    for r in 0..s {
        for c in 0..s {
            a[r * 2 * s + c] = m[r * s + c];
        }
        a[r * 2 * s + s + r] = 1.0;
    }
    for col in 0..s {
        let mut piv = col;
        let mut best = a[col * 2 * s + col].abs();
        for r in (col + 1)..s {
            let v = a[r * 2 * s + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-30 {
            let mut out = vec![0.0f64; s * s];
            for k in 0..s {
                let d = m[k * s + k];
                out[k * s + k] = if d.abs() > 1e-30 { 1.0 / d } else { 0.0 };
            }
            return out;
        }
        if piv != col {
            for c in 0..(2 * s) {
                a.swap(col * 2 * s + c, piv * 2 * s + c);
            }
        }
        let d = a[col * 2 * s + col];
        for c in 0..(2 * s) {
            a[col * 2 * s + c] /= d;
        }
        for r in 0..s {
            if r == col {
                continue;
            }
            let f = a[r * 2 * s + col];
            if f != 0.0 {
                for c in 0..(2 * s) {
                    a[r * 2 * s + c] -= f * a[col * 2 * s + c];
                }
            }
        }
    }
    let mut out = vec![0.0f64; s * s];
    for r in 0..s {
        for c in 0..s {
            out[r * s + c] = a[r * 2 * s + s + c];
        }
    }
    out
}

/// A built preconditioner ready to apply to a residual vector.
enum Built {
    BlockJacobi { minv: Vec<Vec<f64>> },
    Schur(SchurData),
}

struct SchurData {
    u_idx: Vec<usize>,
    p: usize,
    u_len: usize,
    /// `1 / A[u_i, u_i]` per (cell, velocity-component).  `ncells * u_len`.
    diag_u_inv: Vec<f64>,
    /// Assembled pressure–pressure 5-band scalar operator.  `ncells * 5`.
    a_pp: Vec<f64>,
    /// `1 / A[p, p]` per cell.  `ncells`.
    p_diag_inv: Vec<f64>,
    omega: f64,
    sweeps: usize,
}

/// The inner heavy-ball sweep count: `min(20 + √n/8, sweeps_cap)` (mirrors
/// `gpu::modules::coupled_schur::default_pressure_sweeps`).
fn default_pressure_sweeps(num_cells: usize, sweeps_cap: u32) -> usize {
    std::env::var("CFD2_STRUCT_SCHUR_SWEEPS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| {
            (20 + (num_cells as f32).sqrt() as usize / 8).min(sweeps_cap.max(1) as usize)
        })
}

/// The heavy-ball relaxation weight (1.0 → auto 1.95; mirrors
/// `gpu::modules::coupled_schur::heavy_ball_omega`).
fn heavy_ball_omega(model_omega: f32) -> f64 {
    let base = if model_omega == 1.0 { 1.95 } else { model_omega };
    std::env::var("CFD2_STRUCT_SCHUR_OMEGA")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(base) as f64
}

fn build(a: &[f32], nx: usize, ny: usize, s: usize, prec: &BandedPrecond) -> Built {
    let ncells = nx * ny;
    match prec {
        BandedPrecond::BlockJacobi => {
            let mut minv = vec![Vec::new(); ncells];
            for p in 0..ncells {
                let mut m = vec![0.0f64; s * s];
                for r in 0..s {
                    for c in 0..s {
                        m[r * s + c] = block(a, s, p, BAND_DIAG, r, c);
                    }
                }
                minv[p] = invert_block(&m, s);
            }
            Built::BlockJacobi { minv }
        }
        BandedPrecond::Schur { u_idx, p: pp, omega, sweeps_cap } => {
            let u_len = u_idx.len();
            let mut diag_u_inv = vec![0.0f64; ncells * u_len];
            let mut a_pp = vec![0.0f64; ncells * 5];
            let mut p_diag_inv = vec![0.0f64; ncells];
            for cell in 0..ncells {
                for (i, &u) in u_idx.iter().enumerate() {
                    let d = block(a, s, cell, BAND_DIAG, u, u);
                    diag_u_inv[cell * u_len + i] = if d.abs() > 1e-30 { 1.0 / d } else { 0.0 };
                }
                for band in 0..5 {
                    a_pp[cell * 5 + band] = block(a, s, cell, band, *pp, *pp);
                }
                let dp = a_pp[cell * 5 + BAND_DIAG];
                p_diag_inv[cell] = if dp.abs() > 1e-30 { 1.0 / dp } else { 0.0 };
            }
            Built::Schur(SchurData {
                u_idx: u_idx.clone(),
                p: *pp,
                u_len,
                diag_u_inv,
                a_pp,
                p_diag_inv,
                omega: heavy_ball_omega(*omega),
                sweeps: default_pressure_sweeps(ncells, *sweeps_cap),
            })
        }
    }
}

/// `A_pp · x` (scalar 5-band pressure operator).
fn a_pp_spmv(sd: &SchurData, nx: usize, ny: usize, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0f64; nx * ny];
    for j in 0..ny {
        for i in 0..nx {
            let cell = j * nx + i;
            let mut acc = 0.0;
            for &(band, q) in neighbors(i, j, nx, ny).iter() {
                acc += sd.a_pp[cell * 5 + band] * x[q];
            }
            y[cell] = acc;
        }
    }
    y
}

/// Second-order Richardson (heavy ball) on `A_pp x = g` from `x_0 = 0`:
/// `x_{k+1} = (1-ω) x_{k-1} + ω (x_k + D⁻¹(g − A_pp x_k))`, with the same
/// stability safeguard the unstructured `cpu::linalg::heavy_ball_solve` uses
/// (f64 path): the heavy-ball ellipse collapses as ω→2, so on SUSTAINED residual
/// growth (two checks with no new best) halve the momentum and restart from the
/// best iterate; after two decays bail with the best. Returns the best iterate.
/// This safeguard makes the apply DATA-DEPENDENT, so the outer solve must be
/// flexible (FGMRES) — which is why Schur uses [`banded_fgmres`].
fn heavy_ball_pressure(sd: &SchurData, nx: usize, ny: usize, g: &[f64]) -> Vec<f64> {
    let ncells = nx * ny;
    let gnorm = norm(g).max(1e-300);
    let mut cur = vec![0.0f64; ncells]; // x_k
    let mut prev = vec![0.0f64; ncells]; // x_{k-1}  (also receives x_{k+1})
    let mut best = vec![0.0f64; ncells];
    let mut best_rel = 1.0f64; // x = 0 has relative residual exactly 1
    let mut omega = sd.omega;
    let inner_tol = 1e-3; // loose: this is a preconditioner, FGMRES is flexible
    let (mut strikes, mut decays) = (0u32, 0u32);
    let mut next_check = 4usize.min(sd.sweeps.max(1));
    let mut swept = 0usize;
    while swept < sd.sweeps {
        let (om, one_m_om) = (omega, 1.0 - omega);
        let ax = a_pp_spmv(sd, nx, ny, &cur);
        for cell in 0..ncells {
            let hat = cur[cell] + (g[cell] - ax[cell]) * sd.p_diag_inv[cell];
            prev[cell] = one_m_om * prev[cell] + om * hat;
        }
        std::mem::swap(&mut cur, &mut prev);
        swept += 1;
        if swept == next_check || swept == sd.sweeps {
            let ax = a_pp_spmv(sd, nx, ny, &cur);
            let mut rr = 0.0f64;
            for cell in 0..ncells {
                let d = g[cell] - ax[cell];
                rr += d * d;
            }
            let rel = rr.sqrt() / gnorm;
            if rel <= inner_tol {
                best.copy_from_slice(&cur);
                break;
            }
            if rel < best_rel {
                best_rel = rel;
                best.copy_from_slice(&cur);
                strikes = 0;
            } else {
                // Heavy-ball residuals overshoot transiently; only SUSTAINED
                // growth means momentum is amplifying part of the spectrum.
                strikes += 1;
                if strikes >= 2 {
                    strikes = 0;
                    decays += 1;
                    if decays >= 2 {
                        break;
                    }
                    omega = 1.0 + (omega - 1.0) * 0.5;
                    cur.copy_from_slice(&best);
                    prev.copy_from_slice(&best);
                }
            }
            next_check = (next_check * 2).min(sd.sweeps.max(1));
        }
    }
    best
}

fn apply(built: &Built, a: &[f32], nx: usize, ny: usize, s: usize, r: &[f64]) -> Vec<f64> {
    match built {
        Built::BlockJacobi { minv } => {
            let ncells = nx * ny;
            let mut z = vec![0.0f64; ncells * s];
            for p in 0..ncells {
                for i in 0..s {
                    let mut acc = 0.0;
                    for k in 0..s {
                        acc += minv[p][i * s + k] * r[p * s + k];
                    }
                    z[p * s + i] = acc;
                }
            }
            z
        }
        Built::Schur(sd) => {
            let ncells = nx * ny;
            let (u_idx, p, u_len) = (&sd.u_idx, sd.p, sd.u_len);
            // Identity on any unknown that is neither velocity-like nor pressure.
            let mut z = r.to_vec();

            // 1./2. velocity predict + Schur RHS g_p = r_p − A_pu diag(A_uu)⁻¹ r_u.
            let mut gp = vec![0.0f64; ncells];
            for j in 0..ny {
                for i in 0..nx {
                    let cell = j * nx + i;
                    for (ii, &u) in u_idx.iter().enumerate() {
                        z[cell * s + u] = sd.diag_u_inv[cell * u_len + ii] * r[cell * s + u];
                    }
                    z[cell * s + p] = 0.0;
                    let mut g = r[cell * s + p];
                    for &(band, q) in neighbors(i, j, nx, ny).iter() {
                        for (ii, &u) in u_idx.iter().enumerate() {
                            let a_pu = block(a, s, cell, band, p, u);
                            g -= a_pu * sd.diag_u_inv[q * u_len + ii] * r[q * s + u];
                        }
                    }
                    gp[cell] = g;
                }
            }

            // 3. inner pressure solve  Ŝ psol = g_p   (Ŝ = A_pp).
            let psol = heavy_ball_pressure(sd, nx, ny, &gp);

            // 4. velocity correct  z_u -= diag(A_uu)⁻¹ A_up psol ;  z_p = psol.
            for j in 0..ny {
                for i in 0..nx {
                    let cell = j * nx + i;
                    for (ii, &u) in u_idx.iter().enumerate() {
                        let mut corr = 0.0;
                        for &(band, q) in neighbors(i, j, nx, ny).iter() {
                            corr += block(a, s, cell, band, u, p) * psol[q];
                        }
                        z[cell * s + u] -= sd.diag_u_inv[cell * u_len + ii] * corr;
                    }
                    z[cell * s + p] = psol[cell];
                }
            }
            z
        }
    }
}

// Small dense-vector helpers (f64).
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}
#[inline]
fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}
#[inline]
fn scale(a: &[f64], s: f64) -> Vec<f64> {
    a.iter().map(|&x| x * s).collect()
}
#[inline]
fn axpy(y: &mut [f64], a: f64, x: &[f64]) {
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

/// Restarted GMRES on the banded block operator. Block-Jacobi (a fixed linear
/// operator) runs plain LEFT-preconditioned GMRES; the SIMPLE Schur (whose inner
/// heavy-ball safeguard makes it data-dependent) runs flexible RIGHT-
/// preconditioned GMRES ([`banded_fgmres`]). Returns `(x_f32, relative residual)`.
pub fn banded_gmres(
    a: &[f32],
    nx: usize,
    ny: usize,
    s: usize,
    b: &[f32],
    precond: &BandedPrecond,
    restart: usize,
    max_outer: usize,
    tol: f64,
) -> (Vec<f32>, f64) {
    let built = build(a, nx, ny, s, precond);
    if matches!(built, Built::Schur(_)) {
        return banded_fgmres(a, nx, ny, s, b, &built, restart, max_outer, tol);
    }
    let n = nx * ny * s;
    let b64: Vec<f64> = b.iter().map(|&v| v as f64).collect();
    let bnorm = norm(&b64).max(1e-30);
    let mut x = vec![0.0f64; n];

    for _outer in 0..max_outer {
        // r0 = M^{-1}(b - A x)
        let ax = spmv(a, nx, ny, s, &x);
        let r0: Vec<f64> = (0..n).map(|i| b64[i] - ax[i]).collect();
        let r = apply(&built, a, nx, ny, s, &r0);
        let beta = norm(&r);
        if beta / bnorm <= tol {
            return (x.iter().map(|&v| v as f32).collect(), beta / bnorm);
        }

        let m = restart;
        let mut v: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
        v.push(scale(&r, 1.0 / beta));
        let mut h = vec![vec![0.0f64; m]; m + 1];
        let mut g = vec![0.0f64; m + 1];
        g[0] = beta;
        let mut cs = vec![0.0f64; m];
        let mut sn = vec![0.0f64; m];
        let mut k_used = 0;

        for k in 0..m {
            // w = M^{-1} A v_k
            let av = spmv(a, nx, ny, s, &v[k]);
            let mut w = apply(&built, a, nx, ny, s, &av);
            for i in 0..=k {
                h[i][k] = dot(&w, &v[i]);
                axpy(&mut w, -h[i][k], &v[i]);
            }
            h[k + 1][k] = norm(&w);
            if h[k + 1][k] > 1e-14 {
                v.push(scale(&w, 1.0 / h[k + 1][k]));
            } else {
                v.push(vec![0.0f64; n]);
            }
            for i in 0..k {
                let temp = cs[i] * h[i][k] + sn[i] * h[i + 1][k];
                h[i + 1][k] = -sn[i] * h[i][k] + cs[i] * h[i + 1][k];
                h[i][k] = temp;
            }
            let denom = (h[k][k] * h[k][k] + h[k + 1][k] * h[k + 1][k]).sqrt();
            if denom < 1e-300 {
                k_used = k;
                break;
            }
            cs[k] = h[k][k] / denom;
            sn[k] = h[k + 1][k] / denom;
            h[k][k] = cs[k] * h[k][k] + sn[k] * h[k + 1][k];
            h[k + 1][k] = 0.0;
            g[k + 1] = -sn[k] * g[k];
            g[k] = cs[k] * g[k];
            k_used = k + 1;
            if g[k + 1].abs() / bnorm <= tol {
                break;
            }
        }

        let kk = k_used;
        let mut y = vec![0.0f64; kk];
        for i in (0..kk).rev() {
            let mut sm = g[i];
            for jj in (i + 1)..kk {
                sm -= h[i][jj] * y[jj];
            }
            y[i] = if h[i][i].abs() > 1e-300 { sm / h[i][i] } else { 0.0 };
        }
        for i in 0..kk {
            axpy(&mut x, y[i], &v[i]);
        }

        let ax = spmv(a, nx, ny, s, &x);
        let res: Vec<f64> = (0..n).map(|i| b64[i] - ax[i]).collect();
        if norm(&res) / bnorm <= tol {
            return (x.iter().map(|&v| v as f32).collect(), norm(&res) / bnorm);
        }
    }
    let ax = spmv(a, nx, ny, s, &x);
    let res: Vec<f64> = (0..n).map(|i| b64[i] - ax[i]).collect();
    (x.iter().map(|&v| v as f32).collect(), norm(&res) / bnorm)
}

/// Restarted, RIGHT-preconditioned FLEXIBLE GMRES — FGMRES(`restart`) — for the
/// Schur preconditioner, whose inner heavy-ball safeguard makes `M⁻¹` vary per
/// apply. Stores the preconditioned Krylov vectors `z_k = M⁻¹ v_k` and updates
/// `x = x0 + Z y`, so a data-dependent preconditioner does not corrupt the
/// Arnoldi recurrence (unlike plain GMRES). Mirrors `cpu::linalg::fgmres`.
fn banded_fgmres(
    a: &[f32],
    nx: usize,
    ny: usize,
    s: usize,
    b: &[f32],
    built: &Built,
    restart: usize,
    max_outer: usize,
    tol: f64,
) -> (Vec<f32>, f64) {
    let n = nx * ny * s;
    let b64: Vec<f64> = b.iter().map(|&v| v as f64).collect();
    let bnorm = norm(&b64).max(1e-30);
    let mut x = vec![0.0f64; n];

    for _outer in 0..max_outer {
        // Right-preconditioned: the Arnoldi space is built on the UNpreconditioned
        // residual r0 = b - A x.
        let ax = spmv(a, nx, ny, s, &x);
        let r0: Vec<f64> = (0..n).map(|i| b64[i] - ax[i]).collect();
        let beta = norm(&r0);
        if beta / bnorm <= tol {
            return (x.iter().map(|&v| v as f32).collect(), beta / bnorm);
        }

        let m = restart;
        let mut v: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
        v.push(scale(&r0, 1.0 / beta));
        let mut z: Vec<Vec<f64>> = Vec::with_capacity(m); // preconditioned basis
        let mut h = vec![vec![0.0f64; m]; m + 1];
        let mut g = vec![0.0f64; m + 1];
        g[0] = beta;
        let mut cs = vec![0.0f64; m];
        let mut sn = vec![0.0f64; m];
        let mut k_used = 0;

        for k in 0..m {
            // z_k = M^{-1} v_k ; w = A z_k.
            let zk = apply(built, a, nx, ny, s, &v[k]);
            let mut w = spmv(a, nx, ny, s, &zk);
            z.push(zk);
            for i in 0..=k {
                h[i][k] = dot(&w, &v[i]);
                axpy(&mut w, -h[i][k], &v[i]);
            }
            h[k + 1][k] = norm(&w);
            if h[k + 1][k] > 1e-14 {
                v.push(scale(&w, 1.0 / h[k + 1][k]));
            } else {
                v.push(vec![0.0f64; n]);
            }
            for i in 0..k {
                let temp = cs[i] * h[i][k] + sn[i] * h[i + 1][k];
                h[i + 1][k] = -sn[i] * h[i][k] + cs[i] * h[i + 1][k];
                h[i][k] = temp;
            }
            let denom = (h[k][k] * h[k][k] + h[k + 1][k] * h[k + 1][k]).sqrt();
            if denom < 1e-300 {
                k_used = k;
                break;
            }
            cs[k] = h[k][k] / denom;
            sn[k] = h[k + 1][k] / denom;
            h[k][k] = cs[k] * h[k][k] + sn[k] * h[k + 1][k];
            h[k + 1][k] = 0.0;
            g[k + 1] = -sn[k] * g[k];
            g[k] = cs[k] * g[k];
            k_used = k + 1;
            if g[k + 1].abs() / bnorm <= tol {
                break;
            }
        }

        // Back-substitute for y, update x = x + Z y (preconditioned basis).
        let kk = k_used;
        let mut y = vec![0.0f64; kk];
        for i in (0..kk).rev() {
            let mut sm = g[i];
            for jj in (i + 1)..kk {
                sm -= h[i][jj] * y[jj];
            }
            y[i] = if h[i][i].abs() > 1e-300 { sm / h[i][i] } else { 0.0 };
        }
        for i in 0..kk {
            axpy(&mut x, y[i], &z[i]);
        }

        let ax = spmv(a, nx, ny, s, &x);
        let res: Vec<f64> = (0..n).map(|i| b64[i] - ax[i]).collect();
        if norm(&res) / bnorm <= tol {
            return (x.iter().map(|&v| v as f32).collect(), norm(&res) / bnorm);
        }
    }
    let ax = spmv(a, nx, ny, s, &x);
    let res: Vec<f64> = (0..n).map(|i| b64[i] - ax[i]).collect();
    (x.iter().map(|&v| v as f32).collect(), norm(&res) / bnorm)
}

/// Read a model's declared Schur block layout, if any, into a [`BandedPrecond`].
/// Returns `None` when the model declares no Schur preconditioner (e.g. the
/// density-based compressible model) — the caller then uses block-Jacobi.
pub fn schur_from_model(model: &crate::solver::model::ModelSpec) -> Option<BandedPrecond> {
    model.linear_solver.and_then(|ls| match ls.preconditioner {
        crate::solver::model::ModelPreconditionerSpec::Schur { omega, sweeps_cap, layout } => {
            Some(BandedPrecond::Schur {
                u_idx: layout.u_indices().iter().map(|&u| u as usize).collect(),
                p: layout.p as usize,
                omega,
                sweeps_cap,
            })
        }
        _ => None,
    })
}
