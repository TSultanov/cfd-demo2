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
    /// used verbatim. `sweeps_cap` caps the inner pressure sweeps. `pressure_amg`
    /// selects the inner `A_pp` (pressure-Poisson) solve: `false` = the
    /// safeguarded heavy-ball smoother, `true` = an algebraic-multigrid V-cycle
    /// (the SAME `cpu::amg` the unstructured Schur uses; requires the `cpu`
    /// feature, else falls back to heavy-ball) — the structured analogue of the
    /// unstructured "Chebyshev vs AMG pressure solve" selector.
    Schur { u_idx: Vec<usize>, p: usize, omega: f32, sweeps_cap: u32, pressure_amg: bool },
}

/// The coupled-solve preconditioner choice the structured solvers expose (drives
/// the GUI radio + `set_preconditioner`). Mirrors the unstructured menu: block-
/// Jacobi, the model-owned SIMPLE Schur, or Schur with an AMG pressure solve.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CoupledPrecondKind {
    BlockJacobi,
    Schur,
    SchurAmg,
}

// ---- Threading helpers ------------------------------------------------------
// The banded solve is the structured coupled bottleneck. Its per-cell passes
// (SpMV, preconditioner apply) and the Arnoldi dots parallelize over DISJOINT
// cell chunks. cfg-gated: the CPU backend has the persistent worker pool
// (`cpu::parallel`); the pure-GPU build (no `cpu` feature) runs serial (the GPU
// host solve is a small read-back system anyway).

/// Fill `y` (`ncells * width`, cell-major) in parallel: `f(cell0, y_chunk)` owns
/// the disjoint slice for cells `[cell0, cell0 + y_chunk.len()/width)`.
#[cfg(feature = "cpu")]
#[inline]
fn cells_mut<F: Fn(usize, &mut [f64]) + Sync>(
    ncells: usize,
    width: usize,
    threads: usize,
    y: &mut [f64],
    f: F,
) {
    crate::solver::cpu::parallel::parallel_cell_chunks_mut(ncells, width, threads, y, f);
}
#[cfg(not(feature = "cpu"))]
#[inline]
fn cells_mut<F: Fn(usize, &mut [f64])>(_n: usize, _w: usize, _t: usize, y: &mut [f64], f: F) {
    f(0, y);
}

/// Parallel dot product (deterministic per fixed 8192-chunk).
#[cfg(feature = "cpu")]
#[inline]
fn pdot(threads: usize, a: &[f64], b: &[f64]) -> f64 {
    crate::solver::cpu::parallel::par_dot(threads, a, b)
}
#[cfg(not(feature = "cpu"))]
#[inline]
fn pdot(_threads: usize, a: &[f64], b: &[f64]) -> f64 {
    dot(a, b)
}
#[inline]
fn pnorm(threads: usize, a: &[f64]) -> f64 {
    pdot(threads, a, a).sqrt()
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
/// known operator and need `b = A x*`; serial).
pub fn spmv(a: &[f32], nx: usize, ny: usize, s: usize, x: &[f64]) -> Vec<f64> {
    spmv_t(a, nx, ny, s, x, 1)
}

/// `y = A x`, parallel over disjoint cell chunks (`threads`).
fn spmv_t(a: &[f32], nx: usize, ny: usize, s: usize, x: &[f64], threads: usize) -> Vec<f64> {
    let n = nx * ny;
    let mut y = vec![0.0f64; n * s];
    cells_mut(n, s, threads, &mut y, |cell0, chunk| {
        for li in 0..chunk.len() / s {
            let p = cell0 + li;
            let (i, j) = (p % nx, p / nx);
            for r in 0..s {
                let mut acc = 0.0;
                for &(band, q) in neighbors(i, j, nx, ny).iter() {
                    for c in 0..s {
                        acc += block(a, s, p, band, r, c) * x[q * s + c];
                    }
                }
                chunk[li * s + r] = acc;
            }
        }
    });
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
    /// Whether the caller requested the AMG V-cycle pressure solve (else the
    /// safeguarded heavy-ball). The AMG *hierarchy* is now cached across solves in
    /// a [`StructuredAmgCache`]; only the current `A_pp` Galerkin values are
    /// re-assembled per solve inside `banded_fgmres`.
    pressure_amg: bool,
}

/// Cross-solve cache for the structured Schur AMG pressure hierarchy. The
/// hierarchy (aggregation + Galerkin scatter maps) is SPARSITY-only — a fixed
/// 5-point stencil on the fixed `nx×ny` grid — so it is built ONCE from a
/// canonical Poisson seed (independent of any solve's values) and reused across
/// every solve and step; only the Galerkin VALUES are re-assembled each solve
/// from the current `A_pp`. Structured mirror of `CpuSolver::amg_hier`.
/// Empty / no-op without the `cpu` feature.
#[derive(Default)]
pub struct StructuredAmgCache {
    // `OnceLock` (not `cell::OnceCell`) so the enclosing solvers stay `Sync` — the
    // GPU `BandedGpuLinAlg` is held across threads. Same `get_or_init` semantics.
    #[cfg(feature = "cpu")]
    hier: std::sync::OnceLock<crate::solver::cpu::amg::AmgHierarchy>,
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

/// Relative-residual tolerance for the coupled banded inner solve inside the
/// Picard/outer loop. INEXACT-PICARD: the outer loop re-linearizes every sweep,
/// so driving each linearization past ~1e-4 of its initial residual is wasted
/// work — this matches the unstructured model's declared 1e-4 linear tolerance
/// (`ModelLinearSolverSettings::default`) and cuts the inner iteration count
/// several-fold vs. an exact 1e-9 solve. `CFD2_STRUCT_LINEAR_TOL` overrides (e.g.
/// tighten for an MMS order study). Both the CPU and GPU host coupled solves use
/// it, so they stay bit-identical.
pub fn default_step_tol() -> f64 {
    std::env::var("CFD2_STRUCT_LINEAR_TOL")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|t| *t > 0.0)
        .unwrap_or(1e-4)
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
        BandedPrecond::Schur { u_idx, p: pp, omega, sweeps_cap, pressure_amg } => {
            let u_len = u_idx.len();
            let mut diag_u_inv = vec![0.0f64; ncells * u_len];
            let mut a_pp = vec![0.0f64; ncells * 5];
            let mut p_diag_inv = vec![0.0f64; ncells];
            for cell in 0..ncells {
                for (i, &u) in u_idx.iter().enumerate() {
                    let d = block(a, s, cell, BAND_DIAG, u, u);
                    // Guard on a POSITIVE diagonal, not |d|. The momentum diagonal
                    // `ddt + diffusion + convection − bounded_sum_phi` (OpenFOAM
                    // bounded-Gauss net-outflux correction) can transiently go
                    // NEGATIVE during a through-flow startup (net outward flux
                    // exceeds ddt+diffusion). `1/d` with `d<0` is a wrong-sign,
                    // amplifying preconditioner that stalls the momentum solve
                    // (rel_res≈1). Zeroing it there leaves those few cells
                    // unpreconditioned (safe) instead of anti-preconditioned.
                    // For the well-posed `d>0` case this is bit-identical to before.
                    diag_u_inv[cell * u_len + i] = if d > 1e-30 { 1.0 / d } else { 0.0 };
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
                pressure_amg: *pressure_amg,
            })
        }
    }
}

/// Build the AMG aggregation hierarchy over the structured pressure Poisson's
/// FIXED 5-point sparsity, seeded with a CANONICAL Poisson value set (diagonal =
/// in-grid-neighbour count, off-diagonals = −1) rather than any solve's actual
/// `A_pp`.
///
/// Anti-hang rationale: seeding the hierarchy from a real `A_pp` deadlocks the
/// solve when that matrix is the atypical FIRST outer iteration (`A_pp` ~all-zero
/// before `d_p` is populated) — a zero diagonal makes strength-of-connection
/// reject every edge, aggregation yields `nc==n` singletons, `AmgHierarchy::build`
/// bails with ZERO coarsening levels, and the "coarsest = finest" level is
/// dense-inverted (O(n³)) then V-cycled to no effect every apply, so FGMRES never
/// contracts. The canonical seed is guaranteed strongly connected + non-singular,
/// so it ALWAYS coarsens; on the uniform structured grid it is the exact discrete
/// Laplacian, so the aggregation equals what the real values would produce. The
/// hierarchy is value-INDEPENDENT (aggregation + Galerkin scatter maps only); the
/// real `A_pp` values enter per solve via [`amg_csr_values`] + `AmgSolver::assemble`.
#[cfg(feature = "cpu")]
fn build_pressure_amg_hierarchy(nx: usize, ny: usize) -> crate::solver::cpu::amg::AmgHierarchy {
    let ncells = nx * ny;
    let mut row_offsets = Vec::with_capacity(ncells + 1);
    let mut col_indices: Vec<u32> = Vec::with_capacity(ncells * 5);
    let mut values: Vec<f32> = Vec::with_capacity(ncells * 5);
    row_offsets.push(0u32);
    for j in 0..ny {
        for i in 0..nx {
            let nb = neighbors(i, j, nx, ny);
            let ndeg = nb.len - 1; // in-grid neighbours (the diagonal is push #0)
            for &(band, q) in nb.iter() {
                col_indices.push(q as u32);
                values.push(if band == BAND_DIAG { ndeg as f32 } else { -1.0 });
            }
            row_offsets.push(col_indices.len() as u32);
        }
    }
    crate::solver::cpu::amg::AmgHierarchy::build(&row_offsets, &col_indices, &values)
}

/// Finest-level CSR VALUES of the current `A_pp`, in the exact per-row order
/// [`build_pressure_amg_hierarchy`] laid out the pattern (`neighbors()`: diagonal
/// first, then in-grid S/W/E/N), so they line up index-for-index with the cached
/// hierarchy. Recomputed every solve; `AmgSolver::assemble` Galerkin-coarsens
/// them through the cached scatter maps, so the V-cycle operator tracks the
/// current `A_pp` exactly (only the aggregation PATTERN is frozen).
#[cfg(feature = "cpu")]
fn amg_csr_values(a_pp: &[f64], nx: usize, ny: usize) -> Vec<f32> {
    let ncells = nx * ny;
    let mut values: Vec<f32> = Vec::with_capacity(ncells * 5);
    for j in 0..ny {
        for i in 0..nx {
            let cell = j * nx + i;
            // Floor each row to a symmetric M-matrix (off-diagonals ≤ 0, diagonal ≥
            // Σ|off-diag|) before handing it to the AMG. A from-rest through-flow
            // startup transiently drives the Rhie–Chow d_p NEGATIVE, flipping the
            // pressure-Laplacian off-diagonals POSITIVE → A_pp indefinite → the
            // SPD-assuming AMG V-cycle amplifies and FGMRES stalls (SchurAmg
            // diverged from rest). The AMG is only a PRECONDITIONER — FGMRES builds
            // its Krylov space on the TRUE residual (b − A x), so an M-matrix
            // surrogate here just changes the preconditioner quality, never the
            // converged solution. On the DEVELOPED (already-SPD, diagonally-dominant
            // Laplacian) A_pp every clamp is inert → bit-identical to the raw values.
            let nb = neighbors(i, j, nx, ny);
            let mut row: [f32; 5] = [0.0; 5];
            let mut off_abs_sum = 0.0f64;
            let mut len = 0usize;
            for &(band, _q) in nb.iter() {
                if band == BAND_DIAG {
                    row[len] = 0.0; // diagonal is push #0 — filled in after the sum
                } else {
                    let v = a_pp[cell * 5 + band].min(0.0); // off-diag must be ≤ 0
                    off_abs_sum += -v;
                    row[len] = v as f32;
                }
                len += 1;
            }
            row[0] = a_pp[cell * 5 + BAND_DIAG].max(off_abs_sum + 1e-12) as f32;
            values.extend_from_slice(&row[..len]);
        }
    }
    values
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

/// `pre_pressure`, when supplied, is the inner `A_pp` pressure solve prepared ONCE
/// per outer solve (an AMG V-cycle over a hierarchy assembled a single time) —
/// the Schur branch calls it instead of re-assembling / heavy-ball per apply.
#[allow(clippy::type_complexity)]
fn apply(
    built: &Built,
    a: &[f32],
    nx: usize,
    ny: usize,
    s: usize,
    r: &[f64],
    threads: usize,
    pre_pressure: Option<&dyn Fn(&[f64]) -> Vec<f64>>,
) -> Vec<f64> {
    match built {
        Built::BlockJacobi { minv } => {
            let ncells = nx * ny;
            let mut z = vec![0.0f64; ncells * s];
            cells_mut(ncells, s, threads, &mut z, |cell0, chunk| {
                for li in 0..chunk.len() / s {
                    let p = cell0 + li;
                    for i in 0..s {
                        let mut acc = 0.0;
                        for k in 0..s {
                            acc += minv[p][i * s + k] * r[p * s + k];
                        }
                        chunk[li * s + i] = acc;
                    }
                }
            });
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

            // 3. inner pressure solve  Ŝ psol = g_p   (Ŝ = A_pp): the pre-built
            //    AMG V-cycle if supplied (assembled ONCE per outer solve), else
            //    the safeguarded heavy-ball.
            let psol = if let Some(psolve) = pre_pressure {
                psolve(&gp)
            } else {
                heavy_ball_pressure(sd, nx, ny, &gp)
            };

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

/// Restarted GMRES on the banded block operator. BOTH preconditioners run
/// through the RIGHT-preconditioned flexible solve ([`banded_fgmres`]): a fixed
/// linear operator (block-Jacobi) is a valid FGMRES preconditioner, and right
/// preconditioning keeps every convergence decision on the TRUE residual
/// `||b − A x||` — the old LEFT-preconditioned block-Jacobi path decided on
/// `||M⁻¹(b − A x)|| / ||b||`, a mixed norm that is not scale-invariant (block
/// diagonals ≫ 1, e.g. small-dt `vol/dt`, produced FALSE convergence; diagonals
/// ≪ 1, e.g. all-Mach `psi`~1e-5 pressure rows, burned the whole budget).
/// Returns `(x_f32, relative residual)`.
#[allow(clippy::too_many_arguments)]
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
    let (x, res, _iters) = banded_gmres_t(a, nx, ny, s, b, precond, restart, max_outer, tol, 1, None);
    (x, res)
}

/// Threaded variant: `threads` parallelizes the per-cell SpMV / preconditioner
/// apply and the Arnoldi dots over disjoint cell chunks. `amg_cache`, when
/// supplied, holds the cross-solve Schur AMG hierarchy (built once, reused every
/// solve); pass `None` for the non-AMG path or a self-contained one-off solve.
#[allow(clippy::too_many_arguments)]
pub fn banded_gmres_t(
    a: &[f32],
    nx: usize,
    ny: usize,
    s: usize,
    b: &[f32],
    precond: &BandedPrecond,
    restart: usize,
    max_outer: usize,
    tol: f64,
    threads: usize,
    amg_cache: Option<&StructuredAmgCache>,
) -> (Vec<f32>, f64, u32) {
    let built = build(a, nx, ny, s, precond);
    banded_fgmres(a, nx, ny, s, b, &built, restart, max_outer, tol, threads, amg_cache)
}

/// TOTAL Arnoldi-iteration budget for one banded coupled solve. The budget used
/// to be `max_outer` restart CYCLES (200 x 60 = 12,000 iterations): a
/// pathological solve (non-finite operator, indefinite startup transient)
/// stalled for minutes before returning. The unstructured fgmres budget is 200
/// TOTAL iterations; the structured solve starts cold (x0 = 0, no warm start)
/// and its per-block exit is stricter, so it gets several restart cycles more.
/// `CFD2_STRUCT_LINEAR_MAXIT` overrides.
fn max_total_iters(restart: usize, max_outer: usize) -> u32 {
    std::env::var("CFD2_STRUCT_LINEAR_MAXIT")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&v| v > 0)
        .unwrap_or_else(|| (restart.saturating_mul(max_outer)).min(600) as u32)
        .max(1)
}

/// Per-unknown-rank residual scales: `scale[c] = ||b_c||` over cells (the
/// unstructured convention is `min(||b||, ||r0||)`; the structured solve starts
/// from `x0 = 0`, so `r0 == b` and the min is `||b||`). A rank whose RHS is
/// identically zero falls back to the AGGREGATE `bnorm`, so cross-coupling
/// residuals in that block are still checked (loosely, exactly as the old
/// aggregate test did) without stalling on a 0/0.
fn block_scales(b: &[f64], s: usize, bnorm: f64) -> Vec<f64> {
    let mut sums = vec![0.0f64; s];
    for (i, &v) in b.iter().enumerate() {
        sums[i % s] += v * v;
    }
    sums.iter()
        .map(|&q| {
            let n = q.sqrt();
            if n > 1e-300 {
                n
            } else {
                bnorm
            }
        })
        .collect()
}

/// Max over unknown ranks of `||r_c|| / scale_c` — the per-BLOCK true relative
/// residual every convergence decision uses. The AGGREGATE norm is dominated by
/// the momentum rows; requiring each block to meet tol against its own scale is
/// what stops the pressure block from being left with O(1) relative error (the
/// "converged" channel with a ~30% delta-p bias). Returns `+inf` on any
/// non-finite entry (a poisoned residual must never read as converged).
fn block_rel_residual(r: &[f64], scales: &[f64], s: usize) -> f64 {
    let mut sums = vec![0.0f64; s];
    for (i, &v) in r.iter().enumerate() {
        sums[i % s] += v * v;
    }
    let mut worst = 0.0f64;
    for (c, &q) in sums.iter().enumerate() {
        let rel = q.sqrt() / scales[c];
        if !rel.is_finite() {
            return f64::INFINITY;
        }
        worst = worst.max(rel);
    }
    worst
}

/// Restarted, RIGHT-preconditioned FLEXIBLE GMRES — FGMRES(`restart`) — the
/// single banded coupled solve (both block-Jacobi and the Schur variants; the
/// Schur inner heavy-ball safeguard makes `M⁻¹` vary per apply, which plain
/// GMRES cannot tolerate but FGMRES can — and a FIXED `M⁻¹` is trivially valid).
/// Stores the preconditioned Krylov vectors `z_k = M⁻¹ v_k` and updates
/// `x = x0 + Z y`. Mirrors `cpu::linalg::fgmres` (loop shape, best-iterate
/// restore, non-finite bail, projection-trust), with one structured addition:
/// convergence is decided PER UNKNOWN BLOCK on the true residual
/// (see [`block_rel_residual`]), and the returned rel-res is the max over
/// blocks.
#[allow(clippy::too_many_arguments)]
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
    threads: usize,
    amg_cache: Option<&StructuredAmgCache>,
) -> (Vec<f32>, f64, u32) {
    let n = nx * ny * s;
    let b64: Vec<f64> = b.iter().map(|&v| v as f64).collect();
    let bnorm = pnorm(threads, &b64).max(1e-300);
    // Per-block scales (pressure rows get their OWN scale) + the strict scale: a
    // projection residual below `tol * strict_scale` implies EVERY block is under
    // tol (||r_c|| <= ||r||), so it is an always-trusted in-cycle exit.
    let blk_scales = block_scales(&b64, s, bnorm);
    let strict_scale = blk_scales.iter().cloned().fold(f64::INFINITY, f64::min).max(1e-300);
    let mut x = vec![0.0f64; n];
    // Total inner (Arnoldi) iterations across restart cycles — the GMRES iteration
    // count reported to the GUI "Linear: iters=.." readout.
    let mut total_iters = 0u32;
    let max_total = max_total_iters(restart, max_outer);

    // Best-iterate / monotonicity guard (mirrors the unstructured fgmres). A
    // data-dependent preconditioner — the Schur heavy-ball safeguard, or the AMG
    // V-cycle on a TRANSIENTLY-indefinite A_pp during a from-rest startup — can
    // occasionally return a correction that GROWS the true residual (the observed
    // SchurAmg umax~1e25 transient). Track the lowest-residual iterate; if an outer
    // head sees the residual go non-finite or grow past 1.25x the best, restore the
    // best iterate and stop instead of compounding the blow-up. Precision-safe.
    let mut best_x = x.clone();
    let mut best_beta = f64::INFINITY;
    const RESTART_GROWTH_TOL: f64 = 1.25;
    // Projection trust (unstructured-fgmres pattern): the cheap in-cycle Givens
    // estimate is an AGGREGATE norm, so it may propose a cycle break that the
    // head's per-block check rejects. One rejection flips trust off for the rest
    // of the solve — later cycles run until the strict (per-block-sufficient)
    // projection threshold or the restart length, so an aggregate-vs-block gap
    // cannot ping-pong in 1-iteration cycles.
    let mut trust_projection = true;
    let mut proj_broke_early = false;

    // Reuse the CACHED AMG hierarchy (aggregation is sparsity-only, built once from
    // a canonical Poisson seed — see `build_pressure_amg_hierarchy`) and re-Galerkin
    // only the CURRENT `A_pp` values this solve. A per-call fallback hierarchy covers
    // no-cache callers (`banded_gmres`, one-off tests). The assembled `AmgSolver`
    // V-cycle is then reused across every preconditioner apply in this solve.
    let ncells = nx * ny;
    #[cfg(feature = "cpu")]
    let mut fallback_hier: Option<crate::solver::cpu::amg::AmgHierarchy> = None;
    #[cfg(feature = "cpu")]
    let amg_solver = match built {
        Built::Schur(sd) if sd.pressure_amg => {
            let hier: &crate::solver::cpu::amg::AmgHierarchy = match amg_cache {
                Some(cache) => cache.hier.get_or_init(|| build_pressure_amg_hierarchy(nx, ny)),
                None => fallback_hier.insert(build_pressure_amg_hierarchy(nx, ny)),
            };
            let csr = amg_csr_values(&sd.a_pp, nx, ny);
            Some(crate::solver::cpu::amg::AmgSolver::assemble(hier, &csr, threads, false))
        }
        _ => None,
    };
    #[cfg(feature = "cpu")]
    let psolve_owned = amg_solver.as_ref().map(|solver| {
        move |gp: &[f64]| -> Vec<f64> {
            let mut o = vec![0.0f64; ncells];
            solver.vcycle(gp, &mut o);
            o
        }
    });
    #[cfg(feature = "cpu")]
    let psolve: Option<&dyn Fn(&[f64]) -> Vec<f64>> =
        psolve_owned.as_ref().map(|f| f as &dyn Fn(&[f64]) -> Vec<f64>);
    #[cfg(not(feature = "cpu"))]
    let psolve: Option<&dyn Fn(&[f64]) -> Vec<f64>> = {
        let _ = (ncells, amg_cache);
        None
    };

    // Per-block true relative residual of the CANDIDATE iterate (`x` at the exit
    // points below; every loop `break` assigns it first) and the aggregate scale
    // for the trusted in-cycle fast-path.
    let mut rel;
    let mut agg_scale: Option<f64> = None;
    // Recompute `rel` for a RESTORED iterate on the bail paths (one extra SpMV;
    // the common converged/budget exits reuse the head's residual).
    let rel_at = |xv: &[f64]| -> f64 {
        let ax = spmv_t(a, nx, ny, s, xv, threads);
        let r: Vec<f64> = (0..n).map(|i| b64[i] - ax[i]).collect();
        block_rel_residual(&r, &blk_scales, s)
    };

    loop {
        // Restart head: TRUE residual r0 = b - A x. Every convergence / budget /
        // monotonicity decision is made here (the in-cycle projection may only
        // PROPOSE a cycle break, verified at the next head).
        let ax = spmv_t(a, nx, ny, s, &x, threads);
        let r0: Vec<f64> = (0..n).map(|i| b64[i] - ax[i]).collect();
        let beta = pnorm(threads, &r0);
        // Non-finite bail (unstructured linalg::fgmres parity): a poisoned
        // operator/rhs must not run the remaining budget nor leak NaN into the
        // state. Restore the best (finite) iterate — the zero initial guess if
        // nothing better was seen — and report a clearly-unconverged residual.
        if !beta.is_finite() {
            if proj_broke_early {
                // A distrusted projection break may have corrupted x; retry from
                // the best iterate with the projection distrusted.
                x.copy_from_slice(&best_x);
                trust_projection = false;
                proj_broke_early = false;
                continue;
            }
            if std::env::var("CFD2_STRUCT_SOLVE_DEBUG").is_ok() {
                let bad_b = b64.iter().filter(|v| !v.is_finite()).count();
                let bad_a = a.iter().filter(|v| !v.is_finite()).count();
                eprintln!(
                    "[banded] NON-FINITE head beta={beta} iters={total_iters} \
                     (non-finite entries: b {bad_b}/{}, a {bad_a}/{})",
                    b64.len(),
                    a.len()
                );
            }
            x.copy_from_slice(&best_x);
            rel = if best_beta.is_finite() { rel_at(&x) } else { f64::INFINITY };
            break;
        }
        // Monotonicity guard: the previous cycle's (data-dependent) correction
        // GREW the true residual past 1.25x the best seen — restore and stop
        // instead of compounding (the observed SchurAmg umax~1e25 transient).
        if beta > best_beta * RESTART_GROWTH_TOL {
            if proj_broke_early {
                x.copy_from_slice(&best_x);
                trust_projection = false;
                proj_broke_early = false;
                continue;
            }
            x.copy_from_slice(&best_x);
            rel = rel_at(&x);
            break;
        }
        if beta < best_beta {
            best_beta = beta;
            best_x.copy_from_slice(&x);
        }
        // Per-block convergence decision on the true residual.
        rel = block_rel_residual(&r0, &blk_scales, s);
        if rel <= tol || total_iters >= max_total {
            break;
        }
        if proj_broke_early {
            // The aggregate projection claimed convergence but the per-block head
            // check disagrees: stop trusting the aggregate fast-path this solve.
            trust_projection = false;
            proj_broke_early = false;
        }
        let rs = *agg_scale.get_or_insert_with(|| bnorm.min(beta).max(1e-300));

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
            let zk = apply(built, a, nx, ny, s, &v[k], threads, psolve);
            let mut w = spmv_t(a, nx, ny, s, &zk, threads);
            z.push(zk);
            for i in 0..=k {
                h[i][k] = pdot(threads, &w, &v[i]);
                axpy(&mut w, -h[i][k], &v[i]);
            }
            h[k + 1][k] = pnorm(threads, &w);
            if !h[k + 1][k].is_finite() {
                // Poisoned Arnoldi vector: drop the column, let the head decide
                // on whatever progress the earlier columns made.
                k_used = k;
                break;
            }
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
            total_iters += 1;
            // In-cycle exits: the strict threshold (projection <= tol times the
            // SMALLEST block scale) is SUFFICIENT for per-block convergence, so
            // it is always trusted; the aggregate threshold is the optimistic
            // fast-path, verified at the head (see `trust_projection`).
            let proj = g[k + 1].abs();
            let strict_conv = proj <= tol * strict_scale;
            let agg_conv = trust_projection && proj <= tol * rs;
            if strict_conv || agg_conv || !proj.is_finite() || total_iters >= max_total {
                proj_broke_early = agg_conv && !strict_conv;
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
        if kk == 0 {
            // Breakdown with no progress: the head would loop forever on the
            // same x. Keep the head-computed `rel` for this x and stop.
            break;
        }
        // Loop back: the head recomputes the true residual at the restarted
        // iterate and applies the convergence / budget / monotonicity guards.
    }

    if rel > tol && std::env::var("CFD2_STRUCT_SOLVE_DEBUG").is_ok() {
        eprintln!(
            "[banded] EXIT unconverged iters={total_iters}/{max_total} restart={restart} \
             rel_res={rel:.3e} (tol={tol:.1e} not reached)"
        );
    }
    (x.iter().map(|&v| v as f32).collect(), rel, total_iters)
}

/// A model's declared Schur block layout, extracted once so a solver can rebuild
/// its preconditioner for either pressure-solve choice (heavy-ball vs AMG)
/// without re-reading the model.
#[derive(Clone, Debug)]
pub struct SchurLayout {
    pub u_idx: Vec<usize>,
    pub p: usize,
    pub omega: f32,
    pub sweeps_cap: u32,
}

/// Per-step convergence telemetry surfaced to the GUI by the structured banded
/// solvers (CPU [`StructuredModelSolver`] and GPU `StructuredGpuSolver`). Lives
/// here — in the always-compiled solve module, not the feature-gated `cpu`
/// module — so both backends can produce it. Mirrors what the unstructured
/// `step_stats()` reports: `outer_iters`, the last inner solve's linear
/// rel-residual, and the L-infinity coupled-increment (Picard) residuals over
/// the velocity / pressure slots.
#[derive(Default, Clone, Copy)]
pub struct StructuredStepStats {
    pub outer_iters: u32,
    /// Inner GMRES iterations of the last outer's linear solve.
    pub linear_iters: u32,
    pub linear_res: f32,
    pub outer_du: f32,
    pub outer_dp: f32,
}

/// Per-FIELD Picard (outer) residual of a structured step, shared by the CPU and
/// GPU structured solvers so their early-exit decisions cannot drift apart.
///
/// `groups` are the state offsets of each solved field's components (see
/// `model_unknown_state_offset_groups`). Returns, per field, the L-infinity
/// applied correction `max|state − state_iter|` divided by that field's OWN
/// L-infinity scale `max|state|`.
///
/// The scale is NOT floored at 1.0. A `max(scale, 1.0)` floor (the convention the
/// GPU `OuterConvergenceMonitor` inherited) silently turns the relative test into
/// an ABSOLUTE one for any field whose magnitude is below unity — and the shipped
/// GUI channel flow has `max|U| ≈ 0.02` and gauge `max|p| ≈ 2e-3`. At `tol = 1e-3`
/// the pressure was then allowed to move by half its own range and still count as
/// converged, so the loop exited after the 2-iteration minimum with an
/// under-solved field: non-monotone stagnation velocity at an immersed boundary,
/// pressure inside the Brinkman solid an order of magnitude outside the fluid
/// range, and ~10% mass-flux imbalance. Only the near-zero case needs a guard, and
/// there the correction is near-zero too, so a tiny floor suffices.
pub fn structured_outer_residuals(
    state: &[f32],
    state_iter: &[f32],
    stride: usize,
    groups: &[Vec<usize>],
) -> Vec<f32> {
    /// Guards `0/0` for a field that is identically zero (its correction is zero
    /// too, so the ratio is 0 and the field reads as converged).
    const SCALE_FLOOR: f32 = 1e-30;

    let n_cells = state.len() / stride.max(1);
    groups
        .iter()
        .map(|comps| {
            let (mut delta, mut scale) = (0.0f32, 0.0f32);
            let mut finite = true;
            for cell in 0..n_cells {
                let base = cell * stride;
                for &off in comps {
                    let a = state[base + off];
                    let b = state_iter[base + off];
                    // NaN-blindness guard: `f32::max` IGNORES a NaN operand, so a
                    // diverged (NaN) state would read as delta 0 = "converged" and
                    // the Picard loop would early-exit on garbage. A non-finite
                    // entry must read as NOT converged.
                    if !a.is_finite() || !b.is_finite() {
                        finite = false;
                    }
                    delta = delta.max((a - b).abs());
                    scale = scale.max(a.abs());
                }
                if !finite {
                    break;
                }
            }
            if finite {
                delta / scale.max(SCALE_FLOOR)
            } else {
                f32::INFINITY
            }
        })
        .collect()
}

impl BandedPrecond {
    /// Build a Schur preconditioner from a [`SchurLayout`], picking the inner
    /// pressure solve (`pressure_amg`: AMG V-cycle vs safeguarded heavy-ball).
    pub fn schur(layout: &SchurLayout, pressure_amg: bool) -> Self {
        BandedPrecond::Schur {
            u_idx: layout.u_idx.clone(),
            p: layout.p,
            omega: layout.omega,
            sweeps_cap: layout.sweeps_cap,
            pressure_amg,
        }
    }
}

/// Read a model's declared Schur block layout, if any. `None` when the model
/// declares no Schur preconditioner (e.g. the density-based compressible model)
/// — the caller then uses block-Jacobi.
pub fn schur_layout_from_model(model: &crate::solver::model::ModelSpec) -> Option<SchurLayout> {
    model.linear_solver.and_then(|ls| match ls.preconditioner {
        crate::solver::model::ModelPreconditionerSpec::Schur { omega, sweeps_cap, layout } => {
            Some(SchurLayout {
                u_idx: layout.u_indices().iter().map(|&u| u as usize).collect(),
                p: layout.p as usize,
                omega,
                sweeps_cap,
            })
        }
        _ => None,
    })
}

/// Resolve the active preconditioner of a built [`BandedPrecond`] back to its
/// [`CoupledPrecondKind`] (for the solver's `set_preconditioner` return value).
pub fn kind_of(precond: &BandedPrecond) -> CoupledPrecondKind {
    match precond {
        BandedPrecond::BlockJacobi => CoupledPrecondKind::BlockJacobi,
        BandedPrecond::Schur { pressure_amg: true, .. } => CoupledPrecondKind::SchurAmg,
        BandedPrecond::Schur { .. } => CoupledPrecondKind::Schur,
    }
}
