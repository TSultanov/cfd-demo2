//! CPU sparse linear algebra for the CPU backend.
//!
//! The GPU backend implements the linear solve as a stack of WGSL kernels
//! (FGMRES/CG/AMG/Schur over a CSR matrix). The CPU backend instead reads the
//! *same assembled CSR system* — `row_offsets` / `col_indices` / `matrix_values`
//! / `rhs`, produced by the generic-coupled assembly kernel — and solves it with
//! a plain Rust iterative solver. The matrix is generally nonsymmetric (the
//! convection operator), so we use **BiCGSTAB** with a **Jacobi (diagonal)**
//! preconditioner. We solve in `f64` for robustness even though the assembled
//! values are `f32`; results are compared to references at tolerance, not
//! bit-exactly against the GPU.

use crate::solver::cpu::parallel::{par_map_into, par_update};

/// A borrowed CSR matrix (single scalar unknown per row).
pub struct CsrView<'a> {
    pub row_offsets: &'a [u32],
    pub col_indices: &'a [u32],
    pub values: &'a [f32],
    /// Worker threads for the parallel matvec (`1` = serial).
    pub threads: usize,
}

impl CsrView<'_> {
    pub fn n(&self) -> usize {
        self.row_offsets.len() - 1
    }

    /// `y = A x` (both length n), computed in f64.
    fn spmv(&self, x: &[f64], y: &mut [f64]) {
        let threads = self.threads.max(1);
        if threads <= 1 {
            self.spmv_range(x, 0, self.n(), y);
            return;
        }
        // Disjoint output rows per worker; shared read-only `x`/`values`.
        crate::solver::cpu::parallel::parallel_cell_chunks_mut(
            self.n(),
            1,
            threads,
            y,
            |row0, yc| self.spmv_range(x, row0, row0 + yc.len(), yc),
        );
    }

    #[inline]
    fn spmv_range(&self, x: &[f64], row0: usize, row1: usize, y_out: &mut [f64]) {
        for row in row0..row1 {
            let start = self.row_offsets[row] as usize;
            let end = self.row_offsets[row + 1] as usize;
            let mut sum = 0.0f64;
            for k in start..end {
                sum += self.values[k] as f64 * x[self.col_indices[k] as usize];
            }
            y_out[row - row0] = sum;
        }
    }

    /// Diagonal entries (for Jacobi preconditioning).
    fn diagonal(&self) -> Vec<f64> {
        let n = self.n();
        let mut d = vec![0.0f64; n];
        for row in 0..n {
            let start = self.row_offsets[row] as usize;
            let end = self.row_offsets[row + 1] as usize;
            for k in start..end {
                if self.col_indices[k] as usize == row {
                    d[row] = self.values[k] as f64;
                }
            }
        }
        d
    }
}

/// Upper bound on the block size S (unknowns per cell) for stack-allocated
/// per-cell scratch. Coupled models here are S <= 4 (U.x,U.y,p,T); 8 leaves head-room
/// while keeping the fixed array small enough to live in registers/L1.
const MAX_BLOCK_S: usize = 8;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// SIMD dot product (4-wide f64). FP reductions don't auto-vectorise (summation
/// is non-associative), so this is the meaningful manual-SIMD path; it differs
/// from the scalar dot only in summation order (matches to rounding).
fn dot_simd(a: &[f64], b: &[f64]) -> f64 {
    use wide::f64x4;
    let n = a.len();
    let mut acc = f64x4::splat(0.0);
    let mut i = 0;
    while i + 4 <= n {
        let va = f64x4::from([a[i], a[i + 1], a[i + 2], a[i + 3]]);
        let vb = f64x4::from([b[i], b[i + 1], b[i + 2], b[i + 3]]);
        acc += va * vb;
        i += 4;
    }
    let lanes = acc.to_array();
    let mut s = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

// ── Block-CSR (coupled systems, unknowns_per_cell = S) ──────────────────────
//
// The coupled assembly kernel writes a block-CSR matrix in the *SoA* layout
// produced by `CoupledAccumulators::declare_start_rows` + `BlockCsrSoaMatrix`
// (crates/cfd2_codegen/.../dsl). For a cell `i` with `S` unknowns:
//   scalar_offset  = scalar_row_offsets[i]                  (block index of row start)
//   num_neighbors  = scalar_row_offsets[i+1] - scalar_offset (blocks in this row)
//   diag_rank      = diagonal_indices[i] - scalar_offset     (position of the diag block)
//   start_row_0    = scalar_offset * S*S
//   start_row_r    = start_row_0 + num_neighbors * S * r     (r in 1..S)
// The block entry (row r, col c) for the block at position `rank` within the row
// (connecting cell i to cell `col_indices[scalar_offset + rank]`) lives at
//   matrix_values[start_row_r + rank*S + c].
// The right-hand side / solution are packed `rhs[i*S + r]`.
//
// This is the CPU mirror of what the WGSL FGMRES/AMG kernel stack reads; the CPU
// reads the *same assembled buffers* and solves with its own Krylov method.

/// A borrowed block-CSR matrix in the assembly kernel's SoA layout.
#[derive(Clone, Copy)]
pub struct BlockCsr<'a> {
    /// Unknowns per cell (block dimension S).
    pub s: usize,
    /// Block-level row offsets, length `num_cells + 1`.
    pub scalar_row_offsets: &'a [u32],
    /// Neighbour cell per block, length `nnz_blocks`.
    pub col_indices: &'a [u32],
    /// Absolute block index of each cell's diagonal block, length `num_cells`.
    pub diagonal_indices: &'a [u32],
    /// Packed block values, length `nnz_blocks * S*S`.
    pub values: &'a [f32],
    /// Worker threads for the parallel matvec / preconditioner (`1` = serial).
    pub threads: usize,
}

impl BlockCsr<'_> {
    pub fn num_cells(&self) -> usize {
        self.scalar_row_offsets.len() - 1
    }

    /// Total scalar unknowns (`num_cells * S`).
    pub fn n(&self) -> usize {
        self.num_cells() * self.s
    }

    #[inline]
    fn scalar_offset(&self, cell: usize) -> usize {
        self.scalar_row_offsets[cell] as usize
    }

    #[inline]
    fn num_neighbors(&self, cell: usize) -> usize {
        self.scalar_row_offsets[cell + 1] as usize - self.scalar_offset(cell)
    }

    /// Base index into `values` of block-row `r` for `cell` (rank 0 column).
    #[inline]
    fn start_row(&self, cell: usize, r: usize) -> usize {
        let s = self.s;
        let scalar_offset = self.scalar_offset(cell);
        let num_neighbors = self.num_neighbors(cell);
        scalar_offset * s * s + num_neighbors * s * r
    }

    /// `y = A x` (both length `n()`), accumulated in f64.
    fn block_spmv(&self, x: &[f64], y: &mut [f64]) {
        let s = self.s;
        let threads = self.threads.max(1);
        if threads <= 1 {
            self.block_spmv_range(x, 0, self.num_cells(), y);
            return;
        }
        // Disjoint output: each worker owns a contiguous cell range and writes only
        // its own `y[cell*s .. ]` slots (shared read-only `x` / `values`). No races.
        crate::solver::cpu::parallel::parallel_cell_chunks_mut(
            self.num_cells(),
            s,
            threads,
            y,
            |cell0, yc| self.block_spmv_range(x, cell0, cell0 + yc.len() / s, yc),
        );
    }

    /// Compute `y_out[.. ] = (A x)` for cells `[cell0, cell1)`, where `y_out` is the
    /// output sub-slice covering exactly those cells (length `(cell1-cell0)*s`).
    #[inline]
    fn block_spmv_range(&self, x: &[f64], cell0: usize, cell1: usize, y_out: &mut [f64]) {
        let s = self.s;
        debug_assert!(s <= MAX_BLOCK_S, "block size {s} exceeds MAX_BLOCK_S");
        for cell in cell0..cell1 {
            let scalar_offset = self.scalar_offset(cell);
            let num_neighbors = self.num_neighbors(cell);
            // Per-row accumulators (stack, not a per-cell heap allocation).
            let mut acc = [0.0f64; MAX_BLOCK_S];
            let acc = &mut acc[..s];
            for rank in 0..num_neighbors {
                let j = self.col_indices[scalar_offset + rank] as usize;
                for (r, a) in acc.iter_mut().enumerate() {
                    let base = self.start_row(cell, r) + rank * s;
                    let mut sum = 0.0f64;
                    for c in 0..s {
                        sum += self.values[base + c] as f64 * x[j * s + c];
                    }
                    *a += sum;
                }
            }
            let out = &mut y_out[(cell - cell0) * s..(cell - cell0) * s + s];
            out.copy_from_slice(acc);
        }
    }

    /// Extract each cell's S×S diagonal block (row-major), length `num_cells*S*S`.
    fn diagonal_blocks(&self) -> Vec<f64> {
        let s = self.s;
        let mut out = vec![0.0f64; self.num_cells() * s * s];
        let extract = |cell0: usize, chunk: &mut [f64]| {
            for (li, block) in chunk.chunks_mut(s * s).enumerate() {
                let cell = cell0 + li;
                let scalar_offset = self.scalar_offset(cell);
                let diag_rank = self.diagonal_indices[cell] as usize - scalar_offset;
                for r in 0..s {
                    let base = self.start_row(cell, r) + diag_rank * s;
                    for c in 0..s {
                        block[r * s + c] = self.values[base + c] as f64;
                    }
                }
            }
        };
        let threads = self.threads.max(1);
        if threads <= 1 {
            extract(0, &mut out);
        } else {
            crate::solver::cpu::parallel::parallel_cell_chunks_mut(
                self.num_cells(),
                s * s,
                threads,
                &mut out,
                extract,
            );
        }
        out
    }
}

/// Invert an `s`×`s` row-major matrix in place into `inv` via Gauss–Jordan with
/// partial pivoting. Falls back to the identity-scaled diagonal if singular so
/// the preconditioner stays defined (it only needs to be an approximation).
fn invert_dense(s: usize, src: &[f64], inv: &mut [f64]) {
    // Working augmented copy [A | I].
    let mut a = src.to_vec();
    // Initialise inv = I.
    for (i, v) in inv.iter_mut().enumerate() {
        *v = if i % (s + 1) == 0 { 1.0 } else { 0.0 };
    }
    for col in 0..s {
        // Partial pivot: find max |a[row][col]| for row >= col.
        let mut piv = col;
        let mut best = a[col * s + col].abs();
        for row in (col + 1)..s {
            let v = a[row * s + col].abs();
            if v > best {
                best = v;
                piv = row;
            }
        }
        if best < 1e-30 {
            // Singular block: leave a diagonal (Jacobi-like) approximation.
            for v in inv.iter_mut() {
                *v = 0.0;
            }
            for i in 0..s {
                let d = src[i * s + i];
                inv[i * s + i] = if d.abs() > 1e-30 { 1.0 / d } else { 1.0 };
            }
            return;
        }
        if piv != col {
            for k in 0..s {
                a.swap(piv * s + k, col * s + k);
                inv.swap(piv * s + k, col * s + k);
            }
        }
        let diag = a[col * s + col];
        let inv_diag = 1.0 / diag;
        for k in 0..s {
            a[col * s + k] *= inv_diag;
            inv[col * s + k] *= inv_diag;
        }
        for row in 0..s {
            if row == col {
                continue;
            }
            let factor = a[row * s + col];
            if factor == 0.0 {
                continue;
            }
            for k in 0..s {
                a[row * s + k] -= factor * a[col * s + k];
                inv[row * s + k] -= factor * inv[col * s + k];
            }
        }
    }
}

/// A left preconditioner `z = M^{-1} r`.
pub trait Preconditioner {
    fn apply(&self, r: &[f64], z: &mut [f64]);
}

/// Block-Jacobi: per-cell S×S diagonal-block inverse.
pub struct BlockJacobi {
    s: usize,
    inv_blocks: Vec<f64>,
    threads: usize,
}

impl BlockJacobi {
    pub fn new(a: &BlockCsr) -> Self {
        let s = a.s;
        let threads = a.threads.max(1);
        let diag = a.diagonal_blocks();
        let mut inv_blocks = vec![0.0f64; diag.len()];
        // Each cell inverts its own disjoint S×S block (reads `diag[base..]`,
        // writes `inv_blocks[base..]`); no cross-cell dependency.
        let invert = |cell0: usize, chunk: &mut [f64]| {
            for (li, block) in chunk.chunks_mut(s * s).enumerate() {
                let base = (cell0 + li) * s * s;
                invert_dense(s, &diag[base..base + s * s], block);
            }
        };
        if threads <= 1 {
            invert(0, &mut inv_blocks);
        } else {
            crate::solver::cpu::parallel::parallel_cell_chunks_mut(
                a.num_cells(),
                s * s,
                threads,
                &mut inv_blocks,
                invert,
            );
        }
        Self { s, inv_blocks, threads }
    }
}

impl Preconditioner for BlockJacobi {
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        let s = self.s;
        let cells = r.len() / s;
        let inv_blocks = &self.inv_blocks;
        let mul = |cell0: usize, zc: &mut [f64]| {
            for (li, zrow) in zc.chunks_mut(s).enumerate() {
                let cell = cell0 + li;
                let base = cell * s * s;
                for row in 0..s {
                    let mut sum = 0.0f64;
                    for col in 0..s {
                        sum += inv_blocks[base + row * s + col] * r[cell * s + col];
                    }
                    zrow[row] = sum;
                }
            }
        };
        if self.threads <= 1 || cells <= 1 {
            mul(0, z);
        } else {
            crate::solver::cpu::parallel::parallel_cell_chunks_mut(cells, s, self.threads, z, mul);
        }
    }
}

/// Point (scalar-diagonal) Jacobi: `z_i = r_i / A_ii`. Weaker than block-Jacobi
/// but mirrors the GPU's generic-coupled `Jacobi` preconditioner
/// (`jacobi_diag_*_inv`). For the marginally-stable compressible Picard system
/// this weaker preconditioner is essential: block-Jacobi over-converges each
/// linear solve into a full Newton/Picard correction that excites the
/// inviscid-margin instability, whereas point-Jacobi + the inexact-Picard stop
/// (1e-4) yields the damped correction the GPU relies on.
pub struct PointJacobi {
    inv_diag: Vec<f64>,
}

impl PointJacobi {
    pub fn new(a: &BlockCsr) -> Self {
        let s = a.s;
        let diag = a.diagonal_blocks();
        let mut inv_diag = vec![0.0f64; a.num_cells() * s];
        for cell in 0..a.num_cells() {
            for r in 0..s {
                let d = diag[cell * s * s + r * s + r];
                inv_diag[cell * s + r] = if d.abs() > 1e-30 { 1.0 / d } else { 1.0 };
            }
        }
        Self { inv_diag }
    }
}

impl Preconditioner for PointJacobi {
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        for i in 0..r.len() {
            z[i] = self.inv_diag[i] * r[i];
        }
    }
}

/// SIMPLE-like Schur-complement preconditioner for saddle-point block systems
/// (incompressible/buoyant), mirroring the GPU generic Schur
/// (`schur_precond_generic.wgsl` + `generic_coupled_schur_setup.wgsl`):
///   1. predict velocity  z_u = diag(A_uu)^-1 r_u,  z_p = 0
///   2. Schur RHS         g_p = r_p - A_pu diag(A_uu)^-1 r_u
///   3. pressure solve    A_pp p = g_p   (A_pp = the pressure-pressure block;
///                        the GPU smooths it, here BiCGSTAB solves it — FGMRES is
///                        flexible, so a variable/accurate inner solve is fine)
///   4. correct velocity  z_u -= diag(A_uu)^-1 A_up p,   z_p = p
/// Built per solve from the assembled block matrix (the matrix changes each
/// outer iteration). `omega` is accepted for parity with the GPU spec but the
/// BiCGSTAB pressure solve makes the relaxation factor moot.
pub struct SchurPrecond<'a> {
    a: BlockCsr<'a>,
    u_idx: Vec<usize>,
    p: usize,
    diag_u_inv: Vec<f64>, // num_cells * u_len
    diag_p_inv: Vec<f64>, // num_cells
    p_values: Vec<f32>,   // A_pp scalar-CSR values (topology = scalar_row_offsets/col_indices)
    simd: bool,
}

impl<'a> SchurPrecond<'a> {
    pub fn new(a: BlockCsr<'a>, u_idx: &[usize], p: usize, _omega: f64, simd: bool) -> Self {
        let s = a.s;
        let cells = a.num_cells();
        let u_len = u_idx.len();
        let mut diag_u_inv = vec![0.0f64; cells * u_len];
        let mut diag_p_inv = vec![0.0f64; cells];
        let nnz = a.col_indices.len();
        let mut p_values = vec![0.0f32; nnz];
        for cell in 0..cells {
            let scalar_offset = a.scalar_offset(cell);
            let num_neighbors = a.num_neighbors(cell);
            let diag_rank = a.diagonal_indices[cell] as usize - scalar_offset;
            // diag(A_pp)
            let base_p = a.start_row(cell, p) + diag_rank * s;
            let dp = a.values[base_p + p] as f64;
            diag_p_inv[cell] = if dp.abs() > 1e-30 { 1.0 / dp } else { 0.0 };
            // diag(A_uu) per velocity component
            for (i, &u) in u_idx.iter().enumerate() {
                let base_u = a.start_row(cell, u) + diag_rank * s;
                let du = a.values[base_u + u] as f64;
                diag_u_inv[cell * u_len + i] = if du.abs() > 1e-30 { 1.0 / du } else { 0.0 };
            }
            // A_pp scalar-CSR row (one value per neighbour block).
            let srp = a.start_row(cell, p);
            for rank in 0..num_neighbors {
                p_values[scalar_offset + rank] = a.values[srp + rank * s + p];
            }
        }
        Self { a, u_idx: u_idx.to_vec(), p, diag_u_inv, diag_p_inv, p_values, simd }
    }
}

impl Preconditioner for SchurPrecond<'_> {
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        let s = self.a.s;
        let cells = self.a.num_cells();
        let u_len = self.u_idx.len();
        let p = self.p;

        // 1. predict velocity + 2. form Schur RHS g_p.
        z.copy_from_slice(r);
        let mut gp = vec![0.0f64; cells];
        for cell in 0..cells {
            for (i, &u) in self.u_idx.iter().enumerate() {
                z[cell * s + u] = self.diag_u_inv[cell * u_len + i] * r[cell * s + u];
            }
            z[cell * s + p] = 0.0;
            let scalar_offset = self.a.scalar_offset(cell);
            let num_neighbors = self.a.num_neighbors(cell);
            let srp = self.a.start_row(cell, p);
            let mut g = r[cell * s + p];
            for rank in 0..num_neighbors {
                let col_cell = self.a.col_indices[scalar_offset + rank] as usize;
                for (i, &u) in self.u_idx.iter().enumerate() {
                    let a_pu = self.a.values[srp + rank * s + u] as f64;
                    g -= a_pu * self.diag_u_inv[col_cell * u_len + i] * r[col_cell * s + u];
                }
            }
            gp[cell] = g;
        }

        // 3. pressure solve A_pp p = g_p (scalar CSR; topology = block topology).
        let pa = CsrView {
            row_offsets: self.a.scalar_row_offsets,
            col_indices: self.a.col_indices,
            values: &self.p_values,
            threads: self.a.threads,
        };
        let gp_f32: Vec<f32> = gp.iter().map(|&v| v as f32).collect();
        let mut psol = vec![0.0f32; cells];
        // Cheap APPROXIMATE pressure solve — this is a preconditioner, and FGMRES
        // is flexible, so a loose inner solve (few iterations) keeps each outer
        // iteration cheap (mirrors the GPU's fixed smoother sweeps).
        bicgstab(&pa, &gp_f32, &mut psol, 40, 1e-2, self.simd);
        let psol: Vec<f64> = psol.iter().map(|&v| v as f64).collect();

        // 4. correct velocity, write pressure.
        for cell in 0..cells {
            let scalar_offset = self.a.scalar_offset(cell);
            let num_neighbors = self.a.num_neighbors(cell);
            for (i, &u) in self.u_idx.iter().enumerate() {
                let sru = self.a.start_row(cell, u);
                let mut corr = 0.0f64;
                for rank in 0..num_neighbors {
                    let col_cell = self.a.col_indices[scalar_offset + rank] as usize;
                    let a_up = self.a.values[sru + rank * s + p] as f64;
                    corr += a_up * psol[col_cell];
                }
                z[cell * s + u] -= self.diag_u_inv[cell * u_len + i] * corr;
            }
            z[cell * s + p] = psol[cell];
        }
    }
}

/// Restarted, flexible GMRES — FGMRES(`restart`) — over a block-CSR matrix with
/// a pluggable (possibly nonlinear/iterative) preconditioner. Mirrors the GPU's
/// FGMRES(60) so a variable preconditioner (the Schur complement smoother in
/// Phase 2) can be slotted in without breaking the Krylov recurrence. `x` is the
/// initial guess and receives the solution. Convergence is purely relative
/// (`||b - A x|| / ||b|| <= tol`), matching the GPU inexact-Picard criterion.
#[allow(clippy::too_many_arguments)]
pub fn fgmres(
    a: &BlockCsr,
    b: &[f32],
    x: &mut [f32],
    precond: &dyn Preconditioner,
    restart: usize,
    max_iter: usize,
    tol: f64,
    simd: bool,
) -> SolveStats {
    let n = a.n();
    assert_eq!(b.len(), n);
    assert_eq!(x.len(), n);
    let m = restart.max(1);
    let threads = a.threads.max(1);

    let vdot = |u: &[f64], w: &[f64]| if simd { dot_simd(u, w) } else { dot(u, w) };
    let vnorm = |u: &[f64]| vdot(u, u).sqrt();

    let bf: Vec<f64> = b.iter().map(|&v| v as f64).collect();
    let bnorm = vnorm(&bf).max(1e-300);
    let mut xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();

    // Krylov / flexible bases: grown LAZILY. The restart cap `m` is 60, but with a
    // warm start + inexact tolerance only a handful of iterations typically run, so
    // eagerly allocating and zeroing `m+1` full length-`n` vectors wastes GBs of
    // memset per solve (2*61*n*8 bytes at n=3M). Each basis vector persists across
    // restart cycles (reused/overwritten), so the bases grow at most to the largest
    // iteration count actually reached. `h` is tiny (m*(m+1)); keep it dense.
    let mut vbasis: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
    let mut zbasis: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut h: Vec<Vec<f64>> = vec![vec![0.0; m + 1]; m];
    let mut cs = vec![0.0f64; m];
    let mut sn = vec![0.0f64; m];
    let mut g = vec![0.0f64; m + 1];

    let mut ax = vec![0.0f64; n];
    // Reused Arnoldi work vector (was `ax.clone()` per iteration).
    let mut w = vec![0.0f64; n];
    // Reused residual buffer (was reallocated each restart).
    let mut r = vec![0.0f64; n];
    let mut total_iters = 0usize;
    let mut res = {
        a.block_spmv(&xf, &mut ax);
        par_map_into(threads, &mut r, |i| bf[i] - ax[i]);
        vnorm(&r)
    };

    while total_iters < max_iter {
        // r0 = b - A x
        a.block_spmv(&xf, &mut ax);
        par_map_into(threads, &mut r, |i| bf[i] - ax[i]);
        let beta = vnorm(&r);
        res = beta;
        if beta / bnorm <= tol {
            break;
        }
        let inv_beta = 1.0 / beta;
        if vbasis.is_empty() {
            vbasis.push(vec![0.0; n]);
        }
        par_map_into(threads, &mut vbasis[0], |i| r[i] * inv_beta);
        for v in g.iter_mut() {
            *v = 0.0;
        }
        g[0] = beta;

        let mut jfin = 0usize;
        for j in 0..m {
            // z_j = M^{-1} v_j ; w = A z_j. Grow the flexible basis lazily.
            if zbasis.len() <= j {
                zbasis.push(vec![0.0; n]);
            }
            precond.apply(&vbasis[j], &mut zbasis[j]);
            a.block_spmv(&zbasis[j], &mut ax);
            w.copy_from_slice(&ax);
            // Modified Gram-Schmidt against v_0..v_j. The dot (reduction) stays
            // serial to keep the scalar path bit-identical across thread counts;
            // the axpy update is elementwise, so it parallelizes bit-exactly.
            for i in 0..=j {
                let hij = vdot(&w, &vbasis[i]);
                h[j][i] = hij;
                let vi = &vbasis[i];
                par_update(threads, &mut w, |k, wk| *wk -= hij * vi[k]);
            }
            let hnext = vnorm(&w);
            h[j][j + 1] = hnext;

            // Apply previous Givens rotations to the new column.
            for i in 0..j {
                let temp = cs[i] * h[j][i] + sn[i] * h[j][i + 1];
                h[j][i + 1] = -sn[i] * h[j][i] + cs[i] * h[j][i + 1];
                h[j][i] = temp;
            }
            // New Givens rotation to zero h[j][j+1].
            let rr = (h[j][j] * h[j][j] + h[j][j + 1] * h[j][j + 1]).sqrt();
            if rr > 1e-300 {
                cs[j] = h[j][j] / rr;
                sn[j] = h[j][j + 1] / rr;
            } else {
                cs[j] = 1.0;
                sn[j] = 0.0;
            }
            h[j][j] = cs[j] * h[j][j] + sn[j] * h[j][j + 1];
            h[j][j + 1] = 0.0;
            // Update the residual projection.
            let gj = g[j];
            g[j] = cs[j] * gj;
            g[j + 1] = -sn[j] * gj;

            total_iters += 1;
            jfin = j + 1;
            res = g[j + 1].abs();
            if res / bnorm <= tol || hnext < 1e-300 || total_iters >= max_iter {
                break;
            }
            let inv_h = 1.0 / hnext;
            if vbasis.len() <= j + 1 {
                vbasis.push(vec![0.0; n]);
            }
            par_map_into(threads, &mut vbasis[j + 1], |k| w[k] * inv_h);
        }

        // Back-substitute H[0..jfin,0..jfin] y = g[0..jfin].
        let mut y = vec![0.0f64; jfin];
        for i in (0..jfin).rev() {
            let mut sum = g[i];
            for k in (i + 1)..jfin {
                sum -= h[k][i] * y[k];
            }
            y[i] = if h[i][i].abs() > 1e-300 {
                sum / h[i][i]
            } else {
                0.0
            };
        }
        // x += Z y  (flexible: use z basis, not v basis).
        for j in 0..jfin {
            let yj = y[j];
            if yj != 0.0 {
                let zj = &zbasis[j];
                par_update(threads, &mut xf, |k, xk| *xk += yj * zj[k]);
            }
        }

        // Recompute residual at the restarted iterate for the loop guard.
        a.block_spmv(&xf, &mut ax);
        par_map_into(threads, &mut r, |i| bf[i] - ax[i]);
        res = vnorm(&r);
        if res / bnorm <= tol {
            break;
        }
    }

    for i in 0..n {
        x[i] = xf[i] as f32;
    }
    SolveStats {
        iters: total_iters,
        rel_residual: res / bnorm,
        converged: res / bnorm <= tol,
    }
}

/// Outcome of a linear solve.
#[derive(Debug, Clone, Copy)]
pub struct SolveStats {
    pub iters: usize,
    /// Final relative residual `||b - A x|| / ||b||`.
    pub rel_residual: f64,
    pub converged: bool,
}

/// Solve `A x = b` with preconditioned BiCGSTAB. `x` is used as the initial guess
/// and overwritten with the solution. Returns convergence statistics.
pub fn bicgstab(
    a: &CsrView,
    b: &[f32],
    x: &mut [f32],
    max_iter: usize,
    tol: f64,
    simd: bool,
) -> SolveStats {
    let n = a.n();
    assert_eq!(b.len(), n);
    assert_eq!(x.len(), n);

    // The SIMD switch only changes the dot/norm reduction implementation.
    let vdot = |a: &[f64], b: &[f64]| if simd { dot_simd(a, b) } else { dot(a, b) };
    let vnorm = |a: &[f64]| vdot(a, a).sqrt();

    let diag = a.diagonal();
    let minv = |v: &[f64], out: &mut [f64]| {
        for i in 0..n {
            // Jacobi; guard a (pathological) zero diagonal.
            out[i] = if diag[i] != 0.0 { v[i] / diag[i] } else { v[i] };
        }
    };

    let bf: Vec<f64> = b.iter().map(|&v| v as f64).collect();
    let bnorm = vnorm(&bf).max(1e-300);

    let mut xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();

    // r = b - A x
    let mut ax = vec![0.0f64; n];
    a.spmv(&xf, &mut ax);
    let mut r: Vec<f64> = (0..n).map(|i| bf[i] - ax[i]).collect();

    let finish = |xf: &[f64], x: &mut [f32], iters: usize, res: f64, conv: bool| -> SolveStats {
        for i in 0..n {
            x[i] = xf[i] as f32;
        }
        SolveStats {
            iters,
            rel_residual: res / bnorm,
            converged: conv,
        }
    };

    let mut res = vnorm(&r);
    if res / bnorm <= tol {
        return finish(&xf, x, 0, res, true);
    }

    let rhat = r.clone();
    let mut rho = 1.0f64;
    let mut alpha = 1.0f64;
    let mut omega = 1.0f64;
    let mut v = vec![0.0f64; n];
    let mut p = vec![0.0f64; n];
    let (mut phat, mut shat, mut t) = (vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]);

    for iter in 1..=max_iter {
        let rho_new = vdot(&rhat, &r);
        if rho_new.abs() < 1e-300 {
            // Breakdown; restart from the current residual.
            return finish(&xf, x, iter, res, res / bnorm <= tol);
        }
        let beta = (rho_new / rho) * (alpha / omega);
        for i in 0..n {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }
        minv(&p, &mut phat);
        a.spmv(&phat, &mut v);
        let rhat_v = vdot(&rhat, &v);
        alpha = rho_new / rhat_v;

        // s = r - alpha v  (reuse r as s after recording)
        let mut s = vec![0.0f64; n];
        for i in 0..n {
            s[i] = r[i] - alpha * v[i];
        }
        let snorm = vnorm(&s);
        if snorm / bnorm <= tol {
            for i in 0..n {
                xf[i] += alpha * phat[i];
            }
            return finish(&xf, x, iter, snorm, true);
        }

        minv(&s, &mut shat);
        a.spmv(&shat, &mut t);
        let tt = vdot(&t, &t).max(1e-300);
        omega = vdot(&t, &s) / tt;

        for i in 0..n {
            xf[i] += alpha * phat[i] + omega * shat[i];
            r[i] = s[i] - omega * t[i];
        }
        res = vnorm(&r);
        if res / bnorm <= tol {
            return finish(&xf, x, iter, res, true);
        }
        if omega.abs() < 1e-300 {
            return finish(&xf, x, iter, res, res / bnorm <= tol);
        }
        rho = rho_new;
    }

    finish(&xf, x, max_iter, res, res / bnorm <= tol)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn residual_norm(a: &CsrView, b: &[f32], x: &[f32]) -> f64 {
        let n = a.n();
        let mut max = 0.0f64;
        for row in 0..n {
            let start = a.row_offsets[row] as usize;
            let end = a.row_offsets[row + 1] as usize;
            let mut s = 0.0f64;
            for k in start..end {
                s += a.values[k] as f64 * x[a.col_indices[k] as usize] as f64;
            }
            max = max.max((s - b[row] as f64).abs());
        }
        max
    }

    #[test]
    fn bicgstab_solves_nonsymmetric_m_matrix() {
        // [ 4 -1  0 ] [x0]   [1]
        // [-2  4 -1 ] [x1] = [2]
        // [ 0 -1  3 ] [x2]   [3]
        let row_offsets = [0u32, 2, 5, 7];
        let col_indices = [0u32, 1, 0, 1, 2, 1, 2];
        let values = [4.0f32, -1.0, -2.0, 4.0, -1.0, -1.0, 3.0];
        let a = CsrView {
            row_offsets: &row_offsets,
            col_indices: &col_indices,
            values: &values,
            threads: 1,
        };
        let b = [1.0f32, 2.0, 3.0];
        let mut x = [0.0f32; 3];
        let stats = bicgstab(&a, &b, &mut x, 100, 1e-10, false);
        assert!(stats.converged, "did not converge: {stats:?}");
        assert!(
            residual_norm(&a, &b, &x) < 1e-5,
            "residual too large: {}",
            residual_norm(&a, &b, &x)
        );
    }

    #[test]
    fn bicgstab_solves_larger_diffusion_like_system() {
        // 1D Laplacian-ish tridiagonal with Dirichlet-ish diagonal boost: SPD,
        // well-conditioned — exercises convergence on a bigger system.
        let n = 200usize;
        let mut row_offsets = vec![0u32];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for i in 0..n {
            if i > 0 {
                col_indices.push((i - 1) as u32);
                values.push(-1.0f32);
            }
            col_indices.push(i as u32);
            values.push(4.0f32);
            if i + 1 < n {
                col_indices.push((i + 1) as u32);
                values.push(-1.0f32);
            }
            row_offsets.push(col_indices.len() as u32);
        }
        let a = CsrView {
            row_offsets: &row_offsets,
            col_indices: &col_indices,
            values: &values,
            threads: 1,
        };
        let b = vec![1.0f32; n];
        let mut x = vec![0.0f32; n];
        let stats = bicgstab(&a, &b, &mut x, 500, 1e-10, false);
        assert!(stats.converged, "did not converge: {stats:?}");
        assert!(residual_norm(&a, &b, &x) < 1e-4);
    }
}

#[cfg(test)]
mod block_tests {
    use super::*;

    // A hand-built S=2, 2-cell block-CSR in the exact assembly SoA layout.
    // Dense system:
    //   [ 4 -1 | 1  0 ]
    //   [-1  4 | 0  1 ]
    //   [ 1  0 | 5 -1 ]
    //   [ 0  1 |-2  5 ]
    // See the layout comment on BlockCsr for the index derivation.
    fn fixture() -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<f32>) {
        let scalar_row_offsets = vec![0u32, 2, 4];
        let col_indices = vec![0u32, 1, 0, 1];
        let diagonal_indices = vec![0u32, 3];
        let values = vec![
            4.0f32, -1.0, 1.0, 0.0, // [0..4]  cell0 r0 rank0(c0)/rank1(c1)
            -1.0, 4.0, 0.0, 1.0, //   [4..8]  cell0 r1 rank0/rank1
            1.0, 0.0, 5.0, -1.0, //   [8..12] cell1 r0 rank0(c0)/rank1(c1)
            0.0, 1.0, -2.0, 5.0, //   [12..16] cell1 r1
        ];
        (scalar_row_offsets, col_indices, diagonal_indices, values)
    }

    fn dense() -> [[f64; 4]; 4] {
        [
            [4.0, -1.0, 1.0, 0.0],
            [-1.0, 4.0, 0.0, 1.0],
            [1.0, 0.0, 5.0, -1.0],
            [0.0, 1.0, -2.0, 5.0],
        ]
    }

    /// A diagonally-dominant 1D chain of `nc` block cells (S=`s`), each coupled to
    /// itself + left + right, laid out in the assembly SoA format. Big enough that
    /// the parallel cell-chunk split is exercised (workers get disjoint ranges).
    fn banded_block_system(nc: usize, s: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<f32>) {
        let neigh: Vec<Vec<usize>> = (0..nc)
            .map(|i| {
                let mut ns = Vec::new();
                if i > 0 {
                    ns.push(i - 1);
                }
                ns.push(i);
                if i < nc - 1 {
                    ns.push(i + 1);
                }
                ns
            })
            .collect();
        let mut sro = vec![0u32];
        let mut col = Vec::new();
        let mut diag = Vec::new();
        let mut blk = 0u32;
        for (i, ns) in neigh.iter().enumerate() {
            for (rank, &c) in ns.iter().enumerate() {
                col.push(c as u32);
                if c == i {
                    diag.push(blk + rank as u32);
                }
            }
            blk += ns.len() as u32;
            sro.push(blk);
        }
        let mut vals = vec![0.0f32; col.len() * s * s];
        for (i, ns) in neigh.iter().enumerate() {
            let scalar_offset = sro[i] as usize;
            let nn = ns.len();
            for (rank, &c) in ns.iter().enumerate() {
                for r in 0..s {
                    let start_row = scalar_offset * s * s + nn * s * r;
                    for cc in 0..s {
                        let v = if c == i {
                            if r == cc {
                                8.0 + r as f32 + (i % 3) as f32
                            } else {
                                0.3
                            }
                        } else if r == cc {
                            -0.7
                        } else {
                            0.1
                        };
                        vals[start_row + rank * s + cc] = v;
                    }
                }
            }
        }
        (sro, col, diag, vals)
    }

    #[test]
    fn block_fgmres_multithread_bit_identical() {
        // The block FGMRES path (block_spmv + BlockJacobi build/apply + the
        // elementwise Krylov updates) must be BIT-IDENTICAL across thread counts:
        // parallel work is over disjoint cell/index ranges (reductions stay serial).
        let (nc, s) = (200usize, 3usize);
        let (sro, col, diag, vals) = banded_block_system(nc, s);
        let n = nc * s;
        let b: Vec<f32> = (0..n).map(|k| ((k * 7 % 13) as f32) - 6.0).collect();
        let solve = |threads: usize| {
            let a = BlockCsr {
                s,
                scalar_row_offsets: &sro,
                col_indices: &col,
                diagonal_indices: &diag,
                values: &vals,
                threads,
            };
            let pc = BlockJacobi::new(&a);
            let mut x = vec![0.0f32; n];
            let stats = fgmres(&a, &b, &mut x, &pc, 30, 1000, 1e-10, false);
            (x, stats)
        };
        let (x1, s1) = solve(1);
        let (x4, s4) = solve(4);
        assert!(s1.converged && s4.converged, "did not converge: {s1:?} {s4:?}");
        let maxd = x1
            .iter()
            .zip(&x4)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(maxd, 0.0, "block fgmres threads=1 vs 4 differ: max|diff|={maxd:e}");
    }

    #[test]
    fn block_spmv_matches_dense() {
        let (sro, ci, di, vals) = fixture();
        let a = BlockCsr {
            s: 2,
            scalar_row_offsets: &sro,
            col_indices: &ci,
            diagonal_indices: &di,
            values: &vals,
            threads: 1,
        };
        let x = [1.0f64, -2.0, 3.0, 0.5];
        let mut y = [0.0f64; 4];
        a.block_spmv(&x, &mut y);
        let d = dense();
        for r in 0..4 {
            let expect: f64 = (0..4).map(|c| d[r][c] * x[c]).sum();
            assert!(
                (y[r] - expect).abs() < 1e-12,
                "row {r}: spmv={} dense={}",
                y[r],
                expect
            );
        }
    }

    #[test]
    fn diagonal_blocks_extracted() {
        let (sro, ci, di, vals) = fixture();
        let a = BlockCsr {
            s: 2,
            scalar_row_offsets: &sro,
            col_indices: &ci,
            diagonal_indices: &di,
            values: &vals,
            threads: 1,
        };
        let diag = a.diagonal_blocks();
        // cell0 diagonal = [[4,-1],[-1,4]], cell1 = [[5,-1],[-2,5]].
        assert_eq!(&diag[0..4], &[4.0, -1.0, -1.0, 4.0]);
        assert_eq!(&diag[4..8], &[5.0, -1.0, -2.0, 5.0]);
    }

    #[test]
    fn invert_dense_2x2() {
        let src = [4.0, -1.0, -1.0, 4.0];
        let mut inv = [0.0f64; 4];
        invert_dense(2, &src, &mut inv);
        // A A^{-1} = I.
        let prod = [
            src[0] * inv[0] + src[1] * inv[2],
            src[0] * inv[1] + src[1] * inv[3],
            src[2] * inv[0] + src[3] * inv[2],
            src[2] * inv[1] + src[3] * inv[3],
        ];
        for (i, &v) in prod.iter().enumerate() {
            let want = if i % 3 == 0 { 1.0 } else { 0.0 };
            assert!((v - want).abs() < 1e-12, "prod[{i}]={v}");
        }
    }

    #[test]
    fn fgmres_block_jacobi_solves() {
        let (sro, ci, di, vals) = fixture();
        let a = BlockCsr {
            s: 2,
            scalar_row_offsets: &sro,
            col_indices: &ci,
            diagonal_indices: &di,
            values: &vals,
            threads: 1,
        };
        let b = [1.0f32, 2.0, 3.0, 4.0];
        let mut x = [0.0f32; 4];
        let m = BlockJacobi::new(&a);
        let stats = fgmres(&a, &b, &mut x, &m, 30, 200, 1e-12, false);
        assert!(stats.converged, "fgmres did not converge: {stats:?}");
        // Verify residual against the dense system.
        let d = dense();
        let mut max_r = 0.0f64;
        for r in 0..4 {
            let ax: f64 = (0..4).map(|c| d[r][c] * x[c] as f64).sum();
            max_r = max_r.max((ax - b[r] as f64).abs());
        }
        assert!(max_r < 1e-5, "residual too large: {max_r}");
    }

    #[test]
    fn fgmres_scalar_block_matches_bicgstab() {
        // S=1 block-CSR is an ordinary CSR; FGMRES+BlockJacobi must solve it.
        let n = 50usize;
        let mut sro = vec![0u32];
        let mut ci = Vec::new();
        let mut di = vec![0u32; n];
        let mut vals = Vec::new();
        for i in 0..n {
            // diagonal first (rank 0), matches build_csr_topology.
            di[i] = ci.len() as u32;
            ci.push(i as u32);
            vals.push(4.0f32);
            if i > 0 {
                ci.push((i - 1) as u32);
                vals.push(-1.0f32);
            }
            if i + 1 < n {
                ci.push((i + 1) as u32);
                vals.push(-1.0f32);
            }
            sro.push(ci.len() as u32);
        }
        let a = BlockCsr {
            s: 1,
            scalar_row_offsets: &sro,
            col_indices: &ci,
            diagonal_indices: &di,
            values: &vals,
            threads: 1,
        };
        let b = vec![1.0f32; n];
        let mut x = vec![0.0f32; n];
        let m = BlockJacobi::new(&a);
        let stats = fgmres(&a, &b, &mut x, &m, 40, 400, 1e-10, true);
        assert!(stats.converged, "did not converge: {stats:?}");
        // residual
        let mut max_r = 0.0f64;
        for i in 0..n {
            let start = sro[i] as usize;
            let end = sro[i + 1] as usize;
            let ax: f64 = (start..end).map(|k| vals[k] as f64 * x[ci[k] as usize] as f64).sum();
            max_r = max_r.max((ax - b[i] as f64).abs());
        }
        assert!(max_r < 1e-4, "residual too large: {max_r}");
    }
}
