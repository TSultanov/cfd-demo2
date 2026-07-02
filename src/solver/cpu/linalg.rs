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

use crate::solver::cpu::parallel::{
    par_dot, par_map_into, par_update, parallel_cell_chunks_mut, parallel_cell_chunks_mut2,
};

/// Fine-grained linear-solve profiling (enabled by `CFD2_CPU_PROFILE` via
/// `CpuSolver::step`). Global atomic accumulators: the linear solve runs once
/// per outer iteration on the solver thread, so plain relaxed adds are enough.
/// The coarse step profile attributes the whole solve to one bucket; these
/// split it into marshal / preconditioner build+apply / spmv / dots / axpys so
/// the serial-vs-parallel and bandwidth-vs-latency structure is visible.
pub mod prof {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

    pub static ENABLED: AtomicBool = AtomicBool::new(false);
    /// Atomic-store → Vec marshalling (matrix/rhs/x in, x out).
    pub static MARSHAL: AtomicU64 = AtomicU64::new(0);
    /// Preconditioner construction (SchurPrecond::new / BlockJacobi::new).
    pub static PC_BUILD: AtomicU64 = AtomicU64::new(0);
    /// FGMRES-level block spmv.
    pub static SPMV: AtomicU64 = AtomicU64::new(0);
    /// FGMRES-level dot/norm reductions (serial by design).
    pub static DOT: AtomicU64 = AtomicU64::new(0);
    /// FGMRES-level elementwise vector ops (parallel, bit-exact).
    pub static AXPY: AtomicU64 = AtomicU64::new(0);
    /// Whole preconditioner application (includes the Schur sub-phases below).
    pub static PC_APPLY: AtomicU64 = AtomicU64::new(0);
    /// Schur apply: velocity predict + Schur RHS (serial cell loop).
    pub static SCHUR_PRE: AtomicU64 = AtomicU64::new(0);
    /// Schur apply: inner pressure BiCGSTAB.
    pub static SCHUR_SOLVE: AtomicU64 = AtomicU64::new(0);
    /// Schur apply: velocity correction (serial cell loop).
    pub static SCHUR_POST: AtomicU64 = AtomicU64::new(0);
    /// Outer FGMRES iterations this step.
    pub static FGMRES_ITERS: AtomicU64 = AtomicU64::new(0);
    /// Inner (Schur pressure) BiCGSTAB iterations this step.
    pub static INNER_ITERS: AtomicU64 = AtomicU64::new(0);

    #[inline]
    pub fn time<T>(acc: &AtomicU64, f: impl FnOnce() -> T) -> T {
        if !ENABLED.load(Relaxed) {
            return f();
        }
        let t = std::time::Instant::now();
        let r = f();
        acc.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        r
    }

    /// Format the accumulated breakdown and reset all counters.
    pub fn report_and_reset() -> String {
        let take = |a: &AtomicU64| a.swap(0, Relaxed);
        let ms = |v: u64| v as f64 / 1e6;
        format!(
            "marshal={:.0}ms pc_build={:.0}ms spmv={:.0}ms dot={:.0}ms axpy={:.0}ms \
             pc_apply={:.0}ms (schur pre={:.0}ms solve={:.0}ms post={:.0}ms) \
             iters fgmres={} inner={}",
            ms(take(&MARSHAL)),
            ms(take(&PC_BUILD)),
            ms(take(&SPMV)),
            ms(take(&DOT)),
            ms(take(&AXPY)),
            ms(take(&PC_APPLY)),
            ms(take(&SCHUR_PRE)),
            ms(take(&SCHUR_SOLVE)),
            ms(take(&SCHUR_POST)),
            take(&FGMRES_ITERS),
            take(&INNER_ITERS),
        )
    }
}

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
/// per-cell scratch. The biharmonic-compressible model carries S = 12
/// (4 conserved + 8 `lap_*` unknowns) — 8 was too small and made the release
/// build panic on the `acc[..s]` slice (the debug_assert only fires in debug).
/// 16 keeps the fixed array at 128 bytes: still registers/L1-friendly.
const MAX_BLOCK_S: usize = 16;

// Dot products / norms in the solvers below go through `par_dot`
// (parallel.rs): a FIXED-CHUNK deterministic reduction — bit-identical across
// thread counts, 4-wide SIMD within chunks, multi-core across chunks. The
// historical `simd` knob on `fgmres`/`bicgstab` is accepted but no longer
// changes the reduction (the chunked kernel is strictly better and equally
// deterministic).

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
    /// Explicit-SIMD matvec (the GUI "CPU Transpiled (SIMD linear)" option):
    /// `block_spmv` uses monomorphized fixed-S kernels with `wide::f64x4`
    /// vector accumulators for S = 3/4 instead of the runtime-`s` scalar
    /// loop (which LLVM cannot fully vectorize at a variable trip count).
    /// Changes the per-row summation ORDER (per-lane partials + one
    /// horizontal add instead of a sequential scalar sum), i.e. a
    /// rounding-level result change — validated by the tolerance suites,
    /// still bit-exact across thread counts (chunking untouched).
    pub simd: bool,
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
        if self.simd {
            match self.s {
                4 => return self.block_spmv_range_simd::<4>(x, cell0, cell1, y_out),
                3 => return self.block_spmv_range_simd::<3>(x, cell0, cell1, y_out),
                _ => {} // uncommon block sizes keep the scalar loop
            }
        }
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

    /// Explicit-SIMD `block_spmv_range` for compile-time block size `S`
    /// (3 or 4; see the `simd` field docs). Layout facts it exploits: the S
    /// columns of a block row are contiguous in `values`, and `x[j*S..]` is
    /// contiguous per neighbour — so each neighbour contributes one f64x4
    /// FMA per block row (S = 3 pads lane 3 with zeros, which contribute
    /// exactly 0.0 to the horizontal sum). The x vector is loaded ONCE per
    /// neighbour and reused across all S block rows (the scalar loop reloads
    /// it per row).
    fn block_spmv_range_simd<const S: usize>(
        &self,
        x: &[f64],
        cell0: usize,
        cell1: usize,
        y_out: &mut [f64],
    ) {
        use wide::f64x4;
        debug_assert_eq!(self.s, S);
        debug_assert!(S == 3 || S == 4);
        for cell in cell0..cell1 {
            let scalar_offset = self.scalar_offset(cell);
            let num_neighbors = self.num_neighbors(cell);
            let row_stride = num_neighbors * S;
            let row0_base = scalar_offset * S * S;
            let mut accv = [f64x4::splat(0.0); 4];
            for rank in 0..num_neighbors {
                let j = self.col_indices[scalar_offset + rank] as usize * S;
                let xv = if S == 4 {
                    f64x4::from([x[j], x[j + 1], x[j + 2], x[j + 3]])
                } else {
                    f64x4::from([x[j], x[j + 1], x[j + 2], 0.0])
                };
                for (r, a) in accv.iter_mut().enumerate().take(S) {
                    let base = row0_base + row_stride * r + rank * S;
                    let vv = if S == 4 {
                        f64x4::from([
                            self.values[base] as f64,
                            self.values[base + 1] as f64,
                            self.values[base + 2] as f64,
                            self.values[base + 3] as f64,
                        ])
                    } else {
                        f64x4::from([
                            self.values[base] as f64,
                            self.values[base + 1] as f64,
                            self.values[base + 2] as f64,
                            0.0,
                        ])
                    };
                    *a = vv.mul_add(xv, *a);
                }
            }
            let out = &mut y_out[(cell - cell0) * S..(cell - cell0) * S + S];
            for (r, o) in out.iter_mut().enumerate() {
                let l = accv[r].to_array();
                *o = (l[0] + l[1]) + (l[2] + l[3]);
            }
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
/// (Also used by `cpu::amg` for the coarsest-level dense solve.)
pub(crate) fn invert_dense(s: usize, src: &[f64], inv: &mut [f64]) {
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

/// Extract the pressure-pressure scalar-CSR values (`A_pp`) from an assembled
/// block matrix; the scalar topology is the block topology
/// (`scalar_row_offsets`/`col_indices`). Used to seed the AMG hierarchy's
/// strength-of-connection aggregation.
pub fn extract_p_values(a: &BlockCsr, p: usize) -> Vec<f32> {
    let s = a.s;
    let mut p_values = vec![0.0f32; a.col_indices.len()];
    for cell in 0..a.num_cells() {
        let scalar_offset = a.scalar_offset(cell);
        let num_neighbors = a.num_neighbors(cell);
        let srp = a.start_row(cell, p);
        for rank in 0..num_neighbors {
            p_values[scalar_offset + rank] = a.values[srp + rank * s + p];
        }
    }
    p_values
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

/// Inner pressure-solve algorithm for [`SchurPrecond`]
/// (`CFD2_CPU_SCHUR_INNER=heavyball|bicgstab|vcycle`, default `heavyball`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum SchurInner {
    /// Fused heavy-ball (second-order Richardson) sweeps — the CPU mirror of the
    /// GPU `relax_pressure` ping-pong. One parallel pass per sweep, no
    /// reductions in the sweep loop; residual checked at geometrically spaced
    /// sweeps (4, 8, 16, …) for early exit on easy blocks. Default: the inner
    /// solve is launch-overhead-bound at bench sizes (BiCGSTAB pays ~13
    /// parallel ops/iteration and anti-scales beyond 4 threads), so fewer,
    /// fatter parallel passes win even at a worse per-pass contraction rate.
    HeavyBall,
    /// Jacobi-preconditioned BiCGSTAB (the pre-heavy-ball default; also the
    /// inner solve whenever an AMG hierarchy is active).
    BiCgStab,
    /// Raw AMG V-cycle(s) as the pressure solve (requires the AMG hierarchy) —
    /// fixed work per apply, no reductions.
    VCycle,
}

/// SIMPLE-like Schur-complement preconditioner for saddle-point block systems
/// (incompressible/buoyant), mirroring the GPU generic Schur
/// (`schur_precond_generic.wgsl` + `generic_coupled_schur_setup.wgsl`):
///   1. predict velocity  z_u = diag(A_uu)^-1 r_u,  z_p = 0
///   2. Schur RHS         g_p = r_p - A_pu diag(A_uu)^-1 r_u
///   3. pressure solve    A_pp p = g_p   (A_pp = the pressure-pressure block;
///                        heavy-ball sweeps by default — the GPU shape — with
///                        BiCGSTAB/AMG fallbacks; FGMRES is flexible, so a
///                        variable/approximate inner solve is fine)
///   4. correct velocity  z_u -= diag(A_uu)^-1 A_up p,   z_p = p
/// Built per solve from the assembled block matrix (the matrix changes each
/// outer iteration). `omega`/`sweeps_cap` follow the GPU spec semantics
/// (`heavy_ball_omega` / `default_pressure_sweeps`, same env overrides).
pub struct SchurPrecond<'a> {
    a: BlockCsr<'a>,
    u_idx: Vec<usize>,
    p: usize,
    diag_u_inv: Vec<f64>, // num_cells * u_len
    p_values: Vec<f32>,   // A_pp scalar-CSR values (topology = scalar_row_offsets/col_indices)
    p_diag_inv: Vec<f64>, // 1 / diag(A_pp) per cell (heavy-ball sweep scaling)
    /// f32 copy of `p_diag_inv` for the mixed-precision sweep (built only
    /// when `a.simd`; empty otherwise).
    p_diag_inv_f32: Vec<f32>,
    /// Heavy-ball relaxation weight (model spec omega via `heavy_ball_omega`).
    omega: f64,
    /// Heavy-ball sweep budget (`default_pressure_sweeps(num_cells, sweeps_cap)`).
    sweeps: usize,
    /// Inner pressure-solve budget/tolerance (env-overridable experiment knobs:
    /// `CFD2_CPU_SCHUR_INNER_ITERS` / `CFD2_CPU_SCHUR_INNER_TOL`).
    inner_iters: usize,
    inner_tol: f64,
    /// Compact `A[p_row, u_col]` values, `u_len` per scalar-CSR entry
    /// (layout `[(scalar_offset + rank) * u_len + i]`). The Schur pre pass
    /// previously strided the FULL block values array to pick `u_len` floats
    /// out of every `s*s`-value block (~2 cache lines touched per block to
    /// use a handful of bytes, per apply, per FGMRES iteration); the compact
    /// array streams linearly. Same values, same arithmetic — bit-exact.
    pu_values: Vec<f32>,
    /// Compact `A[u_row, p_col]` values, same layout (the post/velocity-
    /// correction pass's column).
    up_values: Vec<f32>,
    /// Inner algorithm selection (see [`SchurInner`]).
    inner: SchurInner,
    /// AMG operator for the pressure block (assembled from `p_values` when the
    /// caller supplies a hierarchy): preconditions the inner BiCGSTAB so its
    /// iteration count stays mesh-independent instead of growing ~h^-2.
    amg: Option<crate::solver::cpu::amg::AmgSolver<'a>>,
    /// Inner-solve outcome counters for the caller's adaptive AMG policy:
    /// (applies, applies that failed to converge — cap-out or stagnation).
    applies: std::cell::Cell<u32>,
    inner_failures: std::cell::Cell<u32>,
    /// Reusable apply-path scratch (see [`SchurWork`]): every apply
    /// previously allocated + zero-filled fresh vectors (gp/psol + f32
    /// mirrors + the heavy-ball ping-pong quad = ~24 MB per apply on the
    /// 750k nozzle, ~50 applies/step of pure alloc/page-fault churn).
    work: std::cell::RefCell<SchurWork>,
    /// AMG-path Krylov selection. PCG was REFUTED as the default by
    /// measurement (July 2026, 118k cut-cell obstacle): the "near-SPD"
    /// premise fails on the real block — cut-cell/BC/deferred-correction
    /// nonsymmetry makes CG stall at rel 0.4-1.0 within ~4 iterations
    /// (sometimes diverging past 1.0), every apply becomes a nearly-useless
    /// preconditioner application, and the warm step regressed 0.25 -> 0.82s
    /// (3.2x) while BiCGSTAB's extra spmv/V-cycle per iteration buys real
    /// reduction. Default false (BiCGSTAB);
    /// `CFD2_CPU_SCHUR_AMG_KRYLOV=cg` opts into the experiment (curvature
    /// breakdown still one-way-flips back).
    amg_use_cg: std::cell::Cell<bool>,
}

/// Per-apply scratch reused across [`SchurPrecond::apply`] calls. All
/// buffers are fully overwritten before use (the zero fills that carried
/// semantics — heavy-ball's from-zero start, the inner solves' x0 = 0 —
/// are now explicit `fill(0.0)` at the use sites), so reuse is bit-exact.
#[derive(Default)]
struct SchurWork {
    gp: Vec<f64>,
    psol: Vec<f64>,
    gp_f32: Vec<f32>,
    psol_f32: Vec<f32>,
    /// heavy-ball cur/prev/scratch/best, each `cells` long.
    hb: Vec<f64>,
    /// f32 heavy-ball workspace (cur/prev/scratch/best/g32) for the
    /// mixed-precision path.
    hb32: Vec<f32>,
}

impl<'a> SchurPrecond<'a> {
    pub fn new(
        a: BlockCsr<'a>,
        u_idx: &[usize],
        p: usize,
        omega: f64,
        sweeps_cap: u32,
        _simd: bool,
        amg_hier: Option<&'a crate::solver::cpu::amg::AmgHierarchy>,
    ) -> Self {
        let s = a.s;
        let cells = a.num_cells();
        let u_len = u_idx.len();
        let threads = a.threads.max(1);
        let mut diag_u_inv = vec![0.0f64; cells * u_len];
        let nnz = a.col_indices.len();
        let mut p_values = vec![0.0f32; nnz];
        let mut p_diag_inv = vec![0.0f64; cells];
        let mut pu_values = vec![0.0f32; nnz * u_len];
        let mut up_values = vec![0.0f32; nnz * u_len];
        // Extract diag(A_uu)^-1, the A_pp scalar-CSR values and diag(A_pp)^-1.
        // Each cell writes only its own slots (diag_u_inv/p_diag_inv are
        // cell-strided; a cell's p_values live in its scalar CSR row range, and
        // row ranges are contiguous and monotone in cell), so the fill
        // parallelizes over disjoint cell ranges bit-exactly. p_values chunks
        // are split at scalar-row-offset boundaries (uneven widths).
        let fill = |cell0: usize,
                    du_chunk: &mut [f64],
                    pd_chunk: &mut [f64],
                    pv_chunk: &mut [f32],
                    pu_chunk: &mut [f32],
                    up_chunk: &mut [f32]| {
            let pv_base = a.scalar_offset(cell0);
            for li in 0..pd_chunk.len() {
                let cell = cell0 + li;
                let scalar_offset = a.scalar_offset(cell);
                let num_neighbors = a.num_neighbors(cell);
                let diag_rank = a.diagonal_indices[cell] as usize - scalar_offset;
                // diag(A_uu) per velocity component
                for (i, &u) in u_idx.iter().enumerate() {
                    let base_u = a.start_row(cell, u) + diag_rank * s;
                    let du = a.values[base_u + u] as f64;
                    du_chunk[li * u_len + i] = if du.abs() > 1e-30 { 1.0 / du } else { 0.0 };
                }
                // A_pp scalar-CSR row (one value per neighbour block), plus the
                // compact A_pu / A_up sub-operator rows (see the field docs).
                let srp = a.start_row(cell, p);
                for rank in 0..num_neighbors {
                    pv_chunk[scalar_offset - pv_base + rank] = a.values[srp + rank * s + p];
                    let cbase = (scalar_offset - pv_base + rank) * u_len;
                    for (i, &u) in u_idx.iter().enumerate() {
                        pu_chunk[cbase + i] = a.values[srp + rank * s + u];
                        up_chunk[cbase + i] = a.values[a.start_row(cell, u) + rank * s + p];
                    }
                }
                let dp = a.values[srp + diag_rank * s + p] as f64;
                pd_chunk[li] = if dp.abs() > 1e-30 { 1.0 / dp } else { 0.0 };
            }
        };
        // ~16k cells per worker minimum: below that, region-launch overhead
        // exceeds the extraction work itself (same rationale as the BLAS-1
        // helpers' MIN_ELEMS_PER_WORKER).
        let workers = threads.min(cells.div_ceil(16 * 1024)).max(1);
        if workers <= 1 {
            fill(
                0,
                &mut diag_u_inv,
                &mut p_diag_inv,
                &mut p_values,
                &mut pu_values,
                &mut up_values,
            );
        } else {
            let chunk = cells.div_ceil(workers * crate::solver::cpu::pool::OVERSPLIT).max(1);
            let tasks = cells.div_ceil(chunk);
            let base_du = crate::solver::cpu::pool::MutSlicePtr::new(&mut diag_u_inv);
            let base_pd = crate::solver::cpu::pool::MutSlicePtr::new(&mut p_diag_inv);
            let base_pv = crate::solver::cpu::pool::MutSlicePtr::new(&mut p_values);
            let base_pu = crate::solver::cpu::pool::MutSlicePtr::new(&mut pu_values);
            let base_up = crate::solver::cpu::pool::MutSlicePtr::new(&mut up_values);
            crate::solver::cpu::pool::run(tasks, workers, |w| {
                let start = w * chunk;
                let end = (start + chunk).min(cells);
                let pv_start = a.scalar_offset(start);
                let pv_take = a.scalar_offset(end) - pv_start;
                // SAFETY: tasks own disjoint cell ranges; the p/pu/up splits
                // follow the monotone scalar-row-offset boundaries, so the
                // reconstructed sub-slices never overlap across tasks and
                // all outlive the (blocking) pool::run call.
                let du = unsafe { base_du.slice(start * u_len, (end - start) * u_len) };
                let pd = unsafe { base_pd.slice(start, end - start) };
                let pv = unsafe { base_pv.slice(pv_start, pv_take) };
                let pu = unsafe { base_pu.slice(pv_start * u_len, pv_take * u_len) };
                let up = unsafe { base_up.slice(pv_start * u_len, pv_take * u_len) };
                fill(start, du, pd, pv, pu, up);
            });
        }
        let p_diag_inv_f32: Vec<f32> = if a.simd {
            p_diag_inv.iter().map(|&v| v as f32).collect()
        } else {
            Vec::new()
        };
        let inner_iters = std::env::var("CFD2_CPU_SCHUR_INNER_ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(40);
        // 1e-1 measured best on BOTH benchmark cases (obstacle 2.19→1.59s,
        // nozzle 2.49→2.17s warm step): the apply is a preconditioner, so a
        // loose pressure solve suffices; the outer FGMRES count barely moves
        // while inner iterations halve. (1e-2 was the pre-AMG default.)
        let inner_tol = std::env::var("CFD2_CPU_SCHUR_INNER_TOL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1e-1);
        let inner = match std::env::var("CFD2_CPU_SCHUR_INNER").as_deref() {
            Ok(v) if v.eq_ignore_ascii_case("vcycle") => SchurInner::VCycle,
            Ok(v) if v.eq_ignore_ascii_case("bicgstab") => SchurInner::BiCgStab,
            _ => SchurInner::HeavyBall,
        };
        if std::env::var("CFD2_CPU_SCHUR_DEBUG").is_ok() {
            let zeros = p_diag_inv.iter().filter(|&&v| v == 0.0).count();
            let dmin = p_diag_inv.iter().cloned().fold(f64::INFINITY, f64::min);
            let dmax = p_diag_inv.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            // Power iteration on D^-1 A_pp: heavy-ball/Jacobi stability needs
            // the Jacobi-preconditioned spectrum inside (0, 2).
            let pa = CsrView {
                row_offsets: a.scalar_row_offsets,
                col_indices: a.col_indices,
                values: &p_values,
                threads,
            };
            let mut v: Vec<f64> = (0..cells).map(|i| ((i % 13) as f64) - 6.0).collect();
            let n0 = par_dot(threads, &v, &v).sqrt().max(1e-300);
            for x in v.iter_mut() {
                *x /= n0;
            }
            let mut w = vec![0.0f64; cells];
            let mut lam = 0.0f64;
            for _ in 0..40 {
                pa.spmv(&v, &mut w);
                for i in 0..cells {
                    w[i] *= p_diag_inv[i];
                }
                lam = par_dot(threads, &w, &w).sqrt().max(1e-300);
                for i in 0..cells {
                    v[i] = w[i] / lam;
                }
            }
            // Spectral radius of the plain-Jacobi iteration matrix G = I - D^-1 A
            // (rho(G) > 1 means Jacobi-family smoothing genuinely diverges).
            let mut g: Vec<f64> = (0..cells).map(|i| ((i % 7) as f64) - 3.0).collect();
            let g0 = par_dot(threads, &g, &g).sqrt().max(1e-300);
            for x in g.iter_mut() {
                *x /= g0;
            }
            let mut rho_g = 0.0f64;
            for _ in 0..100 {
                pa.spmv(&g, &mut w);
                for i in 0..cells {
                    w[i] = g[i] - w[i] * p_diag_inv[i];
                }
                rho_g = par_dot(threads, &w, &w).sqrt().max(1e-300);
                for i in 0..cells {
                    g[i] = w[i] / rho_g;
                }
            }
            eprintln!(
                "[schur-build] cells={cells} p_diag_inv zeros={zeros} min={dmin:.3e} \
                 max={dmax:.3e} lam_max(DinvA)~{lam:.3} rho(I-DinvA)~{rho_g:.4}"
            );
        }
        // Same weight/sweep policy (and env overrides) as the GPU Schur:
        // spec omega 1.0 = auto-1.95 (symmetric blocks), verbatim otherwise
        // (all-Mach declares 1.6); sweeps = min(20 + sqrt(n)/8, sweeps_cap).
        let omega =
            crate::solver::gpu::modules::coupled_schur::heavy_ball_omega(omega as f32) as f64;
        let sweeps = crate::solver::gpu::modules::coupled_schur::default_pressure_sweeps(
            cells as u32,
            sweeps_cap,
        );
        // (Counted under PC_BUILD via the caller's wrap around `new`.)
        let amg = amg_hier.map(|h| {
            crate::solver::cpu::amg::AmgSolver::assemble(h, &p_values, a.threads.max(1), a.simd)
        });
        Self {
            a,
            u_idx: u_idx.to_vec(),
            p,
            diag_u_inv,
            p_values,
            pu_values,
            up_values,
            p_diag_inv,
            p_diag_inv_f32,
            omega,
            sweeps,
            inner_iters,
            inner_tol,
            inner,
            amg,
            applies: std::cell::Cell::new(0),
            inner_failures: std::cell::Cell::new(0),
            amg_use_cg: std::cell::Cell::new(
                std::env::var("CFD2_CPU_SCHUR_AMG_KRYLOV").map_or(false, |v| v == "cg"),
            ),
            work: std::cell::RefCell::new(SchurWork::default()),
        }
    }

    /// (inner applies, inner applies that failed to converge) so far — drives
    /// the solver's adaptive Jacobi→AMG switch for the pressure block.
    pub fn inner_outcomes(&self) -> (u32, u32) {
        (self.applies.get(), self.inner_failures.get())
    }
}

impl Preconditioner for SchurPrecond<'_> {
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        let s = self.a.s;
        let cells = self.a.num_cells();
        let u_len = self.u_idx.len();
        let p = self.p;

        // 1. predict velocity + 2. form Schur RHS g_p. Each cell writes only its
        // own z/gp slots (neighbour data is read-only `r`/`values`), so the pass
        // parallelizes over disjoint cell chunks bit-exactly. (Fields are
        // hoisted into locals so the Sync closure doesn't capture `self`, which
        // holds non-Sync outcome counters.)
        z.copy_from_slice(r);
        let mut work = self.work.borrow_mut();
        let work = &mut *work;
        work.gp.resize(cells, 0.0);
        // gp is fully written by the pre pass below; no zeroing needed.
        let gp = &mut work.gp;
        let (aa, u_idx, diag_u_inv) = (&self.a, &self.u_idx, &self.diag_u_inv);
        let (pu_values, up_values) = (&self.pu_values, &self.up_values);
        let simd = aa.simd && u_len <= 4;
        // NOTE: a lane-per-component SIMD variant of the PRE pass was
        // implemented and MEASURED SLOWER (nozzle 49 -> 78 ms/step): the
        // r-vector lanes need per-component scalar gathers (u_idx is not
        // contiguous for the thermal models), so assembling the vectors costs
        // more than the u_len-long scalar FMA chain it replaces. The POST
        // pass below vectorizes cleanly (contiguous A_up run x broadcast
        // psol) and keeps its SIMD variant.
        prof::time(&prof::SCHUR_PRE, || {
            {
                parallel_cell_chunks_mut2(cells, s, 1, aa.threads, z, gp, |cell0, zc, gpc| {
                    for li in 0..gpc.len() {
                        let cell = cell0 + li;
                        for (i, &u) in u_idx.iter().enumerate() {
                            zc[li * s + u] = diag_u_inv[cell * u_len + i] * r[cell * s + u];
                        }
                        zc[li * s + p] = 0.0;
                        let scalar_offset = aa.scalar_offset(cell);
                        let num_neighbors = aa.num_neighbors(cell);
                        let mut g = r[cell * s + p];
                        for rank in 0..num_neighbors {
                            let col_cell = aa.col_indices[scalar_offset + rank] as usize;
                            let cbase = (scalar_offset + rank) * u_len;
                            for (i, &u) in u_idx.iter().enumerate() {
                                // Compact A_pu stream (bit-equal to the block
                                // values it was extracted from).
                                let a_pu = pu_values[cbase + i] as f64;
                                g -= a_pu
                                    * diag_u_inv[col_cell * u_len + i]
                                    * r[col_cell * s + u];
                            }
                        }
                        gpc[li] = g;
                    }
                });
            }
        });

        // 3. pressure solve A_pp p = g_p (scalar CSR; topology = block topology).
        let pa = CsrView {
            row_offsets: self.a.scalar_row_offsets,
            col_indices: self.a.col_indices,
            values: &self.p_values,
            threads: self.a.threads,
        };
        // Cheap APPROXIMATE pressure solve — this is a preconditioner, and FGMRES
        // is flexible, so a loose inner solve (few iterations) keeps each outer
        // iteration cheap (mirrors the GPU's fixed smoother sweeps).
        // The BiCGSTAB paths run with the stagnation early-exit: bailing on a
        // stalled block beats burning the budget; failures feed the adaptive
        // AMG switch.
        work.psol.resize(cells, 0.0);
        let psol = &mut work.psol;
        let (gp_f32_buf, psol_f32_buf, hb_buf, hb32_buf) = (
            &mut work.gp_f32,
            &mut work.psol_f32,
            &mut work.hb,
            &mut work.hb32,
        );
        let gp: &[f64] = gp;
        let heavy_ball = self.amg.is_none() && self.inner != SchurInner::BiCgStab;
        let stats = prof::time(&prof::SCHUR_SOLVE, || match &self.amg {
            Some(amg) if self.inner == SchurInner::VCycle => {
                // Fixed-work apply: z_p = V-cycle(g_p) (vcycle overwrites).
                amg.vcycle(gp, psol);
                SolveStats { iters: 1, rel_residual: f64::NAN, converged: true }
            }
            Some(amg) => {
                gp_f32_buf.resize(cells, 0.0);
                psol_f32_buf.resize(cells, 0.0);
                for (o, &v) in gp_f32_buf.iter_mut().zip(gp.iter()) {
                    *o = v as f32;
                }
                psol_f32_buf.fill(0.0); // inner-solve x0 = 0, as the fresh alloc had
                let gp_f32: &[f32] = gp_f32_buf;
                let psol_f32: &mut [f32] = psol_f32_buf;
                // BiCGSTAB by default; PCG opt-in only (see the
                // `amg_use_cg` field doc for the measured refutation). On a
                // CG curvature breakdown the apply re-runs with BiCGSTAB so
                // FGMRES never sees the aborted iterate.
                let st = if self.amg_use_cg.get() {
                    let (st, spd_breakdown) = cg_pc_opts(
                        &pa,
                        gp_f32,
                        psol_f32,
                        self.inner_iters,
                        self.inner_tol,
                        &|r, z| amg.vcycle(r, z),
                        true,
                    );
                    if spd_breakdown {
                        self.amg_use_cg.set(false);
                        if std::env::var("CFD2_CPU_SCHUR_DEBUG").is_ok() {
                            eprintln!(
                                "[schur-inner] CG curvature breakdown at apply {} -> BiCGSTAB (one-way)",
                                self.applies.get() + 1
                            );
                        }
                        psol_f32.fill(0.0);
                        bicgstab_pc_opts(
                            &pa,
                            gp_f32,
                            psol_f32,
                            self.inner_iters,
                            self.inner_tol,
                            &|r, z| amg.vcycle(r, z),
                            true,
                        )
                    } else {
                        st
                    }
                } else {
                    bicgstab_pc_opts(
                        &pa,
                        gp_f32,
                        psol_f32,
                        self.inner_iters,
                        self.inner_tol,
                        &|r, z| amg.vcycle(r, z),
                        true,
                    )
                };
                for i in 0..cells {
                    psol[i] = psol_f32[i] as f64;
                }
                st
            }
            None if self.inner == SchurInner::BiCgStab => {
                let threads = pa.threads.max(1);
                let diag = pa.diagonal();
                let minv = move |v: &[f64], out: &mut [f64]| {
                    par_map_into(threads, out, |i| {
                        if diag[i] != 0.0 {
                            v[i] / diag[i]
                        } else {
                            v[i]
                        }
                    });
                };
                gp_f32_buf.resize(cells, 0.0);
                psol_f32_buf.resize(cells, 0.0);
                for (o, &v) in gp_f32_buf.iter_mut().zip(gp.iter()) {
                    *o = v as f32;
                }
                psol_f32_buf.fill(0.0); // inner-solve x0 = 0
                let st = bicgstab_pc_opts(
                    &pa,
                    gp_f32_buf,
                    psol_f32_buf,
                    self.inner_iters,
                    self.inner_tol,
                    &minv,
                    true,
                );
                for i in 0..cells {
                    psol[i] = psol_f32_buf[i] as f64;
                }
                st
            }
            None if self.a.simd => heavy_ball_solve_f32(
                &pa,
                gp,
                psol,
                &self.p_diag_inv_f32,
                self.omega,
                self.sweeps,
                self.inner_tol,
                hb32_buf,
            ),
            None => heavy_ball_solve(
                &pa,
                gp,
                psol,
                &self.p_diag_inv,
                self.omega,
                self.sweeps,
                self.inner_tol,
                hb_buf,
            ),
        });
        self.applies.set(self.applies.get() + 1);
        // Failure accounting drives the one-way AMG switch. Heavy-ball is a
        // fixed-sweep smoother (GPU semantics): missing `inner_tol` at the cap
        // is NORMAL on hard Poisson blocks (rate ~ sqrt(omega-1) per sweep ⇒
        // ~0.2 residual reduction at 63 sweeps) and FGMRES converges fine with
        // it — only a near-total stall (barely any reduction) means the block
        // genuinely needs the AMG hierarchy.
        let failed = if heavy_ball {
            !stats.converged && stats.rel_residual > 0.7
        } else {
            !stats.converged
        };
        if failed {
            self.inner_failures.set(self.inner_failures.get() + 1);
        }
        if std::env::var("CFD2_CPU_SCHUR_DEBUG").is_ok() {
            eprintln!(
                "[schur-inner] apply={} mode={} iters={} rel={:.3e} conv={}",
                self.applies.get(),
                if heavy_ball { "heavyball" } else { "bicgstab/amg" },
                stats.iters,
                stats.rel_residual,
                stats.converged
            );
        }
        prof::INNER_ITERS.fetch_add(stats.iters as u64, std::sync::atomic::Ordering::Relaxed);

        // 4. correct velocity, write pressure. Cell-disjoint writes into z;
        // parallel over cell chunks, bit-exact.
        prof::time(&prof::SCHUR_POST, || {
            if simd {
                // SIMD variant: the compact A_up run is contiguous per
                // neighbour, so all u_len corrections accumulate as one
                // padded f64x4 FMA against the broadcast psol[col].
                parallel_cell_chunks_mut(cells, s, aa.threads, z, |cell0, zc| {
                    use wide::f64x4;
                    for li in 0..zc.len() / s {
                        let cell = cell0 + li;
                        let scalar_offset = aa.scalar_offset(cell);
                        let num_neighbors = aa.num_neighbors(cell);
                        let mut corr_v = f64x4::splat(0.0);
                        for rank in 0..num_neighbors {
                            let col_cell = aa.col_indices[scalar_offset + rank] as usize;
                            let cbase = (scalar_offset + rank) * u_len;
                            let up_v = pad4_f32(&up_values[cbase..cbase + u_len]);
                            corr_v = up_v.mul_add(f64x4::splat(psol[col_cell]), corr_v);
                        }
                        let corr = corr_v.to_array();
                        for (i, &u) in u_idx.iter().enumerate() {
                            zc[li * s + u] -= diag_u_inv[cell * u_len + i] * corr[i];
                        }
                        zc[li * s + p] = psol[cell];
                    }
                });
            } else {
                parallel_cell_chunks_mut(cells, s, aa.threads, z, |cell0, zc| {
                    for li in 0..zc.len() / s {
                        let cell = cell0 + li;
                        let scalar_offset = aa.scalar_offset(cell);
                        let num_neighbors = aa.num_neighbors(cell);
                        for (i, &u) in u_idx.iter().enumerate() {
                            let mut corr = 0.0f64;
                            for rank in 0..num_neighbors {
                                let col_cell = aa.col_indices[scalar_offset + rank] as usize;
                                let a_up = up_values[(scalar_offset + rank) * u_len + i] as f64;
                                corr += a_up * psol[col_cell];
                            }
                            zc[li * s + u] -= diag_u_inv[cell * u_len + i] * corr;
                        }
                        zc[li * s + p] = psol[cell];
                    }
                });
            }
        });
    }
}

/// Restarted, flexible GMRES — FGMRES(`restart`) — over a block-CSR matrix with
/// a pluggable (possibly nonlinear/iterative) preconditioner. Mirrors the GPU's
/// FGMRES(60) so a variable preconditioner (the Schur complement smoother in
/// Phase 2) can be slotted in without breaking the Krylov recurrence. `x` is the
/// initial guess and receives the solution. Convergence is relative to
/// `rel_scale = min(||b||, ||r0||)` — the GPU `clamp_rel_scale` semantics.
/// The min matters for warm-started coupled solves whose RHS is dominated by
/// large ddt/BDF2 terms: there `||r0|| << ||b||`, and a plain `||b||` scale
/// declares convergence at zero iterations without computing any correction
/// (measured: the allmach_thermal obstacle case froze bit-exact at its
/// initial condition on the CPU backend — rel-to-b residual 6.5e-4 was
/// already under the EW first-outer 1e-2 — while the GPU, with the clamp,
/// evolved normally).
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

    let _ = simd;
    let vdot = |u: &[f64], w: &[f64]| prof::time(&prof::DOT, || par_dot(threads, u, w));
    let vnorm = |u: &[f64]| vdot(u, u).sqrt();
    let spmv = |x: &[f64], y: &mut [f64]| prof::time(&prof::SPMV, || a.block_spmv(x, y));

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
    let mut res;
    // GPU-parity relative scale: min(||b||, ||r0||), fixed at the FIRST true
    // residual (see the function docs).
    let mut rel_scale: Option<f64> = None;

    loop {
        // r0 = b - A x. This head residual doubles as the restart-cycle
        // convergence/budget guard, so each cycle costs exactly ONE true
        // residual evaluation (the historical shape recomputed it up to three
        // times per cycle: before the loop, at the head, and after the
        // restarted update).
        spmv(&xf, &mut ax);
        prof::time(&prof::AXPY, || par_map_into(threads, &mut r, |i| bf[i] - ax[i]));
        let beta = vnorm(&r);
        res = beta;
        let rs = *rel_scale.get_or_insert_with(|| bnorm.min(beta).max(1e-300));
        if beta / rs <= tol || total_iters >= max_iter {
            break;
        }
        let inv_beta = 1.0 / beta;
        if vbasis.is_empty() {
            vbasis.push(vec![0.0; n]);
        }
        prof::time(&prof::AXPY, || {
            par_map_into(threads, &mut vbasis[0], |i| r[i] * inv_beta)
        });
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
            prof::time(&prof::PC_APPLY, || precond.apply(&vbasis[j], &mut zbasis[j]));
            spmv(&zbasis[j], &mut ax);
            w.copy_from_slice(&ax);
            // Modified Gram-Schmidt against v_0..v_j. The dot goes through
            // `par_dot` (fixed-chunk deterministic reduction — thread-count
            // invariant); the axpy update is elementwise, so it parallelizes
            // bit-exactly.
            for i in 0..=j {
                let hij = vdot(&w, &vbasis[i]);
                h[j][i] = hij;
                let vi = &vbasis[i];
                prof::time(&prof::AXPY, || {
                    par_update(threads, &mut w, |k, wk| *wk -= hij * vi[k])
                });
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
            if res / rs <= tol || hnext < 1e-300 || total_iters >= max_iter {
                break;
            }
            let inv_h = 1.0 / hnext;
            if vbasis.len() <= j + 1 {
                vbasis.push(vec![0.0; n]);
            }
            prof::time(&prof::AXPY, || {
                par_map_into(threads, &mut vbasis[j + 1], |k| w[k] * inv_h)
            });
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
                prof::time(&prof::AXPY, || {
                    par_update(threads, &mut xf, |k, xk| *xk += yj * zj[k])
                });
            }
        }
        // Loop back: the head recomputes the true residual at the restarted
        // iterate and applies the convergence/budget guard.
    }

    prof::FGMRES_ITERS.fetch_add(total_iters as u64, std::sync::atomic::Ordering::Relaxed);
    for i in 0..n {
        x[i] = xf[i] as f32;
    }
    let rs = rel_scale.unwrap_or(bnorm);
    SolveStats {
        iters: total_iters,
        rel_residual: res / rs,
        converged: res / rs <= tol,
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

/// Solve `A x = b` with Jacobi-preconditioned BiCGSTAB. `x` is used as the
/// initial guess and overwritten with the solution. Returns convergence
/// statistics.
pub fn bicgstab(
    a: &CsrView,
    b: &[f32],
    x: &mut [f32],
    max_iter: usize,
    tol: f64,
    simd: bool,
) -> SolveStats {
    let _ = simd;
    let threads = a.threads.max(1);
    let diag = a.diagonal();
    let minv = move |v: &[f64], out: &mut [f64]| {
        // Jacobi; guard a (pathological) zero diagonal. Elementwise → parallel
        // bit-exactly.
        par_map_into(threads, out, |i| if diag[i] != 0.0 { v[i] / diag[i] } else { v[i] });
    };
    bicgstab_pc(a, b, x, max_iter, tol, &minv)
}

/// Fused heavy-ball (second-order Richardson) sweeps on a scalar CSR system —
/// the CPU mirror of the GPU Schur `relax_pressure` ping-pong:
///   x_{k+1}[i] = (1-w)·x_{k-1}[i] + w·(x_k[i] + (g[i] - Σ_j A_ij·x_k[j]) / A_ii)
/// starting from x_{-1} = x_0 = 0 (both GPU relax buffers are zeroed).
///
/// Each sweep is ONE parallel pass (residual, diagonal scale and momentum
/// update fused per row) with no reductions — the whole point: the inner solve
/// at bench sizes is scoped-thread launch-overhead-bound, and BiCGSTAB pays
/// ~13 launches per iteration. The residual is measured only at geometrically
/// spaced sweeps (4, 8, 16, …, max_sweeps) so easy blocks (the all-Mach
/// diagonal-boosted pressure row) exit after a few sweeps while hard Poisson
/// blocks run the full budget as a FIXED linear operator — which is exactly
/// what FGMRES wants from a preconditioner.
///
/// Deterministic: sweeps write each row once from read-only inputs (ping-pong
/// buffers), and the exit decision comes from `par_dot` — bit-identical across
/// thread counts.
#[allow(clippy::too_many_arguments)]
fn heavy_ball_solve(
    pa: &CsrView,
    g: &[f64],
    x: &mut [f64],
    diag_inv: &[f64],
    omega: f64,
    max_sweeps: usize,
    tol: f64,
    work: &mut Vec<f64>,
) -> SolveStats {
    let n = pa.n();
    debug_assert_eq!(g.len(), n);
    debug_assert_eq!(x.len(), n);
    let threads = pa.threads.max(1);
    let gnorm = par_dot(threads, g, g).sqrt().max(1e-300);

    // Ping-pong pair: `cur` holds x_k, `prev` holds x_{k-1} and RECEIVES
    // x_{k+1} (its own element is read for the momentum term before being
    // overwritten — element-local, so the chunked write stays race-free;
    // neighbour reads touch only `cur`).
    //
    // The four sweep buffers live in the caller's reusable workspace
    // (allocating ~4n per apply measured as ~24 MB/apply x ~50 applies/step
    // of alloc + page-fault churn on the 750k nozzle); the explicit zero
    // fills reproduce the fresh-alloc from-zero start bit-exactly.
    work.resize(4 * n, 0.0);
    work.fill(0.0);
    let (mut cur, rest) = work.split_at_mut(n);
    let (mut prev, rest) = rest.split_at_mut(n);
    let (scratch, best) = rest.split_at_mut(n);

    // Safeguard state: the GPU runs these sweeps blind (no readbacks), but on
    // the CPU a residual check is cheap, so we use it to make high-omega
    // momentum SAFE. The heavy-ball stability ellipse collapses onto the real
    // axis as omega -> 2; pressure blocks with an upwinded `div_flux(phi, p)`
    // (slightly complex spectrum) or near-null gauge modes can be AMPLIFIED at
    // omega 1.95 even though plain Jacobi converges — and once one apply
    // amplifies, FGMRES feeds the unstable modes right back (measured: apply 1
    // rel 0.56, applies 2+ rel ~22 on the cut-cell obstacle). On growth vs the
    // best iterate: halve the momentum and restart from the best; after
    // repeated decays bail with the best iterate (feeds the AMG switch).
    let mut best_rel = 1.0f64; // x = 0 has relative residual exactly 1
    let mut omega = omega;

    let mut sweeps_done = 0usize;
    let mut next_check = 4usize.min(max_sweeps);
    let mut strikes = 0u32;
    let mut decays = 0u32;
    let mut converged = false;
    while sweeps_done < max_sweeps {
        {
            let cur_ref: &[f64] = cur;
            parallel_cell_chunks_mut(n, 1, threads, prev, |row0, chunk| {
                for (li, slot) in chunk.iter_mut().enumerate() {
                    let row = row0 + li;
                    let start = pa.row_offsets[row] as usize;
                    let end = pa.row_offsets[row + 1] as usize;
                    let mut sum = 0.0f64;
                    for k in start..end {
                        sum += pa.values[k] as f64 * cur_ref[pa.col_indices[k] as usize];
                    }
                    let hat = cur_ref[row] + (g[row] - sum) * diag_inv[row];
                    *slot = (1.0 - omega) * *slot + omega * hat;
                }
            });
        }
        std::mem::swap(&mut cur, &mut prev);
        sweeps_done += 1;
        if sweeps_done == next_check || sweeps_done == max_sweeps {
            pa.spmv(cur, scratch);
            par_update(threads, scratch, |i, v| *v = g[i] - *v);
            let rel = par_dot(threads, scratch, scratch).sqrt() / gnorm;
            if rel <= tol {
                converged = true;
                best_rel = rel;
                best.copy_from_slice(cur);
                break;
            }
            if rel < best_rel {
                best_rel = rel;
                best.copy_from_slice(cur);
                strikes = 0;
            } else {
                // Non-improving check. Heavy-ball residuals overshoot
                // TRANSIENTLY (non-normal iteration matrix), so a single bad
                // check is normal; only SUSTAINED growth (two consecutive
                // checks without a new best) means momentum is amplifying part
                // of the spectrum. Then: halve the momentum and restart from
                // the best iterate; after two decays give up and return the
                // best (the failure feeds the caller's adaptive AMG switch).
                strikes += 1;
                if strikes >= 2 {
                    strikes = 0;
                    decays += 1;
                    if decays >= 2 {
                        break;
                    }
                    omega = 1.0 + (omega - 1.0) * 0.5;
                    cur.copy_from_slice(best);
                    prev.copy_from_slice(best);
                }
            }
            next_check = (next_check * 2).min(max_sweeps);
        }
    }
    x.copy_from_slice(best);
    SolveStats { iters: sweeps_done, rel_residual: best_rel, converged }
}

/// Deterministic parallel dot over f32 slices with f64 accumulation: the
/// mixed-precision twin of `par_dot` (same fixed 8192-element chunking and
/// serial chunk-order reduction, so the result is thread-count invariant).
pub fn par_dot_f32(threads: usize, a: &[f32], b: &[f32]) -> f64 {
    const DOT_CHUNK: usize = 8192;
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let nchunks = n.div_ceil(DOT_CHUNK).max(1);
    let chunk_partial = |c: usize| -> f64 {
        let s = c * DOT_CHUNK;
        let e = (s + DOT_CHUNK).min(n);
        let (a, b) = (&a[s..e], &b[s..e]);
        use wide::f64x4;
        let mut acc = f64x4::splat(0.0);
        let mut i = 0;
        while i + 4 <= a.len() {
            let va = f64x4::from([a[i] as f64, a[i + 1] as f64, a[i + 2] as f64, a[i + 3] as f64]);
            let vb = f64x4::from([b[i] as f64, b[i + 1] as f64, b[i + 2] as f64, b[i + 3] as f64]);
            acc = va.mul_add(vb, acc);
            i += 4;
        }
        let l = acc.to_array();
        let mut sum = l[0] + l[1] + l[2] + l[3];
        while i < a.len() {
            sum += a[i] as f64 * b[i] as f64;
            i += 1;
        }
        sum
    };
    let workers = crate::solver::cpu::parallel::par_dot_workers(threads, n);
    if workers <= 1 || nchunks == 1 {
        return (0..nchunks).map(chunk_partial).sum();
    }
    let mut partials = vec![0.0f64; nchunks];
    let per = nchunks.div_ceil(workers * crate::solver::cpu::pool::OVERSPLIT).max(1);
    let tasks = nchunks.div_ceil(per);
    let base = crate::solver::cpu::pool::MutSlicePtr::new(&mut partials);
    crate::solver::cpu::pool::run(tasks, workers, |w| {
        let c0 = w * per;
        let c1 = (c0 + per).min(nchunks);
        // SAFETY: disjoint partial ranges per task; `partials` outlives run.
        let head = unsafe { base.slice(c0, c1 - c0) };
        for (li, o) in head.iter_mut().enumerate() {
            *o = chunk_partial(c0 + li);
        }
    });
    partials.iter().sum()
}

/// Zero-padded f64x4 load from an up-to-4-long f32 slice (SIMD Schur lanes).
#[inline]
fn pad4_f32(v: &[f32]) -> wide::f64x4 {
    let mut l = [0.0f64; 4];
    for (o, &x) in l.iter_mut().zip(v.iter()) {
        *o = x as f64;
    }
    wide::f64x4::from(l)
}

/// Mixed-precision heavy-ball inner solve (the SIMD/mixed-precision option):
/// identical iteration to [`heavy_ball_solve`] but with the sweep state
/// (`cur`/`prev`/`scratch`/`best`) and RHS in f32 storage — the sweep is
/// memory-BANDWIDTH-bound (measured ~54 MB/sweep at ~74 GB/s on the 750k
/// nozzle), so halving the vector bytes is the lever f64 SIMD arithmetic
/// could not be. The inner tolerance is 1e-1 (a preconditioner apply under
/// flexible FGMRES), leaving orders of magnitude of accuracy budget; the
/// residual-check norms accumulate in f64 via the deterministic
/// [`par_dot_f32`], and the safeguard logic is unchanged.
#[allow(clippy::too_many_arguments)]
fn heavy_ball_solve_f32(
    pa: &CsrView,
    g: &[f64],
    x: &mut [f64],
    diag_inv: &[f32],
    omega: f64,
    max_sweeps: usize,
    tol: f64,
    work: &mut Vec<f32>,
) -> SolveStats {
    let n = pa.n();
    debug_assert_eq!(g.len(), n);
    debug_assert_eq!(x.len(), n);
    let threads = pa.threads.max(1);
    let gnorm = par_dot(threads, g, g).sqrt().max(1e-300);

    work.resize(5 * n, 0.0);
    work.fill(0.0);
    let (mut cur, rest) = work.split_at_mut(n);
    let (mut prev, rest) = rest.split_at_mut(n);
    let (scratch, rest) = rest.split_at_mut(n);
    let (best, g32) = rest.split_at_mut(n);
    parallel_cell_chunks_mut(n, 1, threads, g32, |i0, chunk| {
        for (li, o) in chunk.iter_mut().enumerate() {
            *o = g[i0 + li] as f32;
        }
    });
    let g32: &[f32] = g32;

    let mut best_rel = 1.0f64;
    let mut omega = omega;
    let mut sweeps_done = 0usize;
    let mut next_check = 4usize.min(max_sweeps);
    let mut strikes = 0u32;
    let mut decays = 0u32;
    let mut converged = false;
    while sweeps_done < max_sweeps {
        {
            let cur_ref: &[f32] = cur;
            let om = omega as f32;
            parallel_cell_chunks_mut(n, 1, threads, prev, |row0, chunk| {
                for (li, slot) in chunk.iter_mut().enumerate() {
                    let row = row0 + li;
                    let start = pa.row_offsets[row] as usize;
                    let end = pa.row_offsets[row + 1] as usize;
                    let mut sum = 0.0f32;
                    for k in start..end {
                        sum += pa.values[k] * cur_ref[pa.col_indices[k] as usize];
                    }
                    let hat = cur_ref[row] + (g32[row] - sum) * diag_inv[row];
                    *slot = (1.0 - om) * *slot + om * hat;
                }
            });
        }
        std::mem::swap(&mut cur, &mut prev);
        sweeps_done += 1;
        if sweeps_done == next_check || sweeps_done == max_sweeps {
            {
                let cur_ref: &[f32] = cur;
                parallel_cell_chunks_mut(n, 1, threads, scratch, |row0, chunk| {
                    for (li, slot) in chunk.iter_mut().enumerate() {
                        let row = row0 + li;
                        let start = pa.row_offsets[row] as usize;
                        let end = pa.row_offsets[row + 1] as usize;
                        let mut sum = 0.0f32;
                        for k in start..end {
                            sum += pa.values[k] * cur_ref[pa.col_indices[k] as usize];
                        }
                        *slot = g32[row] - sum;
                    }
                });
            }
            let rel = par_dot_f32(threads, scratch, scratch).sqrt() / gnorm;
            if rel <= tol {
                converged = true;
                best_rel = rel;
                best.copy_from_slice(cur);
                break;
            }
            if rel < best_rel {
                best_rel = rel;
                best.copy_from_slice(cur);
                strikes = 0;
            } else {
                strikes += 1;
                if strikes >= 2 {
                    strikes = 0;
                    decays += 1;
                    if decays >= 2 {
                        break;
                    }
                    omega = 1.0 + (omega - 1.0) * 0.5;
                    cur.copy_from_slice(best);
                    prev.copy_from_slice(best);
                }
            }
            next_check = (next_check * 2).min(max_sweeps);
        }
    }
    let best: &[f32] = best;
    parallel_cell_chunks_mut(n, 1, threads, x, |i0, chunk| {
        for (li, o) in chunk.iter_mut().enumerate() {
            *o = best[i0 + li] as f64;
        }
    });
    SolveStats { iters: sweeps_done, rel_residual: best_rel, converged }
}

/// BiCGSTAB with a caller-supplied left preconditioner `minv(r, z)` (e.g. the
/// AMG V-cycle for the Schur pressure block).
pub fn bicgstab_pc(
    a: &CsrView,
    b: &[f32],
    x: &mut [f32],
    max_iter: usize,
    tol: f64,
    minv: &dyn Fn(&[f64], &mut [f64]),
) -> SolveStats {
    bicgstab_pc_opts(a, b, x, max_iter, tol, minv, false)
}

/// [`bicgstab_pc`] with a stagnation early-exit option. When `stagnation_exit`
/// is set and the residual stops improving (two consecutive iterations with
/// less than 2% reduction over the best seen), the solve returns early with
/// the current iterate. ONLY safe for preconditioner-apply usage (the Schur
/// inner solve): the caller (flexible FGMRES) tolerates an approximate z, and
/// bailing beats burning the full budget on an unconvergeable block (e.g. the
/// from-rest step-0 pressure system). Deterministic: the residual norms that
/// drive the exit come from `par_dot`.
#[allow(clippy::too_many_arguments)]
pub fn bicgstab_pc_opts(
    a: &CsrView,
    b: &[f32],
    x: &mut [f32],
    max_iter: usize,
    tol: f64,
    minv: &dyn Fn(&[f64], &mut [f64]),
    stagnation_exit: bool,
) -> SolveStats {
    let n = a.n();
    assert_eq!(b.len(), n);
    assert_eq!(x.len(), n);
    let threads = a.threads.max(1);

    let vdot = |a: &[f64], b: &[f64]| par_dot(threads, a, b);
    let vnorm = |a: &[f64]| vdot(a, a).sqrt();

    let bf: Vec<f64> = b.iter().map(|&v| v as f64).collect();
    let bnorm = vnorm(&bf).max(1e-300);

    let mut xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();

    // r = b - A x
    let mut ax = vec![0.0f64; n];
    a.spmv(&xf, &mut ax);
    let mut r = vec![0.0f64; n];
    par_map_into(threads, &mut r, |i| bf[i] - ax[i]);

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
    // Hoisted s-vector (was a fresh allocation every iteration).
    let mut sv = vec![0.0f64; n];
    // Stagnation tracking (see `stagnation_exit`).
    let mut best_res = res;
    let mut stalled = 0u32;

    for iter in 1..=max_iter {
        let rho_new = vdot(&rhat, &r);
        if rho_new.abs() < 1e-300 {
            // Breakdown; restart from the current residual.
            return finish(&xf, x, iter, res, res / bnorm <= tol);
        }
        let beta = (rho_new / rho) * (alpha / omega);
        {
            let (r, v) = (&r, &v);
            par_update(threads, &mut p, |i, pi| *pi = r[i] + beta * (*pi - omega * v[i]));
        }
        minv(&p, &mut phat);
        a.spmv(&phat, &mut v);
        let rhat_v = vdot(&rhat, &v);
        alpha = rho_new / rhat_v;

        // s = r - alpha v
        {
            let (r, v) = (&r, &v);
            par_map_into(threads, &mut sv, |i| r[i] - alpha * v[i]);
        }
        let snorm = vnorm(&sv);
        if snorm / bnorm <= tol {
            let phat = &phat;
            par_update(threads, &mut xf, |i, xi| *xi += alpha * phat[i]);
            return finish(&xf, x, iter, snorm, true);
        }

        minv(&sv, &mut shat);
        a.spmv(&shat, &mut t);
        let tt = vdot(&t, &t).max(1e-300);
        omega = vdot(&t, &sv) / tt;

        {
            let (phat, shat) = (&phat, &shat);
            par_update(threads, &mut xf, |i, xi| *xi += alpha * phat[i] + omega * shat[i]);
        }
        {
            let (sv, t) = (&sv, &t);
            par_map_into(threads, &mut r, |i| sv[i] - omega * t[i]);
        }
        res = vnorm(&r);
        if res / bnorm <= tol {
            return finish(&xf, x, iter, res, true);
        }
        if omega.abs() < 1e-300 {
            return finish(&xf, x, iter, res, res / bnorm <= tol);
        }
        if stagnation_exit {
            if res > 0.98 * best_res {
                stalled += 1;
                if stalled >= 2 {
                    return finish(&xf, x, iter, res, false);
                }
            } else {
                stalled = 0;
            }
            best_res = best_res.min(res);
        }
        rho = rho_new;
    }

    finish(&xf, x, max_iter, res, res / bnorm <= tol)
}

/// Preconditioned conjugate gradient with the same calling convention as
/// [`bicgstab_pc_opts`] (f32 in/out, f64 internals, deterministic `par_dot`
/// reductions, stagnation exit). Per iteration: 1 spmv + 1 preconditioner
/// apply + 3 dots + 3 fused updates — HALF the spmvs/preconditioner applies
/// (and most of the parallel-region launches) of a BiCGSTAB iteration, at
/// comparable convergence per matvec on the near-SPD pressure block with the
/// symmetric V(1,1) damped-Jacobi AMG cycle as `minv`.
///
/// Returns `(stats, spd_breakdown)`: `spd_breakdown = true` means a
/// non-positive curvature `<p, A p>` was met — the block is materially
/// non-symmetric/indefinite along the search direction and the caller should
/// fall back to BiCGSTAB (one-way, mirroring the adaptive-switch style).
pub fn cg_pc_opts(
    a: &CsrView,
    b: &[f32],
    x: &mut [f32],
    max_iter: usize,
    tol: f64,
    minv: &dyn Fn(&[f64], &mut [f64]),
    stagnation_exit: bool,
) -> (SolveStats, bool) {
    let n = a.n();
    assert_eq!(b.len(), n);
    assert_eq!(x.len(), n);
    let threads = a.threads.max(1);

    let vdot = |a: &[f64], b: &[f64]| par_dot(threads, a, b);
    let vnorm = |a: &[f64]| vdot(a, a).sqrt();

    let bf: Vec<f64> = b.iter().map(|&v| v as f64).collect();
    let bnorm = vnorm(&bf).max(1e-300);
    let mut xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();

    // r = b - A x
    let mut ax = vec![0.0f64; n];
    a.spmv(&xf, &mut ax);
    let mut r = vec![0.0f64; n];
    par_map_into(threads, &mut r, |i| bf[i] - ax[i]);

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
        return (finish(&xf, x, 0, res, true), false);
    }

    let mut z = vec![0.0f64; n];
    minv(&r, &mut z);
    let mut p = z.clone();
    let mut ap = vec![0.0f64; n];
    let mut rz = vdot(&r, &z);
    let mut best_res = res;
    let mut stalled = 0u32;

    for iter in 1..=max_iter {
        a.spmv(&p, &mut ap);
        let pap = vdot(&p, &ap);
        if !(pap > 0.0) || !rz.is_finite() {
            // Non-SPD curvature (or numeric junk): hand back the best-effort
            // iterate and tell the caller to switch algorithms.
            return (finish(&xf, x, iter, res, res / bnorm <= tol), true);
        }
        let alpha = rz / pap;
        {
            let p = &p;
            par_update(threads, &mut xf, |i, xi| *xi += alpha * p[i]);
        }
        {
            let ap = &ap;
            par_update(threads, &mut r, |i, ri| *ri -= alpha * ap[i]);
        }
        res = vnorm(&r);
        if res / bnorm <= tol {
            return (finish(&xf, x, iter, res, true), false);
        }
        if stagnation_exit {
            if res > 0.98 * best_res {
                stalled += 1;
                if stalled >= 2 {
                    return (finish(&xf, x, iter, res, false), false);
                }
            } else {
                stalled = 0;
            }
            best_res = best_res.min(res);
        }
        minv(&r, &mut z);
        let rz_new = vdot(&r, &z);
        let beta = rz_new / rz;
        {
            let z = &z;
            par_update(threads, &mut p, |i, pi| *pi = z[i] + beta * *pi);
        }
        rz = rz_new;
    }

    (finish(&xf, x, max_iter, res, res / bnorm <= tol), false)
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
                simd: false,
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
    fn block_spmv_simd_matches_scalar() {
        // The explicit-SIMD kernels (S = 3, 4) change only the per-row
        // summation ORDER, so they must match the scalar loop to f64
        // rounding accuracy — and stay bit-identical across thread counts.
        for s in [3usize, 4] {
            let nc = 257; // uneven vs chunk sizes
            let (sro, col, diag, vals) = banded_block_system(nc, s);
            let n = nc * s;
            let x: Vec<f64> = (0..n).map(|k| ((k * 11 % 17) as f64) * 0.31 - 2.0).collect();
            let run = |simd: bool, threads: usize| {
                let a = BlockCsr {
                    s,
                    scalar_row_offsets: &sro,
                    col_indices: &col,
                    diagonal_indices: &diag,
                    values: &vals,
                    threads,
                    simd,
                };
                let mut y = vec![0.0f64; n];
                a.block_spmv(&x, &mut y);
                y
            };
            let y_scalar = run(false, 1);
            let y_simd = run(true, 1);
            let scale = y_scalar.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1.0);
            let maxd = y_scalar
                .iter()
                .zip(&y_simd)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            assert!(
                maxd / scale < 1e-13,
                "S={s}: simd vs scalar rel diff {:.3e}",
                maxd / scale
            );
            let y_simd4 = run(true, 4);
            assert_eq!(
                y_simd.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                y_simd4.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "S={s}: SIMD path must be bit-identical across thread counts"
            );
        }
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
            simd: false,
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
            simd: false,
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
            simd: false,
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
            simd: false,
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
