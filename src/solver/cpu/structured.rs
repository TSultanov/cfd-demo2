//! Structured (uniform Cartesian, dense-array) CPU solver path.
//!
//! This is the runtime counterpart of the `TopologyMode::Structured2D` codegen
//! branch. Where [`super::solver::CpuSolver`] builds an unstructured
//! face-connectivity mesh, a sorted-adjacency block CSR and the full FGMRES/AMG
//! solver stack, this path represents the grid as a *dense array with no
//! connectivity indirection*: the field is `state[j*nx + i]`, neighbours are
//! `p±1` / `p±nx`, and the assembled operator is a fixed 5-point **band**
//! (`matrix_values` sized `N*5`, ranks `[S, W, diag, E, N]`).
//!
//! The assembly and update kernels are the *same* codegen-emitted
//! [`KernelProgram`]s the GPU path uses — interpreted here over the dense index
//! space — so the operator expansion is genuinely produced by the IR, not
//! hand-written. The only bespoke numerics is the matrix-free banded conjugate
//! gradient that replaces the CSR Krylov stack (the linear system for
//! `ddt + laplacian` diffusion is symmetric positive-definite).
//!
//! Obstacles are immersed, never cut out of the grid (see the Brinkman
//! penalisation path); a structured grid cannot delete a cell.

use std::collections::HashMap;

use crate::solver::cpu::interpreter::{Buffers, Ctx, Frame, Interpreter, Value};
use crate::solver::cpu::lowering::model_kernel_programs;
use crate::solver::gpu::recipe::{KernelPhase, SolverRecipe, SteppingMode};
use crate::solver::gpu::structs::GpuConstants;
use crate::solver::model::backend::SchemeRegistry;
use crate::solver::model::ModelSpec;
use crate::solver::scheme::Scheme;
use crate::solver::{PreconditionerType, TimeScheme};
use cfd2_ir::ast::Stmt;

/// Band ranks in the fixed 5-point stencil row (ascending column order for the
/// row-major cell numbering `p = j*nx + i`: `p-nx < p-1 < p < p+1 < p+nx`).
const BAND_SOUTH: usize = 0;
const BAND_WEST: usize = 1;
const BAND_DIAG: usize = 2;
const BAND_EAST: usize = 3;
const BAND_NORTH: usize = 4;
const BAND_STRIDE: usize = 5;

/// A uniform Cartesian grid stored implicitly: `nx * ny` cells, spacing
/// `(dx, dy)`, cell `p = j*nx + i` centred at `((i+0.5)dx, (j+0.5)dy)`. Carries
/// no per-cell/face arrays — everything downstream is index arithmetic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StructuredGrid {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,
}

impl StructuredGrid {
    /// A grid spanning `[0, length] x [0, height]` with `nx * ny` cells.
    pub fn new(nx: usize, ny: usize, length: f64, height: f64) -> Self {
        assert!(nx > 0 && ny > 0, "grid must have at least one cell per axis");
        assert!(length > 0.0 && height > 0.0, "grid extents must be positive");
        Self {
            nx,
            ny,
            dx: length / nx as f64,
            dy: height / ny as f64,
        }
    }

    #[inline]
    pub fn num_cells(&self) -> usize {
        self.nx * self.ny
    }

    /// Cell centre coordinates for linear index `p = j*nx + i`.
    #[inline]
    pub fn cell_center(&self, p: usize) -> (f64, f64) {
        let i = p % self.nx;
        let j = p / self.nx;
        ((i as f64 + 0.5) * self.dx, (j as f64 + 0.5) * self.dy)
    }
}

/// Which domain edge a boundary face sits on (used to key the per-face BC).
#[derive(Clone, Copy, Debug)]
pub enum Edge {
    Left,
    Right,
    Bottom,
    Top,
}

/// A structured-grid CPU solver: drives the codegen `Structured2D` assembly +
/// update kernels through the interpreter and closes the loop with a matrix-free
/// banded CG.
pub struct StructuredCpuSolver {
    grid: StructuredGrid,
    /// The solved scalar's stride within the state layout (state is
    /// `[phi, ...]`; phi lives at component 0). Equals 1 for the plain diffusion
    /// demo, 2 when an MMS source field is packed alongside.
    state_stride: usize,
    assembly_stmts: Vec<Stmt>,
    update_stmts: Vec<Stmt>,
    buffers: Buffers,
    dt: f64,
    threads: usize,
}

impl StructuredCpuSolver {
    /// Build a structured solver for `model` (which must have been declared with
    /// `TopologyMode::Structured2D`) on `grid`.
    pub fn new(grid: StructuredGrid, model: &ModelSpec, dt: f64) -> Result<Self, String> {
        let schemes = SchemeRegistry::new(Scheme::Upwind);
        let (programs, _wgsl_only) = model_kernel_programs(model, &schemes)?;
        let program_map: HashMap<String, _> = programs
            .into_iter()
            .map(|(id, p)| (id.as_str().to_string(), p))
            .collect();

        let extract = |id: &str| -> Result<Vec<Stmt>, String> {
            let p = program_map
                .get(id)
                .ok_or_else(|| format!("structured model missing kernel `{id}`"))?;
            let mut stmts =
                Vec::with_capacity(p.indexing.len() + p.preamble.len() + p.body.len());
            stmts.extend_from_slice(&p.indexing);
            stmts.extend_from_slice(&p.preamble);
            stmts.extend_from_slice(&p.body);
            Ok(stmts)
        };
        // `RequiresNoGradState` diffusion selects the plain (non-grad) assembly.
        let assembly_stmts = extract("generic_coupled_assembly")?;
        let update_stmts = extract("generic_coupled_update")?;

        let state_stride = model.state_layout.stride() as usize;
        let n = grid.num_cells();

        let mut buffers = Buffers::new();
        buffers.insert_f32("state", vec![0.0; n * state_stride]);
        buffers.insert_f32("state_old", vec![0.0; n * state_stride]);
        buffers.insert_f32("state_old_old", vec![0.0; n * state_stride]);
        buffers.insert_f32("state_iter", vec![0.0; n * state_stride]);
        // Banded operator: one 1x1 block per stencil slot, 5 slots per cell.
        buffers.insert_f32("matrix_values", vec![0.0; n * BAND_STRIDE]);
        buffers.insert_f32("rhs", vec![0.0; n]);
        buffers.insert_f32("x", vec![0.0; n]);
        // BC table is keyed by the per-(cell, direction) face id `idx*4 + k`.
        buffers.insert_u32("bc_kind", vec![0u32; n * 4]);
        buffers.insert_f32("bc_value", vec![0.0; n * 4]);

        Ok(Self {
            grid,
            state_stride,
            assembly_stmts,
            update_stmts,
            buffers,
            dt,
            threads: 1,
        })
    }

    /// Seed the solved scalar from a closure of the cell-centre coordinates.
    pub fn set_scalar<F: Fn(f64, f64) -> f64>(&mut self, f: F) {
        let n = self.grid.num_cells();
        for p in 0..n {
            let (x, y) = self.grid.cell_center(p);
            let v = f(x, y) as f32;
            self.buffers.set_f32("state", p * self.state_stride, v);
            self.buffers.set_f32("state_old", p * self.state_stride, v);
            self.buffers
                .set_f32("state_old_old", p * self.state_stride, v);
        }
    }

    /// Seed a packed auxiliary state field (e.g. an MMS source) at `component`.
    pub fn set_state_component<F: Fn(f64, f64) -> f64>(&mut self, component: usize, f: F) {
        let n = self.grid.num_cells();
        for p in 0..n {
            let (x, y) = self.grid.cell_center(p);
            let v = f(x, y) as f32;
            self.buffers
                .set_f32("state", p * self.state_stride + component, v);
            self.buffers
                .set_f32("state_old", p * self.state_stride + component, v);
            self.buffers
                .set_f32("state_old_old", p * self.state_stride + component, v);
        }
    }

    /// Impose a Dirichlet value on every boundary face, from a closure of the
    /// FACE-centre coordinates (the ghost sits on the domain edge, half a cell
    /// outside the boundary cell centre).
    pub fn set_dirichlet<F: Fn(f64, f64) -> f64>(&mut self, f: F) {
        let (nx, ny) = (self.grid.nx, self.grid.ny);
        let (dx, dy) = (self.grid.dx, self.grid.dy);
        let n = self.grid.num_cells();
        let mut bc_kind = vec![0u32; n * 4];
        let mut bc_value = vec![0.0f32; n * 4];
        for j in 0..ny {
            for i in 0..nx {
                let p = j * nx + i;
                let (cx, cy) = self.grid.cell_center(p);
                // Faces in the codegen order: S=0, W=1, E=2, N=3.
                let faces = [
                    (BAND_SOUTH_FACE, j == 0, cx, cy - 0.5 * dy),
                    (BAND_WEST_FACE, i == 0, cx - 0.5 * dx, cy),
                    (BAND_EAST_FACE, i == nx - 1, cx + 0.5 * dx, cy),
                    (BAND_NORTH_FACE, j == ny - 1, cx, cy + 0.5 * dy),
                ];
                for (k, is_boundary, fx, fy) in faces {
                    if is_boundary {
                        let fid = p * 4 + k;
                        bc_kind[fid] = 1; // GpuBcKind::Dirichlet
                        bc_value[fid] = f(fx, fy) as f32;
                    }
                }
            }
        }
        self.buffers.insert_u32("bc_kind", bc_kind);
        self.buffers.insert_f32("bc_value", bc_value);
    }

    /// Run one implicit (backward-Euler) time step: advance history, assemble the
    /// banded operator with the interpreted structured kernel, solve, and apply.
    /// Returns the L-infinity change in the solved scalar.
    pub fn step(&mut self) -> f64 {
        let n = self.grid.num_cells();
        let stride = self.state_stride;

        // Advance history (Euler uses state_old; keep old_old coherent for BDF2).
        let state = self.buffers.f32_vec("state");
        let old = self.buffers.f32_vec("state_old");
        self.buffers.copy_into_f32("state_old_old", &old);
        self.buffers.copy_into_f32("state_old", &state);

        // Assemble A (banded, N*5) and b (rhs, N) by interpreting the structured
        // assembly kernel over every cell.
        let ctx = self.build_ctx();
        run_over_cells(&self.buffers, &ctx, &self.assembly_stmts, n, self.threads);

        // Solve A x = b (symmetric SPD) matrix-free over the 5 bands.
        let a = self.buffers.f32_vec("matrix_values");
        let b = self.buffers.f32_vec("rhs");
        let x = banded_cg(&a, &b, self.grid.nx, self.grid.ny, 1e-12, 5000);
        self.buffers.copy_into_f32("x", &x);

        // Apply: state = x (the codegen update kernel, per-cell).
        run_over_cells(&self.buffers, &ctx, &self.update_stmts, n, self.threads);

        // Report convergence in the solved component.
        let new_state = self.buffers.f32_vec("state");
        let mut linf = 0.0f64;
        for p in 0..n {
            let d = (new_state[p * stride] - state[p * stride]).abs() as f64;
            if d > linf {
                linf = d;
            }
        }
        linf
    }

    /// Iterate [`step`](Self::step) until the L-infinity change falls below `tol`
    /// (steady state) or `max_steps` is reached. Returns the step count.
    pub fn solve_to_steady(&mut self, tol: f64, max_steps: usize) -> usize {
        for s in 1..=max_steps {
            if self.step() < tol {
                return s;
            }
        }
        max_steps
    }

    /// Read the solved scalar field (length `nx*ny`, row-major `p = j*nx+i`).
    pub fn scalar_field(&self) -> Vec<f64> {
        let n = self.grid.num_cells();
        let state = self.buffers.f32_vec("state");
        (0..n).map(|p| state[p * self.state_stride] as f64).collect()
    }

    /// Read the assembled banded operator (`N*5`, ranks `[S,W,diag,E,N]`).
    pub fn matrix_bands(&self) -> Vec<f32> {
        self.buffers.f32_vec("matrix_values")
    }

    /// Assemble once (advancing history) without solving — for operator tests.
    pub fn assemble_only(&mut self) {
        let n = self.grid.num_cells();
        let state = self.buffers.f32_vec("state");
        self.buffers.copy_into_f32("state_old", &state);
        let ctx = self.build_ctx();
        run_over_cells(&self.buffers, &ctx, &self.assembly_stmts, n, self.threads);
    }

    fn build_ctx(&self) -> Ctx {
        Ctx::new()
            .with_constant("grid", "nx", Value::U32(self.grid.nx as u32))
            .with_constant("grid", "ny", Value::U32(self.grid.ny as u32))
            .with_constant("grid", "dx", Value::F32(self.grid.dx as f32))
            .with_constant("grid", "dy", Value::F32(self.grid.dy as f32))
            .with_constant("constants", "dt", Value::F32(self.dt as f32))
            .with_constant("constants", "dt_old", Value::F32(self.dt as f32))
            .with_constant("constants", "dtau", Value::F32(0.0))
            .with_constant("constants", "time_scheme", Value::U32(0)) // Euler
            .with_constant("constants", "stride_x", Value::U32(self.grid.nx as u32))
    }
}

// Face-slot ids (the `k` order in the structured face loop: S, W, E, N).
const BAND_SOUTH_FACE: usize = 0;
const BAND_WEST_FACE: usize = 1;
const BAND_EAST_FACE: usize = 2;
const BAND_NORTH_FACE: usize = 3;

/// Interpret `stmts` over cell indices `0..n`, binding `idx` and `global_id`
/// exactly as the CPU dispatch loop does (the launch guard is skipped because we
/// iterate the exact domain).
fn run_over_cells(buffers: &Buffers, ctx: &Ctx, stmts: &[Stmt], n: usize, threads: usize) {
    crate::solver::cpu::parallel::parallel_for(n, threads, |idx| {
        let mut frame = Frame::new()
            .with_local("idx", Value::U32(idx as u32))
            .with_local("global_id", Value::Vec3([idx as f32, 0.0, 0.0]));
        Interpreter::new(buffers, ctx).run(stmts, &mut frame);
    });
}

/// Matrix-free banded SpMV: `y = A x` over the fixed 5-point stencil. Off-grid
/// neighbours (domain edges) contribute nothing — their band was left zero by
/// the assembly (boundary faces close via the diagonal + RHS, not a band).
fn banded_spmv(a: &[f32], x: &[f64], nx: usize, ny: usize) -> Vec<f64> {
    let n = nx * ny;
    let mut y = vec![0.0f64; n];
    for j in 0..ny {
        for i in 0..nx {
            let p = j * nx + i;
            let base = p * BAND_STRIDE;
            let mut acc = a[base + BAND_DIAG] as f64 * x[p];
            if j > 0 {
                acc += a[base + BAND_SOUTH] as f64 * x[p - nx];
            }
            if i > 0 {
                acc += a[base + BAND_WEST] as f64 * x[p - 1];
            }
            if i + 1 < nx {
                acc += a[base + BAND_EAST] as f64 * x[p + 1];
            }
            if j + 1 < ny {
                acc += a[base + BAND_NORTH] as f64 * x[p + nx];
            }
            y[p] = acc;
        }
    }
    y
}

/// Jacobi-preconditioned conjugate gradient on the banded SPD operator. Returns
/// the solution `x` (length `nx*ny`). Diffusion assembles a symmetric,
/// diagonally-dominant matrix, so CG is the natural matrix-free solver.
fn banded_cg(a: &[f32], b: &[f32], nx: usize, ny: usize, tol: f64, max_iter: usize) -> Vec<f32> {
    let n = nx * ny;
    let mut x = vec![0.0f64; n];
    // Jacobi preconditioner: inverse diagonal.
    let minv: Vec<f64> = (0..n)
        .map(|p| {
            let d = a[p * BAND_STRIDE + BAND_DIAG] as f64;
            if d.abs() > 1e-30 {
                1.0 / d
            } else {
                0.0
            }
        })
        .collect();

    let ax = banded_spmv(a, &x, nx, ny);
    let mut r: Vec<f64> = (0..n).map(|p| b[p] as f64 - ax[p]).collect();
    let mut z: Vec<f64> = (0..n).map(|p| minv[p] * r[p]).collect();
    let mut p_dir = z.clone();
    let mut rz: f64 = r.iter().zip(&z).map(|(&ri, &zi)| ri * zi).sum();

    let bnorm: f64 = b.iter().map(|&bi| (bi as f64) * (bi as f64)).sum::<f64>().sqrt();
    let thresh = tol * bnorm.max(1e-30);

    for _ in 0..max_iter {
        let rnorm: f64 = r.iter().map(|&ri| ri * ri).sum::<f64>().sqrt();
        if rnorm <= thresh {
            break;
        }
        let ap = banded_spmv(a, &p_dir, nx, ny);
        let p_ap: f64 = p_dir.iter().zip(&ap).map(|(&pi, &api)| pi * api).sum();
        if p_ap.abs() < 1e-300 {
            break;
        }
        let alpha = rz / p_ap;
        for i in 0..n {
            x[i] += alpha * p_dir[i];
            r[i] -= alpha * ap[i];
        }
        for i in 0..n {
            z[i] = minv[i] * r[i];
        }
        let rz_new: f64 = r.iter().zip(&z).map(|(&ri, &zi)| ri * zi).sum();
        let beta = rz_new / rz;
        for i in 0..n {
            p_dir[i] = z[i] + beta * p_dir[i];
        }
        rz = rz_new;
    }

    x.iter().map(|&v| v as f32).collect()
}

// ===========================================================================
// Coupled (block) banded matrix-free solver — for the structured incompressible
// momentum system (block stride 3: Ux, Uy, p). The assembled operator is a
// fixed 5-point band of `s x s` blocks: `matrix_values[p*5*s*s + r*(5*s) + b*s + c]`
// is block-band `b` (S,W,diag,E,N), row `r`, col `c` of cell `p` — the SoA layout
// the codegen emits (start_row_r = p*5*s*s + 5*s*r). The U–p system is a
// saddle point (indefinite), so we use restarted GMRES with a block-Jacobi
// (per-cell s x s block inverse) preconditioner rather than CG.
// ===========================================================================

/// Block-banded matrix-free operator over a structured grid.
pub struct BandedBlockOperator<'a> {
    a: &'a [f32],
    nx: usize,
    ny: usize,
    s: usize, // block stride (unknowns per cell)
}

impl<'a> BandedBlockOperator<'a> {
    fn n(&self) -> usize {
        self.nx * self.ny * self.s
    }

    #[inline]
    fn block(&self, p: usize, band: usize, r: usize, c: usize) -> f64 {
        // start_row_r = p*5*s*s + 5*s*r ; entry = start_row_r + band*s + c
        let s = self.s;
        self.a[p * 5 * s * s + 5 * s * r + band * s + c] as f64
    }

    /// `y = A x` over the 5-point block stencil (edge neighbours skipped — their
    /// bands are zero, closed via the diagonal + RHS by the assembly).
    fn spmv(&self, x: &[f64]) -> Vec<f64> {
        let (nx, ny, s) = (self.nx, self.ny, self.s);
        let mut y = vec![0.0f64; self.n()];
        for j in 0..ny {
            for i in 0..nx {
                let p = j * nx + i;
                // (band, neighbour cell) pairs present at this cell.
                let mut nbrs: Vec<(usize, usize)> = vec![(BAND_DIAG, p)];
                if j > 0 {
                    nbrs.push((BAND_SOUTH, p - nx));
                }
                if i > 0 {
                    nbrs.push((BAND_WEST, p - 1));
                }
                if i + 1 < nx {
                    nbrs.push((BAND_EAST, p + 1));
                }
                if j + 1 < ny {
                    nbrs.push((BAND_NORTH, p + nx));
                }
                for r in 0..s {
                    let mut acc = 0.0;
                    for &(band, q) in &nbrs {
                        for c in 0..s {
                            acc += self.block(p, band, r, c) * x[q * s + c];
                        }
                    }
                    y[p * s + r] = acc;
                }
            }
        }
        y
    }

    /// Per-cell block-Jacobi preconditioner: the inverse of each cell's diagonal
    /// `s x s` block, stored as a flat `s*s` slice per cell (any block size).
    fn block_jacobi_inverses(&self) -> Vec<Vec<f64>> {
        let ncells = self.nx * self.ny;
        let s = self.s;
        let mut inv = vec![Vec::new(); ncells];
        for p in 0..ncells {
            let mut m = vec![0.0f64; s * s];
            for r in 0..s {
                for c in 0..s {
                    m[r * s + c] = self.block(p, BAND_DIAG, r, c);
                }
            }
            inv[p] = invert_block(&m, s);
        }
        inv
    }

    fn apply_block_jacobi(&self, minv: &[Vec<f64>], r: &[f64]) -> Vec<f64> {
        let s = self.s;
        let ncells = self.nx * self.ny;
        let mut z = vec![0.0f64; self.n()];
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
}

/// Invert a general `s x s` matrix (row-major) via Gauss–Jordan with partial
/// pivoting; falls back to the (pseudo-)diagonal inverse for singular blocks so
/// the preconditioner stays well-defined on weak saddle rows.
fn invert_block(m: &[f64], s: usize) -> Vec<f64> {
    // Augmented [m | I].
    let mut a = vec![0.0f64; s * 2 * s];
    for r in 0..s {
        for c in 0..s {
            a[r * 2 * s + c] = m[r * s + c];
        }
        a[r * 2 * s + s + r] = 1.0;
    }
    for col in 0..s {
        // Partial pivot.
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
            // Singular: diagonal fallback.
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

/// Restarted, block-Jacobi-preconditioned GMRES on the banded block operator.
/// Handles the indefinite (saddle-point) U–p system the scalar CG cannot.
/// Returns `(x, relative_residual)`.
fn banded_block_gmres(
    op: &BandedBlockOperator,
    b: &[f32],
    restart: usize,
    max_outer: usize,
    tol: f64,
) -> (Vec<f32>, f64) {
    let n = op.n();
    let minv = op.block_jacobi_inverses();
    let b64: Vec<f64> = b.iter().map(|&v| v as f64).collect();
    let bnorm = norm(&b64).max(1e-30);
    let mut x = vec![0.0f64; n];

    for _outer in 0..max_outer {
        // r0 = M^{-1}(b - A x)
        let ax = op.spmv(&x);
        let r0: Vec<f64> = (0..n).map(|i| b64[i] - ax[i]).collect();
        let mut r = op.apply_block_jacobi(&minv, &r0);
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
            let av = op.spmv(&v[k]);
            let mut w = op.apply_block_jacobi(&minv, &av);
            // Arnoldi (modified Gram–Schmidt).
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
            // Apply previous Givens rotations, then a new one.
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

        // Back-substitute for y, update x = x + V y.
        let kk = k_used;
        let mut y = vec![0.0f64; kk];
        for i in (0..kk).rev() {
            let mut s = g[i];
            for j in (i + 1)..kk {
                s -= h[i][j] * y[j];
            }
            y[i] = if h[i][i].abs() > 1e-300 { s / h[i][i] } else { 0.0 };
        }
        for i in 0..kk {
            axpy(&mut x, y[i], &v[i]);
        }

        // Convergence check on the true residual.
        let ax = op.spmv(&x);
        let res: Vec<f64> = (0..n).map(|i| b64[i] - ax[i]).collect();
        if norm(&res) / bnorm <= tol {
            return (x.iter().map(|&v| v as f32).collect(), norm(&res) / bnorm);
        }
    }
    let ax = op.spmv(&x);
    let res: Vec<f64> = (0..n).map(|i| b64[i] - ax[i]).collect();
    (x.iter().map(|&v| v as f32).collect(), norm(&res) / bnorm)
}

// Small dense-vector helpers (f64).
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}
fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}
fn scale(a: &[f64], s: f64) -> Vec<f64> {
    a.iter().map(|&x| x * s).collect()
}
fn axpy(y: &mut [f64], a: f64, x: &[f64]) {
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

// ===========================================================================
// General coupled structured model solver: runs ANY `TopologyMode::Structured2D`
// model's recipe schedule (flux → gradients → assembly → banded solve → update)
// through the interpreter, on the dense grid, with the matrix-free banded
// GMRES+block-Jacobi solve replacing the CSR Krylov stack. This is the runtime
// that carries incompressible momentum (and, with their per-model kernels
// converted, the all-Mach / compressible families) on the structured grid.
// ===========================================================================

/// Boundary condition (kind, value) for one unknown component on one face.
/// `kind`: 0 = none/interior, 1 = Dirichlet, 2 = Neumann (matches `GpuBcKind`).
#[derive(Clone, Copy)]
pub struct BcComp {
    pub kind: u32,
    pub value: f32,
}

pub struct StructuredModelSolver {
    grid: StructuredGrid,
    s: usize,            // coupled unknowns per cell (banded block stride)
    state_stride: usize, // full state layout stride
    layout: crate::solver::model::backend::state_layout::StateLayout,
    prep: Vec<String>,
    per_iter: Vec<String>,
    update: Vec<String>,
    kernels: HashMap<String, Vec<Stmt>>,
    buffers: Buffers,
    constants: GpuConstants,
    outer_iters: usize,
    dt: f64,
    threads: usize,
}

impl StructuredModelSolver {
    /// Build a structured solver for a coupled model on `grid`. `outer_iters` is
    /// the number of Picard/Newton sweeps per time step.
    pub fn new(
        grid: StructuredGrid,
        model: &ModelSpec,
        dt: f64,
        outer_iters: usize,
    ) -> Result<Self, String> {
        let scheme = Scheme::Upwind;
        let recipe = SolverRecipe::from_model(
            model,
            scheme,
            TimeScheme::Euler,
            PreconditionerType::Jacobi,
            SteppingMode::Coupled,
        )?;
        let schemes = SchemeRegistry::new(scheme);
        let (programs, _wgsl_only) = model_kernel_programs(model, &schemes)?;
        let program_map: HashMap<String, _> = programs
            .into_iter()
            .map(|(id, p)| (id.as_str().to_string(), p))
            .collect();

        let n = grid.num_cells();
        let s = model.system.unknowns_per_cell() as usize;
        let state_stride = model.state_layout.stride() as usize;
        let flux_stride = recipe.flux.map(|f| f.stride as usize).unwrap_or(1);

        // Group the schedule exactly like the CPU/GPU driver (ScheduleGroups):
        // prep (Preparation, non-bc_expr) once; per-iter gradients/flux/assembly;
        // update/recovery. bc_expr (compressible boundary closure) is not yet
        // structured, so structured models must not schedule it.
        let mut prep = Vec::new();
        let mut per_iter = Vec::new();
        let mut update = Vec::new();
        let mut kernels: HashMap<String, Vec<Stmt>> = HashMap::new();
        for k in &recipe.kernels {
            let id = k.id.as_str().to_string();
            match k.phase {
                KernelPhase::Preparation => {
                    if id.contains("bc_expr") {
                        return Err(format!(
                            "structured model schedules bc_expr `{id}` (not yet structured)"
                        ));
                    }
                    prep.push(id.clone());
                }
                KernelPhase::Gradients | KernelPhase::FluxComputation | KernelPhase::Assembly => {
                    per_iter.push(id.clone());
                }
                KernelPhase::Update | KernelPhase::PrimitiveRecovery => update.push(id.clone()),
                // LinearSolve is replaced by the banded GMRES; Apply is a
                // monitor-only matvec (skipped); everything else is not run here.
                _ => continue,
            }
            if !kernels.contains_key(&id) {
                if let Some(p) = program_map.get(&id) {
                    let mut stmts =
                        Vec::with_capacity(p.indexing.len() + p.preamble.len() + p.body.len());
                    stmts.extend_from_slice(&p.indexing);
                    stmts.extend_from_slice(&p.preamble);
                    stmts.extend_from_slice(&p.body);
                    kernels.insert(id.clone(), stmts);
                } else {
                    return Err(format!(
                        "structured model `{}` schedules `{id}` with no CPU-executable program",
                        model.id
                    ));
                }
            }
        }

        let mut buffers = Buffers::new();
        buffers.insert_f32("state", vec![0.0; n * state_stride]);
        buffers.insert_f32("state_old", vec![0.0; n * state_stride]);
        buffers.insert_f32("state_old_old", vec![0.0; n * state_stride]);
        buffers.insert_f32("state_iter", vec![0.0; n * state_stride]);
        buffers.insert_f32("matrix_values", vec![0.0; n * BAND_STRIDE * s * s]);
        buffers.insert_f32("rhs", vec![0.0; n * s]);
        buffers.insert_f32("x", vec![0.0; n * s]);
        buffers.insert_f32("y", vec![0.0; n * s]);
        buffers.insert_f32("fluxes", vec![0.0; n * 4 * flux_stride]);
        buffers.insert_vec2("grad_state", vec![0.0; n * state_stride * 2]);
        buffers.insert_u32("bc_kind", vec![0u32; n * 4 * s]);
        buffers.insert_f32("bc_value", vec![0.0; n * 4 * s]);

        let mut constants = recipe.initial_constants;
        constants.dtau = 0.0;
        constants.time_scheme = 0; // Euler
        constants.stride_x = grid.nx as u32;

        Ok(Self {
            grid,
            s,
            state_stride,
            layout: model.state_layout.clone(),
            prep,
            per_iter,
            update,
            kernels,
            buffers,
            constants,
            outer_iters: outer_iters.max(1),
            dt,
            threads: 1,
        })
    }

    /// Seed a state component (by state-layout offset) from cell-centre coords.
    pub fn set_state<F: Fn(f64, f64) -> f64>(&mut self, offset: usize, f: F) {
        for p in 0..self.grid.num_cells() {
            let (x, y) = self.grid.cell_center(p);
            let v = f(x, y) as f32;
            for buf in ["state", "state_old", "state_old_old"] {
                self.buffers.set_f32(buf, p * self.state_stride + offset, v);
            }
        }
    }

    /// Set the per-(cell,direction,unknown) BC table from a closure of the edge
    /// and face-centre coords. `edge` is which domain edge the boundary face sits
    /// on; the closure returns one [`BcComp`] per coupled unknown (length `s`).
    pub fn set_boundaries<F: Fn(Edge, f64, f64) -> Vec<BcComp>>(&mut self, f: F) {
        let (nx, ny, s) = (self.grid.nx, self.grid.ny, self.s);
        let (dx, dy) = (self.grid.dx, self.grid.dy);
        let n = self.grid.num_cells();
        let mut bc_kind = vec![0u32; n * 4 * s];
        let mut bc_value = vec![0.0f32; n * 4 * s];
        for j in 0..ny {
            for i in 0..nx {
                let p = j * nx + i;
                let (cx, cy) = self.grid.cell_center(p);
                // Direction order S=0, W=1, E=2, N=3 (the codegen `k`).
                let faces = [
                    (0usize, j == 0, Edge::Bottom, cx, cy - 0.5 * dy),
                    (1, i == 0, Edge::Left, cx - 0.5 * dx, cy),
                    (2, i == nx - 1, Edge::Right, cx + 0.5 * dx, cy),
                    (3, j == ny - 1, Edge::Top, cx, cy + 0.5 * dy),
                ];
                for (k, is_b, edge, fx, fy) in faces {
                    if !is_b {
                        continue;
                    }
                    let comps = f(edge, fx, fy);
                    for (c, bc) in comps.iter().enumerate().take(s) {
                        let idx = (p * 4 + k) * s + c;
                        bc_kind[idx] = bc.kind;
                        bc_value[idx] = bc.value;
                    }
                }
            }
        }
        self.buffers.insert_u32("bc_kind", bc_kind);
        self.buffers.insert_f32("bc_value", bc_value);
    }

    fn build_ctx(&self) -> Ctx {
        let c = &self.constants;
        Ctx::new()
            .with_constant("grid", "nx", Value::U32(self.grid.nx as u32))
            .with_constant("grid", "ny", Value::U32(self.grid.ny as u32))
            .with_constant("grid", "dx", Value::F32(self.grid.dx as f32))
            .with_constant("grid", "dy", Value::F32(self.grid.dy as f32))
            .with_constant("constants", "dt", Value::F32(c.dt))
            .with_constant("constants", "dt_old", Value::F32(c.dt_old))
            .with_constant("constants", "dtau", Value::F32(c.dtau))
            .with_constant("constants", "time", Value::F32(c.time))
            .with_constant("constants", "viscosity", Value::F32(c.viscosity))
            .with_constant("constants", "density", Value::F32(c.density))
            .with_constant("constants", "component", Value::U32(c.component))
            .with_constant("constants", "alpha_p", Value::F32(c.alpha_p))
            .with_constant("constants", "scheme", Value::U32(c.scheme))
            .with_constant("constants", "alpha_u", Value::F32(c.alpha_u))
            .with_constant("constants", "stride_x", Value::U32(c.stride_x))
            .with_constant("constants", "time_scheme", Value::U32(c.time_scheme))
            .with_constant("constants", "eos_gamma", Value::F32(c.eos_gamma))
            .with_constant("constants", "eos_gm1", Value::F32(c.eos_gm1))
            .with_constant("constants", "eos_r", Value::F32(c.eos_r))
            .with_constant("constants", "eos_dp_drho", Value::F32(c.eos_dp_drho))
            .with_constant("constants", "eos_p_offset", Value::F32(c.eos_p_offset))
            .with_constant("constants", "eos_theta_ref", Value::F32(c.eos_theta_ref))
            .with_constant("constants", "buoyant_beta_g", Value::F32(c.buoyant_beta_g))
            .with_constant("constants", "buoyant_t0", Value::F32(c.buoyant_t0))
            .with_constant("constants", "buoyant_k_over_cp", Value::F32(c.buoyant_k_over_cp))
            .with_constant("low_mach_params", "model", Value::U32(0))
            .with_constant("low_mach_params", "theta_floor", Value::F32(0.0))
            .with_constant("low_mach_params", "pressure_coupling_alpha", Value::F32(0.0))
            .with_constant("low_mach_params", "eps4", Value::F32(0.0))
    }

    /// One implicit time step: `outer_iters` Picard sweeps of
    /// flux → gradients → assembly → banded GMRES → update. Returns the
    /// L-infinity change of the state across the step.
    pub fn step(&mut self) {
        let n = self.grid.num_cells();
        self.constants.dt = self.dt as f32;
        self.constants.dt_old = self.dt as f32;
        // Advance history.
        let cur = self.buffers.f32_vec("state");
        let old = self.buffers.f32_vec("state_old");
        self.buffers.copy_into_f32("state_old_old", &old);
        self.buffers.copy_into_f32("state_old", &cur);

        let ctx = self.build_ctx();
        let ids_prep = self.prep.clone();
        for id in &ids_prep {
            self.run(id, n, &ctx);
        }
        let per = self.per_iter.clone();
        let upd = self.update.clone();
        for _outer in 0..self.outer_iters {
            let snap = self.buffers.f32_vec("state");
            self.buffers.copy_into_f32("state_iter", &snap);
            for id in &per {
                self.run(id, n, &ctx);
            }
            // Banded GMRES on the assembled block-banded system.
            let a = self.buffers.f32_vec("matrix_values");
            let b = self.buffers.f32_vec("rhs");
            let op = BandedBlockOperator {
                a: &a,
                nx: self.grid.nx,
                ny: self.grid.ny,
                s: self.s,
            };
            let (x, _res) = banded_block_gmres(&op, &b, 60, 200, 1e-9);
            self.buffers.copy_into_f32("x", &x);
            for id in &upd {
                self.run(id, n, &ctx);
            }
        }
    }

    fn run(&self, id: &str, n: usize, ctx: &Ctx) {
        let stmts = self
            .kernels
            .get(id)
            .unwrap_or_else(|| panic!("structured solver missing kernel `{id}`"));
        run_over_cells(&self.buffers, ctx, stmts, n, self.threads);
    }

    /// Read a state component field (length `nx*ny`).
    pub fn state_field(&self, offset: usize) -> Vec<f64> {
        let st = self.buffers.f32_vec("state");
        (0..self.grid.num_cells())
            .map(|p| st[p * self.state_stride + offset] as f64)
            .collect()
    }

    /// Set the (uniform) fluid density and dynamic viscosity.
    pub fn set_fluid(&mut self, density: f64, viscosity: f64) {
        self.constants.density = density as f32;
        self.constants.viscosity = viscosity as f32;
    }

    /// Seed a named state field from a closure of the cell-centre coords (writes
    /// all history buffers — IC semantics). Panics if the field is absent.
    pub fn set_named_field<F: Fn(f64, f64) -> f64>(&mut self, name: &str, f: F) {
        let off = self
            .layout
            .offset_for(name)
            .unwrap_or_else(|| panic!("structured solver: no state field `{name}`"))
            as usize;
        self.set_state(off, f);
    }

    /// The state-layout offset of a named field, if present.
    pub fn field_offset(&self, name: &str) -> Option<usize> {
        self.layout.offset_for(name).map(|o| o as usize)
    }

    /// Number of coupled unknowns per cell (the banded block stride).
    pub fn unknowns(&self) -> usize {
        self.s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::model::generic_diffusion_demo_structured_model;

    /// The structured assembly must emit the textbook 5-point Laplacian: for an
    /// interior cell of a uniform grid, `diag = vol/dt + 2(dy/dx + dx/dy)` and the
    /// off-diagonal bands are `-dy/dx` (E/W) and `-dx/dy` (N/S). This validates the
    /// operator expansion produced by the IR — no hand-written stencil.
    #[test]
    fn structured_assembly_is_the_5point_laplacian() {
        let grid = StructuredGrid::new(6, 5, 6.0, 5.0); // dx = dy = 1.0
        let dt = 0.5;
        let model = generic_diffusion_demo_structured_model().unwrap();
        let mut solver = StructuredCpuSolver::new(grid, &model, dt).unwrap();
        solver.set_scalar(|_, _| 0.0);
        solver.set_dirichlet(|_, _| 0.0);
        solver.assemble_only();
        let a = solver.matrix_bands();

        let (dx, dy) = (grid.dx, grid.dy);
        let vol = dx * dy;
        let cx = dy / dx; // E/W coefficient magnitude
        let cy = dx / dy; // N/S coefficient magnitude
        // Interior cell (i=2, j=2).
        let p = 2 * grid.nx + 2;
        let base = p * BAND_STRIDE;
        let diag = a[base + BAND_DIAG] as f64;
        assert!(
            (diag - (vol / dt + 2.0 * cx + 2.0 * cy)).abs() < 1e-5,
            "interior diag {diag} != {}",
            vol / dt + 2.0 * cx + 2.0 * cy
        );
        assert!((a[base + BAND_WEST] as f64 + cx).abs() < 1e-5, "W band");
        assert!((a[base + BAND_EAST] as f64 + cx).abs() < 1e-5, "E band");
        assert!((a[base + BAND_SOUTH] as f64 + cy).abs() < 1e-5, "S band");
        assert!((a[base + BAND_NORTH] as f64 + cy).abs() < 1e-5, "N band");
    }

    /// A pure-diffusion steady state with linearly-varying Dirichlet data is the
    /// harmonic function `phi = x` (Laplace's equation with linear BCs). The
    /// structured solve must recover it across the interior.
    #[test]
    fn structured_diffusion_recovers_linear_harmonic_field() {
        let (nx, ny) = (20, 16);
        let (lx, ly) = (2.0, 1.6);
        let grid = StructuredGrid::new(nx, ny, lx, ly);
        let model = generic_diffusion_demo_structured_model().unwrap();
        let mut solver = StructuredCpuSolver::new(grid, &model, 5.0).unwrap();
        // Dirichlet phi = x on all edges; harmonic interior solution phi = x.
        solver.set_scalar(|_, _| 0.0);
        solver.set_dirichlet(|x, _| x);
        let steps = solver.solve_to_steady(1e-9, 4000);
        assert!(steps < 4000, "did not reach steady state ({steps} steps)");

        let field = solver.scalar_field();
        let mut max_err = 0.0f64;
        for p in 0..grid.num_cells() {
            let (x, _) = grid.cell_center(p);
            max_err = max_err.max((field[p] - x).abs());
        }
        assert!(
            max_err < 1e-3,
            "structured diffusion did not recover phi=x (max err {max_err})"
        );
    }

    /// Method-of-manufactured-solutions convergence order. For the steady
    /// diffusion `-lap(phi) = S` on the unit square with `phi* = sin(pi x)
    /// sin(pi y)` (so `S = 2 pi^2 phi*`) and homogeneous Dirichlet data, a
    /// second-order FV discretisation must show the cell-centre L2 error halving
    /// by ~4 as the grid doubles. This proves the structured operator is not just
    /// consistent but SECOND-ORDER accurate.
    #[test]
    fn structured_diffusion_mms_is_second_order() {
        use crate::solver::model::generic_diffusion_demo_structured_mms_model;
        use std::f64::consts::PI;

        let phi_star = |x: f64, y: f64| (PI * x).sin() * (PI * y).sin();
        let source = |x: f64, y: f64| 2.0 * PI * PI * (PI * x).sin() * (PI * y).sin();

        let l2_error = |n: usize| -> f64 {
            let grid = StructuredGrid::new(n, n, 1.0, 1.0);
            let model = generic_diffusion_demo_structured_mms_model().unwrap();
            // Large dt ⇒ the implicit step is essentially the steady Laplace solve.
            let mut solver = StructuredCpuSolver::new(grid, &model, 1.0e6).unwrap();
            solver.set_scalar(|_, _| 0.0);
            solver.set_state_component(1, source); // MMS source field
            solver.set_dirichlet(phi_star); // = 0 on the unit-square edges
            solver.solve_to_steady(1e-12, 20000);

            let field = solver.scalar_field();
            let mut sse = 0.0;
            for p in 0..grid.num_cells() {
                let (x, y) = grid.cell_center(p);
                let e = field[p] - phi_star(x, y);
                sse += e * e;
            }
            (sse / grid.num_cells() as f64).sqrt()
        };

        let e_coarse = l2_error(16);
        let e_fine = l2_error(32);
        let order = (e_coarse / e_fine).log2();
        assert!(
            order > 1.8 && order < 2.25,
            "structured diffusion MMS order {order} (e16={e_coarse:.3e}, e32={e_fine:.3e})"
        );
    }

    /// Immersed-boundary method on the structured grid: a cold circular obstacle
    /// (`phi = 0` inside) is imposed on a hot domain (`phi = 1` on all edges) by
    /// Brinkman volume penalisation — WITHOUT removing any cell. The implicit
    /// source assembles as `diag -= Sp*V` (FV `Sp` convention), so a sink driving
    /// `phi -> 0` uses a large NEGATIVE penalty coefficient (`Sp = -chi/eta`)
    /// inside the solid and `0` in the fluid. This is the whole point of IBM on a
    /// dense array: the grid stays complete and regular; the obstacle lives in a
    /// per-cell mask.
    #[test]
    fn structured_ibm_penalises_a_cold_immersed_obstacle() {
        use crate::solver::model::generic_diffusion_demo_structured_ibm_model;

        let (nx, ny) = (40, 40);
        let grid = StructuredGrid::new(nx, ny, 1.0, 1.0);
        let model = generic_diffusion_demo_structured_ibm_model().unwrap();
        let mut solver = StructuredCpuSolver::new(grid, &model, 1.0e6).unwrap();

        // Immersed cold disk (radius 0.15 about the domain centre).
        let (cx, cy, r) = (0.5, 0.5, 0.15);
        let sdf = |x: f64, y: f64| ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() - r;
        let sp_sink = -1.0e6; // Sp < 0 ⇒ sink ⇒ phi -> 0 inside the solid.

        solver.set_scalar(|_, _| 0.0);
        solver.set_state_component(1, |x, y| if sdf(x, y) < 0.0 { sp_sink } else { 0.0 });
        solver.set_dirichlet(|_, _| 1.0); // hot boundary everywhere
        let steps = solver.solve_to_steady(1e-10, 20000);
        assert!(steps < 20000, "IBM solve did not converge ({steps} steps)");

        let field = solver.scalar_field();
        // The full Cartesian grid is intact — NO cell was cut out.
        assert_eq!(field.len(), nx * ny, "structured IBM must keep every cell");

        // Deep inside the obstacle, phi is driven to ~0.
        let center = (ny / 2) * nx + nx / 2;
        assert!(
            field[center].abs() < 0.05,
            "obstacle core not masked: phi = {}",
            field[center]
        );

        // A fluid cell near a hot corner stays hot (phi ~ 1); the field is bounded.
        let corner = 2 * nx + 2;
        assert!(
            field[corner] > 0.6,
            "hot fluid corner unexpectedly cold: phi = {}",
            field[corner]
        );
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for &v in &field {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        assert!(lo > -0.05 && hi < 1.05, "field out of bounds: [{lo}, {hi}]");

        // Fluid cells far from the disk are hotter than the obstacle core:
        // the penalisation created a genuine cold region inside a warm field.
        let n_cold_in_solid = (0..grid.num_cells())
            .filter(|&p| {
                let (x, y) = grid.cell_center(p);
                sdf(x, y) < -0.02 && field[p] < 0.1
            })
            .count();
        let n_solid = (0..grid.num_cells())
            .filter(|&p| {
                let (x, y) = grid.cell_center(p);
                sdf(x, y) < -0.02
            })
            .count();
        assert!(
            n_cold_in_solid as f64 > 0.9 * n_solid as f64,
            "only {n_cold_in_solid}/{n_solid} interior-solid cells were masked"
        );
    }

    /// The block-banded GMRES + block-Jacobi solver must invert a coupled
    /// (velocity–pressure-like) 3x3-block 5-point system — the indefinite saddle
    /// structure the scalar CG cannot handle. Build a diagonally-dominant block
    /// operator with intra-block U–p coupling, pick a known solution, and verify
    /// GMRES recovers it (and drives the residual down).
    #[test]
    fn banded_block_gmres_inverts_a_coupled_system() {
        let (nx, ny, s) = (6usize, 5usize, 3usize);
        let ncells = nx * ny;
        // Diagonal block with U(0,1)–p(2) coupling (non-symmetric, saddle-like).
        let dblock = [6.0, 0.0, 1.0, 0.0, 6.0, 1.0, -1.0, -1.0, 6.0];
        let mut a = vec![0.0f32; ncells * 5 * s * s];
        let set = |a: &mut [f32], p: usize, band: usize, r: usize, c: usize, v: f32| {
            a[p * 5 * s * s + 5 * s * r + band * s + c] = v;
        };
        for j in 0..ny {
            for i in 0..nx {
                let p = j * nx + i;
                for r in 0..s {
                    for c in 0..s {
                        set(&mut a, p, BAND_DIAG, r, c, dblock[r * s + c] as f32);
                    }
                }
                // Off-diagonal bands: -I on existing neighbours (a block Laplacian).
                let mut band_of = |band: usize, exists: bool| {
                    if exists {
                        for d in 0..s {
                            set(&mut a, p, band, d, d, -1.0);
                        }
                    }
                };
                band_of(BAND_SOUTH, j > 0);
                band_of(BAND_WEST, i > 0);
                band_of(BAND_EAST, i + 1 < nx);
                band_of(BAND_NORTH, j + 1 < ny);
            }
        }

        // Known solution, deterministic; b = A x*.
        let xstar: Vec<f64> = (0..ncells * s)
            .map(|k| ((k * 37 % 11) as f64 - 5.0) * 0.1)
            .collect();
        let op = BandedBlockOperator { a: &a, nx, ny, s };
        let b64 = op.spmv(&xstar);
        let b: Vec<f32> = b64.iter().map(|&v| v as f32).collect();

        let (x, rel_res) = banded_block_gmres(&op, &b, 40, 200, 1e-10);
        assert!(rel_res < 1e-8, "GMRES residual too large: {rel_res}");
        let mut max_err = 0.0f64;
        for k in 0..ncells * s {
            max_err = max_err.max((x[k] as f64 - xstar[k]).abs());
        }
        assert!(
            max_err < 1e-5,
            "GMRES did not recover the known solution (max err {max_err}, res {rel_res})"
        );
    }

    /// End-to-end smoke of the COUPLED structured pipeline: a lid-driven cavity
    /// on the dense Cartesian grid runs the full flux → gradients → assembly →
    /// banded GMRES → update loop (all the structured, indirection-free kernels)
    /// and must develop the top-driven flow — the near-lid x-velocity turns
    /// positive and the field stays bounded. This proves incompressible momentum
    /// SOLVES structured on CPU.
    #[test]
    fn structured_incompressible_lid_cavity_runs() {
        use crate::solver::model::incompressible_momentum_structured_model;
        let (nx, ny) = (24, 24);
        let grid = StructuredGrid::new(nx, ny, 1.0, 1.0);
        let model = incompressible_momentum_structured_model().unwrap();
        let mut solver = StructuredModelSolver::new(grid, &model, 0.05, 3).unwrap();
        assert_eq!(solver.unknowns(), 3, "coupled Ux/Uy/p");
        solver.set_fluid(1.0, 0.01); // Re = U*L/nu = 1*1/0.01 = 100
        // IC: rest.
        solver.set_state(0, |_, _| 0.0);
        solver.set_state(1, |_, _| 0.0);
        solver.set_state(2, |_, _| 0.0);
        // Lid cavity BCs: no-slip walls; top wall slides at u=1. Pressure
        // zero-gradient (Neumann 0) everywhere. Components: 0=Ux, 1=Uy, 2=p.
        solver.set_boundaries(|edge, _x, _y| {
            let u_wall = if matches!(edge, Edge::Top) { 1.0 } else { 0.0 };
            vec![
                BcComp { kind: 1, value: u_wall }, // Ux Dirichlet
                BcComp { kind: 1, value: 0.0 },    // Uy Dirichlet 0
                BcComp { kind: 2, value: 0.0 },    // p Neumann 0
            ]
        });

        for _ in 0..40 {
            solver.step();
        }

        let ux = solver.state_field(0);
        let uy = solver.state_field(1);
        // Field is finite and bounded.
        let mut umax = 0.0f64;
        for (&a, &b) in ux.iter().zip(&uy) {
            assert!(a.is_finite() && b.is_finite(), "velocity diverged");
            umax = umax.max(a.hypot(b));
        }
        assert!(umax > 0.05 && umax < 5.0, "unphysical lid-cavity speed {umax}");
        // The top rows (near the moving lid) must be dragged in +x.
        let top_row_mean_ux: f64 = (0..nx)
            .map(|i| ux[(ny - 1) * nx + i])
            .sum::<f64>()
            / nx as f64;
        assert!(
            top_row_mean_ux > 0.1,
            "near-lid x-velocity did not develop ({top_row_mean_ux})"
        );
    }

    /// End-to-end smoke of the ALL-MACH THERMAL structured pipeline: the
    /// pressure-based compressible solver (momentum + continuity + temperature,
    /// on-device EOS density recovery `rho = rho_t_ref/T + psi*p`) runs its full
    /// structured, indirection-free kernel schedule on the dense grid and stays
    /// bounded, with the lid-driven flow developing. Proves all-Mach thermal
    /// SOLVES structured on CPU.
    #[test]
    fn structured_allmach_thermal_lid_cavity_runs() {
        use crate::solver::model::allmach_thermal_structured_model;
        let (nx, ny) = (20, 20);
        let grid = StructuredGrid::new(nx, ny, 1.0, 1.0);
        let model = allmach_thermal_structured_model().unwrap();
        let s = model.system.unknowns_per_cell() as usize; // Ux, Uy, p, T
        let mut solver = StructuredModelSolver::new(grid, &model, 0.02, 3).unwrap();
        solver.set_fluid(1.0, 0.02);
        // All-Mach extra state seeds (mirror driver.rs): low-Mach compressibility,
        // reference density/temperature for the on-device EOS recovery.
        let psi = 0.5;
        solver.set_named_field("psi", |_, _| psi);
        solver.set_named_field("psi_precond", |_, _| psi.max(1.0));
        solver.set_named_field("rho", |_, _| 1.0);
        solver.set_named_field("rho_t_ref", |_, _| 1.0); // density * T_ref (T_ref = 1)
        solver.set_named_field("T", |_, _| 1.0);
        if solver.field_offset("t_ref").is_some() {
            solver.set_named_field("t_ref", |_, _| 1.0);
        }
        if solver.field_offset("rho_floor").is_some() {
            solver.set_named_field("rho_floor", |_, _| psi * 1.0e-5);
        }
        // Lid cavity BCs (Ux, Uy, p, T): moving top lid, no-slip walls, p & T
        // zero-gradient (adiabatic).
        solver.set_boundaries(move |edge, _x, _y| {
            let u_wall = if matches!(edge, Edge::Top) { 1.0 } else { 0.0 };
            let mut v = vec![
                BcComp { kind: 1, value: u_wall },
                BcComp { kind: 1, value: 0.0 },
                BcComp { kind: 2, value: 0.0 },
            ];
            if s >= 4 {
                v.push(BcComp { kind: 2, value: 0.0 }); // T Neumann 0
            }
            v
        });

        for _ in 0..30 {
            solver.step();
        }
        let ux = solver.state_field(0);
        let temp_off = solver.field_offset("T").unwrap();
        let t = solver.state_field(temp_off);
        let mut umax = 0.0f64;
        for (&a, &ti) in ux.iter().zip(&t) {
            assert!(a.is_finite() && ti.is_finite(), "all-Mach state diverged");
            umax = umax.max(a.abs());
        }
        assert!(umax > 0.05 && umax < 5.0, "unphysical all-Mach lid speed {umax}");
        // Temperature stays near the reference (adiabatic, low-Mach): bounded.
        for &ti in &t {
            assert!(ti > 0.5 && ti < 2.0, "all-Mach temperature out of band: {ti}");
        }
    }

    /// Not an assertion — prints an ASCII heat map of the structured IBM solve so
    /// the immersed cold obstacle in a hot field is visible by eye. Run with:
    /// `cargo test --features cpu --lib structured::tests::structured_ibm_ascii -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn structured_ibm_ascii_heatmap() {
        use crate::solver::model::generic_diffusion_demo_structured_ibm_model;
        let (nx, ny) = (64, 40);
        let grid = StructuredGrid::new(nx, ny, 1.6, 1.0);
        let model = generic_diffusion_demo_structured_ibm_model().unwrap();
        let mut solver = StructuredCpuSolver::new(grid, &model, 1.0e6).unwrap();
        let (cx, cy, r) = (0.6, 0.5, 0.18);
        let sdf = |x: f64, y: f64| ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() - r;
        solver.set_scalar(|_, _| 0.0);
        solver.set_state_component(1, |x, y| if sdf(x, y) < 0.0 { -1.0e6 } else { 0.0 });
        solver.set_dirichlet(|_, _| 1.0);
        solver.solve_to_steady(1e-10, 20000);
        let field = solver.scalar_field();
        let ramp = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];
        println!("\nStructured IBM: cold immersed disk in a hot field ({nx}x{ny}, no cells cut)");
        for j in (0..ny).rev() {
            let mut line = String::with_capacity(nx);
            for i in 0..nx {
                let v = field[j * nx + i].clamp(0.0, 1.0);
                line.push(ramp[((v * (ramp.len() - 1) as f64).round()) as usize]);
            }
            println!("{line}");
        }
    }
}
