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
use crate::solver::model::backend::SchemeRegistry;
use crate::solver::model::ModelSpec;
use crate::solver::scheme::Scheme;
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
