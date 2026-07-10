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
///
/// The dense-grid geometry types are shared with the GPU structured path
/// (always compiled), so both backends key cells/faces identically.
pub use crate::solver::gpu::structured::{BcComp, Edge, StructuredGrid};

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
        // Boundary TYPE per (cell, dir); unused by the scalar diffusion operator
        // (no SlipWall / boundary-type flux), but the assembly reads it, so it
        // must exist. Walls carry type 3, matching the unstructured face_boundary.
        let mut face_boundary = vec![0u32; n * 4];
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let p = j * grid.nx + i;
                if j == 0 {
                    face_boundary[p * 4] = 3;
                }
                if i == 0 {
                    face_boundary[p * 4 + 1] = 3;
                }
                if i == grid.nx - 1 {
                    face_boundary[p * 4 + 2] = 3;
                }
                if j == grid.ny - 1 {
                    face_boundary[p * 4 + 3] = 3;
                }
            }
        }
        buffers.insert_u32("face_boundary", face_boundary);

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
// General coupled structured model solver: runs ANY `TopologyMode::Structured2D`
// model's recipe schedule (flux → gradients → assembly → banded solve → update)
// through the interpreter, on the dense grid, with the matrix-free banded
// GMRES+block-Jacobi solve replacing the CSR Krylov stack. This is the runtime
// that carries incompressible momentum (and, with their per-model kernels
// converted, the all-Mach / compressible families) on the structured grid.
// ===========================================================================

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
    /// When true, the Picard loop may exit early once the relative state change
    /// drops below [`Self::outer_tol`] (mirrors unstructured `outer_auto_converge`).
    outer_auto_converge: bool,
    /// Relative correction-norm threshold for opportunistic outer early-exit
    /// (GPU generic-coupled default is `1e-3`; `0` disables the break even when
    /// `outer_auto_converge` is set).
    outer_tol: f32,
    dt: f64,
    /// Previous step's `dt` — BDF2 variable-step coefficients read `r = dt/dt_old`.
    /// Must lag `dt` across adaptive-dt changes (unstructured `CpuSolver::dt_old`).
    dt_old: f64,
    /// Committed steps — drives the BDF2→Euler startup fallback (`step_count == 0`).
    step_count: u64,
    /// Requested time scheme (BDF2 falls back to Euler on step 0).
    time_scheme: TimeScheme,
    /// State-layout offsets of the coupled unknowns, GROUPED by solved field
    /// (`[[Ux, Uy], [p], [T]]`). Used for the outer residual / early-exit, so that
    /// storage fields like `ibm_penalty_U` (~1e5) cannot dominate `max|state|` and
    /// force a false 1-outer exit on immersed-obstacle cases, and so each field is
    /// measured against its own scale.
    unknown_state_groups: Vec<Vec<usize>>,
    threads: usize,
    /// Kernel engine: the interpreter (default, correctness oracle) or the
    /// compiled-Rust transpiled kernels (speed path; `generated::lookup_structured`
    /// with per-kernel interpreter fallback — same contract as the unstructured
    /// path). Set by the GUI's CPU-Transpiled backend selection.
    engine: crate::solver::cpu::CpuEngine,
    /// Active coupled-solve preconditioner (shared banded routine): block-Jacobi,
    /// the model-owned Schur, or Schur + AMG pressure solve.
    precond: crate::solver::banded_schur::BandedPrecond,
    /// Cross-step cache for the Schur AMG pressure hierarchy (built once from the
    /// grid sparsity, reused every solve; only the Galerkin values re-assemble).
    amg_cache: crate::solver::banded_schur::StructuredAmgCache,
    /// ADAPTIVE AMG activation (mirrors the unstructured `CpuSolver::schur_amg_active`
    /// one-way latch). When the preconditioner is Schur+AMG we do NOT run the AMG
    /// V-cycle from the start: the from-rest startup drives A_pp transiently
    /// indefinite, on which the AMG blows up. We run the robust heavy-ball Schur
    /// first and flip this latch ON (permanently, until a rebuild) only once the
    /// heavy-ball inner solve genuinely STALLS (the h-dependent fine-mesh regime,
    /// where AMG's mesh-independence pays off) — giving robust-from-rest AND
    /// h-independent-once-developed without the user pre-committing to fragile AMG.
    amg_active: std::sync::atomic::AtomicBool,
    /// The model's declared Schur layout (`None` if it declares none), so
    /// `set_preconditioner` can rebuild for any kind without re-reading the model.
    schur_layout: Option<crate::solver::banded_schur::SchurLayout>,
    /// Model id + accumulated sim time (GUI parity with `StructuredGpuSolver`).
    model_id: &'static str,
    time: f64,
    /// Convergence telemetry from the last [`step`](Self::step) (GUI readout):
    /// `(outer_iters, linear rel-residual of the last inner solve, max |ΔU| and
    /// max |Δp| across the last outer sweep = the nonlinear/Picard residual)`.
    last_stats: StructuredStepStats,
}

/// Per-step convergence telemetry surfaced to the GUI (mirrors what the
/// unstructured `step_stats()` reports). Defined in the always-compiled
/// [`banded_schur`](crate::solver::banded_schur) module so the GPU backend (which
/// compiles without the `cpu` feature) can produce the same shape; re-exported
/// here for the CPU solver's callers.
pub use crate::solver::banded_schur::StructuredStepStats;

impl StructuredModelSolver {
    /// Build a structured solver for a coupled model on `grid`. `outer_iters` is
    /// the number of Picard/Newton sweeps per time step. Defaults to first-order
    /// Upwind advection + backward-Euler time integration; use
    /// [`Self::with_config`] to select any of the runtime schemes.
    pub fn new(
        grid: StructuredGrid,
        model: &ModelSpec,
        dt: f64,
        outer_iters: usize,
    ) -> Result<Self, String> {
        Self::with_config(grid, model, dt, outer_iters, Scheme::Upwind, TimeScheme::Euler)
    }

    /// Build a structured coupled solver with an explicit advection scheme and
    /// time-integration scheme. Both are honoured at RUNTIME by the same codegen
    /// kernels the GPU path runs (`constants.scheme` selects the deferred-
    /// correction reconstruction; `constants.time_scheme==1` is BDF2) — full
    /// parity with [`crate::solver::gpu::structured::StructuredGpuSolver::with_config`].
    pub fn with_config(
        grid: StructuredGrid,
        model: &ModelSpec,
        dt: f64,
        outer_iters: usize,
        scheme: Scheme,
        time_scheme: TimeScheme,
    ) -> Result<Self, String> {
        let recipe = SolverRecipe::from_model(
            model,
            scheme,
            time_scheme,
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
        // prep (Preparation, incl. the now-structured `bc_expr` expression-valued
        // boundary closure) once; per-iter gradients/flux/assembly; update/recovery.
        let mut prep = Vec::new();
        let mut per_iter = Vec::new();
        let mut update = Vec::new();
        let mut kernels: HashMap<String, Vec<Stmt>> = HashMap::new();
        for k in &recipe.kernels {
            let id = k.id.as_str().to_string();
            match k.phase {
                KernelPhase::Preparation => {
                    // `bc_expr` is cell-dispatched structured (walks each cell's 4
                    // faces, rewriting `bc_value` from interior state on boundary
                    // faces) — full parity with the unstructured face-dispatch.
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
        // Boundary TYPE per (cell, dir), matching the unstructured
        // `face_boundary` encoding (bc_table_index: 0 interior, 1 Inlet, 2 Outlet,
        // 3 Wall, 4 SlipWall, 5 MovingWall).
        buffers.insert_u32("face_boundary", vec![0u32; n * 4]);

        let mut constants = recipe.initial_constants;
        constants.dtau = 0.0;
        // `time_scheme` (and `scheme`) come from `from_model` via the args — the
        // runtime kernels branch on them (BDF2 at time_scheme==1). Do NOT override.
        constants.stride_x = grid.nx as u32;

        let schur_layout = crate::solver::banded_schur::schur_layout_from_model(model);
        // Pressure under-relaxation, mirroring the driver's incompressible/all-Mach
        // default (alpha_p=0.3, alpha_u=0.7): the coupled update kernel applies
        // `phi = phi_old + alpha*(x - phi_old)`, so a sub-1 alpha_p damps the
        // saddle-point OUTER-iteration instability. Only the Schur-layout (coupled
        // incompressible / all-Mach) models get it; compressible keeps its declared
        // 1.0 (recipe default). No kernel change — the update already consumes it.
        if schur_layout.is_some() {
            constants.alpha_p = 0.3;
            constants.alpha_u = 0.7;
        }

        let unknown_state_groups: Vec<Vec<usize>> =
            crate::solver::model::kernel::model_unknown_state_offset_groups(model)?
                .into_iter()
                .map(|g| g.into_iter().map(|o| o as usize).collect())
                .collect();

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
            outer_auto_converge: false,
            outer_tol: 1e-3,
            dt,
            dt_old: dt,
            step_count: 0,
            time_scheme,
            unknown_state_groups,
            threads: 1,
            engine: crate::solver::cpu::CpuEngine::Interpreter,
            precond: crate::solver::banded_schur::BandedPrecond::BlockJacobi,
            amg_cache: crate::solver::banded_schur::StructuredAmgCache::default(),
            amg_active: std::sync::atomic::AtomicBool::new(false),
            schur_layout,
            model_id: model.id,
            time: 0.0,
            last_stats: StructuredStepStats::default(),
        })
    }

    /// Select the kernel engine (interpreter vs compiled-Rust transpiled) and the
    /// worker-thread count for the per-cell kernel passes. Transpiled runs the
    /// `generated::lookup_structured` kernels where available (interpreter
    /// fallback otherwise); the banded linear solve is unaffected.
    pub fn set_engine(&mut self, engine: crate::solver::cpu::CpuEngine, threads: usize) {
        self.engine = engine;
        self.threads = threads.max(1);
    }

    /// Cap on Picard (outer) sweeps per time step.
    pub fn set_outer_iters(&mut self, n: usize) {
        self.outer_iters = n.max(1);
    }

    /// Enable opportunistic Picard early-exit when the relative state change
    /// falls below [`Self::set_outer_tolerance`].
    pub fn set_outer_auto_converge(&mut self, enable: bool) {
        self.outer_auto_converge = enable;
    }

    /// Relative correction-norm threshold for outer early-exit (`1e-3` default,
    /// matching the unstructured GPU generic-coupled gate).
    pub fn set_outer_tolerance(&mut self, tol: f32) {
        self.outer_tol = tol.max(0.0);
    }

    /// Select the coupled-solve preconditioner (block-Jacobi / Schur / Schur+AMG).
    /// A Schur variant on a model without a layout keeps block-Jacobi. Returns the
    /// EFFECTIVE kind. Mirrors the GPU `StructuredGpuSolver::set_preconditioner`.
    pub fn set_preconditioner(
        &mut self,
        kind: crate::solver::banded_schur::CoupledPrecondKind,
    ) -> crate::solver::banded_schur::CoupledPrecondKind {
        use crate::solver::banded_schur::{BandedPrecond, CoupledPrecondKind as K};
        self.precond = match (kind, &self.schur_layout) {
            (K::Schur, Some(l)) => BandedPrecond::schur(l, false),
            (K::SchurAmg, Some(l)) => BandedPrecond::schur(l, true),
            _ => BandedPrecond::BlockJacobi,
        };
        self.amg_active.store(false, std::sync::atomic::Ordering::Relaxed); // re-arm the adaptive latch on a precond change
        crate::solver::banded_schur::kind_of(&self.precond)
    }

    /// The preconditioner to actually run this step. A Schur+AMG request runs the
    /// robust heavy-ball Schur until the adaptive latch ([`amg_active`]) flips; all
    /// other kinds pass through unchanged.
    fn effective_precond(&self) -> crate::solver::banded_schur::BandedPrecond {
        use crate::solver::banded_schur::BandedPrecond;
        match &self.precond {
            BandedPrecond::Schur {
                u_idx,
                p,
                omega,
                sweeps_cap,
                pressure_amg: true,
            } if !self.amg_active.load(std::sync::atomic::Ordering::Relaxed) => BandedPrecond::Schur {
                u_idx: u_idx.clone(),
                p: *p,
                omega: *omega,
                sweeps_cap: *sweeps_cap,
                pressure_amg: false, // heavy-ball until the latch activates AMG
            },
            other => other.clone(),
        }
    }

    /// Whether the adaptive AMG latch has activated (diagnostics/tests).
    pub fn amg_is_active(&self) -> bool {
        self.amg_active.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Whether this model declares a Schur preconditioner.
    pub fn supports_schur(&self) -> bool {
        self.schur_layout.is_some()
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

    /// Set the per-(cell,direction) boundary TYPE + per-unknown BC table from a
    /// closure of the edge and face-centre coords. The closure returns the
    /// boundary type ([`GpuBoundaryType`] code: 1 Inlet, 2 Outlet, 3 Wall,
    /// 4 SlipWall, 5 MovingWall) and one [`BcComp`] per coupled unknown (length
    /// `s`) — full parity with the unstructured `face_boundary` + `bc_kind/value`.
    pub fn set_boundaries<F: Fn(Edge, f64, f64) -> (u32, Vec<BcComp>)>(&mut self, f: F) {
        let (nx, ny, s) = (self.grid.nx, self.grid.ny, self.s);
        let (dx, dy) = (self.grid.dx, self.grid.dy);
        let n = self.grid.num_cells();
        let mut bc_kind = vec![0u32; n * 4 * s];
        let mut bc_value = vec![0.0f32; n * 4 * s];
        let mut face_boundary = vec![0u32; n * 4];
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
                    let (btype, comps) = f(edge, fx, fy);
                    face_boundary[p * 4 + k] = btype;
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
        self.buffers.insert_u32("face_boundary", face_boundary);
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
        // Variable-dt BDF2: use the PREVIOUS step's dt as `dt_old` (unstructured
        // CpuSolver parity). Overwriting `dt_old = dt` every step forced r=1 and
        // broke adaptive-dt BDF2 (pressure oscillations / stalled development).
        self.constants.dt = self.dt as f32;
        self.constants.dt_old = self.dt_old as f32;
        // BDF2 OpenFOAM-style startup: first step is Euler (no valid n-2 state).
        self.constants.time_scheme = if self.time_scheme == TimeScheme::BDF2 && self.step_count == 0
        {
            TimeScheme::Euler as u32
        } else {
            self.time_scheme as u32
        };
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
        // Adaptive-AMG bookkeeping: while a Schur+AMG request is still running the
        // heavy-ball fallback, count near-total inner stalls (`rel > 0.7`) vs total
        // applies; flip the latch to AMG after this step if the heavy-ball majority-
        // stalled (the h-dependent regime AMG cures). Mirrors the unstructured
        // `failures*2 >= applies` rule.
        // Effective preconditioner can flip mid-step once the adaptive AMG latch
        // fires — re-read via `effective_precond()` after each outer when still
        // counting stalls so later Picard sweeps of the SAME step use AMG.
        let mut precond = self.effective_precond();
        let mut counting_stalls = matches!(
            &self.precond,
            crate::solver::banded_schur::BandedPrecond::Schur { pressure_amg: true, .. }
        ) && !self.amg_active.load(std::sync::atomic::Ordering::Relaxed);
        // Outer-convergence telemetry for the GUI readout (mirrors the unstructured
        // "Coupled: N iters, U:.. P:.." / "Linear: .. res=.." lines). The banded
        // solve returns the coupled increment `x`; its L-infinity over the velocity
        // slots / pressure slot is the Picard outer residual, and `res` is the linear
        // relative residual. We keep the LAST outer's values (early outers are large
        // by construction) so a small readout means the step actually converged.
        // Pressure coupled-rank (== state offset for current models) for the
        // GUI P residual line; `None` → treat state offset 2 as p.
        let p_slot: Option<usize> = self.schur_layout.as_ref().map(|l| l.p);
        let (mut last_res, mut last_du, mut last_dp) = (0f32, 0f32, 0f32);
        let mut last_iters = 0u32;
        let mut outers_done = 0u32;
        // Shared tolerance/plateau exit (GPU parity via the one shared type).
        let mut outer_exit = crate::solver::banded_schur::StructuredOuterExit::default();
        for _outer in 0..self.outer_iters {
            let snap = self.buffers.f32_vec("state");
            self.buffers.copy_into_f32("state_iter", &snap);
            for id in &per {
                self.run(id, n, &ctx);
            }
            // Banded GMRES on the assembled block-banded system (shared routine —
            // block-Jacobi or the model-owned Schur; bit-identical to the GPU
            // host coupled solve).
            let a = self.buffers.f32_vec("matrix_values");
            let b = self.buffers.f32_vec("rhs");
            let (x, res, iters) = crate::solver::banded_schur::banded_gmres_t(
                &a,
                self.grid.nx,
                self.grid.ny,
                self.s,
                &b,
                &precond,
                60,
                200,
                crate::solver::banded_schur::default_step_tol(),
                self.threads,
                Some(&self.amg_cache),
            );
            // NON-FINITE system (e.g. a NaN auxiliary poisoning the assembly —
            // the observed case: an un-seeded `u_ref` makes the on-device
            // psi_precond recovery emit NaN at quiescent cells): the solve
            // bailed without a usable correction (x = the zero initial guess).
            // Applying it would DRIVE THE STATE TO ZERO through the
            // under-relaxed update. Match the unstructured failure convention
            // (fgmres restores its warm start = current state): FREEZE this
            // step — skip the update — and surface the failure via the stats
            // (linear_res = inf on the GUI readout).
            if !res.is_finite() {
                last_res = f32::INFINITY;
                last_iters = iters;
                outers_done += 1;
                break;
            }
            if counting_stalls {
                // Flip AMG as soon as heavy-ball either fails to reduce the residual
                // OR converges only after burning a full GMRES restart budget
                // (iters>60). Waiting until end-of-step left the remaining Picard
                // outers of the wake-hit step on multi-second heavy-ball applies —
                // the GUI freeze — even though the residual eventually met tol.
                // One such outer is enough: the latch is one-way.
                if res > 0.7 || iters > 60 {
                    self.amg_active
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    precond = self.effective_precond();
                    counting_stalls = false;
                }
            }
            last_res = res as f32;
            last_iters = iters;
            self.buffers.copy_into_f32("x", &x);
            for id in &upd {
                self.run(id, n, &ctx);
            }
            outers_done += 1;

            // Picard residual = APPLIED under-relaxed change |state − state_iter|,
            // per FIELD, relative to that field's own scale (see
            // `structured_outer_residuals` for why the scale must NOT be floored
            // at 1.0, and why `Ux`/`Uy` share one scale).
            let st = self.buffers.f32_vec("state");
            let it = self.buffers.f32_vec("state_iter");
            let scaled = crate::solver::banded_schur::structured_outer_residuals(
                &st,
                &it,
                self.state_stride,
                &self.unknown_state_groups,
            );

            // GUI "Coupled: U / P": the velocity field's residual (not T), and p.
            let (mut du, mut dp) = (0.0f32, 0.0f32);
            for (g, comps) in self.unknown_state_groups.iter().enumerate() {
                // Velocity: state offsets 0 and 1 (U vector2).
                if comps.first() == Some(&0) {
                    du = du.max(scaled[g]);
                }
                // p_slot is the COUPLED rank; for pressure-based models it equals
                // the state offset of p (2).
                let p_off = p_slot.unwrap_or(2);
                if comps.first() == Some(&p_off) {
                    dp = dp.max(scaled[g]);
                }
            }
            last_du = du;
            last_dp = dp;
            if std::env::var("CFD2_STRUCT_OUTER_DEBUG").is_ok() {
                eprintln!(
                    "[outer] step {} outer {} lin={iters} res={res:.2e} scaled={:?}",
                    self.step_count,
                    outers_done,
                    scaled.iter().map(|v| format!("{v:.3e}")).collect::<Vec<_>>()
                );
            }

            // Early-exit when EVERY unknown's per-field scaled residual is
            // under tol (from outer 2), or when every field is under tol,
            // stalled, or provably unable to reach tol within the cap (from
            // outer 5) — see `StructuredOuterExit` for why an under-relaxation-
            // limited residual must break instead of burning the outer cap.
            if self.outer_auto_converge
                && self.outer_tol > 0.0
                && outer_exit.should_break(
                    &scaled,
                    outers_done,
                    self.outer_iters as u32,
                    self.outer_tol,
                )
            {
                break;
            }
        }
        self.last_stats = StructuredStepStats {
            outer_iters: outers_done,
            linear_iters: last_iters,
            linear_res: last_res,
            outer_du: last_du,
            outer_dp: last_dp,
        };
        // Commit the step's dt into dt_old for the next step's BDF2 ratio, and
        // bump the step counter (unstructured `CpuSolver` end-of-step bookkeeping).
        self.dt_old = self.dt;
        self.step_count = self.step_count.saturating_add(1);
        self.time += self.dt;
    }

    /// Convergence telemetry from the most recent [`step`](Self::step) — the GUI
    /// readout mirrors the unstructured "Coupled/Linear" lines.
    pub fn last_stats(&self) -> StructuredStepStats {
        self.last_stats
    }

    /// Model id (GUI model-echo / caps).
    pub fn model_id(&self) -> &'static str {
        self.model_id
    }

    /// Accumulated simulation time.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// The implicit time-step size.
    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Velocity under-relaxation for the coupled update (`phi = phi_old + α·Δ`).
    pub fn set_alpha_u(&mut self, alpha_u: f32) {
        self.constants.alpha_u = alpha_u;
    }

    /// Pressure under-relaxation for the coupled update — the driver's
    /// incompressible/all-Mach default is 0.3 (damps the saddle outer iteration).
    pub fn set_alpha_p(&mut self, alpha_p: f32) {
        self.constants.alpha_p = alpha_p;
    }

    /// Set the implicit time-step size (GUI timestep slider / adaptive CFL).
    /// On the very first configuration (`step_count == 0`) also seeds `dt_old`
    /// so BDF2 starts with `r = dt/dt_old = 1` (unstructured TimeIntegrationModule).
    pub fn set_dt(&mut self, dt: f64) {
        self.dt = dt;
        if self.step_count == 0 {
            self.dt_old = dt;
        }
    }

    /// Live time-scheme switch (GUI); resets the BDF2 Euler-startup counter only
    /// if the caller rebuilds history — step_count is left intact so a mid-run
    /// scheme flip does not re-fire step-0 Euler.
    pub fn set_time_scheme(&mut self, scheme: TimeScheme) {
        self.time_scheme = scheme;
        self.constants.time_scheme = scheme as u32;
    }

    /// The dense grid.
    pub fn grid(&self) -> StructuredGrid {
        self.grid
    }

    /// The state layout (drives the UI ports).
    pub fn state_layout(&self) -> &crate::solver::model::backend::state_layout::StateLayout {
        &self.layout
    }

    /// The packed `f32` state (`n * state_stride`, cell-major) — the same layout
    /// the GPU solver's `state` buffer holds, for uploading to a renderer viz
    /// buffer (the GUI CPU backend feeds the GPU-direct renderer this way).
    pub fn packed_state_f32(&self) -> Vec<f32> {
        self.buffers.f32_vec("state")
    }

    /// Paired velocity `(Ux, Uy)` per cell — the GUI readback shape.
    pub fn get_u(&self, u_offset: usize) -> Vec<(f64, f64)> {
        let st = self.buffers.f32_vec("state");
        (0..self.grid.num_cells())
            .map(|p| {
                let b = p * self.state_stride + u_offset;
                (st[b] as f64, st[b + 1] as f64)
            })
            .collect()
    }

    /// A single scalar field per cell — the GUI readback shape.
    pub fn get_scalar(&self, offset: usize) -> Vec<f64> {
        self.state_field(offset)
    }

    fn run(&self, id: &str, n: usize, ctx: &Ctx) {
        // Transpiled (compiled-Rust) engine: run the structured transpiled kernel
        // if one was generated for this (model, kernel), passing the dense-grid
        // geometry as the extra `grid` param; otherwise fall through to the
        // interpreter (the correctness oracle). Same contract as the unstructured
        // `CpuSolver::run_kernel`.
        if self.engine == crate::solver::cpu::CpuEngine::Transpiled {
            if let Some(f) = crate::solver::cpu::generated::lookup_structured(self.model_id, id) {
                let grid = crate::solver::cpu::transpile_rt::StructuredGridRt {
                    nx: self.grid.nx as u32,
                    ny: self.grid.ny as u32,
                    dx: self.grid.dx as f32,
                    dy: self.grid.dy as f32,
                };
                // Structured binds the NEUTRAL low-Mach params (model=0), matching
                // the interpreter's build_ctx — so an all-zero uniform.
                let low_mach: crate::solver::gpu::structs::GpuLowMachParams =
                    bytemuck::Zeroable::zeroed();
                crate::solver::cpu::parallel::parallel_ranges(n, self.threads, |start, end| {
                    f(&self.buffers, start as u32, end as u32, &self.constants, &grid, &low_mach);
                });
                return;
            }
        }
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

    /// Assemble the banded operator once (prep + one flux/gradients/assembly
    /// sweep against the current state) WITHOUT solving — for GPU parity tests.
    pub fn assemble_only(&mut self) {
        let n = self.grid.num_cells();
        // Match `step()`: the time-step size drives the `ddt` diagonal; the
        // constructor leaves `constants.dt` at the recipe default until a step.
        self.constants.dt = self.dt as f32;
        self.constants.dt_old = self.dt_old as f32;
        let cur = self.buffers.f32_vec("state");
        self.buffers.copy_into_f32("state_old", &cur);
        self.buffers.copy_into_f32("state_iter", &cur);
        let ctx = self.build_ctx();
        let prep = self.prep.clone();
        for id in &prep {
            self.run(id, n, &ctx);
        }
        let per = self.per_iter.clone();
        for id in &per {
            self.run(id, n, &ctx);
        }
    }

    /// Read the assembled banded operator (`N*BAND*s*s`).
    pub fn matrix_values(&self) -> Vec<f32> {
        self.buffers.f32_vec("matrix_values")
    }

    /// Read the assembled RHS (`N*s`).
    pub fn rhs(&self) -> Vec<f32> {
        self.buffers.f32_vec("rhs")
    }

    /// Read a named buffer (debug/parity).
    pub fn read_buffer(&self, name: &str) -> Vec<f32> {
        self.buffers.f32_vec(name)
    }

    /// Run only the Preparation-phase kernels once (including the structured
    /// expression-BC closure `bc_expr`) against the currently-seeded state — for
    /// tests that assert what `bc_expr` writes into `bc_value` before any solve.
    #[cfg(test)]
    pub fn run_prep_for_test(&mut self) {
        let n = self.grid.num_cells();
        self.constants.dt = self.dt as f32;
        self.constants.dt_old = self.dt as f32;
        let ctx = self.build_ctx();
        let ids = self.prep.clone();
        for id in &ids {
            self.run(id, n, &ctx);
        }
    }

    /// Read one `bc_value` table entry for `(cell, dir, unknown)` — the structured
    /// key is `(cell*4 + dir)*s + unknown` (parity with `set_boundaries`).
    #[cfg(test)]
    pub fn bc_value_at(&self, cell: usize, dir: usize, unknown: usize) -> f64 {
        let bcv = self.buffers.f32_vec("bc_value");
        bcv[(cell * 4 + dir) * self.s + unknown] as f64
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
        let b64 = crate::solver::banded_schur::spmv(&a, nx, ny, s, &xstar);
        let b: Vec<f32> = b64.iter().map(|&v| v as f32).collect();

        let (x, rel_res) = crate::solver::banded_schur::banded_gmres(
            &a,
            nx,
            ny,
            s,
            &b,
            &crate::solver::banded_schur::BandedPrecond::BlockJacobi,
            40,
            200,
            1e-10,
        );
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
            // Top = MovingWall (5), other sides = Wall (3); velocity via bc_value.
            let btype = if matches!(edge, Edge::Top) { 5 } else { 3 };
            (btype, vec![
                BcComp { kind: 1, value: u_wall }, // Ux Dirichlet
                BcComp { kind: 1, value: 0.0 },    // Uy Dirichlet 0
                BcComp { kind: 2, value: 0.0 },    // p Neumann 0
            ])
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
            let btype = if matches!(edge, Edge::Top) { 5 } else { 3 };
            let mut v = vec![
                BcComp { kind: 1, value: u_wall },
                BcComp { kind: 1, value: 0.0 },
                BcComp { kind: 2, value: 0.0 },
            ];
            if s >= 4 {
                v.push(BcComp { kind: 2, value: 0.0 }); // T Neumann 0
            }
            (btype, v)
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

    /// End-to-end smoke of the density-based COMPRESSIBLE structured pipeline:
    /// a uniform gas at rest (rho=1, rho_u=0, rho_e=2.5 for p=1, gamma=1.4) in a
    /// box with the boundaries pinned to that state must stay uniform and bounded
    /// while the full structured conserved (rho, rho_u, rho_e) central-upwind
    /// operator runs. Proves compressible SOLVES structured on CPU (the
    /// structured expression-BC closure `bc_expr` runs in the prep phase).
    #[test]
    fn structured_compressible_uniform_box_runs() {
        use crate::solver::model::compressible_structured_model;
        let (nx, ny) = (16, 16);
        let grid = StructuredGrid::new(nx, ny, 1.0, 1.0);
        let model = compressible_structured_model().unwrap();
        let s = model.system.unknowns_per_cell() as usize; // rho, rho_ux, rho_uy, rho_e
        let mut solver = StructuredModelSolver::new(grid, &model, 0.01, 1).unwrap();
        solver.set_fluid(1.0, 0.0); // inviscid
        // Gas at rest: rho=1, rho_u=0, rho_e = p/(g-1) = 1/0.4 = 2.5.
        let (rho0, e0, p0) = (1.0, 2.5, 1.0);
        solver.set_named_field("rho", |_, _| rho0);
        solver.set_named_field("rho_e", |_, _| e0);
        solver.set_named_field("p", |_, _| p0);
        solver.set_named_field("T", |_, _| 1.0);
        if solver.field_offset("mu").is_some() {
            solver.set_named_field("mu", |_, _| 0.0);
        }
        // Boundaries pinned (Dirichlet) to the uniform conserved state.
        solver.set_boundaries(move |_edge, _x, _y| {
            let mut v = vec![
                BcComp { kind: 1, value: rho0 as f32 },
                BcComp { kind: 1, value: 0.0 },
                BcComp { kind: 1, value: 0.0 },
            ];
            if s >= 4 {
                v.push(BcComp { kind: 1, value: e0 as f32 });
            }
            (3, v) // Wall
        });

        for _ in 0..20 {
            solver.step();
        }
        let rho = solver.state_field(solver.field_offset("rho").unwrap());
        let rho_e = solver.state_field(solver.field_offset("rho_e").unwrap());
        for (&r, &re) in rho.iter().zip(&rho_e) {
            assert!(r.is_finite() && re.is_finite(), "compressible state diverged");
            assert!(r > 0.5 && r < 2.0, "compressible density drifted: {r}");
            assert!(re > 1.0 && re < 5.0, "compressible energy drifted: {re}");
        }
    }

    /// BC PARITY: the structured `bc_expr` closure must extrapolate outlet state
    /// from the interior exactly as the unstructured (face-dispatched) path does.
    /// The compressible outlet declares `rho := max(interior(rho), 1e-6)`; seed a
    /// non-uniform interior, tag every boundary Outlet with a sentinel `bc_value`,
    /// run the Preparation phase (which fires `bc_expr`), then assert each outlet
    /// face's `bc_value` now equals that owner cell's interior density — proving
    /// the cell-dispatched 4-face loop reads interior state and keys `bc_value` by
    /// `sfd_face_id` correctly (the sentinel is overwritten).
    #[test]
    fn structured_bc_expr_extrapolates_outlet_from_interior() {
        use crate::solver::model::compressible_structured_model;
        let (nx, ny) = (8, 6);
        let grid = StructuredGrid::new(nx, ny, 1.0, 1.0);
        let model = compressible_structured_model().unwrap();
        let s = model.system.unknowns_per_cell() as usize;
        let mut solver = StructuredModelSolver::new(grid, &model, 0.01, 1).unwrap();
        solver.set_fluid(1.0, 0.0);

        // Non-uniform interior density rho(x) = 1 + 0.3 x (all > 1e-6, so the
        // clamp is a no-op); the rest of the state kept thermodynamically sane.
        let rho_field = |x: f64, _y: f64| 1.0 + 0.3 * x;
        solver.set_named_field("rho", rho_field);
        solver.set_named_field("rho_e", |_, _| 2.5);
        solver.set_named_field("p", |_, _| 1.0);
        solver.set_named_field("T", |_, _| 1.0);
        if solver.field_offset("mu").is_some() {
            solver.set_named_field("mu", |_, _| 0.0);
        }

        // Every boundary = Outlet (type 2); the rho slot seeded with a sentinel
        // that `bc_expr` must overwrite from the interior.
        const SENTINEL: f32 = -999.0;
        solver.set_boundaries(move |_edge, _x, _y| {
            let mut v = vec![
                BcComp { kind: 0, value: SENTINEL },
                BcComp { kind: 0, value: 0.0 },
                BcComp { kind: 0, value: 0.0 },
            ];
            if s >= 4 {
                v.push(BcComp { kind: 0, value: 0.0 });
            }
            (2, v) // Outlet
        });

        solver.run_prep_for_test();

        // The rho coupled-unknown slot (parity with what `bc_expr` targeted).
        let flux_layout = crate::solver::model::FluxLayout::from_system(&model.system);
        let rho_unknown = flux_layout.offset_for("rho").expect("rho unknown") as usize;

        // Right-edge cells (i = nx-1): the East face (dir 2) is an outlet.
        let mut checked = 0;
        for j in 0..ny {
            let cell = j * nx + (nx - 1);
            let (cx, _cy) = solver.grid.cell_center(cell);
            let expected = rho_field(cx, 0.0).max(1e-6);
            let got = solver.bc_value_at(cell, 2, rho_unknown);
            assert!(
                (got - expected).abs() < 1e-5,
                "outlet rho bc mismatch at cell {cell}: got {got}, expected {expected}"
            );
            assert!(
                (got - SENTINEL as f64).abs() > 1.0,
                "bc_expr left the sentinel at cell {cell} — the closure did not run"
            );
            checked += 1;
        }
        assert_eq!(checked, ny, "expected one outlet face per right-edge row");
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
