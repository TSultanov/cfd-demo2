//! `CpuSolver`: the CPU backend's solve driver.
//!
//! Reuses the model's backend-agnostic IR end-to-end: it builds the same
//! `SolverRecipe` the GPU path builds, regenerates the model's kernels as typed
//! `KernelProgram`s, allocates CPU-side buffers from the mesh + state layout, and
//! runs the coupled solve loop by *interpreting* the kernels (flux → gradients →
//! assembly) and solving the assembled CSR system with a CPU iterative solver
//! (replacing the GPU FGMRES/AMG kernel stack), then interpreting the update
//! kernel.
//!
//! The public surface mirrors the subset of `GpuUnifiedSolver` the MMS tests use,
//! so the same manufactured-solution references validate this backend.

use std::collections::HashMap;

use cfd2_ir::ast::Stmt;

use crate::solver::cpu::interpreter::{Buffers, Ctx, Frame, Interpreter, Value};
use crate::solver::cpu::linalg::{
    bicgstab, fgmres, BlockCsr, BlockJacobi, CsrView, PointJacobi, Preconditioner, SchurPrecond,
};
use crate::solver::cpu::lowering::model_kernel_programs;
use crate::solver::cpu::{CpuBackendConfig, CpuEngine};
use crate::solver::gpu::enums::GpuBoundaryType;
use crate::solver::gpu::recipe::{KernelPhase, SolverRecipe, SteppingMode};
use crate::solver::gpu::structs::{GpuConstants, GpuLowMachParams, PreconditionerType};
use crate::solver::ir::DispatchDomain;
use crate::solver::mesh::Mesh;
use crate::solver::model::backend::SchemeRegistry;
use crate::solver::model::ModelSpec;
use crate::solver::scheme::Scheme;
use crate::solver::TimeScheme;

/// Linear-solve budget per outer iteration.
const LINEAR_MAX_ITERS: usize = 5000;
/// Fallback relative tolerance for the CPU linear solve (scalar path); the block
/// path uses the model's `linear_solver.tolerance` (the GPU inexact-Picard 1e-4).
const LINEAR_TOL: f64 = 1e-9;

/// A model kernel prepared for interpretation: its dispatch domain plus the
/// concatenated `indexing ++ preamble ++ body` statement list. (Execution
/// phase + ordering live in the `schedule`.)
struct CpuKernel {
    domain: DispatchDomain,
    stmts: Vec<Stmt>,
}

pub struct CpuSolver {
    model_id: &'static str,
    num_cells: usize,
    num_faces: usize,
    state_stride: u32,
    unknowns_per_cell: usize,

    buffers: Buffers,
    /// Kernels keyed by id (the program store).
    kernels: HashMap<String, CpuKernel>,
    /// The scheduled execution order (recipe order), with phases.
    schedule: Vec<ScheduledKernel>,

    // Block-CSR topology (block-level; shared by kernels and the CPU solver).
    scalar_row_offsets: Vec<u32>,
    col_indices: Vec<u32>,
    diagonal_indices: Vec<u32>,

    // Coupled-unknown base offset per equation-target field name.
    coupled_offsets: HashMap<String, u32>,

    // Faces grouped by boundary type index (`GpuBoundaryType as u32`).
    boundary_faces: Vec<Vec<u32>>,

    constants: GpuConstants,
    /// Low-Mach preconditioning + biharmonic-dissipation uniform (a separate
    /// uniform buffer on the GPU; bound by the compressible flux kernels).
    low_mach: GpuLowMachParams,
    state_layout: crate::solver::model::backend::StateLayout,

    stepping: SteppingMode,
    outer_iters: usize,
    /// Relative correction-norm tolerance for the adaptive outer break
    /// (0 = never break; matches the GPU outer gate pinned open).
    outer_tol: f64,
    /// CPU linear-solve relative tolerance (block path; GPU inexact-Picard).
    linear_tol: f64,
    linear_restart: usize,
    /// Block-system preconditioner choice (mirrors the recipe/runtime config).
    precond: PreconditionerType,
    /// Model-owned Schur preconditioner spec, when present: (velocity unknown
    /// indices, pressure unknown index, omega). Saddle-point models
    /// (incompressible/buoyant) use the CPU Schur preconditioner.
    schur: Option<(Vec<usize>, usize, f32)>,
    dt: f32,
    dt_old: f32,
    dtau: f32,
    time: f32,
    time_scheme: TimeScheme,
    /// Number of committed steps. Drives the BDF2 Euler startup (the first step,
    /// step_count == 0, falls back to Euler — there is no valid two-steps-ago
    /// state yet — exactly as the GPU's generic-coupled program does).
    step_count: u64,
    /// Relative per-step state change |x_n - x_{n-1}| / |x_n| from the last step,
    /// used for GUI steady-state auto-pause parity (see `should_stop`).
    last_rel_delta: f64,
    #[allow(dead_code)]
    config: CpuBackendConfig,
}

/// One scheduled dispatch: a kernel id and its phase (recipe order).
#[derive(Clone)]
struct ScheduledKernel {
    id: String,
    phase: KernelPhase,
}

impl CpuSolver {
    pub fn new(
        mesh: &Mesh,
        model: ModelSpec,
        advection_scheme: Scheme,
        time_scheme: TimeScheme,
        config: CpuBackendConfig,
    ) -> Result<Self, String> {
        Self::with_stepping(
            mesh,
            model,
            advection_scheme,
            time_scheme,
            SteppingMode::Coupled,
            config,
        )
    }

    pub fn with_stepping(
        mesh: &Mesh,
        model: ModelSpec,
        advection_scheme: Scheme,
        time_scheme: TimeScheme,
        stepping: SteppingMode,
        config: CpuBackendConfig,
    ) -> Result<Self, String> {
        // Kernel fusion is a GPU dispatch-overhead optimization; the CPU
        // interprets one kernel at a time, so disable it. The unfused schedule
        // is exactly the set of per-module kernel generators (each a typed
        // `DslProgram`), so every scheduled compute kernel is CPU-executable.
        let mut model = model;
        {
            let mut ls = model.linear_solver.unwrap_or_default();
            ls.solver.kernel_fusion_policy =
                crate::solver::model::kernel::KernelFusionPolicy::Off;
            model.linear_solver = Some(ls);
        }
        let recipe = SolverRecipe::from_model(
            &model,
            advection_scheme,
            time_scheme,
            PreconditionerType::Jacobi,
            stepping,
        )?;
        let schemes = SchemeRegistry::new(advection_scheme);
        // All model-module kernels as typed programs (a superset of the
        // scheduled set; the recipe selects + orders the ones that run).
        let (programs, _wgsl_only) = model_kernel_programs(&model, &schemes)?;
        let program_map: HashMap<String, _> =
            programs.into_iter().map(|(id, p)| (id.as_str().to_string(), p)).collect();

        // Phases that the CPU executes by interpreting a model kernel. The
        // LinearSolve phase (FGMRES/AMG/Schur WGSL infrastructure) is replaced
        // by the CPU's own solver; Apply is a monitor-only matvec (writes `y`,
        // not state) so it is skipped.
        let is_compute_phase = |p: KernelPhase| {
            matches!(
                p,
                KernelPhase::Preparation
                    | KernelPhase::Gradients
                    | KernelPhase::FluxComputation
                    | KernelPhase::ExplicitUpdate
                    | KernelPhase::Assembly
                    | KernelPhase::Update
                    | KernelPhase::PrimitiveRecovery
            )
        };

        let mut kernels: HashMap<String, CpuKernel> = HashMap::new();
        let mut schedule: Vec<ScheduledKernel> = Vec::new();
        for kspec in &recipe.kernels {
            let id = kspec.id.as_str().to_string();
            schedule.push(ScheduledKernel {
                id: id.clone(),
                phase: kspec.phase,
            });
            if kernels.contains_key(&id) {
                continue;
            }
            match program_map.get(&id) {
                Some(prog) => {
                    let mut stmts = Vec::with_capacity(
                        prog.indexing.len() + prog.preamble.len() + prog.body.len(),
                    );
                    stmts.extend_from_slice(&prog.indexing);
                    stmts.extend_from_slice(&prog.preamble);
                    stmts.extend_from_slice(&prog.body);
                    kernels.insert(
                        id.clone(),
                        CpuKernel {
                            domain: prog.dispatch.clone(),
                            stmts,
                        },
                    );
                }
                None => {
                    // A scheduled kernel with no typed program: tolerable only
                    // for the linear-solve / apply infrastructure the CPU
                    // replaces. A missing *compute* kernel is a real gap.
                    if is_compute_phase(kspec.phase) {
                        return Err(format!(
                            "model `{}` schedules compute kernel `{}` ({:?}) with no CPU-executable program",
                            model.id, id, kspec.phase
                        ));
                    }
                }
            }
        }

        let state_layout = model.state_layout.clone();
        let state_stride = state_layout.stride();
        let unknowns_per_cell = recipe.unknowns_per_cell;
        let s = unknowns_per_cell;

        // Coupled-unknown base offsets (equation-target order), mirroring
        // `coupled_offsets` in the codegen: field name -> first u_idx.
        let mut coupled_offsets: HashMap<String, u32> = HashMap::new();
        {
            let mut cur = 0u32;
            for eq in model.system.equations() {
                coupled_offsets.insert(eq.target().name().to_string(), cur);
                cur += eq.target().kind().component_count() as u32;
            }
        }

        let num_cells = mesh.num_cells();
        let num_faces = mesh.num_faces();

        let (scalar_row_offsets, col_indices, diagonal_indices, cell_face_matrix_indices) =
            build_csr_topology(mesh);
        let nnz_blocks = *scalar_row_offsets.last().unwrap() as usize;

        let mut buffers = Buffers::new();
        upload_mesh(&mut buffers, mesh);
        buffers.insert_u32("scalar_row_offsets", scalar_row_offsets.clone());
        buffers.insert_u32("row_offsets", scalar_row_offsets.clone());
        buffers.insert_u32("col_indices", col_indices.clone());
        buffers.insert_u32("diagonal_indices", diagonal_indices.clone());
        buffers.insert_u32("cell_face_matrix_indices", cell_face_matrix_indices);

        // State + history + per-iteration snapshot.
        let state_len = num_cells * state_stride as usize;
        buffers.insert_f32("state", vec![0.0; state_len]);
        buffers.insert_f32("state_old", vec![0.0; state_len]);
        buffers.insert_f32("state_old_old", vec![0.0; state_len]);
        buffers.insert_f32("state_iter", vec![0.0; state_len]);

        // Flux table: `flux_stride` floats per face (the packed coupled-unknown
        // layout); falls back to 1 for models with no flux buffer.
        let flux_stride = recipe.flux.map(|f| f.stride as usize).unwrap_or(1);
        buffers.insert_f32("fluxes", vec![0.0; num_faces * flux_stride]);
        // grad_state mirrors the state layout: one Vector2 gradient per state slot
        // (indexed `grad_state[cell * stride + component]`), so it holds
        // `num_cells * stride` Vector2 elements (× 2 floats each).
        buffers.insert_vec2(
            "grad_state",
            vec![0.0; num_cells * state_stride as usize * 2],
        );
        // Block-CSR: nnz_blocks * S*S matrix entries; rhs/x packed `[cell*S + u]`.
        buffers.insert_f32("matrix_values", vec![0.0; nnz_blocks * s * s]);
        buffers.insert_f32("rhs", vec![0.0; num_cells * s]);
        buffers.insert_f32("x", vec![0.0; num_cells * s]);
        buffers.insert_f32("y", vec![0.0; num_cells * s]);

        // Boundary conditions: per face x coupled-unknown component.
        let (bc_kind, bc_value) = build_bc_tables(mesh, &model, s)?;
        buffers.insert_u32("bc_kind", bc_kind);
        buffers.insert_f32("bc_value", bc_value);

        let boundary_faces = group_boundary_faces(mesh);

        let mut constants = recipe.initial_constants;
        constants.dtau = 0.0; // dual-time off by default
        constants.time_scheme = time_scheme as u32;

        let outer_iters = match stepping {
            SteppingMode::Implicit { outer_iters } => outer_iters.max(1),
            _ => 2,
        };

        // Model-owned Schur preconditioner (saddle-point models).
        let schur = match model.linear_solver.and_then(|ls| match ls.preconditioner {
            crate::solver::model::ModelPreconditionerSpec::Schur { omega, layout } => {
                Some((layout, omega))
            }
            _ => None,
        }) {
            Some((layout, omega)) => Some((
                layout.u_indices().iter().map(|&u| u as usize).collect::<Vec<usize>>(),
                layout.p as usize,
                omega,
            )),
            None => None,
        };

        Ok(Self {
            model_id: model.id,
            num_cells,
            num_faces,
            state_stride,
            unknowns_per_cell,
            buffers,
            kernels,
            schedule,
            scalar_row_offsets,
            col_indices,
            diagonal_indices,
            coupled_offsets,
            boundary_faces,
            constants,
            low_mach: GpuLowMachParams::default(),
            state_layout,
            stepping,
            outer_iters,
            outer_tol: 0.0,
            linear_tol: recipe.linear_solver.tolerance as f64,
            linear_restart: match recipe.linear_solver.solver_type {
                crate::solver::gpu::recipe::LinearSolverType::Fgmres { max_restart } => max_restart,
                crate::solver::gpu::recipe::LinearSolverType::Cg => 60,
            },
            precond: recipe.linear_solver.preconditioner,
            schur,
            dt: 0.01,
            dt_old: 0.01,
            dtau: 0.0,
            time: 0.0,
            time_scheme,
            step_count: 0,
            last_rel_delta: f64::INFINITY,
            config,
        })
    }

    // ── configuration ────────────────────────────────────────────────────

    pub fn set_outer_iters(&mut self, n: usize) {
        self.outer_iters = n.max(1);
    }
    pub fn set_dt(&mut self, dt: f32) {
        self.dt = dt;
    }
    pub fn set_dtau(&mut self, dtau: f32) {
        self.dtau = dtau;
    }
    /// Relative correction-norm tolerance for the adaptive outer-iteration
    /// break (0 keeps every requested iteration, matching the GPU outer gate
    /// pinned open). Mirrors `set_outer_tolerance` on the GPU solver.
    pub fn set_outer_tolerance(&mut self, tol: f64) {
        self.outer_tol = tol;
    }
    pub fn set_time_scheme(&mut self, scheme: TimeScheme) {
        self.time_scheme = scheme;
        self.constants.time_scheme = scheme as u32;
    }
    /// The time scheme actually applied to the current step's assembly: BDF2
    /// falls back to Euler on the very first step (no valid two-steps-ago state),
    /// matching the GPU generic-coupled program's OpenFOAM-style backward startup.
    fn effective_time_scheme(&self) -> TimeScheme {
        if self.time_scheme == TimeScheme::BDF2 && self.step_count == 0 {
            TimeScheme::Euler
        } else {
            self.time_scheme
        }
    }
    pub fn set_advection_scheme(&mut self, scheme: Scheme) {
        self.constants.scheme = scheme.gpu_id();
    }
    pub fn set_viscosity(&mut self, mu: f32) {
        self.constants.viscosity = mu;
    }
    pub fn set_density(&mut self, rho: f32) {
        self.constants.density = rho;
    }
    /// Update an EOS runtime constant by its `eos.<field>` param name, mirroring
    /// the GPU `set_eos` (which writes the same constants the assembly reads). Lets
    /// the GUI's fluid/EOS tuning take effect on the CPU backend too. Returns
    /// whether the name was a recognized EOS field.
    pub fn set_eos_param(&mut self, name: &str, v: f32) -> bool {
        match name {
            "eos.gamma" => self.constants.eos_gamma = v,
            "eos.gm1" => self.constants.eos_gm1 = v,
            "eos.r" => self.constants.eos_r = v,
            "eos.dp_drho" => self.constants.eos_dp_drho = v,
            "eos.p_offset" => self.constants.eos_p_offset = v,
            "eos.theta_ref" => self.constants.eos_theta_ref = v,
            _ => return false,
        }
        true
    }
    pub fn set_alpha_u(&mut self, alpha: f32) {
        self.constants.alpha_u = alpha;
    }
    pub fn set_alpha_p(&mut self, alpha: f32) {
        self.constants.alpha_p = alpha;
    }
    /// Biharmonic-dissipation coefficient (the `compressible_mms_biharmonic`
    /// model's `+eps4*(lap_neigh-lap_own)` flux term). Mirrors the GPU's
    /// `set_named_param("low_mach.eps4", …)`.
    pub fn set_eps4(&mut self, eps4: f32) {
        self.low_mach.eps4 = eps4;
    }

    // ── field I/O ─────────────────────────────────────────────────────────

    pub fn set_field_scalar(&mut self, field: &str, values: &[f64]) -> Result<(), String> {
        let off = self
            .state_layout
            .offset_for(field)
            .ok_or_else(|| format!("unknown field `{field}`"))? as usize;
        let stride = self.state_stride as usize;
        for (i, &v) in values.iter().enumerate() {
            self.buffers.set_f32("state", i * stride + off, v as f32);
        }
        Ok(())
    }

    pub fn set_field_scalar_current(&mut self, field: &str, values: &[f64]) -> Result<(), String> {
        self.set_field_scalar(field, values)
    }

    pub fn set_field_vec2(&mut self, field: &str, values: &[(f64, f64)]) -> Result<(), String> {
        let off = self
            .state_layout
            .offset_for(field)
            .ok_or_else(|| format!("unknown field `{field}`"))? as usize;
        let stride = self.state_stride as usize;
        for (i, &(x, y)) in values.iter().enumerate() {
            self.buffers.set_f32("state", i * stride + off, x as f32);
            self.buffers.set_f32("state", i * stride + off + 1, y as f32);
        }
        Ok(())
    }

    pub fn get_field_scalar(&self, field: &str) -> Result<Vec<f64>, String> {
        let off = self
            .state_layout
            .offset_for(field)
            .ok_or_else(|| format!("unknown field `{field}`"))? as usize;
        let stride = self.state_stride as usize;
        Ok((0..self.num_cells)
            .map(|i| self.buffers.get_f32("state", i * stride + off) as f64)
            .collect())
    }

    pub fn get_field_vec2(&self, field: &str) -> Result<Vec<(f64, f64)>, String> {
        let off = self
            .state_layout
            .offset_for(field)
            .ok_or_else(|| format!("unknown field `{field}`"))? as usize;
        let stride = self.state_stride as usize;
        Ok((0..self.num_cells)
            .map(|i| {
                (
                    self.buffers.get_f32("state", i * stride + off) as f64,
                    self.buffers.get_f32("state", i * stride + off + 1) as f64,
                )
            })
            .collect())
    }

    // ── backend-routing accessors (used by UnifiedSolver) ──────────────────

    pub fn num_cells(&self) -> u32 {
        self.num_cells as u32
    }
    pub fn time(&self) -> f32 {
        self.time
    }
    pub fn dt(&self) -> f32 {
        self.dt
    }
    pub fn state_stride(&self) -> u32 {
        self.state_stride
    }

    /// Full packed state (all components, `num_cells * stride` floats).
    pub fn read_state_f32(&self) -> Vec<f32> {
        self.buffers.f32_vec("state")
    }

    /// Overwrite the full packed state.
    pub fn write_state_f32(&self, state: &[f32]) -> Result<(), String> {
        let expected = self.num_cells * self.state_stride as usize;
        if state.len() != expected {
            return Err(format!(
                "state length {} != expected {expected}",
                state.len()
            ));
        }
        self.buffers.copy_into_f32("state", state);
        Ok(())
    }

    // ── boundary conditions ───────────────────────────────────────────────

    pub fn set_boundary_values_per_face(
        &mut self,
        boundary: GpuBoundaryType,
        field: &str,
        component: u32,
        value_for_face: &dyn Fn(u32) -> f32,
    ) -> Result<(), String> {
        let bidx = boundary as usize;
        let stride = self.unknowns_per_cell;
        // Map field+component to the coupled-unknown index. Equation-target
        // fields use their coupled offset; a field that is not itself a coupled
        // unknown (e.g. a primitive whose BC seeds an expression closure) falls
        // back to the raw component (single-unknown / scalar models).
        let u_idx = match self.coupled_offsets.get(field) {
            Some(&base) => (base + component) as usize,
            None => component as usize,
        };
        if u_idx >= stride {
            return Err(format!(
                "boundary field `{field}` component {component} maps to u_idx {u_idx} >= stride {stride}"
            ));
        }
        let faces = self
            .boundary_faces
            .get(bidx)
            .ok_or_else(|| format!("invalid boundary index {bidx}"))?
            .clone();
        for face in faces {
            self.buffers
                .set_f32("bc_value", face as usize * stride + u_idx, value_for_face(face));
        }
        Ok(())
    }

    pub fn set_boundary_scalar(
        &mut self,
        boundary: GpuBoundaryType,
        field: &str,
        value: f32,
    ) -> Result<(), String> {
        self.set_boundary_values_per_face(boundary, field, 0, &|_| value)
    }

    // ── stepping ──────────────────────────────────────────────────────────

    pub fn initialize_history(&self) {
        let state = self.buffers.f32_vec("state");
        self.buffers.copy_into_f32("state_old", &state);
        self.buffers.copy_into_f32("state_old_old", &state);
        self.buffers.copy_into_f32("state_iter", &state);
        // Warm-start the solve buffer `x` from the coupled unknowns in the state.
        // The compressible EOS-coupled block system is rank-deficient (the
        // recovery rows admit a null-space the residual does not pin), so a zero
        // initial guess lets the FIRST solve wander along the null-space and seed
        // a drift that the marginally-stable MMS march then amplifies (the GPU,
        // whose `x` persists in sync with the state, does not). Pin `x` to the
        // initial state so the first warm-start fixes the null-space component.
        self.sync_x_from_state();
    }

    /// Pack the coupled unknowns out of the packed state into the solve buffer
    /// `x` (block-CSR unknown order). Each coupled field's `x` base comes from
    /// `coupled_offsets`; its component width is the gap to the next base (last
    /// to `unknowns_per_cell`); its source offset in the state from the layout.
    fn sync_x_from_state(&self) {
        let s = self.unknowns_per_cell;
        if s <= 1 {
            return; // scalar path solves for the field directly; no packing.
        }
        let stride = self.state_stride as usize;
        let mut offs: Vec<(u32, String)> = self
            .coupled_offsets
            .iter()
            .map(|(k, &v)| (v, k.clone()))
            .collect();
        offs.sort();
        let state = self.buffers.f32_vec("state");
        let mut x = vec![0.0f32; self.num_cells * s];
        for (i, (xbase, field)) in offs.iter().enumerate() {
            let xbase = *xbase as usize;
            let next = offs.get(i + 1).map(|(o, _)| *o as usize).unwrap_or(s);
            let width = next - xbase;
            let Some(soff) = self.state_layout.offset_for(field) else {
                continue;
            };
            let soff = soff as usize;
            for cell in 0..self.num_cells {
                for c in 0..width {
                    x[cell * s + xbase + c] = state[cell * stride + soff + c];
                }
            }
        }
        self.buffers.copy_into_f32("x", &x);
    }

    pub fn step(&mut self) {
        // Rotate time history: old_old <- old, old <- current state.
        let old = self.buffers.f32_vec("state_old");
        self.buffers.copy_into_f32("state_old_old", &old);
        let cur = self.buffers.f32_vec("state");
        self.buffers.copy_into_f32("state_old", &cur);

        self.constants.dt = self.dt;
        self.constants.dt_old = self.dt_old;
        self.constants.dtau = self.dtau;
        self.time += self.dt;
        self.constants.time = self.time;
        self.constants.time_scheme = self.effective_time_scheme() as u32;
        let ctx = constants_ctx(&self.constants, &self.low_mach);

        // Drive in RECIPE (schedule) ORDER, which is the order the GPU executes.
        // Per-iteration kernels are the Gradients/FluxComputation/Assembly passes;
        // the recurring `bc_expr` runs at the END of each outer iteration (after
        // the update), NOT before the assembly. This timing is load-bearing and
        // matches the GPU's generic-coupled loop: the recurring boundary-closure
        // refresh prepares the ghosts for the NEXT iteration/step, so an outer
        // iteration's gradients/flux/assembly all see the ghosts produced by the
        // PRIOR iteration (seeded values on the very first step). Running bc_expr
        // before the assembly instead fed step-1 the refreshed ghosts and biased
        // the boundary energy row (rho_e drift that broke the long MMS march).
        // Once-only Preparation kernels (e.g. rhie_chow dp_init) run before the
        // loop; the CPU linear solve replaces the LinearSolve phase after
        // assembly; Apply is monitor-only (skipped); Update/PrimitiveRecovery
        // apply the solution.
        let is_bc_expr = |id: &str| id.contains("bc_expr");
        let per_iter: Vec<String> = self
            .schedule
            .iter()
            .filter(|s| {
                matches!(
                    s.phase,
                    KernelPhase::Gradients | KernelPhase::FluxComputation | KernelPhase::Assembly
                )
            })
            .map(|s| s.id.clone())
            .collect();
        let bc_expr_ids: Vec<String> = self
            .schedule
            .iter()
            .filter(|s| s.phase == KernelPhase::Preparation && is_bc_expr(&s.id))
            .map(|s| s.id.clone())
            .collect();
        let prep_once: Vec<String> = self
            .schedule
            .iter()
            .filter(|s| s.phase == KernelPhase::Preparation && !is_bc_expr(&s.id))
            .map(|s| s.id.clone())
            .collect();
        let update_group: Vec<String> = self
            .schedule
            .iter()
            .filter(|s| matches!(s.phase, KernelPhase::Update | KernelPhase::PrimitiveRecovery))
            .map(|s| s.id.clone())
            .collect();

        let threads = self.config.threads;
        let engine = self.config.engine;
        let model_id = self.model_id;
        let nf = self.num_faces;
        let nc = self.num_cells;
        let run = |id: &str| {
            run_kernel(
                &self.buffers,
                &ctx,
                &self.kernels,
                id,
                nf,
                nc,
                threads,
                engine,
                model_id,
                &self.constants,
            );
        };

        // Optional per-phase wall-time profiling (CFD2_CPU_PROFILE=1). Attributes
        // step wall time to the parallel dispatch groups vs. the (serial) CPU
        // linear solve, so the parallel/serial split is visible directly.
        let profile = std::env::var("CFD2_CPU_PROFILE").is_ok();
        let (mut t_prep, mut t_asm, mut t_lin, mut t_upd, mut t_bc) = (
            std::time::Duration::ZERO,
            std::time::Duration::ZERO,
            std::time::Duration::ZERO,
            std::time::Duration::ZERO,
            std::time::Duration::ZERO,
        );
        macro_rules! timed {
            ($acc:expr, $body:expr) => {{
                if profile {
                    let _t0 = std::time::Instant::now();
                    let _r = $body;
                    $acc += _t0.elapsed();
                    _r
                } else {
                    $body
                }
            }};
        }

        // Prepare once per step (non-bc_expr Preparation kernels).
        timed!(t_prep, {
            for id in &prep_once {
                run(id);
            }
        });

        for _ in 0..self.outer_iters {
            // Snapshot current iterate (dual-time reference + outer-break delta).
            let snap = self.buffers.f32_vec("state");
            self.buffers.copy_into_f32("state_iter", &snap);

            // Per-iteration kernels in schedule order (gradients/flux/assembly),
            // then the CPU linear solve (replacing LinearSolve), then the update
            // group, then the recurring boundary-closure refresh (bc_expr) which
            // prepares the ghosts for the next iteration/step.
            timed!(t_asm, {
                for id in &per_iter {
                    run(id);
                }
            });
            timed!(t_lin, self.linear_solve());
            timed!(t_upd, {
                for id in &update_group {
                    run(id);
                }
            });
            timed!(t_bc, {
                for id in &bc_expr_ids {
                    run(id);
                }
            });

            // Adaptive outer break (off when outer_tol == 0).
            if self.outer_tol > 0.0 {
                let cur = self.buffers.f32_vec("state");
                let (mut maxd, mut maxs) = (0.0f32, 0.0f32);
                for i in 0..cur.len() {
                    maxd = maxd.max((cur[i] - snap[i]).abs());
                    maxs = maxs.max(cur[i].abs());
                }
                if (maxd as f64) <= self.outer_tol * (maxs as f64 + 1e-30) {
                    break;
                }
            }
        }

        if profile {
            let tot = t_prep + t_asm + t_lin + t_upd + t_bc;
            let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
            let pct = |d: std::time::Duration| 100.0 * d.as_secs_f64() / tot.as_secs_f64().max(1e-30);
            eprintln!(
                "[cpu-profile] step {} threads={} tot={:.1}ms | assembly(par)={:.1}ms({:.0}%) \
                 linsolve(serial)={:.1}ms({:.0}%) update(par)={:.1}ms({:.0}%) \
                 bc_expr(interp)={:.1}ms({:.0}%) prep={:.1}ms({:.0}%)",
                self.step_count, self.config.threads, ms(tot),
                ms(t_asm), pct(t_asm), ms(t_lin), pct(t_lin), ms(t_upd), pct(t_upd),
                ms(t_bc), pct(t_bc), ms(t_prep), pct(t_prep),
            );
        }

        // Per-step relative state change, for the GUI steady-state auto-pause
        // (`should_stop`). `state_old` holds the pre-step state (rotated in at the
        // top of this step), so this is |x_n - x_{n-1}| / |x_n|.
        {
            let cur = self.buffers.f32_vec("state");
            let old = self.buffers.f32_vec("state_old");
            let (mut maxd, mut maxs) = (0.0f64, 0.0f64);
            for i in 0..cur.len().min(old.len()) {
                maxd = maxd.max((cur[i] - old[i]).abs() as f64);
                maxs = maxs.max(cur[i].abs() as f64);
            }
            self.last_rel_delta = maxd / (maxs + 1e-30);
        }

        self.dt_old = self.dt;
        self.step_count += 1;
    }

    /// Steady-state auto-pause for the GUI, mirroring the GPU's convergence-monitor
    /// `should_stop` (which is only active under pseudo-transient continuation):
    /// when `dtau > 0` and the per-step relative state change has fallen below a
    /// small threshold after a few steps, the run is steady and the GUI may pause.
    /// With `dtau == 0` (plain transient) it never fires, matching the GPU.
    pub fn should_stop(&self) -> bool {
        self.dtau > 0.0 && self.step_count >= 5 && self.last_rel_delta < 1e-6
    }

    /// Debug: run prepare + the assembly group (gradients, flux, assembly) at the
    /// current state and return the assembled `(matrix_values, rhs)`. Used by the
    /// MMS consistency check to compute the per-equation discrete residual at the
    /// exact solution. Does NOT solve or update.
    pub fn debug_assemble(&mut self) -> (Vec<f32>, Vec<f32>) {
        self.constants.dt = self.dt;
        self.constants.dt_old = self.dt_old;
        self.constants.dtau = self.dtau;
        self.constants.time = self.time;
        self.constants.time_scheme = self.effective_time_scheme() as u32;
        let ctx = constants_ctx(&self.constants, &self.low_mach);
        let collect = |phases: &[KernelPhase]| -> Vec<String> {
            self.schedule
                .iter()
                .filter(|s| phases.contains(&s.phase))
                .map(|s| s.id.clone())
                .collect()
        };
        // Exclude bc_expr: like a real step-1 outer iteration, the boundary
        // closure refresh runs at the END (it prepares the NEXT iter), so the
        // assembly sees the SEEDED ghosts. Including it here would assemble
        // against refreshed ghosts and disagree with the marched step.
        let prep: Vec<String> = collect(&[KernelPhase::Preparation])
            .into_iter()
            .filter(|id| !id.contains("bc_expr"))
            .collect();
        let assembly_group = collect(&[
            KernelPhase::Gradients,
            KernelPhase::FluxComputation,
            KernelPhase::Assembly,
        ]);
        let (threads, engine, model_id, nf, nc) = (
            self.config.threads,
            self.config.engine,
            self.model_id,
            self.num_faces,
            self.num_cells,
        );
        let run = |id: &str| {
            run_kernel(
                &self.buffers, &ctx, &self.kernels, id, nf, nc, threads, engine, model_id,
                &self.constants,
            );
        };
        for id in &prep {
            run(id);
        }
        for id in &assembly_group {
            run(id);
        }
        (self.buffers.f32_vec("matrix_values"), self.buffers.f32_vec("rhs"))
    }

    /// Debug: read the per-face `bc_value` buffer (length `num_faces * S`).
    pub fn debug_bc_value(&self) -> Vec<f32> {
        self.buffers.f32_vec("bc_value")
    }

    /// Block-CSR topology accessors (for the MMS residual consistency check).
    pub fn debug_topology(&self) -> (&[u32], &[u32], &[u32], usize) {
        (
            &self.scalar_row_offsets,
            &self.col_indices,
            &self.diagonal_indices,
            self.unknowns_per_cell,
        )
    }

    /// Solve the assembled block system `A x = rhs` on the CPU; the result lands
    /// in the `x` buffer that the update/apply kernel consumes. Takes `&self`: it
    /// mutates only the (interior-mutable atomic) buffers, not solver fields.
    ///
    /// `x` starts at zero each solve: the assembled system has a unique solution
    /// independent of the guess, so this is correct whether the model's update is
    /// Picard (x = new state) or Newton (x = correction). FGMRES + block-Jacobi
    /// mirrors the GPU FGMRES(restart); the scalar (`S==1`) path keeps the
    /// validated BiCGSTAB.
    fn linear_solve(&self) {
        let s = self.unknowns_per_cell;
        let n = self.num_cells * s;
        // The assembled matrix is the largest buffer (nnz_blocks * S*S entries);
        // marshal it out of the atomic store in parallel. `rhs`/`x` are O(n), small.
        let matrix = self.buffers.f32_vec_threaded("matrix_values", self.config.threads);
        let rhs = self.buffers.f32_vec("rhs");
        // Warm-start from the persisted `x` buffer (the previous solve's
        // solution, which the update kernel keeps in sync with the state). The
        // EOS-coupled compressible block system is rank-deficient (the recovery
        // rows admit a null-space the residual does not pin), so a zero guess
        // lets the Krylov solve wander along the null-space and drift the
        // marched solution; warm-starting pins the null-space component to the
        // current state, matching the GPU (whose `x` buffer likewise persists).
        let mut x = self.buffers.f32_vec("x");
        if x.len() != n {
            x = vec![0.0f32; n];
        }

        let threads = self.config.threads;
        if s == 1 {
            let a = CsrView {
                row_offsets: &self.scalar_row_offsets,
                col_indices: &self.col_indices,
                values: &matrix,
                threads,
            };
            bicgstab(&a, &rhs, &mut x, LINEAR_MAX_ITERS, LINEAR_TOL, self.config.simd);
        } else {
            let a = BlockCsr {
                s,
                scalar_row_offsets: &self.scalar_row_offsets,
                col_indices: &self.col_indices,
                diagonal_indices: &self.diagonal_indices,
                values: &matrix,
                threads,
            };
            // Preconditioner: the model-owned Schur complement for saddle-point
            // systems (incompressible/buoyant); otherwise the per-cell block
            // Jacobi — the GPU's generic-coupled FGMRES preconditioner for ALL
            // coupled (S>1) systems is `block_precond.wgsl` (a full b×b block
            // inverse), NOT a scalar point Jacobi. The compressible EOS recovery
            // rows (p, T) couple strongly WITHIN a cell and leave the block
            // system rank-deficient; a scalar diagonal mishandles that coupling,
            // so the inexact solve drifts and the marginally-stable MMS march
            // diverges where the block-Jacobi GPU saturates. Point-Jacobi is kept
            // only as a debug toggle (CFD2_CPU_POINT_JACOBI).
            let block_pc: Box<dyn Preconditioner> = match &self.schur {
                Some((u_idx, p, omega)) => {
                    Box::new(SchurPrecond::new(a, u_idx, *p, *omega as f64, self.config.simd))
                }
                None => {
                    if std::env::var("CFD2_CPU_POINT_JACOBI").is_ok() {
                        Box::new(PointJacobi::new(&a))
                    } else {
                        Box::new(BlockJacobi::new(&a))
                    }
                }
            };
            let _ = self.precond;
            let precond = block_pc.as_ref();
            let tol = std::env::var("CFD2_CPU_LINTOL")
                .ok()
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(self.linear_tol);
            let stats = fgmres(
                &a,
                &rhs,
                &mut x,
                precond,
                self.linear_restart,
                LINEAR_MAX_ITERS,
                tol,
                self.config.simd,
            );
            if std::env::var("CFD2_CPU_DEBUG_SOLVE").is_ok() {
                eprintln!(
                    "[cpu-solve] block S={s} n={n} iters={} rel_res={:.3e} conv={}",
                    stats.iters, stats.rel_residual, stats.converged
                );
            }
        }

        self.buffers.copy_into_f32("x", &x);
    }
}

// ── helpers ───────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_kernel(
    buffers: &Buffers,
    ctx: &Ctx,
    kernels: &HashMap<String, CpuKernel>,
    id: &str,
    num_faces: usize,
    num_cells: usize,
    threads: usize,
    engine: CpuEngine,
    model_id: &str,
    constants: &GpuConstants,
) {
    let kernel = kernels
        .get(id)
        .unwrap_or_else(|| panic!("CPU backend missing kernel `{id}`"));
    let domain = match kernel.domain {
        DispatchDomain::Faces => num_faces,
        DispatchDomain::Cells => num_cells,
        DispatchDomain::Custom(_) => panic!("custom dispatch domain unsupported on CPU"),
    };

    // Transpiled engine: run the compiled-Rust kernel if one was generated for
    // this (model, kernel); otherwise fall back to the interpreter.
    if engine == CpuEngine::Transpiled {
        if let Some(f) = crate::solver::cpu::generated::lookup(model_id, id) {
            crate::solver::cpu::parallel::parallel_for(domain, threads, |idx| {
                f(buffers, idx as u32, constants);
            });
            return;
        }
    }

    let stmts = &kernel.stmts;
    crate::solver::cpu::parallel::parallel_for(domain, threads, |idx| {
        // The launch wrapper's `let idx = <invocation_index_expr>;` and the
        // `if (idx >= bound) return;` guard are synthesized by the WGSL emitter
        // from LaunchSemantics, not stored in the kernel body. On CPU we own the
        // dispatch loop, so bind `idx` directly (and `global_id` for any kernel
        // that reads it). We iterate exactly `[0, domain)`, so the bound guard is
        // always false and can be skipped.
        let mut frame = Frame::new()
            .with_local("idx", Value::U32(idx as u32))
            .with_local("global_id", Value::Vec3([idx as f32, 0.0, 0.0]));
        Interpreter::new(buffers, ctx).run(stmts, &mut frame);
    });
}

/// Build the uniform structs the kernels read as interpreter `Ctx` structs:
/// `constants.*` (GpuConstants incl. EOS + buoyant tail) and `low_mach_params.*`
/// (low-Mach preconditioning + biharmonic dissipation).
fn constants_ctx(c: &GpuConstants, lm: &GpuLowMachParams) -> Ctx {
    Ctx::new()
        .with_constant("low_mach_params", "model", Value::U32(lm.model))
        .with_constant("low_mach_params", "theta_floor", Value::F32(lm.theta_floor))
        .with_constant(
            "low_mach_params",
            "pressure_coupling_alpha",
            Value::F32(lm.pressure_coupling_alpha),
        )
        .with_constant("low_mach_params", "eps4", Value::F32(lm.eps4))
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
        // Buoyant Boussinesq tail (mirrors the buoyant uniform port manifest).
        .with_constant("constants", "buoyant_beta_g", Value::F32(c.buoyant_beta_g))
        .with_constant("constants", "buoyant_t0", Value::F32(c.buoyant_t0))
        .with_constant("constants", "buoyant_k_over_cp", Value::F32(c.buoyant_k_over_cp))
}

/// Upload mesh geometry/topology into named CPU buffers matching kernel bindings.
fn upload_mesh(buffers: &mut Buffers, mesh: &Mesh) {
    let nf = mesh.num_faces();
    buffers.insert_u32(
        "face_owner",
        mesh.face_owner.iter().map(|&o| o as u32).collect(),
    );
    buffers.insert_i32(
        "face_neighbor",
        mesh.face_neighbor
            .iter()
            .map(|n| n.map(|v| v as i32).unwrap_or(-1))
            .collect(),
    );
    buffers.insert_f32("face_areas", mesh.face_area.iter().map(|&a| a as f32).collect());
    buffers.insert_vec2("face_normals", interleave(&mesh.face_nx, &mesh.face_ny));
    buffers.insert_vec2("cell_centers", interleave(&mesh.cell_cx, &mesh.cell_cy));
    buffers.insert_vec2("face_centers", interleave(&mesh.face_cx, &mesh.face_cy));
    buffers.insert_f32("cell_vols", mesh.cell_vol.iter().map(|&v| v as f32).collect());
    buffers.insert_u32(
        "cell_face_offsets",
        mesh.cell_face_offsets.iter().map(|&o| o as u32).collect(),
    );
    buffers.insert_u32(
        "cell_faces",
        mesh.cell_faces.iter().map(|&f| f as u32).collect(),
    );
    buffers.insert_u32(
        "face_boundary",
        mesh.face_boundary
            .iter()
            .map(|b| b.map(|t| t.bc_table_index() as u32).unwrap_or(0))
            .collect(),
    );
    // face_wrap_shift is empty on non-periodic meshes (treat as zeros).
    let wrap: Vec<f32> = if mesh.face_wrap_shift.is_empty() {
        vec![0.0; nf * 2]
    } else {
        mesh.face_wrap_shift
            .iter()
            .flat_map(|s| [s[0] as f32, s[1] as f32])
            .collect()
    };
    buffers.insert_vec2("face_wrap_shift", wrap);
}

fn interleave(xs: &[f64], ys: &[f64]) -> Vec<f32> {
    xs.iter()
        .zip(ys)
        .flat_map(|(&x, &y)| [x as f32, y as f32])
        .collect()
}

/// Construct the scalar CSR topology consistent with the assembly kernel's index
/// maps: each row holds the diagonal (rank 0) followed by one entry per interior
/// face. Returns `(row_offsets, col_indices, diagonal_indices,
/// cell_face_matrix_indices)`.
fn build_csr_topology(mesh: &Mesh) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let n = mesh.num_cells();
    let mut row_offsets = vec![0u32; n + 1];
    for i in 0..n {
        let start = mesh.cell_face_offsets[i];
        let end = mesh.cell_face_offsets[i + 1];
        let interior = (start..end)
            .filter(|&k| mesh.face_neighbor[mesh.cell_faces[k]].is_some())
            .count();
        row_offsets[i + 1] = row_offsets[i] + 1 + interior as u32;
    }
    let nnz = *row_offsets.last().unwrap() as usize;
    let mut col_indices = vec![0u32; nnz];
    let mut diagonal_indices = vec![0u32; n];
    let mut cell_face_matrix_indices = vec![0u32; mesh.cell_faces.len()];

    for i in 0..n {
        let base = row_offsets[i] as usize;
        col_indices[base] = i as u32;
        diagonal_indices[i] = base as u32;
        let mut pos = base + 1;
        let start = mesh.cell_face_offsets[i];
        let end = mesh.cell_face_offsets[i + 1];
        for k in start..end {
            let f = mesh.cell_faces[k];
            match mesh.face_neighbor[f] {
                Some(nb) => {
                    let other = if mesh.face_owner[f] == i { nb } else { mesh.face_owner[f] };
                    col_indices[pos] = other as u32;
                    cell_face_matrix_indices[k] = pos as u32;
                    pos += 1;
                }
                None => {
                    // Boundary face: no column entry; point at the diagonal so any
                    // stray read is harmless (the kernel guards with is_boundary).
                    cell_face_matrix_indices[k] = base as u32;
                }
            }
        }
    }
    (row_offsets, col_indices, diagonal_indices, cell_face_matrix_indices)
}

/// Build per-face `(bc_kind, bc_value)` tables (length `num_faces * S`) by
/// scattering the model's per-boundary-type tables onto boundary faces — exactly
/// as the GPU generic-coupled backend does (`row_base(i) = i * S`). Interior and
/// `None`-typed faces stay zero (kernels guard with `is_boundary`). The values
/// are *seeds*; `set_boundary_values_per_face` overrides them at runtime and the
/// `bc_expr` kernel refreshes expression-valued entries each iteration.
fn build_bc_tables(
    mesh: &Mesh,
    model: &ModelSpec,
    s: usize,
) -> Result<(Vec<u32>, Vec<f32>), String> {
    let (kind_by_type, value_by_type) = model
        .boundaries
        .to_gpu_tables(&model.system)
        .map_err(|e| format!("failed to build BC tables: {e}"))?;
    let num_faces = mesh.num_faces();
    let mut bc_kind = vec![0u32; num_faces * s];
    let mut bc_value = vec![0.0f32; num_faces * s];
    for f in 0..num_faces {
        if mesh.face_neighbor[f].is_some() {
            continue; // interior
        }
        let boundary_idx = match mesh.face_boundary.get(f).copied().flatten() {
            None => 0usize,
            Some(bt) => bt.bc_table_index(),
        };
        if boundary_idx == 0 {
            continue;
        }
        let src = boundary_idx * s;
        let dst = f * s;
        bc_kind[dst..dst + s].copy_from_slice(&kind_by_type[src..src + s]);
        bc_value[dst..dst + s].copy_from_slice(&value_by_type[src..src + s]);
    }
    Ok((bc_kind, bc_value))
}

/// Group boundary faces by `BoundaryType::bc_table_index()` (== `GpuBoundaryType
/// as u32`); index 0 is "None"/interior.
fn group_boundary_faces(mesh: &Mesh) -> Vec<Vec<u32>> {
    let mut groups: Vec<Vec<u32>> = vec![Vec::new(); 8];
    for f in 0..mesh.num_faces() {
        if let Some(bt) = mesh.face_boundary[f] {
            groups[bt.bc_table_index()].push(f as u32);
        }
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType};
    use crate::solver::model::{
        scalar_transport_model, ADVECTING_VELOCITY_FIELD, SCALAR_TRANSPORT_FIELD,
        SCALAR_TRANSPORT_KAPPA, SCALAR_TRANSPORT_MMS_SOURCE_FIELD,
    };
    use std::f64::consts::PI;

    fn unit_square(n: usize) -> Mesh {
        generate_structured_rect_mesh(
            n,
            n,
            1.0,
            1.0,
            BoundarySides {
                left: BoundaryType::Inlet,
                right: BoundaryType::Outlet,
                bottom: BoundaryType::Wall,
                top: BoundaryType::Wall,
            },
        )
    }

    // Manufactured solution: T* = sin(pi x) cos(pi y), U = (4, 2) constant,
    // S = U.grad(T*) - kappa lap(T*).  (Mirrors mms_scalar_transport_order_test.)
    fn exact(x: f64, y: f64) -> f64 {
        (PI * x).sin() * (PI * y).cos()
    }
    fn source(x: f64, y: f64) -> f64 {
        let (ux, uy) = (4.0, 2.0);
        let k = SCALAR_TRANSPORT_KAPPA;
        let dtdx = PI * (PI * x).cos() * (PI * y).cos();
        let dtdy = -PI * (PI * x).sin() * (PI * y).sin();
        let lap = -2.0 * PI * PI * exact(x, y);
        ux * dtdx + uy * dtdy - k * lap
    }

    fn l2_error(mesh: &Mesh, t: &[f64]) -> f64 {
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..mesh.num_cells() {
            let e = t[i] - exact(mesh.cell_cx[i], mesh.cell_cy[i]);
            num += e * e * mesh.cell_vol[i];
            den += mesh.cell_vol[i];
        }
        (num / den).sqrt()
    }

    /// Least-squares slope of log(err) vs log(h).
    fn fit_order(hs: &[f64], errs: &[f64]) -> f64 {
        let n = hs.len() as f64;
        let lx: Vec<f64> = hs.iter().map(|h| h.ln()).collect();
        let ly: Vec<f64> = errs.iter().map(|e| e.ln()).collect();
        let sx: f64 = lx.iter().sum();
        let sy: f64 = ly.iter().sum();
        let sxx: f64 = lx.iter().map(|x| x * x).sum();
        let sxy: f64 = lx.iter().zip(&ly).map(|(x, y)| x * y).sum();
        (n * sxy - sx * sy) / (n * sxx - sx * sx)
    }

    fn solve_steady(n: usize, scheme: Scheme) -> (Mesh, Vec<f64>) {
        solve_steady_cfg(n, scheme, CpuBackendConfig::default())
    }

    fn solve_steady_cfg(n: usize, scheme: Scheme, config: CpuBackendConfig) -> (Mesh, Vec<f64>) {
        let mesh = unit_square(n);
        let model = scalar_transport_model().expect("model");
        let mut solver = CpuSolver::new(&mesh, model, scheme, TimeScheme::Euler, config)
            .expect("create cpu solver");
        solver.set_outer_iters(2);
        solver.set_dt(0.2);

        let face_value = |face: u32| exact(mesh.face_cx[face as usize], mesh.face_cy[face as usize]) as f32;
        for b in [
            GpuBoundaryType::Inlet,
            GpuBoundaryType::Outlet,
            GpuBoundaryType::Wall,
        ] {
            solver
                .set_boundary_values_per_face(b, SCALAR_TRANSPORT_FIELD, 0, &face_value)
                .expect("bc");
        }

        let u: Vec<(f64, f64)> = vec![(4.0, 2.0); mesh.num_cells()];
        solver.set_field_vec2(ADVECTING_VELOCITY_FIELD, &u).expect("U");
        let src: Vec<f64> = (0..mesh.num_cells())
            .map(|i| source(mesh.cell_cx[i], mesh.cell_cy[i]))
            .collect();
        solver
            .set_field_scalar(SCALAR_TRANSPORT_MMS_SOURCE_FIELD, &src)
            .expect("src");
        solver
            .set_field_scalar(SCALAR_TRANSPORT_FIELD, &vec![0.0; mesh.num_cells()])
            .expect("init");
        solver.initialize_history();

        let mut prev = solver.get_field_scalar(SCALAR_TRANSPORT_FIELD).unwrap();
        for _ in 0..400 {
            solver.step();
            let cur = solver.get_field_scalar(SCALAR_TRANSPORT_FIELD).unwrap();
            let delta = cur
                .iter()
                .zip(&prev)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            prev = cur;
            if delta < 4e-6 {
                break;
            }
        }
        (mesh, prev)
    }

    #[test]
    fn cpu_scalar_transport_upwind_converges() {
        let levels = [16usize, 32, 64];
        let mut hs = Vec::new();
        let mut errs = Vec::new();
        for &n in &levels {
            let (mesh, t) = solve_steady(n, Scheme::Upwind);
            let e = l2_error(&mesh, &t);
            println!("[cpu-mms][upwind] n={n} l2={e:.3e}");
            hs.push(1.0 / n as f64);
            errs.push(e);
        }
        // Errors must decrease monotonically and the finest must be small.
        assert!(errs[1] < errs[0] && errs[2] < errs[1], "errors not decreasing: {errs:?}");
        assert!(*errs.last().unwrap() < 1e-2, "finest error too large: {errs:?}");
        let order = fit_order(&hs, &errs);
        println!("[cpu-mms][upwind] observed order = {order:.3}");
        // Upwind advection + 2nd-order diffusion: mixed order, ~1 (matches the
        // GPU MMS test's 1.0 ± 0.2 window, with slack for the coarse 16..64 fit).
        assert!(
            (0.6..=1.6).contains(&order),
            "implausible order {order:.3} for upwind advection-diffusion"
        );
    }

    #[test]
    fn cpu_scalar_transport_sou_converges() {
        // Second-order upwind exercises the gradient path (packed_state_gradients
        // + generic_coupled_assembly_grad_state). Expect ~2nd order.
        let levels = [8usize, 16, 32, 64];
        let mut hs = Vec::new();
        let mut errs = Vec::new();
        for &n in &levels {
            let (mesh, t) = solve_steady(n, Scheme::SecondOrderUpwind);
            let e = l2_error(&mesh, &t);
            println!("[cpu-mms][sou] n={n} l2={e:.3e}");
            hs.push(1.0 / n as f64);
            errs.push(e);
        }
        let order = fit_order(&hs, &errs);
        println!("[cpu-mms][sou] observed order = {order:.3}");
        assert!(
            (1.6..=2.4).contains(&order),
            "implausible SOU order {order:.3} (expected ~2)"
        );
        assert!(*errs.last().unwrap() < 1e-3, "finest SOU error too large: {errs:?}");
    }

    #[test]
    fn cpu_multithread_matches_singlethread() {
        // Runtime-switchable multithreading must not change results: relaxed-atomic
        // buffers + disjoint per-cell writes make the parallel run deterministic and
        // identical to the serial run.
        let n = 32;
        let (_, t1) = solve_steady_cfg(n, Scheme::Upwind, CpuBackendConfig { engine: CpuEngine::Interpreter, threads: 1, simd: false });
        let (_, t4) = solve_steady_cfg(n, Scheme::Upwind, CpuBackendConfig { engine: CpuEngine::Interpreter, threads: 4, simd: false });
        let max_diff = t1
            .iter()
            .zip(&t4)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        println!("[cpu-mt] n={n} max|1thread - 4thread| = {max_diff:.3e}");
        assert!(
            max_diff == 0.0,
            "multithreaded result differs from serial: max|diff|={max_diff:.3e}"
        );
    }

    #[test]
    fn cpu_compute_options_all_agree() {
        // Validate every runtime CPU computation option against the reference
        // {1 thread, scalar}: {1,4 threads} × {scalar, SIMD}. SIMD reorders the
        // reduction summation so it matches to rounding (not bit-exact); threads
        // are bit-identical.
        let n = 32;
        let base = solve_steady_cfg(n, Scheme::Upwind, CpuBackendConfig { engine: CpuEngine::Interpreter, threads: 1, simd: false }).1;
        for (threads, simd, tol) in [
            (4usize, false, 0.0f64),
            (1, true, 1e-4),
            (4, true, 1e-4),
        ] {
            let t = solve_steady_cfg(n, Scheme::Upwind, CpuBackendConfig { engine: CpuEngine::Interpreter, threads, simd }).1;
            let max_diff = base
                .iter()
                .zip(&t)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            println!("[cpu-opts] threads={threads} simd={simd} max|diff vs ref|={max_diff:.3e}");
            assert!(
                max_diff <= tol,
                "option (threads={threads}, simd={simd}) diverges: {max_diff:.3e} > {tol:.1e}"
            );
        }
    }

    #[test]
    fn cpu_transpiled_matches_interpreter() {
        // The compiled-Rust (transpiled) engine must agree with the interpreter
        // for both schemes and across the thread/SIMD options.
        for scheme in [Scheme::Upwind, Scheme::SecondOrderUpwind] {
            let n = 32;
            let interp = solve_steady_cfg(
                n,
                scheme,
                CpuBackendConfig { engine: CpuEngine::Interpreter, threads: 1, simd: false },
            )
            .1;
            for (label, cfg) in [
                ("transpiled/1t", CpuBackendConfig { engine: CpuEngine::Transpiled, threads: 1, simd: false }),
                ("transpiled/4t/simd", CpuBackendConfig { engine: CpuEngine::Transpiled, threads: 4, simd: true }),
            ] {
                let t = solve_steady_cfg(n, scheme, cfg).1;
                let d = interp
                    .iter()
                    .zip(&t)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f64, f64::max);
                println!("[cpu-transpiled] scheme={scheme:?} {label} max|interp-transpiled|={d:.3e}");
                assert!(d < 1e-5, "transpiled {label} diverges for {scheme:?}: {d:.3e}");
            }
        }
    }
}
