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
use crate::solver::mesh::refresh::{mesh_geometry_f32, MeshRefreshReport, MeshTopology};
use crate::solver::snapshot::SolverStateSnapshot;
use crate::solver::mesh::Mesh;
use crate::solver::model::backend::SchemeRegistry;
use crate::solver::model::ModelSpec;
use crate::solver::scheme::Scheme;
use crate::solver::TimeScheme;

/// Linear-solve budget per outer iteration (scalar/BiCGSTAB path only; the
/// block/FGMRES path uses the model's `linear_solver.max_iters` — the same
/// budget the GPU enforces, which caps pathological from-rest solves instead of
/// letting them burn thousands of iterations the GPU would never run).
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

/// Kernel-id groups partitioned from the (immutable) schedule once at
/// construction. The step loop drives these in recipe order (the GPU's
/// execution order). Timing contract (load-bearing): the recurring `bc_expr`
/// refresh runs at the END of each outer iteration — after the update, NOT
/// before the assembly — so an iteration's gradients/flux/assembly see the
/// ghosts produced by the PRIOR iteration (seeded values on the first step).
/// Once-only Preparation kernels (e.g. rhie_chow dp_init) run before the loop;
/// the CPU linear solve replaces the LinearSolve phase after assembly; Apply is
/// monitor-only (skipped); Update/PrimitiveRecovery apply the solution.
struct ScheduleGroups {
    /// Once-per-step Preparation kernels (non-bc_expr).
    prep_once: Vec<String>,
    /// Recurring boundary-closure refresh (end of each outer iteration).
    bc_expr: Vec<String>,
    /// Gradients + FluxComputation + Assembly, first outer iteration.
    per_iter: Vec<String>,
    /// `per_iter` for outer iterations AFTER the first: without
    /// [`KernelId::FLUX_MODULE_GRADIENTS`] when the update group contains a
    /// Rhie-Chow grad_p refresher ([`KernelId::refreshes_grad_p`]) — that
    /// kernel already wrote the identical Green-Gauss pressure gradient
    /// (same stencil, same boundary closure) and nothing modifies `p` in
    /// between, so the recompute is redundant. `CFD2_NO_GRADP_SKIP=1` disables
    /// (then this equals `per_iter`).
    per_iter_tail: Vec<String>,
    /// Update + PrimitiveRecovery kernels.
    update: Vec<String>,
    /// `per_iter_tail` with the Assembly kernels replaced by their RHS-only
    /// variants (KernelPhaseId::AssemblyRhsOnly): re-assembles the RHS but
    /// leaves the frozen matrix untouched. Used for outer iterations where
    /// `matrix_freeze_period` (env `CFD2_MATRIX_FREEZE`, default 0 = off)
    /// skips re-linearization; empty when the model has no RHS-only kernels.
    per_iter_frozen: Vec<String>,
}

impl ScheduleGroups {
    fn from_schedule(schedule: &[ScheduledKernel]) -> Self {
        let is_bc_expr = |id: &str| id.contains("bc_expr");
        let ids_in = |pred: &dyn Fn(&ScheduledKernel) -> bool| -> Vec<String> {
            schedule.iter().filter(|s| pred(s)).map(|s| s.id.clone()).collect()
        };
        let per_iter = ids_in(&|s| {
            matches!(
                s.phase,
                KernelPhase::Gradients | KernelPhase::FluxComputation | KernelPhase::Assembly
            )
        });
        let update = ids_in(&|s| {
            matches!(s.phase, KernelPhase::Update | KernelPhase::PrimitiveRecovery)
        });
        let has_grad_p_refresh = update
            .iter()
            .any(|id| crate::solver::model::KernelId::id_refreshes_grad_p(id))
            && !std::env::var("CFD2_NO_GRADP_SKIP").is_ok_and(|v| v == "1");
        let per_iter_tail = if has_grad_p_refresh {
            per_iter
                .iter()
                .filter(|id| {
                    id.as_str() != crate::solver::model::KernelId::FLUX_MODULE_GRADIENTS.as_str()
                })
                .cloned()
                .collect()
        } else {
            per_iter.clone()
        };
        // Frozen-matrix variant: the AssemblyRhsOnly kernels ONLY. Freezing
        // the fluxes/gradients WITH the matrix keeps every implicit/explicit
        // deferred-correction pair consistent by construction (re-running the
        // Gradients/FluxComputation kernels would recompute the dc RHS from
        // fresh fluxes against a matrix built from old ones). Empty when the
        // model declares no RHS-only kernels.
        let has_rhs_only = schedule.iter().any(|s| s.phase == KernelPhase::AssemblyRhsOnly);
        let per_iter_frozen = if has_rhs_only {
            schedule
                .iter()
                .filter(|s| s.phase == KernelPhase::AssemblyRhsOnly)
                .map(|s| s.id.clone())
                .collect()
        } else {
            Vec::new()
        };
        Self {
            prep_once: ids_in(&|s| s.phase == KernelPhase::Preparation && !is_bc_expr(&s.id)),
            bc_expr: ids_in(&|s| s.phase == KernelPhase::Preparation && is_bc_expr(&s.id)),
            per_iter,
            per_iter_tail,
            update,
            per_iter_frozen,
        }
    }
}

pub struct CpuSolver {
    model_id: &'static str,
    /// Matrix-freeze eligibility: models with `linearize_pressure_flux`
    /// terms are EXCLUDED — that term's RHS piece is recomputed from live
    /// state inside the RHS-only kernel, so it would decouple from the
    /// frozen matrix's Jacobian.
    freeze_eligible: bool,
    num_cells: usize,
    num_faces: usize,
    state_stride: u32,
    unknowns_per_cell: usize,

    buffers: Buffers,
    /// Kernels keyed by id (the program store).
    kernels: HashMap<String, CpuKernel>,
    /// The scheduled execution order (recipe order), with phases.
    schedule: Vec<ScheduledKernel>,
    /// Kernel-id groups partitioned from the (immutable) schedule once at
    /// construction — see [`ScheduleGroups`].
    groups: ScheduleGroups,

    // Block-CSR topology (block-level; shared by kernels and the CPU solver).
    scalar_row_offsets: Vec<u32>,
    col_indices: Vec<u32>,
    diagonal_indices: Vec<u32>,

    // Coupled-unknown base offset per equation-target field name.
    coupled_offsets: HashMap<String, u32>,

    // Faces grouped by boundary type index (`GpuBoundaryType as u32`).
    boundary_faces: Vec<Vec<u32>>,

    /// Flux-table stride (floats per face). Kept so a topology refresh can
    /// re-size the face-indexed `fluxes` buffer (the face count changes).
    flux_stride: usize,
    /// Per-boundary-type BC tables (`ModelSpec::boundaries::to_gpu_tables`),
    /// kept so a topology refresh can re-scatter them onto the NEW face
    /// indexing without the full `ModelSpec`. Length `BOUNDARY_TYPE_COUNT * S`.
    bc_kind_by_type: Vec<u32>,
    bc_value_by_type: Vec<f32>,

    /// Host snapshot of the topology this solver was built on, kept so a
    /// `Geometry`-level `refresh_mesh` can validate the incoming mesh is
    /// topology-identical before overwriting the geometry buffers in place.
    mesh_topology: MeshTopology,

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
    /// Enables the per-outer correction-norm collection and the plateau
    /// early-exit (the CPU port of the GPU generic-coupled detector). Set by
    /// the driver from `outer_auto_converge || log_convergence`, exactly like
    /// the GPU `collect_convergence_stats`.
    collect_convergence_stats: bool,
    /// State offsets of the solved unknowns (unknown rank -> state slot), for
    /// the plateau detector's per-field state scaling.
    unknown_offsets: Vec<u32>,
    /// Outer iterations actually executed by the last step (== `outer_iters`
    /// unless the plateau detector exited early).
    outer_iterations_done: u32,
    /// The last outer iteration's per-unknown-row SCALED correction maxima
    /// (`max|x| / max|state|` per row — the same norm the GPU convergence
    /// monitor reduces), captured when `collect_convergence_stats` is on.
    /// Empty otherwise. Feeds the GUI's per-step U/p residual readout, which
    /// previously stayed at its 0.0 default on the CPU backend.
    last_outer_scaled: Vec<f32>,
    /// CPU linear-solve relative tolerance (block path; GPU inexact-Picard).
    linear_tol: f64,
    linear_restart: usize,
    /// Block-path iteration budget per solve (the model/GPU `max_iters`).
    linear_max_iters: usize,
    /// Block-system preconditioner choice (mirrors the recipe/runtime config).
    precond: PreconditionerType,
    /// Model-owned Schur preconditioner spec, when present: (velocity unknown
    /// indices, pressure unknown index, omega, sweeps_cap). Saddle-point models
    /// (incompressible/buoyant) use the CPU Schur preconditioner.
    schur: Option<(Vec<usize>, usize, f32, u32)>,
    /// AMG hierarchy for the Schur pressure block, built lazily on first use
    /// (aggregation seeded by that solve's pressure-block values; the pattern
    /// never changes).
    amg_hier: std::cell::OnceCell<crate::solver::cpu::amg::AmgHierarchy>,
    /// Adaptive inner-solve mode for the Schur pressure block: starts cheap
    /// (Jacobi-BiCGSTAB) and flips ONE-WAY to AMG once inner solves fail to
    /// converge (mesh too fine for the Jacobi inner solve). Overridable via
    /// `CFD2_CPU_SCHUR_AMG=0|1`.
    schur_amg_active: std::cell::Cell<bool>,
    /// True while `linear_solve` serves the FIRST outer iteration of a
    /// multi-outer step (set by the step loop; `Cell` because the loop holds
    /// `&self` borrows) — enables the loosened Eisenstat-Walker first solve.
    ew_first_outer: std::cell::Cell<bool>,
    /// Full-EW forcing tolerance for THIS outer (0.0 = model tolerance): set
    /// per outer from the previous outer's worst scaled correction — see the
    /// GPU `outer_forcing_tolerance` mirror in the step loop.
    ew_outer_tol: std::cell::Cell<f64>,
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
        // ALE buffers (mirroring the GPU `MeshResources`): always allocated,
        // bound only by *_ale model kernels. `mesh_fluxes` zero-filled — a
        // static mesh has zero swept rate, so an ALE model that never uploads
        // reproduces static physics bitwise. Volume history seeded equal to
        // the current volumes (re-seeded by `initialize_history`).
        buffers.insert_f32("mesh_fluxes", vec![0.0; num_faces]);
        let vols = buffers.f32_vec("cell_vols");
        buffers.insert_f32("cell_vols_old", vols.clone());
        buffers.insert_f32("cell_vols_old_old", vols);
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

        // Boundary conditions: per face x coupled-unknown component. The
        // per-boundary-type tables are kept (a topology refresh
        // re-scatters them onto the new faces); the scatter yields the
        // per-face seed the kernels read.
        let (bc_kind_by_type, bc_value_by_type) = model_bc_type_tables(&model, s)?;
        let (bc_kind, bc_value) = scatter_bc_tables(mesh, &bc_kind_by_type, &bc_value_by_type, s);
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
            crate::solver::model::ModelPreconditionerSpec::Schur { omega, sweeps_cap, layout } => {
                Some((layout, omega, sweeps_cap))
            }
            _ => None,
        }) {
            Some((layout, omega, sweeps_cap)) => Some((
                layout.u_indices().iter().map(|&u| u as usize).collect::<Vec<usize>>(),
                layout.p as usize,
                omega,
                sweeps_cap,
            )),
            None => None,
        };

        // Solved-unknown state offsets for the plateau detector's state scale.
        let unknown_offsets =
            crate::solver::model::kernel::model_unknown_state_offsets(&model)?;
        let groups = ScheduleGroups::from_schedule(&schedule);

        Ok(Self {
            model_id: model.id,
            freeze_eligible: !model.system.equations().iter().any(|eq| {
                eq.terms().iter().any(|t| t.linearize_pressure_flux.is_some())
            }),
            num_cells,
            num_faces,
            state_stride,
            unknowns_per_cell,
            buffers,
            kernels,
            schedule,
            groups,
            scalar_row_offsets,
            col_indices,
            diagonal_indices,
            coupled_offsets,
            boundary_faces,
            flux_stride,
            bc_kind_by_type,
            bc_value_by_type,
            mesh_topology: MeshTopology::from_mesh(mesh),
            constants,
            low_mach: GpuLowMachParams::default(),
            state_layout,
            stepping,
            outer_iters,
            outer_tol: 0.0,
            collect_convergence_stats: false,
            unknown_offsets,
            outer_iterations_done: 0,
            last_outer_scaled: Vec::new(),
            linear_tol: recipe.linear_solver.tolerance as f64,
            linear_restart: match recipe.linear_solver.solver_type {
                crate::solver::gpu::recipe::LinearSolverType::Fgmres { max_restart } => max_restart,
                crate::solver::gpu::recipe::LinearSolverType::Cg => 60,
            },
            linear_max_iters: (recipe.linear_solver.max_iters as usize).max(1),
            precond: recipe.linear_solver.preconditioner,
            schur,
            amg_hier: std::cell::OnceCell::new(),
            schur_amg_active: std::cell::Cell::new(false),
            ew_first_outer: std::cell::Cell::new(false),
            ew_outer_tol: std::cell::Cell::new(0.0),
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

    // ── mesh refresh ─────────────────────────────────────────────────────

    /// Geometry-only mesh refresh: overwrite the six geometry entries of
    /// `Buffers` (`face_areas`/`face_normals`/`face_centers`/`face_wrap_shift`/
    /// `cell_centers`/`cell_vols`) in place with the shared f64→f32 cast —
    /// exactly the entries `upload_mesh` populates, from the same
    /// [`mesh_geometry_f32`] helper, so init and refresh (and the GPU backend)
    /// consume bit-identical geometry. Topology must be identical to the
    /// build-time mesh (validated against the stored snapshot; full array
    /// compare — see [`MeshTopology::validate_matches`] for the cost note).
    ///
    /// The `AtomicU32` backing stores are updated in place (`copy_into_f32`),
    /// so the transpiled engine's per-run buffer handles stay valid.
    pub fn refresh_mesh_geometry(&mut self, mesh: &Mesh) -> Result<(), String> {
        self.mesh_topology.validate_matches(mesh)?;
        let geo = mesh_geometry_f32(mesh);
        self.buffers.copy_into_f32("face_areas", &geo.face_areas);
        self.buffers.copy_into_f32("face_normals", &geo.face_normals);
        self.buffers.copy_into_f32("face_centers", &geo.face_centers);
        self.buffers.copy_into_f32("face_wrap_shift", &geo.face_wrap_shift);
        self.buffers.copy_into_f32("cell_centers", &geo.cell_centers);
        self.buffers.copy_into_f32("cell_vols", &geo.cell_vols);
        Ok(())
    }

    /// ALE step entry: rotate the volume history, THEN upload the new
    /// geometry, THEN upload the (f32-closed) mesh face fluxes. Ordering
    /// contract: the rotation must capture the CURRENT `cell_vols` as `V^n`
    /// before `refresh_mesh_geometry` overwrites them with `V^{n+1}`, which is
    /// why this seam — not the `step()` prologue (it runs after the upload) —
    /// owns the rotation. Call once per step, before `step()`.
    pub fn begin_ale_step(&mut self, mesh: &Mesh, mesh_fluxes: &[f32]) -> Result<(), String> {
        if mesh_fluxes.len() != self.num_faces {
            return Err(format!(
                "begin_ale_step: mesh_fluxes has {} entries, mesh has {} faces",
                mesh_fluxes.len(),
                self.num_faces
            ));
        }
        // 1. Rotate the volume history: old_old <- old, old <- current.
        let old = self.buffers.f32_vec("cell_vols_old");
        self.buffers.copy_into_f32("cell_vols_old_old", &old);
        let cur = self.buffers.f32_vec("cell_vols");
        self.buffers.copy_into_f32("cell_vols_old", &cur);
        // 2. Upload the new geometry (validates topology-identity).
        self.refresh_mesh_geometry(mesh)?;
        // 3. Upload the closed mesh fluxes verbatim (already f32 — the SCL
        //    closure must survive byte-exactly).
        self.buffers.copy_into_f32("mesh_fluxes", mesh_fluxes);
        Ok(())
    }

    /// ALE step entry for a **topology-changing** move: rotate the volume
    /// history, rebuild every mesh-topology-derived CPU resource, then upload
    /// the f32-closed mesh fluxes — in that order.
    ///
    /// Ordering: the rotation captures the CURRENT `cell_vols` as
    /// `V^n` into `cell_vols_old` BEFORE `refresh_mesh_topology` overwrites
    /// `cell_vols` with `V^{n+1}` (via `upload_mesh`). The two history buffers
    /// are cell-indexed and never touched by the topology rebuild (cell count
    /// invariant), so the rotated `V^n`/`V^{n-1}` survive it. The rebuild
    /// reallocates `mesh_fluxes` zero-filled at the NEW face count; step 3
    /// fills it with the closed swept fluxes.
    ///
    /// Returns the topology-refresh report (`bc_overrides_reset`).
    pub fn begin_ale_step_topology(
        &mut self,
        mesh: &Mesh,
        mesh_fluxes: &[f32],
    ) -> Result<MeshRefreshReport, String> {
        if mesh_fluxes.len() != mesh.num_faces() {
            return Err(format!(
                "begin_ale_step_topology: mesh_fluxes has {} entries, mesh has {} faces",
                mesh_fluxes.len(),
                mesh.num_faces()
            ));
        }
        // 1. Rotate the volume history on the CURRENT buffers: old_old <- old,
        //    old <- current (V^n). Both are cell-indexed → survive step 2.
        let old = self.buffers.f32_vec("cell_vols_old");
        self.buffers.copy_into_f32("cell_vols_old_old", &old);
        let cur = self.buffers.f32_vec("cell_vols");
        self.buffers.copy_into_f32("cell_vols_old", &cur);
        // 2. Rebuild the topology-derived stack (uploads V^{n+1}; reallocates
        //    mesh_fluxes zero-filled at the new face count).
        let report = self.refresh_mesh_topology(mesh)?;
        // 3. Upload the closed mesh fluxes verbatim (already f32).
        self.buffers.copy_into_f32("mesh_fluxes", mesh_fluxes);
        Ok(report)
    }

    /// Topology refresh: rebuild every mesh-topology-derived CPU resource for a
    /// new mesh with the SAME cell count but a possibly changed face set /
    /// adjacency / boundary classification / nnz. Cell-indexed state (`state`
    /// ×3, `grad_state`, warm-start `x`, `cell_vols_old{,_old}`) is left
    /// UNTOUCHED — a cell keeps its identity across the refresh. The CPU solver
    /// has no baked pipelines, so this refresh is fully surgical and
    /// byte-invisible at ANY step, not just step 0.
    ///
    /// Buffer-swap safety: each rebuilt buffer is REPLACED in the `Buffers` map
    /// via `insert_*` (a fresh `AtomicU32` backing of the new length). No stale
    /// handle survives: the interpreter resolves buffers by NAME on every
    /// load/store (`Buffers::load`/`store`), and the transpiled engine calls
    /// `Buffers::atom(name)` once per DISPATCH inside `run_kernel` (never cached
    /// across steps) — so the next step sees the new backing.
    ///
    /// Runtime per-face BC overrides (keyed by the OLD face indices) are lost;
    /// the returned [`MeshRefreshReport`] flags `bc_overrides_reset` so the
    /// caller re-applies them against the new faces.
    pub fn refresh_mesh_topology(&mut self, mesh: &Mesh) -> Result<MeshRefreshReport, String> {
        // Cell count is the invariant (seed↔cell identity is what lets
        // cell-indexed state survive untouched); faces/adjacency/nnz may change.
        if mesh.num_cells() != self.num_cells {
            return Err(format!(
                "topology refresh requires an unchanged cell count ({} -> {})",
                self.num_cells,
                mesh.num_cells()
            ));
        }
        let s = self.unknowns_per_cell;
        let num_faces = mesh.num_faces();

        // 1. Rebuild the diag-first scalar CSR from the factored builder (the
        //    single source of truth the build path uses — byte-identical
        //    structure for a no-op refresh).
        let (scalar_row_offsets, col_indices, diagonal_indices, cell_face_matrix_indices) =
            build_csr_topology(mesh);
        let nnz_blocks = *scalar_row_offsets.last().unwrap() as usize;

        // 2. Update the struct fields: `num_faces` drives the face-kernel
        //    dispatch count, and the block linear solvers read the CSR
        //    duplicates directly (not via `Buffers`).
        self.num_faces = num_faces;
        self.scalar_row_offsets = scalar_row_offsets.clone();
        self.col_indices = col_indices.clone();
        self.diagonal_indices = diagonal_indices.clone();

        // 3. Re-upload the mesh geometry/topology `Buffers` entries + the CSR
        //    index buffers. `upload_mesh` replaces the Stores (new backing of
        //    the new length); the cell-indexed Stores it does NOT touch (state
        //    ×3, state_iter, grad_state, x/rhs/y, cell_vols_old{,_old}) are left
        //    in place — cell count is invariant.
        upload_mesh(&mut self.buffers, mesh);
        self.buffers.insert_u32("scalar_row_offsets", scalar_row_offsets.clone());
        self.buffers.insert_u32("row_offsets", scalar_row_offsets);
        self.buffers.insert_u32("col_indices", col_indices);
        self.buffers.insert_u32("diagonal_indices", diagonal_indices);
        self.buffers.insert_u32("cell_face_matrix_indices", cell_face_matrix_indices);

        // 4. Resize the face-indexed + nnz-sized data buffers (zero-filled:
        //    `fluxes` and the assembled `matrix_values` are recomputed every
        //    outer iteration; a static mesh's zero `mesh_fluxes` reproduce
        //    static physics bitwise, exactly as the build path seeds them).
        self.buffers.insert_f32("fluxes", vec![0.0; num_faces * self.flux_stride]);
        self.buffers.insert_f32("mesh_fluxes", vec![0.0; num_faces]);
        self.buffers.insert_f32("matrix_values", vec![0.0; nnz_blocks * s * s]);

        // 5. Re-scatter the bc tables from the stored per-type tables onto the
        //    new faces + rebuild the boundary-face groups. Runtime per-face BC
        //    overrides are lost (see the method doc / `bc_overrides_reset`).
        let (bc_kind, bc_value) =
            scatter_bc_tables(mesh, &self.bc_kind_by_type, &self.bc_value_by_type, s);
        self.buffers.insert_u32("bc_kind", bc_kind);
        self.buffers.insert_f32("bc_value", bc_value);
        self.boundary_faces = group_boundary_faces(mesh);

        // 6. Refresh the topology snapshot (so a later Geometry refresh
        //    validates against the CURRENT topology) and clear the AMG caches:
        //    the lazily-built Schur pressure-block hierarchy was aggregated on
        //    the OLD CSR pattern, and the adaptive Jacobi→AMG flip re-evaluates
        //    from cheap on the new sparsity.
        self.mesh_topology = MeshTopology::from_mesh(mesh);
        self.amg_hier = std::cell::OnceCell::new();
        self.schur_amg_active.set(false);

        Ok(MeshRefreshReport {
            bc_overrides_reset: true,
        })
    }

    // ── snapshot / restore ───────────────────────────────────────────────

    /// Capture the full stepping state (see [`SolverStateSnapshot`] for the
    /// inventory + exclusions). Every field is filled (`has_history = true`), so
    /// a fresh solver built on the same mesh + restored reproduces the next step
    /// byte-identically.
    pub fn snapshot(&self) -> SolverStateSnapshot {
        SolverStateSnapshot {
            num_cells: self.num_cells,
            num_faces: self.num_faces,
            state_stride: self.state_stride,
            unknowns_per_cell: self.unknowns_per_cell,
            state: self.buffers.f32_vec("state"),
            state_old: self.buffers.f32_vec("state_old"),
            state_old_old: self.buffers.f32_vec("state_old_old"),
            x: self.buffers.f32_vec("x"),
            cell_vols: self.buffers.f32_vec("cell_vols"),
            cell_vols_old: self.buffers.f32_vec("cell_vols_old"),
            cell_vols_old_old: self.buffers.f32_vec("cell_vols_old_old"),
            mesh_fluxes: self.buffers.f32_vec("mesh_fluxes"),
            time: self.time,
            dt: self.dt,
            dt_old: self.dt_old,
            dtau: self.dtau,
            step_count: self.step_count,
            last_rel_delta: self.last_rel_delta,
            schur_amg_active: self.schur_amg_active.get(),
            has_history: true,
        }
    }

    /// Restore a snapshot into this solver. The cell layout must match
    /// (`num_cells`, `state_stride`); the face-indexed `mesh_fluxes` is only
    /// restored when the face count also matches (a remesh recomputes it).
    pub fn restore(&mut self, snap: &SolverStateSnapshot) -> Result<(), String> {
        snap.check_compatible(self.num_cells, self.state_stride)?;
        self.buffers.copy_into_f32("state", &snap.state);
        if snap.has_history {
            self.buffers.copy_into_f32("state_old", &snap.state_old);
            self.buffers.copy_into_f32("state_old_old", &snap.state_old_old);
            self.buffers.copy_into_f32("x", &snap.x);
            if !snap.cell_vols.is_empty() {
                self.buffers.copy_into_f32("cell_vols", &snap.cell_vols);
            }
            self.buffers.copy_into_f32("cell_vols_old", &snap.cell_vols_old);
            self.buffers.copy_into_f32("cell_vols_old_old", &snap.cell_vols_old_old);
            if snap.num_faces == self.num_faces {
                self.buffers.copy_into_f32("mesh_fluxes", &snap.mesh_fluxes);
            }
        } else {
            // Current-state-only snapshot (GPU capture): re-seed the history
            // from the restored state (IC semantics) so a subsequent step has a
            // consistent `state_old`/`x`. Exact for single-step schemes.
            self.buffers.copy_into_f32("state_old", &snap.state);
            self.buffers.copy_into_f32("state_old_old", &snap.state);
            self.sync_x_from_state();
        }
        self.time = snap.time;
        self.dt = snap.dt;
        self.dt_old = snap.dt_old;
        self.dtau = snap.dtau;
        self.step_count = snap.step_count;
        self.last_rel_delta = snap.last_rel_delta;
        self.schur_amg_active.set(snap.schur_amg_active);
        Ok(())
    }

    // ── configuration ────────────────────────────────────────────────────

    pub fn set_outer_iters(&mut self, n: usize) {
        self.outer_iters = n.max(1);
    }
    /// Enable per-outer correction-norm collection + the plateau early-exit
    /// (mirrors `GpuProgramPlan::collect_convergence_stats`; the driver sets it
    /// from `outer_auto_converge || log_convergence` on both backends).
    pub fn set_collect_convergence_stats(&mut self, enable: bool) {
        self.collect_convergence_stats = enable;
    }
    /// Outer iterations executed by the last step (fewer than `outer_iters`
    /// when the plateau detector exited early).
    pub fn outer_iterations_done(&self) -> u32 {
        self.outer_iterations_done
    }

    /// The last outer iteration's per-unknown-row scaled correction maxima
    /// (empty unless `collect_convergence_stats` is on) plus the state
    /// offset of each row — the caller maps rows onto model fields.
    pub fn outer_scaled_corrections(&self) -> (&[f32], &[u32]) {
        (&self.last_outer_scaled, &self.unknown_offsets)
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

    /// Vector2 twin of [`Self::set_field_scalar_current`]: on the CPU the
    /// plain setter already writes only the current `state` buffer (history
    /// lives in `state_old`/`state_old_old`), so this is the same operation
    /// under the history-preserving name the `UnifiedSolver` API routes to.
    pub fn set_field_vec2_current(&mut self, field: &str, values: &[(f64, f64)]) -> Result<(), String> {
        self.set_field_vec2(field, values)
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

    /// Any non-finite value in the packed state? Clone-free host-side scan
    /// over the atomic backing store — the CPU backend's per-step divergence
    /// probe. The engine's per-solve linear stats are not threaded out of the
    /// solve loop yet (see `solve_block`), so without this a blown CPU solve
    /// keeps stepping on inf/NaN silently: the driver's `LinearSolver`
    /// detection sees an empty stats vec and its `NonFinite` readback scan
    /// only runs at readback cadence.
    pub fn state_has_nonfinite(&self) -> bool {
        use std::sync::atomic::Ordering;
        self.buffers
            .atom("state")
            .iter()
            .any(|a| !f32::from_bits(a.load(Ordering::Relaxed)).is_finite())
    }

    /// Permute every CELL-indexed store by the gather map `perm`
    /// (`new[i] = old[perm[i]]`): all state time levels, the cell volumes +
    /// ALE volume history, and the warm-start x rows. Face-indexed stores are
    /// NOT touched — the caller must rebuild them (the moving driver's next
    /// topology seam does) before any solve. The mesh-reordering seam: long
    /// FlowCoupled runs with recycling degrade the initial Morton locality
    /// toward random (measured on this codebase: +30% CPU / +55% GPU step
    /// cost), and a periodic relabel restores it.
    pub fn permute_cells(&self, perm: &[usize]) -> Result<(), String> {
        if perm.len() != self.num_cells {
            return Err(format!(
                "permute_cells: perm length {} != num_cells {}",
                perm.len(),
                self.num_cells
            ));
        }
        let mut seen = vec![false; self.num_cells];
        for &src in perm {
            if src >= self.num_cells || seen[src] {
                return Err("permute_cells: perm is not a bijection".into());
            }
            seen[src] = true;
        }
        let permute = |name: &str, width: usize| {
            let old = self.buffers.f32_vec(name);
            let mut new = vec![0.0f32; old.len()];
            for (i, &src) in perm.iter().enumerate() {
                new[i * width..(i + 1) * width]
                    .copy_from_slice(&old[src * width..(src + 1) * width]);
            }
            self.buffers.copy_into_f32(name, &new);
        };
        let stride = self.state_stride as usize;
        for b in ["state", "state_old", "state_old_old", "state_iter"] {
            permute(b, stride);
        }
        for b in ["cell_vols", "cell_vols_old", "cell_vols_old_old"] {
            permute(b, 1);
        }
        permute("x", self.unknowns_per_cell.max(1));
        Ok(())
    }

    /// Re-initialize a SUBSET of cells as FRESH fluid parcels (seed-recycling
    /// seam): overwrite the packed state row in EVERY time level
    /// (`state`/`state_old`/`state_old_old`/`state_iter` — the cell's ddt sees
    /// a zero rate), zero the ALE volume-history rate
    /// (`cell_vols_old = cell_vols_old_old = new_vol`; the CURRENT `cell_vols`
    /// was already refreshed by the ALE seam), and re-pack the cell's
    /// warm-start `x` rows from the new state (same unknown packing as
    /// `sync_x_from_state`). Per-cell writes on purpose: `write_state_f32`
    /// has initial-condition semantics and would reset the time history of
    /// EVERY cell. `rows` is `cells.len() × state_stride`; `new_vols` one
    /// volume per cell.
    pub fn reinit_cells(
        &self,
        cells: &[usize],
        rows: &[f32],
        new_vols: &[f64],
    ) -> Result<(), String> {
        let stride = self.state_stride as usize;
        if rows.len() != cells.len() * stride {
            return Err(format!(
                "reinit_cells: rows length {} != cells*stride {}",
                rows.len(),
                cells.len() * stride
            ));
        }
        if new_vols.len() != cells.len() {
            return Err(format!(
                "reinit_cells: new_vols length {} != cells {}",
                new_vols.len(),
                cells.len()
            ));
        }
        // Unknown-order offsets (equation-target order), as in
        // `sync_x_from_state`.
        let s = self.unknowns_per_cell;
        let mut offs: Vec<(u32, String)> = self
            .coupled_offsets
            .iter()
            .map(|(k, &v)| (v, k.clone()))
            .collect();
        offs.sort();
        for (k, &cell) in cells.iter().enumerate() {
            if cell >= self.num_cells {
                return Err(format!(
                    "reinit_cells: cell {cell} out of range ({} cells)",
                    self.num_cells
                ));
            }
            let row = &rows[k * stride..(k + 1) * stride];
            for (c, &v) in row.iter().enumerate() {
                for buf in ["state", "state_old", "state_old_old", "state_iter"] {
                    self.buffers.set_f32(buf, cell * stride + c, v);
                }
            }
            let v = new_vols[k] as f32;
            self.buffers.set_f32("cell_vols_old", cell, v);
            self.buffers.set_f32("cell_vols_old_old", cell, v);
            if s > 1 {
                for (i, (xbase, field)) in offs.iter().enumerate() {
                    let xbase = *xbase as usize;
                    let next = offs.get(i + 1).map(|(o, _)| *o as usize).unwrap_or(s);
                    let width = next - xbase;
                    let Some(soff) = self.state_layout.offset_for(field) else {
                        continue;
                    };
                    for c in 0..width {
                        self.buffers
                            .set_f32("x", cell * s + xbase + c, row[soff as usize + c]);
                    }
                }
            }
        }
        Ok(())
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

    /// Test/diagnostic accessor: the current scalar-CSR topology as the four
    /// arrays `(row_offsets, col_indices, diagonal_indices,
    /// cell_face_matrix_indices)`. `row_offsets`/`col_indices`/`diagonal_indices`
    /// come from the struct fields; `cell_face_matrix_indices` from the
    /// `Buffers` map (the assembly kernels consume it there).
    #[doc(hidden)]
    pub fn debug_scalar_csr(&self) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
        (
            self.scalar_row_offsets.clone(),
            self.col_indices.clone(),
            self.diagonal_indices.clone(),
            self.buffers.u32_vec("cell_face_matrix_indices"),
        )
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
        // ALE volume history: `cell_vols_old == cell_vols_old_old ==
        // cell_vols` at t=0 (also seeded at build; re-copied here in case a
        // geometry refresh rewrote `cell_vols` before initialization). A
        // numeric no-op for static models — only *_ale kernels bind these.
        let vols = self.buffers.f32_vec("cell_vols");
        self.buffers.copy_into_f32("cell_vols_old", &vols);
        self.buffers.copy_into_f32("cell_vols_old_old", &vols);
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

        // Drive in RECIPE (schedule) ORDER, which is the order the GPU executes
        // (see ScheduleGroups for the partition and its timing contract).
        let prep_once = &self.groups.prep_once;
        let bc_expr_ids = &self.groups.bc_expr;
        let per_iter = &self.groups.per_iter;
        let per_iter_tail = &self.groups.per_iter_tail;
        let update_group = &self.groups.update;
        // Matrix freezing (default off): re-linearize on outer 0 and every
        // `matrix_freeze_period`-th outer; frozen outers re-run ONLY the
        // RHS-only assembly against the frozen matrix AND the frozen
        // fluxes/gradients, so every deferred-correction implicit/explicit
        // pair stays consistent. Outer 1 always re-linearizes: step 0's first
        // matrix has a zero pressure diagonal (d_p is seeded by the first
        // Update), and freezing it poisoned the AMG hierarchy. Ineligible for
        // models with `linearize_pressure_flux` (see the field doc).
        // `CFD2_MATRIX_FREEZE=k` enables.
        let freeze_period: usize = std::env::var("CFD2_MATRIX_FREEZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|_| !self.groups.per_iter_frozen.is_empty() && self.freeze_eligible)
            .unwrap_or(0);

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
        let profile_env = std::env::var("CFD2_CPU_PROFILE").ok();
        let profile = profile_env.is_some();
        // CFD2_CPU_PROFILE=2 additionally reports per-kernel wall time.
        let profile_kernels = profile_env.as_deref() == Some("2");
        crate::solver::cpu::linalg::prof::ENABLED
            .store(profile, std::sync::atomic::Ordering::Relaxed);
        let mut kernel_times: Vec<(String, std::time::Duration)> = Vec::new();
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

        macro_rules! run_t {
            ($id:expr) => {{
                if profile_kernels {
                    let _t0 = std::time::Instant::now();
                    run($id);
                    kernel_times.push(($id.clone(), _t0.elapsed()));
                } else {
                    run($id);
                }
            }};
        }

        // Prepare once per step (non-bc_expr Preparation kernels).
        timed!(t_prep, {
            for id in prep_once {
                run_t!(id);
            }
        });

        // Adaptive outer-loop plateau detector — the CPU port of the GPU
        // generic-coupled detector (`outer_plateau_active` /
        // `outer_corrections_plateaued`): per-field maxima of the linear-solve
        // solution `x` scaled by per-field state maxima (scale computed once
        // per step, floored at 1.0); exit once every solved field is either
        // under tolerance or has STALLED (adjacent-ratio inside
        // [0.98, 1.01]) after at least 5 sweeps. Same exclusions as the GPU:
        // density-based `compressible` (conserved-target convergence, single
        // outer), `_mms` order-verification variants (fixed iterations),
        // pseudo-transient (dtau > 0), and single-outer configs.
        // `CFD2_CPU_OUTER_BREAK=0` pins the loop to the fixed count.
        const OUTER_PLATEAU_MIN_ITERS: usize = 5;
        const OUTER_TOL_EXIT_MIN_ITERS: usize = 2;
        const OUTER_PLATEAU_FACTOR: f32 = 0.98;
        const OUTER_PLATEAU_CEILING: f32 = 1.01;
        // Residual COLLECTION is broader than the plateau BREAK: the scaled
        // correction norms feed the GUI readout for every collecting config
        // (incl. pseudo-transient / compressible / MMS runs, where the break
        // stays disabled and `prev_scaled` — the EW forcing input — must
        // remain untouched to keep those paths byte-identical).
        let stats_active =
            self.collect_convergence_stats && !self.unknown_offsets.is_empty();
        let plateau_active = stats_active
            && self.outer_iters > 1
            && self.dtau <= 0.0
            && self.model_id != "compressible"
            && !self.model_id.ends_with("_mms")
            && std::env::var("CFD2_CPU_OUTER_BREAK").map_or(true, |v| v != "0");
        let s_unk = self.unknowns_per_cell;
        let stride = self.state_stride as usize;
        // Per-field max |state| (the correction scale), computed once per step.
        let mut plateau_scale: Option<Vec<f32>> = None;
        let mut prev_scaled: Vec<f32> = Vec::new();
        let mut outer_iters_done = 0u32;
        self.last_outer_scaled.clear();

        for outer_idx in 0..self.outer_iters {
            // Snapshot current iterate (dual-time reference + outer-break delta).
            let snap = self.buffers.f32_vec_threaded("state", self.config.threads);
            self.buffers
                .copy_into_f32_threaded("state_iter", &snap, self.config.threads);

            // Per-iteration kernels in schedule order (gradients/flux/assembly),
            // then the CPU linear solve (replacing LinearSolve), then the update
            // group, then the recurring boundary-closure refresh (bc_expr) which
            // prepares the ghosts for the next iteration/step.
            timed!(t_asm, {
                let group = if outer_idx == 0 {
                    per_iter
                } else if freeze_period > 0 && outer_idx > 1 && outer_idx % freeze_period != 0 {
                    &self.groups.per_iter_frozen
                } else {
                    per_iter_tail
                };
                for id in group {
                    run_t!(id);
                }
            });
            self.ew_first_outer.set(outer_idx == 0 && self.outer_iters > 1);
            // Full-EW forcing for outers 2..N (GPU `outer_forcing_tolerance`
            // mirror): the linear tolerance tracks the previous outer's worst
            // scaled correction — eta = clamp(0.1 * prev_err, tol, 1e-2) —
            // so middle solves stop two decades short of nothing while late
            // outers still get the full model tolerance.
            // `CFD2_NO_EW_FULL=1` restores first-outer-only.
            self.ew_outer_tol.set(0.0);
            if outer_idx > 0 && self.outer_iters > 1 {
                let prev_err = prev_scaled
                    .iter()
                    .map(|&v| v as f64)
                    .fold(f64::NAN, f64::max);
                if prev_err.is_finite()
                    && !std::env::var("CFD2_NO_EW_FULL").is_ok_and(|v| v == "1")
                {
                    self.ew_outer_tol.set(0.1 * prev_err);
                }
            }
            timed!(t_lin, self.linear_solve());
            timed!(t_upd, {
                for id in update_group {
                    run_t!(id);
                }
            });
            timed!(t_bc, {
                for id in bc_expr_ids {
                    run_t!(id);
                }
            });

            outer_iters_done = outer_idx as u32 + 1;

            // Correction-norm collection + plateau detector (see the block
            // comment above the loop).
            if stats_active {
                // Per-field max |x| — the outer correction norm the GPU
                // monitor reduces (`delta_maxima` over the solve solution).
                let x = self.buffers.f32_vec_threaded("x", self.config.threads);
                let mut delta = vec![0.0f32; s_unk];
                if x.len() == self.num_cells * s_unk {
                    for cell in 0..self.num_cells {
                        for (r, d) in delta.iter_mut().enumerate() {
                            *d = d.max(x[cell * s_unk + r].abs());
                        }
                    }
                }
                let scale = plateau_scale.get_or_insert_with(|| {
                    let state = self.buffers.f32_vec_threaded("state", self.config.threads);
                    let mut sc = vec![0.0f32; s_unk];
                    for cell in 0..self.num_cells {
                        for (r, &off) in self.unknown_offsets.iter().enumerate() {
                            sc[r] = sc[r].max(state[cell * stride + off as usize].abs());
                        }
                    }
                    sc
                });
                let scaled: Vec<f32> = delta
                    .iter()
                    .zip(scale.iter())
                    .map(|(&d, &s)| d / s.max(1.0))
                    .collect();
                self.last_outer_scaled.clear();
                self.last_outer_scaled.extend_from_slice(&scaled);
                if plateau_active {
                    let tol_rel = self.outer_tol.max(0.0) as f32;
                    // Tolerance exit below the stall floor (mirrors the GPU
                    // detector): every field's scaled correction under
                    // tolerance is a genuine convergence criterion, valid
                    // from the second sweep; the 5-sweep floor guards only
                    // the STALL exit.
                    let under_tol = outer_idx + 1 >= OUTER_TOL_EXIT_MIN_ITERS
                        && scaled.iter().all(|&cur| cur <= tol_rel);
                    let plateaued = under_tol
                        || (outer_idx + 1 >= OUTER_PLATEAU_MIN_ITERS
                            && !prev_scaled.is_empty()
                            && scaled.iter().zip(prev_scaled.iter()).all(|(&cur, &prev)| {
                                if cur <= tol_rel {
                                    return true;
                                }
                                let ratio = cur / prev.max(1e-30);
                                (OUTER_PLATEAU_FACTOR..=OUTER_PLATEAU_CEILING)
                                    .contains(&ratio)
                            }));
                    prev_scaled = scaled;
                    if plateaued {
                        break;
                    }
                }
            }

            // Adaptive outer break (off when outer_tol == 0).
            if self.outer_tol > 0.0 {
                let cur = self.buffers.f32_vec_threaded("state", self.config.threads);
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
        self.outer_iterations_done = outer_iters_done;

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
            eprintln!(
                "[cpu-profile]   linsolve: {}",
                crate::solver::cpu::linalg::prof::report_and_reset()
            );
            if profile_kernels {
                // Aggregate per-kernel wall time across the step, sorted desc.
                let mut agg: Vec<(String, std::time::Duration, u32)> = Vec::new();
                for (id, d) in kernel_times.drain(..) {
                    match agg.iter_mut().find(|(a, _, _)| *a == id) {
                        Some((_, total, count)) => {
                            *total += d;
                            *count += 1;
                        }
                        None => agg.push((id, d, 1)),
                    }
                }
                agg.sort_by(|a, b| b.1.cmp(&a.1));
                for (id, total, count) in agg.iter().take(12) {
                    eprintln!(
                        "[cpu-profile]   kernel {:>7.1}ms x{count} {id}",
                        total.as_secs_f64() * 1e3,
                    );
                }
            }
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

    /// Debug: read the per-face `bc_kind` buffer (length `num_faces * S`,
    /// `GpuBcKind` raw values: 0 = ZeroGradient, 1 = Dirichlet, 2 = Neumann).
    pub fn debug_bc_kind(&self) -> Vec<u32> {
        self.buffers.u32_vec("bc_kind")
    }

    /// The coupled-system rank (block-row index) of a field's FIRST component
    /// — e.g. `"U" -> 0`, `"p" -> 2` for the incompressible family. Ranks are
    /// equation-declaration order, NOT state-layout offsets.
    pub fn coupled_rank_of(&self, field: &str) -> Option<u32> {
        self.coupled_offsets.get(field).copied()
    }

    /// Per coupled-unknown rank, the state-layout offset it reads/updates
    /// (the rank -> state-slot map the update kernel uses).
    pub fn unknown_state_offsets(&self) -> &[u32] {
        &self.unknown_offsets
    }

    /// The uniform density constant the kernels see (`constants.density`) —
    /// the mass-flux `rho_f` for models without a `rho` state field.
    pub fn density_constant(&self) -> f32 {
        self.constants.density
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
        use crate::solver::cpu::linalg::prof;
        let s = self.unknowns_per_cell;
        let n = self.num_cells * s;
        // The assembled matrix is the largest buffer (nnz_blocks * S*S entries);
        // marshal it out of the atomic store in parallel. `rhs`/`x` are O(n), small.
        let matrix = prof::time(&prof::MARSHAL, || {
            self.buffers.f32_vec_threaded("matrix_values", self.config.threads)
        });
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
            // Scalar path honours the precision option too.
            match self.config.precision {
                crate::solver::cpu::CpuPrecision::F64 => {
                    bicgstab::<f64>(&a, &rhs, &mut x, LINEAR_MAX_ITERS, LINEAR_TOL, self.config.simd)
                }
                crate::solver::cpu::CpuPrecision::F32 => {
                    bicgstab::<f32>(&a, &rhs, &mut x, LINEAR_MAX_ITERS, LINEAR_TOL, self.config.simd)
                }
            };
        } else {
            // Coupled block system: solve in the configured PRECISION (the
            // f64 instantiation is the bit-identical reference; f32 mirrors
            // the GPU arithmetic). The scalar (s == 1) path above keeps the
            // validated f64 BiCGSTAB.
            match self.config.precision {
                crate::solver::cpu::CpuPrecision::F64 => {
                    self.solve_block_system::<f64>(&matrix, &rhs, &mut x)
                }
                crate::solver::cpu::CpuPrecision::F32 => {
                    self.solve_block_system::<f32>(&matrix, &rhs, &mut x)
                }
            }
        }

        prof::time(&prof::MARSHAL, || self.buffers.copy_into_f32("x", &x));
    }

    /// The coupled (S > 1) linear solve at precision `T` — see `linear_solve`.
    fn solve_block_system<T: crate::solver::cpu::linalg::Real>(
        &self,
        matrix: &[f32],
        rhs: &[f32],
        x: &mut [f32],
    ) {
        use crate::solver::cpu::linalg::prof;
        let s = self.unknowns_per_cell;
        let n = self.num_cells * s;
        let threads = self.config.threads;
        {
            let a = BlockCsr {
                s,
                scalar_row_offsets: &self.scalar_row_offsets,
                col_indices: &self.col_indices,
                diagonal_indices: &self.diagonal_indices,
                values: matrix,
                threads,
                simd: self.config.simd,
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
            let _ = self.precond;
            let tol = std::env::var("CFD2_CPU_LINTOL")
                .ok()
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(self.linear_tol);
            // Eisenstat-Walker-style loosened first-outer solve (GPU
            // `first_outer_tolerance` mirror): the first outer iteration of a
            // multi-outer step re-linearizes immediately afterwards, so its
            // linearization error is O(1) and solving past ~1e-2 relative is
            // over-solving; later outers keep the model tolerance, so the
            // converged step is unchanged. `CFD2_NO_EW_FIRST=1` disables.
            let tol = if self.ew_first_outer.get()
                && !std::env::var("CFD2_NO_EW_FIRST").is_ok_and(|v| v == "1")
            {
                tol.max(1e-2)
            } else if self.ew_outer_tol.get() > 0.0 {
                // Full-EW middle-outer forcing (set per outer in the step
                // loop; clamped to [model tol, 1e-2]).
                self.ew_outer_tol.get().clamp(tol, tol.max(1e-2))
            } else {
                tol
            };
            let stats = match &self.schur {
                Some((u_idx, p, omega, sweeps_cap)) => {
                    // Adaptive inner solve for the Schur pressure block: start
                    // with the cheap Jacobi-BiCGSTAB (wins when the block is
                    // easy, e.g. the nozzle); flip ONE-WAY to the AMG-
                    // preconditioned inner solve once inner solves fail to
                    // converge (fine-mesh Poisson blocks, where Jacobi inner
                    // iteration counts grow ~h^-2). `CFD2_CPU_SCHUR_AMG=0|1`
                    // forces the mode.
                    let amg_mode = std::env::var("CFD2_CPU_SCHUR_AMG").ok();
                    let use_amg = match amg_mode.as_deref() {
                        Some("0") => false,
                        Some(_) => true,
                        None => self.schur_amg_active.get(),
                    };
                    let amg_hier = if use_amg {
                        Some(self.amg_hier.get_or_init(|| {
                            crate::solver::cpu::amg::AmgHierarchy::build(
                                &self.scalar_row_offsets,
                                &self.col_indices,
                                &crate::solver::cpu::linalg::extract_p_values(&a, *p),
                            )
                        }))
                    } else {
                        None
                    };
                    let pc = prof::time(&prof::PC_BUILD, || {
                        SchurPrecond::new(
                            a,
                            u_idx,
                            *p,
                            *omega as f64,
                            *sweeps_cap,
                            self.config.simd,
                            amg_hier,
                        )
                    });
                    let stats = fgmres::<T>(
                        &a,
                        rhs,
                        x,
                        &pc,
                        self.linear_restart,
                        self.linear_max_iters,
                        tol,
                        self.config.simd,
                    );
                    if amg_mode.is_none() && !use_amg {
                        let (applies, failures) = pc.inner_outcomes();
                        if applies > 0 && failures * 2 >= applies {
                            self.schur_amg_active.set(true);
                        }
                    }
                    stats
                }
                None => {
                    let block_pc: Box<dyn Preconditioner<T>> =
                        prof::time(&prof::PC_BUILD, || {
                            if std::env::var("CFD2_CPU_POINT_JACOBI").is_ok() {
                                Box::new(PointJacobi::<T>::new(&a)) as Box<dyn Preconditioner<T>>
                            } else {
                                Box::new(BlockJacobi::<T>::new(&a))
                            }
                        });
                    fgmres::<T>(
                        &a,
                        rhs,
                        x,
                        block_pc.as_ref(),
                        self.linear_restart,
                        self.linear_max_iters,
                        tol,
                        self.config.simd,
                    )
                }
            };
            if std::env::var("CFD2_CPU_DEBUG_SOLVE").is_ok() {
                eprintln!(
                    "[cpu-solve] block S={s} n={n} iters={} rel_res={:.3e} conv={}",
                    stats.iters, stats.rel_residual, stats.converged
                );
            }
        }
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
    // this (model, kernel); otherwise fall back to the interpreter. The
    // generated entry points take an index RANGE so buffer handles resolve
    // once per chunk, not once per index (a HashMap lookup per handle).
    if engine == CpuEngine::Transpiled {
        if let Some(f) = crate::solver::cpu::generated::lookup(model_id, id) {
            crate::solver::cpu::parallel::parallel_ranges(domain, threads, |start, end| {
                f(buffers, start as u32, end as u32, constants);
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
///
/// Geometry entries come from the shared [`mesh_geometry_f32`] cast — the same
/// arrays the GPU `init_mesh` uploads and both backends' `refresh_mesh` paths
/// rewrite, so init/refresh and CPU/GPU consume bit-identical f32 geometry.
fn upload_mesh(buffers: &mut Buffers, mesh: &Mesh) {
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
    let geo = mesh_geometry_f32(mesh);
    buffers.insert_f32("face_areas", geo.face_areas);
    buffers.insert_vec2("face_normals", geo.face_normals);
    buffers.insert_vec2("cell_centers", geo.cell_centers);
    buffers.insert_vec2("face_centers", geo.face_centers);
    buffers.insert_f32("cell_vols", geo.cell_vols);
    // face_wrap_shift is empty on non-periodic meshes (all-zero in the cast).
    buffers.insert_vec2("face_wrap_shift", geo.face_wrap_shift);
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
}

/// Construct the scalar CSR topology consistent with the assembly kernel's index
/// maps: each row holds the diagonal (rank 0) followed by one entry per interior
/// face. Returns `(row_offsets, col_indices, diagonal_indices,
/// cell_face_matrix_indices)`. Delegates to the factored builder
/// (`solver::mesh::csr::build_diag_first_scalar_csr`) so init and a topology
/// refresh rebuild byte-identical structure from one source of truth.
fn build_csr_topology(mesh: &Mesh) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let csr = crate::solver::mesh::csr::build_diag_first_scalar_csr(mesh);
    (
        csr.row_offsets,
        csr.col_indices,
        csr.diagonal_indices,
        csr.cell_face_matrix_indices,
    )
}

/// The model's per-boundary-type `(bc_kind, bc_value)` tables (length
/// `BOUNDARY_TYPE_COUNT * S`), the source both the build-time scatter and a
/// topology re-scatter consume. Kept on the solver so a topology refresh
/// can re-derive the per-face tables without the full `ModelSpec`.
fn model_bc_type_tables(model: &ModelSpec, _s: usize) -> Result<(Vec<u32>, Vec<f32>), String> {
    model
        .boundaries
        .to_gpu_tables(&model.system)
        .map_err(|e| format!("failed to build BC tables: {e}"))
}

/// Scatter the model's per-boundary-type tables onto per-face `(bc_kind,
/// bc_value)` seeds (length `num_faces * S`) — exactly as the GPU generic-coupled
/// backend does (`row_base(i) = i * S`). Interior and `None`-typed faces stay
/// zero (kernels guard with `is_boundary`). The values are *seeds*;
/// `set_boundary_values_per_face` overrides them at runtime and the `bc_expr`
/// kernel refreshes expression-valued entries each iteration. Deterministic
/// (dense face sweep, no hash iteration) so a no-op refresh is byte-identical.
fn scatter_bc_tables(
    mesh: &Mesh,
    kind_by_type: &[u32],
    value_by_type: &[f32],
    s: usize,
) -> (Vec<u32>, Vec<f32>) {
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
    (bc_kind, bc_value)
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
        let (_, t1) = solve_steady_cfg(n, Scheme::Upwind, CpuBackendConfig { engine: CpuEngine::Interpreter, threads: 1, simd: false, precision: Default::default(), });
        let (_, t4) = solve_steady_cfg(n, Scheme::Upwind, CpuBackendConfig { engine: CpuEngine::Interpreter, threads: 4, simd: false, precision: Default::default(), });
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
        let base = solve_steady_cfg(n, Scheme::Upwind, CpuBackendConfig { engine: CpuEngine::Interpreter, threads: 1, simd: false, precision: Default::default(), }).1;
        for (threads, simd, tol) in [
            (4usize, false, 0.0f64),
            (1, true, 1e-4),
            (4, true, 1e-4),
        ] {
            let t = solve_steady_cfg(n, Scheme::Upwind, CpuBackendConfig { engine: CpuEngine::Interpreter, threads, simd, precision: Default::default(), }).1;
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
                CpuBackendConfig { engine: CpuEngine::Interpreter, threads: 1, simd: false, precision: Default::default(), },
            )
            .1;
            for (label, cfg) in [
                ("transpiled/1t", CpuBackendConfig { engine: CpuEngine::Transpiled, threads: 1, simd: false, precision: Default::default(), }),
                ("transpiled/4t/simd", CpuBackendConfig { engine: CpuEngine::Transpiled, threads: 4, simd: true, precision: Default::default(), }),
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
