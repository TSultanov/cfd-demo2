use crate::solver::gpu::execution_plan::{run_module_graph, GraphDetail, GraphExecMode};
use crate::solver::gpu::linear_solver::fgmres::FgmresWorkspace;
use crate::solver::gpu::modules::coupled_schur::CoupledPressureSolveKind;
use crate::solver::gpu::modules::generated_kernels::GeneratedKernelsModule;
use crate::solver::gpu::modules::generic_coupled_schur::{
    GenericCoupledSchurPreconditioner, GenericCoupledSchurPreconditionerInputs,
    GenericCoupledSchurSetupBindGroupInputs,
};
use crate::solver::gpu::modules::graph::{DispatchKind, ModuleGraph, RuntimeDims};
use crate::solver::gpu::modules::resource_registry::ResourceRegistry;
use crate::solver::gpu::program::generic_coupled_backend::scatter_bc_tables;
use crate::solver::mesh::MeshRefreshReport;
use crate::solver::gpu::modules::krylov_precond::{DispatchGrids, KrylovDispatch};
use crate::solver::gpu::modules::krylov_solve::KrylovSolveModule;
use crate::solver::gpu::modules::linear_solver::{
    solve_fgmres, submit_solve_cg_fixed_iterations_chunked,
    submit_solve_fgmres_fixed_iterations_chunked, SolveFgmresArgs,
};
use crate::solver::gpu::modules::linear_system::LinearSystemView;
use crate::solver::gpu::modules::outer_convergence::OuterConvergenceMonitor;
use crate::solver::gpu::modules::outer_gate::OuterAdaptiveGate;
use crate::solver::gpu::modules::runtime_preconditioner::{
    RuntimePreconditionerInputs, RuntimePreconditionerModule,
};
use crate::solver::gpu::modules::time_integration::TimeIntegrationModule;
use crate::solver::gpu::modules::unified_field_resources::UnifiedFieldResources;
use crate::solver::gpu::modules::unified_graph::{
    build_graph_for_phases, build_optional_graph_for_phase, build_optional_graph_for_phases,
};
use crate::solver::gpu::program::plan::{GpuProgramPlan, ProgramParamHandler};
use crate::solver::gpu::program::plan_instance::{
    OuterStepStatus, PlanFuture, PlanLinearSystemDebug, PlanParamValue,
};
use crate::solver::gpu::recipe::{KernelPhase, LinearSolverType, SolverRecipe};
use crate::solver::gpu::runtime::GpuCsrRuntime;
use crate::solver::gpu::structs::{
    GpuGenericCoupledSchurSetupParams, GpuSchurPrecondGenericParams, LinearSolverStats,
};
use crate::solver::model::backend::ast::FieldKind;
use crate::solver::model::ports::PortRegistry;
use crate::solver::model::{ModelPreconditionerSpec, ModelSpec};
use bytemuck::bytes_of;
use wgpu::util::DeviceExt;
use cfd2_codegen::solver::codegen::bc_table::{HostBcTable, BOUNDARY_TYPE_COUNT};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

const RHO_POSITIVITY_FLOOR: f32 = 1.0e-8;
const PRESSURE_POSITIVITY_FLOOR: f32 = 0.0;

#[derive(Debug, Clone, Copy, Default)]
struct StepPositivityReport {
    min_rho: f32,
    min_p: f32,
    rho_undershoot_count: u32,
    pressure_undershoot_count: u32,
}

impl StepPositivityReport {
    fn has_violation(self) -> bool {
        self.rho_undershoot_count > 0 || self.pressure_undershoot_count > 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PositivityFallbackAction {
    None,
    RetryReject,
    RollbackAccept,
}

#[derive(Debug, Clone, Copy)]
struct PositivityFieldOffsets {
    rho: usize,
    p: usize,
}

/// Pre-resolved mapping from unknown indices to state layout slots.
///
/// This is computed once at model build time and stored in the program resources
/// to avoid repeated StateLayout lookups during GPU operations.
#[derive(Debug, Clone)]
pub struct ResolvedUnknownMapping {
    /// Flat, indexed by `equation_index * max_components + component_index`.
    pub offsets: Vec<u32>,
    pub num_equations: usize,
    pub max_components: usize,
}

impl ResolvedUnknownMapping {
    const UNMAPPED_OFFSET: u32 = u32::MAX;

    pub fn get_offset(&self, equation: usize, component: usize) -> Option<u32> {
        if equation >= self.num_equations || component >= self.max_components {
            return None;
        }
        let idx = equation * self.max_components + component;
        let offset = self.offsets[idx];
        (offset != Self::UNMAPPED_OFFSET).then_some(offset)
    }
}

/// Resolve the unknown-to-state mapping from equation targets, extracting
/// offsets from the field ports registered in the `PortRegistry`.
pub fn resolve_unknown_mapping_runtime(
    model: &ModelSpec,
    port_registry: &PortRegistry,
) -> Result<ResolvedUnknownMapping, String> {
    let equations: Vec<_> = model.system.equations().iter().collect();
    let num_equations = equations.len();
    let max_components = equations
        .iter()
        .map(|eq| eq.target().kind().component_count())
        .max()
        .unwrap_or(1);

    let mut offsets = vec![ResolvedUnknownMapping::UNMAPPED_OFFSET; num_equations * max_components];

    for (eq_idx, eq) in equations.iter().enumerate() {
        let target = eq.target();
        let name = target.name();
        let kind = target.kind();
        let comps = kind.component_count();

        match kind {
            FieldKind::Scalar => {
                let port = port_registry
                    .get_field_entry_by_name(name)
                    .ok_or_else(|| format!("Field '{}' not found in port registry", name))?;
                offsets[eq_idx * max_components] = port.offset();
            }
            _ => {
                for comp in 0..comps {
                    let port = port_registry
                        .get_field_entry_by_name(name)
                        .ok_or_else(|| format!("Field '{}' not found in port registry", name))?;
                    offsets[eq_idx * max_components + comp] = port.offset() + comp as u32;
                }
            }
        }
    }

    Ok(ResolvedUnknownMapping {
        offsets,
        num_equations,
        max_components,
    })
}

/// Boundary condition data bundled to reduce constructor argument count.
pub struct BoundaryConditionData {
    pub b_bc_kind: wgpu::Buffer,
    pub b_bc_value: wgpu::Buffer,
    pub boundary_faces: Vec<Vec<u32>>,
}

pub(crate) struct GenericCoupledProgramResources {
    runtime: GpuCsrRuntime,
    fields: UnifiedFieldResources,
    time_integration: TimeIntegrationModule,
    requested_time_scheme: crate::solver::gpu::enums::TimeScheme,
    kernels: GeneratedKernelsModule,
    init_prepare_graph: ModuleGraph<GeneratedKernelsModule>,
    dp_init_enabled: bool,
    dp_init_needed: AtomicBool,
    recurring_prepare_enabled: bool,
    assembly_graph: ModuleGraph<GeneratedKernelsModule>,
    /// Assembly graph for outer iterations AFTER the first, when the model's
    /// Update phase runs `rhie_chow/grad_p_update`: that kernel already wrote
    /// the identical Green-Gauss pressure gradient the Gradients-phase
    /// `flux_module_gradients` would recompute, and nothing between them
    /// modifies `p`. `None` when the model lacks the refresher or
    /// `CFD2_NO_GRADP_SKIP=1`.
    assembly_graph_tail: Option<ModuleGraph<GeneratedKernelsModule>>,
    /// RHS-only assembly variant: re-assembles the RHS against a FROZEN matrix
    /// AND frozen fluxes/gradients (freezing them together keeps the
    /// deferred-correction implicit/explicit pairing consistent). Used on outer
    /// iterations where `matrix_freeze_period` skips re-linearization; None when
    /// the recipe has no RHS-only kernels or the model is freeze-ineligible.
    assembly_graph_frozen: Option<ModuleGraph<GeneratedKernelsModule>>,
    /// Matrix-freeze period (default 0 = off; env `CFD2_MATRIX_FREEZE=k`
    /// re-linearizes on outers 0, 1 and every k-th after). Applied on the
    /// non-batched path and the direct batched path; the adaptive-indirect
    /// batched sub-path keeps full re-assembly. `linearize_pressure_flux`
    /// models are ineligible: their RHS piece re-reads live state and would
    /// decouple from the frozen matrix's Jacobian.
    matrix_freeze_period: u32,
    apply_graph: ModuleGraph<GeneratedKernelsModule>,
    update_graph: ModuleGraph<GeneratedKernelsModule>,
    explicit_graph: ModuleGraph<GeneratedKernelsModule>,
    outer_iters: usize,
    outer_tol: f32,
    outer_tol_abs: f32,
    outer_break_enabled: bool,
    outer_batched_mode: bool,
    nonconverged_relax: f32,
    nonconverged_dt_scale: f32,
    nonconverged_dtau_scale: f32,
    nonconverged_retry_enabled: bool,
    nonconverged_retry_max_attempts: usize,
    implicit_base_alpha_u: Option<f32>,
    linear_solver: crate::solver::gpu::recipe::LinearSolverSpec,
    schur: Option<GenericCoupledSchurResources>,
    krylov: Option<GenericCoupledKrylovResources>,
    outer_convergence: Option<OuterConvergenceMonitor>,
    outer_gate: Option<OuterAdaptiveGate>,
    _b_bc_kind: wgpu::Buffer,
    _b_bc_value: wgpu::Buffer,
    boundary_faces: Vec<Vec<u32>>,
    /// Model + recipe clones kept so a topology refresh can reconstruct the
    /// mesh-topology-derived resources (bc scatter needs `model.boundaries` +
    /// `model.system`; the generated-kernel bind-group rebuild needs `model.id`
    /// + `recipe.kernels`; the Schur/krylov + convergence-monitor rebuild needs
    /// both). Cheap relative to the GPU buffers they gate.
    model: ModelSpec,
    recipe: SolverRecipe,
}

/// Controls whether coupled outer iterations use host-side adaptive break logic.
///
/// `true` keeps existing behavior (evaluate correction norms and stop early).
/// `false` forces fixed outer-iteration count and skips adaptive early break.
const DEFAULT_OUTER_BREAK_ENABLED: bool = true;
const DEFAULT_OUTER_BATCHED_MODE: bool = true;

struct GenericCoupledSchurResources {
    solver: KrylovSolveModule<GenericCoupledSchurPreconditioner>,
    dispatch: KrylovDispatch,
    _b_diag_u: wgpu::Buffer,
    _b_diag_p: wgpu::Buffer,
    _b_precond_params: wgpu::Buffer,
    _b_p_matrix_values: wgpu::Buffer,
}

struct GenericCoupledKrylovResources {
    solver: KrylovSolveModule<RuntimePreconditionerModule>,
    dispatch: KrylovDispatch,
    _b_diag_u: wgpu::Buffer,
    _b_diag_v: wgpu::Buffer,
    _b_diag_p: wgpu::Buffer,
}


impl GenericCoupledProgramResources {
    pub(crate) fn new(
        runtime: GpuCsrRuntime,
        fields: UnifiedFieldResources,
        kernels: GeneratedKernelsModule,
        model: &ModelSpec,
        recipe: &SolverRecipe,
        bc_data: BoundaryConditionData,
    ) -> Result<Self, String> {
        let BoundaryConditionData {
            b_bc_kind,
            b_bc_value,
            boundary_faces,
        } = bc_data;
        // Some models (e.g., compressible KT flux) require a gradient stage before flux.
        // Keep gradients optional so diffusion-only models don't fail graph construction.
        let init_prepare_graph = build_optional_graph_for_phase(
            recipe,
            KernelPhase::Preparation,
            &kernels,
            "generic_coupled",
        )?
        .unwrap_or_else(|| ModuleGraph::new(Vec::new()));
        let dp_init_enabled = recipe
            .kernels
            .iter()
            .any(|k| k.phase == KernelPhase::Preparation);
        // Expression-valued boundary conditions read interior state, so
        // their refresh kernel must re-run every outer iteration (not just
        // at step start).
        let recurring_prepare_enabled = recipe
            .kernels
            .iter()
            .any(|k| k.id == crate::solver::model::KernelId::BC_EXPR_UPDATE);

        let assembly_graph = build_graph_for_phases(
            recipe,
            &[
                KernelPhase::Gradients,
                KernelPhase::FluxComputation,
                KernelPhase::Assembly,
            ],
            &kernels,
            "generic_coupled",
        )?;

        // Tail-iteration assembly variant: drop `flux_module_gradients` when
        // the Update phase's `rhie_chow/grad_p_update` already refreshes the
        // same state grad_p slots each outer iteration (see the field doc).
        let has_grad_p_refresh = recipe.kernels.iter().any(|k| k.id.refreshes_grad_p());
        let grad_p_skip_disabled = std::env::var("CFD2_NO_GRADP_SKIP").is_ok_and(|v| v == "1");
        let assembly_graph_tail = (has_grad_p_refresh && !grad_p_skip_disabled).then(|| {
            // Node labels are "{prefix}:{kernel id}" (unified_graph::kernel_label).
            assembly_graph.clone_filtered(|label| {
                !label.ends_with(crate::solver::model::KernelId::FLUX_MODULE_GRADIENTS.as_str())
            })
        });

        // RHS-only assembly graph for matrix-frozen outer iterations (see the
        // field docs; kernels exist only for the generic coupled models).
        let freeze_eligible = !model.system.equations().iter().any(|eq| {
            eq.terms().iter().any(|t| t.linearize_pressure_flux.is_some())
        });
        let assembly_graph_frozen = build_optional_graph_for_phases(
            recipe,
            &[KernelPhase::AssemblyRhsOnly],
            &kernels,
            "generic_coupled",
        )?
        .filter(|_| {
            freeze_eligible
                && recipe.kernels_for_phase(KernelPhase::AssemblyRhsOnly).next().is_some()
        });
        let matrix_freeze_period: u32 = std::env::var("CFD2_MATRIX_FREEZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|_| assembly_graph_frozen.is_some())
            .unwrap_or(0);

        // Apply and update are optional depending on the stepping mode.
        // (For implicit outer-iteration recipes, update may be executed in the "apply" stage.)
        let apply_graph = build_optional_graph_for_phase(
            recipe,
            KernelPhase::Apply,
            &kernels,
            "generic_coupled",
        )?
        .unwrap_or_else(|| ModuleGraph::new(Vec::new()));

        let update_graph = build_optional_graph_for_phase(
            recipe,
            KernelPhase::Update,
            &kernels,
            "generic_coupled",
        )?
        .unwrap_or_else(|| ModuleGraph::new(Vec::new()));

        // Explicit stepping uses a single graph op; build a combined graph covering all
        // compute phases that might be present in explicit recipes.
        let explicit_graph = build_optional_graph_for_phases(
            recipe,
            &[
                KernelPhase::Gradients,
                KernelPhase::FluxComputation,
                KernelPhase::Assembly,
                KernelPhase::Apply,
                KernelPhase::Update,
            ],
            &kernels,
            "generic_coupled",
        )?
        .unwrap_or_else(|| ModuleGraph::new(Vec::new()));

        let outer_iters = match recipe.stepping {
            crate::solver::gpu::recipe::SteppingMode::Implicit { outer_iters } => outer_iters,
            _ => 1,
        };

        let linear_solver = recipe.linear_solver.clone();
        let scalar_row_offsets = &runtime.common.mesh.b_scalar_row_offsets;
        let scalar_col_indices = &runtime.common.mesh.b_scalar_col_indices;
        let schur = build_generic_schur(
            model,
            recipe,
            &runtime,
            scalar_row_offsets,
            scalar_col_indices,
        )?;
        let krylov = if schur.is_some() {
            None
        } else {
            build_generic_krylov(recipe, &runtime)?
        };

        let unknown_mapping = resolve_unknown_mapping_runtime(model, &recipe.port_registry)?;

        let outer_convergence = OuterConvergenceMonitor::new(
            &runtime.common.context.device,
            &runtime.common.context.queue,
            &runtime.common.context.pipeline_cache,
            model,
            runtime.common.num_cells,
            runtime.linear_port_space.buffer(runtime.linear_ports.x),
            &unknown_mapping,
        )?;

        let outer_gate = outer_convergence.as_ref().map(|oc| {
            OuterAdaptiveGate::new(
                &runtime.common.context.device,
                &runtime.common.context.queue,
                &runtime.common.context.pipeline_cache,
                runtime.common.num_cells,
                runtime.common.num_faces,
                runtime
                    .common
                    .context
                    .device
                    .limits()
                    .max_compute_workgroups_per_dimension,
                &oc.b_break_status,
            )
        });

        let requested_time_scheme = match recipe.initial_constants.time_scheme {
            0 => crate::solver::gpu::enums::TimeScheme::Euler,
            1 => crate::solver::gpu::enums::TimeScheme::BDF2,
            other => return Err(format!("unknown time_scheme id {other}")),
        };

        Ok(Self {
            nonconverged_relax: if model.id == "compressible" { 1.0 } else { 1.0 },
            nonconverged_dt_scale: if model.id == "compressible" { 0.5 } else { 1.0 },
            nonconverged_dtau_scale: if model.id == "compressible" { 0.5 } else { 1.0 },
            nonconverged_retry_enabled: model.id == "compressible",
            nonconverged_retry_max_attempts: 1,
            runtime,
            fields,
            time_integration: TimeIntegrationModule::new(),
            requested_time_scheme,
            kernels,
            init_prepare_graph,
            dp_init_enabled,
            dp_init_needed: AtomicBool::new(dp_init_enabled),
            recurring_prepare_enabled,
            assembly_graph,
            assembly_graph_tail,
            assembly_graph_frozen,
            matrix_freeze_period,
            apply_graph,
            update_graph,
            explicit_graph,
            outer_iters,
            outer_tol: 1e-3,
            outer_tol_abs: 1e-6,
            outer_break_enabled: DEFAULT_OUTER_BREAK_ENABLED,
            outer_batched_mode: DEFAULT_OUTER_BATCHED_MODE,
            implicit_base_alpha_u: None,
            linear_solver,
            schur,
            krylov,
            outer_convergence,
            outer_gate,
            _b_bc_kind: b_bc_kind,
            _b_bc_value: b_bc_value,
            boundary_faces,
            model: model.clone(),
            recipe: recipe.clone(),
        })
    }
}

impl GenericCoupledProgramResources {
    fn runtime_dims(&self) -> RuntimeDims {
        RuntimeDims {
            num_cells: self.runtime.common.num_cells,
            num_faces: self.runtime.common.num_faces,
        }
    }

    /// Geometry-only mesh refresh: rewrite the six geometry buffers in
    /// place (see [`MeshResources::refresh_geometry`]). Topology-identical
    /// meshes only (validated); bind groups, CSR structures, bc tables, AMG
    /// and FGMRES/Schur workspaces are untouched — they are all
    /// topology-derived, which is exactly what a `Geometry` refresh leaves
    /// unchanged.
    pub(crate) fn refresh_mesh_geometry(
        &self,
        mesh: &crate::solver::mesh::Mesh,
    ) -> Result<(), String> {
        let common = &self.runtime.common;
        common.mesh.refresh_geometry(&common.context.queue, mesh)
    }

    /// Topology refresh: rebuild every mesh-topology-derived GPU
    /// resource for a new mesh with the SAME cell count but a possibly changed
    /// face set / adjacency / boundary classification / nnz. Cell-indexed
    /// solver state (`state` ×3, gradients, iteration snapshot, warm-start is
    /// re-zeroed — see below) survives untouched; a cell keeps its identity.
    ///
    /// Sequence (each stage depends on the previous):
    /// 1. `runtime.refresh_topology` — reallocate mesh buffers + rebuild the
    ///    host/device CSR + block CSR + the scalar-CG linear system.
    /// 2. `fields.refresh_face_count` — reallocate the per-face flux buffer
    ///    (zero-filled; recomputed every outer iteration).
    /// 3. Re-scatter the bc tables from the model spec + new `face_boundary`
    ///    and rebuild `boundary_faces`. **Runtime per-face BC overrides are
    ///    lost** (they were keyed by the old face indices) — reported via
    ///    `bc_overrides_reset` so the caller re-applies them.
    /// 4. Rebuild the Schur / krylov preconditioner + FGMRES workspace and the
    ///    outer-convergence monitor/gate against the refreshed linear system.
    /// 5. Rebuild every generated-kernel bind group in place, WITHOUT
    ///    recompiling any generated pipeline (the WGSL is unchanged — only the
    ///    buffers moved).
    ///
    /// Stages 1 and 4 reconstruct the static, hand-written linear-algebra
    /// modules (scalar CG, FGMRES, Schur/krylov, AMG, monitors), recompiling
    /// their pipelines; only the generated model kernels take the in-place
    /// bind-group rebuild (stage 5). Reconstruction also makes an in-place
    /// Schur `reset()` unnecessary: the rebuilt Schur has a fresh AMG hierarchy
    /// (`prepares_seen = 0`), so coarse operators cannot go stale.
    pub(crate) fn refresh_mesh_topology(
        &mut self,
        mesh: &crate::solver::mesh::Mesh,
    ) -> Result<MeshRefreshReport, String> {
        let upc = self.model.system.unknowns_per_cell();

        let _prof = std::env::var("CFD2_REFRESH_PROFILE").is_ok();
        macro_rules! tick { ($t:expr, $label:literal) => { if _prof { eprintln!("[refresh-prof] {}: {:.2} ms", $label, $t.elapsed().as_secs_f64()*1e3); $t = std::time::Instant::now(); } }; }
        let mut _t = std::time::Instant::now();

        // Warm-start carry-forward: the coupled FGMRES uses the `x` buffer as
        // its initial guess (absolute unknown values, not a correction — see
        // `solve_fgmres`). A cell keeps its identity across a topology refresh
        // (cell count invariant, so `num_dofs` is invariant), so the previous
        // step's converged iterate is still a good seed. Capture the OLD `x` (an
        // Arc-backed handle that outlives the reallocation) here, BEFORE step 1
        // reallocates the linear system, then copy it into the fresh `x` below.
        // At step 0 the old `x` is still zero, so this is byte-identical to a
        // fresh build; the benefit is mid-run only.
        let old_x = self
            .runtime
            .linear_port_space
            .buffer(self.runtime.linear_ports.x)
            .clone();
        let x_carry_bytes = (self.runtime.num_dofs as u64) * 4;

        // 1. Mesh buffers + CSR + block CSR + scalar-CG linear system.
        self.runtime.refresh_topology(mesh, upc)?;

        // Copy the captured warm-start into the freshly-reallocated `x`. Both
        // buffers are sized for the invariant `num_dofs`; the copy is a device
        // buffer→buffer blit (no CPU readback).
        {
            let new_x = self
                .runtime
                .linear_port_space
                .buffer(self.runtime.linear_ports.x);
            let mut enc = self.runtime.common.context.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("GenericCoupled refresh: warm-start x carry"),
                },
            );
            enc.copy_buffer_to_buffer(&old_x, 0, new_x, 0, x_carry_bytes);
            self.runtime.common.context.queue.submit(Some(enc.finish()));
        }
        tick!(_t, "1.runtime.refresh_topology(csr+scalar_cg)");

        let device = self.runtime.common.context.device.clone();
        let queue = self.runtime.common.context.queue.clone();
        let num_cells = self.runtime.common.num_cells;
        let num_faces = self.runtime.common.num_faces;

        // 2. Per-face flux buffer (only face-indexed field buffer).
        self.fields.refresh_face_count(&device, num_faces);
        tick!(_t, "2.refresh_face_count");

        // 3. Re-scatter bc tables + boundary_faces from the model spec.
        let scattered = scatter_bc_tables(mesh, &self.model, num_faces as usize, upc)?;
        self._b_bc_kind = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GenericCoupled bc_kind (refresh)"),
            contents: bytemuck::cast_slice(&scattered.bc_kind),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        self._b_bc_value = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GenericCoupled bc_value (refresh)"),
            contents: bytemuck::cast_slice(&scattered.bc_value),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        self.boundary_faces = scattered.boundary_faces;
        tick!(_t, "3.scatter_bc_tables");

        // 4a. Preconditioner (Schur or plain FGMRES/krylov) over the new system.
        let schur = build_generic_schur(
            &self.model,
            &self.recipe,
            &self.runtime,
            &self.runtime.common.mesh.b_scalar_row_offsets,
            &self.runtime.common.mesh.b_scalar_col_indices,
        )?;
        let krylov = if schur.is_some() {
            None
        } else {
            build_generic_krylov(&self.recipe, &self.runtime)?
        };
        self.schur = schur;
        self.krylov = krylov;
        tick!(_t, "4a.build_schur/krylov(fgmres+amg)");

        // 4b. Outer-convergence monitor + adaptive gate (the monitor captures
        //     the warm-start `x` buffer, which the linear-system rebuild
        //     reallocated; the gate is sized by num_faces).
        let unknown_mapping = resolve_unknown_mapping_runtime(&self.model, &self.recipe.port_registry)?;
        let outer_convergence = OuterConvergenceMonitor::new(
            &device,
            &queue,
            &self.runtime.common.context.pipeline_cache,
            &self.model,
            num_cells,
            self.runtime.linear_port_space.buffer(self.runtime.linear_ports.x),
            &unknown_mapping,
        )?;
        let outer_gate = outer_convergence.as_ref().map(|oc| {
            OuterAdaptiveGate::new(
                &device,
                &queue,
                &self.runtime.common.context.pipeline_cache,
                num_cells,
                num_faces,
                device.limits().max_compute_workgroups_per_dimension,
                &oc.b_break_status,
            )
        });
        self.outer_convergence = outer_convergence;
        self.outer_gate = outer_gate;
        tick!(_t, "4b.outer_convergence+gate");

        // 5. Generated-kernel bind groups: in-place rebuild over the refreshed
        //    buffers, cached pipelines reused (no generated-WGSL recompile).
        //    Direct field access keeps the `&mut self.kernels` borrow disjoint
        //    from the immutable registry borrows of the other fields.
        let registry = ResourceRegistry::new()
            .with_mesh(&self.runtime.common.mesh)
            .with_unified_fields(&self.fields)
            .with_buffer(
                "matrix_values",
                self.runtime.linear_port_space.buffer(self.runtime.linear_ports.values),
            )
            .with_buffer(
                "rhs",
                self.runtime.linear_port_space.buffer(self.runtime.linear_ports.rhs),
            )
            .with_buffer(
                "x",
                self.runtime.linear_port_space.buffer(self.runtime.linear_ports.x),
            )
            .with_buffer(
                "row_offsets",
                self.runtime.linear_port_space.buffer(self.runtime.linear_ports.row_offsets),
            )
            .with_buffer(
                "col_indices",
                self.runtime.linear_port_space.buffer(self.runtime.linear_ports.col_indices),
            )
            .with_buffer("bc_kind", &self._b_bc_kind)
            .with_buffer("bc_value", &self._b_bc_value)
            .with_buffer(
                "y",
                self.runtime.linear_port_space.buffer(self.runtime.linear_ports.rhs),
            );
        self.kernels
            .rebuild_bind_groups(&device, self.model.id, &self.recipe, &registry)?;
        tick!(_t, "5.rebuild_bind_groups(generated)");

        Ok(MeshRefreshReport {
            bc_overrides_reset: true,
        })
    }

    /// ALE step entry: rotate the volume history, THEN upload the new
    /// geometry, THEN upload the closed mesh fluxes (single owner of that
    /// ordering — see [`MeshResources::begin_ale_step`] for why
    /// `host_prepare_step` cannot do the rotation). Scope guard: the dual-time
    /// step retry/rollback machinery is compressible-only
    /// (`plan.model.id == "compressible"` gates in this file) while ALE is
    /// incompressible-only, so a rejected-step re-run against an
    /// already-advanced mesh cannot occur; compressible ALE will need mesh
    /// rollback (seed snapshot + regen) before those paths may fire.
    pub(crate) fn begin_ale_step(
        &self,
        mesh: &crate::solver::mesh::Mesh,
        mesh_fluxes: &[f32],
    ) -> Result<(), String> {
        let common = &self.runtime.common;
        common.mesh.begin_ale_step(
            &common.context.device,
            &common.context.queue,
            mesh,
            mesh_fluxes,
        )
    }

    /// ALE step entry for a **topology-changing** move: the mesh's
    /// face set / adjacency / nnz may differ (same cell count), so the whole
    /// mesh-topology-derived GPU stack must be rebuilt AND the volume history
    /// rotated — in this order (each step depends on the previous):
    ///   1. `rotate_volume_history` — capture the CURRENT `cell_vols` as `V^n`
    ///      into `cell_vols_old`, BEFORE the rebuild replaces `MeshResources`
    ///      (the rebuild carries the two history buffers over via swap).
    ///   2. `refresh_mesh_topology` — reallocate every topology-derived buffer
    ///      (CSR, block CSR, linear system, bc tables, `mesh_fluxes` zero-
    ///      filled), rebuild the preconditioner + bind groups. This also
    ///      uploads the NEW geometry (`cell_vols = V^{n+1}`).
    ///   3. `upload_mesh_fluxes` — write the f32-closed swept fluxes into the
    ///      freshly-reallocated `b_mesh_fluxes`.
    ///
    /// Returns the topology-refresh report (`bc_overrides_reset` — the caller
    /// re-applies any per-face BC overrides against the new faces).
    ///
    /// Flux-length is validated against the NEW face count up front so a
    /// mismatched flux vector fails before any buffer is mutated.
    pub(crate) fn begin_ale_step_topology(
        &mut self,
        mesh: &crate::solver::mesh::Mesh,
        mesh_fluxes: &[f32],
    ) -> Result<MeshRefreshReport, String> {
        if mesh_fluxes.len() != mesh.num_faces() {
            return Err(format!(
                "begin_ale_step_topology: mesh_fluxes has {} entries, mesh has {} faces",
                mesh_fluxes.len(),
                mesh.num_faces()
            ));
        }
        // 1. Rotate the volume history on the CURRENT (pre-rebuild) buffers.
        {
            let common = &self.runtime.common;
            common
                .mesh
                .rotate_volume_history(&common.context.device, &common.context.queue);
        }
        // 2. Rebuild the whole topology-derived stack (preserves the rotated
        //    history via swap; uploads V^{n+1}; reallocates mesh_fluxes zero).
        let report = self.refresh_mesh_topology(mesh)?;
        // 3. Upload the closed mesh fluxes into the new b_mesh_fluxes.
        {
            let common = &self.runtime.common;
            common
                .mesh
                .upload_mesh_fluxes(&common.context.queue, mesh_fluxes);
        }
        Ok(report)
    }

    /// Seed the ALE volume history buffers (`cell_vols_old{,_old}` :=
    /// `cell_vols`); see [`MeshResources::seed_volume_history`]. Invoked from
    /// `GpuProgramPlan::initialize_history` for every model — numerically a
    /// no-op unless the model's kernels bind the history (only `*_ale`
    /// variants do).
    pub(crate) fn seed_volume_history(&self) {
        let common = &self.runtime.common;
        common
            .mesh
            .seed_volume_history(&common.context.device, &common.context.queue);
    }
}

fn validate_schur_model(
    model: &ModelSpec,
    unknown_mapping: &ResolvedUnknownMapping,
) -> Result<(f32, u32, crate::solver::model::SchurBlockLayout), String> {
    let Some(solver) = model.linear_solver else {
        return Err("model does not define a linear solver spec".into());
    };
    let ModelPreconditionerSpec::Schur { omega, sweeps_cap, layout } = solver.preconditioner
    else {
        return Err("model does not request Schur preconditioning".into());
    };

    let method = model.method()?;
    if !matches!(method, crate::solver::model::method::MethodSpec::Coupled(_)) {
        return Err("Schur preconditioner is only wired for the coupled pipeline".to_string());
    }
    layout.validate(model.system.unknowns_per_cell())?;

    // Validate the layout against the equation targets used to assemble the system.
    //
    // The Schur bridge is N-generic: the u-block may contain any number of
    // non-pressure unknowns (up to SCHUR_MAX_U), with a single pressure-like
    // scalar; the layout must cover exactly the model's equation targets
    // (e.g. buoyant: u = [U_x, U_y, T], p).
    //
    // The layout indexes the packed coupled x-vector (0..unknowns_per_cell),
    // so targets must be resolved through the coupled FluxLayout — NOT the
    // state layout. The two coincide only when the solved unknowns form a
    // prefix of the state layout (e.g. incompressible_momentum).
    let _ = unknown_mapping;
    let flux_layout = crate::solver::ir::FluxLayout::from_system(&model.system);
    let mut target_indices = std::collections::BTreeSet::new();
    let mut scalar_targets = std::collections::BTreeSet::new();
    for eq in model.system.equations() {
        let target = eq.target();
        let comps = target.kind().component_count();

        match target.kind() {
            crate::solver::model::backend::ast::FieldKind::Scalar => {
                let idx = flux_layout
                    .offset_for_field_component(*target, 0)
                    .ok_or_else(|| format!("missing '{}' in coupled layout", target.name()))?;
                target_indices.insert(idx);
                scalar_targets.insert(idx);
            }
            _ => {
                for comp in 0..comps {
                    let idx = flux_layout
                        .offset_for_field_component(*target, comp as u32)
                        .ok_or_else(|| {
                            format!(
                                "missing '{}' component {} in coupled layout",
                                target.name(),
                                comp
                            )
                        })?;
                    target_indices.insert(idx);
                }
            }
        }
    }

    if !scalar_targets.contains(&layout.p) {
        return Err(format!(
            "SchurBlockLayout {:?} pressure index does not match any scalar equation target",
            layout
        ));
    }

    let mut layout_indices = std::collections::BTreeSet::new();
    for &u in layout.u_indices() {
        layout_indices.insert(u);
    }
    layout_indices.insert(layout.p);

    if layout_indices != target_indices {
        return Err(format!(
            "SchurBlockLayout {:?} must cover exactly the model equation targets (layout={:?}, targets={:?})",
            layout,
            layout_indices,
            target_indices
        ));
    }

    Ok((omega, sweeps_cap, layout))
}

fn build_generic_schur(
    model: &ModelSpec,
    recipe: &SolverRecipe,
    runtime: &GpuCsrRuntime,
    scalar_row_offsets: &wgpu::Buffer,
    scalar_col_indices: &wgpu::Buffer,
) -> Result<Option<GenericCoupledSchurResources>, String> {
    let Some(spec) = model.linear_solver else {
        return Ok(None);
    };
    match spec.preconditioner {
        ModelPreconditionerSpec::Default => return Ok(None),
        ModelPreconditionerSpec::Schur { .. } => {}
    }

    // Compute the unknown mapping for Schur validation
    let unknown_mapping = resolve_unknown_mapping_runtime(model, &recipe.port_registry)?;
    let (omega, sweeps_cap, layout) = validate_schur_model(model, &unknown_mapping)?;

    let LinearSolverType::Fgmres { max_restart } = recipe.linear_solver.solver_type else {
        return Err(
            "Schur preconditioner requires LinearSolverType::Fgmres in the recipe".to_string(),
        );
    };

    let device = &runtime.common.context.device;
    let cache = &runtime.common.context.pipeline_cache;
    let num_cells = runtime.common.num_cells;
    let num_dofs = runtime.num_dofs;
    let scalar_nnz = runtime.common.mesh.scalar_col_indices.len() as u64;

    let u_len = layout.u_len;
    let mut u0123 = [0u32; 4];
    let mut u4567 = [0u32; 4];
    for (i, &u) in layout.u_indices().iter().enumerate() {
        if i < 4 {
            u0123[i] = u;
        } else {
            u4567[i - 4] = u;
        }
    }

    let b_diag_u = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("GenericCoupled Schur diag_u_inv"),
        size: (num_cells as u64) * (u_len as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let b_diag_p = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("GenericCoupled Schur diag_p_inv"),
        size: (num_cells as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let b_precond_params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("GenericCoupled Schur precond_params"),
        size: std::mem::size_of::<GpuSchurPrecondGenericParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // The model-spec omega of 1.0 means "auto": use the heavy-ball weight that
    // turns the relax_pressure ping-pong into a second-order Richardson
    // iteration (see coupled_schur::heavy_ball_omega for the math + numbers).
    let omega = crate::solver::gpu::modules::coupled_schur::heavy_ball_omega(omega);
    let params = GpuSchurPrecondGenericParams {
        n: num_dofs,
        num_cells,
        omega,
        unknowns_per_cell: model.system.unknowns_per_cell(),
        p: layout.p,
        u_len,
        _pad0: 0,
        _pad1: 0,
        u0123,
        u4567,
    };
    runtime
        .common
        .context
        .queue
        .write_buffer(&b_precond_params, 0, bytes_of(&params));

    let b_p_matrix_values = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("GenericCoupled Schur p_matrix_values"),
        size: scalar_nnz * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let b_setup_params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("GenericCoupled Schur setup_params"),
        size: std::mem::size_of::<GpuGenericCoupledSchurSetupParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let setup_pipeline = GenericCoupledSchurPreconditioner::build_setup_pipeline(device, cache)?;
    let matrix_values = runtime
        .linear_port_space
        .buffer(runtime.linear_ports.values);
    let diagonal_indices = runtime
        .common
        .mesh
        .buffer_for_binding_name("diagonal_indices")
        .ok_or_else(|| "missing diagonal_indices mesh buffer".to_string())?;

    let setup_bg = GenericCoupledSchurPreconditioner::build_setup_bind_group(
        device,
        &setup_pipeline,
        GenericCoupledSchurSetupBindGroupInputs {
            scalar_row_offsets,
            diagonal_indices,
            matrix_values,
            diag_u_inv: &b_diag_u,
            diag_p_inv: &b_diag_p,
            p_matrix_values: &b_p_matrix_values,
            setup_params: &b_setup_params,
        },
    )?;

    let system = LinearSystemView {
        ports: runtime.linear_ports,
        space: &runtime.linear_port_space,
    };

    let precond_bg = FgmresWorkspace::build_precond_bind_group(
        device,
        cache,
        "generic_coupled FGMRES precond BG",
        |name| match name {
            "diag_u" => Some(b_diag_u.as_entire_binding()),
            "diag_v" => Some(b_diag_p.as_entire_binding()),
            "diag_p" => Some(b_diag_p.as_entire_binding()),
            _ => None,
        },
    )?;
    let fgmres = FgmresWorkspace::new_from_system(
        device,
        cache,
        num_dofs,
        num_cells,
        max_restart,
        recipe.linear_solver.update_strategy,
        system,
        precond_bg,
        "generic_coupled",
    )?;

    let precond = GenericCoupledSchurPreconditioner::new(
        device,
        cache.clone(),
        GenericCoupledSchurPreconditionerInputs {
            num_cells,
            pressure_row_offsets: scalar_row_offsets,
            pressure_col_indices: scalar_col_indices,
            pressure_values: &b_p_matrix_values,
            pressure_row_offsets_host: runtime.common.mesh.scalar_row_offsets.clone(),
            pressure_col_indices_host: runtime.common.mesh.scalar_col_indices.clone(),
            pressure_num_nonzeros: scalar_nnz,
            diag_u_inv: &b_diag_u,
            diag_p_inv: &b_diag_p,
            precond_params: &b_precond_params,
            setup_bg,
            setup_pipeline,
            setup_params: b_setup_params,
            unknowns_per_cell: model.system.unknowns_per_cell(),
            p: layout.p,
            u_len,
            u0123,
            u4567,
            pressure_kind: CoupledPressureSolveKind::from_config(
                recipe.linear_solver.preconditioner,
            ),
            sweeps_cap,
        },
    )?;

    let dispatch = DispatchGrids::for_sizes(num_dofs, num_cells);

    Ok(Some(GenericCoupledSchurResources {
        solver: KrylovSolveModule::new(fgmres, precond),
        dispatch,
        _b_diag_u: b_diag_u,
        _b_diag_p: b_diag_p,
        _b_precond_params: b_precond_params,
        _b_p_matrix_values: b_p_matrix_values,
    }))
}

fn build_generic_krylov(
    recipe: &SolverRecipe,
    runtime: &GpuCsrRuntime,
) -> Result<Option<GenericCoupledKrylovResources>, String> {
    let LinearSolverType::Fgmres { max_restart } = recipe.linear_solver.solver_type else {
        return Ok(None);
    };

    let device = &runtime.common.context.device;
    let cache = &runtime.common.context.pipeline_cache;
    let num_cells = runtime.common.num_cells;
    let n = runtime.num_dofs;

    let diag_bytes = (n.max(1) as u64) * 4;

    let b_diag_u = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("generic_coupled:jacobi_diag_u_inv"),
        size: diag_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let b_diag_v = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("generic_coupled:jacobi_diag_v_inv"),
        size: diag_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let b_diag_p = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("generic_coupled:jacobi_diag_p_inv"),
        size: diag_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let system = LinearSystemView {
        ports: runtime.linear_ports,
        space: &runtime.linear_port_space,
    };

    let precond_bg = FgmresWorkspace::build_precond_bind_group(
        device,
        cache,
        "generic_coupled FGMRES precond BG",
        |name| match name {
            "diag_u" => Some(b_diag_u.as_entire_binding()),
            "diag_v" => Some(b_diag_v.as_entire_binding()),
            "diag_p" => Some(b_diag_p.as_entire_binding()),
            _ => None,
        },
    )?;

    let fgmres = FgmresWorkspace::new_from_system(
        device,
        cache,
        n,
        num_cells,
        max_restart.max(1),
        recipe.linear_solver.update_strategy,
        system,
        precond_bg,
        "generic_coupled",
    )?;

    let unknowns_per_cell: u32 = recipe
        .unknowns_per_cell
        .try_into()
        .map_err(|_| "recipe.unknowns_per_cell overflows u32".to_string())?;
    let (row_offsets, col_indices) = crate::solver::gpu::csr::build_block_csr(
        &runtime.common.mesh.scalar_row_offsets,
        &runtime.common.mesh.scalar_col_indices,
        unknowns_per_cell,
    );

    let precond_inputs = RuntimePreconditionerInputs {
        kind: recipe.linear_solver.preconditioner,
        num_cells,
        num_dofs: n,
        num_nonzeros: runtime.num_nonzeros,
        row_offsets,
        col_indices,
        matrix_values: runtime
            .linear_port_space
            .buffer(runtime.linear_ports.values)
            .clone(),
    };
    let solver = KrylovSolveModule::new(
        fgmres,
        RuntimePreconditionerModule::new(
            device,
            runtime.common.context.pipeline_cache.clone(),
            precond_inputs,
        ),
    );
    let dispatch = DispatchGrids::for_sizes(n, num_cells);

    Ok(Some(GenericCoupledKrylovResources {
        solver,
        dispatch,
        _b_diag_u: b_diag_u,
        _b_diag_v: b_diag_v,
        _b_diag_p: b_diag_p,
    }))
}

impl PlanLinearSystemDebug for GenericCoupledProgramResources {
    fn set_linear_system(&self, matrix_values: &[f32], rhs: &[f32]) -> Result<(), String> {
        self.runtime.set_linear_system(matrix_values, rhs)
    }

    fn solve_linear_system_with_size(
        &mut self,
        n: u32,
        max_iters: u32,
        tol: f32,
    ) -> Result<LinearSolverStats, String> {
        if n != self.runtime.num_dofs {
            return Err(format!(
                "requested solve size {} does not match num_dofs {}",
                n, self.runtime.num_dofs
            ));
        }

        if let Some(schur) = &mut self.schur {
            let system = LinearSystemView {
                ports: self.runtime.linear_ports,
                space: &self.runtime.linear_port_space,
            };

            // Map the debug max_iters into a maximum restart size.
            let max_restart = (max_iters as usize)
                .max(1)
                .min(schur.solver.fgmres.max_restart());

            Ok(solve_fgmres(
                &mut schur.solver,
                SolveFgmresArgs {
                    context: &self.runtime.common.context,
                    system,
                    n,
                    num_cells: self.runtime.common.num_cells,
                    dispatch: schur.dispatch,
                    max_restart,
                    max_iters,
                    tol,
                    tol_abs: tol * 1e-4,
                    precond_label: "GenericCoupled Schur (debug)",
                    use_encoded_seed_basis0: false,
                    tight_budget: false,
                },
            ))
        } else if let Some(krylov) = &mut self.krylov {
            let system = LinearSystemView {
                ports: self.runtime.linear_ports,
                space: &self.runtime.linear_port_space,
            };

            let max_restart = match self.linear_solver.solver_type {
                LinearSolverType::Fgmres { max_restart } => max_restart,
                _ => 30,
            };
            Ok(solve_fgmres(
                &mut krylov.solver,
                SolveFgmresArgs {
                    context: &self.runtime.common.context,
                    system,
                    n,
                    num_cells: self.runtime.common.num_cells,
                    dispatch: krylov.dispatch,
                    max_restart: max_restart.max(1),
                    max_iters,
                    tol,
                    tol_abs: tol * 1e-4,
                    precond_label: "GenericCoupled FGMRES (debug)",
                    use_encoded_seed_basis0: false,
                    tight_budget: false,
                },
            ))
        } else {
            Ok(self.runtime.solve_linear_system_cg(max_iters, tol))
        }
    }

    fn get_linear_solution(&self) -> PlanFuture<'_, Result<Vec<f32>, String>> {
        Box::pin(async move {
            let raw = self
                .runtime
                .common
                .read_buffer(
                    self.runtime
                        .linear_port_space
                        .buffer(self.runtime.linear_ports.x),
                    (self.runtime.num_dofs as u64) * 4,
                    "GenericCoupled CSR Runtime Staging Buffer (cached)",
                )
                .await;
            Ok(bytemuck::cast_slice(&raw).to_vec())
        })
    }

    fn get_linear_matrix(&self) -> PlanFuture<'_, Result<Vec<f32>, String>> {
        Box::pin(async move {
            let raw = self
                .runtime
                .common
                .read_buffer(
                    self.runtime
                        .linear_port_space
                        .buffer(self.runtime.linear_ports.values),
                    (self.runtime.num_nonzeros as u64) * 4,
                    "GenericCoupled matrix_values readback",
                )
                .await;
            Ok(bytemuck::cast_slice(&raw).to_vec())
        })
    }

    fn get_linear_rhs(&self) -> PlanFuture<'_, Result<Vec<f32>, String>> {
        Box::pin(async move {
            let raw = self
                .runtime
                .common
                .read_buffer(
                    self.runtime
                        .linear_port_space
                        .buffer(self.runtime.linear_ports.rhs),
                    (self.runtime.num_dofs as u64) * 4,
                    "GenericCoupled rhs readback",
                )
                .await;
            Ok(bytemuck::cast_slice(&raw).to_vec())
        })
    }
}

fn res(plan: &GpuProgramPlan) -> &GenericCoupledProgramResources {
    &plan.resources.backend
}

fn res_mut(plan: &mut GpuProgramPlan) -> &mut GenericCoupledProgramResources {
    &mut plan.resources.backend
}

pub(crate) fn named_params_for_recipe(
    model: &crate::solver::model::ModelSpec,
    _recipe: &SolverRecipe,
) -> Result<HashMap<&'static str, ProgramParamHandler>, String> {
    crate::solver::gpu::lowering::named_params::named_params_for_model(model)
}

pub(crate) fn spec_num_cells(plan: &GpuProgramPlan) -> u32 {
    res(plan).runtime.common.num_cells
}

pub(crate) fn spec_time(plan: &GpuProgramPlan) -> f32 {
    res(plan).time_integration.time as f32
}

pub(crate) fn spec_dt(plan: &GpuProgramPlan) -> f32 {
    res(plan).time_integration.dt
}

pub(crate) fn spec_state_buffer(plan: &GpuProgramPlan) -> &wgpu::Buffer {
    res(plan).fields.current_state()
}

pub(crate) fn spec_write_state_bytes(plan: &GpuProgramPlan, bytes: &[u8]) -> Result<(), String> {
    res(plan)
        .fields
        .write_state_bytes(&plan.context.queue, bytes);
    Ok(())
}

pub(crate) fn spec_write_state_bytes_current(
    plan: &GpuProgramPlan,
    bytes: &[u8],
) -> Result<(), String> {
    res(plan)
        .fields
        .write_state_bytes_current(&plan.context.queue, bytes);
    Ok(())
}

pub(crate) fn spec_set_bc_value(
    plan: &GpuProgramPlan,
    boundary: crate::solver::gpu::enums::GpuBoundaryType,
    unknown_component: u32,
    value: f32,
) -> Result<(), String> {
    spec_set_bc_values_per_face(plan, boundary, unknown_component, &|_face_idx| value)
}

pub(crate) fn spec_set_bc_values_per_face(
    plan: &GpuProgramPlan,
    boundary: crate::solver::gpu::enums::GpuBoundaryType,
    unknown_component: u32,
    value_for_face: &dyn Fn(u32) -> f32,
) -> Result<(), String> {
    let coupled_stride = plan.model.system.unknowns_per_cell();
    if coupled_stride == 0 {
        return Err("model has no coupled unknowns".into());
    }
    if unknown_component >= coupled_stride {
        return Err(format!(
            "unknown_component {unknown_component} out of range (stride={coupled_stride})"
        ));
    }

    let boundary_idx = boundary as u32;
    if boundary_idx as usize >= BOUNDARY_TYPE_COUNT {
        return Err(format!("invalid boundary type index {boundary_idx}"));
    }

    let table = HostBcTable::new(coupled_stride as usize);

    // Apply the boundary value to all boundary faces of this type.
    // (Boundary index 0 is reserved for "None" and should have no boundary faces.)
    let faces = res(plan)
        .boundary_faces
        .get(boundary_idx as usize)
        .ok_or_else(|| format!("missing boundary_faces[{boundary_idx}]"))?;
    for &face_idx in faces {
        let value = value_for_face(face_idx);
        let offset_bytes = table.byte_offset(face_idx as usize, unknown_component as usize);
        plan.context
            .queue
            .write_buffer(&res(plan)._b_bc_value, offset_bytes, bytes_of(&value));
    }
    Ok(())
}

pub(crate) fn host_prepare_step(plan: &mut GpuProgramPlan) {
    plan.step_linear_stats.clear();
    plan.step_graph_timings.clear();
    plan.outer_iterations = 0;
    plan.outer_residual_u = None;
    plan.outer_residual_p = None;
    plan.outer_step_status = None;
    plan.outer_field_residuals.clear();
    plan.outer_field_residuals_scaled.clear();
    plan.prev_outer_field_residuals_scaled.clear();
    plan.repeat_break = false;
    plan.positivity_min_rho = None;
    plan.positivity_min_p = None;
    plan.positivity_rho_undershoot_count = 0;
    plan.positivity_pressure_undershoot_count = 0;

    let device = plan.context.device.clone();
    let queue = plan.context.queue.clone();
    plan.current_dtau = Some(res(plan).fields.constants.values().dtau);
    let r = res_mut(plan);
    if let Some(monitor) = r.outer_convergence.as_mut() {
        monitor.reset_step();
    }
    r.fields.advance_step();

    // OpenFOAM-style `backward` startup: the first step falls back to Euler
    // because the `n-1` history is not yet meaningful; once at least one step
    // has advanced, switch back to the requested scheme (BDF2).
    let requested = r.requested_time_scheme;
    let effective = if requested == crate::solver::gpu::enums::TimeScheme::BDF2
        && r.time_integration.step_count == 0
    {
        crate::solver::gpu::enums::TimeScheme::Euler
    } else {
        requested
    };
    {
        let values = r.fields.constants.values_mut();
        values.time_scheme = effective as u32;
    }

    // Seed the writable `state` buffer with the previous state so kernels that
    // read from `state` (e.g. gradient/flux stages during implicit outer
    // iterations) start from a consistent iterate, not stale data from the
    // rotated ping-pong buffer.
    let size = r.fields.state_size_bytes();
    let src = r.fields.previous_state();
    let dst = r.fields.current_state();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("generic_coupled:pre_step_copy"),
    });
    encoder.copy_buffer_to_buffer(src, 0, dst, 0, size);
    if r.fields.constants.values().dtau > 0.0 {
        r.fields.snapshot_for_iteration(&mut encoder);
    }
    queue.submit(Some(encoder.finish()));
    crate::count_submission!("Generic Coupled", "pre_step_copy");
    r.time_integration
        .prepare_step(&mut r.fields.constants, &queue);
}

pub(crate) fn host_finalize_step(plan: &mut GpuProgramPlan) {
    let positivity_action = evaluate_step_positivity(plan).map_or(PositivityFallbackAction::None, |report| {
        plan.positivity_min_rho = Some(report.min_rho);
        plan.positivity_min_p = Some(report.min_p);
        plan.positivity_rho_undershoot_count = report.rho_undershoot_count;
        plan.positivity_pressure_undershoot_count = report.pressure_undershoot_count;

        if !report.has_violation() {
            PositivityFallbackAction::None
        } else if should_retry_nonconverged_step(plan, false) {
            plan.outer_step_status = Some(OuterStepStatus::RejectedRetry);
            PositivityFallbackAction::RetryReject
        } else {
            plan.outer_step_status = Some(OuterStepStatus::AcceptedNonconverged);
            PositivityFallbackAction::RollbackAccept
        }
    });

    let model_id = plan.model.id;
    let outer_step_status = plan.outer_step_status;
    if outer_step_status == Some(OuterStepStatus::RejectedRetry) {
        plan.rejected_retry_count = plan.rejected_retry_count.saturating_add(1);
        rollback_rejected_dual_time_step(plan);
        plan.retry_step = true;
        return;
    }

    if positivity_action == PositivityFallbackAction::RollbackAccept {
        rollback_rejected_dual_time_step(plan);
        return;
    }

    let queue = plan.context.queue.clone();
    let current_dtau = {
        let r = res_mut(plan);
        {
            let values = r.fields.constants.values_mut();
            values.time_scheme = r.requested_time_scheme as u32;
        }
        r.time_integration
            .finalize_step(&mut r.fields.constants, &queue);

        if let Some((next_dt, next_dtau)) = next_step_backoff_targets(
            model_id,
            outer_step_status,
            r.time_integration.dt,
            r.fields.constants.values().dtau,
            r.nonconverged_dt_scale,
            r.nonconverged_dtau_scale,
        ) {
            r.time_integration.set_dt(next_dt, &mut r.fields.constants, &queue);
            {
                let values = r.fields.constants.values_mut();
                values.dtau = next_dtau;
            }
            r.fields.constants.write(&queue);
        }
        r.fields.constants.values().dtau
    };
    plan.current_dtau = Some(current_dtau);
}

fn rollback_rejected_dual_time_step(plan: &mut GpuProgramPlan) {
    let device = plan.context.device.clone();
    let queue = plan.context.queue.clone();
    let model_id = plan.model.id;

    let (current_dt, current_dtau, dt_scale, dtau_scale, step_handle) = {
        let r = res(plan);
        (
            r.time_integration.dt,
            r.fields.constants.values().dtau,
            r.nonconverged_dt_scale,
            r.nonconverged_dtau_scale,
            r.fields.step_handle(),
        )
    };

    let current_dtau = {
        let r = res_mut(plan);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("generic_coupled:rollback_rejected_step"),
        });
        r.fields.restore_from_snapshot(&mut encoder);
        queue.submit(Some(encoder.finish()));
        crate::count_submission!("Generic Coupled", "rollback_rejected_step");

        let previous_step_index = (step_handle.load(Ordering::Relaxed) + 2) % 3;
        step_handle.store(previous_step_index, Ordering::Relaxed);
        r.time_integration
            .rollback_prepare_step(&mut r.fields.constants, &queue);

        if let Some((next_dt, next_dtau)) = next_step_backoff_targets(
            model_id,
            Some(OuterStepStatus::RejectedRetry),
            current_dt,
            current_dtau,
            dt_scale,
            dtau_scale,
        ) {
            r.time_integration.set_dt(next_dt, &mut r.fields.constants, &queue);
            {
                let values = r.fields.constants.values_mut();
                values.dtau = next_dtau;
            }
            r.fields.constants.write(&queue);
        }
        r.fields.constants.values().dtau
    };
    plan.current_dtau = Some(current_dtau);
}

pub(crate) fn host_solve_linear_system(plan: &mut GpuProgramPlan) {
    let context = crate::solver::gpu::context::GpuContext {
        device: plan.context.device.clone(),
        queue: plan.context.queue.clone(),
        timestamp_query: plan.context.timestamp_query,
        timestamps_inside_encoders: plan.context.timestamps_inside_encoders,
        timestamp_period_ns: plan.context.timestamp_period_ns,
        pipeline_cache: plan.context.pipeline_cache.clone(),
    };

    let is_first_outer = plan.step_linear_stats.is_empty();
    // Previous outer's worst scaled correction (the plateau detector's own
    // metric), the nonlinear-error estimate the EW forcing scales from.
    let prev_outer_err = plan
        .outer_field_residuals_scaled
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NAN, f32::max);
    let r = res_mut(plan);
    // Enable encoded basis seeding by default for multi-outer host-driven solves.
    // Keep single-outer implicit solves opt-in via env var because that path is
    // still more sensitive in OpenFOAM parity diagnostics.
    let use_encoded_seed_basis0 = encoded_seed_basis0_enabled(r.outer_iters > 1);
    let tol = outer_forcing_tolerance(
        r.linear_solver.tolerance,
        is_first_outer,
        r.outer_iters > 1,
        prev_outer_err,
    );
    // Multi-outer host-driven solves route through the chunked one-submission
    // machinery (see `host_chunked_solve_enabled`); it requires the encoded
    // seed, whose availability `use_encoded_seed_basis0` already gates.
    let use_chunked = use_encoded_seed_basis0 && host_chunked_solve_enabled();

    if let Some(schur) = &mut r.schur {
        let system = LinearSystemView {
            ports: r.runtime.linear_ports,
            space: &r.runtime.linear_port_space,
        };

        let max_restart = match r.linear_solver.solver_type {
            LinearSolverType::Fgmres { max_restart } => max_restart,
            _ => 30,
        };
        let max_restart = max_restart.max(1);

        let args = SolveFgmresArgs {
            context: &context,
            system,
            n: r.runtime.num_dofs,
            num_cells: r.runtime.common.num_cells,
            dispatch: schur.dispatch,
            max_restart,
            max_iters: r.linear_solver.max_iters,
            tol,
            tol_abs: r.linear_solver.tolerance_abs,
            precond_label: "generic_coupled:schur",
            use_encoded_seed_basis0,
            tight_budget: use_chunked,
        };
        let stats = if use_chunked {
            submit_solve_fgmres_fixed_iterations_chunked(
                &mut schur.solver,
                args,
                &mut |_| {},
                &mut |_| {},
            )
        } else {
            solve_fgmres(&mut schur.solver, args)
        };
        plan.last_linear_stats = stats;
        plan.step_linear_stats.push(stats);
        return;
    }

    if let Some(krylov) = &mut r.krylov {
        let system = LinearSystemView {
            ports: r.runtime.linear_ports,
            space: &r.runtime.linear_port_space,
        };

        let max_restart = match r.linear_solver.solver_type {
            LinearSolverType::Fgmres { max_restart } => max_restart,
            _ => 30,
        };

        let args = SolveFgmresArgs {
            context: &context,
            system,
            n: r.runtime.num_dofs,
            num_cells: r.runtime.common.num_cells,
            dispatch: krylov.dispatch,
            max_restart: max_restart.max(1),
            max_iters: r.linear_solver.max_iters,
            tol,
            tol_abs: r.linear_solver.tolerance_abs,
            precond_label: "generic_coupled:fgmres",
            use_encoded_seed_basis0,
            tight_budget: use_chunked,
        };
        let stats = if use_chunked {
            submit_solve_fgmres_fixed_iterations_chunked(
                &mut krylov.solver,
                args,
                &mut |_| {},
                &mut |_| {},
            )
        } else {
            solve_fgmres(&mut krylov.solver, args)
        };
        plan.last_linear_stats = stats;
        plan.step_linear_stats.push(stats);
        return;
    }

    let stats = r.runtime.solve_linear_system_cg(r.linear_solver.max_iters, tol);
    plan.last_linear_stats = stats;
    plan.step_linear_stats.push(stats);
}

/// Route multi-outer host-driven solves through the chunked one-submission
/// machinery (GPU-side seed + rel-scale clamp + restart guard + stall +
/// convergence flag, ONE submission and ONE blocking scalar readback per
/// chunk) instead of the host restart loop, which pays a host residual
/// recompute + blocking readback per restart cycle. Semantics are preserved
/// GPU-side: `clamp_rel_scale` implements the same
/// `rel_scale = min(||b||, ||r0||)`, the encoded stall matches the host
/// checkpoint stall, and the restart guard replicates the snapshot/restore
/// monotonicity logic. `CFD2_NO_HOST_CHUNKED=1` restores the host loop.
fn host_chunked_solve_enabled() -> bool {
    !std::env::var("CFD2_NO_HOST_CHUNKED").is_ok_and(|v| v == "1")
}

/// Eisenstat-Walker-style loosened tolerance for the FIRST outer iteration of
/// a multi-outer step: the step re-linearizes immediately after it, so its
/// linearization error is O(1) and solving past ~1e-2 relative is pure
/// over-solving (the later outers still run at the model tolerance, so the
/// converged step is unchanged). Biggest effect on from-rest first solves that
/// otherwise burn the whole FGMRES budget at the tight tolerance.
/// `CFD2_NO_EW_FIRST=1` disables.
fn first_outer_tolerance(base_tol: f32, is_first_outer: bool, multi_outer: bool) -> f32 {
    if !is_first_outer
        || !multi_outer
        || std::env::var("CFD2_NO_EW_FIRST").is_ok_and(|v| v == "1")
    {
        return base_tol;
    }
    base_tol.max(1e-2)
}

/// Full Eisenstat-Walker-style forcing for outers 2..N of a multi-outer step:
/// the linear tolerance tracks the OUTER Picard error instead of solving every
/// middle system two decades past it. `prev_outer_err` is the previous outer's
/// worst per-field scaled correction (the plateau detector's own metric);
/// eta_k = clamp(0.1 * prev_err, base_tol, 1e-2) solves one decade below the
/// current nonlinear error, so late outers (small corrections) still get the
/// full model tolerance and the CONVERGED step is unchanged. The first outer
/// keeps the EW-lite 1e-2 (its linearization error is O(1)); when no
/// correction data exists (stats not collected / first step), outers 2..N fall
/// back to the base tolerance. `CFD2_NO_EW_FULL=1` restores first-outer-only.
fn outer_forcing_tolerance(
    base_tol: f32,
    is_first_outer: bool,
    multi_outer: bool,
    prev_outer_err: f32,
) -> f32 {
    if is_first_outer || !multi_outer {
        return first_outer_tolerance(base_tol, is_first_outer, multi_outer);
    }
    if !prev_outer_err.is_finite() || std::env::var("CFD2_NO_EW_FULL").is_ok_and(|v| v == "1") {
        return base_tol;
    }
    let hi = base_tol.max(1e-2);
    (0.1 * prev_outer_err).clamp(base_tol, hi)
}

/// True when the adaptive outer-loop *plateau* detector drives this step: ANY
/// coupled SIMPLE solver (incompressible_momentum, allmach_pressure/thermal,
/// buoyant_incompressible, …) with a real outer loop (`outer_iters > 1`), adaptive
/// outer convergence requested (`collect_convergence_stats` / `outer_break_enabled`,
/// which the GUI defaults enable), and NOT pseudo-transient (`dtau > 0`, where the
/// correction-norm plateau means something different).
///
/// Routing to the non-batched path (which can skip the remaining, unencoded outer
/// iterations) and the detector itself are gated on this SAME condition so they
/// never disagree. Explicitly excluded: the density-based `compressible` solver
/// (its own conserved-target convergence + single outer iteration) and every
/// `_mms` order-verification variant (which pins fixed iterations). Any
/// pseudo-transient, fixed-iteration, or break-disabled config also stays on the
/// batched, fixed-count path.
fn outer_plateau_active(plan: &GpuProgramPlan) -> bool {
    let id = plan.model.id;
    if id == "compressible" || id.ends_with("_mms") {
        return false;
    }
    plan.collect_convergence_stats && {
        let r = res(plan);
        r.outer_break_enabled
            && r.outer_iters > 1
            && r.fields.constants.values().dtau <= 0.0
    }
}

/// Minimum outer sweeps before the plateau detector may take the STALL exit —
/// an empirically validated floor (Ghia validates at 5). Below this the
/// correction phase is not yet complete, so a stall-exit could change the
/// physics. The TOLERANCE exit (every field's scaled correction under
/// `outer_tol`) is a genuine convergence criterion and is allowed below the
/// floor (from 2 sweeps).
const OUTER_PLATEAU_MIN_ITERS: usize = 5;
/// Minimum outer sweeps before the TOLERANCE exit: at least one re-linearized
/// second sweep must confirm the first's correction, so a single lucky
/// first-outer solve cannot end the step.
const OUTER_TOL_EXIT_MIN_ITERS: usize = 2;
/// A field has "stalled" when its scaled correction stopped shrinking by more than
/// (1 - factor) per sweep AND is not growing past the ceiling. The band
/// `[factor, ceiling]` treats the settled-but-slightly-drifting velocity residual
/// as done while refusing to call a growing correction converged.
const OUTER_PLATEAU_FACTOR: f32 = 0.98;
const OUTER_PLATEAU_CEILING: f32 = 1.01;

/// Adaptive outer-loop plateau detector. Returns `true` when every solved field's
/// scaled outer-correction has stalled (or is already under tolerance) so further
/// sweeps would not change the solution — the signal to stop the outer loop.
/// Requires at least [`OUTER_PLATEAU_MIN_ITERS`] sweeps and a previous residual to
/// compare against; a field still meaningfully decreasing OR growing blocks the exit.
fn outer_corrections_plateaued(plan: &GpuProgramPlan, iters_done: usize) -> bool {
    let cur = &plan.outer_field_residuals_scaled;
    if cur.is_empty() {
        return false;
    }
    let (tol_rel, tol_abs) = {
        let r = res(plan);
        (r.outer_tol.max(0.0), r.outer_tol_abs.max(0.0))
    };
    // Tolerance exit: every solved field's scaled correction is already under
    // tolerance — allowed below the stall floor (see the const docs).
    if iters_done >= OUTER_TOL_EXIT_MIN_ITERS
        && cur.iter().all(|(_, r)| *r <= tol_rel || *r <= tol_abs)
    {
        return true;
    }
    if iters_done < OUTER_PLATEAU_MIN_ITERS {
        return false;
    }
    let prev = &plan.prev_outer_field_residuals_scaled;
    if prev.is_empty() {
        return false;
    }
    // Every field must be DONE (converged or plateaued); if any is missing a prior
    // value, still improving, or growing, do not exit.
    for (name, r_cur) in cur.iter() {
        let r_cur = *r_cur;
        // Already below tolerance -> done regardless of a noisy ratio near zero.
        if r_cur <= tol_rel || r_cur <= tol_abs {
            continue;
        }
        let Some((_, r_prev)) = prev.iter().find(|(n, _)| n == name) else {
            return false;
        };
        let ratio = r_cur / r_prev.max(1e-30);
        // Still improving (< factor) or growing (> ceiling) -> block the exit.
        if ratio < OUTER_PLATEAU_FACTOR || ratio > OUTER_PLATEAU_CEILING {
            return false;
        }
    }
    true
}

pub(crate) fn host_after_solve(plan: &mut GpuProgramPlan) {
    let iters_done = plan.step_linear_stats.len();
    plan.outer_iterations = iters_done as u32;

    let (outer_iters, outer_break_enabled, collect_convergence_stats) = {
        let r = res(plan);
        (
            r.outer_iters,
            r.outer_break_enabled,
            plan.collect_convergence_stats,
        )
    };
    let is_final_outer_iter = iters_done >= outer_iters;

    let break_should_run = outer_break_enabled
        && outer_iters > 1
        && plan.last_linear_stats.converged
        && !plan.repeat_break;

    if !break_should_run && !collect_convergence_stats {
        if is_final_outer_iter {
            finalize_outer_step_status(plan, None);
        }
        return;
    }

    let mut break_converged = None;

    if collect_convergence_stats {
        let (delta, scale) = match compute_outer_residuals(plan) {
            Some(result) => result,
            None => {
                if is_final_outer_iter {
                    finalize_outer_step_status(plan, None);
                }
                return;
            }
        };

        // Adaptive outer-loop plateau detector (incompressible_momentum default).
        // Handles convergence entirely here — the GPU break below is
        // converged-gated and therefore unreachable for the plateauing SIMPLE
        // corrections. `prev` is refreshed every iter so ratios stay adjacent.
        if outer_plateau_active(plan) {
            let plateaued = outer_corrections_plateaued(plan, iters_done);
            plan.prev_outer_field_residuals_scaled = plan.outer_field_residuals_scaled.clone();
            if plateaued {
                plan.repeat_break = true;
            }
            if plan.repeat_break || is_final_outer_iter {
                finalize_outer_step_status(plan, plateaued.then_some(true));
            }
            return;
        }

        if !break_should_run {
            if is_final_outer_iter {
                finalize_outer_step_status(plan, None);
            }
            return;
        }
        let Some(ref scale) = scale else {
            if is_final_outer_iter {
                finalize_outer_step_status(plan, None);
            }
            return;
        };
        if scale.len() != delta.len() {
            if is_final_outer_iter {
                finalize_outer_step_status(plan, None);
            }
            return;
        }

        let tol_rel = res(plan).outer_tol;
        let tol_abs = res(plan).outer_tol_abs;
        let monitor = res_mut(plan).outer_convergence.take();
        let Some(monitor) = monitor else {
            return;
        };
        match monitor.evaluate_break_from_current_buffers(plan, tol_rel, tol_abs) {
            Ok(converged) => {
                break_converged = Some(converged);
                if converged {
                    plan.repeat_break = true;
                }
            }
            Err(err) => {
                eprintln!("[cfd2][outer] failed to evaluate break status on gpu: {err}");
            }
        }

        res_mut(plan).outer_convergence = Some(monitor);
        if break_converged == Some(true) || is_final_outer_iter {
            finalize_outer_step_status(plan, break_converged);
        }
        return;
    }

    if !break_should_run {
        if is_final_outer_iter {
            finalize_outer_step_status(plan, None);
        }
        return;
    }

    let state = {
        let r = res_mut(plan);
        r.fields.current_state().clone()
    };

    let tol_rel = res(plan).outer_tol;
    let tol_abs = res(plan).outer_tol_abs;
    let monitor = res_mut(plan).outer_convergence.take();
    let Some(mut monitor) = monitor else {
        return;
    };

    match monitor.evaluate_break_from_state_on_gpu(plan, &state, tol_rel, tol_abs) {
        Ok(converged) => {
            break_converged = Some(converged);
            if converged {
                plan.repeat_break = true;
            }
        }
        Err(err) => {
            eprintln!("[cfd2][outer] failed to evaluate break status on gpu: {err}");
        }
    }

    res_mut(plan).outer_convergence = Some(monitor);
    if break_converged == Some(true) || is_final_outer_iter {
        finalize_outer_step_status(plan, break_converged);
    }
}

fn should_track_outer_step_status(plan: &GpuProgramPlan) -> bool {
    res(plan).fields.constants.values().dtau > 0.0
}

fn positivity_field_offsets(registry: &PortRegistry) -> Option<PositivityFieldOffsets> {
    let rho = registry.get_field_entry_by_name("rho")?;
    let p = registry.get_field_entry_by_name("p")?;
    if rho.component_count() != 1 || p.component_count() != 1 {
        return None;
    }
    Some(PositivityFieldOffsets {
        rho: rho.offset() as usize,
        p: p.offset() as usize,
    })
}

fn evaluate_step_positivity(plan: &GpuProgramPlan) -> Option<StepPositivityReport> {
    if plan.model.id != "compressible" || !should_track_outer_step_status(plan) {
        return None;
    }

    let offsets = positivity_field_offsets(&plan.resources.port_registry)?;
    let stride = plan.model.state_layout.stride() as usize;
    let num_cells = plan.num_cells() as usize;
    let bytes = res(plan).fields.state_size_bytes();
    let raw = pollster::block_on(plan.read_state_bytes(bytes));
    let expected_bytes = num_cells.checked_mul(stride)?.checked_mul(4)?;
    if raw.len() < expected_bytes {
        eprintln!(
            "[cfd2][positivity] state readback too short: got {} bytes expected {}",
            raw.len(),
            expected_bytes
        );
        return None;
    }

    let mut report = StepPositivityReport {
        min_rho: f32::INFINITY,
        min_p: f32::INFINITY,
        ..Default::default()
    };

    for cell in 0..num_cells {
        let base = cell * stride;
        let rho_idx = (base + offsets.rho) * 4;
        let p_idx = (base + offsets.p) * 4;
        let rho = f32::from_ne_bytes(raw[rho_idx..rho_idx + 4].try_into().ok()?);
        let p = f32::from_ne_bytes(raw[p_idx..p_idx + 4].try_into().ok()?);

        if rho.is_finite() {
            report.min_rho = report.min_rho.min(rho);
        }
        if p.is_finite() {
            report.min_p = report.min_p.min(p);
        }

        if !rho.is_finite() || rho <= RHO_POSITIVITY_FLOOR {
            report.rho_undershoot_count = report.rho_undershoot_count.saturating_add(1);
        }
        if !p.is_finite() || p <= PRESSURE_POSITIVITY_FLOOR {
            report.pressure_undershoot_count = report.pressure_undershoot_count.saturating_add(1);
        }
    }

    if !report.min_rho.is_finite() {
        report.min_rho = f32::NEG_INFINITY;
    }
    if !report.min_p.is_finite() {
        report.min_p = f32::NEG_INFINITY;
    }

    Some(report)
}

fn should_retry_nonconverged_step(plan: &GpuProgramPlan, converged: bool) -> bool {
    if converged || plan.model.id != "compressible" || !should_track_outer_step_status(plan) {
        return false;
    }

    let r = res(plan);
    r.nonconverged_retry_enabled && plan.step_attempt_index < r.nonconverged_retry_max_attempts
}

fn scaled_outer_targets_converged(plan: &GpuProgramPlan) -> Option<bool> {
    if plan.outer_field_residuals.is_empty() {
        return None;
    }

    let tol_rel = res(plan).outer_tol.max(0.0);
    let tol_abs = res(plan).outer_tol_abs.max(0.0);
    let scaled_by_name: HashMap<&str, f32> = plan
        .outer_field_residuals_scaled
        .iter()
        .map(|(name, value)| (name.as_str(), *value))
        .collect();
    let conserved_targets = ["rho", "rho_u", "rho_e"];
    let use_conserved_targets = conserved_targets
        .iter()
        .all(|target| plan.outer_field_residuals.iter().any(|(name, _)| name == target));

    let mut matched = false;
    for (name, abs_residual) in &plan.outer_field_residuals {
        if use_conserved_targets && !conserved_targets.contains(&name.as_str()) {
            continue;
        }

        matched = true;
        let scaled_residual = scaled_by_name
            .get(name.as_str())
            .copied()
            .unwrap_or(f32::INFINITY);
        if scaled_residual > tol_rel && *abs_residual > tol_abs {
            return Some(false);
        }
    }

    matched.then_some(true)
}

fn finalize_outer_step_status(plan: &mut GpuProgramPlan, fallback_converged: Option<bool>) {
    if !should_track_outer_step_status(plan) {
        return;
    }

    if plan.outer_field_residuals.is_empty() {
        let _ = compute_outer_residuals(plan);
    }

    let converged = scaled_outer_targets_converged(plan)
        .or(fallback_converged)
        .unwrap_or(false);
    plan.outer_step_status = Some(if converged {
        OuterStepStatus::AcceptedConverged
    } else if should_retry_nonconverged_step(plan, converged) {
        OuterStepStatus::RejectedRetry
    } else {
        OuterStepStatus::AcceptedNonconverged
    });
}

fn should_relax_nonconverged_apply(
    dtau: f32,
    last_linear_stats: LinearSolverStats,
    outer_step_status: Option<OuterStepStatus>,
) -> bool {
    if dtau <= 0.0 {
        return !last_linear_stats.converged;
    }

    match outer_step_status {
        Some(OuterStepStatus::AcceptedConverged) => false,
        Some(OuterStepStatus::AcceptedNonconverged | OuterStepStatus::RejectedRetry) => true,
        None => !last_linear_stats.converged,
    }
}

fn next_step_backoff_targets(
    model_id: &str,
    outer_step_status: Option<OuterStepStatus>,
    current_dt: f32,
    current_dtau: f32,
    dt_scale: f32,
    dtau_scale: f32,
) -> Option<(f32, f32)> {
    if model_id != "compressible" || current_dtau <= 0.0 {
        return None;
    }

    match outer_step_status {
        Some(OuterStepStatus::AcceptedNonconverged | OuterStepStatus::RejectedRetry) => {
            let dt_scale = dt_scale.clamp(0.0, 1.0);
            let dtau_scale = dtau_scale.clamp(0.0, 1.0);
            let next_dt = (current_dt * dt_scale).clamp(1e-9, current_dt.max(1e-9));
            let next_dtau = (current_dtau * dtau_scale).clamp(1e-9, current_dtau.max(1e-9));
            if (next_dt - current_dt).abs() > 1e-12 || (next_dtau - current_dtau).abs() > 1e-12 {
                Some((next_dt, next_dtau))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Compute outer-loop correction norms (field residuals) via the
/// `OuterConvergenceMonitor` and populate `plan.outer_field_residuals`,
/// `plan.outer_field_residuals_scaled`, `plan.outer_residual_u`, and
/// `plan.outer_residual_p`.
///
/// Returns `Some((delta, scale))` on success, `None` if no monitor is
/// available or a readback fails.  The monitor is always returned to the
/// resource state regardless of success or failure.
fn compute_outer_residuals(plan: &mut GpuProgramPlan) -> Option<(Vec<f32>, Option<Vec<f32>>)> {
    let state = {
        let r = res_mut(plan);
        r.fields.current_state().clone()
    };
    let monitor = res_mut(plan).outer_convergence.take();
    let mut monitor = monitor?;

    if let Err(err) = monitor.ensure_state_scale(plan, &state) {
        eprintln!("[cfd2][outer] failed to compute state scale: {err}");
    }

    let delta = match monitor.delta_maxima(plan) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("[cfd2][outer] failed to compute correction norms: {err}");
            res_mut(plan).outer_convergence = Some(monitor);
            return None;
        }
    };

    if delta.len() != monitor.target_names().len() {
        eprintln!(
            "[cfd2][outer] correction norm length mismatch: got {} expected {}",
            delta.len(),
            monitor.target_names().len()
        );
        res_mut(plan).outer_convergence = Some(monitor);
        return None;
    }

    if !delta.is_empty() {
        plan.context
            .queue
            .write_buffer(&monitor.b_delta, 0, bytemuck::cast_slice(&delta));
    }

    // Always compute scaled residuals (state scale is populated by ensure_state_scale above).
    let scale: Option<Vec<f32>> = monitor.state_scale().map(|s| s.to_vec());

    plan.outer_field_residuals.clear();
    plan.outer_field_residuals.extend(
        monitor
            .target_names()
            .iter()
            .cloned()
            .zip(delta.iter().copied()),
    );

    plan.outer_field_residuals_scaled.clear();
    if let Some(ref s) = scale {
        if s.len() == delta.len() {
            plan.outer_field_residuals_scaled.extend(
                monitor.target_names().iter().cloned().zip(
                    delta
                        .iter()
                        .zip(s.iter())
                        .map(|(&d, &s_val)| d / s_val.max(1.0)),
                ),
            );
        }
    }

    let mut residual_u: Option<f32> = None;
    let mut residual_p: Option<f32> = None;
    for (name, &v) in monitor.target_names().iter().zip(delta.iter()) {
        if name == "p" {
            residual_p = Some(v);
        } else if name == "u" || name == "U" {
            residual_u = Some(v);
        }
    }
    plan.outer_residual_u = residual_u;
    plan.outer_residual_p = residual_p;

    if std::env::var("CFD2_LOG_OUTER").is_ok() {
        let scaled: Vec<String> = plan
            .outer_field_residuals_scaled
            .iter()
            .map(|(n, r)| format!("{n}={r:.3e}"))
            .collect();
        eprintln!("[outer-resid] iter#{} {}", plan.step_linear_stats.len(), scaled.join(" "));
    }

    res_mut(plan).outer_convergence = Some(monitor);
    Some((delta, scale))
}

fn try_host_coupled_batch_tail_one_submission(plan: &mut GpuProgramPlan, remaining: usize) -> bool {
    if remaining == 0 {
        return false;
    }
    // Plateau-driven models (GUI incompressible defaults + benches) run their
    // outer-exit logic in the break kernel's plateau mode (GPU port of the
    // host detector) so they can use this batched path too.
    let plateau_mode_active = outer_plateau_active(plan);

    let device = plan.context.device.clone();
    let queue = plan.context.queue.clone();
    let context = crate::solver::gpu::context::GpuContext {
        device: device.clone(),
        queue: queue.clone(),
        timestamp_query: plan.context.timestamp_query,
        timestamps_inside_encoders: plan.context.timestamps_inside_encoders,
        timestamp_period_ns: plan.context.timestamp_period_ns,
        pipeline_cache: plan.context.pipeline_cache.clone(),
    };

    let start = std::time::Instant::now();
    // Opt-in phase breakdown of the batched coupled step (setup / FGMRES outer
    // loop / post-processing), to localize the step cost outside the FGMRES solves.
    let profile_phases = std::env::var("CFD2_PROFILE_FGMRES").is_ok();
    let mut setup_ms = 0.0f64;
    let mut loop_ms = 0.0f64;

    let mut encoded_tail_stats = Vec::with_capacity(remaining);
    let mut adaptive_iter_count: Option<u32> = None;
    let iter_counter_buf: Option<std::sync::Arc<wgpu::Buffer>>;
    {
        let r = res_mut(plan);
        let (max_restart, solver_is_cg) = match r.linear_solver.solver_type {
            LinearSolverType::Fgmres { max_restart } => (max_restart.max(1), false),
            LinearSolverType::Cg => (0, true),
        };

        // Split borrows: we need mutable access to the solver (schur or krylov)
        // and immutable access to assembly/update graphs + kernels + runtime dims.
        let assembly_graph = &r.assembly_graph;
        // Outer iterations after the first can use the tail variant (skips the
        // grad_p recompute that grad_p_update already performed — see the
        // `assembly_graph_tail` field doc).
        let assembly_graph_tail = r.assembly_graph_tail.as_ref().unwrap_or(assembly_graph);
        let assembly_graph_frozen = r.assembly_graph_frozen.as_ref();
        let matrix_freeze_period = r.matrix_freeze_period;
        let update_graph = &r.update_graph;
        let kernels = &r.kernels;
        let runtime_dims = r.runtime_dims();

        let system = LinearSystemView {
            ports: r.runtime.linear_ports,
            space: &r.runtime.linear_port_space,
        };
        let n = r.runtime.num_dofs;
        let num_cells = r.runtime.common.num_cells;
        let max_iters = r.linear_solver.max_iters;
        let tol = r.linear_solver.tolerance;
        let tol_abs = r.linear_solver.tolerance_abs;

        let supports_adaptive = r.outer_break_enabled
            && r.outer_gate.is_some()
            && r.outer_convergence.is_some()
            && remaining > 1;

        let use_adaptive = supports_adaptive;

        let adaptive_resources = if use_adaptive {
            let gate = r.outer_gate.as_ref()
                .expect("outer_gate must be Some when use_adaptive is true (checked above)");
            let monitor = r.outer_convergence.as_ref()
                .expect("outer_convergence must be Some when use_adaptive is true (checked above)");

            // The indirect assembly graph is only ever encoded for iter_idx > 0
            // (the direct graph covers the first iteration), so it derives from
            // the TAIL variant.
            let indirect_cells = gate.b_indirect_args_cells.clone();
            let indirect_faces = gate.b_indirect_args_faces.clone();
            let assembly_graph_indirect =
                assembly_graph_tail.clone_with_indirect_dispatch(|kind| match kind {
                    DispatchKind::Faces => (indirect_faces.clone(), 0),
                    _ => (indirect_cells.clone(), 0),
                });
            let update_graph_indirect =
                update_graph.clone_with_indirect_dispatch(|kind| match kind {
                    DispatchKind::Faces => (indirect_faces.clone(), 0),
                    _ => (indirect_cells.clone(), 0),
                });

            let state = r.fields.current_state();
            let bg_state = monitor.create_state_bind_group(&device, state);

            // Upload break params (plateau mode for plateau-driven models:
            // same band/floors as the host detector, evaluated on-device).
            if plateau_mode_active {
                monitor.upload_break_params_plateau(
                    &queue,
                    r.outer_tol,
                    r.outer_tol_abs,
                    OUTER_PLATEAU_FACTOR,
                    OUTER_PLATEAU_CEILING,
                    OUTER_TOL_EXIT_MIN_ITERS as u32,
                    OUTER_PLATEAU_MIN_ITERS as u32,
                );
            } else {
                monitor.upload_break_params(&queue, r.outer_tol, r.outer_tol_abs);
            }

            let zero: u32 = 0;
            queue.write_buffer(&gate.b_iter_counter, 0, bytemuck::bytes_of(&zero));

            let stop_inject_bg = if solver_is_cg {
                let b_scalars = r.runtime.scalar_cg.scalars();
                gate.create_stop_inject_bind_group_cg(&device, &monitor.b_break_status, b_scalars)
            } else {
                let b_scalars = if let Some(schur) = &r.schur {
                    schur.solver.fgmres.scalars_buffer()
                } else if let Some(krylov) = &r.krylov {
                    krylov.solver.fgmres.scalars_buffer()
                } else {
                    return false;
                };
                gate.create_stop_inject_bind_group(&device, &monitor.b_break_status, b_scalars)
            };

            Some((
                assembly_graph_indirect,
                update_graph_indirect,
                bg_state,
                stop_inject_bg,
            ))
        } else {
            None
        };

        if profile_phases {
            setup_ms = start.elapsed().as_secs_f64() * 1e3;
        }

        // Use chunked submission to avoid Metal hangs.  Each FGMRES restart
        // chunk gets its own encoder → submit cycle.  Assembly is prepended to
        // the first chunk and update is appended to the last chunk of each
        // outer iteration.
        for iter_idx in 0..remaining {
            let is_indirect = use_adaptive && iter_idx > 0;
            // Tolerances are fixed at encode time on this path, so the full-EW
            // forcing uses a static geometric continuation 1e-2 -> base across
            // the encoded outers (first outer = the EW-lite 1e-2, last outer =
            // the model tolerance, log-linear in between). `CFD2_NO_EW_FULL=1`
            // restores first-outer-only.
            let tol_iter = if iter_idx == 0 || remaining <= 1 {
                first_outer_tolerance(tol, iter_idx == 0, remaining > 1)
            } else if std::env::var("CFD2_NO_EW_FULL").is_ok_and(|v| v == "1") {
                tol
            } else {
                let hi = tol.max(1e-2);
                let frac = iter_idx as f32 / (remaining - 1) as f32;
                (hi * (tol / hi).powf(frac)).clamp(tol, hi)
            };

            let mut pre = |encoder: &mut wgpu::CommandEncoder| {
                if let Some((ref asm_indirect, _, _, ref stop_bg)) = adaptive_resources {
                    if is_indirect {
                        // Inject STOP scalar so the linear solver becomes zero-cost when converged
                        let gate = r.outer_gate.as_ref()
                            .expect("outer_gate must be Some in adaptive path");
                        if solver_is_cg {
                            gate.encode_stop_inject_cg_into(encoder, stop_bg);
                        } else {
                            gate.encode_stop_inject_into(encoder, stop_bg);
                        }
                        asm_indirect.encode_into(encoder, kernels, runtime_dims);
                        return;
                    }
                }
                if iter_idx == 0 {
                    assembly_graph.encode_into(encoder, kernels, runtime_dims);
                } else if matrix_freeze_period > 0
                    && iter_idx > 1
                    && iter_idx % matrix_freeze_period as usize != 0
                    && assembly_graph_frozen.is_some()
                {
                    assembly_graph_frozen
                        .expect("checked is_some")
                        .encode_into(encoder, kernels, runtime_dims);
                } else {
                    assembly_graph_tail.encode_into(encoder, kernels, runtime_dims);
                }
            };
            let mut post = |encoder: &mut wgpu::CommandEncoder| {
                if let Some((_, ref upd_indirect, ref bg_state, _)) = adaptive_resources {
                    if is_indirect {
                        upd_indirect.encode_into(encoder, kernels, runtime_dims);
                    } else {
                        update_graph.encode_into(encoder, kernels, runtime_dims);
                    }
                    // Convergence check + gate after every iteration in adaptive mode
                    let monitor = r.outer_convergence.as_ref()
                        .expect("outer_convergence must be Some in adaptive path");
                    let gate = r.outer_gate.as_ref()
                        .expect("outer_gate must be Some in adaptive path");
                    let first_iter = iter_idx == 0;
                    monitor.encode_convergence_check(encoder, bg_state, first_iter);
                    gate.encode_gate_into(encoder);
                } else {
                    update_graph.encode_into(encoder, kernels, runtime_dims);
                }
            };

            if let Some(schur) = &mut r.schur {
                let stats = submit_solve_fgmres_fixed_iterations_chunked(
                    &mut schur.solver,
                    SolveFgmresArgs {
                        context: &context,
                        system,
                        n,
                        num_cells,
                        dispatch: schur.dispatch,
                        max_restart,
                        max_iters,
                        tol: tol_iter,
                        tol_abs,
                        precond_label: "generic_coupled:schur(batch_tail)",
                        use_encoded_seed_basis0: true,
                        // Plateau-routed models keep the tight budget policy;
                        // the long-batched models (pseudo-transient nozzle etc.)
                        // keep the validated floor.
                        tight_budget: plateau_mode_active,
                    },
                    &mut pre,
                    &mut post,
                );
                encoded_tail_stats.push(stats);
            } else if let Some(krylov) = &mut r.krylov {
                let stats = submit_solve_fgmres_fixed_iterations_chunked(
                    &mut krylov.solver,
                    SolveFgmresArgs {
                        context: &context,
                        system,
                        n,
                        num_cells,
                        dispatch: krylov.dispatch,
                        max_restart,
                        max_iters,
                        tol: tol_iter,
                        tol_abs,
                        precond_label: "generic_coupled:fgmres(batch_tail)",
                        use_encoded_seed_basis0: true,
                        tight_budget: plateau_mode_active,
                    },
                    &mut pre,
                    &mut post,
                );
                encoded_tail_stats.push(stats);
            } else if solver_is_cg {
                let stats = submit_solve_cg_fixed_iterations_chunked(
                    &r.runtime.scalar_cg,
                    &context,
                    n,
                    max_iters,
                    &mut pre,
                    &mut post,
                );
                encoded_tail_stats.push(stats);
            } else {
                return false;
            }
        }

        if profile_phases {
            loop_ms = start.elapsed().as_secs_f64() * 1e3;
        }

        // Clone the iter counter buffer so we can read it back after dropping `r`.
        iter_counter_buf = if use_adaptive {
            r.outer_gate.as_ref().map(|g| g.b_iter_counter.clone())
        } else {
            None
        };
    }

    // Read back adaptive iteration counter OUTSIDE the `r` borrow scope
    // so we can access `plan.staging_cache`.
    if let Some(b_iter_counter) = iter_counter_buf {
        let staging =
            plan.staging_cache
                .take_or_create(&device, 4, "outer_gate:iter_counter_readback");
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("outer_gate:counter_readback"),
        });
        encoder.copy_buffer_to_buffer(&b_iter_counter, 0, &staging, 0, 4);
        let sub_idx = queue.submit(Some(encoder.finish()));
        crate::count_submission!("Generic Coupled", "outer_gate:counter_readback");

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: Some(sub_idx),
            timeout: None,
        });
        if let Ok(Ok(())) = rx.recv() {
            let data = slice.get_mapped_range();
            let words: &[u32] = bytemuck::cast_slice(&data);
            // iter_counter counts the number of times the gate
            // dispatched (i.e. not-converged iterations). This includes
            // iteration 0 (always runs) plus any unconverged indirect
            // iterations.
            let gpu_iters = words.first().copied().unwrap_or(remaining as u32);
            adaptive_iter_count = Some(gpu_iters);
            drop(data);
            staging.unmap();
        }
        plan.staging_cache.put(4, staging);
    }

    if let Some(last) = encoded_tail_stats.last().copied() {
        plan.last_linear_stats = last;
    }
    plan.step_linear_stats.extend(encoded_tail_stats);

    if let Some(gpu_iters) = adaptive_iter_count {
        // Use the GPU-reported iteration count instead of the encoded count
        plan.outer_iterations = gpu_iters;
    } else {
        plan.outer_iterations = plan.step_linear_stats.len() as u32;
    }

    // Post-submission: compute outer-loop correction norms so that
    // outer_field_residuals, outer_residual_u and outer_residual_p are
    // populated even when the one-submission path is used.  This adds a
    // small number of GPU dispatches (two reduction kernels + readback)
    // once at the end of the step — the per-iteration savings from the
    // encoded path are preserved.
    compute_outer_residuals(plan);
    if profile_phases {
        let total_ms = start.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "[coupled-phases] setup={:.2}ms fgmres_loop={:.2}ms post={:.2}ms total={:.2}ms (outer_iters={remaining})",
            setup_ms,
            loop_ms - setup_ms,
            total_ms - loop_ms,
            total_ms,
        );
    }
    let adaptive_break_converged = adaptive_iter_count.map(|iters| iters < remaining as u32);
    finalize_outer_step_status(plan, adaptive_break_converged);

    if plan.collect_trace {
        plan.step_graph_timings
            .push(crate::solver::gpu::program::plan::StepGraphTiming {
                label: "coupled:one_submission_chunked",
                seconds: start.elapsed().as_secs_f64(),
                detail: None,
            });
    }

    plan.repeat_break = true;
    true
}

pub(crate) fn host_implicit_set_alpha_for_apply(plan: &mut GpuProgramPlan) {
    let queue = plan.context.queue.clone();
    let should_relax = should_relax_nonconverged_apply(
        res(plan).fields.constants.values().dtau,
        plan.last_linear_stats,
        plan.outer_step_status,
    );
    let r = res_mut(plan);

    let base_alpha_u = r.fields.constants.values().alpha_u;
    r.implicit_base_alpha_u = Some(base_alpha_u);

    if !should_relax {
        return;
    }

    let apply_alpha_u = base_alpha_u * r.nonconverged_relax.clamp(0.0, 1.0);
    if (apply_alpha_u - base_alpha_u).abs() > 1e-6 {
        {
            let values = r.fields.constants.values_mut();
            values.alpha_u = apply_alpha_u;
        }
        r.fields.constants.write(&queue);
    }
}

pub(crate) fn host_implicit_restore_alpha(plan: &mut GpuProgramPlan) {
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    let Some(base_alpha_u) = r.implicit_base_alpha_u.take() else {
        return;
    };

    let current_alpha_u = r.fields.constants.values().alpha_u;
    if (current_alpha_u - base_alpha_u).abs() > 1e-6 {
        {
            let values = r.fields.constants.values_mut();
            values.alpha_u = base_alpha_u;
        }
        r.fields.constants.write(&queue);
    }
}

pub(crate) fn assembly_graph_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let r = res(plan);
    // Outer iterations after the first (one linear-stats entry per completed
    // solve this step) use the tail variant, which skips the grad_p recompute
    // that the Update phase's grad_p_update already performed — or, under
    // matrix freezing, the RHS-only variant on non-re-linearization outers.
    // Out-of-step callers (debug assembly, parity harnesses) see empty stats
    // -> full graph.
    let iters_done = plan.step_linear_stats.len();
    let graph = if iters_done == 0 {
        &r.assembly_graph
    } else if r.matrix_freeze_period > 0
        && iters_done > 1
        && iters_done % r.matrix_freeze_period as usize != 0
        && r.assembly_graph_frozen.is_some()
    {
        r.assembly_graph_frozen.as_ref().expect("checked is_some")
    } else {
        r.assembly_graph_tail.as_ref().unwrap_or(&r.assembly_graph)
    };
    run_module_graph(graph, context, &r.kernels, r.runtime_dims(), mode)
}

pub(crate) fn init_prepare_graph_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let r = res(plan);
    if !r.dp_init_enabled {
        return (0.0, None);
    }
    if !r.recurring_prepare_enabled && !r.dp_init_needed.load(Ordering::Relaxed) {
        return (0.0, None);
    }
    run_module_graph(
        &r.init_prepare_graph,
        context,
        &r.kernels,
        r.runtime_dims(),
        mode,
    )
}

pub(crate) fn clear_dp_init_needed(plan: &mut GpuProgramPlan) {
    let r = res_mut(plan);
    if r.dp_init_enabled {
        r.dp_init_needed.store(false, Ordering::Relaxed);
    }
}

/// Runs the Preparation-phase graph once per outer iteration when
/// `recurring_prepare_enabled` is true (i.e. the model declares
/// expression-valued boundary conditions, lowered to the generic
/// `bc_expr_update` kernel).  This keeps boundary-table values synchronized
/// with the evolving interior state within the outer corrector loop.
pub(crate) fn iter_prepare_graph_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let r = res(plan);
    if !r.recurring_prepare_enabled {
        return (0.0, None);
    }
    run_module_graph(
        &r.init_prepare_graph,
        context,
        &r.kernels,
        r.runtime_dims(),
        mode,
    )
}

pub(crate) fn host_coupled_before_iter(plan: &mut GpuProgramPlan) {
    // After the first iteration begins, we can stop running one-time preparation
    // kernels (e.g. `dp_init`) on subsequent steps unless a parameter change
    // re-enables them.
    clear_dp_init_needed(plan);

    // If batched mode is enabled, attempt to run the full outer
    // loop here in one encoded submission and skip the remaining per-iteration
    // nodes in this repeat-body execution.
    let (outer_batched_mode, outer_iters) = {
        let r = res(plan);
        (r.outer_batched_mode, r.outer_iters.max(1))
    };
    // Plateau-driven models can run batched via the break kernel's plateau
    // mode (`CFD2_GPU_PLATEAU=0` restores the host detector on the non-batched
    // per-iteration loop, which computes per-iter residuals host-side and skips
    // the remaining UNENCODED sweeps). `CFD2_NO_BATCH` forces non-batched.
    let outer_batched_mode = outer_batched_mode
        && (!outer_plateau_active(plan) || gpu_plateau_enabled())
        && std::env::var("CFD2_NO_BATCH").is_err();
    if !outer_batched_mode || outer_iters <= 1 {
        return;
    }
    if !plan.step_linear_stats.is_empty() {
        // Only the first outer-loop iteration can consume the full batch.
        return;
    }

    if try_host_coupled_batch_tail_one_submission(plan, outer_iters) {
        plan.skip_remaining_block = true;
    }
}

fn encoded_seed_basis0_enabled(default_enabled: bool) -> bool {
    std::env::var("CFD2_ENABLE_ENCODED_SEED_BASIS0")
        .map(|v| v != "0")
        .unwrap_or(default_enabled)
}

/// On-device plateau detection (the break kernel's plateau mode), letting
/// plateau-driven models use the batched one-submission outer path.
/// `CFD2_GPU_PLATEAU=1` opts in; DEFAULT OFF because the per-outer host route
/// (chunked one-submission solves with the tight budget) beats it: the batched
/// tail must encode ALL `outer_iters` up front while the plateau typically
/// exits early, and STOP-frozen iterations still cost their encoding and no-op
/// dispatches. The host detector also keeps per-iteration residuals for the
/// GUI. The break kernel's plateau mode replicates the host detector exactly
/// (tolerance + stall exits, same band/floors).
fn gpu_plateau_enabled() -> bool {
    std::env::var("CFD2_GPU_PLATEAU").is_ok_and(|v| v == "1")
}

pub(crate) fn update_graph_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let r = res(plan);
    run_module_graph(&r.update_graph, context, &r.kernels, r.runtime_dims(), mode)
}

pub(crate) fn explicit_graph_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let r = res(plan);
    run_module_graph(
        &r.explicit_graph,
        context,
        &r.kernels,
        r.runtime_dims(),
        mode,
    )
}

pub(crate) fn apply_graph_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let r = res(plan);
    run_module_graph(&r.apply_graph, context, &r.kernels, r.runtime_dims(), mode)
}

pub(crate) fn implicit_snapshot_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    _mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let r = res(plan);
    if plan.repeat_break {
        return (0.0, None);
    }
    if r.fields.constants.values().dtau <= 0.0 {
        return (0.0, None);
    }
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("generic_coupled:implicit_snapshot"),
        });
    r.fields.snapshot_for_iteration(&mut encoder);
    context.queue.submit(Some(encoder.finish()));
    crate::count_submission!("Generic Coupled", "implicit_snapshot");
    (0.0, None)
}

pub(crate) fn count_outer_iters(plan: &GpuProgramPlan) -> usize {
    res(plan).outer_iters.max(1)
}

pub(crate) fn param_outer_iters(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::Usize(iters) = value else {
        return Err("OuterIters expects Usize".to_string());
    };
    res_mut(plan).outer_iters = iters.max(1);
    Ok(())
}

pub(crate) fn param_outer_tol(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::F32(tol) = value else {
        return Err("OuterTol expects F32".to_string());
    };
    res_mut(plan).outer_tol = tol.max(0.0);
    Ok(())
}

pub(crate) fn param_outer_tol_abs(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::F32(tol_abs) = value else {
        return Err("OuterTolAbs expects F32".to_string());
    };
    res_mut(plan).outer_tol_abs = tol_abs.max(0.0);
    Ok(())
}

pub(crate) fn param_outer_fixed_iterations_mode(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::Bool(enabled) = value else {
        return Err("OuterFixedIterationsMode expects Bool".to_string());
    };
    // API is framed positively for callers: `true` means fixed-iteration mode.
    res_mut(plan).outer_break_enabled = !enabled;
    Ok(())
}

pub(crate) fn param_outer_batched_mode(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::Bool(enabled) = value else {
        return Err("OuterBatchedMode expects Bool".to_string());
    };
    res_mut(plan).outer_batched_mode = enabled;
    Ok(())
}

pub(crate) fn param_nonconverged_relax(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::F32(alpha) = value else {
        return Err("NonconvergedRelax expects F32".to_string());
    };
    res_mut(plan).nonconverged_relax = alpha.clamp(0.0, 1.0);
    Ok(())
}

pub(crate) fn param_nonconverged_dt_scale(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::F32(scale) = value else {
        return Err("NonconvergedDtScale expects F32".to_string());
    };
    res_mut(plan).nonconverged_dt_scale = scale.clamp(0.0, 1.0);
    Ok(())
}

pub(crate) fn param_nonconverged_dtau_scale(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::F32(scale) = value else {
        return Err("NonconvergedDtauScale expects F32".to_string());
    };
    res_mut(plan).nonconverged_dtau_scale = scale.clamp(0.0, 1.0);
    Ok(())
}

pub(crate) fn param_nonconverged_retry_enabled(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::Bool(enabled) = value else {
        return Err("NonconvergedRetryEnabled expects Bool".to_string());
    };
    res_mut(plan).nonconverged_retry_enabled = enabled;
    Ok(())
}

pub(crate) fn param_nonconverged_retry_max_attempts(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::Usize(attempts) = value else {
        return Err("NonconvergedRetryMaxAttempts expects Usize".to_string());
    };
    res_mut(plan).nonconverged_retry_max_attempts = attempts;
    Ok(())
}

pub(crate) fn param_dt(plan: &mut GpuProgramPlan, value: PlanParamValue) -> Result<(), String> {
    let PlanParamValue::F32(dt) = value else {
        return Err("invalid value type".into());
    };
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    r.time_integration
        .set_dt(dt, &mut r.fields.constants, &queue);
    if r.dp_init_enabled {
        r.dp_init_needed.store(true, Ordering::Relaxed);
    }
    Ok(())
}

pub(crate) fn param_dtau(plan: &mut GpuProgramPlan, value: PlanParamValue) -> Result<(), String> {
    let PlanParamValue::F32(dtau) = value else {
        return Err("invalid value type".into());
    };
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    {
        let values = r.fields.constants.values_mut();
        values.dtau = dtau;
    }
    r.fields.constants.write(&queue);
    Ok(())
}

pub(crate) fn param_advection_scheme(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::Scheme(scheme) = value else {
        return Err("invalid value type".into());
    };
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    {
        let values = r.fields.constants.values_mut();
        values.scheme = scheme.gpu_id();
    }
    r.fields.constants.write(&queue);
    Ok(())
}

pub(crate) fn param_time_scheme(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::TimeScheme(scheme) = value else {
        return Err("invalid value type".into());
    };
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    r.requested_time_scheme = scheme;
    {
        let values = r.fields.constants.values_mut();
        values.time_scheme = scheme as u32;
    }
    r.fields.constants.write(&queue);
    Ok(())
}

pub(crate) fn param_viscosity(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::F32(mu) = value else {
        return Err("invalid value type".into());
    };
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    {
        let values = r.fields.constants.values_mut();
        values.viscosity = mu;
    }
    r.fields.constants.write(&queue);
    Ok(())
}

pub(crate) fn param_density(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::F32(rho) = value else {
        return Err("invalid value type".into());
    };
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    {
        let values = r.fields.constants.values_mut();
        values.density = rho;
    }
    r.fields.constants.write(&queue);
    if r.dp_init_enabled {
        r.dp_init_needed.store(true, Ordering::Relaxed);
    }
    Ok(())
}

pub(crate) fn param_eos_gamma(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::F32(gamma) = value else {
        return Err("invalid value type".into());
    };
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    {
        let values = r.fields.constants.values_mut();
        values.eos_gamma = gamma;
    }
    r.fields.constants.write(&queue);
    Ok(())
}

pub(crate) fn param_eos_gm1(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::F32(gm1) = value else {
        return Err("invalid value type".into());
    };
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    {
        let values = r.fields.constants.values_mut();
        values.eos_gm1 = gm1;
    }
    r.fields.constants.write(&queue);
    Ok(())
}

pub(crate) fn param_eos_r(plan: &mut GpuProgramPlan, value: PlanParamValue) -> Result<(), String> {
    let PlanParamValue::F32(r_gas) = value else {
        return Err("invalid value type".into());
    };
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    {
        let values = r.fields.constants.values_mut();
        values.eos_r = r_gas;
    }
    r.fields.constants.write(&queue);
    Ok(())
}

pub(crate) fn param_eos_dp_drho(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::F32(dp_drho) = value else {
        return Err("invalid value type".into());
    };
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    {
        let values = r.fields.constants.values_mut();
        values.eos_dp_drho = dp_drho;
    }
    r.fields.constants.write(&queue);
    Ok(())
}

pub(crate) fn param_eos_p_offset(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::F32(p_offset) = value else {
        return Err("invalid value type".into());
    };
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    {
        let values = r.fields.constants.values_mut();
        values.eos_p_offset = p_offset;
    }
    r.fields.constants.write(&queue);
    Ok(())
}

pub(crate) fn param_eos_theta_ref(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::F32(theta) = value else {
        return Err("invalid value type".into());
    };
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    {
        let values = r.fields.constants.values_mut();
        values.eos_theta_ref = theta;
    }
    r.fields.constants.write(&queue);
    Ok(())
}

pub(crate) fn param_buoyant_beta_g(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::F32(beta_g) = value else {
        return Err("invalid value type".into());
    };
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    {
        let values = r.fields.constants.values_mut();
        values.buoyant_beta_g = beta_g;
    }
    r.fields.constants.write(&queue);
    Ok(())
}

pub(crate) fn param_buoyant_t0(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::F32(t0) = value else {
        return Err("invalid value type".into());
    };
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    {
        let values = r.fields.constants.values_mut();
        values.buoyant_t0 = t0;
    }
    r.fields.constants.write(&queue);
    Ok(())
}

pub(crate) fn param_buoyant_k_over_cp(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::F32(k_over_cp) = value else {
        return Err("invalid value type".into());
    };
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    {
        let values = r.fields.constants.values_mut();
        values.buoyant_k_over_cp = k_over_cp;
    }
    r.fields.constants.write(&queue);
    Ok(())
}

pub(crate) fn param_alpha_u(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::F32(alpha) = value else {
        return Err("invalid value type".into());
    };
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    {
        let values = r.fields.constants.values_mut();
        values.alpha_u = alpha;
    }
    r.fields.constants.write(&queue);
    Ok(())
}

pub(crate) fn param_alpha_p(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::F32(alpha) = value else {
        return Err("invalid value type".into());
    };
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    {
        let values = r.fields.constants.values_mut();
        values.alpha_p = alpha;
    }
    r.fields.constants.write(&queue);
    Ok(())
}

pub(crate) fn param_preconditioner(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::Preconditioner(preconditioner) = value else {
        return Err("invalid value type".into());
    };

    let r = res_mut(plan);
    r.linear_solver.preconditioner = preconditioner;

    if let Some(schur) = &mut r.schur {
        schur
            .solver
            .precond
            .set_pressure_kind(CoupledPressureSolveKind::from_config(preconditioner));
    } else if let Some(krylov) = &mut r.krylov {
        krylov.solver.precond.set_kind(preconditioner);
    }
    Ok(())
}

pub(crate) fn param_linear_solver_max_restart(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::Usize(max_restart) = value else {
        return Err("linear_solver.max_restart expects Usize".to_string());
    };

    let r = res_mut(plan);
    let max_restart = max_restart.max(1);

    let capacity = if let Some(schur) = &r.schur {
        schur.solver.fgmres.max_restart()
    } else if let Some(krylov) = &r.krylov {
        krylov.solver.fgmres.max_restart()
    } else {
        return Err("linear_solver.max_restart requires an FGMRES workspace".to_string());
    };

    let LinearSolverType::Fgmres { .. } = r.linear_solver.solver_type else {
        return Err("linear_solver.max_restart requires LinearSolverType::Fgmres".to_string());
    };
    r.linear_solver.solver_type = LinearSolverType::Fgmres {
        max_restart: max_restart.min(capacity).max(1),
    };
    Ok(())
}

pub(crate) fn param_linear_solver_max_iters(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::U32(max_iters) = value else {
        return Err("linear_solver.max_iters expects U32".to_string());
    };

    res_mut(plan).linear_solver.max_iters = max_iters.max(1);
    Ok(())
}

pub(crate) fn param_linear_solver_tolerance(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::F32(tol) = value else {
        return Err("linear_solver.tolerance expects F32".to_string());
    };

    res_mut(plan).linear_solver.tolerance = tol.max(0.0);
    Ok(())
}

pub(crate) fn param_linear_solver_tolerance_abs(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::F32(tol_abs) = value else {
        return Err("linear_solver.tolerance_abs expects F32".to_string());
    };

    res_mut(plan).linear_solver.tolerance_abs = tol_abs.max(0.0);
    Ok(())
}

pub(crate) fn param_linear_solver_solution_update_strategy(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::FgmresSolutionUpdateStrategy(strategy) = value else {
        return Err(
            "linear_solver.solution_update_strategy expects FgmresSolutionUpdateStrategy"
                .to_string(),
        );
    };

    let r = res_mut(plan);
    r.linear_solver.update_strategy = strategy;

    if let Some(schur) = &mut r.schur {
        schur.solver.fgmres.set_solution_update_strategy(strategy);
    } else if let Some(krylov) = &mut r.krylov {
        krylov.solver.fgmres.set_solution_update_strategy(strategy);
    }

    Ok(())
}

pub(crate) fn param_detailed_profiling(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::Bool(enable) = value else {
        return Err("invalid value type".into());
    };
    if enable {
        plan.profiling_stats.enable();
    } else {
        plan.profiling_stats.disable();
    }
    Ok(())
}

pub(crate) fn param_low_mach_model(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::LowMachModel(model) = value else {
        return Err("invalid value type".into());
    };
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    let Some(_) = r.fields.low_mach_params_buffer() else {
        return Err("model does not allocate low-mach params".to_string());
    };
    r.fields.low_mach_params_mut().model = model as u32;
    r.fields.update_low_mach_params(&queue);
    Ok(())
}

pub(crate) fn param_low_mach_theta_floor(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::F32(theta) = value else {
        return Err("invalid value type".into());
    };
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    let Some(_) = r.fields.low_mach_params_buffer() else {
        return Err("model does not allocate low-mach params".to_string());
    };
    r.fields.low_mach_params_mut().theta_floor = theta;
    r.fields.update_low_mach_params(&queue);
    Ok(())
}

pub(crate) fn param_low_mach_pressure_coupling_alpha(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::F32(alpha) = value else {
        return Err("invalid value type".into());
    };
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    let Some(_) = r.fields.low_mach_params_buffer() else {
        return Err("model does not allocate low-mach params".to_string());
    };
    r.fields.low_mach_params_mut().pressure_coupling_alpha = alpha.max(0.0);
    r.fields.update_low_mach_params(&queue);
    Ok(())
}

pub(crate) fn param_low_mach_eps4(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::F32(eps4) = value else {
        return Err("invalid value type".into());
    };
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    let Some(_) = r.fields.low_mach_params_buffer() else {
        return Err("model does not allocate low-mach params".to_string());
    };
    r.fields.low_mach_params_mut().eps4 = eps4.max(0.0);
    r.fields.update_low_mach_params(&queue);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::gpu::structs::LinearSolverStats;
    use crate::solver::dimensions::{Pressure, UnitDimension, Velocity};
    use crate::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType};
    use crate::solver::model::backend::ast::{fvm, vol_scalar, vol_vector3, EquationSystem};
    use crate::solver::model::ports::PortRegistry;
    use crate::solver::model::{eos, primitives};
    use crate::solver::model::{
        incompressible_momentum_model, BoundarySpec, ModelLinearSolverSpec,
        ModelPreconditionerSpec, SchurBlockLayout,
    };

    /// Helper to create a PortRegistry with all state fields registered.
    fn create_port_registry_for_test(model: &ModelSpec) -> PortRegistry {
        let mut registry = PortRegistry::new(model.state_layout.clone());
        for field in model.state_layout.fields() {
            registry
                .register_state_field(field.name())
                .expect("Failed to register field");
        }
        registry
    }

    #[test]
    fn schur_rejects_invalid_layout_indices() {
        let mut model = incompressible_momentum_model().expect("model");
        let Some(spec) = &mut model.linear_solver else {
            panic!("missing linear_solver spec");
        };
        let ModelPreconditionerSpec::Schur { layout, .. } = &mut spec.preconditioner else {
            panic!("expected Schur preconditioner");
        };
        // Distinct/in-range, but doesn't cover the full set of equation target indices.
        *layout = SchurBlockLayout::from_u_p(&[0], 2).expect("layout build failed");

        // Test runtime path: create PortRegistry with all fields registered
        let registry = create_port_registry_for_test(&model);
        let unknown_mapping =
            resolve_unknown_mapping_runtime(&model, &registry).expect("runtime mapping failed");
        let err = validate_schur_model(&model, &unknown_mapping).unwrap_err();
        assert!(err.contains("equation targets"), "unexpected error: {err}");
    }

    #[test]
    fn schur_accepts_vector3_velocity_layout() {
        let u = vol_vector3("U", Velocity::UNIT);
        let p = vol_scalar("p", Pressure::UNIT);

        let mut system = EquationSystem::new();
        system.add_equation(fvm::ddt(u).eqn(u));
        system.add_equation(fvm::ddt(p).eqn(p));

        let layout = crate::solver::model::backend::StateLayout::new(vec![u, p]);
        assert_eq!(layout.stride(), 4);
        assert_eq!(system.unknowns_per_cell(), 4);

        let model = crate::solver::model::ModelSpec {
            id: "schur_vector3_test",
            system,
            state_layout: layout,
            boundaries: BoundarySpec::default(),

            modules: vec![
                crate::solver::model::modules::eos::eos_module(eos::EosSpec::Constant),
                crate::solver::model::modules::generic_coupled::generic_coupled_module(
                    crate::solver::model::method::MethodSpec::Coupled(
                        crate::solver::model::method::CoupledCapabilities::default(),
                    ),
                ),
            ],
            linear_solver: Some(ModelLinearSolverSpec {
                preconditioner: ModelPreconditionerSpec::Schur {
                    omega: 1.0,
                    sweeps_cap: 64,
                    layout: SchurBlockLayout::from_u_p(&[0, 1, 2], 3).expect("layout build failed"),
                },
                ..Default::default()
            }),
            primitives: primitives::PrimitiveDerivations::default(),
        };

        // Test runtime path: create PortRegistry with all fields registered
        let registry = create_port_registry_for_test(&model);
        let unknown_mapping =
            resolve_unknown_mapping_runtime(&model, &registry).expect("runtime mapping failed");
        validate_schur_model(&model, &unknown_mapping)
            .expect("Vector3 velocity Schur layout should validate");
    }

    #[test]
    fn runtime_mapping_fails_when_field_not_registered() {
        // Create a model with fields
        let u = vol_vector3("U", Velocity::UNIT);
        let p = vol_scalar("p", Pressure::UNIT);

        let mut system = EquationSystem::new();
        system.add_equation(fvm::ddt(u).eqn(u));
        system.add_equation(fvm::ddt(p).eqn(p));

        let layout = crate::solver::model::backend::StateLayout::new(vec![u, p]);
        let model = crate::solver::model::ModelSpec {
            id: "test_missing_field",
            system,
            state_layout: layout,
            boundaries: BoundarySpec::default(),
            modules: vec![],
            linear_solver: None,
            primitives: primitives::PrimitiveDerivations::default(),
        };

        // Create a PortRegistry WITHOUT registering the fields
        let empty_registry = PortRegistry::new(model.state_layout.clone());

        // Runtime mapping should fail because fields are not registered
        let err = resolve_unknown_mapping_runtime(&model, &empty_registry).unwrap_err();
        assert!(
            err.contains("Field 'U' not found in port registry"),
            "Expected error about missing field 'U', got: {}",
            err
        );
    }

    #[test]
    fn runtime_mapping_succeeds_with_all_fields_registered() {
        // Create a model with scalar and vector fields
        let u = vol_vector3("U", Velocity::UNIT);
        let p = vol_scalar("p", Pressure::UNIT);

        let mut system = EquationSystem::new();
        system.add_equation(fvm::ddt(u).eqn(u));
        system.add_equation(fvm::ddt(p).eqn(p));

        let layout = crate::solver::model::backend::StateLayout::new(vec![u, p]);
        let model = crate::solver::model::ModelSpec {
            id: "test_all_fields",
            system,
            state_layout: layout.clone(),
            boundaries: BoundarySpec::default(),
            modules: vec![],
            linear_solver: None,
            primitives: primitives::PrimitiveDerivations::default(),
        };

        // Create a PortRegistry and register all fields
        let registry = create_port_registry_for_test(&model);

        // Runtime mapping should succeed
        let mapping = resolve_unknown_mapping_runtime(&model, &registry)
            .expect("Runtime mapping should succeed with registered fields");

        // Verify the mapping is correct
        assert_eq!(mapping.num_equations, 2);
        assert_eq!(mapping.max_components, 3);

        // Verify offsets match StateLayout
        // U is at offset 0, p is at offset 3 (after U's 3 components)
        assert_eq!(mapping.get_offset(0, 0), Some(0)); // U x
        assert_eq!(mapping.get_offset(0, 1), Some(1)); // U y
        assert_eq!(mapping.get_offset(0, 2), Some(2)); // U z
        assert_eq!(mapping.get_offset(1, 0), Some(3)); // p
        assert_eq!(mapping.get_offset(1, 1), None);
    }

    #[test]
    fn before_iter_does_not_skip_block_when_one_submission_fails() {
        let mesh = generate_structured_rect_mesh(
            6,
            4,
            1.0,
            0.4,
            BoundarySides {
                left: BoundaryType::Inlet,
                right: BoundaryType::Outlet,
                bottom: BoundaryType::Wall,
                top: BoundaryType::Wall,
            },
        );
        let model = incompressible_momentum_model().expect("model");

        let mut plan = pollster::block_on(crate::solver::gpu::lowering::lower_program_plan(
            &mesh,
            &model,
            crate::solver::gpu::program::plan_instance::PlanInitConfig {
                advection_scheme: crate::solver::scheme::Scheme::Upwind,
                time_scheme: crate::solver::gpu::enums::TimeScheme::BDF2,
                preconditioner: crate::solver::gpu::structs::PreconditionerType::Jacobi,
                stepping: crate::solver::gpu::recipe::SteppingMode::Coupled,
            },
            None,
            None,
        ))
        .expect("build generic coupled plan");

        {
            let r = res_mut(&mut plan);
            r.outer_batched_mode = true;
            r.outer_iters = 2;
            r.linear_solver.solver_type =
                crate::solver::gpu::recipe::LinearSolverType::Fgmres { max_restart: 8 };

            // Construct an intentionally unsupported/inconsistent state for the
            // one-submission encoder: FGMRES selected but no Schur/Krylov resources.
            r.schur = None;
            r.krylov = None;
        }

        plan.step_linear_stats.clear();
        plan.skip_remaining_block = false;

        host_coupled_before_iter(&mut plan);

        assert!(
            !plan.skip_remaining_block,
            "before_iter must not skip the block when one-submission encoding fails"
        );
    }

    #[test]
    fn nonconverged_relaxation_prefers_outer_step_status_for_dual_time() {
        let converged_linear = LinearSolverStats {
            converged: true,
            ..Default::default()
        };

        assert!(!should_relax_nonconverged_apply(
            1.0e-5,
            converged_linear,
            Some(OuterStepStatus::AcceptedConverged),
        ));
        assert!(should_relax_nonconverged_apply(
            1.0e-5,
            converged_linear,
            Some(OuterStepStatus::AcceptedNonconverged),
        ));
        assert!(should_relax_nonconverged_apply(
            1.0e-5,
            converged_linear,
            Some(OuterStepStatus::RejectedRetry),
        ));
    }

    #[test]
    fn nonconverged_relaxation_falls_back_to_linear_convergence_without_dual_time_status() {
        let converged_linear = LinearSolverStats {
            converged: true,
            ..Default::default()
        };
        let stalled_linear = LinearSolverStats {
            converged: false,
            ..Default::default()
        };

        assert!(!should_relax_nonconverged_apply(1.0e-5, converged_linear, None));
        assert!(should_relax_nonconverged_apply(1.0e-5, stalled_linear, None));
        assert!(should_relax_nonconverged_apply(0.0, stalled_linear, None));
        assert!(!should_relax_nonconverged_apply(0.0, converged_linear, None));
    }

    #[test]
    fn next_step_backoff_targets_only_trigger_for_nonconverged_compressible_dual_time() {
        assert_eq!(
            next_step_backoff_targets(
                "compressible",
                Some(OuterStepStatus::AcceptedConverged),
                1.0e-3,
                1.0e-5,
                0.5,
                0.5,
            ),
            None
        );
        assert_eq!(
            next_step_backoff_targets(
                "compressible",
                Some(OuterStepStatus::AcceptedNonconverged),
                1.0e-3,
                1.0e-5,
                0.5,
                0.5,
            ),
            Some((5.0e-4, 5.0e-6))
        );
        assert_eq!(
            next_step_backoff_targets(
                "incompressible_momentum",
                Some(OuterStepStatus::AcceptedNonconverged),
                1.0e-3,
                1.0e-5,
                0.5,
                0.5,
            ),
            None
        );
        assert_eq!(
            next_step_backoff_targets(
                "compressible",
                Some(OuterStepStatus::AcceptedNonconverged),
                1.0e-3,
                0.0,
                0.5,
                0.5,
            ),
            None
        );
    }

    #[test]
    fn nonconverged_dual_time_retries_default_on_for_compressible_and_bounded() {
        let mesh = generate_structured_rect_mesh(
            4,
            3,
            1.0,
            0.4,
            BoundarySides {
                left: BoundaryType::Inlet,
                right: BoundaryType::Outlet,
                bottom: BoundaryType::Wall,
                top: BoundaryType::Wall,
            },
        );
        let model = crate::solver::model::compressible_model().expect("model");

        let mut plan = pollster::block_on(crate::solver::gpu::lowering::lower_program_plan(
            &mesh,
            &model,
            crate::solver::gpu::program::plan_instance::PlanInitConfig {
                advection_scheme: crate::solver::scheme::Scheme::Upwind,
                time_scheme: crate::solver::gpu::enums::TimeScheme::Euler,
                preconditioner: crate::solver::gpu::structs::PreconditionerType::Jacobi,
                stepping: crate::solver::gpu::recipe::SteppingMode::Coupled,
            },
            None,
            None,
        ))
        .expect("build generic coupled plan");

        {
            let r = res_mut(&mut plan);
            assert!(
                r.nonconverged_retry_enabled,
                "compressible dual-time retry should be enabled by default"
            );
            r.nonconverged_retry_max_attempts = 1;
            let values = r.fields.constants.values_mut();
            values.dtau = 1.0e-5;
        }

        plan.step_attempt_index = 0;
        plan.outer_field_residuals = vec![
            ("rho".to_string(), 1.0),
            ("rho_u".to_string(), 1.0),
            ("rho_e".to_string(), 1.0),
        ];
        plan.outer_field_residuals_scaled = vec![
            ("rho".to_string(), 1.0),
            ("rho_u".to_string(), 1.0),
            ("rho_e".to_string(), 1.0),
        ];
        finalize_outer_step_status(&mut plan, Some(false));
        assert_eq!(plan.outer_step_status, Some(OuterStepStatus::RejectedRetry));

        plan.step_attempt_index = 1;
        finalize_outer_step_status(&mut plan, Some(false));
        assert_eq!(
            plan.outer_step_status,
            Some(OuterStepStatus::AcceptedNonconverged)
        );
    }
}
