use crate::solver::gpu::execution_plan::{run_module_graph, GraphDetail, GraphExecMode};
use crate::solver::gpu::linear_solver::fgmres::{FgmresPrecondBindings, FgmresWorkspace};
use super::universal::UniversalProgramResources;
use crate::solver::gpu::modules::generated_kernels::GeneratedKernelsModule;
use crate::solver::gpu::modules::generic_coupled_schur::GenericCoupledSchurPreconditioner;
use crate::solver::gpu::modules::coupled_schur::CoupledPressureSolveKind;
use crate::solver::gpu::modules::graph::{ModuleGraph, RuntimeDims};
use crate::solver::gpu::modules::krylov_precond::{DispatchGrids, KrylovDispatch};
use crate::solver::gpu::modules::krylov_solve::KrylovSolveModule;
use crate::solver::gpu::modules::linear_solver::solve_fgmres;
use crate::solver::gpu::modules::linear_system::LinearSystemView;
use crate::solver::gpu::modules::runtime_preconditioner::RuntimePreconditionerModule;
use crate::solver::gpu::modules::time_integration::TimeIntegrationModule;
use crate::solver::gpu::modules::unified_field_resources::UnifiedFieldResources;
use crate::solver::gpu::modules::unified_graph::{
    build_graph_for_phases, build_optional_graph_for_phase, build_optional_graph_for_phases,
};
use crate::solver::gpu::lowering::kernel_registry;
use crate::solver::gpu::program::plan::{GpuProgramPlan, ProgramParamHandler};
use crate::solver::gpu::program::plan_instance::{PlanFuture, PlanLinearSystemDebug, PlanParamValue};
use crate::solver::gpu::readback::read_buffer_cached;
use crate::solver::gpu::recipe::{KernelPhase, LinearSolverType, SolverRecipe};
use crate::solver::gpu::runtime::GpuCsrRuntime;
use crate::solver::gpu::structs::{
    GpuGenericCoupledSchurSetupParams, GpuSchurPrecondGenericParams, LinearSolverStats,
};
use crate::solver::model::{ModelPreconditionerSpec, ModelSpec};
use crate::solver::model::backend::ast::FieldKind;
use crate::solver::model::ports::PortRegistry;
use bytemuck::{bytes_of, Pod, Zeroable};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Pre-resolved mapping from unknown indices to state layout slots.
///
/// This is computed once at model build time and stored in the program resources
/// to avoid repeated StateLayout lookups during GPU operations.
#[derive(Debug, Clone)]
pub struct ResolvedUnknownMapping {
    /// Maps (equation_index, component_index) -> state_offset
    /// Stored as a flat Vec where index = equation_index * max_components + component_index
    pub offsets: Vec<u32>,
    /// Number of equations
    pub num_equations: usize,
    /// Maximum components per equation (for indexing)
    pub max_components: usize,
    /// Target names for each equation
    pub target_names: Vec<String>,
    /// Component counts for each equation
    pub component_counts: Vec<usize>,
}

impl ResolvedUnknownMapping {
    /// Get the state offset for a given equation and component.
    pub fn get_offset(&self, equation: usize, component: usize) -> Option<u32> {
        if equation >= self.num_equations || component >= self.component_counts[equation] {
            return None;
        }
        let idx = equation * self.max_components + component;
        Some(self.offsets[idx])
    }

    /// Get the target name for a given equation.
    pub fn get_target_name(&self, equation: usize) -> Option<&str> {
        self.target_names.get(equation).map(|s| s.as_str())
    }

    /// Get the component count for a given equation.
    pub fn get_component_count(&self, equation: usize) -> Option<usize> {
        self.component_counts.get(equation).copied()
    }
}

/// Resolve the unknown-to-state mapping from equation targets using PortRegistry.
///
/// This is the runtime path (used during actual GPU execution) that registers
/// field ports and extracts their offsets.
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

    let mut offsets = vec![0u32; num_equations * max_components];
    let mut target_names = Vec::with_capacity(num_equations);
    let mut component_counts = Vec::with_capacity(num_equations);

    for (eq_idx, eq) in equations.iter().enumerate() {
        let target = eq.target();
        let name = target.name();
        let kind = target.kind();
        let comps = kind.component_count();

        target_names.push(name.to_string());
        component_counts.push(comps);

        // Get offsets from PortRegistry
        match kind {
            FieldKind::Scalar => {
                let port = port_registry
                    .get_field_entry_by_name(name)
                    .ok_or_else(|| format!("Field '{}' not found in port registry", name))?;
                offsets[eq_idx * max_components] = port.offset();
            }
            _ => {
                for comp in 0..comps {
                    // For vector fields, we need to get the component offset
                    // The PortRegistry stores these as separate entries or we compute from base
                    let port = port_registry
                        .get_field_entry_by_name(name)
                        .ok_or_else(|| format!("Field '{}' not found in port registry", name))?;
                    // Component offset = base offset + component index
                    offsets[eq_idx * max_components + comp] = port.offset() + comp as u32;
                }
            }
        }
    }

    Ok(ResolvedUnknownMapping {
        offsets,
        num_equations,
        max_components,
        target_names,
        component_counts,
    })
}

/// Resolve the unknown-to-state mapping from equation targets using StateLayout.
///
/// This is the build-script fallback path (used during code generation) that
/// directly queries the StateLayout without PortRegistry.
pub fn resolve_unknown_mapping_build_script(
    model: &ModelSpec,
) -> Result<ResolvedUnknownMapping, String> {
    let layout = &model.state_layout;
    let equations: Vec<_> = model.system.equations().iter().collect();
    let num_equations = equations.len();
    let max_components = equations
        .iter()
        .map(|eq| eq.target().kind().component_count())
        .max()
        .unwrap_or(1);

    let mut offsets = vec![0u32; num_equations * max_components];
    let mut target_names = Vec::with_capacity(num_equations);
    let mut component_counts = Vec::with_capacity(num_equations);

    for (eq_idx, eq) in equations.iter().enumerate() {
        let target = eq.target();
        let name = target.name();
        let kind = target.kind();
        let comps = kind.component_count();

        target_names.push(name.to_string());
        component_counts.push(comps);

        match kind {
            FieldKind::Scalar => {
                let off = layout
                    .offset_for(name)
                    .ok_or_else(|| format!("Missing '{}' in state layout", name))?;
                offsets[eq_idx * max_components] = off;
            }
            _ => {
                for comp in 0..comps {
                    let off = layout
                        .component_offset(name, comp as u32)
                        .ok_or_else(|| format!("Missing '{}' component {} in state layout", name, comp))?;
                    offsets[eq_idx * max_components + comp] = off;
                }
            }
        }
    }

    Ok(ResolvedUnknownMapping {
        offsets,
        num_equations,
        max_components,
        target_names,
        component_counts,
    })
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
    assembly_graph: ModuleGraph<GeneratedKernelsModule>,
    apply_graph: ModuleGraph<GeneratedKernelsModule>,
    update_graph: ModuleGraph<GeneratedKernelsModule>,
    explicit_graph: ModuleGraph<GeneratedKernelsModule>,
    outer_iters: usize,
    nonconverged_relax: f32,
    implicit_base_alpha_u: Option<f32>,
    linear_solver: crate::solver::gpu::recipe::LinearSolverSpec,
    schur: Option<GenericCoupledSchurResources>,
    krylov: Option<GenericCoupledKrylovResources>,
    outer_convergence: Option<OuterConvergenceMonitor>,
    _b_bc_kind: wgpu::Buffer,
    _b_bc_value: wgpu::Buffer,
    boundary_faces: Vec<Vec<u32>>,
    /// Pre-resolved unknown-to-state mapping for GPU operations.
    /// This is computed at model build time to avoid repeated StateLayout lookups.
    unknown_mapping: ResolvedUnknownMapping,
}

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

const OUTER_CONVERGENCE_WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuOuterConvergenceParams {
    num_cells: u32,
    stride: u32,
    num_targets: u32,
    _pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuOuterConvergenceTargetDesc {
    offsets: [u32; 4],
    num_comps: u32,
    _pad0: [u32; 3],
}

struct OuterConvergenceMonitor {
    target_names: Vec<String>,
    pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    _b_params_x: wgpu::Buffer,
    b_params_state: wgpu::Buffer,
    #[allow(dead_code)]
    _b_descs_x: wgpu::Buffer,
    b_descs_state: wgpu::Buffer,
    b_out_bits: wgpu::Buffer,
    bg_x: wgpu::BindGroup,
    zero_out_words: Vec<u32>,
    dispatch_cells: u32,
    state_scale: Option<Vec<f32>>,
}

impl OuterConvergenceMonitor {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &ModelSpec,
        num_cells: u32,
        x: &wgpu::Buffer,
        unknown_mapping: &ResolvedUnknownMapping,
    ) -> Result<Option<Self>, String> {
        let stride_x = model.system.unknowns_per_cell();
        let stride_state = model.state_layout.stride();
        if stride_x == 0 || stride_state == 0 || num_cells == 0 {
            return Ok(None);
        }

        let mut target_names: Vec<String> = Vec::new();
        let mut target_descs_x: Vec<GpuOuterConvergenceTargetDesc> = Vec::new();
        let mut target_descs_state: Vec<GpuOuterConvergenceTargetDesc> = Vec::new();

        let mut unknown_offset_cursor: u32 = 0;
        for (eq_idx, eqn) in model.system.equations().iter().enumerate() {
            let target = eqn.target();
            let name = target.name();

            let kind = target.kind();
            let comps = kind.component_count() as usize;
            if comps == 0 {
                continue;
            }
            if comps > 4 {
                return Err(format!(
                    "outer convergence monitor only supports up to 4 components per target (got {comps} for '{name}')"
                ));
            }

            let mut offsets_x = [0u32; 4];
            for comp in 0..comps {
                offsets_x[comp] = unknown_offset_cursor + comp as u32;
            }
            unknown_offset_cursor += comps as u32;

            // Get state offsets from the pre-resolved mapping
            let mut offsets_state = [0u32; 4];
            let mut has_all_offsets = true;
            for comp in 0..comps {
                match unknown_mapping.get_offset(eq_idx, comp) {
                    Some(off) => offsets_state[comp] = off,
                    None => {
                        has_all_offsets = false;
                        break;
                    }
                }
            }

            if !has_all_offsets {
                continue;
            }

            target_names.push(name.to_string());
            target_descs_x.push(GpuOuterConvergenceTargetDesc {
                offsets: offsets_x,
                num_comps: comps as u32,
                _pad0: [0u32; 3],
            });
            target_descs_state.push(GpuOuterConvergenceTargetDesc {
                offsets: offsets_state,
                num_comps: comps as u32,
                _pad0: [0u32; 3],
            });
        }

        let num_targets = target_descs_x.len() as u32;
        if num_targets == 0 {
            return Ok(None);
        }
        if target_descs_state.len() != target_descs_x.len() {
            return Err(format!(
                "outer convergence monitor target count mismatch: x_descs={} state_descs={}",
                target_descs_x.len(),
                target_descs_state.len()
            ));
        }

        let pipeline = {
            let src = kernel_registry::kernel_source_by_id(
                "",
                crate::solver::model::KernelId::OUTER_CONVERGENCE,
            )?;
            (src.create_pipeline)(device)
        };
        let bgl = pipeline.get_bind_group_layout(0);

        let b_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outer_convergence:params_x"),
            size: std::mem::size_of::<GpuOuterConvergenceParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_x = GpuOuterConvergenceParams {
            num_cells,
            stride: stride_x,
            num_targets,
            _pad0: 0,
        };
        queue.write_buffer(&b_params, 0, bytemuck::bytes_of(&params_x));

        let b_params_state = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outer_convergence:params_state"),
            size: std::mem::size_of::<GpuOuterConvergenceParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_state = GpuOuterConvergenceParams {
            num_cells,
            stride: stride_state,
            num_targets,
            _pad0: 0,
        };
        queue.write_buffer(&b_params_state, 0, bytemuck::bytes_of(&params_state));

        let b_descs_x = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outer_convergence:target_descs_x"),
            size: (target_descs_x.len() as u64)
                * std::mem::size_of::<GpuOuterConvergenceTargetDesc>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&b_descs_x, 0, bytemuck::cast_slice(&target_descs_x));

        let b_descs_state = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outer_convergence:target_descs_state"),
            size: (target_descs_state.len() as u64)
                * std::mem::size_of::<GpuOuterConvergenceTargetDesc>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &b_descs_state,
            0,
            bytemuck::cast_slice(&target_descs_state),
        );

        let b_out_bits = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outer_convergence:out_bits"),
            size: (num_targets as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bg_x = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("outer_convergence:bg_x"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_descs_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_out_bits.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_params.as_entire_binding(),
                },
            ],
        });

        let zero_out_words = vec![0u32; target_descs_x.len()];
        let dispatch_cells = (num_cells + OUTER_CONVERGENCE_WORKGROUP_SIZE - 1)
            / OUTER_CONVERGENCE_WORKGROUP_SIZE;

        Ok(Some(Self {
            target_names,
            pipeline,
            _b_params_x: b_params,
            b_params_state,
            _b_descs_x: b_descs_x,
            b_descs_state,
            b_out_bits,
            bg_x,
            zero_out_words,
            dispatch_cells,
            state_scale: None,
        }))
    }

    fn reset_step(&mut self) {
        self.state_scale = None;
    }

    fn ensure_state_scale(
        &mut self,
        plan: &GpuProgramPlan,
        state: &wgpu::Buffer,
    ) -> Result<(), String> {
        if self.state_scale.is_some() {
            return Ok(());
        }
        let scale = self.compute_maxima_with_params_and_descs(
            plan,
            state,
            &self.b_descs_state,
            &self.b_params_state,
            "outer_convergence:state",
        )?;
        self.state_scale = Some(scale);
        Ok(())
    }

    fn delta_maxima(&self, plan: &GpuProgramPlan) -> Result<Vec<f32>, String> {
        self.compute_maxima_from_bind_group(plan, &self.bg_x, "outer_convergence:delta")
    }

    fn compute_maxima_with_params_and_descs(
        &self,
        plan: &GpuProgramPlan,
        input: &wgpu::Buffer,
        descs: &wgpu::Buffer,
        params: &wgpu::Buffer,
        label_prefix: &'static str,
    ) -> Result<Vec<f32>, String> {
        let bgl = self.pipeline.get_bind_group_layout(0);
        let bg = plan.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label_prefix),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: descs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.b_out_bits.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        self.compute_maxima_from_bind_group(plan, &bg, label_prefix)
    }

    fn compute_maxima_from_bind_group(
        &self,
        plan: &GpuProgramPlan,
        bind_group: &wgpu::BindGroup,
        label_prefix: &'static str,
    ) -> Result<Vec<f32>, String> {
        if self.zero_out_words.is_empty() {
            return Ok(Vec::new());
        }

        plan.context.queue.write_buffer(
            &self.b_out_bits,
            0,
            bytemuck::cast_slice(&self.zero_out_words),
        );

        let mut encoder = plan
            .context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(label_prefix),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label_prefix),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(self.dispatch_cells.max(1), 1, 1);
        }
        plan.context.queue.submit(Some(encoder.finish()));

        let out_bytes = (self.zero_out_words.len() as u64) * 4;
        let raw = pollster::block_on(read_buffer_cached(
            &plan.context,
            &plan.staging_cache,
            &plan.profiling_stats,
            &self.b_out_bits,
            out_bytes,
            "outer_convergence:out_bits (cached)",
        ));
        if raw.len() != out_bytes as usize {
            return Err(format!(
                "outer convergence readback size mismatch: got {} expected {}",
                raw.len(),
                out_bytes
            ));
        }
        let words: &[u32] = bytemuck::cast_slice(&raw);
        Ok(words.iter().map(|&w| f32::from_bits(w)).collect())
    }

    fn target_names(&self) -> &[String] {
        &self.target_names
    }

    fn state_scale(&self) -> Option<&[f32]> {
        self.state_scale.as_deref()
    }
}

impl GenericCoupledProgramResources {
    pub(crate) fn new(
        runtime: GpuCsrRuntime,
        fields: UnifiedFieldResources,
        kernels: GeneratedKernelsModule,
        model: &ModelSpec,
        recipe: &SolverRecipe,
        b_bc_kind: wgpu::Buffer,
        b_bc_value: wgpu::Buffer,
        boundary_faces: Vec<Vec<u32>>,
    ) -> Result<Self, String> {
        // Build graphs from recipe using unified graph builder.
        //
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

        // Resolve unknown-to-state mapping using PortRegistry (runtime path)
        let unknown_mapping = resolve_unknown_mapping_runtime(model, &recipe.port_registry)?;

        let outer_convergence = OuterConvergenceMonitor::new(
            &runtime.common.context.device,
            &runtime.common.context.queue,
            model,
            runtime.common.num_cells,
            runtime.linear_port_space.buffer(runtime.linear_ports.x),
            &unknown_mapping,
        )?;

        let requested_time_scheme = match recipe.initial_constants.time_scheme {
            0 => crate::solver::gpu::enums::TimeScheme::Euler,
            1 => crate::solver::gpu::enums::TimeScheme::BDF2,
            other => return Err(format!("unknown time_scheme id {other}")),
        };

        Ok(Self {
            runtime,
            fields,
            time_integration: TimeIntegrationModule::new(),
            requested_time_scheme,
            kernels,
            init_prepare_graph,
            dp_init_enabled,
            dp_init_needed: AtomicBool::new(dp_init_enabled),
            assembly_graph,
            apply_graph,
            update_graph,
            explicit_graph,
            outer_iters,
            nonconverged_relax: 1.0,
            implicit_base_alpha_u: None,
            linear_solver,
            schur,
            krylov,
            outer_convergence,
            _b_bc_kind: b_bc_kind,
            _b_bc_value: b_bc_value,
            boundary_faces,
            unknown_mapping,
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
}

fn validate_schur_model(
    model: &ModelSpec,
    unknown_mapping: &ResolvedUnknownMapping,
) -> Result<(f32, crate::solver::model::SchurBlockLayout), String> {
    let Some(solver) = model.linear_solver else {
        return Err("model does not define a linear solver spec".into());
    };
    let ModelPreconditionerSpec::Schur { omega, layout } = solver.preconditioner else {
        return Err("model does not request Schur preconditioning".into());
    };

    let method = model.method()?;
    if !matches!(method, crate::solver::model::method::MethodSpec::Coupled(_)) {
        return Err("Schur preconditioner is only wired for the coupled pipeline".to_string());
    }
    layout.validate(model.system.unknowns_per_cell())?;

    // Validate the layout against the equation targets used to assemble the system.
    //
    // For the current Schur bridge, the linear system is assumed to consist only of a
    // velocity-like block and a single pressure-like scalar.
    //
    // Use the pre-resolved unknown_mapping instead of querying StateLayout directly.
    let mut target_indices = std::collections::BTreeSet::new();
    let mut scalar_targets = std::collections::BTreeSet::new();
    for (eq_idx, eq) in model.system.equations().iter().enumerate() {
        let target = eq.target();
        let comps = target.kind().component_count() as usize;
        
        match target.kind() {
            crate::solver::model::backend::ast::FieldKind::Scalar => {
                let idx = unknown_mapping
                    .get_offset(eq_idx, 0)
                    .ok_or_else(|| format!("missing '{}' in unknown mapping", target.name()))?;
                target_indices.insert(idx);
                scalar_targets.insert(idx);
            }
            _ => {
                for comp in 0..comps {
                    let idx = unknown_mapping
                        .get_offset(eq_idx, comp)
                        .ok_or_else(|| {
                            format!(
                                "missing '{}' component {} in unknown mapping",
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

    Ok((omega, layout))
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
    let (omega, layout) = validate_schur_model(model, &unknown_mapping)?;

    let LinearSolverType::Fgmres { max_restart } = recipe.linear_solver.solver_type else {
        return Err(
            "Schur preconditioner requires LinearSolverType::Fgmres in the recipe".to_string(),
        );
    };

    let device = &runtime.common.context.device;
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

    let b_precond_params =
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GenericCoupled Schur precond_params"),
            size: std::mem::size_of::<GpuSchurPrecondGenericParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

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

    let setup_pipeline = GenericCoupledSchurPreconditioner::build_setup_pipeline(device);
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
        scalar_row_offsets,
        diagonal_indices,
        matrix_values,
        &b_diag_u,
        &b_diag_p,
        &b_p_matrix_values,
        &b_setup_params,
    )?;

    let system = LinearSystemView {
        ports: runtime.linear_ports,
        space: &runtime.linear_port_space,
    };

    let precond_bindings = FgmresPrecondBindings::SchurWithParams {
        diag_u: &b_diag_u,
        diag_p: &b_diag_p,
        precond_params: &b_precond_params,
    };
    let fgmres = FgmresWorkspace::new_from_system(
        device,
        num_dofs,
        num_cells,
        max_restart,
        system,
        precond_bindings,
        "generic_coupled",
    );

    let precond = GenericCoupledSchurPreconditioner::new(
        device,
        num_cells,
        scalar_row_offsets,
        scalar_col_indices,
        &b_p_matrix_values,
        runtime.common.mesh.scalar_row_offsets.clone(),
        runtime.common.mesh.scalar_col_indices.clone(),
        scalar_nnz,
        &b_diag_u,
        &b_diag_p,
        &b_precond_params,
        setup_bg,
        setup_pipeline,
        b_setup_params,
        model.system.unknowns_per_cell(),
        layout.p,
        u_len,
        u0123,
        u4567,
        CoupledPressureSolveKind::from_config(recipe.linear_solver.preconditioner),
    );

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

    let precond_bindings = FgmresPrecondBindings::Diag {
        diag_u: &b_diag_u,
        diag_v: &b_diag_v,
        diag_p: &b_diag_p,
    };

    let fgmres = FgmresWorkspace::new_from_system(
        device,
        n,
        num_cells,
        max_restart.max(1),
        system,
        precond_bindings,
        "generic_coupled",
    );

    let unknowns_per_cell: u32 = recipe
        .unknowns_per_cell
        .try_into()
        .map_err(|_| "recipe.unknowns_per_cell overflows u32".to_string())?;
    let (row_offsets, col_indices) = crate::solver::gpu::csr::build_block_csr(
        &runtime.common.mesh.scalar_row_offsets,
        &runtime.common.mesh.scalar_col_indices,
        unknowns_per_cell,
    );

    let solver = KrylovSolveModule::new(
        fgmres,
        RuntimePreconditionerModule::new(
            device,
            recipe.linear_solver.preconditioner,
            num_cells,
            n,
            runtime.num_nonzeros,
            row_offsets,
            col_indices,
            runtime
                .linear_port_space
                .buffer(runtime.linear_ports.values)
                .clone(),
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
                &self.runtime.common.context,
                &mut schur.solver,
                system,
                n,
                self.runtime.common.num_cells,
                schur.dispatch,
                max_restart,
                max_iters,
                tol,
                tol * 1e-4,
                "GenericCoupled Schur (debug)",
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
                &self.runtime.common.context,
                &mut krylov.solver,
                system,
                n,
                self.runtime.common.num_cells,
                krylov.dispatch,
                max_restart.max(1),
                max_iters,
                tol,
                tol * 1e-4,
                "GenericCoupled FGMRES (debug)",
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
}

fn res(plan: &GpuProgramPlan) -> &GenericCoupledProgramResources {
    plan.resources
        .get::<UniversalProgramResources>()
        .and_then(|u| u.generic_coupled())
        .expect("missing GenericCoupledProgramResources backend")
}

fn res_mut(plan: &mut GpuProgramPlan) -> &mut GenericCoupledProgramResources {
    plan.resources
        .get_mut::<UniversalProgramResources>()
        .and_then(|u| u.generic_coupled_mut())
        .expect("missing GenericCoupledProgramResources backend")
}

/// Register ops using the unified registry builder.
/// The recipe's stepping mode determines which ops are registered.

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

pub(crate) fn spec_set_bc_value(
    plan: &GpuProgramPlan,
    boundary: crate::solver::gpu::enums::GpuBoundaryType,
    unknown_component: u32,
    value: f32,
) -> Result<(), String> {
    let coupled_stride = plan.model.system.unknowns_per_cell() as u32;
    if coupled_stride == 0 {
        return Err("model has no coupled unknowns".into());
    }
    if unknown_component >= coupled_stride {
        return Err(format!(
            "unknown_component {unknown_component} out of range (stride={coupled_stride})"
        ));
    }

    let boundary_count = 6u32; // None, Inlet, Outlet, Wall, SlipWall, MovingWall
    let boundary_idx = boundary as u32;
    if boundary_idx >= boundary_count {
        return Err(format!("invalid boundary type index {boundary_idx}"));
    }

    // Per-face BC storage: apply the boundary value to all boundary faces of this type.
    // (Boundary index 0 is reserved for "None" and should have no boundary faces.)
    let faces = res(plan)
        .boundary_faces
        .get(boundary_idx as usize)
        .ok_or_else(|| format!("missing boundary_faces[{boundary_idx}]"))?;
    for &face_idx in faces {
        let offset_bytes = ((face_idx as u64 * coupled_stride as u64 + unknown_component as u64)
            * 4) as u64;
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
    plan.outer_field_residuals.clear();
    plan.repeat_break = false;

    let device = plan.context.device.clone();
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    if let Some(monitor) = r.outer_convergence.as_mut() {
        monitor.reset_step();
    }
    r.fields.advance_step();

    // OpenFOAM-style `backward` startup: the first step falls back to Euler because the
    // `n-1` history is not yet meaningful. Once we have advanced at least one step, switch
    // back to the requested scheme (BDF2).
    //
    // This keeps the scheme selection stable for steady runs and only affects `BDF2` at the
    // very beginning of a simulation.
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
    // iterations) start from a consistent iterate.
    //
    // This mirrors the EI solver's pre-step copy and avoids reading stale data
    // from the rotated ping-pong buffer.
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
    r.time_integration
        .prepare_step(&mut r.fields.constants, &queue);
}

pub(crate) fn host_finalize_step(plan: &mut GpuProgramPlan) {
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    {
        let values = r.fields.constants.values_mut();
        values.time_scheme = r.requested_time_scheme as u32;
    }
    r.time_integration
        .finalize_step(&mut r.fields.constants, &queue);
}

pub(crate) fn host_solve_linear_system(plan: &mut GpuProgramPlan) {
    let context = crate::solver::gpu::context::GpuContext {
        device: plan.context.device.clone(),
        queue: plan.context.queue.clone(),
    };

    let r = res_mut(plan);

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

        let stats = solve_fgmres(
            &context,
            &mut schur.solver,
            system,
            r.runtime.num_dofs,
            r.runtime.common.num_cells,
            schur.dispatch,
            max_restart,
            r.linear_solver.max_iters,
            r.linear_solver.tolerance,
            r.linear_solver.tolerance_abs,
            "generic_coupled:schur",
        );
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

        let stats = solve_fgmres(
            &context,
            &mut krylov.solver,
            system,
            r.runtime.num_dofs,
            r.runtime.common.num_cells,
            krylov.dispatch,
            max_restart.max(1),
            r.linear_solver.max_iters,
            r.linear_solver.tolerance,
            r.linear_solver.tolerance_abs,
            "generic_coupled:fgmres",
        );
        plan.last_linear_stats = stats;
        plan.step_linear_stats.push(stats);
        return;
    }

    let stats = r
        .runtime
        .solve_linear_system_cg(r.linear_solver.max_iters, r.linear_solver.tolerance);
    plan.last_linear_stats = stats;
    plan.step_linear_stats.push(stats);
}

pub(crate) fn host_after_solve(plan: &mut GpuProgramPlan) {
    let iters_done = plan.step_linear_stats.len();
    plan.outer_iterations = iters_done as u32;

    let outer_iters = res(plan).outer_iters;
    if outer_iters <= 1 && !plan.collect_convergence_stats {
        return;
    }

    let (lin_tol, lin_tol_abs, state, monitor) = {
        let r = res_mut(plan);
        (
            r.linear_solver.tolerance,
            r.linear_solver.tolerance_abs,
            r.fields.current_state().clone(),
            r.outer_convergence.take(),
        )
    };

    let Some(mut monitor) = monitor else {
        return;
    };

    if outer_iters > 1 {
        if let Err(err) = monitor.ensure_state_scale(plan, &state) {
            eprintln!("[cfd2][outer] failed to compute state scale: {err}");
        }
    }

    let delta = match monitor.delta_maxima(plan) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("[cfd2][outer] failed to compute correction norms: {err}");
            res_mut(plan).outer_convergence = Some(monitor);
            return;
        }
    };

    if delta.len() != monitor.target_names().len() {
        eprintln!(
            "[cfd2][outer] correction norm length mismatch: got {} expected {}",
            delta.len(),
            monitor.target_names().len()
        );
        res_mut(plan).outer_convergence = Some(monitor);
        return;
    }

    plan.outer_field_residuals.clear();
    plan.outer_field_residuals.extend(
        monitor
            .target_names()
            .iter()
            .cloned()
            .zip(delta.iter().copied()),
    );

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

    if outer_iters <= 1 || !plan.last_linear_stats.converged || plan.repeat_break {
        res_mut(plan).outer_convergence = Some(monitor);
        return;
    }

    let Some(scale) = monitor.state_scale() else {
        res_mut(plan).outer_convergence = Some(monitor);
        return;
    };
    if scale.len() != delta.len() {
        res_mut(plan).outer_convergence = Some(monitor);
        return;
    }

    // Outer-loop convergence: stop iterating once the maximum correction magnitude is
    // sufficiently small relative to the current state scale.
    let tol_rel = (lin_tol * 10.0).clamp(1e-8, 1e-2);
    let tol_abs = (lin_tol_abs * 10.0).max(0.0);

    let mut converged = true;
    for (&d, &s) in delta.iter().zip(scale.iter()) {
        if !d.is_finite() || !s.is_finite() {
            converged = false;
            break;
        }
        let s = s.max(1.0);
        let tol = tol_abs + tol_rel * s;
        if d > tol {
            converged = false;
            break;
        }
    }

    if converged {
        plan.repeat_break = true;
    }

    res_mut(plan).outer_convergence = Some(monitor);
}

pub(crate) fn host_implicit_set_alpha_for_apply(plan: &mut GpuProgramPlan) {
    let queue = plan.context.queue.clone();
    let converged = plan.last_linear_stats.converged;
    let r = res_mut(plan);

    let base_alpha_u = r.fields.constants.values().alpha_u;
    r.implicit_base_alpha_u = Some(base_alpha_u);

    if converged {
        return;
    }

    let apply_alpha_u = base_alpha_u * r.nonconverged_relax.max(0.0);
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
    run_module_graph(
        &r.assembly_graph,
        context,
        &r.kernels,
        r.runtime_dims(),
        mode,
    )
}

pub(crate) fn init_prepare_graph_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let r = res(plan);
    if !r.dp_init_enabled || !r.dp_init_needed.load(Ordering::Relaxed) {
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

pub(crate) fn param_nonconverged_relax(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::F32(alpha) = value else {
        return Err("NonconvergedRelax expects F32".to_string());
    };
    res_mut(plan).nonconverged_relax = alpha.max(0.0);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::model::backend::ast::{fvm, vol_scalar, vol_vector3, EquationSystem};
    use crate::solver::model::{eos, primitives};
    use crate::solver::model::{
        incompressible_momentum_model, BoundarySpec, ModelLinearSolverSpec,
        ModelPreconditionerSpec, SchurBlockLayout,
    };
    use crate::solver::units::si;

    #[test]
    fn schur_rejects_invalid_layout_indices() {
        let mut model = incompressible_momentum_model();
        let Some(spec) = &mut model.linear_solver else {
            panic!("missing linear_solver spec");
        };
        let ModelPreconditionerSpec::Schur { layout, .. } = &mut spec.preconditioner else {
            panic!("expected Schur preconditioner");
        };
        // Distinct/in-range, but doesn't cover the full set of equation target indices.
        *layout = SchurBlockLayout::from_u_p(&[0], 2).expect("layout build failed");

        let unknown_mapping = resolve_unknown_mapping_build_script(&model).expect("mapping failed");
        let err = validate_schur_model(&model, &unknown_mapping).unwrap_err();
        assert!(err.contains("equation targets"), "unexpected error: {err}");
    }

    #[test]
    fn schur_accepts_vector3_velocity_layout() {
        let u = vol_vector3("U", si::VELOCITY);
        let p = vol_scalar("p", si::PRESSURE);

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
                    layout: SchurBlockLayout::from_u_p(&[0, 1, 2], 3).expect("layout build failed"),
                },
                ..Default::default()
            }),
            primitives: primitives::PrimitiveDerivations::default(),
        };

        let unknown_mapping = resolve_unknown_mapping_build_script(&model).expect("mapping failed");
        validate_schur_model(&model, &unknown_mapping).expect("Vector3 velocity Schur layout should validate");
    }
}
