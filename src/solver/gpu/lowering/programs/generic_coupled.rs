use super::universal::UniversalProgramResources;
use crate::solver::gpu::execution_plan::{run_module_graph, GraphDetail, GraphExecMode};
use crate::solver::gpu::linear_solver::fgmres::{FgmresPrecondBindings, FgmresWorkspace};
use crate::solver::gpu::lowering::kernel_registry;
use crate::solver::gpu::modules::coupled_schur::CoupledPressureSolveKind;
use crate::solver::gpu::modules::generated_kernels::GeneratedKernelsModule;
use crate::solver::gpu::modules::generic_coupled_schur::{
    GenericCoupledSchurPreconditioner, GenericCoupledSchurPreconditionerInputs,
    GenericCoupledSchurSetupBindGroupInputs,
};
use crate::solver::gpu::modules::graph::{DispatchKind, ModuleGraph, RuntimeDims};
use crate::solver::gpu::modules::krylov_precond::{DispatchGrids, KrylovDispatch};
use crate::solver::gpu::modules::krylov_solve::KrylovSolveModule;
use crate::solver::gpu::modules::linear_solver::{
    solve_fgmres, submit_solve_cg_fixed_iterations_chunked,
    submit_solve_fgmres_fixed_iterations_chunked, SolveFgmresArgs,
};
use crate::solver::gpu::modules::linear_system::LinearSystemView;
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
    PlanFuture, PlanLinearSystemDebug, PlanParamValue,
};
use crate::solver::gpu::recipe::{KernelPhase, LinearSolverType, SolverRecipe};
use crate::solver::gpu::runtime::GpuCsrRuntime;
use crate::solver::gpu::structs::{
    GpuGenericCoupledSchurSetupParams, GpuSchurPrecondGenericParams, LinearSolverStats,
};
use crate::solver::model::backend::ast::FieldKind;
use crate::solver::model::ports::PortRegistry;
use crate::solver::model::{ModelPreconditionerSpec, ModelSpec};
use bytemuck::{bytes_of, Pod, Zeroable};
use cfd2_codegen::solver::codegen::bc_table::{HostBcTable, BOUNDARY_TYPE_COUNT};
use cfd2_codegen::solver::codegen::wgsl_ast::*;
use cfd2_codegen::solver::codegen::wgsl_dsl::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

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
}

impl ResolvedUnknownMapping {
    const UNMAPPED_OFFSET: u32 = u32::MAX;

    /// Get the state offset for a given equation and component.
    pub fn get_offset(&self, equation: usize, component: usize) -> Option<u32> {
        if equation >= self.num_equations || component >= self.max_components {
            return None;
        }
        let idx = equation * self.max_components + component;
        let offset = self.offsets[idx];
        (offset != Self::UNMAPPED_OFFSET).then_some(offset)
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

    let mut offsets = vec![ResolvedUnknownMapping::UNMAPPED_OFFSET; num_equations * max_components];

    for (eq_idx, eq) in equations.iter().enumerate() {
        let target = eq.target();
        let name = target.name();
        let kind = target.kind();
        let comps = kind.component_count();

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
    assembly_graph: ModuleGraph<GeneratedKernelsModule>,
    apply_graph: ModuleGraph<GeneratedKernelsModule>,
    update_graph: ModuleGraph<GeneratedKernelsModule>,
    explicit_graph: ModuleGraph<GeneratedKernelsModule>,
    outer_iters: usize,
    outer_tol: f32,
    outer_tol_abs: f32,
    outer_break_enabled: bool,
    outer_batched_mode: bool,
    nonconverged_relax: f32,
    implicit_base_alpha_u: Option<f32>,
    linear_solver: crate::solver::gpu::recipe::LinearSolverSpec,
    schur: Option<GenericCoupledSchurResources>,
    krylov: Option<GenericCoupledKrylovResources>,
    outer_convergence: Option<OuterConvergenceMonitor>,
    outer_gate: Option<OuterAdaptiveGate>,
    _b_bc_kind: wgpu::Buffer,
    _b_bc_value: wgpu::Buffer,
    boundary_faces: Vec<Vec<u32>>,
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuOuterConvergenceBreakParams {
    count: u32,
    tol_rel: f32,
    tol_abs: f32,
    _pad0: u32,
}

/// Builds the outer convergence break kernel WGSL via the structured DSL.
///
/// Single-thread kernel that checks whether all (delta, scale) pairs satisfy the
/// convergence criterion, writing `1u` (converged) or `0u` (not converged) into
/// `status[0]`.
fn build_outer_convergence_break_wgsl() -> String {
    let mut m = Module::new();

    m.push(Item::Struct(StructDef::new(
        "BreakParams",
        vec![
            StructField::new("count", Type::U32),
            StructField::new("tol_rel", Type::F32),
            StructField::new("tol_abs", Type::F32),
            StructField::new("_pad0", Type::U32),
        ],
    )));

    m.push(Item::GlobalVar(GlobalVar::new(
        "delta",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(0), Attribute::Binding(0)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "scale",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(0), Attribute::Binding(1)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "status",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(2)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "params",
        Type::Custom("BreakParams".into()),
        StorageClass::Uniform,
        None,
        vec![Attribute::Group(0), Attribute::Binding(3)],
    )));

    let global_id = Expr::ident("global_id");
    let params = Expr::ident("params");
    let delta = Expr::ident("delta");
    let scale = Expr::ident("scale");
    let status = Expr::ident("status");
    let i = Expr::ident("i");
    let d = Expr::ident("d");
    let s_raw = Expr::ident("s_raw");
    let bad_d = Expr::ident("bad_d");
    let bad_s = Expr::ident("bad_s");
    let s = Expr::ident("s");
    let tol = Expr::ident("tol");
    let converged = Expr::ident("converged");

    let body = block(vec![
        // if (global_id.x != 0u) { return; }
        if_block_expr(
            global_id.clone().field("x").ne(Expr::lit_u32(0)),
            block(vec![return_void()]),
            None,
        ),
        // var converged: u32 = 1u;
        var_typed_expr("converged", Type::U32, Some(Expr::lit_u32(1))),
        // for (var i: u32 = 0u; i < params.count; i = i + 1u) { ... }
        for_loop_expr(
            for_init_var_typed_expr("i", Type::U32, Expr::lit_u32(0)),
            i.clone().lt(params.clone().field("count")),
            for_step_increment_expr(i.clone()),
            block(vec![
                // let d = delta[i];
                let_expr("d", delta.clone().index(i.clone())),
                // let s_raw = scale[i];
                let_expr("s_raw", scale.clone().index(i.clone())),
                // let bad_d = (!(d <= d)) || (abs(d) > 1.0e30);
                let_expr(
                    "bad_d",
                    (!d.clone().le(d.clone())) | abs(d.clone()).gt(Expr::lit_f32(1.0e30)),
                ),
                // let bad_s = (!(s_raw <= s_raw)) || (abs(s_raw) > 1.0e30);
                let_expr(
                    "bad_s",
                    (!s_raw.clone().le(s_raw.clone()))
                        | abs(s_raw.clone()).gt(Expr::lit_f32(1.0e30)),
                ),
                // if (bad_d || bad_s) { converged = 0u; break; }
                if_block_expr(
                    bad_d.clone() | bad_s.clone(),
                    block(vec![
                        assign_expr(converged.clone(), Expr::lit_u32(0)),
                        break_stmt(),
                    ]),
                    None,
                ),
                // let s = max(s_raw, 1.0);
                let_expr("s", max(s_raw.clone(), Expr::lit_f32(1.0))),
                // let tol = params.tol_abs + params.tol_rel * s;
                let_expr(
                    "tol",
                    params.clone().field("tol_abs") + params.clone().field("tol_rel") * s.clone(),
                ),
                // if (d > tol) { converged = 0u; break; }
                if_block_expr(
                    d.clone().gt(tol.clone()),
                    block(vec![
                        assign_expr(converged.clone(), Expr::lit_u32(0)),
                        break_stmt(),
                    ]),
                    None,
                ),
            ]),
        ),
        // status[0] = converged;
        assign_expr(status.clone().index(Expr::lit_u32(0)), converged.clone()),
    ]);

    m.push(Item::Function(Function::new(
        "main",
        vec![Param::new(
            "global_id",
            Type::vec3_u32(),
            vec![Attribute::Builtin("global_invocation_id".into())],
        )],
        None,
        vec![Attribute::Compute, Attribute::WorkgroupSize3(1, 1, 1)],
        body,
    )));

    m.to_wgsl()
}

/// Builds the outer gate kernel WGSL via the structured DSL.
///
/// Single-thread kernel that reads `break_status[0]`:
/// - If NOT converged (0): copies real dispatch args to indirect args for cells/faces,
///   and atomically increments the iteration counter.
/// - If converged (1): writes zeros to indirect args (zero-cost dispatch).
fn build_outer_gate_wgsl() -> String {
    let mut m = Module::new();

    m.push(Item::GlobalVar(GlobalVar::new(
        "break_status",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(0), Attribute::Binding(0)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "real_args_cells",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(0), Attribute::Binding(1)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "indirect_args_cells",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(2)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "real_args_faces",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(0), Attribute::Binding(3)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "indirect_args_faces",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(4)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "iter_counter",
        Type::array(Type::atomic(Type::U32)),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(5)],
    )));

    let global_id = Expr::ident("global_id");
    let break_status = Expr::ident("break_status");
    let real_args_cells = Expr::ident("real_args_cells");
    let indirect_args_cells = Expr::ident("indirect_args_cells");
    let real_args_faces = Expr::ident("real_args_faces");
    let indirect_args_faces = Expr::ident("indirect_args_faces");
    let iter_counter = Expr::ident("iter_counter");
    let converged = Expr::ident("converged");

    let body = block(vec![
        // if (global_id.x != 0u) { return; }
        if_block_expr(
            global_id.clone().field("x").ne(Expr::lit_u32(0)),
            block(vec![return_void()]),
            None,
        ),
        // let converged = break_status[0];
        let_expr("converged", break_status.clone().index(Expr::lit_u32(0))),
        // if (converged == 0u) { ... } else { ... }
        if_block_expr(
            converged.clone().eq(Expr::lit_u32(0)),
            block(vec![
                // Not converged: enable dispatches with real args
                assign_expr(
                    indirect_args_cells.clone().index(Expr::lit_u32(0)),
                    real_args_cells.clone().index(Expr::lit_u32(0)),
                ),
                assign_expr(
                    indirect_args_cells.clone().index(Expr::lit_u32(1)),
                    real_args_cells.clone().index(Expr::lit_u32(1)),
                ),
                assign_expr(
                    indirect_args_cells.clone().index(Expr::lit_u32(2)),
                    real_args_cells.clone().index(Expr::lit_u32(2)),
                ),
                assign_expr(
                    indirect_args_faces.clone().index(Expr::lit_u32(0)),
                    real_args_faces.clone().index(Expr::lit_u32(0)),
                ),
                assign_expr(
                    indirect_args_faces.clone().index(Expr::lit_u32(1)),
                    real_args_faces.clone().index(Expr::lit_u32(1)),
                ),
                assign_expr(
                    indirect_args_faces.clone().index(Expr::lit_u32(2)),
                    real_args_faces.clone().index(Expr::lit_u32(2)),
                ),
                // Increment iteration counter
                call_stmt_expr(atomic_add(
                    iter_counter.clone().index(Expr::lit_u32(0)).addr_of(),
                    Expr::lit_u32(1),
                )),
            ]),
            Some(block(vec![
                // Converged: zero out dispatches
                assign_expr(
                    indirect_args_cells.clone().index(Expr::lit_u32(0)),
                    Expr::lit_u32(0),
                ),
                assign_expr(
                    indirect_args_cells.clone().index(Expr::lit_u32(1)),
                    Expr::lit_u32(0),
                ),
                assign_expr(
                    indirect_args_cells.clone().index(Expr::lit_u32(2)),
                    Expr::lit_u32(0),
                ),
                assign_expr(
                    indirect_args_faces.clone().index(Expr::lit_u32(0)),
                    Expr::lit_u32(0),
                ),
                assign_expr(
                    indirect_args_faces.clone().index(Expr::lit_u32(1)),
                    Expr::lit_u32(0),
                ),
                assign_expr(
                    indirect_args_faces.clone().index(Expr::lit_u32(2)),
                    Expr::lit_u32(0),
                ),
            ])),
        ),
    ]);

    m.push(Item::Function(Function::new(
        "main",
        vec![Param::new(
            "global_id",
            Type::vec3_u32(),
            vec![Attribute::Builtin("global_invocation_id".into())],
        )],
        None,
        vec![Attribute::Compute, Attribute::WorkgroupSize3(1, 1, 1)],
        body,
    )));

    m.to_wgsl()
}

/// Builds the STOP-inject kernel WGSL via the structured DSL.
///
/// Single-thread kernel that reads `break_status[0]` and writes `f32(break_status[0])`
/// into the scalars buffer at the given `scalar_stop` index.
fn build_outer_stop_inject_wgsl(scalar_stop: usize) -> String {
    let mut m = Module::new();

    m.push(Item::GlobalVar(GlobalVar::new(
        "break_status",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(0), Attribute::Binding(0)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "scalars",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(1)],
    )));

    let global_id = Expr::ident("global_id");
    let break_status = Expr::ident("break_status");
    let scalars = Expr::ident("scalars");

    let body = block(vec![
        // if (global_id.x != 0u) { return; }
        if_block_expr(
            global_id.clone().field("x").ne(Expr::lit_u32(0)),
            block(vec![return_void()]),
            None,
        ),
        // scalars[SCALAR_STOP] = f32(break_status[0]);
        assign_expr(
            scalars.clone().index(Expr::lit_u32(scalar_stop as u32)),
            f32_cast(break_status.clone().index(Expr::lit_u32(0))),
        ),
    ]);

    m.push(Item::Function(Function::new(
        "main",
        vec![Param::new(
            "global_id",
            Type::vec3_u32(),
            vec![Attribute::Builtin("global_invocation_id".into())],
        )],
        None,
        vec![Attribute::Compute, Attribute::WorkgroupSize3(1, 1, 1)],
        body,
    )));

    m.to_wgsl()
}

/// GPU-side adaptive outer break gate resources.
///
/// This struct holds the indirect dispatch argument buffers, the gate kernel
/// pipeline, and the iteration counter. It is created when one-submission mode
/// with adaptive break is enabled.
struct OuterAdaptiveGate {
    /// Gate kernel pipeline (reads break_status, writes indirect args / counter)
    gate_pipeline: wgpu::ComputePipeline,
    gate_bg: wgpu::BindGroup,
    /// STOP-inject kernel pipeline (writes break_status into FGMRES scalars[SCALAR_STOP])
    stop_inject_pipeline: wgpu::ComputePipeline,
    /// STOP-inject kernel pipeline for CG (writes break_status into CG scalars[CG_SCALAR_STOP])
    cg_stop_inject_pipeline: wgpu::ComputePipeline,
    /// Indirect dispatch args for Cells-dispatched kernels (3 × u32)
    b_indirect_args_cells: std::sync::Arc<wgpu::Buffer>,
    /// Indirect dispatch args for Faces-dispatched kernels (3 × u32)
    b_indirect_args_faces: std::sync::Arc<wgpu::Buffer>,
    /// GPU-side iteration counter (atomically incremented by gate kernel)
    b_iter_counter: std::sync::Arc<wgpu::Buffer>,
}

impl OuterAdaptiveGate {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        num_cells: u32,
        num_faces: u32,
        max_workgroups_per_dim: u32,
        b_break_status: &wgpu::Buffer,
    ) -> Self {
        // Compute real dispatch args using the same formula as GeneratedKernelsModule
        const WORKGROUP_SIZE_X: u32 = 64;
        let compute_dispatch = |items: u32| -> [u32; 3] {
            let groups = items.div_ceil(WORKGROUP_SIZE_X);
            if groups <= max_workgroups_per_dim {
                [groups.max(1), 1, 1]
            } else {
                let x = max_workgroups_per_dim;
                let y = groups.div_ceil(x);
                [x, y, 1]
            }
        };
        let real_cells = compute_dispatch(num_cells);
        let real_faces = compute_dispatch(num_faces);

        let b_real_args_cells = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outer_gate:real_args_cells"),
            size: 12,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&b_real_args_cells, 0, bytemuck::cast_slice(&real_cells));

        let b_real_args_faces = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outer_gate:real_args_faces"),
            size: 12,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&b_real_args_faces, 0, bytemuck::cast_slice(&real_faces));

        let b_indirect_args_cells =
            std::sync::Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("outer_gate:indirect_args_cells"),
                size: 12,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
                mapped_at_creation: false,
            }));

        let b_indirect_args_faces =
            std::sync::Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("outer_gate:indirect_args_faces"),
                size: 12,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
                mapped_at_creation: false,
            }));

        let b_iter_counter = std::sync::Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outer_gate:iter_counter"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        // Gate pipeline
        let gate_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("outer_gate:gate"),
            source: wgpu::ShaderSource::Wgsl(build_outer_gate_wgsl().into()),
        });
        let gate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("outer_gate:gate"),
            layout: None,
            module: &gate_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let gate_bgl = gate_pipeline.get_bind_group_layout(0);
        let gate_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("outer_gate:gate_bg"),
            layout: &gate_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_break_status.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_real_args_cells.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_indirect_args_cells.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_real_args_faces.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: b_indirect_args_faces.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: b_iter_counter.as_entire_binding(),
                },
            ],
        });

        // STOP-inject pipeline (bind group created per-FGMRES workspace)
        let stop_inject_src = build_outer_stop_inject_wgsl(FGMRES_SCALAR_STOP);
        let stop_inject_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("outer_gate:stop_inject"),
            source: wgpu::ShaderSource::Wgsl(stop_inject_src.into()),
        });
        let stop_inject_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("outer_gate:stop_inject"),
                layout: None,
                module: &stop_inject_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // STOP-inject pipeline for CG (same shader, different scalar offset)
        let cg_stop_inject_src = build_outer_stop_inject_wgsl(CG_SCALAR_STOP);
        let cg_stop_inject_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("outer_gate:cg_stop_inject"),
            source: wgpu::ShaderSource::Wgsl(cg_stop_inject_src.into()),
        });
        let cg_stop_inject_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("outer_gate:cg_stop_inject"),
                layout: None,
                module: &cg_stop_inject_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            gate_pipeline,
            gate_bg,
            stop_inject_pipeline,
            cg_stop_inject_pipeline,
            b_indirect_args_cells,
            b_indirect_args_faces,
            b_iter_counter,
        }
    }

    /// Create a bind group for the STOP-inject kernel with a specific FGMRES scalars buffer.
    fn create_stop_inject_bind_group(
        &self,
        device: &wgpu::Device,
        b_break_status: &wgpu::Buffer,
        b_scalars: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let bgl = self.stop_inject_pipeline.get_bind_group_layout(0);
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("outer_gate:stop_inject_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_break_status.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_scalars.as_entire_binding(),
                },
            ],
        })
    }

    /// Encode the gate kernel dispatch into a command encoder.
    fn encode_gate_into(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("outer_gate:gate"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.gate_pipeline);
        pass.set_bind_group(0, &self.gate_bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Encode the STOP-inject kernel dispatch into a command encoder.
    fn encode_stop_inject_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        stop_inject_bg: &wgpu::BindGroup,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("outer_gate:stop_inject"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.stop_inject_pipeline);
        pass.set_bind_group(0, stop_inject_bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Create a bind group for the CG STOP-inject kernel with a specific CG scalars buffer.
    fn create_stop_inject_bind_group_cg(
        &self,
        device: &wgpu::Device,
        b_break_status: &wgpu::Buffer,
        b_scalars: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let bgl = self.cg_stop_inject_pipeline.get_bind_group_layout(0);
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("outer_gate:cg_stop_inject_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_break_status.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_scalars.as_entire_binding(),
                },
            ],
        })
    }

    /// Encode the CG STOP-inject kernel dispatch into a command encoder.
    fn encode_stop_inject_cg_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        stop_inject_bg: &wgpu::BindGroup,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("outer_gate:cg_stop_inject"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.cg_stop_inject_pipeline);
        pass.set_bind_group(0, stop_inject_bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
}

/// Constant for FGMRES scalar STOP index, used in STOP-inject WGSL.
use crate::solver::gpu::linear_solver::fgmres::FGMRES_SCALAR_STOP;
/// Constant for CG scalar STOP index, used in STOP-inject WGSL.
use crate::solver::gpu::modules::scalar_cg::CG_SCALAR_STOP;

struct OuterConvergenceMonitor {
    target_names: Vec<String>,
    pipeline: wgpu::ComputePipeline,
    break_pipeline: wgpu::ComputePipeline,
    _b_params_x: wgpu::Buffer,
    b_params_state: wgpu::Buffer,
    _b_descs_x: wgpu::Buffer,
    b_descs_state: wgpu::Buffer,
    b_out_bits: wgpu::Buffer,
    b_delta: wgpu::Buffer,
    b_scale: wgpu::Buffer,
    b_break_status: wgpu::Buffer,
    b_break_params: wgpu::Buffer,
    bg_x: wgpu::BindGroup,
    break_bg: wgpu::BindGroup,
    zero_out_words: Vec<u32>,
    dispatch_cells: u32,
    state_scale: Option<Vec<f32>>,
    state_scale_ready: bool,
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
            let comps = kind.component_count();
            if comps == 0 {
                continue;
            }
            if comps > 4 {
                return Err(format!(
                    "outer convergence monitor only supports up to 4 components per target (got {comps} for '{name}')"
                ));
            }

            let mut offsets_x = [0u32; 4];
            for (comp, offset) in offsets_x.iter_mut().enumerate().take(comps) {
                *offset = unknown_offset_cursor + comp as u32;
            }
            unknown_offset_cursor += comps as u32;

            // Get state offsets from the pre-resolved mapping
            let mut offsets_state = [0u32; 4];
            let mut has_all_offsets = true;
            for (comp, offset) in offsets_state.iter_mut().enumerate().take(comps) {
                match unknown_mapping.get_offset(eq_idx, comp) {
                    Some(off) => *offset = off,
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
        let break_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("outer_convergence:break"),
            source: wgpu::ShaderSource::Wgsl(build_outer_convergence_break_wgsl().into()),
        });
        let break_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("outer_convergence:break"),
            layout: None,
            module: &break_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

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
        queue.write_buffer(&b_descs_state, 0, bytemuck::cast_slice(&target_descs_state));

        let b_out_bits = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outer_convergence:out_bits"),
            size: (num_targets as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let b_delta = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outer_convergence:delta"),
            size: (num_targets as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let b_scale = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outer_convergence:scale"),
            size: (num_targets as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let b_break_status = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outer_convergence:break_status"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let b_break_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outer_convergence:break_params"),
            size: std::mem::size_of::<GpuOuterConvergenceBreakParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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
        let break_bgl = break_pipeline.get_bind_group_layout(0);
        let break_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("outer_convergence:break_bg"),
            layout: &break_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_delta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_scale.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_break_status.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_break_params.as_entire_binding(),
                },
            ],
        });

        let zero_out_words = vec![0u32; target_descs_x.len()];
        let dispatch_cells = num_cells.div_ceil(OUTER_CONVERGENCE_WORKGROUP_SIZE);

        Ok(Some(Self {
            target_names,
            pipeline,
            break_pipeline,
            _b_params_x: b_params,
            b_params_state,
            _b_descs_x: b_descs_x,
            b_descs_state,
            b_out_bits,
            b_delta,
            b_scale,
            b_break_status,
            b_break_params,
            bg_x,
            break_bg,
            zero_out_words,
            dispatch_cells,
            state_scale: None,
            state_scale_ready: false,
        }))
    }

    fn reset_step(&mut self) {
        self.state_scale = None;
        self.state_scale_ready = false;
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
        if !scale.is_empty() {
            plan.context
                .queue
                .write_buffer(&self.b_scale, 0, bytemuck::cast_slice(&scale));
        }
        self.state_scale = Some(scale);
        self.state_scale_ready = true;
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
        let bg = plan
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
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

        let mut encoder =
            plan.context
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

        let out_bytes = (self.zero_out_words.len() as u64) * 4;
        let staging_buffer = plan.staging_cache.take_or_create(
            &plan.context.device,
            out_bytes,
            "outer_convergence:out_bits (cached)",
        );
        encoder.copy_buffer_to_buffer(&self.b_out_bits, 0, &staging_buffer, 0, out_bytes);
        let submission_index = plan.context.queue.submit(Some(encoder.finish()));
        crate::count_submission!("Generic Coupled", label_prefix);

        let raw_result: Result<Vec<u8>, String> = (|| {
            let slice = staging_buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });

            let _ = plan.context.device.poll(wgpu::PollType::Wait {
                submission_index: Some(submission_index),
                timeout: None,
            });

            let map_result = rx
                .recv()
                .map_err(|_| "outer convergence readback map channel closed".to_string())?;
            map_result.map_err(|err| format!("outer convergence readback map failed: {err:?}"))?;

            let data = slice.get_mapped_range();
            let raw = data.to_vec();
            drop(data);
            staging_buffer.unmap();
            Ok(raw)
        })();
        plan.staging_cache.put(out_bytes, staging_buffer);
        let raw = raw_result?;

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

    fn submit_break_eval_and_read_status(
        &self,
        plan: &GpuProgramPlan,
        mut encoder: wgpu::CommandEncoder,
        submission_label: &'static str,
    ) -> Result<bool, String> {
        let out_bytes = 4u64;
        let staging_buffer = plan.staging_cache.take_or_create(
            &plan.context.device,
            out_bytes,
            "outer_convergence:break_status (cached)",
        );
        encoder.copy_buffer_to_buffer(&self.b_break_status, 0, &staging_buffer, 0, out_bytes);
        let submission_index = plan.context.queue.submit(Some(encoder.finish()));
        crate::count_submission!("Generic Coupled", submission_label);

        let raw_result: Result<Vec<u8>, String> = (|| {
            let slice = staging_buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });

            let _ = plan.context.device.poll(wgpu::PollType::Wait {
                submission_index: Some(submission_index),
                timeout: None,
            });

            let map_result = rx
                .recv()
                .map_err(|_| "outer convergence break map channel closed".to_string())?;
            map_result.map_err(|err| format!("outer convergence break map failed: {err:?}"))?;

            let data = slice.get_mapped_range();
            let raw = data.to_vec();
            drop(data);
            staging_buffer.unmap();
            Ok(raw)
        })();
        plan.staging_cache.put(out_bytes, staging_buffer);
        let raw = raw_result?;

        if raw.len() != out_bytes as usize {
            return Err(format!(
                "outer convergence break readback size mismatch: got {} expected {}",
                raw.len(),
                out_bytes
            ));
        }
        let words: &[u32] = bytemuck::cast_slice(&raw);
        Ok(words.first().copied().unwrap_or(0) != 0)
    }

    fn evaluate_break_from_current_buffers(
        &self,
        plan: &GpuProgramPlan,
        tol_rel: f32,
        tol_abs: f32,
    ) -> Result<bool, String> {
        let params = GpuOuterConvergenceBreakParams {
            count: self.target_names.len() as u32,
            tol_rel,
            tol_abs,
            _pad0: 0,
        };
        plan.context
            .queue
            .write_buffer(&self.b_break_params, 0, bytes_of(&params));

        let mut encoder =
            plan.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("outer_convergence:break_eval_buffered"),
                });
        self.encode_break_eval_into(&mut encoder);
        self.submit_break_eval_and_read_status(
            plan,
            encoder,
            "outer_convergence:break_eval_buffered",
        )
    }

    fn evaluate_break_from_state_on_gpu(
        &mut self,
        plan: &GpuProgramPlan,
        state: &wgpu::Buffer,
        tol_rel: f32,
        tol_abs: f32,
    ) -> Result<bool, String> {
        let params = GpuOuterConvergenceBreakParams {
            count: self.target_names.len() as u32,
            tol_rel,
            tol_abs,
            _pad0: 0,
        };
        plan.context
            .queue
            .write_buffer(&self.b_break_params, 0, bytes_of(&params));

        let mut encoder =
            plan.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("outer_convergence:break_eval_no_readback"),
                });

        let seeded_scale = if self.state_scale_ready {
            false
        } else {
            let bg_state = self.create_state_bind_group(&plan.context.device, state);
            self.encode_state_scale_into(&mut encoder, &bg_state);
            true
        };

        self.encode_delta_maxima_into(&mut encoder);
        self.encode_break_eval_into(&mut encoder);

        let converged = self.submit_break_eval_and_read_status(
            plan,
            encoder,
            "outer_convergence:break_eval_no_readback",
        )?;
        if seeded_scale {
            self.state_scale_ready = true;
        }
        Ok(converged)
    }

    // --- Encode-only methods for GPU-driven adaptive outer break ---

    /// Create a bind group for the reduction pipeline bound to the state buffer.
    /// Call this once before the one-submission encoder loop.
    fn create_state_bind_group(
        &self,
        device: &wgpu::Device,
        state: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let bgl = self.pipeline.get_bind_group_layout(0);
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("outer_convergence:bg_state"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.b_descs_state.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.b_out_bits.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.b_params_state.as_entire_binding(),
                },
            ],
        })
    }

    /// Upload break parameters into the GPU buffer. Call once before the encoder loop.
    fn upload_break_params(&self, queue: &wgpu::Queue, tol_rel: f32, tol_abs: f32) {
        let params = GpuOuterConvergenceBreakParams {
            count: self.target_names.len() as u32,
            tol_rel,
            tol_abs,
            _pad0: 0,
        };
        queue.write_buffer(&self.b_break_params, 0, bytes_of(&params));
    }

    /// Encode the delta-maxima reduction: clear out_bits, dispatch reduction, copy → b_delta.
    fn encode_delta_maxima_into(&self, encoder: &mut wgpu::CommandEncoder) {
        let out_bytes = (self.zero_out_words.len() as u64) * 4;
        encoder.clear_buffer(&self.b_out_bits, 0, Some(out_bytes));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("outer_convergence:delta_encode"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bg_x, &[]);
            pass.dispatch_workgroups(self.dispatch_cells.max(1), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.b_out_bits, 0, &self.b_delta, 0, out_bytes);
    }

    /// Encode the state-scale reduction: clear out_bits, dispatch reduction, copy → b_scale.
    fn encode_state_scale_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bg_state: &wgpu::BindGroup,
    ) {
        let out_bytes = (self.zero_out_words.len() as u64) * 4;
        encoder.clear_buffer(&self.b_out_bits, 0, Some(out_bytes));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("outer_convergence:scale_encode"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bg_state, &[]);
            pass.dispatch_workgroups(self.dispatch_cells.max(1), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.b_out_bits, 0, &self.b_scale, 0, out_bytes);
    }

    /// Encode the break evaluation: clear break_status, dispatch break kernel.
    fn encode_break_eval_into(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.b_break_status, 0, Some(4));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("outer_convergence:break_eval_encode"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.break_pipeline);
            pass.set_bind_group(0, &self.break_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }

    /// Encode a full convergence check sequence for one outer iteration.
    ///
    /// If `first_iter` is true, also encodes the state-scale reduction.
    /// After this, `b_break_status` contains the convergence result on the GPU.
    fn encode_convergence_check(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bg_state: &wgpu::BindGroup,
        first_iter: bool,
    ) {
        if first_iter {
            self.encode_state_scale_into(encoder, bg_state);
        }
        self.encode_delta_maxima_into(encoder);
        self.encode_break_eval_into(encoder);
    }
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

        let outer_gate = outer_convergence.as_ref().map(|oc| {
            OuterAdaptiveGate::new(
                &runtime.common.context.device,
                &runtime.common.context.queue,
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
            outer_tol: 1e-3,
            outer_tol_abs: 1e-6,
            outer_break_enabled: DEFAULT_OUTER_BREAK_ENABLED,
            outer_batched_mode: DEFAULT_OUTER_BATCHED_MODE,
            nonconverged_relax: 1.0,
            implicit_base_alpha_u: None,
            linear_solver,
            schur,
            krylov,
            outer_convergence,
            outer_gate,
            _b_bc_kind: b_bc_kind,
            _b_bc_value: b_bc_value,
            boundary_faces,
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
        let comps = target.kind().component_count();

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
                    let idx = unknown_mapping.get_offset(eq_idx, comp).ok_or_else(|| {
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

    let b_precond_params = device.create_buffer(&wgpu::BufferDescriptor {
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
        recipe.linear_solver.update_strategy,
        system,
        precond_bindings,
        "generic_coupled",
    );

    let precond = GenericCoupledSchurPreconditioner::new(
        device,
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
        },
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
        recipe.linear_solver.update_strategy,
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
        RuntimePreconditionerModule::new(device, precond_inputs),
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

    // Per-face BC storage: apply the boundary value to all boundary faces of this type.
    // (Boundary index 0 is reserved for "None" and should have no boundary faces.)
    let faces = res(plan)
        .boundary_faces
        .get(boundary_idx as usize)
        .ok_or_else(|| format!("missing boundary_faces[{boundary_idx}]"))?;
    for &face_idx in faces {
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
    plan.outer_field_residuals.clear();
    plan.outer_field_residuals_scaled.clear();
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
    crate::count_submission!("Generic Coupled", "pre_step_copy");
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
    // Enable encoded basis seeding by default for multi-outer host-driven solves.
    // Keep single-outer implicit solves opt-in via env var because that path is
    // still more sensitive in OpenFOAM parity diagnostics.
    let use_encoded_seed_basis0 = encoded_seed_basis0_enabled(r.outer_iters > 1);

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
            &mut schur.solver,
            SolveFgmresArgs {
                context: &context,
                system,
                n: r.runtime.num_dofs,
                num_cells: r.runtime.common.num_cells,
                dispatch: schur.dispatch,
                max_restart,
                max_iters: r.linear_solver.max_iters,
                tol: r.linear_solver.tolerance,
                tol_abs: r.linear_solver.tolerance_abs,
                precond_label: "generic_coupled:schur",
                use_encoded_seed_basis0,
            },
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
            &mut krylov.solver,
            SolveFgmresArgs {
                context: &context,
                system,
                n: r.runtime.num_dofs,
                num_cells: r.runtime.common.num_cells,
                dispatch: krylov.dispatch,
                max_restart: max_restart.max(1),
                max_iters: r.linear_solver.max_iters,
                tol: r.linear_solver.tolerance,
                tol_abs: r.linear_solver.tolerance_abs,
                precond_label: "generic_coupled:fgmres",
                use_encoded_seed_basis0,
            },
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

    let (outer_iters, outer_break_enabled, collect_convergence_stats) = {
        let r = res(plan);
        (
            r.outer_iters,
            r.outer_break_enabled,
            plan.collect_convergence_stats,
        )
    };

    let break_should_run = outer_break_enabled
        && outer_iters > 1
        && plan.last_linear_stats.converged
        && !plan.repeat_break;

    if !break_should_run && !collect_convergence_stats {
        return;
    }

    if collect_convergence_stats {
        let (delta, scale) = match compute_outer_residuals(plan) {
            Some(result) => result,
            None => return,
        };
        if !break_should_run {
            return;
        }
        let Some(ref scale) = scale else {
            return;
        };
        if scale.len() != delta.len() {
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
                if converged {
                    plan.repeat_break = true;
                }
            }
            Err(err) => {
                eprintln!("[cfd2][outer] failed to evaluate break status on gpu: {err}");
            }
        }

        res_mut(plan).outer_convergence = Some(monitor);
        return;
    }

    if !break_should_run {
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
            if converged {
                plan.repeat_break = true;
            }
        }
        Err(err) => {
            eprintln!("[cfd2][outer] failed to evaluate break status on gpu: {err}");
        }
    }

    res_mut(plan).outer_convergence = Some(monitor);
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

    // Store absolute residuals
    plan.outer_field_residuals.clear();
    plan.outer_field_residuals.extend(
        monitor
            .target_names()
            .iter()
            .cloned()
            .zip(delta.iter().copied()),
    );

    // Store scaled residuals (normalized by state scale)
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

    res_mut(plan).outer_convergence = Some(monitor);
    Some((delta, scale))
}

pub(crate) fn host_coupled_batch_tail(plan: &mut GpuProgramPlan) {
    let (outer_batched_mode, outer_iters) = {
        let r = res(plan);
        (r.outer_batched_mode, r.outer_iters.max(1))
    };
    if !outer_batched_mode || outer_iters <= 1 {
        return;
    }

    // This host op is placed at the end of the per-iteration block.
    // When enabled, run all remaining fixed outer iterations here and break
    // the recipe-level repeat loop to avoid duplicated outer passes.
    let iters_done = plan.step_linear_stats.len();
    if iters_done != 1 {
        return;
    }

    let remaining = outer_iters.saturating_sub(iters_done);
    if remaining == 0 {
        return;
    }

    // Encode all remaining outer iterations into one submission.
    // When adaptive break is enabled, uses indirect dispatch + GPU-side convergence
    // gating so converged iterations become zero-cost dispatches.
    if !try_host_coupled_batch_tail_one_submission(plan, remaining) {
        // One-submission encode failed (unsupported/inconsistent solver state).
        // Ensure the recipe-level repeat loop remains active so remaining outer
        // iterations execute through the standard per-iteration path.
        plan.repeat_break = false;
        eprintln!(
            "[cfd2][batch_tail] one-submission batch tail could not be used \
             (unsupported solver config); falling back to recipe-level per-iteration loop"
        );
    }
}

fn try_host_coupled_batch_tail_one_submission(plan: &mut GpuProgramPlan, remaining: usize) -> bool {
    if remaining == 0 {
        return false;
    }

    let device = plan.context.device.clone();
    let queue = plan.context.queue.clone();
    let context = crate::solver::gpu::context::GpuContext {
        device: device.clone(),
        queue: queue.clone(),
    };

    let start = std::time::Instant::now();

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

        // Determine if adaptive outer break is available
        let use_adaptive = r.outer_break_enabled
            && r.outer_gate.is_some()
            && r.outer_convergence.is_some()
            && remaining > 1;

        // Prepare adaptive resources (indirect graphs, bind groups, etc.)
        let adaptive_resources = if use_adaptive {
            let gate = r.outer_gate.as_ref().unwrap();
            let monitor = r.outer_convergence.as_ref().unwrap();

            // Create indirect-dispatch variants of assembly and update graphs
            let indirect_cells = gate.b_indirect_args_cells.clone();
            let indirect_faces = gate.b_indirect_args_faces.clone();
            let assembly_graph_indirect =
                assembly_graph.clone_with_indirect_dispatch(|kind| match kind {
                    DispatchKind::Faces => (indirect_faces.clone(), 0),
                    _ => (indirect_cells.clone(), 0),
                });
            let update_graph_indirect =
                update_graph.clone_with_indirect_dispatch(|kind| match kind {
                    DispatchKind::Faces => (indirect_faces.clone(), 0),
                    _ => (indirect_cells.clone(), 0),
                });

            // Create state bind group for convergence check
            let state = r.fields.current_state();
            let bg_state = monitor.create_state_bind_group(&device, state);

            // Upload break params
            monitor.upload_break_params(&queue, r.outer_tol, r.outer_tol_abs);

            // Clear iteration counter
            let zero: u32 = 0;
            queue.write_buffer(&gate.b_iter_counter, 0, bytemuck::bytes_of(&zero));

            // Create STOP-inject bind group from the solver's scalars buffer
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

        // Use chunked submission to avoid Metal hangs.  Each FGMRES restart
        // chunk gets its own encoder → submit cycle.  Assembly is prepended to
        // the first chunk and update is appended to the last chunk of each
        // outer iteration.
        for iter_idx in 0..remaining {
            let is_indirect = use_adaptive && iter_idx > 0;

            let mut pre = |encoder: &mut wgpu::CommandEncoder| {
                if let Some((ref asm_indirect, _, _, ref stop_bg)) = adaptive_resources {
                    if is_indirect {
                        // Inject STOP scalar so the linear solver becomes zero-cost when converged
                        let gate = r.outer_gate.as_ref().unwrap();
                        if solver_is_cg {
                            gate.encode_stop_inject_cg_into(encoder, stop_bg);
                        } else {
                            gate.encode_stop_inject_into(encoder, stop_bg);
                        }
                        asm_indirect.encode_into(encoder, kernels, runtime_dims);
                        return;
                    }
                }
                assembly_graph.encode_into(encoder, kernels, runtime_dims);
            };
            let mut post = |encoder: &mut wgpu::CommandEncoder| {
                if let Some((_, ref upd_indirect, ref bg_state, _)) = adaptive_resources {
                    if is_indirect {
                        upd_indirect.encode_into(encoder, kernels, runtime_dims);
                    } else {
                        update_graph.encode_into(encoder, kernels, runtime_dims);
                    }
                    // Convergence check + gate after every iteration in adaptive mode
                    let monitor = r.outer_convergence.as_ref().unwrap();
                    let gate = r.outer_gate.as_ref().unwrap();
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
                        tol,
                        tol_abs,
                        precond_label: "generic_coupled:schur(batch_tail)",
                        use_encoded_seed_basis0: true,
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
                        tol,
                        tol_abs,
                        precond_label: "generic_coupled:fgmres(batch_tail)",
                        use_encoded_seed_basis0: true,
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

#[cfg(test)]
mod tests {
    use super::*;
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
        let mut model = incompressible_momentum_model();
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
    fn batch_tail_fallback_clears_repeat_break_for_unsupported_solver_state() {
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
        let model = incompressible_momentum_model();

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
        plan.step_linear_stats.push(
            crate::solver::gpu::structs::LinearSolverStats::max_iterations(
                1,
                1.0,
                std::time::Duration::ZERO,
            ),
        );
        plan.repeat_break = true;

        host_coupled_batch_tail(&mut plan);

        assert!(
            !plan.repeat_break,
            "fallback path must clear repeat_break so recipe-level loop continues"
        );
        assert_eq!(
            plan.step_linear_stats.len(),
            1,
            "fallback path should keep existing per-iteration stats"
        );
    }
}
