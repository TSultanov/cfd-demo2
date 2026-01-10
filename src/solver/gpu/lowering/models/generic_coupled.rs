use crate::solver::gpu::execution_plan::{run_module_graph, GraphDetail, GraphExecMode};
use crate::solver::gpu::modules::generic_coupled_kernels::{
    GenericCoupledBindGroups, GenericCoupledKernelsModule, GenericCoupledPipeline,
};
use crate::solver::gpu::modules::graph::{
    ComputeSpec, DispatchKind, ModuleGraph, ModuleNode, RuntimeDims,
};
use crate::solver::gpu::modules::unified_field_resources::UnifiedFieldResources;
use crate::solver::gpu::plans::plan_instance::{PlanFuture, PlanLinearSystemDebug, PlanParamValue};
use crate::solver::gpu::plans::program::{GpuProgramPlan, ProgramOpRegistry};
use crate::solver::gpu::runtime::GpuScalarRuntime;
use crate::solver::gpu::structs::LinearSolverStats;

use crate::solver::gpu::lowering::templates::generic_coupled_scalar as op_ids;

pub(crate) struct GenericCoupledProgramResources {
    runtime: GpuScalarRuntime,
    fields: UnifiedFieldResources,
    kernels: GenericCoupledKernelsModule,
    assembly_graph: ModuleGraph<GenericCoupledKernelsModule>,
    update_graph: ModuleGraph<GenericCoupledKernelsModule>,
    _b_bc_kind: wgpu::Buffer,
    _b_bc_value: wgpu::Buffer,
}

impl GenericCoupledProgramResources {
    pub(crate) fn new(
        runtime: GpuScalarRuntime,
        fields: UnifiedFieldResources,
        kernels: GenericCoupledKernelsModule,
        b_bc_kind: wgpu::Buffer,
        b_bc_value: wgpu::Buffer,
    ) -> Self {
        Self {
            runtime,
            fields,
            kernels,
            assembly_graph: build_assembly_graph(),
            update_graph: build_update_graph(),
            _b_bc_kind: b_bc_kind,
            _b_bc_value: b_bc_value,
        }
    }
}

impl GenericCoupledProgramResources {
    fn runtime_dims(&self) -> RuntimeDims {
        RuntimeDims {
            num_cells: self.runtime.common.num_cells,
            num_faces: 0,
        }
    }
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
        if n != self.runtime.common.num_cells {
            return Err(format!(
                "requested solve size {} does not match num_cells {}",
                n, self.runtime.common.num_cells
            ));
        }
        Ok(self
            .runtime
            .solve_linear_system_cg_with_size(n, max_iters, tol))
    }

    fn get_linear_solution(&self) -> PlanFuture<'_, Result<Vec<f32>, String>> {
        Box::pin(async move {
            self.runtime
                .get_linear_solution(self.runtime.common.num_cells)
                .await
        })
    }
}

fn res(plan: &GpuProgramPlan) -> &GenericCoupledProgramResources {
    plan.resources
        .get::<GenericCoupledProgramResources>()
        .expect("missing GenericCoupledProgramResources")
}

fn res_mut(plan: &mut GpuProgramPlan) -> &mut GenericCoupledProgramResources {
    plan.resources
        .get_mut::<GenericCoupledProgramResources>()
        .expect("missing GenericCoupledProgramResources")
}

pub(crate) fn register_ops(registry: &mut ProgramOpRegistry) -> Result<(), String> {
    registry.register_graph(op_ids::G_ASSEMBLY, assembly_graph_run)?;
    registry.register_graph(op_ids::G_UPDATE, update_graph_run)?;

    registry.register_host(op_ids::H_PREPARE, host_prepare_step)?;
    registry.register_host(op_ids::H_SOLVE, host_solve_linear_system)?;
    registry.register_host(op_ids::H_FINALIZE, host_finalize_step)?;

    Ok(())
}

pub(crate) fn spec_num_cells(plan: &GpuProgramPlan) -> u32 {
    res(plan).runtime.common.num_cells
}

pub(crate) fn spec_time(plan: &GpuProgramPlan) -> f32 {
    res(plan).runtime.time_integration.time as f32
}

pub(crate) fn spec_dt(plan: &GpuProgramPlan) -> f32 {
    res(plan).runtime.time_integration.dt
}

pub(crate) fn spec_state_buffer(plan: &GpuProgramPlan) -> &wgpu::Buffer {
    res(plan).fields.current_state()
}

pub(crate) fn spec_write_state_bytes(plan: &GpuProgramPlan, bytes: &[u8]) -> Result<(), String> {
    res(plan).fields.write_state_bytes(&plan.context.queue, bytes);
    Ok(())
}

pub(crate) fn host_prepare_step(plan: &mut GpuProgramPlan) {
    let r = res_mut(plan);
    r.fields.advance_step();
    r.runtime.advance_time();
}

pub(crate) fn host_finalize_step(plan: &mut GpuProgramPlan) {
    let queue = plan.context.queue.clone();
    let r = res_mut(plan);
    r.runtime
        .time_integration
        .finalize_step(&mut r.runtime.constants, &queue);
}

pub(crate) fn host_solve_linear_system(plan: &mut GpuProgramPlan) {
    let r = res(plan);
    let stats = r.runtime.solve_linear_system_cg(400, 1e-6);
    plan.last_linear_stats = stats;
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

pub(crate) fn update_graph_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let r = res(plan);
    run_module_graph(&r.update_graph, context, &r.kernels, r.runtime_dims(), mode)
}

pub(crate) fn param_dt(plan: &mut GpuProgramPlan, value: PlanParamValue) -> Result<(), String> {
    let PlanParamValue::F32(dt) = value else {
        return Err("invalid value type".into());
    };
    res_mut(plan).runtime.set_dt(dt);
    Ok(())
}

pub(crate) fn param_advection_scheme(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::Scheme(scheme) = value else {
        return Err("invalid value type".into());
    };
    res_mut(plan).runtime.set_scheme(scheme.gpu_id());
    Ok(())
}

pub(crate) fn param_time_scheme(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::TimeScheme(scheme) = value else {
        return Err("invalid value type".into());
    };
    res_mut(plan).runtime.set_time_scheme(scheme as u32);
    Ok(())
}

pub(crate) fn param_preconditioner(
    _plan: &mut GpuProgramPlan,
    _value: PlanParamValue,
) -> Result<(), String> {
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

pub(crate) fn linear_debug_provider(
    plan: &mut GpuProgramPlan,
) -> Option<&mut dyn PlanLinearSystemDebug> {
    Some(res_mut(plan) as &mut dyn PlanLinearSystemDebug)
}

fn build_assembly_graph() -> ModuleGraph<GenericCoupledKernelsModule> {
    ModuleGraph::new(vec![ModuleNode::Compute(ComputeSpec {
        label: "generic_coupled:assembly",
        pipeline: GenericCoupledPipeline::Assembly,
        bind: GenericCoupledBindGroups::Assembly,
        dispatch: DispatchKind::Cells,
    })])
}

fn build_update_graph() -> ModuleGraph<GenericCoupledKernelsModule> {
    ModuleGraph::new(vec![ModuleNode::Compute(ComputeSpec {
        label: "generic_coupled:update",
        pipeline: GenericCoupledPipeline::Update,
        bind: GenericCoupledBindGroups::Update,
        dispatch: DispatchKind::Cells,
    })])
}
