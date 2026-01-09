use crate::solver::gpu::plans::compressible::CompressiblePlanResources;
use crate::solver::gpu::plans::plan_instance::{
    GpuPlanInstance, PlanFuture, PlanLinearSystemDebug, PlanParam, PlanParamValue, PlanStepStats,
};
use crate::solver::gpu::plans::program::{
    GpuProgramPlan, ModelGpuProgramSpec, ProgramExecutionPlan, ProgramNode, ProgramResources,
    ProgramSetParamFallback, ProgramStepStatsFn, ProgramStepWithStatsFn,
};
use crate::solver::gpu::execution_plan::{run_module_graph, GraphDetail, GraphExecMode};
use crate::solver::gpu::structs::LinearSolverStats;
use crate::solver::mesh::Mesh;
use crate::solver::model::ModelSpec;
use std::sync::Arc;

struct CompressibleProgramResources {
    plan: CompressiblePlanResources,
}

impl PlanLinearSystemDebug for CompressibleProgramResources {
    fn set_linear_system(&self, matrix_values: &[f32], rhs: &[f32]) -> Result<(), String> {
        PlanLinearSystemDebug::set_linear_system(&self.plan, matrix_values, rhs)
    }

    fn solve_linear_system_with_size(
        &mut self,
        n: u32,
        max_iters: u32,
        tol: f32,
    ) -> Result<LinearSolverStats, String> {
        PlanLinearSystemDebug::solve_linear_system_with_size(&mut self.plan, n, max_iters, tol)
    }

    fn get_linear_solution(&self) -> PlanFuture<'_, Result<Vec<f32>, String>> {
        PlanLinearSystemDebug::get_linear_solution(&self.plan)
    }
}

fn res(plan: &GpuProgramPlan) -> &CompressiblePlanResources {
    &plan
        .resources
        .get::<CompressibleProgramResources>()
        .expect("missing CompressibleProgramResources")
        .plan
}

fn res_mut(plan: &mut GpuProgramPlan) -> &mut CompressiblePlanResources {
    &mut plan
        .resources
        .get_mut::<CompressibleProgramResources>()
        .expect("missing CompressibleProgramResources")
        .plan
}

fn spec_num_cells(plan: &GpuProgramPlan) -> u32 {
    res(plan).num_cells()
}

fn spec_time(plan: &GpuProgramPlan) -> f32 {
    res(plan).time()
}

fn spec_dt(plan: &GpuProgramPlan) -> f32 {
    res(plan).dt()
}

fn spec_state_buffer(plan: &GpuProgramPlan) -> &wgpu::Buffer {
    res(plan).state_buffer()
}

fn spec_write_state_bytes(plan: &GpuProgramPlan, bytes: &[u8]) -> Result<(), String> {
    GpuPlanInstance::write_state_bytes(res(plan), bytes)
}

fn host_step(plan: &mut GpuProgramPlan) {
    res_mut(plan).step();
}

fn should_use_explicit(plan: &GpuProgramPlan) -> bool {
    res(plan).should_use_explicit()
}

fn host_explicit_prepare(plan: &mut GpuProgramPlan) {
    let solver = res_mut(plan);
    solver.advance_ping_pong_and_time();
    solver.pre_step_copy();
}

fn explicit_graph_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let solver = res(plan);
    run_module_graph(
        solver.explicit_graph(),
        context,
        &solver.kernels,
        solver.runtime_dims(),
        mode,
    )
}

fn host_explicit_finalize(plan: &mut GpuProgramPlan) {
    res_mut(plan).finalize_dt_old();
}

fn init_history(plan: &GpuProgramPlan) {
    res(plan).initialize_history();
}

fn step_stats(plan: &GpuProgramPlan) -> PlanStepStats {
    res(plan).step_stats()
}

fn step_with_stats(plan: &mut GpuProgramPlan) -> Result<Vec<LinearSolverStats>, String> {
    if should_use_explicit(plan) {
        plan.step();
        Ok(Vec::new())
    } else {
        GpuPlanInstance::step_with_stats(res_mut(plan))
    }
}

fn set_param_fallback(
    plan: &mut GpuProgramPlan,
    param: PlanParam,
    value: PlanParamValue,
) -> Result<(), String> {
    res_mut(plan).set_param(param, value)
}

fn linear_debug_provider(plan: &mut GpuProgramPlan) -> Option<&mut dyn PlanLinearSystemDebug> {
    Some(
        plan.resources
            .get_mut::<CompressibleProgramResources>()?
            as &mut dyn PlanLinearSystemDebug,
    )
}

pub(crate) async fn lower_compressible_program(
    mesh: &Mesh,
    model: ModelSpec,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
) -> Result<Box<dyn GpuPlanInstance>, String> {
    let plan = CompressiblePlanResources::new(mesh, model.clone(), device, queue).await?;

    let context = crate::solver::gpu::context::GpuContext {
        device: plan.common.context.device.clone(),
        queue: plan.common.context.queue.clone(),
    };
    let profiling_stats = Arc::clone(&plan.common.profiling_stats);

    let mut resources = ProgramResources::new();
    resources.insert(CompressibleProgramResources { plan });

    let explicit_plan = Arc::new(ProgramExecutionPlan::new(vec![
        ProgramNode::Host {
            label: "compressible:explicit_prepare",
            run: host_explicit_prepare,
        },
        ProgramNode::Graph {
            label: "compressible:explicit_graph",
            run: explicit_graph_run,
            mode: GraphExecMode::SplitTimed,
        },
        ProgramNode::Host {
            label: "compressible:explicit_finalize",
            run: host_explicit_finalize,
        },
    ]));

    let implicit_plan = Arc::new(ProgramExecutionPlan::new(vec![ProgramNode::Host {
        label: "compressible:implicit_step_legacy",
        run: host_step,
    }]));

    let step = Arc::new(ProgramExecutionPlan::new(vec![ProgramNode::If {
        label: "compressible:select_step_path",
        cond: should_use_explicit,
        then_plan: explicit_plan,
        else_plan: Some(implicit_plan),
    }]));

    let spec = ModelGpuProgramSpec {
        num_cells: spec_num_cells,
        time: spec_time,
        dt: spec_dt,
        state_buffer: spec_state_buffer,
        write_state_bytes: spec_write_state_bytes,
        step,
        initialize_history: Some(init_history),
        params: std::collections::HashMap::new(),
        set_param_fallback: Some(set_param_fallback as ProgramSetParamFallback),
        step_stats: Some(step_stats as ProgramStepStatsFn),
        step_with_stats: Some(step_with_stats as ProgramStepWithStatsFn),
        linear_debug: Some(linear_debug_provider),
    };

    Ok(Box::new(GpuProgramPlan::new(
        model,
        context,
        profiling_stats,
        resources,
        spec,
    )))
}
