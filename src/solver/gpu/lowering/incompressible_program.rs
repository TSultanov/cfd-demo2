use crate::solver::gpu::plans::plan_instance::{
    GpuPlanInstance, PlanFuture, PlanLinearSystemDebug, PlanParam, PlanParamValue, PlanStepStats,
};
use crate::solver::gpu::plans::program::{
    GpuProgramPlan, ModelGpuProgramSpec, ProgramExecutionPlan, ProgramResources,
    ProgramSetParamFallback, ProgramStepStatsFn, ProgramStepWithStatsFn,
};
use crate::solver::gpu::structs::{GpuSolver, LinearSolverStats};
use crate::solver::mesh::Mesh;
use crate::solver::model::ModelSpec;
use std::sync::Arc;

struct IncompressibleProgramResources {
    plan: GpuSolver,
}

impl PlanLinearSystemDebug for IncompressibleProgramResources {
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

fn res(plan: &GpuProgramPlan) -> &GpuSolver {
    &plan
        .resources
        .get::<IncompressibleProgramResources>()
        .expect("missing IncompressibleProgramResources")
        .plan
}

fn res_mut(plan: &mut GpuProgramPlan) -> &mut GpuSolver {
    &mut plan
        .resources
        .get_mut::<IncompressibleProgramResources>()
        .expect("missing IncompressibleProgramResources")
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
    res(plan).write_state_bytes(bytes)
}

fn host_step(plan: &mut GpuProgramPlan) {
    res_mut(plan).step();
}

fn init_history(plan: &GpuProgramPlan) {
    res(plan).initialize_history();
}

fn step_stats(plan: &GpuProgramPlan) -> PlanStepStats {
    res(plan).step_stats()
}

fn step_with_stats(plan: &mut GpuProgramPlan) -> Result<Vec<LinearSolverStats>, String> {
    GpuPlanInstance::step_with_stats(res_mut(plan))
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
            .get_mut::<IncompressibleProgramResources>()?
            as &mut dyn PlanLinearSystemDebug,
    )
}

pub(crate) async fn lower_incompressible_program(
    mesh: &Mesh,
    model: ModelSpec,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
) -> Result<Box<dyn GpuPlanInstance>, String> {
    let plan = GpuSolver::new(mesh, model.clone(), device, queue).await?;

    let context = crate::solver::gpu::context::GpuContext {
        device: plan.common.context.device.clone(),
        queue: plan.common.context.queue.clone(),
    };
    let profiling_stats = Arc::clone(&plan.common.profiling_stats);

    let mut resources = ProgramResources::new();
    resources.insert(IncompressibleProgramResources { plan });

    let step = Arc::new(ProgramExecutionPlan::new(vec![
        crate::solver::gpu::plans::program::ProgramNode::Host {
            label: "incompressible:step",
            run: host_step,
        },
    ]));

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
