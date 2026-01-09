use crate::solver::gpu::execution_plan::{run_module_graph, GraphDetail, GraphExecMode};
use crate::solver::gpu::plans::compressible::CompressiblePlanResources;
use crate::solver::gpu::plans::plan_instance::{
    PlanFuture, PlanLinearSystemDebug, PlanParam, PlanParamValue, PlanStepStats,
};
use crate::solver::gpu::plans::program::{
    CondOpKind, CountOpKind, GpuProgramPlan, GraphOpKind, HostOpKind, ProgramOpRegistry,
};
use crate::solver::gpu::structs::LinearSolverStats;
use std::env;

pub(in crate::solver::gpu::lowering) struct CompressibleProgramResources {
    plan: CompressiblePlanResources,
    implicit_outer_idx: usize,
    implicit_stats: Vec<LinearSolverStats>,
}

impl CompressibleProgramResources {
    pub(in crate::solver::gpu::lowering) fn new(plan: CompressiblePlanResources) -> Self {
        Self {
            plan,
            implicit_outer_idx: 0,
            implicit_stats: Vec::new(),
        }
    }
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

fn res_wrap_mut(plan: &mut GpuProgramPlan) -> &mut CompressibleProgramResources {
    plan.resources
        .get_mut::<CompressibleProgramResources>()
        .expect("missing CompressibleProgramResources")
}

pub(in crate::solver::gpu::lowering) fn register_ops(
    registry: &mut ProgramOpRegistry,
) -> Result<(), String> {
    registry.register_graph(GraphOpKind::CompressibleExplicitGraph, explicit_graph_run)?;
    registry.register_graph(
        GraphOpKind::CompressibleImplicitGradAssembly,
        implicit_grad_assembly_graph_run,
    )?;
    registry.register_graph(GraphOpKind::CompressibleImplicitSnapshot, implicit_snapshot_run)?;
    registry.register_graph(GraphOpKind::CompressibleImplicitApply, implicit_apply_graph_run)?;
    registry.register_graph(
        GraphOpKind::CompressiblePrimitiveUpdate,
        primitive_update_graph_run,
    )?;

    registry.register_host(HostOpKind::CompressibleExplicitPrepare, host_explicit_prepare)?;
    registry.register_host(HostOpKind::CompressibleExplicitFinalize, host_explicit_finalize)?;
    registry.register_host(HostOpKind::CompressibleImplicitPrepare, host_implicit_prepare)?;
    registry.register_host(
        HostOpKind::CompressibleImplicitSetIterParams,
        host_implicit_set_iter_params,
    )?;
    registry.register_host(
        HostOpKind::CompressibleImplicitSolveFgmres,
        host_implicit_solve_fgmres,
    )?;
    registry.register_host(
        HostOpKind::CompressibleImplicitRecordStats,
        host_implicit_record_stats,
    )?;
    registry.register_host(
        HostOpKind::CompressibleImplicitSetAlpha,
        host_implicit_set_alpha_for_apply,
    )?;
    registry.register_host(
        HostOpKind::CompressibleImplicitRestoreAlpha,
        host_implicit_restore_alpha,
    )?;
    registry.register_host(
        HostOpKind::CompressibleImplicitAdvanceOuterIdx,
        host_implicit_advance_outer_idx,
    )?;
    registry.register_host(HostOpKind::CompressibleImplicitFinalize, host_implicit_finalize)?;

    registry.register_cond(CondOpKind::CompressibleShouldUseExplicit, should_use_explicit)?;
    registry.register_count(
        CountOpKind::CompressibleImplicitOuterIters,
        implicit_outer_iters,
    )?;

    Ok(())
}

pub(in crate::solver::gpu::lowering) fn spec_num_cells(plan: &GpuProgramPlan) -> u32 {
    res(plan).num_cells
}

pub(in crate::solver::gpu::lowering) fn spec_time(plan: &GpuProgramPlan) -> f32 {
    res(plan).time_integration.time as f32
}

pub(in crate::solver::gpu::lowering) fn spec_dt(plan: &GpuProgramPlan) -> f32 {
    res(plan).time_integration.dt
}

pub(in crate::solver::gpu::lowering) fn spec_state_buffer(plan: &GpuProgramPlan) -> &wgpu::Buffer {
    res(plan).fields.state.state()
}

pub(in crate::solver::gpu::lowering) fn spec_write_state_bytes(
    plan: &GpuProgramPlan,
    bytes: &[u8],
) -> Result<(), String> {
    res(plan).write_state_bytes(bytes);
    Ok(())
}

pub(in crate::solver::gpu::lowering) fn should_use_explicit(plan: &GpuProgramPlan) -> bool {
    res(plan).should_use_explicit()
}

pub(in crate::solver::gpu::lowering) fn host_explicit_prepare(plan: &mut GpuProgramPlan) {
    let solver = res_mut(plan);
    solver.advance_ping_pong_and_time();
    solver.pre_step_copy();
}

pub(in crate::solver::gpu::lowering) fn explicit_graph_run(
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

pub(in crate::solver::gpu::lowering) fn host_explicit_finalize(plan: &mut GpuProgramPlan) {
    res_mut(plan).finalize_dt_old();
}

pub(in crate::solver::gpu::lowering) fn implicit_outer_iters(plan: &GpuProgramPlan) -> usize {
    res(plan).outer_iters
}

pub(in crate::solver::gpu::lowering) fn host_implicit_prepare(plan: &mut GpuProgramPlan) {
    let solver = res_mut(plan);
    solver.advance_ping_pong_and_time();
    solver.pre_step_copy();
    solver.implicit_set_base_alpha();
    let wrap = res_wrap_mut(plan);
    wrap.implicit_outer_idx = 0;
    wrap.implicit_stats.clear();
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|val| val.parse().ok())
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    env::var(name)
        .ok()
        .and_then(|val| val.parse().ok())
        .unwrap_or(default)
}

pub(in crate::solver::gpu::lowering) fn host_implicit_set_iter_params(plan: &mut GpuProgramPlan) {
    let tol_base = env_f32("CFD2_COMP_FGMRES_TOL", 1e-8);
    let warm_scale = env_f32("CFD2_COMP_FGMRES_WARM_SCALE", 100.0).max(1.0);
    let warm_iters = env_usize("CFD2_COMP_FGMRES_WARM_ITERS", 4);
    let retry_scale = env_f32("CFD2_COMP_FGMRES_RETRY_SCALE", 0.5).clamp(0.0, 1.0);
    let max_restart = env_usize("CFD2_COMP_FGMRES_MAX_RESTART", 80).max(1);
    let retry_restart = env_usize("CFD2_COMP_FGMRES_RETRY_RESTART", 160).max(1);

    let idx = res_wrap_mut(plan).implicit_outer_idx;
    let tol = if idx < warm_iters {
        tol_base * warm_scale
    } else {
        tol_base
    };
    let retry_tol = (tol * retry_scale).min(tol_base);
    res_mut(plan).implicit_set_iteration_params(tol, retry_tol, max_restart, retry_restart);
}

pub(in crate::solver::gpu::lowering) fn implicit_grad_assembly_graph_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let solver = res(plan);
    run_module_graph(
        solver.implicit_grad_assembly_graph(),
        context,
        &solver.kernels,
        solver.runtime_dims(),
        mode,
    )
}

pub(in crate::solver::gpu::lowering) fn host_implicit_solve_fgmres(plan: &mut GpuProgramPlan) {
    res_mut(plan).implicit_solve_fgmres();
}

pub(in crate::solver::gpu::lowering) fn host_implicit_record_stats(plan: &mut GpuProgramPlan) {
    let stats = res(plan).implicit_last_stats();
    let wrap = res_wrap_mut(plan);
    wrap.implicit_stats.push(stats);
}

pub(in crate::solver::gpu::lowering) fn implicit_snapshot_run(
    plan: &GpuProgramPlan,
    _context: &crate::solver::gpu::context::GpuContext,
    _mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let start = std::time::Instant::now();
    res(plan).implicit_snapshot();
    (start.elapsed().as_secs_f64(), None)
}

pub(in crate::solver::gpu::lowering) fn host_implicit_set_alpha_for_apply(
    plan: &mut GpuProgramPlan,
) {
    res_mut(plan).implicit_set_alpha_for_apply();
}

pub(in crate::solver::gpu::lowering) fn implicit_apply_graph_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let solver = res(plan);
    run_module_graph(
        solver.implicit_apply_graph(),
        context,
        &solver.kernels,
        solver.runtime_dims(),
        mode,
    )
}

pub(in crate::solver::gpu::lowering) fn host_implicit_restore_alpha(plan: &mut GpuProgramPlan) {
    res_mut(plan).implicit_restore_alpha();
}

pub(in crate::solver::gpu::lowering) fn host_implicit_advance_outer_idx(plan: &mut GpuProgramPlan) {
    res_wrap_mut(plan).implicit_outer_idx += 1;
}

pub(in crate::solver::gpu::lowering) fn primitive_update_graph_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let solver = res(plan);
    run_module_graph(
        solver.primitive_update_graph(),
        context,
        &solver.kernels,
        solver.runtime_dims(),
        mode,
    )
}

pub(in crate::solver::gpu::lowering) fn host_implicit_finalize(plan: &mut GpuProgramPlan) {
    res_mut(plan).finalize_dt_old();
}

pub(in crate::solver::gpu::lowering) fn init_history(plan: &GpuProgramPlan) {
    res(plan).initialize_history();
}

pub(in crate::solver::gpu::lowering) fn step_stats(_plan: &GpuProgramPlan) -> PlanStepStats {
    PlanStepStats::default()
}

pub(in crate::solver::gpu::lowering) fn step_with_stats(
    plan: &mut GpuProgramPlan,
) -> Result<Vec<LinearSolverStats>, String> {
    if should_use_explicit(plan) {
        plan.step();
        Ok(Vec::new())
    } else {
        plan.step();
        Ok(std::mem::take(&mut res_wrap_mut(plan).implicit_stats))
    }
}

pub(in crate::solver::gpu::lowering) fn set_param_fallback(
    plan: &mut GpuProgramPlan,
    param: PlanParam,
    value: PlanParamValue,
) -> Result<(), String> {
    let solver = res_mut(plan);
    match (param, value) {
        (PlanParam::Dt, PlanParamValue::F32(dt)) => {
            solver.set_dt(dt);
            Ok(())
        }
        (PlanParam::AdvectionScheme, PlanParamValue::Scheme(scheme)) => {
            solver.set_scheme(scheme.gpu_id());
            Ok(())
        }
        (PlanParam::TimeScheme, PlanParamValue::TimeScheme(scheme)) => {
            solver.set_time_scheme(scheme as u32);
            Ok(())
        }
        (PlanParam::Preconditioner, PlanParamValue::Preconditioner(preconditioner)) => {
            solver.set_precond_type(preconditioner);
            Ok(())
        }
        (PlanParam::Viscosity, PlanParamValue::F32(mu)) => {
            solver.set_viscosity(mu);
            Ok(())
        }
        (PlanParam::AlphaU, PlanParamValue::F32(alpha)) => {
            solver.set_alpha_u(alpha);
            Ok(())
        }
        (PlanParam::InletVelocity, PlanParamValue::F32(velocity)) => {
            solver.set_inlet_velocity(velocity);
            Ok(())
        }
        (PlanParam::Dtau, PlanParamValue::F32(dtau)) => {
            solver.set_dtau(dtau);
            Ok(())
        }
        (PlanParam::OuterIters, PlanParamValue::Usize(iters)) => {
            solver.set_outer_iters(iters);
            Ok(())
        }
        (PlanParam::LowMachModel, PlanParamValue::LowMachModel(model)) => {
            solver.set_precond_model(model as u32);
            Ok(())
        }
        (PlanParam::LowMachThetaFloor, PlanParamValue::F32(theta)) => {
            solver.set_precond_theta_floor(theta);
            Ok(())
        }
        (PlanParam::NonconvergedRelax, PlanParamValue::F32(relax)) => {
            solver.set_nonconverged_relax(relax);
            Ok(())
        }
        (PlanParam::DetailedProfilingEnabled, PlanParamValue::Bool(enable)) => {
            if enable {
                solver.common.profiling_stats.enable();
            } else {
                solver.common.profiling_stats.disable();
            }
            Ok(())
        }
        _ => Err("parameter is not supported by this plan".into()),
    }
}

pub(in crate::solver::gpu::lowering) fn linear_debug_provider(
    plan: &mut GpuProgramPlan,
) -> Option<&mut dyn PlanLinearSystemDebug> {
    Some(plan.resources.get_mut::<CompressibleProgramResources>()? as &mut dyn PlanLinearSystemDebug)
}
