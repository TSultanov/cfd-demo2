use crate::solver::gpu::execution_plan::{GraphDetail, GraphExecMode};
use crate::solver::gpu::recipe::{SolverRecipe, SteppingMode};
use crate::solver::gpu::plans::compressible::CompressiblePlanResources;
use crate::solver::gpu::plans::plan_instance::{
    PlanFuture, PlanLinearSystemDebug, PlanParam, PlanParamValue, PlanStepStats,
};
use crate::solver::gpu::plans::program::{GpuProgramPlan, ProgramOpRegistry};
use crate::solver::gpu::structs::LinearSolverStats;
use std::env;

mod op_ids {
    use crate::solver::gpu::plans::program::{CondOpKind, CountOpKind, GraphOpKind, HostOpKind};

    pub(crate) const G_EXPLICIT_GRAPH: GraphOpKind = GraphOpKind("compressible:explicit_graph");
    pub(crate) const G_IMPLICIT_GRAD_ASSEMBLY: GraphOpKind =
        GraphOpKind("compressible:implicit_grad_assembly");
    pub(crate) const G_IMPLICIT_SNAPSHOT: GraphOpKind = GraphOpKind("compressible:implicit_snapshot");
    pub(crate) const G_IMPLICIT_APPLY: GraphOpKind = GraphOpKind("compressible:implicit_apply");
    pub(crate) const G_PRIMITIVE_UPDATE: GraphOpKind = GraphOpKind("compressible:primitive_update");

    pub(crate) const H_EXPLICIT_PREPARE: HostOpKind = HostOpKind("compressible:explicit_prepare");
    pub(crate) const H_EXPLICIT_FINALIZE: HostOpKind = HostOpKind("compressible:explicit_finalize");
    pub(crate) const H_IMPLICIT_PREPARE: HostOpKind = HostOpKind("compressible:implicit_prepare");
    pub(crate) const H_IMPLICIT_SET_ITER_PARAMS: HostOpKind =
        HostOpKind("compressible:implicit_set_iter_params");
    pub(crate) const H_IMPLICIT_SOLVE_FGMRES: HostOpKind =
        HostOpKind("compressible:implicit_solve_fgmres");
    pub(crate) const H_IMPLICIT_RECORD_STATS: HostOpKind =
        HostOpKind("compressible:implicit_record_stats");
    pub(crate) const H_IMPLICIT_SET_ALPHA: HostOpKind = HostOpKind("compressible:implicit_set_alpha");
    pub(crate) const H_IMPLICIT_RESTORE_ALPHA: HostOpKind =
        HostOpKind("compressible:implicit_restore_alpha");
    pub(crate) const H_IMPLICIT_ADVANCE_OUTER_IDX: HostOpKind =
        HostOpKind("compressible:implicit_advance_outer_idx");
    pub(crate) const H_IMPLICIT_FINALIZE: HostOpKind = HostOpKind("compressible:implicit_finalize");

    pub(crate) const C_SHOULD_USE_EXPLICIT: CondOpKind = CondOpKind("compressible:should_use_explicit");
    pub(crate) const N_IMPLICIT_OUTER_ITERS: CountOpKind =
        CountOpKind("compressible:implicit_outer_iters");
}

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
    registry.register_graph(op_ids::G_EXPLICIT_GRAPH, explicit_graph_run)?;
    registry.register_graph(
        op_ids::G_IMPLICIT_GRAD_ASSEMBLY,
        implicit_grad_assembly_graph_run,
    )?;
    registry.register_graph(op_ids::G_IMPLICIT_SNAPSHOT, implicit_snapshot_run)?;
    registry.register_graph(op_ids::G_IMPLICIT_APPLY, implicit_apply_graph_run)?;
    registry.register_graph(
        op_ids::G_PRIMITIVE_UPDATE,
        primitive_update_graph_run,
    )?;

    registry.register_host(op_ids::H_EXPLICIT_PREPARE, host_explicit_prepare)?;
    registry.register_host(op_ids::H_EXPLICIT_FINALIZE, host_explicit_finalize)?;
    registry.register_host(op_ids::H_IMPLICIT_PREPARE, host_implicit_prepare)?;
    registry.register_host(
        op_ids::H_IMPLICIT_SET_ITER_PARAMS,
        host_implicit_set_iter_params,
    )?;
    registry.register_host(
        op_ids::H_IMPLICIT_SOLVE_FGMRES,
        host_implicit_solve_fgmres,
    )?;
    registry.register_host(
        op_ids::H_IMPLICIT_RECORD_STATS,
        host_implicit_record_stats,
    )?;
    registry.register_host(
        op_ids::H_IMPLICIT_SET_ALPHA,
        host_implicit_set_alpha_for_apply,
    )?;
    registry.register_host(
        op_ids::H_IMPLICIT_RESTORE_ALPHA,
        host_implicit_restore_alpha,
    )?;
    registry.register_host(
        op_ids::H_IMPLICIT_ADVANCE_OUTER_IDX,
        host_implicit_advance_outer_idx,
    )?;
    registry.register_host(op_ids::H_IMPLICIT_FINALIZE, host_implicit_finalize)?;

    registry.register_cond(op_ids::C_SHOULD_USE_EXPLICIT, should_use_explicit)?;
    registry.register_count(
        op_ids::N_IMPLICIT_OUTER_ITERS,
        implicit_outer_iters,
    )?;

    Ok(())
}

/// Register ops based on the solver recipe.
///
/// Today the compressible program spec is still template-based and expects the full
/// set of compressible ops (explicit + implicit + primitive recovery). We use the
/// recipe here primarily as a correctness check and as a hook point for future
/// recipe-driven op selection.
pub(in crate::solver::gpu::lowering) fn register_ops_from_recipe(
    recipe: &SolverRecipe,
    registry: &mut ProgramOpRegistry,
) -> Result<(), String> {
    if matches!(recipe.stepping, SteppingMode::Coupled { .. }) {
        return Err(format!(
            "compressible lowering does not support coupled stepping (recipe.model_id='{}')",
            recipe.model_id
        ));
    }

    // For now, the template spec requires all ops to be registered.
    register_ops(registry)
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
    solver.graphs.run_explicit(
        context,
        &solver.kernels,
        solver.runtime_dims(),
        mode,
        solver.needs_gradients,
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
    solver.graphs.run_implicit_grad_assembly(
        context,
        &solver.kernels,
        solver.runtime_dims(),
        mode,
        solver.needs_gradients,
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
    solver.graphs.run_implicit_apply(
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
    solver.graphs.run_primitive_update(
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
