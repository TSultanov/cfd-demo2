use crate::solver::gpu::execution_plan::{run_module_graph, GraphDetail, GraphExecMode};
use crate::solver::gpu::plans::compressible::CompressiblePlanResources;
use crate::solver::gpu::plans::plan_instance::{
    PlanFuture, PlanLinearSystemDebug, PlanParam, PlanParamValue, PlanStepStats,
};
use crate::solver::gpu::plans::program::{
    GpuProgramPlan, ModelGpuProgramSpec, ProgramCondId, ProgramCountId, ProgramExecutionPlan,
    ProgramGraphId, ProgramHostId, ProgramNode, ProgramResources, ProgramSetParamFallback,
    ProgramStepStatsFn, ProgramStepWithStatsFn,
};
use crate::solver::gpu::structs::LinearSolverStats;
use crate::solver::mesh::Mesh;
use crate::solver::model::ModelSpec;
use std::env;
use std::sync::Arc;

struct CompressibleProgramResources {
    plan: CompressiblePlanResources,
    implicit_outer_idx: usize,
    implicit_stats: Vec<LinearSolverStats>,
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

fn spec_num_cells(plan: &GpuProgramPlan) -> u32 {
    res(plan).num_cells
}

fn spec_time(plan: &GpuProgramPlan) -> f32 {
    res(plan).constants.time
}

fn spec_dt(plan: &GpuProgramPlan) -> f32 {
    res(plan).constants.dt
}

fn spec_state_buffer(plan: &GpuProgramPlan) -> &wgpu::Buffer {
    &res(plan).b_state
}

fn spec_write_state_bytes(plan: &GpuProgramPlan, bytes: &[u8]) -> Result<(), String> {
    res(plan).write_state_bytes(bytes);
    Ok(())
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

fn implicit_outer_iters(plan: &GpuProgramPlan) -> usize {
    res(plan).outer_iters
}

fn host_implicit_prepare(plan: &mut GpuProgramPlan) {
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

fn host_implicit_set_iter_params(plan: &mut GpuProgramPlan) {
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

fn implicit_grad_assembly_graph_run(
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

fn host_implicit_solve_fgmres(plan: &mut GpuProgramPlan) {
    res_mut(plan).implicit_solve_fgmres();
}

fn host_implicit_record_stats(plan: &mut GpuProgramPlan) {
    let stats = res(plan).implicit_last_stats();
    let wrap = res_wrap_mut(plan);
    wrap.implicit_stats.push(stats);
}

fn implicit_snapshot_run(
    plan: &GpuProgramPlan,
    _context: &crate::solver::gpu::context::GpuContext,
    _mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let start = std::time::Instant::now();
    res(plan).implicit_snapshot();
    (start.elapsed().as_secs_f64(), None)
}

fn host_implicit_set_alpha_for_apply(plan: &mut GpuProgramPlan) {
    res_mut(plan).implicit_set_alpha_for_apply();
}

fn implicit_apply_graph_run(
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

fn host_implicit_restore_alpha(plan: &mut GpuProgramPlan) {
    res_mut(plan).implicit_restore_alpha();
}

fn host_implicit_advance_outer_idx(plan: &mut GpuProgramPlan) {
    res_wrap_mut(plan).implicit_outer_idx += 1;
}

fn primitive_update_graph_run(
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

fn host_implicit_finalize(plan: &mut GpuProgramPlan) {
    res_mut(plan).finalize_dt_old();
}

fn init_history(plan: &GpuProgramPlan) {
    res(plan).initialize_history();
}

fn step_stats(_plan: &GpuProgramPlan) -> PlanStepStats {
    PlanStepStats::default()
}

fn step_with_stats(plan: &mut GpuProgramPlan) -> Result<Vec<LinearSolverStats>, String> {
    if should_use_explicit(plan) {
        plan.step();
        Ok(Vec::new())
    } else {
        plan.step();
        Ok(std::mem::take(&mut res_wrap_mut(plan).implicit_stats))
    }
}

fn set_param_fallback(
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

fn linear_debug_provider(plan: &mut GpuProgramPlan) -> Option<&mut dyn PlanLinearSystemDebug> {
    Some(plan.resources.get_mut::<CompressibleProgramResources>()? as &mut dyn PlanLinearSystemDebug)
}

const G_EXPLICIT_GRAPH: ProgramGraphId = ProgramGraphId(0);
const G_IMPLICIT_GRAD_ASSEMBLY: ProgramGraphId = ProgramGraphId(1);
const G_IMPLICIT_SNAPSHOT: ProgramGraphId = ProgramGraphId(2);
const G_IMPLICIT_APPLY: ProgramGraphId = ProgramGraphId(3);
const G_PRIMITIVE_UPDATE: ProgramGraphId = ProgramGraphId(4);

const H_EXPLICIT_PREPARE: ProgramHostId = ProgramHostId(0);
const H_EXPLICIT_FINALIZE: ProgramHostId = ProgramHostId(1);
const H_IMPLICIT_PREPARE: ProgramHostId = ProgramHostId(2);
const H_IMPLICIT_SET_ITER_PARAMS: ProgramHostId = ProgramHostId(3);
const H_IMPLICIT_SOLVE_FGMRES: ProgramHostId = ProgramHostId(4);
const H_IMPLICIT_RECORD_STATS: ProgramHostId = ProgramHostId(5);
const H_IMPLICIT_SET_ALPHA: ProgramHostId = ProgramHostId(6);
const H_IMPLICIT_RESTORE_ALPHA: ProgramHostId = ProgramHostId(7);
const H_IMPLICIT_ADVANCE_OUTER_IDX: ProgramHostId = ProgramHostId(8);
const H_IMPLICIT_FINALIZE: ProgramHostId = ProgramHostId(9);

const C_SHOULD_USE_EXPLICIT: ProgramCondId = ProgramCondId(0);
const N_IMPLICIT_OUTER_ITERS: ProgramCountId = ProgramCountId(0);

pub(crate) async fn lower_compressible_program(
    mesh: &Mesh,
    model: ModelSpec,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
) -> Result<GpuProgramPlan, String> {
    let plan = CompressiblePlanResources::new(mesh, model.clone(), device, queue).await?;

    let context = crate::solver::gpu::context::GpuContext {
        device: plan.common.context.device.clone(),
        queue: plan.common.context.queue.clone(),
    };
    let profiling_stats = Arc::clone(&plan.common.profiling_stats);

    let mut resources = ProgramResources::new();
    resources.insert(CompressibleProgramResources {
        plan,
        implicit_outer_idx: 0,
        implicit_stats: Vec::new(),
    });

    let mut graph_ops = std::collections::HashMap::new();
    graph_ops.insert(G_EXPLICIT_GRAPH, explicit_graph_run as _);
    graph_ops.insert(
        G_IMPLICIT_GRAD_ASSEMBLY,
        implicit_grad_assembly_graph_run as _,
    );
    graph_ops.insert(G_IMPLICIT_SNAPSHOT, implicit_snapshot_run as _);
    graph_ops.insert(G_IMPLICIT_APPLY, implicit_apply_graph_run as _);
    graph_ops.insert(G_PRIMITIVE_UPDATE, primitive_update_graph_run as _);

    let mut host_ops = std::collections::HashMap::new();
    host_ops.insert(H_EXPLICIT_PREPARE, host_explicit_prepare as _);
    host_ops.insert(H_EXPLICIT_FINALIZE, host_explicit_finalize as _);
    host_ops.insert(H_IMPLICIT_PREPARE, host_implicit_prepare as _);
    host_ops.insert(
        H_IMPLICIT_SET_ITER_PARAMS,
        host_implicit_set_iter_params as _,
    );
    host_ops.insert(H_IMPLICIT_SOLVE_FGMRES, host_implicit_solve_fgmres as _);
    host_ops.insert(H_IMPLICIT_RECORD_STATS, host_implicit_record_stats as _);
    host_ops.insert(H_IMPLICIT_SET_ALPHA, host_implicit_set_alpha_for_apply as _);
    host_ops.insert(H_IMPLICIT_RESTORE_ALPHA, host_implicit_restore_alpha as _);
    host_ops.insert(
        H_IMPLICIT_ADVANCE_OUTER_IDX,
        host_implicit_advance_outer_idx as _,
    );
    host_ops.insert(H_IMPLICIT_FINALIZE, host_implicit_finalize as _);

    let mut cond_ops = std::collections::HashMap::new();
    cond_ops.insert(C_SHOULD_USE_EXPLICIT, should_use_explicit as _);

    let mut count_ops = std::collections::HashMap::new();
    count_ops.insert(N_IMPLICIT_OUTER_ITERS, implicit_outer_iters as _);

    let explicit_plan = Arc::new(ProgramExecutionPlan::new(vec![
        ProgramNode::Host {
            label: "compressible:explicit_prepare",
            id: H_EXPLICIT_PREPARE,
        },
        ProgramNode::Graph {
            label: "compressible:explicit_graph",
            id: G_EXPLICIT_GRAPH,
            mode: GraphExecMode::SplitTimed,
        },
        ProgramNode::Host {
            label: "compressible:explicit_finalize",
            id: H_EXPLICIT_FINALIZE,
        },
    ]));

    let implicit_iter_body = Arc::new(ProgramExecutionPlan::new(vec![
        ProgramNode::Host {
            label: "compressible:implicit_set_iter_params",
            id: H_IMPLICIT_SET_ITER_PARAMS,
        },
        ProgramNode::Graph {
            label: "compressible:implicit_grad_assembly",
            id: G_IMPLICIT_GRAD_ASSEMBLY,
            mode: GraphExecMode::SplitTimed,
        },
        ProgramNode::Host {
            label: "compressible:implicit_fgmres",
            id: H_IMPLICIT_SOLVE_FGMRES,
        },
        ProgramNode::Host {
            label: "compressible:implicit_record_stats",
            id: H_IMPLICIT_RECORD_STATS,
        },
        ProgramNode::Graph {
            label: "compressible:implicit_snapshot",
            id: G_IMPLICIT_SNAPSHOT,
            mode: GraphExecMode::SingleSubmit,
        },
        ProgramNode::Host {
            label: "compressible:implicit_set_alpha",
            id: H_IMPLICIT_SET_ALPHA,
        },
        ProgramNode::Graph {
            label: "compressible:implicit_apply",
            id: G_IMPLICIT_APPLY,
            mode: GraphExecMode::SingleSubmit,
        },
        ProgramNode::Host {
            label: "compressible:implicit_restore_alpha",
            id: H_IMPLICIT_RESTORE_ALPHA,
        },
        ProgramNode::Host {
            label: "compressible:implicit_outer_idx_inc",
            id: H_IMPLICIT_ADVANCE_OUTER_IDX,
        },
    ]));

    let implicit_plan = Arc::new(ProgramExecutionPlan::new(vec![
        ProgramNode::Host {
            label: "compressible:implicit_prepare",
            id: H_IMPLICIT_PREPARE,
        },
        ProgramNode::Repeat {
            label: "compressible:implicit_outer_loop",
            times: N_IMPLICIT_OUTER_ITERS,
            body: implicit_iter_body,
        },
        ProgramNode::Graph {
            label: "compressible:primitive_update",
            id: G_PRIMITIVE_UPDATE,
            mode: GraphExecMode::SingleSubmit,
        },
        ProgramNode::Host {
            label: "compressible:implicit_finalize",
            id: H_IMPLICIT_FINALIZE,
        },
    ]));

    let step = Arc::new(ProgramExecutionPlan::new(vec![ProgramNode::If {
        label: "compressible:select_step_path",
        cond: C_SHOULD_USE_EXPLICIT,
        then_plan: explicit_plan,
        else_plan: Some(implicit_plan),
    }]));

    let spec = ModelGpuProgramSpec {
        graph_ops,
        host_ops,
        cond_ops,
        count_ops,
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

    Ok(GpuProgramPlan::new(
        model,
        context,
        profiling_stats,
        resources,
        spec,
    ))
}
