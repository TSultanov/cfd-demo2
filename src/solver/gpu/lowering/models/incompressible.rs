use crate::solver::gpu::execution_plan::{run_module_graph, GraphDetail, GraphExecMode};
use crate::solver::gpu::recipe::{SolverRecipe, SteppingMode};
use crate::solver::gpu::plans::plan_instance::{
    PlanFuture, PlanLinearSystemDebug, PlanParam, PlanParamValue, PlanStepStats,
};
use crate::solver::gpu::plans::program::{GpuProgramPlan, ProgramOpRegistry};
use crate::solver::gpu::structs::{GpuSolver, LinearSolverStats};

mod op_ids {
    use crate::solver::gpu::plans::program::{CondOpKind, CountOpKind, GraphOpKind, HostOpKind};

    pub(crate) const G_COUPLED_PREPARE_ASSEMBLY: GraphOpKind =
        GraphOpKind("incompressible:coupled_prepare_assembly");
    pub(crate) const G_COUPLED_ASSEMBLY: GraphOpKind = GraphOpKind("incompressible:coupled_assembly");
    pub(crate) const G_COUPLED_UPDATE: GraphOpKind = GraphOpKind("incompressible:coupled_update");
    pub(crate) const G_COUPLED_INIT_PREPARE: GraphOpKind =
        GraphOpKind("incompressible:coupled_init_prepare");

    pub(crate) const H_COUPLED_BEGIN_STEP: HostOpKind = HostOpKind("incompressible:coupled_begin_step");
    pub(crate) const H_COUPLED_BEFORE_ITER: HostOpKind =
        HostOpKind("incompressible:coupled_before_iter");
    pub(crate) const H_COUPLED_SOLVE: HostOpKind = HostOpKind("incompressible:coupled_solve");
    pub(crate) const H_COUPLED_CLEAR_MAX_DIFF: HostOpKind =
        HostOpKind("incompressible:coupled_clear_max_diff");
    pub(crate) const H_COUPLED_CONVERGENCE_ADVANCE: HostOpKind =
        HostOpKind("incompressible:coupled_convergence_advance");
    pub(crate) const H_COUPLED_FINALIZE_STEP: HostOpKind =
        HostOpKind("incompressible:coupled_finalize_step");

    pub(crate) const C_HAS_COUPLED_RESOURCES: CondOpKind =
        CondOpKind("incompressible:has_coupled_resources");
    pub(crate) const C_COUPLED_NEEDS_PREPARE: CondOpKind =
        CondOpKind("incompressible:coupled_needs_prepare");
    pub(crate) const C_COUPLED_SHOULD_CONTINUE: CondOpKind =
        CondOpKind("incompressible:coupled_should_continue");

    pub(crate) const N_COUPLED_MAX_ITERS: CountOpKind = CountOpKind("incompressible:coupled_max_iters");
}

pub(in crate::solver::gpu::lowering) struct IncompressibleProgramResources {
    plan: GpuSolver,
    coupled_outer_iter: usize,
    coupled_continue: bool,
    coupled_prev_residual_u: f64,
    coupled_prev_residual_p: f64,
}

impl IncompressibleProgramResources {
    pub(in crate::solver::gpu::lowering) fn new(plan: GpuSolver) -> Self {
        Self {
            plan,
            coupled_outer_iter: 0,
            coupled_continue: true,
            coupled_prev_residual_u: f64::MAX,
            coupled_prev_residual_p: f64::MAX,
        }
    }
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

fn wrap(plan: &GpuProgramPlan) -> &IncompressibleProgramResources {
    plan.resources
        .get::<IncompressibleProgramResources>()
        .expect("missing IncompressibleProgramResources")
}

fn wrap_mut(plan: &mut GpuProgramPlan) -> &mut IncompressibleProgramResources {
    plan.resources
        .get_mut::<IncompressibleProgramResources>()
        .expect("missing IncompressibleProgramResources")
}

fn res(plan: &GpuProgramPlan) -> &GpuSolver {
    &wrap(plan).plan
}

fn res_mut(plan: &mut GpuProgramPlan) -> &mut GpuSolver {
    &mut wrap_mut(plan).plan
}

pub(in crate::solver::gpu::lowering) fn register_ops(
    registry: &mut ProgramOpRegistry,
) -> Result<(), String> {
    registry.register_graph(
        op_ids::G_COUPLED_INIT_PREPARE,
        coupled_graph_init_prepare_run,
    )?;
    registry.register_graph(
        op_ids::G_COUPLED_PREPARE_ASSEMBLY,
        coupled_graph_prepare_assembly_run,
    )?;
    registry.register_graph(
        op_ids::G_COUPLED_ASSEMBLY,
        coupled_graph_assembly_run,
    )?;
    registry.register_graph(
        op_ids::G_COUPLED_UPDATE,
        coupled_graph_update_run,
    )?;

    registry.register_host(op_ids::H_COUPLED_BEGIN_STEP, host_coupled_begin_step)?;
    registry.register_host(
        op_ids::H_COUPLED_BEFORE_ITER,
        host_coupled_before_iter,
    )?;
    registry.register_host(op_ids::H_COUPLED_SOLVE, host_coupled_solve)?;
    registry.register_host(
        op_ids::H_COUPLED_CLEAR_MAX_DIFF,
        host_coupled_clear_max_diff,
    )?;
    registry.register_host(
        op_ids::H_COUPLED_CONVERGENCE_ADVANCE,
        host_coupled_convergence_and_advance,
    )?;
    registry.register_host(
        op_ids::H_COUPLED_FINALIZE_STEP,
        host_coupled_finalize_step,
    )?;

    registry.register_cond(op_ids::C_HAS_COUPLED_RESOURCES, has_coupled_resources)?;
    registry.register_cond(
        op_ids::C_COUPLED_NEEDS_PREPARE,
        coupled_needs_prepare,
    )?;
    registry.register_cond(
        op_ids::C_COUPLED_SHOULD_CONTINUE,
        coupled_should_continue,
    )?;

    registry.register_count(op_ids::N_COUPLED_MAX_ITERS, coupled_max_iters)?;

    Ok(())
}

/// Register ops based on the solver recipe.
///
/// IncompressibleCoupled is expected to run in `SteppingMode::Coupled`.
pub(in crate::solver::gpu::lowering) fn register_ops_from_recipe(
    recipe: &SolverRecipe,
    registry: &mut ProgramOpRegistry,
) -> Result<(), String> {
    if !matches!(recipe.stepping, SteppingMode::Coupled { .. }) {
        return Err(format!(
            "incompressible_coupled lowering expects coupled stepping (recipe.model_id='{}', stepping={:?})",
            recipe.model_id, recipe.stepping
        ));
    }

    // The program spec is still template-based, so register the full coupled op set.
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
    res(plan).fields.state.write_all(&plan.context.queue, bytes);
    Ok(())
}

pub(in crate::solver::gpu::lowering) fn has_coupled_resources(plan: &GpuProgramPlan) -> bool {
    res(plan).coupled_resources.is_some()
}

fn coupled_runtime(plan: &GpuProgramPlan) -> crate::solver::gpu::modules::graph::RuntimeDims {
    let solver = res(plan);
    crate::solver::gpu::modules::graph::RuntimeDims {
        num_cells: solver.num_cells,
        num_faces: solver.num_faces,
    }
}

pub(in crate::solver::gpu::lowering) fn coupled_graph_init_prepare_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let solver = res(plan);
    run_module_graph(
        &solver.coupled_init_prepare_graph,
        context,
        &solver.kernels,
        coupled_runtime(plan),
        mode,
    )
}

pub(in crate::solver::gpu::lowering) fn coupled_graph_prepare_assembly_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let solver = res(plan);
    run_module_graph(
        &solver.coupled_prepare_assembly_graph,
        context,
        &solver.kernels,
        coupled_runtime(plan),
        mode,
    )
}

pub(in crate::solver::gpu::lowering) fn coupled_graph_assembly_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let solver = res(plan);
    run_module_graph(
        &solver.coupled_assembly_graph,
        context,
        &solver.kernels,
        coupled_runtime(plan),
        mode,
    )
}

pub(in crate::solver::gpu::lowering) fn coupled_graph_update_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let solver = res(plan);
    run_module_graph(
        &solver.coupled_update_graph,
        context,
        &solver.kernels,
        coupled_runtime(plan),
        mode,
    )
}

pub(in crate::solver::gpu::lowering) fn coupled_needs_prepare(plan: &GpuProgramPlan) -> bool {
    let wrap = wrap(plan);
    wrap.coupled_outer_iter > 0 || res(plan).scheme_needs_gradients
}

pub(in crate::solver::gpu::lowering) fn coupled_max_iters(plan: &GpuProgramPlan) -> usize {
    let solver = res(plan);
    (solver.n_outer_correctors.max(10)) as usize
}

pub(in crate::solver::gpu::lowering) fn coupled_should_continue(plan: &GpuProgramPlan) -> bool {
    wrap(plan).coupled_continue
}

pub(in crate::solver::gpu::lowering) fn host_coupled_begin_step(plan: &mut GpuProgramPlan) {
    let wrap = wrap_mut(plan);
    let solver = &mut wrap.plan;

    // Mirror legacy behavior: no-op if resources were not initialized.
    if solver.coupled_resources.is_none() {
        wrap.coupled_continue = false;
        return;
    }

    wrap.coupled_outer_iter = 0;
    wrap.coupled_continue = !solver.should_stop;
    wrap.coupled_prev_residual_u = f64::MAX;
    wrap.coupled_prev_residual_p = f64::MAX;

    // Reset async reader to clear old values.
    if let Some(res) = solver.coupled_resources.as_ref() {
        res.async_scalar_reader.borrow_mut().reset();
    }

    // Ping-pong rotation (shared with kernels via PingPongState handle).
    solver.fields.state.advance();
    solver.time_integration.prepare_step(
        &mut solver.fields.constants,
        &solver.common.context.queue,
    );

    // Initialize fluxes and d_p (and gradients) expects `component = 0`.
    {
        let values = solver.fields.constants.values_mut();
        values.component = 0;
    }
    solver.fields.constants.write(&solver.common.context.queue);
}

pub(in crate::solver::gpu::lowering) fn host_coupled_before_iter(plan: &mut GpuProgramPlan) {
    let wrap = wrap_mut(plan);
    let iter = wrap.coupled_outer_iter;
    let solver = &mut wrap.plan;
    if iter > 0 {
        {
            let values = solver.fields.constants.values_mut();
            values.component = 0;
        }
        solver.fields.constants.write(&solver.common.context.queue);
    }
}

pub(in crate::solver::gpu::lowering) fn host_coupled_solve(plan: &mut GpuProgramPlan) {
    let solver = &mut wrap_mut(plan).plan;
    let stats = solver.solve_coupled_system();
    solver.coupled_last_linear_stats = stats;
    *solver.stats_p.lock().unwrap() = stats;
    if stats.residual.is_nan() {
        panic!("Coupled Linear Solver Diverged: NaN detected in linear residual");
    }
}

pub(in crate::solver::gpu::lowering) fn host_coupled_clear_max_diff(plan: &mut GpuProgramPlan) {
    let wrap = wrap_mut(plan);
    let iter = wrap.coupled_outer_iter;
    let solver = &mut wrap.plan;
    if iter == 0 {
        return;
    }
    let Some(res) = solver.coupled_resources.as_ref() else {
        return;
    };
    solver
        .common
        .context
        .queue
        .write_buffer(&res.b_max_diff_result, 0, &[0u8; 8]);
}

pub(in crate::solver::gpu::lowering) fn host_coupled_convergence_and_advance(
    plan: &mut GpuProgramPlan,
) {
    let wrap = wrap_mut(plan);
    let solver = &mut wrap.plan;
    let iter = wrap.coupled_outer_iter;

    if iter == 0 {
        *solver.outer_residual_u.lock().unwrap() = f32::MAX;
        *solver.outer_residual_p.lock().unwrap() = f32::MAX;
        *solver.outer_iterations.lock().unwrap() = 1;
        wrap.coupled_outer_iter = 1;
        return;
    }

    let Some(res) = solver.coupled_resources.as_ref() else {
        wrap.coupled_continue = false;
        return;
    };

    // Start async read for CURRENT iteration, then poll for completion of previous reads.
    {
        let mut reader = res.async_scalar_reader.borrow_mut();
        reader.start_read(
            &solver.common.context.device,
            &solver.common.context.queue,
            &res.b_max_diff_result,
            0,
        );
        reader.poll();

        if let Some(results) = reader.get_last_value_vec(2) {
            let max_diff_u = results[0] as f64;
            let max_diff_p = results[1] as f64;

            if max_diff_u.is_nan() || max_diff_p.is_nan() {
                panic!(
                    "Coupled Solver Diverged: NaN detected in outer residuals (U: {}, P: {})",
                    max_diff_u, max_diff_p
                );
            }

            *solver.outer_residual_u.lock().unwrap() = max_diff_u as f32;
            *solver.outer_residual_p.lock().unwrap() = max_diff_p as f32;
            *solver.outer_iterations.lock().unwrap() = (iter + 1) as u32;

            let convergence_tol_u = 1e-5;
            let convergence_tol_p = 1e-4;
            if max_diff_u < convergence_tol_u && max_diff_p < convergence_tol_p {
                wrap.coupled_continue = false;
            }

            // Stagnation check (mirror legacy thresholds).
            let stagnation_factor = 1e-2;
            let rel_u = if wrap.coupled_prev_residual_u.is_finite()
                && wrap.coupled_prev_residual_u.abs() > 1e-14
            {
                ((max_diff_u - wrap.coupled_prev_residual_u) / wrap.coupled_prev_residual_u).abs()
            } else {
                f64::INFINITY
            };
            let rel_p = if wrap.coupled_prev_residual_p.is_finite()
                && wrap.coupled_prev_residual_p.abs() > 1e-14
            {
                ((max_diff_p - wrap.coupled_prev_residual_p) / wrap.coupled_prev_residual_p).abs()
            } else {
                f64::INFINITY
            };

            if rel_u < stagnation_factor && rel_p < stagnation_factor && iter > 2 {
                wrap.coupled_continue = false;
            }

            wrap.coupled_prev_residual_u = max_diff_u;
            wrap.coupled_prev_residual_p = max_diff_p;
        }
    }

    wrap.coupled_outer_iter += 1;
}

pub(in crate::solver::gpu::lowering) fn init_history(plan: &GpuProgramPlan) {
    res(plan).initialize_history();
}

pub(in crate::solver::gpu::lowering) fn step_stats(plan: &GpuProgramPlan) -> PlanStepStats {
    let solver = res(plan);
    PlanStepStats {
        should_stop: Some(solver.should_stop),
        degenerate_count: Some(solver.degenerate_count),
        outer_iterations: Some(*solver.outer_iterations.lock().unwrap()),
        outer_residual_u: Some(*solver.outer_residual_u.lock().unwrap()),
        outer_residual_p: Some(*solver.outer_residual_p.lock().unwrap()),
        linear_stats: Some((
            *solver.stats_ux.lock().unwrap(),
            *solver.stats_uy.lock().unwrap(),
            *solver.stats_p.lock().unwrap(),
        )),
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
            solver.time_integration.set_dt(
                dt,
                &mut solver.fields.constants,
                &solver.common.context.queue,
            );
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
        (PlanParam::Density, PlanParamValue::F32(rho)) => {
            solver.set_density(rho);
            Ok(())
        }
        (PlanParam::AlphaU, PlanParamValue::F32(alpha)) => {
            solver.set_alpha_u(alpha);
            Ok(())
        }
        (PlanParam::AlphaP, PlanParamValue::F32(alpha)) => {
            solver.set_alpha_p(alpha);
            Ok(())
        }
        (PlanParam::InletVelocity, PlanParamValue::F32(velocity)) => {
            solver.set_inlet_velocity(velocity);
            Ok(())
        }
        (PlanParam::RampTime, PlanParamValue::F32(time)) => {
            solver.set_ramp_time(time);
            Ok(())
        }
        (PlanParam::IncompressibleOuterCorrectors, PlanParamValue::U32(iters)) => {
            solver.n_outer_correctors = iters.max(1);
            Ok(())
        }
        (PlanParam::IncompressibleShouldStop, PlanParamValue::Bool(value)) => {
            solver.should_stop = value;
            Ok(())
        }
        (PlanParam::DetailedProfilingEnabled, PlanParamValue::Bool(enable)) => {
            solver.enable_detailed_profiling(enable);
            Ok(())
        }
        _ => Err("parameter is not supported by this plan".into()),
    }
}

pub(in crate::solver::gpu::lowering) fn linear_debug_provider(
    plan: &mut GpuProgramPlan,
) -> Option<&mut dyn PlanLinearSystemDebug> {
    Some(plan.resources.get_mut::<IncompressibleProgramResources>()?
        as &mut dyn PlanLinearSystemDebug)
}

pub(in crate::solver::gpu::lowering) fn host_coupled_finalize_step(plan: &mut GpuProgramPlan) {
    let solver = res_mut(plan);
    if solver.coupled_resources.is_none() {
        return;
    }

    solver.time_integration.finalize_step(
        &mut solver.fields.constants,
        &solver.common.context.queue,
    );

    solver.check_evolution();

    let _ = solver
        .common
        .context
        .device
        .poll(wgpu::PollType::wait_indefinitely());
}