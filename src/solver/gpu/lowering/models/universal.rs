use crate::solver::gpu::execution_plan::{run_module_graph, GraphDetail, GraphExecMode};
use crate::solver::gpu::lowering::unified_registry::UnifiedOpRegistryConfig;
use crate::solver::gpu::plans::compressible::CompressiblePlanResources;
use crate::solver::gpu::plans::plan_instance::{
    PlanFuture, PlanLinearSystemDebug, PlanParam, PlanParamValue, PlanStepStats,
};
use crate::solver::gpu::plans::program::{GpuProgramPlan, ProgramOpRegistry};
use crate::solver::gpu::recipe::{SolverRecipe, SteppingMode};
use crate::solver::gpu::structs::{GpuSolver, LinearSolverStats};
use std::env;

// --- Single universal program resource ---

pub(in crate::solver::gpu::lowering) struct UniversalProgramResources {
    backend: UniversalBackend,
}

enum UniversalBackend {
    ExplicitImplicit(ExplicitImplicitBackend),
    Coupled(CoupledBackend),
}

struct ExplicitImplicitBackend {
    plan: CompressiblePlanResources,
    implicit_outer_idx: usize,
    implicit_stats: Vec<LinearSolverStats>,
}

struct CoupledBackend {
    plan: GpuSolver,
    coupled_outer_iter: usize,
    coupled_continue: bool,
    coupled_prev_residual_u: f64,
    coupled_prev_residual_p: f64,
}

impl UniversalProgramResources {
    pub(in crate::solver::gpu::lowering) fn new_explicit_implicit(
        plan: CompressiblePlanResources,
    ) -> Self {
        Self {
            backend: UniversalBackend::ExplicitImplicit(ExplicitImplicitBackend {
                plan,
                implicit_outer_idx: 0,
                implicit_stats: Vec::new(),
            }),
        }
    }

    pub(in crate::solver::gpu::lowering) fn new_coupled(plan: GpuSolver) -> Self {
        Self {
            backend: UniversalBackend::Coupled(CoupledBackend {
                plan,
                coupled_outer_iter: 0,
                coupled_continue: true,
                coupled_prev_residual_u: f64::MAX,
                coupled_prev_residual_p: f64::MAX,
            }),
        }
    }
}

impl PlanLinearSystemDebug for UniversalProgramResources {
    fn set_linear_system(&self, matrix_values: &[f32], rhs: &[f32]) -> Result<(), String> {
        match &self.backend {
            UniversalBackend::ExplicitImplicit(b) => {
                PlanLinearSystemDebug::set_linear_system(&b.plan, matrix_values, rhs)
            }
            UniversalBackend::Coupled(b) => {
                PlanLinearSystemDebug::set_linear_system(&b.plan, matrix_values, rhs)
            }
        }
    }

    fn solve_linear_system_with_size(
        &mut self,
        n: u32,
        max_iters: u32,
        tol: f32,
    ) -> Result<LinearSolverStats, String> {
        match &mut self.backend {
            UniversalBackend::ExplicitImplicit(b) => {
                PlanLinearSystemDebug::solve_linear_system_with_size(&mut b.plan, n, max_iters, tol)
            }
            UniversalBackend::Coupled(b) => {
                PlanLinearSystemDebug::solve_linear_system_with_size(&mut b.plan, n, max_iters, tol)
            }
        }
    }

    fn get_linear_solution(&self) -> PlanFuture<'_, Result<Vec<f32>, String>> {
        match &self.backend {
            UniversalBackend::ExplicitImplicit(b) => {
                PlanLinearSystemDebug::get_linear_solution(&b.plan)
            }
            UniversalBackend::Coupled(b) => PlanLinearSystemDebug::get_linear_solution(&b.plan),
        }
    }
}

// --- Universal resource access helpers ---

fn wrap(plan: &GpuProgramPlan) -> &UniversalProgramResources {
    plan.resources
        .get::<UniversalProgramResources>()
        .expect("missing UniversalProgramResources")
}

fn wrap_mut(plan: &mut GpuProgramPlan) -> &mut UniversalProgramResources {
    plan.resources
        .get_mut::<UniversalProgramResources>()
        .expect("missing UniversalProgramResources")
}

fn explicit_implicit(plan: &GpuProgramPlan) -> Option<&CompressiblePlanResources> {
    match &wrap(plan).backend {
        UniversalBackend::ExplicitImplicit(b) => Some(&b.plan),
        UniversalBackend::Coupled(_) => None,
    }
}

fn explicit_implicit_mut(plan: &mut GpuProgramPlan) -> Option<&mut CompressiblePlanResources> {
    match &mut wrap_mut(plan).backend {
        UniversalBackend::ExplicitImplicit(b) => Some(&mut b.plan),
        UniversalBackend::Coupled(_) => None,
    }
}

fn explicit_implicit_backend_mut(
    plan: &mut GpuProgramPlan,
) -> Option<&mut ExplicitImplicitBackend> {
    match &mut wrap_mut(plan).backend {
        UniversalBackend::ExplicitImplicit(b) => Some(b),
        UniversalBackend::Coupled(_) => None,
    }
}

fn coupled(plan: &GpuProgramPlan) -> Option<&GpuSolver> {
    match &wrap(plan).backend {
        UniversalBackend::Coupled(b) => Some(&b.plan),
        UniversalBackend::ExplicitImplicit(_) => None,
    }
}

fn coupled_mut(plan: &mut GpuProgramPlan) -> Option<&mut GpuSolver> {
    match &mut wrap_mut(plan).backend {
        UniversalBackend::Coupled(b) => Some(&mut b.plan),
        UniversalBackend::ExplicitImplicit(_) => None,
    }
}

fn coupled_backend(plan: &GpuProgramPlan) -> Option<&CoupledBackend> {
    match &wrap(plan).backend {
        UniversalBackend::Coupled(b) => Some(b),
        UniversalBackend::ExplicitImplicit(_) => None,
    }
}

fn coupled_backend_mut(plan: &mut GpuProgramPlan) -> Option<&mut CoupledBackend> {
    match &mut wrap_mut(plan).backend {
        UniversalBackend::Coupled(b) => Some(b),
        UniversalBackend::ExplicitImplicit(_) => None,
    }
}

// --- Universal op registration ---

/// Single universal lowering path: register the unified op kinds emitted by `SolverRecipe::build_program_spec()`.
///
/// The actual host/graph handlers are implemented against type-erased `ProgramResources` and downcast at runtime.
pub(in crate::solver::gpu::lowering) fn register_ops_from_recipe(
    recipe: &SolverRecipe,
    registry: &mut ProgramOpRegistry,
) -> Result<(), String> {
    let config = match recipe.stepping {
        SteppingMode::Explicit => UnifiedOpRegistryConfig {
            prepare: Some(host_explicit_prepare),
            finalize: Some(host_explicit_finalize),
            update_graph: Some(explicit_graph_run),
            ..Default::default()
        },
        SteppingMode::Implicit { .. } => UnifiedOpRegistryConfig {
            prepare: Some(host_implicit_prepare),
            finalize: Some(host_implicit_finalize),
            solve: Some(host_implicit_solve_fgmres),

            // Compressible implicit combines gradients+assembly.
            assembly_graph: Some(implicit_grad_assembly_graph_run),
            apply_graph: Some(implicit_apply_graph_run),
            implicit_snapshot_graph: Some(implicit_snapshot_run),
            implicit_update_graph: Some(primitive_update_graph_run),

            implicit_before_iter: Some(host_implicit_set_iter_params),
            implicit_after_solve: Some(host_implicit_record_stats),
            implicit_before_apply: Some(host_implicit_set_alpha_for_apply),
            implicit_after_apply: Some(host_implicit_restore_alpha),
            implicit_advance_outer_idx: Some(host_implicit_advance_outer_idx),
            implicit_outer_iters: Some(implicit_outer_iters),
            ..Default::default()
        },
        SteppingMode::Coupled { .. } => {
            fn coupled_graph_assembly_select_run(
                plan: &GpuProgramPlan,
                context: &crate::solver::gpu::context::GpuContext,
                mode: GraphExecMode,
            ) -> (f64, Option<GraphDetail>) {
                if coupled_needs_prepare(plan) {
                    coupled_graph_prepare_assembly_run(plan, context, mode)
                } else {
                    coupled_graph_assembly_run(plan, context, mode)
                }
            }

            UnifiedOpRegistryConfig {
                // These map onto the unified coupled program.
                prepare: Some(host_coupled_begin_step),
                finalize: Some(host_coupled_finalize_step),
                solve: Some(host_coupled_solve),

                coupled_enabled: Some(has_coupled_resources),
                coupled_init_prepare_graph: Some(coupled_graph_init_prepare_run),
                coupled_before_iter: Some(host_coupled_before_iter),
                coupled_clear_max_diff: Some(host_coupled_clear_max_diff),
                coupled_convergence_advance: Some(host_coupled_convergence_and_advance),
                coupled_should_continue: Some(coupled_should_continue),
                coupled_max_iters: Some(coupled_max_iters),

                assembly_graph: Some(coupled_graph_assembly_select_run),
                update_graph: Some(coupled_graph_update_run),
                ..Default::default()
            }
        }
    };

    let built =
        crate::solver::gpu::lowering::unified_registry::build_unified_registry(recipe, config)?;
    registry.merge(built)
}

// --- Universal program spec callbacks ---

pub(in crate::solver::gpu::lowering) fn spec_num_cells(plan: &GpuProgramPlan) -> u32 {
    if let Some(solver) = explicit_implicit(plan) {
        return solver.num_cells;
    }
    if let Some(solver) = coupled(plan) {
        return solver.num_cells;
    }
    panic!("missing solver resources (num_cells)");
}

pub(in crate::solver::gpu::lowering) fn spec_time(plan: &GpuProgramPlan) -> f32 {
    if let Some(solver) = explicit_implicit(plan) {
        return solver.time_integration.time as f32;
    }
    if let Some(solver) = coupled(plan) {
        return solver.time_integration.time as f32;
    }
    panic!("missing solver resources (time)");
}

pub(in crate::solver::gpu::lowering) fn spec_dt(plan: &GpuProgramPlan) -> f32 {
    if let Some(solver) = explicit_implicit(plan) {
        return solver.time_integration.dt;
    }
    if let Some(solver) = coupled(plan) {
        return solver.time_integration.dt;
    }
    panic!("missing solver resources (dt)");
}

pub(in crate::solver::gpu::lowering) fn spec_state_buffer(plan: &GpuProgramPlan) -> &wgpu::Buffer {
    if let Some(solver) = explicit_implicit(plan) {
        return solver.fields.state.state();
    }
    if let Some(solver) = coupled(plan) {
        return solver.fields.state.state();
    }
    panic!("missing solver resources (state_buffer)");
}

pub(in crate::solver::gpu::lowering) fn spec_write_state_bytes(
    plan: &GpuProgramPlan,
    bytes: &[u8],
) -> Result<(), String> {
    if let Some(solver) = explicit_implicit(plan) {
        solver.write_state_bytes(bytes);
        return Ok(());
    }
    if let Some(solver) = coupled(plan) {
        solver.fields.state.write_all(&plan.context.queue, bytes);
        return Ok(());
    }
    Err("missing solver resources (write_state_bytes)".into())
}

pub(in crate::solver::gpu::lowering) fn init_history(plan: &GpuProgramPlan) {
    if let Some(solver) = explicit_implicit(plan) {
        solver.initialize_history();
        return;
    }
    if let Some(solver) = coupled(plan) {
        solver.initialize_history();
        return;
    }
    panic!("missing solver resources (initialize_history)");
}

pub(in crate::solver::gpu::lowering) fn step_stats(plan: &GpuProgramPlan) -> PlanStepStats {
    if coupled(plan).is_some() {
        return step_stats_coupled(plan);
    }
    PlanStepStats::default()
}

pub(in crate::solver::gpu::lowering) fn set_param_fallback(
    plan: &mut GpuProgramPlan,
    param: PlanParam,
    value: PlanParamValue,
) -> Result<(), String> {
    if explicit_implicit_mut(plan).is_some() {
        return set_param_fallback_compressible(plan, param, value);
    }
    if coupled_mut(plan).is_some() {
        return set_param_fallback_coupled(plan, param, value);
    }
    Err("missing solver resources (set_param_fallback)".into())
}

pub(in crate::solver::gpu::lowering) fn step_with_stats(
    plan: &mut GpuProgramPlan,
) -> Result<Vec<LinearSolverStats>, String> {
    if explicit_implicit_mut(plan).is_some() {
        return step_with_stats_compressible(plan);
    }
    Err("step_with_stats is only supported for non-coupled stepping".into())
}

pub(in crate::solver::gpu::lowering) fn linear_debug_provider(
    plan: &mut GpuProgramPlan,
) -> Option<&mut dyn PlanLinearSystemDebug> {
    Some(plan.resources.get_mut::<UniversalProgramResources>()? as &mut dyn PlanLinearSystemDebug)
}

// --- Compressible handlers (explicit/implicit) ---

fn should_use_explicit(plan: &GpuProgramPlan) -> bool {
    explicit_implicit(plan)
        .expect("missing universal explicit/implicit backend")
        .should_use_explicit()
}

fn host_explicit_prepare(plan: &mut GpuProgramPlan) {
    let solver = explicit_implicit_mut(plan).expect("missing universal explicit/implicit backend");
    solver.advance_ping_pong_and_time();
    solver.pre_step_copy();
}

fn explicit_graph_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let solver = explicit_implicit(plan).expect("missing universal explicit/implicit backend");
    solver.graphs.run_explicit(
        context,
        &solver.kernels,
        solver.runtime_dims(),
        mode,
        solver.needs_gradients,
    )
}

fn host_explicit_finalize(plan: &mut GpuProgramPlan) {
    explicit_implicit_mut(plan)
        .expect("missing universal explicit/implicit backend")
        .finalize_dt_old();
}

fn implicit_outer_iters(plan: &GpuProgramPlan) -> usize {
    explicit_implicit(plan)
        .expect("missing universal explicit/implicit backend")
        .outer_iters
}

fn host_implicit_prepare(plan: &mut GpuProgramPlan) {
    let solver = explicit_implicit_mut(plan).expect("missing universal explicit/implicit backend");
    solver.advance_ping_pong_and_time();
    solver.pre_step_copy();
    solver.implicit_set_base_alpha();

    let wrap =
        explicit_implicit_backend_mut(plan).expect("missing universal explicit/implicit backend");
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

    let idx = explicit_implicit_backend_mut(plan)
        .expect("missing universal explicit/implicit backend")
        .implicit_outer_idx;
    let tol = if idx < warm_iters {
        tol_base * warm_scale
    } else {
        tol_base
    };
    let retry_tol = (tol * retry_scale).min(tol_base);

    explicit_implicit_mut(plan)
        .expect("missing universal explicit/implicit backend")
        .implicit_set_iteration_params(tol, retry_tol, max_restart, retry_restart);
}

fn implicit_grad_assembly_graph_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let solver = explicit_implicit(plan).expect("missing universal explicit/implicit backend");
    solver.graphs.run_implicit_grad_assembly(
        context,
        &solver.kernels,
        solver.runtime_dims(),
        mode,
        solver.needs_gradients,
    )
}

fn host_implicit_solve_fgmres(plan: &mut GpuProgramPlan) {
    explicit_implicit_mut(plan)
        .expect("missing universal explicit/implicit backend")
        .implicit_solve_fgmres();
}

fn host_implicit_record_stats(plan: &mut GpuProgramPlan) {
    let stats = explicit_implicit(plan)
        .expect("missing universal explicit/implicit backend")
        .implicit_last_stats();
    explicit_implicit_backend_mut(plan)
        .expect("missing universal explicit/implicit backend")
        .implicit_stats
        .push(stats);
}

fn implicit_snapshot_run(
    plan: &GpuProgramPlan,
    _context: &crate::solver::gpu::context::GpuContext,
    _mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let start = std::time::Instant::now();
    explicit_implicit(plan)
        .expect("missing universal explicit/implicit backend")
        .implicit_snapshot();
    (start.elapsed().as_secs_f64(), None)
}

fn host_implicit_set_alpha_for_apply(plan: &mut GpuProgramPlan) {
    explicit_implicit_mut(plan)
        .expect("missing universal explicit/implicit backend")
        .implicit_set_alpha_for_apply();
}

fn implicit_apply_graph_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let solver = explicit_implicit(plan).expect("missing universal explicit/implicit backend");
    solver
        .graphs
        .run_implicit_apply(context, &solver.kernels, solver.runtime_dims(), mode)
}

fn host_implicit_restore_alpha(plan: &mut GpuProgramPlan) {
    explicit_implicit_mut(plan)
        .expect("missing universal explicit/implicit backend")
        .implicit_restore_alpha();
}

fn host_implicit_advance_outer_idx(plan: &mut GpuProgramPlan) {
    explicit_implicit_backend_mut(plan)
        .expect("missing universal explicit/implicit backend")
        .implicit_outer_idx += 1;
}

fn primitive_update_graph_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let solver = explicit_implicit(plan).expect("missing universal explicit/implicit backend");
    solver
        .graphs
        .run_primitive_update(context, &solver.kernels, solver.runtime_dims(), mode)
}

fn host_implicit_finalize(plan: &mut GpuProgramPlan) {
    explicit_implicit_mut(plan)
        .expect("missing universal explicit/implicit backend")
        .finalize_dt_old();
}

fn step_with_stats_compressible(
    plan: &mut GpuProgramPlan,
) -> Result<Vec<LinearSolverStats>, String> {
    if should_use_explicit(plan) {
        plan.step();
        Ok(Vec::new())
    } else {
        plan.step();
        Ok(std::mem::take(
            &mut explicit_implicit_backend_mut(plan)
                .expect("missing universal explicit/implicit backend")
                .implicit_stats,
        ))
    }
}

fn set_param_fallback_compressible(
    plan: &mut GpuProgramPlan,
    param: PlanParam,
    value: PlanParamValue,
) -> Result<(), String> {
    let solver = explicit_implicit_mut(plan).expect("missing universal explicit/implicit backend");
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

// --- Coupled handlers ---

fn has_coupled_resources(plan: &GpuProgramPlan) -> bool {
    coupled(plan)
        .expect("missing universal coupled backend")
        .coupled_resources
        .is_some()
}

fn coupled_runtime(plan: &GpuProgramPlan) -> crate::solver::gpu::modules::graph::RuntimeDims {
    let solver = coupled(plan).expect("missing universal coupled backend");
    crate::solver::gpu::modules::graph::RuntimeDims {
        num_cells: solver.num_cells,
        num_faces: solver.num_faces,
    }
}

fn coupled_graph_init_prepare_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let solver = coupled(plan).expect("missing universal coupled backend");
    run_module_graph(
        &solver.coupled_init_prepare_graph,
        context,
        &solver.kernels,
        coupled_runtime(plan),
        mode,
    )
}

fn coupled_graph_prepare_assembly_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let solver = coupled(plan).expect("missing universal coupled backend");
    run_module_graph(
        &solver.coupled_prepare_assembly_graph,
        context,
        &solver.kernels,
        coupled_runtime(plan),
        mode,
    )
}

fn coupled_graph_assembly_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let solver = coupled(plan).expect("missing universal coupled backend");
    run_module_graph(
        &solver.coupled_assembly_graph,
        context,
        &solver.kernels,
        coupled_runtime(plan),
        mode,
    )
}

fn coupled_graph_update_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let solver = coupled(plan).expect("missing universal coupled backend");
    run_module_graph(
        &solver.coupled_update_graph,
        context,
        &solver.kernels,
        coupled_runtime(plan),
        mode,
    )
}

fn coupled_needs_prepare(plan: &GpuProgramPlan) -> bool {
    let wrap = coupled_backend(plan).expect("missing universal coupled backend");
    wrap.coupled_outer_iter > 0
        || coupled(plan)
            .expect("missing universal coupled backend")
            .scheme_needs_gradients
}

fn coupled_max_iters(plan: &GpuProgramPlan) -> usize {
    let solver = coupled(plan).expect("missing universal coupled backend");
    (solver.n_outer_correctors.max(10)) as usize
}

fn coupled_should_continue(plan: &GpuProgramPlan) -> bool {
    coupled_backend(plan)
        .expect("missing universal coupled backend")
        .coupled_continue
}

fn host_coupled_begin_step(plan: &mut GpuProgramPlan) {
    let wrap = coupled_backend_mut(plan).expect("missing universal coupled backend");
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
    solver
        .time_integration
        .prepare_step(&mut solver.fields.constants, &solver.common.context.queue);

    // Initialize fluxes and d_p (and gradients) expects `component = 0`.
    {
        let values = solver.fields.constants.values_mut();
        values.component = 0;
    }
    solver.fields.constants.write(&solver.common.context.queue);
}

fn host_coupled_before_iter(plan: &mut GpuProgramPlan) {
    let wrap = coupled_backend_mut(plan).expect("missing universal coupled backend");
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

fn host_coupled_solve(plan: &mut GpuProgramPlan) {
    let solver = coupled_mut(plan).expect("missing universal coupled backend");
    let stats = solver.solve_coupled_system();
    solver.coupled_last_linear_stats = stats;
    *solver.stats_p.lock().unwrap() = stats;
    if stats.residual.is_nan() {
        panic!("Coupled Linear Solver Diverged: NaN detected in linear residual");
    }
}

fn host_coupled_clear_max_diff(plan: &mut GpuProgramPlan) {
    let wrap = coupled_backend_mut(plan).expect("missing universal coupled backend");
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

fn host_coupled_convergence_and_advance(plan: &mut GpuProgramPlan) {
    let wrap = coupled_backend_mut(plan).expect("missing universal coupled backend");
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

fn host_coupled_finalize_step(plan: &mut GpuProgramPlan) {
    let solver = coupled_mut(plan).expect("missing universal coupled backend");
    if solver.coupled_resources.is_none() {
        return;
    }

    solver
        .time_integration
        .finalize_step(&mut solver.fields.constants, &solver.common.context.queue);

    solver.check_evolution();

    let _ = solver
        .common
        .context
        .device
        .poll(wgpu::PollType::wait_indefinitely());
}

fn step_stats_coupled(plan: &GpuProgramPlan) -> PlanStepStats {
    let solver = coupled(plan).expect("missing universal coupled backend");
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

fn set_param_fallback_coupled(
    plan: &mut GpuProgramPlan,
    param: PlanParam,
    value: PlanParamValue,
) -> Result<(), String> {
    let solver = coupled_mut(plan).expect("missing universal coupled backend");
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
