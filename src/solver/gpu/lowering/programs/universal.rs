use super::generic_coupled as generic_coupled_program;
use crate::solver::gpu::execution_plan::{GraphDetail, GraphExecMode};
use crate::solver::gpu::lowering::unified_registry::UnifiedOpRegistryConfig;
use crate::solver::gpu::program::plan::{GpuProgramPlan, ProgramOpRegistry};
use crate::solver::gpu::program::plan_instance::{
    PlanLinearSystemDebug, PlanStepStats,
};
use crate::solver::gpu::recipe::{SolverRecipe, SteppingMode};

// --- Universal op registration ---

/// Single universal lowering path: register the unified op kinds emitted by `SolverRecipe::build_program_spec()`.
///
/// The actual host/graph handlers access the strongly-typed `PlanResources` fields directly.
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
        SteppingMode::Coupled => {
            UnifiedOpRegistryConfig {
                // These map onto the unified coupled program.
                prepare: Some(host_coupled_begin_step),
                finalize: Some(host_coupled_finalize_step),
                solve: Some(host_coupled_solve),
                coupled_init_prepare_graph: Some(coupled_graph_init_prepare_run),
                coupled_before_iter: Some(host_coupled_before_iter),
                coupled_outer_iters: Some(coupled_outer_iters),

                assembly_graph: Some(generic_coupled_program::assembly_graph_run),
                update_graph: Some(generic_coupled_program::update_graph_run),
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
    generic_coupled_program::spec_num_cells(plan)
}

pub(in crate::solver::gpu::lowering) fn spec_time(plan: &GpuProgramPlan) -> f32 {
    generic_coupled_program::spec_time(plan)
}

pub(in crate::solver::gpu::lowering) fn spec_dt(plan: &GpuProgramPlan) -> f32 {
    generic_coupled_program::spec_dt(plan)
}

pub(in crate::solver::gpu::lowering) fn spec_state_buffer(plan: &GpuProgramPlan) -> &wgpu::Buffer {
    generic_coupled_program::spec_state_buffer(plan)
}

pub(in crate::solver::gpu::lowering) fn spec_write_state_bytes(
    plan: &GpuProgramPlan,
    bytes: &[u8],
) -> Result<(), String> {
    generic_coupled_program::spec_write_state_bytes(plan, bytes)
}

pub(in crate::solver::gpu::lowering) fn spec_set_bc_value(
    plan: &GpuProgramPlan,
    boundary: crate::solver::gpu::enums::GpuBoundaryType,
    unknown_component: u32,
    value: f32,
) -> Result<(), String> {
    generic_coupled_program::spec_set_bc_value(plan, boundary, unknown_component, value)
}

pub(in crate::solver::gpu::lowering) fn step_stats(plan: &GpuProgramPlan) -> PlanStepStats {
    let linear_stats = if !plan.step_linear_stats.is_empty() {
        let first = plan.step_linear_stats[0];
        let last = *plan
            .step_linear_stats
            .last()
            .unwrap_or(&plan.last_linear_stats);
        let best = plan
            .step_linear_stats
            .iter()
            .copied()
            .min_by(|a, b| {
                a.residual
                    .partial_cmp(&b.residual)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(last);
        Some((first, best, last))
    } else if plan.last_linear_stats.iterations > 0
        || plan.last_linear_stats.converged
        || plan.last_linear_stats.diverged
    {
        let s = plan.last_linear_stats;
        Some((s, s, s))
    } else {
        None
    };

    PlanStepStats {
        outer_iterations: (plan.outer_iterations > 0).then_some(plan.outer_iterations),
        outer_residual_u: plan.outer_residual_u,
        outer_residual_p: plan.outer_residual_p,
        linear_stats,
        ..Default::default()
    }
}

pub(in crate::solver::gpu::lowering) fn linear_debug_provider(
    plan: &mut GpuProgramPlan,
) -> Option<&mut dyn PlanLinearSystemDebug> {
    Some(&mut plan.resources.backend as &mut dyn PlanLinearSystemDebug)
}

// --- Generic-coupled handlers (explicit/implicit) ---

fn host_explicit_prepare(plan: &mut GpuProgramPlan) {
    generic_coupled_program::host_prepare_step(plan);
}

fn explicit_graph_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    generic_coupled_program::explicit_graph_run(plan, context, mode)
}

fn host_explicit_finalize(plan: &mut GpuProgramPlan) {
    generic_coupled_program::host_finalize_step(plan);
}

fn implicit_outer_iters(plan: &GpuProgramPlan) -> usize {
    generic_coupled_program::count_outer_iters(plan)
}

fn host_implicit_prepare(plan: &mut GpuProgramPlan) {
    generic_coupled_program::host_prepare_step(plan);
}

fn host_implicit_set_iter_params(_plan: &mut GpuProgramPlan) {}

fn implicit_grad_assembly_graph_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    generic_coupled_program::assembly_graph_run(plan, context, mode)
}

fn host_implicit_solve_fgmres(plan: &mut GpuProgramPlan) {
    generic_coupled_program::host_solve_linear_system(plan);
}

fn host_implicit_record_stats(plan: &mut GpuProgramPlan) {
    generic_coupled_program::host_after_solve(plan);
}

fn implicit_snapshot_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    _mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    generic_coupled_program::implicit_snapshot_run(plan, context, GraphExecMode::SingleSubmit)
}

fn host_implicit_set_alpha_for_apply(plan: &mut GpuProgramPlan) {
    generic_coupled_program::host_implicit_set_alpha_for_apply(plan);
}

fn implicit_apply_graph_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    generic_coupled_program::apply_graph_run(plan, context, mode)
}

fn host_implicit_restore_alpha(plan: &mut GpuProgramPlan) {
    generic_coupled_program::host_implicit_restore_alpha(plan);
}

fn host_implicit_advance_outer_idx(_plan: &mut GpuProgramPlan) {}

fn primitive_update_graph_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    generic_coupled_program::update_graph_run(plan, context, mode)
}

fn host_implicit_finalize(plan: &mut GpuProgramPlan) {
    generic_coupled_program::host_finalize_step(plan);
}

// --- Coupled handlers ---

fn coupled_graph_init_prepare_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    generic_coupled_program::init_prepare_graph_run(plan, context, mode)
}

fn coupled_outer_iters(plan: &GpuProgramPlan) -> usize {
    // Use the same `OuterIters` knob as the implicit path so tests/UI can control the number of
    // nonlinear corrector iterations per step for coupled methods (e.g. incompressible SIMPLE).
    generic_coupled_program::count_outer_iters(plan)
}

fn host_coupled_begin_step(plan: &mut GpuProgramPlan) {
    generic_coupled_program::host_prepare_step(plan);
}

fn host_coupled_before_iter(plan: &mut GpuProgramPlan) {
    generic_coupled_program::host_coupled_before_iter(plan);
}

fn host_coupled_solve(plan: &mut GpuProgramPlan) {
    generic_coupled_program::host_solve_linear_system(plan);
    generic_coupled_program::host_after_solve(plan);
}

fn host_coupled_finalize_step(plan: &mut GpuProgramPlan) {
    generic_coupled_program::host_finalize_step(plan);
}
