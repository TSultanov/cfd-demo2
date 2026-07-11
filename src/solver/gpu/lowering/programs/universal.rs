use super::generic_coupled as generic_coupled_program;
use crate::solver::gpu::execution_plan::{GraphDetail, GraphExecMode};
use crate::solver::gpu::lowering::unified_registry::UnifiedOpRegistryConfig;
use crate::solver::gpu::program::plan::{GpuProgramPlan, ProgramOpRegistry};
use crate::solver::gpu::program::plan_instance::{
    PlanLinearSystemDebug, PlanStepStats,
};
use crate::solver::gpu::recipe::{SolverRecipe, SteppingMode};

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
            explicit_stage_time: [
                Some(host_explicit_stage_1_time),
                Some(host_explicit_stage_2_time),
                Some(host_explicit_stage_3_time),
                Some(host_explicit_stage_4_time),
            ],
            explicit_residual_graph: Some(explicit_residual_graph_run),
            explicit_stage_graph: [
                Some(explicit_stage_1_graph_run),
                Some(explicit_stage_2_graph_run),
                Some(explicit_stage_3_graph_run),
                Some(explicit_stage_4_graph_run),
            ],
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
                prepare: Some(host_coupled_begin_step),
                finalize: Some(host_coupled_finalize_step),
                solve: Some(host_coupled_solve),
                coupled_init_prepare_graph: Some(coupled_graph_init_prepare_run),
                coupled_iter_prepare_graph: Some(coupled_graph_iter_prepare_run),
                coupled_before_iter: Some(host_coupled_before_iter),
                coupled_outer_iters: Some(coupled_outer_iters),

                assembly_graph: Some(coupled_assembly_graph_run),
                update_graph: Some(coupled_update_graph_run),
                ..Default::default()
            }
        }
    };

    let built =
        crate::solver::gpu::lowering::unified_registry::build_unified_registry(recipe, config)?;
    registry.merge(built)
}

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

pub(in crate::solver::gpu::lowering) fn spec_write_state_bytes_current(
    plan: &GpuProgramPlan,
    bytes: &[u8],
) -> Result<(), String> {
    generic_coupled_program::spec_write_state_bytes_current(plan, bytes)
}

pub(in crate::solver::gpu::lowering) fn spec_reinit_cells(
    plan: &GpuProgramPlan,
    cells: &[u32],
    rows: &[f32],
    new_vols: &[f64],
) -> Result<(), String> {
    generic_coupled_program::spec_reinit_cells(plan, cells, rows, new_vols)
}

pub(in crate::solver::gpu::lowering) fn spec_permute_cells(
    plan: &GpuProgramPlan,
    perm: &[u32],
) -> Result<(), String> {
    generic_coupled_program::spec_permute_cells(plan, perm)
}

pub(in crate::solver::gpu::lowering) fn spec_set_bc_value(
    plan: &GpuProgramPlan,
    boundary: crate::solver::gpu::enums::GpuBoundaryType,
    unknown_component: u32,
    value: f32,
) -> Result<(), String> {
    generic_coupled_program::spec_set_bc_value(plan, boundary, unknown_component, value)
}

pub(in crate::solver::gpu::lowering) fn spec_set_bc_values_per_face(
    plan: &GpuProgramPlan,
    boundary: crate::solver::gpu::enums::GpuBoundaryType,
    unknown_component: u32,
    value_for_face: &dyn Fn(u32) -> f32,
) -> Result<(), String> {
    generic_coupled_program::spec_set_bc_values_per_face(
        plan,
        boundary,
        unknown_component,
        value_for_face,
    )
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
        outer_step_status: plan.outer_step_status,
        step_attempt_count: (plan.step_attempt_count > 0).then_some(plan.step_attempt_count),
        rejected_retry_count: (plan.rejected_retry_count > 0)
            .then_some(plan.rejected_retry_count),
        current_dt: Some(plan.dt()),
        current_dtau: plan.current_dtau,
        positivity_min_rho: plan.positivity_min_rho,
        positivity_min_p: plan.positivity_min_p,
        positivity_rho_undershoot_count: Some(plan.positivity_rho_undershoot_count),
        positivity_pressure_undershoot_count: Some(plan.positivity_pressure_undershoot_count),
        linear_stats,
        ..Default::default()
    }
}

pub(in crate::solver::gpu::lowering) fn linear_debug_provider(
    plan: &mut GpuProgramPlan,
) -> Option<&mut dyn PlanLinearSystemDebug> {
    plan.resources
        .backend
        .has_linear_system()
        .then_some(&mut plan.resources.backend as &mut dyn PlanLinearSystemDebug)
}

fn host_explicit_prepare(plan: &mut GpuProgramPlan) {
    generic_coupled_program::host_prepare_explicit_step(plan);
}

fn host_explicit_stage_1_time(plan: &mut GpuProgramPlan) {
    generic_coupled_program::host_set_explicit_stage_1_time(plan);
}

fn host_explicit_stage_2_time(plan: &mut GpuProgramPlan) {
    generic_coupled_program::host_set_explicit_stage_2_time(plan);
}

fn host_explicit_stage_3_time(plan: &mut GpuProgramPlan) {
    generic_coupled_program::host_set_explicit_stage_3_time(plan);
}

fn host_explicit_stage_4_time(plan: &mut GpuProgramPlan) {
    generic_coupled_program::host_set_explicit_stage_4_time(plan);
}

fn explicit_residual_graph_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    generic_coupled_program::explicit_residual_graph_run(plan, context, mode)
}

macro_rules! explicit_stage_graph_runner {
    ($name:ident, $target:ident) => {
        fn $name(
            plan: &GpuProgramPlan,
            context: &crate::solver::gpu::context::GpuContext,
            mode: GraphExecMode,
        ) -> (f64, Option<GraphDetail>) {
            generic_coupled_program::$target(plan, context, mode)
        }
    };
}

explicit_stage_graph_runner!(explicit_stage_1_graph_run, explicit_stage_1_graph_run);
explicit_stage_graph_runner!(explicit_stage_2_graph_run, explicit_stage_2_graph_run);
explicit_stage_graph_runner!(explicit_stage_3_graph_run, explicit_stage_3_graph_run);
explicit_stage_graph_runner!(explicit_stage_4_graph_run, explicit_stage_4_graph_run);

fn host_explicit_finalize(plan: &mut GpuProgramPlan) {
    generic_coupled_program::host_finalize_explicit_step(plan);
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

fn coupled_graph_init_prepare_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    generic_coupled_program::init_prepare_graph_run(plan, context, mode)
}

fn coupled_graph_iter_prepare_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    // Fused host path: encoded into the solve submission instead (see
    // `try_host_coupled_solve_fused`).
    if generic_coupled_program::coupled_host_fusion_active(plan) {
        return (0.0, None);
    }
    generic_coupled_program::iter_prepare_graph_run(plan, context, mode)
}

fn coupled_assembly_graph_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    // Fused host path: encoded into the solve submission instead (see
    // `try_host_coupled_solve_fused`).
    if generic_coupled_program::coupled_host_fusion_active(plan) {
        return (0.0, None);
    }
    generic_coupled_program::assembly_graph_run(plan, context, mode)
}

fn coupled_update_graph_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    // Fused host path: encoded into the solve submission instead (see
    // `try_host_coupled_solve_fused`).
    if generic_coupled_program::coupled_host_fusion_active(plan) {
        return (0.0, None);
    }
    generic_coupled_program::update_graph_run(plan, context, mode)
}

fn coupled_outer_iters(plan: &GpuProgramPlan) -> usize {
    // Reuse the implicit path's `OuterIters` knob to control nonlinear corrector iterations per step.
    generic_coupled_program::count_outer_iters(plan)
}

fn host_coupled_begin_step(plan: &mut GpuProgramPlan) {
    generic_coupled_program::host_prepare_step(plan);
}

fn host_coupled_before_iter(plan: &mut GpuProgramPlan) {
    generic_coupled_program::host_coupled_before_iter(plan);
}

fn host_coupled_solve(plan: &mut GpuProgramPlan) {
    if !generic_coupled_program::try_host_coupled_solve_fused(plan) {
        generic_coupled_program::host_solve_linear_system(plan);
    }
    generic_coupled_program::host_after_solve(plan);
}

fn host_coupled_finalize_step(plan: &mut GpuProgramPlan) {
    generic_coupled_program::host_finalize_step(plan);
}
