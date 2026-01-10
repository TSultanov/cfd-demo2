use crate::solver::gpu::plans::plan_instance::{PlanInitConfig, PlanParam, PlanParamValue};
use crate::solver::gpu::plans::program::{GpuProgramPlan, ProgramOpRegistry};
use crate::solver::gpu::recipe::SolverRecipe;
use crate::solver::mesh::Mesh;
use crate::solver::model::{KernelId, ModelSpec};
use std::collections::HashMap;

use super::models;
use super::types::{LoweredProgramParts, ModelGpuProgramSpecParts};

pub(crate) async fn lower_program_model_driven(
    mesh: &Mesh,
    model: &ModelSpec,
    config: PlanInitConfig,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
) -> Result<GpuProgramPlan, String> {
    // Derive the solver recipe from model + config
    let recipe = SolverRecipe::from_model(
        model,
        config.advection_scheme,
        config.time_scheme,
        config.preconditioner,
    )?;

    let parts = lower_parts_for_model(mesh, model, recipe.clone(), device, queue).await?;

    // Program spec is now always recipe-driven (with stable legacy templates
    // emitted when the kernel set matches those families).
    let program = recipe.build_program_spec();
    let spec = parts.spec.into_spec(program)?;
    let mut plan = GpuProgramPlan::new(
        parts.model,
        parts.context,
        parts.profiling_stats,
        parts.resources,
        spec,
    );

    plan.set_param(
        PlanParam::AdvectionScheme,
        PlanParamValue::Scheme(config.advection_scheme),
    )?;
    plan.set_param(
        PlanParam::TimeScheme,
        PlanParamValue::TimeScheme(config.time_scheme),
    )?;
    plan.set_param(
        PlanParam::Preconditioner,
        PlanParamValue::Preconditioner(config.preconditioner),
    )?;

    Ok(plan)
}

async fn lower_parts_for_model(
    mesh: &Mesh,
    model: &ModelSpec,
    recipe: SolverRecipe,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
) -> Result<LoweredProgramParts, String> {
    let has_kernel = |id: KernelId| recipe.kernels.iter().any(|k| k.id == id);

    // Generic coupled plan currently has its own dedicated lowering path.
    if has_kernel(KernelId::GENERIC_COUPLED_ASSEMBLY) {
        return crate::solver::gpu::plans::generic_coupled::GenericCoupledPlanResources::new(
            mesh,
            model.clone(),
            recipe,
            device,
            queue,
        )
        .await;
    }

    // Otherwise, select the runtime backend based on the derived recipe structure.
    match recipe.stepping {
        crate::solver::gpu::recipe::SteppingMode::Coupled { .. } => {
            let solver = crate::solver::gpu::structs::GpuSolver::new(
                mesh,
                model.clone(),
                recipe.clone(),
                device,
                queue,
            )
            .await?;

            let context = crate::solver::gpu::context::GpuContext {
                device: solver.common.context.device.clone(),
                queue: solver.common.context.queue.clone(),
            };
            let profiling_stats = std::sync::Arc::clone(&solver.common.profiling_stats);

            let mut resources = crate::solver::gpu::plans::program::ProgramResources::new();
            resources.insert(models::universal::UniversalProgramResources::new_coupled(
                solver,
            ));

            let mut ops = ProgramOpRegistry::new();
            models::universal::register_ops_from_recipe(&recipe, &mut ops)?;

            Ok(LoweredProgramParts {
                model: model.clone(),
                context,
                profiling_stats,
                resources,
                spec: ModelGpuProgramSpecParts {
                    ops,
                    num_cells: models::universal::spec_num_cells,
                    time: models::universal::spec_time,
                    dt: models::universal::spec_dt,
                    state_buffer: models::universal::spec_state_buffer,
                    write_state_bytes: models::universal::spec_write_state_bytes,
                    initialize_history: Some(models::universal::init_history),
                    params: HashMap::new(),
                    set_param_fallback: Some(models::universal::set_param_fallback),
                    step_stats: Some(models::universal::step_stats),
                    step_with_stats: None,
                    linear_debug: Some(models::universal::linear_debug_provider),
                },
            })
        }
        _ => {
            let plan = crate::solver::gpu::plans::compressible::CompressiblePlanResources::new(
                mesh,
                model.clone(),
                recipe.clone(),
                device,
                queue,
            )
            .await?;

            let context = crate::solver::gpu::context::GpuContext {
                device: plan.common.context.device.clone(),
                queue: plan.common.context.queue.clone(),
            };
            let profiling_stats = std::sync::Arc::clone(&plan.common.profiling_stats);

            let mut resources = crate::solver::gpu::plans::program::ProgramResources::new();
            resources
                .insert(models::universal::UniversalProgramResources::new_explicit_implicit(plan));

            let mut ops = ProgramOpRegistry::new();
            models::universal::register_ops_from_recipe(&recipe, &mut ops)?;

            Ok(LoweredProgramParts {
                model: model.clone(),
                context,
                profiling_stats,
                resources,
                spec: ModelGpuProgramSpecParts {
                    ops,
                    num_cells: models::universal::spec_num_cells,
                    time: models::universal::spec_time,
                    dt: models::universal::spec_dt,
                    state_buffer: models::universal::spec_state_buffer,
                    write_state_bytes: models::universal::spec_write_state_bytes,
                    initialize_history: Some(models::universal::init_history),
                    params: HashMap::new(),
                    set_param_fallback: Some(models::universal::set_param_fallback),
                    step_stats: Some(models::universal::step_stats),
                    step_with_stats: Some(models::universal::step_with_stats),
                    linear_debug: Some(models::universal::linear_debug_provider),
                },
            })
        }
    }
}
