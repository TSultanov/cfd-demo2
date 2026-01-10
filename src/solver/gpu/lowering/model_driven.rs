use crate::solver::gpu::plans::plan_instance::{PlanInitConfig, PlanParam, PlanParamValue};
use crate::solver::gpu::plans::program::{GpuProgramPlan, ProgramOpRegistry};
use crate::solver::gpu::recipe::SolverRecipe;
use crate::solver::mesh::Mesh;
use crate::solver::model::ModelSpec;
use std::collections::HashMap;

use super::models;
use super::templates::{build_program_spec, ProgramTemplateKind};
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

    let template = ProgramTemplateKind::for_model(model)?;
    let parts = lower_parts_for_template(template, mesh, model, recipe, device, queue).await?;

    let program = build_program_spec(template);
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

async fn lower_parts_for_template(
    template: ProgramTemplateKind,
    mesh: &Mesh,
    model: &ModelSpec,
    recipe: SolverRecipe,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
) -> Result<LoweredProgramParts, String> {
    match template {
        ProgramTemplateKind::Compressible => {
            let plan = crate::solver::gpu::plans::compressible::CompressiblePlanResources::new(
                mesh,
                model.clone(),
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
            resources.insert(models::compressible::CompressibleProgramResources::new(
                plan,
            ));

            let mut ops = ProgramOpRegistry::new();
            models::compressible::register_ops(&mut ops)?;

            Ok(LoweredProgramParts {
                model: model.clone(),
                recipe,
                context,
                profiling_stats,
                resources,
                spec: ModelGpuProgramSpecParts {
                    ops,
                    num_cells: models::compressible::spec_num_cells,
                    time: models::compressible::spec_time,
                    dt: models::compressible::spec_dt,
                    state_buffer: models::compressible::spec_state_buffer,
                    write_state_bytes: models::compressible::spec_write_state_bytes,
                    initialize_history: Some(models::compressible::init_history),
                    params: HashMap::new(),
                    set_param_fallback: Some(models::compressible::set_param_fallback),
                    step_stats: Some(models::compressible::step_stats),
                    step_with_stats: Some(models::compressible::step_with_stats),
                    linear_debug: Some(models::compressible::linear_debug_provider),
                },
            })
        }
        ProgramTemplateKind::IncompressibleCoupled => {
            let plan =
                crate::solver::gpu::structs::GpuSolver::new(mesh, model.clone(), device, queue)
                    .await?;

            let context = crate::solver::gpu::context::GpuContext {
                device: plan.common.context.device.clone(),
                queue: plan.common.context.queue.clone(),
            };
            let profiling_stats = std::sync::Arc::clone(&plan.common.profiling_stats);

            let mut resources = crate::solver::gpu::plans::program::ProgramResources::new();
            resources.insert(models::incompressible::IncompressibleProgramResources::new(
                plan,
            ));

            let mut ops = ProgramOpRegistry::new();
            models::incompressible::register_ops(&mut ops)?;

            Ok(LoweredProgramParts {
                model: model.clone(),
                recipe,
                context,
                profiling_stats,
                resources,
                spec: ModelGpuProgramSpecParts {
                    ops,
                    num_cells: models::incompressible::spec_num_cells,
                    time: models::incompressible::spec_time,
                    dt: models::incompressible::spec_dt,
                    state_buffer: models::incompressible::spec_state_buffer,
                    write_state_bytes: models::incompressible::spec_write_state_bytes,
                    initialize_history: Some(models::incompressible::init_history),
                    params: HashMap::new(),
                    set_param_fallback: Some(models::incompressible::set_param_fallback),
                    step_stats: Some(models::incompressible::step_stats),
                    step_with_stats: None,
                    linear_debug: Some(models::incompressible::linear_debug_provider),
                },
            })
        }
        ProgramTemplateKind::GenericCoupledScalar => {
            crate::solver::gpu::plans::generic_coupled::GenericCoupledPlanResources::new(
                mesh,
                model.clone(),
                recipe,
                device,
                queue,
            )
            .await
        }
    }
}
