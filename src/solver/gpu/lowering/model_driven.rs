use crate::solver::gpu::plans::plan_instance::{PlanInitConfig, PlanParam, PlanParamValue};
use crate::solver::gpu::plans::program::{GpuProgramPlan, ProgramOpRegistry};
use crate::solver::gpu::recipe::SolverRecipe;
use crate::solver::mesh::Mesh;
use crate::solver::model::ModelSpec;
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
    validate_model_owned_preconditioner_config(model, config.preconditioner)?;

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

    // If the model owns the preconditioner choice, do not allow runtime/config overrides.
    if should_set_preconditioner_param(model) {
        plan.set_param(
            PlanParam::Preconditioner,
            PlanParamValue::Preconditioner(config.preconditioner),
        )?;
    }

    Ok(plan)
}

fn should_set_preconditioner_param(model: &ModelSpec) -> bool {
    let Some(solver) = model.linear_solver else {
        return true;
    };
    // Any explicit model-owned selection disables plan/runtime overrides.
    !matches!(solver.preconditioner, crate::solver::model::ModelPreconditionerSpec::Schur { .. })
}

pub(crate) fn validate_model_owned_preconditioner_config(
    model: &ModelSpec,
    config_preconditioner: crate::solver::gpu::structs::PreconditionerType,
) -> Result<(), String> {
    let Some(solver) = model.linear_solver else {
        return Ok(());
    };
    match solver.preconditioner {
        crate::solver::model::ModelPreconditionerSpec::Default => Ok(()),
        crate::solver::model::ModelPreconditionerSpec::Schur { .. } => {
            // SolverConfig's `PreconditionerType` is not the same concept as the model-owned
            // Schur preconditioner. Until we remove SolverConfig-level preconditioners entirely,
            // disallow non-default values when a model-owned preconditioner is active.
            if config_preconditioner != crate::solver::gpu::structs::PreconditionerType::Jacobi {
                return Err(
                    "preconditioner is model-owned for this model; do not set SolverConfig.preconditioner".to_string(),
                );
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::gpu::structs::PreconditionerType;
    use crate::solver::model::{incompressible_momentum_generic_model, ModelPreconditionerSpec};

    #[test]
    fn model_owned_schur_rejects_nondefault_preconditioner_config() {
        let model = incompressible_momentum_generic_model();
        assert!(matches!(
            model.linear_solver.unwrap().preconditioner,
            ModelPreconditionerSpec::Schur { .. }
        ));
        assert!(validate_model_owned_preconditioner_config(&model, PreconditionerType::Amg)
            .unwrap_err()
            .contains("model-owned"));
    }

    #[test]
    fn model_owned_schur_skips_setting_plan_preconditioner_param() {
        let model = incompressible_momentum_generic_model();
        assert!(!should_set_preconditioner_param(&model));
    }
}

async fn lower_parts_for_model(
    mesh: &Mesh,
    model: &ModelSpec,
    recipe: SolverRecipe,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
) -> Result<LoweredProgramParts, String> {
    // Otherwise, select the runtime backend based on the derived recipe structure.
    match recipe.stepping {
        crate::solver::gpu::recipe::SteppingMode::Coupled { .. } => {
            if model.method != crate::solver::model::MethodSpec::CoupledIncompressible {
                let built = crate::solver::gpu::plans::generic_coupled::build_generic_coupled_backend(
                    mesh,
                    model.clone(),
                    recipe.clone(),
                    device,
                    queue,
                )
                .await?;

                let mut ops = ProgramOpRegistry::new();
                models::universal::register_ops_from_recipe(&recipe, &mut ops)?;

                let mut resources = crate::solver::gpu::plans::program::ProgramResources::new();
                resources.insert(models::universal::UniversalProgramResources::new_generic_coupled(
                    built.backend,
                ));

                return Ok(LoweredProgramParts {
                    model: built.model,
                    context: built.context,
                    profiling_stats: built.profiling_stats,
                    resources,
                    spec: ModelGpuProgramSpecParts {
                        ops,
                        num_cells: models::universal::spec_num_cells,
                        time: models::universal::spec_time,
                        dt: models::universal::spec_dt,
                        state_buffer: models::universal::spec_state_buffer,
                        write_state_bytes: models::universal::spec_write_state_bytes,
                        initialize_history: None,
                        params: HashMap::new(),
                        set_param_fallback: Some(models::universal::set_param_fallback),
                        step_stats: Some(models::universal::step_stats),
                        step_with_stats: None,
                        linear_debug: Some(models::universal::linear_debug_provider),
                    },
                });
            }

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
        crate::solver::gpu::recipe::SteppingMode::Implicit { .. } => {
            let built = crate::solver::gpu::plans::generic_coupled::build_generic_coupled_backend(
                mesh,
                model.clone(),
                recipe.clone(),
                device,
                queue,
            )
            .await?;

            let mut ops = ProgramOpRegistry::new();
            models::universal::register_ops_from_recipe(&recipe, &mut ops)?;

            let mut resources = crate::solver::gpu::plans::program::ProgramResources::new();
            resources.insert(models::universal::UniversalProgramResources::new_generic_coupled(
                built.backend,
            ));

            Ok(LoweredProgramParts {
                model: built.model,
                context: built.context,
                profiling_stats: built.profiling_stats,
                resources,
                spec: ModelGpuProgramSpecParts {
                    ops,
                    num_cells: models::universal::spec_num_cells,
                    time: models::universal::spec_time,
                    dt: models::universal::spec_dt,
                    state_buffer: models::universal::spec_state_buffer,
                    write_state_bytes: models::universal::spec_write_state_bytes,
                    initialize_history: None,
                    params: HashMap::new(),
                    set_param_fallback: Some(models::universal::set_param_fallback),
                    step_stats: Some(models::universal::step_stats),
                    step_with_stats: None,
                    linear_debug: Some(models::universal::linear_debug_provider),
                },
            })
        }
        crate::solver::gpu::recipe::SteppingMode::Explicit => {
            let built = crate::solver::gpu::plans::generic_coupled::build_generic_coupled_backend(
                mesh,
                model.clone(),
                recipe.clone(),
                device,
                queue,
            )
            .await?;

            let mut ops = ProgramOpRegistry::new();
            models::universal::register_ops_from_recipe(&recipe, &mut ops)?;

            let mut resources = crate::solver::gpu::plans::program::ProgramResources::new();
            resources.insert(models::universal::UniversalProgramResources::new_generic_coupled(
                built.backend,
            ));

            Ok(LoweredProgramParts {
                model: built.model,
                context: built.context,
                profiling_stats: built.profiling_stats,
                resources,
                spec: ModelGpuProgramSpecParts {
                    ops,
                    num_cells: models::universal::spec_num_cells,
                    time: models::universal::spec_time,
                    dt: models::universal::spec_dt,
                    state_buffer: models::universal::spec_state_buffer,
                    write_state_bytes: models::universal::spec_write_state_bytes,
                    initialize_history: None,
                    params: HashMap::new(),
                    set_param_fallback: Some(models::universal::set_param_fallback),
                    step_stats: Some(models::universal::step_stats),
                    step_with_stats: None,
                    linear_debug: Some(models::universal::linear_debug_provider),
                },
            })
        }
    }
}
