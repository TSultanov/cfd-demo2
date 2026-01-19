use crate::solver::gpu::plans::plan_instance::{PlanInitConfig, PlanParamValue};
use crate::solver::gpu::plans::program::{GpuProgramPlan, ProgramOpRegistry};
use crate::solver::gpu::recipe::SolverRecipe;
use crate::solver::mesh::Mesh;
use crate::solver::model::ModelSpec;

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
        config.stepping,
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

    // Apply the preconditioner config only if the model declares a handler for it.
    if model.named_param_keys().into_iter().any(|k| k == "preconditioner") {
        plan.set_named_param(
            "preconditioner",
            PlanParamValue::Preconditioner(config.preconditioner),
        )?;
    }

    Ok(plan)
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
            // Model-owned Schur remains authoritative, but we still allow runtime configuration
            // to select the pressure solve strategy (Chebyshev vs AMG).
            match config_preconditioner {
                crate::solver::gpu::structs::PreconditionerType::Jacobi
                | crate::solver::gpu::structs::PreconditionerType::Amg
                | crate::solver::gpu::structs::PreconditionerType::BlockJacobi => Ok(()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::gpu::structs::PreconditionerType;
    use crate::solver::model::{incompressible_momentum_generic_model, ModelPreconditionerSpec};

    #[test]
    fn model_owned_schur_allows_preconditioner_config() {
        let model = incompressible_momentum_generic_model();
        assert!(matches!(
            model.linear_solver.unwrap().preconditioner,
            ModelPreconditionerSpec::Schur { .. }
        ));
        validate_model_owned_preconditioner_config(&model, PreconditionerType::Amg)
            .expect("AMG pressure solve should be allowed for model-owned Schur");
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

    let named_params = models::generic_coupled::named_params_for_recipe(&built.model, &recipe)?;

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
            set_bc_value: Some(models::universal::spec_set_bc_value),
            initialize_history: None,
            named_params,
            set_named_param_fallback: None,
            step_stats: Some(models::universal::step_stats),
            step_with_stats: None,
            linear_debug: Some(models::universal::linear_debug_provider),
        },
    })
}
