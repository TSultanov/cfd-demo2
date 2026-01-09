use crate::solver::gpu::plans::plan_instance::{PlanInitConfig, PlanParam, PlanParamValue};
use crate::solver::gpu::plans::program::GpuProgramPlan;
use crate::solver::mesh::Mesh;
use crate::solver::model::{ModelFields, ModelSpec};

mod compressible_program;
mod generic_coupled_program;
mod incompressible_program;

async fn lower_program(
    mesh: &Mesh,
    model: &ModelSpec,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
) -> Result<GpuProgramPlan, String> {
    match &model.fields {
        ModelFields::Incompressible(_) => {
            incompressible_program::lower_incompressible_program(mesh, model.clone(), device, queue)
                .await
        }
        ModelFields::Compressible(_) => {
            compressible_program::lower_compressible_program(mesh, model.clone(), device, queue)
                .await
        }
        ModelFields::GenericCoupled(_) => {
            generic_coupled_program::lower_generic_coupled_program(
                mesh,
                model.clone(),
                device,
                queue,
            )
            .await
        }
    }
}

pub(crate) async fn lower_plan_instance(
    mesh: &Mesh,
    model: &ModelSpec,
    config: PlanInitConfig,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
) -> Result<GpuProgramPlan, String> {
    let mut plan = lower_program(mesh, model, device, queue).await?;
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
