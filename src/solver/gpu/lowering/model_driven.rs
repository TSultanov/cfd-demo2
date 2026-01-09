use crate::solver::gpu::plans::plan_instance::{PlanInitConfig, PlanParam, PlanParamValue};
use crate::solver::gpu::plans::program::GpuProgramPlan;
use crate::solver::mesh::Mesh;
use crate::solver::model::ModelSpec;

use super::models;
use super::templates::{build_program_spec, ProgramTemplateKind};
use super::types::LoweredProgramParts;

pub(crate) async fn lower_program_model_driven(
    mesh: &Mesh,
    model: &ModelSpec,
    config: PlanInitConfig,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
) -> Result<GpuProgramPlan, String> {
    let template = ProgramTemplateKind::for_model(model)?;
    let parts: LoweredProgramParts = match template {
        ProgramTemplateKind::Compressible => {
            models::compressible::lower_parts(mesh, model.clone(), device, queue).await?
        }
        ProgramTemplateKind::IncompressibleCoupled => {
            models::incompressible::lower_parts(mesh, model.clone(), device, queue).await?
        }
        ProgramTemplateKind::GenericCoupledScalar => {
            models::generic_coupled::lower_parts(mesh, model.clone(), device, queue).await?
        }
    };

    let program = build_program_spec(template);
    let spec = parts.spec.into_spec(program);
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

