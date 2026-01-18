use crate::solver::gpu::plans::plan_instance::PlanInitConfig;
use crate::solver::gpu::plans::program::GpuProgramPlan;
use crate::solver::mesh::Mesh;
use crate::solver::model::ModelSpec;

pub(crate) mod kernel_registry;
pub(crate) mod named_params;
mod model_driven;
pub(crate) mod models;
pub(crate) mod types;
pub(crate) mod unified_registry;

pub(crate) use model_driven::validate_model_owned_preconditioner_config;

pub(crate) async fn lower_plan_instance(
    mesh: &Mesh,
    model: &ModelSpec,
    config: PlanInitConfig,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
) -> Result<GpuProgramPlan, String> {
    model_driven::lower_program_model_driven(mesh, model, config, device, queue).await
}
