pub(crate) mod generic_coupled_backend;
pub(crate) mod plan;
pub(crate) mod plan_instance;

use crate::solver::gpu::program::plan::GpuProgramPlan;
use crate::solver::gpu::program::plan_instance::PlanInitConfig;
use crate::solver::mesh::Mesh;
use crate::solver::model::ModelSpec;

pub(crate) async fn build_program_plan(
    mesh: &Mesh,
    model: &ModelSpec,
    config: PlanInitConfig,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
) -> Result<GpuProgramPlan, String> {
    crate::solver::gpu::lowering::lower_program_plan(mesh, model, config, device, queue).await
}
