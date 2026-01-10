pub(crate) mod explicit_implicit;
pub(crate) mod generic_coupled;
pub(crate) mod plan_instance;
pub(crate) mod program;

use crate::solver::gpu::plans::plan_instance::PlanInitConfig;
use crate::solver::gpu::plans::program::GpuProgramPlan;
use crate::solver::mesh::Mesh;
use crate::solver::model::ModelSpec;

pub(crate) async fn build_plan_instance(
    mesh: &Mesh,
    model: &ModelSpec,
    config: PlanInitConfig,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
) -> Result<GpuProgramPlan, String> {
    crate::solver::gpu::lowering::lower_plan_instance(mesh, model, config, device, queue).await
}
