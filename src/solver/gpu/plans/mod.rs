pub(crate) mod compressible;
pub(crate) mod compressible_fgmres;
pub(crate) mod coupled;
pub(crate) mod coupled_fgmres;
pub(crate) mod generic_coupled;
pub(crate) mod incompressible;
pub(crate) mod plan_instance;

use crate::solver::gpu::plans::compressible::CompressiblePlanResources;
use crate::solver::gpu::plans::generic_coupled::GpuGenericCoupledSolver;
use crate::solver::gpu::plans::plan_instance::GpuPlanInstance;
use crate::solver::gpu::structs::GpuSolver;
use crate::solver::mesh::Mesh;
use crate::solver::model::{ModelFields, ModelSpec};

pub(crate) async fn build_plan_instance(
    mesh: &Mesh,
    model: &ModelSpec,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
) -> Result<Box<dyn GpuPlanInstance>, String> {
    let plan: Box<dyn GpuPlanInstance> = match &model.fields {
        ModelFields::Incompressible(_) => Box::new(GpuSolver::new(mesh, device, queue).await),
        ModelFields::Compressible(_) => {
            Box::new(CompressiblePlanResources::new(mesh, device, queue).await)
        }
        ModelFields::GenericCoupled(_) => {
            let solver = GpuGenericCoupledSolver::new(mesh, model.clone(), device, queue).await?;
            Box::new(solver)
        }
    };
    Ok(plan)
}
