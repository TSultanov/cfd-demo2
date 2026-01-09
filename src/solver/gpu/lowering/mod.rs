use crate::solver::gpu::plans::compressible::CompressiblePlanResources;
use crate::solver::gpu::plans::generic_coupled::GpuGenericCoupledSolver;
use crate::solver::gpu::plans::plan_instance::{GpuPlanInstance, PlanFuture};
use crate::solver::gpu::structs::GpuSolver;
use crate::solver::mesh::Mesh;
use crate::solver::model::{ModelFields, ModelSpec};

pub(crate) trait ModelLowerer: Send + Sync {
    fn can_lower(&self, model: &ModelSpec) -> bool;

    fn lower<'a>(
        &'a self,
        mesh: &'a Mesh,
        model: &'a ModelSpec,
        device: Option<wgpu::Device>,
        queue: Option<wgpu::Queue>,
    ) -> PlanFuture<'a, Result<Box<dyn GpuPlanInstance>, String>>;
}

struct IncompressibleLowerer;

impl ModelLowerer for IncompressibleLowerer {
    fn can_lower(&self, model: &ModelSpec) -> bool {
        matches!(model.fields, ModelFields::Incompressible(_))
    }

    fn lower<'a>(
        &'a self,
        mesh: &'a Mesh,
        _model: &'a ModelSpec,
        device: Option<wgpu::Device>,
        queue: Option<wgpu::Queue>,
    ) -> PlanFuture<'a, Result<Box<dyn GpuPlanInstance>, String>> {
        Box::pin(async move {
            Ok(Box::new(GpuSolver::new(mesh, device, queue).await) as Box<dyn GpuPlanInstance>)
        })
    }
}

struct CompressibleLowerer;

impl ModelLowerer for CompressibleLowerer {
    fn can_lower(&self, model: &ModelSpec) -> bool {
        matches!(model.fields, ModelFields::Compressible(_))
    }

    fn lower<'a>(
        &'a self,
        mesh: &'a Mesh,
        _model: &'a ModelSpec,
        device: Option<wgpu::Device>,
        queue: Option<wgpu::Queue>,
    ) -> PlanFuture<'a, Result<Box<dyn GpuPlanInstance>, String>> {
        Box::pin(async move {
            Ok(Box::new(CompressiblePlanResources::new(mesh, device, queue).await)
                as Box<dyn GpuPlanInstance>)
        })
    }
}

struct GenericCoupledLowerer;

impl ModelLowerer for GenericCoupledLowerer {
    fn can_lower(&self, model: &ModelSpec) -> bool {
        matches!(model.fields, ModelFields::GenericCoupled(_))
    }

    fn lower<'a>(
        &'a self,
        mesh: &'a Mesh,
        model: &'a ModelSpec,
        device: Option<wgpu::Device>,
        queue: Option<wgpu::Queue>,
    ) -> PlanFuture<'a, Result<Box<dyn GpuPlanInstance>, String>> {
        Box::pin(async move {
            let solver = GpuGenericCoupledSolver::new(mesh, model.clone(), device, queue).await?;
            Ok(Box::new(solver) as Box<dyn GpuPlanInstance>)
        })
    }
}

fn registry() -> &'static [Box<dyn ModelLowerer>] {
    static REGISTRY: std::sync::OnceLock<Vec<Box<dyn ModelLowerer>>> = std::sync::OnceLock::new();
    REGISTRY.get_or_init(|| {
        vec![
            Box::new(IncompressibleLowerer),
            Box::new(CompressibleLowerer),
            Box::new(GenericCoupledLowerer),
        ]
    })
}

pub(crate) async fn lower_plan_instance(
    mesh: &Mesh,
    model: &ModelSpec,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
) -> Result<Box<dyn GpuPlanInstance>, String> {
    let lowerer = registry()
        .iter()
        .find(|lowerer| lowerer.can_lower(model))
        .ok_or_else(|| format!("no model lowerer registered for model fields: {:?}", model.fields))?;
    lowerer.lower(mesh, model, device, queue).await
}
