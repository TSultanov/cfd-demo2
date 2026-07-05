use crate::solver::gpu::lowering::kernel_registry;
use crate::solver::gpu::modules::resource_registry::ResourceRegistry;
use crate::solver::gpu::pipeline_cache::PipelineCache;
use crate::solver::gpu::wgsl_reflect;
use crate::solver::model::KernelId;

pub struct ScalarResources {
    pub bg_scalars: wgpu::BindGroup,
    pub pipeline_init_cg_scalars: wgpu::ComputePipeline,
    pub pipeline_reduce_rho_new_r_r: wgpu::ComputePipeline,
    pub pipeline_reduce_r0_v: wgpu::ComputePipeline,
}

pub fn init_scalars(
    device: &wgpu::Device,
    cache: &PipelineCache,
    b_scalars: &wgpu::Buffer,
    b_dot_result: &wgpu::Buffer,
    b_dot_result_2: &wgpu::Buffer,
    b_solver_params: &wgpu::Buffer,
) -> Result<ScalarResources, String> {
    let init_cg_src = kernel_registry::kernel_source_by_id("", KernelId::SCALARS_INIT_CG)
        .map_err(|e| format!("missing scalars/init_cg_scalars kernel: {e}"))?;
    let pipeline_init_cg_scalars = cache.pipeline(device, "", KernelId::SCALARS_INIT_CG)?;

    let bgl_scalars = pipeline_init_cg_scalars.get_bind_group_layout(0);
    let registry = ResourceRegistry::new()
        .with_buffer("scalars", b_scalars)
        .with_buffer("dot_result_1", b_dot_result)
        .with_buffer("dot_result_2", b_dot_result_2)
        .with_buffer("params", b_solver_params);
    let bg_scalars = wgsl_reflect::create_bind_group_from_bindings(
        device,
        "Scalars Bind Group",
        &bgl_scalars,
        init_cg_src.bindings,
        0,
        |name| registry.resolve(name),
    )
    .map_err(|e| format!("failed to create scalars bind group: {e}"))?;

    let pipeline_reduce_rho_new_r_r =
        cache.pipeline(device, "", KernelId::SCALARS_REDUCE_RHO_NEW_R_R)?;
    let pipeline_reduce_r0_v = cache.pipeline(device, "", KernelId::SCALARS_REDUCE_R0_V)?;

    Ok(ScalarResources {
        bg_scalars,
        pipeline_init_cg_scalars,
        pipeline_reduce_rho_new_r_r,
        pipeline_reduce_r0_v,
    })
}
