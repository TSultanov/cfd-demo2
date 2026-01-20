use crate::solver::gpu::lowering::kernel_registry;
use crate::solver::gpu::modules::resource_registry::ResourceRegistry;
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
    b_scalars: &wgpu::Buffer,
    b_dot_result: &wgpu::Buffer,
    b_dot_result_2: &wgpu::Buffer,
    b_solver_params: &wgpu::Buffer,
) -> ScalarResources {
    let init_cg_src = kernel_registry::kernel_source_by_id("", KernelId::SCALARS_INIT_CG)
        .unwrap_or_else(|e| panic!("missing scalars/init_cg_scalars kernel: {e}"));
    let pipeline_init_cg_scalars = (init_cg_src.create_pipeline)(device);

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
    .unwrap_or_else(|err| panic!("failed to create scalars bind group: {err}"));

    let pipeline_reduce_rho_new_r_r = {
        let src = kernel_registry::kernel_source_by_id("", KernelId::SCALARS_REDUCE_RHO_NEW_R_R)
            .unwrap_or_else(|e| panic!("missing scalars/reduce_rho_new_r_r kernel: {e}"));
        (src.create_pipeline)(device)
    };
    let pipeline_reduce_r0_v = {
        let src = kernel_registry::kernel_source_by_id("", KernelId::SCALARS_REDUCE_R0_V)
            .unwrap_or_else(|e| panic!("missing scalars/reduce_r0_v kernel: {e}"));
        (src.create_pipeline)(device)
    };

    ScalarResources {
        bg_scalars,
        pipeline_init_cg_scalars,
        pipeline_reduce_rho_new_r_r,
        pipeline_reduce_r0_v,
    }
}
