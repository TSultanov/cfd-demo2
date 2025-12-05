use crate::solver::gpu::bindings::scalars;

pub struct ScalarResources {
    pub bg_scalars: wgpu::BindGroup,
    pub pipeline_init_scalars: wgpu::ComputePipeline,
    pub pipeline_init_cg_scalars: wgpu::ComputePipeline,
    pub pipeline_reduce_rho_new_r_r: wgpu::ComputePipeline,
    pub pipeline_reduce_r0_v: wgpu::ComputePipeline,
    pub pipeline_reduce_t_s_t_t: wgpu::ComputePipeline,
    pub pipeline_update_cg_alpha: wgpu::ComputePipeline,
    pub pipeline_update_cg_beta: wgpu::ComputePipeline,
    pub pipeline_update_rho_old: wgpu::ComputePipeline,
}

pub fn init_scalars(
    device: &wgpu::Device,
    b_scalars: &wgpu::Buffer,
    b_dot_result: &wgpu::Buffer,
    b_dot_result_2: &wgpu::Buffer,
    b_solver_params: &wgpu::Buffer,
) -> ScalarResources {
    // Group 0: Scalars
    let bgl_scalars = device.create_bind_group_layout(&scalars::WgpuBindGroup0::LAYOUT_DESCRIPTOR);

    let bg_scalars = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Scalars Bind Group"),
        layout: &bgl_scalars,
        entries: &scalars::WgpuBindGroup0Entries::new(scalars::WgpuBindGroup0EntriesParams {
            scalars: b_scalars.as_entire_buffer_binding(),
            dot_result_1: b_dot_result.as_entire_buffer_binding(),
            dot_result_2: b_dot_result_2.as_entire_buffer_binding(),
            params: b_solver_params.as_entire_buffer_binding(),
        })
        .into_array(),
    });

    let pipeline_init_scalars = scalars::compute::create_init_scalars_pipeline_embed_source(device);
    let pipeline_init_cg_scalars =
        scalars::compute::create_init_cg_scalars_pipeline_embed_source(device);
    let pipeline_reduce_rho_new_r_r =
        scalars::compute::create_reduce_rho_new_r_r_pipeline_embed_source(device);
    let pipeline_reduce_r0_v = scalars::compute::create_reduce_r0_v_pipeline_embed_source(device);
    let pipeline_reduce_t_s_t_t =
        scalars::compute::create_reduce_t_s_t_t_pipeline_embed_source(device);
    let pipeline_update_cg_alpha =
        scalars::compute::create_update_cg_alpha_pipeline_embed_source(device);
    let pipeline_update_cg_beta =
        scalars::compute::create_update_cg_beta_pipeline_embed_source(device);
    let pipeline_update_rho_old =
        scalars::compute::create_update_rho_old_pipeline_embed_source(device);

    ScalarResources {
        bg_scalars,
        pipeline_init_scalars,
        pipeline_init_cg_scalars,
        pipeline_reduce_rho_new_r_r,
        pipeline_reduce_r0_v,
        pipeline_reduce_t_s_t_t,
        pipeline_update_cg_alpha,
        pipeline_update_cg_beta,
        pipeline_update_rho_old,
    }
}
