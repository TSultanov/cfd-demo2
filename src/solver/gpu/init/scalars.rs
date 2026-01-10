use crate::solver::gpu::bindings::scalars;
use crate::solver::gpu::lowering::kernel_registry;
use crate::solver::model::KernelId;

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

    let pipeline_init_scalars = {
        let src = kernel_registry::kernel_source_by_id("", KernelId::SCALARS_INIT)
            .unwrap_or_else(|e| panic!("missing scalars/init_scalars kernel: {e}"));
        (src.create_pipeline)(device)
    };
    let pipeline_init_cg_scalars = {
        let src = kernel_registry::kernel_source_by_id("", KernelId::SCALARS_INIT_CG)
            .unwrap_or_else(|e| panic!("missing scalars/init_cg_scalars kernel: {e}"));
        (src.create_pipeline)(device)
    };
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
    let pipeline_reduce_t_s_t_t = {
        let src = kernel_registry::kernel_source_by_id("", KernelId::SCALARS_REDUCE_T_S_T_T)
            .unwrap_or_else(|e| panic!("missing scalars/reduce_t_s_t_t kernel: {e}"));
        (src.create_pipeline)(device)
    };
    let pipeline_update_cg_alpha = {
        let src = kernel_registry::kernel_source_by_id("", KernelId::SCALARS_UPDATE_CG_ALPHA)
            .unwrap_or_else(|e| panic!("missing scalars/update_cg_alpha kernel: {e}"));
        (src.create_pipeline)(device)
    };
    let pipeline_update_cg_beta = {
        let src = kernel_registry::kernel_source_by_id("", KernelId::SCALARS_UPDATE_CG_BETA)
            .unwrap_or_else(|e| panic!("missing scalars/update_cg_beta kernel: {e}"));
        (src.create_pipeline)(device)
    };
    let pipeline_update_rho_old = {
        let src = kernel_registry::kernel_source_by_id("", KernelId::SCALARS_UPDATE_RHO_OLD)
            .unwrap_or_else(|e| panic!("missing scalars/update_rho_old kernel: {e}"));
        (src.create_pipeline)(device)
    };

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
