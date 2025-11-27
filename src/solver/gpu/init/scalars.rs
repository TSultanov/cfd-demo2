use std::borrow::Cow;

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
    // Scalar Pipelines
    let shader_scalars = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Scalars Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../../shaders/scalars.wgsl"))),
    });

    let bgl_scalars = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Scalars Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bg_scalars = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Scalars Bind Group"),
        layout: &bgl_scalars,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: b_scalars.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_dot_result.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: b_dot_result_2.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: b_solver_params.as_entire_binding(),
            },
        ],
    });

    let pl_scalars = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Scalars Pipeline Layout"),
        bind_group_layouts: &[&bgl_scalars],
        push_constant_ranges: &[],
    });

    let pipeline_init_scalars = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Init Scalars Pipeline"),
        layout: Some(&pl_scalars),
        module: &shader_scalars,
        entry_point: "init_scalars",
    });

    let pipeline_init_cg_scalars =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Init CG Scalars Pipeline"),
            layout: Some(&pl_scalars),
            module: &shader_scalars,
            entry_point: "init_cg_scalars",
        });

    let pipeline_reduce_rho_new_r_r =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Reduce Rho New R R Pipeline"),
            layout: Some(&pl_scalars),
            module: &shader_scalars,
            entry_point: "reduce_rho_new_r_r",
        });

    let pipeline_reduce_r0_v = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Reduce R0 V Pipeline"),
        layout: Some(&pl_scalars),
        module: &shader_scalars,
        entry_point: "reduce_r0_v",
    });

    let pipeline_reduce_t_s_t_t =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Reduce T S T T Pipeline"),
            layout: Some(&pl_scalars),
            module: &shader_scalars,
            entry_point: "reduce_t_s_t_t",
        });

    let pipeline_update_cg_alpha =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Update CG Alpha Pipeline"),
            layout: Some(&pl_scalars),
            module: &shader_scalars,
            entry_point: "update_cg_alpha",
        });

    let pipeline_update_cg_beta =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Update CG Beta Pipeline"),
            layout: Some(&pl_scalars),
            module: &shader_scalars,
            entry_point: "update_cg_beta",
        });

    let pipeline_update_rho_old =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Update Rho Old Pipeline"),
            layout: Some(&pl_scalars),
            module: &shader_scalars,
            entry_point: "update_rho_old",
        });

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
