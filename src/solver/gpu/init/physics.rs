use std::borrow::Cow;

pub struct PhysicsPipelines {
    pub pipeline_gradient: wgpu::ComputePipeline,
    pub pipeline_flux: wgpu::ComputePipeline,
    pub pipeline_momentum_assembly: wgpu::ComputePipeline,
    pub pipeline_pressure_assembly: wgpu::ComputePipeline,
    pub pipeline_pressure_assembly_with_grad: wgpu::ComputePipeline,
    pub pipeline_flux_rhie_chow: wgpu::ComputePipeline,
    pub pipeline_velocity_correction: wgpu::ComputePipeline,
    pub pipeline_update_u_component: wgpu::ComputePipeline,
}

pub fn init_physics_pipelines(
    device: &wgpu::Device,
    bgl_mesh: &wgpu::BindGroupLayout,
    bgl_fields: &wgpu::BindGroupLayout,
    bgl_solver: &wgpu::BindGroupLayout,
    bgl_linear_state_ro: &wgpu::BindGroupLayout,
) -> PhysicsPipelines {
    // Shaders
    let shader_gradient = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Gradient Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/gradient.wgsl"))),
    });

    let pl_gradient = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Gradient Pipeline Layout"),
        bind_group_layouts: &[bgl_mesh, bgl_fields], // Needs mesh and fields
        push_constant_ranges: &[],
    });

    let pl_mesh_fields_state = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Mesh Fields State Pipeline Layout"),
        bind_group_layouts: &[bgl_mesh, bgl_fields, bgl_linear_state_ro],
        push_constant_ranges: &[],
    });

    let pipeline_gradient = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Gradient Pipeline"),
        layout: Some(&pl_gradient),
        module: &shader_gradient,
        entry_point: "main",
    });

    let pl_matrix = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Matrix Assembly Pipeline Layout"),
        bind_group_layouts: &[bgl_mesh, bgl_fields, bgl_solver],
        push_constant_ranges: &[],
    });

    let shader_flux = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Flux Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/flux.wgsl"))),
    });

    let pipeline_flux = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Flux Pipeline"),
        layout: Some(&pl_gradient),
        module: &shader_flux,
        entry_point: "main",
    });

    let shader_momentum = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Momentum Assembly Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
            "../shaders/momentum_assembly_v2.wgsl"
        ))),
    });

    let pipeline_momentum_assembly =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Momentum Assembly Pipeline"),
            layout: Some(&pl_matrix),
            module: &shader_momentum,
            entry_point: "main",
        });

    let shader_pressure = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Pressure Assembly Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
            "../shaders/pressure_assembly.wgsl"
        ))),
    });

    let pipeline_pressure_assembly =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Pressure Assembly Pipeline"),
            layout: Some(&pl_matrix),
            module: &shader_pressure,
            entry_point: "main",
        });

    // Combined pressure assembly + gradient shader (optimization)
    let shader_pressure_with_grad = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Pressure Assembly With Grad Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
            "../shaders/pressure_assembly_with_grad.wgsl"
        ))),
    });

    let pipeline_pressure_assembly_with_grad =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Pressure Assembly With Grad Pipeline"),
            layout: Some(&pl_matrix),
            module: &shader_pressure_with_grad,
            entry_point: "main",
        });

    let shader_flux_rhie_chow = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Flux Rhie-Chow Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
            "../shaders/flux_rhie_chow.wgsl"
        ))),
    });

    let pipeline_flux_rhie_chow =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Flux Rhie-Chow Pipeline"),
            layout: Some(&pl_gradient), // Uses mesh and fields (Group 0, 1)
            module: &shader_flux_rhie_chow,
            entry_point: "main",
        });

    let shader_velocity_correction = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Velocity Correction Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
            "../shaders/velocity_correction.wgsl"
        ))),
    });

    let pipeline_velocity_correction =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Velocity Correction Pipeline"),
            layout: Some(&pl_mesh_fields_state),
            module: &shader_velocity_correction,
            entry_point: "main",
        });

    let shader_update_u = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Update U Component Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
            "../shaders/update_u_component.wgsl"
        ))),
    });

    let pipeline_update_u_component =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Update U Component Pipeline"),
            layout: Some(&pl_mesh_fields_state),
            module: &shader_update_u,
            entry_point: "main",
        });

    PhysicsPipelines {
        pipeline_gradient,
        pipeline_flux,
        pipeline_momentum_assembly,
        pipeline_pressure_assembly,
        pipeline_pressure_assembly_with_grad,
        pipeline_flux_rhie_chow,
        pipeline_velocity_correction,
        pipeline_update_u_component,
    }
}
