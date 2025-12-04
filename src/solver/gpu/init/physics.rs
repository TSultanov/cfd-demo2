use std::borrow::Cow;

pub struct PhysicsPipelines {
    pub pipeline_gradient: wgpu::ComputePipeline,
    pub pipeline_gradient_coupled: wgpu::ComputePipeline,
    pub pipeline_momentum_assembly: wgpu::ComputePipeline,
    pub pipeline_pressure_assembly: wgpu::ComputePipeline,
    pub pipeline_flux_rhie_chow: wgpu::ComputePipeline,
    pub pipeline_coupled_assembly: wgpu::ComputePipeline,
    pub pipeline_update_from_coupled: wgpu::ComputePipeline,
}

pub fn init_physics_pipelines(
    device: &wgpu::Device,
    bgl_mesh: &wgpu::BindGroupLayout,
    bgl_fields: &wgpu::BindGroupLayout,
    bgl_solver: &wgpu::BindGroupLayout,
    bgl_coupled_solver: &wgpu::BindGroupLayout,
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

    let pipeline_gradient = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Gradient Pipeline"),
        layout: Some(&pl_gradient),
        module: &shader_gradient,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let shader_gradient_coupled = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Gradient Coupled Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/gradient_coupled.wgsl"))),
    });

    let pl_gradient_coupled = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Gradient Coupled Pipeline Layout"),
        bind_group_layouts: &[bgl_mesh, bgl_fields, bgl_coupled_solver],
        push_constant_ranges: &[],
    });

    let pipeline_gradient_coupled = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Gradient Coupled Pipeline"),
        layout: Some(&pl_gradient_coupled),
        module: &shader_gradient_coupled,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let pl_matrix = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Matrix Assembly Pipeline Layout"),
        bind_group_layouts: &[bgl_mesh, bgl_fields, bgl_solver],
        push_constant_ranges: &[],
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
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
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
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
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
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let pl_coupled = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Coupled Assembly Pipeline Layout"),
        bind_group_layouts: &[bgl_mesh, bgl_fields, bgl_coupled_solver],
        push_constant_ranges: &[],
    });

    let shader_coupled = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Coupled Assembly Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
            "../shaders/coupled_assembly.wgsl"
        ))),
    });

    let pipeline_coupled_assembly =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Coupled Assembly Pipeline"),
            layout: Some(&pl_coupled),
            module: &shader_coupled,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let shader_update_coupled = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Update From Coupled Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
            "../shaders/update_fields_from_coupled.wgsl"
        ))),
    });

    let bgl_coupled_solution = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Coupled Solution Layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let pl_update_coupled = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Update From Coupled Pipeline Layout"),
        bind_group_layouts: &[bgl_mesh, bgl_fields, &bgl_coupled_solution],
        push_constant_ranges: &[],
    });

    let pipeline_update_from_coupled =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Update From Coupled Pipeline"),
            layout: Some(&pl_update_coupled),
            module: &shader_update_coupled,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    PhysicsPipelines {
        pipeline_gradient,
        pipeline_gradient_coupled,
        pipeline_momentum_assembly,
        pipeline_pressure_assembly,
        pipeline_flux_rhie_chow,
        pipeline_coupled_assembly,
        pipeline_update_from_coupled,
    }
}
