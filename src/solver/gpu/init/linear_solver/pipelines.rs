use super::matrix::MatrixResources;
use super::state::StateResources;
use std::borrow::Cow;

pub struct PipelineResources {
    pub bg_solver: wgpu::BindGroup,
    pub bg_linear_matrix: wgpu::BindGroup,
    pub bg_linear_state: wgpu::BindGroup,
    pub bg_linear_state_ro: wgpu::BindGroup,
    pub bg_dot_params: wgpu::BindGroup,
    pub bg_dot_r0_v: wgpu::BindGroup,
    pub bg_dot_p_v: wgpu::BindGroup,
    pub bg_dot_r_r: wgpu::BindGroup,
    pub bg_dot_pair_r0r_rr: wgpu::BindGroup,
    pub bg_dot_pair_tstt: wgpu::BindGroup,
    pub bgl_solver: wgpu::BindGroupLayout,
    pub bgl_linear_matrix: wgpu::BindGroupLayout,
    pub bgl_linear_state: wgpu::BindGroupLayout,
    pub bgl_linear_state_ro: wgpu::BindGroupLayout,
    pub pipeline_spmv_p_v: wgpu::ComputePipeline,
    pub pipeline_spmv_s_t: wgpu::ComputePipeline,
    pub pipeline_dot: wgpu::ComputePipeline,
    pub pipeline_dot_pair: wgpu::ComputePipeline,
    pub pipeline_bicgstab_update_x_r: wgpu::ComputePipeline,
    pub pipeline_bicgstab_update_p: wgpu::ComputePipeline,
    pub pipeline_bicgstab_update_s: wgpu::ComputePipeline,
    pub pipeline_cg_update_x_r: wgpu::ComputePipeline,
    pub pipeline_cg_update_p: wgpu::ComputePipeline,

    pub bgl_dot_params: wgpu::BindGroupLayout,
    pub bgl_dot_inputs: wgpu::BindGroupLayout,
    pub bgl_dot_pair_inputs: wgpu::BindGroupLayout,
}

pub fn init_pipelines(
    device: &wgpu::Device,
    matrix: &MatrixResources,
    state: &StateResources,
    bgl_mesh: &wgpu::BindGroupLayout,
) -> PipelineResources {
    // Group 2: Solver (Read/Write)
    let bgl_solver = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Solver Bind Group Layout"),
        entries: &[
            // 0: Matrix Values
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
            // 1: RHS
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 2: X
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 3: R
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 4: P_Solver
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 5: V
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 6: S
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 7: T
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bg_solver = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Solver Bind Group"),
        layout: &bgl_solver,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: matrix.b_matrix_values.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: state.b_rhs.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: state.b_x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: state.b_r.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: state.b_p_solver.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: state.b_v.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: state.b_s.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: state.b_t.as_entire_binding(),
            },
        ],
    });

    // Linear Solver Layouts
    let bgl_linear_matrix = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Linear Matrix Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
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

    let bgl_linear_state = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Linear State Bind Group Layout"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bgl_linear_state_ro = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Linear State RO Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bgl_dot_params = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Dot Params Bind Group Layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let bgl_dot_inputs = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Dot Inputs Bind Group Layout"),
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
        ],
    });

    let bgl_dot_pair_inputs = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Dot Pair Inputs Bind Group Layout"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bg_dot_params = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Dot Params Bind Group"),
        layout: &bgl_dot_params,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: state.b_solver_params.as_entire_binding(),
        }],
    });

    let bg_linear_matrix = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Linear Matrix Bind Group"),
        layout: &bgl_linear_matrix,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: matrix.b_row_offsets.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: matrix.b_col_indices.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: matrix.b_matrix_values.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: state.b_scalars.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: state.b_solver_params.as_entire_binding(),
            },
        ],
    });

    let bg_linear_state = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Linear State Bind Group"),
        layout: &bgl_linear_state,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: state.b_x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: state.b_r.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: state.b_p_solver.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: state.b_v.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: state.b_s.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: state.b_t.as_entire_binding(),
            },
        ],
    });

    let bg_linear_state_ro = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Linear State RO Bind Group"),
        layout: &bgl_linear_state_ro,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: state.b_x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: state.b_r.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: state.b_p_solver.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: state.b_v.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: state.b_s.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: state.b_t.as_entire_binding(),
            },
        ],
    });

    let bg_dot_r0_v = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Dot R0 V Bind Group"),
        layout: &bgl_dot_inputs,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: state.b_dot_result.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: state.b_r0.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: state.b_v.as_entire_binding(),
            },
        ],
    });

    let bg_dot_p_v = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Dot P V Bind Group"),
        layout: &bgl_dot_inputs,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: state.b_dot_result.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: state.b_p_solver.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: state.b_v.as_entire_binding(),
            },
        ],
    });

    let bg_dot_r_r = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Dot R R Bind Group"),
        layout: &bgl_dot_inputs,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: state.b_dot_result.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: state.b_r.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: state.b_r.as_entire_binding(),
            },
        ],
    });

    let bg_dot_pair_r0r_rr = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Dot Pair R0R & RR Bind Group"),
        layout: &bgl_dot_pair_inputs,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: state.b_dot_result.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: state.b_dot_result_2.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: state.b_r0.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: state.b_r.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: state.b_r.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: state.b_r.as_entire_binding(),
            },
        ],
    });

    let bg_dot_pair_tstt = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Dot Pair TS & TT Bind Group"),
        layout: &bgl_dot_pair_inputs,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: state.b_dot_result.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: state.b_dot_result_2.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: state.b_t.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: state.b_s.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: state.b_t.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: state.b_t.as_entire_binding(),
            },
        ],
    });

    // Shaders
    let shader_linear = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Linear Solver Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
            "../../shaders/linear_solver.wgsl"
        ))),
    });

    let pl_linear = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Linear Solver Pipeline Layout"),
        bind_group_layouts: &[&bgl_mesh, &bgl_linear_state, &bgl_linear_matrix],
        push_constant_ranges: &[],
    });

    let pl_dot = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Dot Product Pipeline Layout"),
        bind_group_layouts: &[&bgl_dot_params, &bgl_dot_inputs],
        push_constant_ranges: &[],
    });

    let pl_dot_pair = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Dot Pair Pipeline Layout"),
        bind_group_layouts: &[&bgl_dot_params, &bgl_dot_pair_inputs],
        push_constant_ranges: &[],
    });

    let pipeline_spmv_p_v = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("SPMV P->V Pipeline"),
        layout: Some(&pl_linear),
        module: &shader_linear,
        entry_point: "spmv_p_v",
    });

    let pipeline_spmv_s_t = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("SPMV S->T Pipeline"),
        layout: Some(&pl_linear),
        module: &shader_linear,
        entry_point: "spmv_s_t",
    });

    let shader_dot = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Dot Product Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
            "../../shaders/dot_product.wgsl"
        ))),
    });

    let pipeline_dot = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Dot Product Pipeline"),
        layout: Some(&pl_dot),
        module: &shader_dot,
        entry_point: "main",
    });

    let shader_dot_pair = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Dot Pair Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
            "../../shaders/dot_product_pair.wgsl"
        ))),
    });

    let pipeline_dot_pair = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Dot Pair Pipeline"),
        layout: Some(&pl_dot_pair),
        module: &shader_dot_pair,
        entry_point: "main",
    });

    let pipeline_bicgstab_update_x_r =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("BiCGStab Update X and R Pipeline"),
            layout: Some(&pl_linear),
            module: &shader_linear,
            entry_point: "bicgstab_update_x_r",
        });

    let pipeline_bicgstab_update_p =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("BiCGStab Update P Pipeline"),
            layout: Some(&pl_linear),
            module: &shader_linear,
            entry_point: "bicgstab_update_p",
        });

    let pipeline_bicgstab_update_s =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("BiCGStab Update S Pipeline"),
            layout: Some(&pl_linear),
            module: &shader_linear,
            entry_point: "bicgstab_update_s",
        });

    let pipeline_cg_update_x_r = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("CG Update X R Pipeline"),
        layout: Some(&pl_linear),
        module: &shader_linear,
        entry_point: "cg_update_x_r",
    });

    let pipeline_cg_update_p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("CG Update P Pipeline"),
        layout: Some(&pl_linear),
        module: &shader_linear,
        entry_point: "cg_update_p",
    });

    PipelineResources {
        bg_solver,
        bg_linear_matrix,
        bg_linear_state,
        bg_linear_state_ro,
        bg_dot_params,
        bg_dot_r0_v,
        bg_dot_p_v,
        bg_dot_r_r,
        bg_dot_pair_r0r_rr,
        bg_dot_pair_tstt,
        bgl_solver,
        bgl_linear_matrix,
        bgl_linear_state,
        bgl_linear_state_ro,
        pipeline_spmv_p_v,
        pipeline_spmv_s_t,
        pipeline_dot,
        pipeline_dot_pair,
        pipeline_bicgstab_update_x_r,
        pipeline_bicgstab_update_p,
        pipeline_bicgstab_update_s,
        pipeline_cg_update_x_r,
        pipeline_cg_update_p,
        bgl_dot_params,
        bgl_dot_inputs,
        bgl_dot_pair_inputs,
    }
}
