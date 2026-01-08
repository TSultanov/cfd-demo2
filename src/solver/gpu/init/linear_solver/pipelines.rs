use super::matrix::MatrixResources;
use super::state::StateResources;
use crate::solver::gpu::bindings::dot_product;
use crate::solver::gpu::bindings::dot_product_pair;
use crate::solver::gpu::bindings::linear_solver;

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
) -> PipelineResources {
    // Use generated layouts
    // Group 0: Linear State (Vectors)
    let bgl_linear_state =
        device.create_bind_group_layout(&linear_solver::WgpuBindGroup0::LAYOUT_DESCRIPTOR);
    // Group 1: Linear Matrix (Matrix & Params)
    let bgl_linear_matrix =
        device.create_bind_group_layout(&linear_solver::WgpuBindGroup1::LAYOUT_DESCRIPTOR);

    // Dot Product Layouts
    let bgl_dot_params =
        device.create_bind_group_layout(&dot_product::WgpuBindGroup0::LAYOUT_DESCRIPTOR);
    let bgl_dot_inputs =
        device.create_bind_group_layout(&dot_product::WgpuBindGroup1::LAYOUT_DESCRIPTOR);
    let bgl_dot_pair_inputs =
        device.create_bind_group_layout(&dot_product_pair::WgpuBindGroup1::LAYOUT_DESCRIPTOR);

    // Create Bind Groups
    let entries_1 = matrix.as_bind_group_1_entries(
        state.b_scalars.as_entire_buffer_binding(),
        state.b_solver_params.as_entire_buffer_binding(),
    );
    let entries_struct_1 = linear_solver::WgpuBindGroup1Entries::new(entries_1);
    let entries_array_1 = entries_struct_1.into_array();
    let bg_linear_matrix = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Linear Matrix Bind Group"),
        layout: &bgl_linear_matrix,
        entries: &entries_array_1,
    });

    let entries_0 = state.as_bind_group_0_entries();
    let entries_struct_0 = linear_solver::WgpuBindGroup0Entries::new(entries_0);
    let entries_array_0 = entries_struct_0.into_array();
    let bg_linear_state = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Linear State Bind Group"),
        layout: &bgl_linear_state,
        entries: &entries_array_0,
    });

    // Manual creation of RO layout and bind group
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

    let bg_dot_params = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Dot Params Bind Group"),
        layout: &bgl_dot_params,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: state.b_solver_params.as_entire_binding(),
        }],
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

    // Pipelines
    let pipeline_spmv_p_v = linear_solver::compute::create_spmv_p_v_pipeline_embed_source(device);
    let pipeline_spmv_s_t = linear_solver::compute::create_spmv_s_t_pipeline_embed_source(device);

    let pipeline_dot = dot_product::compute::create_main_pipeline_embed_source(device);
    let pipeline_dot_pair = dot_product_pair::compute::create_main_pipeline_embed_source(device);

    let pipeline_bicgstab_update_x_r =
        linear_solver::compute::create_bicgstab_update_x_r_pipeline_embed_source(device);
    let pipeline_bicgstab_update_p =
        linear_solver::compute::create_bicgstab_update_p_pipeline_embed_source(device);
    let pipeline_bicgstab_update_s =
        linear_solver::compute::create_bicgstab_update_s_pipeline_embed_source(device);
    let pipeline_cg_update_x_r =
        linear_solver::compute::create_cg_update_x_r_pipeline_embed_source(device);
    let pipeline_cg_update_p =
        linear_solver::compute::create_cg_update_p_pipeline_embed_source(device);

    PipelineResources {
        bg_solver: bg_linear_state.clone(),
        bg_linear_matrix,
        bg_linear_state: bg_linear_state.clone(),
        bg_linear_state_ro,
        bg_dot_params,
        bg_dot_r0_v,
        bg_dot_p_v,
        bg_dot_r_r,
        bg_dot_pair_r0r_rr,
        bg_dot_pair_tstt,
        bgl_solver: bgl_linear_state.clone(),
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
