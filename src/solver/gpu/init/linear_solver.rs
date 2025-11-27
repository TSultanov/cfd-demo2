use crate::solver::gpu::structs::SolverParams;
use crate::solver::mesh::Mesh;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

pub struct LinearSolverResources {
    pub b_row_offsets: wgpu::Buffer,
    pub b_col_indices: wgpu::Buffer,
    pub num_nonzeros: u32,
    pub b_matrix_values: wgpu::Buffer,
    pub b_rhs: wgpu::Buffer,
    pub b_x: wgpu::Buffer,
    pub b_r: wgpu::Buffer,
    pub b_r0: wgpu::Buffer,
    pub b_p_solver: wgpu::Buffer,
    pub b_v: wgpu::Buffer,
    pub b_s: wgpu::Buffer,
    pub b_t: wgpu::Buffer,
    pub b_dot_result: wgpu::Buffer,
    pub b_dot_result_2: wgpu::Buffer,
    pub b_scalars: wgpu::Buffer,
    pub b_staging_scalar: wgpu::Buffer,
    pub b_solver_params: wgpu::Buffer,
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
    pub row_offsets: Vec<u32>,
    pub col_indices: Vec<u32>,
    pub num_groups: u32,
}

pub fn init_linear_solver(
    device: &wgpu::Device,
    mesh: &Mesh,
    num_cells: u32,
    bgl_mesh: &wgpu::BindGroupLayout,
) -> LinearSolverResources {
    // --- CSR Matrix Structure ---
    let mut row_offsets = vec![0u32; num_cells as usize + 1];
    let mut col_indices = Vec::new();

    let mut adj = vec![Vec::new(); num_cells as usize];
    for (i, &owner) in mesh.face_owner.iter().enumerate() {
        if let Some(neighbor) = mesh.face_neighbor[i] {
            adj[owner].push(neighbor);
            adj[neighbor].push(owner);
        }
    }

    for (i, list) in adj.iter_mut().enumerate() {
        list.push(i); // Add diagonal
        list.sort();
        list.dedup();
    }

    let mut current_offset = 0;
    for (i, list) in adj.iter().enumerate() {
        row_offsets[i] = current_offset;
        for &neighbor in list {
            col_indices.push(neighbor as u32);
        }
        current_offset += list.len() as u32;
    }
    row_offsets[num_cells as usize] = current_offset;
    let num_nonzeros = current_offset;

    let b_row_offsets = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Row Offsets Buffer"),
        contents: bytemuck::cast_slice(&row_offsets),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let b_col_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Col Indices Buffer"),
        contents: bytemuck::cast_slice(&col_indices),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    // --- Solver Buffers ---
    let b_matrix_values = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Matrix Values Buffer"),
        size: (num_nonzeros as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let b_rhs = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("RHS Buffer"),
        size: (num_cells as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let b_x = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("X Buffer"),
        size: (num_cells as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let b_r = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("R Buffer"),
        size: (num_cells as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let b_r0 = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("R0 Buffer"),
        size: (num_cells as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let b_p_solver = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("P Solver Buffer"),
        size: (num_cells as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let b_v = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("V Buffer"),
        size: (num_cells as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let b_s = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("S Buffer"),
        size: (num_cells as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let b_t = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("T Buffer"),
        size: (num_cells as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    // Dot Product & Params
    let workgroup_size = 64;
    let num_groups = num_cells.div_ceil(workgroup_size);

    let b_dot_result = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Dot Result Buffer"),
        size: (num_groups as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let b_dot_result_2 = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Dot Result Buffer 2"),
        size: (num_groups as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let b_scalars = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Scalars Buffer"),
        size: 64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let b_staging_scalar = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer Scalar"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let solver_params = SolverParams {
        n: num_cells,
        num_groups,
        padding: [0; 2],
    };
    let b_solver_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Solver Params Buffer"),
        contents: bytemuck::bytes_of(&solver_params),
        usage: wgpu::BufferUsages::UNIFORM
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

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
                resource: b_matrix_values.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_rhs.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: b_x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: b_r.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: b_p_solver.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: b_v.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: b_s.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: b_t.as_entire_binding(),
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
            resource: b_solver_params.as_entire_binding(),
        }],
    });

    let bg_linear_matrix = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Linear Matrix Bind Group"),
        layout: &bgl_linear_matrix,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: b_row_offsets.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_col_indices.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: b_matrix_values.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: b_scalars.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: b_solver_params.as_entire_binding(),
            },
        ],
    });

    let bg_linear_state = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Linear State Bind Group"),
        layout: &bgl_linear_state,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: b_x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_r.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: b_p_solver.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: b_v.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: b_s.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: b_t.as_entire_binding(),
            },
        ],
    });

    let bg_linear_state_ro = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Linear State RO Bind Group"),
        layout: &bgl_linear_state_ro,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: b_x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_r.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: b_p_solver.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: b_v.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: b_s.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: b_t.as_entire_binding(),
            },
        ],
    });

    let bg_dot_r0_v = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Dot R0 V Bind Group"),
        layout: &bgl_dot_inputs,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: b_dot_result.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_r0.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: b_v.as_entire_binding(),
            },
        ],
    });

    let bg_dot_p_v = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Dot P V Bind Group"),
        layout: &bgl_dot_inputs,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: b_dot_result.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_p_solver.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: b_v.as_entire_binding(),
            },
        ],
    });

    let bg_dot_r_r = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Dot R R Bind Group"),
        layout: &bgl_dot_inputs,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: b_dot_result.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_r.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: b_r.as_entire_binding(),
            },
        ],
    });

    let bg_dot_pair_r0r_rr = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Dot Pair R0R & RR Bind Group"),
        layout: &bgl_dot_pair_inputs,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: b_dot_result.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_dot_result_2.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: b_r0.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: b_r.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: b_r.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: b_r.as_entire_binding(),
            },
        ],
    });

    let bg_dot_pair_tstt = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Dot Pair TS & TT Bind Group"),
        layout: &bgl_dot_pair_inputs,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: b_dot_result.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_dot_result_2.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: b_t.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: b_s.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: b_t.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: b_t.as_entire_binding(),
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

    LinearSolverResources {
        b_row_offsets,
        b_col_indices,
        num_nonzeros,
        b_matrix_values,
        b_rhs,
        b_x,
        b_r,
        b_r0,
        b_p_solver,
        b_v,
        b_s,
        b_t,
        b_dot_result,
        b_dot_result_2,
        b_scalars,
        b_staging_scalar,
        b_solver_params,
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
        row_offsets,
        col_indices,
        num_groups,
    }
}
