pub mod matrix;
pub mod pipelines;
pub mod state;

use crate::solver::gpu::async_buffer::AsyncScalarReader;
use crate::solver::gpu::structs::{CoupledSolverResources, PreconditionerParams};
use crate::solver::mesh::Mesh;
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
    pub bgl_dot_inputs: wgpu::BindGroupLayout,
    pub bgl_dot_pair_inputs: wgpu::BindGroupLayout,
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
    pub coupled_resources: CoupledSolverResources,
}

pub fn init_linear_solver(
    device: &wgpu::Device,
    mesh: &Mesh,
    num_cells: u32,
    bgl_mesh: &wgpu::BindGroupLayout,
) -> LinearSolverResources {
    // 1. Initialize Matrix Resources (Buffers)
    // We recalculate the CSR structure here to ensure it matches the mesh connectivity.
    // Ideally, this should be passed from the mesh initialization, but for now we
    // recompute it to keep the initialization logic self-contained.

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

    let matrix_res = matrix::init_matrix(device, &row_offsets, &col_indices);

    // 2. Initialize State Resources
    let state_res = state::init_state(device, num_cells);

    // 3. Initialize Pipelines
    let pipeline_res = pipelines::init_pipelines(device, &matrix_res, &state_res, bgl_mesh);

    let coupled_resources = init_coupled_resources(
        device,
        mesh,
        num_cells,
        &pipeline_res,
        &matrix_res.b_row_offsets,
        &matrix_res.b_matrix_values,
    );

    LinearSolverResources {
        b_row_offsets: matrix_res.b_row_offsets,
        b_col_indices: matrix_res.b_col_indices,
        num_nonzeros: matrix_res.num_nonzeros,
        b_matrix_values: matrix_res.b_matrix_values,

        b_rhs: state_res.b_rhs,
        b_x: state_res.b_x,
        b_r: state_res.b_r,
        b_r0: state_res.b_r0,
        b_p_solver: state_res.b_p_solver,
        b_v: state_res.b_v,
        b_s: state_res.b_s,
        b_t: state_res.b_t,
        b_dot_result: state_res.b_dot_result,
        b_dot_result_2: state_res.b_dot_result_2,
        b_scalars: state_res.b_scalars,
        b_staging_scalar: state_res.b_staging_scalar,
        b_solver_params: state_res.b_solver_params,

        bg_solver: pipeline_res.bg_solver,
        bg_linear_matrix: pipeline_res.bg_linear_matrix,
        bg_linear_state: pipeline_res.bg_linear_state,
        bg_linear_state_ro: pipeline_res.bg_linear_state_ro,
        bg_dot_params: pipeline_res.bg_dot_params,
        bg_dot_r0_v: pipeline_res.bg_dot_r0_v,
        bg_dot_p_v: pipeline_res.bg_dot_p_v,
        bg_dot_r_r: pipeline_res.bg_dot_r_r,
        bg_dot_pair_r0r_rr: pipeline_res.bg_dot_pair_r0r_rr,
        bg_dot_pair_tstt: pipeline_res.bg_dot_pair_tstt,

        bgl_solver: pipeline_res.bgl_solver,
        bgl_linear_matrix: pipeline_res.bgl_linear_matrix,
        bgl_linear_state: pipeline_res.bgl_linear_state,
        bgl_linear_state_ro: pipeline_res.bgl_linear_state_ro,
        bgl_dot_inputs: pipeline_res.bgl_dot_inputs,
        bgl_dot_pair_inputs: pipeline_res.bgl_dot_pair_inputs,

        pipeline_spmv_p_v: pipeline_res.pipeline_spmv_p_v,
        pipeline_spmv_s_t: pipeline_res.pipeline_spmv_s_t,
        pipeline_dot: pipeline_res.pipeline_dot,
        pipeline_dot_pair: pipeline_res.pipeline_dot_pair,
        pipeline_bicgstab_update_x_r: pipeline_res.pipeline_bicgstab_update_x_r,
        pipeline_bicgstab_update_p: pipeline_res.pipeline_bicgstab_update_p,
        pipeline_bicgstab_update_s: pipeline_res.pipeline_bicgstab_update_s,
        pipeline_cg_update_x_r: pipeline_res.pipeline_cg_update_x_r,
        pipeline_cg_update_p: pipeline_res.pipeline_cg_update_p,

        row_offsets,
        col_indices,
        num_groups: state_res.num_groups,
        coupled_resources,
    }
}

fn init_coupled_resources(
    device: &wgpu::Device,
    mesh: &Mesh,
    num_cells: u32,
    pipeline_res: &pipelines::PipelineResources,
    scalar_row_offsets_buffer: &wgpu::Buffer,
    b_scalar_matrix_values: &wgpu::Buffer,
) -> CoupledSolverResources {
    // 1. Compute Coupled CSR Structure
    let num_coupled_cells = num_cells * 3;
    let mut row_offsets = vec![0u32; num_coupled_cells as usize + 1];
    let mut col_indices = Vec::new();

    // Build adjacency list first (same as scalar)
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

    // Now expand to coupled system
    let mut current_offset = 0;
    for i in 0..num_cells as usize {
        let neighbors = &adj[i];
        // For each of the 3 rows for cell i
        for _row_sub in 0..3 {
            row_offsets[3 * i + _row_sub] = current_offset;
            // For each neighbor cell j (including i itself)
            for &j in neighbors {
                // Add 3 columns: 3*j, 3*j+1, 3*j+2
                col_indices.push((3 * j) as u32);
                col_indices.push((3 * j + 1) as u32);
                col_indices.push((3 * j + 2) as u32);
            }
            current_offset += (neighbors.len() * 3) as u32;
        }
    }
    row_offsets[num_coupled_cells as usize] = current_offset;

    // 2. Init Matrix Buffers
    let matrix_res = matrix::init_matrix(device, &row_offsets, &col_indices);

    // Init Gradient Buffers for Coupled Solver (for higher order schemes)
    let b_grad_u = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Coupled Grad U"),
        size: (num_cells as u64) * 8, // Vector2<f32>
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let b_grad_v = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Coupled Grad V"),
        size: (num_cells as u64) * 8, // Vector2<f32>
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Init Max-Diff Convergence Check Buffers
    let workgroup_size = 64u32;
    let num_max_diff_groups = num_cells.div_ceil(workgroup_size);

    let b_max_diff_result = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Max Diff Result"),
        size: 8, // 2 floats: max_u, max_p
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // 3. Init State Buffers (size * 3)
    let state_res = state::init_state(device, num_coupled_cells);

    // Create preconditioner buffers (Moved up for bg_solver)
    let b_diag_inv = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Coupled Block Inverse"),
        size: (num_cells as u64) * 9 * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let b_diag_u = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Coupled Diag U"),
        size: (num_cells as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let b_diag_v = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Coupled Diag V"),
        size: (num_cells as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let b_diag_p = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Coupled Diag P"),
        size: (num_cells as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let b_p_hat = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Coupled P Hat"),
        size: (num_coupled_cells as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let b_s_hat = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Coupled S Hat"),
        size: (num_coupled_cells as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let b_precond_rhs = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Coupled Precond RHS"),
        size: (num_coupled_cells as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let precond_params = PreconditionerParams::default();
    let b_precond_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Preconditioner Params"),
        contents: bytemuck::bytes_of(&precond_params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // 4. Create Bind Groups
    // Reuse layouts from pipeline_res

    // Create custom layout for coupled solver assembly
    let bgl_coupled_solver = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Coupled Solver Layout"),
        entries: &[
            // 0: Matrix Values (RW)
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
            // 1: RHS (RW)
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
            // 2: Scalar Row Offsets (Read)
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
            // 3: Grad U (Read/Write)
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
            // 4: Grad V (Read/Write)
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
            // 5: Scalar Matrix Values (Read/Write)
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
            // 6: Diag U (Write)
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
            // 7: Diag V (Write)
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
            // 8: Diag P (Write)
            wgpu::BindGroupLayoutEntry {
                binding: 8,
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
        label: Some("Coupled Solver Bind Group"),
        layout: &bgl_coupled_solver,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: matrix_res.b_matrix_values.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: state_res.b_rhs.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: scalar_row_offsets_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: b_grad_u.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: b_grad_v.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: b_scalar_matrix_values.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: b_diag_u.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: b_diag_v.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: b_diag_p.as_entire_binding(),
            },
        ],
    });

    let bg_linear_matrix = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Coupled Linear Matrix Bind Group"),
        layout: &pipeline_res.bgl_linear_matrix,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: matrix_res.b_row_offsets.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: matrix_res.b_col_indices.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: matrix_res.b_matrix_values.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: state_res.b_scalars.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: state_res.b_solver_params.as_entire_binding(),
            },
        ],
    });

    let bg_linear_state = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Coupled Linear State Bind Group"),
        layout: &pipeline_res.bgl_linear_state,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: state_res.b_x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: state_res.b_r.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: state_res.b_p_solver.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: state_res.b_v.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: state_res.b_s.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: state_res.b_t.as_entire_binding(),
            },
        ],
    });

    let bg_linear_state_ro = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Coupled Linear State RO Bind Group"),
        layout: &pipeline_res.bgl_linear_state_ro,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: state_res.b_x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: state_res.b_r.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: state_res.b_p_solver.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: state_res.b_v.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: state_res.b_s.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: state_res.b_t.as_entire_binding(),
            },
        ],
    });

    let bg_dot_p_v = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Coupled Dot P V Bind Group"),
        layout: &pipeline_res.bgl_dot_inputs,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: state_res.b_dot_result.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: state_res.b_p_solver.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: state_res.b_v.as_entire_binding(),
            },
        ],
    });

    let bg_dot_r_r = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Coupled Dot R R Bind Group"),
        layout: &pipeline_res.bgl_dot_inputs,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: state_res.b_dot_result.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: state_res.b_r.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: state_res.b_r.as_entire_binding(),
            },
        ],
    });

    // Coupled solution bind group
    let bgl_coupled_solution = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Coupled Solution Layout"),
        entries: &[
            // 0: Solution X (Read Only)
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
            // 1: Max Diff Result (Atomic)
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
        ],
    });

    let bg_coupled_solution = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Coupled Solution Bind Group"),
        layout: &bgl_coupled_solution,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: state_res.b_x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_max_diff_result.as_entire_binding(),
            },
        ],
    });

    // Scalars Bind Group (Recreated layout to match init_scalars.rs)
    let bgl_scalars = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Coupled Scalars Bind Group Layout"),
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
        label: Some("Coupled Scalars Bind Group"),
        layout: &bgl_scalars,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: state_res.b_scalars.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: state_res.b_dot_result.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: state_res.b_dot_result_2.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: state_res.b_solver_params.as_entire_binding(),
            },
        ],
    });

    let bg_dot_params = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Coupled Dot Params Bind Group"),
        layout: &pipeline_res.bgl_dot_params,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: state_res.b_solver_params.as_entire_binding(),
        }],
    });

    // ========== Preconditioner Setup ==========

    // Preconditioner buffers moved up

    // Preconditioner bind group layout
    let bgl_precond = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Preconditioner Bind Group Layout"),
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
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bg_precond = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Preconditioner Bind Group"),
        layout: &bgl_precond,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: b_diag_inv.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_p_hat.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: b_s_hat.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: b_precond_rhs.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: b_precond_params.as_entire_binding(),
            },
        ],
    });

    // Create preconditioner shader and pipelines
    let shader_precond = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Preconditioner Shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "../../shaders/preconditioner.wgsl"
        ))),
    });

    let pl_precond = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Preconditioner Pipeline Layout"),
        bind_group_layouts: &[
            &pipeline_res.bgl_linear_state,
            &pipeline_res.bgl_linear_matrix,
            &bgl_precond,
        ],
        push_constant_ranges: &[],
    });

    let pipeline_build_schur_rhs =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Build Schur RHS Pipeline"),
            layout: Some(&pl_precond),
            module: &shader_precond,
            entry_point: Some("build_schur_rhs"),
            compilation_options: Default::default(),
            cache: None,
        });

    let pipeline_finalize_precond =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Finalize Preconditioner Pipeline"),
            layout: Some(&pl_precond),
            module: &shader_precond,
            entry_point: Some("finalize_precond"),
            compilation_options: Default::default(),
            cache: None,
        });

    let pipeline_spmv_phat_v = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("SpMV Phat V Pipeline"),
        layout: Some(&pl_precond),
        module: &shader_precond,
        entry_point: Some("spmv_phat_v"),
        compilation_options: Default::default(),
        cache: None,
    });

    let pipeline_spmv_shat_t = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("SpMV Shat T Pipeline"),
        layout: Some(&pl_precond),
        module: &shader_precond,
        entry_point: Some("spmv_shat_t"),
        compilation_options: Default::default(),
        cache: None,
    });

    CoupledSolverResources {
        b_row_offsets: matrix_res.b_row_offsets,
        b_col_indices: matrix_res.b_col_indices,
        b_matrix_values: matrix_res.b_matrix_values,
        b_rhs: state_res.b_rhs,
        b_x: state_res.b_x,
        b_r: state_res.b_r,
        b_r0: state_res.b_r0,
        b_p_solver: state_res.b_p_solver,
        b_v: state_res.b_v,
        b_s: state_res.b_s,
        b_t: state_res.b_t,
        b_scalars: state_res.b_scalars,
        b_staging_scalar: state_res.b_staging_scalar,
        num_nonzeros: matrix_res.num_nonzeros,
        b_diag_inv,
        b_diag_u,
        b_diag_v,
        b_diag_p,
        b_p_hat,
        b_s_hat,
        b_precond_rhs,
        b_precond_params,
        b_grad_u,
        b_grad_v,
        // Max-diff convergence check
        b_max_diff_result,
        num_max_diff_groups,
        bg_solver,
        bg_linear_matrix,
        bg_linear_state,
        bg_linear_state_ro,
        bg_dot_p_v,
        bg_dot_r_r,
        bg_coupled_solution,
        bg_scalars,
        bg_dot_params,
        bg_precond,
        bgl_coupled_solver,
        bgl_coupled_solution,
        bgl_precond,
        // Preconditioner pipelines
        pipeline_build_schur_rhs,
        pipeline_finalize_precond,
        pipeline_spmv_phat_v,
        pipeline_spmv_shat_t,
        async_scalar_reader: std::cell::RefCell::new(AsyncScalarReader::new(device, 8)),
    }
}
