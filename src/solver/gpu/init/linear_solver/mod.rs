pub mod matrix;
pub mod pipelines;
pub mod state;

use crate::solver::mesh::Mesh;

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
    // 1. Initialize Matrix Resources (Buffers)
    // Note: row_offsets and col_indices are passed in, but we need to create buffers for them.
    // Wait, init_mesh returns row_offsets and col_indices as Vec<u32>.
    // But init_matrix expects them as arguments.
    // We need to pass them from the caller (init/mod.rs) or recalculate them.
    // The previous implementation calculated them inside init_linear_solver.
    // But init_mesh also calculates them.
    // Let's check init/mod.rs. It calls init_mesh, which returns MeshResources.
    // MeshResources contains row_offsets and col_indices.
    // So we should pass them to init_linear_solver.
    // However, the signature of init_linear_solver in init/mod.rs currently is:
    // init_linear_solver(device, mesh, num_cells, bgl_mesh)
    // It doesn't take row_offsets and col_indices.
    // Ah, in the previous refactoring (Step 78), I added row_offsets and col_indices to MeshResources.
    // But I didn't update init_linear_solver to take them.
    // Instead, init_linear_solver was recalculating them!
    // That's inefficient.
    // I should update init_linear_solver to take row_offsets and col_indices.
    // But for now, to match the existing signature (and avoid changing init/mod.rs too much yet),
    // I will recalculate them OR use the ones from MeshResources if I update init/mod.rs.
    // The plan says "Update init/mod.rs to use new structure".
    // So I can change the signature.

    // Let's recalculate them here for now to match the previous behavior of init_linear_solver,
    // OR better, since I am refactoring, I should use the ones from MeshResources.
    // But init_linear_solver signature in init/mod.rs is:
    // pub fn init_linear_solver(..., mesh: &Mesh, ...)
    // It has access to mesh.

    // Let's look at what I wrote in init/mesh.rs in Step 78.
    // It returns MeshResources which has row_offsets and col_indices.

    // So I will update init_linear_solver to take row_offsets and col_indices.

    // Recalculating for now to keep it simple and consistent with the previous file content I'm splitting.
    // The previous file `init/linear_solver.rs` calculated them.

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
    }
}
