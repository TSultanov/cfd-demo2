pub mod matrix;
pub mod pipelines;
pub mod state;

use crate::solver::gpu::init::scalars;
use crate::solver::gpu::modules::linear_system::LinearSystemPorts;
use crate::solver::gpu::modules::ports::{BufF32, BufU32, PortSpace};
use crate::solver::gpu::modules::scalar_cg::ScalarCgModule;

pub struct ScalarLinearSolverResources {
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
    pub num_groups: u32,
    pub bg_linear_matrix: wgpu::BindGroup,
    pub bg_linear_state: wgpu::BindGroup,
    pub bg_dot_params: wgpu::BindGroup,
    pub bg_dot_p_v: wgpu::BindGroup,
    pub bg_dot_r_r: wgpu::BindGroup,
    pub bgl_dot_pair_inputs: wgpu::BindGroupLayout,
    pub pipeline_spmv_p_v: wgpu::ComputePipeline,
    pub pipeline_dot: wgpu::ComputePipeline,
    pub pipeline_dot_pair: wgpu::ComputePipeline,
    pub pipeline_cg_update_x_r: wgpu::ComputePipeline,
    pub pipeline_cg_update_p: wgpu::ComputePipeline,
    pub ports: LinearSystemPorts,
    pub port_space: PortSpace,
    pipeline_res: pipelines::PipelineResources,
}

pub struct ScalarCgInit {
    pub num_nonzeros: u32,
    pub ports: LinearSystemPorts,
    pub port_space: PortSpace,
    pub scalar_cg: ScalarCgModule,
}

pub fn init_scalar_linear_solver(
    device: &wgpu::Device,
    num_cells: u32,
    scalar_row_offsets: &[u32],
    scalar_col_indices: &[u32],
) -> ScalarLinearSolverResources {
    let mut port_space = PortSpace::new();

    debug_assert_eq!(
        scalar_row_offsets.len(),
        num_cells as usize + 1,
        "unexpected CSR row_offsets length"
    );
    let matrix_res = matrix::init_matrix(device, scalar_row_offsets, scalar_col_indices);
    let state_res = state::init_state(device, num_cells);
    let ports = {
        let row_offsets_port = port_space.port::<BufU32>("linear:row_offsets");
        port_space.insert(row_offsets_port, matrix_res.b_row_offsets.clone());
        let col_indices_port = port_space.port::<BufU32>("linear:col_indices");
        port_space.insert(col_indices_port, matrix_res.b_col_indices.clone());
        let values_port = port_space.port::<BufF32>("linear:matrix_values");
        port_space.insert(values_port, matrix_res.b_matrix_values.clone());
        let rhs_port = port_space.port::<BufF32>("linear:rhs");
        port_space.insert(rhs_port, state_res.b_rhs.clone());
        let x_port = port_space.port::<BufF32>("linear:x");
        port_space.insert(x_port, state_res.b_x.clone());
        LinearSystemPorts {
            row_offsets: row_offsets_port,
            col_indices: col_indices_port,
            values: values_port,
            rhs: rhs_port,
            x: x_port,
        }
    };

    let pipeline_res = pipelines::init_pipelines(device, &matrix_res, &state_res);

    ScalarLinearSolverResources {
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
        num_groups: state_res.num_groups,
        bg_linear_matrix: pipeline_res.bg_linear_matrix.clone(),
        bg_linear_state: pipeline_res.bg_linear_state.clone(),
        bg_dot_params: pipeline_res.bg_dot_params.clone(),
        bg_dot_p_v: pipeline_res.bg_dot_p_v.clone(),
        bg_dot_r_r: pipeline_res.bg_dot_r_r.clone(),
        bgl_dot_pair_inputs: pipeline_res.bgl_dot_pair_inputs.clone(),
        pipeline_spmv_p_v: pipeline_res.pipeline_spmv_p_v.clone(),
        pipeline_dot: pipeline_res.pipeline_dot.clone(),
        pipeline_dot_pair: pipeline_res.pipeline_dot_pair.clone(),
        pipeline_cg_update_x_r: pipeline_res.pipeline_cg_update_x_r.clone(),
        pipeline_cg_update_p: pipeline_res.pipeline_cg_update_p.clone(),
        ports,
        port_space,
        pipeline_res,
    }
}

pub fn init_scalar_cg(
    device: &wgpu::Device,
    num_cells: u32,
    scalar_row_offsets: &[u32],
    scalar_col_indices: &[u32],
) -> ScalarCgInit {
    let linear_res =
        init_scalar_linear_solver(device, num_cells, scalar_row_offsets, scalar_col_indices);
    let scalar_cg = build_scalar_cg(device, num_cells, &linear_res);

    ScalarCgInit {
        num_nonzeros: linear_res.num_nonzeros,
        ports: linear_res.ports,
        port_space: linear_res.port_space,
        scalar_cg,
    }
}

fn build_scalar_cg(
    device: &wgpu::Device,
    num_cells: u32,
    linear_res: &ScalarLinearSolverResources,
) -> ScalarCgModule {
    let scalar_res = scalars::init_scalars(
        device,
        &linear_res.b_scalars,
        &linear_res.b_dot_result,
        &linear_res.b_dot_result_2,
        &linear_res.b_solver_params,
    );

    ScalarCgModule::new(
        num_cells,
        &linear_res.b_rhs,
        &linear_res.b_x,
        &linear_res.b_matrix_values,
        &linear_res.b_r,
        &linear_res.b_r0,
        &linear_res.b_p_solver,
        &linear_res.b_v,
        &linear_res.b_s,
        &linear_res.b_t,
        &linear_res.b_dot_result,
        &linear_res.b_dot_result_2,
        &linear_res.b_scalars,
        &linear_res.b_solver_params,
        &linear_res.b_staging_scalar,
        &linear_res.bg_linear_matrix,
        &linear_res.bg_linear_state,
        &linear_res.bg_dot_params,
        &linear_res.bg_dot_p_v,
        &linear_res.bg_dot_r_r,
        &scalar_res.bg_scalars,
        &linear_res.bgl_dot_pair_inputs,
        &linear_res.pipeline_spmv_p_v,
        &linear_res.pipeline_dot,
        &linear_res.pipeline_dot_pair,
        &linear_res.pipeline_cg_update_x_r,
        &linear_res.pipeline_cg_update_p,
        &scalar_res.pipeline_init_cg_scalars,
        &scalar_res.pipeline_reduce_r0_v,
        &scalar_res.pipeline_reduce_rho_new_r_r,
        device,
    )
}
