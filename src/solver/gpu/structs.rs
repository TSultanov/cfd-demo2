use super::context::GpuContext;
use bytemuck::{Pod, Zeroable};
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;

#[derive(Default, Clone, Copy, Debug)]
pub struct LinearSolverStats {
    pub iterations: u32,
    pub residual: f32,
    pub converged: bool,
    pub time: std::time::Duration,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuConstants {
    pub dt: f32,
    pub time: f32,
    pub viscosity: f32,
    pub density: f32,
    pub component: u32, // 0: x, 1: y
    pub alpha_p: f32,   // Pressure relaxation
    pub scheme: u32,    // 0: Upwind, 1: SOU, 2: QUICK
    pub stride_x: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SolverParams {
    pub n: u32,
    pub num_groups: u32,
    pub padding: [u32; 2],
}

pub struct GpuSolver {
    pub context: GpuContext,

    // Mesh buffers
    pub b_face_owner: wgpu::Buffer,
    pub b_face_neighbor: wgpu::Buffer,
    pub b_face_boundary: wgpu::Buffer,
    pub b_face_areas: wgpu::Buffer,
    pub b_face_normals: wgpu::Buffer,
    pub b_face_centers: wgpu::Buffer,
    pub b_cell_centers: wgpu::Buffer,
    pub b_cell_vols: wgpu::Buffer,

    // Connectivity
    pub b_cell_face_offsets: wgpu::Buffer,
    pub b_cell_faces: wgpu::Buffer,
    pub b_cell_face_matrix_indices: wgpu::Buffer,
    pub b_diagonal_indices: wgpu::Buffer,

    // Field buffers
    pub b_u: wgpu::Buffer,
    pub b_p: wgpu::Buffer,
    pub b_d_p: wgpu::Buffer,
    pub b_fluxes: wgpu::Buffer,
    pub b_grad_p: wgpu::Buffer,
    pub b_grad_component: wgpu::Buffer,

    // Matrix Structure (CSR)
    pub b_row_offsets: wgpu::Buffer,
    pub b_col_indices: wgpu::Buffer,
    pub num_nonzeros: u32,

    // Linear Solver Buffers
    pub b_matrix_values: wgpu::Buffer,
    pub b_rhs: wgpu::Buffer,
    pub b_x: wgpu::Buffer,
    pub b_r: wgpu::Buffer,
    pub b_r0: wgpu::Buffer,
    pub b_p_solver: wgpu::Buffer,
    pub b_v: wgpu::Buffer,
    pub b_s: wgpu::Buffer,
    pub b_t: wgpu::Buffer,

    // Dot Product & Params
    pub b_dot_result: wgpu::Buffer,
    pub b_dot_result_2: wgpu::Buffer,
    pub b_scalars: wgpu::Buffer,
    pub b_staging_scalar: wgpu::Buffer,
    pub b_solver_params: wgpu::Buffer,

    // Constants
    pub b_constants: wgpu::Buffer,

    // Bind Groups & Pipelines
    pub bg_mesh: wgpu::BindGroup,
    pub bg_fields: wgpu::BindGroup,
    pub bg_solver: wgpu::BindGroup,

    // Linear Solver Bind Groups
    pub bg_linear_matrix: wgpu::BindGroup,
    pub bg_linear_state: wgpu::BindGroup,
    pub bg_linear_state_ro: wgpu::BindGroup,
    pub bg_dot_params: wgpu::BindGroup,
    pub bg_dot_r0_v: wgpu::BindGroup,
    pub bg_dot_pair_r0r_rr: wgpu::BindGroup,
    pub bg_dot_pair_tstt: wgpu::BindGroup,
    pub bg_scalars: wgpu::BindGroup,
    pub bg_empty: wgpu::BindGroup,

    pub pipeline_gradient: wgpu::ComputePipeline,
    pub pipeline_spmv_p_v: wgpu::ComputePipeline,
    pub pipeline_spmv_s_t: wgpu::ComputePipeline,
    pub pipeline_dot: wgpu::ComputePipeline,
    pub pipeline_dot_pair: wgpu::ComputePipeline,
    pub pipeline_bicgstab_update_x_r: wgpu::ComputePipeline,
    pub pipeline_bicgstab_update_p: wgpu::ComputePipeline,
    pub pipeline_bicgstab_update_s: wgpu::ComputePipeline,

    // Scalar Pipelines
    pub pipeline_init_scalars: wgpu::ComputePipeline,
    pub pipeline_reduce_rho_new_r_r: wgpu::ComputePipeline,
    pub pipeline_reduce_r0_v: wgpu::ComputePipeline,
    pub pipeline_reduce_t_s_t_t: wgpu::ComputePipeline,

    pub pipeline_flux: wgpu::ComputePipeline,
    pub pipeline_momentum_assembly: wgpu::ComputePipeline,
    pub pipeline_pressure_assembly: wgpu::ComputePipeline,
    pub pipeline_flux_rhie_chow: wgpu::ComputePipeline,
    pub pipeline_velocity_correction: wgpu::ComputePipeline,
    pub pipeline_update_u_component: wgpu::ComputePipeline,

    pub num_cells: u32,
    pub num_faces: u32,

    pub constants: GpuConstants,

    // Profiling
    pub profiling_enabled: AtomicBool,
    pub time_compute: Mutex<std::time::Duration>,
    pub time_spmv: Mutex<std::time::Duration>,
    pub time_dot: Mutex<std::time::Duration>,

    // Solver Stats
    pub stats_ux: Mutex<LinearSolverStats>,
    pub stats_uy: Mutex<LinearSolverStats>,
    pub stats_p: Mutex<LinearSolverStats>,
}
