use super::async_buffer::AsyncScalarReader;
use super::context::GpuContext;
use super::coupled_solver_fgmres::FgmresResources;
use super::profiling::ProfilingStats;
use bytemuck::{Pod, Zeroable};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

#[derive(Default, Clone, Copy, Debug)]
pub struct LinearSolverStats {
    pub iterations: u32,
    pub residual: f32,
    pub converged: bool,
    pub diverged: bool,
    pub time: std::time::Duration,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PreconditionerType {
    Jacobi = 0,
    Amg = 1,
}

pub struct CoupledSolverResources {
    pub b_row_offsets: wgpu::Buffer,
    pub b_col_indices: wgpu::Buffer,
    pub b_matrix_values: wgpu::Buffer,
    pub b_rhs: wgpu::Buffer,
    pub b_x: wgpu::Buffer,
    pub b_r: wgpu::Buffer,
    pub b_r0: wgpu::Buffer,
    pub b_p_solver: wgpu::Buffer,
    pub b_v: wgpu::Buffer,
    pub b_s: wgpu::Buffer,
    pub b_t: wgpu::Buffer,
    pub b_scalars: wgpu::Buffer,
    pub b_staging_scalar: wgpu::Buffer,
    pub num_nonzeros: u32,

    // Preconditioner buffers
    pub b_diag_inv: wgpu::Buffer, // 3x3 block inverse for block-Jacobi preconditioner
    pub b_diag_u: wgpu::Buffer,   // Diagonal inverse for U (FGMRES-Schur)
    pub b_diag_v: wgpu::Buffer,   // Diagonal inverse for V (FGMRES-Schur)
    pub b_diag_p: wgpu::Buffer,   // Diagonal inverse for P (FGMRES-Schur)
    pub b_p_hat: wgpu::Buffer,    // M^{-1} * p
    pub b_s_hat: wgpu::Buffer,    // M^{-1} * s
    pub b_precond_rhs: wgpu::Buffer, // Schur RHS per DOF
    pub b_precond_params: wgpu::Buffer,

    // Gradient buffers for higher order schemes
    pub b_grad_u: wgpu::Buffer,
    pub b_grad_v: wgpu::Buffer,

    // Convergence check buffers (GPU max-diff)
    pub b_max_diff_partial_u: wgpu::Buffer, // Partial max values for U from workgroups
    pub b_max_diff_partial_p: wgpu::Buffer, // Partial max values for P from workgroups
    pub b_max_diff_result: wgpu::Buffer,    // Final max-diff result (2 floats: max_u, max_p)
    pub num_max_diff_groups: u32,           // Number of workgroups for max-diff reduction

    pub bg_reduce: wgpu::BindGroup,

    pub bg_solver: wgpu::BindGroup,
    pub bg_linear_matrix: wgpu::BindGroup,
    pub bg_linear_state: wgpu::BindGroup,
    pub bg_linear_state_ro: wgpu::BindGroup,
    pub bg_dot_p_v: wgpu::BindGroup,
    pub bg_dot_r_r: wgpu::BindGroup,
    pub bg_coupled_solution: wgpu::BindGroup,
    pub bg_scalars: wgpu::BindGroup,
    pub bg_dot_params: wgpu::BindGroup,
    pub bg_precond: wgpu::BindGroup, // Preconditioner bind group

    pub bgl_coupled_solver: wgpu::BindGroupLayout,
    pub bgl_coupled_solution: wgpu::BindGroupLayout,
    pub bgl_precond: wgpu::BindGroupLayout, // Preconditioner bind group layout

    // Max-diff convergence check resources
    pub bgl_max_diff: wgpu::BindGroupLayout,
    pub bgl_max_diff_params: wgpu::BindGroupLayout,
    pub b_max_diff_params: wgpu::Buffer,
    pub bg_max_diff_params: wgpu::BindGroup,
    pub pipeline_max_diff_reduce: wgpu::ComputePipeline,

    // Preconditioner pipelines
    pub pipeline_build_schur_rhs: wgpu::ComputePipeline,
    pub pipeline_finalize_precond: wgpu::ComputePipeline,
    pub pipeline_spmv_phat_v: wgpu::ComputePipeline,
    pub pipeline_spmv_shat_t: wgpu::ComputePipeline,
    /// Async scalar reader for non-blocking convergence checks
    pub async_scalar_reader: RefCell<AsyncScalarReader>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuConstants {
    pub dt: f32,
    pub dt_old: f32,
    pub time: f32,
    pub viscosity: f32,
    pub density: f32,
    pub component: u32, // 0: x, 1: y
    pub alpha_p: f32,   // Pressure relaxation
    pub scheme: u32,    // 0: Upwind, 1: SOU, 2: QUICK
    pub alpha_u: f32,   // Velocity under-relaxation
    pub stride_x: u32,
    pub time_scheme: u32, // 0: Euler, 1: BDF2
    pub inlet_velocity: f32,
    pub ramp_time: f32,
    pub precond_type: u32, // 0: Jacobi, 1: AMG
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SolverParams {
    pub n: u32,
    pub num_groups: u32,
    pub padding: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct PreconditionerParams {
    pub n: u32,
    pub num_cells: u32,
    pub omega: f32,
    pub precond_type: u32, // 0: Jacobi, 1: AMG
}

impl Default for PreconditionerParams {
    fn default() -> Self {
        Self {
            n: 0,
            num_cells: 0,
            omega: 1.0,
            precond_type: 0,
        }
    }
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
    pub b_u_old: wgpu::Buffer,     // For velocity under-relaxation
    pub b_u_old_old: wgpu::Buffer, // For 2nd order time stepping
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
    pub bg_scalars: wgpu::BindGroup,
    pub bg_dot_p_v: wgpu::BindGroup,
    pub bg_dot_r_r: wgpu::BindGroup, // For CG r.r
    pub bgl_dot_inputs: wgpu::BindGroupLayout,
    pub bgl_dot_pair_inputs: wgpu::BindGroupLayout,

    pub pipeline_spmv_p_v: wgpu::ComputePipeline,
    pub pipeline_dot: wgpu::ComputePipeline,
    pub pipeline_dot_pair: wgpu::ComputePipeline,
    pub pipeline_cg_update_x_r: wgpu::ComputePipeline,
    pub pipeline_cg_update_p: wgpu::ComputePipeline,
    pub pipeline_update_cg_alpha: wgpu::ComputePipeline,
    pub pipeline_update_cg_beta: wgpu::ComputePipeline,
    pub pipeline_update_rho_old: wgpu::ComputePipeline,

    // Scalar Pipelines
    pub pipeline_init_scalars: wgpu::ComputePipeline,
    pub pipeline_reduce_rho_new_r_r: wgpu::ComputePipeline,
    pub pipeline_reduce_r0_v: wgpu::ComputePipeline,

    pub pipeline_momentum_assembly: wgpu::ComputePipeline,
    pub pipeline_pressure_assembly: wgpu::ComputePipeline,
    pub pipeline_flux_rhie_chow: wgpu::ComputePipeline,
    pub pipeline_coupled_assembly_merged: wgpu::ComputePipeline,
    pub pipeline_update_from_coupled: wgpu::ComputePipeline,
    pub pipeline_prepare_coupled: wgpu::ComputePipeline,
    pub pipeline_init_cg_scalars: wgpu::ComputePipeline,

    pub num_cells: u32,
    pub num_faces: u32,

    pub constants: GpuConstants,

    // Solver Stats
    pub stats_ux: Mutex<LinearSolverStats>,
    pub stats_uy: Mutex<LinearSolverStats>,
    pub stats_p: Mutex<LinearSolverStats>,
    pub outer_residual_u: Mutex<f32>,
    pub outer_residual_p: Mutex<f32>,
    pub outer_iterations: Mutex<u32>,

    pub fgmres_resources: Option<FgmresResources>,

    pub n_outer_correctors: u32,

    pub coupled_resources: Option<CoupledSolverResources>,

    /// Detailed profiling statistics for GPU-CPU communication analysis
    pub profiling_stats: Arc<ProfilingStats>,

    // Cache of persistent staging buffers for GPU->CPU reads, keyed by exact size in bytes
    pub staging_buffers: Mutex<HashMap<u64, wgpu::Buffer>>,
}
