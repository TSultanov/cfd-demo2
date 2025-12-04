pub mod fields;
pub mod linear_solver;
pub mod mesh;
pub mod physics;
pub mod scalars;

use crate::solver::mesh::Mesh;
use std::sync::Arc;
use std::sync::Mutex;

use super::profiling::ProfilingStats;
use super::structs::GpuSolver;

impl GpuSolver {
    pub async fn new(mesh: &Mesh) -> Self {
        let context = super::context::GpuContext::new().await;

        let num_cells = mesh.cell_cx.len() as u32;
        let num_faces = mesh.face_owner.len() as u32;

        // 1. Initialize Mesh
        let mesh_res = mesh::init_mesh(&context.device, mesh);

        // 2. Initialize Fields
        let fields_res = fields::init_fields(&context.device, num_cells, num_faces);

        // 3. Initialize Linear Solver
        let linear_res = linear_solver::init_linear_solver(
            &context.device,
            mesh,
            num_cells,
            &mesh_res.bgl_mesh,
            &fields_res.b_u,
            &fields_res.b_p,
        );

        // 4. Initialize Scalars
        let scalar_res = scalars::init_scalars(
            &context.device,
            &linear_res.b_scalars,
            &linear_res.b_dot_result,
            &linear_res.b_dot_result_2,
            &linear_res.b_solver_params,
        );

        // 5. Initialize Physics Pipelines
        let physics_res = physics::init_physics_pipelines(
            &context.device,
            &mesh_res.bgl_mesh,
            &fields_res.bgl_fields,
            &linear_res.bgl_solver,
            &linear_res.coupled_resources.bgl_coupled_solver,
            &linear_res.coupled_resources.bgl_coupled_solution, // Pass the solution layout
        );

        // Misc
        Self {
            context,
            // Mesh
            b_face_owner: mesh_res.b_face_owner,
            b_face_neighbor: mesh_res.b_face_neighbor,
            b_face_boundary: mesh_res.b_face_boundary,
            b_face_areas: mesh_res.b_face_areas,
            b_face_normals: mesh_res.b_face_normals,
            b_face_centers: mesh_res.b_face_centers,
            b_cell_centers: mesh_res.b_cell_centers,
            b_cell_vols: mesh_res.b_cell_vols,
            b_cell_face_offsets: mesh_res.b_cell_face_offsets,
            b_cell_faces: mesh_res.b_cell_faces,
            b_cell_face_matrix_indices: mesh_res.b_cell_face_matrix_indices,
            b_diagonal_indices: mesh_res.b_diagonal_indices,
            bg_mesh: mesh_res.bg_mesh,

            // Fields
            b_u: fields_res.b_u,
            b_u_old: fields_res.b_u_old,
            b_u_old_old: fields_res.b_u_old_old,
            b_p: fields_res.b_p,
            b_d_p: fields_res.b_d_p,
            b_fluxes: fields_res.b_fluxes,
            b_grad_p: fields_res.b_grad_p,
            b_grad_component: fields_res.b_grad_component,
            b_constants: fields_res.b_constants,
            bg_fields: fields_res.bg_fields,
            constants: fields_res.constants,

            // Linear Solver
            b_row_offsets: linear_res.b_row_offsets,
            b_col_indices: linear_res.b_col_indices,
            num_nonzeros: linear_res.num_nonzeros,
            b_matrix_values: linear_res.b_matrix_values,
            b_rhs: linear_res.b_rhs,
            b_x: linear_res.b_x,
            b_r: linear_res.b_r,
            b_r0: linear_res.b_r0,
            b_p_solver: linear_res.b_p_solver,
            b_v: linear_res.b_v,
            b_s: linear_res.b_s,
            b_t: linear_res.b_t,
            b_dot_result: linear_res.b_dot_result,
            b_dot_result_2: linear_res.b_dot_result_2,
            b_scalars: linear_res.b_scalars,
            b_staging_scalar: linear_res.b_staging_scalar,
            b_solver_params: linear_res.b_solver_params,
            bg_solver: linear_res.bg_solver,
            bg_linear_matrix: linear_res.bg_linear_matrix,
            bg_linear_state: linear_res.bg_linear_state,
            bg_linear_state_ro: linear_res.bg_linear_state_ro,
            bg_dot_params: linear_res.bg_dot_params,
            bg_dot_p_v: linear_res.bg_dot_p_v,
            bg_dot_r_r: linear_res.bg_dot_r_r,
            bgl_dot_inputs: linear_res.bgl_dot_inputs,
            bgl_dot_pair_inputs: linear_res.bgl_dot_pair_inputs,
            pipeline_spmv_p_v: linear_res.pipeline_spmv_p_v,
            pipeline_dot: linear_res.pipeline_dot,
            pipeline_dot_pair: linear_res.pipeline_dot_pair,
            pipeline_cg_update_x_r: linear_res.pipeline_cg_update_x_r,
            pipeline_cg_update_p: linear_res.pipeline_cg_update_p,

            // Scalars
            bg_scalars: scalar_res.bg_scalars,
            pipeline_init_scalars: scalar_res.pipeline_init_scalars,
            pipeline_init_cg_scalars: scalar_res.pipeline_init_cg_scalars,
            pipeline_reduce_rho_new_r_r: scalar_res.pipeline_reduce_rho_new_r_r,
            pipeline_reduce_r0_v: scalar_res.pipeline_reduce_r0_v,
            pipeline_update_cg_alpha: scalar_res.pipeline_update_cg_alpha,
            pipeline_update_cg_beta: scalar_res.pipeline_update_cg_beta,
            pipeline_update_rho_old: scalar_res.pipeline_update_rho_old,

            // Physics
            pipeline_gradient_coupled: physics_res.pipeline_gradient_coupled,
            pipeline_momentum_assembly: physics_res.pipeline_momentum_assembly,
            pipeline_pressure_assembly: physics_res.pipeline_pressure_assembly,
            pipeline_flux_rhie_chow: physics_res.pipeline_flux_rhie_chow,
            pipeline_coupled_assembly: physics_res.pipeline_coupled_assembly,
            pipeline_update_from_coupled: physics_res.pipeline_update_from_coupled,

            // Misc
            num_cells,
            num_faces,
            stats_ux: Mutex::new(Default::default()),
            stats_uy: Mutex::new(Default::default()),
            stats_p: Mutex::new(Default::default()),
            outer_residual_u: Mutex::new(0.0),
            outer_residual_p: Mutex::new(0.0),
            outer_iterations: Mutex::new(0),
            fgmres_resources: None,
            n_outer_correctors: 20,
            coupled_resources: Some(linear_res.coupled_resources),
            profiling_stats: Arc::new(ProfilingStats::new()),
            staging_buffers: Mutex::new(std::collections::HashMap::new()),
        }
    }
}
