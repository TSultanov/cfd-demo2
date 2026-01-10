// Force recompile 2
use std::sync::Arc;

use super::incompressible_linear_solver::FgmresResources;
use crate::solver::gpu::plans::plan_instance::{PlanFuture, PlanLinearSystemDebug};
use crate::solver::gpu::profiling::ProfilingStats;
use crate::solver::gpu::structs::{GpuSolver, LinearSolverStats};
use crate::solver::model::backend::{expand_schemes, SchemeRegistry};
use crate::solver::scheme::Scheme;

impl GpuSolver {
    pub(crate) fn update_needs_gradients(&mut self) {
        let scheme =
            Scheme::from_gpu_id(self.fields.constants.values().scheme).unwrap_or(Scheme::Upwind);
        let registry = SchemeRegistry::new(scheme);
        self.scheme_needs_gradients = expand_schemes(&self.model.system, &registry)
            .map(|expansion| expansion.needs_gradients())
            .unwrap_or(true);
    }

    pub fn set_u(&self, u: &[(f64, f64)]) {
        let layout = &self.model.state_layout;
        let stride = layout.stride() as usize;
        let u_offset = layout.offset_for("U").unwrap_or(0) as usize;
        let mut state = vec![0.0f32; self.num_cells as usize * stride];
        for (i, &(ux, uy)) in u.iter().enumerate() {
            let base = i * stride + u_offset;
            state[base] = ux as f32;
            state[base + 1] = uy as f32;
        }
        self.common.context.queue.write_buffer(
            self.fields.state.state(),
            0,
            bytemuck::cast_slice(&state),
        );
    }

    pub fn set_p(&self, p: &[f64]) {
        let layout = &self.model.state_layout;
        let stride = layout.stride() as usize;
        let p_offset = layout.offset_for("p").unwrap_or(0) as usize;
        let mut state = vec![0.0f32; self.num_cells as usize * stride];
        for (i, &pval) in p.iter().enumerate() {
            state[i * stride + p_offset] = pval as f32;
        }
        self.common.context.queue.write_buffer(
            self.fields.state.state(),
            0,
            bytemuck::cast_slice(&state),
        );
    }

    pub fn set_dt(&mut self, dt: f32) {
        self.time_integration
            .set_dt(dt, &mut self.fields.constants, &self.common.context.queue);
    }

    pub fn set_viscosity(&mut self, mu: f32) {
        {
            let values = self.fields.constants.values_mut();
            values.viscosity = mu;
        }
        self.fields.constants.write(&self.common.context.queue);
    }

    pub fn set_alpha_p(&mut self, alpha_p: f32) {
        {
            let values = self.fields.constants.values_mut();
            values.alpha_p = alpha_p;
        }
        self.fields.constants.write(&self.common.context.queue);
    }

    pub fn set_alpha_u(&mut self, alpha_u: f32) {
        {
            let values = self.fields.constants.values_mut();
            values.alpha_u = alpha_u;
        }
        self.fields.constants.write(&self.common.context.queue);
    }

    pub fn set_density(&mut self, rho: f32) {
        {
            let values = self.fields.constants.values_mut();
            values.density = rho;
        }
        self.fields.constants.write(&self.common.context.queue);
    }

    pub fn set_scheme(&mut self, scheme: u32) {
        {
            let values = self.fields.constants.values_mut();
            values.scheme = scheme;
        }
        self.fields.constants.write(&self.common.context.queue);
        self.update_needs_gradients();
    }

    pub fn set_time_scheme(&mut self, scheme: u32) {
        {
            let values = self.fields.constants.values_mut();
            values.time_scheme = scheme;
        }
        self.fields.constants.write(&self.common.context.queue);
    }

    pub fn set_inlet_velocity(&mut self, velocity: f32) {
        {
            let values = self.fields.constants.values_mut();
            values.inlet_velocity = velocity;
        }
        self.fields.constants.write(&self.common.context.queue);
    }

    pub fn set_ramp_time(&mut self, time: f32) {
        {
            let values = self.fields.constants.values_mut();
            values.ramp_time = time;
        }
        self.fields.constants.write(&self.common.context.queue);
    }

    pub fn set_precond_type(
        &mut self,
        precond_type: crate::solver::gpu::structs::PreconditionerType,
    ) {
        self.preconditioner = precond_type;
    }

    pub async fn get_u(&self) -> Vec<(f64, f64)> {
        let layout = &self.model.state_layout;
        let stride = layout.stride() as usize;
        let u_offset = layout.offset_for("U").unwrap_or(0) as usize;
        let data = self
            .read_buffer(
                self.fields.state.state(),
                (self.num_cells as u64) * stride as u64 * 4,
            )
            .await;
        let state: &[f32] = bytemuck::cast_slice(&data);
        (0..self.num_cells as usize)
            .map(|i| {
                let base = i * stride + u_offset;
                (state[base] as f64, state[base + 1] as f64)
            })
            .collect()
    }

    pub async fn get_p(&self) -> Vec<f64> {
        let layout = &self.model.state_layout;
        let stride = layout.stride() as usize;
        let p_offset = layout.offset_for("p").unwrap_or(0) as usize;
        let data = self
            .read_buffer(
                self.fields.state.state(),
                (self.num_cells as u64) * stride as u64 * 4,
            )
            .await;
        let state: &[f32] = bytemuck::cast_slice(&data);
        (0..self.num_cells as usize)
            .map(|i| state[i * stride + p_offset] as f64)
            .collect()
    }

    pub async fn get_d_p(&self) -> Vec<f64> {
        let layout = &self.model.state_layout;
        let stride = layout.stride() as usize;
        let dp_offset = layout.offset_for("d_p").unwrap_or(0) as usize;
        let data = self
            .read_buffer(
                self.fields.state.state(),
                (self.num_cells as u64) * stride as u64 * 4,
            )
            .await;
        let state: &[f32] = bytemuck::cast_slice(&data);
        (0..self.num_cells as usize)
            .map(|i| state[i * stride + dp_offset] as f64)
            .collect()
    }

    pub(crate) async fn read_buffer(&self, buffer: &wgpu::Buffer, size: u64) -> Vec<u8> {
        self.common
            .read_buffer(buffer, size, "Staging Buffer (cached)")
            .await
    }

    pub(crate) async fn read_buffer_f32(&self, buffer: &wgpu::Buffer, count: u32) -> Vec<f32> {
        let data = self.read_buffer(buffer, (count as u64) * 4).await;
        bytemuck::cast_slice(&data).to_vec()
    }

    pub(crate) async fn read_buffer_u32(&self, buffer: &wgpu::Buffer, count: u32) -> Vec<u32> {
        let data = self.read_buffer(buffer, (count as u64) * 4).await;
        bytemuck::cast_slice(&data).to_vec()
    }

    /// Get a reference to the detailed profiling statistics
    pub fn get_profiling_stats(&self) -> Arc<ProfilingStats> {
        Arc::clone(&self.common.profiling_stats)
    }

    /// Enable detailed GPU-CPU communication profiling
    pub fn enable_detailed_profiling(&self, enable: bool) {
        if enable {
            self.common.profiling_stats.enable();
        } else {
            self.common.profiling_stats.disable();
        }
    }

    /// Start a profiling session
    pub fn start_profiling_session(&self) {
        self.common.profiling_stats.start_session();
        self.record_initial_allocations();
    }

    /// End a profiling session and get the report
    pub fn end_profiling_session(&self) {
        self.common.profiling_stats.end_session();
    }

    /// Print the profiling report
    pub fn print_profiling_report(&self) {
        self.common.profiling_stats.print_report();
    }

    pub fn initialize_history(&self) {
        let mut encoder =
            self.common
                .context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Initialize History Encoder"),
                });

        let stride = self.model.state_layout.stride() as u64;
        let state_size = (self.num_cells as u64) * stride * 4;

        // Copy state to state_old
        encoder.copy_buffer_to_buffer(
            self.fields.state.state(),
            0,
            self.fields.state.state_old(),
            0,
            state_size,
        );

        // Copy state to state_old_old
        encoder.copy_buffer_to_buffer(
            self.fields.state.state(),
            0,
            self.fields.state.state_old_old(),
            0,
            state_size,
        );

        self.common.context.queue.submit(Some(encoder.finish()));
    }

    fn record_initial_allocations(&self) {
        if !self.common.profiling_stats.is_enabled() {
            return;
        }

        let record = |solver: &Self, label: &str, buf: &wgpu::Buffer| {
            solver
                .common
                .profiling_stats
                .record_gpu_alloc(label, buf.size());
        };

        // Mesh
        record(self, "mesh:face_owner", &self.common.mesh.b_face_owner);
        record(
            self,
            "mesh:face_neighbor",
            &self.common.mesh.b_face_neighbor,
        );
        record(
            self,
            "mesh:face_boundary",
            &self.common.mesh.b_face_boundary,
        );
        record(self, "mesh:face_areas", &self.common.mesh.b_face_areas);
        record(self, "mesh:face_normals", &self.common.mesh.b_face_normals);
        record(self, "mesh:face_centers", &self.common.mesh.b_face_centers);
        record(self, "mesh:cell_centers", &self.common.mesh.b_cell_centers);
        record(self, "mesh:cell_vols", &self.common.mesh.b_cell_vols);
        record(
            self,
            "mesh:cell_face_offsets",
            &self.common.mesh.b_cell_face_offsets,
        );
        record(self, "mesh:cell_faces", &self.common.mesh.b_cell_faces);
        record(
            self,
            "mesh:cell_face_matrix_indices",
            &self.common.mesh.b_cell_face_matrix_indices,
        );
        record(
            self,
            "mesh:diagonal_indices",
            &self.common.mesh.b_diagonal_indices,
        );

        // Fields (consolidated FluidState buffers)
        let state_buffers = self.fields.state.buffers();
        record(self, "fields:state", &state_buffers[0]);
        record(self, "fields:state_old", &state_buffers[1]);
        record(self, "fields:state_old_old", &state_buffers[2]);
        if let Some(fluxes) = self.fields.flux_buffer.as_ref() {
            record(self, "fields:fluxes", fluxes);
        }
        record(self, "fields:constants", self.fields.constants.buffer());

        // Matrix / linear solver
        record(
            self,
            "linear:row_offsets",
            self.linear_port_space.buffer(self.linear_ports.row_offsets),
        );
        record(
            self,
            "linear:col_indices",
            self.linear_port_space.buffer(self.linear_ports.col_indices),
        );
        record(
            self,
            "linear:matrix_values",
            self.linear_port_space.buffer(self.linear_ports.values),
        );
        record(
            self,
            "linear:rhs",
            self.linear_port_space.buffer(self.linear_ports.rhs),
        );
        record(
            self,
            "linear:x",
            self.linear_port_space.buffer(self.linear_ports.x),
        );
        record(self, "linear:r", self.scalar_cg.r());
        record(self, "linear:r0", self.scalar_cg.r0());
        record(self, "linear:p_solver", self.scalar_cg.p());
        record(self, "linear:v", self.scalar_cg.v());
        record(self, "linear:s", self.scalar_cg.s());
        record(self, "linear:t", self.scalar_cg.t());
        record(self, "linear:dot_result", self.scalar_cg.dot_result());
        record(self, "linear:dot_result_2", self.scalar_cg.dot_result_2());
        record(self, "linear:scalars", self.scalar_cg.scalars());
        record(
            self,
            "linear:staging_scalar",
            self.scalar_cg.staging_scalar(),
        );
        record(self, "linear:solver_params", self.scalar_cg.solver_params());

        if let Some(c) = &self.coupled_resources {
            for (label, buf) in [
                ("coupled:diag_inv", &c.b_diag_inv),
                ("coupled:diag_u", &c.b_diag_u),
                ("coupled:diag_v", &c.b_diag_v),
                ("coupled:diag_p", &c.b_diag_p),
                ("coupled:p_hat", &c.b_p_hat),
                ("coupled:s_hat", &c.b_s_hat),
                ("coupled:precond_rhs", &c.b_precond_rhs),
                ("coupled:precond_params", &c.b_precond_params),
                ("coupled:grad_u", &c.b_grad_u),
                ("coupled:grad_v", &c.b_grad_v),
                ("coupled:max_diff_result", &c.b_max_diff_result),
            ] {
                record(self, label, buf);
            }
        }

        if let Some(fgmres) = &self.linear_solver.fgmres_resources {
            self.record_fgmres_allocations(fgmres);
        }
    }

    pub(crate) fn record_fgmres_allocations(&self, fgmres: &FgmresResources) {
        if !self.common.profiling_stats.is_enabled() {
            return;
        }

        let record = |solver: &Self, label: &str, buf: &wgpu::Buffer| {
            solver
                .common
                .profiling_stats
                .record_gpu_alloc(label, buf.size());
        };

        record(self, "fgmres:basis", fgmres.fgmres.basis_buffer());
        for (i, buf) in fgmres.fgmres.z_vectors().iter().enumerate() {
            record(self, &format!("fgmres:z_{}", i), buf);
        }
        record(self, "fgmres:w", fgmres.fgmres.w_buffer());
        record(self, "fgmres:temp", fgmres.fgmres.temp_buffer());
        record(
            self,
            "fgmres:dot_partial",
            fgmres.fgmres.dot_partial_buffer(),
        );
        record(self, "fgmres:scalars", fgmres.fgmres.scalars_buffer());
        record(self, "fgmres:temp_p", fgmres.precond.b_temp_p());
        record(self, "fgmres:p_sol", fgmres.precond.b_p_sol());
        record(self, "fgmres:params", fgmres.fgmres.params_buffer());
        record(self, "fgmres:hessenberg", fgmres.fgmres.hessenberg_buffer());
        record(self, "fgmres:givens", fgmres.fgmres.givens_buffer());
        record(self, "fgmres:g", fgmres.fgmres.g_buffer());
        record(self, "fgmres:y", fgmres.fgmres.y_buffer());
        record(
            self,
            "fgmres:iter_params",
            fgmres.fgmres.iter_params_buffer(),
        );
        record(
            self,
            "fgmres:staging_scalar",
            fgmres.fgmres.staging_scalar_buffer(),
        );
    }
}

impl PlanLinearSystemDebug for GpuSolver {
    fn set_linear_system(&self, matrix_values: &[f32], rhs: &[f32]) -> Result<(), String> {
        GpuSolver::set_linear_system(self, matrix_values, rhs);
        Ok(())
    }

    fn solve_linear_system_with_size(
        &mut self,
        n: u32,
        max_iters: u32,
        tol: f32,
    ) -> Result<LinearSolverStats, String> {
        Ok(GpuSolver::solve_linear_system_cg_with_size(
            self, n, max_iters, tol,
        ))
    }

    fn get_linear_solution(&self) -> PlanFuture<'_, Result<Vec<f32>, String>> {
        Box::pin(async move { Ok(GpuSolver::get_linear_solution(self).await) })
    }
}
