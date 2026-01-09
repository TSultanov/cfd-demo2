// Force recompile 2
use std::sync::Arc;

use super::coupled_fgmres::FgmresResources;
use crate::solver::gpu::model_defaults::default_incompressible_model;
use crate::solver::gpu::plans::plan_instance::{
    GpuPlanInstance, PlanAction, PlanFuture, PlanLinearSystemDebug, PlanParam,
    PlanParamValue, PlanStepStats,
};
use crate::solver::gpu::profiling::ProfilingStats;
use crate::solver::gpu::structs::{GpuSolver, LinearSolverStats};
use crate::solver::model::backend::{expand_schemes, SchemeRegistry};
use crate::solver::scheme::Scheme;

impl GpuSolver {
    pub(crate) fn update_needs_gradients(&mut self) {
        let scheme = Scheme::from_gpu_id(self.constants.scheme).unwrap_or(Scheme::Upwind);
        let registry = SchemeRegistry::new(scheme);
        self.scheme_needs_gradients = expand_schemes(&default_incompressible_model().system, &registry)
            .map(|expansion| expansion.needs_gradients())
            .unwrap_or(true);
    }

    pub fn set_u(&self, u: &[(f64, f64)]) {
        let layout = &default_incompressible_model().state_layout;
        let stride = layout.stride() as usize;
        let u_offset = layout.offset_for("U").unwrap_or(0) as usize;
        let mut state = vec![0.0f32; self.num_cells as usize * stride];
        for (i, &(ux, uy)) in u.iter().enumerate() {
            let base = i * stride + u_offset;
            state[base] = ux as f32;
            state[base + 1] = uy as f32;
        }
        self.common
            .context
            .queue
            .write_buffer(&self.b_state, 0, bytemuck::cast_slice(&state));
    }

    pub fn set_p(&self, p: &[f64]) {
        let layout = &default_incompressible_model().state_layout;
        let stride = layout.stride() as usize;
        let p_offset = layout.offset_for("p").unwrap_or(0) as usize;
        let mut state = vec![0.0f32; self.num_cells as usize * stride];
        for (i, &pval) in p.iter().enumerate() {
            state[i * stride + p_offset] = pval as f32;
        }
        self.common
            .context
            .queue
            .write_buffer(&self.b_state, 0, bytemuck::cast_slice(&state));
    }

    pub fn set_dt(&mut self, dt: f32) {
        if self.constants.time <= 0.0 {
            self.constants.dt_old = dt;
        } else {
            self.constants.dt_old = self.constants.dt;
        }
        self.constants.dt = dt;
        self.update_constants();
    }

    pub fn set_viscosity(&mut self, mu: f32) {
        self.constants.viscosity = mu;
        self.update_constants();
    }

    pub fn set_alpha_p(&mut self, alpha_p: f32) {
        self.constants.alpha_p = alpha_p;
        self.update_constants();
    }

    pub fn set_alpha_u(&mut self, alpha_u: f32) {
        self.constants.alpha_u = alpha_u;
        self.update_constants();
    }

    pub fn set_density(&mut self, rho: f32) {
        self.constants.density = rho;
        self.update_constants();
    }

    pub fn set_scheme(&mut self, scheme: u32) {
        self.constants.scheme = scheme;
        self.update_constants();
        self.update_needs_gradients();
    }

    pub fn set_time_scheme(&mut self, scheme: u32) {
        self.constants.time_scheme = scheme;
        self.update_constants();
    }

    pub fn set_inlet_velocity(&mut self, velocity: f32) {
        self.constants.inlet_velocity = velocity;
        self.update_constants();
    }

    pub fn set_ramp_time(&mut self, time: f32) {
        self.constants.ramp_time = time;
        self.update_constants();
    }

    pub fn set_precond_type(&mut self, precond_type: crate::solver::gpu::structs::PreconditionerType) {
        self.preconditioner = precond_type;
    }

    pub fn update_constants(&self) {
        self.common
            .context
            .queue
            .write_buffer(&self.b_constants, 0, bytemuck::bytes_of(&self.constants));
    }

    pub async fn get_u(&self) -> Vec<(f64, f64)> {
        let layout = &default_incompressible_model().state_layout;
        let stride = layout.stride() as usize;
        let u_offset = layout.offset_for("U").unwrap_or(0) as usize;
        let data = self
            .read_buffer(&self.b_state, (self.num_cells as u64) * stride as u64 * 4)
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
        let layout = &default_incompressible_model().state_layout;
        let stride = layout.stride() as usize;
        let p_offset = layout.offset_for("p").unwrap_or(0) as usize;
        let data = self
            .read_buffer(&self.b_state, (self.num_cells as u64) * stride as u64 * 4)
            .await;
        let state: &[f32] = bytemuck::cast_slice(&data);
        (0..self.num_cells as usize)
            .map(|i| state[i * stride + p_offset] as f64)
            .collect()
    }

    pub async fn get_d_p(&self) -> Vec<f64> {
        let layout = &default_incompressible_model().state_layout;
        let stride = layout.stride() as usize;
        let dp_offset = layout.offset_for("d_p").unwrap_or(0) as usize;
        let data = self
            .read_buffer(&self.b_state, (self.num_cells as u64) * stride as u64 * 4)
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

    /// Perform a single timestep of the coupled solver
    pub fn step(&mut self) {
        self.step_coupled();
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

        let stride = default_incompressible_model().state_layout.stride() as u64;
        let state_size = (self.num_cells as u64) * stride * 4;

        // Copy state to state_old
        encoder.copy_buffer_to_buffer(&self.b_state, 0, &self.b_state_old, 0, state_size);

        // Copy state to state_old_old
        encoder.copy_buffer_to_buffer(&self.b_state, 0, &self.b_state_old_old, 0, state_size);

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
        record(self, "mesh:face_neighbor", &self.common.mesh.b_face_neighbor);
        record(self, "mesh:face_boundary", &self.common.mesh.b_face_boundary);
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
        record(self, "mesh:diagonal_indices", &self.common.mesh.b_diagonal_indices);

        // Fields (consolidated FluidState buffers)
        record(self, "fields:state", &self.b_state);
        record(self, "fields:state_old", &self.b_state_old);
        record(self, "fields:state_old_old", &self.b_state_old_old);
        for (i, buf) in self.state_buffers.iter().enumerate() {
            record(self, &format!("fields:state_buffer_{}", i), buf);
        }
        record(self, "fields:fluxes", &self.b_fluxes);
        record(self, "fields:constants", &self.b_constants);

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
        record(self, "linear:staging_scalar", self.scalar_cg.staging_scalar());
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

        if let Some(fgmres) = &self.fgmres_resources {
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
        record(self, "fgmres:dot_partial", fgmres.fgmres.dot_partial_buffer());
        record(self, "fgmres:scalars", fgmres.fgmres.scalars_buffer());
        record(self, "fgmres:temp_p", fgmres.precond.b_temp_p());
        record(self, "fgmres:p_sol", fgmres.precond.b_p_sol());
        record(self, "fgmres:params", fgmres.fgmres.params_buffer());
        record(self, "fgmres:hessenberg", fgmres.fgmres.hessenberg_buffer());
        record(self, "fgmres:givens", fgmres.fgmres.givens_buffer());
        record(self, "fgmres:g", fgmres.fgmres.g_buffer());
        record(self, "fgmres:y", fgmres.fgmres.y_buffer());
        record(self, "fgmres:iter_params", fgmres.fgmres.iter_params_buffer());
        record(self, "fgmres:staging_scalar", fgmres.fgmres.staging_scalar_buffer());
    }
}

impl GpuPlanInstance for GpuSolver {
    fn num_cells(&self) -> u32 {
        self.num_cells
    }

    fn time(&self) -> f32 {
        self.constants.time
    }

    fn dt(&self) -> f32 {
        self.constants.dt
    }

    fn state_buffer(&self) -> &wgpu::Buffer {
        &self.b_state
    }

    fn set_param(&mut self, param: PlanParam, value: PlanParamValue) -> Result<(), String> {
        match (param, value) {
            (PlanParam::Dt, PlanParamValue::F32(dt)) => {
                self.set_dt(dt);
                Ok(())
            }
            (PlanParam::AdvectionScheme, PlanParamValue::Scheme(scheme)) => {
                self.set_scheme(scheme.gpu_id());
                Ok(())
            }
            (PlanParam::TimeScheme, PlanParamValue::TimeScheme(scheme)) => {
                self.set_time_scheme(scheme as u32);
                Ok(())
            }
            (PlanParam::Preconditioner, PlanParamValue::Preconditioner(preconditioner)) => {
                self.set_precond_type(preconditioner);
                Ok(())
            }
            (PlanParam::Viscosity, PlanParamValue::F32(mu)) => {
                self.set_viscosity(mu);
                Ok(())
            }
            (PlanParam::Density, PlanParamValue::F32(rho)) => {
                self.set_density(rho);
                Ok(())
            }
            (PlanParam::AlphaU, PlanParamValue::F32(alpha)) => {
                self.set_alpha_u(alpha);
                Ok(())
            }
            (PlanParam::AlphaP, PlanParamValue::F32(alpha)) => {
                self.set_alpha_p(alpha);
                Ok(())
            }
            (PlanParam::InletVelocity, PlanParamValue::F32(velocity)) => {
                self.set_inlet_velocity(velocity);
                Ok(())
            }
            (PlanParam::RampTime, PlanParamValue::F32(time)) => {
                self.set_ramp_time(time);
                Ok(())
            }
            (PlanParam::IncompressibleOuterCorrectors, PlanParamValue::U32(iters)) => {
                self.n_outer_correctors = iters.max(1);
                Ok(())
            }
            (PlanParam::IncompressibleShouldStop, PlanParamValue::Bool(value)) => {
                self.should_stop = value;
                Ok(())
            }
            (PlanParam::DetailedProfilingEnabled, PlanParamValue::Bool(enable)) => {
                self.enable_detailed_profiling(enable);
                Ok(())
            }
            _ => Err("parameter is not supported by this plan".into()),
        }
    }

    fn write_state_bytes(&self, bytes: &[u8]) -> Result<(), String> {
        for buffer in &self.state_buffers {
            self.common.context.queue.write_buffer(buffer, 0, bytes);
        }
        Ok(())
    }

    fn step_stats(&self) -> PlanStepStats {
        PlanStepStats {
            should_stop: Some(self.should_stop),
            degenerate_count: Some(self.degenerate_count),
            outer_iterations: Some(*self.outer_iterations.lock().unwrap()),
            outer_residual_u: Some(*self.outer_residual_u.lock().unwrap()),
            outer_residual_p: Some(*self.outer_residual_p.lock().unwrap()),
            linear_stats: Some((
                *self.stats_ux.lock().unwrap(),
                *self.stats_uy.lock().unwrap(),
                *self.stats_p.lock().unwrap(),
            )),
        }
    }

    fn perform(&self, action: PlanAction) -> Result<(), String> {
        match action {
            PlanAction::StartProfilingSession => {
                self.start_profiling_session();
                Ok(())
            }
            PlanAction::EndProfilingSession => {
                self.end_profiling_session();
                Ok(())
            }
            PlanAction::PrintProfilingReport => {
                self.print_profiling_report();
                Ok(())
            }
        }
    }

    fn profiling_stats(&self) -> Arc<ProfilingStats> {
        self.get_profiling_stats()
    }

    fn linear_system_debug(&mut self) -> Option<&mut dyn PlanLinearSystemDebug> {
        Some(self)
    }

    fn step(&mut self) {
        crate::solver::gpu::plans::coupled::plan::step_coupled(self);
    }

    fn initialize_history(&self) {
        GpuSolver::initialize_history(self);
    }

    fn read_state_bytes(&self, bytes: u64) -> PlanFuture<'_, Vec<u8>> {
        Box::pin(async move { self.read_buffer(self.state_buffer(), bytes).await })
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
        Ok(GpuSolver::solve_linear_system_cg_with_size(self, n, max_iters, tol))
    }

    fn get_linear_solution(&self) -> PlanFuture<'_, Result<Vec<f32>, String>> {
        Box::pin(async move { Ok(GpuSolver::get_linear_solution(self).await) })
    }
}
