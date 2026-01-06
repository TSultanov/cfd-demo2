// Force recompile 2
use std::sync::Arc;

use super::coupled_solver_fgmres::FgmresResources;
use super::profiling::ProfilingStats;
use super::structs::GpuSolver;

impl GpuSolver {
    pub fn set_u(&self, u: &[(f64, f64)]) {
        use super::init::fields::FluidState;
        // Read existing state, update u, write back
        // FluidState is 32 bytes per cell: u(8), p(4), d_p(4), grad_p(8), grad_component(8)
        let mut states = vec![FluidState::default(); self.num_cells as usize];
        for (i, &(ux, uy)) in u.iter().enumerate() {
            states[i].u = [ux as f32, uy as f32];
        }
        // Write full state buffer (for initialization)
        self.context
            .queue
            .write_buffer(&self.b_state, 0, bytemuck::cast_slice(&states));
    }

    pub fn set_p(&self, p: &[f64]) {
        use super::init::fields::FluidState;
        // Read existing state, update p, write back
        let mut states = vec![FluidState::default(); self.num_cells as usize];
        for (i, &pval) in p.iter().enumerate() {
            states[i].p = pval as f32;
        }
        // Write full state buffer (for initialization)
        self.context
            .queue
            .write_buffer(&self.b_state, 0, bytemuck::cast_slice(&states));
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

    pub fn set_precond_type(&mut self, precond_type: super::structs::PreconditionerType) {
        self.constants.precond_type = precond_type as u32;
        self.update_constants();
    }

    pub fn update_constants(&self) {
        self.context
            .queue
            .write_buffer(&self.b_constants, 0, bytemuck::bytes_of(&self.constants));
    }

    pub async fn get_u(&self) -> Vec<(f64, f64)> {
        use super::init::fields::FluidState;
        // FluidState is 32 bytes per cell
        let data = self
            .read_buffer(&self.b_state, (self.num_cells as u64) * 32)
            .await;
        let states: &[FluidState] = bytemuck::cast_slice(&data);
        states
            .iter()
            .map(|s| (s.u[0] as f64, s.u[1] as f64))
            .collect()
    }

    pub async fn get_p(&self) -> Vec<f64> {
        use super::init::fields::FluidState;
        // FluidState is 32 bytes per cell
        let data = self
            .read_buffer(&self.b_state, (self.num_cells as u64) * 32)
            .await;
        let states: &[FluidState] = bytemuck::cast_slice(&data);
        states.iter().map(|s| s.p as f64).collect()
    }

    pub async fn get_d_p(&self) -> Vec<f64> {
        use super::init::fields::FluidState;
        // FluidState is 32 bytes per cell
        let data = self
            .read_buffer(&self.b_state, (self.num_cells as u64) * 32)
            .await;
        let states: &[FluidState] = bytemuck::cast_slice(&data);
        states.iter().map(|s| s.d_p as f64).collect()
    }

    pub(crate) async fn read_buffer(&self, buffer: &wgpu::Buffer, size: u64) -> Vec<u8> {
        use super::profiling::ProfileCategory;
        use std::time::Instant;

        // 1. Obtain a cached staging buffer or create one if absent
        let staging_buffer = {
            if let Some(buf) = self.staging_buffers.lock().unwrap().remove(&size) {
                buf
            } else {
                let t0 = Instant::now();
                let buf = self.context.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Staging Buffer (cached)"),
                    size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.profiling_stats
                    .record_gpu_alloc("read_buffer:staging", size);
                self.profiling_stats.record_location(
                    "read_buffer:create_staging",
                    ProfileCategory::GpuResourceCreation,
                    t0.elapsed(),
                    0,
                );
                buf
            }
        };

        // 2. Encode and submit copy command
        let t1 = Instant::now();
        let mut encoder = self
            .context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
        self.context.queue.submit(Some(encoder.finish()));
        self.profiling_stats.record_location(
            "read_buffer:submit_copy",
            ProfileCategory::GpuDispatch,
            t1.elapsed(),
            0,
        );

        // 3. Request async map
        let t2 = Instant::now();
        let slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        self.profiling_stats.record_location(
            "read_buffer:map_async_request",
            ProfileCategory::Other,
            t2.elapsed(),
            0,
        );

        // 4. Poll/wait for GPU - THIS IS THE BLOCKING CALL
        let t3 = Instant::now();
        let _ = self
            .context
            .device
            .poll(wgpu::PollType::wait_indefinitely());
        self.profiling_stats.record_location(
            "read_buffer:device_poll_wait",
            ProfileCategory::GpuSync,
            t3.elapsed(),
            0,
        );

        // 5. Wait for channel (should be instant after poll)
        let t4 = Instant::now();
        rx.recv().unwrap().unwrap();
        self.profiling_stats.record_location(
            "read_buffer:channel_recv",
            ProfileCategory::Other,
            t4.elapsed(),
            0,
        );

        // 6. Read mapped data
        let t5 = Instant::now();
        let data = slice.get_mapped_range();
        let result = data.to_vec();
        drop(data);
        staging_buffer.unmap();
        self.profiling_stats
            .record_cpu_alloc("read_buffer:cpu_copy", size);
        self.profiling_stats.record_location(
            "read_buffer:memcpy",
            ProfileCategory::Other,
            t5.elapsed(),
            size,
        );
        // 7. Return staging buffer to cache for reuse
        {
            let mut cache = self.staging_buffers.lock().unwrap();
            cache.insert(size, staging_buffer);
        }

        result
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
        Arc::clone(&self.profiling_stats)
    }

    /// Enable detailed GPU-CPU communication profiling
    pub fn enable_detailed_profiling(&self, enable: bool) {
        if enable {
            self.profiling_stats.enable();
        } else {
            self.profiling_stats.disable();
        }
    }

    /// Start a profiling session
    pub fn start_profiling_session(&self) {
        self.profiling_stats.start_session();
        self.record_initial_allocations();
    }

    /// End a profiling session and get the report
    pub fn end_profiling_session(&self) {
        self.profiling_stats.end_session();
    }

    /// Print the profiling report
    pub fn print_profiling_report(&self) {
        self.profiling_stats.print_report();
    }

    pub fn initialize_history(&self) {
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Initialize History Encoder"),
                });

        // FluidState is 32 bytes per cell
        let state_size = (self.num_cells as u64) * 32;

        // Copy state to state_old
        encoder.copy_buffer_to_buffer(&self.b_state, 0, &self.b_state_old, 0, state_size);

        // Copy state to state_old_old
        encoder.copy_buffer_to_buffer(&self.b_state, 0, &self.b_state_old_old, 0, state_size);

        self.context.queue.submit(Some(encoder.finish()));
    }

    fn record_initial_allocations(&self) {
        if !self.profiling_stats.is_enabled() {
            return;
        }

        let record = |solver: &Self, label: &str, buf: &wgpu::Buffer| {
            solver.profiling_stats.record_gpu_alloc(label, buf.size());
        };

        // Mesh
        record(self, "mesh:face_owner", &self.b_face_owner);
        record(self, "mesh:face_neighbor", &self.b_face_neighbor);
        record(self, "mesh:face_boundary", &self.b_face_boundary);
        record(self, "mesh:face_areas", &self.b_face_areas);
        record(self, "mesh:face_normals", &self.b_face_normals);
        record(self, "mesh:face_centers", &self.b_face_centers);
        record(self, "mesh:cell_centers", &self.b_cell_centers);
        record(self, "mesh:cell_vols", &self.b_cell_vols);
        record(self, "mesh:cell_face_offsets", &self.b_cell_face_offsets);
        record(self, "mesh:cell_faces", &self.b_cell_faces);
        record(
            self,
            "mesh:cell_face_matrix_indices",
            &self.b_cell_face_matrix_indices,
        );
        record(self, "mesh:diagonal_indices", &self.b_diagonal_indices);

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
        record(self, "linear:row_offsets", &self.b_row_offsets);
        record(self, "linear:col_indices", &self.b_col_indices);
        record(self, "linear:matrix_values", &self.b_matrix_values);
        record(self, "linear:rhs", &self.b_rhs);
        record(self, "linear:x", &self.b_x);
        record(self, "linear:r", &self.b_r);
        record(self, "linear:r0", &self.b_r0);
        record(self, "linear:p_solver", &self.b_p_solver);
        record(self, "linear:v", &self.b_v);
        record(self, "linear:s", &self.b_s);
        record(self, "linear:t", &self.b_t);
        record(self, "linear:dot_result", &self.b_dot_result);
        record(self, "linear:dot_result_2", &self.b_dot_result_2);
        record(self, "linear:scalars", &self.b_scalars);
        record(self, "linear:staging_scalar", &self.b_staging_scalar);
        record(self, "linear:solver_params", &self.b_solver_params);

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
        if !self.profiling_stats.is_enabled() {
            return;
        }

        let record = |solver: &Self, label: &str, buf: &wgpu::Buffer| {
            solver.profiling_stats.record_gpu_alloc(label, buf.size());
        };

        record(self, "fgmres:basis", &fgmres.b_basis);
        for (i, buf) in fgmres.z_vectors.iter().enumerate() {
            record(self, &format!("fgmres:z_{}", i), buf);
        }
        record(self, "fgmres:w", &fgmres.b_w);
        record(self, "fgmres:temp", &fgmres.b_temp);
        record(self, "fgmres:dot_partial", &fgmres.b_dot_partial);
        record(self, "fgmres:scalars", &fgmres.b_scalars);
        record(self, "fgmres:temp_p", &fgmres.b_temp_p);
        record(self, "fgmres:p_sol", &fgmres.b_p_sol);
        record(self, "fgmres:params", &fgmres.b_params);
        record(self, "fgmres:precond_params", &fgmres.b_precond_params);
        record(self, "fgmres:hessenberg", &fgmres.b_hessenberg);
        record(self, "fgmres:givens", &fgmres.b_givens);
        record(self, "fgmres:g", &fgmres.b_g);
        record(self, "fgmres:y", &fgmres.b_y);
        record(self, "fgmres:iter_params", &fgmres.b_iter_params);
        record(self, "fgmres:staging_scalar", &fgmres.b_staging_scalar);
    }
}
