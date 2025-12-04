// Force recompile 2
use std::sync::Arc;

use super::profiling::ProfilingStats;
use super::structs::GpuSolver;

impl GpuSolver {
    pub fn set_u(&self, u: &[(f64, f64)]) {
        let u_f32: Vec<[f32; 2]> = u.iter().map(|&(x, y)| [x as f32, y as f32]).collect();
        self.context
            .queue
            .write_buffer(&self.b_u, 0, bytemuck::cast_slice(&u_f32));
    }

    pub fn set_p(&self, p: &[f64]) {
        let p_f32: Vec<f32> = p.iter().map(|&x| x as f32).collect();
        self.context
            .queue
            .write_buffer(&self.b_p, 0, bytemuck::cast_slice(&p_f32));
    }

    pub fn set_fluxes(&self, fluxes: &[f64]) {
        let fluxes_f32: Vec<f32> = fluxes.iter().map(|&x| x as f32).collect();
        self.context
            .queue
            .write_buffer(&self.b_fluxes, 0, bytemuck::cast_slice(&fluxes_f32));
    }

    pub fn set_dt(&mut self, dt: f32) {
        if self.constants.dt > 0.0 {
            self.constants.dt_old = self.constants.dt;
        } else {
            self.constants.dt_old = dt;
        }
        self.constants.dt = dt;
        self.update_constants();
    }

    pub fn set_viscosity(&mut self, nu: f32) {
        self.constants.viscosity = nu;
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

    pub fn set_uniform_u(&self, ux: f64, uy: f64) {
        let u_data = vec![[ux as f32, uy as f32]; self.num_cells as usize];
        self.context
            .queue
            .write_buffer(&self.b_u, 0, bytemuck::cast_slice(&u_data));
    }

    pub fn update_constants(&self) {
        self.context
            .queue
            .write_buffer(&self.b_constants, 0, bytemuck::bytes_of(&self.constants));
    }

    pub async fn get_u(&self) -> Vec<(f64, f64)> {
        let data = self
            .read_buffer(&self.b_u, (self.num_cells as u64) * 8)
            .await; // 2 floats * 4 bytes = 8
        let u_f32: &[f32] = bytemuck::cast_slice(&data);
        u_f32
            .chunks(2)
            .map(|c| (c[0] as f64, c[1] as f64))
            .collect()
    }

    pub async fn get_grad_p(&self) -> Vec<(f64, f64)> {
        let data = self
            .read_buffer(&self.b_grad_p, (self.num_cells as u64) * 8)
            .await;
        let u_f32: &[f32] = bytemuck::cast_slice(&data);
        u_f32
            .chunks(2)
            .map(|c| (c[0] as f64, c[1] as f64))
            .collect()
    }

    pub async fn get_matrix_values(&self) -> Vec<f64> {
        let data = self
            .read_buffer(&self.b_matrix_values, (self.num_nonzeros as u64) * 4)
            .await;
        let vals_f32: &[f32] = bytemuck::cast_slice(&data);
        vals_f32.iter().map(|&x| x as f64).collect()
    }

    pub async fn get_x(&self) -> Vec<f32> {
        let data = self
            .read_buffer(&self.b_x, (self.num_cells as u64) * 4)
            .await;
        let vals_f32: &[f32] = bytemuck::cast_slice(&data);
        vals_f32.to_vec()
    }

    pub async fn get_p(&self) -> Vec<f64> {
        let data = self
            .read_buffer(&self.b_p, (self.num_cells as u64) * 4)
            .await;
        let p_f32: &[f32] = bytemuck::cast_slice(&data);
        p_f32.iter().map(|&x| x as f64).collect()
    }

    pub async fn get_rhs(&self) -> Vec<f64> {
        let data = self
            .read_buffer(&self.b_rhs, (self.num_cells as u64) * 4)
            .await;
        let vals_f32: &[f32] = bytemuck::cast_slice(&data);
        vals_f32.iter().map(|&x| x as f64).collect()
    }

    pub async fn get_row_offsets(&self) -> Vec<u32> {
        let data = self
            .read_buffer(&self.b_row_offsets, ((self.num_cells as u64) + 1) * 4)
            .await;
        let vals_u32: &[u32] = bytemuck::cast_slice(&data);
        vals_u32.to_vec()
    }

    pub async fn get_col_indices(&self) -> Vec<u32> {
        let data = self
            .read_buffer(&self.b_col_indices, (self.num_nonzeros as u64) * 4)
            .await;
        let vals_u32: &[u32] = bytemuck::cast_slice(&data);
        vals_u32.to_vec()
    }

    pub async fn get_cell_vols(&self) -> Vec<f32> {
        let data = self
            .read_buffer(&self.b_cell_vols, (self.num_cells as u64) * 4)
            .await;
        let vals_f32: &[f32] = bytemuck::cast_slice(&data);
        vals_f32.to_vec()
    }

    pub async fn get_diagonal_indices(&self) -> Vec<u32> {
        let data = self
            .read_buffer(&self.b_diagonal_indices, (self.num_cells as u64) * 4)
            .await;
        let vals_u32: &[u32] = bytemuck::cast_slice(&data);
        vals_u32.to_vec()
    }

    pub async fn get_d_p(&self) -> Vec<f64> {
        let data = self
            .read_buffer(&self.b_d_p, (self.num_cells as u64) * 4)
            .await;
        let vals_f32: &[f32] = bytemuck::cast_slice(&data);
        vals_f32.iter().map(|&x| x as f64).collect()
    }

    pub async fn get_fluxes(&self) -> Vec<f32> {
        let data = self
            .read_buffer(&self.b_fluxes, (self.num_faces as u64) * 4)
            .await;
        let vals_f32: &[f32] = bytemuck::cast_slice(&data);
        vals_f32.to_vec()
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

    /// Copy current velocity to u_old buffer for under-relaxation
    pub(crate) fn copy_u_to_u_old(&self) {
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Copy U to U_old Encoder"),
                });

        // Copy u_old to u_old_old first
        encoder.copy_buffer_to_buffer(
            &self.b_u_old,
            0,
            &self.b_u_old_old,
            0,
            (self.num_cells as u64) * 8,
        );

        encoder.copy_buffer_to_buffer(
            &self.b_u,
            0,
            &self.b_u_old,
            0,
            (self.num_cells as u64) * 8, // 2 floats per cell
        );
        self.context.queue.submit(Some(encoder.finish()));
    }

    /// Initialize d_p values before the pressure solve by running momentum assembly
    /// This ensures d_p is non-zero even at the first timestep
    pub fn initialize_d_p(&mut self, num_groups: u32) {
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Initial D_P Assembly Encoder"),
                });
        self.initialize_d_p_with_encoder(num_groups, &mut encoder);
        self.context.queue.submit(Some(encoder.finish()));
    }

    pub fn initialize_d_p_with_encoder(
        &mut self,
        num_groups: u32,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        // Run momentum assembly for component 0 just to compute d_p
        // We don't solve the system, just assemble to get the diagonal coefficients
        self.constants.component = 0;
        self.update_constants();

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Initial Momentum Assembly Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline_momentum_assembly);
            cpass.set_bind_group(0, &self.bg_mesh, &[]);
            cpass.set_bind_group(1, &self.bg_fields, &[]);
            cpass.set_bind_group(2, &self.bg_solver, &[]);
            cpass.dispatch_workgroups(num_groups, 1, 1);
        }
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
    }

    /// End a profiling session and get the report
    pub fn end_profiling_session(&self) {
        self.profiling_stats.end_session();
    }

    /// Print the profiling report
    pub fn print_profiling_report(&self) {
        self.profiling_stats.print_report();
    }

    pub fn compute_fluxes(&mut self) {
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Flux Computation Encoder"),
                });
        self.compute_fluxes_with_encoder(&mut encoder);
        self.context.queue.submit(Some(encoder.finish()));
    }

    pub fn compute_fluxes_with_encoder(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let workgroup_size = 64;
        let num_groups_faces = self.num_faces.div_ceil(workgroup_size);
        let max_groups_x = 65535;
        let dispatch_faces_x = num_groups_faces.min(max_groups_x);
        let dispatch_faces_y = num_groups_faces.div_ceil(max_groups_x);

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Flux RC Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline_flux_rhie_chow);
            cpass.set_bind_group(0, &self.bg_mesh, &[]);
            cpass.set_bind_group(1, &self.bg_fields, &[]);
            cpass.dispatch_workgroups(dispatch_faces_x, dispatch_faces_y, 1);
        }
    }

    pub fn compute_fluxes_and_dp_with_encoder(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let workgroup_size = 64;
        let num_groups_cells = self.num_cells.div_ceil(workgroup_size);
        
        // Ensure component is 0 for d_p calculation (if needed by shader logic)
        self.constants.component = 0;
        self.update_constants();

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Flux and DP Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline_flux_and_dp);
            cpass.set_bind_group(0, &self.bg_mesh, &[]);
            cpass.set_bind_group(1, &self.bg_fields, &[]);
            cpass.set_bind_group(2, &self.bg_solver, &[]); // Needed for layout compatibility
            cpass.dispatch_workgroups(num_groups_cells, 1, 1);
        }
    }

    pub fn initialize_history(&self) {
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Initialize History Encoder"),
                });

        // Copy u to u_old
        encoder.copy_buffer_to_buffer(&self.b_u, 0, &self.b_u_old, 0, (self.num_cells as u64) * 8);

        // Copy u to u_old_old
        encoder.copy_buffer_to_buffer(
            &self.b_u,
            0,
            &self.b_u_old_old,
            0,
            (self.num_cells as u64) * 8,
        );
        self.context.queue.submit(Some(encoder.finish()));
    }
}
