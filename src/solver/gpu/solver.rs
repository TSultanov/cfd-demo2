// Force recompile 2
use std::sync::atomic::Ordering;

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

    pub fn set_dt(&mut self, dt: f32) {
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

    pub fn set_density(&mut self, rho: f32) {
        self.constants.density = rho;
        self.update_constants();
    }

    fn update_constants(&self) {
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
        let mut encoder = self
            .context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let staging_buffer = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
        self.context.queue.submit(Some(encoder.finish()));

        let slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        self.context.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result = data.to_vec();
        drop(data);
        staging_buffer.unmap();
        result
    }

    pub fn step(&mut self) {
        let workgroup_size = 64;
        let num_groups_cells = (self.num_cells + workgroup_size - 1) / workgroup_size;
        let num_groups_faces = (self.num_faces + workgroup_size - 1) / workgroup_size;

        // 1. Momentum Predictor

        // 1. Momentum Predictor

        // Gradient P is now computed inside momentum_assembly (component 0)

        // Solve Ux (Component 0)

        // Solve Ux (Component 0)
        self.solve_momentum(0, num_groups_cells);

        // Solve Uy (Component 1)
        self.solve_momentum(1, num_groups_cells);

        // 2. Pressure Corrector

        // PISO Loop
        for _ in 0..2 {
            // Gradient, flux interpolation, and pressure assembly
            {
                let mut encoder =
                    self.context
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("PISO Pre-Solve Encoder"),
                        });
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Gradient Pass"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&self.pipeline_gradient);
                    cpass.set_bind_group(0, &self.bg_mesh, &[]);
                    cpass.set_bind_group(1, &self.bg_fields, &[]);
                    cpass.dispatch_workgroups(num_groups_cells, 1, 1);
                }
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Flux RC Pass"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&self.pipeline_flux_rhie_chow);
                    cpass.set_bind_group(0, &self.bg_mesh, &[]);
                    cpass.set_bind_group(1, &self.bg_fields, &[]);
                    cpass.dispatch_workgroups(num_groups_faces, 1, 1);
                }
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Pressure Assembly Pass"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&self.pipeline_pressure_assembly);
                    cpass.set_bind_group(0, &self.bg_mesh, &[]);
                    cpass.set_bind_group(1, &self.bg_fields, &[]);
                    cpass.set_bind_group(2, &self.bg_solver, &[]);
                    cpass.dispatch_workgroups(num_groups_cells, 1, 1);
                }
                self.context.queue.submit(Some(encoder.finish()));
            }

            // Solve Pressure (p_prime)
            self.zero_buffer(&self.b_x, (self.num_cells as u64) * 4);
            let stats = pollster::block_on(self.solve());
            *self.stats_p.lock().unwrap() = stats;

            // Velocity Correction
            {
                let mut encoder =
                    self.context
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Velocity Correction Encoder"),
                        });
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Velocity Correction Pass"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&self.pipeline_velocity_correction);
                    cpass.set_bind_group(0, &self.bg_mesh, &[]);
                    cpass.set_bind_group(1, &self.bg_fields, &[]);
                    cpass.set_bind_group(2, &self.bg_linear_state_ro, &[]);
                    cpass.dispatch_workgroups(num_groups_cells, 1, 1);
                }
                self.context.queue.submit(Some(encoder.finish()));
            }
        }

        self.context.device.poll(wgpu::Maintain::Wait);
    }

    fn solve_momentum(&mut self, component: u32, num_groups: u32) {
        self.constants.component = component;
        self.update_constants();

        {
            let mut encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Momentum Assembly Encoder"),
                    });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Momentum Assembly Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipeline_momentum_assembly);
                cpass.set_bind_group(0, &self.bg_mesh, &[]);
                cpass.set_bind_group(1, &self.bg_fields, &[]);
                cpass.set_bind_group(2, &self.bg_solver, &[]);
                cpass.dispatch_workgroups(num_groups, 1, 1);
            }
            self.context.queue.submit(Some(encoder.finish()));
        }

        self.zero_buffer(&self.b_x, (self.num_cells as u64) * 4);
        let stats = pollster::block_on(self.solve());
        if component == 0 {
            *self.stats_ux.lock().unwrap() = stats;
        } else {
            *self.stats_uy.lock().unwrap() = stats;
        }

        {
            let mut encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Update U Encoder"),
                    });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Update U Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipeline_update_u_component);
                cpass.set_bind_group(0, &self.bg_mesh, &[]);
                cpass.set_bind_group(1, &self.bg_fields, &[]);
                cpass.set_bind_group(2, &self.bg_linear_state_ro, &[]);
                cpass.dispatch_workgroups(num_groups, 1, 1);
            }
            self.context.queue.submit(Some(encoder.finish()));
        }
    }

    pub fn enable_profiling(&self, enable: bool) {
        self.profiling_enabled.store(enable, Ordering::Relaxed);
    }

    pub fn get_profiling_data(
        &self,
    ) -> (
        std::time::Duration,
        std::time::Duration,
        std::time::Duration,
        std::time::Duration,
    ) {
        let compute = *self.time_compute.lock().unwrap();
        let spmv = *self.time_spmv.lock().unwrap();
        let dot = *self.time_dot.lock().unwrap();
        (dot, compute, spmv, std::time::Duration::new(0, 0))
    }

    pub fn compute_gradient(&self) {
        let workgroup_size = 64;
        let num_groups_cells = (self.num_cells + workgroup_size - 1) / workgroup_size;

        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Gradient Encoder"),
                });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Gradient Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline_gradient);
            cpass.set_bind_group(0, &self.bg_mesh, &[]);
            cpass.set_bind_group(1, &self.bg_fields, &[]);
            cpass.dispatch_workgroups(num_groups_cells, 1, 1);
        }
        self.context.queue.submit(Some(encoder.finish()));
        self.context.device.poll(wgpu::Maintain::Wait);
    }
}
