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

    pub fn set_amg_cycle(
        &mut self,
        field: &str,
        cycle_type: crate::solver::gpu::multigrid_solver::CycleType,
    ) {
        if let Some(amg) = &mut self.amg_solver {
            match field {
                "Ux" => amg.cycle_type_ux = cycle_type,
                "Uy" => amg.cycle_type_uy = cycle_type,
                "p" => amg.cycle_type_p = cycle_type,
                _ => println!("Unknown field for AMG cycle: {}", field),
            }
            println!("Set AMG cycle for {}: {:?}", field, cycle_type);
        } else {
            println!("AMG solver is None, cannot set cycle type");
        }
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
        let num_groups_cells = self.num_cells.div_ceil(workgroup_size);
        let num_groups_faces = self.num_faces.div_ceil(workgroup_size);

        let max_groups_x = 65535;
        let dispatch_faces_x = num_groups_faces.min(max_groups_x);
        let dispatch_faces_y = num_groups_faces.div_ceil(max_groups_x);

        // Save old velocity for under-relaxation
        self.copy_u_to_u_old();

        // Save old pressure for restart
        self.copy_p_to_p_old();

        // Initialize fluxes based on current U
        self.compute_fluxes();

        // Initialize d_p by running a preliminary momentum assembly
        // This is crucial for the first timestep where d_p would otherwise be zero
        self.initialize_d_p(num_groups_cells);

        // PIMPLE Loop (outer iterations with convergence control)
        let max_outer_iters = self.n_outer_correctors;
        let outer_tol_u = 1e-3;
        let outer_tol_p = 1e-3;
        let stagnation_tolerance = 1e-2;
        let stagnation_factor = 1e-2; // relative change threshold

        let mut use_cg_for_p = false;

        'restart_loop: loop {
            let mut prev_residual_u = f64::MAX;
            let mut prev_residual_p = f64::MAX;

            for outer_iter in 0..max_outer_iters {
                println!("PIMPLE Outer Iter: {}", outer_iter + 1);
                if outer_iter > 0 {
                    // Refresh Rhie-Chow fluxes so each predictor uses latest velocity field
                    self.compute_fluxes();
                }
                // 1. Momentum Predictor

                // Save velocity and pressure before solve for convergence check
                let (u_before, p_before) = if outer_iter > 0 {
                    (
                        Some(pollster::block_on(self.get_u())),
                        Some(pollster::block_on(self.get_p())),
                    )
                } else {
                    (None, None)
                };

                // Solve Ux (Component 0) - also computes d_p and grad_p
                self.solve_momentum(0, num_groups_cells);

                // Solve Uy (Component 1)
                self.solve_momentum(1, num_groups_cells);

                // 2. Pressure Corrector (PISO inner loop)
                let need_extra_piso = prev_residual_u > outer_tol_u * 0.5
                    || prev_residual_p > outer_tol_p * 0.5;
                let num_piso_iters = if outer_iter == 0 || need_extra_piso { 2 } else { 1 };

                for piso_iter in 0..num_piso_iters {
                    // Set component to 2 for Pressure Gradient calculation
                    self.constants.component = 2;
                    self.update_constants();

                    // Update gradient if P has changed (after first iteration)
                    if piso_iter > 0 {
                        self.compute_gradient();
                    }

                    // Flux interpolation with Rhie-Chow, then combined gradient + pressure assembly
                    {
                        let mut encoder = self.context.device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("PISO Pre-Solve Encoder"),
                            },
                        );
                        {
                            let mut cpass =
                                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                    label: Some("Flux RC Pass"),
                                    timestamp_writes: None,
                                });
                            cpass.set_pipeline(&self.pipeline_flux_rhie_chow);
                            cpass.set_bind_group(0, &self.bg_mesh, &[]);
                            cpass.set_bind_group(1, &self.bg_fields, &[]);
                            cpass.dispatch_workgroups(dispatch_faces_x, dispatch_faces_y, 1);
                        }
                        {
                            // Combined gradient + pressure assembly (merged kernel)
                            let mut cpass =
                                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                    label: Some("Pressure Assembly With Grad Pass"),
                                    timestamp_writes: None,
                                });
                            cpass.set_pipeline(&self.pipeline_pressure_assembly_with_grad);
                            cpass.set_bind_group(0, &self.bg_mesh, &[]);
                            cpass.set_bind_group(1, &self.bg_fields, &[]);
                            cpass.set_bind_group(2, &self.bg_solver, &[]);
                            cpass.dispatch_workgroups(num_groups_cells, 1, 1);
                        }
                        self.context.queue.submit(Some(encoder.finish()));
                    }

                    // Solve Pressure (p_prime)
                    self.zero_buffer(&self.b_x, (self.num_cells as u64) * 4);

                    let stats = if use_cg_for_p {
                        pollster::block_on(self.solve_cg("p"))
                    } else {
                        pollster::block_on(self.solve("p"))
                    };

                    if stats.diverged {
                        if use_cg_for_p {
                            println!("CG solver also diverged. Aborting PIMPLE loop.");
                            break;
                        }
                        println!("Pressure solver diverged. Restarting PIMPLE loop with CG...");
                        use_cg_for_p = true;
                        self.restore_u_from_u_old();
                        self.restore_p_from_p_old();
                        self.compute_fluxes();
                        self.initialize_d_p(num_groups_cells);
                        continue 'restart_loop;
                    }

                    *self.stats_p.lock().unwrap() = stats;

                    // Velocity Correction
                    {
                        let mut encoder = self.context.device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("Velocity Correction Encoder"),
                            },
                        );
                        {
                            let mut cpass =
                                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
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

                // Check for outer loop convergence (after first iteration)
                if let (Some(u_before), Some(p_before)) = (u_before, p_before) {
                    let u_after = pollster::block_on(self.get_u());
                    let p_after = pollster::block_on(self.get_p());

                    // Calculate velocity residual (max change)
                    let mut max_diff_u = 0.0f64;
                    for (before, after) in u_before.iter().zip(u_after.iter()) {
                        let diff_x = (after.0 - before.0).abs();
                        let diff_y = (after.1 - before.1).abs();
                        if diff_x.is_nan() || diff_y.is_nan() {
                            max_diff_u = f64::NAN;
                            break;
                        }
                        max_diff_u = max_diff_u.max(diff_x).max(diff_y);
                    }

                    // Calculate pressure residual (max change)
                    let mut max_diff_p = 0.0f64;
                    for (before, after) in p_before.iter().zip(p_after.iter()) {
                        let diff = (after - before).abs();
                        if diff.is_nan() {
                            max_diff_p = f64::NAN;
                            break;
                        }
                        max_diff_p = max_diff_p.max(diff);
                    }

                    // Store outer loop stats
                    *self.outer_residual_u.lock().unwrap() = max_diff_u as f32;
                    *self.outer_residual_p.lock().unwrap() = max_diff_p as f32;
                    *self.outer_iterations.lock().unwrap() = outer_iter + 1;

                    println!(
                        "PIMPLE Residuals - U: {:.2e}, P: {:.2e}",
                        max_diff_u, max_diff_p
                    );

                    // Converged if both U and P are below tolerance
                    if max_diff_u < outer_tol_u && max_diff_p < outer_tol_p {
                        println!("PIMPLE Converged in {} iterations", outer_iter + 1);
                        break;
                    }

                    // Stagnation check: stop if relative residual change drops below threshold
                    let rel_u = if prev_residual_u.is_finite() && prev_residual_u.abs() > 1e-14 {
                        ((max_diff_u - prev_residual_u) / prev_residual_u).abs()
                    } else {
                        f64::INFINITY
                    };
                    let rel_p = if prev_residual_p.is_finite() && prev_residual_p.abs() > 1e-14 {
                        ((max_diff_p - prev_residual_p) / prev_residual_p).abs()
                    } else {
                        f64::INFINITY
                    };
                    let u_stagnated = rel_u < stagnation_factor;
                    let p_stagnated = rel_p < stagnation_factor;
                    if u_stagnated
                        && p_stagnated
                        && outer_iter > 2
                        && max_diff_u < stagnation_tolerance
                        && max_diff_p < stagnation_tolerance
                    {
                        println!(
                            "PIMPLE stagnated at iter {}: U={:.2e} (prev {:.2e}), P={:.2e} (prev {:.2e})",
                            outer_iter + 1, max_diff_u, prev_residual_u, max_diff_p, prev_residual_p
                        );
                        break;
                    }

                    prev_residual_u = max_diff_u;
                    prev_residual_p = max_diff_p;
                } else {
                    // First iteration - store initial values
                    *self.outer_residual_u.lock().unwrap() = f32::MAX;
                    *self.outer_residual_p.lock().unwrap() = f32::MAX;
                    *self.outer_iterations.lock().unwrap() = 1;
                }
            }
            break;
        }

        // Update time
        self.constants.time += self.constants.dt;
        self.update_constants();

        self.context.device.poll(wgpu::Maintain::Wait);
    }

    /// Copy current velocity to u_old buffer for under-relaxation
    fn copy_u_to_u_old(&self) {
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

    /// Initialize d_p values before PISO loop by running momentum assembly
    /// This ensures d_p is non-zero even at the first timestep
    fn initialize_d_p(&mut self, num_groups: u32) {
        // Run momentum assembly for component 0 just to compute d_p
        // We don't solve the system, just assemble to get the diagonal coefficients
        self.constants.component = 0;
        self.update_constants();

        {
            let mut encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Initial D_P Assembly Encoder"),
                    });
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
            self.context.queue.submit(Some(encoder.finish()));
        }
        self.context.device.poll(wgpu::Maintain::Wait);
    }

    fn solve_momentum(&mut self, component: u32, num_groups: u32) {
        self.constants.component = component;
        self.update_constants();

        // Pre-assembly: gradient (for higher order schemes) + momentum assembly
        // Combined into single encoder submission for efficiency
        {
            let mut encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Momentum Pre-Solve Encoder"),
                    });

            // Compute gradients BEFORE assembly for higher order schemes
            // The gradient of the OLD field is needed for deferred correction
            if self.constants.scheme != 0 {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Gradient U Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipeline_gradient);
                cpass.set_bind_group(0, &self.bg_mesh, &[]);
                cpass.set_bind_group(1, &self.bg_fields, &[]);
                cpass.dispatch_workgroups(num_groups, 1, 1);
            }

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

        // Use combined solve + update_u to avoid separate kernel dispatch
        let field_name = if component == 0 { "Ux" } else { "Uy" };
        let stats = pollster::block_on(self.solve_and_update_u(field_name));
        if component == 0 {
            *self.stats_ux.lock().unwrap() = stats;
        } else {
            *self.stats_uy.lock().unwrap() = stats;
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

    pub fn compute_gradient(&mut self) {
        self.constants.component = 2;
        self.update_constants();

        let workgroup_size = 64;
        let num_groups_cells = self.num_cells.div_ceil(workgroup_size);

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

    pub fn compute_velocity_gradient(&mut self, component: u32) {
        self.constants.component = component;
        self.update_constants();

        let workgroup_size = 64;
        let num_groups_cells = self.num_cells.div_ceil(workgroup_size);

        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Velocity Gradient Encoder"),
                });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Velocity Gradient Pass"),
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

    pub fn assemble_momentum(&mut self, component: u32) {
        self.constants.component = component;
        self.update_constants();

        let workgroup_size = 64;
        let num_groups_cells = self.num_cells.div_ceil(workgroup_size);

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
            cpass.dispatch_workgroups(num_groups_cells, 1, 1);
        }
        self.context.queue.submit(Some(encoder.finish()));
        self.context.device.poll(wgpu::Maintain::Wait);
    }

    pub fn compute_fluxes(&mut self) {
        let workgroup_size = 64;
        let num_groups_faces = self.num_faces.div_ceil(workgroup_size);
        let max_groups_x = 65535;
        let dispatch_faces_x = num_groups_faces.min(max_groups_x);
        let dispatch_faces_y = num_groups_faces.div_ceil(max_groups_x);

        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Flux Computation Encoder"),
                });
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
        self.context.queue.submit(Some(encoder.finish()));
        self.context.device.poll(wgpu::Maintain::Wait);
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
        self.context.device.poll(wgpu::Maintain::Wait);
    }

    fn copy_p_to_p_old(&self) {
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Copy P to P_old Encoder"),
                });
        encoder.copy_buffer_to_buffer(&self.b_p, 0, &self.b_p_old, 0, (self.num_cells as u64) * 4);
        self.context.queue.submit(Some(encoder.finish()));
    }

    fn restore_p_from_p_old(&self) {
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Restore P from P_old Encoder"),
                });
        encoder.copy_buffer_to_buffer(&self.b_p_old, 0, &self.b_p, 0, (self.num_cells as u64) * 4);
        self.context.queue.submit(Some(encoder.finish()));
    }

    fn restore_u_from_u_old(&self) {
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Restore U from U_old Encoder"),
                });
        encoder.copy_buffer_to_buffer(&self.b_u_old, 0, &self.b_u, 0, (self.num_cells as u64) * 8);
        self.context.queue.submit(Some(encoder.finish()));
    }
}
