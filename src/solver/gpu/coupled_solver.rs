// Coupled Solver for CFD
//
// This solver solves the momentum and continuity equations simultaneously
// in a block-coupled manner, as opposed to the segregated PISO approach.
//
// The coupled approach forms a larger block system:
// | A_u  G  | | u |   | b_u |
// | D    0  | | p | = | 0   |
//
// Where:
// - A_u is the momentum matrix (discretized convection + diffusion + time derivative)
// - G is the gradient operator (pressure gradient contribution to momentum)
// - D is the divergence operator (mass flux in continuity)
// - u is the velocity field
// - p is the pressure field
// - b_u is the momentum source term

use super::structs::{GpuSolver, LinearSolverStats};

impl GpuSolver {
    /// Performs a single timestep using the coupled solver approach.
    ///
    /// Unlike PISO which iterates between momentum prediction and pressure correction,
    /// the coupled solver assembles and solves the full block system in one go,
    /// with outer iterations for non-linearity.
    pub fn step_coupled(&mut self) {
        let workgroup_size = 64;
        let num_groups_cells = self.num_cells.div_ceil(workgroup_size);

        // Save old velocity for under-relaxation and time derivative
        self.copy_u_to_u_old();
        self.copy_p_to_p_old();

        // Initialize fluxes based on current U
        self.compute_fluxes();

        // Initialize d_p by running a preliminary momentum assembly
        // This is crucial for Rhie-Chow interpolation in the coupled solver
        self.initialize_d_p(num_groups_cells);

        // Coupled iteration loop - use more iterations for robustness
        let max_coupled_iters = self.n_outer_correctors.max(10);
        let convergence_tol_u = 1e-5;
        let convergence_tol_p = 1e-4;

        let mut prev_residual_u = f64::MAX;
        let mut prev_residual_p = f64::MAX;

        // We need to access coupled resources. If not available, return.
        if self.coupled_resources.is_none() {
            println!("Coupled resources not initialized!");
            return;
        }

        for iter in 0..max_coupled_iters {
            println!("Coupled Iteration: {}", iter + 1);

            // Save state for convergence check
            let (u_before, p_before) = if iter > 0 {
                (
                    Some(pollster::block_on(self.get_u())),
                    Some(pollster::block_on(self.get_p())),
                )
            } else {
                (None, None)
            };

            // Update fluxes with current velocity for advection terms
            if iter > 0 {
                self.compute_fluxes();
                // Re-compute d_p based on current velocity/fluxes
                self.initialize_d_p(num_groups_cells);
            }

            // Debug: Check d_p values on first iteration
            if iter == 0 {
                let d_p_vals = pollster::block_on(self.get_d_p());
                let min_dp = d_p_vals.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_dp = d_p_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let avg_dp: f64 = d_p_vals.iter().sum::<f64>() / d_p_vals.len() as f64;
                println!(
                    "d_p stats: min={:.2e}, max={:.2e}, avg={:.2e}",
                    min_dp, max_dp, avg_dp
                );
            }

            // 1. Assemble Coupled System
            // We need to zero the matrix and RHS first
            {
                let res = self.coupled_resources.as_ref().unwrap();
                self.zero_buffer(&res.b_matrix_values, (res.num_nonzeros as u64) * 4);
                self.zero_buffer(&res.b_rhs, (self.num_cells as u64) * 3 * 4);

                // Dispatch Assembly
                let mut encoder =
                    self.context
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Coupled Assembly Encoder"),
                        });
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Coupled Assembly Pass"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&self.pipeline_coupled_assembly);
                    cpass.set_bind_group(0, &self.bg_mesh, &[]);
                    cpass.set_bind_group(1, &self.bg_fields, &[]);
                    cpass.set_bind_group(2, &res.bg_solver, &[]);
                    cpass.dispatch_workgroups(num_groups_cells, 1, 1);
                }
                self.context.queue.submit(Some(encoder.finish()));
            }

            // Debug: Check matrix and RHS after assembly on first iteration
            if iter == 0 {
                self.context.device.poll(wgpu::Maintain::Wait);
                let res = self.coupled_resources.as_ref().unwrap();

                // Read diagonal entries (sample first few)
                let matrix_vals = pollster::block_on(
                    self.read_buffer_f32(&res.b_matrix_values, res.num_nonzeros),
                );
                let rhs_vals =
                    pollster::block_on(self.read_buffer_f32(&res.b_rhs, self.num_cells * 3));
                let col_indices =
                    pollster::block_on(self.read_buffer_u32(&res.b_col_indices, res.num_nonzeros));

                // Find diagonal statistics
                let num_cells = self.num_cells as usize;
                let row_offsets = pollster::block_on(
                    self.read_buffer_u32(&res.b_row_offsets, self.num_cells * 3 + 1),
                );

                // Sample diagonal for each equation type - look for actual diagonal
                let mut diag_u_sample = Vec::new();
                let mut diag_v_sample = Vec::new();
                let mut diag_p_sample = Vec::new();

                for cell_idx in 0..5.min(num_cells) {
                    // Find diagonal for u equation (row = cell_idx*3, col = cell_idx*3)
                    let row_u = cell_idx * 3;
                    let target_col_u = cell_idx * 3;
                    let start_u = row_offsets[row_u] as usize;
                    let end_u = row_offsets[row_u + 1] as usize;
                    for k in start_u..end_u {
                        if col_indices[k] as usize == target_col_u {
                            diag_u_sample.push(matrix_vals[k]);
                            break;
                        }
                    }

                    // Find diagonal for v equation (row = cell_idx*3+1, col = cell_idx*3+1)
                    let row_v = cell_idx * 3 + 1;
                    let target_col_v = cell_idx * 3 + 1;
                    let start_v = row_offsets[row_v] as usize;
                    let end_v = row_offsets[row_v + 1] as usize;
                    for k in start_v..end_v {
                        if col_indices[k] as usize == target_col_v {
                            diag_v_sample.push(matrix_vals[k]);
                            break;
                        }
                    }

                    // Find diagonal for p equation (row = cell_idx*3+2, col = cell_idx*3+2)
                    let row_p = cell_idx * 3 + 2;
                    let target_col_p = cell_idx * 3 + 2;
                    let start_p = row_offsets[row_p] as usize;
                    let end_p = row_offsets[row_p + 1] as usize;
                    for k in start_p..end_p {
                        if col_indices[k] as usize == target_col_p {
                            diag_p_sample.push(matrix_vals[k]);
                            break;
                        }
                    }
                }

                println!("Matrix diag_u sample: {:?}", diag_u_sample);
                println!("Matrix diag_v sample: {:?}", diag_v_sample);
                println!("Matrix diag_p sample: {:?}", diag_p_sample);

                // Check RHS for NaN
                let rhs_has_nan = rhs_vals.iter().any(|v| v.is_nan());
                let rhs_u: Vec<_> = (0..5.min(num_cells)).map(|i| rhs_vals[i * 3]).collect();
                let rhs_p: Vec<_> = (0..5.min(num_cells)).map(|i| rhs_vals[i * 3 + 2]).collect();
                println!("RHS u sample: {:?}", rhs_u);
                println!("RHS p sample: {:?}", rhs_p);
                println!("RHS has NaN: {}", rhs_has_nan);
            }

            // 1.5. Extract diagonal for Jacobi preconditioner
            {
                let res = self.coupled_resources.as_ref().unwrap();

                let mut encoder =
                    self.context
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Extract Diagonal Encoder"),
                        });
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Extract Diagonal Pass"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&res.pipeline_extract_diagonal);
                    cpass.set_bind_group(0, &res.bg_linear_state, &[]);
                    cpass.set_bind_group(1, &res.bg_linear_matrix, &[]);
                    cpass.set_bind_group(2, &res.bg_precond, &[]);
                    cpass.dispatch_workgroups(num_groups_cells, 1, 1);
                }
                self.context.queue.submit(Some(encoder.finish()));
            }

            // 2. Solve Coupled System entirely on the GPU via BiCGStab + Schur preconditioner
            let stats = self.solve_coupled_system();
            *self.stats_p.lock().unwrap() = stats;
            println!(
                "Coupled linear solve: {} iterations, residual {:.2e}, converged={}",
                stats.iterations, stats.residual, stats.converged
            );

            // 3. Update Fields
            {
                let res = self.coupled_resources.as_ref().unwrap();
                let mut encoder =
                    self.context
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Update Fields Encoder"),
                        });
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Update Fields Pass"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&self.pipeline_update_from_coupled);
                    cpass.set_bind_group(0, &self.bg_mesh, &[]); // Just for size
                    cpass.set_bind_group(1, &self.bg_fields, &[]);
                    cpass.set_bind_group(2, &res.bg_coupled_solution, &[]);
                    cpass.dispatch_workgroups(num_groups_cells, 1, 1);
                }
                self.context.queue.submit(Some(encoder.finish()));
            }

            // Check convergence
            if let (Some(u_before), Some(p_before)) = (u_before, p_before) {
                let u_after = pollster::block_on(self.get_u());
                let p_after = pollster::block_on(self.get_p());

                // Calculate velocity residual
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

                // Calculate pressure residual
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
                *self.outer_iterations.lock().unwrap() = iter + 1;

                println!(
                    "Coupled Residuals - U: {:.2e}, P: {:.2e}",
                    max_diff_u, max_diff_p
                );

                // Converged if both U and P are below tolerance
                if max_diff_u < convergence_tol_u && max_diff_p < convergence_tol_p {
                    println!("Coupled Solver Converged in {} iterations", iter + 1);
                    break;
                }

                // Stagnation check
                let stagnation_factor = 1e-2;
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

                if rel_u < stagnation_factor && rel_p < stagnation_factor && iter > 2 {
                    println!(
                        "Coupled solver stagnated at iter {}: U={:.2e}, P={:.2e}",
                        iter + 1,
                        max_diff_u,
                        max_diff_p
                    );
                    break;
                }

                prev_residual_u = max_diff_u;
                prev_residual_p = max_diff_p;
            } else {
                // First iteration - initialize stats
                *self.outer_residual_u.lock().unwrap() = f32::MAX;
                *self.outer_residual_p.lock().unwrap() = f32::MAX;
                *self.outer_iterations.lock().unwrap() = 1;
            }
        }

        // Update time
        self.constants.time += self.constants.dt;
        self.update_constants();

        self.context.device.poll(wgpu::Maintain::Wait);
    }

    fn solve_coupled_system(&mut self) -> LinearSolverStats {
        // Take AMG solver out to satisfy borrow checker
        let mut amg_solver = self.amg_solver.take();

        // Build AMG hierarchy if available
        if let Some(amg) = &mut amg_solver {
            println!("Building AMG hierarchy for Coupled Solver...");
            amg.build_amg_hierarchy(self);
        }

        // Reduced iterations per outer loop - rely on outer iterations for convergence
        // This is more stable for saddle-point systems
        let max_iter = 100;
        let abs_tol = 1e-4; // Relaxed - outer iteration handles fine convergence
        let rel_tol = 1e-2; // Relaxed relative tolerance

        let res = self.coupled_resources.as_ref().unwrap();
        let num_coupled_cells = self.num_cells * 3;
        let workgroup_size = 64;
        let num_groups = num_coupled_cells.div_ceil(workgroup_size);

        // Initialize r = b - Ax. Since x=0 (we zeroed it), r = b.
        let size = (num_coupled_cells as u64) * 4;

        {
            let mut encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Init Coupled Solver Encoder"),
                    });
            encoder.copy_buffer_to_buffer(&res.b_rhs, 0, &res.b_r, 0, size);
            encoder.copy_buffer_to_buffer(&res.b_rhs, 0, &res.b_r0, 0, size);
            encoder.clear_buffer(&res.b_p_solver, 0, None);
            encoder.clear_buffer(&res.b_v, 0, None);
            encoder.clear_buffer(&res.b_x, 0, None); // Ensure x is zero

            // Init scalars: rho_new=1, rho_old=1, alpha=1, beta=0, omega=1
            let init_scalars = [1.0f32, 1.0, 1.0, 0.0, 1.0];
            self.context
                .queue
                .write_buffer(&res.b_scalars, 0, bytemuck::cast_slice(&init_scalars));

            self.context.queue.submit(Some(encoder.finish()));
        }

        let mut init_resid = 0.0;
        let mut final_resid = f32::INFINITY;
        let mut converged = false;
        let mut final_iter = max_iter;
        let mut min_resid = f32::INFINITY; // Track minimum residual

        fn check_vector_for_nan(
            solver: &GpuSolver,
            buffer: &wgpu::Buffer,
            count: u32,
            label: &str,
            iter_idx: usize,
        ) -> bool {
            solver.context.device.poll(wgpu::Maintain::Wait);
            let values = pollster::block_on(solver.read_buffer_f32(buffer, count));
            if let Some(idx) = values.iter().position(|v| v.is_nan()) {
                let cell = idx / 3;
                let component = idx % 3;
                println!(
                    "NaN detected in {} at solver iter {} (cell {}, component {})",
                    label, iter_idx, cell, component
                );
                let num_cells = solver.num_cells as usize;
                let clamped_cell = cell.min(num_cells.saturating_sub(1));
                let start_cell = clamped_cell.saturating_sub(1);
                let end_cell = (clamped_cell + 2).min(num_cells);
                let start_idx = start_cell * 3;
                let end_idx = end_cell * 3;
                println!(
                    "{} sample (cells {}-{}): {:?}",
                    label,
                    start_cell,
                    end_cell - 1,
                    &values[start_idx..end_idx]
                );
                return true;
            }
            false
        }

        for iter in 0..max_iter {
            let mut encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Coupled Solver Loop Encoder"),
                    });

            // 1. rho_new = <r0, r>
            //    r_r = <r, r> (for convergence check)
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Dot Pair R0R RR"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipeline_dot_pair);
                cpass.set_bind_group(0, &res.bg_dot_params, &[]); // Params
                cpass.set_bind_group(1, &res.bg_dot_pair_r0r_rr, &[]);
                cpass.dispatch_workgroups(num_groups, 1, 1);
            }

            // Reduce rho_new and r_r
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Reduce Rho New R R"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipeline_reduce_rho_new_r_r);
                cpass.set_bind_group(0, &res.bg_scalars, &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }

            // 2. Update Beta and P
            // beta = (rho_new / rho_old) * (alpha / omega)
            // p = r + beta * (p - omega * v)
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Update Beta and P"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipeline_bicgstab_update_p);
                cpass.set_bind_group(0, &self.bg_mesh, &[]); // Unused but required by layout
                cpass.set_bind_group(1, &res.bg_linear_state, &[]);
                cpass.set_bind_group(2, &res.bg_linear_matrix, &[]);
                cpass.dispatch_workgroups(num_groups, 1, 1);
            }

            // 2.5. Apply preconditioner: p_hat = M^{-1} * p
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Apply Precond P"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&res.pipeline_apply_precond_p);
                cpass.set_bind_group(0, &res.bg_linear_state, &[]);
                cpass.set_bind_group(1, &res.bg_linear_matrix, &[]);
                cpass.set_bind_group(2, &res.bg_precond, &[]);
                cpass.dispatch_workgroups(num_groups, 1, 1);
            }

            // Submit current encoder to ensure Jacobi is done before AMG
            self.context.queue.submit(Some(encoder.finish()));

            if check_vector_for_nan(
                self,
                &res.b_p_hat,
                num_coupled_cells,
                "p_hat after block preconditioner",
                iter,
            ) {
                final_resid = f32::NAN;
                final_iter = iter + 1;
                break;
            }

            // AMG Preconditioner for Pressure
            if let Some(amg) = &mut amg_solver {
                let res = self.coupled_resources.as_ref().unwrap();
                amg.solve_coupled_pressure(self, &res.b_p_solver, &res.b_p_hat);
            }

            if check_vector_for_nan(
                self,
                &res.b_p_hat,
                num_coupled_cells,
                "p_hat after AMG",
                iter,
            ) {
                final_resid = f32::NAN;
                final_iter = iter + 1;
                break;
            }

            // Start new encoder
            let mut encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Coupled Solver Loop Encoder 2"),
                    });

            // 3. v = A * p_hat (preconditioned)
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SpMV Phat->V"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&res.pipeline_spmv_phat_v);
                cpass.set_bind_group(0, &res.bg_linear_state, &[]);
                cpass.set_bind_group(1, &res.bg_linear_matrix, &[]);
                cpass.set_bind_group(2, &res.bg_precond, &[]);
                cpass.dispatch_workgroups(num_groups, 1, 1);
            }

            // 4. alpha = rho_new / <r0, v>
            //    Compute <r0, v>
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Dot R0 V"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipeline_dot);
                cpass.set_bind_group(0, &res.bg_dot_params, &[]); // Params
                cpass.set_bind_group(1, &res.bg_dot_r0_v, &[]);
                cpass.dispatch_workgroups(num_groups, 1, 1);
            }

            // Reduce <r0, v> and update alpha
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Reduce R0 V"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipeline_reduce_r0_v);
                cpass.set_bind_group(0, &res.bg_scalars, &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }

            // 5. s = r - alpha * v
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Update S"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipeline_bicgstab_update_s);
                cpass.set_bind_group(0, &self.bg_mesh, &[]);
                cpass.set_bind_group(1, &res.bg_linear_state, &[]);
                cpass.set_bind_group(2, &res.bg_linear_matrix, &[]);
                cpass.dispatch_workgroups(num_groups, 1, 1);
            }

            // 5.5. Apply preconditioner: s_hat = M^{-1} * s
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Apply Precond S"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&res.pipeline_apply_precond_s);
                cpass.set_bind_group(0, &res.bg_linear_state, &[]);
                cpass.set_bind_group(1, &res.bg_linear_matrix, &[]);
                cpass.set_bind_group(2, &res.bg_precond, &[]);
                cpass.dispatch_workgroups(num_groups, 1, 1);
            }

            // Submit current encoder to ensure Jacobi is done before AMG
            self.context.queue.submit(Some(encoder.finish()));

            if check_vector_for_nan(
                self,
                &res.b_s_hat,
                num_coupled_cells,
                "s_hat after block preconditioner",
                iter,
            ) {
                final_resid = f32::NAN;
                final_iter = iter + 1;
                break;
            }

            // AMG Preconditioner for Pressure
            if let Some(amg) = &mut amg_solver {
                let res = self.coupled_resources.as_ref().unwrap();
                amg.solve_coupled_pressure(self, &res.b_s, &res.b_s_hat);
            }

            if check_vector_for_nan(
                self,
                &res.b_s_hat,
                num_coupled_cells,
                "s_hat after AMG",
                iter,
            ) {
                final_resid = f32::NAN;
                final_iter = iter + 1;
                break;
            }

            // Start new encoder
            let mut encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Coupled Solver Loop Encoder 3"),
                    });

            // 6. t = A * s_hat (preconditioned)
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SpMV Shat->T"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&res.pipeline_spmv_shat_t);
                cpass.set_bind_group(0, &res.bg_linear_state, &[]);
                cpass.set_bind_group(1, &res.bg_linear_matrix, &[]);
                cpass.set_bind_group(2, &res.bg_precond, &[]);
                cpass.dispatch_workgroups(num_groups, 1, 1);
            }

            // 7. omega = <t, s> / <t, t>
            //    Compute <t, s> and <t, t>
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Dot Pair TS TT"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipeline_dot_pair);
                cpass.set_bind_group(0, &res.bg_dot_params, &[]); // Params
                cpass.set_bind_group(1, &res.bg_dot_pair_tstt, &[]);
                cpass.dispatch_workgroups(num_groups, 1, 1);
            }

            // Reduce and update omega
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Reduce TS TT"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipeline_reduce_t_s_t_t);
                cpass.set_bind_group(0, &res.bg_scalars, &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }

            // 8. x = x + alpha * p_hat + omega * s_hat (using preconditioned search directions)
            //    r = s - omega * t
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Update X and R"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&res.pipeline_bicgstab_precond_update_x_r);
                cpass.set_bind_group(0, &res.bg_linear_state, &[]);
                cpass.set_bind_group(1, &res.bg_linear_matrix, &[]);
                cpass.set_bind_group(2, &res.bg_precond, &[]);
                cpass.dispatch_workgroups(num_groups, 1, 1);
            }

            // 9. Update rho_old = rho_new
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Update Rho Old"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipeline_update_rho_old);
                cpass.set_bind_group(0, &res.bg_scalars, &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }

            self.context.queue.submit(Some(encoder.finish()));

            self.context.device.poll(wgpu::Maintain::Wait);
            let r_vals = pollster::block_on(self.read_buffer_f32(&res.b_r, num_coupled_cells));
            if let Some(idx) = r_vals.iter().position(|v| v.is_nan()) {
                let cell = idx / 3;
                let component = idx % 3;
                println!(
                    "Detected NaN in residual at iter {} (cell {}, component {})",
                    iter, cell, component
                );

                let sample = |data: &[f32], name: &str| {
                    let num_cells = self.num_cells as usize;
                    let cell = cell.min(num_cells.saturating_sub(1));
                    let start_cell = cell.saturating_sub(1);
                    let end_cell = (cell + 2).min(num_cells);
                    let start_idx = start_cell * 3;
                    let end_idx = end_cell * 3;
                    let has_nan = data.iter().any(|v| v.is_nan());
                    let slice = &data[start_idx..end_idx];
                    println!(
                        "{} sample (cells {}-{}): {:?}",
                        name,
                        start_cell,
                        end_cell - 1,
                        slice
                    );
                    println!("{} has NaN: {}", name, has_nan);
                };

                let p_vals =
                    pollster::block_on(self.read_buffer_f32(&res.b_p_solver, num_coupled_cells));
                let v_vals =
                    pollster::block_on(self.read_buffer_f32(&res.b_v, num_coupled_cells));
                let s_vals =
                    pollster::block_on(self.read_buffer_f32(&res.b_s, num_coupled_cells));
                let t_vals =
                    pollster::block_on(self.read_buffer_f32(&res.b_t, num_coupled_cells));
                let scalars_snapshot =
                    pollster::block_on(self.read_buffer_f32(&res.b_scalars, 16));

                sample(&r_vals, "r");
                sample(&p_vals, "p");
                sample(&v_vals, "v");
                sample(&s_vals, "s");
                sample(&t_vals, "t");
                let scalar_len = scalars_snapshot.len().min(9);
                println!("Scalars snapshot: {:?}", &scalars_snapshot[..scalar_len]);

                final_resid = f32::NAN;
                final_iter = iter + 1;
                break;
            }

            // Check convergence every 10 iterations
            if iter % 10 == 0 {
                // Copy to staging buffer
                let mut encoder =
                    self.context
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Copy Scalars to Staging"),
                        });
                encoder.copy_buffer_to_buffer(&res.b_scalars, 0, &res.b_staging_scalar, 0, 64);
                self.context.queue.submit(Some(encoder.finish()));

                // Read from staging buffer
                let buffer_slice = res.b_staging_scalar.slice(..);
                let (sender, receiver) = std::sync::mpsc::channel();
                buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
                self.context.device.poll(wgpu::Maintain::Wait);

                if let Ok(Ok(())) = receiver.recv() {
                    let data = buffer_slice.get_mapped_range();
                    let scalars: &[f32] = bytemuck::cast_slice(&data);
                    let r_r = scalars[8];
                    let resid = r_r.sqrt();

                    drop(data);
                    res.b_staging_scalar.unmap();

                    if iter == 0 {
                        init_resid = resid;
                        println!("Coupled Solver Initial Residual: {:.2e}", init_resid);
                    }

                    final_resid = resid;
                    min_resid = min_resid.min(resid);
                    
                    // Only print every 20 iterations to reduce noise
                    if iter % 20 == 0 {
                        println!("Coupled Solver Iter: {}, Residual: {:.2e}", iter, resid);
                    }

                    if resid < abs_tol || (init_resid > 0.0 && resid / init_resid < rel_tol) {
                        converged = true;
                        final_iter = iter + 1;
                        println!("Coupled Solver converged at iter {}, Residual: {:.2e}", iter, resid);
                        break;
                    }

                    // Exit early if diverging - be more aggressive
                    if resid > 10.0 * min_resid && iter > 20 {
                        println!("Coupled Solver diverging at iter {}, stopping (resid={:.2e}, min={:.2e})", 
                                 iter, resid, min_resid);
                        final_iter = iter + 1;
                        break;
                    }
                    
                    // Also exit if residual is extremely large
                    if resid > 1e6 {
                        println!("Coupled Solver residual too large ({:.2e}), stopping", resid);
                        final_iter = iter + 1;
                        break;
                    }

                    if resid.is_nan() {
                        println!("Coupled Solver NaN detected at iter {}", iter);
                        final_iter = iter + 1;
                        break;
                    }
                }
            }
        }

        // Restore AMG solver
        self.amg_solver = amg_solver;

        LinearSolverStats {
            iterations: final_iter as u32,
            residual: final_resid,
            converged,
            diverged: !converged && final_resid.is_nan(),
            time: std::time::Duration::default(),
        }
    }
}
