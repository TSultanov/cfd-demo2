// Coupled Solver for CFD
//
// This solver solves the momentum and continuity equations simultaneously
// in a block-coupled manner, as opposed to segregated predictor-corrector approaches.
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

use super::profiling::ProfileCategory;
use super::structs::{GpuSolver, LinearSolverStats};
use std::time::Instant;

/// Enable debug reads for diagnosing matrix assembly issues.
/// Set to true to enable debug output on the first iteration.
/// WARNING: This adds significant GPU-CPU synchronization overhead (~65ms per step).
const DEBUG_READS_ENABLED: bool = false;

impl GpuSolver {
    /// Performs a single timestep using the coupled solver approach.
    ///
    /// Unlike segregated predictors that iterate between momentum and pressure solves,
    /// the coupled solver assembles and solves the full block system in one go,
    /// with outer iterations for non-linearity.
    pub fn step_coupled(&mut self) {
        let workgroup_size = 64;
        let num_groups_cells = self.num_cells.div_ceil(workgroup_size);

        // Save old velocity for under-relaxation and time derivative
        self.copy_u_to_u_old();

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

        // Reset async reader to clear old values
        if let Some(res) = self.coupled_resources.as_ref() {
            res.async_scalar_reader.borrow_mut().reset();
        }

        for iter in 0..max_coupled_iters {
            self.profiling_stats.increment_iteration();
            println!("Coupled Iteration: {}", iter + 1);

            // Update fluxes with current velocity for advection terms
            if iter > 0 {
                self.compute_fluxes();
                // Re-compute d_p based on current velocity/fluxes
                self.initialize_d_p(num_groups_cells);
            }

            // Compute Gradients for Higher Order Schemes (2nd Order Upwind / QUICK)
            // We need to compute grad U and grad V and store them in the coupled resources
            // This is done inside the loop to use the latest U field for deferred correction
            if let Some(res) = self.coupled_resources.as_ref() {
                let dispatch_start = Instant::now();
                let mut encoder =
                    self.context
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Gradient Coupled Encoder"),
                        });
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Gradient Coupled Pass"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&self.pipeline_gradient_coupled);
                    cpass.set_bind_group(0, &self.bg_mesh, &[]);
                    cpass.set_bind_group(1, &self.bg_fields, &[]);
                    // Bind Group 2: Output buffers (grad_u, grad_v)
                    // We reuse bg_solver which has them at bindings 3 and 4
                    cpass.set_bind_group(2, &res.bg_solver, &[]);
                    cpass.dispatch_workgroups(num_groups_cells, 1, 1);
                }
                self.context.queue.submit(Some(encoder.finish()));
                self.profiling_stats.record_location(
                    "coupled:gradient_coupled",
                    ProfileCategory::GpuDispatch,
                    dispatch_start.elapsed(),
                    0,
                );
            }

            // Debug: Check d_p values on first iteration - DEBUG READ
            if DEBUG_READS_ENABLED && iter == 0 {
                let read_start = Instant::now();
                let d_p_vals = pollster::block_on(self.get_d_p());
                self.profiling_stats.record_location(
                    "coupled:debug_get_d_p",
                    ProfileCategory::GpuRead,
                    read_start.elapsed(),
                    (self.num_cells as u64) * 4,
                );
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
                let dispatch_start = Instant::now();
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
                self.profiling_stats.record_location(
                    "coupled:assembly",
                    ProfileCategory::GpuDispatch,
                    dispatch_start.elapsed(),
                    0,
                );
            }

            // Debug: Check matrix and RHS after assembly on first iteration - DEBUG READS
            if DEBUG_READS_ENABLED && iter == 0 {
                let sync_start = Instant::now();
                let _ = self
                    .context
                    .device
                    .poll(wgpu::PollType::wait_indefinitely());
                self.profiling_stats.record_location(
                    "coupled:debug_sync",
                    ProfileCategory::GpuSync,
                    sync_start.elapsed(),
                    0,
                );
                let res = self.coupled_resources.as_ref().unwrap();

                // Read diagonal entries (sample first few) - DEBUG READS
                let read_start = Instant::now();
                let matrix_vals = pollster::block_on(
                    self.read_buffer_f32(&res.b_matrix_values, res.num_nonzeros),
                );
                self.profiling_stats.record_location(
                    "coupled:debug_read_matrix",
                    ProfileCategory::GpuRead,
                    read_start.elapsed(),
                    (res.num_nonzeros as u64) * 4,
                );

                let read_start = Instant::now();
                let rhs_vals =
                    pollster::block_on(self.read_buffer_f32(&res.b_rhs, self.num_cells * 3));
                self.profiling_stats.record_location(
                    "coupled:debug_read_rhs",
                    ProfileCategory::GpuRead,
                    read_start.elapsed(),
                    (self.num_cells as u64) * 3 * 4,
                );

                let read_start = Instant::now();
                let col_indices =
                    pollster::block_on(self.read_buffer_u32(&res.b_col_indices, res.num_nonzeros));
                self.profiling_stats.record_location(
                    "coupled:debug_read_col_indices",
                    ProfileCategory::GpuRead,
                    read_start.elapsed(),
                    (res.num_nonzeros as u64) * 4,
                );

                // Find diagonal statistics
                let num_cells = self.num_cells as usize;
                let read_start = Instant::now();
                let row_offsets = pollster::block_on(
                    self.read_buffer_u32(&res.b_row_offsets, self.num_cells * 3 + 1),
                );
                self.profiling_stats.record_location(
                    "coupled:debug_read_row_offsets",
                    ProfileCategory::GpuRead,
                    read_start.elapsed(),
                    ((self.num_cells * 3 + 1) as u64) * 4,
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

            // 1.5 & 1.8. Assemble Scalar Pressure Matrix
            {
                let _res = self.coupled_resources.as_ref().unwrap();

                let mut encoder =
                    self.context
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Precond Setup Encoder"),
                        });

                // Pass 2: Scalar Pressure Assembly
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Scalar Pressure Assembly Pass"),
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

            // 2. Solve Coupled System using FGMRES-based coupled solver (CPU-side Krylov, GPU SpMV/precond)
            let stats = self.solve_coupled_system();
            *self.stats_p.lock().unwrap() = stats;
            println!(
                "Coupled linear solve: {} iterations, residual {:.2e}, converged={}",
                stats.iterations, stats.residual, stats.converged
            );

            // 3. Update Fields
            {
                let res = self.coupled_resources.as_ref().unwrap();
                let dispatch_start = Instant::now();
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
                self.profiling_stats.record_location(
                    "coupled:update_fields",
                    ProfileCategory::GpuDispatch,
                    dispatch_start.elapsed(),
                    0,
                );
            }

            // Check convergence using GPU-computed max-diff (Async)
            if iter > 0 {
                // Start async computation and read for CURRENT iteration
                self.start_compute_max_diff_gpu();

                // Check if PREVIOUS iteration's result is ready
                if let Some(res) = self.coupled_resources.as_ref() {
                    let mut reader = res.async_scalar_reader.borrow_mut();
                    reader.poll(); // Poll for completion of previous reads

                    // Check if we have a result available
                    if let Some(results) = reader.get_last_value_vec(2) {
                        let max_diff_u = results[0] as f64;
                        let max_diff_p = results[1] as f64;

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
                        let rel_u = if prev_residual_u.is_finite() && prev_residual_u.abs() > 1e-14
                        {
                            ((max_diff_u - prev_residual_u) / prev_residual_u).abs()
                        } else {
                            f64::INFINITY
                        };
                        let rel_p = if prev_residual_p.is_finite() && prev_residual_p.abs() > 1e-14
                        {
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
                    }
                }
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

        let _ = self
            .context
            .device
            .poll(wgpu::PollType::wait_indefinitely());
    }

    fn solve_coupled_system(&mut self) -> LinearSolverStats {
        // Use the FGMRES-based coupled solver implementation from
        // `coupled_solver_fgmres.rs` to solve the coupled block system.
        self.solve_coupled_fgmres()
    }

    /// Start computing max-diff on GPU and initiate async read
    fn start_compute_max_diff_gpu(&self) {
        let res = match self.coupled_resources.as_ref() {
            Some(r) => r,
            None => return,
        };

        let dispatch_start = Instant::now();

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct MaxDiffParams {
            n: u32,
            _pad1: u32,
            _pad2: u32,
            _pad3: u32,
        }

        let workgroup_size = 64u32;
        let num_groups = self.num_cells.div_ceil(workgroup_size);

        // Write params for partial reduction (n = num_cells)
        let params_partial = MaxDiffParams {
            n: self.num_cells,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };

        // Write params for final reduction (n = num_groups)
        let params_reduce = MaxDiffParams {
            n: num_groups,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };

        // Write both params to the buffer at different offsets
        // Offset 0: Partial
        // Offset 256: Final
        self.context.queue.write_buffer(
            &res.b_max_diff_params,
            0,
            bytemuck::bytes_of(&params_partial),
        );
        self.context.queue.write_buffer(
            &res.b_max_diff_params,
            256,
            bytemuck::bytes_of(&params_reduce),
        );

        // Create single command encoder for all GPU work
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Max Diff Batched Encoder"),
                });

        // Pass 1: Partial reduction for U
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Max Diff U Partial Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&res.pipeline_max_diff_u_partial);
            cpass.set_bind_group(0, &res.bg_max_diff_u, &[]);
            // Use dynamic offset 0
            cpass.set_bind_group(1, &res.bg_max_diff_params, &[0]);
            cpass.dispatch_workgroups(num_groups, 1, 1);
        }

        // Pass 2: Partial reduction for P
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Max Diff P Partial Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&res.pipeline_max_diff_p_partial);
            cpass.set_bind_group(0, &res.bg_max_diff_p, &[]);
            // Use dynamic offset 0
            cpass.set_bind_group(1, &res.bg_max_diff_params, &[0]);
            cpass.dispatch_workgroups(num_groups, 1, 1);
        }

        // Pass 3: Final reduction for U (writes to result[0])
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Max Diff U Final Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&res.pipeline_max_diff_reduce);
            cpass.set_bind_group(0, &res.bg_reduce_u, &[]);
            // Use dynamic offset 256
            cpass.set_bind_group(1, &res.bg_max_diff_params, &[256]);
            cpass.dispatch_workgroups(1, 1, 1);
        }

        // Pass 4: Final reduction for P (writes to result[1])
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Max Diff P Final Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&res.pipeline_max_diff_reduce_p);
            cpass.set_bind_group(0, &res.bg_reduce_p, &[]);
            // Use dynamic offset 256
            cpass.set_bind_group(1, &res.bg_max_diff_params, &[256]);
            cpass.dispatch_workgroups(1, 1, 1);
        }

        // Submit all reductions
        self.context.queue.submit(Some(encoder.finish()));

        self.profiling_stats.record_location(
            "coupled:gpu_max_diff_dispatch",
            ProfileCategory::GpuDispatch,
            dispatch_start.elapsed(),
            0,
        );

        // Start async read
        let mut reader = res.async_scalar_reader.borrow_mut();
        reader.start_read(
            &self.context.device,
            &self.context.queue,
            &res.b_max_diff_result,
            0,
        );
    }
}
