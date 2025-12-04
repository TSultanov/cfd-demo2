use crate::solver::gpu::profiling::ProfileCategory;
use crate::solver::gpu::structs::GpuSolver;

impl GpuSolver {
    pub async fn solve(
        &mut self,
        field_name: &str,
    ) -> crate::solver::gpu::structs::LinearSolverStats {
        let start_time = std::time::Instant::now();
        let max_iter = 1000;
        let abs_tol = 1e-6;
        let rel_tol = 1e-4;
        let stagnation_tolerance = 1e-3;
        let stagnation_factor = 1e-4; // relative change threshold
        let n = self.num_cells;
        let workgroup_size = 64;
        let num_groups = n.div_ceil(workgroup_size);

        // Initialize r = b - Ax. Since x=0, r = b.
        let size = (n as u64) * 4;
        self.zero_buffer(&self.b_x, size);
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Init Solver Encoder"),
                });
        encoder.copy_buffer_to_buffer(&self.b_rhs, 0, &self.b_r, 0, size);
        encoder.copy_buffer_to_buffer(&self.b_rhs, 0, &self.b_r0, 0, size);
        encoder.clear_buffer(&self.b_p_solver, 0, None);
        encoder.clear_buffer(&self.b_v, 0, None);

        // Init scalars
        self.encode_scalar_compute(&mut encoder, &self.pipeline_init_scalars);

        self.context.queue.submit(Some(encoder.finish()));

        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Solver Loop Encoder"),
                });
        let mut pending_commands = false;
        let mut init_resid = 0.0;
        let mut converged = false;
        let mut final_iter = max_iter;
        let mut prev_res = f32::MAX;

        for iter in 0..max_iter {
            // 1. rho_new = (r0, r) -> b_dot_result
            // 2. r_r = (r, r) -> b_dot_result_2
            self.encode_dot_pair(&mut encoder, &self.bg_dot_pair_r0r_rr, num_groups);

            // Reduce rho_new and r_r
            self.encode_scalar_reduce(&mut encoder, &self.pipeline_reduce_rho_new_r_r, num_groups);

            pending_commands = true;

            // Check convergence every iteration
            {
                // Submit pending commands before reading back
                if pending_commands {
                    // Submit pending commands before reading scalar from GPU
                    let dispatch_start = std::time::Instant::now();
                    self.context.queue.submit(Some(encoder.finish()));
                    self.profiling_stats.record_location(
                        "linear_solver:submit_pending",
                        ProfileCategory::GpuDispatch,
                        dispatch_start.elapsed(),
                        0,
                    );
                    encoder = self.context.device.create_command_encoder(
                        &wgpu::CommandEncoderDescriptor {
                            label: Some("Solver Loop Encoder"),
                        },
                    );
                    pending_commands = false;
                }

                let r_r = self.read_scalar_r_r().await;
                let res = r_r.sqrt();

                if iter == 0 {
                    init_resid = res;
                }

                if res < abs_tol || (iter > 0 && res < rel_tol * init_resid) {
                    converged = true;
                    final_iter = iter + 1;
                    break;
                }

                // Stagnation check (relative)
                if iter > 0 {
                    let rel_change = if prev_res.is_finite() && prev_res > 1e-20 {
                        ((res - prev_res) / prev_res).abs()
                    } else {
                        f32::INFINITY
                    };
                    if rel_change < stagnation_factor && res < stagnation_tolerance {
                        final_iter = iter + 1;
                        break;
                    }
                }
                prev_res = res;
            }

            // p = r + beta * (p - omega * v)
            self.encode_compute(
                &mut encoder,
                &self.pipeline_bicgstab_update_p,
                &self.bg_linear_state,
                num_groups,
            );

            // v = A * p
            self.encode_spmv(&mut encoder, &self.pipeline_spmv_p_v, num_groups);

            // r0_v = (r0, v)
            self.encode_dot(&mut encoder, &self.bg_dot_r0_v, num_groups);
            self.encode_scalar_reduce(&mut encoder, &self.pipeline_reduce_r0_v, num_groups);

            // s = r - alpha * v
            self.encode_compute(
                &mut encoder,
                &self.pipeline_bicgstab_update_s,
                &self.bg_linear_state,
                num_groups,
            );

            // t = A * s
            self.encode_spmv(&mut encoder, &self.pipeline_spmv_s_t, num_groups);

            // t_s = (t, s) -> b_dot_result
            // t_t = (t, t) -> b_dot_result_2
            self.encode_dot_pair(&mut encoder, &self.bg_dot_pair_tstt, num_groups);
            self.encode_scalar_reduce(&mut encoder, &self.pipeline_reduce_t_s_t_t, num_groups);

            // x = x + alpha * p + omega * s
            // r = s - omega * t
            self.encode_compute(
                &mut encoder,
                &self.pipeline_bicgstab_update_x_r,
                &self.bg_linear_state,
                num_groups,
            );

            pending_commands = true;
        }

        if pending_commands {
            self.context.queue.submit(Some(encoder.finish()));
        }

        // Refresh residual using the latest r to avoid reporting stale values.
        let final_resid = {
            let mut encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Final Residual Encoder"),
                    });
            self.encode_dot_pair(&mut encoder, &self.bg_dot_pair_r0r_rr, num_groups);
            self.encode_scalar_reduce(&mut encoder, &self.pipeline_reduce_rho_new_r_r, num_groups);
            self.context.queue.submit(Some(encoder.finish()));

            let r_r = self.read_scalar_r_r().await;
            r_r.sqrt()
        };

        // Check for divergence
        let diverged = !converged
            && (final_resid.is_nan()
                || final_resid.is_infinite()
                || final_resid > init_resid * 10.0);

        if diverged {
            println!(
                "BiCGStab diverged for {}. Residual: {:.2e}.",
                field_name, final_resid
            );
        } else {
            println!(
                "Solved {}: {} iterations, Residual: {:.2e}",
                field_name, final_iter, final_resid
            );
        }

        crate::solver::gpu::structs::LinearSolverStats {
            iterations: final_iter,
            residual: final_resid,
            converged,
            diverged,
            time: start_time.elapsed(),
        }
    }
}
