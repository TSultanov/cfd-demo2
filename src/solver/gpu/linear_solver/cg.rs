use crate::solver::gpu::structs::GpuSolver;

impl GpuSolver {
    pub async fn solve_cg(
        &self,
        field_name: &str,
    ) -> crate::solver::gpu::structs::LinearSolverStats {
        let start_time = std::time::Instant::now();
        let max_iter = 1000;
        let abs_tol = 1e-6;
        let rel_tol = 1e-4;
        let n = self.num_cells;
        let workgroup_size = 64;
        let num_groups = n.div_ceil(workgroup_size);

        // Initialize r = b - Ax. Since x=0, r = b.
        let size = (n as u64) * 4;
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Init CG Solver Encoder"),
                });
        encoder.copy_buffer_to_buffer(&self.b_rhs, 0, &self.b_r, 0, size);
        encoder.copy_buffer_to_buffer(&self.b_rhs, 0, &self.b_p_solver, 0, size); // p = r
        encoder.clear_buffer(&self.b_x, 0, None); // Ensure x starts at 0
        encoder.clear_buffer(&self.b_v, 0, None);

        // Init scalars: rho_old = r.r
        // Compute r.r -> dot_result_1
        self.encode_dot(&mut encoder, &self.bg_dot_r_r, num_groups);
        // Init scalars (sums dot_result_1 to rho_old)
        self.encode_scalar_compute(&mut encoder, &self.pipeline_init_cg_scalars);

        self.context.queue.submit(Some(encoder.finish()));

        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("CG Solver Loop Encoder"),
                });
        let mut pending_commands = false;
        let mut init_resid = 0.0;
        let mut final_resid = 0.0;
        let mut converged = false;
        let mut final_iter = max_iter;

        for iter in 0..max_iter {
            // v = Ap
            self.encode_spmv(&mut encoder, &self.pipeline_spmv_p_v, num_groups);

            // d = p.v -> dot_result_1
            self.encode_dot(&mut encoder, &self.bg_dot_p_v, num_groups);

            // Reduce d to r0_v (reusing r0_v slot for p.v)
            self.encode_scalar_reduce(&mut encoder, &self.pipeline_reduce_r0_v, num_groups);

            // alpha = rho_old / d
            self.encode_scalar_compute(&mut encoder, &self.pipeline_update_cg_alpha);

            // x += alpha * p
            // r -= alpha * v
            self.encode_compute(
                &mut encoder,
                &self.pipeline_cg_update_x_r,
                &self.bg_linear_state,
                num_groups,
            );

            // rho_new = r.r -> dot_result_1
            self.encode_dot(&mut encoder, &self.bg_dot_r_r, num_groups);

            // Reduce rho_new
            self.encode_scalar_reduce(&mut encoder, &self.pipeline_reduce_rho_new_r_r, num_groups);

            pending_commands = true;

            // Check convergence
            if iter % 1 == 0 {
                if pending_commands {
                    self.context.queue.submit(Some(encoder.finish()));
                    encoder = self.context.device.create_command_encoder(
                        &wgpu::CommandEncoderDescriptor {
                            label: Some("CG Solver Loop Encoder"),
                        },
                    );
                    pending_commands = false;
                }

                // Read rho_new (which is in scalars.rho_new)
                // We can reuse read_scalar_r_r? No, that reads r_r.
                // We need to read rho_new.
                // rho_new is at offset 4 (f32).
                // Let's check GpuScalars struct in scalars.wgsl.
                // struct GpuScalars { rho_old, rho_new, ... }
                // rho_old is 0, rho_new is 4.

                let rho_new = self.read_scalar_rho_new().await;
                let res = rho_new.sqrt();
                final_resid = res;

                if iter == 0 {
                    init_resid = res;
                }

                if res < abs_tol || (iter > 0 && res < rel_tol * init_resid) {
                    converged = true;
                    final_iter = iter + 1;
                    break;
                }
            }

            // beta = rho_new / rho_old
            self.encode_scalar_compute(&mut encoder, &self.pipeline_update_cg_beta);

            // p = r + beta * p
            self.encode_compute(
                &mut encoder,
                &self.pipeline_cg_update_p,
                &self.bg_linear_state,
                num_groups,
            );

            // rho_old = rho_new
            self.encode_scalar_compute(&mut encoder, &self.pipeline_update_rho_old);
        }

        if pending_commands {
            self.context.queue.submit(Some(encoder.finish()));
        }

        println!(
            "Solved CG {}: {} iterations, Residual: {:.2e}",
            field_name, final_iter, final_resid
        );

        let diverged = final_resid.is_nan() || final_resid.is_infinite();

        crate::solver::gpu::structs::LinearSolverStats {
            iterations: final_iter,
            residual: final_resid,
            converged,
            diverged,
            time: start_time.elapsed(),
        }
    }
}
