use super::structs::GpuSolver;
use std::sync::atomic::Ordering;

impl GpuSolver {
    pub fn zero_buffer(&self, buffer: &wgpu::Buffer, size: u64) {
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Zero Buffer Encoder"),
                });
        encoder.clear_buffer(buffer, 0, Some(size));
        self.context.queue.submit(Some(encoder.finish()));
    }

    pub async fn solve(&self) {
        let max_iter = 1000;
        let tol = 1e-6;
        let n = self.num_cells;
        let workgroup_size = 64;
        let num_groups = (n + workgroup_size - 1) / workgroup_size;

        // Initialize r = b - Ax. Since x=0, r = b.
        let size = (n as u64) * 4;
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

        for _iter in 0..max_iter {
            // 1. rho_new = (r0, r) -> b_dot_result
            // 2. r_r = (r, r) -> b_dot_result_2
            self.encode_dot_pair(
                &mut encoder,
                &self.bg_dot_pair_r0r_rr,
                num_groups,
            );

            // Reduce rho_new and r_r
            self.encode_scalar_reduce(&mut encoder, &self.pipeline_reduce_rho_new_r_r, num_groups);

            pending_commands = true;

            // Check convergence every 50 iterations
            if _iter % 50 == 0 {
                // Submit pending commands before reading back
                if pending_commands {
                    // println!("Submitting batch at iter {}", _iter);
                    let start = if self.profiling_enabled.load(Ordering::Relaxed) {
                        Some(std::time::Instant::now())
                    } else {
                        None
                    };
                    self.context.queue.submit(Some(encoder.finish()));
                    if let Some(start) = start {
                        *self.time_compute.lock().unwrap() += start.elapsed();
                    }
                    encoder = self.context.device.create_command_encoder(
                        &wgpu::CommandEncoderDescriptor {
                            label: Some("Solver Loop Encoder"),
                        },
                    );
                    pending_commands = false;
                }

                let r_r = self.read_scalar_r_r().await;
                let res = r_r.sqrt();
                if res < tol {
                    println!("Converged at iter {} with residual {:e}", _iter, res);
                    break;
                }
                if _iter == max_iter - 1 {
                    println!("Solver failed to converge! Final residual: {:e}", res);
                }
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
    }

    fn encode_compute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        num_groups: u32,
    ) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &self.bg_mesh, &[]);
        cpass.set_bind_group(1, bind_group, &[]); // linear_state
        cpass.set_bind_group(2, &self.bg_linear_matrix, &[]);
        cpass.dispatch_workgroups(num_groups, 1, 1);
    }

    fn encode_spmv(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        num_groups: u32,
    ) {
        self.encode_compute(encoder, pipeline, &self.bg_linear_state, num_groups);
    }

    fn encode_dot(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bg_dot_inputs: &wgpu::BindGroup,
        num_groups: u32,
    ) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Dot Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline_dot);
        cpass.set_bind_group(0, &self.bg_dot_params, &[]);
        cpass.set_bind_group(1, bg_dot_inputs, &[]);
        cpass.dispatch_workgroups(num_groups, 1, 1);
    }

    fn encode_dot_pair(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        num_groups: u32,
    ) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Dot Pair Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline_dot_pair);
        cpass.set_bind_group(0, &self.bg_dot_params, &[]);
        cpass.set_bind_group(1, bind_group, &[]);
        cpass.dispatch_workgroups(num_groups, 1, 1);
    }

    fn encode_scalar_compute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
    ) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Scalar Compute Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &self.bg_scalars, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }

    fn encode_scalar_reduce(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        _num_groups: u32,
    ) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Scalar Reduce Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &self.bg_scalars, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }

    async fn read_scalar_r_r(&self) -> f32 {
        // Read r_r from b_scalars.
        // Offset of r_r is 8 * 4 = 32 bytes.
        let offset = 32;
        let size = 4;

        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Read Scalar Encoder"),
                });
        encoder.copy_buffer_to_buffer(&self.b_scalars, offset, &self.b_staging_scalar, 0, size);
        self.context.queue.submit(Some(encoder.finish()));

        let slice = self.b_staging_scalar.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        self.context.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let val = f32::from_ne_bytes(data[0..4].try_into().unwrap());
        drop(data);
        self.b_staging_scalar.unmap();

        val
    }
}
