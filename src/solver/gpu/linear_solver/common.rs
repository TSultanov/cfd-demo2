use crate::solver::gpu::structs::{GpuSolver, LinearSolverStats, SolverParams};
use std::time::Instant;

const LINEAR_WORKGROUP_SIZE: u32 = 64;

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Default)]
struct GpuLinearScalars {
    rho_old: f32,
    rho_new: f32,
    alpha: f32,
    beta: f32,
    omega: f32,
    r0_v: f32,
    t_s: f32,
    t_t: f32,
    r_r: f32,
}

impl GpuLinearScalars {
    fn from_slice(values: &[f32]) -> Self {
        Self {
            rho_old: values.get(0).copied().unwrap_or(0.0),
            rho_new: values.get(1).copied().unwrap_or(0.0),
            alpha: values.get(2).copied().unwrap_or(0.0),
            beta: values.get(3).copied().unwrap_or(0.0),
            omega: values.get(4).copied().unwrap_or(0.0),
            r0_v: values.get(5).copied().unwrap_or(0.0),
            t_s: values.get(6).copied().unwrap_or(0.0),
            t_t: values.get(7).copied().unwrap_or(0.0),
            r_r: values.get(8).copied().unwrap_or(0.0),
        }
    }
}

impl GpuSolver {
    pub fn set_linear_system(&self, matrix_values: &[f32], rhs: &[f32]) {
        assert_eq!(
            matrix_values.len(),
            self.num_nonzeros as usize,
            "matrix_values length mismatch"
        );
        assert_eq!(
            rhs.len(),
            self.num_cells as usize,
            "rhs length mismatch"
        );
        self.context
            .queue
            .write_buffer(&self.b_matrix_values, 0, bytemuck::cast_slice(matrix_values));
        self.context
            .queue
            .write_buffer(&self.b_rhs, 0, bytemuck::cast_slice(rhs));
    }

    pub async fn get_linear_solution(&self) -> Vec<f32> {
        let data = self
            .read_buffer(&self.b_x, (self.num_cells as u64) * 4)
            .await;
        bytemuck::cast_slice(&data).to_vec()
    }

    pub fn solve_linear_system_cg(&self, max_iters: u32, tol: f32) -> LinearSolverStats {
        self.solve_linear_system_cg_with_size(self.num_cells, max_iters, tol)
    }

    pub fn solve_linear_system_cg_with_size(
        &self,
        n: u32,
        max_iters: u32,
        tol: f32,
    ) -> LinearSolverStats {
        if n > self.num_cells {
            panic!(
                "requested solve size {} exceeds allocated size {}",
                n, self.num_cells
            );
        }
        let num_groups = self.update_linear_solver_params(n);
        let start = Instant::now();
        let buffer_size = (n as u64) * 4;

        let zeros = vec![0.0_f32; n as usize];
        self.context
            .queue
            .write_buffer(&self.b_x, 0, bytemuck::cast_slice(&zeros));

        self.copy_buffer(&self.b_rhs, &self.b_r, buffer_size);
        self.copy_buffer(&self.b_rhs, &self.b_p_solver, buffer_size);
        self.copy_buffer(&self.b_rhs, &self.b_r0, buffer_size);

        let bg_dot_pair_r0r_rr = self.context.device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: Some("Dot Pair R0R RR Bind Group (CG)"),
                layout: &self.bgl_dot_pair_inputs,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.b_dot_result.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.b_dot_result_2.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.b_r0.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.b_r.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.b_r.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.b_r.as_entire_binding(),
                    },
                ],
            },
        );

        let mut stats = LinearSolverStats::default();
        stats.converged = false;

        {
            let mut encoder = self
                .context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CG init dot r.r"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_dot);
                pass.set_bind_group(0, &self.bg_dot_params, &[]);
                pass.set_bind_group(1, &self.bg_dot_r_r, &[]);
                pass.dispatch_workgroups(num_groups, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CG init reduce"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_init_cg_scalars);
                pass.set_bind_group(0, &self.bg_scalars, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            self.context.queue.submit(Some(encoder.finish()));
        }

        for iter in 0..max_iters {
            let mut encoder = self
                .context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CG spmv p->v"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_spmv_p_v);
                pass.set_bind_group(0, &self.bg_linear_state, &[]);
                pass.set_bind_group(1, &self.bg_linear_matrix, &[]);
                pass.dispatch_workgroups(num_groups, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CG dot p.v"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_dot);
                pass.set_bind_group(0, &self.bg_dot_params, &[]);
                pass.set_bind_group(1, &self.bg_dot_p_v, &[]);
                pass.dispatch_workgroups(num_groups, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CG reduce r0_v"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_reduce_r0_v);
                pass.set_bind_group(0, &self.bg_scalars, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CG update x,r"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_cg_update_x_r);
                pass.set_bind_group(0, &self.bg_linear_state, &[]);
                pass.set_bind_group(1, &self.bg_linear_matrix, &[]);
                pass.dispatch_workgroups(num_groups, 1, 1);
            }

            encoder.copy_buffer_to_buffer(&self.b_r, 0, &self.b_r0, 0, buffer_size);

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CG dot pair r.r"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_dot_pair);
                pass.set_bind_group(0, &self.bg_dot_params, &[]);
                pass.set_bind_group(1, &bg_dot_pair_r0r_rr, &[]);
                pass.dispatch_workgroups(num_groups, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CG reduce rho_new"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_reduce_rho_new_r_r);
                pass.set_bind_group(0, &self.bg_scalars, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CG update p"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_cg_update_p);
                pass.set_bind_group(0, &self.bg_linear_state, &[]);
                pass.set_bind_group(1, &self.bg_linear_matrix, &[]);
                pass.dispatch_workgroups(num_groups, 1, 1);
            }

            self.context.queue.submit(Some(encoder.finish()));

            let scalars = self.read_linear_scalars();
            let residual = scalars.r_r.abs().sqrt();
            stats.iterations = iter + 1;
            stats.residual = residual;

            if !residual.is_finite() {
                stats.diverged = true;
                break;
            }
            if residual <= tol {
                stats.converged = true;
                break;
            }
        }

        stats.time = start.elapsed();
        stats
    }

    fn update_linear_solver_params(&self, n: u32) -> u32 {
        let num_groups = (n + LINEAR_WORKGROUP_SIZE - 1) / LINEAR_WORKGROUP_SIZE;
        let params = SolverParams {
            n,
            num_groups,
            padding: [0; 2],
        };
        self.context
            .queue
            .write_buffer(&self.b_solver_params, 0, bytemuck::bytes_of(&params));
        num_groups
    }

    fn read_linear_scalars(&self) -> GpuLinearScalars {
        let data = pollster::block_on(self.read_buffer(&self.b_scalars, 64));
        let values: &[f32] = bytemuck::cast_slice(&data);
        GpuLinearScalars::from_slice(values)
    }
}
