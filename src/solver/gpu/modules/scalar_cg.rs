use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::structs::{LinearSolverStats, SolverParams};

pub struct ScalarCgModule {
    capacity: u32,

    // External IO buffers (part of the linear system)
    b_rhs: wgpu::Buffer,
    b_x: wgpu::Buffer,
    b_matrix_values: wgpu::Buffer,

    // Internal workspace buffers
    b_r: wgpu::Buffer,
    b_r0: wgpu::Buffer,
    b_p: wgpu::Buffer,
    b_v: wgpu::Buffer,
    b_s: wgpu::Buffer,
    b_t: wgpu::Buffer,

    b_dot_result: wgpu::Buffer,
    b_dot_result_2: wgpu::Buffer,
    b_scalars: wgpu::Buffer,
    b_solver_params: wgpu::Buffer,
    b_staging_scalar: wgpu::Buffer,

    bg_linear_matrix: wgpu::BindGroup,
    bg_linear_state: wgpu::BindGroup,
    bg_dot_params: wgpu::BindGroup,
    bg_dot_p_v: wgpu::BindGroup,
    bg_dot_r_r: wgpu::BindGroup,
    bg_dot_pair_r0r_rr: wgpu::BindGroup,
    bg_scalars: wgpu::BindGroup,

    bgl_dot_pair_inputs: wgpu::BindGroupLayout,

    pipeline_spmv_p_v: wgpu::ComputePipeline,
    pipeline_dot: wgpu::ComputePipeline,
    pipeline_dot_pair: wgpu::ComputePipeline,
    pipeline_cg_update_x_r: wgpu::ComputePipeline,
    pipeline_cg_update_p: wgpu::ComputePipeline,

    pipeline_init_cg_scalars: wgpu::ComputePipeline,
    pipeline_reduce_r0_v: wgpu::ComputePipeline,
    pipeline_reduce_rho_new_r_r: wgpu::ComputePipeline,
}

impl ScalarCgModule {
    const WORKGROUP_SIZE: u32 = 64;

    pub fn new(
        capacity: u32,
        b_rhs: &wgpu::Buffer,
        b_x: &wgpu::Buffer,
        b_matrix_values: &wgpu::Buffer,
        b_r: &wgpu::Buffer,
        b_r0: &wgpu::Buffer,
        b_p: &wgpu::Buffer,
        b_v: &wgpu::Buffer,
        b_s: &wgpu::Buffer,
        b_t: &wgpu::Buffer,
        b_dot_result: &wgpu::Buffer,
        b_dot_result_2: &wgpu::Buffer,
        b_scalars: &wgpu::Buffer,
        b_solver_params: &wgpu::Buffer,
        b_staging_scalar: &wgpu::Buffer,
        bg_linear_matrix: &wgpu::BindGroup,
        bg_linear_state: &wgpu::BindGroup,
        bg_dot_params: &wgpu::BindGroup,
        bg_dot_p_v: &wgpu::BindGroup,
        bg_dot_r_r: &wgpu::BindGroup,
        bg_scalars: &wgpu::BindGroup,
        bgl_dot_pair_inputs: &wgpu::BindGroupLayout,
        pipeline_spmv_p_v: &wgpu::ComputePipeline,
        pipeline_dot: &wgpu::ComputePipeline,
        pipeline_dot_pair: &wgpu::ComputePipeline,
        pipeline_cg_update_x_r: &wgpu::ComputePipeline,
        pipeline_cg_update_p: &wgpu::ComputePipeline,
        pipeline_init_cg_scalars: &wgpu::ComputePipeline,
        pipeline_reduce_r0_v: &wgpu::ComputePipeline,
        pipeline_reduce_rho_new_r_r: &wgpu::ComputePipeline,
        device: &wgpu::Device,
    ) -> Self {
        let bg_dot_pair_r0r_rr = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dot Pair R0R RR Bind Group (CG module)"),
            layout: bgl_dot_pair_inputs,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_dot_result.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_dot_result_2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_r0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_r.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: b_r.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: b_r.as_entire_binding(),
                },
            ],
        });

        Self {
            capacity,
            b_rhs: b_rhs.clone(),
            b_x: b_x.clone(),
            b_matrix_values: b_matrix_values.clone(),
            b_r: b_r.clone(),
            b_r0: b_r0.clone(),
            b_p: b_p.clone(),
            b_v: b_v.clone(),
            b_s: b_s.clone(),
            b_t: b_t.clone(),
            b_dot_result: b_dot_result.clone(),
            b_dot_result_2: b_dot_result_2.clone(),
            b_scalars: b_scalars.clone(),
            b_solver_params: b_solver_params.clone(),
            b_staging_scalar: b_staging_scalar.clone(),
            bg_linear_matrix: bg_linear_matrix.clone(),
            bg_linear_state: bg_linear_state.clone(),
            bg_dot_params: bg_dot_params.clone(),
            bg_dot_p_v: bg_dot_p_v.clone(),
            bg_dot_r_r: bg_dot_r_r.clone(),
            bg_dot_pair_r0r_rr,
            bg_scalars: bg_scalars.clone(),
            bgl_dot_pair_inputs: bgl_dot_pair_inputs.clone(),
            pipeline_spmv_p_v: pipeline_spmv_p_v.clone(),
            pipeline_dot: pipeline_dot.clone(),
            pipeline_dot_pair: pipeline_dot_pair.clone(),
            pipeline_cg_update_x_r: pipeline_cg_update_x_r.clone(),
            pipeline_cg_update_p: pipeline_cg_update_p.clone(),
            pipeline_init_cg_scalars: pipeline_init_cg_scalars.clone(),
            pipeline_reduce_r0_v: pipeline_reduce_r0_v.clone(),
            pipeline_reduce_rho_new_r_r: pipeline_reduce_rho_new_r_r.clone(),
        }
    }

    pub fn matrix_values(&self) -> &wgpu::Buffer {
        &self.b_matrix_values
    }

    pub fn rhs(&self) -> &wgpu::Buffer {
        &self.b_rhs
    }

    pub fn x(&self) -> &wgpu::Buffer {
        &self.b_x
    }

    pub fn r(&self) -> &wgpu::Buffer {
        &self.b_r
    }

    pub fn r0(&self) -> &wgpu::Buffer {
        &self.b_r0
    }

    pub fn p(&self) -> &wgpu::Buffer {
        &self.b_p
    }

    pub fn v(&self) -> &wgpu::Buffer {
        &self.b_v
    }

    pub fn s(&self) -> &wgpu::Buffer {
        &self.b_s
    }

    pub fn t(&self) -> &wgpu::Buffer {
        &self.b_t
    }

    pub fn dot_result(&self) -> &wgpu::Buffer {
        &self.b_dot_result
    }

    pub fn dot_result_2(&self) -> &wgpu::Buffer {
        &self.b_dot_result_2
    }

    pub fn scalars(&self) -> &wgpu::Buffer {
        &self.b_scalars
    }

    pub fn solver_params(&self) -> &wgpu::Buffer {
        &self.b_solver_params
    }

    pub fn staging_scalar(&self) -> &wgpu::Buffer {
        &self.b_staging_scalar
    }

    pub fn solve(
        &self,
        context: &GpuContext,
        n: u32,
        max_iters: u32,
        tol: f32,
    ) -> LinearSolverStats {
        if n > self.capacity {
            panic!(
                "requested solve size {} exceeds allocated size {}",
                n, self.capacity
            );
        }

        let start = std::time::Instant::now();
        let num_groups = self.update_params(context, n);
        let buffer_size = (n as u64) * 4;

        let zeros = vec![0.0_f32; n as usize];
        context
            .queue
            .write_buffer(&self.b_x, 0, bytemuck::cast_slice(&zeros));

        self.copy_buffer(context, &self.b_rhs, &self.b_r, buffer_size);
        self.copy_buffer(context, &self.b_rhs, &self.b_p, buffer_size);
        self.copy_buffer(context, &self.b_rhs, &self.b_r0, buffer_size);

        let mut stats = LinearSolverStats::default();
        stats.converged = false;

        {
            let mut encoder = context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CG init dot r.r (module)"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_dot);
                pass.set_bind_group(0, &self.bg_dot_params, &[]);
                pass.set_bind_group(1, &self.bg_dot_r_r, &[]);
                pass.dispatch_workgroups(num_groups, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CG init reduce (module)"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_init_cg_scalars);
                pass.set_bind_group(0, &self.bg_scalars, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            context.queue.submit(Some(encoder.finish()));
        }

        for iter in 0..max_iters {
            let mut encoder = context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CG spmv p->v (module)"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_spmv_p_v);
                pass.set_bind_group(0, &self.bg_linear_state, &[]);
                pass.set_bind_group(1, &self.bg_linear_matrix, &[]);
                pass.dispatch_workgroups(num_groups, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CG dot p.v (module)"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_dot);
                pass.set_bind_group(0, &self.bg_dot_params, &[]);
                pass.set_bind_group(1, &self.bg_dot_p_v, &[]);
                pass.dispatch_workgroups(num_groups, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CG reduce r0_v (module)"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_reduce_r0_v);
                pass.set_bind_group(0, &self.bg_scalars, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CG update x,r (module)"),
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
                    label: Some("CG dot pair r0r rr (module)"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_dot_pair);
                pass.set_bind_group(0, &self.bg_dot_params, &[]);
                pass.set_bind_group(1, &self.bg_dot_pair_r0r_rr, &[]);
                pass.dispatch_workgroups(num_groups, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CG reduce rho_new (module)"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_reduce_rho_new_r_r);
                pass.set_bind_group(0, &self.bg_scalars, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CG update p (module)"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_cg_update_p);
                pass.set_bind_group(0, &self.bg_linear_state, &[]);
                pass.set_bind_group(1, &self.bg_linear_matrix, &[]);
                pass.dispatch_workgroups(num_groups, 1, 1);
            }

            context.queue.submit(Some(encoder.finish()));

            let residual = self.read_residual(context).abs().sqrt();
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

    fn update_params(&self, context: &GpuContext, n: u32) -> u32 {
        let num_groups = n.div_ceil(Self::WORKGROUP_SIZE);
        let params = SolverParams {
            n,
            num_groups,
            padding: [0; 2],
        };
        context
            .queue
            .write_buffer(&self.b_solver_params, 0, bytemuck::bytes_of(&params));
        num_groups
    }

    fn copy_buffer(&self, context: &GpuContext, src: &wgpu::Buffer, dst: &wgpu::Buffer, size: u64) {
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(src, 0, dst, 0, size);
        context.queue.submit(Some(encoder.finish()));
    }

    fn read_residual(&self, context: &GpuContext) -> f32 {
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.b_scalars, 0, &self.b_staging_scalar, 0, 64);
        let submission_index = context.queue.submit(Some(encoder.finish()));

        let slice = self.b_staging_scalar.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        let _ = context.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: None,
        });
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let values: &[f32] = bytemuck::cast_slice(&data);
        let r_r = values.get(8).copied().unwrap_or(0.0);
        drop(data);
        self.b_staging_scalar.unmap();
        r_r
    }
}
