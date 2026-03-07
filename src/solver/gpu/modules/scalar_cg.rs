use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::linear_solver::fgmres::{dispatch_2d, DEFAULT_WORKGROUP_SIZE};
use crate::solver::gpu::lowering::kernel_registry;
use crate::solver::gpu::modules::resource_registry::ResourceRegistry;
use crate::solver::gpu::structs::{LinearSolverStats, SolverParams};
use crate::solver::gpu::wgsl_reflect;
use crate::solver::model::KernelId;

/// Index of the `stop` field within the CG `GpuScalars` struct (byte offset = 6 * 4 = 24).
pub const CG_SCALAR_STOP: usize = 6;

/// Input resources for constructing a [`ScalarCgModule`].
pub struct ScalarCgModuleInputs<'a> {
    pub capacity: u32,
    pub b_rhs: &'a wgpu::Buffer,
    pub b_x: &'a wgpu::Buffer,
    pub b_matrix_values: &'a wgpu::Buffer,
    pub b_r: &'a wgpu::Buffer,
    pub b_r0: &'a wgpu::Buffer,
    pub b_p: &'a wgpu::Buffer,
    pub b_v: &'a wgpu::Buffer,
    pub b_dot_result: &'a wgpu::Buffer,
    pub b_dot_result_2: &'a wgpu::Buffer,
    pub b_scalars: &'a wgpu::Buffer,
    pub b_solver_params: &'a wgpu::Buffer,
    pub b_staging_scalar: &'a wgpu::Buffer,
    pub bg_linear_matrix: &'a wgpu::BindGroup,
    pub bg_linear_state: &'a wgpu::BindGroup,
    pub bg_dot_params: &'a wgpu::BindGroup,
    pub bg_dot_p_v: &'a wgpu::BindGroup,
    pub bg_dot_r_r: &'a wgpu::BindGroup,
    pub bg_scalars: &'a wgpu::BindGroup,
    pub pipeline_spmv_p_v: &'a wgpu::ComputePipeline,
    pub pipeline_dot: &'a wgpu::ComputePipeline,
    pub pipeline_dot_pair: &'a wgpu::ComputePipeline,
    pub pipeline_cg_update_x_r: &'a wgpu::ComputePipeline,
    pub pipeline_cg_update_p: &'a wgpu::ComputePipeline,
    pub pipeline_init_cg_scalars: &'a wgpu::ComputePipeline,
    pub pipeline_reduce_r0_v: &'a wgpu::ComputePipeline,
    pub pipeline_reduce_rho_new_r_r: &'a wgpu::ComputePipeline,
    pub device: &'a wgpu::Device,
}

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
    const WORKGROUP_SIZE: u32 = DEFAULT_WORKGROUP_SIZE;

    pub fn new(inputs: &ScalarCgModuleInputs<'_>) -> Self {
        let bgl_dot_pair_inputs = inputs.pipeline_dot_pair.get_bind_group_layout(1);
        let dot_pair_src = kernel_registry::kernel_source_by_id("", KernelId::DOT_PRODUCT_PAIR)
            .unwrap_or_else(|err| panic!("missing dot_product_pair kernel: {err}"));
        let registry = ResourceRegistry::new()
            .with_buffer("dot_result_a", inputs.b_dot_result)
            .with_buffer("dot_result_b", inputs.b_dot_result_2)
            .with_buffer("dot_a0", inputs.b_r0)
            .with_buffer("dot_b0", inputs.b_r)
            .with_buffer("dot_a1", inputs.b_r)
            .with_buffer("dot_b1", inputs.b_r);
        let bg_dot_pair_r0r_rr = wgsl_reflect::create_bind_group_from_bindings(
            inputs.device,
            "Dot Pair R0R RR Bind Group (CG module)",
            &bgl_dot_pair_inputs,
            dot_pair_src.bindings,
            1,
            |name| registry.resolve(name),
        )
        .unwrap_or_else(|err| panic!("CG dot pair BG creation failed: {err}"));

        Self {
            capacity: inputs.capacity,
            b_rhs: inputs.b_rhs.clone(),
            b_x: inputs.b_x.clone(),
            b_matrix_values: inputs.b_matrix_values.clone(),
            b_r: inputs.b_r.clone(),
            b_r0: inputs.b_r0.clone(),
            b_p: inputs.b_p.clone(),
            b_v: inputs.b_v.clone(),
            b_dot_result: inputs.b_dot_result.clone(),
            b_dot_result_2: inputs.b_dot_result_2.clone(),
            b_scalars: inputs.b_scalars.clone(),
            b_solver_params: inputs.b_solver_params.clone(),
            b_staging_scalar: inputs.b_staging_scalar.clone(),
            bg_linear_matrix: inputs.bg_linear_matrix.clone(),
            bg_linear_state: inputs.bg_linear_state.clone(),
            bg_dot_params: inputs.bg_dot_params.clone(),
            bg_dot_p_v: inputs.bg_dot_p_v.clone(),
            bg_dot_r_r: inputs.bg_dot_r_r.clone(),
            bg_dot_pair_r0r_rr,
            bg_scalars: inputs.bg_scalars.clone(),
            pipeline_spmv_p_v: inputs.pipeline_spmv_p_v.clone(),
            pipeline_dot: inputs.pipeline_dot.clone(),
            pipeline_dot_pair: inputs.pipeline_dot_pair.clone(),
            pipeline_cg_update_x_r: inputs.pipeline_cg_update_x_r.clone(),
            pipeline_cg_update_p: inputs.pipeline_cg_update_p.clone(),
            pipeline_init_cg_scalars: inputs.pipeline_init_cg_scalars.clone(),
            pipeline_reduce_r0_v: inputs.pipeline_reduce_r0_v.clone(),
            pipeline_reduce_rho_new_r_r: inputs.pipeline_reduce_rho_new_r_r.clone(),
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
        let (dispatch_x, dispatch_y) = dispatch_2d(num_groups);
        let buffer_size = (n as u64) * 4;

        let zeros = vec![0.0_f32; n as usize];
        context
            .queue
            .write_buffer(&self.b_x, 0, bytemuck::cast_slice(&zeros));

        self.copy_buffer(context, &self.b_rhs, &self.b_r, buffer_size);
        self.copy_buffer(context, &self.b_rhs, &self.b_p, buffer_size);
        self.copy_buffer(context, &self.b_rhs, &self.b_r0, buffer_size);

        let mut stats = LinearSolverStats {
            converged: false,
            ..LinearSolverStats::default()
        };

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
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
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
            crate::count_submission!("Scalar CG", "init");
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
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CG dot p.v (module)"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_dot);
                pass.set_bind_group(0, &self.bg_dot_params, &[]);
                pass.set_bind_group(1, &self.bg_dot_p_v, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
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
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
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
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
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
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }

            context.queue.submit(Some(encoder.finish()));
            crate::count_submission!("Scalar CG", "iteration");

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

    /// Encode a fixed-iteration CG solve into a caller-provided command encoder.
    ///
    /// Unlike [`solve`], this does **not** read back residuals between iterations
    /// and does **not** submit.  All CG iterations (init + `max_iters` iteration
    /// bodies) are recorded into `encoder`.  The caller is responsible for
    /// submitting the encoder and reading back stats via [`read_last_cg_stats`].
    ///
    /// `params_already_written` should be `false` for the first chunk of a
    /// chunked solve and `true` for subsequent chunks (params and init are only
    /// needed once).
    pub fn encode_solve_cg_fixed_iterations(
        &self,
        context: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        n: u32,
        max_iters: u32,
        include_init: bool,
        capture_scalars: bool,
    ) {
        if n > self.capacity {
            panic!(
                "requested solve size {} exceeds allocated size {}",
                n, self.capacity
            );
        }

        let num_groups = n.div_ceil(Self::WORKGROUP_SIZE);
        let (dispatch_x, dispatch_y) = dispatch_2d(num_groups);
        let buffer_size = (n as u64) * 4;

        if include_init {
            // Write solver params before encoding (constant across all iterations).
            let params = SolverParams {
                n,
                num_groups,
                padding: [0; 2],
            };
            context
                .queue
                .write_buffer(&self.b_solver_params, 0, bytemuck::bytes_of(&params));

            // Zero the solution vector.
            encoder.clear_buffer(&self.b_x, 0, None);

            // Initial copies: rhs -> r, rhs -> p, rhs -> r0
            encoder.copy_buffer_to_buffer(&self.b_rhs, 0, &self.b_r, 0, buffer_size);
            encoder.copy_buffer_to_buffer(&self.b_rhs, 0, &self.b_p, 0, buffer_size);
            encoder.copy_buffer_to_buffer(&self.b_rhs, 0, &self.b_r0, 0, buffer_size);

            // Init passes: dot(r, r) then reduce to set rho_old + clear alpha/beta/stop
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CG encode: init dot r.r"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_dot);
                pass.set_bind_group(0, &self.bg_dot_params, &[]);
                pass.set_bind_group(1, &self.bg_dot_r_r, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CG encode: init reduce"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_init_cg_scalars);
                pass.set_bind_group(0, &self.bg_scalars, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
        }

        // Unroll the iteration loop: each iteration encodes 7 compute passes + 1 buffer copy.
        for _iter in 0..max_iters {
            // 1. SpMV: v = A * p
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CG encode: spmv p->v"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_spmv_p_v);
                pass.set_bind_group(0, &self.bg_linear_state, &[]);
                pass.set_bind_group(1, &self.bg_linear_matrix, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }

            // 2. Dot: partial dot products p.v
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CG encode: dot p.v"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_dot);
                pass.set_bind_group(0, &self.bg_dot_params, &[]);
                pass.set_bind_group(1, &self.bg_dot_p_v, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }

            // 3. Reduce r0_v (= p.v)
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CG encode: reduce r0_v"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_reduce_r0_v);
                pass.set_bind_group(0, &self.bg_scalars, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }

            // 4. Update x and r: x += alpha*p, r -= alpha*v
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CG encode: update x,r"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_cg_update_x_r);
                pass.set_bind_group(0, &self.bg_linear_state, &[]);
                pass.set_bind_group(1, &self.bg_linear_matrix, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }

            // 5. Copy r -> r0
            encoder.copy_buffer_to_buffer(&self.b_r, 0, &self.b_r0, 0, buffer_size);

            // 6. Dual dot: partial dot products (r0.r, r.r)
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CG encode: dot pair r0r rr"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_dot_pair);
                pass.set_bind_group(0, &self.bg_dot_params, &[]);
                pass.set_bind_group(1, &self.bg_dot_pair_r0r_rr, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }

            // 7. Reduce rho_new and r_r
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CG encode: reduce rho_new"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_reduce_rho_new_r_r);
                pass.set_bind_group(0, &self.bg_scalars, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }

            // 8. Update p: p = r + beta*p, update rho_old = rho_new
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CG encode: update p"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_cg_update_p);
                pass.set_bind_group(0, &self.bg_linear_state, &[]);
                pass.set_bind_group(1, &self.bg_linear_matrix, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }
        }

        // Copy scalars to staging so the caller can read back the residual
        // after submission.
        if capture_scalars {
            encoder.copy_buffer_to_buffer(&self.b_scalars, 0, &self.b_staging_scalar, 0, 64);
        }
    }

    /// Read back CG solver stats from the staging buffer after submission.
    ///
    /// This mirrors [`KrylovSolveModule::read_last_solver_stats`] but reads from
    /// the CG scalars layout (`values[5]` = `r_r` = residual norm squared).
    pub fn read_last_cg_stats(
        &self,
        context: &GpuContext,
        submission_index: wgpu::SubmissionIndex,
        iterations: u32,
        time: std::time::Duration,
    ) -> LinearSolverStats {
        let slice = self.b_staging_scalar.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| { let _ = tx.send(v); });
        let _ = context.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: None,
        });
        rx.recv()
            .map_err(|e| format!("CG staging readback recv failed: {e}"))
            .and_then(|r| r.map_err(|e| format!("CG buffer mapping failed: {e:?}")))
            .expect("CG staging readback failed");
        let data = slice.get_mapped_range();
        let values: &[f32] = bytemuck::cast_slice(&data);
        let r_r = values.get(5).copied().unwrap_or(0.0);
        drop(data);
        self.b_staging_scalar.unmap();

        let residual = r_r.abs().sqrt();
        LinearSolverStats {
            iterations,
            residual,
            converged: residual.is_finite(),
            diverged: !residual.is_finite(),
            time,
            ..LinearSolverStats::default()
        }
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
        crate::count_submission!("Scalar CG", "copy_buffer");
    }

    fn read_residual(&self, context: &GpuContext) -> f32 {
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.b_scalars, 0, &self.b_staging_scalar, 0, 64);
        let submission_index = context.queue.submit(Some(encoder.finish()));
        crate::count_submission!("Scalar CG", "read_residual_copy");

        let slice = self.b_staging_scalar.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| { let _ = tx.send(v); });
        let _ = context.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: None,
        });
        rx.recv()
            .map_err(|e| format!("CG staging readback recv failed: {e}"))
            .and_then(|r| r.map_err(|e| format!("CG buffer mapping failed: {e:?}")))
            .expect("CG staging readback failed");
        let data = slice.get_mapped_range();
        let values: &[f32] = bytemuck::cast_slice(&data);
        let r_r = values.get(5).copied().unwrap_or(0.0);
        drop(data);
        self.b_staging_scalar.unmap();
        r_r
    }
}
