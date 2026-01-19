use crate::solver::gpu::linear_solver::amg::{AmgResources, CsrMatrix};
use crate::solver::gpu::lowering::kernel_registry;
use crate::solver::gpu::modules::generic_linear_solver::IdentityPreconditioner;
use crate::solver::gpu::modules::krylov_precond::{DispatchGrids, FgmresPreconditionerModule};
use crate::solver::gpu::structs::PreconditionerType;
use crate::solver::model::KernelId;

pub(crate) struct RuntimePreconditionerModule {
    kind: PreconditionerType,
    identity: IdentityPreconditioner,
    amg: Option<AmgResources>,
    amg_init_failed: bool,
    pipeline_extract_diag_inv: Option<wgpu::ComputePipeline>,
    pipeline_apply_diag_inv: Option<wgpu::ComputePipeline>,
    b_rhs: wgpu::Buffer,
    num_dofs: u32,
    num_nonzeros: u32,
    row_offsets: Vec<u32>,
    col_indices: Vec<u32>,
    matrix_values: wgpu::Buffer,
}

impl RuntimePreconditionerModule {
    pub(crate) fn new(
        device: &wgpu::Device,
        kind: PreconditionerType,
        num_dofs: u32,
        num_nonzeros: u32,
        row_offsets: Vec<u32>,
        col_indices: Vec<u32>,
        matrix_values: wgpu::Buffer,
    ) -> Self {
        let b_rhs = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runtime_preconditioner:rhs"),
            size: (num_dofs as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            kind,
            identity: IdentityPreconditioner::new(),
            amg: None,
            amg_init_failed: false,
            pipeline_extract_diag_inv: None,
            pipeline_apply_diag_inv: None,
            b_rhs,
            num_dofs,
            num_nonzeros,
            row_offsets,
            col_indices,
            matrix_values,
        }
    }

    pub(crate) fn set_kind(&mut self, kind: PreconditionerType) {
        self.kind = kind;
    }

    fn ensure_jacobi_pipelines(&mut self, device: &wgpu::Device) {
        const EXTRACT_DIAG_INV: KernelId = KernelId("gmres_ops/extract_diag_inv");
        const APPLY_DIAG_INV: KernelId = KernelId("gmres_ops/apply_diag_inv");

        if self.pipeline_extract_diag_inv.is_none() {
            let src = kernel_registry::kernel_source_by_id("", EXTRACT_DIAG_INV)
                .expect("gmres_ops/extract_diag_inv shader missing from kernel registry");
            self.pipeline_extract_diag_inv = Some((src.create_pipeline)(device));
        }
        if self.pipeline_apply_diag_inv.is_none() {
            let src = kernel_registry::kernel_source_by_id("", APPLY_DIAG_INV)
                .expect("gmres_ops/apply_diag_inv shader missing from kernel registry");
            self.pipeline_apply_diag_inv = Some((src.create_pipeline)(device));
        }
    }

    fn refresh_jacobi_diag_inv(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        fgmres: &crate::solver::gpu::linear_solver::fgmres::FgmresWorkspace,
        dispatch: DispatchGrids,
    ) {
        self.ensure_jacobi_pipelines(device);

        let Some(pipeline) = self.pipeline_extract_diag_inv.as_ref() else {
            return;
        };

        let vector_bg = fgmres.create_vector_bind_group(
            device,
            fgmres.w_buffer().as_entire_binding(),
            fgmres.temp_buffer().as_entire_binding(),
            fgmres.z_buffer(0).as_entire_binding(),
            "runtime_preconditioner:jacobi_diag_inv_vectors",
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("runtime_preconditioner:jacobi_diag_inv"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runtime_preconditioner:jacobi_diag_inv"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &vector_bg, &[]);
            pass.set_bind_group(1, fgmres.matrix_bg(), &[]);
            pass.set_bind_group(2, fgmres.precond_bg(), &[]);
            pass.set_bind_group(3, fgmres.params_bg(), &[]);
            pass.dispatch_workgroups(dispatch.dofs.0, dispatch.dofs.1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }

    fn read_matrix_values(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<f32>, String> {
        let size = (self.num_nonzeros as u64) * 4;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runtime_preconditioner:matrix_readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("runtime_preconditioner:matrix_readback_copy"),
        });
        encoder.copy_buffer_to_buffer(&self.matrix_values, 0, &staging, 0, size);
        let submission_index = queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: None,
        });
        rx.recv()
            .map_err(|err| format!("matrix readback channel recv failed: {err}"))?
            .map_err(|err| format!("matrix readback failed: {err}"))?;

        let data = slice.get_mapped_range();
        let values = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Ok(values)
    }

    fn ensure_amg(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.kind != PreconditionerType::Amg {
            return;
        }
        if self.amg.is_some() || self.amg_init_failed {
            return;
        }

        let values = match self.read_matrix_values(device, queue) {
            Ok(values) => values,
            Err(_err) => {
                self.amg_init_failed = true;
                return;
            }
        };

        let matrix = CsrMatrix {
            row_offsets: self.row_offsets.clone(),
            col_indices: self.col_indices.clone(),
            values,
            num_rows: self.num_dofs as usize,
            num_cols: self.num_dofs as usize,
        };

        self.amg = Some(AmgResources::new(device, &matrix, 20));
    }

    fn refresh_amg_level0_matrix(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let Some(amg) = &self.amg else {
            return;
        };
        let Some(level0) = amg.levels.first() else {
            return;
        };

        let size = (self.num_nonzeros as u64) * 4;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("runtime_preconditioner:amg_level0_refresh"),
        });
        encoder.copy_buffer_to_buffer(&self.matrix_values, 0, &level0.b_matrix_values, 0, size);
        queue.submit(Some(encoder.finish()));
    }
}

impl FgmresPreconditionerModule for RuntimePreconditionerModule {
    fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        fgmres: &crate::solver::gpu::linear_solver::fgmres::FgmresWorkspace,
        _rhs: wgpu::BindingResource<'_>,
        _dispatch: DispatchGrids,
    ) {
        if self.kind == PreconditionerType::Jacobi {
            self.refresh_jacobi_diag_inv(device, queue, fgmres, _dispatch);
            return;
        }

        self.ensure_amg(device, queue);
        if self.kind == PreconditionerType::Amg {
            self.refresh_amg_level0_matrix(device, queue);
        }
    }

    fn encode_apply(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        fgmres: &crate::solver::gpu::linear_solver::fgmres::FgmresWorkspace,
        input: wgpu::BindingResource<'_>,
        output: &wgpu::Buffer,
        dispatch: DispatchGrids,
    ) {
        if self.kind == PreconditionerType::Jacobi {
            self.ensure_jacobi_pipelines(device);
            let Some(pipeline) = self.pipeline_apply_diag_inv.as_ref() else {
                return self
                    .identity
                    .encode_apply(device, encoder, fgmres, input, output, dispatch);
            };

            let vector_bg = fgmres.create_vector_bind_group(
                device,
                input,
                output.as_entire_binding(),
                output.as_entire_binding(),
                "runtime_preconditioner:jacobi_apply_vectors",
            );

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runtime_preconditioner:jacobi_apply"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &vector_bg, &[]);
            pass.set_bind_group(1, fgmres.matrix_bg(), &[]);
            pass.set_bind_group(2, fgmres.precond_bg(), &[]);
            pass.set_bind_group(3, fgmres.params_bg(), &[]);
            pass.dispatch_workgroups(dispatch.dofs.0, dispatch.dofs.1, 1);
            return;
        }

        if self.kind != PreconditionerType::Amg {
            return self
                .identity
                .encode_apply(device, encoder, fgmres, input, output, dispatch);
        }

        let Some(amg) = &self.amg else {
            return self
                .identity
                .encode_apply(device, encoder, fgmres, input, output, dispatch);
        };

        let wgpu::BindingResource::Buffer(input_binding) = &input else {
            return self
                .identity
                .encode_apply(device, encoder, fgmres, input, output, dispatch);
        };

        let expected_bytes = (self.num_dofs as u64) * 4;
        let available = input_binding.size.map(|s| s.get()).unwrap_or(expected_bytes);
        if available < expected_bytes {
            return self
                .identity
                .encode_apply(device, encoder, fgmres, input, output, dispatch);
        }

        encoder.copy_buffer_to_buffer(
            input_binding.buffer,
            input_binding.offset,
            &self.b_rhs,
            0,
            expected_bytes,
        );
        encoder.clear_buffer(output, 0, Some(expected_bytes));

        let Some(level0) = amg.levels.first() else {
            return self
                .identity
                .encode_apply(device, encoder, fgmres, input, output, dispatch);
        };

        let override_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("runtime_preconditioner:amg_level0_state_override"),
            layout: &amg.bgl_state,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.b_rhs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: level0.b_params.as_entire_binding(),
                },
            ],
        });

        amg.v_cycle(encoder, Some(&override_bg));
    }
}
