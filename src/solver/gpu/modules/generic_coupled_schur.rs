use crate::solver::gpu::linear_solver::amg::CsrMatrix;
use crate::solver::gpu::lowering::kernel_registry;
use crate::solver::gpu::modules::coupled_schur::{
    CoupledPressureSolveKind, CoupledSchurInputs, CoupledSchurKernelIds, CoupledSchurModule,
};
use crate::solver::gpu::modules::krylov_precond::{DispatchGrids, FgmresPreconditionerModule};
use crate::solver::gpu::structs::GpuGenericCoupledSchurSetupParams;
use crate::solver::gpu::wgsl_reflect;
use crate::solver::model::KernelId;

use crate::solver::gpu::modules::resource_registry::ResourceRegistry;

/// Input struct for [`GenericCoupledSchurPreconditioner::new`] to reduce argument count.
pub struct GenericCoupledSchurPreconditionerInputs<'a> {
    pub num_cells: u32,
    pub pressure_row_offsets: &'a wgpu::Buffer,
    pub pressure_col_indices: &'a wgpu::Buffer,
    pub pressure_values: &'a wgpu::Buffer,
    pub pressure_row_offsets_host: Vec<u32>,
    pub pressure_col_indices_host: Vec<u32>,
    pub pressure_num_nonzeros: u64,
    pub diag_u_inv: &'a wgpu::Buffer,
    pub diag_p_inv: &'a wgpu::Buffer,
    pub precond_params: &'a wgpu::Buffer,
    pub setup_bg: wgpu::BindGroup,
    pub setup_pipeline: wgpu::ComputePipeline,
    pub setup_params: wgpu::Buffer,
    pub unknowns_per_cell: u32,
    pub p: u32,
    pub u_len: u32,
    pub u0123: [u32; 4],
    pub u4567: [u32; 4],
    pub pressure_kind: CoupledPressureSolveKind,
}

/// Input struct for [`GenericCoupledSchurPreconditioner::build_setup_bind_group`].
pub struct GenericCoupledSchurSetupBindGroupInputs<'a> {
    pub scalar_row_offsets: &'a wgpu::Buffer,
    pub diagonal_indices: &'a wgpu::Buffer,
    pub matrix_values: &'a wgpu::Buffer,
    pub diag_u_inv: &'a wgpu::Buffer,
    pub diag_p_inv: &'a wgpu::Buffer,
    pub p_matrix_values: &'a wgpu::Buffer,
    pub setup_params: &'a wgpu::Buffer,
}

pub struct GenericCoupledSchurPreconditioner {
    schur: CoupledSchurModule,
    setup_pipeline: wgpu::ComputePipeline,
    setup_bg: wgpu::BindGroup,
    setup_params: wgpu::Buffer,
    pressure_values: wgpu::Buffer,
    pressure_row_offsets: Vec<u32>,
    pressure_col_indices: Vec<u32>,
    pressure_num_nonzeros: u64,
    num_cells: u32,
    unknowns_per_cell: u32,
    p: u32,
    u_len: u32,
    u0123: [u32; 4],
    u4567: [u32; 4],
}

impl GenericCoupledSchurPreconditioner {
    pub fn new(device: &wgpu::Device, inputs: GenericCoupledSchurPreconditionerInputs<'_>) -> Self {
        Self {
            schur: CoupledSchurModule::new(
                device,
                CoupledSchurInputs {
                    num_cells: inputs.num_cells,
                    pressure_row_offsets: inputs.pressure_row_offsets,
                    pressure_col_indices: inputs.pressure_col_indices,
                    pressure_values: inputs.pressure_values,
                    diag_u_inv: inputs.diag_u_inv,
                    diag_p_inv: inputs.diag_p_inv,
                    precond_params: inputs.precond_params,
                    pressure_kind: inputs.pressure_kind,
                    kernels: CoupledSchurKernelIds {
                        predict_and_form: KernelId::SCHUR_GENERIC_PRECOND_PREDICT_AND_FORM,
                        relax_pressure: KernelId::SCHUR_GENERIC_PRECOND_RELAX_PRESSURE,
                        correct_velocity: KernelId::SCHUR_GENERIC_PRECOND_CORRECT_VELOCITY,
                    },
                },
            ),
            setup_pipeline: inputs.setup_pipeline,
            setup_bg: inputs.setup_bg,
            setup_params: inputs.setup_params,
            pressure_values: inputs.pressure_values.clone(),
            pressure_row_offsets: inputs.pressure_row_offsets_host,
            pressure_col_indices: inputs.pressure_col_indices_host,
            pressure_num_nonzeros: inputs.pressure_num_nonzeros,
            num_cells: inputs.num_cells,
            unknowns_per_cell: inputs.unknowns_per_cell,
            p: inputs.p,
            u_len: inputs.u_len,
            u0123: inputs.u0123,
            u4567: inputs.u4567,
        }
    }

    pub fn build_setup_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
        let src = kernel_registry::kernel_source_by_id(
            "",
            KernelId::GENERIC_COUPLED_SCHUR_SETUP_BUILD_DIAG_AND_PRESSURE,
        )
        .expect("generic_coupled_schur_setup shader missing from kernel registry");
        (src.create_pipeline)(device)
    }

    pub fn build_setup_bind_group(
        device: &wgpu::Device,
        pipeline: &wgpu::ComputePipeline,
        inputs: GenericCoupledSchurSetupBindGroupInputs<'_>,
    ) -> Result<wgpu::BindGroup, String> {
        let src = kernel_registry::kernel_source_by_id(
            "",
            KernelId::GENERIC_COUPLED_SCHUR_SETUP_BUILD_DIAG_AND_PRESSURE,
        )?;
        let bgl = pipeline.get_bind_group_layout(0);

        let base_registry = ResourceRegistry::new()
            .with_buffer("scalar_row_offsets", inputs.scalar_row_offsets)
            .with_buffer("diagonal_indices", inputs.diagonal_indices)
            .with_buffer("matrix_values", inputs.matrix_values)
            .with_buffer("diag_u_inv", inputs.diag_u_inv)
            .with_buffer("diag_p_inv", inputs.diag_p_inv)
            .with_buffer("p_matrix_values", inputs.p_matrix_values)
            .with_buffer("params", inputs.setup_params);

        wgsl_reflect::create_bind_group_from_bindings(
            device,
            "Generic Coupled Schur Setup BG",
            &bgl,
            src.bindings,
            0,
            |name| base_registry.resolve(name),
        )
    }

    pub fn set_pressure_kind(&mut self, kind: CoupledPressureSolveKind) {
        self.schur.set_pressure_kind(kind);
    }

    pub fn ensure_amg_resources(&mut self, device: &wgpu::Device, matrix: CsrMatrix) {
        self.schur.ensure_amg_resources(device, matrix);
    }

    fn read_pressure_values(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<f32>, String> {
        let size = self.pressure_num_nonzeros * 4;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("generic_coupled_schur:pressure_matrix_readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("generic_coupled_schur:pressure_matrix_readback_copy"),
        });
        encoder.copy_buffer_to_buffer(&self.pressure_values, 0, &staging, 0, size);
        let submission_index = queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: None,
        });
        rx.recv()
            .map_err(|err| format!("pressure matrix readback channel recv failed: {err}"))?
            .map_err(|err| format!("pressure matrix readback failed: {err}"))?;

        let data = slice.get_mapped_range();
        let values = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Ok(values)
    }
}

impl FgmresPreconditionerModule for GenericCoupledSchurPreconditioner {
    fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _fgmres: &crate::solver::gpu::linear_solver::fgmres::FgmresWorkspace,
        _rhs: wgpu::BindingResource<'_>,
        dispatch: DispatchGrids,
    ) {
        let params = GpuGenericCoupledSchurSetupParams {
            num_cells: self.num_cells,
            unknowns_per_cell: self.unknowns_per_cell,
            p: self.p,
            u_len: self.u_len,
            u0123: self.u0123,
            u4567: self.u4567,
        };
        queue.write_buffer(&self.setup_params, 0, bytemuck::bytes_of(&params));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Generic Coupled Schur Setup"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Generic Coupled Schur Setup"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.setup_pipeline);
            pass.set_bind_group(0, &self.setup_bg, &[]);
            pass.dispatch_workgroups(dispatch.cells.0, dispatch.cells.1, 1);
        }
        queue.submit(Some(encoder.finish()));

        if self.schur.pressure_kind() == CoupledPressureSolveKind::Amg {
            if !self.schur.has_amg_resources() {
                if let Ok(values) = self.read_pressure_values(device, queue) {
                    self.schur.ensure_amg_resources(
                        device,
                        CsrMatrix {
                            row_offsets: self.pressure_row_offsets.clone(),
                            col_indices: self.pressure_col_indices.clone(),
                            values,
                            num_rows: self.num_cells as usize,
                            num_cols: self.num_cells as usize,
                        },
                    );
                } else {
                    self.schur
                        .set_pressure_kind(CoupledPressureSolveKind::Chebyshev);
                }
            } else {
                self.schur.refresh_amg_level0_matrix(
                    device,
                    queue,
                    &self.pressure_values,
                    self.pressure_num_nonzeros,
                );
            }
        }
    }

    fn encode_apply(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        fgmres: &crate::solver::gpu::linear_solver::fgmres::FgmresWorkspace,
        input: wgpu::BindingResource<'_>,
        output: wgpu::BindingResource<'_>,
        dispatch: DispatchGrids,
    ) {
        self.schur
            .encode_apply(device, encoder, fgmres, input, output, dispatch);
    }
}
