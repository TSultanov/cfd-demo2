use crate::solver::gpu::linear_solver::amg::CsrMatrix;
use crate::solver::gpu::lowering::kernel_registry;
use crate::solver::gpu::modules::coupled_schur::{
    CoupledPressureSolveKind, CoupledSchurKernelIds, CoupledSchurModule,
};
use crate::solver::gpu::modules::krylov_precond::{DispatchGrids, FgmresPreconditionerModule};
use crate::solver::gpu::structs::GpuGenericCoupledSchurSetupParams;
use crate::solver::gpu::wgsl_reflect;
use crate::solver::model::KernelId;

use crate::solver::gpu::modules::resource_registry::ResourceRegistry;

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
    pub fn new(
        device: &wgpu::Device,
        num_cells: u32,
        pressure_row_offsets: &wgpu::Buffer,
        pressure_col_indices: &wgpu::Buffer,
        pressure_values: &wgpu::Buffer,
        pressure_row_offsets_host: Vec<u32>,
        pressure_col_indices_host: Vec<u32>,
        pressure_num_nonzeros: u64,
        diag_u_inv: &wgpu::Buffer,
        diag_p_inv: &wgpu::Buffer,
        precond_params: &wgpu::Buffer,
        setup_bg: wgpu::BindGroup,
        setup_pipeline: wgpu::ComputePipeline,
        setup_params: wgpu::Buffer,
        unknowns_per_cell: u32,
        p: u32,
        u_len: u32,
        u0123: [u32; 4],
        u4567: [u32; 4],
        pressure_kind: CoupledPressureSolveKind,
    ) -> Self {
        Self {
            schur: CoupledSchurModule::new(
                device,
                num_cells,
                pressure_row_offsets,
                pressure_col_indices,
                pressure_values,
                diag_u_inv,
                diag_p_inv,
                precond_params,
                pressure_kind,
                CoupledSchurKernelIds {
                    predict_and_form: KernelId::SCHUR_GENERIC_PRECOND_PREDICT_AND_FORM,
                    relax_pressure: KernelId::SCHUR_GENERIC_PRECOND_RELAX_PRESSURE,
                    correct_velocity: KernelId::SCHUR_GENERIC_PRECOND_CORRECT_VELOCITY,
                },
            ),
            setup_pipeline,
            setup_bg,
            setup_params,
            pressure_values: pressure_values.clone(),
            pressure_row_offsets: pressure_row_offsets_host,
            pressure_col_indices: pressure_col_indices_host,
            pressure_num_nonzeros,
            num_cells,
            unknowns_per_cell,
            p,
            u_len,
            u0123,
            u4567,
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
        scalar_row_offsets: &wgpu::Buffer,
        diagonal_indices: &wgpu::Buffer,
        matrix_values: &wgpu::Buffer,
        diag_u_inv: &wgpu::Buffer,
        diag_p_inv: &wgpu::Buffer,
        p_matrix_values: &wgpu::Buffer,
        setup_params: &wgpu::Buffer,
    ) -> Result<wgpu::BindGroup, String> {
        let src = kernel_registry::kernel_source_by_id(
            "",
            KernelId::GENERIC_COUPLED_SCHUR_SETUP_BUILD_DIAG_AND_PRESSURE,
        )?;
        let bgl = pipeline.get_bind_group_layout(0);

        let base_registry = ResourceRegistry::new()
            .with_buffer("scalar_row_offsets", scalar_row_offsets)
            .with_buffer("diagonal_indices", diagonal_indices)
            .with_buffer("matrix_values", matrix_values)
            .with_buffer("diag_u_inv", diag_u_inv)
            .with_buffer("diag_p_inv", diag_p_inv)
            .with_buffer("p_matrix_values", p_matrix_values)
            .with_buffer("params", setup_params);

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
                    self.schur.set_pressure_kind(CoupledPressureSolveKind::Chebyshev);
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
        output: &wgpu::Buffer,
        dispatch: DispatchGrids,
    ) {
        self.schur
            .encode_apply(device, encoder, fgmres, input, output, dispatch);
    }
}
