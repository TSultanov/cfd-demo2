use crate::solver::gpu::linear_solver::amg::CsrMatrix;
use crate::solver::gpu::lowering::kernel_registry;
use crate::solver::gpu::modules::coupled_schur::{CoupledPressureSolveKind, CoupledSchurModule};
use crate::solver::gpu::modules::krylov_precond::{DispatchGrids, FgmresPreconditionerModule};
use crate::solver::gpu::modules::resource_registry::ResourceRegistry;
use crate::solver::gpu::structs::GpuGenericCoupledSchurSetupParams;
use crate::solver::gpu::wgsl_reflect;
use crate::solver::model::KernelId;
const WORKGROUP_SIZE: u32 = 64;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub struct GenericCoupledSchurPreconditioner {
    schur: CoupledSchurModule,
    setup_pipeline: wgpu::ComputePipeline,
    setup_bg_ping_pong: Vec<wgpu::BindGroup>,
    setup_params: wgpu::Buffer,
    state_step_index: Arc<AtomicUsize>,
    state_stride: u32,
    d_p_offset: u32,
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
        fgmres: &crate::solver::gpu::linear_solver::fgmres::FgmresWorkspace,
        num_cells: u32,
        pressure_row_offsets: &wgpu::Buffer,
        pressure_col_indices: &wgpu::Buffer,
        pressure_values: &wgpu::Buffer,
        setup_bg_ping_pong: Vec<wgpu::BindGroup>,
        setup_pipeline: wgpu::ComputePipeline,
        setup_params: wgpu::Buffer,
        state_step_index: Arc<AtomicUsize>,
        state_stride: u32,
        d_p_offset: u32,
        unknowns_per_cell: u32,
        p: u32,
        u_len: u32,
        u0123: [u32; 4],
        u4567: [u32; 4],
    ) -> Self {
        Self {
            schur: CoupledSchurModule::new(
                device,
                fgmres,
                num_cells,
                pressure_row_offsets,
                pressure_col_indices,
                pressure_values,
                CoupledPressureSolveKind::Chebyshev,
                crate::solver::model::KernelId::SCHUR_GENERIC_PRECOND_PREDICT_AND_FORM,
            ),
            setup_pipeline,
            setup_bg_ping_pong,
            setup_params,
            state_step_index,
            state_stride,
            d_p_offset,
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
        state_buffers: &[wgpu::Buffer; 3],
        cell_vols: &wgpu::Buffer,
        setup_params: &wgpu::Buffer,
    ) -> Result<Vec<wgpu::BindGroup>, String> {
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
            .with_buffer("cell_vols", cell_vols)
            .with_buffer("params", setup_params);

        let mut out = Vec::with_capacity(3);
        for phase in 0..3usize {
            let (idx_state, _, _) = crate::solver::gpu::modules::state::ping_pong_indices(phase);
            let state = &state_buffers[idx_state];
            let registry = base_registry.clone().with_buffer("state", state);

            let label = format!("Generic Coupled Schur Setup BG (phase {phase})");
            let bg = wgsl_reflect::create_bind_group_from_bindings(
                device,
                &label,
                &bgl,
                src.bindings,
                0,
                |name| registry.resolve(name),
            )?;
            out.push(bg);
        }

        Ok(out)
    }

    pub fn set_pressure_kind(&mut self, kind: CoupledPressureSolveKind) {
        self.schur.set_pressure_kind(kind);
    }

    pub fn ensure_amg_resources(&mut self, device: &wgpu::Device, matrix: CsrMatrix) {
        self.schur.ensure_amg_resources(device, matrix);
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
            dispatch_x: dispatch.cells.0 * WORKGROUP_SIZE,
            num_cells: self.num_cells,
            unknowns_per_cell: self.unknowns_per_cell,
            p: self.p,
            u_len: self.u_len,
            state_stride: self.state_stride,
            d_p_offset: self.d_p_offset,
            _pad0: 0,
            u0123: self.u0123,
            u4567: self.u4567,
        };
        queue.write_buffer(&self.setup_params, 0, bytemuck::bytes_of(&params));

        let phase = self.state_step_index.load(Ordering::Relaxed) % 3;
        let setup_bg = self
            .setup_bg_ping_pong
            .get(phase)
            .unwrap_or_else(|| panic!("missing schur setup bind group for phase {phase}"));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Generic Coupled Schur Setup"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Generic Coupled Schur Setup"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.setup_pipeline);
            pass.set_bind_group(0, setup_bg, &[]);
            pass.dispatch_workgroups(dispatch.cells.0, dispatch.cells.1, 1);
        }
        queue.submit(Some(encoder.finish()));
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
