use crate::solver::gpu::linear_solver::amg::CsrMatrix;
use crate::solver::gpu::modules::coupled_schur::{CoupledPressureSolveKind, CoupledSchurModule};
use crate::solver::gpu::modules::krylov_precond::{DispatchGrids, FgmresPreconditionerModule};
const WORKGROUP_SIZE: u32 = 64;
const SCHUR_SETUP_WGSL: &str = include_str!("../shaders/generic_coupled_schur_setup.wgsl");

use crate::solver::gpu::bindings::generic_coupled_schur_setup::SetupParams;

pub struct GenericCoupledSchurPreconditioner {
    schur: CoupledSchurModule,
    setup_pipeline: wgpu::ComputePipeline,
    setup_bg: wgpu::BindGroup,
    setup_params: wgpu::Buffer,
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
        setup_bg: wgpu::BindGroup,
        setup_pipeline: wgpu::ComputePipeline,
        setup_params: wgpu::Buffer,
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
            setup_bg,
            setup_params,
            num_cells,
            unknowns_per_cell,
            p,
            u_len,
            u0123,
            u4567,
        }
    }

    pub fn build_setup_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Generic Coupled Schur Setup"),
            source: wgpu::ShaderSource::Wgsl(SCHUR_SETUP_WGSL.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Generic Coupled Schur Setup BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Generic Coupled Schur Setup Layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Generic Coupled Schur Setup"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("build_diag_and_pressure"),
            compilation_options: Default::default(),
            cache: None,
        })
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
    ) -> wgpu::BindGroup {
        let bgl = pipeline.get_bind_group_layout(0);
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Generic Coupled Schur Setup BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scalar_row_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: diagonal_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: matrix_values.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: diag_u_inv.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: diag_p_inv.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: p_matrix_values.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: setup_params.as_entire_binding(),
                },
            ],
        })
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
        let params = SetupParams::new(
            dispatch.cells.0 * WORKGROUP_SIZE,
            self.num_cells,
            self.unknowns_per_cell,
            self.p,
            self.u_len,
            0,
            0,
            self.u0123,
            self.u4567,
        );
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
