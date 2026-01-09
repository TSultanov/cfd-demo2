use crate::solver::gpu::bindings;
use crate::solver::gpu::linear_solver::amg::{AmgResources, CsrMatrix};
use crate::solver::gpu::linear_solver::fgmres::FgmresWorkspace;
use crate::solver::gpu::structs::PreconditionerType;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::krylov_precond::{DispatchGrids, FgmresPreconditionerModule};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompressibleKrylovPreconditionerKind {
    Identity,
    BlockJacobi,
    Amg,
}

impl CompressibleKrylovPreconditionerKind {
    pub fn select(preconditioner: PreconditionerType, force_amg: bool, enable_block: bool) -> Self {
        if preconditioner == PreconditionerType::Amg || force_amg {
            return Self::Amg;
        }
        if enable_block {
            return Self::BlockJacobi;
        }
        Self::Identity
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PackParams {
    num_cells: u32,
    component: u32,
    _pad1: u32,
    _pad2: u32,
}

pub struct CompressibleKrylovModule {
    kind: CompressibleKrylovPreconditionerKind,

    b_diag_u: wgpu::Buffer,
    b_diag_v: wgpu::Buffer,
    b_diag_p: wgpu::Buffer,

    b_block_inv: wgpu::Buffer,
    bgl_block_precond: wgpu::BindGroupLayout,
    bg_block_precond: wgpu::BindGroup,
    pipeline_build_block_inv: wgpu::ComputePipeline,
    pipeline_apply_block_precond: wgpu::ComputePipeline,

    bgl_pack: wgpu::BindGroupLayout,
    b_pack_params: [wgpu::Buffer; 4],
    bg_pack_params: [wgpu::BindGroup; 4],
    pipeline_pack_component: wgpu::ComputePipeline,
    pipeline_unpack_component: wgpu::ComputePipeline,

    amg: Option<AmgResources>,
}

impl CompressibleKrylovModule {
    pub fn create_diag_buffers(
        device: &wgpu::Device,
        n: u32,
    ) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
        let b_diag_u = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compressible FGMRES diag u"),
            size: n as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let b_diag_v = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compressible FGMRES diag v"),
            size: n as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let b_diag_p = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compressible FGMRES diag p"),
            size: n as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        (b_diag_u, b_diag_v, b_diag_p)
    }

    pub fn new(
        device: &wgpu::Device,
        fgmres: &FgmresWorkspace,
        num_cells: u32,
        kind: CompressibleKrylovPreconditionerKind,
        b_diag_u: wgpu::Buffer,
        b_diag_v: wgpu::Buffer,
        b_diag_p: wgpu::Buffer,
    ) -> Self {
        let b_block_inv = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compressible FGMRES block inv"),
            size: num_cells as u64 * 16 * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bgl_block_precond = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compressible FGMRES block precond BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bg_block_precond = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compressible FGMRES block precond BG"),
            layout: &bgl_block_precond,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: b_block_inv.as_entire_binding(),
            }],
        });

        let pipeline_layout_precond =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compressible precond pipeline layout"),
                bind_group_layouts: &[
                    fgmres.vectors_layout(),
                    fgmres.matrix_layout(),
                    &bgl_block_precond,
                    fgmres.params_layout(),
                ],
                push_constant_ranges: &[],
            });

        let shader_precond =
            bindings::compressible_precond::create_shader_module_embed_source(device);
        let make_precond_pipeline =
            |label: &str, entry: &str, layout: &wgpu::PipelineLayout| -> wgpu::ComputePipeline {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(layout),
                    module: &shader_precond,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            };
        let pipeline_build_block_inv = make_precond_pipeline(
            "Compressible Precond Build Block Inv",
            "build_block_inv",
            &pipeline_layout_precond,
        );
        let pipeline_apply_block_precond = make_precond_pipeline(
            "Compressible Precond Apply Block",
            "apply_block_precond",
            &pipeline_layout_precond,
        );

        let bgl_pack = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compressible FGMRES pack BGL"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bgl_pack_params = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compressible FGMRES pack params BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout_pack = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compressible FGMRES pack pipeline layout"),
            bind_group_layouts: &[&bgl_pack, &bgl_pack_params],
            push_constant_ranges: &[],
        });
        let shader_pack =
            bindings::compressible_amg_pack::create_shader_module_embed_source(device);
        let make_pack_pipeline = |label: &str, entry: &str| -> wgpu::ComputePipeline {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout_pack),
                module: &shader_pack,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipeline_pack_component =
            make_pack_pipeline("Compressible AMG Pack Component", "pack_component");
        let pipeline_unpack_component =
            make_pack_pipeline("Compressible AMG Unpack Component", "unpack_component");

        let mut b_pack_params_vec = Vec::with_capacity(4);
        let mut bg_pack_params_vec = Vec::with_capacity(4);
        for component in 0..4u32 {
            let pack_params = PackParams {
                num_cells,
                component,
                _pad1: 0,
                _pad2: 0,
            };
            let b_pack_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Compressible FGMRES pack params"),
                contents: bytemuck::bytes_of(&pack_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg_pack_params = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Compressible FGMRES pack params BG"),
                layout: &bgl_pack_params,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_pack_params.as_entire_binding(),
                }],
            });
            b_pack_params_vec.push(b_pack_params);
            bg_pack_params_vec.push(bg_pack_params);
        }

        let b_pack_params: [wgpu::Buffer; 4] = b_pack_params_vec.try_into().unwrap();
        let bg_pack_params: [wgpu::BindGroup; 4] = bg_pack_params_vec.try_into().unwrap();

        Self {
            kind,
            b_diag_u,
            b_diag_v,
            b_diag_p,
            b_block_inv,
            bgl_block_precond,
            bg_block_precond,
            pipeline_build_block_inv,
            pipeline_apply_block_precond,
            bgl_pack,
            b_pack_params,
            bg_pack_params,
            pipeline_pack_component,
            pipeline_unpack_component,
            amg: None,
        }
    }

    pub fn set_kind(&mut self, kind: CompressibleKrylovPreconditionerKind) {
        self.kind = kind;
    }

    pub fn kind(&self) -> CompressibleKrylovPreconditionerKind {
        self.kind
    }

    pub fn ensure_amg_resources(
        &mut self,
        device: &wgpu::Device,
        matrix: CsrMatrix,
        levels: usize,
    ) {
        if self.amg.is_some() {
            return;
        }
        self.amg = Some(AmgResources::new(device, &matrix, levels));
    }

    pub fn has_amg_resources(&self) -> bool {
        self.amg.is_some()
    }

    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        fgmres: &FgmresWorkspace,
        rhs: wgpu::BindingResource<'_>,
        cells_dispatch: (u32, u32),
    ) {
        if self.kind != CompressibleKrylovPreconditionerKind::BlockJacobi {
            return;
        }

        let vector_bg = fgmres.create_vector_bind_group(
            device,
            rhs,
            fgmres.w_buffer().as_entire_binding(),
            fgmres.temp_buffer().as_entire_binding(),
            "Compressible precond build vector BG",
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compressible Precond Build"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compressible Precond Build Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_build_block_inv);
            pass.set_bind_group(0, &vector_bg, &[]);
            pass.set_bind_group(1, fgmres.matrix_bg(), &[]);
            pass.set_bind_group(2, &self.bg_block_precond, &[]);
            pass.set_bind_group(3, fgmres.params_bg(), &[]);
            pass.dispatch_workgroups(cells_dispatch.0, cells_dispatch.1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }

    fn encode_identity(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        fgmres: &FgmresWorkspace,
        device: &wgpu::Device,
        input: wgpu::BindingResource<'_>,
        output: &wgpu::Buffer,
        dispatch: (u32, u32),
    ) {
        let vector_bg = fgmres.create_vector_bind_group(
            device,
            input,
            output.as_entire_binding(),
            fgmres.temp_buffer().as_entire_binding(),
            "Compressible precond identity vector BG",
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compressible Precond Identity"),
            timestamp_writes: None,
        });
        pass.set_pipeline(fgmres.pipeline_copy());
        pass.set_bind_group(0, &vector_bg, &[]);
        pass.set_bind_group(1, fgmres.matrix_bg(), &[]);
        pass.set_bind_group(2, fgmres.precond_bg(), &[]);
        pass.set_bind_group(3, fgmres.params_bg(), &[]);
        pass.dispatch_workgroups(dispatch.0, dispatch.1, 1);
    }

    fn create_pack_bind_group<'a>(
        &self,
        device: &wgpu::Device,
        input: wgpu::BindingResource<'a>,
        output: wgpu::BindingResource<'a>,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compressible AMG pack BG"),
            layout: &self.bgl_pack,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input,
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output,
                },
            ],
        })
    }
}

impl FgmresPreconditionerModule for CompressibleKrylovModule {
    fn encode_apply(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        fgmres: &FgmresWorkspace,
        input: wgpu::BindingResource<'_>,
        output: &wgpu::Buffer,
        dispatch: DispatchGrids,
    ) {
        match self.kind {
            CompressibleKrylovPreconditionerKind::Identity => {
                self.encode_identity(encoder, fgmres, device, input, output, dispatch.dofs);
            }
            CompressibleKrylovPreconditionerKind::BlockJacobi => {
                let vector_bg = fgmres.create_vector_bind_group(
                    device,
                    input,
                    output.as_entire_binding(),
                    fgmres.temp_buffer().as_entire_binding(),
                    "Compressible precond block vector BG",
                );
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compressible Precond Apply Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_apply_block_precond);
                pass.set_bind_group(0, &vector_bg, &[]);
                pass.set_bind_group(1, fgmres.matrix_bg(), &[]);
                pass.set_bind_group(2, &self.bg_block_precond, &[]);
                pass.set_bind_group(3, fgmres.params_bg(), &[]);
                pass.dispatch_workgroups(dispatch.cells.0, dispatch.cells.1, 1);
            }
            CompressibleKrylovPreconditionerKind::Amg => {
                let Some(amg) = &self.amg else {
                    self.encode_identity(encoder, fgmres, device, input, output, dispatch.dofs);
                    return;
                };

                encoder.clear_buffer(output, 0, None);

                let pack_bg = self.create_pack_bind_group(
                    device,
                    input,
                    amg.levels[0].b_b.as_entire_binding(),
                );
                let unpack_bg = self.create_pack_bind_group(
                    device,
                    amg.levels[0].b_x.as_entire_binding(),
                    output.as_entire_binding(),
                );

                for component in 0..4usize {
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Compressible AMG pack"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.pipeline_pack_component);
                        pass.set_bind_group(0, &pack_bg, &[]);
                        pass.set_bind_group(1, &self.bg_pack_params[component], &[]);
                        pass.dispatch_workgroups(dispatch.cells.0, dispatch.cells.1, 1);
                    }

                    encoder.clear_buffer(&amg.levels[0].b_x, 0, None);
                    amg.v_cycle(encoder, None);

                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Compressible AMG unpack"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.pipeline_unpack_component);
                        pass.set_bind_group(0, &unpack_bg, &[]);
                        pass.set_bind_group(1, &self.bg_pack_params[component], &[]);
                        pass.dispatch_workgroups(dispatch.cells.0, dispatch.cells.1, 1);
                    }
                }
            }
        }
    }
}
