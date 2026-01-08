use crate::solver::gpu::bindings;
use super::compressible::CompressiblePlanResources;
use crate::solver::gpu::linear_solver::amg::{AmgResources, CsrMatrix};
use crate::solver::gpu::linear_solver::fgmres::{
    fgmres_solve_once_with_preconditioner, write_params, write_scalars, write_zeros,
    FgmresPrecondBindings, FgmresSolveOnceConfig, FgmresWorkspace, IterParams, RawFgmresParams,
};
use crate::solver::gpu::preconditioners::CompressibleKrylovPreconditioner;
use crate::solver::gpu::structs::LinearSolverStats;
use bytemuck::{bytes_of, Pod, Zeroable};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

pub struct CompressibleFgmresResources {
    pub fgmres: FgmresWorkspace,
    pub b_diag_u: wgpu::Buffer,
    pub b_diag_v: wgpu::Buffer,
    pub b_diag_p: wgpu::Buffer,
    pub b_block_inv: wgpu::Buffer,
    pub b_pack_params: wgpu::Buffer,
    pub bgl_block_precond: wgpu::BindGroupLayout,
    pub bgl_pack: wgpu::BindGroupLayout,
    pub bgl_pack_params: wgpu::BindGroupLayout,
    pub bg_block_precond: wgpu::BindGroup,
    pub bg_pack_params: wgpu::BindGroup,
    pub pipeline_build_block_inv: wgpu::ComputePipeline,
    pub pipeline_apply_block_precond: wgpu::ComputePipeline,
    pub pipeline_pack_component: wgpu::ComputePipeline,
    pub pipeline_unpack_component: wgpu::ComputePipeline,
    pub amg_resources: Option<AmgResources>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PackParams {
    num_cells: u32,
    component: u32,
    _pad1: u32,
    _pad2: u32,
}

impl CompressiblePlanResources {
    const MAX_WORKGROUPS_PER_DIMENSION: u32 = 65535;
    const WORKGROUP_SIZE: u32 = 64;

    pub(crate) fn ensure_fgmres_resources(&mut self, max_restart: usize) {
        let n = self.num_unknowns;
        let rebuild = match &self.fgmres_resources {
            Some(existing) => {
                existing.fgmres.max_restart() < max_restart || existing.fgmres.n() != n
            }
            None => true,
        };
        if rebuild {
            let resources = self.init_fgmres_resources(max_restart);
            self.fgmres_resources = Some(resources);
        }
    }

    pub async fn ensure_amg_resources(&mut self) {
        let needs_init = matches!(
            self.fgmres_resources.as_ref(),
            Some(res) if res.amg_resources.is_none()
        );
        if !needs_init {
            return;
        }

        let values = self
            .read_buffer_f32(&self.b_matrix_values, self.block_col_indices.len())
            .await;

        let num_cells = self.num_cells as usize;
        let mut scalar_values = vec![0.0f32; self.scalar_col_indices.len()];

        for cell in 0..num_cells {
            let start = self.scalar_row_offsets[cell] as usize;
            let end = self.scalar_row_offsets[cell + 1] as usize;
            let mut map = HashMap::with_capacity(end - start);
            for idx in start..end {
                map.insert(self.scalar_col_indices[idx] as usize, idx);
            }

            for row in 0..4usize {
                let block_row = cell * 4 + row;
                let bstart = self.block_row_offsets[block_row] as usize;
                let bend = self.block_row_offsets[block_row + 1] as usize;
                for k in bstart..bend {
                    let col_cell = self.block_col_indices[k] as usize / 4;
                    if let Some(&pos) = map.get(&col_cell) {
                        scalar_values[pos] += values[k].abs();
                    }
                }
            }

            if let Some(&diag_pos) = map.get(&cell) {
                if scalar_values[diag_pos].abs() < 1e-12 {
                    scalar_values[diag_pos] = 1.0;
                }
            }
        }

        let matrix = CsrMatrix {
            row_offsets: self.scalar_row_offsets.clone(),
            col_indices: self.scalar_col_indices.clone(),
            values: scalar_values,
            num_rows: num_cells,
            num_cols: num_cells,
        };

        let amg = AmgResources::new(&self.context.device, &matrix, 20);
        if let Some(fgmres) = &mut self.fgmres_resources {
            fgmres.amg_resources = Some(amg);
        }
    }

    fn init_fgmres_resources(&self, max_restart: usize) -> CompressibleFgmresResources {
        let device = &self.context.device;
        let n = self.num_unknowns;
        let num_cells = self.num_cells;
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
        let b_block_inv = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compressible FGMRES block inv"),
            size: num_cells as u64 * 16 * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let pack_params = PackParams {
            num_cells: self.num_cells,
            component: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let b_pack_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Compressible FGMRES pack params"),
            contents: bytemuck::bytes_of(&pack_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let fgmres = FgmresWorkspace::new(
            device,
            n,
            num_cells,
            max_restart,
            &self.b_row_offsets,
            &self.b_col_indices,
            &self.b_matrix_values,
            FgmresPrecondBindings::Diag {
                diag_u: &b_diag_u,
                diag_v: &b_diag_v,
                diag_p: &b_diag_p,
            },
            "Compressible",
        );
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

        let bg_block_precond = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compressible FGMRES block precond BG"),
            layout: &bgl_block_precond,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: b_block_inv.as_entire_binding(),
            }],
        });
        let bg_pack_params = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compressible FGMRES pack params BG"),
            layout: &bgl_pack_params,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: b_pack_params.as_entire_binding(),
            }],
        });
        let pipeline_layout_precond_build =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compressible precond build pipeline layout"),
                bind_group_layouts: &[
                    fgmres.vectors_layout(),
                    fgmres.matrix_layout(),
                    &bgl_block_precond,
                    fgmres.params_layout(),
                ],
                push_constant_ranges: &[],
            });
        let pipeline_layout_precond_apply =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compressible precond apply pipeline layout"),
                bind_group_layouts: &[
                    fgmres.vectors_layout(),
                    fgmres.matrix_layout(),
                    &bgl_block_precond,
                    fgmres.params_layout(),
                ],
                push_constant_ranges: &[],
            });
        let pipeline_layout_pack = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compressible FGMRES pack pipeline layout"),
            bind_group_layouts: &[&bgl_pack, &bgl_pack_params],
            push_constant_ranges: &[],
        });

        let shader_precond =
            bindings::compressible_precond::create_shader_module_embed_source(device);
        let shader_pack =
            bindings::compressible_amg_pack::create_shader_module_embed_source(device);
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

        let pipeline_build_block_inv = make_precond_pipeline(
            "Compressible Precond Build Block Inv",
            "build_block_inv",
            &pipeline_layout_precond_build,
        );
        let pipeline_apply_block_precond = make_precond_pipeline(
            "Compressible Precond Apply Block",
            "apply_block_precond",
            &pipeline_layout_precond_apply,
        );
        let pipeline_pack_component =
            make_pack_pipeline("Compressible AMG Pack Component", "pack_component");
        let pipeline_unpack_component =
            make_pack_pipeline("Compressible AMG Unpack Component", "unpack_component");

        CompressibleFgmresResources {
            fgmres,
            b_diag_u,
            b_diag_v,
            b_diag_p,
            b_block_inv,
            b_pack_params,
            bgl_block_precond,
            bgl_pack,
            bgl_pack_params,
            bg_block_precond,
            bg_pack_params,
            pipeline_build_block_inv,
            pipeline_apply_block_precond,
            pipeline_pack_component,
            pipeline_unpack_component,
            amg_resources: None,
        }
    }

    pub(crate) fn solve_compressible_fgmres(
        &mut self,
        max_restart: usize,
        tol: f32,
    ) -> LinearSolverStats {
        let start = std::time::Instant::now();
        self.ensure_fgmres_resources(max_restart);
        let precond = CompressibleKrylovPreconditioner::select(
            self.preconditioner,
            env_flag("CFD2_COMP_AMG", false),
            env_flag("CFD2_COMP_BLOCK_PRECOND", true),
        );
        precond.ensure_resources(self);
        let Some(fgmres) = &self.fgmres_resources else {
            return LinearSolverStats::default();
        };

        let n = self.num_unknowns;
        let workgroups = self.workgroups_for_size(n);
        let (dispatch_x, dispatch_y) = self.dispatch_2d(workgroups);
        let dispatch_x_threads = self.dispatch_x_threads(workgroups);
        let block_workgroups = self.workgroups_for_size(self.num_cells);
        let (block_dispatch_x, block_dispatch_y) = self.dispatch_2d(block_workgroups);

        self.zero_buffer(&self.b_x, n);

        let tol_abs = 1e-6f32;
        let rhs_norm = self.gpu_norm(fgmres, self.b_rhs.as_entire_binding(), n);
        if rhs_norm <= tol_abs {
            let stats = LinearSolverStats {
                iterations: 0,
                residual: rhs_norm,
                converged: true,
                diverged: false,
                time: start.elapsed(),
            };
            if std::env::var("CFD2_DEBUG_FGMRES").ok().as_deref() == Some("1") {
                eprintln!(
                    "fgmres: iters=0 rhs_norm={:.3e} residual={:.3e} converged=true",
                    rhs_norm, rhs_norm
                );
            }
            return stats;
        }

        let params = RawFgmresParams {
            n,
            num_cells: self.num_cells,
            num_iters: 0,
            omega: 1.0,
            dispatch_x: dispatch_x_threads,
            max_restart: fgmres.fgmres.max_restart() as u32,
            column_offset: 0,
            _pad3: 0,
        };
        let core = fgmres.fgmres.core(&self.context.device, &self.context.queue);
        write_params(&core, &params);
        precond.prepare(self, fgmres, block_workgroups);

        let iter_params = IterParams {
            current_idx: 0,
            max_restart: fgmres.fgmres.max_restart() as u32,
            _pad1: 0,
            _pad2: 0,
        };

        write_zeros(&core, fgmres.fgmres.hessenberg_buffer());
        write_zeros(&core, fgmres.fgmres.givens_buffer());
        write_zeros(&core, fgmres.fgmres.y_buffer());
        let mut g_init = vec![0.0f32; fgmres.fgmres.max_restart() + 1];
        g_init[0] = rhs_norm;
        self.context
            .queue
            .write_buffer(fgmres.fgmres.g_buffer(), 0, bytemuck::cast_slice(&g_init));

        let basis0 = self.basis_binding(fgmres, 0);
        self.dispatch_vector_pipeline(
            fgmres.fgmres.pipeline_copy(),
            fgmres,
            &self.create_vector_bind_group(
                fgmres,
                self.b_rhs.as_entire_binding(),
                basis0,
                fgmres.fgmres.temp_buffer().as_entire_binding(),
            ),
            fgmres.fgmres.params_bg(),
            dispatch_x,
            dispatch_y,
            );

        write_scalars(&core, &[1.0 / rhs_norm]);
        let basis0_y = self.basis_binding(fgmres, 0);
        self.dispatch_vector_pipeline(
            fgmres.fgmres.pipeline_scale_in_place(),
            fgmres,
            &self.create_vector_bind_group(
                fgmres,
                fgmres.fgmres.w_buffer().as_entire_binding(),
                basis0_y,
                fgmres.fgmres.temp_buffer().as_entire_binding(),
            ),
            fgmres.fgmres.params_bg(),
            dispatch_x,
            dispatch_y,
        );

        // Solve (single restart) using the shared GPU FGMRES core.
        let solve = fgmres_solve_once_with_preconditioner(
            &core,
            &self.b_x,
            rhs_norm,
            params,
            iter_params,
            FgmresSolveOnceConfig {
                tol_rel: tol,
                tol_abs,
                reset_x_before_update: true,
            },
            |_j, vj, z_buf| {
                precond.apply(
                    self,
                    fgmres,
                    vj,
                    z_buf,
                    dispatch_x,
                    dispatch_y,
                    block_dispatch_x,
                    block_dispatch_y,
                );
            },
        );
        let basis_size = solve.basis_size;
        let final_residual = solve.residual_est;
        let converged = solve.converged;

        let stats = LinearSolverStats {
            iterations: basis_size as u32,
            residual: final_residual,
            converged,
            diverged: false,
            time: start.elapsed(),
        };
        if std::env::var("CFD2_DEBUG_FGMRES").ok().as_deref() == Some("1") {
            eprintln!(
                "fgmres: iters={} rhs_norm={:.3e} residual={:.3e} converged={}",
                stats.iterations, rhs_norm, stats.residual, stats.converged
            );
        }
        stats
    }

    fn basis_binding<'a>(
        &self,
        fgmres: &'a CompressibleFgmresResources,
        idx: usize,
    ) -> wgpu::BindingResource<'a> {
        fgmres.fgmres.basis_binding(idx)
    }

    pub(crate) fn create_vector_bind_group<'a>(
        &self,
        fgmres: &CompressibleFgmresResources,
        x: wgpu::BindingResource<'a>,
        y: wgpu::BindingResource<'a>,
        z: wgpu::BindingResource<'a>,
    ) -> wgpu::BindGroup {
        fgmres.fgmres.create_vector_bind_group(
            &self.context.device,
            x,
            y,
            z,
            "Compressible FGMRES vector BG",
        )
    }

    fn create_pack_bind_group<'a>(
        &self,
        fgmres: &CompressibleFgmresResources,
        input: wgpu::BindingResource<'a>,
        output: wgpu::BindingResource<'a>,
    ) -> wgpu::BindGroup {
        self.context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Compressible AMG pack BG"),
                layout: &fgmres.bgl_pack,
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

    pub(crate) fn dispatch_vector_pipeline(
        &self,
        pipeline: &wgpu::ComputePipeline,
        fgmres: &CompressibleFgmresResources,
        vector_bg: &wgpu::BindGroup,
        group3_bg: &wgpu::BindGroup,
        dispatch_x: u32,
        dispatch_y: u32,
    ) {
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Compressible FGMRES vector pass"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compressible FGMRES vector"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, vector_bg, &[]);
            pass.set_bind_group(1, fgmres.fgmres.matrix_bg(), &[]);
            pass.set_bind_group(2, fgmres.fgmres.precond_bg(), &[]);
            pass.set_bind_group(3, group3_bg, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
        self.context.queue.submit(Some(encoder.finish()));
    }

    fn dispatch_pack_pipeline(
        &self,
        pipeline: &wgpu::ComputePipeline,
        fgmres: &CompressibleFgmresResources,
        pack_bg: &wgpu::BindGroup,
        dispatch_x: u32,
        dispatch_y: u32,
    ) {
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Compressible AMG pack pass"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compressible AMG pack"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, pack_bg, &[]);
            pass.set_bind_group(1, &fgmres.bg_pack_params, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
        self.context.queue.submit(Some(encoder.finish()));
    }

    pub(crate) fn dispatch_precond_build(
        &self,
        fgmres: &CompressibleFgmresResources,
        vector_bg: &wgpu::BindGroup,
        workgroups: u32,
    ) {
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Compressible Precond Build"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compressible Precond Build Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&fgmres.pipeline_build_block_inv);
            pass.set_bind_group(0, vector_bg, &[]);
            pass.set_bind_group(1, fgmres.fgmres.matrix_bg(), &[]);
            pass.set_bind_group(2, &fgmres.bg_block_precond, &[]);
            pass.set_bind_group(3, fgmres.fgmres.params_bg(), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.context.queue.submit(Some(encoder.finish()));
    }

    pub(crate) fn dispatch_precond_apply(
        &self,
        fgmres: &CompressibleFgmresResources,
        vector_bg: &wgpu::BindGroup,
        dispatch_x: u32,
        dispatch_y: u32,
    ) {
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Compressible Precond Apply"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compressible Precond Apply Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&fgmres.pipeline_apply_block_precond);
            pass.set_bind_group(0, vector_bg, &[]);
            pass.set_bind_group(1, fgmres.fgmres.matrix_bg(), &[]);
            pass.set_bind_group(2, &fgmres.bg_block_precond, &[]);
            pass.set_bind_group(3, fgmres.fgmres.params_bg(), &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
        self.context.queue.submit(Some(encoder.finish()));
    }

    pub(crate) fn apply_amg_precond<'a>(
        &self,
        fgmres: &CompressibleFgmresResources,
        input: wgpu::BindingResource<'a>,
        output: &wgpu::Buffer,
        dispatch_x: u32,
        dispatch_y: u32,
    ) {
        let Some(amg) = &fgmres.amg_resources else {
            return;
        };
        let num_cells = self.num_cells;
        let n = self.num_unknowns;
        self.zero_buffer(output, n);

        let mut params = PackParams {
            num_cells,
            component: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let pack_bg = self.create_pack_bind_group(
            fgmres,
            input,
            amg.levels[0].b_b.as_entire_binding(),
        );
        let unpack_bg = self.create_pack_bind_group(
            fgmres,
            amg.levels[0].b_x.as_entire_binding(),
            output.as_entire_binding(),
        );

        for component in 0..4u32 {
            params.component = component;
            self.context
                .queue
                .write_buffer(&fgmres.b_pack_params, 0, bytes_of(&params));

            self.dispatch_pack_pipeline(
                &fgmres.pipeline_pack_component,
                fgmres,
                &pack_bg,
                dispatch_x,
                dispatch_y,
            );

            self.zero_buffer(&amg.levels[0].b_x, num_cells);

            let mut encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Compressible AMG V-Cycle"),
                    });
            amg.v_cycle(&mut encoder, None);
            self.context.queue.submit(Some(encoder.finish()));

            self.dispatch_pack_pipeline(
                &fgmres.pipeline_unpack_component,
                fgmres,
                &unpack_bg,
                dispatch_x,
                dispatch_y,
            );
        }
    }

    fn workgroups_for_size(&self, n: u32) -> u32 {
        n.div_ceil(Self::WORKGROUP_SIZE)
    }

    fn dispatch_2d(&self, workgroups: u32) -> (u32, u32) {
        if workgroups <= Self::MAX_WORKGROUPS_PER_DIMENSION {
            (workgroups, 1)
        } else {
            let dispatch_y = workgroups.div_ceil(Self::MAX_WORKGROUPS_PER_DIMENSION);
            let dispatch_x = workgroups.div_ceil(dispatch_y);
            (dispatch_x, dispatch_y)
        }
    }

    fn dispatch_x_threads(&self, workgroups: u32) -> u32 {
        let (dispatch_x, _) = self.dispatch_2d(workgroups);
        dispatch_x * Self::WORKGROUP_SIZE
    }

    fn write_scalars(&self, fgmres: &CompressibleFgmresResources, scalars: &[f32]) {
        self.context
            .queue
            .write_buffer(fgmres.fgmres.scalars_buffer(), 0, bytemuck::cast_slice(scalars));
    }

    fn zero_buffer(&self, buffer: &wgpu::Buffer, n: u32) {
        let zeros = vec![0.0f32; n as usize];
        self.context
            .queue
            .write_buffer(buffer, 0, bytemuck::cast_slice(&zeros));
    }

    fn read_scalar(&self, fgmres: &CompressibleFgmresResources) -> f32 {
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Compressible FGMRES read scalar"),
                });
        encoder.copy_buffer_to_buffer(
            fgmres.fgmres.scalars_buffer(),
            0,
            fgmres.fgmres.staging_scalar_buffer(),
            0,
            4,
        );
        self.context.queue.submit(Some(encoder.finish()));

        let slice = fgmres.fgmres.staging_scalar_buffer().slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = tx.send(v);
        });
        let _ = self
            .context
            .device
            .poll(wgpu::PollType::wait_indefinitely());
        rx.recv().ok().and_then(|v| v.ok()).unwrap();
        let data = slice.get_mapped_range();
        let value = f32::from_ne_bytes([data[0], data[1], data[2], data[3]]);
        drop(data);
        fgmres.fgmres.staging_scalar_buffer().unmap();
        value
    }

    fn gpu_norm<'a>(
        &self,
        fgmres: &CompressibleFgmresResources,
        x: wgpu::BindingResource<'a>,
        n: u32,
    ) -> f32 {
        let workgroups = self.workgroups_for_size(n);
        let (dispatch_x, dispatch_y) = self.dispatch_2d(workgroups);

        let vector_bg = self.create_vector_bind_group(
            fgmres,
            x,
            fgmres.fgmres.temp_buffer().as_entire_binding(),
            fgmres.fgmres.dot_partial_buffer().as_entire_binding(),
        );

        let reduce_bg = self.create_vector_bind_group(
            fgmres,
            fgmres.fgmres.dot_partial_buffer().as_entire_binding(),
            fgmres.fgmres.temp_buffer().as_entire_binding(),
            fgmres.fgmres.temp_buffer().as_entire_binding(),
        );

        let params = RawFgmresParams {
            n,
            num_cells: self.num_cells,
            num_iters: 0,
            omega: 1.0,
            dispatch_x: self.dispatch_x_threads(workgroups),
            max_restart: fgmres.fgmres.max_restart() as u32,
            column_offset: 0,
            _pad3: 0,
        };
        self.context
            .queue
            .write_buffer(fgmres.fgmres.params_buffer(), 0, bytes_of(&params));
        let iter_params = IterParams {
            current_idx: 0,
            max_restart: fgmres.fgmres.max_restart() as u32,
            _pad1: 0,
            _pad2: 0,
        };
        self.context
            .queue
            .write_buffer(fgmres.fgmres.iter_params_buffer(), 0, bytes_of(&iter_params));

        self.dispatch_vector_pipeline(
            fgmres.fgmres.pipeline_norm_sq(),
            fgmres,
            &vector_bg,
            fgmres.fgmres.params_bg(),
            dispatch_x,
            dispatch_y,
        );

        let reduce_params = RawFgmresParams {
            n: fgmres.fgmres.num_dot_groups(),
            num_cells: 0,
            num_iters: 0,
            omega: 0.0,
            dispatch_x: Self::WORKGROUP_SIZE,
            max_restart: 0,
            column_offset: 0,
            _pad3: 0,
        };
        self.context
            .queue
            .write_buffer(fgmres.fgmres.params_buffer(), 0, bytes_of(&reduce_params));

        self.dispatch_vector_pipeline(
            fgmres.fgmres.pipeline_reduce_final(),
            fgmres,
            &reduce_bg,
            fgmres.fgmres.params_bg(),
            1,
            1,
        );

        let norm_sq = self.read_scalar(fgmres);
        norm_sq.sqrt()
    }
}

fn env_flag(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|val| {
            let val = val.to_ascii_lowercase();
            matches!(val.as_str(), "1" | "true" | "yes" | "y" | "on")
        })
        .unwrap_or(default)
}
