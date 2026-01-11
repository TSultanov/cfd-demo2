use crate::solver::gpu::lowering::kernel_registry;
use crate::solver::gpu::linear_solver::amg::{AmgResources, CsrMatrix};
use crate::solver::gpu::linear_solver::fgmres::FgmresWorkspace;
use crate::solver::model::KernelId;
use crate::solver::gpu::modules::krylov_precond::{DispatchGrids, FgmresPreconditionerModule};
use crate::solver::gpu::structs::PreconditionerType;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoupledPressureSolveKind {
    Chebyshev,
    Amg,
}

impl CoupledPressureSolveKind {
    pub fn from_config(preconditioner: PreconditionerType) -> Self {
        match preconditioner {
            PreconditionerType::Jacobi => Self::Chebyshev,
            PreconditionerType::Amg => Self::Amg,
        }
    }
}

pub struct CoupledSchurModule {
    num_cells: u32,
    pressure_kind: CoupledPressureSolveKind,

    /// Temporary buffer for pressure RHS (r_p')
    b_temp_p: wgpu::Buffer,
    /// Pressure solution buffer.
    b_p_sol: wgpu::Buffer,

    /// Schur Vector bind group layout (r_in, z_out, temp_p, p_sol, aux)
    bgl_schur_vectors: wgpu::BindGroupLayout,

    /// Bind group for pressure matrix CSR (Group 3)
    bg_pressure_matrix: wgpu::BindGroup,

    /// Schur complement preconditioner pipelines.
    pipeline_predict_and_form: wgpu::ComputePipeline,
    pipeline_relax_pressure: wgpu::ComputePipeline,
    pipeline_correct_vel: wgpu::ComputePipeline,

    amg: Option<AmgResources>,
    amg_level0_state_override: Option<wgpu::BindGroup>,
}

impl CoupledSchurModule {
    pub fn new(
        device: &wgpu::Device,
        fgmres: &FgmresWorkspace,
        num_cells: u32,
        pressure_row_offsets: &wgpu::Buffer,
        pressure_col_indices: &wgpu::Buffer,
        pressure_values: &wgpu::Buffer,
        pressure_kind: CoupledPressureSolveKind,
        shader_id: KernelId,
    ) -> Self {
        let b_temp_p = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Schur temp_p"),
            size: (num_cells as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let b_p_sol = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Schur p_sol"),
            size: (num_cells as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bgl_pressure_matrix =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Schur Pressure Matrix BGL"),
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
                ],
            });

        let bg_pressure_matrix = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Schur Pressure Matrix BG"),
            layout: &bgl_pressure_matrix,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pressure_row_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pressure_col_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pressure_values.as_entire_binding(),
                },
            ],
        });

        let bgl_schur_vectors = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Schur Vectors BGL"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
            ],
        });

        let pipeline_layout_schur =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Schur Pipeline Layout"),
                bind_group_layouts: &[
                    &bgl_schur_vectors,
                    fgmres.matrix_layout(),
                    fgmres.precond_layout(),
                    &bgl_pressure_matrix,
                ],
                push_constant_ranges: &[],
            });

        let shader_schur = kernel_registry::kernel_shader_module_by_id(device, "", shader_id)
            .expect("schur_precond shader missing from kernel registry");
        let make_schur_pipeline = |label: &str, entry: &str| -> wgpu::ComputePipeline {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout_schur),
                module: &shader_schur,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        Self {
            num_cells,
            pressure_kind,
            b_temp_p,
            b_p_sol,
            bgl_schur_vectors,
            bg_pressure_matrix,
            pipeline_predict_and_form: make_schur_pipeline(
                "Schur Predict & Form",
                "predict_and_form_schur",
            ),
            pipeline_relax_pressure: make_schur_pipeline("Schur Relax P", "relax_pressure"),
            pipeline_correct_vel: make_schur_pipeline("Schur Correct Vel", "correct_velocity"),
            amg: None,
            amg_level0_state_override: None,
        }
    }

    pub fn set_pressure_kind(&mut self, kind: CoupledPressureSolveKind) {
        self.pressure_kind = kind;
    }

    pub fn pressure_kind(&self) -> CoupledPressureSolveKind {
        self.pressure_kind
    }

    pub fn b_temp_p(&self) -> &wgpu::Buffer {
        &self.b_temp_p
    }

    pub fn b_p_sol(&self) -> &wgpu::Buffer {
        &self.b_p_sol
    }

    pub fn bg_pressure_matrix(&self) -> &wgpu::BindGroup {
        &self.bg_pressure_matrix
    }

    pub fn ensure_amg_resources(&mut self, device: &wgpu::Device, matrix: CsrMatrix) {
        if self.amg.is_some() {
            return;
        }
        let amg = AmgResources::new(device, &matrix, 20);

        let override_bg = {
            let level0 = &amg.levels[0];
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Schur AMG Level0 State Override"),
                layout: &amg.bgl_state,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.b_p_sol.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.b_temp_p.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: level0.b_params.as_entire_binding(),
                    },
                ],
            })
        };

        self.amg = Some(amg);
        self.amg_level0_state_override = Some(override_bg);
    }

    fn dispatch_schur(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        fgmres: &FgmresWorkspace,
        schur_bg: &wgpu::BindGroup,
        dispatch: (u32, u32),
        label: &str,
    ) {
        let (dispatch_x, dispatch_y) = dispatch;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, schur_bg, &[]);
        pass.set_bind_group(1, fgmres.matrix_bg(), &[]);
        pass.set_bind_group(2, fgmres.precond_bg(), &[]);
        pass.set_bind_group(3, &self.bg_pressure_matrix, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    fn create_schur_bg<'a>(
        &'a self,
        device: &wgpu::Device,
        v: wgpu::BindingResource<'a>,
        z: wgpu::BindingResource<'a>,
        p_sol: wgpu::BindingResource<'a>,
        aux: wgpu::BindingResource<'a>,
        label: &str,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.bgl_schur_vectors,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: v,
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: z,
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.b_temp_p.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: p_sol,
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: aux,
                },
            ],
        })
    }
}

impl FgmresPreconditionerModule for CoupledSchurModule {
    fn encode_apply(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        fgmres: &FgmresWorkspace,
        input: wgpu::BindingResource<'_>,
        output: &wgpu::Buffer,
        dispatch: DispatchGrids,
    ) {
        let aux = fgmres.temp_buffer().as_entire_binding();

        let current_bg = self.create_schur_bg(
            device,
            input.clone(),
            output.as_entire_binding(),
            self.b_p_sol.as_entire_binding(),
            aux.clone(),
            "Schur BG",
        );
        let swap_bg = self.create_schur_bg(
            device,
            input,
            output.as_entire_binding(),
            aux.clone(),
            self.b_p_sol.as_entire_binding(),
            "Schur Swap BG",
        );

        self.dispatch_schur(
            encoder,
            &self.pipeline_predict_and_form,
            fgmres,
            &current_bg,
            dispatch.cells,
            "Schur Predict & Form",
        );

        match self.pressure_kind {
            CoupledPressureSolveKind::Amg => {
                if let Some(amg) = &self.amg {
                    let override_bg = self
                        .amg_level0_state_override
                        .as_ref()
                        .expect("AMG override bind group missing");
                    amg.v_cycle(encoder, Some(override_bg));
                }
            }
            CoupledPressureSolveKind::Chebyshev => {
                let p_iters = (20 + (self.num_cells as f32).sqrt() as usize / 2)
                    .min(200)
                    .saturating_sub(1);
                if p_iters == 0 {
                    return;
                }

                let mut p_result_in_sol = true;
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Schur Relax P"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_relax_pressure);
                pass.set_bind_group(1, fgmres.matrix_bg(), &[]);
                pass.set_bind_group(2, fgmres.precond_bg(), &[]);
                pass.set_bind_group(3, &self.bg_pressure_matrix, &[]);

                for _ in 0..p_iters {
                    let bg = if p_result_in_sol {
                        &current_bg
                    } else {
                        &swap_bg
                    };
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(dispatch.cells.0, dispatch.cells.1, 1);
                    p_result_in_sol = !p_result_in_sol;
                }

                drop(pass);

                let correct_bg = if p_result_in_sol {
                    &current_bg
                } else {
                    &swap_bg
                };
                self.dispatch_schur(
                    encoder,
                    &self.pipeline_correct_vel,
                    fgmres,
                    correct_bg,
                    dispatch.cells,
                    "Schur Correct Vel",
                );
                return;
            }
        }

        // AMG leaves p in `b_p_sol`.
        self.dispatch_schur(
            encoder,
            &self.pipeline_correct_vel,
            fgmres,
            &current_bg,
            dispatch.cells,
            "Schur Correct Vel",
        );
    }
}
