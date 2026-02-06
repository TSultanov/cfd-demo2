use crate::solver::gpu::linear_solver::amg::{AmgResources, CsrMatrix};
use crate::solver::gpu::linear_solver::fgmres::FgmresWorkspace;
use crate::solver::gpu::lowering::kernel_registry;
use crate::solver::gpu::modules::krylov_precond::{DispatchGrids, FgmresPreconditionerModule};
use crate::solver::gpu::modules::resource_registry::ResourceRegistry;
use crate::solver::gpu::structs::PreconditionerType;
use crate::solver::gpu::wgsl_reflect;
use crate::solver::model::KernelId;

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
            PreconditionerType::BlockJacobi => Self::Chebyshev,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CoupledSchurKernelIds {
    pub predict_and_form: KernelId,
    pub relax_pressure: KernelId,
    pub correct_velocity: KernelId,
}

/// Input parameters for [`CoupledSchurModule`].
pub struct CoupledSchurInputs<'a> {
    pub num_cells: u32,
    pub pressure_row_offsets: &'a wgpu::Buffer,
    pub pressure_col_indices: &'a wgpu::Buffer,
    pub pressure_values: &'a wgpu::Buffer,
    pub diag_u_inv: &'a wgpu::Buffer,
    pub diag_p_inv: &'a wgpu::Buffer,
    pub precond_params: &'a wgpu::Buffer,
    pub pressure_kind: CoupledPressureSolveKind,
    pub kernels: CoupledSchurKernelIds,
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
    schur_bindings: &'static [wgsl_reflect::WgslBindingDesc],

    /// Schur complement diagonals/params bind group (Group 2)
    bg_schur_precond: wgpu::BindGroup,

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
    pub fn new(device: &wgpu::Device, inputs: CoupledSchurInputs<'_>) -> Self {
        let b_temp_p = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Schur temp_p"),
            size: (inputs.num_cells as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let b_p_sol = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Schur p_sol"),
            size: (inputs.num_cells as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let schur_src = kernel_registry::kernel_source_by_id("", inputs.kernels.predict_and_form)
            .expect("schur_precond predict_and_form shader missing from kernel registry");
        let schur_bindings = schur_src.bindings;

        let pipeline_predict_and_form = (schur_src.create_pipeline)(device);
        let pipeline_relax_pressure = {
            let src = kernel_registry::kernel_source_by_id("", inputs.kernels.relax_pressure)
                .expect("schur_precond relax_pressure shader missing from kernel registry");
            (src.create_pipeline)(device)
        };
        let pipeline_correct_vel = {
            let src = kernel_registry::kernel_source_by_id("", inputs.kernels.correct_velocity)
                .expect("schur_precond correct_velocity shader missing from kernel registry");
            (src.create_pipeline)(device)
        };

        let bgl_schur_vectors = pipeline_predict_and_form.get_bind_group_layout(0);
        let bgl_schur_precond = pipeline_predict_and_form.get_bind_group_layout(2);
        let bgl_pressure_matrix = pipeline_predict_and_form.get_bind_group_layout(3);
        let bg_schur_precond = {
            let registry = ResourceRegistry::new()
                .with_buffer("diag_u_inv", inputs.diag_u_inv)
                .with_buffer("diag_p_inv", inputs.diag_p_inv)
                .with_buffer("params", inputs.precond_params);
            wgsl_reflect::create_bind_group_from_bindings(
                device,
                "Schur Precond BG",
                &bgl_schur_precond,
                schur_bindings,
                2,
                |name| registry.resolve(name),
            )
            .unwrap_or_else(|err| panic!("Schur precond BG creation failed: {err}"))
        };
        let bg_pressure_matrix = {
            let registry = ResourceRegistry::new()
                .with_buffer("p_row_offsets", inputs.pressure_row_offsets)
                .with_buffer("p_col_indices", inputs.pressure_col_indices)
                .with_buffer("p_matrix_values", inputs.pressure_values);
            wgsl_reflect::create_bind_group_from_bindings(
                device,
                "Schur Pressure Matrix BG",
                &bgl_pressure_matrix,
                schur_bindings,
                3,
                |name| registry.resolve(name),
            )
            .unwrap_or_else(|err| panic!("Schur pressure matrix BG creation failed: {err}"))
        };

        Self {
            num_cells: inputs.num_cells,
            pressure_kind: inputs.pressure_kind,
            b_temp_p,
            b_p_sol,
            bgl_schur_vectors,
            schur_bindings,
            bg_schur_precond,
            bg_pressure_matrix,
            pipeline_predict_and_form,
            pipeline_relax_pressure,
            pipeline_correct_vel,
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

    pub fn has_amg_resources(&self) -> bool {
        self.amg.is_some()
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

        let level0 = &amg.levels[0];
        let override_bg = amg.create_state_override_bind_group(
            device,
            &self.b_p_sol,
            &self.b_temp_p,
            &level0.b_params,
            "Schur AMG Level0 State Override",
        );

        self.amg = Some(amg);
        self.amg_level0_state_override = Some(override_bg);
    }

    pub fn refresh_amg_level0_matrix(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        matrix_values: &wgpu::Buffer,
        num_nonzeros: u64,
    ) {
        let Some(amg) = &self.amg else {
            return;
        };
        let Some(level0) = amg.levels.first() else {
            return;
        };

        let size = num_nonzeros * 4;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("schur:refresh_amg_level0_matrix"),
        });
        encoder.copy_buffer_to_buffer(matrix_values, 0, &level0.b_matrix_values, 0, size);
        queue.submit(Some(encoder.finish()));
    }

    fn dispatch_schur(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        fgmres: &FgmresWorkspace,
        schur_bg: &wgpu::BindGroup,
        _dispatch: (u32, u32),
        label: &str,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, schur_bg, &[]);
        pass.set_bind_group(1, fgmres.matrix_bg(), &[]);
        pass.set_bind_group(2, &self.bg_schur_precond, &[]);
        pass.set_bind_group(3, &self.bg_pressure_matrix, &[]);
        pass.dispatch_workgroups_indirect(
            fgmres.indirect_args_buffer(),
            FgmresWorkspace::indirect_dispatch_cells_offset(),
        );
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
        wgsl_reflect::create_bind_group_from_bindings(
            device,
            label,
            &self.bgl_schur_vectors,
            self.schur_bindings,
            0,
            |name| match name {
                "r_in" => Some(v.clone()),
                "z_out" => Some(z.clone()),
                "temp_p" => Some(self.b_temp_p.as_entire_binding()),
                "p_sol" => Some(p_sol.clone()),
                "p_prev" => Some(aux.clone()),
                _ => None,
            },
        )
        .unwrap_or_else(|err| panic!("Schur vectors BG creation failed: {err}"))
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
                    amg.sync_control_scalars(encoder, fgmres.scalars_buffer());
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
                pass.set_bind_group(2, &self.bg_schur_precond, &[]);
                pass.set_bind_group(3, &self.bg_pressure_matrix, &[]);

                for _ in 0..p_iters {
                    let bg = if p_result_in_sol {
                        &current_bg
                    } else {
                        &swap_bg
                    };
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups_indirect(
                        fgmres.indirect_args_buffer(),
                        FgmresWorkspace::indirect_dispatch_cells_offset(),
                    );
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
