use super::compressible_fgmres::CompressibleFgmresResources;
use super::compressible_solver::GpuCompressibleSolver;
use super::coupled_solver_fgmres::FgmresResources;
use super::structs::{GpuSolver, PreconditionerType};

pub(crate) enum CoupledPressurePreconditioner {
    Chebyshev,
    Amg,
}

impl CoupledPressurePreconditioner {
    pub fn from_config(preconditioner: PreconditionerType) -> Self {
        match preconditioner {
            PreconditionerType::Jacobi => Self::Chebyshev,
            PreconditionerType::Amg => Self::Amg,
        }
    }

    pub fn ensure_resources(&self, solver: &mut GpuSolver) {
        if matches!(self, Self::Amg) {
            pollster::block_on(solver.ensure_amg_resources());
        }
    }

    pub fn encode(
        &self,
        solver: &GpuSolver,
        encoder: &mut wgpu::CommandEncoder,
        fgmres: &FgmresResources,
        amg_level0_state_override: Option<&wgpu::BindGroup>,
        current_bg: &wgpu::BindGroup,
        swap_bg: &wgpu::BindGroup,
        _workgroups_cells: u32,
        dispatch_x: u32,
        dispatch_y: u32,
        p_result_in_sol: &mut bool,
    ) {
        match self {
            Self::Amg => {
                // AMG doesn't need ping-pong.
                *p_result_in_sol = true;
                if let Some(amg) = &fgmres.amg_resources {
                    amg.v_cycle(encoder, amg_level0_state_override);
                }
            }
            Self::Chebyshev => {
                let num_cells = solver.num_cells;
                let p_iters = (20 + (num_cells as f32).sqrt() as usize / 2)
                    .min(200)
                    .saturating_sub(1);
                if p_iters == 0 {
                    return;
                }

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Schur Relax P"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&fgmres.pipeline_relax_pressure);
                // Layout requires these bind groups even though relax_pressure doesn't use group 1.
                pass.set_bind_group(1, &fgmres.bg_matrix, &[]);
                pass.set_bind_group(2, &fgmres.bg_precond, &[]);
                pass.set_bind_group(3, &fgmres.bg_pressure_matrix, &[]);

                for _ in 0..p_iters {
                    let bg = if *p_result_in_sol { current_bg } else { swap_bg };
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
                    *p_result_in_sol = !*p_result_in_sol;
                }
            }
        }
    }
}

pub(crate) enum CompressibleKrylovPreconditioner {
    Identity,
    BlockJacobi,
    Amg,
}

impl CompressibleKrylovPreconditioner {
    pub fn select(preconditioner: PreconditionerType, force_amg: bool, enable_block: bool) -> Self {
        if preconditioner == PreconditionerType::Amg || force_amg {
            return Self::Amg;
        }
        if enable_block {
            return Self::BlockJacobi;
        }
        Self::Identity
    }

    pub fn ensure_resources(&self, solver: &mut GpuCompressibleSolver) {
        if matches!(self, Self::Amg) {
            pollster::block_on(solver.ensure_amg_resources());
        }
    }

    pub fn prepare(
        &self,
        solver: &GpuCompressibleSolver,
        fgmres: &CompressibleFgmresResources,
        workgroups_cells: u32,
    ) {
        if matches!(self, Self::BlockJacobi) {
            let precond_vectors = solver.create_vector_bind_group(
                fgmres,
                solver.b_rhs.as_entire_binding(),
                fgmres.b_w.as_entire_binding(),
                fgmres.b_temp.as_entire_binding(),
            );
            solver.dispatch_precond_build(fgmres, &precond_vectors, workgroups_cells);
        }
    }

    pub fn apply(
        &self,
        solver: &GpuCompressibleSolver,
        fgmres: &CompressibleFgmresResources,
        vj: wgpu::BindingResource<'_>,
        z_out: &wgpu::Buffer,
        dispatch_x: u32,
        dispatch_y: u32,
        block_dispatch_x: u32,
        block_dispatch_y: u32,
    ) {
        match self {
            Self::Amg => solver.apply_amg_precond(
                fgmres,
                vj,
                z_out,
                block_dispatch_x,
                block_dispatch_y,
            ),
            Self::BlockJacobi => solver.dispatch_precond_apply(
                fgmres,
                &solver.create_vector_bind_group(
                    fgmres,
                    vj,
                    z_out.as_entire_binding(),
                    fgmres.b_temp.as_entire_binding(),
                ),
                block_dispatch_x,
                block_dispatch_y,
            ),
            Self::Identity => solver.dispatch_vector_pipeline(
                &fgmres.pipeline_copy,
                fgmres,
                &solver.create_vector_bind_group(
                    fgmres,
                    vj,
                    z_out.as_entire_binding(),
                    fgmres.b_temp.as_entire_binding(),
                ),
                &fgmres.bg_params,
                dispatch_x,
                dispatch_y,
            ),
        }
    }
}
