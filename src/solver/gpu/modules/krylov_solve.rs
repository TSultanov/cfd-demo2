use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::linear_solver::fgmres::{
    fgmres_solve_once_with_preconditioner, FgmresSolveOnceConfig, FgmresSolveOnceResult,
    FgmresWorkspace, IterParams, RawFgmresParams,
};
use crate::solver::gpu::modules::krylov_precond::{DispatchGrids, FgmresPreconditionerModule};
use crate::solver::gpu::modules::linear_system::LinearSystemView;

pub struct KrylovSolveModule<P> {
    pub fgmres: FgmresWorkspace,
    pub precond: P,
}

impl<P> KrylovSolveModule<P> {
    pub fn new(fgmres: FgmresWorkspace, precond: P) -> Self {
        Self { fgmres, precond }
    }

    pub fn rhs_norm(
        &self,
        context: &GpuContext,
        system: LinearSystemView<'_>,
        n: u32,
    ) -> f32 {
        self.fgmres.gpu_norm(
            &context.device,
            &context.queue,
            system.rhs().as_entire_binding(),
            n,
        )
    }
}

impl<P: FgmresPreconditionerModule> KrylovSolveModule<P> {
    pub fn solve_once(
        &mut self,
        context: &GpuContext,
        system: LinearSystemView<'_>,
        rhs_norm: f32,
        params: RawFgmresParams,
        iter_params: IterParams,
        config: FgmresSolveOnceConfig,
        dispatch: DispatchGrids,
        precond_label: &str,
    ) -> FgmresSolveOnceResult {
        let core = self.fgmres.core(&context.device, &context.queue);
        fgmres_solve_once_with_preconditioner(
            &core,
            system.x(),
            rhs_norm,
            params,
            iter_params,
            config,
            |_j, vj, z_buf| {
                let mut encoder = context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some(precond_label),
                    });
                self.precond.encode_apply(
                    &context.device,
                    &mut encoder,
                    &self.fgmres,
                    vj,
                    z_buf,
                    dispatch,
                );
                context.queue.submit(Some(encoder.finish()));
            },
        )
    }
}
