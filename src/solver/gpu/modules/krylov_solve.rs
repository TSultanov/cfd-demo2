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

/// Arguments for the `solve_once` method to reduce parameter count.
pub struct SolveOnceArgs<'a> {
    pub context: &'a GpuContext,
    pub system: LinearSystemView<'a>,
    pub rhs_norm: f32,
    pub params: RawFgmresParams,
    pub iter_params: IterParams,
    pub config: FgmresSolveOnceConfig,
    pub dispatch: DispatchGrids,
    pub precond_label: &'a str,
}

impl<P> KrylovSolveModule<P> {
    pub fn new(fgmres: FgmresWorkspace, precond: P) -> Self {
        Self { fgmres, precond }
    }

    pub fn rhs_norm(&self, context: &GpuContext, system: LinearSystemView<'_>, n: u32) -> f32 {
        self.fgmres.gpu_norm(
            &context.device,
            &context.queue,
            system.rhs().as_entire_binding(),
            n,
        )
    }
}

impl<P: FgmresPreconditionerModule> KrylovSolveModule<P> {
    pub fn solve_once(&mut self, args: SolveOnceArgs<'_>) -> FgmresSolveOnceResult {
        let SolveOnceArgs {
            context,
            system,
            rhs_norm,
            params,
            iter_params,
            config,
            dispatch,
            precond_label,
        } = args;
        let core = self.fgmres.core(&context.device, &context.queue);
        fgmres_solve_once_with_preconditioner(
            &core,
            system.x(),
            rhs_norm,
            params,
            iter_params,
            config,
            |_j, encoder, vj, z_buf| {
                encoder.push_debug_group(precond_label);
                self.precond.encode_apply(
                    &context.device,
                    encoder,
                    &self.fgmres,
                    vj,
                    z_buf,
                    dispatch,
                );
                encoder.pop_debug_group();
            },
        )
    }
}
