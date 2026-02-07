use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::linear_solver::fgmres::{
    encode_fgmres_seed_basis0_from_system, encode_fgmres_solve_once_with_preconditioner,
    encode_write_params, solve_once_from_encoded_status, submit_fgmres_encoded_pass,
    FgmresSolveOnceConfig, FgmresSolveOnceResult, FgmresWorkspace, IterParams, RawFgmresParams,
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

pub struct EncodeSolveOnceArgs<'a> {
    pub context: &'a GpuContext,
    pub system: LinearSystemView<'a>,
    pub rhs_norm: f32,
    pub params: RawFgmresParams,
    pub iter_params: IterParams,
    pub config: FgmresSolveOnceConfig,
    pub dispatch: DispatchGrids,
    pub precond_label: &'a str,
    pub capture_solver_scalars: bool,
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
        self.solve_once_with_prepare(args, false, false)
    }

    pub fn solve_once_with_prepare(
        &mut self,
        args: SolveOnceArgs<'_>,
        prepare_preconditioner: bool,
        seed_basis_from_system: bool,
    ) -> FgmresSolveOnceResult {
        let context = args.context;
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("FGMRES restart body"),
            });
        let max_restart = self.encode_solve_once_with_prepare(
            EncodeSolveOnceArgs::from_solve_once_args(args, true),
            &mut encoder,
            prepare_preconditioner,
            seed_basis_from_system,
        );
        self.finish_encoded_solve_once(context, encoder, max_restart, true)
    }

    pub fn encode_solve_once_with_prepare(
        &mut self,
        args: EncodeSolveOnceArgs<'_>,
        encoder: &mut wgpu::CommandEncoder,
        prepare_preconditioner: bool,
        seed_basis_from_system: bool,
    ) -> usize {
        let context = args.context;
        let system = args.system;
        let dispatch = args.dispatch;
        let seed_max_restart = args.iter_params.max_restart;
        let core = self.fgmres.core(&context.device, &context.queue);

        // Keep params deterministic for any preconditioner prepare kernels that consume
        // fgmres `params` before the restart body starts.
        encode_write_params(&core, encoder, &args.params);

        if prepare_preconditioner {
            self.precond.encode_prepare(
                &context.device,
                &context.queue,
                encoder,
                &self.fgmres,
                system.rhs().as_entire_binding(),
                dispatch,
            );
        }
        if seed_basis_from_system {
            encode_fgmres_seed_basis0_from_system(&core, encoder, system, seed_max_restart);
        }
        self.encode_solve_once(args, encoder)
    }

    pub fn encode_solve_once(
        &mut self,
        args: EncodeSolveOnceArgs<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> usize {
        let EncodeSolveOnceArgs {
            context,
            system,
            rhs_norm,
            params,
            iter_params,
            config,
            dispatch,
            precond_label,
            capture_solver_scalars,
        } = args;
        let core = self.fgmres.core(&context.device, &context.queue);
        let encoded = encode_fgmres_solve_once_with_preconditioner(
            &core,
            encoder,
            system.x(),
            rhs_norm,
            params,
            iter_params,
            config,
            capture_solver_scalars,
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
        );
        encoded.max_restart
    }

    pub fn finish_encoded_solve_once(
        &mut self,
        context: &GpuContext,
        encoder: wgpu::CommandEncoder,
        max_restart: usize,
        capture_solver_scalars: bool,
    ) -> FgmresSolveOnceResult {
        let core = self.fgmres.core(&context.device, &context.queue);
        let submission_index = submit_fgmres_encoded_pass(&core, encoder, "restart_body");
        if capture_solver_scalars {
            solve_once_from_encoded_status(&core, submission_index, max_restart)
        } else {
            FgmresSolveOnceResult {
                basis_size: max_restart.max(1),
                residual_est: f32::INFINITY,
                converged: false,
            }
        }
    }
}

impl<'a> EncodeSolveOnceArgs<'a> {
    pub fn from_solve_once_args(args: SolveOnceArgs<'a>, capture_solver_scalars: bool) -> Self {
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
        Self {
            context,
            system,
            rhs_norm,
            params,
            iter_params,
            config,
            dispatch,
            precond_label,
            capture_solver_scalars,
        }
    }
}
