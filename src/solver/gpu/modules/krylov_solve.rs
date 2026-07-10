use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::linear_solver::fgmres::{
    encode_fgmres_seed_basis0_from_system, encode_fgmres_solve_once_with_preconditioner,
    encode_rhs_norm_into_scalars, encode_write_params, read_solver_scalars_after_submit,
    solve_once_from_encoded_status, submit_fgmres_encoded_pass, FgmresSolveOnceConfig,
    FgmresSolveOnceResult, FgmresWorkspace, IterParams, PrecondTarget, RawFgmresParams,
    FGMRES_SCALAR_CONVERGED, FGMRES_SCALAR_RESIDUAL_EST, FGMRES_SCALAR_STOP_PUB,
    FGMRES_SCALAR_TOTAL_ITERS,
};
use crate::solver::gpu::modules::krylov_precond::{DispatchGrids, PreconditionerModule};
use crate::solver::gpu::modules::linear_system::LinearSystemView;
use crate::solver::gpu::structs::LinearSolverStats;

pub struct KrylovSolveModule<P> {
    pub fgmres: FgmresWorkspace,
    pub precond: P,
    /// Adaptive iteration budget for the chunked encoded path: the next
    /// solve encodes only ~what the previous solve actually needed (the
    /// stall-stop makes "actually needed" observable). None = no history
    /// yet; the configured max_iters always caps it. See
    /// `submit_solve_fgmres_fixed_iterations_chunked`.
    pub adaptive_budget: Option<u32>,
    /// AUTO-mode CGS2 arming (see `cgs2_mode` in `linear_solver.rs`): set
    /// when the previous solve's actual iterations exceeded the arming
    /// fraction of max_iters — a late-converging solver benefits from CGS2
    /// re-orthogonalization on its next solve, a fast one only pays its
    /// overhead. Per-module state, like `adaptive_budget`: each solver
    /// arms independently.
    pub cgs2_auto_engaged: bool,
}

/// Extra information read back alongside [`LinearSolverStats`] from the
/// encoded path (see [`KrylovSolveModule::read_last_solver_stats`]).
pub struct EncodedSolveInfo {
    /// Actual Arnoldi iterations executed (SCALAR_TOTAL_ITERS), as opposed
    /// to the encoded budget.
    pub actual_iters: u32,
    /// Whether the solve stopped itself (convergence, guard freeze, or
    /// stall) before exhausting the encoded budget.
    pub stopped_early: bool,
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
    pub preserve_convergence_state: bool,
    /// When true, compute `||rhs||` on GPU and store it in `scalars[RHS_NORM]`
    /// after the scalars buffer has been initialised.  This must happen after
    /// `seed_basis0` and after the full-scalars init in the non-preserve path
    /// so that the copy into slot 14 is not clobbered.
    pub compute_rhs_norm_on_gpu: bool,
}

impl<P> KrylovSolveModule<P> {
    pub fn new(fgmres: FgmresWorkspace, precond: P) -> Self {
        Self {
            fgmres,
            precond,
            adaptive_budget: None,
            cgs2_auto_engaged: false,
        }
    }

    pub fn rhs_norm(&self, context: &GpuContext, system: LinearSystemView<'_>, n: u32) -> f32 {
        self.fgmres.gpu_norm(
            &context.device,
            &context.queue,
            system.rhs().as_entire_binding(),
            n,
        )
    }

    /// Encode a GPU-side `||rhs||` computation into `scalars[FGMRES_SCALAR_RHS_NORM]`.
    ///
    /// No host readback is performed — the result stays GPU-resident for use by
    /// subsequent encoded kernels.
    pub fn encode_rhs_norm(
        &self,
        context: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        system: LinearSystemView<'_>,
    ) {
        let core = self.fgmres.core(&context.device, &context.queue);
        encode_rhs_norm_into_scalars(&core, encoder, system);
    }

    /// Read back the FGMRES solver scalars after a submission and return a
    /// [`LinearSolverStats`] with the real GPU-computed residual and convergence
    /// flag.  `iterations` and `time` are supplied by the caller because they
    /// are not tracked in the GPU scalars buffer.
    pub fn read_last_solver_stats(
        &self,
        context: &GpuContext,
        submission_index: wgpu::SubmissionIndex,
        iterations: u32,
        time: std::time::Duration,
    ) -> (LinearSolverStats, EncodedSolveInfo) {
        let core = self.fgmres.core(&context.device, &context.queue);
        let scalars = read_solver_scalars_after_submit(&core, submission_index);
        let residual_est = scalars[FGMRES_SCALAR_RESIDUAL_EST];
        let converged = scalars[FGMRES_SCALAR_CONVERGED] > 0.5;
        let total = scalars[FGMRES_SCALAR_TOTAL_ITERS];
        let actual_iters = if total.is_finite() && total >= 1.0 {
            (total.round() as u32).min(iterations)
        } else {
            iterations
        };
        let info = EncodedSolveInfo {
            actual_iters,
            stopped_early: scalars[FGMRES_SCALAR_STOP_PUB] > 0.5,
        };

        let stats = if !residual_est.is_finite() {
            LinearSolverStats::diverged(actual_iters, residual_est, time)
        } else if converged {
            LinearSolverStats::converged(actual_iters, residual_est, time)
        } else {
            LinearSolverStats::max_iterations(actual_iters, residual_est, time)
        };
        (stats, info)
    }
}

impl<P: PreconditionerModule> KrylovSolveModule<P> {
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
            let ctx = self.fgmres.precond_context(dispatch);
            self.precond.encode_prepare(
                &context.device,
                &context.queue,
                encoder,
                &ctx,
                system.rhs().as_entire_binding(),
            );
        }
        if seed_basis_from_system {
            encode_fgmres_seed_basis0_from_system(
                &core,
                encoder,
                system,
                seed_max_restart,
                args.preserve_convergence_state,
            );
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
            preserve_convergence_state,
            compute_rhs_norm_on_gpu,
        } = args;
        let rhs_norm_system = if compute_rhs_norm_on_gpu {
            Some(system)
        } else {
            None
        };
        // In-pass preconditioner applies let the restart loop encode as one
        // long compute pass (see `PreconditionerModule::begin_in_pass_applies`).
        let precond_in_pass = !std::env::var("CFD2_FGMRES_MEGA_PASS")
            .is_ok_and(|v| v == "0")
            && self.precond.begin_in_pass_applies(&context.device);
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
            preserve_convergence_state,
            rhs_norm_system,
            precond_in_pass,
            |_j, target, vj, z_buf| {
                let ctx = self.fgmres.precond_context(dispatch);
                match target {
                    PrecondTarget::Encoder(encoder) => {
                        encoder.push_debug_group(precond_label);
                        self.precond.encode_apply(
                            &context.device,
                            encoder,
                            &ctx,
                            vj,
                            z_buf,
                        );
                        encoder.pop_debug_group();
                    }
                    PrecondTarget::Pass(pass) => {
                        self.precond.encode_apply_in_pass(
                            &context.device,
                            pass,
                            &ctx,
                            vj,
                            z_buf,
                        );
                    }
                }
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
            preserve_convergence_state: false,
            compute_rhs_norm_on_gpu: false,
        }
    }
}
