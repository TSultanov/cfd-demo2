use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::linear_solver::fgmres::{
    write_params, FgmresSolveOnceConfig, IterParams, RawFgmresParams,
};
use crate::solver::gpu::modules::krylov_precond::{FgmresPreconditionerModule, KrylovDispatch};
use crate::solver::gpu::modules::krylov_solve::KrylovSolveModule;
use crate::solver::gpu::modules::linear_system::LinearSystemView;
use crate::solver::gpu::structs::LinearSolverStats;
use std::time::Instant;

pub fn solve_fgmres<P: FgmresPreconditionerModule>(
    context: &GpuContext,
    krylov: &mut KrylovSolveModule<P>,
    system: LinearSystemView<'_>,
    n: u32,
    num_cells: u32,
    dispatch: KrylovDispatch,
    max_restart: usize,
    tol: f32,
    tol_abs: f32,
    precond_label: &str,
) -> LinearSolverStats {
    let start = Instant::now();

    // 1. Zero initial guess (assuming x0 = 0 for now).
    zero_buffer(context, system.x(), n);

    let rhs_norm = krylov.rhs_norm(context, system, n);
    if !rhs_norm.is_finite() {
        let stats = LinearSolverStats {
            iterations: 0,
            residual: rhs_norm,
            converged: false,
            diverged: true,
            time: start.elapsed(),
        };
        debug_log(0, rhs_norm, rhs_norm, false);
        return stats;
    }
    if rhs_norm <= tol_abs {
        let stats = LinearSolverStats {
            iterations: 0,
            residual: rhs_norm,
            converged: true,
            diverged: false,
            time: start.elapsed(),
        };
        debug_log(0, rhs_norm, rhs_norm, true);
        return stats;
    }

    let capacity = krylov.fgmres.max_restart() as u32;
    let iterations_per_cycle = (max_restart as u32).min(capacity);

    let params = RawFgmresParams {
        n,
        num_cells,
        num_iters: 0,
        omega: 1.0,
        dispatch_x: dispatch.dofs_dispatch_x_threads,
        max_restart: iterations_per_cycle,
        column_offset: 0,
        _pad3: 0,
    };
    let core = krylov.fgmres.core(&context.device, &context.queue);

    // Initial parameter write
    write_params(&core, &params);

    // Preconditioner setup
    krylov.precond.prepare(
        &context.device,
        &context.queue,
        &krylov.fgmres,
        system.rhs().as_entire_binding(),
        dispatch.grids,
    );

    let iter_params = IterParams {
        current_idx: 0,
        max_restart: iterations_per_cycle,
        _pad1: 0,
        _pad2: 0,
    };

    // FGMRES initialization
    krylov.fgmres.clear_restart_aux(&core);
    krylov.fgmres.write_g0(&context.queue, rhs_norm);
    krylov.fgmres.init_basis0_from_vector_normalized(
        &core,
        system.rhs().as_entire_binding(),
        1.0 / rhs_norm,
        "Generic FGMRES",
    );

    // Solve (single restart)
    let solve = krylov.solve_once(
        context,
        system,
        rhs_norm,
        params,
        iter_params,
        FgmresSolveOnceConfig {
            tol_rel: tol,
            tol_abs,
            reset_x_before_update: true,
        },
        dispatch.grids,
        precond_label,
    );

    let stats = LinearSolverStats {
        iterations: solve.basis_size as u32,
        residual: solve.residual_est,
        converged: solve.converged,
        diverged: false,
        time: start.elapsed(),
    };
    debug_log(stats.iterations, rhs_norm, stats.residual, stats.converged);
    stats
}

fn zero_buffer(context: &GpuContext, buffer: &wgpu::Buffer, n: u32) {
    let zeros = vec![0.0f32; n as usize];
    context
        .queue
        .write_buffer(buffer, 0, bytemuck::cast_slice(&zeros));
}

fn debug_log(iters: u32, rhs_norm: f32, residual: f32, converged: bool) {
    const DEBUG_FGMRES: bool = false;
    if DEBUG_FGMRES {
        eprintln!(
            "fgmres: iters={} rhs_norm={:.3e} residual={:.3e} converged={}",
            iters, rhs_norm, residual, converged
        );
    }
}
