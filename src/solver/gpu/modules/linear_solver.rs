use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::linear_solver::fgmres::{
    write_params, FgmresSolveOnceConfig, IterParams, RawFgmresParams,
};
use crate::solver::gpu::modules::krylov_precond::{FgmresPreconditionerModule, KrylovDispatch};
use crate::solver::gpu::modules::krylov_solve::{KrylovSolveModule, SolveOnceArgs};
use crate::solver::gpu::modules::linear_system::LinearSystemView;
use crate::solver::gpu::structs::LinearSolverStats;
use std::time::Instant;

/// Arguments for the `solve_fgmres` function to reduce parameter count.
pub struct SolveFgmresArgs<'a> {
    pub context: &'a GpuContext,
    pub system: LinearSystemView<'a>,
    pub n: u32,
    pub num_cells: u32,
    pub dispatch: KrylovDispatch,
    pub max_restart: usize,
    pub max_iters: u32,
    pub tol: f32,
    pub tol_abs: f32,
    pub precond_label: &'a str,
}

pub fn solve_fgmres<P: FgmresPreconditionerModule>(
    krylov: &mut KrylovSolveModule<P>,
    args: SolveFgmresArgs<'_>,
) -> LinearSolverStats {
    let SolveFgmresArgs {
        context,
        system,
        n,
        num_cells,
        dispatch,
        max_restart,
        max_iters,
        tol,
        tol_abs,
        precond_label,
    } = args;
    let start = Instant::now();

    let debug_fgmres = std::env::var("CFD2_DEBUG_FGMRES")
        .map(|v| v != "0")
        .unwrap_or(false);

    let rhs_norm = krylov.rhs_norm(context, system, n);
    if debug_fgmres {
        eprintln!(
            "[cfd2][fgmres] n={n} rhs_norm={rhs_norm:.3e} tol={tol:.3e} tol_abs={tol_abs:.3e} precond={precond_label}"
        );
    }
    if !rhs_norm.is_finite() {
        let stats = LinearSolverStats {
            iterations: 0,
            residual: rhs_norm,
            converged: false,
            diverged: true,
            time: start.elapsed(),
        };
        if debug_fgmres {
            eprintln!("[cfd2][fgmres] early-exit: rhs_norm non-finite");
        }
        return stats;
    }

    // Use the existing `x` buffer contents as the initial guess.
    //
    // This is important for coupled solvers where `x` stores the current iterate (absolute
    // unknown values, not a correction). Starting from a good initial guess can dramatically
    // reduce the number of Krylov iterations needed to reach tight tolerances.
    //
    // Note: wgpu ensures newly-created buffers are initialized before use, so the first solve
    // still effectively starts from x0=0 unless something has written into `x`.

    let max_iters = max_iters.max(1);
    let capacity = krylov.fgmres.max_restart();
    let restart_len = max_restart.max(1).min(capacity);

    let mut params = RawFgmresParams {
        n,
        num_cells,
        num_iters: 0,
        omega: 1.0,
        dispatch_x: dispatch.dofs_dispatch_x_threads,
        max_restart: restart_len as u32,
        column_offset: 0,
        _pad3: 0,
    };

    // Initial parameter write (required for vector ops used before the first solve_once call).
    {
        let core = krylov.fgmres.core(&context.device, &context.queue);
        write_params(&core, &params);
    }

    // Preconditioner setup
    krylov.precond.prepare(
        &context.device,
        &context.queue,
        &krylov.fgmres,
        system.rhs().as_entire_binding(),
        dispatch.grids,
    );

    let mut total_iters: u32 = 0;
    let mut residual = rhs_norm;
    let mut converged = false;
    let mut rel_scale: Option<f32> = None;

    while total_iters < max_iters {
        // Seed basis0 with the current residual (rhs - A*x).
        residual = {
            let core = krylov.fgmres.core(&context.device, &context.queue);
            krylov.fgmres.compute_residual_norm_into(
                &core,
                system,
                krylov.fgmres.basis_binding(0),
                "Generic FGMRES",
            )
        };
        if !residual.is_finite() {
            let stats = LinearSolverStats {
                iterations: total_iters,
                residual,
                converged: false,
                diverged: true,
                time: start.elapsed(),
            };
            if debug_fgmres {
                eprintln!("[cfd2][fgmres] diverged: residual non-finite at iters={total_iters}");
            }
            return stats;
        }
        if rel_scale.is_none() {
            // Avoid declaring convergence purely because the RHS happens to have a much larger
            // norm than the current residual (e.g., when `x` is already close for the
            // dominant-magnitude unknowns). Use the smaller of ||b|| and ||r0|| for the
            // relative tolerance scale.
            rel_scale = Some(rhs_norm.min(residual));
        }
        let rel_scale = rel_scale.unwrap();

        if residual <= tol * rel_scale || residual <= tol_abs {
            converged = true;
            break;
        }

        let remaining = (max_iters - total_iters) as usize;
        let iter_restart = restart_len.min(remaining).max(1);
        params.max_restart = iter_restart as u32;

        // Normalize basis0 and initialize restart-local aux buffers.
        {
            let core = krylov.fgmres.core(&context.device, &context.queue);
            write_params(&core, &params);
            krylov.fgmres.scale_in_place(
                &core,
                krylov.fgmres.basis_binding(0),
                1.0 / residual,
                "Generic FGMRES basis0 normalize",
            );
            krylov.fgmres.clear_restart_aux(&core);
        }
        krylov.fgmres.write_g0(&context.queue, residual);

        let iter_params = IterParams {
            current_idx: 0,
            max_restart: iter_restart as u32,
            _pad1: 0,
            _pad2: 0,
        };

        let solve = krylov.solve_once(SolveOnceArgs {
            context,
            system,
            rhs_norm: rel_scale,
            params,
            iter_params,
            config: FgmresSolveOnceConfig {
                tol_rel: tol,
                tol_abs,
                reset_x_before_update: false,
            },
            dispatch: dispatch.grids,
            precond_label,
        });

        total_iters = total_iters.saturating_add(solve.basis_size as u32);
        residual = solve.residual_est;
        if solve.converged {
            converged = true;
            break;
        }
    }

    if !converged {
        // Provide an accurate residual for reporting when the iteration cap is hit.
        residual = {
            let core = krylov.fgmres.core(&context.device, &context.queue);
            krylov.fgmres.compute_residual_norm_into(
                &core,
                system,
                krylov.fgmres.basis_binding(0),
                "Generic FGMRES final",
            )
        };
    }

    let stats = LinearSolverStats {
        iterations: total_iters,
        residual,
        converged,
        diverged: false,
        time: start.elapsed(),
    };
    if debug_fgmres {
        let rel_scale = rel_scale.unwrap_or(rhs_norm);
        eprintln!(
            "[cfd2][fgmres] done: iters={} rhs_norm={:.3e} rel_scale={:.3e} residual={:.3e} converged={}",
            stats.iterations, rhs_norm, rel_scale, stats.residual, stats.converged
        );
    }
    stats
}

// NOTE: Prefer `CFD2_DEBUG_FGMRES=1` over hardcoded debug logging.
