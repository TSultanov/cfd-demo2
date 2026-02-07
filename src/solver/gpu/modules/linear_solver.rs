use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::linear_solver::fgmres::{
    write_params, FgmresSolveOnceConfig, IterParams, RawFgmresParams,
};
use crate::solver::gpu::modules::krylov_precond::{FgmresPreconditionerModule, KrylovDispatch};
use crate::solver::gpu::modules::krylov_solve::{
    EncodeSolveOnceArgs, KrylovSolveModule, SolveOnceArgs,
};
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
    pub use_encoded_seed_basis0: bool,
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
        use_encoded_seed_basis0,
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
        if debug_fgmres {
            eprintln!("[cfd2][fgmres] early-exit: rhs_norm non-finite");
        }
        return LinearSolverStats::diverged(0, rhs_norm, start.elapsed());
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

    let mut total_iters: u32 = 0;
    let mut residual = rhs_norm;
    let mut converged = false;
    let mut rel_scale: Option<f32> = None;
    let mut precond_prepared = false;

    while total_iters < max_iters {
        let remaining = (max_iters - total_iters) as usize;
        let iter_restart = restart_len.min(remaining).max(1);
        params.max_restart = iter_restart as u32;

        // Seed basis0 with the current residual (rhs - A*x).
        //
        // Even when using the encoded-seed restart body, we still evaluate ||r0|| on host here
        // so convergence checks and relative scaling match the non-encoded path.
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
            if debug_fgmres {
                eprintln!("[cfd2][fgmres] diverged: residual non-finite at iters={total_iters}");
            }
            return LinearSolverStats::diverged(total_iters, residual, start.elapsed());
        }
        if rel_scale.is_none() {
            // Avoid declaring convergence purely because the RHS happens to have a much larger
            // norm than the current residual (e.g., when `x` is already close for the
            // dominant-magnitude unknowns). Use the smaller of ||b|| and ||r0|| for the
            // relative tolerance scale.
            rel_scale = Some(rhs_norm.min(residual));
        }
        let rel_scale_for_restart = rel_scale.unwrap();

        if residual <= tol * rel_scale_for_restart || residual <= tol_abs {
            converged = true;
            break;
        }

        if !use_encoded_seed_basis0 {
            // Non-encoded path seeds and normalizes basis0 on host before the restart body.
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
        }

        let iter_params = IterParams {
            current_idx: 0,
            max_restart: iter_restart as u32,
            _pad1: 0,
            _pad2: 0,
        };

        let prepare_this_restart = !precond_prepared;
        let solve = krylov.solve_once_with_prepare(
            SolveOnceArgs {
                context,
                system,
                rhs_norm: rel_scale_for_restart,
                params,
                iter_params,
                config: FgmresSolveOnceConfig {
                    tol_rel: tol,
                    tol_abs,
                    reset_x_before_update: false,
                },
                dispatch: dispatch.grids,
                precond_label,
            },
            prepare_this_restart,
            use_encoded_seed_basis0,
        );
        if prepare_this_restart {
            precond_prepared = true;
        }

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

    let stats = if converged {
        LinearSolverStats::converged(total_iters, residual, start.elapsed())
    } else {
        LinearSolverStats::max_iterations(total_iters, residual, start.elapsed())
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

/// Encode a fixed-budget FGMRES solve into an existing command encoder.
///
/// This path is intended for one-submission outer-loop batching where host readbacks inside the
/// linear solve must be avoided. It does not perform host-side residual checks between restarts.
pub fn encode_solve_fgmres_fixed_iterations<P: FgmresPreconditionerModule>(
    krylov: &mut KrylovSolveModule<P>,
    args: SolveFgmresArgs<'_>,
    encoder: &mut wgpu::CommandEncoder,
) -> LinearSolverStats {
    let SolveFgmresArgs {
        context,
        system,
        n,
        num_cells,
        dispatch,
        max_restart,
        max_iters,
        tol: _tol,
        tol_abs: _tol_abs,
        precond_label,
        use_encoded_seed_basis0,
    } = args;
    let start = Instant::now();

    let max_iters = max_iters.max(1);
    let capacity = krylov.fgmres.max_restart();
    let restart_len = max_restart.max(1).min(capacity);
    let restart_budget = std::env::var("CFD2_ONE_SUBMISSION_RESTART_BUDGET")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(12)
        .max(1);
    let total_iter_budget = std::env::var("CFD2_ONE_SUBMISSION_TOTAL_ITERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(27)
        .max(1);
    let iter_restart = restart_len
        .min(max_iters as usize)
        .min(restart_budget)
        .max(1);
    let total_iters_to_encode = (max_iters as usize).min(total_iter_budget).max(1);
    let min_tail_chunk = std::env::var("CFD2_ONE_SUBMISSION_MIN_TAIL")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);
    let one_submission_solution_omega = std::env::var("CFD2_ONE_SUBMISSION_SOLUTION_OMEGA")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(1.0)
        .clamp(0.0, 2.0);
    let one_submission_tail_omega = std::env::var("CFD2_ONE_SUBMISSION_TAIL_OMEGA")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(one_submission_solution_omega)
        .clamp(0.0, 2.0);
    let explicit_chunks: Option<Vec<usize>> = std::env::var("CFD2_ONE_SUBMISSION_CHUNKS")
        .ok()
        .and_then(|raw| {
            let parsed: Vec<usize> = raw
                .split(',')
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .filter(|&v| v > 0)
                .collect();
            if parsed.is_empty() {
                None
            } else {
                Some(parsed)
            }
        });
    let has_explicit_chunks = explicit_chunks.is_some();

    let mut params = RawFgmresParams {
        n,
        num_cells,
        num_iters: 0,
        omega: one_submission_solution_omega,
        dispatch_x: dispatch.dofs_dispatch_x_threads,
        max_restart: 0,
        column_offset: 0,
        _pad3: 0,
    };

    let mut chunk_sizes: Vec<usize> = if let Some(mut chunks) = explicit_chunks {
        let mut normalized: Vec<usize> = Vec::new();
        let mut remaining = total_iters_to_encode;
        for c in chunks.drain(..) {
            if remaining == 0 {
                break;
            }
            let chunk = c.min(iter_restart).min(remaining).max(1);
            normalized.push(chunk);
            remaining -= chunk;
        }
        while remaining > 0 {
            let chunk = iter_restart.min(remaining).max(1);
            normalized.push(chunk);
            remaining -= chunk;
        }
        normalized
    } else {
        let mut defaults: Vec<usize> = Vec::new();
        let mut remaining = total_iters_to_encode;
        while remaining > 0 {
            let chunk = iter_restart.min(remaining).max(1);
            defaults.push(chunk);
            remaining -= chunk;
        }
        defaults
    };
    if !has_explicit_chunks && chunk_sizes.len() >= 2 {
        let last_idx = chunk_sizes.len() - 1;
        if chunk_sizes[last_idx] < min_tail_chunk {
            let mut need = min_tail_chunk - chunk_sizes[last_idx];
            for donor_idx in 0..last_idx {
                if need == 0 {
                    break;
                }
                let donor_can_give = chunk_sizes[donor_idx].saturating_sub(1);
                if donor_can_give == 0 {
                    continue;
                }
                let give = donor_can_give.min(need);
                chunk_sizes[donor_idx] -= give;
                chunk_sizes[last_idx] += give;
                need -= give;
            }
        }
    }

    let mut encoded_total = 0usize;
    let chunk_count = chunk_sizes.len();
    for (chunk_idx, &chunk_restart) in chunk_sizes.iter().enumerate() {
        params.max_restart = chunk_restart as u32;
        params.omega = if chunk_idx + 1 == chunk_count {
            one_submission_tail_omega
        } else {
            one_submission_solution_omega
        };
        let iter_params = IterParams {
            current_idx: 0,
            max_restart: chunk_restart as u32,
            _pad1: 0,
            _pad2: 0,
        };
        let is_first_chunk = chunk_idx == 0;
        let encoded = krylov.encode_solve_once_with_prepare(
            EncodeSolveOnceArgs {
                context,
                system,
                rhs_norm: 1.0,
                params,
                iter_params,
                config: FgmresSolveOnceConfig {
                    tol_rel: 0.0,
                    tol_abs: 0.0,
                    reset_x_before_update: false,
                },
                dispatch: dispatch.grids,
                precond_label,
                capture_solver_scalars: false,
            },
            encoder,
            is_first_chunk,
            use_encoded_seed_basis0,
        );
        let consumed = encoded.max(1);
        encoded_total = encoded_total.saturating_add(consumed);
    }

    LinearSolverStats::max_iterations(encoded_total as u32, f32::INFINITY, start.elapsed())
}
