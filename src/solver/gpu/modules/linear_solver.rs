//! Linear solver encoding helpers for one-submission outer-loop batching.
//!
//! This module provides three families of linear solver execution:
//!
//! 1. **Host-driven** (`solve_fgmres`) — Standard FGMRES solve with host readbacks
//!    between restarts.  Used for the per-iteration recipe-level path.
//!
//! 2. **Encoded** (`encode_solve_fgmres_fixed_iterations`) — Encodes a fixed budget
//!    of FGMRES restarts into a caller-provided `CommandEncoder` without host
//!    readbacks.  Used for parity testing and the single-encoder batched path.
//!
//! 3. **Chunked submit** (`submit_solve_fgmres_fixed_iterations_chunked`,
//!    `submit_solve_cg_fixed_iterations_chunked`) — Splits the iteration budget
//!    across multiple encoder→submit cycles to avoid Metal backend command buffer
//!    size limits.  Used by the one-submission outer-loop orchestration.
//!
//! All three share the same `FgmresChunkLayout` computation for consistency.

use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::linear_solver::fgmres::{
    encode_fgmres_seed_basis0_from_system, encode_restart_guard, write_params,
    FgmresSolveOnceConfig, IterParams, RawFgmresParams, FGMRES_SCALAR_COUNT,
};
use crate::solver::gpu::modules::krylov_precond::{KrylovDispatch, PreconditionerModule};
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

#[derive(Debug, Clone)]
struct OneSubmissionEnvTunables {
    restart_budget: Option<usize>,
    total_iter_budget: Option<usize>,
    min_tail_chunk: Option<usize>,
    cg_chunk_size: Option<usize>,
}

impl OneSubmissionEnvTunables {
    fn from_env() -> Self {
        Self {
            restart_budget: parse_usize_env("CFD2_ONE_SUBMISSION_RESTART_BUDGET"),
            total_iter_budget: parse_usize_env("CFD2_ONE_SUBMISSION_TOTAL_ITERS"),
            min_tail_chunk: parse_usize_env("CFD2_ONE_SUBMISSION_MIN_TAIL"),
            cg_chunk_size: parse_usize_env("CFD2_ONE_SUBMISSION_CG_CHUNK_SIZE"),
        }
    }
}

/// CGS2 re-orthogonalization A/B switch (default OFF while measurements
/// accumulate; see the roadmap's Arc 2). Read per solve, not cached: a
/// process-wide cache froze the first test's environment (see
/// `one_submission_env_tunables`).
fn cgs2_enabled() -> bool {
    std::env::var("CFD2_FGMRES_CGS2")
        .map(|v| v != "0")
        .unwrap_or(false)
}

/// Stall-stop level factor (0.0 disables; see the stall branch in
/// gmres_logic/restart_guard and the host loop in `solve_fgmres`).
/// Default OFF while measurements accumulate (roadmap Arc 3). Read per
/// solve, not cached (see `one_submission_env_tunables`).
fn stall_level_rel() -> f32 {
    std::env::var("CFD2_FGMRES_STALL_REL")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.0)
}

/// Linear-solve relative-tolerance override (sweep/diagnostic knob for the
/// inexact-Picard tolerance arc). When set, overrides the model-declared
/// FGMRES relative tolerance at every solve entry point. Read per solve,
/// not cached (see `one_submission_env_tunables`).
fn lin_tol_override() -> Option<f32> {
    std::env::var("CFD2_LIN_TOL")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| *v > 0.0)
}

fn parse_usize_env(key: &str) -> Option<usize> {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
}

/// Read the one-submission env tunables.
///
/// Deliberately *not* a process-global cache: both call sites run once per linear
/// solve (chunk-layout computation), so the env reads are cheap there, and a
/// process-wide `OnceLock` froze the first test's environment for the whole test
/// process, making env-guard-based tests order-dependent.
fn one_submission_env_tunables() -> OneSubmissionEnvTunables {
    OneSubmissionEnvTunables::from_env()
}

/// Pre-computed FGMRES chunk layout for encoded/one-submission solves.
struct FgmresChunkLayout {
    params: RawFgmresParams,
    chunk_sizes: Vec<usize>,
}

/// Compute the FGMRES iteration chunk sizes and base params for encoded solves.
fn compute_fgmres_chunk_layout(
    n: u32,
    num_cells: u32,
    dispatch_x: u32,
    max_restart: usize,
    max_iters: u32,
    capacity: usize,
) -> FgmresChunkLayout {
    let max_iters = max_iters.max(1);
    let restart_len = max_restart.max(1).min(capacity);
    let tunables = one_submission_env_tunables();
    let restart_budget = tunables.restart_budget.unwrap_or(restart_len).max(1);
    let total_iter_budget = tunables
        .total_iter_budget
        .unwrap_or(max_iters as usize)
        .max(1);
    let iter_restart = restart_len
        .min(max_iters as usize)
        .min(restart_budget)
        .max(1);
    let total_iters_to_encode = (max_iters as usize).min(total_iter_budget).max(1);
    let min_tail_chunk = tunables.min_tail_chunk.unwrap_or(1).max(1);

    let params = RawFgmresParams {
        n,
        num_cells,
        num_iters: 0,
        omega: 1.0,
        dispatch_x,
        max_restart: 0,
        column_offset: 0,
        _pad3: 0,
    };

    let mut chunk_sizes: Vec<usize> = {
        let mut defaults: Vec<usize> = Vec::new();
        let mut remaining = total_iters_to_encode;
        while remaining > 0 {
            let chunk = iter_restart.min(remaining).max(1);
            defaults.push(chunk);
            remaining -= chunk;
        }
        defaults
    };
    if chunk_sizes.len() >= 2 {
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

    FgmresChunkLayout {
        params,
        chunk_sizes,
    }
}

pub fn solve_fgmres<P: PreconditionerModule>(
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
    let tol = lin_tol_override().unwrap_or(tol);
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

    // Restart-boundary monotonicity guard. A restart cycle whose f32 Arnoldi
    // basis lost orthogonality can APPLY an update that increases the true
    // residual; left unguarded this compounds across restarts (observed:
    // residual growth by orders of magnitude, ending in NaN, on the coupled
    // incompressible system — see tests/dp_diag_probe.rs). Track the
    // best-so-far x at the true-residual checkpoints and restore it when a
    // cycle made things worse.
    let mut best_residual = f32::INFINITY;
    let mut have_snapshot = false;
    const RESTART_GROWTH_TOL: f32 = 1.25;

    // Stall-stop (mirrors the GPU-side stall branch in
    // gmres_logic/restart_guard — keep the two in sync): with an
    // unreachable tolerance every solve burns to the iteration cap at its
    // f32 floor; stop when the checkpoint residual improves <2% twice in a
    // row AND is already below stall_rel * rel_scale. 0.0 disables.
    let stall_rel = stall_level_rel();
    let mut prev_checkpoint_residual: Option<f32> = None;
    let mut stall_count = 0u32;
    const STALL_IMPROVEMENT_TOL: f32 = 0.02;

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
            if have_snapshot {
                // The last cycle corrupted x; hand back the best iterate
                // instead of the non-finite one.
                let core = krylov.fgmres.core(&context.device, &context.queue);
                krylov
                    .fgmres
                    .restore_x(&core, system.x(), "Generic FGMRES x restore");
                return LinearSolverStats::max_iterations(
                    total_iters,
                    best_residual,
                    start.elapsed(),
                );
            }
            return LinearSolverStats::diverged(total_iters, residual, start.elapsed());
        }
        if residual < best_residual {
            best_residual = residual;
            let core = krylov.fgmres.core(&context.device, &context.queue);
            krylov
                .fgmres
                .snapshot_x(&core, system.x(), "Generic FGMRES x snapshot");
            have_snapshot = true;
        } else if have_snapshot && residual > best_residual * RESTART_GROWTH_TOL {
            if debug_fgmres {
                eprintln!(
                    "[cfd2][fgmres] restart residual grew {best_residual:.3e} -> {residual:.3e}; restoring best x and stopping"
                );
            }
            let core = krylov.fgmres.core(&context.device, &context.queue);
            krylov
                .fgmres
                .restore_x(&core, system.x(), "Generic FGMRES x restore");
            residual = best_residual;
            break;
        }
        if rel_scale.is_none() {
            // Avoid declaring convergence purely because the RHS happens to have a much larger
            // norm than the current residual (e.g., when `x` is already close for the
            // dominant-magnitude unknowns). Use the smaller of ||b|| and ||r0|| for the
            // relative tolerance scale.
            rel_scale = Some(rhs_norm.min(residual));
        }
        let rel_scale_for_restart = rel_scale
            .expect("rel_scale must be Some after the is_none guard above");

        if residual <= tol * rel_scale_for_restart || residual <= tol_abs {
            converged = true;
            break;
        }

        if stall_rel > 0.0 {
            if let Some(prev) = prev_checkpoint_residual {
                let no_improve = residual > prev * (1.0 - STALL_IMPROVEMENT_TOL);
                // Level against rel_scale (= min(||b||, ||r0||)), matching the
                // GPU path where clamp_rel_scale folds ||r0|| into RHS_NORM.
                let level_ok = residual <= stall_rel * rel_scale_for_restart;
                if no_improve && level_ok {
                    stall_count += 1;
                } else {
                    stall_count = 0;
                }
                if stall_count >= 2 {
                    if debug_fgmres {
                        eprintln!(
                            "[cfd2][fgmres] stall-stop at iters={total_iters} residual={residual:.3e}"
                        );
                    }
                    // x is the current iterate; the !converged epilogue
                    // recomputes the true residual and restores the best
                    // snapshot if this one is worse.
                    break;
                }
            }
            prev_checkpoint_residual = Some(residual);
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
                    enable_cgs2: cgs2_enabled(),
                    // The GPU-side MID-CYCLE stall applies to the encoded
                    // cycle bodies on every path; the host loop adds its own
                    // checkpoint stall and snapshot/restore guard on top
                    // (the GPU restart_guard stays off here).
                    stall_level_rel: stall_level_rel(),
                    enable_restart_guard: false,
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
        // The final (uncheckpointed) cycle may also have corrupted x.
        if have_snapshot && (!residual.is_finite() || residual > best_residual) {
            let core = krylov.fgmres.core(&context.device, &context.queue);
            krylov
                .fgmres
                .restore_x(&core, system.x(), "Generic FGMRES x restore final");
            residual = best_residual;
        }
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
pub fn encode_solve_fgmres_fixed_iterations<P: PreconditionerModule>(
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
        tol,
        tol_abs,
        precond_label,
        use_encoded_seed_basis0,
    } = args;
    let tol = lin_tol_override().unwrap_or(tol);
    let start = Instant::now();

    let FgmresChunkLayout {
        mut params,
        chunk_sizes,
    } = compute_fgmres_chunk_layout(
        n,
        num_cells,
        dispatch.dofs_dispatch_x_threads,
        max_restart,
        max_iters,
        krylov.fgmres.max_restart(),
    );

    let num_chunks = chunk_sizes.len();
    let mut encoded_total = 0usize;
    for (chunk_idx, &chunk_restart) in chunk_sizes.iter().enumerate() {
        params.max_restart = chunk_restart as u32;
        let iter_params = IterParams {
            current_idx: 0,
            max_restart: chunk_restart as u32,
            _pad1: 0,
            _pad2: 0,
        };
        let is_first_chunk = chunk_idx == 0;
        let is_last_chunk = chunk_idx + 1 == num_chunks;
        let preserve = !is_first_chunk;
        // Capture solver scalars on the last chunk so the caller can read back
        // the real residual and convergence flag after submission.
        let capture_scalars = is_last_chunk;
        let encoded = krylov.encode_solve_once_with_prepare(
            EncodeSolveOnceArgs {
                context,
                system,
                rhs_norm: 1.0, // GPU-resident; shader multiplies TOL_REL_RHS * scalars[RHS_NORM]
                params,
                iter_params,
                config: FgmresSolveOnceConfig {
                    tol_rel: tol,
                    tol_abs,
                    reset_x_before_update: false,
                    enable_cgs2: cgs2_enabled(),
                    stall_level_rel: stall_level_rel(),
                    // Encoded-seed path: GPU-side monotonicity guard.
                    enable_restart_guard: use_encoded_seed_basis0,
                },
                dispatch: dispatch.grids,
                precond_label,
                capture_solver_scalars: capture_scalars,
                preserve_convergence_state: preserve,
                // Compute ||rhs|| on GPU only for the first chunk, after seed_basis0
                // and scalars init.  Subsequent chunks preserve the value via the
                // preserve_convergence_state path.
                compute_rhs_norm_on_gpu: is_first_chunk,
            },
            encoder,
            is_first_chunk,
            use_encoded_seed_basis0,
        );
        let consumed = encoded.max(1);
        encoded_total = encoded_total.saturating_add(consumed);
    }

    // Note: this function only encodes into the caller's encoder — the actual
    // submission and readback happen externally.  The last chunk has
    // `capture_solver_scalars: true`, so after submitting the caller can use
    // `KrylovSolveModule::read_last_solver_stats` to get a real residual.
    // We still return a placeholder here because we cannot read back without
    // a submission index.
    LinearSolverStats::max_iterations(encoded_total as u32, f32::INFINITY, start.elapsed())
}

/// Like [`encode_solve_fgmres_fixed_iterations`], but submits a separate command buffer for each
/// restart chunk instead of encoding everything into one caller-provided encoder.
///
/// This avoids Metal/backend limits on command-buffer size (the `encoder.finish()` call can block
/// when tens of thousands of compute dispatches are recorded into a single buffer).  Each chunk
/// gets its own encoder → submit cycle, but the GPU-side solver state (buffers, convergence
/// flags) persists between submissions because the same device/queue is used.
///
/// The caller can also encode assembly/update graphs into the returned encoders via the
/// `pre_encode` and `post_encode` callbacks.
pub fn submit_solve_fgmres_fixed_iterations_chunked<P: PreconditionerModule>(
    krylov: &mut KrylovSolveModule<P>,
    args: SolveFgmresArgs<'_>,
    pre_encode: &mut dyn FnMut(&mut wgpu::CommandEncoder),
    post_encode: &mut dyn FnMut(&mut wgpu::CommandEncoder),
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
    let tol = lin_tol_override().unwrap_or(tol);
    let start = Instant::now();

    // Adaptive iteration budget: encode only ~what the previous solve
    // actually needed. Engages whenever solves stop early — convergence
    // (reachable tolerance), stall, or guard; the configured max_iters
    // always caps it, and a solve that exhausts its budget without
    // stopping doubles the next budget (AIMD). Inert at an unreachable
    // tolerance with the stall off: stopped_early never fires, so the
    // budget stays max_iters. The wall-time win is on the HOST side:
    // frozen chunks execute as GPU no-ops either way, but encoding them
    // is what dominates small-system solves.
    let restart_len_u32 = max_restart.max(1) as u32;
    let budget = krylov
        .adaptive_budget
        .map(|b| b.clamp(restart_len_u32, max_iters.max(1)))
        .unwrap_or(max_iters);

    let FgmresChunkLayout {
        mut params,
        chunk_sizes,
    } = compute_fgmres_chunk_layout(
        n,
        num_cells,
        dispatch.dofs_dispatch_x_threads,
        max_restart,
        budget,
        krylov.fgmres.max_restart(),
    );

    let num_chunks = chunk_sizes.len();
    let mut encoded_total = 0usize;
    let mut last_submission_index: Option<wgpu::SubmissionIndex> = None;
    for (chunk_idx, &chunk_restart) in chunk_sizes.iter().enumerate() {
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fgmres:one_submission_chunk"),
            });

        // Let the caller encode assembly/setup commands before the solve chunk.
        if chunk_idx == 0 {
            pre_encode(&mut encoder);
        }

        params.max_restart = chunk_restart as u32;
        let iter_params = IterParams {
            current_idx: 0,
            max_restart: chunk_restart as u32,
            _pad1: 0,
            _pad2: 0,
        };
        let is_first_chunk = chunk_idx == 0;
        let is_last_chunk = chunk_idx + 1 == num_chunks;
        let preserve = !is_first_chunk;
        // Capture solver scalars on the last chunk so we can read back a real
        // residual and convergence flag instead of returning f32::INFINITY.
        let capture_scalars = is_last_chunk;
        let encoded = krylov.encode_solve_once_with_prepare(
            EncodeSolveOnceArgs {
                context,
                system,
                rhs_norm: 1.0,
                params,
                iter_params,
                config: FgmresSolveOnceConfig {
                    tol_rel: tol,
                    tol_abs,
                    reset_x_before_update: false,
                    enable_cgs2: cgs2_enabled(),
                    stall_level_rel: stall_level_rel(),
                    // Encoded-seed path: GPU-side monotonicity guard.
                    enable_restart_guard: use_encoded_seed_basis0,
                },
                dispatch: dispatch.grids,
                precond_label,
                capture_solver_scalars: capture_scalars,
                preserve_convergence_state: preserve,
                compute_rhs_norm_on_gpu: is_first_chunk,
            },
            &mut encoder,
            is_first_chunk,
            use_encoded_seed_basis0,
        );
        let consumed = encoded.max(1);
        encoded_total = encoded_total.saturating_add(consumed);

        // Final verification: recompute the true residual and let the guard
        // restore the best-so-far x if the LAST cycle corrupted it (cycles
        // before the last are covered by the per-chunk guard at the next
        // seed). Must run before post_encode so the model's update kernels
        // consume the verified x. Re-capture scalars afterwards so the
        // readback reflects the guard's verdict.
        if is_last_chunk && use_encoded_seed_basis0 {
            let core = krylov.fgmres.core(&context.device, &context.queue);
            encode_fgmres_seed_basis0_from_system(&core, &mut encoder, system, 1, true);
            encode_restart_guard(&core, &mut encoder, system.x(), &params);
            encoder.copy_buffer_to_buffer(
                core.b_scalars,
                0,
                core.b_staging_scalar,
                0,
                (FGMRES_SCALAR_COUNT as u64) * 4,
            );
        }

        // Let the caller encode update/assembly commands after the last solve chunk.
        if is_last_chunk {
            post_encode(&mut encoder);
        }

        let sub_idx = context.queue.submit(Some(encoder.finish()));
        crate::count_submission!("Generic Coupled", "fgmres:one_submission_chunk");
        if is_last_chunk {
            last_submission_index = Some(sub_idx);
        }
    }

    // Read back solver scalars from the last chunk to produce a meaningful
    // LinearSolverStats with the real GPU-computed residual.
    if let Some(sub_idx) = last_submission_index {
        let (stats, info) =
            krylov.read_last_solver_stats(context, sub_idx, encoded_total as u32, start.elapsed());
        if std::env::var("CFD2_DEBUG_FGMRES").map(|v| v != "0").unwrap_or(false) {
            eprintln!(
                "[cfd2][fgmres][chunked] budget={budget} actual={} stopped_early={} resid={:.3e}",
                info.actual_iters, info.stopped_early, stats.residual
            );
        }
        let next = if info.stopped_early {
            // Stall/guard/convergence ended the solve: next budget =
            // actual work + one restart cycle of margin.
            (info.actual_iters.saturating_add(restart_len_u32))
                .clamp(restart_len_u32, max_iters.max(1))
        } else {
            // Budget exhausted without stopping: grow back quickly.
            budget.saturating_mul(2).clamp(restart_len_u32, max_iters.max(1))
        };
        krylov.adaptive_budget = Some(next);
        stats
    } else {
        LinearSolverStats::max_iterations(encoded_total as u32, f32::INFINITY, start.elapsed())
    }
}

/// Default number of CG iterations per submission chunk.
///
/// This is analogous to the FGMRES restart-budget chunking: splitting many
/// iterations across separate command-buffer submissions avoids Metal backend
/// hangs when tens of thousands of compute dispatches are recorded into one
/// command buffer.
const DEFAULT_CG_CHUNK_SIZE: usize = 50;

/// Submit a fixed-iteration CG solve in chunked submissions, with assembly/update
/// callbacks analogous to [`submit_solve_fgmres_fixed_iterations_chunked`].
///
/// Each chunk gets its own encoder → submit cycle.  `pre_encode` is called on the
/// first chunk's encoder (to record assembly dispatches) and `post_encode` is called
/// on the last chunk's encoder (to record update dispatches).
pub fn submit_solve_cg_fixed_iterations_chunked(
    cg: &crate::solver::gpu::modules::scalar_cg::ScalarCgModule,
    context: &GpuContext,
    n: u32,
    max_iters: u32,
    pre_encode: &mut dyn FnMut(&mut wgpu::CommandEncoder),
    post_encode: &mut dyn FnMut(&mut wgpu::CommandEncoder),
) -> LinearSolverStats {
    let start = Instant::now();
    let max_iters = max_iters.max(1) as usize;

    let chunk_size = one_submission_env_tunables()
        .cg_chunk_size
        .unwrap_or(DEFAULT_CG_CHUNK_SIZE)
        .max(1);

    // Build chunk layout.
    let mut chunk_sizes: Vec<usize> = Vec::new();
    let mut remaining = max_iters;
    while remaining > 0 {
        let chunk = chunk_size.min(remaining).max(1);
        chunk_sizes.push(chunk);
        remaining -= chunk;
    }

    let num_chunks = chunk_sizes.len();
    let mut last_submission_index: Option<wgpu::SubmissionIndex> = None;

    for (chunk_idx, &iters_in_chunk) in chunk_sizes.iter().enumerate() {
        let is_first_chunk = chunk_idx == 0;
        let is_last_chunk = chunk_idx + 1 == num_chunks;

        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cg:one_submission_chunk"),
            });

        // Assembly: prepend to the first chunk.
        if is_first_chunk {
            pre_encode(&mut encoder);
        }

        // Encode CG iterations for this chunk.
        cg.encode_solve_cg_fixed_iterations(
            context,
            &mut encoder,
            n,
            iters_in_chunk as u32,
            is_first_chunk, // include_init only on first chunk
            is_last_chunk,  // capture_scalars only on last chunk
        );

        // Update: append to the last chunk.
        if is_last_chunk {
            post_encode(&mut encoder);
        }

        let sub_idx = context.queue.submit(Some(encoder.finish()));
        crate::count_submission!("Generic Coupled", "cg:one_submission_chunk");
        if is_last_chunk {
            last_submission_index = Some(sub_idx);
        }
    }

    // Read back solver scalars from the last chunk.
    if let Some(sub_idx) = last_submission_index {
        cg.read_last_cg_stats(context, sub_idx, max_iters as u32, start.elapsed())
    } else {
        LinearSolverStats::max_iterations(max_iters as u32, f32::INFINITY, start.elapsed())
    }
}
