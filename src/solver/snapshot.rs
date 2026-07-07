//! Solver state snapshot / restore.
//!
//! A [`SolverStateSnapshot`] captures the *stepping state* of a solver — every
//! buffer + scalar counter the next `step()` reads before it writes — so a
//! fresh solver built on the same (or a remeshed, same-cell-count) mesh can be
//! restored to reproduce the run.
//!
//! ## What the next step reads (CPU `CpuSolver::step`)
//!
//! - **`state`** (current) — rotated into `state_old` at the top of the step.
//! - **`state_old`, `state_old_old`** — the BDF2 time history (Euler uses only
//!   `state_old`; `state_old_old` matters once `step_count >= 1`).
//! - **`x`** — the warm-start linear-solve solution. The Krylov basis is
//!   allocated fresh inside each `linear_solve` and never persists, so `x` is
//!   the entire cross-step linear-solver state.
//! - **`cell_vols_old`, `cell_vols_old_old`** — the ALE moving-volume history
//!   (a numeric no-op for static models, whose kernels never bind them).
//! - **`mesh_fluxes`** — face-indexed ALE swept rates (zero on static meshes);
//!   restore is length-guarded on face count (a remesh recomputes them anyway).
//! - **scalars** `time, dt, dt_old, dtau, step_count` — `step_count` selects the
//!   BDF2→Euler startup fallback (step 0), so a restored solver must carry the
//!   real count or its first step would wrongly fall back to Euler.
//! - **`schur_amg_active`** — the CPU Schur inner-solver flip (Jacobi→AMG); it
//!   changes the inner solve path, hence the byte pattern of `x`, so it is
//!   captured for byte-exact reproduction. (The AMG hierarchy itself is rebuilt
//!   deterministically from the CSR pattern + pressure values.)
//!
//! ## Excluded (derived — written before read each step)
//!
//! `state_iter`, `grad_state`, `fluxes`, `rhs`, `y`, `matrix_values`: all
//! recomputed from `state` at the head of each outer iteration. Model
//! *configuration* (viscosity, density, alpha, EOS, schemes, BC seeds) is NOT in
//! the snapshot — the caller re-applies it to the fresh solver via the same
//! setters/`apply_params` it used at build (a snapshot restores state, not the
//! model).
//!
//! ## Backend completeness
//!
//! BOTH backends fill every field (`has_history == true`). The **CPU**
//! restore reproduces the next step byte-identically. The **GPU** snapshot
//! reads back all three ping-pong state levels, the warm-start `x`, the
//! volume history and the mesh fluxes, and its restore re-uploads them plus
//! the scalar counters — the seam the moving-mesh resize discipline (BDF
//! continuity + two-level re-solve) rides on either backend. Two CPU-only
//! diagnostics have no GPU analog: `schur_amg_active` (the GPU Schur
//! Chebyshev→AMG cadence counter lives inside the preconditioner and resets
//! with any rebuild) and `last_rel_delta`; a GPU-origin restore therefore
//! resumes the run exactly in stepping state, while the first linear solve
//! after a REBUILD may take a different (equally converged) iterate path.
//! Restoring a `has_history == false` capture re-seeds the history from the
//! current state (IC semantics) — exact for single-step schemes, a
//! startup-fallback for BDF2.

/// Captured stepping state of a solver — see the module docs for the inventory
/// and the exclusion rationale.
#[derive(Debug, Clone)]
pub struct SolverStateSnapshot {
    /// Cell count the snapshot was taken at (the restore invariant).
    pub num_cells: usize,
    /// Face count at snapshot time (guards the face-indexed `mesh_fluxes`
    /// restore; `0` when the backend did not capture it).
    pub num_faces: usize,
    /// Packed-state stride (floats per cell) — restore requires a match.
    pub state_stride: u32,
    /// Coupled unknowns per cell (width of `x`); `0` for scalar models.
    pub unknowns_per_cell: usize,

    // ── cell-indexed buffers (exact f32 bits) ──
    /// Current packed state, `num_cells * state_stride`.
    pub state: Vec<f32>,
    /// One-step-old history (`num_cells * state_stride`); empty if `!has_history`.
    pub state_old: Vec<f32>,
    /// Two-steps-old history; empty if `!has_history`.
    pub state_old_old: Vec<f32>,
    /// Warm-start linear-solve solution, `num_cells * unknowns_per_cell`; empty
    /// if `!has_history`.
    pub x: Vec<f32>,
    /// CURRENT cell volumes (`num_cells`); empty if `!has_history` or on
    /// captures predating the field. The ALE seam's volume-history ROTATION
    /// input — an implicit-mesh-motion retry must rewind it along with the
    /// history, or the next attempt rotates a stale trial mesh's volumes
    /// into `V^n`.
    pub cell_vols: Vec<f32>,
    /// ALE volume history `V^n` (`num_cells`); empty if `!has_history`.
    pub cell_vols_old: Vec<f32>,
    /// ALE volume history `V^{n-1}` (`num_cells`); empty if `!has_history`.
    pub cell_vols_old_old: Vec<f32>,
    /// ALE per-face swept rates (`num_faces`); empty if `!has_history` or on a
    /// static (non-ALE) capture where it is uniformly zero.
    pub mesh_fluxes: Vec<f32>,

    // ── scalar counters ──
    pub time: f32,
    pub dt: f32,
    pub dt_old: f32,
    pub dtau: f32,
    pub step_count: u64,
    pub last_rel_delta: f64,
    /// CPU Schur inner-solver adaptivity flip (Jacobi→AMG); `false` on GPU.
    pub schur_amg_active: bool,

    /// `true` when the full time/warm-start/volume history was captured (CPU);
    /// `false` for a current-state-only capture (GPU) whose restore re-seeds the
    /// history from `state`.
    pub has_history: bool,
}

impl SolverStateSnapshot {
    /// Error string if this snapshot's cell layout does not match a solver with
    /// `num_cells` / `state_stride`. Shared by both backends' `restore`.
    pub fn check_compatible(&self, num_cells: usize, state_stride: u32) -> Result<(), String> {
        if self.num_cells != num_cells {
            return Err(format!(
                "snapshot restore requires an unchanged cell count ({} != {})",
                self.num_cells, num_cells
            ));
        }
        if self.state_stride != state_stride {
            return Err(format!(
                "snapshot restore requires an unchanged state stride ({} != {})",
                self.state_stride, state_stride
            ));
        }
        if self.state.len() != num_cells * state_stride as usize {
            return Err(format!(
                "snapshot state length {} != num_cells {} * stride {}",
                self.state.len(),
                num_cells,
                state_stride
            ));
        }
        Ok(())
    }
}
