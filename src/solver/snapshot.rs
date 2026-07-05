//! Solver state snapshot / restore (M2 Tier B stage 3).
//!
//! A [`SolverStateSnapshot`] captures the *stepping state* of a solver — every
//! buffer + scalar counter the next `step()` reads before it writes — so a
//! fresh solver built on the same (or a remeshed, same-cell-count) mesh can be
//! restored to reproduce the run. It is the foundation for the M4 remesh loop:
//! snapshot → rebuild on the new mesh → restore the cell-indexed state.
//!
//! ## What the next step reads (inventory, CPU `CpuSolver::step`)
//!
//! - **`state`** (current) — rotated into `state_old` at the top of the step.
//! - **`state_old`, `state_old_old`** — the BDF2 time history (Euler uses only
//!   `state_old`; `state_old_old` matters once `step_count >= 1`).
//! - **`x`** — the warm-start / persisted linear-solve solution. The FGMRES /
//!   BiCGSTAB Krylov *basis* (V, H, Givens rotations) is allocated fresh inside
//!   each `linear_solve` call and never persists across steps, so `x` is the
//!   **entire** cross-step linear-solver state (roadmap open question 1: the
//!   solves cold-start every step apart from this warm guess — verified in
//!   `cpu::solver::linear_solve`, which marshals `x` out of the buffer and back).
//! - **`cell_vols_old`, `cell_vols_old_old`** — the ALE moving-volume history
//!   (a numeric no-op for static models, whose kernels never bind them).
//! - **`mesh_fluxes`** — the face-indexed ALE swept rates (zero on static
//!   meshes). Face-indexed, so restore is length-guarded: it only lands when the
//!   target solver has the same face count (a remesh recomputes them anyway).
//! - **scalars** `time, dt, dt_old, dtau, step_count` — `step_count` is
//!   load-bearing: it selects the BDF2→Euler startup fallback (step 0), so a
//!   fresh solver (`step_count == 0`) restored mid-run must carry the real count
//!   or its first restored step would wrongly fall back to Euler.
//! - **`schur_amg_active`** — the CPU Schur inner-solver adaptivity flip
//!   (Jacobi→AMG). It changes the *inner* solve path, hence the byte pattern of
//!   the produced `x`, so it is captured for byte-exact reproduction. (The AMG
//!   *hierarchy* itself is rebuilt deterministically from the CSR pattern +
//!   pressure values, so whether it is "already built" does not change results —
//!   only `schur_amg_active` does.)
//!
//! ## Excluded (derived — written before read each step, so irrelevant)
//!
//! `state_iter`, `grad_state`, `fluxes`, `rhs`, `y`, `matrix_values`: all
//! recomputed from `state` at the head of each outer iteration. Model
//! *configuration* (viscosity, density, alpha, EOS, advection/time scheme, BC
//! seeds) is NOT in the snapshot — the caller re-applies it to the fresh solver
//! via the same setters/`apply_params` it used at build (a snapshot restores
//! *state*, not the model).
//!
//! ## Backend completeness
//!
//! The **CPU** backend fills every field (`has_history == true`) and its restore
//! reproduces the next step byte-identically (CPU `read_state_f32` is an exact
//! f32-bit marshal — gated in `tests/mesh_refresh_identity_test.rs`).
//!
//! The **GPU** backend currently captures the *current state* + scalar counters
//! only (`has_history == false`): the history / warm-start / volume buffers have
//! no readback plumbing yet, so a GPU restore re-seeds the history from the
//! current state (IC semantics) — exact for single-step schemes, a documented
//! startup-fallback for BDF2. Full GPU history capture is a later stage.
//!
//! ## M5 stage 2 — the moving-mesh loop does NOT need GPU history readback
//!
//! The [`crate::sim::MovingMeshDriver`] preserves ALL cross-step state
//! *surgically in place* across the per-step topology refresh — it never uses
//! `snapshot`/`restore`. On the GPU backend:
//!   * `state`, `state_old`, `state_old_old` are cell-indexed field buffers the
//!     topology refresh never reallocates (only the face-flux buffer is resized),
//!     so the BDF2 time history survives a topology change untouched;
//!   * the ALE volume history `cell_vols_old{,_old}` is rotated then carried over
//!     by swap in `MeshResources::refresh_topology`;
//!   * the warm-start `x` is blitted across the reallocation (M5 stage 1).
//! So the moving loop preserves BDF2 history across steps with NO readback — a
//! device→device path strictly cheaper than a snapshot round-trip. This is
//! proven by `tests/gpu_moving_mesh_test.rs::gpu_moving_mesh_gcl_bdf2` (a lost
//! `state_old_old` would silently cold-start BDF2 to Euler each step and the GCL
//! drift would compound; it does not). GPU `has_history==true` readback is only
//! needed for a snapshot/restore that crosses a *rebuild boundary* (e.g. a
//! fresh-build-equivalence check), which the moving loop does not exercise, and
//! remains deferred.

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
