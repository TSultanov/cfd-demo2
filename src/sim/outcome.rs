//! Structured results the shared [`crate::sim::SolverDriver`] returns, so the GUI
//! worker and the headless tests react to divergence / steady-state / field bounds
//! through one code path instead of re-deriving them.

use crate::solver::LinearSolverStats;

/// Why a step is considered diverged.
#[derive(Debug, Clone)]
pub enum DivergeReason {
    /// The linear solve diverged or returned a non-finite / very large residual
    /// (mirrors the GUI worker's `linear_diverged` predicate).
    LinearSolver,
    /// A readback found non-finite velocity and/or pressure.
    NonFinite { u: usize, p: usize },
    /// `UnifiedSolver::step_with_stats` returned an error.
    StepError(String),
}

impl DivergeReason {
    /// Human-readable message matching the GUI worker's `SolverWorkerEvent::Error`
    /// strings, so the error surface is unchanged after routing through the driver.
    pub fn message(&self) -> String {
        match self {
            DivergeReason::LinearSolver => "divergence detected (linear solver)".to_string(),
            DivergeReason::NonFinite { u, p } => {
                format!("divergence detected (nonfinite u={u}, p={p})")
            }
            DivergeReason::StepError(err) => format!("solver step failed: {err}"),
        }
    }
}

/// Cheap field statistics computed when a step is asked to read back state.
#[derive(Debug, Clone, Copy)]
pub struct FieldStats {
    pub max_vel: f64,
    pub nonfinite_u: usize,
    pub p_min: f64,
    pub p_max: f64,
    pub p_finite: bool,
    pub nonfinite_p: usize,
    /// Compressible only: `(min, max)` density. `None` for incompressible — the
    /// driver only reads density when the model carries a `rho` equation.
    pub rho: Option<(f64, f64)>,
}

/// The result of one [`crate::sim::SolverDriver::step`].
///
/// `linear_stats` is moved out of `step_with_stats` for the caller's stats / trace.
/// `diverged` / `should_stop` are computed by the driver so neither the GUI nor the
/// tests re-derive them. `readback` is populated only on steps where readback was
/// requested; `outer_iters` is the post-step outer-iteration count (self-contained
/// for `run_steps` callers, who cannot borrow the driver inside their callback).
pub struct StepOutcome {
    pub dt: f32,
    /// Backend-local accepted-step latency. The generic driver times
    /// `step_with_stats`; structured explicit stepping also includes its
    /// mandatory completion/finite-state audit. Optional GUI snapshot packaging
    /// remains excluded. The GUI does not present this directly: asynchronous
    /// GPU/CPU comparison uses its completion-fenced worker throughput window.
    pub step_time_ms: f32,
    pub linear_stats: Vec<LinearSolverStats>,
    pub outer_iters: Option<u32>,
    /// Outer (Picard) residuals for the GUI "Coupled: U/P" readout. Populated by
    /// the structured banded path (whose solver is not a [`UnifiedSolver`], so the
    /// worker cannot read them off `step_stats()`); `None` on the unstructured path,
    /// where the worker sources them from the `UnifiedSolver` directly.
    pub outer_residual_u: Option<f32>,
    pub outer_residual_p: Option<f32>,
    pub diverged: Option<DivergeReason>,
    pub should_stop: bool,
    pub readback: Option<Readback>,
}

/// Field data read back from the solver on a readback step: the raw velocity and
/// pressure fields (the GUI ships these to the renderer / UI snapshot) plus the
/// summary [`FieldStats`] (which the tests assert on).
pub struct Readback {
    pub u: Vec<(f64, f64)>,
    pub p: Vec<f64>,
    pub stats: FieldStats,
}

/// Summary of a [`crate::sim::SolverDriver::run_steps`] loop.
///
/// `run_steps` stops early only on hard divergence or a caller `ControlFlow::Break`
/// (it does **not** auto-stop on `should_stop`, so it preserves the gate's
/// "run the full count unless something goes wrong" semantics); callers that want
/// steady-state auto-stop inspect `StepOutcome::should_stop` and return `Break`.
#[derive(Debug, Clone, Default)]
pub struct RunResult {
    /// Number of steps actually executed.
    pub executed_steps: usize,
    /// Set if a step hard-diverged (linear / non-finite / step error).
    pub diverged: Option<DivergeReason>,
    /// Step index at which the loop stopped early (divergence or a caller
    /// `ControlFlow::Break`); `None` if it ran the full requested count.
    pub stop_step: Option<usize>,
    /// The caller's callback requested an early stop.
    pub stopped_by_caller: bool,
}
