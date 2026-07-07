//! Backend-agnostic, GUI-independent solver driver shared by the desktop GUI
//! worker and the headless tests.
//!
//! The inner step (`UnifiedSolver::step`/`step_with_stats`) is already unified
//! across the CPU and GPU backends. This module unifies the *driver layer*
//! wrapped around it — timestep control (incl. the acoustic-aware adaptive dt),
//! runtime parameter application, the initial/boundary-condition setup, and the
//! divergence / steady-state stop detection — so the GUI and the tests run the
//! exact same physics-affecting code path instead of two hand-written copies.
//!
//! It depends only on `crate::solver` types (never on `egui`/`eframe` or the
//! `ui`-gated `Fluid`), so it is usable from `meshgen`-only tests.

mod driver;
#[cfg(feature = "cpu")]
mod mass_projection;
mod moving_mesh_driver;
mod outcome;
mod params;

pub use driver::{DriverBuild, SolverDriver};
pub use moving_mesh_driver::{
    BoundaryMotionSpec, MeshMotionSpec, MovingMeshDriver, MovingMeshStats, OscAxis,
    RegenBackend, ResizeTiming, ADAPT_BUDGET_MAX_FACTOR, DEFAULT_MESH_CFL,
    OSC_AMPLITUDE_CELL_FRACTION,
};
pub use outcome::{DivergeReason, FieldStats, Readback, RunResult, StepOutcome};
pub use params::RuntimeParams;
