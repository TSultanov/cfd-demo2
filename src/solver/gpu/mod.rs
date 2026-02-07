pub mod bindings;
pub mod buffers;
pub mod context;
pub mod csr;
pub mod dispatch_counter;
pub mod enums;
pub mod execution_plan;
#[cfg(feature = "profiling")]
pub mod gpu_timestamp_profiler;
pub mod init;
pub mod linear_solver;
pub(crate) mod lowering;
pub mod modules;
pub mod profiling;
pub(crate) mod program;
pub mod readback;
pub mod recipe;
pub(crate) mod runtime;
pub(crate) mod runtime_common;
pub mod structs;
pub mod submission_counter;
pub mod unified_solver;
pub(crate) mod wgsl_reflect;

pub use recipe::{LinearSolverSpec, SolverRecipe, TimeIntegrationSpec};
pub use unified_solver::{FgmresSizing, GpuUnifiedSolver, SolverConfig, UiPortSet};
