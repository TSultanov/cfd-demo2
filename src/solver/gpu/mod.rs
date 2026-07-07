pub mod bindings;
pub mod buffers;
pub mod capacity;
pub mod context;
pub mod csr;
pub mod dispatch_counter;
pub mod enums;
pub mod execution_plan;
pub mod gpu_timer;
pub mod init;
pub mod linear_solver;
pub(crate) mod lowering;
pub mod modules;
pub mod pipeline_cache;
pub mod profiling_types;
pub mod profiling;
pub(crate) mod program;
pub mod readback;
pub mod recipe;
pub(crate) mod runtime;
pub(crate) mod runtime_common;
pub mod srd;
pub mod structs;
pub mod structured;
pub mod submission_counter;
pub mod unified_solver;
/// GPU meshless Voronoi engine; depends on the CPU engine's shared types, hence the `meshgen` gate.
#[cfg(feature = "meshgen")]
pub mod voronoi;
pub(crate) mod wgsl_reflect;

pub use program::plan_instance::OuterStepStatus;
pub use recipe::{LinearSolverSpec, SolverRecipe, TimeIntegrationSpec};
pub use unified_solver::{FgmresSizing, GpuUnifiedSolver, SolverConfig, UiPortSet};
