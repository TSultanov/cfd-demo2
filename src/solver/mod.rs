pub use cfd2_codegen::solver::codegen;
pub use cfd2_codegen::compiler;
pub mod gpu;
pub(crate) mod ir;
pub mod kernels;
pub mod mesh;
pub mod model;
pub mod scheme;
pub mod shared;
pub mod units;

pub use gpu::{FgmresSizing, GpuUnifiedSolver as UnifiedSolver, SolverConfig};
pub use gpu::enums::{GpuBcKind, GpuBoundaryType, GpuLowMachPrecondModel, TimeScheme};
pub use gpu::profiling::{ProfileCategory, ProfilingStats};
pub use gpu::recipe::SteppingMode;
pub use gpu::structs::{LinearSolverStats, PreconditionerType};
