pub mod backend;
pub use cfd2_codegen::solver::codegen;
pub use cfd2_codegen::compiler;
pub mod gpu;
pub(crate) mod ir;
pub mod kernels;
pub mod mesh;
pub mod model;
pub mod options;
pub mod profiling;
pub mod scheme;
pub mod shared;
pub mod units;

pub use backend::{FgmresSizing, SolverConfig, UnifiedSolver};
