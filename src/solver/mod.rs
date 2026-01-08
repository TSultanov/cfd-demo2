mod gpu;
mod codegen;
pub mod backend;
pub mod compiler;
pub mod kernels;
pub mod model;
pub mod mesh;
pub mod options;
pub mod profiling;
pub mod scheme;
pub mod units;

pub use backend::{FgmresSizing, SolverConfig, UnifiedSolver};
