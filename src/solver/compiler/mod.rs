pub mod emit;

pub use emit::*;

// Re-export core codegen helpers (WGSL generators + lowering) for convenience.
pub use crate::solver::codegen::*;
