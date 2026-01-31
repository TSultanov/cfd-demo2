pub mod coeff_expr;
pub mod constants;
pub mod dsl;
pub mod generic_coupled_kernels;
pub mod ir;
pub mod kernel_wgsl;
pub mod packed_state_gradients;
pub mod plan;
pub mod primitive_expr;
pub mod reconstruction;
pub mod state_access;
pub mod unified_assembly;
pub mod wgsl;
pub mod wgsl_ast;
pub mod wgsl_dsl;

pub use kernel_wgsl::KernelWgsl;
pub use ir::{
    lower_system, lower_system_unchecked, DiscreteEquation, DiscreteOp, DiscreteOpKind, DiscreteSystem,
};
pub use packed_state_gradients::generate_packed_state_gradients_wgsl;
// Slot-based state access helpers are publicly available via state_access module
pub use wgsl::{generate_wgsl, generate_wgsl_library};
