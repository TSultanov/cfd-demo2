pub mod bc_table;
pub mod coeff_expr;
pub mod constants;
pub mod coupled_common;
pub mod dsl;
pub mod fusion;
pub mod generic_coupled_kernels;
pub mod infrastructure_kernels;
pub mod ir;
pub mod kernel_wgsl;
pub mod packed_state_gradients;
pub mod plan;
pub mod primitive_expr;
pub mod reconstruction;
pub mod rhs_only;
pub mod rust_emit;
pub mod state_access;
pub mod time_integration;
pub mod unified_assembly;
pub mod wgsl;
pub mod wgsl_ast;
pub mod wgsl_bindings;
pub mod wgsl_dsl;

pub use ir::{
    lower_system, lower_system_unchecked, DiscreteEquation, DiscreteOp, DiscreteOpKind,
    DiscreteSystem,
};
pub use kernel_wgsl::{BindingDesc, KernelWgsl};
pub use packed_state_gradients::generate_packed_state_gradients_kernel_program;
pub use packed_state_gradients::generate_packed_state_gradients_wgsl;
pub use wgsl::{generate_wgsl, generate_wgsl_library};
