pub mod coeff_expr;
pub mod dsl;
pub mod flux_module;
pub mod generic_coupled_kernels;
pub mod ir;
pub mod plan;
pub mod primitive_expr;
pub mod reconstruction;
pub mod state_access;
pub mod unified_assembly;
pub mod wgsl;
pub mod wgsl_ast;
pub mod wgsl_dsl;

pub use flux_module::generate_flux_module_wgsl;
pub use ir::{lower_system, DiscreteEquation, DiscreteOp, DiscreteOpKind, DiscreteSystem};
pub use state_access::{state_component_expr, state_scalar_expr, state_vec2_expr};
pub use wgsl::{generate_wgsl, generate_wgsl_library};
