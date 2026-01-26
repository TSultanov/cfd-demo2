pub mod coeff_expr;
pub mod dp_init;
pub mod dp_update;
pub mod dsl;
pub mod flux_module;
pub mod flux_module_gradients;
pub mod generic_coupled_kernels;
pub mod ir;
pub mod packed_state_gradients;
pub mod plan;
pub mod primitive_expr;
pub mod reconstruction;
pub mod rhie_chow_correct_velocity;
pub mod state_access;
pub mod unified_assembly;
pub mod wgsl;
pub mod wgsl_ast;
pub mod wgsl_dsl;

pub use dp_init::generate_dp_init_wgsl;
pub use dp_update::generate_dp_update_from_diag_wgsl;
pub use flux_module::generate_flux_module_wgsl;
pub use flux_module_gradients::generate_flux_module_gradients_wgsl;
pub use ir::{lower_system, DiscreteEquation, DiscreteOp, DiscreteOpKind, DiscreteSystem};
pub use packed_state_gradients::generate_packed_state_gradients_wgsl;
pub use rhie_chow_correct_velocity::{
    generate_rhie_chow_correct_velocity_delta_wgsl, generate_rhie_chow_correct_velocity_wgsl,
    generate_rhie_chow_store_grad_p_wgsl,
};
pub use state_access::{state_component_expr, state_scalar_expr, state_vec2_expr};
pub use wgsl::{generate_wgsl, generate_wgsl_library};
