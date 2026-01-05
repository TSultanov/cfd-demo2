pub mod ast;
pub mod coupled_assembly;
pub mod emit;
pub mod ir;
pub mod model;
pub mod prepare_coupled;
pub mod pressure_assembly;
pub mod scheme;
pub mod update_fields_from_coupled;
pub mod state_access;
pub mod state_layout;
pub mod wgsl;
pub mod wgsl_ast;
pub mod wgsl_dsl;

pub use ast::{
    fvc, fvm, Coefficient, Discretization, Equation, EquationSystem, FieldKind, FieldRef, FluxRef,
    Term, TermOp,
};
pub use coupled_assembly::generate_coupled_assembly_wgsl;
pub use emit::write_wgsl_file;
pub use emit::emit_coupled_assembly_codegen_wgsl;
pub use emit::emit_coupled_assembly_codegen_wgsl_with_schemes;
pub use emit::emit_incompressible_momentum_wgsl;
pub use emit::emit_incompressible_momentum_wgsl_with_schemes;
pub use emit::emit_prepare_coupled_codegen_wgsl;
pub use emit::emit_pressure_assembly_codegen_wgsl;
pub use emit::emit_update_fields_from_coupled_codegen_wgsl;
pub use ir::{lower_system, DiscreteEquation, DiscreteOp, DiscreteOpKind, DiscreteSystem};
pub use model::{incompressible_momentum_model, incompressible_momentum_system, ModelSpec};
pub use prepare_coupled::generate_prepare_coupled_wgsl;
pub use pressure_assembly::generate_pressure_assembly_wgsl;
pub use update_fields_from_coupled::generate_update_fields_from_coupled_wgsl;
pub use scheme::{SchemeRegistry, TermKey};
pub use state_access::{state_component_expr, state_scalar_expr, state_vec2_expr};
pub use state_layout::{StateField, StateLayout};
pub use wgsl::{generate_wgsl, generate_wgsl_library};
