#[doc(hidden)]
pub mod backend;
mod definitions;
pub mod eos;
pub mod flux_layout;
pub mod flux_module;
pub mod flux_schemes;
pub mod gpu_spec;
pub mod helpers;
pub mod kernel;
pub mod linear_solver;
pub mod method;
pub mod primitives;

pub use definitions::{
    all_models, compressible_model, compressible_system, generic_diffusion_demo_model,
    generic_diffusion_demo_neumann_model, incompressible_momentum_generic_model,
    incompressible_momentum_model, incompressible_momentum_system, BoundaryCondition, BoundarySpec,
    CompressibleFields, FieldBoundarySpec, GenericCoupledFields, IncompressibleMomentumFields,
    ModelSpec,
};
pub use eos::EosSpec;
pub use flux_layout::{FluxComponent, FluxLayout};
pub use flux_module::{FluxModuleSpec, LimiterSpec, ReconstructionSpec};
pub use gpu_spec::{expand_field_components, FluxSpec, GradientStorage, ModelGpuSpec};
pub use kernel::KernelId;
pub use linear_solver::SchurBlockLayout;
pub use linear_solver::{ModelLinearSolverSpec, ModelPreconditionerSpec};
pub use method::MethodSpec;
pub use primitives::PrimitiveDerivations;
