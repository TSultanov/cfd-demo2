#[doc(hidden)]
pub mod backend;
mod definitions;
pub mod gpu_spec;
pub mod kernel;
pub mod method;
pub mod flux_layout;
pub mod eos;
pub mod flux_module;
pub mod primitives;
pub mod linear_solver;

pub use definitions::{
    compressible_model, compressible_system, generic_diffusion_demo_model,
    generic_diffusion_demo_neumann_model, incompressible_momentum_model,
    incompressible_momentum_generic_model,
    incompressible_momentum_system, BoundaryCondition, BoundarySpec, CompressibleFields,
    FieldBoundarySpec, GenericCoupledFields, IncompressibleMomentumFields, ModelSpec,
};
pub use flux_layout::{FluxComponent, FluxLayout};
pub use gpu_spec::{expand_field_components, FluxSpec, GradientStorage, ModelGpuSpec};
pub use kernel::derive_kernel_plan_for_model;
pub use kernel::{KernelId, KernelKind, KernelPlan};
pub use method::MethodSpec;
pub use eos::EosSpec;
pub use flux_module::{FluxModuleSpec, ReconstructionSpec, LimiterSpec};
pub use primitives::PrimitiveDerivations;
pub use linear_solver::{ModelLinearSolverSpec, ModelPreconditionerSpec};
pub use linear_solver::SchurBlockLayout;
