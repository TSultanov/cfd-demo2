#[doc(hidden)]
pub mod backend;
mod definitions;
pub mod kernel;

pub use definitions::{
    compressible_model, compressible_system, generic_diffusion_demo_model,
    generic_diffusion_demo_neumann_model, incompressible_momentum_model,
    incompressible_momentum_system, BoundaryCondition, BoundarySpec, CompressibleFields,
    FieldBoundarySpec, GenericCoupledFields, IncompressibleMomentumFields, ModelFields, ModelSpec,
};
pub use kernel::{KernelKind, KernelPlan};
