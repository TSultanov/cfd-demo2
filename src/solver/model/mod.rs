#[doc(hidden)]
pub mod backend;
mod definitions;
pub mod kernel;

pub use definitions::{
    compressible_model, compressible_system, incompressible_momentum_model,
    incompressible_momentum_system, generic_diffusion_demo_model, CompressibleFields,
    generic_diffusion_demo_neumann_model,
    GenericCoupledFields, IncompressibleMomentumFields, ModelFields, ModelSpec,
    BoundaryCondition, BoundarySpec, FieldBoundarySpec,
};
pub use kernel::{KernelKind, KernelPlan};
