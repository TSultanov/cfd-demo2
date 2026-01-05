#[doc(hidden)]
pub mod backend;
mod definitions;

pub use definitions::{
    incompressible_momentum_model, incompressible_momentum_system, IncompressibleMomentumFields,
    ModelSpec,
};
