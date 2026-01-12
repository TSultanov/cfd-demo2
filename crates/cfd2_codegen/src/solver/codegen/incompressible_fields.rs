use crate::solver::ir::{vol_scalar, vol_vector, FieldRef};
use crate::solver::units::si;

/// Minimal field bundle used by legacy incompressible codegen modules.
///
/// This intentionally lives in `codegen` so those generators don't need to depend on
/// model-layer types.
#[derive(Debug, Clone)]
pub struct CodegenIncompressibleMomentumFields {
    pub u: FieldRef,
    pub p: FieldRef,
    pub d_p: FieldRef,
    pub grad_p: FieldRef,
}

impl CodegenIncompressibleMomentumFields {
    pub fn new() -> Self {
        Self {
            u: vol_vector("U", si::VELOCITY),
            p: vol_scalar("p", si::PRESSURE),
            d_p: vol_scalar("d_p", si::D_P),
            grad_p: vol_vector("grad_p", si::PRESSURE_GRADIENT),
        }
    }
}
