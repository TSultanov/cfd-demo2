use crate::solver::model::backend::{EquationSystem, StateLayout};

/// Explicit-Implicit (EI) method: apply linear-solver delta into the packed state.
///
/// Transitional wrapper: implementation currently lives in the legacy
/// `compressible_apply` module.
pub fn generate_ei_apply_wgsl(layout: &StateLayout, system: &EquationSystem) -> String {
    super::super::compressible_apply::generate_ei_apply_wgsl(layout, system)
}
