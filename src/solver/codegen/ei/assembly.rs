use crate::solver::model::backend::{EquationSystem, StateLayout};

/// Explicit-Implicit (EI) method: implicit assembly kernel.
///
/// Transitional wrapper: implementation currently lives in the legacy
/// `compressible_assembly` module.
pub fn generate_ei_assembly_wgsl(layout: &StateLayout, system: &EquationSystem) -> String {
    super::super::compressible_assembly::generate_ei_assembly_wgsl(layout, system)
}
