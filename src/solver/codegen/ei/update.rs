use crate::solver::model::backend::StateLayout;

/// Explicit-Implicit (EI) method: primitive recovery / update kernel.
///
/// Transitional wrapper: implementation currently lives in the legacy
/// `compressible_update` module.
pub fn generate_ei_update_wgsl(layout: &StateLayout) -> String {
    super::super::compressible_update::generate_ei_update_wgsl(layout)
}
