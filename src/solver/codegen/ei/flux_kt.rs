use crate::solver::model::backend::StateLayout;
use crate::solver::model::FluxLayout;

/// Explicit-Implicit (EI) method: KT (Kurganov–Tadmor) face flux kernel.
///
/// Transitional wrapper: implementation currently lives in the legacy
/// `compressible_flux_kt` module.
pub fn generate_ei_flux_kt_wgsl(layout: &StateLayout, flux_layout: &FluxLayout) -> String {
    super::super::compressible_flux_kt::generate_ei_flux_kt_wgsl(layout, flux_layout)
}
