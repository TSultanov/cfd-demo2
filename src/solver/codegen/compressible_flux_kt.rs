use crate::solver::model::backend::StateLayout;
use crate::solver::model::FluxLayout;

pub fn generate_ei_flux_kt_wgsl(layout: &StateLayout, flux_layout: &FluxLayout) -> String {
    super::ei::flux_kt::generate_ei_flux_kt_wgsl(layout, flux_layout)
}

pub fn generate_compressible_flux_kt_wgsl(
    layout: &StateLayout,
    flux_layout: &FluxLayout,
) -> String {
    generate_ei_flux_kt_wgsl(layout, flux_layout)
}
