use crate::solver::model::backend::StateLayout;
use crate::solver::model::FluxLayout;

pub fn generate_ei_explicit_update_wgsl(
    layout: &StateLayout,
    flux_layout: &FluxLayout,
) -> String {
    super::ei::explicit_update::generate_ei_explicit_update_wgsl(layout, flux_layout)
}

pub fn generate_compressible_explicit_update_wgsl(
    layout: &StateLayout,
    flux_layout: &FluxLayout,
) -> String {
    generate_ei_explicit_update_wgsl(layout, flux_layout)
}
