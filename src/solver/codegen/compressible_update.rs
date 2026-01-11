use crate::solver::model::backend::StateLayout;

pub fn generate_ei_update_wgsl(layout: &StateLayout) -> String {
    super::ei::update::generate_ei_update_wgsl(layout)
}

pub fn generate_compressible_update_wgsl(
    layout: &StateLayout,
) -> String {
    generate_ei_update_wgsl(layout)
}
