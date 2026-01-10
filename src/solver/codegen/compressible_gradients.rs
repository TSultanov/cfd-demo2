use crate::solver::model::backend::StateLayout;

pub fn generate_ei_gradients_wgsl(layout: &StateLayout) -> String {
    super::ei::gradients::generate_ei_gradients_wgsl(layout)
}

pub fn generate_compressible_gradients_wgsl(layout: &StateLayout) -> String {
    generate_ei_gradients_wgsl(layout)
}
