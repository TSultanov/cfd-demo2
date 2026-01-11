use crate::solver::model::backend::{EquationSystem, StateLayout};

pub fn generate_ei_apply_wgsl(layout: &StateLayout, system: &EquationSystem) -> String {
    super::ei::apply::generate_ei_apply_wgsl(layout, system)
}

pub fn generate_compressible_apply_wgsl(
    layout: &StateLayout,
    system: &EquationSystem,
) -> String {
    generate_ei_apply_wgsl(layout, system)
}
