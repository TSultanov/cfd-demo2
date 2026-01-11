use crate::solver::model::backend::{EquationSystem, StateLayout};

pub fn generate_ei_assembly_wgsl(layout: &StateLayout, system: &EquationSystem) -> String {
    super::ei::assembly::generate_ei_assembly_wgsl(layout, system)
}

pub fn generate_compressible_assembly_wgsl(
    layout: &StateLayout,
    system: &EquationSystem,
) -> String {
    generate_ei_assembly_wgsl(layout, system)
}
