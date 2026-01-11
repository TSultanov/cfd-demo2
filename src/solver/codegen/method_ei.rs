use crate::solver::model::{FluxLayout, ModelSpec};

use super::ei;

pub fn generate_ei_assembly_wgsl(model: &ModelSpec) -> String {
    ei::assembly::generate_ei_assembly_wgsl(&model.state_layout, &model.system, &model.eos)
}

pub fn generate_ei_apply_wgsl(model: &ModelSpec) -> String {
    ei::apply::generate_ei_apply_wgsl(&model.state_layout, &model.system)
}

pub fn generate_ei_gradients_wgsl(model: &ModelSpec) -> String {
    ei::gradients::generate_ei_gradients_wgsl(&model.state_layout)
}

pub fn generate_ei_explicit_update_wgsl(model: &ModelSpec) -> String {
    let flux_layout = FluxLayout::from_system(&model.system);
    ei::explicit_update::generate_ei_explicit_update_wgsl(&model.state_layout, &flux_layout)
}

pub fn generate_ei_update_wgsl(model: &ModelSpec) -> String {
    ei::update::generate_ei_update_wgsl(&model.state_layout, &model.eos)
}

pub fn generate_ei_flux_kt_wgsl(model: &ModelSpec) -> String {
    let flux_layout = FluxLayout::from_system(&model.system);
    ei::flux_kt::generate_ei_flux_kt_wgsl(&model.state_layout, &flux_layout, &model.eos)
}
