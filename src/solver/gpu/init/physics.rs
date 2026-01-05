use crate::solver::gpu::bindings::{
    coupled_assembly_merged as manual_coupled_assembly,
    flux_rhie_chow as manual_flux_rhie_chow,
    prepare_coupled as manual_prepare_coupled,
    pressure_assembly as manual_pressure_assembly,
    update_fields_from_coupled as manual_update_fields,
    generated::coupled_assembly_merged as generated_coupled_assembly,
    generated::flux_rhie_chow as generated_flux_rhie_chow,
    generated::prepare_coupled as generated_prepare_coupled,
    generated::pressure_assembly as generated_pressure_assembly,
    generated::update_fields_from_coupled as generated_update_fields,
};
use crate::solver::gpu::init::ShaderVariant;

pub struct PhysicsPipelines {
    pub pipeline_pressure_assembly: wgpu::ComputePipeline,
    pub pipeline_flux_rhie_chow: wgpu::ComputePipeline,
    pub pipeline_coupled_assembly_merged: wgpu::ComputePipeline,
    pub pipeline_update_from_coupled: wgpu::ComputePipeline,
    pub pipeline_prepare_coupled: wgpu::ComputePipeline,
}

pub fn init_physics_pipelines(
    device: &wgpu::Device,
    shader_variant: ShaderVariant,
) -> PhysicsPipelines {
    // Shaders
    let (
        pipeline_pressure_assembly,
        pipeline_flux_rhie_chow,
        pipeline_coupled_assembly_merged,
        pipeline_update_from_coupled,
        pipeline_prepare_coupled,
    ) = match shader_variant {
        ShaderVariant::Generated => (
            generated_pressure_assembly::compute::create_main_pipeline_embed_source(device),
            generated_flux_rhie_chow::compute::create_main_pipeline_embed_source(device),
            generated_coupled_assembly::compute::create_main_pipeline_embed_source(device),
            generated_update_fields::compute::create_main_pipeline_embed_source(device),
            generated_prepare_coupled::compute::create_main_pipeline_embed_source(device),
        ),
        ShaderVariant::Manual => (
            manual_pressure_assembly::compute::create_main_pipeline_embed_source(device),
            manual_flux_rhie_chow::compute::create_main_pipeline_embed_source(device),
            manual_coupled_assembly::compute::create_main_pipeline_embed_source(device),
            manual_update_fields::compute::create_main_pipeline_embed_source(device),
            manual_prepare_coupled::compute::create_main_pipeline_embed_source(device),
        ),
    };

    PhysicsPipelines {
        pipeline_pressure_assembly,
        pipeline_flux_rhie_chow,
        pipeline_coupled_assembly_merged,
        pipeline_update_from_coupled,
        pipeline_prepare_coupled,
    }
}
