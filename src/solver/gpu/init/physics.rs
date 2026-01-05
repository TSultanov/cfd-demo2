use crate::solver::gpu::bindings::generated::{
    coupled_assembly_merged as generated_coupled_assembly,
    flux_rhie_chow as generated_flux_rhie_chow,
    prepare_coupled as generated_prepare_coupled,
    pressure_assembly as generated_pressure_assembly,
    update_fields_from_coupled as generated_update_fields,
};

pub struct PhysicsPipelines {
    pub pipeline_pressure_assembly: wgpu::ComputePipeline,
    pub pipeline_flux_rhie_chow: wgpu::ComputePipeline,
    pub pipeline_coupled_assembly_merged: wgpu::ComputePipeline,
    pub pipeline_update_from_coupled: wgpu::ComputePipeline,
    pub pipeline_prepare_coupled: wgpu::ComputePipeline,
}

pub fn init_physics_pipelines(device: &wgpu::Device) -> PhysicsPipelines {
    let pipeline_pressure_assembly =
        generated_pressure_assembly::compute::create_main_pipeline_embed_source(device);
    let pipeline_flux_rhie_chow =
        generated_flux_rhie_chow::compute::create_main_pipeline_embed_source(device);
    let pipeline_coupled_assembly_merged =
        generated_coupled_assembly::compute::create_main_pipeline_embed_source(device);
    let pipeline_update_from_coupled =
        generated_update_fields::compute::create_main_pipeline_embed_source(device);
    let pipeline_prepare_coupled =
        generated_prepare_coupled::compute::create_main_pipeline_embed_source(device);

    PhysicsPipelines {
        pipeline_pressure_assembly,
        pipeline_flux_rhie_chow,
        pipeline_coupled_assembly_merged,
        pipeline_update_from_coupled,
        pipeline_prepare_coupled,
    }
}
