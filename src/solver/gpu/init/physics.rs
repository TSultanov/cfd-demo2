use crate::solver::gpu::bindings::generated::coupled_assembly_merged;
use crate::solver::gpu::bindings::generated::flux_rhie_chow;
use crate::solver::gpu::bindings::generated::prepare_coupled;
use crate::solver::gpu::bindings::generated::pressure_assembly;
use crate::solver::gpu::bindings::generated::update_fields_from_coupled;

pub struct PhysicsPipelines {
    pub pipeline_pressure_assembly: wgpu::ComputePipeline,
    pub pipeline_flux_rhie_chow: wgpu::ComputePipeline,
    pub pipeline_coupled_assembly_merged: wgpu::ComputePipeline,
    pub pipeline_update_from_coupled: wgpu::ComputePipeline,
    pub pipeline_prepare_coupled: wgpu::ComputePipeline,
}

pub fn init_physics_pipelines(device: &wgpu::Device) -> PhysicsPipelines {
    // Shaders
    let pipeline_pressure_assembly =
        pressure_assembly::compute::create_main_pipeline_embed_source(device);
    let pipeline_flux_rhie_chow =
        flux_rhie_chow::compute::create_main_pipeline_embed_source(device);
    let pipeline_coupled_assembly_merged =
        coupled_assembly_merged::compute::create_main_pipeline_embed_source(device);
    let pipeline_update_from_coupled =
        update_fields_from_coupled::compute::create_main_pipeline_embed_source(device);
    let pipeline_prepare_coupled =
        prepare_coupled::compute::create_main_pipeline_embed_source(device);

    PhysicsPipelines {
        pipeline_pressure_assembly,
        pipeline_flux_rhie_chow,
        pipeline_coupled_assembly_merged,
        pipeline_update_from_coupled,
        pipeline_prepare_coupled,
    }
}
