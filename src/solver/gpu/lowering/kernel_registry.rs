use crate::solver::gpu::wgsl_reflect::WgslBindingDesc;
use crate::solver::model::KernelKind;

pub(crate) struct KernelSource {
    pub kind: KernelKind,
    pub shader: &'static str,
    pub bindings: &'static [WgslBindingDesc],
    pub create_pipeline: fn(&wgpu::Device) -> wgpu::ComputePipeline,
}

pub(crate) struct GenericCoupledKernelSources {
    pub assembly: KernelSource,
    pub update: KernelSource,
}

mod generated {
    include!(concat!(env!("OUT_DIR"), "/generic_coupled_registry.rs"));
}

pub(crate) fn generic_coupled_sources(
    model_id: &str,
) -> Result<GenericCoupledKernelSources, String> {
    let Some((
        assembly_shader,
        assembly_pipeline,
        assembly_bindings,
        update_shader,
        update_pipeline,
        update_bindings,
    )) = generated::generic_coupled_pair(model_id)
    else {
        return Err(format!(
            "GpuProgramPlan does not have generated generic-coupled kernels for model id '{model_id}'"
        ));
    };

    Ok(GenericCoupledKernelSources {
        assembly: KernelSource {
            kind: KernelKind::GenericCoupledAssembly,
            shader: assembly_shader,
            bindings: assembly_bindings,
            create_pipeline: assembly_pipeline,
        },
        update: KernelSource {
            kind: KernelKind::GenericCoupledUpdate,
            shader: update_shader,
            bindings: update_bindings,
            create_pipeline: update_pipeline,
        },
    })
}
