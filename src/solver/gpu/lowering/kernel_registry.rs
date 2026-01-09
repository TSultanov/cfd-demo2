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
    include!(concat!(env!("OUT_DIR"), "/kernel_registry_map.rs"));
}

pub(crate) fn generic_coupled_sources(
    model_id: &str,
) -> Result<GenericCoupledKernelSources, String> {
    Ok(GenericCoupledKernelSources {
        assembly: kernel_source(model_id, KernelKind::GenericCoupledAssembly)?,
        update: kernel_source(model_id, KernelKind::GenericCoupledUpdate)?,
    })
}

pub(crate) fn kernel_source(model_id: &str, kind: KernelKind) -> Result<KernelSource, String> {
    if matches!(
        kind,
        KernelKind::GenericCoupledAssembly | KernelKind::GenericCoupledUpdate
    ) {
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

        if kind == KernelKind::GenericCoupledAssembly {
            return Ok(KernelSource {
                kind,
                shader: assembly_shader,
                bindings: assembly_bindings,
                create_pipeline: assembly_pipeline,
            });
        }

        return Ok(KernelSource {
            kind,
            shader: update_shader,
            bindings: update_bindings,
            create_pipeline: update_pipeline,
        });
    }

    if let Some((shader, create_pipeline, bindings)) = generated::kernel_entry(kind) {
        return Ok(KernelSource {
            kind,
            shader,
            bindings,
            create_pipeline,
        });
    }

    match kind {
        KernelKind::IncompressibleMomentum => Err(
            "KernelKind::IncompressibleMomentum is not a compute kernel (no generated pipeline entrypoint)"
                .to_string(),
        ),
        _ => Err(format!(
            "KernelKind::{kind:?} does not have a generated kernel source entry"
        )),
    }
}
