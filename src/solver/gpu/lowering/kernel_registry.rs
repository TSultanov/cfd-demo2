use crate::solver::gpu::wgsl_reflect::WgslBindingDesc;
use crate::solver::model::{KernelId, KernelKind};

pub(crate) struct KernelSource {
    pub bindings: &'static [WgslBindingDesc],
    pub create_pipeline: fn(&wgpu::Device) -> wgpu::ComputePipeline,
}

mod generated {
    include!(concat!(env!("OUT_DIR"), "/generic_coupled_registry.rs"));
    include!(concat!(env!("OUT_DIR"), "/kernel_registry_map.rs"));
}

#[allow(dead_code)]
pub(crate) fn kernel_source(model_id: &str, kind: KernelKind) -> Result<KernelSource, String> {
    kernel_source_by_id(model_id, KernelId::from(kind))
}

pub(crate) fn kernel_source_by_id(
    model_id: &str,
    kernel_id: KernelId,
) -> Result<KernelSource, String> {
    // Generic-coupled kernels are emitted per-model, so they require a model-specific lookup.
    if matches!(
        kernel_id,
        KernelId::GENERIC_COUPLED_ASSEMBLY | KernelId::GENERIC_COUPLED_UPDATE
    ) {
        let Some((
            _assembly_shader,
            assembly_pipeline,
            assembly_bindings,
            _update_shader,
            update_pipeline,
            update_bindings,
        )) = generated::generic_coupled_pair(model_id)
        else {
            return Err(format!(
                "GpuProgramPlan does not have generated generic-coupled kernels for model id '{model_id}'"
            ));
        };

        if kernel_id == KernelId::GENERIC_COUPLED_ASSEMBLY {
            return Ok(KernelSource {
                bindings: assembly_bindings,
                create_pipeline: assembly_pipeline,
            });
        }

        return Ok(KernelSource {
            bindings: update_bindings,
            create_pipeline: update_pipeline,
        });
    }

    if let Some((_shader, create_pipeline, bindings)) =
        generated::kernel_entry_by_id(kernel_id.as_str())
    {
        return Ok(KernelSource {
            bindings,
            create_pipeline,
        });
    }

    Err(format!(
        "KernelId '{}' does not have a generated kernel source entry",
        kernel_id.as_str()
    ))
}
