use crate::solver::gpu::wgsl_reflect::WgslBindingDesc;
use crate::solver::model::KernelId;

pub(crate) struct KernelSource {
    pub bindings: &'static [WgslBindingDesc],
    pub create_pipeline: fn(&wgpu::Device) -> wgpu::ComputePipeline,
}

mod generated {
    include!(concat!(env!("OUT_DIR"), "/kernel_registry_map.rs"));
}

pub(crate) fn kernel_source_by_id(
    model_id: &str,
    kernel_id: KernelId,
) -> Result<KernelSource, String> {
    if let Some((_shader, create_pipeline, bindings)) =
        generated::kernel_entry_by_id(model_id, kernel_id.as_str())
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
