use crate::solver::gpu::wgsl_reflect::WgslBindingDesc;
use crate::solver::model::KernelId;
use std::borrow::Cow;

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

pub(crate) fn kernel_wgsl_source_by_id(
    model_id: &str,
    kernel_id: KernelId,
) -> Result<&'static str, String> {
    if let Some((shader, _create_pipeline, _bindings)) =
        generated::kernel_entry_by_id(model_id, kernel_id.as_str())
    {
        return Ok(shader);
    }

    Err(format!(
        "KernelId '{}' does not have a generated kernel source entry",
        kernel_id.as_str()
    ))
}

pub(crate) fn kernel_shader_module_by_id(
    device: &wgpu::Device,
    model_id: &str,
    kernel_id: KernelId,
) -> Result<wgpu::ShaderModule, String> {
    let wgsl = kernel_wgsl_source_by_id(model_id, kernel_id)?;
    Ok(device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(kernel_id.as_str()),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(wgsl)),
    }))
}
