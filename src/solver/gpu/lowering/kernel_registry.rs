use crate::solver::gpu::wgsl_reflect::WgslBindingDesc;
use crate::solver::gpu::{wgsl_meta, bindings};
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
    Ok(GenericCoupledKernelSources {
        assembly: kernel_source(model_id, KernelKind::GenericCoupledAssembly)?,
        update: kernel_source(model_id, KernelKind::GenericCoupledUpdate)?,
    })
}

pub(crate) fn kernel_source(model_id: &str, kind: KernelKind) -> Result<KernelSource, String> {
    match kind {
        KernelKind::GenericCoupledAssembly | KernelKind::GenericCoupledUpdate => {
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

            Ok(KernelSource {
                kind,
                shader: update_shader,
                bindings: update_bindings,
                create_pipeline: update_pipeline,
            })
        }

        KernelKind::CompressibleAssembly => Ok(KernelSource {
            kind,
            shader: bindings::generated::compressible_assembly::SHADER_STRING,
            bindings: wgsl_meta::COMPRESSIBLE_ASSEMBLY_BINDINGS,
            create_pipeline: bindings::generated::compressible_assembly::compute::create_main_pipeline_embed_source,
        }),
        KernelKind::CompressibleApply => Ok(KernelSource {
            kind,
            shader: bindings::generated::compressible_apply::SHADER_STRING,
            bindings: wgsl_meta::COMPRESSIBLE_APPLY_BINDINGS,
            create_pipeline: bindings::generated::compressible_apply::compute::create_main_pipeline_embed_source,
        }),
        KernelKind::CompressibleFluxKt => Ok(KernelSource {
            kind,
            shader: bindings::generated::compressible_flux_kt::SHADER_STRING,
            bindings: wgsl_meta::COMPRESSIBLE_FLUX_KT_BINDINGS,
            create_pipeline: bindings::generated::compressible_flux_kt::compute::create_main_pipeline_embed_source,
        }),
        KernelKind::CompressibleGradients => Ok(KernelSource {
            kind,
            shader: bindings::generated::compressible_gradients::SHADER_STRING,
            bindings: wgsl_meta::COMPRESSIBLE_GRADIENTS_BINDINGS,
            create_pipeline: bindings::generated::compressible_gradients::compute::create_main_pipeline_embed_source,
        }),
        KernelKind::CompressibleUpdate => Ok(KernelSource {
            kind,
            shader: bindings::generated::compressible_update::SHADER_STRING,
            bindings: wgsl_meta::COMPRESSIBLE_UPDATE_BINDINGS,
            create_pipeline: bindings::generated::compressible_update::compute::create_main_pipeline_embed_source,
        }),

        KernelKind::PrepareCoupled => Ok(KernelSource {
            kind,
            shader: bindings::generated::prepare_coupled::SHADER_STRING,
            bindings: wgsl_meta::PREPARE_COUPLED_BINDINGS,
            create_pipeline: bindings::generated::prepare_coupled::compute::create_main_pipeline_embed_source,
        }),
        KernelKind::CoupledAssembly => Ok(KernelSource {
            kind,
            shader: bindings::generated::coupled_assembly_merged::SHADER_STRING,
            bindings: wgsl_meta::COUPLED_ASSEMBLY_MERGED_BINDINGS,
            create_pipeline: bindings::generated::coupled_assembly_merged::compute::create_main_pipeline_embed_source,
        }),
        KernelKind::PressureAssembly => Ok(KernelSource {
            kind,
            shader: bindings::generated::pressure_assembly::SHADER_STRING,
            bindings: wgsl_meta::PRESSURE_ASSEMBLY_BINDINGS,
            create_pipeline: bindings::generated::pressure_assembly::compute::create_main_pipeline_embed_source,
        }),
        KernelKind::UpdateFieldsFromCoupled => Ok(KernelSource {
            kind,
            shader: bindings::generated::update_fields_from_coupled::SHADER_STRING,
            bindings: wgsl_meta::UPDATE_FIELDS_FROM_COUPLED_BINDINGS,
            create_pipeline: bindings::generated::update_fields_from_coupled::compute::create_main_pipeline_embed_source,
        }),
        KernelKind::FluxRhieChow => Ok(KernelSource {
            kind,
            shader: bindings::generated::flux_rhie_chow::SHADER_STRING,
            bindings: wgsl_meta::FLUX_RHIE_CHOW_BINDINGS,
            create_pipeline: bindings::generated::flux_rhie_chow::compute::create_main_pipeline_embed_source,
        }),

        KernelKind::GenericCoupledApply => Ok(KernelSource {
            kind,
            shader: bindings::generated::generic_coupled_apply::SHADER_STRING,
            bindings: wgsl_meta::GENERIC_COUPLED_APPLY_BINDINGS,
            create_pipeline: bindings::generated::generic_coupled_apply::compute::create_main_pipeline_embed_source,
        }),

        KernelKind::IncompressibleMomentum => Err(
            "KernelKind::IncompressibleMomentum is not a compute kernel (no generated pipeline entrypoint)"
                .to_string(),
        ),
    }
}
