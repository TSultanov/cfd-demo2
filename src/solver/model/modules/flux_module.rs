use crate::solver::model::flux_module::FluxModuleSpec;
use crate::solver::model::kernel::{
    DispatchKindId, KernelConditionId, KernelPhaseId, ModelKernelGeneratorSpec, ModelKernelSpec,
};
use crate::solver::model::module::{KernelBundleModule, ModuleManifest};
use crate::solver::model::KernelId;

use cfd2_codegen::solver::codegen::KernelWgsl;

mod wgsl_flux {
    include!("flux_module_wgsl.rs");
}

mod wgsl_gradients {
    include!("flux_module_gradients_wgsl.rs");
}

pub(crate) use wgsl_gradients::generate_flux_module_gradients_wgsl;

pub fn flux_module_module(flux: FluxModuleSpec) -> Result<KernelBundleModule, String> {
    let has_gradients = match &flux {
        FluxModuleSpec::Kernel { gradients, .. } => gradients.is_some(),
        FluxModuleSpec::Scheme { gradients, .. } => gradients.is_some(),
    };

    let mut out = KernelBundleModule {
        name: "flux_module",
        kernels: Vec::new(),
        generators: Vec::new(),
        manifest: ModuleManifest {
            flux_module: Some(flux),
            ..Default::default()
        },
        ..Default::default()
    };

    if has_gradients {
        out.kernels.push(ModelKernelSpec {
            id: KernelId::FLUX_MODULE_GRADIENTS,
            phase: KernelPhaseId::Gradients,
            dispatch: DispatchKindId::Cells,
            condition: KernelConditionId::Always,
        });
        out.generators.push(ModelKernelGeneratorSpec::new(
            KernelId::FLUX_MODULE_GRADIENTS,
            generate_flux_module_gradients_kernel_wgsl,
        ));
    }

    out.kernels.push(ModelKernelSpec {
        id: KernelId::FLUX_MODULE,
        phase: KernelPhaseId::FluxComputation,
        dispatch: DispatchKindId::Faces,
        condition: KernelConditionId::Always,
    });
    out.generators.push(ModelKernelGeneratorSpec::new(
        KernelId::FLUX_MODULE,
        generate_flux_module_kernel_wgsl,
    ));

    Ok(out)
}

fn generate_flux_module_gradients_kernel_wgsl(
    model: &crate::solver::model::ModelSpec,
    _schemes: &crate::solver::ir::SchemeRegistry,
) -> Result<KernelWgsl, String> {
    let flux = model
        .flux_module()
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "flux_module_gradients requested but model has no flux module".to_string())?;

    match flux {
        crate::solver::model::flux_module::FluxModuleSpec::Kernel {
            gradients: Some(_),
            ..
        }
        | crate::solver::model::flux_module::FluxModuleSpec::Scheme {
            gradients: Some(_),
            ..
        } => {
            let flux_layout = crate::solver::ir::FluxLayout::from_system(&model.system);
            generate_flux_module_gradients_wgsl(&model.state_layout, &flux_layout)
                .map_err(|e| e.to_string())
        }
        _ => Err("flux_module_gradients requested but model has no gradients stage".to_string()),
    }
}

fn generate_flux_module_kernel_wgsl(
    model: &crate::solver::model::ModelSpec,
    _schemes: &crate::solver::ir::SchemeRegistry,
) -> Result<KernelWgsl, String> {
    let flux_layout = crate::solver::ir::FluxLayout::from_system(&model.system);
    let flux_stride = model.system.unknowns_per_cell();
    let prims = model
        .primitives
        .ordered()
        .map_err(|e| format!("primitive recovery ordering failed: {e}"))?;

    let flux = model
        .flux_module()
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "flux_module requested but model has no flux module".to_string())?;

    match flux {
        crate::solver::model::flux_module::FluxModuleSpec::Kernel { kernel, .. } => Ok(
            wgsl_flux::generate_flux_module_wgsl(
                &model.state_layout,
                &flux_layout,
                flux_stride,
                &prims,
                kernel,
            ),
        ),
        crate::solver::model::flux_module::FluxModuleSpec::Scheme { scheme, .. } => {
            use crate::solver::scheme::Scheme;

            let schemes = [
                Scheme::Upwind,
                Scheme::SecondOrderUpwind,
                Scheme::QUICK,
                Scheme::SecondOrderUpwindMinMod,
                Scheme::SecondOrderUpwindVanLeer,
                Scheme::QUICKMinMod,
                Scheme::QUICKVanLeer,
            ];

            let mut variants = Vec::new();
            for reconstruction in schemes {
                let kernel = crate::solver::model::flux_schemes::lower_flux_scheme(
                    scheme,
                    &model.system,
                    reconstruction,
                )
                .map_err(|e| format!("flux scheme lowering failed: {e}"))?;
                variants.push((reconstruction, kernel));
            }

            Ok(wgsl_flux::generate_flux_module_wgsl_runtime_scheme(
                &model.state_layout,
                &flux_layout,
                flux_stride,
                &prims,
                &variants,
            ))
        }
    }
}
