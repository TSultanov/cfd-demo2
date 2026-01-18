use crate::solver::model::flux_module::FluxModuleSpec;
use crate::solver::model::kernel::{DispatchKindId, KernelPhaseId, ModelKernelGeneratorSpec, ModelKernelSpec};
use crate::solver::model::module::{KernelBundleModule, ModuleManifest};
use crate::solver::model::KernelId;

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
        });
        out.generators.push(ModelKernelGeneratorSpec {
            id: KernelId::FLUX_MODULE_GRADIENTS,
            generator: crate::solver::model::kernel::generate_flux_module_gradients_kernel_wgsl,
        });
    }

    out.kernels.push(ModelKernelSpec {
        id: KernelId::FLUX_MODULE,
        phase: KernelPhaseId::FluxComputation,
        dispatch: DispatchKindId::Faces,
    });
    out.generators.push(ModelKernelGeneratorSpec {
        id: KernelId::FLUX_MODULE,
        generator: crate::solver::model::kernel::generate_flux_module_kernel_wgsl,
    });

    Ok(out)
}
