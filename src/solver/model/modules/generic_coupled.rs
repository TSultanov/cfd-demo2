use crate::solver::model::kernel::{DispatchKindId, KernelPhaseId, ModelKernelGeneratorSpec, ModelKernelSpec};
use crate::solver::model::module::{KernelBundleModule, ModuleManifest, NamedParamKey};
use crate::solver::model::{KernelId, MethodSpec};

pub fn generic_coupled_module(method: MethodSpec) -> KernelBundleModule {
    KernelBundleModule {
        name: "generic_coupled",
        kernels: vec![
            ModelKernelSpec {
                id: KernelId::GENERIC_COUPLED_ASSEMBLY,
                phase: KernelPhaseId::Assembly,
                dispatch: DispatchKindId::Cells,
            },
            ModelKernelSpec {
                id: KernelId::GENERIC_COUPLED_UPDATE,
                phase: KernelPhaseId::Update,
                dispatch: DispatchKindId::Cells,
            },
        ],
        generators: vec![
            ModelKernelGeneratorSpec {
                id: KernelId::GENERIC_COUPLED_ASSEMBLY,
                generator: crate::solver::model::kernel::generate_generic_coupled_assembly_kernel_wgsl,
            },
            ModelKernelGeneratorSpec {
                id: KernelId::GENERIC_COUPLED_UPDATE,
                generator: crate::solver::model::kernel::generate_generic_coupled_update_kernel_wgsl,
            },
        ],
        manifest: ModuleManifest {
            method: Some(method),
            named_params: vec![
                NamedParamKey::Key("dt"),
                NamedParamKey::Key("dtau"),
                NamedParamKey::Key("advection_scheme"),
                NamedParamKey::Key("time_scheme"),
                NamedParamKey::Key("preconditioner"),
                NamedParamKey::Key("viscosity"),
                NamedParamKey::Key("density"),
                NamedParamKey::Key("alpha_u"),
                NamedParamKey::Key("alpha_p"),
                NamedParamKey::Key("nonconverged_relax"),
                NamedParamKey::Key("outer_iters"),
                NamedParamKey::Key("detailed_profiling_enabled"),
            ],
            ..Default::default()
        },
        ..Default::default()
    }
}
