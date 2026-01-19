use crate::solver::model::kernel::{
    DispatchKindId, KernelConditionId, KernelPhaseId, ModelKernelGeneratorSpec, ModelKernelSpec,
};
use crate::solver::model::module::{KernelBundleModule, ModuleManifest, NamedParamKey};
use crate::solver::model::{KernelId, MethodSpec};

pub fn generic_coupled_module(method: MethodSpec) -> KernelBundleModule {
    const PACKED_STATE_GRADIENTS: KernelId = KernelId("packed_state_gradients");

    KernelBundleModule {
        name: "generic_coupled",
        kernels: vec![
            ModelKernelSpec {
                id: PACKED_STATE_GRADIENTS,
                phase: KernelPhaseId::Gradients,
                dispatch: DispatchKindId::Cells,
                condition: KernelConditionId::RequiresGradState,
            },
            ModelKernelSpec {
                id: KernelId::GENERIC_COUPLED_ASSEMBLY,
                phase: KernelPhaseId::Assembly,
                dispatch: DispatchKindId::Cells,
                condition: KernelConditionId::RequiresNoGradState,
            },
            ModelKernelSpec {
                id: KernelId::GENERIC_COUPLED_ASSEMBLY_GRAD_STATE,
                phase: KernelPhaseId::Assembly,
                dispatch: DispatchKindId::Cells,
                condition: KernelConditionId::RequiresGradState,
            },
            ModelKernelSpec {
                id: KernelId::GENERIC_COUPLED_APPLY,
                phase: KernelPhaseId::Apply,
                dispatch: DispatchKindId::Cells,
                condition: KernelConditionId::RequiresImplicitStepping,
            },
            ModelKernelSpec {
                id: KernelId::GENERIC_COUPLED_UPDATE,
                phase: KernelPhaseId::Update,
                dispatch: DispatchKindId::Cells,
                condition: KernelConditionId::Always,
            },
        ],
        generators: vec![
            ModelKernelGeneratorSpec {
                id: PACKED_STATE_GRADIENTS,
                generator: crate::solver::model::kernel::generate_packed_state_gradients_kernel_wgsl,
            },
            ModelKernelGeneratorSpec {
                id: KernelId::GENERIC_COUPLED_ASSEMBLY,
                generator: crate::solver::model::kernel::generate_generic_coupled_assembly_kernel_wgsl,
            },
            ModelKernelGeneratorSpec {
                id: KernelId::GENERIC_COUPLED_ASSEMBLY_GRAD_STATE,
                generator:
                    crate::solver::model::kernel::generate_generic_coupled_assembly_grad_state_kernel_wgsl,
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
                NamedParamKey::Key("linear_solver.max_restart"),
                NamedParamKey::Key("linear_solver.max_iters"),
                NamedParamKey::Key("linear_solver.tolerance"),
                NamedParamKey::Key("linear_solver.tolerance_abs"),
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
