use crate::solver::model::kernel::{
    DispatchKindId, KernelConditionId, KernelPhaseId, ModelKernelGeneratorSpec, ModelKernelSpec,
};
use crate::solver::model::module::{KernelBundleModule, ModuleManifest, NamedParamKey};
use crate::solver::model::{KernelId, MethodSpec};

pub fn generic_coupled_module(method: MethodSpec) -> KernelBundleModule {
    const PACKED_STATE_GRADIENTS: KernelId = KernelId("packed_state_gradients");

    let apply_relaxation_in_update = match method {
        MethodSpec::Coupled(caps) => caps.apply_relaxation_in_update,
    };

    // Set the port-based manifest for uniform params
    // This is done via a helper function in a separate module to avoid
    // proc-macro issues in build scripts
    #[cfg(cfd2_build_script)]
    let port_manifest = Some(
        crate::solver::model::modules::generic_coupled_ports::generic_coupled_uniform_port_manifest(
            apply_relaxation_in_update,
        ),
    );
    #[cfg(not(cfd2_build_script))]
    let port_manifest = None;

    // Keep named_params for host-only parameters
    // Uniform params (dt, dtau, viscosity, density, schemes, relaxation) are now
    // declared via port_manifest
    let mut named_params = vec![
        NamedParamKey::Key("preconditioner"),
        NamedParamKey::Key("linear_solver.max_restart"),
        NamedParamKey::Key("linear_solver.max_iters"),
        NamedParamKey::Key("linear_solver.tolerance"),
        NamedParamKey::Key("linear_solver.tolerance_abs"),
        NamedParamKey::Key("outer_iters"),
        NamedParamKey::Key("detailed_profiling_enabled"),
    ];

    if apply_relaxation_in_update {
        named_params.push(NamedParamKey::Key("nonconverged_relax"));
    }

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
            ModelKernelGeneratorSpec::new(
                PACKED_STATE_GRADIENTS,
                crate::solver::model::kernel::generate_packed_state_gradients_kernel_wgsl,
            ),
            ModelKernelGeneratorSpec::new(
                KernelId::GENERIC_COUPLED_ASSEMBLY,
                crate::solver::model::kernel::generate_generic_coupled_assembly_kernel_wgsl,
            ),
            ModelKernelGeneratorSpec::new(
                KernelId::GENERIC_COUPLED_ASSEMBLY_GRAD_STATE,
                crate::solver::model::kernel::generate_generic_coupled_assembly_grad_state_kernel_wgsl,
            ),
            ModelKernelGeneratorSpec::new_shared(
                KernelId::GENERIC_COUPLED_APPLY,
                |_model, _schemes| {
                    Ok(cfd2_codegen::solver::codegen::generic_coupled_kernels::generate_generic_coupled_apply_wgsl())
                },
            ),
            ModelKernelGeneratorSpec::new(
                KernelId::GENERIC_COUPLED_UPDATE,
                crate::solver::model::kernel::generate_generic_coupled_update_kernel_wgsl,
            ),
        ],
        manifest: ModuleManifest {
            method: Some(method),
            named_params,
            port_manifest,
            ..Default::default()
        },
        ..Default::default()
    }
}
