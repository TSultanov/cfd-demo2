use crate::solver::model::kernel::{
    DispatchKindId, KernelConditionId, KernelPhaseId, ModelKernelGeneratorSpec, ModelKernelSpec,
};
use crate::solver::model::module::{KernelBundleModule, PortManifest};
use crate::solver::model::{KernelId, MethodSpec};

pub fn generic_coupled_module(method: MethodSpec) -> KernelBundleModule {
    const PACKED_STATE_GRADIENTS: KernelId = KernelId("packed_state_gradients");

    let apply_relaxation_in_update = match method {
        MethodSpec::Coupled(caps) => caps.apply_relaxation_in_update,
    };

    // Set the port-based manifest for uniform params
    // This is done via a helper function in a separate module to avoid
    // proc-macro issues in build scripts
    let port_manifest: Option<PortManifest> = Some(
        crate::solver::model::modules::generic_coupled_ports::generic_coupled_uniform_port_manifest(
            apply_relaxation_in_update,
        ),
    );

    // Keep named_params for host-only parameters
    // Uniform params (dt, dtau, viscosity, density, schemes, relaxation) are now
    // declared via port_manifest
    let mut named_params = vec![
        "preconditioner",
        "linear_solver.max_restart",
        "linear_solver.max_iters",
        "linear_solver.tolerance",
        "linear_solver.tolerance_abs",
        "linear_solver.solution_update_strategy",
        "outer_iters",
        "outer_tol",
        "outer_tol_abs",
        "detailed_profiling_enabled",
    ];

    if apply_relaxation_in_update {
        named_params.push("nonconverged_relax");
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
            ModelKernelGeneratorSpec::new_dsl(
                KernelId::GENERIC_COUPLED_ASSEMBLY,
                crate::solver::model::kernel::generate_generic_coupled_assembly_kernel_program,
            ),
            ModelKernelGeneratorSpec::new_dsl(
                KernelId::GENERIC_COUPLED_ASSEMBLY_GRAD_STATE,
                crate::solver::model::kernel::generate_generic_coupled_assembly_grad_state_kernel_program,
            ),
            ModelKernelGeneratorSpec::new_shared(
                KernelId::GENERIC_COUPLED_APPLY,
                |_model, _schemes| {
                    // Shared kernel must use canonical EOS params to generate
                    // identical WGSL across all models
                    let eos_params = crate::solver::model::modules::eos_ports::eos_uniform_port_manifest()
                        .params;
                    Ok(cfd2_codegen::solver::codegen::generic_coupled_kernels::generate_generic_coupled_apply_wgsl(&eos_params))
                },
            ),
            ModelKernelGeneratorSpec::new_dsl(
                KernelId::GENERIC_COUPLED_UPDATE,
                crate::solver::model::kernel::generate_generic_coupled_update_kernel_program,
            ),
        ],
        method: Some(method),
        named_params,
        port_manifest,
        ..Default::default()
    }
}
