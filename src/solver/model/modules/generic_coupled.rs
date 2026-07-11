use crate::solver::model::kernel::{
    DispatchKindId, FusionGuard, KernelConditionId, KernelFusionPolicy, KernelFusionStepping,
    KernelPatternAtom, KernelPhaseId, ModelKernelFusionRule, ModelKernelGeneratorSpec,
    ModelKernelSpec,
};
use crate::solver::model::module::{KernelBundleModule, PortManifest};
use crate::solver::model::{KernelId, MethodSpec};
use cfd2_codegen::solver::codegen::fusion::BindingRemap;

pub fn generic_coupled_module(method: MethodSpec) -> KernelBundleModule {
    const PACKED_STATE_GRADIENTS: KernelId = KernelId("packed_state_gradients");
    const FUSED_GRADIENTS_ASSEMBLY: KernelId =
        KernelId("fusion/packed_state_gradients_assembly_grad_state");

    let apply_relaxation_in_update = match method {
        MethodSpec::Coupled(caps) => caps.apply_relaxation_in_update,
    };

    // Built via a helper in a separate module to avoid proc-macro issues in
    // build scripts.
    let port_manifest: Option<PortManifest> = Some(
        crate::solver::model::modules::generic_coupled_ports::generic_coupled_uniform_port_manifest(
            apply_relaxation_in_update,
        ),
    );

    // Host-only parameters; uniform params (dt, dtau, viscosity, density,
    // schemes, relaxation) are declared via port_manifest instead.
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
        "outer_fixed_iterations_mode",
        "outer_batched_mode",
        "detailed_profiling_enabled",
    ];

    if apply_relaxation_in_update {
        named_params.push("nonconverged_relax");
        named_params.push("nonconverged_dt_scale");
        named_params.push("nonconverged_dtau_scale");
        named_params.push("nonconverged_retry_enabled");
        named_params.push("nonconverged_retry_max_attempts");
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
                condition: KernelConditionId::RequiresNoGradStateAndImplicitStepping,
            },
            ModelKernelSpec {
                id: KernelId::GENERIC_COUPLED_ASSEMBLY_GRAD_STATE,
                phase: KernelPhaseId::Assembly,
                dispatch: DispatchKindId::Cells,
                condition: KernelConditionId::RequiresGradStateAndImplicitStepping,
            },
            // RHS-only assembly variants for matrix-frozen outer iterations
            // (scheduled only by the freeze paths; default off).
            ModelKernelSpec {
                id: KernelId::GENERIC_COUPLED_ASSEMBLY_RHS_ONLY,
                phase: KernelPhaseId::AssemblyRhsOnly,
                dispatch: DispatchKindId::Cells,
                condition: KernelConditionId::RequiresNoGradStateAndImplicitStepping,
            },
            ModelKernelSpec {
                id: KernelId::GENERIC_COUPLED_ASSEMBLY_GRAD_STATE_RHS_ONLY,
                phase: KernelPhaseId::AssemblyRhsOnly,
                dispatch: DispatchKindId::Cells,
                condition: KernelConditionId::RequiresGradStateAndImplicitStepping,
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
                condition: KernelConditionId::RequiresImplicitStepping,
            },
            ModelKernelSpec {
                id: KernelId::EXPLICIT_RESIDUAL,
                phase: KernelPhaseId::Assembly,
                dispatch: DispatchKindId::Cells,
                condition: KernelConditionId::RequiresNoGradStateAndExplicitStepping,
            },
            ModelKernelSpec {
                id: KernelId::EXPLICIT_RESIDUAL_GRAD_STATE,
                phase: KernelPhaseId::Assembly,
                dispatch: DispatchKindId::Cells,
                condition: KernelConditionId::RequiresGradStateAndExplicitStepping,
            },
            ModelKernelSpec {
                id: KernelId::EXPLICIT_RK4_STAGE_1,
                phase: KernelPhaseId::Update,
                dispatch: DispatchKindId::Cells,
                condition: KernelConditionId::RequiresExplicitStepping,
            },
            ModelKernelSpec {
                id: KernelId::EXPLICIT_RK4_STAGE_2,
                phase: KernelPhaseId::Update,
                dispatch: DispatchKindId::Cells,
                condition: KernelConditionId::RequiresExplicitStepping,
            },
            ModelKernelSpec {
                id: KernelId::EXPLICIT_RK4_STAGE_3,
                phase: KernelPhaseId::Update,
                dispatch: DispatchKindId::Cells,
                condition: KernelConditionId::RequiresExplicitStepping,
            },
            ModelKernelSpec {
                id: KernelId::EXPLICIT_RK4_STAGE_4,
                phase: KernelPhaseId::Update,
                dispatch: DispatchKindId::Cells,
                condition: KernelConditionId::RequiresExplicitStepping,
            },
        ],
        generators: vec![
            ModelKernelGeneratorSpec::new_dsl(
                PACKED_STATE_GRADIENTS,
                crate::solver::model::kernel::generate_packed_state_gradients_kernel_program,
            ),
            ModelKernelGeneratorSpec::new_dsl(
                KernelId::GENERIC_COUPLED_ASSEMBLY,
                crate::solver::model::kernel::generate_generic_coupled_assembly_kernel_program,
            ),
            ModelKernelGeneratorSpec::new_dsl(
                KernelId::GENERIC_COUPLED_ASSEMBLY_GRAD_STATE,
                crate::solver::model::kernel::generate_generic_coupled_assembly_grad_state_kernel_program,
            ),
            ModelKernelGeneratorSpec::new_dsl(
                KernelId::GENERIC_COUPLED_ASSEMBLY_RHS_ONLY,
                crate::solver::model::kernel::generate_generic_coupled_assembly_rhs_only_kernel_program,
            ),
            ModelKernelGeneratorSpec::new_dsl(
                KernelId::GENERIC_COUPLED_ASSEMBLY_GRAD_STATE_RHS_ONLY,
                crate::solver::model::kernel::generate_generic_coupled_assembly_grad_state_rhs_only_kernel_program,
            ),
            ModelKernelGeneratorSpec::new_shared_dsl(
                KernelId::GENERIC_COUPLED_APPLY,
                |_model, _schemes| {
                    // Shared kernel must use canonical EOS params to generate
                    // identical WGSL across all models
                    let eos_params = crate::solver::model::modules::eos_ports::eos_uniform_port_manifest()
                        .params;
                    cfd2_codegen::solver::codegen::generic_coupled_kernels::generate_generic_coupled_apply_kernel_program(
                        crate::solver::model::KernelId::GENERIC_COUPLED_APPLY.as_str(),
                        &eos_params,
                    )
                },
            ),
            ModelKernelGeneratorSpec::new_dsl(
                KernelId::GENERIC_COUPLED_UPDATE,
                crate::solver::model::kernel::generate_generic_coupled_update_kernel_program,
            ),
            ModelKernelGeneratorSpec::new_explicit_rk4_dsl(
                KernelId::EXPLICIT_RESIDUAL,
                crate::solver::model::kernel::generate_explicit_residual_kernel_program,
            ),
            ModelKernelGeneratorSpec::new_explicit_rk4_dsl(
                KernelId::EXPLICIT_RESIDUAL_GRAD_STATE,
                crate::solver::model::kernel::generate_explicit_residual_grad_state_kernel_program,
            ),
            ModelKernelGeneratorSpec::new_explicit_rk4_dsl(
                KernelId::EXPLICIT_RK4_STAGE_1,
                crate::solver::model::kernel::generate_explicit_rk4_stage_1_kernel_program,
            ),
            ModelKernelGeneratorSpec::new_explicit_rk4_dsl(
                KernelId::EXPLICIT_RK4_STAGE_2,
                crate::solver::model::kernel::generate_explicit_rk4_stage_2_kernel_program,
            ),
            ModelKernelGeneratorSpec::new_explicit_rk4_dsl(
                KernelId::EXPLICIT_RK4_STAGE_3,
                crate::solver::model::kernel::generate_explicit_rk4_stage_3_kernel_program,
            ),
            ModelKernelGeneratorSpec::new_explicit_rk4_dsl(
                KernelId::EXPLICIT_RK4_STAGE_4,
                crate::solver::model::kernel::generate_explicit_rk4_stage_4_kernel_program,
            ),
        ],
        fusion_rules: vec![
            // Cross-phase fusion: packed_state_gradients (Gradients) + assembly_grad_state (Assembly)
            // These are directly adjacent in has_grad_state=true schedules for all coupled models.
            ModelKernelFusionRule {
                name: "generic_coupled:gradients_assembly_grad_state_v1",
                priority: 100,
                phase: KernelPhaseId::Gradients,
                pattern: vec![
                    KernelPatternAtom::with_phase(
                        PACKED_STATE_GRADIENTS,
                        DispatchKindId::Cells,
                        KernelPhaseId::Gradients,
                    ),
                    KernelPatternAtom::with_phase(
                        KernelId::GENERIC_COUPLED_ASSEMBLY_GRAD_STATE,
                        DispatchKindId::Cells,
                        KernelPhaseId::Assembly,
                    ),
                ],
                replacement: ModelKernelSpec {
                    id: FUSED_GRADIENTS_ASSEMBLY,
                    phase: KernelPhaseId::Gradients,
                    dispatch: DispatchKindId::Cells,
                    condition: KernelConditionId::RequiresGradState,
                },
                guards: vec![
                    FusionGuard::RequiresGradState,
                    FusionGuard::RequiresStepping(KernelFusionStepping::Coupled),
                    FusionGuard::RequiresModule("generic_coupled"),
                    FusionGuard::MinPolicy(KernelFusionPolicy::Safe),
                    // Fusing gradients into the assembly dispatch makes
                    // NEIGHBOR grad_state reads racy (fresh-or-stale within
                    // the dispatch). Reconstruction reads tolerate that as a
                    // lagged correction; transpose_dev2 terms consume the
                    // values directly, so the fusion must not apply.
                    FusionGuard::RequiresNoNeighborGradConsumers,
                ],
                // The gradients kernel (program 0) has a different bind layout
                // than assembly (program 1). Remap program 0's slots to match
                // assembly's superset layout before merging:
                //   grad_state: (1,4) → (1,5)  [assembly stores grad_state at slot 5]
                //   bc_kind:    (2,0) → (3,0)  [assembly has BCs in group 3]
                //   bc_value:   (2,1) → (3,1)
                binding_remaps: vec![
                    BindingRemap {
                        program_index: 0,
                        from_group: 1,
                        from_binding: 4,
                        to_group: 1,
                        to_binding: 5,
                    },
                    BindingRemap {
                        program_index: 0,
                        from_group: 2,
                        from_binding: 0,
                        to_group: 3,
                        to_binding: 0,
                    },
                    BindingRemap {
                        program_index: 0,
                        from_group: 2,
                        from_binding: 1,
                        to_group: 3,
                        to_binding: 1,
                    },
                ],
                expected_hazards: vec![],
            },
        ],
        method: Some(method),
        named_params,
        port_manifest,
        ..Default::default()
    }
}
