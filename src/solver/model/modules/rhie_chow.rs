use crate::solver::model::kernel::{
    DispatchKindId, FusionGuard, KernelConditionId, KernelFusionStepping, KernelPatternAtom,
    KernelPhaseId, ModelKernelFusionRule, ModelKernelGeneratorSpec, ModelKernelSpec,
};
use crate::solver::model::module::{KernelBundleModule, ModuleInvariant};
use crate::solver::model::KernelId;

use cfd2_codegen::solver::codegen::fusion::{ExpectedHazard, HazardKind};

use cfd2_codegen::solver::codegen::{
    bc_table::BcTable,
    dsl::XY,
    wgsl_ast::{AssignOp, Expr, ForStep, Type},
    wgsl_dsl as dsl,
};
use cfd2_ir::kernel::{
    BindingAccess, DispatchDomain, EffectResource, KernelBinding, KernelProgram, LaunchSemantics,
};

/// How the Rhie–Chow coupling coefficient `d_p` is computed each outer
/// iteration (the `dp_update_from_diag` kernel).
///
/// `ClosedForm` is the historical default: a uniform `alpha_u * dt / rho`.
/// `FromAssembledDiagonal` is the OpenFOAM `rAU` analogue: `V / a_P` from
/// the assembled momentum diagonal (averaged over the two momentum
/// components), optionally scaled by `alpha_u` (`include_relaxation`;
/// OpenFOAM PISO mode uses the UNRELAXED 1/a_P, SIMPLE mode the relaxed
/// one), and damped across outer iterations
/// (`d_p <- theta*new + (1-theta)*old`) to stabilize the lagged
/// d_p -> assembly -> d_p feedback loop. With this formulation `dp_init`
/// seeds the closed form once (when d_p is still zero) instead of zeroing,
/// so the damped update has a sane starting value and persists across
/// iterations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DpFormulation {
    /// `d_p = alpha_u * dt / rho` (uniform; historical default).
    ClosedForm,
    /// `d_p = [alpha_u *] V / a_P` from the assembled momentum diagonal,
    /// damped across outer iterations by `theta`.
    FromAssembledDiagonal {
        include_relaxation: bool,
        theta: f32,
    },
    /// SIMPLEC-consistent: `d_p = V / Σ_rank a(c,c)` — the momentum-row sum
    /// over same-component columns (diagonal + signed off-diagonals).
    /// Conservative transport rows sum to the ddt coefficient plus
    /// eliminated-boundary contributions, so d_p stays at the proven-stable
    /// `dt/rho` scale in the interior while shrinking near Dirichlet
    /// boundaries where wall friction inflates the diagonal. No `alpha_u`
    /// factor anywhere: SIMPLEC's point is relaxation-consistency. Same
    /// damped update across outer iterations as `FromAssembledDiagonal`.
    ///
    /// Rationale (June 2026): the plain `V / a_P` formulation is
    /// kernel-correct but its `~1/d_p` outer-loop gain is unstable at the
    /// Schur-consistent scale even with f32-floor linear solves
    /// (tests/dp_diag_probe.rs), and a pressure-row equilibration cannot
    /// help — row scaling leaves exact-solve outer dynamics unchanged.
    /// The row-sum denominator avoids that scale by construction.
    FromAssembledRowSum { theta: f32 },
}

impl Default for DpFormulation {
    fn default() -> Self {
        Self::ClosedForm
    }
}

impl DpFormulation {
    /// True for the formulations that read the assembled momentum rows and
    /// persist d_p across outer iterations (seed-if-zero init + damped
    /// update), as opposed to the recomputed-each-iteration closed form.
    pub fn uses_assembled_matrix(&self) -> bool {
        matches!(
            self,
            Self::FromAssembledDiagonal { .. } | Self::FromAssembledRowSum { .. }
        )
    }
}

/// Rhie-Chow auxiliary module that manages pressure correction and velocity correction.
///
/// This module provides kernels for:
/// - `dp_init`: Initialize the pressure correction field
/// - `dp_update_from_diag`: Update dp using diagonal coefficients from the linear system
/// - `rhie_chow/store_grad_p`: Store the current pressure gradient for later use
/// - `rhie_chow/grad_p_update`: Recompute the pressure gradient after pressure update
/// - `rhie_chow/correct_velocity_delta`: Apply Rhie-Chow velocity correction
///
/// # Arguments
///
/// * `system` - The equation system (needed to infer momentum-pressure coupling)
/// * `dp_field` - The name of the pressure correction field in the state layout
/// * `require_vector2_momentum` - Whether to require Vector2 momentum field
/// * `require_pressure_gradient` - Whether to require pressure gradient fields
/// * `dp_formulation` - How `d_p` is computed (see [`DpFormulation`])
///
/// # Errors
///
/// Returns an error if the momentum-pressure coupling cannot be inferred from the system.
pub fn rhie_chow_aux_module(
    system: &crate::solver::model::backend::ast::EquationSystem,
    dp_field: &'static str,
    require_vector2_momentum: bool,
    require_pressure_gradient: bool,
    dp_formulation: DpFormulation,
) -> Result<KernelBundleModule, String> {
    // Infer coupling once at module construction time
    let coupling =
        crate::solver::model::invariants::infer_unique_momentum_pressure_coupling_referencing_dp_system(
            system, dp_field,
        )
        .map_err(|e| format!("rhie_chow_aux_module: {e}"))?;

    let pressure_name = coupling.pressure.name();

    // Precompute derived gradient field names once
    // These are interned/leaked to obtain &'static str for PortManifest
    let grad_p_name: &'static str = Box::leak(format!("grad_{}", pressure_name).into_boxed_str());
    let grad_p_old_name: &'static str =
        Box::leak(format!("grad_{}_old", pressure_name).into_boxed_str());

    let kernel_dp_init = KernelId("dp_init");
    let kernel_dp_update_from_diag = KernelId("dp_update_from_diag");
    let kernel_dp_update_store_grad_p_fused = KernelId("rhie_chow/dp_update_store_grad_p_fused");
    let kernel_dp_update_store_grad_p_grad_p_update_fused =
        KernelId("rhie_chow/dp_update_store_grad_p_grad_p_update_fused");
    let kernel_dp_update_store_grad_p_grad_p_update_correct_velocity_delta_fused =
        KernelId("rhie_chow/dp_update_store_grad_p_grad_p_update_correct_velocity_delta_fused");
    let kernel_generic_coupled_update_dp_init_fused =
        KernelId("generic_coupled/update_dp_init_fused");
    let kernel_dp_init_dp_update_store_grad_p_grad_p_update_correct_velocity_delta_fused =
        KernelId::RHIE_CHOW_DP_INIT_DP_UPDATE_STORE_GRAD_P_GRAD_P_UPDATE_CORRECT_VELOCITY_DELTA_FUSED;
    let kernel_store_grad_p = KernelId("rhie_chow/store_grad_p");
    let kernel_grad_p_update = KernelId::RHIE_CHOW_GRAD_P_UPDATE;
    let kernel_rhie_chow_correct_velocity_delta = KernelId("rhie_chow/correct_velocity_delta");
    let kernel_grad_p_update_correct_velocity_delta_fused =
        KernelId::RHIE_CHOW_GRAD_P_UPDATE_CORRECT_VELOCITY_DELTA_FUSED;
    let kernel_store_grad_p_grad_p_update_fused =
        KernelId::RHIE_CHOW_STORE_GRAD_P_GRAD_P_UPDATE_FUSED;

    let kernels = vec![
        ModelKernelSpec {
            id: kernel_dp_init,
            phase: KernelPhaseId::Update,
            dispatch: DispatchKindId::Cells,
            condition: KernelConditionId::Always,
        },
        ModelKernelSpec {
            id: kernel_dp_update_from_diag,
            phase: KernelPhaseId::Update,
            dispatch: DispatchKindId::Cells,
            condition: KernelConditionId::Always,
        },
        // Snapshot the pressure gradient before it is recomputed so velocity correction can use
        // the change in pressure gradient within the same nonlinear iteration.
        ModelKernelSpec {
            id: kernel_store_grad_p,
            phase: KernelPhaseId::Update,
            dispatch: DispatchKindId::Cells,
            condition: KernelConditionId::Always,
        },
        // Recompute `grad(p)` after the pressure update so velocity correction can use the
        // change in pressure gradient within the same nonlinear iteration.
        ModelKernelSpec {
            id: kernel_grad_p_update,
            phase: KernelPhaseId::Update,
            dispatch: DispatchKindId::Cells,
            condition: KernelConditionId::Always,
        },
        ModelKernelSpec {
            id: kernel_rhie_chow_correct_velocity_delta,
            phase: KernelPhaseId::Update,
            dispatch: DispatchKindId::Cells,
            condition: KernelConditionId::Always,
        },
    ];

    // Clone coupling data for the generator closures
    let coupling_for_dp_update = coupling;
    let coupling_for_grad_update = coupling;
    let coupling_for_correct = coupling;

    let generators = vec![
        ModelKernelGeneratorSpec::new_dsl(kernel_dp_init, move |model, _schemes| {
            generate_dp_init_kernel_program(model, dp_field, dp_formulation)
        }),
        ModelKernelGeneratorSpec::new_dsl(kernel_dp_update_from_diag, move |model, _schemes| {
            generate_dp_update_from_diag_kernel_program(
                model,
                dp_field,
                coupling_for_dp_update,
                dp_formulation,
            )
        }),
        ModelKernelGeneratorSpec::new_dsl(kernel_store_grad_p, move |model, _schemes| {
            generate_rhie_chow_store_grad_p_kernel_program(model, grad_p_name, grad_p_old_name)
        }),
        ModelKernelGeneratorSpec::new_dsl(kernel_grad_p_update, move |model, _schemes| {
            generate_rhie_chow_grad_p_update_kernel_program(
                model,
                coupling_for_grad_update,
                grad_p_name,
            )
        }),
        ModelKernelGeneratorSpec::new_dsl(
            kernel_rhie_chow_correct_velocity_delta,
            move |model, _schemes| {
                generate_rhie_chow_correct_velocity_delta_kernel_program(
                    model,
                    dp_field,
                    coupling_for_correct,
                    grad_p_name,
                    grad_p_old_name,
                )
            },
        ),
    ];

    // Build PortManifest with required fields
    use crate::solver::dimensions::{PressureGradient, UnitDimension, D_P};
    use crate::solver::ir::ports::{FieldSpec, PortFieldKind, PortManifest};

    let port_manifest = Some(PortManifest {
        fields: vec![
            // dp field: Scalar with D_P unit
            FieldSpec {
                name: dp_field,
                kind: PortFieldKind::Scalar,
                unit: D_P::UNIT,
            },
            // grad_p field: Vector2 with PRESSURE_GRADIENT unit
            FieldSpec {
                name: grad_p_name,
                kind: PortFieldKind::Vector2,
                unit: PressureGradient::UNIT,
            },
            // grad_p_old field: Vector2 with PRESSURE_GRADIENT unit
            FieldSpec {
                name: grad_p_old_name,
                kind: PortFieldKind::Vector2,
                unit: PressureGradient::UNIT,
            },
            // momentum field: Vector2 with ANY_DIMENSION (dynamic dimension)
            FieldSpec {
                name: coupling.momentum.name(),
                kind: PortFieldKind::Vector2,
                unit: crate::solver::ir::ports::ANY_DIMENSION,
            },
        ],
        ..Default::default()
    });

    let fusion_rules = vec![
        ModelKernelFusionRule {
            name: "generic_coupled:update_dp_init_v1",
            priority: 140,
            phase: KernelPhaseId::Update,
            pattern: vec![
                KernelPatternAtom::with_dispatch(
                    KernelId::GENERIC_COUPLED_UPDATE,
                    DispatchKindId::Cells,
                ),
                KernelPatternAtom::with_dispatch(kernel_dp_init, DispatchKindId::Cells),
            ],
            replacement: ModelKernelSpec {
                id: kernel_generic_coupled_update_dp_init_fused,
                phase: KernelPhaseId::Update,
                dispatch: DispatchKindId::Cells,
                condition: KernelConditionId::Always,
            },
            guards: vec![
                FusionGuard::RequiresStepping(KernelFusionStepping::Coupled),
                FusionGuard::RequiresModule("generic_coupled"),
                FusionGuard::RequiresModule("rhie_chow_aux"),
                FusionGuard::MinPolicy(crate::solver::model::kernel::KernelFusionPolicy::Safe),
                FusionGuard::ExactPolicy(crate::solver::model::kernel::KernelFusionPolicy::Safe),
            ],
            binding_remaps: vec![],
            expected_hazards: vec![],
        },
        ModelKernelFusionRule {
            name: "rhie_chow:dp_update_store_grad_p_v1",
            priority: 100,
            phase: KernelPhaseId::Update,
            pattern: vec![
                KernelPatternAtom::with_dispatch(kernel_dp_update_from_diag, DispatchKindId::Cells),
                KernelPatternAtom::with_dispatch(kernel_store_grad_p, DispatchKindId::Cells),
            ],
            replacement: ModelKernelSpec {
                id: kernel_dp_update_store_grad_p_fused,
                phase: KernelPhaseId::Update,
                dispatch: DispatchKindId::Cells,
                condition: KernelConditionId::Always,
            },
            guards: vec![
                FusionGuard::RequiresStepping(KernelFusionStepping::Coupled),
                FusionGuard::RequiresModule("rhie_chow_aux"),
                FusionGuard::MinPolicy(crate::solver::model::kernel::KernelFusionPolicy::Safe),
            ],
            binding_remaps: vec![],
            expected_hazards: vec![],
        },
        ModelKernelFusionRule {
            name:
                "rhie_chow:dp_init_dp_update_store_grad_p_grad_p_update_correct_velocity_delta_v1",
            priority: 130,
            phase: KernelPhaseId::Update,
            pattern: vec![
                KernelPatternAtom::with_dispatch(kernel_dp_init, DispatchKindId::Cells),
                KernelPatternAtom::with_dispatch(kernel_dp_update_from_diag, DispatchKindId::Cells),
                KernelPatternAtom::with_dispatch(kernel_store_grad_p, DispatchKindId::Cells),
                KernelPatternAtom::with_dispatch(kernel_grad_p_update, DispatchKindId::Cells),
                KernelPatternAtom::with_dispatch(
                    kernel_rhie_chow_correct_velocity_delta,
                    DispatchKindId::Cells,
                ),
            ],
            replacement: ModelKernelSpec {
                id:
                    kernel_dp_init_dp_update_store_grad_p_grad_p_update_correct_velocity_delta_fused,
                phase: KernelPhaseId::Update,
                dispatch: DispatchKindId::Cells,
                condition: KernelConditionId::Always,
            },
            guards: vec![
                FusionGuard::RequiresStepping(KernelFusionStepping::Coupled),
                FusionGuard::RequiresModule("rhie_chow_aux"),
                FusionGuard::MinPolicy(
                    crate::solver::model::kernel::KernelFusionPolicy::Aggressive,
                ),
            ],
            binding_remaps: vec![],
            expected_hazards: {
                let mut hazards = vec![ExpectedHazard {
                    kind: HazardKind::WAW,
                    kernel_id: "dp_update_from_diag",
                    justification: "dp_init and dp_update both write d_p at same index; \
                                    dp_update overwrites dp_init's zero, last-writer-wins is correct \
                                    because both are per-cell scalar writes at idx*stride+offset",
                }];
                // Only the assembled-matrix formulations read d_p in
                // dp_init (seed-if-zero) and dp_update (damped update); the
                // whitelist must stay exact per formulation.
                if dp_formulation.uses_assembled_matrix() {
                    hazards.push(ExpectedHazard {
                        kind: HazardKind::RAW,
                        kernel_id: "dp_update_from_diag",
                        justification: "with an assembled-matrix DpFormulation, dp_update reads \
                                        the d_p seeded by dp_init for its damped update; both access \
                                        the same cell at idx*stride+offset, so the sequential \
                                        read-after-write within the fused body is the intended \
                                        semantics",
                    });
                    hazards.push(ExpectedHazard {
                        kind: HazardKind::WAR,
                        kernel_id: "dp_update_from_diag",
                        justification: "with an assembled-matrix DpFormulation, dp_init reads d_p \
                                        (seed-if-zero) before dp_update overwrites it; per-cell at \
                                        idx*stride+offset, sequential within the fused body",
                    });
                }
                hazards.push(ExpectedHazard {
                    kind: HazardKind::WAR,
                    kernel_id: "rhie_chow/grad_p_update",
                    justification: "store_grad_p reads grad_p before grad_p_update writes it; \
                                    safe because both operate on the same cell index (idx) and \
                                    store_grad_p's read is sequentially before grad_p_update's write \
                                    in the fused body",
                });
                hazards.push(ExpectedHazard {
                    kind: HazardKind::RAW,
                    kernel_id: "rhie_chow/correct_velocity_delta",
                    justification: "correct_velocity_delta reads d_p, grad_p, grad_p_old written \
                                    by earlier segments; safe because all accesses are per-cell at \
                                    idx*stride+offset with no cross-cell dependencies",
                });
                hazards
            },
        },
        ModelKernelFusionRule {
            name: "rhie_chow:dp_update_store_grad_p_grad_p_update_correct_velocity_delta_v1",
            priority: 120,
            phase: KernelPhaseId::Update,
            pattern: vec![
                KernelPatternAtom::with_dispatch(kernel_dp_update_from_diag, DispatchKindId::Cells),
                KernelPatternAtom::with_dispatch(kernel_store_grad_p, DispatchKindId::Cells),
                KernelPatternAtom::with_dispatch(kernel_grad_p_update, DispatchKindId::Cells),
                KernelPatternAtom::with_dispatch(
                    kernel_rhie_chow_correct_velocity_delta,
                    DispatchKindId::Cells,
                ),
            ],
            replacement: ModelKernelSpec {
                id: kernel_dp_update_store_grad_p_grad_p_update_correct_velocity_delta_fused,
                phase: KernelPhaseId::Update,
                dispatch: DispatchKindId::Cells,
                condition: KernelConditionId::Always,
            },
            guards: vec![
                FusionGuard::RequiresStepping(KernelFusionStepping::Coupled),
                FusionGuard::RequiresModule("rhie_chow_aux"),
                FusionGuard::MinPolicy(
                    crate::solver::model::kernel::KernelFusionPolicy::Aggressive,
                ),
            ],
            binding_remaps: vec![],
            expected_hazards: vec![
                ExpectedHazard {
                    kind: HazardKind::WAR,
                    kernel_id: "rhie_chow/grad_p_update",
                    justification: "store_grad_p reads grad_p before grad_p_update writes it; \
                                    safe because both operate on the same cell index",
                },
                ExpectedHazard {
                    kind: HazardKind::RAW,
                    kernel_id: "rhie_chow/correct_velocity_delta",
                    justification: "correct_velocity_delta reads d_p, grad_p, grad_p_old written \
                                    by earlier segments; safe because all accesses are per-cell",
                },
            ],
        },
        ModelKernelFusionRule {
            name: "rhie_chow:dp_update_store_grad_p_grad_p_update_v1",
            priority: 110,
            phase: KernelPhaseId::Update,
            pattern: vec![
                KernelPatternAtom::with_dispatch(kernel_dp_update_from_diag, DispatchKindId::Cells),
                KernelPatternAtom::with_dispatch(kernel_store_grad_p, DispatchKindId::Cells),
                KernelPatternAtom::with_dispatch(kernel_grad_p_update, DispatchKindId::Cells),
            ],
            replacement: ModelKernelSpec {
                id: kernel_dp_update_store_grad_p_grad_p_update_fused,
                phase: KernelPhaseId::Update,
                dispatch: DispatchKindId::Cells,
                condition: KernelConditionId::Always,
            },
            guards: vec![
                FusionGuard::RequiresStepping(KernelFusionStepping::Coupled),
                FusionGuard::RequiresModule("rhie_chow_aux"),
                FusionGuard::MinPolicy(
                    crate::solver::model::kernel::KernelFusionPolicy::Aggressive,
                ),
            ],
            binding_remaps: vec![],
            expected_hazards: vec![
                ExpectedHazard {
                    kind: HazardKind::WAR,
                    kernel_id: "rhie_chow/grad_p_update",
                    justification: "store_grad_p reads grad_p before grad_p_update writes it; \
                                    safe because both operate on the same cell index",
                },
            ],
        },
        // Standalone fusion rule for grad_p_update + correct_velocity_delta (aggressive-only)
        ModelKernelFusionRule {
            name: "rhie_chow:grad_p_update_correct_velocity_delta_v1",
            priority: 105,
            phase: KernelPhaseId::Update,
            pattern: vec![
                KernelPatternAtom::with_dispatch(kernel_grad_p_update, DispatchKindId::Cells),
                KernelPatternAtom::with_dispatch(
                    kernel_rhie_chow_correct_velocity_delta,
                    DispatchKindId::Cells,
                ),
            ],
            replacement: ModelKernelSpec {
                id: kernel_grad_p_update_correct_velocity_delta_fused,
                phase: KernelPhaseId::Update,
                dispatch: DispatchKindId::Cells,
                condition: KernelConditionId::Always,
            },
            guards: vec![
                FusionGuard::RequiresStepping(KernelFusionStepping::Coupled),
                FusionGuard::RequiresModule("rhie_chow_aux"),
                FusionGuard::MinPolicy(
                    crate::solver::model::kernel::KernelFusionPolicy::Aggressive,
                ),
            ],
            binding_remaps: vec![],
            expected_hazards: vec![
                ExpectedHazard {
                    kind: HazardKind::RAW,
                    kernel_id: "rhie_chow/correct_velocity_delta",
                    justification: "correct_velocity_delta reads grad_p written by grad_p_update; \
                                    safe because both operate on the same cell index",
                },
            ],
        },
        // Standalone fusion rule for store_grad_p + grad_p_update (aggressive-only)
        ModelKernelFusionRule {
            name: "rhie_chow:store_grad_p_grad_p_update_v1",
            priority: 106,
            phase: KernelPhaseId::Update,
            pattern: vec![
                KernelPatternAtom::with_dispatch(kernel_store_grad_p, DispatchKindId::Cells),
                KernelPatternAtom::with_dispatch(kernel_grad_p_update, DispatchKindId::Cells),
            ],
            replacement: ModelKernelSpec {
                id: kernel_store_grad_p_grad_p_update_fused,
                phase: KernelPhaseId::Update,
                dispatch: DispatchKindId::Cells,
                condition: KernelConditionId::Always,
            },
            guards: vec![
                FusionGuard::RequiresStepping(KernelFusionStepping::Coupled),
                FusionGuard::RequiresModule("rhie_chow_aux"),
                FusionGuard::MinPolicy(
                    crate::solver::model::kernel::KernelFusionPolicy::Aggressive,
                ),
            ],
            binding_remaps: vec![],
            expected_hazards: vec![
                ExpectedHazard {
                    kind: HazardKind::WAR,
                    kernel_id: "rhie_chow/grad_p_update",
                    justification: "store_grad_p reads grad_p before grad_p_update writes it; \
                                    safe because both operate on the same cell index",
                },
            ],
        },
    ];

    Ok(KernelBundleModule {
        name: "rhie_chow_aux",
        kernels,
        generators,
        fusion_rules,
        invariants: vec![
            ModuleInvariant::RequireUniqueMomentumPressureCouplingReferencingDp {
                dp_field,
                require_vector2_momentum,
                require_pressure_gradient,
            },
        ],
        port_manifest,
        ..Default::default()
    })
}

fn generate_dp_init_kernel_program(
    model: &crate::solver::model::ModelSpec,
    dp_field: &str,
    dp_formulation: DpFormulation,
) -> Result<KernelProgram, String> {
    use crate::solver::model::ports::dimensions::D_P;
    use crate::solver::model::ports::PortRegistry;

    let mut registry = PortRegistry::new(model.state_layout.clone());

    let d_p = registry
        .register_scalar_field::<D_P>(dp_field)
        .map_err(|e| format!("dp_init: {e}"))?;

    let stride = registry.state_layout().stride();
    let d_p_offset = d_p.offset();
    let mut program = KernelProgram::new(
        "dp_init",
        DispatchDomain::Cells,
        rhie_chow_state_launch(stride),
        rhie_chow_state_bindings(),
    );
    let indexing_stmts = vec![dsl::let_expr("base", Expr::ident("idx") * stride)];
    let body_stmts = match dp_formulation {
        DpFormulation::ClosedForm => vec![dsl::assign_expr(
            dsl::array_access("state", Expr::ident("base") + d_p_offset),
            Expr::lit_f32(0.0),
        )],
        // The assembled-matrix formulations damp d_p across outer
        // iterations, so the field must PERSIST between updates: seed the
        // closed form once (while d_p is still zero from state init)
        // instead of re-zeroing.
        DpFormulation::FromAssembledDiagonal { .. } | DpFormulation::FromAssembledRowSum { .. } => {
            let d_p_old = dsl::array_access("state", Expr::ident("base") + d_p_offset);
            let rho = dsl::max(
                Expr::ident("constants").field("density"),
                Expr::lit_f32(1e-12),
            );
            let dt = dsl::max(Expr::ident("constants").field("dt"), Expr::lit_f32(0.0));
            let seed = Expr::ident("constants").field("alpha_u") * dt / rho;
            vec![dsl::assign_expr(
                dsl::array_access("state", Expr::ident("base") + d_p_offset),
                dsl::select(d_p_old.clone(), seed, d_p_old.eq(Expr::lit_f32(0.0))),
            )]
        }
    };
    program.indexing = indexing_stmts;
    program.body = body_stmts;
    program
        .side_effects
        .read_set
        .insert(EffectResource::binding(0, 1));
    if dp_formulation.uses_assembled_matrix() {
        program
            .side_effects
            .read_set
            .insert(EffectResource::component(
                0,
                0,
                format!("state:{d_p_offset}"),
            ));
    }
    program
        .side_effects
        .write_set
        .insert(EffectResource::component(
            0,
            0,
            format!("state:{d_p_offset}"),
        ));
    Ok(program)
}

fn rhie_chow_state_launch(state_stride: u32) -> LaunchSemantics {
    LaunchSemantics::new(
        [64, 1, 1],
        "global_id.y * constants.stride_x + global_id.x",
        Some(format!("idx >= (arrayLength(&state) / {state_stride}u)")),
    )
}

fn rhie_chow_state_bindings() -> Vec<KernelBinding> {
    vec![
        KernelBinding::new(0, 0, "state", "array<f32>", BindingAccess::ReadWriteStorage),
        KernelBinding::new(0, 1, "constants", "Constants", BindingAccess::Uniform),
    ]
}

fn rhie_chow_grad_p_update_bindings() -> Vec<KernelBinding> {
    vec![
        KernelBinding::new(0, 0, "state", "array<f32>", BindingAccess::ReadWriteStorage),
        KernelBinding::new(0, 1, "constants", "Constants", BindingAccess::Uniform),
        KernelBinding::new(
            1,
            0,
            "face_owner",
            "array<u32>",
            BindingAccess::ReadOnlyStorage,
        ),
        KernelBinding::new(
            1,
            1,
            "face_neighbor",
            "array<i32>",
            BindingAccess::ReadOnlyStorage,
        ),
        KernelBinding::new(
            1,
            2,
            "face_areas",
            "array<f32>",
            BindingAccess::ReadOnlyStorage,
        ),
        KernelBinding::new(
            1,
            3,
            "face_normals",
            "array<Vector2>",
            BindingAccess::ReadOnlyStorage,
        ),
        KernelBinding::new(
            1,
            13,
            "face_centers",
            "array<Vector2>",
            BindingAccess::ReadOnlyStorage,
        ),
        KernelBinding::new(
            1,
            4,
            "cell_centers",
            "array<Vector2>",
            BindingAccess::ReadOnlyStorage,
        ),
        KernelBinding::new(
            1,
            5,
            "cell_vols",
            "array<f32>",
            BindingAccess::ReadOnlyStorage,
        ),
        KernelBinding::new(
            1,
            6,
            "cell_face_offsets",
            "array<u32>",
            BindingAccess::ReadOnlyStorage,
        ),
        KernelBinding::new(
            1,
            7,
            "cell_faces",
            "array<u32>",
            BindingAccess::ReadOnlyStorage,
        ),
        KernelBinding::new(
            1,
            12,
            "face_boundary",
            "array<u32>",
            BindingAccess::ReadOnlyStorage,
        ),
        KernelBinding::new(
            2,
            0,
            "bc_kind",
            "array<u32>",
            BindingAccess::ReadOnlyStorage,
        ),
        KernelBinding::new(
            2,
            1,
            "bc_value",
            "array<f32>",
            BindingAccess::ReadOnlyStorage,
        ),
    ]
}

fn generate_dp_update_from_diag_kernel_program(
    model: &crate::solver::model::ModelSpec,
    dp_field: &str,
    coupling: crate::solver::model::invariants::MomentumPressureCoupling,
    dp_formulation: DpFormulation,
) -> Result<KernelProgram, String> {
    use crate::solver::model::ports::dimensions::{AnyDimension, D_P};
    use crate::solver::model::ports::PortRegistry;

    let mut registry = PortRegistry::new(model.state_layout.clone());

    let d_p = registry
        .register_scalar_field::<D_P>(dp_field)
        .map_err(|e| format!("dp_update_from_diag: {e}"))?;

    let momentum = coupling.momentum;

    // Register momentum field to validate it exists (offset not needed here)
    let _u = registry
        .register_vector2_field::<AnyDimension>(momentum.name())
        .map_err(|e| format!("dp_update_from_diag: {e}"))?;

    let stride = registry.state_layout().stride();
    let d_p_offset = d_p.offset();

    match dp_formulation {
        DpFormulation::ClosedForm => {
            let mut program = KernelProgram::new(
                "dp_update_from_diag",
                DispatchDomain::Cells,
                rhie_chow_state_launch(stride),
                rhie_chow_state_bindings(),
            );
            let indexing_stmts = vec![dsl::let_expr("base", Expr::ident("idx") * stride)];
            let preamble_stmts = vec![
                dsl::let_expr(
                    "rho",
                    dsl::max(
                        Expr::ident("constants").field("density"),
                        Expr::lit_f32(1e-12),
                    ),
                ),
                dsl::let_expr(
                    "dt",
                    dsl::max(Expr::ident("constants").field("dt"), Expr::lit_f32(0.0)),
                ),
                dsl::let_expr(
                    "d_p",
                    Expr::ident("constants").field("alpha_u") * Expr::ident("dt")
                        / Expr::ident("rho"),
                ),
            ];
            let body_stmts = vec![dsl::assign_expr(
                dsl::array_access("state", Expr::ident("base") + d_p_offset),
                Expr::ident("d_p"),
            )];
            program.indexing = indexing_stmts;
            program.preamble = preamble_stmts;
            program.body = body_stmts;
            program
                .side_effects
                .read_set
                .insert(EffectResource::binding(0, 1));
            program
                .side_effects
                .write_set
                .insert(EffectResource::component(
                    0,
                    0,
                    format!("state:{d_p_offset}"),
                ));

            Ok(program)
        }
        DpFormulation::FromAssembledDiagonal {
            include_relaxation,
            theta,
        } => generate_dp_update_from_assembled_diagonal(
            model,
            stride,
            d_p_offset,
            momentum.name(),
            include_relaxation,
            theta,
            DpDenominator::Diagonal,
        ),
        DpFormulation::FromAssembledRowSum { theta } => {
            generate_dp_update_from_assembled_diagonal(
                model,
                stride,
                d_p_offset,
                momentum.name(),
                // SIMPLEC carries no relaxation factor by construction.
                false,
                theta,
                DpDenominator::RowSum,
            )
        }
    }
}

/// Which momentum-row reduction feeds the d_p denominator.
#[derive(Clone, Copy, PartialEq)]
enum DpDenominator {
    /// `a_P` — the same-component diagonal entry (OpenFOAM rAU analogue).
    Diagonal,
    /// `Σ_rank a(c,c)` — same-component row sum (SIMPLEC).
    RowSum,
}

/// Assembled-matrix coupling coefficient, damped across outer iterations
/// (`d_p <- theta*new + (1-theta)*old`):
/// - `DpDenominator::Diagonal`: `d_p = [alpha_u *] V / a_P` with `a_P` the
///   assembled momentum diagonal (OpenFOAM rAU analogue);
/// - `DpDenominator::RowSum`: `d_p = V / Σ_rank a(c,c)` — same-component
///   row sum (SIMPLEC), no relaxation factor.
/// Either denominator is averaged over the two momentum components.
///
/// The kernel runs in the Update phase, after this outer iteration's
/// assembly and solve, so `matrix_values` holds the matrix assembled with
/// the PREVIOUS iteration's d_p — the damped update converges this lagged
/// loop to the fixed point. The matrix entry layout matches the Schur
/// setup kernel (`generate_generic_coupled_schur_setup`):
///   a(c,c at rank r) = matrix_values[scalar_offset*S^2 + c*num_neighbors*S + r*S + c]
/// with ranks in FluxLayout component order (rank-keyed, like bc tables —
/// NOT state offsets). Falls back to the closed form while the denominator
/// is zero (matrix not yet assembled).
fn generate_dp_update_from_assembled_diagonal(
    model: &crate::solver::model::ModelSpec,
    stride: u32,
    d_p_offset: u32,
    momentum_name: &str,
    include_relaxation: bool,
    theta: f32,
    denominator: DpDenominator,
) -> Result<KernelProgram, String> {
    let flux_layout = crate::solver::ir::FluxLayout::from_system(&model.system);
    let unknowns: Vec<String> = flux_layout
        .components
        .iter()
        .map(|c| c.name.clone())
        .collect();
    let s = unknowns.len() as u32;
    let ux_rank = unknowns
        .iter()
        .position(|n| n == &format!("{momentum_name}_x"))
        .ok_or_else(|| {
            format!("dp_update_from_diag: momentum component {momentum_name}_x not in flux layout")
        })? as u32;
    let uy_rank = unknowns
        .iter()
        .position(|n| n == &format!("{momentum_name}_y"))
        .ok_or_else(|| {
            format!("dp_update_from_diag: momentum component {momentum_name}_y not in flux layout")
        })? as u32;

    let mut bindings = rhie_chow_state_bindings();
    // Slot choices merge with the rhie_chow fused families: cell_vols
    // matches grad_p_update's (1,5); the CSR buffers take the free slots
    // (1,8..10). All names resolve through the generic-coupled backend
    // ResourceRegistry (mesh: scalar_row_offsets/diagonal_indices/cell_vols;
    // linear ports: matrix_values).
    bindings.push(KernelBinding::new(
        1,
        5,
        "cell_vols",
        "array<f32>",
        BindingAccess::ReadOnlyStorage,
    ));
    bindings.push(KernelBinding::new(
        1,
        8,
        "scalar_row_offsets",
        "array<u32>",
        BindingAccess::ReadOnlyStorage,
    ));
    bindings.push(KernelBinding::new(
        1,
        9,
        "diagonal_indices",
        "array<u32>",
        BindingAccess::ReadOnlyStorage,
    ));
    bindings.push(KernelBinding::new(
        1,
        10,
        "matrix_values",
        "array<f32>",
        BindingAccess::ReadOnlyStorage,
    ));

    let mut program = KernelProgram::new(
        "dp_update_from_diag",
        DispatchDomain::Cells,
        rhie_chow_state_launch(stride),
        bindings,
    );
    let indexing_stmts = vec![dsl::let_expr("base", Expr::ident("idx") * stride)];

    let alpha_u = Expr::ident("constants").field("alpha_u");
    let closed_form_scale: Expr = if include_relaxation {
        alpha_u.clone()
    } else {
        Expr::lit_f32(1.0)
    };
    let mut preamble_stmts = vec![
        dsl::let_expr(
            "rho",
            dsl::max(
                Expr::ident("constants").field("density"),
                Expr::lit_f32(1e-12),
            ),
        ),
        dsl::let_expr(
            "dt",
            dsl::max(Expr::ident("constants").field("dt"), Expr::lit_f32(0.0)),
        ),
        // Closed-form fallback (used while the matrix is unassembled). Keep
        // the historical alpha_u scaling here regardless of
        // include_relaxation: it is only the pre-assembly seed magnitude.
        dsl::let_expr(
            "d_p_closed",
            alpha_u * Expr::ident("dt") / Expr::ident("rho"),
        ),
        dsl::let_expr(
            "scalar_offset",
            dsl::array_access("scalar_row_offsets", Expr::ident("idx")),
        ),
        dsl::let_expr(
            "num_neighbors",
            dsl::array_access("scalar_row_offsets", Expr::ident("idx") + 1u32)
                - Expr::ident("scalar_offset"),
        ),
        dsl::let_expr(
            "diag_rank",
            dsl::array_access("diagonal_indices", Expr::ident("idx"))
                - Expr::ident("scalar_offset"),
        ),
        dsl::let_expr(
            "row_stride",
            Expr::ident("num_neighbors") * Expr::lit_u32(s),
        ),
        dsl::let_expr(
            "mat_base",
            Expr::ident("scalar_offset") * Expr::lit_u32(s * s)
                + Expr::ident("diag_rank") * Expr::lit_u32(s),
        ),
    ];
    match denominator {
        DpDenominator::Diagonal => preamble_stmts.extend([
            dsl::let_expr(
                "a_u_x",
                dsl::array_access(
                    "matrix_values",
                    Expr::ident("mat_base")
                        + Expr::lit_u32(ux_rank) * Expr::ident("row_stride")
                        + Expr::lit_u32(ux_rank),
                ),
            ),
            dsl::let_expr(
                "a_u_y",
                dsl::array_access(
                    "matrix_values",
                    Expr::ident("mat_base")
                        + Expr::lit_u32(uy_rank) * Expr::ident("row_stride")
                        + Expr::lit_u32(uy_rank),
                ),
            ),
        ]),
        DpDenominator::RowSum => preamble_stmts.extend([
            dsl::let_expr(
                "row_base_x",
                Expr::ident("scalar_offset") * Expr::lit_u32(s * s)
                    + Expr::lit_u32(ux_rank) * Expr::ident("row_stride"),
            ),
            dsl::let_expr(
                "row_base_y",
                Expr::ident("scalar_offset") * Expr::lit_u32(s * s)
                    + Expr::lit_u32(uy_rank) * Expr::ident("row_stride"),
            ),
            dsl::var_expr("a_u_x", Expr::lit_f32(0.0)),
            dsl::var_expr("a_u_y", Expr::lit_f32(0.0)),
            dsl::for_loop_expr(
                dsl::for_init_var_expr("r", Expr::lit_u32(0)),
                Expr::ident("r").lt(Expr::ident("num_neighbors")),
                ForStep::Increment(Expr::ident("r")),
                dsl::block(vec![
                    dsl::assign_op_expr(
                        AssignOp::Add,
                        Expr::ident("a_u_x"),
                        dsl::array_access(
                            "matrix_values",
                            Expr::ident("row_base_x")
                                + Expr::ident("r") * Expr::lit_u32(s)
                                + Expr::lit_u32(ux_rank),
                        ),
                    ),
                    dsl::assign_op_expr(
                        AssignOp::Add,
                        Expr::ident("a_u_y"),
                        dsl::array_access(
                            "matrix_values",
                            Expr::ident("row_base_y")
                                + Expr::ident("r") * Expr::lit_u32(s)
                                + Expr::lit_u32(uy_rank),
                        ),
                    ),
                ]),
            ),
        ]),
    }
    preamble_stmts.extend([
        dsl::let_expr(
            "a_bar",
            Expr::lit_f32(0.5) * (Expr::ident("a_u_x") + Expr::ident("a_u_y")),
        ),
        dsl::let_expr("vol", dsl::array_access("cell_vols", Expr::ident("idx"))),
        dsl::let_expr(
            "d_p_diag",
            closed_form_scale * Expr::ident("vol")
                / dsl::max(Expr::ident("a_bar"), Expr::lit_f32(1e-30)),
        ),
        dsl::let_expr(
            "d_p_new",
            dsl::select(
                Expr::ident("d_p_closed"),
                Expr::ident("d_p_diag"),
                Expr::ident("a_bar").gt(Expr::lit_f32(1e-30)),
            ),
        ),
        dsl::let_expr(
            "d_p_old",
            dsl::array_access("state", Expr::ident("base") + d_p_offset),
        ),
        dsl::let_expr(
            "d_p",
            Expr::lit_f32(theta) * Expr::ident("d_p_new")
                + Expr::lit_f32(1.0 - theta) * Expr::ident("d_p_old"),
        ),
    ]);
    let body_stmts = vec![dsl::assign_expr(
        dsl::array_access("state", Expr::ident("base") + d_p_offset),
        Expr::ident("d_p"),
    )];
    program.indexing = indexing_stmts;
    program.preamble = preamble_stmts;
    program.body = body_stmts;
    program
        .side_effects
        .read_set
        .insert(EffectResource::binding(0, 1));
    for slot in [5u32, 8, 9, 10] {
        program
            .side_effects
            .read_set
            .insert(EffectResource::binding(1, slot));
    }
    program
        .side_effects
        .read_set
        .insert(EffectResource::component(
            0,
            0,
            format!("state:{d_p_offset}"),
        ));
    program
        .side_effects
        .write_set
        .insert(EffectResource::component(
            0,
            0,
            format!("state:{d_p_offset}"),
        ));

    Ok(program)
}

fn generate_rhie_chow_grad_p_update_kernel_program(
    model: &crate::solver::model::ModelSpec,
    coupling: crate::solver::model::invariants::MomentumPressureCoupling,
    grad_p_name: &'static str,
) -> Result<KernelProgram, String> {
    use crate::solver::model::ports::dimensions::{Pressure, PressureGradient};
    use crate::solver::model::ports::PortRegistry;

    let mut registry = PortRegistry::new(model.state_layout.clone());
    let p = registry
        .register_scalar_field::<Pressure>(coupling.pressure.name())
        .map_err(|e| format!("rhie_chow/grad_p_update: {e}"))?;
    let grad_p = registry
        .register_vector2_field::<PressureGradient>(grad_p_name)
        .map_err(|e| format!("rhie_chow/grad_p_update: {e}"))?;

    let state_stride = registry.state_layout().stride();
    let p_offset = p.offset();
    let grad_p_x = grad_p
        .component(XY::X.to_usize() as u32)
        .map(|c| c.full_offset())
        .ok_or("rhie_chow/grad_p_update: grad_p component 0")?;
    let grad_p_y = grad_p
        .component(XY::Y.to_usize() as u32)
        .map(|c| c.full_offset())
        .ok_or("rhie_chow/grad_p_update: grad_p component 1")?;

    let flux_layout = crate::solver::ir::FluxLayout::from_system(&model.system);
    let p_unknown_offset = flux_layout
        .offset_for_field_component(coupling.pressure, 0)
        .ok_or_else(|| {
            format!(
                "rhie_chow/grad_p_update: missing unknown offset for pressure field '{}'",
                coupling.pressure.name()
            )
        })?;
    let unknowns_per_face = model.system.unknowns_per_cell();
    let bc = BcTable::new(Expr::ident("face_idx"), unknowns_per_face);
    let p_state_expr = dsl::array_access("state", Expr::ident("base") + p_offset);
    let p_other_state_expr =
        dsl::array_access("state", Expr::ident("other_idx") * state_stride + p_offset);
    let p_boundary_expr = bc.ghost_value(p_unknown_offset, p_state_expr.clone(), Expr::ident("d_own"));
    let p_interp_expr = p_state_expr * Expr::ident("lambda")
        + dsl::select(
            p_other_state_expr,
            p_boundary_expr,
            Expr::ident("is_boundary"),
        ) * Expr::ident("lambda_other");

    let mut program = KernelProgram::new(
        "rhie_chow/grad_p_update",
        DispatchDomain::Cells,
        rhie_chow_state_launch(state_stride),
        rhie_chow_grad_p_update_bindings(),
    );
    let indexing_stmts = vec![dsl::let_expr("base", Expr::ident("idx") * state_stride)];
    let preamble_stmts = vec![
        dsl::let_expr(
            "cell_center",
            dsl::array_access("cell_centers", Expr::ident("idx")),
        ),
        dsl::let_typed_expr(
            "cell_center_vec",
            Type::vec2_f32(),
            dsl::vec2_f32(
                Expr::ident("cell_center").field("x"),
                Expr::ident("cell_center").field("y"),
            ),
        ),
        dsl::let_expr("vol", dsl::array_access("cell_vols", Expr::ident("idx"))),
        dsl::let_expr(
            "start",
            dsl::array_access("cell_face_offsets", Expr::ident("idx")),
        ),
        dsl::let_expr(
            "end",
            dsl::array_access("cell_face_offsets", Expr::ident("idx") + 1u32),
        ),
        dsl::var_typed_expr(
            "grad_acc_p",
            Type::vec2_f32(),
            Some(dsl::vec2_f32(0.0, 0.0)),
        ),
    ];
    let body_stmts = vec![
        dsl::for_loop_expr(
            dsl::for_init_var_expr("k", Expr::ident("start")),
            Expr::ident("k").lt(Expr::ident("end")),
            ForStep::Increment(Expr::ident("k")),
            dsl::block(vec![
                dsl::let_expr(
                    "face_idx",
                    dsl::array_access("cell_faces", Expr::ident("k")),
                ),
                dsl::let_expr(
                    "owner",
                    dsl::array_access("face_owner", Expr::ident("face_idx")),
                ),
                dsl::let_expr(
                    "neighbor_raw",
                    dsl::array_access("face_neighbor", Expr::ident("face_idx")),
                ),
                dsl::let_expr(
                    "is_boundary",
                    Expr::ident("neighbor_raw").eq(Expr::lit_i32(-1)),
                ),
                dsl::let_expr(
                    "boundary_type",
                    dsl::array_access("face_boundary", Expr::ident("face_idx")),
                ),
                dsl::let_expr(
                    "area",
                    dsl::array_access("face_areas", Expr::ident("face_idx")),
                ),
                dsl::let_expr(
                    "face_center",
                    dsl::array_access("face_centers", Expr::ident("face_idx")),
                ),
                dsl::let_typed_expr(
                    "face_center_vec",
                    Type::vec2_f32(),
                    dsl::vec2_f32(
                        Expr::ident("face_center").field("x"),
                        Expr::ident("face_center").field("y"),
                    ),
                ),
                dsl::var_typed_expr(
                    "normal_vec",
                    Type::vec2_f32(),
                    Some(dsl::vec2_f32(
                        dsl::array_access("face_normals", Expr::ident("face_idx")).field("x"),
                        dsl::array_access("face_normals", Expr::ident("face_idx")).field("y"),
                    )),
                ),
                dsl::if_block_expr(
                    dsl::dot_expr(
                        Expr::ident("face_center_vec") - Expr::ident("cell_center_vec"),
                        Expr::ident("normal_vec"),
                    )
                    .lt(Expr::lit_f32(0.0)),
                    dsl::block(vec![dsl::assign_expr(
                        Expr::ident("normal_vec"),
                        -Expr::ident("normal_vec"),
                    )]),
                    None,
                ),
                dsl::var_typed_expr("other_idx", Type::U32, Some(Expr::ident("idx"))),
                dsl::var_typed_expr(
                    "other_center_vec",
                    Type::vec2_f32(),
                    Some(Expr::ident("face_center_vec")),
                ),
                dsl::if_block_expr(
                    Expr::ident("neighbor_raw").ne(Expr::lit_i32(-1)),
                    dsl::block(vec![
                        dsl::let_expr(
                            "neighbor",
                            Expr::call_named("u32", vec![Expr::ident("neighbor_raw")]),
                        ),
                        dsl::assign_expr(Expr::ident("other_idx"), Expr::ident("neighbor")),
                        dsl::if_block_expr(
                            Expr::ident("owner").ne(Expr::ident("idx")),
                            dsl::block(vec![dsl::assign_expr(
                                Expr::ident("other_idx"),
                                Expr::ident("owner"),
                            )]),
                            None,
                        ),
                        dsl::let_expr(
                            "other_center",
                            dsl::array_access("cell_centers", Expr::ident("other_idx")),
                        ),
                        dsl::assign_expr(
                            Expr::ident("other_center_vec"),
                            dsl::vec2_f32(
                                Expr::ident("other_center").field("x"),
                                Expr::ident("other_center").field("y"),
                            ),
                        ),
                    ]),
                    None,
                ),
                dsl::let_expr(
                    "d_own",
                    dsl::abs(dsl::dot_expr(
                        Expr::ident("face_center_vec") - Expr::ident("cell_center_vec"),
                        Expr::ident("normal_vec"),
                    )),
                ),
                dsl::let_expr(
                    "d_neigh",
                    dsl::abs(dsl::dot_expr(
                        Expr::ident("other_center_vec") - Expr::ident("face_center_vec"),
                        Expr::ident("normal_vec"),
                    )),
                ),
                dsl::let_expr("total_dist", Expr::ident("d_own") + Expr::ident("d_neigh")),
                dsl::var_typed_expr("lambda", Type::F32, Some(Expr::lit_f32(0.5))),
                dsl::if_block_expr(
                    Expr::ident("total_dist").gt(Expr::lit_f32(0.000001)),
                    dsl::block(vec![dsl::assign_expr(
                        Expr::ident("lambda"),
                        Expr::ident("d_neigh") / Expr::ident("total_dist"),
                    )]),
                    None,
                ),
                dsl::let_expr("lambda_other", Expr::lit_f32(1.0) - Expr::ident("lambda")),
                dsl::let_expr("_unused_boundary_type", Expr::ident("boundary_type")),
                dsl::assign_op_expr(
                    AssignOp::Add,
                    Expr::ident("grad_acc_p"),
                    Expr::ident("normal_vec") * p_interp_expr * Expr::ident("area"),
                ),
            ]),
        ),
        dsl::let_typed_expr(
            "grad_out_p",
            Type::vec2_f32(),
            Expr::ident("grad_acc_p") * Expr::lit_f32(1.0)
                / dsl::max(Expr::ident("vol"), Expr::lit_f32(1e-12)),
        ),
        dsl::assign_expr(
            dsl::array_access("state", Expr::ident("base") + grad_p_x),
            Expr::ident("grad_out_p").field("x"),
        ),
        dsl::assign_expr(
            dsl::array_access("state", Expr::ident("base") + grad_p_y),
            Expr::ident("grad_out_p").field("y"),
        ),
    ];
    program.indexing = indexing_stmts;
    program.preamble = preamble_stmts;
    program.body = body_stmts;
    for (group, binding) in [
        (0u32, 1u32),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (1, 7),
        (1, 12),
        (1, 13),
        (2, 0),
        (2, 1),
    ] {
        program
            .side_effects
            .read_set
            .insert(EffectResource::binding(group, binding));
    }
    program
        .side_effects
        .read_set
        .insert(EffectResource::component(0, 0, format!("state:{p_offset}")));
    program.side_effects.write_set.extend([
        EffectResource::component(0, 0, format!("state:{grad_p_x}")),
        EffectResource::component(0, 0, format!("state:{grad_p_y}")),
    ]);

    Ok(program)
}

fn generate_rhie_chow_store_grad_p_kernel_program(
    model: &crate::solver::model::ModelSpec,
    grad_p_name: &'static str,
    grad_p_old_name: &'static str,
) -> Result<KernelProgram, String> {
    use crate::solver::model::ports::dimensions::PressureGradient;
    use crate::solver::model::ports::PortRegistry;

    let mut registry = PortRegistry::new(model.state_layout.clone());

    // Register gradient fields using pre-computed derived names
    let grad_p = registry
        .register_vector2_field::<PressureGradient>(grad_p_name)
        .map_err(|e| {
            format!(
                "rhie_chow/store_grad_p: missing gradient field '{}': {e}",
                grad_p_name
            )
        })?;

    let grad_old = registry
        .register_vector2_field::<PressureGradient>(grad_p_old_name)
        .map_err(|e| {
            format!(
                "rhie_chow/store_grad_p: missing old gradient field '{}': {e}",
                grad_p_old_name
            )
        })?;

    let stride = registry.state_layout().stride();
    let grad_p_x = grad_p
        .component(XY::X.to_usize() as u32)
        .map(|c| c.full_offset())
        .ok_or("grad_p component 0")?;
    let grad_p_y = grad_p
        .component(XY::Y.to_usize() as u32)
        .map(|c| c.full_offset())
        .ok_or("grad_p component 1")?;
    let grad_old_x = grad_old
        .component(XY::X.to_usize() as u32)
        .map(|c| c.full_offset())
        .ok_or("grad_old component 0")?;
    let grad_old_y = grad_old
        .component(XY::Y.to_usize() as u32)
        .map(|c| c.full_offset())
        .ok_or("grad_old component 1")?;

    let mut program = KernelProgram::new(
        "rhie_chow/store_grad_p",
        DispatchDomain::Cells,
        rhie_chow_state_launch(stride),
        rhie_chow_state_bindings(),
    );
    let indexing_stmts = vec![dsl::let_expr("base", Expr::ident("idx") * stride)];
    let body_stmts = vec![
        dsl::assign_expr(
            dsl::array_access("state", Expr::ident("base") + grad_old_x),
            dsl::array_access("state", Expr::ident("base") + grad_p_x),
        ),
        dsl::assign_expr(
            dsl::array_access("state", Expr::ident("base") + grad_old_y),
            dsl::array_access("state", Expr::ident("base") + grad_p_y),
        ),
    ];
    program.indexing = indexing_stmts;
    program.body = body_stmts;
    program.side_effects.read_set.extend([
        EffectResource::component(0, 0, format!("state:{grad_p_x}")),
        EffectResource::component(0, 0, format!("state:{grad_p_y}")),
    ]);
    program.side_effects.write_set.extend([
        EffectResource::component(0, 0, format!("state:{grad_old_x}")),
        EffectResource::component(0, 0, format!("state:{grad_old_y}")),
    ]);

    Ok(program)
}

fn generate_rhie_chow_correct_velocity_delta_kernel_program(
    model: &crate::solver::model::ModelSpec,
    dp_field: &str,
    coupling: crate::solver::model::invariants::MomentumPressureCoupling,
    grad_p_name: &'static str,
    grad_p_old_name: &'static str,
) -> Result<KernelProgram, String> {
    use crate::solver::model::ports::dimensions::PressureGradient;
    use crate::solver::model::ports::dimensions::{AnyDimension, D_P};
    use crate::solver::model::ports::PortRegistry;

    let mut registry = PortRegistry::new(model.state_layout.clone());

    let d_p = registry
        .register_scalar_field::<D_P>(dp_field)
        .map_err(|e| format!("rhie_chow/correct_velocity_delta: {e}"))?;

    let momentum = coupling.momentum;

    // Register momentum field
    let u = registry
        .register_vector2_field::<AnyDimension>(momentum.name())
        .map_err(|e| format!("rhie_chow/correct_velocity_delta: {e}"))?;

    // Register gradient fields using pre-computed derived names
    let grad_p = registry
        .register_vector2_field::<PressureGradient>(grad_p_name)
        .map_err(|e| {
            format!(
                "rhie_chow/correct_velocity_delta: missing gradient field '{}': {e}",
                grad_p_name
            )
        })?;

    let grad_old = registry
        .register_vector2_field::<PressureGradient>(grad_p_old_name)
        .map_err(|e| {
            format!(
                "rhie_chow/correct_velocity_delta: missing old gradient field '{}': {e}",
                grad_p_old_name
            )
        })?;

    let stride = registry.state_layout().stride();
    let d_p_offset = d_p.offset();
    let u_x = u
        .component(XY::X.to_usize() as u32)
        .map(|c| c.full_offset())
        .ok_or("u component 0")?;
    let u_y = u
        .component(XY::Y.to_usize() as u32)
        .map(|c| c.full_offset())
        .ok_or("u component 1")?;
    let grad_p_x = grad_p
        .component(XY::X.to_usize() as u32)
        .map(|c| c.full_offset())
        .ok_or("grad_p component 0")?;
    let grad_p_y = grad_p
        .component(XY::Y.to_usize() as u32)
        .map(|c| c.full_offset())
        .ok_or("grad_p component 1")?;
    let grad_old_x = grad_old
        .component(XY::X.to_usize() as u32)
        .map(|c| c.full_offset())
        .ok_or("grad_old component 0")?;
    let grad_old_y = grad_old
        .component(XY::Y.to_usize() as u32)
        .map(|c| c.full_offset())
        .ok_or("grad_old component 1")?;

    let mut program = KernelProgram::new(
        "rhie_chow/correct_velocity_delta",
        DispatchDomain::Cells,
        rhie_chow_state_launch(stride),
        rhie_chow_state_bindings(),
    );
    let indexing_stmts = vec![dsl::let_expr("base", Expr::ident("idx") * stride)];
    let preamble_stmts = vec![
        dsl::let_expr(
            "d_p",
            dsl::array_access("state", Expr::ident("base") + d_p_offset),
        ),
        dsl::let_expr(
            "grad_px",
            dsl::array_access("state", Expr::ident("base") + grad_p_x),
        ),
        dsl::let_expr(
            "grad_py",
            dsl::array_access("state", Expr::ident("base") + grad_p_y),
        ),
        dsl::let_expr(
            "grad_old_x",
            dsl::array_access("state", Expr::ident("base") + grad_old_x),
        ),
        dsl::let_expr(
            "grad_old_y",
            dsl::array_access("state", Expr::ident("base") + grad_old_y),
        ),
        dsl::let_expr(
            "corr_x",
            Expr::ident("d_p") * (Expr::ident("grad_px") - Expr::ident("grad_old_x")),
        ),
        dsl::let_expr(
            "corr_y",
            Expr::ident("d_p") * (Expr::ident("grad_py") - Expr::ident("grad_old_y")),
        ),
    ];
    let body_stmts = vec![
        dsl::assign_expr(
            dsl::array_access("state", Expr::ident("base") + u_x),
            dsl::array_access("state", Expr::ident("base") + u_x) - Expr::ident("corr_x"),
        ),
        dsl::assign_expr(
            dsl::array_access("state", Expr::ident("base") + u_y),
            dsl::array_access("state", Expr::ident("base") + u_y) - Expr::ident("corr_y"),
        ),
    ];
    program.indexing = indexing_stmts;
    program.preamble = preamble_stmts;
    program.body = body_stmts;
    program.side_effects.read_set.extend([
        EffectResource::component(0, 0, format!("state:{d_p_offset}")),
        EffectResource::component(0, 0, format!("state:{grad_p_x}")),
        EffectResource::component(0, 0, format!("state:{grad_p_y}")),
        EffectResource::component(0, 0, format!("state:{grad_old_x}")),
        EffectResource::component(0, 0, format!("state:{grad_old_y}")),
        EffectResource::component(0, 0, format!("state:{u_x}")),
        EffectResource::component(0, 0, format!("state:{u_y}")),
    ]);
    program.side_effects.write_set.extend([
        EffectResource::component(0, 0, format!("state:{u_x}")),
        EffectResource::component(0, 0, format!("state:{u_y}")),
    ]);

    Ok(program)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::dimensions::{
        Density, DynamicViscosity, MassFlux, Pressure, PressureGradient, UnitDimension, Velocity,
        D_P,
    };
    use crate::solver::model::backend::ast::{
        fvm, surface_scalar, vol_scalar, vol_vector, Coefficient, EquationSystem,
    };
    use crate::solver::model::backend::state_layout::StateLayout;
    use crate::solver::model::{BoundarySpec, PrimitiveDerivations};

    #[test]
    fn contract_rhie_chow_aux_kernel_generators_honor_dp_field_name() {
        let u = vol_vector("U", Velocity::UNIT);
        let p = vol_scalar("p", Pressure::UNIT);
        let phi = surface_scalar("phi", MassFlux::UNIT);
        let mu = vol_scalar("mu", DynamicViscosity::UNIT);
        let rho = vol_scalar("rho", Density::UNIT);
        let dp_custom = vol_scalar("dp_custom", D_P::UNIT);
        let grad_p = vol_vector("grad_p", PressureGradient::UNIT);
        let grad_p_old = vol_vector("grad_p_old", PressureGradient::UNIT);

        let momentum = (fvm::ddt_coeff(Coefficient::field(rho).expect("rho must be scalar"), u)
            + fvm::div(phi, u)
            + fvm::laplacian(Coefficient::field(mu).expect("mu must be scalar"), u)
            + fvm::grad(p))
        .eqn(u);

        let pressure = (fvm::laplacian(
            Coefficient::product(
                Coefficient::field(rho).expect("rho must be scalar"),
                Coefficient::field(dp_custom).expect("dp_custom must be scalar"),
            )
            .expect("pressure coefficient must be scalar"),
            p,
        ) + fvm::div_flux(phi, p))
        .eqn(p);

        let mut system = EquationSystem::new();
        system.add_equation(momentum);
        system.add_equation(pressure);

        let layout = StateLayout::new(vec![u, p, dp_custom, grad_p, grad_p_old]);

        let module =
            rhie_chow_aux_module(&system, "dp_custom", true, true, DpFormulation::ClosedForm).expect("module creation failed");

        let model = crate::solver::model::ModelSpec {
            id: "rhie_chow_dp_custom_test",
            system,
            state_layout: layout,
            boundaries: BoundarySpec::default(),
            modules: vec![module],
            linear_solver: None,
            primitives: PrimitiveDerivations::identity(),
        };

        let schemes = crate::solver::ir::SchemeRegistry::default();
        for kernel_id in [
            KernelId("dp_init"),
            KernelId("dp_update_from_diag"),
            KernelId("rhie_chow/store_grad_p"),
            KernelId("rhie_chow/grad_p_update"),
            KernelId("rhie_chow/correct_velocity_delta"),
        ] {
            crate::solver::model::kernel::generate_kernel_wgsl_for_model_by_id(
                &model, &schemes, kernel_id,
            )
            .unwrap_or_else(|e| {
                panic!(
                    "generator missing or failed for {}: {e}",
                    kernel_id.as_str()
                )
            });
        }
    }

    #[test]
    fn contract_rhie_chow_fused_kernel_is_synthesized_from_dsl_inputs() {
        let model = crate::solver::model::incompressible_momentum_model().expect("model");
        let rhie_chow_module = model
            .modules
            .iter()
            .find(|module| module.name == "rhie_chow_aux")
            .expect("incompressible model missing rhie_chow_aux module");

        assert!(
            !rhie_chow_module
                .generators
                .iter()
                .any(|gen| gen.id.as_str() == "rhie_chow/dp_update_store_grad_p_fused"),
            "fused replacement should not have a handwritten generator"
        );

        let schemes = crate::solver::ir::SchemeRegistry::default();
        let mut out_dir = std::env::temp_dir();
        out_dir.push(format!(
            "cfd2_rhie_chow_fused_synth_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock before unix epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&out_dir).expect("create temp output dir");

        let emitted = crate::solver::model::kernel::emit_model_kernels_wgsl_with_ids(
            &out_dir, &model, &schemes,
        )
        .expect("emit model kernels");
        let fused_path = emitted
            .iter()
            .find_map(|(id, path)| {
                (id.as_str() == "rhie_chow/dp_update_store_grad_p_fused").then_some(path)
            })
            .expect("synthesized fused rhie-chow kernel path");
        let fused_src = std::fs::read_to_string(fused_path).expect("read synthesized fused kernel");
        assert!(
            fused_src.contains("synthesized by fusion rule: rhie_chow:dp_update_store_grad_p_v1"),
            "expected fusion synthesis marker in fused Rhie-Chow WGSL"
        );

        let update_dp_init_fused_path = emitted
            .iter()
            .find_map(|(id, path)| {
                (id.as_str() == "generic_coupled/update_dp_init_fused").then_some(path)
            })
            .expect("synthesized generic_coupled/update_dp_init fused kernel path");
        let update_dp_init_fused_src = std::fs::read_to_string(update_dp_init_fused_path)
            .expect("read synthesized generic_coupled/update_dp_init fused kernel");
        assert!(
            update_dp_init_fused_src
                .contains("synthesized by fusion rule: generic_coupled:update_dp_init_v1"),
            "expected synthesis marker for generic_coupled:update_dp_init_v1"
        );

        let aggressive_fused_path = emitted
            .iter()
            .find_map(|(id, path)| {
                (id.as_str() == "rhie_chow/dp_update_store_grad_p_grad_p_update_fused")
                    .then_some(path)
            })
            .expect("aggressive synthesized fused rhie-chow kernel path");
        let aggressive_fused_src =
            std::fs::read_to_string(aggressive_fused_path).expect("read aggressive fused kernel");
        assert!(
            aggressive_fused_src.contains(
                "synthesized by fusion rule: rhie_chow:dp_update_store_grad_p_grad_p_update_v1"
            ),
            "expected aggressive fusion synthesis marker in fused Rhie-Chow WGSL"
        );

        let aggressive_full_fused_path = emitted
            .iter()
            .find_map(|(id, path)| {
                (id.as_str()
                    == "rhie_chow/dp_update_store_grad_p_grad_p_update_correct_velocity_delta_fused")
                    .then_some(path)
            })
            .expect("aggressive full synthesized fused rhie-chow kernel path");
        let aggressive_full_fused_src = std::fs::read_to_string(aggressive_full_fused_path)
            .expect("read aggressive full fused kernel");
        assert!(
            aggressive_full_fused_src.contains(
                "synthesized by fusion rule: rhie_chow:dp_update_store_grad_p_grad_p_update_correct_velocity_delta_v1"
            ),
            "expected aggressive full fusion synthesis marker in fused Rhie-Chow WGSL"
        );

        let aggressive_with_dp_init_fused_path = emitted
            .iter()
            .find_map(|(id, path)| {
                (id.as_str()
                    == "rhie_chow/dp_init_dp_update_store_grad_p_grad_p_update_correct_velocity_delta_fused")
                    .then_some(path)
            })
            .expect("aggressive full synthesized fused rhie-chow-with-dp_init kernel path");
        let aggressive_with_dp_init_fused_src =
            std::fs::read_to_string(aggressive_with_dp_init_fused_path)
                .expect("read aggressive dp_init+full fused kernel");
        assert!(
            aggressive_with_dp_init_fused_src.contains(
                "synthesized by fusion rule: rhie_chow:dp_init_dp_update_store_grad_p_grad_p_update_correct_velocity_delta_v1"
            ),
            "expected aggressive dp_init+full fusion synthesis marker in fused Rhie-Chow WGSL"
        );

        // Verify standalone grad_p_update + correct_velocity_delta fused kernel
        let standalone_fused_path = emitted
            .iter()
            .find_map(|(id, path)| {
                (id.as_str() == "rhie_chow/grad_p_update_correct_velocity_delta_fused")
                    .then_some(path)
            })
            .expect("standalone grad_p_update_correct_velocity_delta fused kernel path");
        let standalone_fused_src = std::fs::read_to_string(standalone_fused_path)
            .expect("read standalone grad_p_update_correct_velocity_delta fused kernel");
        assert!(
            standalone_fused_src.contains(
                "synthesized by fusion rule: rhie_chow:grad_p_update_correct_velocity_delta_v1"
            ),
            "expected standalone grad_p_update_correct_velocity_delta fusion synthesis marker in fused Rhie-Chow WGSL"
        );

        // Verify standalone store_grad_p + grad_p_update fused kernel
        let store_grad_p_grad_p_update_fused_path = emitted
            .iter()
            .find_map(|(id, path)| {
                (id.as_str() == "rhie_chow/store_grad_p_grad_p_update_fused").then_some(path)
            })
            .expect("standalone store_grad_p_grad_p_update fused kernel path");
        let store_grad_p_grad_p_update_fused_src =
            std::fs::read_to_string(store_grad_p_grad_p_update_fused_path)
                .expect("read standalone store_grad_p_grad_p_update fused kernel");
        assert!(
            store_grad_p_grad_p_update_fused_src.contains(
                "synthesized by fusion rule: rhie_chow:store_grad_p_grad_p_update_v1"
            ),
            "expected standalone store_grad_p_grad_p_update fusion synthesis marker in fused Rhie-Chow WGSL"
        );

        let _ = std::fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn contract_rhie_chow_dp_init_and_correct_velocity_delta_are_dsl_artifacts() {
        let model = crate::solver::model::incompressible_momentum_model().expect("model");
        let rhie_chow_module = model
            .modules
            .iter()
            .find(|module| module.name == "rhie_chow_aux")
            .expect("incompressible model missing rhie_chow_aux module");
        let schemes = crate::solver::ir::SchemeRegistry::default();

        for kernel_id in ["dp_init", "rhie_chow/correct_velocity_delta"] {
            let generator = rhie_chow_module
                .generators
                .iter()
                .find(|gen| gen.id.as_str() == kernel_id)
                .unwrap_or_else(|| panic!("missing {kernel_id} generator"));
            let artifact = (generator.generator.as_ref())(&model, &schemes)
                .unwrap_or_else(|e| panic!("generate {kernel_id} artifact failed: {e}"));

            assert!(
                matches!(
                    artifact,
                    crate::solver::model::kernel::ModelKernelArtifact::DslProgram(_)
                ),
                "{kernel_id} should be emitted as a DSL artifact"
            );
        }
    }

    #[test]
    fn contract_rhie_chow_local_symbols_are_derived_from_dsl_statements() {
        let model = crate::solver::model::incompressible_momentum_model().expect("model");
        let rhie_chow_module = model
            .modules
            .iter()
            .find(|module| module.name == "rhie_chow_aux")
            .expect("incompressible model missing rhie_chow_aux module");
        let schemes = crate::solver::ir::SchemeRegistry::default();

        for (kernel_id, expected_symbols) in [
            ("dp_update_from_diag", vec!["rho", "dt", "d_p"]),
            ("dp_init", vec![]),
            ("rhie_chow/store_grad_p", vec![]),
            (
                "rhie_chow/grad_p_update",
                vec!["k", "face_idx", "lambda", "grad_out_p"],
            ),
            (
                "rhie_chow/correct_velocity_delta",
                vec!["d_p", "corr_x", "corr_y"],
            ),
        ] {
            let generator = rhie_chow_module
                .generators
                .iter()
                .find(|gen| gen.id.as_str() == kernel_id)
                .unwrap_or_else(|| panic!("missing {kernel_id} generator"));
            let artifact = (generator.generator.as_ref())(&model, &schemes)
                .unwrap_or_else(|e| panic!("generate {kernel_id} artifact failed: {e}"));
            let program = match artifact {
                crate::solver::model::kernel::ModelKernelArtifact::DslProgram(program) => program,
                crate::solver::model::kernel::ModelKernelArtifact::Wgsl(_) => {
                    panic!("{kernel_id} must be a DSL artifact")
                }
            };

            assert!(
                !program.local_symbols().iter().any(|s| s == "base"),
                "{kernel_id}: indexing aliases should not be renamed in fusion"
            );
            for expected in expected_symbols {
                assert!(
                    program.local_symbols().iter().any(|s| s == expected),
                    "{kernel_id}: missing expected local symbol '{expected}'"
                );
            }
        }
    }

    #[test]
    fn missing_grad_p_old_returns_clear_error() {
        // Regression test: when grad_p_old is missing, the generator should
        // return a clear error containing the missing field name.
        let u = vol_vector("U", Velocity::UNIT);
        let p = vol_scalar("p", Pressure::UNIT);
        let phi = surface_scalar("phi", MassFlux::UNIT);
        let mu = vol_scalar("mu", DynamicViscosity::UNIT);
        let rho = vol_scalar("rho", Density::UNIT);
        let dp = vol_scalar("dp", D_P::UNIT);
        let grad_p = vol_vector("grad_p", PressureGradient::UNIT);
        // Note: grad_p_old is intentionally missing

        let momentum = (fvm::ddt_coeff(Coefficient::field(rho).expect("rho must be scalar"), u)
            + fvm::div(phi, u)
            + fvm::laplacian(Coefficient::field(mu).expect("mu must be scalar"), u)
            + fvm::grad(p))
        .eqn(u);

        let pressure = (fvm::laplacian(
            Coefficient::product(
                Coefficient::field(rho).expect("rho must be scalar"),
                Coefficient::field(dp).expect("dp must be scalar"),
            )
            .expect("pressure coefficient must be scalar"),
            p,
        ) + fvm::div_flux(phi, p))
        .eqn(p);

        let mut system = EquationSystem::new();
        system.add_equation(momentum);
        system.add_equation(pressure);

        let layout = StateLayout::new(vec![u, p, dp, grad_p]);

        // Module creation should succeed (no StateLayout validation yet)
        let module =
            rhie_chow_aux_module(&system, "dp", true, true, DpFormulation::ClosedForm).expect("module creation failed");

        let model = crate::solver::model::ModelSpec {
            id: "rhie_chow_missing_grad_p_old_test",
            system,
            state_layout: layout,
            boundaries: BoundarySpec::default(),
            modules: vec![module],
            linear_solver: None,
            primitives: PrimitiveDerivations::identity(),
        };

        let schemes = crate::solver::ir::SchemeRegistry::default();

        // Test store_grad_p kernel - should fail with clear error about missing grad_p_old
        let result = crate::solver::model::kernel::generate_kernel_wgsl_for_model_by_id(
            &model,
            &schemes,
            KernelId("rhie_chow/store_grad_p"),
        );
        let err = result.expect_err("should fail when grad_p_old is missing");
        assert!(
            err.contains("grad_p_old"),
            "error should contain the missing field name 'grad_p_old': {err}"
        );

        // Test correct_velocity_delta kernel - should also fail with clear error
        let result2 = crate::solver::model::kernel::generate_kernel_wgsl_for_model_by_id(
            &model,
            &schemes,
            KernelId("rhie_chow/correct_velocity_delta"),
        );
        let err2 = result2.expect_err("should fail when grad_p_old is missing");
        assert!(
            err2.contains("grad_p_old"),
            "error should contain the missing field name 'grad_p_old': {err2}"
        );
    }

    #[test]
    fn rhie_chow_module_has_port_manifest() {
        // Verify that rhie_chow_aux_module produces a PortManifest with expected fields.
        let u = vol_vector("U", Velocity::UNIT);
        let p = vol_scalar("p", Pressure::UNIT);
        let phi = surface_scalar("phi", MassFlux::UNIT);
        let mu = vol_scalar("mu", DynamicViscosity::UNIT);
        let rho = vol_scalar("rho", Density::UNIT);
        let dp = vol_scalar("dp", D_P::UNIT);
        let grad_p = vol_vector("grad_p", PressureGradient::UNIT);
        let grad_p_old = vol_vector("grad_p_old", PressureGradient::UNIT);

        let momentum = (fvm::ddt_coeff(Coefficient::field(rho).expect("rho must be scalar"), u)
            + fvm::div(phi, u)
            + fvm::laplacian(Coefficient::field(mu).expect("mu must be scalar"), u)
            + fvm::grad(p))
        .eqn(u);

        let pressure = (fvm::laplacian(
            Coefficient::product(
                Coefficient::field(rho).expect("rho must be scalar"),
                Coefficient::field(dp).expect("dp must be scalar"),
            )
            .expect("pressure coefficient must be scalar"),
            p,
        ) + fvm::div_flux(phi, p))
        .eqn(p);

        let mut system = EquationSystem::new();
        system.add_equation(momentum);
        system.add_equation(pressure);

        let _layout = StateLayout::new(vec![u, p, dp, grad_p, grad_p_old]);

        let module = rhie_chow_aux_module(&system, "dp", true, true, DpFormulation::ClosedForm).expect("module creation");

        // Check that port_manifest is present
        let port_manifest = module
            .port_manifest
            .expect("port_manifest should be present");

        // Should have 4 fields: dp, grad_p, grad_p_old, momentum
        assert_eq!(port_manifest.fields.len(), 4, "expected 4 field specs");

        // Verify field specs
        let dp_field = port_manifest
            .fields
            .iter()
            .find(|f| f.name == "dp")
            .expect("dp field spec");
        assert_eq!(
            dp_field.kind,
            crate::solver::ir::ports::PortFieldKind::Scalar
        );
        assert_eq!(dp_field.unit, D_P::UNIT);

        let grad_p_field = port_manifest
            .fields
            .iter()
            .find(|f| f.name == "grad_p")
            .expect("grad_p field spec");
        assert_eq!(
            grad_p_field.kind,
            crate::solver::ir::ports::PortFieldKind::Vector2
        );
        assert_eq!(grad_p_field.unit, PressureGradient::UNIT);

        let grad_p_old_field = port_manifest
            .fields
            .iter()
            .find(|f| f.name == "grad_p_old")
            .expect("grad_p_old field spec");
        assert_eq!(
            grad_p_old_field.kind,
            crate::solver::ir::ports::PortFieldKind::Vector2
        );
        assert_eq!(grad_p_old_field.unit, PressureGradient::UNIT);

        let momentum_field = port_manifest
            .fields
            .iter()
            .find(|f| f.name == "U")
            .expect("momentum field spec");
        assert_eq!(
            momentum_field.kind,
            crate::solver::ir::ports::PortFieldKind::Vector2
        );
        assert_eq!(
            momentum_field.unit,
            crate::solver::ir::ports::ANY_DIMENSION,
            "momentum should use ANY_DIMENSION sentinel"
        );
    }

    /// CI gate: validate that all aggressive fusion rules have accurate hazard
    /// whitelists. Catches both new hazards introduced by code changes AND stale
    /// whitelist entries from refactored kernels.
    #[test]
    fn aggressive_fusion_rules_have_accurate_hazard_whitelists() {
        use crate::solver::model::kernel::*;
        let model = crate::solver::model::incompressible_momentum_model().expect("model");
        let schemes = crate::solver::ir::SchemeRegistry::default();
        let rules = derive_kernel_fusion_rules_for_model(&model);

        for rule in &rules {
            // Only process rules that require Aggressive policy.
            let is_aggressive = rule.guards.iter().any(|g| matches!(
                g,
                FusionGuard::MinPolicy(crate::solver::model::kernel::KernelFusionPolicy::Aggressive)
            ));
            if !is_aggressive {
                continue;
            }

            // Generate DSL programs via module generators.
            let mut programs = Vec::new();
            for module in &model.modules {
                let module: &dyn crate::solver::model::module::ModelModule = module;
                for atom in &rule.pattern {
                    if let Some(gen) = module.kernel_generators().iter().find(|g| g.id == atom.id) {
                        if let Ok(ModelKernelArtifact::DslProgram(p)) =
                            (gen.generator.as_ref())(&model, &schemes)
                        {
                            programs.push(p);
                        }
                    }
                }
            }
            if programs.len() != rule.pattern.len() {
                continue;
            }

            let hazards = cfd2_codegen::solver::codegen::fusion::detect_hazards(&programs);

            // Verify that expected_hazards covers all detected hazards.
            for h in &hazards {
                assert!(
                    rule.expected_hazards.iter().any(|e| e.matches(h)),
                    "Rule '{}': unwhitelisted {} hazard at kernel '{}' — \
                     add an ExpectedHazard entry with a justification",
                    rule.name, h.kind, h.kernel_id
                );
            }
            // Verify no stale entries.
            for e in &rule.expected_hazards {
                assert!(
                    hazards.iter().any(|h| e.matches(h)),
                    "Rule '{}': stale expected_hazards entry: {} at '{}' — \
                     remove it or update the kernel side-effects",
                    rule.name, e.kind, e.kernel_id
                );
            }
        }
    }
}
