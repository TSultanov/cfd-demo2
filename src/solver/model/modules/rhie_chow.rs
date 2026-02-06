use crate::solver::model::kernel::{
    DispatchKindId, FusionGuard, KernelConditionId, KernelFusionStepping, KernelPatternAtom,
    KernelPhaseId, ModelKernelFusionRule, ModelKernelGeneratorSpec, ModelKernelSpec,
};
use crate::solver::model::module::{KernelBundleModule, ModuleInvariant};
use crate::solver::model::KernelId;

use cfd2_codegen::solver::codegen::{
    wgsl_ast::{AssignOp, Block, Expr, ForStep, Stmt, Type},
    wgsl_dsl as dsl,
};
use cfd2_ir::solver::ir::{
    BindingAccess, DispatchDomain, EffectResource, KernelBinding, KernelProgram, LaunchSemantics,
};

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
///
/// # Errors
///
/// Returns an error if the momentum-pressure coupling cannot be inferred from the system.
pub fn rhie_chow_aux_module(
    system: &crate::solver::model::backend::ast::EquationSystem,
    dp_field: &'static str,
    require_vector2_momentum: bool,
    require_pressure_gradient: bool,
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
    let kernel_dp_update_store_grad_p_grad_p_update_correct_velocity_delta_fused = KernelId(
        "rhie_chow/dp_update_store_grad_p_grad_p_update_correct_velocity_delta_fused",
    );
    let kernel_dp_init_dp_update_store_grad_p_grad_p_update_correct_velocity_delta_fused = KernelId(
        "rhie_chow/dp_init_dp_update_store_grad_p_grad_p_update_correct_velocity_delta_fused",
    );
    let kernel_store_grad_p = KernelId("rhie_chow/store_grad_p");
    let kernel_grad_p_update = KernelId("rhie_chow/grad_p_update");
    let kernel_rhie_chow_correct_velocity_delta = KernelId("rhie_chow/correct_velocity_delta");
    let kernel_grad_p_update_correct_velocity_delta_fused =
        KernelId("rhie_chow/grad_p_update_correct_velocity_delta_fused");
    let kernel_store_grad_p_grad_p_update_fused =
        KernelId("rhie_chow/store_grad_p_grad_p_update_fused");

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
            generate_dp_init_kernel_program(model, dp_field)
        }),
        ModelKernelGeneratorSpec::new_dsl(kernel_dp_update_from_diag, move |model, _schemes| {
            generate_dp_update_from_diag_kernel_program(model, dp_field, coupling_for_dp_update)
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
        },
        ModelKernelFusionRule {
            name: "rhie_chow:dp_init_dp_update_store_grad_p_grad_p_update_correct_velocity_delta_v1",
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
                id: kernel_dp_init_dp_update_store_grad_p_grad_p_update_correct_velocity_delta_fused,
                phase: KernelPhaseId::Update,
                dispatch: DispatchKindId::Cells,
                condition: KernelConditionId::Always,
            },
            guards: vec![
                FusionGuard::RequiresStepping(KernelFusionStepping::Coupled),
                FusionGuard::RequiresModule("rhie_chow_aux"),
                FusionGuard::MinPolicy(crate::solver::model::kernel::KernelFusionPolicy::Aggressive),
            ],
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
                FusionGuard::MinPolicy(crate::solver::model::kernel::KernelFusionPolicy::Aggressive),
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
                FusionGuard::MinPolicy(crate::solver::model::kernel::KernelFusionPolicy::Aggressive),
            ],
        },
        // Standalone fusion rule for grad_p_update + correct_velocity_delta (aggressive-only)
        ModelKernelFusionRule {
            name: "rhie_chow:grad_p_update_correct_velocity_delta_v1",
            priority: 105,
            phase: KernelPhaseId::Update,
            pattern: vec![
                KernelPatternAtom::with_dispatch(kernel_grad_p_update, DispatchKindId::Cells),
                KernelPatternAtom::with_dispatch(kernel_rhie_chow_correct_velocity_delta, DispatchKindId::Cells),
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
                FusionGuard::MinPolicy(crate::solver::model::kernel::KernelFusionPolicy::Aggressive),
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
                FusionGuard::MinPolicy(crate::solver::model::kernel::KernelFusionPolicy::Aggressive),
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
    let body_stmts = vec![dsl::assign_expr(
        dsl::array_access("state", Expr::ident("base") + d_p_offset),
        Expr::lit_f32(0.0),
    )];
    program.indexing = rhie_chow_section_lines(indexing_stmts);
    program.body = rhie_chow_section_lines(body_stmts.clone());
    program.local_symbols = rhie_chow_collect_local_symbols_sections(&[&body_stmts]);
    program
        .side_effects
        .read_set
        .insert(EffectResource::binding(0, 1));
    program.side_effects.write_set.insert(EffectResource::component(
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
        Some(format!(
            "idx >= (arrayLength(&state) / max({state_stride}u, 1u))"
        )),
    )
}

fn rhie_chow_state_bindings() -> Vec<KernelBinding> {
    vec![
        KernelBinding::new(0, 0, "state", "array<f32>", BindingAccess::ReadWriteStorage),
        KernelBinding::new(0, 1, "constants", "Constants", BindingAccess::Uniform),
    ]
}

fn rhie_chow_section_lines(stmts: Vec<Stmt>) -> Vec<String> {
    cfd2_codegen::solver::codegen::wgsl_ast::render_block_lines(&Block::new(stmts))
}

fn rhie_chow_collect_local_symbols_sections(sections: &[&[Stmt]]) -> Vec<String> {
    let mut merged = Vec::new();
    for section in sections {
        merged.extend((*section).iter().cloned());
    }
    cfd2_codegen::solver::codegen::wgsl_ast::collect_local_symbols(&merged)
}

fn rhie_chow_grad_p_update_bindings() -> Vec<KernelBinding> {
    vec![
        KernelBinding::new(0, 0, "state", "array<f32>", BindingAccess::ReadWriteStorage),
        KernelBinding::new(0, 1, "constants", "Constants", BindingAccess::Uniform),
        KernelBinding::new(1, 0, "face_owner", "array<u32>", BindingAccess::ReadOnlyStorage),
        KernelBinding::new(1, 1, "face_neighbor", "array<i32>", BindingAccess::ReadOnlyStorage),
        KernelBinding::new(1, 2, "face_areas", "array<f32>", BindingAccess::ReadOnlyStorage),
        KernelBinding::new(1, 3, "face_normals", "array<Vector2>", BindingAccess::ReadOnlyStorage),
        KernelBinding::new(
            1,
            13,
            "face_centers",
            "array<Vector2>",
            BindingAccess::ReadOnlyStorage,
        ),
        KernelBinding::new(1, 4, "cell_centers", "array<Vector2>", BindingAccess::ReadOnlyStorage),
        KernelBinding::new(1, 5, "cell_vols", "array<f32>", BindingAccess::ReadOnlyStorage),
        KernelBinding::new(
            1,
            6,
            "cell_face_offsets",
            "array<u32>",
            BindingAccess::ReadOnlyStorage,
        ),
        KernelBinding::new(1, 7, "cell_faces", "array<u32>", BindingAccess::ReadOnlyStorage),
        KernelBinding::new(
            1,
            12,
            "face_boundary",
            "array<u32>",
            BindingAccess::ReadOnlyStorage,
        ),
        KernelBinding::new(2, 0, "bc_kind", "array<u32>", BindingAccess::ReadOnlyStorage),
        KernelBinding::new(2, 1, "bc_value", "array<f32>", BindingAccess::ReadOnlyStorage),
    ]
}

fn generate_dp_update_from_diag_kernel_program(
    model: &crate::solver::model::ModelSpec,
    dp_field: &str,
    coupling: crate::solver::model::invariants::MomentumPressureCoupling,
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

    let mut program = KernelProgram::new(
        "dp_update_from_diag",
        DispatchDomain::Cells,
        rhie_chow_state_launch(stride),
        rhie_chow_state_bindings(),
    );
    let indexing_stmts = vec![dsl::let_expr(
        "base",
        Expr::ident("idx") * stride,
    )];
    let preamble_stmts = vec![
        dsl::let_expr(
            "rho",
            dsl::max(Expr::ident("constants").field("density"), Expr::lit_f32(1e-12)),
        ),
        dsl::let_expr(
            "dt",
            dsl::max(Expr::ident("constants").field("dt"), Expr::lit_f32(0.0)),
        ),
        dsl::let_expr(
            "d_p",
            Expr::ident("constants").field("alpha_u") * Expr::ident("dt") / Expr::ident("rho"),
        ),
    ];
    let body_stmts = vec![dsl::assign_expr(
        dsl::array_access("state", Expr::ident("base") + d_p_offset),
        Expr::ident("d_p"),
    )];
    program.indexing = rhie_chow_section_lines(indexing_stmts);
    program.preamble = rhie_chow_section_lines(preamble_stmts.clone());
    program.body = rhie_chow_section_lines(body_stmts.clone());
    program.local_symbols =
        rhie_chow_collect_local_symbols_sections(&[&preamble_stmts, &body_stmts]);
    program.side_effects.read_set.insert(EffectResource::binding(0, 1));
    program.side_effects.write_set.insert(EffectResource::component(
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
        .component(0)
        .map(|c| c.full_offset())
        .ok_or("rhie_chow/grad_p_update: grad_p component 0")?;
    let grad_p_y = grad_p
        .component(1)
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
    let bc_idx_expr = Expr::ident("face_idx") * unknowns_per_face + p_unknown_offset;
    let p_state_expr = dsl::array_access("state", Expr::ident("base") + p_offset);
    let p_other_state_expr =
        dsl::array_access("state", Expr::ident("other_idx") * state_stride + p_offset);
    let bc_kind_expr = dsl::array_access("bc_kind", bc_idx_expr);
    let bc_value_expr = dsl::array_access("bc_value", bc_idx_expr);
    let p_boundary_expr = dsl::select(
        dsl::select(p_state_expr, bc_value_expr, bc_kind_expr.eq(1u32)),
        p_state_expr + bc_value_expr * Expr::ident("d_own"),
        bc_kind_expr.eq(2u32),
    );
    let p_interp_expr = p_state_expr * Expr::ident("lambda")
        + dsl::select(p_other_state_expr, p_boundary_expr, Expr::ident("is_boundary"))
            * Expr::ident("lambda_other");

    let mut program = KernelProgram::new(
        "rhie_chow/grad_p_update",
        DispatchDomain::Cells,
        rhie_chow_state_launch(state_stride),
        rhie_chow_grad_p_update_bindings(),
    );
    let indexing_stmts = vec![dsl::let_expr(
        "base",
        Expr::ident("idx") * state_stride,
    )];
    let preamble_stmts = vec![
        dsl::let_expr("cell_center", dsl::array_access("cell_centers", Expr::ident("idx"))),
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
                dsl::let_expr("owner", dsl::array_access("face_owner", Expr::ident("face_idx"))),
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
                dsl::let_expr("area", dsl::array_access("face_areas", Expr::ident("face_idx"))),
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
    program.indexing = rhie_chow_section_lines(indexing_stmts);
    program.preamble = rhie_chow_section_lines(preamble_stmts.clone());
    program.body = rhie_chow_section_lines(body_stmts.clone());
    program.local_symbols =
        rhie_chow_collect_local_symbols_sections(&[&preamble_stmts, &body_stmts]);
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
    program.side_effects.read_set.insert(EffectResource::component(
        0,
        0,
        format!("state:{p_offset}"),
    ));
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
        .component(0)
        .map(|c| c.full_offset())
        .ok_or("grad_p component 0")?;
    let grad_p_y = grad_p
        .component(1)
        .map(|c| c.full_offset())
        .ok_or("grad_p component 1")?;
    let grad_old_x = grad_old
        .component(0)
        .map(|c| c.full_offset())
        .ok_or("grad_old component 0")?;
    let grad_old_y = grad_old
        .component(1)
        .map(|c| c.full_offset())
        .ok_or("grad_old component 1")?;

    let mut program = KernelProgram::new(
        "rhie_chow/store_grad_p",
        DispatchDomain::Cells,
        rhie_chow_state_launch(stride),
        rhie_chow_state_bindings(),
    );
    program.indexing = rhie_chow_section_lines(vec![dsl::let_expr(
        "base",
        Expr::ident("idx") * stride,
    )]);
    program.body = rhie_chow_section_lines(vec![
        dsl::assign_expr(
            dsl::array_access("state", Expr::ident("base") + grad_old_x),
            dsl::array_access("state", Expr::ident("base") + grad_p_x),
        ),
        dsl::assign_expr(
            dsl::array_access("state", Expr::ident("base") + grad_old_y),
            dsl::array_access("state", Expr::ident("base") + grad_p_y),
        ),
    ]);
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
        .component(0)
        .map(|c| c.full_offset())
        .ok_or("u component 0")?;
    let u_y = u
        .component(1)
        .map(|c| c.full_offset())
        .ok_or("u component 1")?;
    let grad_p_x = grad_p
        .component(0)
        .map(|c| c.full_offset())
        .ok_or("grad_p component 0")?;
    let grad_p_y = grad_p
        .component(1)
        .map(|c| c.full_offset())
        .ok_or("grad_p component 1")?;
    let grad_old_x = grad_old
        .component(0)
        .map(|c| c.full_offset())
        .ok_or("grad_old component 0")?;
    let grad_old_y = grad_old
        .component(1)
        .map(|c| c.full_offset())
        .ok_or("grad_old component 1")?;

    let mut program = KernelProgram::new(
        "rhie_chow/correct_velocity_delta",
        DispatchDomain::Cells,
        rhie_chow_state_launch(stride),
        rhie_chow_state_bindings(),
    );
    let indexing_stmts = vec![dsl::let_expr(
        "base",
        Expr::ident("idx") * stride,
    )];
    let preamble_stmts = vec![
        dsl::let_expr("d_p", dsl::array_access("state", Expr::ident("base") + d_p_offset)),
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
    program.indexing = rhie_chow_section_lines(indexing_stmts);
    program.preamble = rhie_chow_section_lines(preamble_stmts.clone());
    program.body = rhie_chow_section_lines(body_stmts.clone());
    program.local_symbols =
        rhie_chow_collect_local_symbols_sections(&[&preamble_stmts, &body_stmts]);
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
            rhie_chow_aux_module(&system, "dp_custom", true, true).expect("module creation failed");

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
        let model = crate::solver::model::incompressible_momentum_model();
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
            aggressive_fused_src
                .contains("synthesized by fusion rule: rhie_chow:dp_update_store_grad_p_grad_p_update_v1"),
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
                (id.as_str() == "rhie_chow/store_grad_p_grad_p_update_fused")
                    .then_some(path)
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
        let model = crate::solver::model::incompressible_momentum_model();
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
        let model = crate::solver::model::incompressible_momentum_model();
        let rhie_chow_module = model
            .modules
            .iter()
            .find(|module| module.name == "rhie_chow_aux")
            .expect("incompressible model missing rhie_chow_aux module");
        let schemes = crate::solver::ir::SchemeRegistry::default();

        for (kernel_id, expected_symbols) in [
            (
                "dp_update_from_diag",
                vec!["rho", "dt", "d_p"],
            ),
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
                !program.local_symbols.iter().any(|s| s == "base"),
                "{kernel_id}: indexing aliases should not be renamed in fusion"
            );
            for expected in expected_symbols {
                assert!(
                    program.local_symbols.iter().any(|s| s == expected),
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
            rhie_chow_aux_module(&system, "dp", true, true).expect("module creation failed");

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

        let module = rhie_chow_aux_module(&system, "dp", true, true).expect("module creation");

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
}
