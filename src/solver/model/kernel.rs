#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelKind {
    PrepareCoupled,
    CoupledAssembly,
    PressureAssembly,
    UpdateFieldsFromCoupled,
    FluxRhieChow,
    SystemMain,

    // Legacy compressible kernels (EI family) with model-agnostic naming.
    ConservativeGradients,
    ConservativeFluxKt,
    ConservativeExplicitUpdate,
    ConservativeAssembly,
    ConservativeApply,
    ConservativeUpdate,

    GenericCoupledAssembly,
    GenericCoupledApply,
    GenericCoupledUpdate,
}

/// Stable identifier for a compute kernel.
///
/// This is used by the unified solver orchestration to decouple scheduling and lookup
/// from handwritten enums/matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelId(pub &'static str);

impl KernelId {
    pub const PREPARE_COUPLED: KernelId = KernelId("prepare_coupled");
    pub const COUPLED_ASSEMBLY: KernelId = KernelId("coupled_assembly");
    pub const PRESSURE_ASSEMBLY: KernelId = KernelId("pressure_assembly");
    pub const UPDATE_FIELDS_FROM_COUPLED: KernelId = KernelId("update_fields_from_coupled");
    pub const FLUX_RHIE_CHOW: KernelId = KernelId("flux_rhie_chow");

    pub const SYSTEM_MAIN: KernelId = KernelId("system_main");

    // Transitional stable ids for legacy EI kernels.
    //
    // We intentionally keep the underlying string ids aligned with the existing generated
    // shader registry entries ("ei_*") while removing EI naming from selection logic.
    pub const CONSERVATIVE_GRADIENTS: KernelId = KernelId("ei_gradients");
    pub const CONSERVATIVE_FLUX_KT: KernelId = KernelId("ei_flux_kt");
    pub const CONSERVATIVE_EXPLICIT_UPDATE: KernelId = KernelId("ei_explicit_update");
    pub const CONSERVATIVE_ASSEMBLY: KernelId = KernelId("ei_assembly");
    pub const CONSERVATIVE_APPLY: KernelId = KernelId("ei_apply");
    pub const CONSERVATIVE_UPDATE: KernelId = KernelId("ei_update");

    pub const GENERIC_COUPLED_ASSEMBLY: KernelId = KernelId("generic_coupled_assembly");
    pub const GENERIC_COUPLED_APPLY: KernelId = KernelId("generic_coupled_apply");
    pub const GENERIC_COUPLED_UPDATE: KernelId = KernelId("generic_coupled_update");

    // Handwritten solver-infrastructure kernels (single-entrypoint compute shaders).
    pub const DOT_PRODUCT: KernelId = KernelId("dot_product");
    pub const DOT_PRODUCT_PAIR: KernelId = KernelId("dot_product_pair");

    pub const SCALARS_INIT: KernelId = KernelId("scalars/init_scalars");
    pub const SCALARS_INIT_CG: KernelId = KernelId("scalars/init_cg_scalars");
    pub const SCALARS_REDUCE_RHO_NEW_R_R: KernelId = KernelId("scalars/reduce_rho_new_r_r");
    pub const SCALARS_REDUCE_R0_V: KernelId = KernelId("scalars/reduce_r0_v");
    pub const SCALARS_REDUCE_T_S_T_T: KernelId = KernelId("scalars/reduce_t_s_t_t");
    pub const SCALARS_UPDATE_CG_ALPHA: KernelId = KernelId("scalars/update_cg_alpha");
    pub const SCALARS_UPDATE_CG_BETA: KernelId = KernelId("scalars/update_cg_beta");
    pub const SCALARS_UPDATE_RHO_OLD: KernelId = KernelId("scalars/update_rho_old");

    pub const LINEAR_SOLVER_SPMV_P_V: KernelId = KernelId("linear_solver/spmv_p_v");
    pub const LINEAR_SOLVER_SPMV_S_T: KernelId = KernelId("linear_solver/spmv_s_t");
    pub const LINEAR_SOLVER_BICGSTAB_UPDATE_X_R: KernelId =
        KernelId("linear_solver/bicgstab_update_x_r");
    pub const LINEAR_SOLVER_BICGSTAB_UPDATE_P: KernelId =
        KernelId("linear_solver/bicgstab_update_p");
    pub const LINEAR_SOLVER_BICGSTAB_UPDATE_S: KernelId =
        KernelId("linear_solver/bicgstab_update_s");
    pub const LINEAR_SOLVER_CG_UPDATE_X_R: KernelId = KernelId("linear_solver/cg_update_x_r");
    pub const LINEAR_SOLVER_CG_UPDATE_P: KernelId = KernelId("linear_solver/cg_update_p");

    // Handwritten solver-infrastructure kernels (multi-entrypoint compute shaders).
    pub const AMG_SMOOTH_OP: KernelId = KernelId("amg/smooth_op");
    pub const AMG_RESTRICT_RESIDUAL: KernelId = KernelId("amg/restrict_residual");
    pub const AMG_PROLONGATE_OP: KernelId = KernelId("amg/prolongate_op");
    pub const AMG_CLEAR: KernelId = KernelId("amg/clear");

    pub const AMG_PACK_PACK_COMPONENT: KernelId = KernelId("amg_pack/pack_component");
    pub const AMG_PACK_UNPACK_COMPONENT: KernelId = KernelId("amg_pack/unpack_component");

    pub const BLOCK_PRECOND_BUILD_BLOCK_INV: KernelId = KernelId("block_precond/build_block_inv");
    pub const BLOCK_PRECOND_APPLY_BLOCK_PRECOND: KernelId =
        KernelId("block_precond/apply_block_precond");

    pub const SCHUR_PRECOND_PREDICT_AND_FORM: KernelId =
        KernelId("schur_precond/predict_and_form_schur");
    pub const SCHUR_PRECOND_RELAX_PRESSURE: KernelId = KernelId("schur_precond/relax_pressure");
    pub const SCHUR_PRECOND_CORRECT_VELOCITY: KernelId =
        KernelId("schur_precond/correct_velocity");

    pub const PRECONDITIONER_BUILD_SCHUR_RHS: KernelId =
        KernelId("preconditioner/build_schur_rhs");
    pub const PRECONDITIONER_FINALIZE_PRECOND: KernelId =
        KernelId("preconditioner/finalize_precond");
    pub const PRECONDITIONER_SPMV_PHAT_V: KernelId = KernelId("preconditioner/spmv_phat_v");
    pub const PRECONDITIONER_SPMV_SHAT_T: KernelId = KernelId("preconditioner/spmv_shat_t");

    pub const GMRES_OPS_SPMV: KernelId = KernelId("gmres_ops/spmv");
    pub const GMRES_OPS_AXPY: KernelId = KernelId("gmres_ops/axpy");
    pub const GMRES_OPS_AXPY_FROM_Y: KernelId = KernelId("gmres_ops/axpy_from_y");
    pub const GMRES_OPS_AXPBY: KernelId = KernelId("gmres_ops/axpby");
    pub const GMRES_OPS_SCALE: KernelId = KernelId("gmres_ops/scale");
    pub const GMRES_OPS_SCALE_IN_PLACE: KernelId = KernelId("gmres_ops/scale_in_place");
    pub const GMRES_OPS_COPY: KernelId = KernelId("gmres_ops/copy");
    pub const GMRES_OPS_DOT_PRODUCT_PARTIAL: KernelId = KernelId("gmres_ops/dot_product_partial");
    pub const GMRES_OPS_NORM_SQ_PARTIAL: KernelId = KernelId("gmres_ops/norm_sq_partial");
    pub const GMRES_OPS_REDUCE_FINAL: KernelId = KernelId("gmres_ops/reduce_final");
    pub const GMRES_OPS_REDUCE_FINAL_AND_FINISH_NORM: KernelId =
        KernelId("gmres_ops/reduce_final_and_finish_norm");

    pub const GMRES_LOGIC_UPDATE_HESSENBERG_GIVENS: KernelId =
        KernelId("gmres_logic/update_hessenberg_givens");
    pub const GMRES_LOGIC_SOLVE_TRIANGULAR: KernelId = KernelId("gmres_logic/solve_triangular");

    pub const GMRES_CGS_CALC_DOTS: KernelId = KernelId("gmres_cgs/calc_dots_cgs");
    pub const GMRES_CGS_REDUCE_DOTS: KernelId = KernelId("gmres_cgs/reduce_dots_cgs");
    pub const GMRES_CGS_UPDATE_W: KernelId = KernelId("gmres_cgs/update_w_cgs");

    pub fn as_str(self) -> &'static str {
        self.0
    }
}

impl From<KernelKind> for KernelId {
    fn from(kind: KernelKind) -> Self {
        match kind {
            KernelKind::PrepareCoupled => KernelId::PREPARE_COUPLED,
            KernelKind::CoupledAssembly => KernelId::COUPLED_ASSEMBLY,
            KernelKind::PressureAssembly => KernelId::PRESSURE_ASSEMBLY,
            KernelKind::UpdateFieldsFromCoupled => KernelId::UPDATE_FIELDS_FROM_COUPLED,
            KernelKind::FluxRhieChow => KernelId::FLUX_RHIE_CHOW,
            KernelKind::SystemMain => KernelId::SYSTEM_MAIN,

            KernelKind::ConservativeGradients => KernelId::CONSERVATIVE_GRADIENTS,
            KernelKind::ConservativeFluxKt => KernelId::CONSERVATIVE_FLUX_KT,
            KernelKind::ConservativeExplicitUpdate => KernelId::CONSERVATIVE_EXPLICIT_UPDATE,
            KernelKind::ConservativeAssembly => KernelId::CONSERVATIVE_ASSEMBLY,
            KernelKind::ConservativeApply => KernelId::CONSERVATIVE_APPLY,
            KernelKind::ConservativeUpdate => KernelId::CONSERVATIVE_UPDATE,

            KernelKind::GenericCoupledAssembly => KernelId::GENERIC_COUPLED_ASSEMBLY,
            KernelKind::GenericCoupledApply => KernelId::GENERIC_COUPLED_APPLY,
            KernelKind::GenericCoupledUpdate => KernelId::GENERIC_COUPLED_UPDATE,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelPlan {
    kernels: Vec<KernelKind>,
}

impl KernelPlan {
    pub fn new(kernels: Vec<KernelKind>) -> Self {
        Self { kernels }
    }

    pub fn kernels(&self) -> &[KernelKind] {
        &self.kernels
    }

    pub fn contains(&self, kind: KernelKind) -> bool {
        self.kernels.contains(&kind)
    }
}

pub fn derive_kernel_plan(system: &crate::solver::model::backend::EquationSystem) -> KernelPlan {
    // Back-compat shim: without a `ModelSpec` we cannot select a method.
    // Prefer the generic coupled path as a conservative default.
    let _ = system;
    KernelPlan::new(vec![
        KernelKind::GenericCoupledAssembly,
        KernelKind::GenericCoupledUpdate,
    ])
}

/// Derive a kernel execution ordering in terms of stable `KernelId`s.
///
/// This is the model-structure-level kernel plan used by the unified solver.
///
/// Notes:
/// - This returns only *which* kernels are required and their order.
/// - Phase assignment and dispatch kind are decided by the solver recipe.
/// - `KernelKind` is retained as a legacy/debug bridge.
pub fn derive_kernel_ids(system: &crate::solver::model::backend::EquationSystem) -> Vec<KernelId> {
    // Back-compat shim: without a `ModelSpec` we cannot select a method.
    // Prefer the generic coupled path as a conservative default.
    let _ = system;
    vec![
        KernelId::GENERIC_COUPLED_ASSEMBLY,
        KernelId::GENERIC_COUPLED_UPDATE,
    ]
}

/// Model-driven kernel plan: selected from `ModelSpec.method`.
pub fn derive_kernel_plan_for_model(model: &crate::solver::model::ModelSpec) -> KernelPlan {
    use crate::solver::model::MethodSpec;

    match model.method {
        MethodSpec::CoupledIncompressible => KernelPlan::new(vec![
            KernelKind::PrepareCoupled,
            KernelKind::FluxRhieChow,
            KernelKind::CoupledAssembly,
            KernelKind::PressureAssembly,
            KernelKind::UpdateFieldsFromCoupled,
        ]),
        MethodSpec::ConservativeCompressible { .. } => KernelPlan::new(vec![
            KernelKind::ConservativeGradients,
            KernelKind::ConservativeFluxKt,
            KernelKind::ConservativeExplicitUpdate,
            KernelKind::ConservativeAssembly,
            KernelKind::ConservativeApply,
            KernelKind::ConservativeUpdate,
        ]),
        MethodSpec::GenericCoupled => {
            let mut kernels = Vec::new();
            // Optional flux module stage.
            //
            // Transitional bridge: reuse the existing Rhie–Chow kernel when requested by the
            // model. This should eventually come from module-driven recipe emission.
            if matches!(
                model.flux_module,
                Some(crate::solver::model::flux_module::FluxModuleSpec::RhieChow { .. })
            ) {
                kernels.push(KernelKind::FluxRhieChow);
            }

            kernels.push(KernelKind::GenericCoupledAssembly);
            kernels.push(KernelKind::GenericCoupledUpdate);
            KernelPlan::new(kernels)
        }
        MethodSpec::GenericCoupledImplicit { .. } => KernelPlan::new(vec![
            // For implicit outer iterations, update is executed in the loop body.
            KernelKind::GenericCoupledAssembly,
            KernelKind::GenericCoupledUpdate,
        ]),
    }
}

/// Model-driven kernel id list: selected from `ModelSpec.method`.
pub fn derive_kernel_ids_for_model(model: &crate::solver::model::ModelSpec) -> Vec<KernelId> {
    use crate::solver::model::MethodSpec;

    match model.method {
        MethodSpec::CoupledIncompressible => vec![
            KernelId::PREPARE_COUPLED,
            KernelId::FLUX_RHIE_CHOW,
            KernelId::COUPLED_ASSEMBLY,
            KernelId::PRESSURE_ASSEMBLY,
            KernelId::UPDATE_FIELDS_FROM_COUPLED,
        ],
        MethodSpec::ConservativeCompressible { .. } => vec![
            KernelId::CONSERVATIVE_GRADIENTS,
            KernelId::CONSERVATIVE_FLUX_KT,
            KernelId::CONSERVATIVE_EXPLICIT_UPDATE,
            KernelId::CONSERVATIVE_ASSEMBLY,
            KernelId::CONSERVATIVE_APPLY,
            KernelId::CONSERVATIVE_UPDATE,
        ],
        MethodSpec::GenericCoupled => {
            let mut ids = Vec::new();
            if matches!(
                model.flux_module,
                Some(crate::solver::model::flux_module::FluxModuleSpec::RhieChow { .. })
            ) {
                ids.push(KernelId::FLUX_RHIE_CHOW);
            }
            ids.push(KernelId::GENERIC_COUPLED_ASSEMBLY);
            ids.push(KernelId::GENERIC_COUPLED_UPDATE);
            ids
        }
        MethodSpec::GenericCoupledImplicit { .. } => {
            vec![KernelId::GENERIC_COUPLED_ASSEMBLY, KernelId::GENERIC_COUPLED_UPDATE]
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PressureCoupling {
    momentum: crate::solver::model::backend::FieldRef,
    pressure: crate::solver::model::backend::FieldRef,
}

#[derive(Debug, Clone, Default)]
struct KernelRequirements {
    pressure_coupling: Option<PressureCoupling>,
}

fn analyze_kernel_requirements(
    system: &crate::solver::model::backend::EquationSystem,
) -> KernelRequirements {
    use crate::solver::model::backend::{FieldKind, FieldRef, TermOp};
    use std::collections::{HashMap, HashSet};

    let equations = system.equations();

    // Build a lookup for equations by target field.
    let mut eq_by_target: HashMap<FieldRef, usize> = HashMap::new();
    for (idx, eq) in equations.iter().enumerate() {
        eq_by_target.insert(*eq.target(), idx);
    }

    // Detect incompressible momentum+pressure coupling structurally:
    // - find a vector-target equation that takes Grad(p) for some scalar p
    // - require some transport operator (Div or Laplacian) on the momentum equation
    // - require a Laplacian term on the pressure equation targeting p
    let mut candidates: Vec<PressureCoupling> = Vec::new();
    for eq in equations {
        if eq.target().kind() != FieldKind::Vector2 {
            continue;
        }

        let momentum = *eq.target();
        let has_transport = eq
            .terms()
            .iter()
            .any(|t| matches!(t.op, TermOp::Div | TermOp::Laplacian));
        if !has_transport {
            continue;
        }

        let mut grad_scalars: HashSet<FieldRef> = HashSet::new();
        for term in eq.terms() {
            if term.op == TermOp::Grad && term.field.kind() == FieldKind::Scalar {
                grad_scalars.insert(term.field);
            }
        }

        for pressure in grad_scalars {
            let Some(&p_eq_idx) = eq_by_target.get(&pressure) else {
                continue;
            };
            let p_eq = &equations[p_eq_idx];
            let p_has_laplacian = p_eq.terms().iter().any(|t| t.op == TermOp::Laplacian);
            if p_has_laplacian {
                candidates.push(PressureCoupling { momentum, pressure });
            }
        }
    }

    KernelRequirements {
        pressure_coupling: (candidates.len() == 1).then(|| candidates[0]),
    }
}

fn synthesize_kernel_plan(req: &KernelRequirements) -> KernelPlan {
    if req.pressure_coupling.is_some() {
        return KernelPlan::new(vec![
            KernelKind::PrepareCoupled,
            KernelKind::FluxRhieChow,
            KernelKind::CoupledAssembly,
            KernelKind::PressureAssembly,
            KernelKind::UpdateFieldsFromCoupled,
        ]);
    }

    KernelPlan::new(vec![
        KernelKind::GenericCoupledAssembly,
        KernelKind::GenericCoupledApply,
        KernelKind::GenericCoupledUpdate,
    ])
}

fn synthesize_kernel_ids(req: &KernelRequirements) -> Vec<KernelId> {
    if req.pressure_coupling.is_some() {
        return vec![
            KernelId::PREPARE_COUPLED,
            KernelId::FLUX_RHIE_CHOW,
            KernelId::COUPLED_ASSEMBLY,
            KernelId::PRESSURE_ASSEMBLY,
            KernelId::UPDATE_FIELDS_FROM_COUPLED,
        ];
    }

    vec![
        KernelId::GENERIC_COUPLED_ASSEMBLY,
        KernelId::GENERIC_COUPLED_APPLY,
        KernelId::GENERIC_COUPLED_UPDATE,
    ]
}
