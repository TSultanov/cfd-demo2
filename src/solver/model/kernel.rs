#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelKind {
    PrepareCoupled,
    CoupledAssembly,
    PressureAssembly,
    UpdateFieldsFromCoupled,
    FluxRhieChow,
    SystemMain,
    EiAssembly,
    EiApply,
    EiGradients,
    EiExplicitUpdate,
    EiUpdate,
    EiFluxKt,
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

    pub const EI_ASSEMBLY: KernelId = KernelId("ei_assembly");
    pub const EI_APPLY: KernelId = KernelId("ei_apply");
    pub const EI_GRADIENTS: KernelId = KernelId("ei_gradients");
    pub const EI_EXPLICIT_UPDATE: KernelId = KernelId("ei_explicit_update");
    pub const EI_UPDATE: KernelId = KernelId("ei_update");
    pub const EI_FLUX_KT: KernelId = KernelId("ei_flux_kt");

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
            KernelKind::EiAssembly => KernelId::EI_ASSEMBLY,
            KernelKind::EiApply => KernelId::EI_APPLY,
            KernelKind::EiGradients => KernelId::EI_GRADIENTS,
            KernelKind::EiExplicitUpdate => KernelId::EI_EXPLICIT_UPDATE,
            KernelKind::EiUpdate => KernelId::EI_UPDATE,
            KernelKind::EiFluxKt => KernelId::EI_FLUX_KT,
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
    let req = analyze_kernel_requirements(system);
    synthesize_kernel_plan(&req)
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
    let req = analyze_kernel_requirements(system);
    synthesize_kernel_ids(&req)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PressureCoupling {
    momentum: crate::solver::model::backend::FieldRef,
    pressure: crate::solver::model::backend::FieldRef,
}

#[derive(Debug, Clone, Default)]
struct KernelRequirements {
    has_div_flux: bool,
    pressure_coupling: Option<PressureCoupling>,
}

fn analyze_kernel_requirements(
    system: &crate::solver::model::backend::EquationSystem,
) -> KernelRequirements {
    use crate::solver::model::backend::{FieldKind, FieldRef, TermOp};
    use std::collections::{HashMap, HashSet};

    let equations = system.equations();

    let has_div_flux = equations
        .iter()
        .any(|eq| eq.terms().iter().any(|t| t.op == TermOp::DivFlux));

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
        has_div_flux,
        pressure_coupling: (candidates.len() == 1).then(|| candidates[0]),
    }
}

fn synthesize_kernel_plan(req: &KernelRequirements) -> KernelPlan {
    // Priority order is intentional: if a system uses DivFlux it should use the compressible
    // path even if it accidentally also matches other coupling patterns.
    if req.has_div_flux {
        return KernelPlan::new(vec![
            KernelKind::EiGradients,
            KernelKind::EiFluxKt,
            KernelKind::EiExplicitUpdate,
            KernelKind::EiAssembly,
            KernelKind::EiApply,
            KernelKind::EiUpdate,
        ]);
    }

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
    // Priority order is intentional: if a system uses DivFlux it should use the compressible
    // path even if it accidentally also matches other coupling patterns.
    if req.has_div_flux {
        return vec![
            KernelId::EI_GRADIENTS,
            KernelId::EI_FLUX_KT,
            KernelId::EI_EXPLICIT_UPDATE,
            KernelId::EI_ASSEMBLY,
            KernelId::EI_APPLY,
            KernelId::EI_UPDATE,
        ];
    }

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
