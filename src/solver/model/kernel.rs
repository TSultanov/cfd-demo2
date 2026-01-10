#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelKind {
    PrepareCoupled,
    CoupledAssembly,
    PressureAssembly,
    UpdateFieldsFromCoupled,
    FluxRhieChow,
    IncompressibleMomentum,
    CompressibleAssembly,
    CompressibleApply,
    CompressibleGradients,
    CompressibleExplicitUpdate,
    CompressibleUpdate,
    CompressibleFluxKt,
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

    pub const COMPRESSIBLE_ASSEMBLY: KernelId = KernelId("compressible_assembly");
    pub const COMPRESSIBLE_APPLY: KernelId = KernelId("compressible_apply");
    pub const COMPRESSIBLE_GRADIENTS: KernelId = KernelId("compressible_gradients");
    pub const COMPRESSIBLE_EXPLICIT_UPDATE: KernelId = KernelId("compressible_explicit_update");
    pub const COMPRESSIBLE_UPDATE: KernelId = KernelId("compressible_update");
    pub const COMPRESSIBLE_FLUX_KT: KernelId = KernelId("compressible_flux_kt");

    pub const GENERIC_COUPLED_ASSEMBLY: KernelId = KernelId("generic_coupled_assembly");
    pub const GENERIC_COUPLED_APPLY: KernelId = KernelId("generic_coupled_apply");
    pub const GENERIC_COUPLED_UPDATE: KernelId = KernelId("generic_coupled_update");

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
            KernelKind::IncompressibleMomentum => KernelId("incompressible_momentum"),
            KernelKind::CompressibleAssembly => KernelId::COMPRESSIBLE_ASSEMBLY,
            KernelKind::CompressibleApply => KernelId::COMPRESSIBLE_APPLY,
            KernelKind::CompressibleGradients => KernelId::COMPRESSIBLE_GRADIENTS,
            KernelKind::CompressibleExplicitUpdate => KernelId::COMPRESSIBLE_EXPLICIT_UPDATE,
            KernelKind::CompressibleUpdate => KernelId::COMPRESSIBLE_UPDATE,
            KernelKind::CompressibleFluxKt => KernelId::COMPRESSIBLE_FLUX_KT,
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

fn analyze_kernel_requirements(system: &crate::solver::model::backend::EquationSystem) -> KernelRequirements {
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
            KernelKind::CompressibleGradients,
            KernelKind::CompressibleFluxKt,
            KernelKind::CompressibleExplicitUpdate,
            KernelKind::CompressibleAssembly,
            KernelKind::CompressibleApply,
            KernelKind::CompressibleUpdate,
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
            KernelId::COMPRESSIBLE_GRADIENTS,
            KernelId::COMPRESSIBLE_FLUX_KT,
            KernelId::COMPRESSIBLE_EXPLICIT_UPDATE,
            KernelId::COMPRESSIBLE_ASSEMBLY,
            KernelId::COMPRESSIBLE_APPLY,
            KernelId::COMPRESSIBLE_UPDATE,
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
