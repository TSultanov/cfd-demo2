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
