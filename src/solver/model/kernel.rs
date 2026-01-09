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
