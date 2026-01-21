use crate::solver::model::gpu_spec::GradientStorage;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MethodSpec {
    /// Coupled multi-unknown method.
    ///
    /// This captures *capabilities* of the coupled discretization/codegen path.
    /// Solver strategy (implicit vs coupled vs explicit) is selected separately via
    /// `SolverConfig.stepping`.
    Coupled(CoupledCapabilities),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CoupledCapabilities {
    /// Whether the model's `generic_coupled_update` kernel should apply under-relaxation.
    pub apply_relaxation_in_update: bool,

    /// Whether under-relaxation should only be applied when dual-time stepping is enabled
    /// (`dtau > 0`).
    ///
    /// This is useful for compressible dual-time stepping where relaxation is used to stabilize
    /// pseudo-time iterations, but should not affect standard single-solve implicit stepping.
    pub relaxation_requires_dtau: bool,

    /// Whether this coupled method requires a `flux_module` to be present.
    pub requires_flux_module: bool,

    /// How gradients should be stored/bound for generated kernels.
    pub gradient_storage: GradientStorage,
}

impl Default for MethodSpec {
    fn default() -> Self {
        MethodSpec::Coupled(CoupledCapabilities::default())
    }
}

impl Default for CoupledCapabilities {
    fn default() -> Self {
        Self {
            apply_relaxation_in_update: false,
            relaxation_requires_dtau: false,
            requires_flux_module: false,
            gradient_storage: GradientStorage::PackedState,
        }
    }
}
