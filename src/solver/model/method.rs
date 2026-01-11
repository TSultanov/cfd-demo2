#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MethodSpec {
    /// Pressure-coupled incompressible momentum + pressure method.
    CoupledIncompressible,

    /// Legacy compressible explicit/implicit family (EI kernels), selected without
    /// EI-specific branches in the solver.
    ConservativeCompressible { outer_iters: usize },

    /// Generic coupled scalar/vector method (unified assembly/apply/update path).
    GenericCoupled,

    /// Generic coupled method with an implicit outer iteration loop.
    ///
    /// This is intended as a transitional configuration while we bring all
    /// multi-unknown systems through the generic discretization/codegen path.
    GenericCoupledImplicit { outer_iters: usize },
}

impl Default for MethodSpec {
    fn default() -> Self {
        // Prefer a safe default that does not assume EI/coupled structure.
        MethodSpec::GenericCoupled
    }
}
