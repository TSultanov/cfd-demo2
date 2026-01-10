#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MethodSpec {
    /// Explicit-Implicit Conservative (EI) method (historically used for the "compressible" demo).
    ExplicitImplicitConservative,

    /// Pressure-coupled incompressible momentum + pressure method.
    CoupledIncompressible,

    /// Generic coupled scalar/vector method (unified assembly/apply/update path).
    GenericCoupled,
}

impl Default for MethodSpec {
    fn default() -> Self {
        // Prefer a safe default that does not assume EI/coupled structure.
        MethodSpec::GenericCoupled
    }
}
