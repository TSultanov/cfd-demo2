#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EosSpec {
    /// Ideal gas equation of state with constant gamma.
    ///
    /// Used by the current EI Euler implementation.
    IdealGas { gamma: f32 },

    /// Incompressible (or otherwise non-thermodynamic) model: EOS does not depend on state.
    ///
    /// This is intentionally minimal for now; methods that require thermodynamic closure
    /// (e.g. Euler EI) should reject this variant.
    Constant,
}

impl Default for EosSpec {
    fn default() -> Self {
        // Treat "no EOS provided" as incompressible/constant.
        EosSpec::Constant
    }
}

impl EosSpec {
    pub fn ideal_gas_gamma(&self) -> Option<f32> {
        match *self {
            EosSpec::IdealGas { gamma } => Some(gamma),
            EosSpec::Constant => None,
        }
    }
}
