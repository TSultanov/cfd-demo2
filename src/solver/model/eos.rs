#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum EosSpec {
    /// Ideal gas equation of state with constant gamma.
    ///
    /// Used by the current EI Euler implementation.
    IdealGas {
        gamma: f64,
        gas_constant: f64,
        temperature: f64,
    },

    /// Barotropic linear compressibility model (often used as a weakly-compressible liquid EOS).
    ///
    /// p = p_ref + K * (rho - rho_ref) / rho_ref
    LinearCompressibility {
        bulk_modulus: f64,
        rho_ref: f64,
        p_ref: f64,
    },

    /// Incompressible (or otherwise non-thermodynamic) model: EOS does not depend on state.
    ///
    /// This is intentionally minimal for now; methods that require thermodynamic closure
    /// (e.g. Euler EI) should reject this variant.
    #[default]
    Constant,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EosRuntimeParams {
    /// Gamma used by the Euler flux wave-speed estimate (gamma*p/rho).
    pub gamma: f32,
    /// (gamma - 1) used by the calorically-perfect-gas closure p=(gamma-1)*(rho_e - 0.5 rho |u|^2).
    pub gm1: f32,
    /// Gas constant R in solver units (has units of p/(rho*T)).
    pub r: f32,
    /// dp/drho contribution for barotropic closures (e.g. linear compressibility).
    pub dp_drho: f32,
    /// p offset for barotropic closures: p = dp_drho * rho - p_offset.
    pub p_offset: f32,
    /// Reference theta = R*T (units of p/rho) so that p = rho*theta for an isothermal ideal gas.
    pub theta_ref: f32,
}


impl EosSpec {
    pub fn ideal_gas_gamma(&self) -> Option<f32> {
        match *self {
            EosSpec::IdealGas { gamma, .. } => Some(gamma as f32),
            EosSpec::LinearCompressibility { .. } | EosSpec::Constant => None,
        }
    }

    pub fn pressure_for_density(&self, rho: f64) -> f64 {
        match *self {
            EosSpec::IdealGas {
                gas_constant,
                temperature,
                ..
            } => rho * gas_constant * temperature,
            EosSpec::LinearCompressibility {
                bulk_modulus,
                rho_ref,
                p_ref,
            } => {
                let denom = rho_ref.abs().max(1e-12);
                p_ref + bulk_modulus * (rho - rho_ref) / denom
            }
            EosSpec::Constant => 0.0,
        }
    }

    pub fn sound_speed(&self, rho: f64) -> f64 {
        match *self {
            EosSpec::IdealGas {
                gamma,
                gas_constant,
                temperature,
            } => (gamma * gas_constant * temperature).sqrt(),
            EosSpec::LinearCompressibility { bulk_modulus, .. } => {
                (bulk_modulus / rho.abs().max(1e-12)).sqrt()
            }
            EosSpec::Constant => 0.0,
        }
    }

    pub fn runtime_params(&self) -> EosRuntimeParams {
        match *self {
            EosSpec::IdealGas {
                gamma,
                gas_constant,
                temperature,
            } => EosRuntimeParams {
                gamma: gamma as f32,
                gm1: (gamma - 1.0) as f32,
                r: gas_constant as f32,
                dp_drho: 0.0,
                p_offset: 0.0,
                theta_ref: (gas_constant * temperature) as f32,
            },
            EosSpec::LinearCompressibility {
                bulk_modulus,
                rho_ref,
                p_ref,
            } => {
                let denom = rho_ref.abs().max(1e-12);
                let dp_drho = bulk_modulus / denom;
                let p_offset = dp_drho * rho_ref - p_ref;
                EosRuntimeParams {
                    gamma: 0.0,
                    gm1: 0.0,
                    r: 1.0,
                    dp_drho: dp_drho as f32,
                    p_offset: p_offset as f32,
                    theta_ref: 0.0,
                }
            }
            EosSpec::Constant => EosRuntimeParams {
                gamma: 0.0,
                gm1: 0.0,
                r: 1.0,
                dp_drho: 0.0,
                p_offset: 0.0,
                theta_ref: 0.0,
            },
        }
    }
}
