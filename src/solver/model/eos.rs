#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum EosSpec {
    /// Ideal gas equation of state with constant gamma.
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
    /// Methods that require thermodynamic closure (e.g. Euler EI) should reject this variant.
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
    /// Reference pressure for a reference-centered barotropic closure:
    /// `p = dp_drho * (rho - rho_ref) + p_ref`.
    pub p_ref: f32,
    /// Reference theta = R*T (units of p/rho) so that p = rho*theta for an isothermal ideal gas.
    pub theta_ref: f32,
    /// Reference density for the centered barotropic pressure evaluation.
    pub rho_ref: f32,
    /// GAUGE STORAGE references (all zero = absolute storage, the historical
    /// semantics). When nonzero, the density-based compressible state fields
    /// store deviations from a constant reference state:
    /// `rho_state = rho_abs - gauge_rho_ref`,
    /// `rho_e_state = rho_e_abs - gauge_e_ref`,
    /// `p_state = p_abs - gauge_p_ref`
    /// (`rho_u`, `T`, `u` remain absolute). This keeps sub-Pa acoustic
    /// perturbations representable in f32 against a large absolute base state
    /// (one f32 ULP of 105 kPa is ~0.0078 Pa). See
    /// docs/compressible-explicit-acoustics.md.
    pub gauge_rho_ref: f32,
    /// Absolute pressure of the gauge reference state (see `gauge_rho_ref`).
    pub gauge_p_ref: f32,
    /// Absolute internal-energy density of the gauge reference state.
    pub gauge_e_ref: f32,
    /// Affine tail of the STATE-form pressure closure, computed in f64 so the
    /// reference constants cancel exactly:
    /// `gauge_p_bias = gm1*gauge_e_ref + p_ref - gauge_p_ref`
    /// (`p_ref` here is the EOS affine reference). Equals `p_ref` when the
    /// gauge is off and exactly zero for a self-consistent gauge, so
    /// `p_state = gm1*(rho_e_state - ke) + dp_drho*(rho_state + (gauge_rho_ref
    /// - rho_ref)) + gauge_p_bias` holds in BOTH conventions.
    pub gauge_p_bias: f32,
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

    pub fn sound_speed(&self, _rho: f64) -> f64 {
        match *self {
            EosSpec::IdealGas {
                gamma,
                gas_constant,
                temperature,
            } => (gamma * gas_constant * temperature).sqrt(),
            EosSpec::LinearCompressibility {
                bulk_modulus,
                rho_ref,
                ..
            } => {
                // dp/drho is constant for the declared affine EOS and is set
                // by its reference density, not by the queried state.
                (bulk_modulus / rho_ref.abs().max(1e-12)).sqrt()
            }
            EosSpec::Constant => 0.0,
        }
    }

    /// Isentropic compressibility `psi = d(rho)/d(p) = 1/c^2` [s^2/m^2] — the
    /// physical value of the all-Mach `psi` field. For `IdealGas` this is
    /// `1/(gamma*R*T)`; for `LinearCompressibility` it is `rho/K = 1/dp_drho`.
    ///
    /// Returns `0.0` for the `Constant` (incompressible) EOS — the `psi = 0` limit
    /// at which the all-Mach pressure equation reduces to incompressible, keeping a
    /// `Constant`-EOS fluid byte-identical to the incompressible solver.
    pub fn compressibility(&self, rho: f64) -> f64 {
        let c = self.sound_speed(rho);
        if c > 0.0 {
            1.0 / (c * c)
        } else {
            0.0
        }
    }

    /// Thermodynamically compatible internal-energy density for a barotropic
    /// linear EOS, including an explicit density-linear energy gauge.
    ///
    /// For `p(rho) = a*rho + b`, barotropic total-energy conservation requires
    /// the specific internal energy to satisfy
    ///
    /// `d(e)/d(rho) = p(rho)/rho^2`.
    ///
    /// Taking the positive reference density `rho0 = abs(rho_ref)`, the
    /// returned energy density is
    ///
    /// `rho*e = a*rho*ln(rho/rho0) + b*(rho/rho0 - 1) + gauge*rho`.
    ///
    /// The final term is the unavoidable density-linear gauge: adding it
    /// shifts specific energy by a constant and therefore leaves the defining
    /// derivative and the conservative dynamics unchanged. `gauge = 0`
    /// chooses zero internal-energy density at the reference state. Returns
    /// `None` for non-barotropic EOS families; a non-positive or non-finite
    /// density deliberately produces a non-finite `Some` value so downstream
    /// state-health checks cannot mistake an invalid state for an ideal gas.
    pub fn barotropic_internal_energy_density(
        &self,
        rho: f64,
        density_linear_gauge: f64,
    ) -> Option<f64> {
        let EosSpec::LinearCompressibility {
            bulk_modulus,
            rho_ref,
            p_ref,
        } = *self
        else {
            return None;
        };

        let rho0 = rho_ref.abs().max(1e-12);
        let a = bulk_modulus / rho0;
        let b = p_ref - a * rho_ref;
        let delta = rho / rho0 - 1.0;

        // Evaluate (1+d)*ln(1+d)-d by series near d=0. The direct form
        // subtracts two O(d) terms and loses precisely the small liquid-energy
        // perturbation this oracle is meant to preserve.
        let shape = if delta.abs() < 1.0e-4 {
            let d2 = delta * delta;
            d2 * (0.5
                + delta
                    * (-1.0 / 6.0
                        + delta
                            * (1.0 / 12.0
                                + delta * (-1.0 / 20.0 + delta * (1.0 / 30.0)))))
        } else {
            (1.0 + delta) * delta.ln_1p() - delta
        };
        let p_at_rho0 = a * rho0 + b;
        Some(
            a * rho0 * shape
                + p_at_rho0 * delta
                + density_linear_gauge * rho,
        )
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
                p_ref: 0.0,
                theta_ref: (gas_constant * temperature) as f32,
                rho_ref: 0.0,
                gauge_rho_ref: 0.0,
                gauge_p_ref: 0.0,
                gauge_e_ref: 0.0,
                gauge_p_bias: 0.0,
            },
            EosSpec::LinearCompressibility {
                bulk_modulus,
                rho_ref,
                p_ref,
            } => {
                let denom = rho_ref.abs().max(1e-12);
                let dp_drho = bulk_modulus / denom;
                EosRuntimeParams {
                    gamma: 0.0,
                    gm1: 0.0,
                    r: 1.0,
                    dp_drho: dp_drho as f32,
                    p_ref: p_ref as f32,
                    theta_ref: 0.0,
                    rho_ref: rho_ref as f32,
                    gauge_rho_ref: 0.0,
                    gauge_p_ref: 0.0,
                    gauge_e_ref: 0.0,
                    // Gauge off: the state-form pressure closure reduces to the
                    // historical absolute form only with bias == p_ref.
                    gauge_p_bias: p_ref as f32,
                }
            }
            EosSpec::Constant => EosRuntimeParams {
                gamma: 0.0,
                gm1: 0.0,
                r: 1.0,
                dp_drho: 0.0,
                p_ref: 0.0,
                theta_ref: 0.0,
                rho_ref: 0.0,
                gauge_rho_ref: 0.0,
                gauge_p_ref: 0.0,
                gauge_e_ref: 0.0,
                gauge_p_bias: 0.0,
            },
        }
    }

    /// Absolute internal-energy density of the quiescent reference state at
    /// density `rho0`: `p(rho0)/(gamma-1)` for an ideal gas, the barotropic
    /// internal-energy oracle (zero-gauged at the EOS reference) for a linear
    /// EOS, and `0` for the constant EOS.
    pub fn internal_energy_density(&self, rho0: f64) -> f64 {
        match *self {
            EosSpec::IdealGas { gamma, .. } => {
                let gm1 = (gamma - 1.0).max(1.0e-12);
                self.pressure_for_density(rho0) / gm1
            }
            EosSpec::LinearCompressibility { .. } => self
                .barotropic_internal_energy_density(rho0, 0.0)
                .unwrap_or(0.0),
            EosSpec::Constant => 0.0,
        }
    }

    /// [`Self::runtime_params`] with GAUGE STORAGE active around the quiescent
    /// reference state at density `rho0` (see `EosRuntimeParams::gauge_rho_ref`).
    ///
    /// All reference values and the pressure-closure bias are computed in f64
    /// so the affine constants of the state-form closure cancel exactly:
    /// for an ideal gas `gauge_p_bias = gm1*e_ref - p_ref_gauge = 0` and for a
    /// linear EOS at its own reference `gauge_p_bias = p_ref - p_ref_gauge = 0`.
    pub fn runtime_params_gauged(&self, rho0: f64) -> EosRuntimeParams {
        let mut params = self.runtime_params();
        if matches!(self, EosSpec::Constant) {
            return params;
        }
        let gauge_p = self.pressure_for_density(rho0);
        // For an ideal gas divide by the f32-ROUNDED gm1 the kernels actually
        // multiply with, so the closure bias `gm1*e_ref - p_ref` cancels to
        // exactly zero and the stored pressure is exactly zero at the
        // reference state.
        let gauge_e = if matches!(self, EosSpec::IdealGas { .. }) && params.gm1 > 0.0 {
            gauge_p / f64::from(params.gm1)
        } else {
            self.internal_energy_density(rho0)
        };
        params.gauge_rho_ref = rho0 as f32;
        params.gauge_p_ref = gauge_p as f32;
        params.gauge_e_ref = gauge_e as f32;
        params.gauge_p_bias = (f64::from(params.gm1) * gauge_e
            + f64::from(params.p_ref)
            - gauge_p) as f32;
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn water_runtime_centered_pressure_matches_declared_eos_without_intercept_cancellation() {
        let eos = EosSpec::LinearCompressibility {
            bulk_modulus: 2.2e9,
            rho_ref: 1000.0,
            p_ref: 1.0e5,
        };
        let params = eos.runtime_params();
        for rho in [1000.0_f32, 1000.0001, 1000.125] {
            let runtime_pressure = params.dp_drho * (rho - params.rho_ref) + params.p_ref;
            let declared_pressure = eos.pressure_for_density(f64::from(rho)) as f32;
            assert!(
                (runtime_pressure - declared_pressure).abs() <= 1.0,
                "runtime centered pressure drift at rho={rho}: {runtime_pressure} vs {declared_pressure}"
            );
        }
        assert_eq!(
            (params.dp_drho * (1000.0 - params.rho_ref) + params.p_ref).to_bits(),
            100_000.0_f32.to_bits(),
        );
        let runtime_sound = params.dp_drho.sqrt();
        let declared_sound = eos.sound_speed(1000.0) as f32;
        assert!((runtime_sound - declared_sound).abs() <= declared_sound * 2.0e-7);
        assert_eq!(
            eos.sound_speed(1000.125).to_bits(),
            eos.sound_speed(1000.0).to_bits(),
            "affine EOS sound speed must equal its constant dp/drho"
        );
    }

    #[test]
    fn ideal_gas_runtime_parameters_remain_unchanged_by_affine_convention() {
        let eos = EosSpec::IdealGas {
            gamma: 1.4,
            gas_constant: 287.0,
            temperature: 300.0,
        };
        let params = eos.runtime_params();
        assert_eq!(params.gamma.to_bits(), 1.4_f32.to_bits());
        assert_eq!(params.gm1.to_bits(), 0.4_f32.to_bits());
        assert_eq!(params.r.to_bits(), 287.0_f32.to_bits());
        assert_eq!(params.dp_drho.to_bits(), 0.0_f32.to_bits());
        assert_eq!(params.p_ref.to_bits(), 0.0_f32.to_bits());
        assert_eq!(params.theta_ref.to_bits(), 86_100.0_f32.to_bits());
        assert_eq!(params.rho_ref.to_bits(), 0.0_f32.to_bits());
        assert_eq!(
            eos.barotropic_internal_energy_density(1.2, 0.0),
            None,
            "the barotropic oracle must not alter ideal-gas seeding"
        );
    }

    #[test]
    fn barotropic_energy_has_reference_gauge_and_pressure_derivative() {
        let eos = EosSpec::LinearCompressibility {
            bulk_modulus: 2.2e9,
            rho_ref: 1000.0,
            p_ref: 1.0e5,
        };
        let rho_ref = 1000.0;
        assert_eq!(
            eos.barotropic_internal_energy_density(rho_ref, 0.0),
            Some(0.0),
            "zero gauge must pin the reference internal energy exactly"
        );

        let rho = 1000.125;
        let h = 1.0e-3;
        let specific_energy = |r: f64| {
            eos.barotropic_internal_energy_density(r, 37.0)
                .expect("linear EOS energy")
                / r
        };
        let derivative = (specific_energy(rho + h) - specific_energy(rho - h)) / (2.0 * h);
        let expected = eos.pressure_for_density(rho) / (rho * rho);
        assert!(
            (derivative - expected).abs() <= expected.abs() * 2.0e-8,
            "de/drho={derivative:e}, p/rho^2={expected:e}"
        );

        let zero_gauge = eos
            .barotropic_internal_energy_density(rho, 0.0)
            .expect("linear EOS energy");
        let shifted = eos
            .barotropic_internal_energy_density(rho, 37.0)
            .expect("linear EOS energy");
        assert!((shifted - zero_gauge - 37.0 * rho).abs() <= 1.0e-8);
    }
}
