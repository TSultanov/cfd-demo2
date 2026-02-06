use crate::solver::model::EosSpec;

#[derive(Clone, PartialEq, Debug)]
pub struct Fluid {
    pub name: String,
    pub density: f64,
    pub viscosity: f64,
    pub eos: EosSpec,
}

impl Fluid {
    pub fn presets() -> Vec<Fluid> {
        vec![
            Fluid {
                name: "Water".into(),
                density: 1000.0,
                viscosity: 0.001,
                eos: EosSpec::LinearCompressibility {
                    bulk_modulus: 2.2e9,
                    rho_ref: 1000.0,
                    p_ref: 1.0e5,
                },
            },
            Fluid {
                name: "Air".into(),
                density: 1.225,
                viscosity: 1.81e-5,
                eos: EosSpec::IdealGas {
                    gamma: 1.4,
                    gas_constant: 287.0,
                    temperature: 300.0,
                },
            },
            Fluid {
                name: "Alcohol".into(),
                density: 789.0,
                viscosity: 0.0012,
                eos: EosSpec::LinearCompressibility {
                    bulk_modulus: 1.0e9,
                    rho_ref: 789.0,
                    p_ref: 1.0e5,
                },
            },
            Fluid {
                name: "Kerosene".into(),
                density: 820.0,
                viscosity: 0.00164,
                eos: EosSpec::LinearCompressibility {
                    bulk_modulus: 1.3e9,
                    rho_ref: 820.0,
                    p_ref: 1.0e5,
                },
            },
            Fluid {
                name: "Mercury".into(),
                density: 13546.0,
                viscosity: 0.001526,
                eos: EosSpec::LinearCompressibility {
                    bulk_modulus: 2.85e10,
                    rho_ref: 13546.0,
                    p_ref: 1.0e5,
                },
            },
            Fluid {
                name: "Custom".into(),
                density: 1.0,
                viscosity: 0.01,
                eos: EosSpec::LinearCompressibility {
                    bulk_modulus: 2.2e9,
                    rho_ref: 1.0,
                    p_ref: 1.0e5,
                },
            },
        ]
    }

    pub fn pressure_for_density(&self, rho: f64) -> f64 {
        self.eos.pressure_for_density(rho)
    }

    pub fn sound_speed(&self) -> f64 {
        self.eos.sound_speed(self.density)
    }
}
