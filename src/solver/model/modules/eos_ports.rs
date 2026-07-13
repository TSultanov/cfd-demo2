/// EOS port definitions - separate from the main eos module to avoid build script issues.
use cfd2_ir::dimensions::{Dimensionless, DivDim, MulDim, Pressure, Density, Temperature, UnitDimension};
use crate::solver::ir::ports::ParamSpec;
use crate::solver::model::module::PortManifest as ModulePortManifest;

pub fn eos_uniform_port_manifest() -> ModulePortManifest {
    // R = P/(rho*T) has units of (ML⁻¹T⁻²)/(ML⁻³·K) = L²T⁻²K⁻¹
    let gas_constant_unit = DivDim::<Pressure, MulDim<Density, Temperature>>::UNIT;

    // dp/drho has units of P/rho = (ML⁻¹T⁻²)/(ML⁻³) = L²T⁻²
    let dp_drho_unit = DivDim::<Pressure, Density>::UNIT;

    ModulePortManifest {
        params: vec![
            ParamSpec {
                key: "eos.gamma",
                wgsl_field: "eos_gamma",
                wgsl_type: "f32",
                unit: Dimensionless::UNIT,
            },
            ParamSpec {
                key: "eos.gm1",
                wgsl_field: "eos_gm1",
                wgsl_type: "f32",
                unit: Dimensionless::UNIT,
            },
            ParamSpec {
                key: "eos.r",
                wgsl_field: "eos_r",
                wgsl_type: "f32",
                unit: gas_constant_unit,
            },
            ParamSpec {
                key: "eos.dp_drho",
                wgsl_field: "eos_dp_drho",
                wgsl_type: "f32",
                unit: dp_drho_unit,
            },
            ParamSpec {
                key: "eos.p_ref",
                wgsl_field: "eos_p_ref",
                wgsl_type: "f32",
                unit: Pressure::UNIT,
            },
            ParamSpec {
                key: "eos.theta_ref",
                wgsl_field: "eos_theta_ref",
                wgsl_type: "f32",
                unit: dp_drho_unit, // theta = P/rho has units L²/T² (specific energy)
            },
            ParamSpec {
                key: "eos.rho_ref",
                wgsl_field: "eos_rho_ref",
                wgsl_type: "f32",
                unit: Density::UNIT,
            },
            // Gauge-storage references (state stores deviations from a constant
            // reference; zero = absolute storage). `e_ref`/`p_bias` share the
            // Pressure unit dimension (J/m^3 == Pa).
            ParamSpec {
                key: "eos.gauge_rho_ref",
                wgsl_field: "eos_gauge_rho_ref",
                wgsl_type: "f32",
                unit: Density::UNIT,
            },
            ParamSpec {
                key: "eos.gauge_p_ref",
                wgsl_field: "eos_gauge_p_ref",
                wgsl_type: "f32",
                unit: Pressure::UNIT,
            },
            ParamSpec {
                key: "eos.gauge_e_ref",
                wgsl_field: "eos_gauge_e_ref",
                wgsl_type: "f32",
                unit: Pressure::UNIT,
            },
            ParamSpec {
                key: "eos.gauge_p_bias",
                wgsl_field: "eos_gauge_p_bias",
                wgsl_type: "f32",
                unit: Pressure::UNIT,
            },
            // Inlet driving mode for the density-based compressible family
            // (0 = velocity inlet, 1 = pressure inlet + floating outlet).
            ParamSpec {
                key: "eos.bc_pressure_inlet",
                wgsl_field: "bc_pressure_inlet",
                wgsl_type: "f32",
                unit: Dimensionless::UNIT,
            },
        ],
        fields: vec![],
        buffers: vec![],
        gradient_targets: vec![],
        resolved_state_slots: None,
    }
}
