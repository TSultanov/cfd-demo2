/// EOS port definitions - separate from the main eos module to avoid build script issues.
use cfd2_ir::solver::dimensions::{Dimensionless, DivDim, MulDim, Pressure, Density, Temperature, UnitDimension};
use crate::solver::ir::ports::ParamSpec;
use crate::solver::model::module::PortManifest as ModulePortManifest;

/// Get the port manifest for EOS uniform params.
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
                key: "eos.p_offset",
                wgsl_field: "eos_p_offset",
                wgsl_type: "f32",
                unit: Pressure::UNIT,
            },
            ParamSpec {
                key: "eos.theta_ref",
                wgsl_field: "eos_theta_ref",
                wgsl_type: "f32",
                unit: dp_drho_unit, // theta = P/rho has units L²/T² (specific energy)
            },
        ],
        fields: vec![],
        buffers: vec![],
        gradient_targets: vec![],
        resolved_state_slots: None,
    }
}
