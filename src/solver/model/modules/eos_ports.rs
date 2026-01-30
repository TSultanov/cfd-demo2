/// EOS port definitions - separate from the main eos module to avoid build script issues.
use crate::solver::ir::ports::{ParamSpec, PortManifest};
use crate::solver::model::module::PortManifest as ModulePortManifest;
use crate::solver::units::UnitDim;

/// Get the port manifest for EOS uniform params.
pub fn eos_uniform_port_manifest() -> ModulePortManifest {
    ModulePortManifest {
        params: vec![
            ParamSpec {
                key: "eos.gamma",
                wgsl_field: "gamma",
                wgsl_type: "f32",
                unit: UnitDim::new(0, 0, 0), // Dimensionless
            },
            ParamSpec {
                key: "eos.gm1",
                wgsl_field: "gm1",
                wgsl_type: "f32",
                unit: UnitDim::new(0, 0, 0), // Dimensionless
            },
            ParamSpec {
                key: "eos.r",
                wgsl_field: "r",
                wgsl_type: "f32",
                unit: UnitDim::new(0, 0, 0), // Dimensionless
            },
            ParamSpec {
                key: "eos.dp_drho",
                wgsl_field: "dp_drho",
                wgsl_type: "f32",
                unit: UnitDim::new(0, 0, 0), // Dimensionless
            },
            ParamSpec {
                key: "eos.p_offset",
                wgsl_field: "p_offset",
                wgsl_type: "f32",
                unit: UnitDim::new(1, -1, -2), // Pressure
            },
            ParamSpec {
                key: "eos.theta_ref",
                wgsl_field: "theta_ref",
                wgsl_type: "f32",
                unit: UnitDim::new(0, 2, -2), // Temperature (as L^2/T^2)
            },
        ],
        fields: vec![],
        buffers: vec![],
    }
}
