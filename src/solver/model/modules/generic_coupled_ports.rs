/// Generic coupled module port definitions - separate from the main module to avoid build script issues.
use cfd2_ir::solver::dimensions::{Dimensionless, Density, DynamicViscosity, Time, UnitDimension};
use crate::solver::ir::ports::ParamSpec;
use crate::solver::model::module::PortManifest as ModulePortManifest;

/// Get the port manifest for generic_coupled uniform params.
pub fn generic_coupled_uniform_port_manifest(
    apply_relaxation_in_update: bool,
) -> ModulePortManifest {
    let mut params = vec![
        ParamSpec {
            key: "dt",
            wgsl_field: "dt",
            wgsl_type: "f32",
            unit: Time::UNIT,
        },
        ParamSpec {
            key: "dtau",
            wgsl_field: "dtau",
            wgsl_type: "f32",
            unit: Time::UNIT,
        },
        ParamSpec {
            key: "viscosity",
            wgsl_field: "viscosity",
            wgsl_type: "f32",
            unit: DynamicViscosity::UNIT,
        },
        ParamSpec {
            key: "density",
            wgsl_field: "density",
            wgsl_type: "f32",
            unit: Density::UNIT,
        },
        ParamSpec {
            key: "advection_scheme",
            wgsl_field: "scheme",
            wgsl_type: "u32",
            unit: Dimensionless::UNIT,
        },
        ParamSpec {
            key: "time_scheme",
            wgsl_field: "time_scheme",
            wgsl_type: "u32",
            unit: Dimensionless::UNIT,
        },
    ];

    if apply_relaxation_in_update {
        params.extend([
            ParamSpec {
                key: "alpha_u",
                wgsl_field: "alpha_u",
                wgsl_type: "f32",
                unit: Dimensionless::UNIT,
            },
            ParamSpec {
                key: "alpha_p",
                wgsl_field: "alpha_p",
                wgsl_type: "f32",
                unit: Dimensionless::UNIT,
            },
        ]);
    }

    ModulePortManifest {
        params,
        fields: vec![],
        buffers: vec![],
        gradient_targets: vec![],
        resolved_state_slots: None,
    }
}
