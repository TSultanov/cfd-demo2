//! Shared helper for building the WGSL `Constants` struct.
//!
//! This module provides a centralized way to construct the `Constants` uniform struct
//! used across generated WGSL kernels, supporting both base constants and optional
//! extra fields from parameter specs (e.g., EOS module parameters).

use super::wgsl_ast::{StructDef, StructField, Type};
use cfd2_ir::solver::ir::ports::ParamSpec;

/// Build the standard WGSL `Constants` struct definition.
///
/// # Arguments
///
/// * `extra_params` - Optional extra parameter specs to append to the base constants.
///   These are typically EOS module parameters (`eos_gamma`, `eos_gm1`, etc.).
///
/// # Returns
///
/// A `StructDef` containing the complete `Constants` struct with all fields.
pub fn constants_struct(extra_params: &[ParamSpec]) -> StructDef {
    let mut fields = base_constant_fields();

    // Append extra params from the slice (e.g., EOS module params)
    for param in extra_params {
        let ty = wgsl_type_to_ast_type(param.wgsl_type);
        fields.push(StructField::new(param.wgsl_field, ty));
    }

    StructDef::new("Constants", fields)
}

/// Returns the base constant fields shared across all kernels.
///
/// These fields are always present in the `Constants` struct:
/// - `dt`, `dt_old`, `dtau`: Time stepping
/// - `time`: Simulation time
/// - `viscosity`, `density`: Physical properties
/// - `component`: Component index for multi-component solves
/// - `alpha_p`, `alpha_u`: Relaxation factors
/// - `scheme`: Advection scheme selector
/// - `stride_x`: Grid stride for 2D indexing
/// - `time_scheme`: Time integration scheme selector
fn base_constant_fields() -> Vec<StructField> {
    vec![
        StructField::new("dt", Type::F32),
        StructField::new("dt_old", Type::F32),
        StructField::new("dtau", Type::F32),
        StructField::new("time", Type::F32),
        StructField::new("viscosity", Type::F32),
        StructField::new("density", Type::F32),
        StructField::new("component", Type::U32),
        StructField::new("alpha_p", Type::F32),
        StructField::new("scheme", Type::U32),
        StructField::new("alpha_u", Type::F32),
        StructField::new("stride_x", Type::U32),
        StructField::new("time_scheme", Type::U32),
    ]
}

/// Convert a WGSL type string to a `Type` AST node.
///
/// # Supported Types
///
/// * `"f32"` -> `Type::F32`
/// * `"u32"` -> `Type::U32`
///
/// # Panics
///
/// Panics if the type string is not supported.
fn wgsl_type_to_ast_type(wgsl_type: &str) -> Type {
    match wgsl_type {
        "f32" => Type::F32,
        "u32" => Type::U32,
        _ => panic!("unsupported WGSL type for constants struct: {wgsl_type}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_struct_with_no_extra_params_has_base_fields_only() {
        let def = constants_struct(&[]);
        assert_eq!(def.name, "Constants");
        assert_eq!(def.fields.len(), 12); // Base fields only

        // Verify first and last base fields
        assert_eq!(def.fields[0].name, "dt");
        assert_eq!(def.fields[11].name, "time_scheme");
    }

    #[test]
    fn constants_struct_with_extra_params_appends_them() {
        let extra = vec![
            ParamSpec {
                key: "eos.gamma",
                wgsl_field: "eos_gamma",
                wgsl_type: "f32",
                unit: cfd2_ir::solver::units::si::DIMENSIONLESS,
            },
            ParamSpec {
                key: "eos.gm1",
                wgsl_field: "eos_gm1",
                wgsl_type: "f32",
                unit: cfd2_ir::solver::units::si::DIMENSIONLESS,
            },
        ];

        let def = constants_struct(&extra);
        assert_eq!(def.fields.len(), 14); // 12 base + 2 extra

        // Verify extra fields are appended after base fields
        assert_eq!(def.fields[12].name, "eos_gamma");
        assert_eq!(def.fields[12].ty, Type::F32);
        assert_eq!(def.fields[13].name, "eos_gm1");
        assert_eq!(def.fields[13].ty, Type::F32);
    }

    #[test]
    fn constants_struct_preserves_order_of_extra_params() {
        let extra = vec![
            ParamSpec {
                key: "eos.first",
                wgsl_field: "eos_first",
                wgsl_type: "f32",
                unit: cfd2_ir::solver::units::si::DIMENSIONLESS,
            },
            ParamSpec {
                key: "eos.second",
                wgsl_field: "eos_second",
                wgsl_type: "u32",
                unit: cfd2_ir::solver::units::si::DIMENSIONLESS,
            },
        ];

        let def = constants_struct(&extra);
        assert_eq!(def.fields[12].name, "eos_first");
        assert_eq!(def.fields[13].name, "eos_second");
    }
}
