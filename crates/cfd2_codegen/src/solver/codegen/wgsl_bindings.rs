//! Shared WGSL binding helpers for code generation.
//!
//! This module provides common functions for generating WGSL variable declarations
//! and struct definitions used across multiple codegen modules.

use super::wgsl_ast::{
    AccessMode, Attribute, GlobalVar, Item, StorageClass, StructDef, StructField, Type,
};

/// Create a storage variable declaration with the given binding attributes.
pub fn storage_var(name: &str, ty: Type, group: u32, binding: u32, access: AccessMode) -> Item {
    Item::GlobalVar(GlobalVar::new(
        name,
        ty,
        StorageClass::Storage,
        Some(access),
        vec![Attribute::Group(group), Attribute::Binding(binding)],
    ))
}

/// Create a uniform variable declaration with the given binding attributes.
pub fn uniform_var(name: &str, ty: Type, group: u32, binding: u32) -> Item {
    Item::GlobalVar(GlobalVar::new(
        name,
        ty,
        StorageClass::Uniform,
        None,
        vec![Attribute::Group(group), Attribute::Binding(binding)],
    ))
}

/// Create the standard Vector2 struct definition.
pub fn vector2_struct() -> StructDef {
    StructDef::new(
        "Vector2",
        vec![
            StructField::new("x", Type::F32),
            StructField::new("y", Type::F32),
        ],
    )
}

/// Create the LowMachParams struct definition used by flux modules.
pub fn low_mach_params_struct() -> StructDef {
    StructDef::new(
        "LowMachParams",
        vec![
            StructField::new("model", Type::U32),
            StructField::new("theta_floor", Type::F32),
            StructField::new("pressure_coupling_alpha", Type::F32),
            StructField::new("_pad0", Type::F32),
        ],
    )
}

/// Create the standard boundary condition bindings (bc_kind and bc_value).
pub fn boundary_bindings() -> Vec<Item> {
    vec![
        storage_var("bc_kind", Type::array(Type::U32), 2, 0, AccessMode::Read),
        storage_var("bc_value", Type::array(Type::F32), 2, 1, AccessMode::Read),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_var() {
        let item = storage_var("test", Type::array(Type::F32), 0, 1, AccessMode::Read);
        match item {
            Item::GlobalVar(gv) => {
                assert_eq!(gv.name, "test");
            }
            _ => panic!("Expected GlobalVar"),
        }
    }

    #[test]
    fn test_uniform_var() {
        let item = uniform_var("constants", Type::Custom("Constants".to_string()), 1, 4);
        match item {
            Item::GlobalVar(gv) => {
                assert_eq!(gv.name, "constants");
            }
            _ => panic!("Expected GlobalVar"),
        }
    }

    #[test]
    fn test_vector2_struct() {
        let s = vector2_struct();
        assert_eq!(s.name, "Vector2");
        assert_eq!(s.fields.len(), 2);
    }
}
