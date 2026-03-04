//! Shared helpers for coupled assembly kernel code generation.
//!
//! Functions extracted from `generic_coupled_kernels.rs` and `unified_assembly.rs`
//! to eliminate duplication.

use std::collections::HashMap;

use super::coeff_expr::coeff_cell_expr;
use super::constants::constants_struct;
use super::wgsl_ast::{AccessMode, Attribute, Expr, Item, StorageClass, Stmt, Type};
use super::wgsl_bindings::{storage_var, uniform_var, vector2_struct};
use crate::solver::codegen::ir::DiscreteSystem;
use crate::solver::ir::ports::{ParamSpec, ResolvedStateSlotsSpec};
use crate::solver::ir::{BindingAccess, Coefficient, KernelBinding};

/// Flatten a coupled system's equations into a list of `(FieldRef, component_index)` pairs.
pub fn coupled_unknown_components(
    system: &DiscreteSystem,
) -> Vec<(crate::solver::ir::FieldRef, u32)> {
    let mut out = Vec::new();
    for equation in &system.equations {
        let count = equation.target.kind().component_count() as u32;
        for component in 0..count {
            out.push((equation.target, component));
        }
    }
    out
}

/// Build a `HashMap<field_name, base_offset>` mapping each equation's target field
/// to its starting index in the coupled unknown vector.
pub fn coupled_offsets(system: &DiscreteSystem) -> HashMap<String, u32> {
    let mut offsets = HashMap::new();
    let mut current = 0u32;
    for equation in &system.equations {
        offsets.insert(equation.target.name().to_string(), current);
        current += equation.target.kind().component_count() as u32;
    }
    offsets
}

/// Generate an expression for reading a coefficient value at a cell.
pub fn coefficient_value_expr(
    slots: &ResolvedStateSlotsSpec,
    coeff: Option<&Coefficient>,
    idx_ident: &str,
    default: Expr,
) -> Expr {
    coeff_cell_expr(slots, coeff, idx_ident, default)
}

/// Emit WGSL global declarations for mesh data (group 0 bindings).
pub fn base_mesh_items(eos_params: &[ParamSpec]) -> Vec<Item> {
    vec![
        Item::Struct(vector2_struct()),
        Item::Struct(constants_struct(eos_params)),
        Item::Comment("Group 0: Mesh".to_string()),
        storage_var("face_owner", Type::array(Type::U32), 0, 0, AccessMode::Read),
        storage_var(
            "face_neighbor",
            Type::array(Type::I32),
            0,
            1,
            AccessMode::Read,
        ),
        storage_var("face_areas", Type::array(Type::F32), 0, 2, AccessMode::Read),
        storage_var(
            "face_normals",
            Type::array(Type::Custom("Vector2".to_string())),
            0,
            3,
            AccessMode::Read,
        ),
        storage_var(
            "face_centers",
            Type::array(Type::Custom("Vector2".to_string())),
            0,
            13,
            AccessMode::Read,
        ),
        storage_var(
            "cell_centers",
            Type::array(Type::Custom("Vector2".to_string())),
            0,
            4,
            AccessMode::Read,
        ),
        storage_var("cell_vols", Type::array(Type::F32), 0, 5, AccessMode::Read),
        storage_var(
            "cell_face_offsets",
            Type::array(Type::U32),
            0,
            6,
            AccessMode::Read,
        ),
        storage_var("cell_faces", Type::array(Type::U32), 0, 7, AccessMode::Read),
        storage_var(
            "cell_face_matrix_indices",
            Type::array(Type::U32),
            0,
            10,
            AccessMode::Read,
        ),
        storage_var(
            "diagonal_indices",
            Type::array(Type::U32),
            0,
            11,
            AccessMode::Read,
        ),
        storage_var(
            "face_boundary",
            Type::array(Type::U32),
            0,
            12,
            AccessMode::Read,
        ),
    ]
}

/// Emit WGSL global declarations for state fields (group 1 bindings).
///
/// When `needs_fluxes` is true, an additional `fluxes` storage buffer is included.
pub fn base_state_items(needs_gradients: bool, needs_fluxes: bool) -> Vec<Item> {
    let mut items = vec![
        Item::Comment("Group 1: Fields".to_string()),
        storage_var("state", Type::array(Type::F32), 1, 0, AccessMode::ReadWrite),
        storage_var("state_old", Type::array(Type::F32), 1, 1, AccessMode::Read),
        storage_var(
            "state_old_old",
            Type::array(Type::F32),
            1,
            2,
            AccessMode::Read,
        ),
        uniform_var("constants", Type::Custom("Constants".to_string()), 1, 3),
        storage_var("state_iter", Type::array(Type::F32), 1, 4, AccessMode::Read),
    ];
    if needs_gradients {
        items.push(storage_var(
            "grad_state",
            Type::array(Type::Custom("Vector2".to_string())),
            1,
            5,
            AccessMode::Read,
        ));
    }
    if needs_fluxes {
        items.push(storage_var(
            "fluxes",
            Type::array(Type::F32),
            1,
            6,
            AccessMode::ReadWrite,
        ));
    }
    items
}

/// Emit all WGSL global items for an assembly kernel: mesh + state + solver + BCs.
pub fn base_assembly_items(
    needs_gradients: bool,
    needs_fluxes: bool,
    eos_params: &[ParamSpec],
) -> Vec<Item> {
    let mut items = Vec::new();
    items.extend(base_mesh_items(eos_params));
    items.extend(base_state_items(needs_gradients, needs_fluxes));
    items.push(Item::Comment(
        "Group 2: Solver (block CSR values + RHS)".to_string(),
    ));
    items.push(storage_var(
        "matrix_values",
        Type::array(Type::F32),
        2,
        0,
        AccessMode::ReadWrite,
    ));
    items.push(storage_var(
        "rhs",
        Type::array(Type::F32),
        2,
        1,
        AccessMode::ReadWrite,
    ));
    items.push(storage_var(
        "scalar_row_offsets",
        Type::array(Type::U32),
        2,
        2,
        AccessMode::Read,
    ));
    items.push(Item::Comment(
        "Group 3: Boundary conditions (per face x unknown)".to_string(),
    ));
    items.push(storage_var(
        "bc_kind",
        Type::array(Type::U32),
        3,
        0,
        AccessMode::Read,
    ));
    items.push(storage_var(
        "bc_value",
        Type::array(Type::F32),
        3,
        1,
        AccessMode::Read,
    ));
    items
}

/// Extract `KernelBinding` metadata from WGSL `Item` AST nodes.
pub fn kernel_bindings_from_items(items: &[Item]) -> Result<Vec<KernelBinding>, String> {
    let mut bindings = Vec::new();
    for item in items {
        let Item::GlobalVar(var) = item else {
            continue;
        };
        let Some((group, binding)) = group_binding_from_attributes(&var.attributes) else {
            continue;
        };
        let access = match var.storage {
            StorageClass::Storage => match var.access {
                Some(AccessMode::Read) => BindingAccess::ReadOnlyStorage,
                Some(AccessMode::ReadWrite) => BindingAccess::ReadWriteStorage,
                None => {
                    return Err(format!(
                        "kernel_bindings_from_items: storage var '{}' missing access mode",
                        var.name
                    ));
                }
            },
            StorageClass::Uniform => BindingAccess::Uniform,
            StorageClass::Workgroup => continue,
        };
        bindings.push(KernelBinding::new(
            group,
            binding,
            &var.name,
            var.ty.to_string(),
            access,
        ));
    }
    Ok(bindings)
}

/// Parse `@group(N) @binding(M)` from an attribute list.
pub fn group_binding_from_attributes(attrs: &[Attribute]) -> Option<(u32, u32)> {
    let mut group = None;
    let mut binding = None;
    for attr in attrs {
        match attr {
            Attribute::Group(value) => group = Some(*value),
            Attribute::Binding(value) => binding = Some(*value),
            _ => {}
        }
    }
    group.zip(binding)
}

// ---------------------------------------------------------------------------
// Time-derivative (ddt) contribution helpers
// ---------------------------------------------------------------------------

/// Emit AST statements for all implicit time-derivative contributions across
/// all equations in the system.
///
/// Delegates to [`super::time_integration::emit_ddt_contributions`] using the
/// default [`super::time_integration::BdfDualTimeIntegrator`].
pub fn emit_ddt_contributions(
    system: &DiscreteSystem,
    slots: &ResolvedStateSlotsSpec,
    offsets: &std::collections::HashMap<String, u32>,
    acc: &super::dsl::CoupledAccumulators,
) -> Vec<Stmt> {
    let integrator = super::time_integration::BdfDualTimeIntegrator;
    super::time_integration::emit_ddt_contributions(system, slots, offsets, acc, &integrator)
}
