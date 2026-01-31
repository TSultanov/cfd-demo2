use crate::solver::codegen::dsl::{DslType, DynExpr, TypedExpr};
use crate::solver::codegen::wgsl_ast::Expr;
use crate::solver::ir::ports::{PortFieldKind, ResolvedStateSlotSpec, ResolvedStateSlotsSpec};
use cfd2_ir::solver::dimensions::UnitDimension;

/// Find a slot by name in the resolved state slots spec.
fn find_slot<'a>(slots: &'a ResolvedStateSlotsSpec, field: &str) -> Option<&'a ResolvedStateSlotSpec> {
    slots.slots.iter().find(|s| s.name == field)
}

/// Compute the offset for a specific component of a field.
/// Panics if the field is not found or component is out of range.
fn compute_offset(slots: &ResolvedStateSlotsSpec, field: &str, component: u32) -> u32 {
    let slot = find_slot(slots, field).unwrap_or_else(|| {
        panic!("missing field '{}' in resolved state slots", field);
    });
    if component >= slot.kind.component_count() {
        panic!(
            "component {} out of range for field '{}' (kind: {:?})",
            component, field, slot.kind
        );
    }
    slot.base_offset + component
}

pub fn state_component_expr(
    slots: &ResolvedStateSlotsSpec,
    buffer: &str,
    idx: &str,
    field: &str,
    component: u32,
) -> String {
    let offset = compute_offset(slots, field, component);
    let stride = slots.stride;
    format!("{}[{} * {}u + {}u]", buffer, idx, stride, offset)
}

pub fn state_scalar_expr(
    slots: &ResolvedStateSlotsSpec,
    buffer: &str,
    idx: &str,
    field: &str,
) -> String {
    state_component_expr(slots, buffer, idx, field, 0)
}

pub fn state_vec2_expr(
    slots: &ResolvedStateSlotsSpec,
    buffer: &str,
    idx: &str,
    field: &str,
) -> String {
    let x_expr = state_component_expr(slots, buffer, idx, field, 0);
    let y_expr = state_component_expr(slots, buffer, idx, field, 1);
    format!("vec2<f32>({}, {})", x_expr, y_expr)
}

pub fn state_vec3_expr(
    slots: &ResolvedStateSlotsSpec,
    buffer: &str,
    idx: &str,
    field: &str,
) -> String {
    let x_expr = state_component_expr(slots, buffer, idx, field, 0);
    let y_expr = state_component_expr(slots, buffer, idx, field, 1);
    let z_expr = state_component_expr(slots, buffer, idx, field, 2);
    format!("vec3<f32>({}, {}, {})", x_expr, y_expr, z_expr)
}

pub fn state_component(
    slots: &ResolvedStateSlotsSpec,
    buffer: &str,
    idx: &str,
    field: &str,
    component: u32,
) -> Expr {
    let offset = compute_offset(slots, field, component);
    let stride = slots.stride;
    let idx_expr = Expr::ident(idx);
    let index_expr = idx_expr * stride + offset;
    Expr::ident(buffer).index(index_expr)
}

pub fn state_scalar(
    slots: &ResolvedStateSlotsSpec,
    buffer: &str,
    idx: &str,
    field: &str,
) -> Expr {
    state_component(slots, buffer, idx, field, 0)
}

pub fn state_vec2(
    slots: &ResolvedStateSlotsSpec,
    buffer: &str,
    idx: &str,
    field: &str,
) -> Expr {
    let x_expr = state_component(slots, buffer, idx, field, 0);
    let y_expr = state_component(slots, buffer, idx, field, 1);
    Expr::call_named("vec2<f32>", vec![x_expr, y_expr])
}

pub fn state_vec3(
    slots: &ResolvedStateSlotsSpec,
    buffer: &str,
    idx: &str,
    field: &str,
) -> Expr {
    let x_expr = state_component(slots, buffer, idx, field, 0);
    let y_expr = state_component(slots, buffer, idx, field, 1);
    let z_expr = state_component(slots, buffer, idx, field, 2);
    Expr::call_named("vec3<f32>", vec![x_expr, y_expr, z_expr])
}

pub fn state_component_typed(
    slots: &ResolvedStateSlotsSpec,
    buffer: &str,
    idx: &str,
    field: &str,
    component: u32,
) -> DynExpr {
    let slot = find_slot(slots, field).unwrap_or_else(|| {
        panic!("missing field '{}' in resolved state slots", field);
    });
    let expr = state_component(slots, buffer, idx, field, component);
    DynExpr::new(expr, DslType::f32(), slot.unit)
}

pub fn state_scalar_typed(
    slots: &ResolvedStateSlotsSpec,
    buffer: &str,
    idx: &str,
    field: &str,
) -> DynExpr {
    let slot = find_slot(slots, field).unwrap_or_else(|| {
        panic!("missing field '{}' in resolved state slots", field);
    });
    if slot.kind != PortFieldKind::Scalar {
        panic!("field '{}' is not scalar (kind: {:?})", field, slot.kind);
    }
    state_component_typed(slots, buffer, idx, field, 0)
}

pub fn state_vec2_typed(
    slots: &ResolvedStateSlotsSpec,
    buffer: &str,
    idx: &str,
    field: &str,
) -> DynExpr {
    let slot = find_slot(slots, field).unwrap_or_else(|| {
        panic!("missing field '{}' in resolved state slots", field);
    });
    if slot.kind != PortFieldKind::Vector2 {
        panic!("field '{}' is not vec2 (kind: {:?})", field, slot.kind);
    }
    let x_expr = state_component(slots, buffer, idx, field, 0);
    let y_expr = state_component(slots, buffer, idx, field, 1);
    DynExpr::new(
        Expr::call_named("vec2<f32>", vec![x_expr, y_expr]),
        DslType::vec2_f32(),
        slot.unit,
    )
}

pub fn state_vec3_typed(
    slots: &ResolvedStateSlotsSpec,
    buffer: &str,
    idx: &str,
    field: &str,
) -> DynExpr {
    let slot = find_slot(slots, field).unwrap_or_else(|| {
        panic!("missing field '{}' in resolved state slots", field);
    });
    if slot.kind != PortFieldKind::Vector3 {
        panic!("field '{}' is not vec3 (kind: {:?})", field, slot.kind);
    }
    let x_expr = state_component(slots, buffer, idx, field, 0);
    let y_expr = state_component(slots, buffer, idx, field, 1);
    let z_expr = state_component(slots, buffer, idx, field, 2);
    DynExpr::new(
        Expr::call_named("vec3<f32>", vec![x_expr, y_expr, z_expr]),
        DslType::vec3_f32(),
        slot.unit,
    )
}

// ============================================================================
// Type-Level Dimension State Access Helpers
// ============================================================================

/// Access a scalar state field with compile-time unit checking.
/// 
/// Panics if:
/// - The field is not found
/// - The field is not a scalar
/// - The slot unit does not match the type-level dimension `D`
pub fn state_scalar_dim<D: UnitDimension>(
    slots: &ResolvedStateSlotsSpec,
    buffer: &str,
    idx: &str,
    field: &str,
) -> TypedExpr<D> {
    let slot = find_slot(slots, field).unwrap_or_else(|| {
        panic!("missing field '{}' in resolved state slots", field);
    });
    if slot.kind != PortFieldKind::Scalar {
        panic!("field '{}' is not scalar (kind: {:?})", field, slot.kind);
    }
    if slot.unit != D::UNIT {
        panic!(
            "unit mismatch for field '{}': expected {:?}, got {:?}",
            field, D::UNIT, slot.unit
        );
    }
    let expr = state_component(slots, buffer, idx, field, 0);
    TypedExpr::new(expr, DslType::f32())
}

/// Access a vec2 state field with compile-time unit checking.
/// 
/// Panics if:
/// - The field is not found
/// - The field is not a Vector2
/// - The slot unit does not match the type-level dimension `D`
pub fn state_vec2_dim<D: UnitDimension>(
    slots: &ResolvedStateSlotsSpec,
    buffer: &str,
    idx: &str,
    field: &str,
) -> TypedExpr<D> {
    let slot = find_slot(slots, field).unwrap_or_else(|| {
        panic!("missing field '{}' in resolved state slots", field);
    });
    if slot.kind != PortFieldKind::Vector2 {
        panic!("field '{}' is not vec2 (kind: {:?})", field, slot.kind);
    }
    if slot.unit != D::UNIT {
        panic!(
            "unit mismatch for field '{}': expected {:?}, got {:?}",
            field, D::UNIT, slot.unit
        );
    }
    let x_expr = state_component(slots, buffer, idx, field, 0);
    let y_expr = state_component(slots, buffer, idx, field, 1);
    let expr = Expr::call_named("vec2<f32>", vec![x_expr, y_expr]);
    TypedExpr::new(expr, DslType::vec2_f32())
}

/// Access a vec3 state field with compile-time unit checking.
/// 
/// Panics if:
/// - The field is not found
/// - The field is not a Vector3
/// - The slot unit does not match the type-level dimension `D`
pub fn state_vec3_dim<D: UnitDimension>(
    slots: &ResolvedStateSlotsSpec,
    buffer: &str,
    idx: &str,
    field: &str,
) -> TypedExpr<D> {
    let slot = find_slot(slots, field).unwrap_or_else(|| {
        panic!("missing field '{}' in resolved state slots", field);
    });
    if slot.kind != PortFieldKind::Vector3 {
        panic!("field '{}' is not vec3 (kind: {:?})", field, slot.kind);
    }
    if slot.unit != D::UNIT {
        panic!(
            "unit mismatch for field '{}': expected {:?}, got {:?}",
            field, D::UNIT, slot.unit
        );
    }
    let x_expr = state_component(slots, buffer, idx, field, 0);
    let y_expr = state_component(slots, buffer, idx, field, 1);
    let z_expr = state_component(slots, buffer, idx, field, 2);
    let expr = Expr::call_named("vec3<f32>", vec![x_expr, y_expr, z_expr]);
    TypedExpr::new(expr, DslType::vec3_f32())
}

/// Access a single component of a state field with compile-time unit checking.
/// 
/// Panics if:
/// - The field is not found
/// - The component is out of range for the field's kind
/// - The slot unit does not match the type-level dimension `D`
pub fn state_component_dim<D: UnitDimension>(
    slots: &ResolvedStateSlotsSpec,
    buffer: &str,
    idx: &str,
    field: &str,
    component: u32,
) -> TypedExpr<D> {
    let slot = find_slot(slots, field).unwrap_or_else(|| {
        panic!("missing field '{}' in resolved state slots", field);
    });
    if component >= slot.kind.component_count() {
        panic!(
            "component {} out of range for field '{}' (kind: {:?})",
            component, field, slot.kind
        );
    }
    if slot.unit != D::UNIT {
        panic!(
            "unit mismatch for field '{}': expected {:?}, got {:?}",
            field, D::UNIT, slot.unit
        );
    }
    let expr = state_component(slots, buffer, idx, field, component);
    TypedExpr::new(expr, DslType::f32())
}

/// Resolve a state offset by field name, supporting component suffixes (e.g., "rho_u_x").
/// Returns the absolute offset in the state array for the given name.
pub fn resolve_state_offset_by_name(slots: &ResolvedStateSlotsSpec, name: &str) -> Option<u32> {
    // First, try direct field lookup
    if let Some(slot) = find_slot(slots, name) {
        return Some(slot.base_offset);
    }

    // Try to parse component suffix (_x, _y, _z)
    let (base, component) = name.rsplit_once('_')?;
    let component = match component {
        "x" => 0,
        "y" => 1,
        "z" => 2,
        _ => return None,
    };

    let slot = find_slot(slots, base)?;
    if component >= slot.kind.component_count() {
        return None;
    }
    Some(slot.base_offset + component)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::ir::ports::PortFieldKind;
    use crate::solver::units::si;
    use cfd2_ir::solver::dimensions::{Pressure, Velocity};

    /// Helper to create a ResolvedStateSlotsSpec from field definitions for testing.
    fn test_slots_from_fields(
        fields: Vec<(&str, PortFieldKind, crate::solver::units::UnitDim)>,
    ) -> ResolvedStateSlotsSpec {
        let mut slots = Vec::new();
        let mut current_offset = 0u32;
        for (name, kind, unit) in fields {
            slots.push(ResolvedStateSlotSpec {
                name: name.to_string(),
                kind,
                unit,
                base_offset: current_offset,
            });
            current_offset += kind.component_count();
        }
        ResolvedStateSlotsSpec {
            stride: current_offset,
            slots,
        }
    }

    #[test]
    fn state_access_builds_scalar_and_vector_exprs() {
        let slots = test_slots_from_fields(vec![
            ("U", PortFieldKind::Vector2, si::VELOCITY),
            ("p", PortFieldKind::Scalar, si::PRESSURE),
        ]);
        let scalar = state_scalar_expr(&slots, "state", "idx", "p");
        assert_eq!(scalar, "state[idx * 3u + 2u]");

        let component = state_component_expr(&slots, "state", "idx", "U", 1);
        assert_eq!(component, "state[idx * 3u + 1u]");

        let vec2 = state_vec2_expr(&slots, "state", "idx", "U");
        assert_eq!(
            vec2,
            "vec2<f32>(state[idx * 3u + 0u], state[idx * 3u + 1u])"
        );
    }

    #[test]
    fn state_access_builds_expr_ast() {
        let slots = test_slots_from_fields(vec![
            ("U", PortFieldKind::Vector2, si::VELOCITY),
            ("p", PortFieldKind::Scalar, si::PRESSURE),
        ]);
        let scalar = state_scalar(&slots, "state", "idx", "p");
        assert_eq!(scalar.to_string(), "state[idx * 3u + 2u]");

        let component = state_component(&slots, "state", "idx", "U", 1);
        assert_eq!(component.to_string(), "state[idx * 3u + 1u]");

        let vec2 = state_vec2(&slots, "state", "idx", "U");
        assert_eq!(
            vec2.to_string(),
            "vec2<f32>(state[idx * 3u + 0u], state[idx * 3u + 1u])"
        );
    }

    #[test]
    fn typed_state_access_tracks_units() {
        let slots = test_slots_from_fields(vec![
            ("U", PortFieldKind::Vector2, si::VELOCITY),
            ("p", PortFieldKind::Scalar, si::PRESSURE),
        ]);
        let p = state_scalar_typed(&slots, "state", "idx", "p");
        assert_eq!(p.ty, DslType::f32());
        assert_eq!(p.unit, si::PRESSURE);
        assert_eq!(p.expr.to_string(), "state[idx * 3u + 2u]");

        let u = state_vec2_typed(&slots, "state", "idx", "U");
        assert_eq!(u.ty, DslType::vec2_f32());
        assert_eq!(u.unit, si::VELOCITY);
        assert_eq!(
            u.expr.to_string(),
            "vec2<f32>(state[idx * 3u + 0u], state[idx * 3u + 1u])"
        );
    }

    #[test]
    fn resolve_state_offset_by_name_handles_components() {
        let slots = test_slots_from_fields(vec![
            ("rho", PortFieldKind::Scalar, si::DENSITY),
            ("rho_u", PortFieldKind::Vector2, si::MOMENTUM_DENSITY),
        ]);

        // Direct field access
        assert_eq!(resolve_state_offset_by_name(&slots, "rho"), Some(0));
        assert_eq!(resolve_state_offset_by_name(&slots, "rho_u"), Some(1));

        // Component access
        assert_eq!(resolve_state_offset_by_name(&slots, "rho_u_x"), Some(1));
        assert_eq!(resolve_state_offset_by_name(&slots, "rho_u_y"), Some(2));

        // Invalid access
        assert_eq!(resolve_state_offset_by_name(&slots, "nonexistent"), None);
        assert_eq!(resolve_state_offset_by_name(&slots, "rho_u_z"), None); // Vector2 has no z
    }

    // ============================================================================
    // Type-Level Dimension State Access Tests
    // ============================================================================

    #[test]
    fn typed_dim_state_scalar_access_works() {
        let slots = test_slots_from_fields(vec![
            ("p", PortFieldKind::Scalar, si::PRESSURE),
        ]);
        
        // Access "p" with Pressure dimension type
        let p: TypedExpr<Pressure> = state_scalar_dim(&slots, "state", "idx", "p");
        
        // Verify the expression string matches
        assert_eq!(p.expr.to_string(), "state[idx * 1u + 0u]");
        
        // Verify into_dyn().unit matches the slot unit
        let dyn_p = p.into_dyn();
        assert_eq!(dyn_p.unit, si::PRESSURE);
        assert_eq!(dyn_p.ty, DslType::f32());
    }

    #[test]
    fn typed_dim_state_vec2_access_works() {
        let slots = test_slots_from_fields(vec![
            ("U", PortFieldKind::Vector2, si::VELOCITY),
        ]);
        
        // Access "U" with Velocity dimension type
        let u: TypedExpr<Velocity> = state_vec2_dim(&slots, "state", "idx", "U");
        
        // Verify the expression string matches
        assert_eq!(
            u.expr.to_string(),
            "vec2<f32>(state[idx * 2u + 0u], state[idx * 2u + 1u])"
        );
        
        // Verify into_dyn().unit matches the slot unit
        let dyn_u = u.into_dyn();
        assert_eq!(dyn_u.unit, si::VELOCITY);
        assert_eq!(dyn_u.ty, DslType::vec2_f32());
    }

    #[test]
    fn typed_dim_state_component_access_works() {
        let slots = test_slots_from_fields(vec![
            ("U", PortFieldKind::Vector2, si::VELOCITY),
        ]);
        
        // Access component 0 of "U" with Velocity dimension type
        let u_x: TypedExpr<Velocity> = state_component_dim(&slots, "state", "idx", "U", 0);
        
        // Verify the expression string matches
        assert_eq!(u_x.expr.to_string(), "state[idx * 2u + 0u]");
        
        // Verify into_dyn().unit matches the slot unit
        let dyn_u_x = u_x.into_dyn();
        assert_eq!(dyn_u_x.unit, si::VELOCITY);
        assert_eq!(dyn_u_x.ty, DslType::f32());
    }

    #[test]
    #[should_panic(expected = "unit mismatch for field 'p'")]
    fn typed_dim_state_scalar_unit_mismatch_panics() {
        let slots = test_slots_from_fields(vec![
            ("p", PortFieldKind::Scalar, si::PRESSURE),
        ]);
        
        // Try to access "p" with Velocity dimension type (should panic)
        let _: TypedExpr<Velocity> = state_scalar_dim(&slots, "state", "idx", "p");
    }

    #[test]
    #[should_panic(expected = "unit mismatch for field 'U'")]
    fn typed_dim_state_vec2_unit_mismatch_panics() {
        let slots = test_slots_from_fields(vec![
            ("U", PortFieldKind::Vector2, si::VELOCITY),
        ]);
        
        // Try to access "U" with Pressure dimension type (should panic)
        let _: TypedExpr<Pressure> = state_vec2_dim(&slots, "state", "idx", "U");
    }

    #[test]
    #[should_panic(expected = "field 'p' is not vec2")]
    fn typed_dim_state_kind_mismatch_scalar_to_vec2_panics() {
        let slots = test_slots_from_fields(vec![
            ("p", PortFieldKind::Scalar, si::PRESSURE),
        ]);
        
        // Try to access scalar "p" as vec2 (should panic)
        let _: TypedExpr<Pressure> = state_vec2_dim(&slots, "state", "idx", "p");
    }

    #[test]
    #[should_panic(expected = "field 'U' is not scalar")]
    fn typed_dim_state_kind_mismatch_vec2_to_scalar_panics() {
        let slots = test_slots_from_fields(vec![
            ("U", PortFieldKind::Vector2, si::VELOCITY),
        ]);
        
        // Try to access vec2 "U" as scalar (should panic)
        let _: TypedExpr<Velocity> = state_scalar_dim(&slots, "state", "idx", "U");
    }

    #[test]
    fn typed_dim_and_dyn_helpers_coexist() {
        // Verify that both typed-dimension and dynamic helpers work together
        let slots = test_slots_from_fields(vec![
            ("U", PortFieldKind::Vector2, si::VELOCITY),
            ("p", PortFieldKind::Scalar, si::PRESSURE),
        ]);
        
        // Typed dimension access
        let p_typed: TypedExpr<Pressure> = state_scalar_dim(&slots, "state", "idx", "p");
        assert_eq!(p_typed.expr.to_string(), "state[idx * 3u + 2u]");
        
        // Dynamic access (escape hatch)
        let p_dyn = state_scalar_typed(&slots, "state", "idx", "p");
        assert_eq!(p_dyn.expr.to_string(), "state[idx * 3u + 2u]");
        assert_eq!(p_dyn.unit, si::PRESSURE);
        
        // They should produce equivalent expressions
        assert_eq!(p_typed.expr.to_string(), p_dyn.expr.to_string());
    }
}
