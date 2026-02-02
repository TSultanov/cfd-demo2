use crate::solver::codegen::dsl::{DslType, DynExpr, TypedExpr};
use crate::solver::codegen::wgsl_ast::Expr;
use crate::solver::ir::ports::{PortFieldKind, ResolvedStateSlotSpec, ResolvedStateSlotsSpec};
use cfd2_ir::solver::dimensions::UnitDimension;

/// Find a slot by name in the resolved state slots spec.
pub fn find_slot<'a>(
    slots: &'a ResolvedStateSlotsSpec,
    field: &str,
) -> Option<&'a ResolvedStateSlotSpec> {
    slots.slots.iter().find(|s| s.name == field)
}

/// Compute the offset for a specific component of a slot.
/// Panics if the component is out of range.
fn compute_offset_for_slot(slot: &ResolvedStateSlotSpec, component: u32) -> u32 {
    if component >= slot.kind.component_count() {
        panic!(
            "component {} out of range for field '{}' (kind: {:?})",
            component, slot.name, slot.kind
        );
    }
    slot.base_offset + component
}

// ============================================================================
// Slot-Based State Access Helpers (no name lookup)
// ============================================================================

/// Access a single component of a state field by slot (no name lookup).
pub fn state_component_slot(
    stride: u32,
    buffer: &str,
    idx: &str,
    slot: &ResolvedStateSlotSpec,
    component: u32,
) -> Expr {
    let offset = compute_offset_for_slot(slot, component);
    let idx_expr = Expr::ident(idx);
    let index_expr = idx_expr * stride + offset;
    Expr::ident(buffer).index(index_expr)
}

/// Access a scalar state field by slot (no name lookup).
pub fn state_scalar_slot(
    stride: u32,
    buffer: &str,
    idx: &str,
    slot: &ResolvedStateSlotSpec,
) -> Expr {
    state_component_slot(stride, buffer, idx, slot, 0)
}

/// Access a vec2 state field by slot (no name lookup).
pub fn state_vec2_slot(stride: u32, buffer: &str, idx: &str, slot: &ResolvedStateSlotSpec) -> Expr {
    let x_expr = state_component_slot(stride, buffer, idx, slot, 0);
    let y_expr = state_component_slot(stride, buffer, idx, slot, 1);
    Expr::call_named("vec2<f32>", vec![x_expr, y_expr])
}

/// Access a vec3 state field by slot (no name lookup).
pub fn state_vec3_slot(stride: u32, buffer: &str, idx: &str, slot: &ResolvedStateSlotSpec) -> Expr {
    let x_expr = state_component_slot(stride, buffer, idx, slot, 0);
    let y_expr = state_component_slot(stride, buffer, idx, slot, 1);
    let z_expr = state_component_slot(stride, buffer, idx, slot, 2);
    Expr::call_named("vec3<f32>", vec![x_expr, y_expr, z_expr])
}

/// Access a single component of a state field by slot with unit tracking (no name lookup).
pub fn state_component_slot_typed(
    stride: u32,
    buffer: &str,
    idx: &str,
    slot: &ResolvedStateSlotSpec,
    component: u32,
) -> DynExpr {
    let expr = state_component_slot(stride, buffer, idx, slot, component);
    DynExpr::new(expr, DslType::f32(), slot.unit)
}

/// Access a scalar state field by slot with unit tracking (no name lookup).
///
/// Panics if the slot is not a scalar.
pub fn state_scalar_slot_typed(
    stride: u32,
    buffer: &str,
    idx: &str,
    slot: &ResolvedStateSlotSpec,
) -> DynExpr {
    if slot.kind != PortFieldKind::Scalar {
        panic!(
            "field '{}' is not scalar (kind: {:?})",
            slot.name, slot.kind
        );
    }
    state_component_slot_typed(stride, buffer, idx, slot, 0)
}

/// Access a vec2 state field by slot with unit tracking (no name lookup).
///
/// Panics if the slot is not a Vector2.
pub fn state_vec2_slot_typed(
    stride: u32,
    buffer: &str,
    idx: &str,
    slot: &ResolvedStateSlotSpec,
) -> DynExpr {
    if slot.kind != PortFieldKind::Vector2 {
        panic!("field '{}' is not vec2 (kind: {:?})", slot.name, slot.kind);
    }
    let x_expr = state_component_slot(stride, buffer, idx, slot, 0);
    let y_expr = state_component_slot(stride, buffer, idx, slot, 1);
    DynExpr::new(
        Expr::call_named("vec2<f32>", vec![x_expr, y_expr]),
        DslType::vec2_f32(),
        slot.unit,
    )
}

/// Access a vec3 state field by slot with unit tracking (no name lookup).
///
/// Panics if the slot is not a Vector3.
pub fn state_vec3_slot_typed(
    stride: u32,
    buffer: &str,
    idx: &str,
    slot: &ResolvedStateSlotSpec,
) -> DynExpr {
    if slot.kind != PortFieldKind::Vector3 {
        panic!("field '{}' is not vec3 (kind: {:?})", slot.name, slot.kind);
    }
    let x_expr = state_component_slot(stride, buffer, idx, slot, 0);
    let y_expr = state_component_slot(stride, buffer, idx, slot, 1);
    let z_expr = state_component_slot(stride, buffer, idx, slot, 2);
    DynExpr::new(
        Expr::call_named("vec3<f32>", vec![x_expr, y_expr, z_expr]),
        DslType::vec3_f32(),
        slot.unit,
    )
}

/// Access a scalar state field by slot with compile-time unit checking (no name lookup).
///
/// Panics if:
/// - The slot is not a scalar
/// - The slot unit does not match the type-level dimension `D`
pub fn state_scalar_slot_dim<D: UnitDimension>(
    stride: u32,
    buffer: &str,
    idx: &str,
    slot: &ResolvedStateSlotSpec,
) -> TypedExpr<D> {
    if slot.kind != PortFieldKind::Scalar {
        panic!(
            "field '{}' is not scalar (kind: {:?})",
            slot.name, slot.kind
        );
    }
    if slot.unit != D::UNIT {
        panic!(
            "unit mismatch for field '{}': expected {:?}, got {:?}",
            slot.name,
            D::UNIT,
            slot.unit
        );
    }
    let expr = state_component_slot(stride, buffer, idx, slot, 0);
    TypedExpr::new(expr, DslType::f32())
}

/// Access a vec2 state field by slot with compile-time unit checking (no name lookup).
///
/// Panics if:
/// - The slot is not a Vector2
/// - The slot unit does not match the type-level dimension `D`
pub fn state_vec2_slot_dim<D: UnitDimension>(
    stride: u32,
    buffer: &str,
    idx: &str,
    slot: &ResolvedStateSlotSpec,
) -> TypedExpr<D> {
    if slot.kind != PortFieldKind::Vector2 {
        panic!("field '{}' is not vec2 (kind: {:?})", slot.name, slot.kind);
    }
    if slot.unit != D::UNIT {
        panic!(
            "unit mismatch for field '{}': expected {:?}, got {:?}",
            slot.name,
            D::UNIT,
            slot.unit
        );
    }
    let x_expr = state_component_slot(stride, buffer, idx, slot, 0);
    let y_expr = state_component_slot(stride, buffer, idx, slot, 1);
    let expr = Expr::call_named("vec2<f32>", vec![x_expr, y_expr]);
    TypedExpr::new(expr, DslType::vec2_f32())
}

/// Access a vec3 state field by slot with compile-time unit checking (no name lookup).
///
/// Panics if:
/// - The slot is not a Vector3
/// - The slot unit does not match the type-level dimension `D`
pub fn state_vec3_slot_dim<D: UnitDimension>(
    stride: u32,
    buffer: &str,
    idx: &str,
    slot: &ResolvedStateSlotSpec,
) -> TypedExpr<D> {
    if slot.kind != PortFieldKind::Vector3 {
        panic!("field '{}' is not vec3 (kind: {:?})", slot.name, slot.kind);
    }
    if slot.unit != D::UNIT {
        panic!(
            "unit mismatch for field '{}': expected {:?}, got {:?}",
            slot.name,
            D::UNIT,
            slot.unit
        );
    }
    let x_expr = state_component_slot(stride, buffer, idx, slot, 0);
    let y_expr = state_component_slot(stride, buffer, idx, slot, 1);
    let z_expr = state_component_slot(stride, buffer, idx, slot, 2);
    let expr = Expr::call_named("vec3<f32>", vec![x_expr, y_expr, z_expr]);
    TypedExpr::new(expr, DslType::vec3_f32())
}

/// Access a single component of a state field by slot with compile-time unit checking (no name lookup).
///
/// Panics if:
/// - The component is out of range for the slot's kind
/// - The slot unit does not match the type-level dimension `D`
pub fn state_component_slot_dim<D: UnitDimension>(
    stride: u32,
    buffer: &str,
    idx: &str,
    slot: &ResolvedStateSlotSpec,
    component: u32,
) -> TypedExpr<D> {
    if component >= slot.kind.component_count() {
        panic!(
            "component {} out of range for field '{}' (kind: {:?})",
            component, slot.name, slot.kind
        );
    }
    if slot.unit != D::UNIT {
        panic!(
            "unit mismatch for field '{}': expected {:?}, got {:?}",
            slot.name,
            D::UNIT,
            slot.unit
        );
    }
    let expr = state_component_slot(stride, buffer, idx, slot, component);
    TypedExpr::new(expr, DslType::f32())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::ir::ports::PortFieldKind;
    use cfd2_ir::solver::dimensions::{Density, MomentumDensity, Pressure, Velocity};

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

    // ============================================================================
    // Slot-Based State Access Tests
    // ============================================================================

    #[test]
    fn slot_based_state_access_builds_scalar_and_vector_exprs() {
        let slots = test_slots_from_fields(vec![
            ("U", PortFieldKind::Vector2, Velocity::UNIT),
            ("p", PortFieldKind::Scalar, Pressure::UNIT),
        ]);

        let p_slot = find_slot(&slots, "p").unwrap();
        let u_slot = find_slot(&slots, "U").unwrap();

        // Test scalar access
        let scalar = state_scalar_slot(slots.stride, "state", "idx", p_slot);
        assert_eq!(scalar.to_string(), "state[idx * 3u + 2u]");

        // Test component access
        let component = state_component_slot(slots.stride, "state", "idx", u_slot, 1);
        assert_eq!(component.to_string(), "state[idx * 3u + 1u]");

        // Test vec2 access
        let vec2 = state_vec2_slot(slots.stride, "state", "idx", u_slot);
        assert_eq!(
            vec2.to_string(),
            "vec2<f32>(state[idx * 3u + 0u], state[idx * 3u + 1u])"
        );
    }

    #[test]
    fn slot_based_typed_state_access_tracks_units() {
        let slots = test_slots_from_fields(vec![
            ("U", PortFieldKind::Vector2, Velocity::UNIT),
            ("p", PortFieldKind::Scalar, Pressure::UNIT),
        ]);

        let p_slot = find_slot(&slots, "p").unwrap();
        let u_slot = find_slot(&slots, "U").unwrap();

        // Test typed scalar access
        let p = state_scalar_slot_typed(slots.stride, "state", "idx", p_slot);
        assert_eq!(p.ty, DslType::f32());
        assert_eq!(p.unit, Pressure::UNIT);
        assert_eq!(p.expr.to_string(), "state[idx * 3u + 2u]");

        // Test typed vec2 access
        let u = state_vec2_slot_typed(slots.stride, "state", "idx", u_slot);
        assert_eq!(u.ty, DslType::vec2_f32());
        assert_eq!(u.unit, Velocity::UNIT);
        assert_eq!(
            u.expr.to_string(),
            "vec2<f32>(state[idx * 3u + 0u], state[idx * 3u + 1u])"
        );
    }

    #[test]
    fn slot_based_typed_dim_state_scalar_access_works() {
        let slots = test_slots_from_fields(vec![("p", PortFieldKind::Scalar, Pressure::UNIT)]);

        let p_slot = find_slot(&slots, "p").unwrap();

        // Access "p" with Pressure dimension type
        let p: TypedExpr<Pressure> = state_scalar_slot_dim(slots.stride, "state", "idx", p_slot);

        // Verify the expression string matches
        assert_eq!(p.expr.to_string(), "state[idx * 1u + 0u]");

        // Verify into_dyn().unit matches the slot unit
        let dyn_p = p.into_dyn();
        assert_eq!(dyn_p.unit, Pressure::UNIT);
        assert_eq!(dyn_p.ty, DslType::f32());
    }

    #[test]
    fn slot_based_typed_dim_state_vec2_access_works() {
        let slots = test_slots_from_fields(vec![("U", PortFieldKind::Vector2, Velocity::UNIT)]);

        let u_slot = find_slot(&slots, "U").unwrap();

        // Access "U" with Velocity dimension type
        let u: TypedExpr<Velocity> = state_vec2_slot_dim(slots.stride, "state", "idx", u_slot);

        // Verify the expression string matches
        assert_eq!(
            u.expr.to_string(),
            "vec2<f32>(state[idx * 2u + 0u], state[idx * 2u + 1u])"
        );

        // Verify into_dyn().unit matches the slot unit
        let dyn_u = u.into_dyn();
        assert_eq!(dyn_u.unit, Velocity::UNIT);
        assert_eq!(dyn_u.ty, DslType::vec2_f32());
    }

    #[test]
    fn slot_based_typed_dim_state_component_access_works() {
        let slots = test_slots_from_fields(vec![("U", PortFieldKind::Vector2, Velocity::UNIT)]);

        let u_slot = find_slot(&slots, "U").unwrap();

        // Access component 0 of "U" with Velocity dimension type
        let u_x: TypedExpr<Velocity> =
            state_component_slot_dim(slots.stride, "state", "idx", u_slot, 0);

        // Verify the expression string matches
        assert_eq!(u_x.expr.to_string(), "state[idx * 2u + 0u]");

        // Verify into_dyn().unit matches the slot unit
        let dyn_u_x = u_x.into_dyn();
        assert_eq!(dyn_u_x.unit, Velocity::UNIT);
        assert_eq!(dyn_u_x.ty, DslType::f32());
    }

    #[test]
    #[should_panic(expected = "unit mismatch for field 'p'")]
    fn slot_based_typed_dim_state_scalar_unit_mismatch_panics() {
        let slots = test_slots_from_fields(vec![("p", PortFieldKind::Scalar, Pressure::UNIT)]);

        let p_slot = find_slot(&slots, "p").unwrap();

        // Try to access "p" with Velocity dimension type (should panic)
        let _: TypedExpr<Velocity> = state_scalar_slot_dim(slots.stride, "state", "idx", p_slot);
    }

    #[test]
    #[should_panic(expected = "unit mismatch for field 'U'")]
    fn slot_based_typed_dim_state_vec2_unit_mismatch_panics() {
        let slots = test_slots_from_fields(vec![("U", PortFieldKind::Vector2, Velocity::UNIT)]);

        let u_slot = find_slot(&slots, "U").unwrap();

        // Try to access "U" with Pressure dimension type (should panic)
        let _: TypedExpr<Pressure> = state_vec2_slot_dim(slots.stride, "state", "idx", u_slot);
    }

    #[test]
    #[should_panic(expected = "field 'p' is not vec2")]
    fn slot_based_typed_dim_state_kind_mismatch_scalar_to_vec2_panics() {
        let slots = test_slots_from_fields(vec![("p", PortFieldKind::Scalar, Pressure::UNIT)]);

        let p_slot = find_slot(&slots, "p").unwrap();

        // Try to access scalar "p" as vec2 (should panic)
        let _: TypedExpr<Pressure> = state_vec2_slot_dim(slots.stride, "state", "idx", p_slot);
    }

    #[test]
    #[should_panic(expected = "field 'U' is not scalar")]
    fn slot_based_typed_dim_state_kind_mismatch_vec2_to_scalar_panics() {
        let slots = test_slots_from_fields(vec![("U", PortFieldKind::Vector2, Velocity::UNIT)]);

        let u_slot = find_slot(&slots, "U").unwrap();

        // Try to access vec2 "U" as scalar (should panic)
        let _: TypedExpr<Velocity> = state_scalar_slot_dim(slots.stride, "state", "idx", u_slot);
    }

    #[test]
    fn slot_based_typed_kind_mismatch_panics() {
        let slots = test_slots_from_fields(vec![("p", PortFieldKind::Scalar, Pressure::UNIT)]);

        let p_slot = find_slot(&slots, "p").unwrap();

        // Try to access scalar "p" as vec2 (should panic)
        let result = std::panic::catch_unwind(|| {
            let _ = state_vec2_slot_typed(slots.stride, "state", "idx", p_slot);
        });
        assert!(result.is_err());
    }

    #[test]
    fn find_slot_locates_correct_slot() {
        let slots = test_slots_from_fields(vec![
            ("rho", PortFieldKind::Scalar, Density::UNIT),
            ("rho_u", PortFieldKind::Vector2, MomentumDensity::UNIT),
        ]);

        // Direct field access
        assert!(find_slot(&slots, "rho").is_some());
        assert!(find_slot(&slots, "rho_u").is_some());
        assert_eq!(find_slot(&slots, "rho").unwrap().base_offset, 0);
        assert_eq!(find_slot(&slots, "rho_u").unwrap().base_offset, 1);

        // Invalid access
        assert!(find_slot(&slots, "nonexistent").is_none());
    }
}
