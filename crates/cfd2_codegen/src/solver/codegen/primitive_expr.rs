use crate::solver::codegen::dsl::{DynExpr, DslType};
use crate::solver::codegen::wgsl_ast::Expr;
use crate::solver::ir::ports::{ResolvedStateSlotSpec, ResolvedStateSlotsSpec};
use crate::solver::shared::PrimitiveExpr;
use crate::solver::units::UnitDim;

/// Resolve a field name to a slot and component index.
///
/// Supports:
/// - Direct field matches: "rho" -> (slot, 0)
/// - Component suffixes: "rho_u_x" -> (slot, 0), "rho_u_y" -> (slot, 1), etc.
///
/// Returns `None` if the field is not found or the component is out of range.
fn resolve_field_slot_component<'a>(
    slots: &'a ResolvedStateSlotsSpec,
    name: &str,
) -> Option<(&'a ResolvedStateSlotSpec, u32)> {
    // First, try direct field lookup
    if let Some(slot) = slots.slots.iter().find(|s| s.name == name) {
        return Some((slot, 0));
    }

    // Try to parse component suffix (_x, _y, _z)
    let (base, suffix) = name.rsplit_once('_')?;
    let component = match suffix {
        "x" => 0,
        "y" => 1,
        "z" => 2,
        _ => return None,
    };

    let slot = slots.slots.iter().find(|s| s.name == base)?;

    // Validate component is within range for this slot's kind
    if component >= slot.kind.component_count() {
        return None;
    }

    Some((slot, component))
}

/// Lower a PrimitiveExpr to a DynExpr with runtime unit checking.
///
/// This function provides unit consistency checking for primitive expression lowering.
/// Operations like `+` and `-` will panic at runtime if the operands have mismatched units.
pub fn lower_primitive_expr_dyn(
    expr: &PrimitiveExpr,
    slots: &ResolvedStateSlotsSpec,
    cell_idx: Expr,
    state_array: &str,
) -> DynExpr {
    match expr {
        PrimitiveExpr::Literal(val) => {
            // Literals are dimensionless (scalar f32)
            DynExpr::f32(*val, UnitDim::dimensionless())
        }

        PrimitiveExpr::Field(name) => {
            let (slot, component) = resolve_field_slot_component(slots, name).unwrap_or_else(|| {
                panic!("primitive field '{}' not found in resolved state slots", name)
            });
            let offset = slot.base_offset + component;
            let stride = slots.stride;
            let expr = Expr::ident(state_array).index(cell_idx * stride + offset);
            DynExpr::new(expr, DslType::f32(), slot.unit)
        }

        PrimitiveExpr::Add(lhs, rhs) => {
            let lhs_dyn = lower_primitive_expr_dyn(lhs, slots, cell_idx.clone(), state_array);
            let rhs_dyn = lower_primitive_expr_dyn(rhs, slots, cell_idx, state_array);
            lhs_dyn + rhs_dyn
        }

        PrimitiveExpr::Sub(lhs, rhs) => {
            let lhs_dyn = lower_primitive_expr_dyn(lhs, slots, cell_idx.clone(), state_array);
            let rhs_dyn = lower_primitive_expr_dyn(rhs, slots, cell_idx, state_array);
            lhs_dyn - rhs_dyn
        }

        PrimitiveExpr::Mul(lhs, rhs) => {
            let lhs_dyn = lower_primitive_expr_dyn(lhs, slots, cell_idx.clone(), state_array);
            let rhs_dyn = lower_primitive_expr_dyn(rhs, slots, cell_idx, state_array);
            lhs_dyn * rhs_dyn
        }

        PrimitiveExpr::Div(lhs, rhs) => {
            let lhs_dyn = lower_primitive_expr_dyn(lhs, slots, cell_idx.clone(), state_array);
            let rhs_dyn = lower_primitive_expr_dyn(rhs, slots, cell_idx, state_array);
            lhs_dyn / rhs_dyn
        }

        PrimitiveExpr::Sqrt(inner) => {
            let inner_dyn = lower_primitive_expr_dyn(inner, slots, cell_idx, state_array);
            inner_dyn.sqrt().expect("sqrt operation failed")
        }

        PrimitiveExpr::Neg(inner) => {
            let inner_dyn = lower_primitive_expr_dyn(inner, slots, cell_idx, state_array);
            -inner_dyn
        }
    }
}

/// Lower a PrimitiveExpr to a WGSL Expr.
///
/// This is the legacy API that delegates to `lower_primitive_expr_dyn` and extracts
/// the underlying expression. Unit checking is performed during lowering and will
/// panic on mismatches.
pub fn lower_primitive_expr(
    expr: &PrimitiveExpr,
    slots: &ResolvedStateSlotsSpec,
    cell_idx: Expr,
    state_array: &str,
) -> Expr {
    lower_primitive_expr_dyn(expr, slots, cell_idx, state_array).expr
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::ir::ports::{PortFieldKind, ResolvedStateSlotSpec, ResolvedStateSlotsSpec};
    use cfd2_ir::solver::dimensions::{Area, Density, Length, MomentumDensity, Pressure, UnitDimension, Velocity};

    /// Helper to create a ResolvedStateSlotsSpec for testing.
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
    fn primitive_expr_lowers_field_access_and_ops() {
        let slots = test_slots_from_fields(vec![
            ("rho", PortFieldKind::Scalar, Density::UNIT),
            ("rho_u", PortFieldKind::Vector2, MomentumDensity::UNIT),
        ]);

        // (rho_u_x / rho) - a dimensionally consistent expression
        // momentum_density / density = velocity
        let expr = PrimitiveExpr::Div(
            Box::new(PrimitiveExpr::field("rho_u_x")),
            Box::new(PrimitiveExpr::field("rho")),
        );

        let cell_idx = Expr::ident("i");
        let wgsl = lower_primitive_expr(&expr, &slots, cell_idx, "state").to_string();
        assert!(wgsl.contains("state[i * 3u + 0u]"));
        assert!(wgsl.contains("state[i * 3u + 1u]"));
        assert!(wgsl.contains("/"));
    }

    #[test]
    fn primitive_expr_dyn_tracks_units() {
        let slots = test_slots_from_fields(vec![
            ("rho", PortFieldKind::Scalar, Density::UNIT),
            ("rho_u", PortFieldKind::Vector2, MomentumDensity::UNIT),
        ]);

        // rho_u_x / rho produces velocity units
        let expr = PrimitiveExpr::Div(
            Box::new(PrimitiveExpr::field("rho_u_x")),
            Box::new(PrimitiveExpr::field("rho")),
        );

        let cell_idx = Expr::ident("i");
        let dyn_expr = lower_primitive_expr_dyn(&expr, &slots, cell_idx, "state");

        // Verify the expression produces the expected WGSL
        assert!(dyn_expr.expr.to_string().contains("state[i * 3u + 1u]"));
        assert!(dyn_expr.expr.to_string().contains("state[i * 3u + 0u]"));

        // Verify the resulting unit is velocity (momentum_density / density = velocity)
        assert_eq!(dyn_expr.unit, Velocity::UNIT);
        assert_eq!(dyn_expr.ty, DslType::f32());
    }

    #[test]
    fn primitive_expr_dyn_tracks_component_units() {
        let slots = test_slots_from_fields(vec![
            ("U", PortFieldKind::Vector2, Velocity::UNIT),
        ]);

        // Access U_x component - should have velocity units
        let expr = PrimitiveExpr::field("U_x");

        let cell_idx = Expr::ident("i");
        let dyn_expr = lower_primitive_expr_dyn(&expr, &slots, cell_idx, "state");

        assert_eq!(dyn_expr.unit, Velocity::UNIT);
        assert_eq!(dyn_expr.expr.to_string(), "state[i * 2u + 0u]");
    }

    #[test]
    fn primitive_expr_dyn_literal_is_dimensionless() {
        let slots = test_slots_from_fields(vec![]);

        let expr = PrimitiveExpr::lit(3.25);
        let cell_idx = Expr::ident("i");
        let dyn_expr = lower_primitive_expr_dyn(&expr, &slots, cell_idx, "state");

        assert_eq!(dyn_expr.unit, UnitDim::dimensionless());
        assert_eq!(dyn_expr.expr.to_string(), "3.25");
    }

    #[test]
    fn primitive_expr_dyn_mul_combines_units() {
        let slots = test_slots_from_fields(vec![
            ("rho", PortFieldKind::Scalar, Density::UNIT),
            ("U", PortFieldKind::Vector2, Velocity::UNIT),
        ]);

        // rho * U_x produces momentum density
        let expr = PrimitiveExpr::Mul(
            Box::new(PrimitiveExpr::field("rho")),
            Box::new(PrimitiveExpr::field("U_x")),
        );

        let cell_idx = Expr::ident("i");
        let dyn_expr = lower_primitive_expr_dyn(&expr, &slots, cell_idx, "state");

        assert_eq!(dyn_expr.unit, MomentumDensity::UNIT);
    }

    #[test]
    fn primitive_expr_dyn_sqrt_applies_sqrt_to_units() {
        let slots = test_slots_from_fields(vec![
            ("area", PortFieldKind::Scalar, Area::UNIT),
        ]);

        // sqrt(area) produces length
        let expr = PrimitiveExpr::Sqrt(Box::new(PrimitiveExpr::field("area")));

        let cell_idx = Expr::ident("i");
        let dyn_expr = lower_primitive_expr_dyn(&expr, &slots, cell_idx, "state");

        assert_eq!(dyn_expr.unit, Length::UNIT);
        assert!(dyn_expr.expr.to_string().contains("sqrt"));
    }

    #[test]
    #[should_panic(expected = "typed add failed")]
    fn primitive_expr_dyn_panics_on_unit_mismatch_add() {
        let slots = test_slots_from_fields(vec![
            ("rho", PortFieldKind::Scalar, Density::UNIT),
            ("U", PortFieldKind::Scalar, Velocity::UNIT),
        ]);

        // rho + U is a unit mismatch (density + velocity)
        let expr = PrimitiveExpr::Add(
            Box::new(PrimitiveExpr::field("rho")),
            Box::new(PrimitiveExpr::field("U")),
        );

        let cell_idx = Expr::ident("i");
        // This should panic due to unit mismatch
        let _ = lower_primitive_expr_dyn(&expr, &slots, cell_idx, "state");
    }

    #[test]
    #[should_panic(expected = "typed sub failed")]
    fn primitive_expr_dyn_panics_on_unit_mismatch_sub() {
        let slots = test_slots_from_fields(vec![
            ("p", PortFieldKind::Scalar, Pressure::UNIT),
            ("rho", PortFieldKind::Scalar, Density::UNIT),
        ]);

        // p - rho is a unit mismatch (pressure - density)
        let expr = PrimitiveExpr::Sub(
            Box::new(PrimitiveExpr::field("p")),
            Box::new(PrimitiveExpr::field("rho")),
        );

        let cell_idx = Expr::ident("i");
        // This should panic due to unit mismatch
        let _ = lower_primitive_expr_dyn(&expr, &slots, cell_idx, "state");
    }
}
