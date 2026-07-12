use crate::solver::codegen::dsl::{DslType, DynExpr};
use crate::solver::codegen::wgsl_ast::Expr;
use crate::solver::ir::ports::{ResolvedStateSlotSpec, ResolvedStateSlotsSpec};
use crate::solver::units::UnitDim;

use cfd2_ir::ast::ExprNode;
use cfd2_ir::dimensions::{
    Density, DivDim, DynamicViscosity, MulDim, Pressure, Temperature, Time, UnitDimension,
};

fn constant_field_unit(name: &str) -> Option<UnitDim> {
    type GasConstant = DivDim<Pressure, MulDim<Density, Temperature>>;
    type CompressibilitySlope = DivDim<Pressure, Density>;
    match name {
        "dt" | "dt_old" | "dtau" | "time" => Some(Time::UNIT),
        "viscosity" => Some(DynamicViscosity::UNIT),
        "density" => Some(Density::UNIT),
        "eos_r" => Some(GasConstant::UNIT),
        "eos_dp_drho" | "eos_theta_ref" => Some(CompressibilitySlope::UNIT),
        "eos_p_ref" => Some(Pressure::UNIT),
        "eos_rho_ref" => Some(Density::UNIT),
        "eos_gauge_rho_ref" => Some(Density::UNIT),
        "eos_gauge_p_ref" | "eos_gauge_e_ref" | "eos_gauge_p_bias" => Some(Pressure::UNIT),
        "eos_gamma" | "eos_gm1" | "component" | "alpha_p" | "scheme"
        | "alpha_u" | "stride_x" | "time_scheme" => Some(UnitDim::dimensionless()),
        _ => None,
    }
}

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
    if let Some(slot) = slots.slots.iter().find(|s| s.name == name) {
        return Some((slot, 0));
    }

    // Parse a component suffix (_x, _y, _z).
    let (base, suffix) = name.rsplit_once('_')?;
    let component = match suffix {
        "x" => 0,
        "y" => 1,
        "z" => 2,
        _ => return None,
    };

    let slot = slots.slots.iter().find(|s| s.name == base)?;

    if component >= slot.kind.component_count() {
        return None;
    }

    Some((slot, component))
}

/// Resolve field references in an `Expr` tree, replacing bare `Ident` nodes matching
/// field names with state array index expressions (`state[idx * stride + offset]`).
///
/// This is a recursive tree walk that also produces a `DynExpr` with runtime unit tracking.
pub fn resolve_field_refs_dyn(
    expr: &Expr,
    slots: &ResolvedStateSlotsSpec,
    cell_idx: Expr,
    state_array: &str,
) -> DynExpr {
    match expr.node() {
        ExprNode::Literal(lit) => {
            // Literals are dimensionless (scalar f32)
            match lit {
                cfd2_ir::ast::Literal::Float(s) => {
                    let val: f32 = s.parse().unwrap_or(0.0);
                    DynExpr::f32(val, UnitDim::dimensionless())
                }
                cfd2_ir::ast::Literal::Int(v) => {
                    DynExpr::f32(*v as f32, UnitDim::dimensionless())
                }
                cfd2_ir::ast::Literal::Uint(v) => {
                    DynExpr::f32(*v as f32, UnitDim::dimensionless())
                }
                cfd2_ir::ast::Literal::Bool(_) => {
                    DynExpr::f32(0.0, UnitDim::dimensionless())
                }
            }
        }

        ExprNode::Ident(name) => {
            if let Some((slot, component)) = resolve_field_slot_component(slots, name) {
                let offset = slot.base_offset + component;
                let stride = slots.stride;
                let resolved = Expr::ident(state_array).index(cell_idx * stride + offset);
                DynExpr::new(resolved, DslType::f32(), slot.unit)
            } else {
                // Not a field: pass through as-is (e.g. a local variable name).
                DynExpr::new(expr.clone(), DslType::f32(), UnitDim::dimensionless())
            }
        }

        ExprNode::Binary { left, op, right } => {
            let lhs_dyn = resolve_field_refs_dyn(left, slots, cell_idx.clone(), state_array);
            let rhs_dyn = resolve_field_refs_dyn(right, slots, cell_idx, state_array);
            match op {
                cfd2_ir::ast::BinaryOp::Add => lhs_dyn + rhs_dyn,
                cfd2_ir::ast::BinaryOp::Sub => lhs_dyn - rhs_dyn,
                cfd2_ir::ast::BinaryOp::Mul => lhs_dyn * rhs_dyn,
                cfd2_ir::ast::BinaryOp::Div => lhs_dyn / rhs_dyn,
                _ => {
                    // For other binary ops, combine as dimensionless
                    let combined = Expr::binary(lhs_dyn.expr, *op, rhs_dyn.expr);
                    DynExpr::new(combined, DslType::f32(), UnitDim::dimensionless())
                }
            }
        }

        ExprNode::Unary { op, expr: inner } => {
            let inner_dyn = resolve_field_refs_dyn(inner, slots, cell_idx, state_array);
            match op {
                cfd2_ir::ast::UnaryOp::Negate => -inner_dyn,
                _ => {
                    let combined = Expr::alloc_node(ExprNode::Unary {
                        op: *op,
                        expr: inner_dyn.expr,
                    });
                    DynExpr::new(combined, inner_dyn.ty, inner_dyn.unit)
                }
            }
        }

        ExprNode::Call { callee, args } => {
            if let ExprNode::Ident(name) = callee.node() {
                if name == "sqrt" && args.len() == 1 {
                    let inner_dyn =
                        resolve_field_refs_dyn(&args[0], slots, cell_idx, state_array);
                    return inner_dyn.sqrt().expect("sqrt operation failed");
                }
                if name == "abs" && args.len() == 1 {
                    let inner_dyn =
                        resolve_field_refs_dyn(&args[0], slots, cell_idx, state_array);
                    let combined = Expr::call_named(name, vec![inner_dyn.expr]);
                    return DynExpr::new(combined, inner_dyn.ty, inner_dyn.unit);
                }
                // `max`/`min` PRESERVE units: clamping a quantity to a floor/ceiling of
                // the SAME unit yields that unit (unlike the dimensionless fallback below,
                // which would reject e.g. an EOS density floor `max(rho, rho_floor)`).
                // Both operands must share a unit — but a dimensionless operand (a bare
                // literal, e.g. `max(x, 0.0)`) adopts the other's unit, so a literal floor
                // of zero still type-checks.
                if (name == "max" || name == "min") && args.len() == 2 {
                    let lhs = resolve_field_refs_dyn(&args[0], slots, cell_idx.clone(), state_array);
                    let rhs = resolve_field_refs_dyn(&args[1], slots, cell_idx, state_array);
                    let unit = if lhs.unit == rhs.unit {
                        lhs.unit.clone()
                    } else if lhs.unit == UnitDim::dimensionless() {
                        rhs.unit.clone()
                    } else if rhs.unit == UnitDim::dimensionless() {
                        lhs.unit.clone()
                    } else {
                        panic!(
                            "{name}() operands must share a unit (got {:?} vs {:?})",
                            lhs.unit, rhs.unit
                        );
                    };
                    let combined = Expr::call_named(name, vec![lhs.expr, rhs.expr]);
                    return DynExpr::new(combined, lhs.ty, unit);
                }
            }
            // For other calls, resolve args but treat as dimensionless
            let resolved_args: Vec<Expr> = args
                .iter()
                .map(|a| resolve_field_refs_dyn(a, slots, cell_idx.clone(), state_array).expr)
                .collect();
            let callee_resolved =
                resolve_field_refs_dyn(callee, slots, cell_idx, state_array).expr;
            let combined = Expr::call(callee_resolved, resolved_args);
            DynExpr::new(combined, DslType::f32(), UnitDim::dimensionless())
        }

        ExprNode::Field { base, field } => {
            if matches!(base.node(), ExprNode::Ident(name) if name == "constants") {
                if let Some(unit) = constant_field_unit(field) {
                    return DynExpr::new(expr.clone(), DslType::f32(), unit);
                }
            }
            let base_dyn = resolve_field_refs_dyn(base, slots, cell_idx, state_array);
            let combined = base_dyn.expr.field(field.clone());
            DynExpr::new(combined, base_dyn.ty, base_dyn.unit)
        }

        ExprNode::Index { base, index } => {
            let base_dyn = resolve_field_refs_dyn(base, slots, cell_idx.clone(), state_array);
            let index_dyn = resolve_field_refs_dyn(index, slots, cell_idx, state_array);
            let combined = base_dyn.expr.index(index_dyn.expr);
            DynExpr::new(combined, base_dyn.ty, base_dyn.unit)
        }
    }
}

/// Resolve field references in an `Expr` tree, replacing bare `Ident` nodes matching
/// field names with state array index expressions.
///
/// This is the simple API that discards unit tracking. For unit-aware resolution,
/// use `resolve_field_refs_dyn`.
pub fn resolve_field_refs(
    expr: &Expr,
    slots: &ResolvedStateSlotsSpec,
    cell_idx: Expr,
    state_array: &str,
) -> Expr {
    resolve_field_refs_dyn(expr, slots, cell_idx, state_array).expr
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::ir::ports::{PortFieldKind, ResolvedStateSlotSpec, ResolvedStateSlotsSpec};
    use cfd2_ir::dimensions::{
        Area, Density, Length, MomentumDensity, Pressure, UnitDimension, Velocity,
    };

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
    fn expr_resolves_field_access_and_ops() {
        let slots = test_slots_from_fields(vec![
            ("rho", PortFieldKind::Scalar, Density::UNIT),
            ("rho_u", PortFieldKind::Vector2, MomentumDensity::UNIT),
        ]);

        let expr = Expr::ident("rho_u_x") / Expr::ident("rho");

        let cell_idx = Expr::ident("i");
        let wgsl = resolve_field_refs(&expr, &slots, cell_idx, "state").to_string();
        assert!(wgsl.contains("state[i * 3u + 0u]"));
        assert!(wgsl.contains("state[i * 3u + 1u]"));
        assert!(wgsl.contains("/"));
    }

    #[test]
    fn expr_dyn_tracks_units() {
        let slots = test_slots_from_fields(vec![
            ("rho", PortFieldKind::Scalar, Density::UNIT),
            ("rho_u", PortFieldKind::Vector2, MomentumDensity::UNIT),
        ]);

        // rho_u_x / rho produces velocity units
        let expr = Expr::ident("rho_u_x") / Expr::ident("rho");

        let cell_idx = Expr::ident("i");
        let dyn_expr = resolve_field_refs_dyn(&expr, &slots, cell_idx, "state");

        assert!(dyn_expr.expr.to_string().contains("state[i * 3u + 1u]"));
        assert!(dyn_expr.expr.to_string().contains("state[i * 3u + 0u]"));

        // momentum_density / density = velocity
        assert_eq!(dyn_expr.unit, Velocity::UNIT);
        assert_eq!(dyn_expr.ty, DslType::f32());
    }

    #[test]
    fn expr_dyn_tracks_component_units() {
        let slots = test_slots_from_fields(vec![("U", PortFieldKind::Vector2, Velocity::UNIT)]);

        let expr = Expr::ident("U_x");

        let cell_idx = Expr::ident("i");
        let dyn_expr = resolve_field_refs_dyn(&expr, &slots, cell_idx, "state");

        assert_eq!(dyn_expr.unit, Velocity::UNIT);
        assert_eq!(dyn_expr.expr.to_string(), "state[i * 2u + 0u]");
    }

    #[test]
    fn expr_dyn_literal_is_dimensionless() {
        let slots = test_slots_from_fields(vec![]);

        let expr = Expr::lit_f32(3.25);
        let cell_idx = Expr::ident("i");
        let dyn_expr = resolve_field_refs_dyn(&expr, &slots, cell_idx, "state");

        assert_eq!(dyn_expr.unit, UnitDim::dimensionless());
        assert_eq!(dyn_expr.expr.to_string(), "3.25");
    }

    #[test]
    fn expr_dyn_mul_combines_units() {
        let slots = test_slots_from_fields(vec![
            ("rho", PortFieldKind::Scalar, Density::UNIT),
            ("U", PortFieldKind::Vector2, Velocity::UNIT),
        ]);

        // rho * U_x produces momentum density
        let expr = Expr::ident("rho") * Expr::ident("U_x");

        let cell_idx = Expr::ident("i");
        let dyn_expr = resolve_field_refs_dyn(&expr, &slots, cell_idx, "state");

        assert_eq!(dyn_expr.unit, MomentumDensity::UNIT);
    }

    #[test]
    fn expr_dyn_max_preserves_units() {
        // max(rho, rho_floor) must stay Density (not collapse to dimensionless) so an
        // on-device EOS density floor type-checks. Both operands are Density fields.
        let slots = test_slots_from_fields(vec![
            ("rho", PortFieldKind::Scalar, Density::UNIT),
            ("rho_floor", PortFieldKind::Scalar, Density::UNIT),
        ]);
        let expr = Expr::call_named("max", vec![Expr::ident("rho"), Expr::ident("rho_floor")]);
        let dyn_expr = resolve_field_refs_dyn(&expr, &slots, Expr::ident("i"), "state");
        assert_eq!(dyn_expr.unit, Density::UNIT);
        assert!(dyn_expr.expr.to_string().starts_with("max("));
    }

    #[test]
    fn expr_dyn_max_with_dimensionless_literal_adopts_field_unit() {
        // A bare literal floor (dimensionless) adopts the field's unit.
        let slots = test_slots_from_fields(vec![("rho", PortFieldKind::Scalar, Density::UNIT)]);
        let expr = Expr::call_named("max", vec![Expr::ident("rho"), Expr::lit_f32(0.0)]);
        let dyn_expr = resolve_field_refs_dyn(&expr, &slots, Expr::ident("i"), "state");
        assert_eq!(dyn_expr.unit, Density::UNIT);
    }

    #[test]
    fn expr_dyn_sqrt_applies_sqrt_to_units() {
        let slots = test_slots_from_fields(vec![("area", PortFieldKind::Scalar, Area::UNIT)]);

        // sqrt(area) produces length
        let expr = Expr::ident("area").sqrt();

        let cell_idx = Expr::ident("i");
        let dyn_expr = resolve_field_refs_dyn(&expr, &slots, cell_idx, "state");

        assert_eq!(dyn_expr.unit, Length::UNIT);
        assert!(dyn_expr.expr.to_string().contains("sqrt"));
    }

    #[test]
    #[should_panic(expected = "typed add failed")]
    fn expr_dyn_panics_on_unit_mismatch_add() {
        let slots = test_slots_from_fields(vec![
            ("rho", PortFieldKind::Scalar, Density::UNIT),
            ("U", PortFieldKind::Scalar, Velocity::UNIT),
        ]);

        // rho + U is a unit mismatch (density + velocity)
        let expr = Expr::ident("rho") + Expr::ident("U");

        let cell_idx = Expr::ident("i");
        let _ = resolve_field_refs_dyn(&expr, &slots, cell_idx, "state");
    }

    #[test]
    #[should_panic(expected = "typed sub failed")]
    fn expr_dyn_panics_on_unit_mismatch_sub() {
        let slots = test_slots_from_fields(vec![
            ("p", PortFieldKind::Scalar, Pressure::UNIT),
            ("rho", PortFieldKind::Scalar, Density::UNIT),
        ]);

        // p - rho is a unit mismatch (pressure - density)
        let expr = Expr::ident("p") - Expr::ident("rho");

        let cell_idx = Expr::ident("i");
        let _ = resolve_field_refs_dyn(&expr, &slots, cell_idx, "state");
    }
}
