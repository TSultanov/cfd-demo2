use crate::solver::codegen::state_access::resolve_state_offset_by_name;
use crate::solver::codegen::wgsl_ast::Expr;
use crate::solver::ir::ports::ResolvedStateSlotsSpec;
use crate::solver::shared::PrimitiveExpr;

pub fn lower_primitive_expr(
    expr: &PrimitiveExpr,
    slots: &ResolvedStateSlotsSpec,
    cell_idx: Expr,
    state_array: &str,
) -> Expr {
    match expr {
        PrimitiveExpr::Literal(val) => Expr::lit_f32(*val),

        PrimitiveExpr::Field(name) => {
            let offset = resolve_state_offset_by_name(slots, name).unwrap_or_else(|| {
                panic!("primitive field '{}' not found in resolved state slots", name)
            });
            let stride = slots.stride;
            Expr::ident(state_array).index(cell_idx * stride + offset)
        }

        PrimitiveExpr::Add(lhs, rhs) => {
            lower_primitive_expr(lhs, slots, cell_idx.clone(), state_array)
                + lower_primitive_expr(rhs, slots, cell_idx, state_array)
        }

        PrimitiveExpr::Sub(lhs, rhs) => {
            lower_primitive_expr(lhs, slots, cell_idx.clone(), state_array)
                - lower_primitive_expr(rhs, slots, cell_idx, state_array)
        }

        PrimitiveExpr::Mul(lhs, rhs) => {
            lower_primitive_expr(lhs, slots, cell_idx.clone(), state_array)
                * lower_primitive_expr(rhs, slots, cell_idx, state_array)
        }

        PrimitiveExpr::Div(lhs, rhs) => {
            lower_primitive_expr(lhs, slots, cell_idx.clone(), state_array)
                / lower_primitive_expr(rhs, slots, cell_idx, state_array)
        }

        PrimitiveExpr::Sqrt(inner) => {
            let inner_expr = lower_primitive_expr(inner, slots, cell_idx, state_array);
            Expr::call_named("sqrt", vec![inner_expr])
        }

        PrimitiveExpr::Neg(inner) => {
            -lower_primitive_expr(inner, slots, cell_idx, state_array)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::ir::ports::{PortFieldKind, ResolvedStateSlotSpec, ResolvedStateSlotsSpec};
    use crate::solver::units::si;

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
            ("rho", PortFieldKind::Scalar, si::DENSITY),
            ("rho_u", PortFieldKind::Vector2, si::MOMENTUM_DENSITY),
        ]);

        // (rho_u_x / rho) + 1
        let expr = PrimitiveExpr::Add(
            Box::new(PrimitiveExpr::Div(
                Box::new(PrimitiveExpr::field("rho_u_x")),
                Box::new(PrimitiveExpr::field("rho")),
            )),
            Box::new(PrimitiveExpr::lit(1.0)),
        );

        let cell_idx = Expr::ident("i");
        let wgsl = lower_primitive_expr(&expr, &slots, cell_idx, "state").to_string();
        assert!(wgsl.contains("state[i * 3u + 0u]"));
        assert!(wgsl.contains("state[i * 3u + 1u]"));
        assert!(wgsl.contains("+ 1.0"));
    }
}
