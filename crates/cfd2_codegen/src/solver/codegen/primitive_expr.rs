use crate::solver::codegen::wgsl_ast::Expr;
use crate::solver::ir::StateLayout;
use crate::solver::shared::PrimitiveExpr;

pub fn lower_primitive_expr(
    expr: &PrimitiveExpr,
    state_layout: &StateLayout,
    cell_idx: Expr,
    state_array: &str,
) -> Expr {
    match expr {
        PrimitiveExpr::Literal(val) => Expr::lit_f32(*val),

        PrimitiveExpr::Field(name) => {
            let offset = resolve_state_offset(state_layout, name).unwrap_or_else(|| {
                panic!("primitive field '{}' not found in state layout", name)
            });
            let stride = state_layout.stride();
            Expr::ident(state_array).index(cell_idx * stride + offset)
        }

        PrimitiveExpr::Add(lhs, rhs) => {
            lower_primitive_expr(lhs, state_layout, cell_idx, state_array)
                + lower_primitive_expr(rhs, state_layout, cell_idx, state_array)
        }

        PrimitiveExpr::Sub(lhs, rhs) => {
            lower_primitive_expr(lhs, state_layout, cell_idx, state_array)
                - lower_primitive_expr(rhs, state_layout, cell_idx, state_array)
        }

        PrimitiveExpr::Mul(lhs, rhs) => {
            lower_primitive_expr(lhs, state_layout, cell_idx, state_array)
                * lower_primitive_expr(rhs, state_layout, cell_idx, state_array)
        }

        PrimitiveExpr::Div(lhs, rhs) => {
            lower_primitive_expr(lhs, state_layout, cell_idx, state_array)
                / lower_primitive_expr(rhs, state_layout, cell_idx, state_array)
        }

        PrimitiveExpr::Sqrt(inner) => {
            let inner_expr = lower_primitive_expr(inner, state_layout, cell_idx, state_array);
            Expr::call_named("sqrt", vec![inner_expr])
        }

        PrimitiveExpr::Neg(inner) => -lower_primitive_expr(inner, state_layout, cell_idx, state_array),
    }
}

fn resolve_state_offset(state_layout: &StateLayout, name: &str) -> Option<u32> {
    if let Some(offset) = state_layout.offset_for(name) {
        return Some(offset);
    }

    let (base, component) = name.rsplit_once('_')?;
    let component = match component {
        "x" => 0,
        "y" => 1,
        "z" => 2,
        _ => return None,
    };
    state_layout.component_offset(base, component)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::ir::{vol_scalar, vol_vector};
    use crate::solver::units::si;

    #[test]
    fn primitive_expr_lowers_field_access_and_ops() {
        let rho = vol_scalar("rho", si::DENSITY);
        let rho_u = vol_vector("rho_u", si::MOMENTUM_DENSITY);
        let layout = StateLayout::new(vec![rho, rho_u]);

        // (rho_u_x / rho) + 1
        let expr = PrimitiveExpr::Add(
            Box::new(PrimitiveExpr::Div(
                Box::new(PrimitiveExpr::field("rho_u_x")),
                Box::new(PrimitiveExpr::field("rho")),
            )),
            Box::new(PrimitiveExpr::lit(1.0)),
        );

        let cell_idx = Expr::ident("i");
        let wgsl = lower_primitive_expr(&expr, &layout, cell_idx, "state").to_string();
        assert!(wgsl.contains("state[i * 3u + 0u]"));
        assert!(wgsl.contains("state[i * 3u + 1u]"));
        assert!(wgsl.contains("+ 1.0"));
    }
}

