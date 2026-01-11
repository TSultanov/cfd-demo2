/// High-level primitive expression language for model-defined derived primitives.
///
/// This bridges the model specification (physics-aware) to the WGSL AST (physics-agnostic).
/// Models define primitive derivations using this algebra, and codegen lowers them to
/// WGSL expressions using state layout information.

use super::wgsl_ast::Expr as WgslExpr;

/// Primitive expression over conserved state fields.
///
/// Example: pressure in ideal gas Euler
/// ```ignore
/// PrimitiveExpr::Mul(
///     Box::new(PrimitiveExpr::Literal(0.4)),  // gamma - 1
///     Box::new(PrimitiveExpr::Sub(
///         Box::new(PrimitiveExpr::Field("rho_e".into())),
///         Box::new(/* 0.5 * rho * u^2 */),
///     )),
/// )
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum PrimitiveExpr {
    /// Constant literal value
    Literal(f32),

    /// Reference to a state field by name (e.g., "rho", "rho_u_x")
    Field(String),

    /// Binary operations
    Add(Box<PrimitiveExpr>, Box<PrimitiveExpr>),
    Sub(Box<PrimitiveExpr>, Box<PrimitiveExpr>),
    Mul(Box<PrimitiveExpr>, Box<PrimitiveExpr>),
    Div(Box<PrimitiveExpr>, Box<PrimitiveExpr>),

    /// Unary operations
    Sqrt(Box<PrimitiveExpr>),
    Neg(Box<PrimitiveExpr>),

    // Future: Max, Min, Clamp, Select, etc.
}

impl PrimitiveExpr {
    /// Lower this expression to a WGSL Expr.
    ///
    /// Requires:
    /// - `state_layout`: mapping from field names to buffer offsets
    /// - `cell_idx`: WGSL expression representing the cell index (e.g., `Expr::ident("cell_id")`)
    /// - `state_array`: name of the state buffer binding (typically "state")
    ///
    /// Returns: WGSL expression that computes this primitive value for the given cell.
    ///
    /// Example:
    /// ```ignore
    /// PrimitiveExpr::Field("rho".into()).to_wgsl_expr(layout, cell_idx, "state")
    /// // → Expr::ident("state").index(cell_idx * stride + offset_for_rho)
    /// ```
    pub fn to_wgsl_expr(
        &self,
        state_layout: &crate::solver::model::backend::StateLayout,
        cell_idx: WgslExpr,
        state_array: &str,
    ) -> WgslExpr {
        match self {
            PrimitiveExpr::Literal(val) => WgslExpr::lit_f32(*val),

            PrimitiveExpr::Field(name) => {
                // Look up field offset in state layout
                let offset = state_layout
                    .offset_for(name)
                    .unwrap_or_else(|| panic!("primitive field '{}' not found in state layout", name));
                let stride = state_layout.stride();

                // state[cell_idx * stride + offset]
                WgslExpr::ident(state_array).index(cell_idx * stride + offset)
            }

            PrimitiveExpr::Add(lhs, rhs) => {
                let lhs_expr = lhs.to_wgsl_expr(state_layout, cell_idx, state_array);
                let rhs_expr = rhs.to_wgsl_expr(state_layout, cell_idx, state_array);
                lhs_expr + rhs_expr
            }

            PrimitiveExpr::Sub(lhs, rhs) => {
                let lhs_expr = lhs.to_wgsl_expr(state_layout, cell_idx, state_array);
                let rhs_expr = rhs.to_wgsl_expr(state_layout, cell_idx, state_array);
                lhs_expr - rhs_expr
            }

            PrimitiveExpr::Mul(lhs, rhs) => {
                let lhs_expr = lhs.to_wgsl_expr(state_layout, cell_idx, state_array);
                let rhs_expr = rhs.to_wgsl_expr(state_layout, cell_idx, state_array);
                lhs_expr * rhs_expr
            }

            PrimitiveExpr::Div(lhs, rhs) => {
                let lhs_expr = lhs.to_wgsl_expr(state_layout, cell_idx, state_array);
                let rhs_expr = rhs.to_wgsl_expr(state_layout, cell_idx, state_array);
                lhs_expr / rhs_expr
            }

            PrimitiveExpr::Sqrt(inner) => {
                let inner_expr = inner.to_wgsl_expr(state_layout, cell_idx, state_array);
                WgslExpr::call_named("sqrt", vec![inner_expr])
            }

            PrimitiveExpr::Neg(inner) => {
                let inner_expr = inner.to_wgsl_expr(state_layout, cell_idx, state_array);
                -inner_expr
            }
        }
    }

    /// Helper: create a field reference
    pub fn field(name: impl Into<String>) -> Self {
        PrimitiveExpr::Field(name.into())
    }

    /// Helper: create a literal
    pub fn lit(val: f32) -> Self {
        PrimitiveExpr::Literal(val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::model::backend::{ast::vol_scalar, StateLayout};
    use crate::solver::units::si;

    #[test]
    fn primitive_expr_lowers_to_wgsl() {
        let rho = vol_scalar("rho", si::DENSITY);
        let rho_e = vol_scalar("rho_e", si::ENERGY_DENSITY);
        let layout = StateLayout::new(vec![rho, rho_e]);

        let cell_idx = WgslExpr::ident("i");

        // rho_e / rho
        let expr = PrimitiveExpr::Div(
            Box::new(PrimitiveExpr::field("rho_e")),
            Box::new(PrimitiveExpr::field("rho")),
        );

        let wgsl = expr.to_wgsl_expr(&layout, cell_idx, "state");
        let rendered = wgsl.to_string();

        // Should be: state[i * 2 + 1] / state[i * 2 + 0]
        assert!(rendered.contains("state[i * 2u + 1u]"));
        assert!(rendered.contains("state[i * 2u + 0u]"));
    }

    #[test]
    fn primitive_expr_handles_literals_and_ops() {
        let rho = vol_scalar("rho", si::DENSITY);
        let layout = StateLayout::new(vec![rho]);
        let cell_idx = WgslExpr::ident("idx");

        // 2.0 * rho + 1.0
        let expr = PrimitiveExpr::Add(
            Box::new(PrimitiveExpr::Mul(
                Box::new(PrimitiveExpr::lit(2.0)),
                Box::new(PrimitiveExpr::field("rho")),
            )),
            Box::new(PrimitiveExpr::lit(1.0)),
        );

        let wgsl = expr.to_wgsl_expr(&layout, cell_idx, "state");
        let rendered = wgsl.to_string();

        assert!(rendered.contains("2.0"));
        assert!(rendered.contains("1.0"));
        assert!(rendered.contains("state[idx"));
    }
}
