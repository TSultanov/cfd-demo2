//! High-level primitive expression language for model-defined derived primitives.
//!
//! Models define primitive derivations using this algebra.
//!
//! Note: this type is intentionally WGSL/AST-agnostic. Lowering to WGSL is performed
//! by the codegen layer to avoid coupling shared/model code to a specific WGSL AST arena.

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
    /// Helper: create a field reference
    pub fn field(name: impl Into<String>) -> Self {
        PrimitiveExpr::Field(name.into())
    }

    /// Helper: create a literal
    pub fn lit(val: f32) -> Self {
        PrimitiveExpr::Literal(val)
    }
}
