use crate::solver::codegen::wgsl_ast::{AssignOp, BinaryOp, Expr, Literal, Stmt};

use super::matrix::BlockCsrSoaEntry;

#[derive(Debug, Clone, PartialEq)]
pub struct VecExpr<const N: usize> {
    expr: Expr,
}

impl<const N: usize> VecExpr<N> {
    fn constructor_name() -> &'static str {
        match N {
            2 => "vec2<f32>",
            3 => "vec3<f32>",
            4 => "vec4<f32>",
            _ => panic!("unsupported vector size {N}; expected 2/3/4"),
        }
    }

    pub fn from_expr(expr: Expr) -> Self {
        Self { expr }
    }

    pub fn expr(&self) -> Expr {
        self.expr.clone()
    }

    pub fn from_components(components: [Expr; N]) -> Self {
        Self::from_expr(Expr::call_named(Self::constructor_name(), components.into()))
    }

    pub fn zeros() -> Self {
        Self::from_components(std::array::from_fn(|_| Expr::lit_f32(0.0)))
    }

    pub fn add(&self, rhs: &Self) -> Self {
        Self::from_expr(Expr::binary(self.expr(), BinaryOp::Add, rhs.expr()))
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        Self::from_expr(Expr::binary(self.expr(), BinaryOp::Sub, rhs.expr()))
    }

    pub fn neg(&self) -> Self {
        Self::from_expr(Expr::unary(crate::solver::codegen::wgsl_ast::UnaryOp::Negate, self.expr()))
    }

    pub fn mul_scalar(&self, scalar: Expr) -> Self {
        Self::from_expr(Expr::binary(self.expr(), BinaryOp::Mul, scalar))
    }

    pub fn div_scalar(&self, scalar: Expr) -> Self {
        Self::from_expr(Expr::binary(self.expr(), BinaryOp::Div, scalar))
    }

    pub fn dot(&self, rhs: &Self) -> Expr {
        Expr::call_named("dot", vec![self.expr(), rhs.expr()])
    }
}

impl VecExpr<2> {
    pub fn from_xy_fields(value: Expr) -> Self {
        Self::from_components([value.clone().field("x"), value.field("y")])
    }

    pub fn to_vector2_struct(&self) -> Expr {
        Expr::call_named(
            "Vector2",
            vec![self.expr().field("x"), self.expr().field("y")],
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatExpr<const R: usize, const C: usize> {
    entries: [[Expr; C]; R],
}

impl<const R: usize, const C: usize> MatExpr<R, C> {
    pub fn from_entries(entries: [[Expr; C]; R]) -> Self {
        Self { entries }
    }

    pub fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> Expr,
    {
        let entries = std::array::from_fn(|row| std::array::from_fn(|col| f(row, col)));
        Self { entries }
    }

    pub fn from_prefix(prefix: &str) -> Self {
        Self::from_fn(|row, col| Expr::ident(format!("{prefix}_{row}{col}")))
    }

    pub fn var_prefix(prefix: &str, init: Expr) -> Vec<Stmt> {
        let mut out = Vec::new();
        for row in 0..R {
            for col in 0..C {
                out.push(Stmt::Var {
                    name: format!("{prefix}_{row}{col}"),
                    ty: None,
                    expr: Some(init.clone()),
                });
            }
        }
        out
    }

    pub fn entry(&self, row: usize, col: usize) -> Expr {
        self.entries[row][col].clone()
    }

    pub fn add(&self, rhs: &Self) -> Self {
        Self::from_fn(|row, col| {
            Expr::binary(self.entry(row, col), BinaryOp::Add, rhs.entry(row, col))
        })
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        Self::from_fn(|row, col| {
            Expr::binary(self.entry(row, col), BinaryOp::Sub, rhs.entry(row, col))
        })
    }

    pub fn mul_scalar(&self, scalar: Expr) -> Self {
        Self::from_fn(|row, col| Expr::binary(self.entry(row, col), BinaryOp::Mul, scalar.clone()))
    }

    pub fn mul_mat<const D: usize>(&self, rhs: &MatExpr<C, D>) -> MatExpr<R, D> {
        MatExpr::<R, D>::from_fn(|row, col| {
            let mut acc: Option<Expr> = None;
            for k in 0..C {
                let lhs = self.entry(row, k);
                let rhs_entry = rhs.entry(k, col);
                if expr_is_zero(&lhs) || expr_is_zero(&rhs_entry) {
                    continue;
                }
                let term = if expr_is_one(&lhs) {
                    rhs_entry
                } else if expr_is_one(&rhs_entry) {
                    lhs
                } else {
                    Expr::binary(lhs, BinaryOp::Mul, rhs_entry)
                };
                acc = Some(match acc {
                    None => term,
                    Some(prev) => Expr::binary(prev, BinaryOp::Add, term),
                });
            }
            acc.unwrap_or_else(|| Expr::lit_f32(0.0))
        })
    }

    pub fn assign_op_diag(&self, op: AssignOp, value: Expr) -> Vec<Stmt> {
        let diag_len = std::cmp::min(R, C);
        let mut out = Vec::with_capacity(diag_len);
        for idx in 0..diag_len {
            out.push(Stmt::AssignOp {
                target: self.entry(idx, idx),
                op,
                value: value.clone(),
            });
        }
        out
    }

    pub fn assign_to_prefix_scaled(&self, prefix: &str, scale: Option<Expr>) -> Vec<Stmt> {
        let mut out = Vec::new();
        for row in 0..R {
            for col in 0..C {
                let target = Expr::ident(format!("{prefix}_{row}{col}"));
                let mut value = self.entry(row, col);
                if let Some(scale) = scale.clone() {
                    value = Expr::binary(value, BinaryOp::Mul, scale);
                }
                out.push(Stmt::Assign { target, value });
            }
        }
        out
    }

    pub fn assign_op_to_prefix_scaled(
        &self,
        op: AssignOp,
        prefix: &str,
        scale: Option<Expr>,
    ) -> Vec<Stmt> {
        let mut out = Vec::new();
        for row in 0..R {
            for col in 0..C {
                let target = Expr::ident(format!("{prefix}_{row}{col}"));
                let mut value = self.entry(row, col);
                if let Some(scale) = scale.clone() {
                    value = Expr::binary(value, BinaryOp::Mul, scale);
                }
                out.push(Stmt::AssignOp { target, op, value });
            }
        }
        out
    }

    pub fn scatter_assign_to_block_entry_scaled(
        &self,
        entry: &BlockCsrSoaEntry,
        scale: Option<Expr>,
    ) -> Vec<Stmt> {
        let mut out = Vec::new();
        for row in 0..R {
            for col in 0..C {
                let row_u8 = row as u8;
                let col_u8 = col as u8;
                let target = entry.access_expr(row_u8, col_u8);
                let mut value = self.entry(row, col);
                if let Some(scale) = scale.clone() {
                    value = Expr::binary(value, BinaryOp::Mul, scale);
                }
                out.push(Stmt::Assign { target, value });
            }
        }
        out
    }
}

impl<const N: usize> MatExpr<N, N> {
    pub fn identity() -> Self {
        Self::from_fn(|row, col| {
            if row == col {
                Expr::lit_f32(1.0)
            } else {
                Expr::lit_f32(0.0)
            }
        })
    }
}

fn expr_is_zero(expr: &Expr) -> bool {
    match expr {
        Expr::Literal(Literal::Float(value)) => match value.parse::<f32>() {
            Ok(v) => v == 0.0,
            Err(_) => false,
        },
        _ => false,
    }
}

fn expr_is_one(expr: &Expr) -> bool {
    match expr {
        Expr::Literal(Literal::Float(value)) => match value.parse::<f32>() {
            Ok(v) => v == 1.0,
            Err(_) => false,
        },
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::codegen::dsl::matrix::BlockShape;
    use crate::solver::codegen::dsl::{BlockCsrSoaMatrix, ScalarType, UnitDim};

    #[test]
    fn mat_expr_builds_prefix_entries() {
        let mat = MatExpr::<2, 3>::from_prefix("a");
        assert_eq!(mat.entry(1, 2).to_string(), "a_12");
    }

    #[test]
    fn mat_expr_scatter_uses_block_entry_indexing() {
        let mat = MatExpr::<2, 2>::from_prefix("jac");
        let block = BlockShape::new(2, 2);
        let soa = BlockCsrSoaMatrix::from_start_row_prefix(
            "matrix_values",
            "start",
            block,
            ScalarType::F32,
            UnitDim::dimensionless(),
        );
        let entry = soa.row_entry(&Expr::ident("rank"));
        let stmts = mat.scatter_assign_to_block_entry_scaled(&entry, Some(Expr::ident("area")));
        assert_eq!(stmts.len(), 4);
        match &stmts[0] {
            Stmt::Assign { target, value } => {
                assert!(target.to_string().starts_with("matrix_values["));
                assert_eq!(value.to_string(), "jac_00 * area");
            }
            _ => panic!("expected assign stmt"),
        }
    }

    #[test]
    fn mat_expr_mul_simplifies_identity() {
        let mat = MatExpr::<2, 2>::from_prefix("a");
        let ident = MatExpr::<2, 2>::identity();
        let prod = mat.mul_mat(&ident);
        assert_eq!(prod.entry(0, 0).to_string(), "a_00");
        assert_eq!(prod.entry(0, 1).to_string(), "a_01");
        assert_eq!(prod.entry(1, 0).to_string(), "a_10");
        assert_eq!(prod.entry(1, 1).to_string(), "a_11");
    }
}
