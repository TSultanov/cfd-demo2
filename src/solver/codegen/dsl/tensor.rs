use crate::solver::codegen::wgsl_ast::{AssignOp, BinaryOp, Expr, Stmt};

use super::matrix::BlockCsrSoaEntry;

#[derive(Debug, Clone, PartialEq)]
pub struct MatExpr<const R: usize, const C: usize> {
    entries: [[Expr; C]; R],
}

impl<const R: usize, const C: usize> MatExpr<R, C> {
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
}

