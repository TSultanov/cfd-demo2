use crate::solver::codegen::wgsl_ast::{AssignOp, BinaryOp, Expr, Literal, Stmt};

use super::matrix::BlockCsrSoaEntry;

pub trait Axis<const N: usize> {
    type Index: Copy;

    fn to_usize(index: Self::Index) -> usize;
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum XY {
    X,
    Y,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct AxisXY;

impl Axis<2> for AxisXY {
    type Index = XY;

    fn to_usize(index: Self::Index) -> usize {
        match index {
            XY::X => 0,
            XY::Y => 1,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Cons {
    Rho,
    Ru,
    Rv,
    Re,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct AxisCons;

impl Axis<4> for AxisCons {
    type Index = Cons;

    fn to_usize(index: Self::Index) -> usize {
        match index {
            Cons::Rho => 0,
            Cons::Ru => 1,
            Cons::Rv => 2,
            Cons::Re => 3,
        }
    }
}

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

    pub fn component(&self, index: usize) -> Expr {
        if index >= N {
            panic!("vector component {index} out of bounds for VecExpr<{N}>");
        }
        let ctor_name = match N {
            2 => Some("vec2<f32>"),
            3 => Some("vec3<f32>"),
            4 => Some("vec4<f32>"),
            _ => None,
        };
        if let Some(ctor_name) = ctor_name {
            if let Expr::Call { callee, args } = &self.expr {
                if let Expr::Ident(name) = callee.as_ref() {
                    if name == ctor_name && args.len() == N {
                        return args[index].clone();
                    }
                }
            }
        }
        self.expr().field(vec_field_name(index))
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
pub struct NamedVecExpr<const N: usize, Ax> {
    vec: VecExpr<N>,
    _axis: std::marker::PhantomData<Ax>,
}

impl<const N: usize, Ax> NamedVecExpr<N, Ax> {
    pub fn from_vec(vec: VecExpr<N>) -> Self {
        Self {
            vec,
            _axis: std::marker::PhantomData,
        }
    }

    pub fn from_expr(expr: Expr) -> Self {
        Self::from_vec(VecExpr::from_expr(expr))
    }

    pub fn expr(&self) -> Expr {
        self.vec.expr()
    }

    pub fn vec_expr(&self) -> VecExpr<N> {
        self.vec.clone()
    }

    pub fn add(&self, rhs: &Self) -> Self {
        Self::from_vec(self.vec.add(&rhs.vec))
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        Self::from_vec(self.vec.sub(&rhs.vec))
    }

    pub fn neg(&self) -> Self {
        Self::from_vec(self.vec.neg())
    }

    pub fn mul_scalar(&self, scalar: Expr) -> Self {
        Self::from_vec(self.vec.mul_scalar(scalar))
    }

    pub fn div_scalar(&self, scalar: Expr) -> Self {
        Self::from_vec(self.vec.div_scalar(scalar))
    }

    pub fn dot(&self, rhs: &Self) -> Expr {
        self.vec.dot(&rhs.vec)
    }

    fn component(&self, index: usize) -> Expr {
        self.vec.component(index)
    }
}

impl<const N: usize, Ax: Axis<N>> NamedVecExpr<N, Ax> {
    pub fn at(&self, index: Ax::Index) -> Expr {
        self.component(Ax::to_usize(index))
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
            let lhs = self.entry(row, col);
            let rhs = rhs.entry(row, col);
            if expr_is_zero(&lhs) {
                return rhs;
            }
            if expr_is_zero(&rhs) {
                return lhs;
            }
            Expr::binary(lhs, BinaryOp::Add, rhs)
        })
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        Self::from_fn(|row, col| {
            let lhs = self.entry(row, col);
            let rhs = rhs.entry(row, col);
            if expr_is_zero(&rhs) {
                return lhs;
            }
            Expr::binary(lhs, BinaryOp::Sub, rhs)
        })
    }

    pub fn mul_scalar(&self, scalar: Expr) -> Self {
        if expr_is_one(&scalar) {
            return self.clone();
        }
        if expr_is_zero(&scalar) {
            return Self::from_fn(|_, _| Expr::lit_f32(0.0));
        }
        Self::from_fn(|row, col| {
            let entry = self.entry(row, col);
            if expr_is_zero(&entry) {
                return Expr::lit_f32(0.0);
            }
            if expr_is_one(&entry) {
                return scalar.clone();
            }
            Expr::binary(entry, BinaryOp::Mul, scalar.clone())
        })
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

#[derive(Debug, Clone, PartialEq)]
pub struct NamedMatExpr<const R: usize, const C: usize, RowAx, ColAx> {
    mat: MatExpr<R, C>,
    _row: std::marker::PhantomData<RowAx>,
    _col: std::marker::PhantomData<ColAx>,
}

impl<const R: usize, const C: usize, RowAx, ColAx> NamedMatExpr<R, C, RowAx, ColAx> {
    pub fn from_mat(mat: MatExpr<R, C>) -> Self {
        Self {
            mat,
            _row: std::marker::PhantomData,
            _col: std::marker::PhantomData,
        }
    }

    pub fn from_entries(entries: [[Expr; C]; R]) -> Self {
        Self::from_mat(MatExpr::from_entries(entries))
    }

    pub fn entry(&self, row: usize, col: usize) -> Expr {
        self.mat.entry(row, col)
    }

    pub fn at(&self, row: RowAx::Index, col: ColAx::Index) -> Expr
    where
        RowAx: Axis<R>,
        ColAx: Axis<C>,
    {
        self.entry(RowAx::to_usize(row), ColAx::to_usize(col))
    }

    pub fn add(&self, rhs: &Self) -> Self {
        Self::from_mat(self.mat.add(&rhs.mat))
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        Self::from_mat(self.mat.sub(&rhs.mat))
    }

    pub fn mul_scalar(&self, scalar: Expr) -> Self {
        Self::from_mat(self.mat.mul_scalar(scalar))
    }

    pub fn mat_expr(&self) -> MatExpr<R, C> {
        self.mat.clone()
    }

    pub fn mul_row_broadcast(&self, row: &NamedVecExpr<R, RowAx>) -> Self {
        Self::from_mat(MatExpr::from_fn(|r, c| {
            let entry = self.entry(r, c);
            let scale = row.component(r);
            if expr_is_zero(&entry) || expr_is_zero(&scale) {
                return Expr::lit_f32(0.0);
            }
            if expr_is_one(&scale) {
                return entry;
            }
            Expr::binary(entry, BinaryOp::Mul, scale)
        }))
    }

    pub fn mul_col_broadcast(&self, col: &NamedVecExpr<C, ColAx>) -> Self {
        Self::from_mat(MatExpr::from_fn(|r, c| {
            let entry = self.entry(r, c);
            let scale = col.component(c);
            if expr_is_zero(&entry) || expr_is_zero(&scale) {
                return Expr::lit_f32(0.0);
            }
            if expr_is_one(&scale) {
                return entry;
            }
            Expr::binary(entry, BinaryOp::Mul, scale)
        }))
    }

    pub fn contract_rows(&self, row: &NamedVecExpr<R, RowAx>) -> NamedVecExpr<C, ColAx> {
        NamedVecExpr::from_vec(VecExpr::<C>::from_components(std::array::from_fn(|c| {
            let mut acc: Option<Expr> = None;
            for r in 0..R {
                let lhs = self.entry(r, c);
                let rhs = row.component(r);
                if expr_is_zero(&lhs) || expr_is_zero(&rhs) {
                    continue;
                }
                let term = if expr_is_one(&lhs) {
                    rhs
                } else if expr_is_one(&rhs) {
                    lhs
                } else {
                    Expr::binary(lhs, BinaryOp::Mul, rhs)
                };
                acc = Some(match acc {
                    None => term,
                    Some(prev) => Expr::binary(prev, BinaryOp::Add, term),
                });
            }
            acc.unwrap_or_else(|| Expr::lit_f32(0.0))
        })))
    }

    pub fn mul_mat<const D: usize, OutColAx>(&self, rhs: &NamedMatExpr<C, D, ColAx, OutColAx>) -> NamedMatExpr<R, D, RowAx, OutColAx> {
        NamedMatExpr::from_mat(self.mat.mul_mat(&rhs.mat))
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

fn vec_field_name(index: usize) -> &'static str {
    match index {
        0 => "x",
        1 => "y",
        2 => "z",
        3 => "w",
        _ => panic!("unsupported vector field {index}; expected 0..4"),
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

    #[test]
    fn named_vec_at_uses_axis_swizzles() {
        let u = NamedVecExpr::<4, AxisCons>::from_expr(Expr::ident("u"));
        assert_eq!(u.at(Cons::Rho).to_string(), "u.x");
        assert_eq!(u.at(Cons::Ru).to_string(), "u.y");
        assert_eq!(u.at(Cons::Rv).to_string(), "u.z");
        assert_eq!(u.at(Cons::Re).to_string(), "u.w");

        let v = NamedVecExpr::<2, AxisXY>::from_expr(Expr::ident("v"));
        assert_eq!(v.at(XY::X).to_string(), "v.x");
        assert_eq!(v.at(XY::Y).to_string(), "v.y");
    }

    #[test]
    fn named_mat_contract_rows_builds_vec() {
        let mat = NamedMatExpr::<2, 4, AxisXY, AxisCons>::from_mat(MatExpr::from_prefix("m"));
        let v = NamedVecExpr::<2, AxisXY>::from_expr(Expr::ident("v"));
        let out = mat.contract_rows(&v);
        assert_eq!(out.at(Cons::Rho).to_string(), "m_00 * v.x + m_10 * v.y");
        assert_eq!(out.at(Cons::Re).to_string(), "m_03 * v.x + m_13 * v.y");
    }
}
