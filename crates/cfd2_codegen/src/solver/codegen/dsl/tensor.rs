use crate::solver::codegen::wgsl_ast::{AssignOp, Expr, Stmt};

use super::matrix::{BlockCsrSoaEntry, NamedBlockCsrSoaEntry};

pub trait Axis<const N: usize> {
    type Index: Copy;

    fn to_usize(index: Self::Index) -> usize;
}

// ── Coupled-system axis types ─────────────────────────────────────────

/// Trait for coupled-system axis types used in block CSR matrices.
///
/// Unlike `Axis<N>` which is const-generic, this trait enables
/// compile-time distinguishing of row vs column indices in block
/// matrix entry calls. The block matrix is always square, so a
/// single axis type describes both rows and columns.
pub trait CoupledAxis: Copy + std::fmt::Debug + 'static {
    /// Number of scalar unknowns per cell for this coupled system.
    const STRIDE: u32;

    /// Convert this axis value to its positional index.
    fn to_u8(self) -> u8;

    /// Convert a numeric index to this axis value.
    ///
    /// Panics if `index >= Self::STRIDE`.
    fn from_u32(index: u32) -> Self;

    /// All variants in index order.
    fn all() -> &'static [Self];
}

/// Newtype wrapper marking a block-matrix **row** index.
///
/// Cannot be used where a `BlockCol` is expected, preventing
/// accidental transposition of row/col arguments.
#[derive(Debug, Clone, Copy)]
pub struct BlockRow<Ax: CoupledAxis>(Ax);

impl<Ax: CoupledAxis> BlockRow<Ax> {
    pub fn new(ax: Ax) -> Self {
        Self(ax)
    }

    pub fn to_u8(self) -> u8 {
        self.0.to_u8()
    }

    pub fn axis(self) -> Ax {
        self.0
    }
}

/// Newtype wrapper marking a block-matrix **column** index.
///
/// Cannot be used where a `BlockRow` is expected, preventing
/// accidental transposition of row/col arguments.
#[derive(Debug, Clone, Copy)]
pub struct BlockCol<Ax: CoupledAxis>(Ax);

impl<Ax: CoupledAxis> BlockCol<Ax> {
    pub fn new(ax: Ax) -> Self {
        Self(ax)
    }

    pub fn to_u8(self) -> u8 {
        self.0.to_u8()
    }

    pub fn axis(self) -> Ax {
        self.0
    }
}

/// Convenience constructors for `BlockRow` / `BlockCol` from numeric indices.
pub fn block_row<Ax: CoupledAxis>(index: u32) -> BlockRow<Ax> {
    BlockRow::new(Ax::from_u32(index))
}

pub fn block_col<Ax: CoupledAxis>(index: u32) -> BlockCol<Ax> {
    BlockCol::new(Ax::from_u32(index))
}

// ── Concrete coupled axis enums ───────────────────────────────────────

/// Single-scalar coupled system (e.g., generic diffusion demo).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ScalarAxis {
    Phi,
}

impl CoupledAxis for ScalarAxis {
    const STRIDE: u32 = 1;

    fn to_u8(self) -> u8 {
        0
    }

    fn from_u32(index: u32) -> Self {
        match index {
            0 => ScalarAxis::Phi,
            _ => panic!("ScalarAxis::from_u32({index}): expected 0"),
        }
    }

    fn all() -> &'static [Self] {
        &[ScalarAxis::Phi]
    }
}

/// 2D incompressible velocity-pressure system (stride 3).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum IncompressibleAxis2D {
    Ux,
    Uy,
    P,
}

impl CoupledAxis for IncompressibleAxis2D {
    const STRIDE: u32 = 3;

    fn to_u8(self) -> u8 {
        match self {
            Self::Ux => 0,
            Self::Uy => 1,
            Self::P => 2,
        }
    }

    fn from_u32(index: u32) -> Self {
        match index {
            0 => Self::Ux,
            1 => Self::Uy,
            2 => Self::P,
            _ => panic!("IncompressibleAxis2D::from_u32({index}): expected 0..3"),
        }
    }

    fn all() -> &'static [Self] {
        &[Self::Ux, Self::Uy, Self::P]
    }
}

/// 3D incompressible velocity-pressure system (stride 4).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum IncompressibleAxis3D {
    Ux,
    Uy,
    Uz,
    P,
}

impl CoupledAxis for IncompressibleAxis3D {
    const STRIDE: u32 = 4;

    fn to_u8(self) -> u8 {
        match self {
            Self::Ux => 0,
            Self::Uy => 1,
            Self::Uz => 2,
            Self::P => 3,
        }
    }

    fn from_u32(index: u32) -> Self {
        match index {
            0 => Self::Ux,
            1 => Self::Uy,
            2 => Self::Uz,
            3 => Self::P,
            _ => panic!("IncompressibleAxis3D::from_u32({index}): expected 0..4"),
        }
    }

    fn all() -> &'static [Self] {
        &[Self::Ux, Self::Uy, Self::Uz, Self::P]
    }
}

/// 2D compressible Euler/Navier-Stokes system (stride 8).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CompressibleAxis2D {
    Rho,
    RhoUx,
    RhoUy,
    RhoE,
    Ux,
    Uy,
    P,
    T,
}

impl CoupledAxis for CompressibleAxis2D {
    const STRIDE: u32 = 8;

    fn to_u8(self) -> u8 {
        match self {
            Self::Rho => 0,
            Self::RhoUx => 1,
            Self::RhoUy => 2,
            Self::RhoE => 3,
            Self::Ux => 4,
            Self::Uy => 5,
            Self::P => 6,
            Self::T => 7,
        }
    }

    fn from_u32(index: u32) -> Self {
        match index {
            0 => Self::Rho,
            1 => Self::RhoUx,
            2 => Self::RhoUy,
            3 => Self::RhoE,
            4 => Self::Ux,
            5 => Self::Uy,
            6 => Self::P,
            7 => Self::T,
            _ => panic!("CompressibleAxis2D::from_u32({index}): expected 0..8"),
        }
    }

    fn all() -> &'static [Self] {
        &[
            Self::Rho,
            Self::RhoUx,
            Self::RhoUy,
            Self::RhoE,
            Self::Ux,
            Self::Uy,
            Self::P,
            Self::T,
        ]
    }
}

/// Dispatch a function call by matching `coupled_stride` to the appropriate
/// concrete `CoupledAxis` type. This is the monomorphization point: the
/// `body` closure is instantiated for each supported stride.
///
/// Returns an error string if the stride is not recognised.
pub fn dispatch_by_coupled_stride<R>(
    coupled_stride: u32,
    body: impl DispatchByStride<R>,
) -> Result<R, String> {
    match coupled_stride {
        1 => Ok(body.call::<ScalarAxis>()),
        3 => Ok(body.call::<IncompressibleAxis2D>()),
        4 => Ok(body.call::<IncompressibleAxis3D>()),
        8 => Ok(body.call::<CompressibleAxis2D>()),
        _ => Err(format!(
            "unsupported coupled_stride {coupled_stride}; expected 1, 3, 4, or 8"
        )),
    }
}

/// Helper trait for `dispatch_by_coupled_stride` to work around Rust's
/// lack of generic closures. Implement this for a struct that captures
/// the closure's environment.
pub trait DispatchByStride<R> {
    fn call<Ax: CoupledAxis>(&self) -> R;
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum XY {
    X,
    Y,
}

impl XY {
    /// Both variants in index order, useful for iteration.
    pub const ALL: [XY; 2] = [XY::X, XY::Y];

    /// Convert a numeric component index to the corresponding axis value.
    ///
    /// Panics if `index > 1`.
    pub fn from_index(index: u32) -> Self {
        match index {
            0 => XY::X,
            1 => XY::Y,
            _ => panic!("XY::from_index({index}): expected 0 or 1"),
        }
    }

    /// Return the positional index (`0` for X, `1` for Y).
    pub fn to_usize(self) -> usize {
        match self {
            XY::X => 0,
            XY::Y => 1,
        }
    }

    /// Short lowercase suffix (`"x"` or `"y"`), matching WGSL swizzle names.
    pub fn suffix(self) -> &'static str {
        match self {
            XY::X => "x",
            XY::Y => "y",
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct AxisXY;

impl Axis<2> for AxisXY {
    type Index = XY;

    fn to_usize(index: Self::Index) -> usize {
        index.to_usize()
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
            if let Some(args) = self.expr.try_call_named(ctor_name) {
                if args.len() == N {
                    return args[index].clone();
                }
            }
        }
        self.expr().field(vec_field_name(index))
    }

    pub fn from_components(components: [Expr; N]) -> Self {
        Self::from_expr(Expr::call_named(
            Self::constructor_name(),
            components.into(),
        ))
    }

    pub fn zeros() -> Self {
        Self::from_components(std::array::from_fn(|_| 0.0.into()))
    }

    pub fn add(&self, rhs: &Self) -> Self {
        Self::from_expr(self.expr() + rhs.expr())
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        Self::from_expr(self.expr() - rhs.expr())
    }

    pub fn neg(&self) -> Self {
        Self::from_expr(-self.expr())
    }

    pub fn mul_scalar(&self, scalar: Expr) -> Self {
        Self::from_expr(self.expr() * scalar)
    }

    pub fn div_scalar(&self, scalar: Expr) -> Self {
        Self::from_expr(self.expr() / scalar)
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

    pub fn from_components(components: [Expr; N]) -> Self {
        Self::from_vec(VecExpr::from_components(components))
    }

    pub fn zeros() -> Self {
        Self::from_vec(VecExpr::zeros())
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
            lhs + rhs
        })
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        Self::from_fn(|row, col| {
            let lhs = self.entry(row, col);
            let rhs = rhs.entry(row, col);
            if expr_is_zero(&rhs) {
                return lhs;
            }
            lhs - rhs
        })
    }

    pub fn mul_scalar(&self, scalar: Expr) -> Self {
        if expr_is_one(&scalar) {
            return self.clone();
        }
        if expr_is_zero(&scalar) {
            return Self::from_fn(|_, _| 0.0.into());
        }
        Self::from_fn(|row, col| {
            let entry = self.entry(row, col);
            if expr_is_zero(&entry) {
                return 0.0.into();
            }
            if expr_is_one(&entry) {
                return scalar.clone();
            }
            entry * scalar.clone()
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
                    lhs * rhs_entry
                };
                acc = Some(match acc {
                    None => term,
                    Some(prev) => prev + term,
                });
            }
            acc.unwrap_or_else(|| 0.0.into())
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
                if let Some(ref scale) = scale {
                    value = value * scale.clone();
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
                if let Some(ref scale) = scale {
                    value = value * scale.clone();
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
                if let Some(ref scale) = scale {
                    value = value * scale.clone();
                }
                out.push(Stmt::Assign { target, value });
            }
        }
        out
    }

    /// Type-safe variant of [`scatter_assign_to_block_entry_scaled`] that
    /// uses [`NamedBlockCsrSoaEntry`] with phantom-typed row/col indices.
    pub fn scatter_assign_to_named_block_entry_scaled<Ax: CoupledAxis>(
        &self,
        entry: &NamedBlockCsrSoaEntry<Ax>,
        scale: Option<Expr>,
    ) -> Vec<Stmt> {
        let mut out = Vec::new();
        for row in 0..R {
            for col in 0..C {
                let target = entry.access_expr(
                    BlockRow::new(Ax::from_u32(row as u32)),
                    BlockCol::new(Ax::from_u32(col as u32)),
                );
                let mut value = self.entry(row, col);
                if let Some(ref scale) = scale {
                    value = value * scale.clone();
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
                return 0.0.into();
            }
            if expr_is_one(&scale) {
                return entry;
            }
            entry * scale
        }))
    }

    pub fn mul_col_broadcast(&self, col: &NamedVecExpr<C, ColAx>) -> Self {
        Self::from_mat(MatExpr::from_fn(|r, c| {
            let entry = self.entry(r, c);
            let scale = col.component(c);
            if expr_is_zero(&entry) || expr_is_zero(&scale) {
                return 0.0.into();
            }
            if expr_is_one(&scale) {
                return entry;
            }
            entry * scale
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
                    lhs * rhs
                };
                acc = Some(match acc {
                    None => term,
                    Some(prev) => prev + term,
                });
            }
            acc.unwrap_or_else(|| 0.0.into())
        })))
    }

    pub fn mul_mat<const D: usize, OutColAx>(
        &self,
        rhs: &NamedMatExpr<C, D, ColAx, OutColAx>,
    ) -> NamedMatExpr<R, D, RowAx, OutColAx> {
        NamedMatExpr::from_mat(self.mat.mul_mat(&rhs.mat))
    }
}

impl<const N: usize> MatExpr<N, N> {
    pub fn identity() -> Self {
        Self::from_fn(|row, col| if row == col { 1.0.into() } else { 0.0.into() })
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
    matches!(expr.try_f32_literal(), Some(value) if value == 0.0)
}

fn expr_is_one(expr: &Expr) -> bool {
    matches!(expr.try_f32_literal(), Some(value) if value == 1.0)
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

    #[test]
    fn xy_from_index_roundtrips() {
        assert_eq!(XY::from_index(0), XY::X);
        assert_eq!(XY::from_index(1), XY::Y);
        assert_eq!(XY::X.to_usize(), 0);
        assert_eq!(XY::Y.to_usize(), 1);
    }

    #[test]
    fn xy_suffix_returns_swizzle_names() {
        assert_eq!(XY::X.suffix(), "x");
        assert_eq!(XY::Y.suffix(), "y");
    }

    #[test]
    fn xy_all_iterates_both_axes() {
        let axes: Vec<_> = XY::ALL.iter().map(|a| a.suffix()).collect();
        assert_eq!(axes, vec!["x", "y"]);
    }

    #[test]
    fn named_vec_from_components_and_zeros() {
        let v = NamedVecExpr::<2, AxisXY>::from_components([Expr::ident("a"), Expr::ident("b")]);
        assert_eq!(v.at(XY::X).to_string(), "a");
        assert_eq!(v.at(XY::Y).to_string(), "b");

        let z = NamedVecExpr::<2, AxisXY>::zeros();
        assert_eq!(z.at(XY::X).to_string(), "0.0");
        assert_eq!(z.at(XY::Y).to_string(), "0.0");
    }

    // ── CoupledAxis enum tests ────────────────────────────────────────

    #[test]
    fn scalar_axis_roundtrips() {
        assert_eq!(ScalarAxis::Phi.to_u8(), 0);
        assert_eq!(ScalarAxis::from_u32(0), ScalarAxis::Phi);
        assert_eq!(ScalarAxis::all(), &[ScalarAxis::Phi]);
        assert_eq!(ScalarAxis::STRIDE, 1);
    }

    #[test]
    #[should_panic(expected = "expected 0")]
    fn scalar_axis_from_u32_panics_on_out_of_range() {
        ScalarAxis::from_u32(1);
    }

    #[test]
    fn incompressible_2d_roundtrips() {
        use IncompressibleAxis2D::*;
        assert_eq!(IncompressibleAxis2D::STRIDE, 3);
        let all = IncompressibleAxis2D::all();
        assert_eq!(all, &[Ux, Uy, P]);
        for (i, &ax) in all.iter().enumerate() {
            assert_eq!(ax.to_u8() as usize, i);
            assert_eq!(IncompressibleAxis2D::from_u32(i as u32), ax);
        }
    }

    #[test]
    #[should_panic(expected = "expected 0..3")]
    fn incompressible_2d_from_u32_panics_on_out_of_range() {
        IncompressibleAxis2D::from_u32(3);
    }

    #[test]
    fn incompressible_3d_roundtrips() {
        use IncompressibleAxis3D::*;
        assert_eq!(IncompressibleAxis3D::STRIDE, 4);
        let all = IncompressibleAxis3D::all();
        assert_eq!(all, &[Ux, Uy, Uz, P]);
        for (i, &ax) in all.iter().enumerate() {
            assert_eq!(ax.to_u8() as usize, i);
            assert_eq!(IncompressibleAxis3D::from_u32(i as u32), ax);
        }
    }

    #[test]
    #[should_panic(expected = "expected 0..4")]
    fn incompressible_3d_from_u32_panics_on_out_of_range() {
        IncompressibleAxis3D::from_u32(4);
    }

    #[test]
    fn compressible_2d_roundtrips() {
        use CompressibleAxis2D::*;
        assert_eq!(CompressibleAxis2D::STRIDE, 8);
        let all = CompressibleAxis2D::all();
        assert_eq!(all, &[Rho, RhoUx, RhoUy, RhoE, Ux, Uy, P, T]);
        for (i, &ax) in all.iter().enumerate() {
            assert_eq!(ax.to_u8() as usize, i);
            assert_eq!(CompressibleAxis2D::from_u32(i as u32), ax);
        }
    }

    #[test]
    #[should_panic(expected = "expected 0..8")]
    fn compressible_2d_from_u32_panics_on_out_of_range() {
        CompressibleAxis2D::from_u32(8);
    }

    // ── BlockRow / BlockCol tests ─────────────────────────────────────

    #[test]
    fn block_row_preserves_axis() {
        let row = BlockRow::new(IncompressibleAxis2D::P);
        assert_eq!(row.to_u8(), 2);
        assert_eq!(row.axis(), IncompressibleAxis2D::P);
    }

    #[test]
    fn block_col_preserves_axis() {
        let col = BlockCol::new(IncompressibleAxis2D::Ux);
        assert_eq!(col.to_u8(), 0);
        assert_eq!(col.axis(), IncompressibleAxis2D::Ux);
    }

    #[test]
    fn block_row_col_free_functions() {
        let row: BlockRow<IncompressibleAxis3D> = block_row(2);
        assert_eq!(row.axis(), IncompressibleAxis3D::Uz);
        assert_eq!(row.to_u8(), 2);

        let col: BlockCol<IncompressibleAxis3D> = block_col(3);
        assert_eq!(col.axis(), IncompressibleAxis3D::P);
        assert_eq!(col.to_u8(), 3);
    }

    // ── dispatch_by_coupled_stride tests ──────────────────────────────

    struct StrideChecker;

    impl DispatchByStride<u32> for StrideChecker {
        fn call<Ax: CoupledAxis>(&self) -> u32 {
            Ax::STRIDE
        }
    }

    #[test]
    fn dispatch_resolves_all_supported_strides() {
        assert_eq!(dispatch_by_coupled_stride(1, StrideChecker), Ok(1));
        assert_eq!(dispatch_by_coupled_stride(3, StrideChecker), Ok(3));
        assert_eq!(dispatch_by_coupled_stride(4, StrideChecker), Ok(4));
        assert_eq!(dispatch_by_coupled_stride(8, StrideChecker), Ok(8));
    }

    #[test]
    fn dispatch_returns_error_for_unsupported_stride() {
        let result = dispatch_by_coupled_stride(5, StrideChecker);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unsupported coupled_stride 5"));
    }

    // ── scatter_assign_to_named_block_entry_scaled tests ──────────────

    #[test]
    fn scatter_named_matches_untyped() {
        let mat = MatExpr::<3, 3>::from_prefix("jac");
        let block = BlockShape::new(3, 3);
        let soa = BlockCsrSoaMatrix::from_start_row_prefix(
            "vals",
            "start",
            block,
            ScalarType::F32,
            UnitDim::dimensionless(),
        );
        let entry = soa.row_entry(&Expr::ident("rank"));

        // Untyped version
        let untyped = mat.scatter_assign_to_block_entry_scaled(&entry, Some(Expr::ident("area")));

        // Typed version
        let named_entry =
            super::super::matrix::NamedBlockCsrSoaEntry::<IncompressibleAxis2D>::new(entry);
        let typed =
            mat.scatter_assign_to_named_block_entry_scaled(&named_entry, Some(Expr::ident("area")));

        assert_eq!(untyped.len(), typed.len());
        for (u, t) in untyped.iter().zip(typed.iter()) {
            match (u, t) {
                (
                    Stmt::Assign {
                        target: ut,
                        value: uv,
                    },
                    Stmt::Assign {
                        target: tt,
                        value: tv,
                    },
                ) => {
                    assert_eq!(ut.to_string(), tt.to_string());
                    assert_eq!(uv.to_string(), tv.to_string());
                }
                _ => panic!("expected Assign stmts"),
            }
        }
    }

    #[test]
    fn scatter_named_without_scale() {
        let mat = MatExpr::<1, 1>::from_prefix("j");
        let block = BlockShape::new(1, 1);
        let soa = BlockCsrSoaMatrix::from_start_row_prefix(
            "vals",
            "start",
            block,
            ScalarType::F32,
            UnitDim::dimensionless(),
        );
        let entry = soa.row_entry(&Expr::ident("rank"));
        let named_entry = super::super::matrix::NamedBlockCsrSoaEntry::<ScalarAxis>::new(entry);
        let stmts = mat.scatter_assign_to_named_block_entry_scaled(&named_entry, None);
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Stmt::Assign { target, value } => {
                assert!(target.to_string().starts_with("vals["));
                assert_eq!(value.to_string(), "j_00");
            }
            _ => panic!("expected assign stmt"),
        }
    }
}
