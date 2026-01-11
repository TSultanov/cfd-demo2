use std::marker::PhantomData;

use crate::solver::codegen::wgsl_ast::Expr;

pub trait WgslEnum: Copy {
    fn wgsl_id(self) -> u32;
}

#[derive(Clone, Copy, Debug)]
pub struct EnumExpr<E: WgslEnum> {
    expr: Expr,
    _marker: PhantomData<E>,
}

impl<E: WgslEnum> EnumExpr<E> {
    pub fn from_expr(expr: Expr) -> Self {
        Self {
            expr,
            _marker: PhantomData,
        }
    }

    pub fn expr(self) -> Expr {
        self.expr
    }

    pub fn eq(self, rhs: E) -> Expr {
        self.expr.eq(rhs.wgsl_id())
    }

    pub fn ne(self, rhs: E) -> Expr {
        self.expr.ne(rhs.wgsl_id())
    }

    pub fn eq_expr(self, rhs: Self) -> Expr {
        self.expr.eq(rhs.expr)
    }

    pub fn ne_expr(self, rhs: Self) -> Expr {
        self.expr.ne(rhs.expr)
    }
}

impl WgslEnum for crate::solver::scheme::Scheme {
    fn wgsl_id(self) -> u32 {
        self.gpu_id()
    }
}

impl WgslEnum for crate::solver::ir::Discretization {
    fn wgsl_id(self) -> u32 {
        match self {
            crate::solver::ir::Discretization::Implicit => 0,
            crate::solver::ir::Discretization::Explicit => 1,
        }
    }
}

impl WgslEnum for crate::solver::gpu::enums::GpuBoundaryType {
    fn wgsl_id(self) -> u32 {
        self as u32
    }
}

impl WgslEnum for crate::solver::gpu::enums::TimeScheme {
    fn wgsl_id(self) -> u32 {
        self as u32
    }
}

impl WgslEnum for crate::solver::gpu::enums::GpuBcKind {
    fn wgsl_id(self) -> u32 {
        self as u32
    }
}

impl WgslEnum for crate::solver::gpu::enums::GpuLowMachPrecondModel {
    fn wgsl_id(self) -> u32 {
        self as u32
    }
}
