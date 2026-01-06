use std::fmt;

use crate::solver::codegen::wgsl_ast::{BinaryOp, Expr, UnaryOp};

use super::{DslType, Shape, UnitDim};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DslError {
    TypeMismatch {
        op: &'static str,
        lhs: DslType,
        rhs: DslType,
    },
    UnitMismatch {
        op: &'static str,
        lhs: UnitDim,
        rhs: UnitDim,
    },
    Unsupported {
        op: &'static str,
        ty: DslType,
    },
}

impl fmt::Display for DslError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DslError::TypeMismatch { op, lhs, rhs } => {
                write!(f, "type mismatch for {op}: {lhs:?} vs {rhs:?}")
            }
            DslError::UnitMismatch { op, lhs, rhs } => {
                write!(f, "unit mismatch for {op}: {lhs} vs {rhs}")
            }
            DslError::Unsupported { op, ty } => write!(f, "unsupported operation {op} for {ty:?}"),
        }
    }
}

impl std::error::Error for DslError {}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedExpr {
    pub expr: Expr,
    pub ty: DslType,
    pub unit: UnitDim,
}

impl TypedExpr {
    pub fn new(expr: Expr, ty: DslType, unit: UnitDim) -> Self {
        Self { expr, ty, unit }
    }

    pub fn f32(value: f32, unit: UnitDim) -> Self {
        Self::new(Expr::lit_f32(value), DslType::f32(), unit)
    }

    pub fn ident(name: &str, ty: DslType, unit: UnitDim) -> Self {
        Self::new(Expr::ident(name), ty, unit)
    }

    pub fn to_wgsl(&self) -> Expr {
        self.expr.clone()
    }

    pub fn add(&self, rhs: &Self) -> Result<Self, DslError> {
        if self.ty != rhs.ty {
            return Err(DslError::TypeMismatch {
                op: "+",
                lhs: self.ty,
                rhs: rhs.ty,
            });
        }
        if self.unit != rhs.unit {
            return Err(DslError::UnitMismatch {
                op: "+",
                lhs: self.unit,
                rhs: rhs.unit,
            });
        }
        Ok(Self::new(
            Expr::binary(self.to_wgsl(), BinaryOp::Add, rhs.to_wgsl()),
            self.ty,
            self.unit,
        ))
    }

    pub fn sub(&self, rhs: &Self) -> Result<Self, DslError> {
        if self.ty != rhs.ty {
            return Err(DslError::TypeMismatch {
                op: "-",
                lhs: self.ty,
                rhs: rhs.ty,
            });
        }
        if self.unit != rhs.unit {
            return Err(DslError::UnitMismatch {
                op: "-",
                lhs: self.unit,
                rhs: rhs.unit,
            });
        }
        Ok(Self::new(
            Expr::binary(self.to_wgsl(), BinaryOp::Sub, rhs.to_wgsl()),
            self.ty,
            self.unit,
        ))
    }

    pub fn mul(&self, rhs: &Self) -> Result<Self, DslError> {
        match (self.ty.shape, rhs.ty.shape) {
            (Shape::Scalar, _) => Ok(Self::new(
                Expr::binary(self.to_wgsl(), BinaryOp::Mul, rhs.to_wgsl()),
                rhs.ty,
                self.unit * rhs.unit,
            )),
            (_, Shape::Scalar) => Ok(Self::new(
                Expr::binary(self.to_wgsl(), BinaryOp::Mul, rhs.to_wgsl()),
                self.ty,
                self.unit * rhs.unit,
            )),
            _ => Err(DslError::Unsupported {
                op: "*",
                ty: self.ty,
            }),
        }
    }

    pub fn div(&self, rhs: &Self) -> Result<Self, DslError> {
        match rhs.ty.shape {
            Shape::Scalar => Ok(Self::new(
                Expr::binary(self.to_wgsl(), BinaryOp::Div, rhs.to_wgsl()),
                self.ty,
                self.unit / rhs.unit,
            )),
            _ => Err(DslError::Unsupported {
                op: "/",
                ty: rhs.ty,
            }),
        }
    }

    pub fn neg(&self) -> Result<Self, DslError> {
        match self.ty.shape {
            Shape::Scalar | Shape::Vec(_) => Ok(Self::new(
                Expr::unary(UnaryOp::Negate, self.to_wgsl()),
                self.ty,
                self.unit,
            )),
            _ => Err(DslError::Unsupported {
                op: "neg",
                ty: self.ty,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::codegen::dsl::units::UnitDim;
    use crate::solver::codegen::dsl::types::{DslType, Shape};

    #[test]
    fn typed_expr_rejects_unit_mismatch_on_add() {
        let a = TypedExpr::f32(1.0, UnitDim::new(0, 1, 0));
        let b = TypedExpr::f32(2.0, UnitDim::new(0, 0, 0));
        let err = a.add(&b).unwrap_err();
        assert!(matches!(err, DslError::UnitMismatch { .. }));
    }

    #[test]
    fn typed_expr_allows_scalar_times_vector() {
        let scalar = TypedExpr::f32(2.0, UnitDim::dimensionless());
        let vec = TypedExpr::ident("u", DslType::vec2_f32(), UnitDim::new(0, 1, -1));
        let out = scalar.mul(&vec).expect("scalar * vec");
        assert_eq!(out.ty.shape, Shape::Vec(2));
        assert_eq!(out.unit, UnitDim::new(0, 1, -1));
        assert_eq!(out.expr.to_string(), "2.0 * u");
    }
}

