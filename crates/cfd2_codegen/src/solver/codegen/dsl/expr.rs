use std::fmt;

use crate::solver::codegen::wgsl_ast::Expr;

use super::{DslType, ScalarType, Shape, UnitDim};

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
        Self::new(value.into(), DslType::f32(), unit)
    }

    pub fn ident(name: &str, ty: DslType, unit: UnitDim) -> Self {
        Self::new(Expr::ident(name), ty, unit)
    }

    pub fn to_wgsl(&self) -> Expr {
        self.expr
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
            self.to_wgsl() + rhs.to_wgsl(),
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
            self.to_wgsl() - rhs.to_wgsl(),
            self.ty,
            self.unit,
        ))
    }

    pub fn mul(&self, rhs: &Self) -> Result<Self, DslError> {
        match (self.ty.shape, rhs.ty.shape) {
            (Shape::Scalar, _) => Ok(Self::new(
                self.to_wgsl() * rhs.to_wgsl(),
                rhs.ty,
                self.unit * rhs.unit,
            )),
            (_, Shape::Scalar) => Ok(Self::new(
                self.to_wgsl() * rhs.to_wgsl(),
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
                self.to_wgsl() / rhs.to_wgsl(),
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
            Shape::Scalar | Shape::Vec(_) => Ok(Self::new(-self.to_wgsl(), self.ty, self.unit)),
            _ => Err(DslError::Unsupported {
                op: "neg",
                ty: self.ty,
            }),
        }
    }

    pub fn sqrt(&self) -> Result<Self, DslError> {
        match (self.ty.scalar, self.ty.shape) {
            (ScalarType::F32, Shape::Scalar | Shape::Vec(_)) => Ok(Self::new(
                Expr::call_named("sqrt", vec![self.to_wgsl()]),
                self.ty,
                self.unit.sqrt(),
            )),
            _ => Err(DslError::Unsupported {
                op: "sqrt",
                ty: self.ty,
            }),
        }
    }

    pub fn abs(&self) -> Result<Self, DslError> {
        match (self.ty.scalar, self.ty.shape) {
            (ScalarType::F32, Shape::Scalar | Shape::Vec(_)) => Ok(Self::new(
                Expr::call_named("abs", vec![self.to_wgsl()]),
                self.ty,
                self.unit,
            )),
            _ => Err(DslError::Unsupported {
                op: "abs",
                ty: self.ty,
            }),
        }
    }

    pub fn min(&self, rhs: &Self) -> Result<Self, DslError> {
        if self.ty != rhs.ty {
            return Err(DslError::TypeMismatch {
                op: "min",
                lhs: self.ty,
                rhs: rhs.ty,
            });
        }
        if self.unit != rhs.unit {
            return Err(DslError::UnitMismatch {
                op: "min",
                lhs: self.unit,
                rhs: rhs.unit,
            });
        }
        Ok(Self::new(
            Expr::call_named("min", vec![self.to_wgsl(), rhs.to_wgsl()]),
            self.ty,
            self.unit,
        ))
    }

    pub fn max(&self, rhs: &Self) -> Result<Self, DslError> {
        if self.ty != rhs.ty {
            return Err(DslError::TypeMismatch {
                op: "max",
                lhs: self.ty,
                rhs: rhs.ty,
            });
        }
        if self.unit != rhs.unit {
            return Err(DslError::UnitMismatch {
                op: "max",
                lhs: self.unit,
                rhs: rhs.unit,
            });
        }
        Ok(Self::new(
            Expr::call_named("max", vec![self.to_wgsl(), rhs.to_wgsl()]),
            self.ty,
            self.unit,
        ))
    }

    pub fn component(&self, index: u8) -> Result<Self, DslError> {
        match (self.ty.scalar, self.ty.shape) {
            (ScalarType::F32, Shape::Vec(2) | Shape::Vec(3) | Shape::Vec(4)) => {
                let field = match index {
                    0 => "x",
                    1 => "y",
                    2 => "z",
                    3 => "w",
                    _ => {
                        return Err(DslError::Unsupported {
                            op: "component",
                            ty: self.ty,
                        })
                    }
                };
                Ok(Self::new(
                    self.to_wgsl().field(field),
                    DslType::f32(),
                    self.unit,
                ))
            }
            _ => Err(DslError::Unsupported {
                op: "component",
                ty: self.ty,
            }),
        }
    }

    pub fn dot(&self, rhs: &Self) -> Result<Self, DslError> {
        match (self.ty.scalar, self.ty.shape, rhs.ty.scalar, rhs.ty.shape) {
            (ScalarType::F32, Shape::Vec(n), ScalarType::F32, Shape::Vec(m)) if n == m => {
                Ok(Self::new(
                    Expr::call_named("dot", vec![self.to_wgsl(), rhs.to_wgsl()]),
                    DslType::f32(),
                    self.unit * rhs.unit,
                ))
            }
            _ => Err(DslError::TypeMismatch {
                op: "dot",
                lhs: self.ty,
                rhs: rhs.ty,
            }),
        }
    }
}

impl std::ops::Add for TypedExpr {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        TypedExpr::add(&self, &rhs).unwrap_or_else(|err| {
            panic!("typed add failed: {err}");
        })
    }
}

impl std::ops::Sub for TypedExpr {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        TypedExpr::sub(&self, &rhs).unwrap_or_else(|err| {
            panic!("typed sub failed: {err}");
        })
    }
}

impl std::ops::Mul for TypedExpr {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        TypedExpr::mul(&self, &rhs).unwrap_or_else(|err| {
            panic!("typed mul failed: {err}");
        })
    }
}

impl std::ops::Div for TypedExpr {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        TypedExpr::div(&self, &rhs).unwrap_or_else(|err| {
            panic!("typed div failed: {err}");
        })
    }
}

impl std::ops::Neg for TypedExpr {
    type Output = Self;

    fn neg(self) -> Self::Output {
        TypedExpr::neg(&self).unwrap_or_else(|err| {
            panic!("typed neg failed: {err}");
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::codegen::dsl::types::{DslType, Shape};
    use crate::solver::codegen::dsl::units::UnitDim;

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

    #[test]
    fn typed_expr_sqrt_emits_fractional_units() {
        let length = TypedExpr::ident("x", DslType::f32(), UnitDim::new(0, 1, 0));
        let sqrt = length.sqrt().expect("sqrt");
        assert_eq!(sqrt.unit * sqrt.unit, length.unit);
        assert_eq!(sqrt.expr.to_string(), "sqrt(x)");
    }
}
