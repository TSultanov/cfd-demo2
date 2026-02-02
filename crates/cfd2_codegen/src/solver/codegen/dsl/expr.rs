use std::fmt;
use std::marker::PhantomData;

use crate::solver::codegen::wgsl_ast::Expr;
use cfd2_ir::solver::dimensions::{DivDim, MulDim, SqrtDim, UnitDimension};

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

// ============================================================================
// Dynamic Expression (runtime unit tracking) - Escape hatch for dynamic units
// ============================================================================

/// A dynamically-typed expression with runtime unit checking.
///
/// This is the escape hatch for when units are not known at compile time,
/// such as when working with runtime slot specifications.
#[derive(Debug, Clone, PartialEq)]
pub struct DynExpr {
    pub expr: Expr,
    pub ty: DslType,
    pub unit: UnitDim,
}

impl DynExpr {
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

    /// Cast this dynamic expression to a typed expression.
    ///
    /// Returns an error if the runtime unit does not match the expected type-level unit.
    pub fn cast_to<D: UnitDimension>(&self) -> Result<TypedExpr<D>, DslError> {
        if self.unit != D::UNIT {
            return Err(DslError::UnitMismatch {
                op: "cast_to",
                lhs: self.unit,
                rhs: D::UNIT,
            });
        }
        Ok(TypedExpr::new(self.expr, self.ty))
    }
}

impl std::ops::Add for DynExpr {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        DynExpr::add(&self, &rhs).unwrap_or_else(|err| {
            panic!("typed add failed: {err}");
        })
    }
}

impl std::ops::Sub for DynExpr {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        DynExpr::sub(&self, &rhs).unwrap_or_else(|err| {
            panic!("typed sub failed: {err}");
        })
    }
}

impl std::ops::Mul for DynExpr {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        DynExpr::mul(&self, &rhs).unwrap_or_else(|err| {
            panic!("typed mul failed: {err}");
        })
    }
}

impl std::ops::Div for DynExpr {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        DynExpr::div(&self, &rhs).unwrap_or_else(|err| {
            panic!("typed div failed: {err}");
        })
    }
}

impl std::ops::Neg for DynExpr {
    type Output = Self;

    fn neg(self) -> Self::Output {
        DynExpr::neg(&self).unwrap_or_else(|err| {
            panic!("typed neg failed: {err}");
        })
    }
}

// ============================================================================
// Typed Expression (compile-time unit checking via type-level dimensions)
// ============================================================================

/// A type-safe expression parameterized by a dimension type `D: UnitDimension`.
///
/// Unit correctness is enforced at compile time by the Rust type system.
/// Operations like `+` and `-` require identical dimensions, while `*` and `/`
/// produce new dimension types via type-level arithmetic.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedExpr<D: UnitDimension> {
    pub expr: Expr,
    pub ty: DslType,
    _phantom: PhantomData<D>,
}

impl<D: UnitDimension> TypedExpr<D> {
    pub fn new(expr: Expr, ty: DslType) -> Self {
        Self {
            expr,
            ty,
            _phantom: PhantomData,
        }
    }

    pub fn f32(value: f32) -> Self {
        Self::new(value.into(), DslType::f32())
    }

    pub fn ident(name: &str, ty: DslType) -> Self {
        Self::new(Expr::ident(name), ty)
    }

    pub fn to_wgsl(&self) -> Expr {
        self.expr
    }

    /// Convert to a dynamic expression, erasing the type-level dimension.
    pub fn into_dyn(self) -> DynExpr {
        DynExpr::new(self.expr, self.ty, D::UNIT)
    }

    /// Get the unit dimension at runtime (for debugging/serialization).
    pub fn unit() -> UnitDim {
        D::UNIT
    }
}

// Addition: only for identical dimensions
impl<D: UnitDimension> std::ops::Add for TypedExpr<D> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.ty, rhs.ty, "type mismatch for +");
        Self::new(self.to_wgsl() + rhs.to_wgsl(), self.ty)
    }
}

// Subtraction: only for identical dimensions
impl<D: UnitDimension> std::ops::Sub for TypedExpr<D> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.ty, rhs.ty, "type mismatch for -");
        Self::new(self.to_wgsl() - rhs.to_wgsl(), self.ty)
    }
}

// Negation
impl<D: UnitDimension> std::ops::Neg for TypedExpr<D> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.to_wgsl(), self.ty)
    }
}

// Multiplication: yields MulDim<A, B>
impl<A: UnitDimension, B: UnitDimension> std::ops::Mul<TypedExpr<B>> for TypedExpr<A> {
    type Output = TypedExpr<MulDim<A, B>>;

    fn mul(self, rhs: TypedExpr<B>) -> Self::Output {
        match (self.ty.shape, rhs.ty.shape) {
            // scalar * anything -> anything's type
            (Shape::Scalar, _) => TypedExpr::new(self.to_wgsl() * rhs.to_wgsl(), rhs.ty),
            // anything * scalar -> self's type
            (_, Shape::Scalar) => TypedExpr::new(self.to_wgsl() * rhs.to_wgsl(), self.ty),
            _ => panic!(
                "unsupported shape combination for *: {:?} and {:?}",
                self.ty.shape, rhs.ty.shape
            ),
        }
    }
}

// Division: yields DivDim<A, B>
impl<A: UnitDimension, B: UnitDimension> std::ops::Div<TypedExpr<B>> for TypedExpr<A> {
    type Output = TypedExpr<DivDim<A, B>>;

    fn div(self, rhs: TypedExpr<B>) -> Self::Output {
        match rhs.ty.shape {
            // anything / scalar -> self's type
            Shape::Scalar => TypedExpr::new(self.to_wgsl() / rhs.to_wgsl(), self.ty),
            _ => panic!("unsupported divisor shape for /: {:?}", rhs.ty.shape),
        }
    }
}

// Extension trait for sqrt operation
pub trait TypedSqrt {
    type Output;
    fn sqrt(self) -> Self::Output;
}

impl<D: UnitDimension> TypedSqrt for TypedExpr<D> {
    type Output = TypedExpr<SqrtDim<D>>;

    fn sqrt(self) -> Self::Output {
        match (self.ty.scalar, self.ty.shape) {
            (ScalarType::F32, Shape::Scalar | Shape::Vec(_)) => {
                TypedExpr::new(Expr::call_named("sqrt", vec![self.to_wgsl()]), self.ty)
            }
            _ => panic!("unsupported type for sqrt: {:?}", self.ty),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::codegen::dsl::types::{DslType, Shape};
    use crate::solver::codegen::dsl::units::UnitDim;
    use cfd2_ir::solver::dimensions::{Dimensionless, Length, Time, Velocity};

    // ============================================================================
    // Dynamic expression tests (runtime unit checking)
    // ============================================================================

    #[test]
    fn dyn_expr_rejects_unit_mismatch_on_add() {
        let a = DynExpr::f32(1.0, UnitDim::new(0, 1, 0));
        let b = DynExpr::f32(2.0, UnitDim::new(0, 0, 0));
        let err = a.add(&b).unwrap_err();
        assert!(matches!(err, DslError::UnitMismatch { .. }));
    }

    #[test]
    fn dyn_expr_allows_scalar_times_vector() {
        let scalar = DynExpr::f32(2.0, UnitDim::dimensionless());
        let vec = DynExpr::ident("u", DslType::vec2_f32(), UnitDim::new(0, 1, -1));
        let out = scalar.mul(&vec).expect("scalar * vec");
        assert_eq!(out.ty.shape, Shape::Vec(2));
        assert_eq!(out.unit, UnitDim::new(0, 1, -1));
        assert_eq!(out.expr.to_string(), "2.0 * u");
    }

    #[test]
    fn dyn_expr_sqrt_emits_fractional_units() {
        let length = DynExpr::ident("x", DslType::f32(), UnitDim::new(0, 1, 0));
        let sqrt = length.sqrt().expect("sqrt");
        assert_eq!(sqrt.unit * sqrt.unit, length.unit);
        assert_eq!(sqrt.expr.to_string(), "sqrt(x)");
    }

    #[test]
    fn dyn_expr_cast_to_typed_succeeds_when_units_match() {
        let dyn_expr = DynExpr::f32(3.25, UnitDim::new(0, 1, 0));
        let typed: TypedExpr<Length> = dyn_expr.cast_to().expect("cast should succeed");
        assert_eq!(typed.ty, DslType::f32());
        assert_eq!(typed.expr.to_string(), "3.25");
    }

    #[test]
    fn dyn_expr_cast_to_typed_fails_when_units_mismatch() {
        let dyn_expr = DynExpr::f32(3.25, UnitDim::new(0, 0, 1)); // Time units
        let result: Result<TypedExpr<Length>, _> = dyn_expr.cast_to();
        assert!(matches!(result, Err(DslError::UnitMismatch { .. })));
    }

    // ============================================================================
    // Typed expression tests (compile-time unit checking)
    // ============================================================================

    #[test]
    fn typed_expr_length_addition_requires_same_dimension() {
        let a: TypedExpr<Length> = TypedExpr::f32(1.0);
        let b: TypedExpr<Length> = TypedExpr::f32(2.0);
        let c = a + b;
        assert_eq!(c.expr.to_string(), "1.0 + 2.0");
    }

    #[test]
    fn typed_expr_length_div_time_is_velocity() {
        let length: TypedExpr<Length> = TypedExpr::ident("x", DslType::f32());
        let time: TypedExpr<Time> = TypedExpr::ident("t", DslType::f32());
        let velocity: TypedExpr<Velocity> = length / time;
        assert_eq!(velocity.expr.to_string(), "x / t");
        // Verify the unit matches Velocity
        assert_eq!(<Velocity as UnitDimension>::UNIT, UnitDim::new(0, 1, -1));
    }

    #[test]
    fn typed_expr_sqrt_produces_sqrt_dim() {
        let length: TypedExpr<Length> = TypedExpr::ident("x", DslType::f32());
        let sqrt_len = length.sqrt();

        // Verify the expression is correct
        assert_eq!(sqrt_len.expr.to_string(), "sqrt(x)");

        // Verify the dimension is SqrtDim<Length>
        // sqrt(Length) has L exponent of 1/2
        assert_eq!(<SqrtDim<Length> as UnitDimension>::L, (1, 2));

        // Verify that (sqrt(Length))^2 = Length
        type SqrtLengthSquared = MulDim<SqrtDim<Length>, SqrtDim<Length>>;
        assert_eq!(<SqrtLengthSquared as UnitDimension>::L, (1, 1));
    }

    #[test]
    fn typed_expr_into_dyn_preserves_unit() {
        let length: TypedExpr<Length> = TypedExpr::f32(5.0);
        let dyn_expr = length.into_dyn();
        assert_eq!(dyn_expr.unit, UnitDim::new(0, 1, 0));
        assert_eq!(dyn_expr.ty, DslType::f32());
    }

    #[test]
    fn typed_expr_multiplication_combines_dimensions() {
        let mass: TypedExpr<cfd2_ir::solver::dimensions::Mass> =
            TypedExpr::ident("m", DslType::f32());
        let velocity: TypedExpr<Velocity> = TypedExpr::ident("v", DslType::f32());

        // Mass * Velocity = MomentumDensity (or Mass * L / T)
        type Momentum = MulDim<cfd2_ir::solver::dimensions::Mass, Velocity>;
        let _momentum: TypedExpr<Momentum> = mass * velocity;

        // Momentum should have M: 1, L: 1, T: -1
        assert_eq!(<Momentum as UnitDimension>::M, (1, 1));
        assert_eq!(<Momentum as UnitDimension>::L, (1, 1));
        assert_eq!(<Momentum as UnitDimension>::T, (-1, 1));
    }

    #[test]
    fn typed_expr_scalar_times_vector_preserves_vector_shape() {
        let scalar: TypedExpr<Dimensionless> = TypedExpr::f32(2.0);
        let vec: TypedExpr<Velocity> = TypedExpr::ident("u", DslType::vec2_f32());
        // scalar * vec produces MulDim<Dimensionless, Velocity> which equals Velocity
        let result: TypedExpr<MulDim<Dimensionless, Velocity>> = scalar * vec;

        assert_eq!(result.ty.shape, Shape::Vec(2));
        assert_eq!(result.expr.to_string(), "2.0 * u");
    }

    #[test]
    fn typed_expr_dimensionless_operations() {
        let a: TypedExpr<Dimensionless> = TypedExpr::f32(0.5);
        let b: TypedExpr<Dimensionless> = TypedExpr::f32(0.3);

        let sum = a.clone() + b.clone();
        assert_eq!(sum.expr.to_string(), "0.5 + 0.3");

        // Dimensionless * Dimensionless produces MulDim<Dimensionless, Dimensionless>
        // which has the same unit (dimensionless)
        let _product: TypedExpr<MulDim<Dimensionless, Dimensionless>> = a * b;
        assert_eq!(
            <Dimensionless as UnitDimension>::UNIT,
            UnitDim::dimensionless()
        );
    }
}
