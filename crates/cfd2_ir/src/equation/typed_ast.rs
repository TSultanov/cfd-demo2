/// Typed IR builder layer for compile-time unit dimension checking.
///
/// This module provides type-level wrappers around the untyped AST that enforce
/// unit consistency at compile time. The typed wrappers erase to the existing
/// untyped IR structs, allowing incremental adoption without changing codegen.
///
/// # Example
///
/// ```rust,ignore
/// use cfd2_ir::equation::typed_ast::*;
/// use cfd2_ir::dimensions::*;
///
/// // Create typed field references
/// let p = TypedFieldRef::<Pressure, Scalar>::new("p");
/// let u = TypedFieldRef::<Velocity, Vector2>::new("U");
///
/// // Build equation with compile-time unit checking
/// let eqn = (typed_fvm::ddt(u.clone()) + typed_fvm::div(phi, u.clone())).eqn(u.clone());
/// ```
use std::marker::PhantomData;
use std::ops::Add;

use crate::dimensions::UnitDimension;
use crate::equation::ast::{
    Coefficient, Discretization, Equation, EquationSystem, FieldKind, FieldRef, FluxRef, Term,
    TermOp, TermSum,
};

// ============================================================================
// Kind Markers
// ============================================================================

/// Type-level marker for scalar fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scalar;

/// Type-level marker for 2D vector fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vector2;

/// Type-level marker for 3D vector fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vector3;

/// Trait for type-level field kinds that map to runtime FieldKind.
pub trait Kind: 'static + Copy + Send + Sync + Eq + PartialEq + std::fmt::Debug {
    /// The corresponding runtime FieldKind.
    const RUNTIME: FieldKind;
}

impl Kind for Scalar {
    const RUNTIME: FieldKind = FieldKind::Scalar;
}

impl Kind for Vector2 {
    const RUNTIME: FieldKind = FieldKind::Vector2;
}

impl Kind for Vector3 {
    const RUNTIME: FieldKind = FieldKind::Vector3;
}

// ============================================================================
// Typed Field and Flux References
// ============================================================================

/// Typed field reference with compile-time dimension and kind checking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TypedFieldRef<D: UnitDimension, K: Kind> {
    name: &'static str,
    _dim: PhantomData<D>,
    _kind: PhantomData<K>,
}

impl<D: UnitDimension, K: Kind> TypedFieldRef<D, K> {
    /// Create a new typed field reference.
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            _dim: PhantomData,
            _kind: PhantomData,
        }
    }

    /// Convert to the underlying untyped FieldRef.
    pub fn to_untyped(self) -> FieldRef {
        FieldRef::new(self.name, K::RUNTIME, D::UNIT)
    }

    /// Get the field name.
    pub fn name(&self) -> &'static str {
        self.name
    }
}

/// Typed flux reference with compile-time dimension and kind checking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TypedFluxRef<D: UnitDimension, K: Kind> {
    name: &'static str,
    _dim: PhantomData<D>,
    _kind: PhantomData<K>,
}

impl<D: UnitDimension, K: Kind> TypedFluxRef<D, K> {
    /// Create a new typed flux reference.
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            _dim: PhantomData,
            _kind: PhantomData,
        }
    }

    /// Convert to the underlying untyped FluxRef.
    pub fn to_untyped(self) -> FluxRef {
        FluxRef::new(self.name, K::RUNTIME, D::UNIT)
    }

    /// Get the flux name.
    pub fn name(&self) -> &'static str {
        self.name
    }
}

// ============================================================================
// Typed Coefficient
// ============================================================================

/// Typed coefficient with compile-time dimension checking.
///
/// Only scalar fields can be used as coefficients (enforced at compile time).
#[derive(Clone, Debug, PartialEq)]
pub struct TypedCoeff<D: UnitDimension> {
    inner: Coefficient,
    _dim: PhantomData<D>,
}

impl<D: UnitDimension> TypedCoeff<D> {
    /// Create from a constant value.
    pub fn constant(value: f64) -> Self {
        Self {
            inner: Coefficient::constant_unit(value, D::UNIT),
            _dim: PhantomData,
        }
    }

    /// Create from a scalar field reference.
    ///
    /// This is only available for scalar-typed fields, enforcing the constraint
    /// that coefficients must be scalars at compile time.
    pub fn from_field(field: TypedFieldRef<D, Scalar>) -> Self {
        // We know the field is scalar by construction, so this always succeeds
        let untyped = field.to_untyped();
        Self {
            inner: Coefficient::Field(untyped),
            _dim: PhantomData,
        }
    }

    /// Create a magnitude-squared coefficient from any field.
    /// Returns a coefficient with squared dimension: D * D
    pub fn mag_sqr<K: Kind>(
        field: TypedFieldRef<D, K>,
    ) -> TypedCoeff<crate::dimensions::MulDim<D, D>> {
        TypedCoeff {
            inner: Coefficient::mag_sqr(field.to_untyped()),
            _dim: PhantomData,
        }
    }

    /// Multiply two coefficients.
    pub fn multiply<OtherD: UnitDimension>(
        self,
        other: TypedCoeff<OtherD>,
    ) -> TypedCoeff<crate::dimensions::MulDim<D, OtherD>> {
        TypedCoeff {
            inner: Coefficient::Product(Box::new(self.inner), Box::new(other.inner)),
            _dim: PhantomData,
        }
    }

    /// Convert to the underlying untyped Coefficient.
    pub fn to_untyped(&self) -> Coefficient {
        self.inner.clone()
    }
}

impl<D, OtherD> std::ops::Mul<TypedCoeff<OtherD>> for TypedCoeff<D>
where
    D: UnitDimension,
    OtherD: UnitDimension,
{
    type Output = TypedCoeff<crate::dimensions::MulDim<D, OtherD>>;

    fn mul(self, other: TypedCoeff<OtherD>) -> Self::Output {
        TypedCoeff {
            inner: Coefficient::Product(Box::new(self.inner), Box::new(other.inner)),
            _dim: PhantomData,
        }
    }
}

// ============================================================================
// Typed Term
// ============================================================================

/// Typed term with compile-time integrated unit dimension checking.
///
/// The `D` type parameter represents the integrated unit dimension of the term
/// (following the `Term::integrated_unit()` logic in ast.rs).
#[derive(Clone, Debug, PartialEq)]
pub struct TypedTerm<D: UnitDimension> {
    inner: Term,
    _dim: PhantomData<D>,
}

impl<D: UnitDimension> TypedTerm<D> {
    /// Convert to the underlying untyped Term.
    pub fn to_untyped(&self) -> Term {
        self.inner.clone()
    }

    /// Declare the discretization scheme for this term as part of the model's
    /// math declaration. Takes precedence over the registry default and the
    /// runtime scheme knob (codegen bakes the declared scheme as a literal).
    pub fn scheme(mut self, scheme: crate::scheme::Scheme) -> Self {
        self.inner = self.inner.with_scheme(scheme);
        self
    }

    /// Declare the bounded convection form: assembly subtracts the
    /// continuity defect `(div phi) * phi_P` from the diagonal so convection
    /// stays bounded while the flux is not exactly divergence-free
    /// (OpenFOAM's `bounded Gauss`). Only meaningful on implicit div terms.
    pub fn bounded(mut self) -> Self {
        self.inner = self.inner.with_bounded();
        self
    }

    /// Create an equation from this single term.
    ///
    /// The target field is the unknown being solved for. Its dimension can be
    /// different from the term's integrated unit dimension.
    pub fn eqn<TargetD: UnitDimension, K: Kind>(
        self,
        target: TypedFieldRef<TargetD, K>,
    ) -> Equation {
        self.inner.eqn(target.to_untyped())
    }

    /// Cast this term to a different dimension type.
    ///
    /// This is useful when two dimension types are semantically equivalent
    /// but different at the type level (e.g., due to unnormalized type expressions).
    /// The cast is safe as long as the runtime unit dimensions match.
    ///
    /// # Panics
    ///
    /// Panics if `DFrom::UNIT != DTo::UNIT`.
    pub fn cast_to<DTo: UnitDimension>(self) -> TypedTerm<DTo> {
        assert!(
            D::UNIT == DTo::UNIT,
            "Cannot cast term from dimension {:?} to {:?} - units do not match",
            D::UNIT,
            DTo::UNIT
        );
        TypedTerm {
            inner: self.inner,
            _dim: PhantomData,
        }
    }
}

impl<D: UnitDimension> Add<TypedTerm<D>> for TypedTerm<D> {
    type Output = TypedTermSum<D>;

    fn add(self, rhs: TypedTerm<D>) -> Self::Output {
        TypedTermSum {
            inner: self.inner + rhs.inner,
            _dim: PhantomData,
        }
    }
}

// ============================================================================
// Typed Term Sum
// ============================================================================

/// Typed term sum with compile-time unit dimension checking.
///
/// All terms in the sum must have the same integrated unit dimension `D`.
#[derive(Clone, Debug, PartialEq)]
pub struct TypedTermSum<D: UnitDimension> {
    inner: TermSum,
    _dim: PhantomData<D>,
}

impl<D: UnitDimension> TypedTermSum<D> {
    /// Convert to the underlying untyped TermSum.
    pub fn to_untyped(&self) -> TermSum {
        self.inner.clone()
    }

    /// Create an equation from this term sum.
    ///
    /// The target field is the unknown being solved for. Its dimension can be
    /// different from the term sum's integrated unit dimension.
    pub fn eqn<TargetD: UnitDimension, K: Kind>(
        self,
        target: TypedFieldRef<TargetD, K>,
    ) -> Equation {
        self.inner.eqn(target.to_untyped())
    }

    /// Cast this term sum to a different dimension type.
    ///
    /// This is useful when two dimension types are semantically equivalent
    /// but different at the type level (e.g., due to unnormalized type expressions).
    /// The cast is safe as long as the runtime unit dimensions match.
    ///
    /// # Panics
    ///
    /// Panics if `DFrom::UNIT != DTo::UNIT`.
    pub fn cast_to<DTo: UnitDimension>(self) -> TypedTermSum<DTo> {
        assert!(
            D::UNIT == DTo::UNIT,
            "Cannot cast term sum from dimension {:?} to {:?} - units do not match",
            D::UNIT,
            DTo::UNIT
        );
        TypedTermSum {
            inner: self.inner,
            _dim: PhantomData,
        }
    }
}

impl<D: UnitDimension> Add<TypedTerm<D>> for TypedTermSum<D> {
    type Output = TypedTermSum<D>;

    fn add(mut self, rhs: TypedTerm<D>) -> Self::Output {
        self.inner = self.inner + rhs.inner;
        self
    }
}

// ============================================================================
// Typed Equation System Builder
// ============================================================================

/// Builder for creating equation systems with typed terms.
#[derive(Clone, Debug, Default)]
pub struct TypedEquationSystem {
    inner: EquationSystem,
}

impl TypedEquationSystem {
    /// Create a new empty typed equation system.
    pub fn new() -> Self {
        Self {
            inner: EquationSystem::new(),
        }
    }

    /// Add an equation to the system.
    pub fn add_equation(&mut self, equation: Equation) {
        self.inner.add_equation(equation);
    }

    /// Add a typed equation to the system.
    pub fn add_typed_equation<D: UnitDimension, K: Kind>(&mut self, equation: TypedEquation<D, K>) {
        self.inner.add_equation(equation.into_untyped());
    }

    /// Convert to the underlying untyped EquationSystem.
    pub fn into_untyped(self) -> EquationSystem {
        self.inner
    }

    /// Validate units in the underlying equation system.
    pub fn validate_units(
        &self,
    ) -> Result<(), crate::equation::ast::UnitValidationError> {
        self.inner.validate_units()
    }
}

/// Typed equation with compile-time unit checking.
#[derive(Clone, Debug, PartialEq)]
pub struct TypedEquation<D: UnitDimension, K: Kind> {
    target: TypedFieldRef<D, K>,
    terms: Vec<Term>,
    _dim: PhantomData<D>,
}

impl<D: UnitDimension, K: Kind> TypedEquation<D, K> {
    /// Create a new typed equation for the given target field.
    pub fn new(target: TypedFieldRef<D, K>) -> Self {
        Self {
            target,
            terms: Vec::new(),
            _dim: PhantomData,
        }
    }

    /// Add a typed term to the equation.
    pub fn with_term(mut self, term: TypedTerm<D>) -> Self {
        self.terms.push(term.to_untyped());
        self
    }

    /// Convert to the underlying untyped Equation.
    pub fn into_untyped(self) -> Equation {
        let mut eqn = Equation::new(self.target.to_untyped());
        for term in self.terms {
            eqn.add_term(term);
        }
        eqn
    }
}

// ============================================================================
// FVM Term Constructors (Implicit Discretization)
// ============================================================================

/// Type-level computation for ddt integrated unit: coeff * field_unit * volume / time
pub type DdtUnit<FieldD, CoeffD> = crate::dimensions::DivDim<
    crate::dimensions::MulDim<
        crate::dimensions::MulDim<CoeffD, FieldD>,
        crate::dimensions::Volume,
    >,
    crate::dimensions::Time,
>;

/// Type-level computation for div integrated unit: flux_unit * field_unit
pub type DivUnit<FluxD, FieldD> = crate::dimensions::MulDim<FluxD, FieldD>;

/// Type-level computation for div_flux integrated unit: flux_unit
pub type DivFluxUnit<FluxD> = FluxD;

/// Type-level computation for grad integrated unit: field_unit * area
pub type GradUnit<FieldD> =
    crate::dimensions::MulDim<FieldD, crate::dimensions::Area>;

/// Type-level computation for laplacian integrated unit: coeff * field_unit * area / length
pub type LaplacianUnit<FieldD, CoeffD> = crate::dimensions::DivDim<
    crate::dimensions::MulDim<
        crate::dimensions::MulDim<CoeffD, FieldD>,
        crate::dimensions::Area,
    >,
    crate::dimensions::Length,
>;

/// Type-level computation for implicit source integrated unit: coeff * field_unit * volume
pub type SourceImplicitUnit<FieldD, CoeffD> = crate::dimensions::MulDim<
    crate::dimensions::MulDim<CoeffD, FieldD>,
    crate::dimensions::Volume,
>;

/// Type-level computation for the static implicit-diagonal integrated unit:
/// coeff * field_unit (the `sp` form drops the volume factor of the ordinary
/// implicit source).
pub type SourceStaticUnit<FieldD, CoeffD> = crate::dimensions::MulDim<CoeffD, FieldD>;

/// Type-level computation for explicit source integrated unit: coeff * volume
pub type SourceExplicitUnit<CoeffD> =
    crate::dimensions::MulDim<CoeffD, crate::dimensions::Volume>;

pub mod typed_fvm {
    use super::*;

    /// Time derivative term: ddt(field) or ddt(coeff * field)
    ///
    /// Integrated unit: coeff_unit * field_unit * volume / time
    pub fn ddt<D: UnitDimension, K: Kind>(
        field: TypedFieldRef<D, K>,
    ) -> TypedTerm<DdtUnit<D, crate::dimensions::Dimensionless>> {
        TypedTerm {
            inner: Term::new(
                TermOp::Ddt,
                Discretization::Implicit,
                field.to_untyped(),
                None,
                None,
            ),
            _dim: PhantomData,
        }
    }

    /// Time derivative term with coefficient: ddt(coeff * field)
    pub fn ddt_coeff<FieldD: UnitDimension, CoeffD: UnitDimension, K: Kind>(
        coeff: TypedCoeff<CoeffD>,
        field: TypedFieldRef<FieldD, K>,
    ) -> TypedTerm<DdtUnit<FieldD, CoeffD>> {
        TypedTerm {
            inner: Term::new(
                TermOp::Ddt,
                Discretization::Implicit,
                field.to_untyped(),
                None,
                Some(coeff.to_untyped()),
            ),
            _dim: PhantomData,
        }
    }

    /// Divergence term: div(flux, field)
    ///
    /// Integrated unit: flux_unit * field_unit
    /// Note: flux and field can have different kinds (e.g., scalar flux with vector field)
    pub fn div<FluxD: UnitDimension, FieldD: UnitDimension, FluxK: Kind, FieldK: Kind>(
        flux: TypedFluxRef<FluxD, FluxK>,
        field: TypedFieldRef<FieldD, FieldK>,
    ) -> TypedTerm<DivUnit<FluxD, FieldD>> {
        TypedTerm {
            inner: Term::new(
                TermOp::Div,
                Discretization::Implicit,
                field.to_untyped(),
                Some(flux.to_untyped()),
                None,
            ),
            _dim: PhantomData,
        }
    }

    /// Divergence of flux term: div_flux(flux, field)
    ///
    /// Integrated unit: flux_unit (requires flux.kind == field.kind)
    pub fn div_flux<FluxD: UnitDimension, FieldD: UnitDimension, K: Kind>(
        flux: TypedFluxRef<FluxD, K>,
        field: TypedFieldRef<FieldD, K>,
    ) -> TypedTerm<DivFluxUnit<FluxD>> {
        TypedTerm {
            inner: Term::new(
                TermOp::DivFlux,
                Discretization::Implicit,
                field.to_untyped(),
                Some(flux.to_untyped()),
                None,
            ),
            _dim: PhantomData,
        }
    }

    /// Gradient term: grad(field)
    ///
    /// Integrated unit: field_unit * area
    pub fn grad<D: UnitDimension, K: Kind>(field: TypedFieldRef<D, K>) -> TypedTerm<GradUnit<D>> {
        TypedTerm {
            inner: Term::new(
                TermOp::Grad,
                Discretization::Implicit,
                field.to_untyped(),
                None,
                None,
            ),
            _dim: PhantomData,
        }
    }

    /// Laplacian term: laplacian(coeff, field)
    ///
    /// Integrated unit: coeff * field_unit * area / length
    pub fn laplacian<FieldD: UnitDimension, CoeffD: UnitDimension, K: Kind>(
        coeff: TypedCoeff<CoeffD>,
        field: TypedFieldRef<FieldD, K>,
    ) -> TypedTerm<LaplacianUnit<FieldD, CoeffD>> {
        TypedTerm {
            inner: Term::new(
                TermOp::Laplacian,
                Discretization::Implicit,
                field.to_untyped(),
                None,
                Some(coeff.to_untyped()),
            ),
            _dim: PhantomData,
        }
    }

    /// Source term: source(field)
    ///
    /// For implicit discretization: field_unit * volume
    pub fn source<D: UnitDimension, K: Kind>(
        field: TypedFieldRef<D, K>,
    ) -> TypedTerm<SourceImplicitUnit<D, crate::dimensions::Dimensionless>> {
        TypedTerm {
            inner: Term::new(
                TermOp::Source,
                Discretization::Implicit,
                field.to_untyped(),
                None,
                None,
            ),
            _dim: PhantomData,
        }
    }

    /// Source term with coefficient: source_coeff(coeff, field)
    ///
    /// For implicit discretization: coeff * field_unit * volume
    pub fn source_coeff<FieldD: UnitDimension, CoeffD: UnitDimension, K: Kind>(
        coeff: TypedCoeff<CoeffD>,
        field: TypedFieldRef<FieldD, K>,
    ) -> TypedTerm<SourceImplicitUnit<FieldD, CoeffD>> {
        TypedTerm {
            inner: Term::new(
                TermOp::Source,
                Discretization::Implicit,
                field.to_untyped(),
                None,
                Some(coeff.to_untyped()),
            ),
            _dim: PhantomData,
        }
    }

    /// Static implicit-diagonal source ("Sp"): contributes `coeff` to the
    /// matrix diagonal of `field` directly, WITHOUT the cell-volume factor of
    /// `source_coeff`. Use this for pointwise identity/constraint rows whose
    /// magnitude must not scale with cell volume — e.g. defining an auxiliary
    /// unknown `lap_X` via `(sp(-1, lap_X) + laplacian(1, X)).eqn(lap_X)`,
    /// where the un-volumed diagonal must balance the surface-flux Laplacian
    /// of `X` on the same row.
    ///
    /// Integrated unit: coeff * field_unit (no volume).
    pub fn sp<FieldD: UnitDimension, CoeffD: UnitDimension, K: Kind>(
        coeff: TypedCoeff<CoeffD>,
        field: TypedFieldRef<FieldD, K>,
    ) -> TypedTerm<SourceStaticUnit<FieldD, CoeffD>> {
        TypedTerm {
            inner: Term::new(
                TermOp::Source,
                Discretization::Implicit,
                field.to_untyped(),
                None,
                Some(coeff.to_untyped()),
            )
            .with_static_diag(),
            _dim: PhantomData,
        }
    }
}

// ============================================================================
// FVC Term Constructors (Explicit Discretization)
// ============================================================================

pub mod typed_fvc {
    use super::*;

    /// Explicit time derivative term.
    pub fn ddt<D: UnitDimension, K: Kind>(
        field: TypedFieldRef<D, K>,
    ) -> TypedTerm<DdtUnit<D, crate::dimensions::Dimensionless>> {
        TypedTerm {
            inner: Term::new(
                TermOp::Ddt,
                Discretization::Explicit,
                field.to_untyped(),
                None,
                None,
            ),
            _dim: PhantomData,
        }
    }

    /// Explicit time derivative term with coefficient.
    pub fn ddt_coeff<FieldD: UnitDimension, CoeffD: UnitDimension, K: Kind>(
        coeff: TypedCoeff<CoeffD>,
        field: TypedFieldRef<FieldD, K>,
    ) -> TypedTerm<DdtUnit<FieldD, CoeffD>> {
        TypedTerm {
            inner: Term::new(
                TermOp::Ddt,
                Discretization::Explicit,
                field.to_untyped(),
                None,
                Some(coeff.to_untyped()),
            ),
            _dim: PhantomData,
        }
    }

    /// Explicit divergence term.
    /// Note: flux and field can have different kinds (e.g., scalar flux with vector field)
    pub fn div<FluxD: UnitDimension, FieldD: UnitDimension, FluxK: Kind, FieldK: Kind>(
        flux: TypedFluxRef<FluxD, FluxK>,
        field: TypedFieldRef<FieldD, FieldK>,
    ) -> TypedTerm<DivUnit<FluxD, FieldD>> {
        TypedTerm {
            inner: Term::new(
                TermOp::Div,
                Discretization::Explicit,
                field.to_untyped(),
                Some(flux.to_untyped()),
                None,
            ),
            _dim: PhantomData,
        }
    }

    /// Explicit divergence of flux term.
    pub fn div_flux<FluxD: UnitDimension, FieldD: UnitDimension, K: Kind>(
        flux: TypedFluxRef<FluxD, K>,
        field: TypedFieldRef<FieldD, K>,
    ) -> TypedTerm<DivFluxUnit<FluxD>> {
        TypedTerm {
            inner: Term::new(
                TermOp::DivFlux,
                Discretization::Explicit,
                field.to_untyped(),
                Some(flux.to_untyped()),
                None,
            ),
            _dim: PhantomData,
        }
    }

    /// Explicit gradient term.
    pub fn grad<D: UnitDimension, K: Kind>(field: TypedFieldRef<D, K>) -> TypedTerm<GradUnit<D>> {
        TypedTerm {
            inner: Term::new(
                TermOp::Grad,
                Discretization::Explicit,
                field.to_untyped(),
                None,
                None,
            ),
            _dim: PhantomData,
        }
    }

    /// Explicit laplacian term.
    pub fn laplacian<FieldD: UnitDimension, CoeffD: UnitDimension, K: Kind>(
        coeff: TypedCoeff<CoeffD>,
        field: TypedFieldRef<FieldD, K>,
    ) -> TypedTerm<LaplacianUnit<FieldD, CoeffD>> {
        TypedTerm {
            inner: Term::new(
                TermOp::Laplacian,
                Discretization::Explicit,
                field.to_untyped(),
                None,
                Some(coeff.to_untyped()),
            ),
            _dim: PhantomData,
        }
    }

    /// Explicit transpose/deviatoric viscous correction:
    /// `-div(coeff * dev2((grad field)^T))` with `dev2(A) = A - (2/3) tr(A) I`.
    ///
    /// Declared alongside `typed_fvm::laplacian(coeff, field)` this completes
    /// the full viscous stress divergence (OpenFOAM laminar UEqn form:
    /// `-fvm::laplacian(nu,U) - fvc::div(nu*dev2(T(fvc::grad(U))))`). The
    /// assembly evaluates the face flux from `grad_state` cell gradients;
    /// declaring the term forces the gradients pipeline on unconditionally.
    /// Vector2 targets only. Shares `LaplacianUnit` (coeff·field·area/length),
    /// so it casts to the equation unit exactly like the laplacian term.
    pub fn div_dev2_grad_transpose<FieldD: UnitDimension, CoeffD: UnitDimension>(
        coeff: TypedCoeff<CoeffD>,
        field: TypedFieldRef<FieldD, Vector2>,
    ) -> TypedTerm<LaplacianUnit<FieldD, CoeffD>> {
        TypedTerm {
            inner: Term::new(
                TermOp::Laplacian,
                Discretization::Explicit,
                field.to_untyped(),
                None,
                Some(coeff.to_untyped()),
            )
            .with_transpose_dev2(),
            _dim: PhantomData,
        }
    }

    /// Explicit source term.
    ///
    /// For explicit discretization: volume (field-independent, coeff = dimensionless)
    pub fn source<D: UnitDimension, K: Kind>(
        field: TypedFieldRef<D, K>,
    ) -> TypedTerm<SourceExplicitUnit<crate::dimensions::Dimensionless>> {
        TypedTerm {
            inner: Term::new(
                TermOp::Source,
                Discretization::Explicit,
                field.to_untyped(),
                None,
                None,
            ),
            _dim: PhantomData,
        }
    }

    /// Explicit source term with coefficient.
    ///
    /// For explicit discretization: coeff * volume (field-independent)
    pub fn source_coeff<FieldD: UnitDimension, CoeffD: UnitDimension, K: Kind>(
        coeff: TypedCoeff<CoeffD>,
        field: TypedFieldRef<FieldD, K>,
    ) -> TypedTerm<SourceExplicitUnit<CoeffD>> {
        TypedTerm {
            inner: Term::new(
                TermOp::Source,
                Discretization::Explicit,
                field.to_untyped(),
                None,
                Some(coeff.to_untyped()),
            ),
            _dim: PhantomData,
        }
    }

    /// Explicit directional source on a vector target: component `c` of the
    /// target receives `coeff * direction[c] * V`. The scalar coefficient
    /// tree is evaluated at the current state (so it may reference solved
    /// fields, Picard-style), and `direction` carries the constant
    /// per-component multipliers (e.g. gravity direction). The declared
    /// coefficient dimension must therefore account for the direction's
    /// physical magnitude (treat `direction` as dimensionless).
    pub fn source_directional<TargetD: UnitDimension, CoeffD: UnitDimension>(
        coeff: TypedCoeff<CoeffD>,
        direction: [f64; 2],
        target: TypedFieldRef<TargetD, Vector2>,
    ) -> TypedTerm<SourceExplicitUnit<CoeffD>> {
        TypedTerm {
            inner: Term::new(
                TermOp::Source,
                Discretization::Explicit,
                target.to_untyped(),
                None,
                Some(coeff.to_untyped()),
            )
            .with_direction(direction.to_vec()),
            _dim: PhantomData,
        }
    }

    /// Explicit per-component source from a vector field.
    ///
    /// Each component of `source` contributes `source_c * V` to the matching
    /// component of `target`'s RHS (the kinds must match; assembly reads the
    /// source field component-wise). This is the vector-equation counterpart
    /// of `source_coeff` with a field coefficient — a scalar coefficient on a
    /// vector target would otherwise be broadcast, which assembly rejects.
    pub fn source_vector<SourceD: UnitDimension, TargetD: UnitDimension, K: Kind>(
        source: TypedFieldRef<SourceD, K>,
        target: TypedFieldRef<TargetD, K>,
    ) -> TypedTerm<SourceExplicitUnit<SourceD>> {
        TypedTerm {
            inner: Term::new(
                TermOp::Source,
                Discretization::Explicit,
                target.to_untyped(),
                None,
                Some(Coefficient::Field(source.to_untyped())),
            ),
            _dim: PhantomData,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimensions::*;
    use crate::units::si;

    #[test]
    fn typed_field_ref_creates_untyped_with_correct_units() {
        let p = TypedFieldRef::<Pressure, Scalar>::new("p");
        let untyped = p.to_untyped();

        assert_eq!(untyped.name(), "p");
        assert_eq!(untyped.kind(), FieldKind::Scalar);
        assert_eq!(untyped.unit(), si::PRESSURE);
    }

    #[test]
    fn typed_flux_ref_creates_untyped_with_correct_units() {
        let phi = TypedFluxRef::<MassFlux, Scalar>::new("phi");
        let untyped = phi.to_untyped();

        assert_eq!(untyped.name(), "phi");
        assert_eq!(untyped.kind(), FieldKind::Scalar);
        assert_eq!(untyped.unit(), si::MASS_FLUX);
    }

    #[test]
    fn source_vector_builds_explicit_term_with_vector_field_coefficient() {
        // Per-component explicit source: the coefficient carries the vector
        // source field; assembly reads it component-wise.
        let src = TypedFieldRef::<Force, Vector2>::new("mms_src_U");
        let u = TypedFieldRef::<Velocity, Vector2>::new("U");
        let term = typed_fvc::source_vector(src, u).to_untyped();

        assert_eq!(term.op, TermOp::Source);
        assert_eq!(term.discretization, Discretization::Explicit);
        assert_eq!(term.field.name(), "U");
        let Some(Coefficient::Field(f)) = term.coeff else {
            panic!("expected vector-field coefficient");
        };
        assert_eq!(f.name(), "mms_src_U");
        assert_eq!(f.kind(), FieldKind::Vector2);
    }

    #[test]
    fn typed_coeff_from_scalar_field() {
        let rho = TypedFieldRef::<Density, Scalar>::new("rho");
        let coeff = TypedCoeff::<Density>::from_field(rho);

        assert_eq!(coeff.to_untyped().unit(), si::DENSITY);
    }

    #[test]
    fn typed_coeff_constant() {
        let coeff = TypedCoeff::<Pressure>::constant(101325.0);

        assert_eq!(coeff.to_untyped().unit(), si::PRESSURE);
    }

    #[test]
    fn typed_ddt_term_has_correct_integrated_unit() {
        // ddt(U) where U is velocity
        // Integrated unit: velocity * volume / time = (m/s) * m^3 / s = m^4 / s^2
        let u = TypedFieldRef::<Velocity, Vector2>::new("U");
        let term = typed_fvm::ddt(u);

        // Verify the term can be created and converted
        let untyped = term.to_untyped();
        assert_eq!(untyped.op, TermOp::Ddt);
        assert_eq!(untyped.discretization, Discretization::Implicit);
    }

    #[test]
    fn typed_div_term_has_correct_integrated_unit() {
        // div(phi, U) where phi is mass flux and U is velocity
        // Integrated unit: mass_flux * velocity = (kg/s) * (m/s) = kg*m/s^2
        // Note: div requires flux.kind == field.kind, so both must be Scalar or both Vector
        let phi = TypedFluxRef::<MassFlux, Vector2>::new("phi");
        let u = TypedFieldRef::<Velocity, Vector2>::new("U");
        let term = typed_fvm::div(phi, u);

        let untyped = term.to_untyped();
        assert_eq!(untyped.op, TermOp::Div);
        assert_eq!(untyped.flux.unwrap().name(), "phi");
    }

    #[test]
    fn typed_terms_can_be_added_when_units_match() {
        // Both terms must have the same integrated unit dimension
        // Use a scalar field for simpler testing
        let p = TypedFieldRef::<Pressure, Scalar>::new("p");

        // Two ddt terms on the same field - these have matching units
        let term1 = typed_fvm::ddt(p);
        let term2 = typed_fvm::ddt(p);
        let sum = term1 + term2;

        assert_eq!(sum.to_untyped().terms().len(), 2);
    }

    #[test]
    fn typed_equation_system_builds_and_validates() {
        // Build a simple pressure equation: ddt(p) = 0 (transient only)
        let p = TypedFieldRef::<Pressure, Scalar>::new("p");

        let eqn = typed_fvm::ddt(p).eqn(p);

        let mut system = TypedEquationSystem::new();
        system.add_equation(eqn);

        // Validate - single term should always pass
        assert!(system.validate_units().is_ok());
    }

    #[test]
    fn typed_equation_with_multiple_terms_validates() {
        // Build a proper momentum equation with consistent units:
        // For incompressible flow: ddt(U) + div(phi, U) - laplacian(nu, U) = -grad(p)
        //
        // All terms should have integrated unit: kg*m/s^2 (force)
        //
        // ddt(rho*U): density * velocity * volume / time = (kg/m^3)*(m/s)*m^3/s = kg*m/s^2
        // div(phi, U): mass_flux * velocity = (kg/s)*(m/s) = kg*m/s^2
        // laplacian(mu, U): viscosity * velocity * area / length = (Pa*s)*(m/s)*m^2/m = kg*m/s^2

        // Let's verify our type system computes these correctly
        type DdtMomentum = DdtUnit<MomentumDensity, Dimensionless>;
        type DivMomentum = DivUnit<MassFlux, Velocity>;

        // Check that these compute to the same dimension
        // DdtMomentum = momentum_density * volume / time = (kg/m^2/s) * m^3 / s = kg*m/s^2
        assert_eq!(DdtMomentum::M, (1, 1)); // kg
        assert_eq!(DdtMomentum::L, (1, 1)); // m
        assert_eq!(DdtMomentum::T, (-2, 1)); // s^-2

        // DivMomentum = mass_flux * velocity = (kg/s) * (m/s) = kg*m/s^2
        assert_eq!(DivMomentum::M, (1, 1)); // kg
        assert_eq!(DivMomentum::L, (1, 1)); // m
        assert_eq!(DivMomentum::T, (-2, 1)); // s^-2
    }

    #[test]
    fn typed_laplacian_term() {
        // laplacian(nu, U) where nu is kinematic viscosity
        let nu = TypedCoeff::<KinematicViscosity>::constant(1e-5);
        let u = TypedFieldRef::<Velocity, Vector2>::new("U");

        let term = typed_fvm::laplacian(nu, u);

        let untyped = term.to_untyped();
        assert_eq!(untyped.op, TermOp::Laplacian);
        assert!(untyped.coeff.is_some());
    }

    #[test]
    fn typed_source_term() {
        let u = TypedFieldRef::<Velocity, Vector2>::new("U");
        let term = typed_fvm::source(u);

        let untyped = term.to_untyped();
        assert_eq!(untyped.op, TermOp::Source);
    }

    #[test]
    fn typed_div_dev2_grad_transpose_term() {
        // Same op + units as the explicit laplacian, distinguished only by
        // the transpose_dev2 metadata flag (no new TermOp: Laplacian's
        // integrated_unit coeff*field*AREA/LENGTH is exactly right).
        let mu = TypedCoeff::<DynamicViscosity>::constant(0.01);
        let u = TypedFieldRef::<Velocity, Vector2>::new("U");

        let dev2 = typed_fvc::div_dev2_grad_transpose(mu, u).to_untyped();
        assert_eq!(dev2.op, TermOp::Laplacian);
        assert_eq!(dev2.discretization, Discretization::Explicit);
        assert!(dev2.transpose_dev2);
        assert!(dev2.coeff.is_some());

        let mu2 = TypedCoeff::<DynamicViscosity>::constant(0.01);
        let lap = typed_fvc::laplacian(mu2, u).to_untyped();
        assert!(!lap.transpose_dev2);
        assert_eq!(
            dev2.integrated_unit().unwrap(),
            lap.integrated_unit().unwrap(),
            "dev2 term must share the laplacian's integrated unit"
        );
    }

    #[test]
    fn typed_grad_term() {
        let p = TypedFieldRef::<Pressure, Scalar>::new("p");
        let term = typed_fvm::grad(p);

        let untyped = term.to_untyped();
        assert_eq!(untyped.op, TermOp::Grad);
    }

    #[test]
    fn typed_div_flux_term() {
        let phi = TypedFluxRef::<MassFlux, Scalar>::new("phi");
        // Note: div_flux requires flux.kind == field.kind, so we need a scalar field
        let rho = TypedFieldRef::<Density, Scalar>::new("rho");

        let term = typed_fvm::div_flux(phi, rho);

        let untyped = term.to_untyped();
        assert_eq!(untyped.op, TermOp::DivFlux);
    }

    #[test]
    fn typed_coefficient_multiplication() {
        let rho = TypedFieldRef::<Density, Scalar>::new("rho");
        let coeff1 = TypedCoeff::<Density>::from_field(rho);
        let coeff2 = TypedCoeff::<Dimensionless>::constant(0.5);

        let product = coeff1 * coeff2;

        // Product should have dimension: Density * Dimensionless = Density
        let untyped = product.to_untyped();
        assert_eq!(untyped.unit(), si::DENSITY);
    }

    #[test]
    fn typed_equation_system_matches_untyped_construction() {
        use crate::units::si;

        // Build the same system using both APIs and verify they match
        let p_typed = TypedFieldRef::<Pressure, Scalar>::new("p");

        // Typed construction - simple pressure equation
        let eqn_typed = typed_fvm::ddt(p_typed).eqn(p_typed);

        let mut typed_system = TypedEquationSystem::new();
        typed_system.add_equation(eqn_typed);

        // Untyped construction
        let p_untyped = FieldRef::new("p", FieldKind::Scalar, si::PRESSURE);

        let eqn_untyped = crate::equation::ast::fvm::ddt(p_untyped).eqn(p_untyped);

        let mut untyped_system = EquationSystem::new();
        untyped_system.add_equation(eqn_untyped);

        // Both should validate successfully
        assert!(typed_system.validate_units().is_ok());
        assert!(untyped_system.validate_units().is_ok());
    }

    #[test]
    fn explicit_terms_use_correct_discretization() {
        let p = TypedFieldRef::<Pressure, Scalar>::new("p");

        let ddt_term = typed_fvc::ddt(p);

        assert_eq!(
            ddt_term.to_untyped().discretization,
            Discretization::Explicit
        );
    }

    #[test]
    fn typed_div_accepts_mismatched_kinds() {
        // div(phi_scalar, u_vector) should compile and work
        // This is used in incompressible momentum: phi (mass flux, scalar) with U (velocity, vector)
        let phi = TypedFluxRef::<MassFlux, Scalar>::new("phi");
        let u = TypedFieldRef::<Velocity, Vector2>::new("U");

        let term = typed_fvm::div(phi, u);

        let untyped = term.to_untyped();
        assert_eq!(untyped.op, TermOp::Div);
        assert_eq!(untyped.discretization, Discretization::Implicit);
        assert_eq!(untyped.flux.unwrap().name(), "phi");
        assert_eq!(untyped.field.name(), "U");
    }

    #[test]
    fn typed_explicit_source_has_correct_unit() {
        // Explicit source: coeff * volume (field-independent)
        let coeff = TypedCoeff::<Pressure>::constant(101325.0);
        let p = TypedFieldRef::<Pressure, Scalar>::new("p");

        let term = typed_fvc::source_coeff(coeff, p);

        let untyped = term.to_untyped();
        assert_eq!(untyped.op, TermOp::Source);
        assert_eq!(untyped.discretization, Discretization::Explicit);

        // Verify integrated unit matches runtime: coeff_unit * volume
        let integrated = untyped.integrated_unit().expect("should compute unit");
        let expected = si::PRESSURE * si::VOLUME;
        assert_eq!(integrated, expected);
    }

    #[test]
    fn typed_explicit_source_without_coeff_has_correct_unit() {
        // Explicit source without coeff: dimensionless * volume
        let p = TypedFieldRef::<Pressure, Scalar>::new("p");

        let term = typed_fvc::source(p);

        let untyped = term.to_untyped();
        assert_eq!(untyped.op, TermOp::Source);
        assert_eq!(untyped.discretization, Discretization::Explicit);

        // Verify integrated unit matches runtime: volume
        let integrated = untyped.integrated_unit().expect("should compute unit");
        assert_eq!(integrated, si::VOLUME);
    }

    #[test]
    fn typed_mag_sqr_produces_squared_unit() {
        // mag_sqr(u) should have unit velocity^2
        let u = TypedFieldRef::<Velocity, Vector2>::new("U");

        let coeff = TypedCoeff::mag_sqr(u);

        let untyped = coeff.to_untyped();
        // Coefficient::mag_sqr produces MagSqr variant
        assert!(matches!(untyped, Coefficient::MagSqr(_)));

        // Verify unit is velocity^2
        let unit = untyped.unit();
        let expected = si::VELOCITY * si::VELOCITY;
        assert_eq!(unit, expected);
    }

    #[test]
    fn typed_div_explicit_accepts_mismatched_kinds() {
        // Explicit div should also accept mismatched kinds
        let phi = TypedFluxRef::<MassFlux, Scalar>::new("phi");
        let u = TypedFieldRef::<Velocity, Vector2>::new("U");

        let term = typed_fvc::div(phi, u);

        let untyped = term.to_untyped();
        assert_eq!(untyped.op, TermOp::Div);
        assert_eq!(untyped.discretization, Discretization::Explicit);
    }
}
