// Algebraic (non-differential) equations for coupled systems.
//
// This module lets a model declare pointwise algebraic relations between
// solved fields -- primitive-variable recovery (`rho*u = rho_u`), equation-of-
// state closures (`p = (gamma-1)*rho_e - ...`), and similar constraints -- as
// math, and mechanically lowers them to the implicit/explicit source rows the
// coupled assembly already understands. This replaces hand-written
// "pseudo-source equation" blocks in model definitions.
//
// Compilation contexts: like `typed_ast.rs`, this file is compiled both as
// `cfd2_ir::equation::algebraic` and (via include!) inside build.rs as a
// sibling of the include!'d `typed_ast` module. Therefore: paths must be
// `super::ast` / `super::typed_ast` / `crate::dimensions`, no inherent impls
// on `ast` types (they are foreign types in the build.rs context), and no
// inner (`//!`) doc comments.
//
// # Lowering semantics
//
// `AlgebraicEquation { target, lhs, rhs }` represents `lhs = rhs`, where
// `target` names the unknown whose block-row of the coupled matrix this
// equation occupies. The residual `lhs - rhs` is flattened into a signed sum
// of products. Each product may reference at most one *linear unknown*: the
// target if it appears as a plain field factor, otherwise the single solved-
// unknown field factor. All other factors -- constants, params, `mag_sqr`,
// and remaining plain fields -- are frozen at the current state (Picard
// linearization). Non-affine forms (sums inside products, an unknown
// appearing twice in one product, two distinct non-target unknowns in one
// product) are rejected; distribute or restructure the declaration instead.
//
// Every row is scaled by the engine coefficient `inv_dt` (1/dt, preferring
// dtau during dual-time stepping) so algebraic rows match the magnitude of
// the `ddt(..)` rows they couple to. The coefficient construction rule is
// deterministic and reproduces the historical hand-written compressible
// recovery rows bit-for-bit:
//
//   - factors multiply left-to-right in declaration order: sign (`-1` if the
//     signed summand is negative), then non-mag_sqr frozen factors, then
//     `inv_dt`, then mag_sqr factors;
//   - except for the row's target term, where a negative sign wraps the
//     whole product: `-1 * (frozen.. * inv_dt)`.
//
// Emission order: the target term first, then implicit terms in declaration
// order, then explicit (unknown-free) terms last.

use std::marker::PhantomData;

use super::ast::{
    Coefficient, Discretization, Equation, EquationSystem, FieldKind, FieldRef, Term, TermOp,
};
use super::typed_ast::{Kind, Scalar, TypedFieldRef};
use crate::dimensions::{Dimensionless, DivDim, InvTime, MulDim, UnitDimension};

// `UnitDim` is the runtime unit representation; reachable in both compilation
// contexts through the `dimensions` re-export.
use crate::dimensions::UnitDim;

// ============================================================================
// Untyped algebraic expression tree
// ============================================================================

/// Reference to a named uniform model parameter (e.g. `eos_gm1`).
///
/// Parameters are uniforms declared by a module port manifest, not state
/// fields. At lowering time a param is represented as the legacy named
/// coefficient (`Coefficient::Field` carrying the param's WGSL field name),
/// which codegen resolves to `constants.<name>`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParamRef {
    name: &'static str,
    unit: UnitDim,
}

impl ParamRef {
    pub fn new(name: &'static str, unit: UnitDim) -> Self {
        Self { name, unit }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn unit(&self) -> UnitDim {
        self.unit
    }
}

/// Pointwise algebraic expression over cell values of fields and params.
#[derive(Debug, Clone, PartialEq)]
pub enum AlgExpr {
    Constant {
        value: f64,
        unit: UnitDim,
    },
    /// Cell value of a state field (scalar or vector).
    Field(FieldRef),
    /// Uniform model parameter.
    Param(ParamRef),
    /// Magnitude squared of a field (scalar: phi^2; vector: |u|^2), always
    /// evaluated from the current state (frozen), like `Coefficient::MagSqr`.
    MagSqr(FieldRef),
    Mul(Box<AlgExpr>, Box<AlgExpr>),
    /// Division. Supported by face-expression lowering (e.g. wave-speed
    /// declarations consumed by flux derivation); REJECTED by the affine
    /// coupled-row lowering (`lower_algebraic_equation`), which requires
    /// clearing denominators in the declaration instead.
    Div(Box<AlgExpr>, Box<AlgExpr>),
    Add(Box<AlgExpr>, Box<AlgExpr>),
    Sub(Box<AlgExpr>, Box<AlgExpr>),
    Neg(Box<AlgExpr>),
}

/// A pointwise algebraic relation `lhs = rhs` occupying the coupled
/// block-row of `target`.
#[derive(Debug, Clone, PartialEq)]
pub struct AlgebraicEquation {
    pub target: FieldRef,
    pub lhs: AlgExpr,
    pub rhs: AlgExpr,
}

// ============================================================================
// Lowering
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
enum Factor {
    Constant { value: f64, unit: UnitDim },
    Param(ParamRef),
    Field(FieldRef),
    MagSqr(FieldRef),
}

impl Factor {
    fn unit(&self) -> UnitDim {
        match self {
            Factor::Constant { unit, .. } => *unit,
            Factor::Param(p) => p.unit(),
            Factor::Field(f) => f.unit(),
            Factor::MagSqr(f) => f.unit() * f.unit(),
        }
    }

    fn to_coeff(self) -> Coefficient {
        match self {
            Factor::Constant { value, unit } => Coefficient::Constant { value, unit },
            Factor::Param(p) => {
                Coefficient::Field(FieldRef::new(p.name(), FieldKind::Scalar, p.unit()))
            }
            Factor::Field(f) => Coefficient::Field(f),
            Factor::MagSqr(f) => Coefficient::MagSqr(f),
        }
    }
}

struct Summand {
    negative: bool,
    factors: Vec<Factor>,
}

fn flatten_sum(expr: &AlgExpr, negate: bool, out: &mut Vec<Summand>) -> Result<(), String> {
    match expr {
        AlgExpr::Add(a, b) => {
            flatten_sum(a, negate, out)?;
            flatten_sum(b, negate, out)
        }
        AlgExpr::Sub(a, b) => {
            flatten_sum(a, negate, out)?;
            flatten_sum(b, !negate, out)
        }
        AlgExpr::Neg(a) => flatten_sum(a, !negate, out),
        product => {
            let mut negative = negate;
            let mut factors = Vec::new();
            flatten_product(product, &mut negative, &mut factors)?;
            out.push(Summand { negative, factors });
            Ok(())
        }
    }
}

fn flatten_product(
    expr: &AlgExpr,
    negative: &mut bool,
    out: &mut Vec<Factor>,
) -> Result<(), String> {
    match expr {
        AlgExpr::Mul(a, b) => {
            flatten_product(a, negative, out)?;
            flatten_product(b, negative, out)
        }
        AlgExpr::Neg(a) => {
            *negative = !*negative;
            flatten_product(a, negative, out)
        }
        AlgExpr::Constant { value, unit } => {
            out.push(Factor::Constant {
                value: *value,
                unit: *unit,
            });
            Ok(())
        }
        AlgExpr::Param(p) => {
            out.push(Factor::Param(*p));
            Ok(())
        }
        AlgExpr::Field(f) => {
            out.push(Factor::Field(*f));
            Ok(())
        }
        AlgExpr::MagSqr(f) => {
            out.push(Factor::MagSqr(*f));
            Ok(())
        }
        AlgExpr::Add(..) | AlgExpr::Sub(..) => Err(
            "non-affine algebraic expression: sum nested inside a product; \
             distribute the product over the sum in the declaration"
                .to_string(),
        ),
        AlgExpr::Div(..) => Err(
            "non-affine algebraic expression: division is not supported in coupled-row \
             lowering; clear the denominator in the declaration (e.g. declare rho*u = rho_u \
             instead of u = rho_u/rho)"
                .to_string(),
        ),
    }
}

struct LoweredSummand {
    negative: bool,
    /// The factor treated as the linear unknown (column of the coupled
    /// matrix); `None` for unknown-free (explicit) summands.
    linear: Option<FieldRef>,
    /// Frozen non-mag_sqr factors, declaration order.
    pre: Vec<Factor>,
    /// Frozen mag_sqr factors, declaration order (multiplied after inv_dt).
    post: Vec<Factor>,
}

fn classify_summand(
    summand: &Summand,
    target: FieldRef,
    known_unknowns: &[FieldRef],
) -> Result<LoweredSummand, String> {
    let plain_fields: Vec<FieldRef> = summand
        .factors
        .iter()
        .filter_map(|f| match f {
            Factor::Field(fr) => Some(*fr),
            _ => None,
        })
        .collect();

    let target_count = plain_fields.iter().filter(|f| **f == target).count();
    if target_count > 1 {
        return Err(format!(
            "non-affine algebraic equation for '{}': target appears {} times in one product",
            target.name(),
            target_count
        ));
    }

    let linear = if target_count == 1 {
        Some(target)
    } else {
        let unknown_occurrences: Vec<FieldRef> = plain_fields
            .iter()
            .filter(|f| known_unknowns.contains(f))
            .copied()
            .collect();
        match unknown_occurrences.len() {
            0 => None,
            1 => Some(unknown_occurrences[0]),
            _ => {
                return Err(format!(
                    "ambiguous/non-affine product in algebraic equation for '{}': solved \
                     unknowns {:?} appear together in one product; at most one plain-field \
                     unknown is allowed per product (use mag_sqr or restructure to freeze \
                     the others)",
                    target.name(),
                    unknown_occurrences
                        .iter()
                        .map(|f| f.name())
                        .collect::<Vec<_>>()
                ))
            }
        }
    };

    if let Some(lin) = linear {
        if lin.kind() != target.kind() {
            return Err(format!(
                "kind mismatch in algebraic equation for '{}' ({}): linear unknown '{}' is {}; \
                 per-component coupling requires matching kinds",
                target.name(),
                target.kind().as_str(),
                lin.name(),
                lin.kind().as_str()
            ));
        }
    } else if target.kind() != FieldKind::Scalar {
        return Err(format!(
            "algebraic equation for vector target '{}' has an unknown-free (explicit) term; \
             per-component explicit sources are not supported yet (planned with the derived \
             flux work)",
            target.name()
        ));
    }

    let mut pre = Vec::new();
    let mut post = Vec::new();
    let mut linear_taken = false;
    for factor in &summand.factors {
        match factor {
            Factor::MagSqr(_) => post.push(*factor),
            Factor::Field(fr) if Some(*fr) == linear && !linear_taken => {
                linear_taken = true;
            }
            _ => pre.push(*factor),
        }
    }

    for factor in &pre {
        if let Factor::Field(fr) = factor {
            if !fr.is_scalar() {
                return Err(format!(
                    "frozen factor '{}' in algebraic equation for '{}' is {}; frozen \
                     (coefficient) factors must be scalar",
                    fr.name(),
                    target.name(),
                    fr.kind().as_str()
                ));
            }
        }
    }

    Ok(LoweredSummand {
        negative: summand.negative,
        linear,
        pre,
        post,
    })
}

fn inv_dt_coeff() -> Coefficient {
    Coefficient::Field(FieldRef::new("inv_dt", FieldKind::Scalar, InvTime::UNIT))
}

fn minus_one_coeff() -> Coefficient {
    Coefficient::Constant {
        value: -1.0,
        unit: Dimensionless::UNIT,
    }
}

fn fold_product(parts: impl IntoIterator<Item = Coefficient>) -> Option<Coefficient> {
    parts
        .into_iter()
        .reduce(|acc, next| Coefficient::Product(Box::new(acc), Box::new(next)))
}

fn summand_coefficient(summand: &LoweredSummand, is_target_term: bool) -> Coefficient {
    let pre = || summand.pre.iter().map(|f| f.to_coeff());
    let post = || summand.post.iter().map(|f| f.to_coeff());
    if is_target_term {
        // The target term wraps a negative sign around the whole product.
        let base = fold_product(pre().chain(std::iter::once(inv_dt_coeff())).chain(post()))
            .expect("base product always contains inv_dt");
        if summand.negative {
            Coefficient::Product(Box::new(minus_one_coeff()), Box::new(base))
        } else {
            base
        }
    } else {
        // Everywhere else the sign folds in at the front (left-to-right).
        let sign = summand.negative.then(minus_one_coeff);
        fold_product(
            sign.into_iter()
                .chain(pre())
                .chain(std::iter::once(inv_dt_coeff()))
                .chain(post()),
        )
        .expect("non-empty product")
    }
}

/// Lower an algebraic relation to implicit/explicit source terms.
///
/// `known_unknowns` lists the solved fields of the coupled system (typically
/// the targets of previously added equations); `eq.target` is always treated
/// as an unknown. See the module docs for semantics and the deterministic
/// coefficient construction rule.
pub fn lower_algebraic_equation(
    eq: &AlgebraicEquation,
    known_unknowns: &[FieldRef],
) -> Result<Equation, String> {
    let target = eq.target;

    let mut summands = Vec::new();
    flatten_sum(&eq.lhs, false, &mut summands)?;
    flatten_sum(&eq.rhs, true, &mut summands)?;

    let mut expected_unit: Option<UnitDim> = None;
    for summand in &summands {
        let unit = summand
            .factors
            .iter()
            .fold(Dimensionless::UNIT, |acc, f| acc * f.unit());
        if let Some(prev) = expected_unit {
            if prev != unit {
                return Err(format!(
                    "unit mismatch in algebraic equation for '{}': term units {} vs {}",
                    target.name(),
                    prev,
                    unit
                ));
            }
        } else {
            expected_unit = Some(unit);
        }
    }

    let classified = summands
        .iter()
        .map(|s| classify_summand(s, target, known_unknowns))
        .collect::<Result<Vec<_>, _>>()?;

    let target_terms = classified
        .iter()
        .filter(|s| s.linear == Some(target))
        .count();
    if target_terms != 1 {
        return Err(format!(
            "algebraic equation for '{}' must contain its target as a plain linear factor in \
             exactly one product (found {})",
            target.name(),
            target_terms
        ));
    }

    let mut terms: Vec<Term> = Vec::new();
    // Target term first.
    for summand in classified.iter().filter(|s| s.linear == Some(target)) {
        terms.push(Term::new(
            TermOp::Source,
            Discretization::Implicit,
            target,
            None,
            Some(summand_coefficient(summand, true)),
        ));
    }
    // Other implicit terms in declaration order.
    for summand in &classified {
        if let Some(linear) = summand.linear {
            if linear != target {
                terms.push(Term::new(
                    TermOp::Source,
                    Discretization::Implicit,
                    linear,
                    None,
                    Some(summand_coefficient(summand, false)),
                ));
            }
        }
    }
    // Explicit (unknown-free) terms last; attached to the target field.
    for summand in classified.iter().filter(|s| s.linear.is_none()) {
        terms.push(Term::new(
            TermOp::Source,
            Discretization::Explicit,
            target,
            None,
            Some(summand_coefficient(summand, false)),
        ));
    }

    let mut equation = Equation::new(target);
    for term in terms {
        equation.add_term(term);
    }
    Ok(equation)
}

/// Lower `eq` against the targets of the equations already in `system` and
/// append the resulting row. Equations whose unknowns this relation couples
/// to implicitly must already be present (order matters: a field not yet
/// known as an unknown is frozen instead of coupled).
pub fn add_algebraic_equation(
    system: &mut EquationSystem,
    eq: &AlgebraicEquation,
) -> Result<(), String> {
    let known_unknowns: Vec<FieldRef> = system
        .equations()
        .iter()
        .map(|e| *e.target())
        .collect();
    let lowered = lower_algebraic_equation(eq, &known_unknowns)?;
    system.add_equation(lowered);
    Ok(())
}

// ============================================================================
// Typed wrappers
// ============================================================================

/// Typed reference to a named uniform model parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypedParamRef<D: UnitDimension> {
    name: &'static str,
    _dim: PhantomData<D>,
}

impl<D: UnitDimension> TypedParamRef<D> {
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            _dim: PhantomData,
        }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn to_untyped(self) -> ParamRef {
        ParamRef::new(self.name, D::UNIT)
    }
}

/// Typed algebraic expression with compile-time dimension and kind tracking.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedAlgExpr<D: UnitDimension, K: Kind> {
    inner: AlgExpr,
    _dim: PhantomData<D>,
    _kind: PhantomData<K>,
}

impl<D: UnitDimension, K: Kind> TypedAlgExpr<D, K> {
    fn wrap(inner: AlgExpr) -> Self {
        Self {
            inner,
            _dim: PhantomData,
            _kind: PhantomData,
        }
    }

    pub fn to_untyped(&self) -> AlgExpr {
        self.inner.clone()
    }

    /// Cast to a different dimension type with identical runtime units
    /// (type-level dimension expressions are not normalized).
    ///
    /// # Panics
    ///
    /// Panics if `D::UNIT != DTo::UNIT`.
    pub fn cast_to<DTo: UnitDimension>(self) -> TypedAlgExpr<DTo, K> {
        assert!(
            D::UNIT == DTo::UNIT,
            "Cannot cast algebraic expression from dimension {:?} to {:?} - units do not match",
            D::UNIT,
            DTo::UNIT
        );
        TypedAlgExpr::wrap(self.inner)
    }
}

impl<D1, D2, K> std::ops::Mul<TypedAlgExpr<D2, K>> for TypedAlgExpr<D1, Scalar>
where
    D1: UnitDimension,
    D2: UnitDimension,
    K: Kind,
{
    type Output = TypedAlgExpr<MulDim<D1, D2>, K>;

    fn mul(self, rhs: TypedAlgExpr<D2, K>) -> Self::Output {
        TypedAlgExpr::wrap(AlgExpr::Mul(Box::new(self.inner), Box::new(rhs.inner)))
    }
}

impl<D1, D2, K> std::ops::Div<TypedAlgExpr<D2, Scalar>> for TypedAlgExpr<D1, K>
where
    D1: UnitDimension,
    D2: UnitDimension,
    K: Kind,
{
    type Output = TypedAlgExpr<DivDim<D1, D2>, K>;

    fn div(self, rhs: TypedAlgExpr<D2, Scalar>) -> Self::Output {
        TypedAlgExpr::wrap(AlgExpr::Div(Box::new(self.inner), Box::new(rhs.inner)))
    }
}

impl<D: UnitDimension, K: Kind> std::ops::Add for TypedAlgExpr<D, K> {
    type Output = TypedAlgExpr<D, K>;

    fn add(self, rhs: Self) -> Self::Output {
        TypedAlgExpr::wrap(AlgExpr::Add(Box::new(self.inner), Box::new(rhs.inner)))
    }
}

impl<D: UnitDimension, K: Kind> std::ops::Sub for TypedAlgExpr<D, K> {
    type Output = TypedAlgExpr<D, K>;

    fn sub(self, rhs: Self) -> Self::Output {
        TypedAlgExpr::wrap(AlgExpr::Sub(Box::new(self.inner), Box::new(rhs.inner)))
    }
}

impl<D: UnitDimension, K: Kind> std::ops::Neg for TypedAlgExpr<D, K> {
    type Output = TypedAlgExpr<D, K>;

    fn neg(self) -> Self::Output {
        TypedAlgExpr::wrap(AlgExpr::Neg(Box::new(self.inner)))
    }
}

/// Constructors for typed algebraic expressions (mirrors `typed_fvm`/`typed_fvc`).
pub mod typed_alg {
    use super::*;

    /// Cell value of a state field.
    pub fn field<D: UnitDimension, K: Kind>(f: TypedFieldRef<D, K>) -> TypedAlgExpr<D, K> {
        TypedAlgExpr::wrap(AlgExpr::Field(f.to_untyped()))
    }

    /// Uniform model parameter.
    pub fn param<D: UnitDimension>(p: TypedParamRef<D>) -> TypedAlgExpr<D, Scalar> {
        TypedAlgExpr::wrap(AlgExpr::Param(p.to_untyped()))
    }

    /// Dimensionless constant.
    pub fn constant(value: f64) -> TypedAlgExpr<Dimensionless, Scalar> {
        TypedAlgExpr::wrap(AlgExpr::Constant {
            value,
            unit: Dimensionless::UNIT,
        })
    }

    /// Magnitude squared of a field, frozen at the current state.
    pub fn mag_sqr<D: UnitDimension, K: Kind>(
        f: TypedFieldRef<D, K>,
    ) -> TypedAlgExpr<MulDim<D, D>, Scalar> {
        TypedAlgExpr::wrap(AlgExpr::MagSqr(f.to_untyped()))
    }

    /// Declare the algebraic relation `lhs = rhs` for the block-row of
    /// `target`. Both sides must agree in dimension and kind, and the target
    /// kind must match the sides' kind.
    pub fn equation<TD: UnitDimension, D: UnitDimension, K: Kind>(
        target: TypedFieldRef<TD, K>,
        lhs: TypedAlgExpr<D, K>,
        rhs: TypedAlgExpr<D, K>,
    ) -> AlgebraicEquation {
        AlgebraicEquation {
            target: target.to_untyped(),
            lhs: lhs.inner,
            rhs: rhs.inner,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::typed_alg::{constant, equation, field, mag_sqr, param};
    use super::*;
    use crate::dimensions::{Density, MomentumDensity, Pressure, Velocity};
    use crate::equation::ast::{vol_scalar_dim, vol_vector_dim};

    fn rho() -> TypedFieldRef<Density, Scalar> {
        TypedFieldRef::new("rho")
    }

    fn u() -> TypedFieldRef<Velocity, super::super::typed_ast::Vector2> {
        TypedFieldRef::new("u")
    }

    fn rho_u() -> TypedFieldRef<MomentumDensity, super::super::typed_ast::Vector2> {
        TypedFieldRef::new("rho_u")
    }

    fn p() -> TypedFieldRef<Pressure, Scalar> {
        TypedFieldRef::new("p")
    }

    fn inv_dt_field() -> FieldRef {
        FieldRef::new("inv_dt", FieldKind::Scalar, InvTime::UNIT)
    }

    #[test]
    fn velocity_recovery_lowers_to_handwritten_shape() {
        // rho_u = rho * u, target u: matches the historical hand-written form
        //   fvm::source_coeff(-1 * (rho * inv_dt), u) + fvm::source_coeff(inv_dt, rho_u)
        let eq = equation(
            u(),
            field(rho_u()),
            (field(rho()) * field(u())).cast_to::<MomentumDensity>(),
        );
        let unknowns = [
            vol_scalar_dim::<Density>("rho"),
            vol_vector_dim::<MomentumDensity>("rho_u"),
        ];
        let lowered = lower_algebraic_equation(&eq, &unknowns).expect("lowering");

        assert_eq!(lowered.target().name(), "u");
        let terms = lowered.terms();
        assert_eq!(terms.len(), 2);

        // Target term first: -1 * (rho * inv_dt), implicit on u.
        assert_eq!(terms[0].op, TermOp::Source);
        assert_eq!(terms[0].discretization, Discretization::Implicit);
        assert_eq!(terms[0].field.name(), "u");
        let expected_target_coeff = Coefficient::Product(
            Box::new(Coefficient::Constant {
                value: -1.0,
                unit: Dimensionless::UNIT,
            }),
            Box::new(Coefficient::Product(
                Box::new(Coefficient::Field(vol_scalar_dim::<Density>("rho"))),
                Box::new(Coefficient::Field(inv_dt_field())),
            )),
        );
        assert_eq!(terms[0].coeff.as_ref(), Some(&expected_target_coeff));

        // Then the rho_u column with bare inv_dt.
        assert_eq!(terms[1].field.name(), "rho_u");
        assert_eq!(terms[1].discretization, Discretization::Implicit);
        assert_eq!(
            terms[1].coeff.as_ref(),
            Some(&Coefficient::Field(inv_dt_field()))
        );
    }

    #[test]
    fn magsqr_factors_multiply_after_inv_dt() {
        // p = 0.5 * |u|^2 * rho (a synthetic EOS-like row): the mag_sqr factor
        // must multiply after inv_dt: ((0.5 * inv_dt) * |u|^2).
        let eq = equation(
            p(),
            field(p()),
            (constant(0.5) * mag_sqr(u()) * field(rho())).cast_to::<Pressure>(),
        );
        let unknowns = [vol_scalar_dim::<Density>("rho")];
        let lowered = lower_algebraic_equation(&eq, &unknowns).expect("lowering");
        let terms = lowered.terms();
        assert_eq!(terms.len(), 2);

        // Target term: bare inv_dt.
        assert_eq!(terms[0].field.name(), "p");
        assert_eq!(
            terms[0].coeff.as_ref(),
            Some(&Coefficient::Field(inv_dt_field()))
        );

        // rho column: -(0.5) folded first, then inv_dt, then |u|^2.
        assert_eq!(terms[1].field.name(), "rho");
        let expected = Coefficient::Product(
            Box::new(Coefficient::Product(
                Box::new(Coefficient::Product(
                    Box::new(Coefficient::Constant {
                        value: -1.0,
                        unit: Dimensionless::UNIT,
                    }),
                    Box::new(Coefficient::Constant {
                        value: 0.5,
                        unit: Dimensionless::UNIT,
                    }),
                )),
                Box::new(Coefficient::Field(inv_dt_field())),
            )),
            Box::new(Coefficient::MagSqr(vol_vector_dim::<Velocity>("u"))),
        );
        assert_eq!(terms[1].coeff.as_ref(), Some(&expected));
    }

    #[test]
    fn unknown_free_summand_becomes_trailing_explicit_source() {
        // p = c with c a param: explicit source on the target, emitted last,
        // with coefficient (-1 * c) * inv_dt.
        let offset = TypedParamRef::<Pressure>::new("p_offset");
        let eq = equation(p(), field(p()), param(offset));
        let lowered = lower_algebraic_equation(&eq, &[]).expect("lowering");
        let terms = lowered.terms();
        assert_eq!(terms.len(), 2);
        assert_eq!(terms[1].discretization, Discretization::Explicit);
        assert_eq!(terms[1].field.name(), "p");
        let expected = Coefficient::Product(
            Box::new(Coefficient::Product(
                Box::new(Coefficient::Constant {
                    value: -1.0,
                    unit: Dimensionless::UNIT,
                }),
                Box::new(Coefficient::Field(FieldRef::new(
                    "p_offset",
                    FieldKind::Scalar,
                    Pressure::UNIT,
                ))),
            )),
            Box::new(Coefficient::Field(inv_dt_field())),
        );
        assert_eq!(terms[1].coeff.as_ref(), Some(&expected));
    }

    #[test]
    fn rejects_division_in_coupled_row_lowering() {
        // u = rho_u / rho must be declared with the denominator cleared
        // (rho * u = rho_u); Div is only for face-expression declarations.
        let rho_f = vol_scalar_dim::<Density>("rho");
        let u_f = vol_vector_dim::<Velocity>("u");
        let rho_u_f = vol_vector_dim::<MomentumDensity>("rho_u");
        let bad = AlgebraicEquation {
            target: u_f,
            lhs: AlgExpr::Field(u_f),
            rhs: AlgExpr::Div(
                Box::new(AlgExpr::Field(rho_u_f)),
                Box::new(AlgExpr::Field(rho_f)),
            ),
        };
        let err = lower_algebraic_equation(&bad, &[rho_f, rho_u_f]).unwrap_err();
        assert!(err.contains("clear the denominator"), "{err}");
    }

    #[test]
    fn rejects_sum_inside_product() {
        let bad = AlgebraicEquation {
            target: vol_scalar_dim::<Pressure>("p"),
            lhs: AlgExpr::Field(vol_scalar_dim::<Pressure>("p")),
            rhs: AlgExpr::Mul(
                Box::new(AlgExpr::Add(
                    Box::new(AlgExpr::Field(vol_scalar_dim::<Pressure>("a"))),
                    Box::new(AlgExpr::Field(vol_scalar_dim::<Pressure>("b"))),
                )),
                Box::new(AlgExpr::Constant {
                    value: 2.0,
                    unit: Dimensionless::UNIT,
                }),
            ),
        };
        let err = lower_algebraic_equation(&bad, &[]).unwrap_err();
        assert!(err.contains("non-affine"), "{err}");
    }

    #[test]
    fn rejects_two_unknowns_in_one_product() {
        let rho_f = vol_scalar_dim::<Density>("rho");
        let t_f = vol_scalar_dim::<crate::dimensions::Temperature>("T");
        // Target carries the product's unit so the ambiguity check (not the
        // unit check) is what fires.
        let target = crate::equation::ast::vol_scalar("q", rho_f.unit() * t_f.unit());
        let bad = AlgebraicEquation {
            target,
            lhs: AlgExpr::Field(target),
            rhs: AlgExpr::Mul(Box::new(AlgExpr::Field(rho_f)), Box::new(AlgExpr::Field(t_f))),
        };
        let err = lower_algebraic_equation(&bad, &[rho_f, t_f]).unwrap_err();
        assert!(err.contains("ambiguous"), "{err}");
    }

    #[test]
    fn rejects_target_squared() {
        let p_f = vol_scalar_dim::<Pressure>("p");
        let bad = AlgebraicEquation {
            target: p_f,
            lhs: AlgExpr::Mul(Box::new(AlgExpr::Field(p_f)), Box::new(AlgExpr::Field(p_f))),
            rhs: AlgExpr::Constant {
                value: 1.0,
                unit: Pressure::UNIT * Pressure::UNIT,
            },
        };
        let err = lower_algebraic_equation(&bad, &[]).unwrap_err();
        assert!(err.contains("target appears 2 times"), "{err}");
    }

    #[test]
    fn rejects_missing_target_term() {
        let bad = AlgebraicEquation {
            target: vol_scalar_dim::<Pressure>("p"),
            lhs: AlgExpr::Field(vol_scalar_dim::<Pressure>("a")),
            rhs: AlgExpr::Field(vol_scalar_dim::<Pressure>("b")),
        };
        let err = lower_algebraic_equation(&bad, &[]).unwrap_err();
        assert!(err.contains("exactly one product (found 0)"), "{err}");
    }

    #[test]
    fn rejects_unit_mismatch_between_sides() {
        let bad = AlgebraicEquation {
            target: vol_scalar_dim::<Pressure>("p"),
            lhs: AlgExpr::Field(vol_scalar_dim::<Pressure>("p")),
            rhs: AlgExpr::Field(vol_scalar_dim::<Density>("rho")),
        };
        let err = lower_algebraic_equation(&bad, &[]).unwrap_err();
        assert!(err.contains("unit mismatch"), "{err}");
    }

    #[test]
    fn rejects_frozen_vector_factor() {
        // Row with a product rho_u * x where rho_u must be frozen (target
        // absent from the product, x is the unknown) -- frozen vectors are
        // not valid coefficients. Target carries the product's unit so the
        // frozen-factor check (not the unit check) is what fires.
        let rho_u_f = vol_vector_dim::<MomentumDensity>("rho_u");
        let x_f = vol_scalar_dim::<Density>("x");
        let target = crate::equation::ast::vol_scalar("q", rho_u_f.unit() * x_f.unit());
        let bad = AlgebraicEquation {
            target,
            lhs: AlgExpr::Field(target),
            rhs: AlgExpr::Mul(
                Box::new(AlgExpr::Field(rho_u_f)),
                Box::new(AlgExpr::Field(x_f)),
            ),
        };
        let err = lower_algebraic_equation(&bad, &[x_f]).unwrap_err();
        assert!(err.contains("frozen factor 'rho_u'"), "{err}");
    }

    #[test]
    fn add_algebraic_equation_uses_existing_targets_as_unknowns() {
        use crate::equation::ast::fvm;

        let rho_f = vol_scalar_dim::<Density>("rho");
        let p_f = vol_scalar_dim::<Pressure>("p");
        let dp_drho = TypedParamRef::<crate::dimensions::DivDim<Pressure, Density>>::new("dp_drho");

        let mut system = EquationSystem::new();
        system.add_equation(fvm::ddt(rho_f).eqn(rho_f));

        // p = dp_drho * rho: rho must be coupled implicitly because its
        // equation is already present.
        let eq = equation(
            p(),
            field(p()),
            (param(dp_drho) * field(rho())).cast_to::<Pressure>(),
        );
        add_algebraic_equation(&mut system, &eq).expect("add");

        let eqs = system.equations();
        assert_eq!(eqs.len(), 2);
        let p_terms = eqs[1].terms();
        assert_eq!(p_terms.len(), 2);
        assert_eq!(p_terms[1].field.name(), "rho");
        assert_eq!(p_terms[1].discretization, Discretization::Implicit);
    }

    #[test]
    #[should_panic(expected = "units do not match")]
    fn cast_to_panics_on_unit_mismatch() {
        let _ = field(rho()).cast_to::<Pressure>();
    }
}
