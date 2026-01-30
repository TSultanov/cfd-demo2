use std::fmt;
use std::ops::Add;

use crate::solver::units::{si, UnitDim};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldKind {
    Scalar,
    Vector2,
    Vector3,
}

impl FieldKind {
    pub fn as_str(self) -> &'static str {
        match self {
            FieldKind::Scalar => "scalar",
            FieldKind::Vector2 => "vector2",
            FieldKind::Vector3 => "vector3",
        }
    }

    pub fn component_count(self) -> usize {
        match self {
            FieldKind::Scalar => 1,
            FieldKind::Vector2 => 2,
            FieldKind::Vector3 => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FieldRef {
    name: &'static str,
    kind: FieldKind,
    unit: UnitDim,
}

impl FieldRef {
    pub fn new(name: &'static str, kind: FieldKind, unit: UnitDim) -> Self {
        Self { name, kind, unit }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn kind(&self) -> FieldKind {
        self.kind
    }

    pub fn unit(&self) -> UnitDim {
        self.unit
    }

    pub fn is_scalar(&self) -> bool {
        self.kind == FieldKind::Scalar
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FluxRef {
    name: &'static str,
    kind: FieldKind,
    unit: UnitDim,
}

impl FluxRef {
    pub fn new(name: &'static str, kind: FieldKind, unit: UnitDim) -> Self {
        Self { name, kind, unit }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn kind(&self) -> FieldKind {
        self.kind
    }

    pub fn unit(&self) -> UnitDim {
        self.unit
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Coefficient {
    Constant {
        value: f64,
        unit: UnitDim,
    },
    Field(FieldRef),
    /// Magnitude-squared of a field (scalar: φ²; vector: |u|²).
    ///
    /// This is treated as a scalar coefficient evaluated from the current state.
    MagSqr(FieldRef),
    Product(Box<Coefficient>, Box<Coefficient>),
}

impl Coefficient {
    pub fn constant(value: f64) -> Self {
        Self::Constant {
            value,
            unit: UnitDim::dimensionless(),
        }
    }

    pub fn constant_unit(value: f64, unit: UnitDim) -> Self {
        Self::Constant { value, unit }
    }

    pub fn field(field: FieldRef) -> Result<Self, CodegenError> {
        if field.is_scalar() {
            Ok(Self::Field(field))
        } else {
            Err(CodegenError::NonScalarCoefficient {
                field: field.name().to_string(),
                kind: field.kind(),
            })
        }
    }

    pub fn mag_sqr(field: FieldRef) -> Self {
        Self::MagSqr(field)
    }

    pub fn product(lhs: Coefficient, rhs: Coefficient) -> Result<Self, CodegenError> {
        lhs.ensure_scalar()?;
        rhs.ensure_scalar()?;
        Ok(Self::Product(Box::new(lhs), Box::new(rhs)))
    }

    fn ensure_scalar(&self) -> Result<(), CodegenError> {
        match self {
            Coefficient::Constant { .. } => Ok(()),
            Coefficient::Field(field) => {
                if field.is_scalar() {
                    Ok(())
                } else {
                    Err(CodegenError::NonScalarCoefficient {
                        field: field.name().to_string(),
                        kind: field.kind(),
                    })
                }
            }
            Coefficient::MagSqr(_) => Ok(()),
            Coefficient::Product(lhs, rhs) => {
                lhs.ensure_scalar()?;
                rhs.ensure_scalar()
            }
        }
    }

    pub fn unit(&self) -> UnitDim {
        match self {
            Coefficient::Constant { unit, .. } => *unit,
            Coefficient::Field(field) => field.unit(),
            Coefficient::MagSqr(field) => field.unit() * field.unit(),
            Coefficient::Product(lhs, rhs) => lhs.unit() * rhs.unit(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TermOp {
    Ddt,
    Div,
    DivFlux,
    Grad,
    Laplacian,
    Source,
}

impl TermOp {
    pub fn as_str(self) -> &'static str {
        match self {
            TermOp::Ddt => "ddt",
            TermOp::Div => "div",
            TermOp::DivFlux => "divFlux",
            TermOp::Grad => "grad",
            TermOp::Laplacian => "laplacian",
            TermOp::Source => "source",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Discretization {
    Implicit,
    Explicit,
}

impl Discretization {
    pub fn as_str(self) -> &'static str {
        match self {
            Discretization::Implicit => "implicit",
            Discretization::Explicit => "explicit",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Term {
    pub op: TermOp,
    pub discretization: Discretization,
    pub field: FieldRef,
    pub flux: Option<FluxRef>,
    pub coeff: Option<Coefficient>,
}

impl Term {
    pub fn new(
        op: TermOp,
        discretization: Discretization,
        field: FieldRef,
        flux: Option<FluxRef>,
        coeff: Option<Coefficient>,
    ) -> Self {
        Self {
            op,
            discretization,
            field,
            flux,
            coeff,
        }
    }

    pub fn eqn(self, target: FieldRef) -> Equation {
        Equation::new(target).with_term(self)
    }

    pub fn integrated_unit(&self) -> Result<UnitDim, UnitValidationError> {
        match self.op {
            TermOp::Ddt => {
                let coeff_unit = self
                    .coeff
                    .as_ref()
                    .map(|value| value.unit())
                    .unwrap_or(si::DIMENSIONLESS);
                Ok(coeff_unit * self.field.unit() * si::VOLUME / si::TIME)
            }
            TermOp::Div => {
                let flux = self
                    .flux
                    .ok_or(UnitValidationError::MissingFlux { op: self.op })?;
                Ok(flux.unit() * self.field.unit())
            }
            TermOp::DivFlux => {
                let flux = self
                    .flux
                    .ok_or(UnitValidationError::MissingFlux { op: self.op })?;
                if flux.kind() != self.field.kind() {
                    return Err(UnitValidationError::FluxKindMismatch {
                        op: self.op,
                        flux: flux.name().to_string(),
                        flux_kind: flux.kind(),
                        field: self.field.name().to_string(),
                        field_kind: self.field.kind(),
                    });
                }
                Ok(flux.unit())
            }
            TermOp::Grad => Ok(self.field.unit() * si::AREA),
            TermOp::Laplacian => {
                let coeff_unit = self
                    .coeff
                    .as_ref()
                    .map(|value| value.unit())
                    .unwrap_or(si::DIMENSIONLESS);
                Ok(coeff_unit * self.field.unit() * si::AREA / si::LENGTH)
            }
            TermOp::Source => {
                let coeff_unit = self
                    .coeff
                    .as_ref()
                    .map(|value| value.unit())
                    .unwrap_or(si::DIMENSIONLESS);
                match self.discretization {
                    // Implicit "source coefficient": S_p * phi.
                    Discretization::Implicit => Ok(coeff_unit * self.field.unit() * si::VOLUME),
                    // Explicit "source term": S_u (independent of unknown state).
                    Discretization::Explicit => Ok(coeff_unit * si::VOLUME),
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TermSum {
    terms: Vec<Term>,
}

impl TermSum {
    pub fn new(first: Term) -> Self {
        Self { terms: vec![first] }
    }

    pub fn terms(&self) -> &[Term] {
        &self.terms
    }

    pub fn eqn(self, target: FieldRef) -> Equation {
        let mut equation = Equation::new(target);
        for term in self.terms {
            equation.add_term(term);
        }
        equation
    }
}

impl Add<Term> for Term {
    type Output = TermSum;

    fn add(self, rhs: Term) -> Self::Output {
        let mut sum = TermSum::new(self);
        sum.terms.push(rhs);
        sum
    }
}

impl Add<Term> for TermSum {
    type Output = TermSum;

    fn add(mut self, rhs: Term) -> Self::Output {
        self.terms.push(rhs);
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Equation {
    target: FieldRef,
    terms: Vec<Term>,
}

impl Equation {
    pub fn new(target: FieldRef) -> Self {
        Self {
            target,
            terms: Vec::new(),
        }
    }

    pub fn target(&self) -> &FieldRef {
        &self.target
    }

    pub fn terms(&self) -> &[Term] {
        &self.terms
    }

    pub fn add_term(&mut self, term: Term) {
        self.terms.push(term);
    }

    pub fn with_term(mut self, term: Term) -> Self {
        self.terms.push(term);
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EquationSystem {
    equations: Vec<Equation>,
}

impl EquationSystem {
    pub fn new() -> Self {
        Self {
            equations: Vec::new(),
        }
    }

    pub fn add_equation(&mut self, equation: Equation) {
        self.equations.push(equation);
    }

    pub fn equations(&self) -> &[Equation] {
        &self.equations
    }

    pub fn unknowns_per_cell(&self) -> u32 {
        self.equations
            .iter()
            .map(|eqn| eqn.target.kind().component_count() as u32)
            .sum()
    }
}

impl Default for EquationSystem {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodegenError {
    NonScalarCoefficient { field: String, kind: FieldKind },
}

impl fmt::Display for CodegenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodegenError::NonScalarCoefficient { field, kind } => write!(
                f,
                "coefficient field must be scalar (field={}, kind={})",
                field,
                kind.as_str()
            ),
        }
    }
}

impl std::error::Error for CodegenError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnitValidationError {
    MissingFlux {
        op: TermOp,
    },
    FluxKindMismatch {
        op: TermOp,
        flux: String,
        flux_kind: FieldKind,
        field: String,
        field_kind: FieldKind,
    },
    TermUnitMismatch {
        equation: String,
        op: TermOp,
        expected: UnitDim,
        found: UnitDim,
    },
}

impl fmt::Display for UnitValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnitValidationError::MissingFlux { op } => {
                write!(f, "missing flux for term op {}", op.as_str())
            }
            UnitValidationError::FluxKindMismatch {
                op,
                flux,
                flux_kind,
                field,
                field_kind,
            } => write!(
                f,
                "flux kind mismatch for {}: flux {} is {}, but field {} is {}",
                op.as_str(),
                flux,
                flux_kind.as_str(),
                field,
                field_kind.as_str()
            ),
            UnitValidationError::TermUnitMismatch {
                equation,
                op,
                expected,
                found,
            } => write!(
                f,
                "unit mismatch in equation {} for term {}: expected {}, found {}",
                equation,
                op.as_str(),
                expected,
                found
            ),
        }
    }
}

impl std::error::Error for UnitValidationError {}

impl EquationSystem {
    pub fn validate_units(&self) -> Result<(), UnitValidationError> {
        for equation in &self.equations {
            let mut expected: Option<UnitDim> = None;
            for term in &equation.terms {
                let unit = term.integrated_unit()?;
                if let Some(prev) = expected {
                    if prev != unit {
                        return Err(UnitValidationError::TermUnitMismatch {
                            equation: equation.target.name().to_string(),
                            op: term.op,
                            expected: prev,
                            found: unit,
                        });
                    }
                } else {
                    expected = Some(unit);
                }
            }
        }
        Ok(())
    }
}

pub fn vol_scalar(name: &'static str, unit: UnitDim) -> FieldRef {
    FieldRef::new(name, FieldKind::Scalar, unit)
}

pub fn vol_vector(name: &'static str, unit: UnitDim) -> FieldRef {
    FieldRef::new(name, FieldKind::Vector2, unit)
}

pub fn vol_vector3(name: &'static str, unit: UnitDim) -> FieldRef {
    FieldRef::new(name, FieldKind::Vector3, unit)
}

pub fn surface_scalar(name: &'static str, unit: UnitDim) -> FluxRef {
    FluxRef::new(name, FieldKind::Scalar, unit)
}

pub fn surface_vector(name: &'static str, unit: UnitDim) -> FluxRef {
    FluxRef::new(name, FieldKind::Vector2, unit)
}

pub fn surface_vector3(name: &'static str, unit: UnitDim) -> FluxRef {
    FluxRef::new(name, FieldKind::Vector3, unit)
}

pub mod fvm {
    use super::{Coefficient, Discretization, FieldRef, FluxRef, Term, TermOp};

    pub fn ddt(field: FieldRef) -> Term {
        Term::new(TermOp::Ddt, Discretization::Implicit, field, None, None)
    }

    pub fn ddt_coeff(coeff: Coefficient, field: FieldRef) -> Term {
        Term::new(
            TermOp::Ddt,
            Discretization::Implicit,
            field,
            None,
            Some(coeff),
        )
    }

    pub fn div(flux: FluxRef, field: FieldRef) -> Term {
        Term::new(
            TermOp::Div,
            Discretization::Implicit,
            field,
            Some(flux),
            None,
        )
    }

    pub fn div_flux(flux: FluxRef, field: FieldRef) -> Term {
        Term::new(
            TermOp::DivFlux,
            Discretization::Implicit,
            field,
            Some(flux),
            None,
        )
    }

    pub fn grad(field: FieldRef) -> Term {
        Term::new(TermOp::Grad, Discretization::Implicit, field, None, None)
    }

    pub fn laplacian(coeff: Coefficient, field: FieldRef) -> Term {
        Term::new(
            TermOp::Laplacian,
            Discretization::Implicit,
            field,
            None,
            Some(coeff),
        )
    }

    pub fn source(field: FieldRef) -> Term {
        Term::new(TermOp::Source, Discretization::Implicit, field, None, None)
    }

    pub fn source_coeff(coeff: Coefficient, field: FieldRef) -> Term {
        Term::new(
            TermOp::Source,
            Discretization::Implicit,
            field,
            None,
            Some(coeff),
        )
    }
}

pub mod fvc {
    use super::{Coefficient, Discretization, FieldRef, FluxRef, Term, TermOp};

    pub fn ddt(field: FieldRef) -> Term {
        Term::new(TermOp::Ddt, Discretization::Explicit, field, None, None)
    }

    pub fn ddt_coeff(coeff: Coefficient, field: FieldRef) -> Term {
        Term::new(
            TermOp::Ddt,
            Discretization::Explicit,
            field,
            None,
            Some(coeff),
        )
    }

    pub fn div(flux: FluxRef, field: FieldRef) -> Term {
        Term::new(
            TermOp::Div,
            Discretization::Explicit,
            field,
            Some(flux),
            None,
        )
    }

    pub fn div_flux(flux: FluxRef, field: FieldRef) -> Term {
        Term::new(
            TermOp::DivFlux,
            Discretization::Explicit,
            field,
            Some(flux),
            None,
        )
    }

    pub fn grad(field: FieldRef) -> Term {
        Term::new(TermOp::Grad, Discretization::Explicit, field, None, None)
    }

    pub fn laplacian(coeff: Coefficient, field: FieldRef) -> Term {
        Term::new(
            TermOp::Laplacian,
            Discretization::Explicit,
            field,
            None,
            Some(coeff),
        )
    }

    pub fn source(field: FieldRef) -> Term {
        Term::new(TermOp::Source, Discretization::Explicit, field, None, None)
    }

    pub fn source_coeff(coeff: Coefficient, field: FieldRef) -> Term {
        Term::new(
            TermOp::Source,
            Discretization::Explicit,
            field,
            None,
            Some(coeff),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_constructors_track_kind() {
        let p = vol_scalar("p", si::PRESSURE);
        let u = vol_vector("U", si::VELOCITY);
        let phi = surface_scalar("phi", si::MASS_FLUX);

        assert_eq!(p.kind(), FieldKind::Scalar);
        assert_eq!(u.kind(), FieldKind::Vector2);
        assert_eq!(phi.name(), "phi");
    }

    #[test]
    fn coefficient_rejects_vector_field() {
        let u = vol_vector("U", si::VELOCITY);
        let err = Coefficient::field(u).unwrap_err();
        assert!(matches!(err, CodegenError::NonScalarCoefficient { .. }));
    }

    #[test]
    fn coefficient_product_rejects_vector_field() {
        let u = vol_vector("U", si::VELOCITY);
        let rho = vol_scalar("rho", si::DENSITY);
        let err = Coefficient::product(Coefficient::field(rho).unwrap(), Coefficient::Field(u))
            .unwrap_err();
        assert!(matches!(err, CodegenError::NonScalarCoefficient { .. }));
    }

    #[test]
    fn term_builders_set_op_and_discretization() {
        let u = vol_vector("U", si::VELOCITY);
        let p = vol_scalar("p", si::PRESSURE);
        let rho = vol_scalar("rho", si::DENSITY);
        let phi = surface_scalar("phi", si::MASS_FLUX);

        let term = fvm::ddt(u.clone());
        assert_eq!(term.op, TermOp::Ddt);
        assert_eq!(term.discretization, Discretization::Implicit);
        assert_eq!(term.field, u);

        let term = fvc::div(phi.clone(), p.clone());
        assert_eq!(term.op, TermOp::Div);
        assert_eq!(term.discretization, Discretization::Explicit);
        assert_eq!(term.flux, Some(phi));

        let coeff = Coefficient::field(rho).unwrap();
        let term = fvm::laplacian(coeff, p.clone());
        assert_eq!(term.op, TermOp::Laplacian);
        assert_eq!(term.coeff.is_some(), true);
        assert_eq!(term.field, p);
    }

    #[test]
    fn equation_system_collects_terms() {
        let u = vol_vector("U", si::VELOCITY);
        let phi = surface_scalar("phi", si::MASS_FLUX);

        let eqn = Equation::new(u.clone())
            .with_term(fvm::ddt(u.clone()))
            .with_term(fvm::div(phi, u.clone()));

        let mut system = EquationSystem::new();
        system.add_equation(eqn);

        assert_eq!(system.equations().len(), 1);
        assert_eq!(system.equations()[0].target(), &u);
        assert_eq!(system.equations()[0].terms().len(), 2);
        assert_eq!(system.equations()[0].terms()[0].field.name(), "U");
    }

    #[test]
    fn term_sum_builds_equation_from_addition() {
        let u = vol_vector("U", si::VELOCITY);
        let phi = surface_scalar("phi", si::MASS_FLUX);

        let eqn = (fvm::ddt(u.clone()) + fvm::div(phi, u.clone())).eqn(u.clone());

        assert_eq!(eqn.target(), &u);
        assert_eq!(eqn.terms().len(), 2);
        assert_eq!(eqn.terms()[0].op, TermOp::Ddt);
        assert_eq!(eqn.terms()[1].op, TermOp::Div);
    }

    #[test]
    fn term_eqn_creates_single_term_equation() {
        let u = vol_vector("U", si::VELOCITY);
        let eqn = fvc::source(u.clone()).eqn(u.clone());

        assert_eq!(eqn.target(), &u);
        assert_eq!(eqn.terms().len(), 1);
        assert_eq!(eqn.terms()[0].op, TermOp::Source);
    }

    #[test]
    fn termop_and_discretization_strings_are_stable() {
        assert_eq!(FieldKind::Scalar.as_str(), "scalar");
        assert_eq!(FieldKind::Vector2.as_str(), "vector2");
        assert_eq!(FieldKind::Vector3.as_str(), "vector3");
        assert_eq!(TermOp::Ddt.as_str(), "ddt");
        assert_eq!(TermOp::Div.as_str(), "div");
        assert_eq!(TermOp::DivFlux.as_str(), "divFlux");
        assert_eq!(TermOp::Grad.as_str(), "grad");
        assert_eq!(TermOp::Laplacian.as_str(), "laplacian");
        assert_eq!(TermOp::Source.as_str(), "source");

        assert_eq!(Discretization::Implicit.as_str(), "implicit");
        assert_eq!(Discretization::Explicit.as_str(), "explicit");
    }

    #[test]
    fn equation_system_default_is_empty() {
        let system = EquationSystem::default();
        assert!(system.equations().is_empty());
    }

    #[test]
    fn error_messages_include_field_name() {
        let u = vol_vector("U", si::VELOCITY);
        let err = Coefficient::field(u).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("field=U"));
    }

    #[test]
    fn validate_units_rejects_mismatched_terms() {
        let u = vol_vector("U", si::VELOCITY);
        let eqn = (fvm::ddt(u.clone()) + fvm::source(u.clone())).eqn(u.clone());

        let mut system = EquationSystem::new();
        system.add_equation(eqn);

        let err = system.validate_units().unwrap_err();
        assert!(matches!(err, UnitValidationError::TermUnitMismatch { .. }));
    }

    #[test]
    fn validate_units_rejects_flux_kind_mismatch() {
        let u = vol_vector("U", si::VELOCITY);
        let phi = surface_scalar("phi", si::MASS_FLUX);
        let eqn = fvm::div_flux(phi, u.clone()).eqn(u.clone());

        let mut system = EquationSystem::new();
        system.add_equation(eqn);

        let err = system.validate_units().unwrap_err();
        assert!(matches!(err, UnitValidationError::FluxKindMismatch { .. }));
    }
}
