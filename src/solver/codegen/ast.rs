use std::fmt;
use std::ops::Add;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldKind {
    Scalar,
    Vector2,
}

impl FieldKind {
    pub fn as_str(self) -> &'static str {
        match self {
            FieldKind::Scalar => "scalar",
            FieldKind::Vector2 => "vector2",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FieldRef {
    name: String,
    kind: FieldKind,
}

impl FieldRef {
    pub fn new(name: impl Into<String>, kind: FieldKind) -> Self {
        Self {
            name: name.into(),
            kind,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn kind(&self) -> FieldKind {
        self.kind
    }

    pub fn is_scalar(&self) -> bool {
        self.kind == FieldKind::Scalar
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FluxRef {
    name: String,
}

impl FluxRef {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Coefficient {
    Constant(f64),
    Field(FieldRef),
}

impl Coefficient {
    pub fn constant(value: f64) -> Self {
        Self::Constant(value)
    }

    pub fn field(field: FieldRef) -> Result<Self, CodegenError> {
        if field.is_scalar() {
            Ok(Self::Field(field))
        } else {
            Err(CodegenError::NonScalarCoefficient {
                field: field.name,
                kind: field.kind,
            })
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TermOp {
    Ddt,
    Div,
    Grad,
    Laplacian,
    Source,
}

impl TermOp {
    pub fn as_str(self) -> &'static str {
        match self {
            TermOp::Ddt => "ddt",
            TermOp::Div => "div",
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
    fn new(
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

pub fn vol_scalar(name: impl Into<String>) -> FieldRef {
    FieldRef::new(name, FieldKind::Scalar)
}

pub fn vol_vector(name: impl Into<String>) -> FieldRef {
    FieldRef::new(name, FieldKind::Vector2)
}

pub fn surface_scalar(name: impl Into<String>) -> FluxRef {
    FluxRef::new(name)
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

    pub fn grad(field: FieldRef) -> Term {
        Term::new(
            TermOp::Grad,
            Discretization::Implicit,
            field,
            None,
            None,
        )
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
        Term::new(
            TermOp::Source,
            Discretization::Implicit,
            field,
            None,
            None,
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

    pub fn grad(field: FieldRef) -> Term {
        Term::new(
            TermOp::Grad,
            Discretization::Explicit,
            field,
            None,
            None,
        )
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
        Term::new(
            TermOp::Source,
            Discretization::Explicit,
            field,
            None,
            None,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_constructors_track_kind() {
        let p = vol_scalar("p");
        let u = vol_vector("U");
        let phi = surface_scalar("phi");

        assert_eq!(p.kind(), FieldKind::Scalar);
        assert_eq!(u.kind(), FieldKind::Vector2);
        assert_eq!(phi.name(), "phi");
    }

    #[test]
    fn coefficient_rejects_vector_field() {
        let u = vol_vector("U");
        let err = Coefficient::field(u).unwrap_err();
        assert!(matches!(
            err,
            CodegenError::NonScalarCoefficient { .. }
        ));
    }

    #[test]
    fn term_builders_set_op_and_discretization() {
        let u = vol_vector("U");
        let p = vol_scalar("p");
        let rho = vol_scalar("rho");
        let phi = surface_scalar("phi");

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
        let u = vol_vector("U");
        let phi = surface_scalar("phi");

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
        let u = vol_vector("U");
        let phi = surface_scalar("phi");

        let eqn = (fvm::ddt(u.clone()) + fvm::div(phi, u.clone())).eqn(u.clone());

        assert_eq!(eqn.target(), &u);
        assert_eq!(eqn.terms().len(), 2);
        assert_eq!(eqn.terms()[0].op, TermOp::Ddt);
        assert_eq!(eqn.terms()[1].op, TermOp::Div);
    }

    #[test]
    fn term_eqn_creates_single_term_equation() {
        let u = vol_vector("U");
        let eqn = fvc::source(u.clone()).eqn(u.clone());

        assert_eq!(eqn.target(), &u);
        assert_eq!(eqn.terms().len(), 1);
        assert_eq!(eqn.terms()[0].op, TermOp::Source);
    }

    #[test]
    fn termop_and_discretization_strings_are_stable() {
        assert_eq!(FieldKind::Scalar.as_str(), "scalar");
        assert_eq!(FieldKind::Vector2.as_str(), "vector2");
        assert_eq!(TermOp::Ddt.as_str(), "ddt");
        assert_eq!(TermOp::Div.as_str(), "div");
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
        let u = vol_vector("U");
        let err = Coefficient::field(u).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("field=U"));
    }
}
