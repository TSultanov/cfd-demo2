use crate::solver::scheme::Scheme;

use crate::solver::model::ast::{Coefficient, Discretization, EquationSystem, FieldRef, FluxRef, Term, TermOp};
use crate::solver::model::SchemeRegistry;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiscreteOpKind {
    TimeDerivative,
    Convection,
    Gradient,
    Diffusion,
    Source,
}

impl DiscreteOpKind {
    pub fn as_str(self) -> &'static str {
        match self {
            DiscreteOpKind::TimeDerivative => "ddt",
            DiscreteOpKind::Convection => "div",
            DiscreteOpKind::Gradient => "grad",
            DiscreteOpKind::Diffusion => "laplacian",
            DiscreteOpKind::Source => "source",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiscreteOp {
    pub target: FieldRef,
    pub kind: DiscreteOpKind,
    pub discretization: Discretization,
    pub scheme: Scheme,
    pub field: FieldRef,
    pub flux: Option<FluxRef>,
    pub coeff: Option<Coefficient>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiscreteEquation {
    pub target: FieldRef,
    pub ops: Vec<DiscreteOp>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiscreteSystem {
    pub equations: Vec<DiscreteEquation>,
}

pub fn lower_system(system: &EquationSystem, schemes: &SchemeRegistry) -> DiscreteSystem {
    let mut equations = Vec::new();
    for equation in system.equations() {
        let mut ops = Vec::new();
        for term in equation.terms() {
            ops.push(lower_term(equation.target(), term, schemes));
        }
        equations.push(DiscreteEquation {
            target: equation.target().clone(),
            ops,
        });
    }
    DiscreteSystem { equations }
}

fn lower_term(target: &FieldRef, term: &Term, schemes: &SchemeRegistry) -> DiscreteOp {
    let kind = match term.op {
        TermOp::Ddt => DiscreteOpKind::TimeDerivative,
        TermOp::Div => DiscreteOpKind::Convection,
        TermOp::Grad => DiscreteOpKind::Gradient,
        TermOp::Laplacian => DiscreteOpKind::Diffusion,
        TermOp::Source => DiscreteOpKind::Source,
    };
    DiscreteOp {
        target: target.clone(),
        kind,
        discretization: term.discretization,
        scheme: schemes.scheme_for(term),
        field: term.field.clone(),
        flux: term.flux.clone(),
        coeff: term.coeff.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::model::ast::{fvc, fvm, surface_scalar, vol_scalar, vol_vector};

    #[test]
    fn lower_system_maps_all_term_kinds() {
        let u = vol_vector("U");
        let p = vol_scalar("p");
        let phi = surface_scalar("phi");
        let mu = vol_scalar("mu");

        let mut eqn = crate::solver::model::ast::Equation::new(u.clone());
        eqn.add_term(fvm::ddt(u.clone()));
        eqn.add_term(fvm::div(phi.clone(), u.clone()));
        eqn.add_term(fvc::grad(p.clone()));
        eqn.add_term(fvm::laplacian(Coefficient::field(mu).unwrap(), u.clone()));
        eqn.add_term(fvc::source(u.clone()));

        let mut system = EquationSystem::new();
        system.add_equation(eqn);

        let registry = SchemeRegistry::new(Scheme::Upwind);
        let discrete = lower_system(&system, &registry);

        assert_eq!(discrete.equations.len(), 1);
        let ops = &discrete.equations[0].ops;
        assert_eq!(ops.len(), 5);
        assert_eq!(ops[0].kind, DiscreteOpKind::TimeDerivative);
        assert_eq!(ops[1].kind, DiscreteOpKind::Convection);
        assert_eq!(ops[2].kind, DiscreteOpKind::Gradient);
        assert_eq!(ops[3].kind, DiscreteOpKind::Diffusion);
        assert_eq!(ops[4].kind, DiscreteOpKind::Source);
        assert_eq!(ops[0].scheme, Scheme::Upwind);
    }

    #[test]
    fn lower_system_applies_scheme_registry() {
        let u = vol_vector("U");
        let phi = surface_scalar("phi");

        let mut eqn = crate::solver::model::ast::Equation::new(u.clone());
        eqn.add_term(fvm::div(phi.clone(), u.clone()));

        let mut system = EquationSystem::new();
        system.add_equation(eqn);

        let mut registry = SchemeRegistry::new(Scheme::Upwind);
        registry.set_for_term(TermOp::Div, Some(&phi), &u, Scheme::QUICK);

        let discrete = lower_system(&system, &registry);
        assert_eq!(discrete.equations[0].ops[0].scheme, Scheme::QUICK);
    }
}
