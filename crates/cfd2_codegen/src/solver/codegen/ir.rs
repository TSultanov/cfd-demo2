use crate::solver::scheme::Scheme;

use crate::solver::ir::{
    Coefficient, Discretization, EquationSystem, FieldRef, FluxRef, SchemeRegistry, Term, TermOp,
    UnitValidationError,
};

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
    pub term_op: TermOp,
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

/// Lower an equation system to a discrete system without validation.
///
/// This is a "trusted" variant for systems that have already been validated
/// (e.g., built with the typed builder). Prefer `lower_system` for untyped
/// construction paths where validation is needed as a backstop.
pub fn lower_system_unchecked(
    system: &EquationSystem,
    schemes: &SchemeRegistry,
) -> DiscreteSystem {
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

/// Lower an equation system to a discrete system with validation.
///
/// This is the standard entry point that validates units before lowering.
/// For systems already known to be valid (e.g., typed-built), use
/// `lower_system_unchecked` to avoid redundant validation.
pub fn lower_system(
    system: &EquationSystem,
    schemes: &SchemeRegistry,
) -> Result<DiscreteSystem, UnitValidationError> {
    system.validate_units()?;
    Ok(lower_system_unchecked(system, schemes))
}

fn lower_term(target: &FieldRef, term: &Term, schemes: &SchemeRegistry) -> DiscreteOp {
    let kind = match term.op {
        TermOp::Ddt => DiscreteOpKind::TimeDerivative,
        TermOp::Div | TermOp::DivFlux => DiscreteOpKind::Convection,
        TermOp::Grad => DiscreteOpKind::Gradient,
        TermOp::Laplacian => DiscreteOpKind::Diffusion,
        TermOp::Source => DiscreteOpKind::Source,
    };
    DiscreteOp {
        target: target.clone(),
        kind,
        term_op: term.op,
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
    use crate::solver::ir::{
        fvc, fvm, surface_scalar_dim, vol_scalar_dim, vol_vector_dim,
    };
    use cfd2_ir::solver::dimensions::{
        Density, DynamicViscosity, MassFlux, Pressure, PressureGradient, UnitDimension, Velocity,
    };

    #[test]
    fn lower_system_maps_all_term_kinds() {
        let u = vol_vector_dim::<Velocity>("U");
        let p = vol_scalar_dim::<Pressure>("p");
        let rho = vol_scalar_dim::<Density>("rho");
        let phi = surface_scalar_dim::<MassFlux>("phi");
        let mu = vol_scalar_dim::<DynamicViscosity>("mu");

        let mut eqn = crate::solver::ir::Equation::new(u.clone());
        eqn.add_term(fvm::ddt_coeff(Coefficient::field(rho).unwrap(), u.clone()));
        eqn.add_term(fvm::div(phi.clone(), u.clone()));
        eqn.add_term(fvc::grad(p.clone()));
        eqn.add_term(fvm::laplacian(Coefficient::field(mu).unwrap(), u.clone()));
        eqn.add_term(fvc::source_coeff(
            Coefficient::constant_unit(1.0, PressureGradient::UNIT),
            u.clone(),
        ));

        let mut system = EquationSystem::new();
        system.add_equation(eqn);

        let registry = SchemeRegistry::new(Scheme::Upwind);
        let discrete = lower_system(&system, &registry).unwrap();

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
        let u = vol_vector_dim::<Velocity>("U");
        let phi = surface_scalar_dim::<MassFlux>("phi");

        let mut eqn = crate::solver::ir::Equation::new(u.clone());
        eqn.add_term(fvm::div(phi.clone(), u.clone()));

        let mut system = EquationSystem::new();
        system.add_equation(eqn);

        let mut registry = SchemeRegistry::new(Scheme::Upwind);
        registry.set_for_term(TermOp::Div, Some(&phi), &u, Scheme::QUICK);

        let discrete = lower_system(&system, &registry).unwrap();
        assert_eq!(discrete.equations[0].ops[0].scheme, Scheme::QUICK);
    }
}
