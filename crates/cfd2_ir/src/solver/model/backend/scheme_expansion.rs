use std::collections::HashSet;

use super::ast::{EquationSystem, FieldRef, TermOp, UnitValidationError};
use super::SchemeRegistry;
use crate::solver::scheme::Scheme;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemeExpansion {
    gradient_fields: Vec<FieldRef>,
}

impl SchemeExpansion {
    pub fn gradient_fields(&self) -> &[FieldRef] {
        &self.gradient_fields
    }

    pub fn needs_gradients(&self) -> bool {
        !self.gradient_fields.is_empty()
    }
}

pub fn expand_schemes(
    system: &EquationSystem,
    schemes: &SchemeRegistry,
) -> Result<SchemeExpansion, UnitValidationError> {
    system.validate_units()?;

    let mut seen: HashSet<String> = HashSet::new();
    let mut gradient_fields = Vec::new();

    for equation in system.equations() {
        for term in equation.terms() {
            let scheme = schemes.scheme_for(term);
            let needs_gradient =
                matches!(term.op, TermOp::Div | TermOp::DivFlux) && scheme != Scheme::Upwind;
            if !needs_gradient {
                continue;
            }

            if seen.insert(term.field.name().to_string()) {
                gradient_fields.push(term.field);
            }
        }
    }

    Ok(SchemeExpansion { gradient_fields })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::model::backend::ast::{fvm, surface_scalar, vol_scalar, vol_vector};
    use crate::solver::units::si;

    #[test]
    fn expand_schemes_reports_no_gradients_for_upwind() {
        let u = vol_vector("U", si::VELOCITY);
        let phi = surface_scalar("phi", si::MASS_FLUX);

        let mut system = EquationSystem::new();
        system.add_equation((fvm::div(phi, u)).eqn(u));

        let registry = SchemeRegistry::new(Scheme::Upwind);
        let expansion = expand_schemes(&system, &registry).unwrap();
        assert!(!expansion.needs_gradients());
    }

    #[test]
    fn expand_schemes_collects_convection_fields_for_high_order() {
        let u = vol_vector("U", si::VELOCITY);
        let p = vol_scalar("p", si::PRESSURE);
        let phi = surface_scalar("phi", si::MASS_FLUX);

        let mut system = EquationSystem::new();
        system.add_equation((fvm::div(phi.clone(), u)).eqn(u));
        system.add_equation((fvm::div(phi, p)).eqn(p));

        let mut registry = SchemeRegistry::new(Scheme::Upwind);
        registry.set_for_op(TermOp::Div, Scheme::SecondOrderUpwind);

        let expansion = expand_schemes(&system, &registry).unwrap();
        assert!(expansion.needs_gradients());
        assert!(expansion.gradient_fields().iter().any(|f| f.name() == "U"));
        assert!(expansion.gradient_fields().iter().any(|f| f.name() == "p"));
    }
}
