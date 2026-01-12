use std::collections::HashMap;

use crate::solver::scheme::Scheme;

use super::ast::{FieldRef, FluxRef, Term, TermOp};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TermKey {
    op: TermOp,
    field: String,
    flux: Option<String>,
}

impl TermKey {
    pub fn new(op: TermOp, field: &FieldRef, flux: Option<&FluxRef>) -> Self {
        Self {
            op,
            field: field.name().to_string(),
            flux: flux.map(|value| value.name().to_string()),
        }
    }

    pub fn from_term(term: &Term) -> Self {
        Self::new(term.op, &term.field, term.flux.as_ref())
    }
}

#[derive(Debug, Clone)]
pub struct SchemeRegistry {
    default_scheme: Scheme,
    schemes_by_term: HashMap<TermKey, Scheme>,
    schemes_by_op: HashMap<TermOp, Scheme>,
}

impl SchemeRegistry {
    pub fn new(default_scheme: Scheme) -> Self {
        Self {
            default_scheme,
            schemes_by_term: HashMap::new(),
            schemes_by_op: HashMap::new(),
        }
    }

    pub fn set_for_term(
        &mut self,
        op: TermOp,
        flux: Option<&FluxRef>,
        field: &FieldRef,
        scheme: Scheme,
    ) {
        let key = TermKey::new(op, field, flux);
        self.schemes_by_term.insert(key, scheme);
    }

    pub fn set_for_term_names(
        &mut self,
        op: TermOp,
        flux: Option<&str>,
        field: &str,
        scheme: Scheme,
    ) {
        let key = TermKey {
            op,
            field: field.to_string(),
            flux: flux.map(|value| value.to_string()),
        };
        self.schemes_by_term.insert(key, scheme);
    }

    pub fn set_for_op(&mut self, op: TermOp, scheme: Scheme) {
        self.schemes_by_op.insert(op, scheme);
    }

    pub fn scheme_for(&self, term: &Term) -> Scheme {
        let key = TermKey::from_term(term);
        if let Some(scheme) = self.schemes_by_term.get(&key) {
            return *scheme;
        }
        if let Some(scheme) = self.schemes_by_op.get(&term.op) {
            return *scheme;
        }
        self.default_scheme
    }
}

impl Default for SchemeRegistry {
    fn default() -> Self {
        Self::new(Scheme::Upwind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::model::backend::ast::{fvm, surface_scalar, vol_scalar};
    use crate::solver::units::si;

    #[test]
    fn scheme_registry_uses_default_when_unset() {
        let registry = SchemeRegistry::new(Scheme::Upwind);
        let term = fvm::div(
            surface_scalar("phi", si::MASS_FLUX),
            vol_scalar("U", si::VELOCITY),
        );
        assert_eq!(registry.scheme_for(&term), Scheme::Upwind);
    }

    #[test]
    fn scheme_registry_uses_op_override() {
        let mut registry = SchemeRegistry::new(Scheme::Upwind);
        registry.set_for_op(TermOp::Div, Scheme::QUICK);

        let term = fvm::div(
            surface_scalar("phi", si::MASS_FLUX),
            vol_scalar("U", si::VELOCITY),
        );
        assert_eq!(registry.scheme_for(&term), Scheme::QUICK);
    }

    #[test]
    fn scheme_registry_prefers_term_override() {
        let mut registry = SchemeRegistry::new(Scheme::Upwind);
        registry.set_for_op(TermOp::Div, Scheme::QUICK);

        let flux = surface_scalar("phi", si::MASS_FLUX);
        let field = vol_scalar("U", si::VELOCITY);
        registry.set_for_term(TermOp::Div, Some(&flux), &field, Scheme::SecondOrderUpwind);

        let term = fvm::div(flux, field);
        assert_eq!(registry.scheme_for(&term), Scheme::SecondOrderUpwind);
    }

    #[test]
    fn scheme_registry_default_impl_uses_upwind() {
        let registry = SchemeRegistry::default();
        let term = fvm::div(
            surface_scalar("phi", si::MASS_FLUX),
            vol_scalar("U", si::VELOCITY),
        );
        assert_eq!(registry.scheme_for(&term), Scheme::Upwind);
    }

    #[test]
    fn scheme_registry_set_for_term_names_uses_string_keys() {
        let mut registry = SchemeRegistry::new(Scheme::Upwind);
        registry.set_for_term_names(TermOp::Div, Some("phi"), "U", Scheme::QUICK);

        let term = fvm::div(
            surface_scalar("phi", si::MASS_FLUX),
            vol_scalar("U", si::VELOCITY),
        );
        assert_eq!(registry.scheme_for(&term), Scheme::QUICK);
    }
}
