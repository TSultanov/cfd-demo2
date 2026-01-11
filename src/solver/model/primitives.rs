/// Derived primitive recovery specification for models.
///
/// Maps primitive field names (e.g., "p", "u_x") to expressions over conserved state.
/// Codegen uses these to generate primitive recovery kernels without hardcoding physics.

use crate::solver::shared::PrimitiveExpr;
use std::collections::HashMap;

/// Derived primitive recovery specification.
///
/// Defines how to compute primitive variables (p, u, T, etc.) from conserved state (rho, rho_u, rho_e).
#[derive(Debug, Clone, PartialEq)]
pub struct PrimitiveDerivations {
    /// Map: primitive_name → expression over state fields
    ///
    /// Example for Euler ideal gas:
    /// - "u_x" → rho_u_x / rho
    /// - "u_y" → rho_u_y / rho
    /// - "p" → (gamma - 1) * (rho_e - 0.5 * rho * u^2)
    pub derivations: HashMap<String, PrimitiveExpr>,
}

impl PrimitiveDerivations {
    /// No derived primitives (incompressible or primitives = state).
    pub fn identity() -> Self {
        Self {
            derivations: HashMap::new(),
        }
    }

    pub fn is_identity(&self) -> bool {
        self.derivations.is_empty()
    }

    /// Euler ideal gas primitive recovery.
    ///
    /// Conserved: rho, rho_u, rho_e
    /// Primitives: rho (identity), u, p
    ///
    /// Relations:
    /// - u = rho_u / rho
    /// - p = (gamma - 1) * (rho_e - 0.5 * rho * |u|^2)
    pub fn euler_ideal_gas(gamma: f32) -> Self {
        use PrimitiveExpr as E;
        let mut derivations = HashMap::new();

        // rho is conserved (identity mapping)
        derivations.insert("rho".into(), E::field("rho"));

        // u_x = rho_u_x / rho
        derivations.insert(
            "u_x".into(),
            E::Div(
                Box::new(E::field("rho_u_x")),
                Box::new(E::field("rho")),
            ),
        );

        // u_y = rho_u_y / rho
        derivations.insert(
            "u_y".into(),
            E::Div(
                Box::new(E::field("rho_u_y")),
                Box::new(E::field("rho")),
            ),
        );

        // kinetic_energy = 0.5 * (rho_u_x^2 + rho_u_y^2) / rho
        //
        // Important: avoid referencing derived primitives (u_x/u_y) when defining `p`.
        // Primitive recovery kernels may compute outputs in any order unless an
        // explicit dependency graph is enforced.
        let rho_u_sq = E::Add(
            Box::new(E::Mul(
                Box::new(E::field("rho_u_x")),
                Box::new(E::field("rho_u_x")),
            )),
            Box::new(E::Mul(
                Box::new(E::field("rho_u_y")),
                Box::new(E::field("rho_u_y")),
            )),
        );
        let ke = E::Mul(
            Box::new(E::lit(0.5)),
            Box::new(E::Div(Box::new(rho_u_sq), Box::new(E::field("rho")))),
        );

        // p = (gamma - 1) * (rho_e - ke)
        derivations.insert(
            "p".into(),
            E::Mul(
                Box::new(E::lit(gamma - 1.0)),
                Box::new(E::Sub(
                    Box::new(E::field("rho_e")),
                    Box::new(ke),
                )),
            ),
        );

        Self { derivations }
    }

    /// Get the primitive expression for a given field name.
    pub fn get(&self, name: &str) -> Option<&PrimitiveExpr> {
        self.derivations.get(name)
    }

    /// Check if a primitive derivation exists for this field.
    pub fn contains(&self, name: &str) -> bool {
        self.derivations.contains_key(name)
    }

    /// Iterator over all (name, expression) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &PrimitiveExpr)> {
        self.derivations.iter()
    }
}

impl Default for PrimitiveDerivations {
    fn default() -> Self {
        Self::identity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euler_ideal_gas_defines_expected_primitives() {
        let prims = PrimitiveDerivations::euler_ideal_gas(1.4);

        assert!(prims.contains("rho"));
        assert!(prims.contains("u_x"));
        assert!(prims.contains("u_y"));
        assert!(prims.contains("p"));

        // rho should be identity (just a field reference)
        match prims.get("rho").unwrap() {
            PrimitiveExpr::Field(name) => assert_eq!(name, "rho"),
            other => panic!("expected Field, got {:?}", other),
        }

        // u_x should be division
        match prims.get("u_x").unwrap() {
            PrimitiveExpr::Div(_, _) => {}
            other => panic!("expected Div, got {:?}", other),
        }

        // p should be multiplication (gamma-1) * (...)
        match prims.get("p").unwrap() {
            PrimitiveExpr::Mul(_, _) => {}
            other => panic!("expected Mul, got {:?}", other),
        }
    }

    #[test]
    fn identity_derivations_are_empty() {
        let prims = PrimitiveDerivations::identity();
        assert!(prims.derivations.is_empty());
    }
}
