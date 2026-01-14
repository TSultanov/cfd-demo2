/// Derived primitive recovery specification for models.
///
/// Maps primitive field names (e.g., "p", "u_x") to expressions over conserved state.
/// Codegen uses these to generate primitive recovery kernels without hardcoding physics.

use crate::solver::shared::PrimitiveExpr;
use std::collections::HashMap;
use std::collections::{BTreeSet, VecDeque};

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

    /// Euler ideal gas derived pressure + temperature, assuming velocity is already present in
    /// the state as a coupled unknown (i.e. no `u_x/u_y` recovery).
    ///
    /// Conserved: rho, rho_u, rho_e
    /// Primitives: rho (identity), p, T
    ///
    /// Relations:
    /// - p = (gamma - 1) * (rho_e - 0.5 * (rho_u_x^2 + rho_u_y^2) / rho)
    /// - T = p / rho  (solver nondimensional units; equivalent to p / (rho * R) with R=1)
    pub fn euler_ideal_gas_pressure_t(gamma: f32) -> Self {
        use PrimitiveExpr as E;
        let mut derivations = HashMap::new();

        // rho is conserved (identity mapping)
        derivations.insert("rho".into(), E::field("rho"));

        // kinetic_energy = 0.5 * (rho_u_x^2 + rho_u_y^2) / rho
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

        // T = p / rho
        derivations.insert(
            "T".into(),
            E::Div(Box::new(E::field("p")), Box::new(E::field("rho"))),
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

    /// Return primitive derivations in a deterministic dependency order.
    ///
    /// This allows derived primitives to reference other derived primitives safely, as long as
    /// the dependency graph is acyclic.
    pub fn ordered(&self) -> Result<Vec<(String, PrimitiveExpr)>, String> {
        fn collect_field_refs(expr: &PrimitiveExpr, out: &mut BTreeSet<String>) {
            match expr {
                PrimitiveExpr::Literal(_) => {}
                PrimitiveExpr::Field(name) => {
                    out.insert(name.clone());
                }
                PrimitiveExpr::Add(lhs, rhs)
                | PrimitiveExpr::Sub(lhs, rhs)
                | PrimitiveExpr::Mul(lhs, rhs)
                | PrimitiveExpr::Div(lhs, rhs) => {
                    collect_field_refs(lhs, out);
                    collect_field_refs(rhs, out);
                }
                PrimitiveExpr::Sqrt(inner) | PrimitiveExpr::Neg(inner) => {
                    collect_field_refs(inner, out);
                }
            }
        }

        let mut nodes: Vec<String> = self.derivations.keys().cloned().collect();
        nodes.sort();

        let mut deps: HashMap<String, BTreeSet<String>> = HashMap::new();
        for name in &nodes {
            let expr = self
                .derivations
                .get(name)
                .ok_or_else(|| format!("missing primitive derivation for '{name}'"))?;

            let mut fields = BTreeSet::new();
            collect_field_refs(expr, &mut fields);

            let mut d = BTreeSet::new();
            for f in fields {
                if self.derivations.contains_key(&f) {
                    if f == *name {
                        // Allow identity mappings like `rho = rho` (these are redundant but harmless).
                        if matches!(expr, PrimitiveExpr::Field(inner) if inner == name) {
                            continue;
                        }
                        return Err(format!("primitive '{name}' depends on itself"));
                    }
                    d.insert(f);
                }
            }
            deps.insert(name.clone(), d);
        }

        // Kahn topo sort with deterministic (lexicographic) queue behavior.
        let mut indegree: HashMap<String, usize> = HashMap::new();
        let mut reverse: HashMap<String, Vec<String>> = HashMap::new();

        for name in &nodes {
            let d = deps.get(name).expect("deps missing for node");
            indegree.insert(name.clone(), d.len());
            for dep in d {
                reverse
                    .entry(dep.clone())
                    .or_default()
                    .push(name.clone());
            }
        }

        for v in reverse.values_mut() {
            v.sort();
        }

        let mut ready = VecDeque::new();
        for name in &nodes {
            if indegree.get(name).copied().unwrap_or(0) == 0 {
                ready.push_back(name.clone());
            }
        }

        let mut ordered = Vec::with_capacity(nodes.len());
        while let Some(name) = ready.pop_front() {
            ordered.push(name.clone());
            if let Some(children) = reverse.get(&name) {
                for child in children {
                    let e = indegree
                        .get_mut(child)
                        .ok_or_else(|| format!("missing indegree for '{child}'"))?;
                    *e = e
                        .checked_sub(1)
                        .ok_or_else(|| format!("indegree underflow for '{child}'"))?;
                    if *e == 0 {
                        // Maintain a stable overall order by inserting then sorting once per push.
                        ready.push_back(child.clone());
                    }
                }
            }
            // Ensure deterministic tie-breaking among newly-ready nodes.
            if ready.len() > 1 {
                let mut tmp: Vec<_> = ready.drain(..).collect();
                tmp.sort();
                ready.extend(tmp);
            }
        }

        if ordered.len() != nodes.len() {
            let mut remaining: Vec<String> = nodes
                .into_iter()
                .filter(|n| !ordered.contains(n))
                .collect();
            remaining.sort();
            return Err(format!(
                "primitive derivations contain a dependency cycle: {:?}",
                remaining
            ));
        }

        let mut out = Vec::with_capacity(ordered.len());
        for name in ordered {
            let expr = self
                .derivations
                .get(&name)
                .ok_or_else(|| format!("missing primitive derivation for '{name}'"))?;
            out.push((name, expr.clone()));
        }
        Ok(out)
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

    #[test]
    fn ordered_allows_derived_to_depend_on_derived() {
        use PrimitiveExpr as E;

        let mut derivations = HashMap::new();
        derivations.insert("rho".into(), E::field("rho"));
        derivations.insert(
            "u_x".into(),
            E::Div(Box::new(E::field("rho_u_x")), Box::new(E::field("rho"))),
        );
        derivations.insert(
            "p".into(),
            E::Mul(Box::new(E::lit(1.0)), Box::new(E::field("u_x"))),
        );

        let prims = PrimitiveDerivations { derivations };
        let ordered = prims.ordered().unwrap();
        let names: Vec<_> = ordered.iter().map(|(n, _)| n.as_str()).collect();

        let rho_i = names.iter().position(|&n| n == "rho").unwrap();
        let u_i = names.iter().position(|&n| n == "u_x").unwrap();
        let p_i = names.iter().position(|&n| n == "p").unwrap();
        assert!(rho_i < u_i);
        assert!(u_i < p_i);
    }

    #[test]
    fn ordered_rejects_cycles() {
        use PrimitiveExpr as E;

        let mut derivations = HashMap::new();
        derivations.insert("a".into(), E::field("b"));
        derivations.insert("b".into(), E::field("a"));

        let prims = PrimitiveDerivations { derivations };
        assert!(prims.ordered().unwrap_err().contains("cycle"));
    }
}
