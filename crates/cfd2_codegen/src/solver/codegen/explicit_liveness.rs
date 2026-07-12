//! Model-derived liveness for matrix-free explicit time integration.
//!
//! The coupled row layout is intentionally broader than the RK workspace
//! layout: a model may append local algebraic closure rows after (or between)
//! its differential equations.  Those rows are useful to the implicit solver,
//! but the explicit path recovers them from `PrimitiveDerivations` and assigns
//! them a zero rate.  Materialising them in `rk_base` / `rk_accum` therefore
//! wastes bandwidth and, more subtly, makes every stage solve a larger local
//! mass block than the method-of-lines system requires.
//!
//! This module derives the compact layout from equation semantics only.  It
//! never switches on a model id or field name.

use super::coupled_common::coupled_unknown_components;
use super::ir::{DiscreteOpKind, DiscreteSystem};
use crate::solver::ir::{EquationSystem, FieldRef, TermOp};
use std::collections::HashMap;

/// One scalar component in the coupled equation layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExplicitComponent {
    /// Field targeted by the equation row.
    pub field: FieldRef,
    /// Component within the target field.
    pub component: u32,
    /// Position in the full coupled residual / face-flux layout.
    pub coupled_rank: u32,
    /// Position in the compact RK workspace, or `None` for an algebraic row.
    pub differential_rank: Option<u32>,
}

/// Exact differential/algebraic partition used by explicit RK.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExplicitRkLayout {
    components: Vec<ExplicitComponent>,
    differential: Vec<ExplicitComponent>,
    coupled_offsets: HashMap<String, u32>,
}

impl ExplicitRkLayout {
    /// Derive liveness from lowered equation IR.
    ///
    /// A target is differential iff its equation declares an own-variable
    /// time derivative.  This is exactly the capability rule enforced by
    /// `ModelSpec::validate_explicit_rk4`; rows without one are recovered by
    /// the model's ordered local primitive closure.
    pub fn from_discrete_system(system: &DiscreteSystem) -> Self {
        let mut components = Vec::new();
        let mut differential = Vec::new();
        let mut coupled_offsets = HashMap::new();
        let mut coupled_rank = 0u32;

        for equation in &system.equations {
            coupled_offsets.insert(equation.target.name().to_string(), coupled_rank);
            let owns_ddt = equation
                .ops
                .iter()
                .any(|op| op.kind == DiscreteOpKind::TimeDerivative && op.field == equation.target);
            for component in 0..equation.target.kind().component_count() as u32 {
                let differential_rank = owns_ddt.then_some(differential.len() as u32);
                let entry = ExplicitComponent {
                    field: equation.target,
                    component,
                    coupled_rank,
                    differential_rank,
                };
                components.push(entry);
                if owns_ddt {
                    differential.push(entry);
                }
                coupled_rank += 1;
            }
        }

        debug_assert_eq!(
            components.len(),
            coupled_unknown_components(system).len(),
            "explicit liveness must preserve the full coupled component order"
        );

        Self {
            components,
            differential,
            coupled_offsets,
        }
    }

    pub fn coupled_stride(&self) -> u32 {
        self.components.len() as u32
    }

    pub fn differential_stride(&self) -> u32 {
        self.differential.len() as u32
    }

    pub fn algebraic_stride(&self) -> u32 {
        self.coupled_stride() - self.differential_stride()
    }

    pub fn components(&self) -> &[ExplicitComponent] {
        &self.components
    }

    pub fn differential_components(&self) -> &[ExplicitComponent] {
        &self.differential
    }

    pub fn differential_rank_for_coupled(&self, coupled_rank: u32) -> Option<u32> {
        self.components
            .get(coupled_rank as usize)
            .and_then(|component| component.differential_rank)
    }

    pub fn coupled_offset(&self, field_name: &str) -> Option<u32> {
        self.coupled_offsets.get(field_name).copied()
    }
}

/// Source-IR counterpart used by runtime resource recipes before lowering.
///
/// Keeping this query next to the lowered layout makes allocation and codegen
/// share one semantic definition without requiring model identifiers.
pub fn differential_component_count(system: &EquationSystem) -> u32 {
    system
        .equations()
        .iter()
        .filter(|equation| {
            equation
                .terms()
                .iter()
                .any(|term| term.op == TermOp::Ddt && term.field == *equation.target())
        })
        .map(|equation| equation.target().kind().component_count() as u32)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::codegen::ir::lower_system;
    use crate::solver::ir::{fvm, vol_scalar_dim, Equation, SchemeRegistry};
    use crate::solver::scheme::Scheme;
    use cfd2_ir::dimensions::Dimensionless;

    fn mixed_system() -> EquationSystem {
        let q = vol_scalar_dim::<Dimensionless>("q");
        let closure = vol_scalar_dim::<Dimensionless>("closure");
        let r = vol_scalar_dim::<Dimensionless>("r");

        let mut q_eq = Equation::new(q);
        q_eq.add_term(fvm::ddt(q));
        // A live differential cross-ddt: the compact mass block must retain it.
        q_eq.add_term(fvm::ddt(r));

        // Local algebraic row: no own ddt.
        let closure_eq = Equation::new(closure);

        let mut r_eq = Equation::new(r);
        r_eq.add_term(fvm::ddt(r));
        // A ddt of an algebraic column has zero rate in the incumbent explicit
        // semantics.  It must not make the closure row live in RK storage.
        r_eq.add_term(fvm::ddt(closure));

        let mut system = EquationSystem::new();
        system.add_equation(q_eq);
        system.add_equation(closure_eq);
        system.add_equation(r_eq);
        system
    }

    #[test]
    fn partitions_interleaved_algebraic_rows_without_model_names() {
        let system = mixed_system();
        assert_eq!(differential_component_count(&system), 2);

        let discrete = lower_system(&system, &SchemeRegistry::new(Scheme::Upwind)).unwrap();
        let layout = ExplicitRkLayout::from_discrete_system(&discrete);
        assert_eq!(layout.coupled_stride(), 3);
        assert_eq!(layout.differential_stride(), 2);
        assert_eq!(layout.algebraic_stride(), 1);
        assert_eq!(layout.differential_rank_for_coupled(0), Some(0));
        assert_eq!(layout.differential_rank_for_coupled(1), None);
        assert_eq!(layout.differential_rank_for_coupled(2), Some(1));
        assert_eq!(layout.coupled_offset("q"), Some(0));
        assert_eq!(layout.coupled_offset("closure"), Some(1));
        assert_eq!(layout.coupled_offset("r"), Some(2));
    }
}
