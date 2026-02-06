use crate::solver::model::backend::ast::EquationSystem;
use crate::solver::model::backend::{Coefficient as BackendCoeff, FieldKind, FieldRef, TermOp};
use crate::solver::model::ModelSpec;

#[derive(Debug, Clone, Copy)]
pub struct MomentumPressureCoupling {
    pub momentum: FieldRef,
    pub pressure: FieldRef,
}

fn collect_coeff_fields(coeff: &BackendCoeff, out: &mut Vec<FieldRef>) {
    match coeff {
        BackendCoeff::Constant { .. } => {}
        BackendCoeff::Field(field) => out.push(*field),
        BackendCoeff::MagSqr(field) => out.push(*field),
        BackendCoeff::Product(lhs, rhs) => {
            collect_coeff_fields(lhs, out);
            collect_coeff_fields(rhs, out);
        }
    }
}

/// Infer the unique (momentum, pressure) coupling from the equation system where:
/// - a vector equation has a `Grad(pressure)` term
/// - the pressure equation has a Laplacian term
/// - the Laplacian coefficient references `dp_field_name`
///
/// This system-only helper works without a full ModelSpec, allowing it to be used
/// at module construction time (before ModelSpec exists).
pub fn infer_unique_momentum_pressure_coupling_referencing_dp_system(
    system: &EquationSystem,
    dp_field_name: &str,
) -> Result<MomentumPressureCoupling, String> {
    let equations = system.equations();
    let mut eq_by_target = std::collections::HashMap::new();
    for (idx, eq) in equations.iter().enumerate() {
        eq_by_target.insert(*eq.target(), idx);
    }

    let mut candidates = Vec::new();
    for eq in equations {
        if !matches!(eq.target().kind(), FieldKind::Vector2 | FieldKind::Vector3) {
            continue;
        }
        for term in eq.terms() {
            if term.op != TermOp::Grad || term.field.kind() != FieldKind::Scalar {
                continue;
            }
            let Some(&p_eq_idx) = eq_by_target.get(&term.field) else {
                continue;
            };
            let p_eq = &system.equations()[p_eq_idx];
            let Some(lap) = p_eq.terms().iter().find(|t| t.op == TermOp::Laplacian) else {
                continue;
            };
            let Some(coeff) = &lap.coeff else {
                continue;
            };
            let mut coeff_fields = Vec::new();
            collect_coeff_fields(coeff, &mut coeff_fields);
            if coeff_fields.iter().any(|f| f.name() == dp_field_name) {
                candidates.push((*eq.target(), term.field));
            }
        }
    }

    match candidates.as_slice() {
        [(m, p)] => Ok(MomentumPressureCoupling {
            momentum: *m,
            pressure: *p,
        }),
        [] => Err(format!(
            "requires a unique momentum-pressure coupling referencing '{dp_field_name}'"
        )),
        many => Err(format!(
            "requires a unique momentum-pressure coupling referencing '{dp_field_name}', found {} candidates",
            many.len()
        )),
    }
}

/// Infer the unique (momentum, pressure) coupling from the equation system where:
/// - a vector equation has a `Grad(pressure)` term
/// - the pressure equation has a Laplacian term
/// - the Laplacian coefficient references `dp_field_name`
///
/// This is used by dp-based Rhie–Chow helpers and the `dp_update_from_diag` kernel.
pub fn infer_unique_momentum_pressure_coupling_referencing_dp(
    model: &ModelSpec,
    dp_field_name: &str,
) -> Result<MomentumPressureCoupling, String> {
    infer_unique_momentum_pressure_coupling_referencing_dp_system(&model.system, dp_field_name)
}
