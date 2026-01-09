use crate::solver::codegen::state_access::state_scalar;
use crate::solver::codegen::wgsl_ast::Expr;
use crate::solver::model::backend::{Coefficient, FieldKind, StateLayout};

#[derive(Clone)]
enum CoeffSample<'a> {
    Cell {
        idx: &'a str,
    },
    Face {
        owner_idx: &'a str,
        neighbor_idx: &'a str,
        interp: Expr,
    },
}

fn f32_literal(value: f64) -> Expr {
    value.into()
}

fn coeff_named_expr(name: &str) -> Option<Expr> {
    match name {
        "rho" => Some(Expr::ident("constants").field("density")),
        // Dynamic viscosity (SI): Pa·s = kg/(m·s). Historically this was called `nu`,
        // but `nu` is conventionally kinematic viscosity; accept both for now.
        "mu" | "nu" => Some(Expr::ident("constants").field("viscosity")),
        _ => None,
    }
}

fn coeff_expr(layout: &StateLayout, coeff: &Coefficient, sample: CoeffSample<'_>) -> Expr {
    match coeff {
        Coefficient::Constant { value, .. } => f32_literal(*value),
        Coefficient::Field(field) => {
            if let Some(state_field) = layout.field(field.name()) {
                if state_field.kind() != FieldKind::Scalar {
                    panic!("coefficient '{}' must be scalar", field.name());
                }
                match sample {
                    CoeffSample::Cell { idx } => state_scalar(layout, "state", idx, field.name()),
                    CoeffSample::Face {
                        owner_idx,
                        neighbor_idx,
                        interp,
                    } => {
                        let own = state_scalar(layout, "state", owner_idx, field.name());
                        let neigh = state_scalar(layout, "state", neighbor_idx, field.name());
                        interp * own + (Expr::from(1.0) - interp) * neigh
                    }
                }
            } else {
                coeff_named_expr(field.name()).unwrap_or_else(|| {
                    panic!(
                        "missing coefficient field '{}' in state layout",
                        field.name()
                    )
                })
            }
        }
        Coefficient::Product(lhs, rhs) => {
            let lhs_expr = coeff_expr(layout, lhs, sample.clone());
            let rhs_expr = coeff_expr(layout, rhs, sample);
            lhs_expr * rhs_expr
        }
    }
}

pub fn coeff_cell_expr(
    layout: &StateLayout,
    coeff: Option<&Coefficient>,
    idx: &str,
    fallback: Expr,
) -> Expr {
    match coeff {
        None => fallback,
        Some(value) => coeff_expr(layout, value, CoeffSample::Cell { idx }),
    }
}

pub fn coeff_face_expr(
    layout: &StateLayout,
    coeff: Option<&Coefficient>,
    owner_idx: &str,
    neighbor_idx: &str,
    interp: Expr,
    fallback: Expr,
) -> Expr {
    match coeff {
        None => fallback,
        Some(value) => coeff_expr(
            layout,
            value,
            CoeffSample::Face {
                owner_idx,
                neighbor_idx,
                interp,
            },
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::model::backend::ast::{vol_scalar, vol_vector};
    use crate::solver::units::si;

    #[test]
    fn coeff_expr_handles_product_and_constants() {
        let rho = vol_scalar("rho", si::DENSITY);
        let d_p = vol_scalar("d_p", si::D_P);
        let layout = StateLayout::new(vec![d_p]);
        let coeff = Coefficient::product(
            Coefficient::field(rho).unwrap(),
            Coefficient::field(vol_scalar("d_p", si::D_P)).unwrap(),
        )
        .unwrap();

        let expr = coeff_face_expr(&layout, Some(&coeff), "i", "j", 0.5.into(), 1.0.into());
        assert_eq!(
            expr.to_string(),
            "constants.density * (0.5 * state[i * 1u + 0u] + (1.0 - 0.5) * state[j * 1u + 0u])"
        );
    }

    #[test]
    fn coeff_expr_rejects_vector_coefficients() {
        let u = vol_vector("U", si::VELOCITY);
        let layout = StateLayout::new(vec![vol_scalar("p", si::PRESSURE)]);
        let err = Coefficient::product(
            Coefficient::field(vol_scalar("p", si::PRESSURE)).unwrap(),
            Coefficient::Field(u),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            crate::solver::model::backend::ast::CodegenError::NonScalarCoefficient { .. }
        ));

        let coeff = Coefficient::field(vol_scalar("p", si::PRESSURE)).unwrap();
        let expr = coeff_cell_expr(&layout, Some(&coeff), "idx", 1.0.into());
        assert_eq!(expr.to_string(), "state[idx * 1u + 0u]");
    }
}
