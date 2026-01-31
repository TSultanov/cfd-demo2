use crate::solver::codegen::state_access::{state_component, state_scalar};
use crate::solver::codegen::wgsl_ast::Expr;
use crate::solver::ir::ports::{PortFieldKind, ResolvedStateSlotsSpec};
use crate::solver::ir::Coefficient;

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
        "inv_dt" => {
            let dt = Expr::ident("constants").field("dt");
            let dtau = Expr::ident("constants").field("dtau");
            let dt_eff = Expr::call_named("select", vec![dt, dtau.clone(), dtau.clone().gt(0.0)]);
            Some(Expr::from(1.0) / dt_eff)
        }
        // Dynamic viscosity (SI): Pa·s = kg/(m·s). Historically this was called `nu`,
        // but `nu` is conventionally kinematic viscosity; accept both for now.
        "mu" | "nu" => Some(Expr::ident("constants").field("viscosity")),

        // Thermal conductivity for ideal-gas (laminar) OpenFOAM reference alignment:
        //   kappa = mu * Cp / Pr
        // with Cp = gamma/(gamma-1) * R.
        //
        // Notes:
        // - kappa has units of W/(m·K) (POWER/(LENGTH*TEMPERATURE)).
        // - Pr is fixed at the OpenFOAM reference value (0.71) for now.
        "kappa" => {
            let mu = Expr::ident("constants").field("viscosity");
            let gamma = Expr::ident("constants").field("eos_gamma");
            let r = Expr::ident("constants").field("eos_r");
            let gm1 = Expr::call_named(
                "max",
                vec![Expr::ident("constants").field("eos_gm1"), Expr::from(1e-12)],
            );
            let pr = Expr::from(0.71);
            let cp = gamma * r / gm1;
            Some(mu * cp / pr)
        }
        "eos_gamma" => Some(Expr::ident("constants").field("eos_gamma")),
        "eos_gm1" => Some(Expr::ident("constants").field("eos_gm1")),
        "eos_r" => Some(Expr::ident("constants").field("eos_r")),
        "eos_dp_drho" => Some(Expr::ident("constants").field("eos_dp_drho")),
        "eos_p_offset" => Some(Expr::ident("constants").field("eos_p_offset")),
        "eos_theta_ref" => Some(Expr::ident("constants").field("eos_theta_ref")),
        _ => None,
    }
}

/// Find a slot by name in the resolved state slots spec.
fn find_slot<'a>(slots: &'a ResolvedStateSlotsSpec, field: &str) -> Option<&'a crate::solver::ir::ports::ResolvedStateSlotSpec> {
    slots.slots.iter().find(|s| s.name == field)
}

fn coeff_expr(slots: &ResolvedStateSlotsSpec, coeff: &Coefficient, sample: CoeffSample<'_>) -> Expr {
    match coeff {
        Coefficient::Constant { value, .. } => f32_literal(*value),
        Coefficient::Field(field) => {
            if let Some(slot) = find_slot(slots, field.name()) {
                if slot.kind != PortFieldKind::Scalar {
                    panic!("coefficient '{}' must be scalar", field.name());
                }
                match sample {
                    CoeffSample::Cell { idx } => state_scalar(slots, "state", idx, field.name()),
                    CoeffSample::Face {
                        owner_idx,
                        neighbor_idx,
                        interp,
                    } => {
                        let own = state_scalar(slots, "state", owner_idx, field.name());
                        let neigh = state_scalar(slots, "state", neighbor_idx, field.name());
                        interp * own + (Expr::from(1.0) - interp) * neigh
                    }
                }
            } else {
                coeff_named_expr(field.name()).unwrap_or_else(|| {
                    panic!(
                        "missing coefficient field '{}' in resolved state slots",
                        field.name()
                    )
                })
            }
        }
        Coefficient::MagSqr(field) => {
            let Some(slot) = find_slot(slots, field.name()) else {
                panic!(
                    "missing coefficient field '{}' in resolved state slots",
                    field.name()
                );
            };

            let mag_sqr_at = |idx: &str| match slot.kind {
                PortFieldKind::Scalar => {
                    let v = state_scalar(slots, "state", idx, field.name());
                    v.clone() * v
                }
                PortFieldKind::Vector2 => {
                    let x = state_component(slots, "state", idx, field.name(), 0);
                    let y = state_component(slots, "state", idx, field.name(), 1);
                    x.clone() * x + y.clone() * y
                }
                PortFieldKind::Vector3 => {
                    let x = state_component(slots, "state", idx, field.name(), 0);
                    let y = state_component(slots, "state", idx, field.name(), 1);
                    let z = state_component(slots, "state", idx, field.name(), 2);
                    x.clone() * x + y.clone() * y + z.clone() * z
                }
            };

            match sample {
                CoeffSample::Cell { idx } => mag_sqr_at(idx),
                CoeffSample::Face {
                    owner_idx,
                    neighbor_idx,
                    interp,
                } => {
                    let own = mag_sqr_at(owner_idx);
                    let neigh = mag_sqr_at(neighbor_idx);
                    interp * own + (Expr::from(1.0) - interp) * neigh
                }
            }
        }
        Coefficient::Product(lhs, rhs) => {
            let lhs_expr = coeff_expr(slots, lhs, sample.clone());
            let rhs_expr = coeff_expr(slots, rhs, sample);
            lhs_expr * rhs_expr
        }
    }
}

pub fn coeff_cell_expr(
    slots: &ResolvedStateSlotsSpec,
    coeff: Option<&Coefficient>,
    idx: &str,
    fallback: Expr,
) -> Expr {
    match coeff {
        None => fallback,
        Some(value) => coeff_expr(slots, value, CoeffSample::Cell { idx }),
    }
}

pub fn coeff_face_expr(
    slots: &ResolvedStateSlotsSpec,
    coeff: Option<&Coefficient>,
    owner_idx: &str,
    neighbor_idx: &str,
    interp: Expr,
    fallback: Expr,
) -> Expr {
    match coeff {
        None => fallback,
        Some(value) => coeff_expr(
            slots,
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
    use crate::solver::ir::ports::{PortFieldKind, ResolvedStateSlotSpec, ResolvedStateSlotsSpec};
    use crate::solver::ir::vol_scalar;
    use crate::solver::units::si;

    /// Helper to create a ResolvedStateSlotsSpec from a StateLayout for testing.
    fn slots_from_layout(layout: &crate::solver::ir::StateLayout) -> ResolvedStateSlotsSpec {
        let mut slots = Vec::new();
        for field in layout.fields() {
            let kind = match field.kind() {
                crate::solver::ir::FieldKind::Scalar => PortFieldKind::Scalar,
                crate::solver::ir::FieldKind::Vector2 => PortFieldKind::Vector2,
                crate::solver::ir::FieldKind::Vector3 => PortFieldKind::Vector3,
            };
            slots.push(ResolvedStateSlotSpec {
                name: field.name().to_string(),
                kind,
                unit: field.unit(),
                base_offset: field.offset(),
            });
        }
        ResolvedStateSlotsSpec {
            stride: layout.stride(),
            slots,
        }
    }

    #[test]
    fn coeff_expr_inv_dt_prefers_dtau_when_set() {
        let inv_dt = coeff_named_expr("inv_dt").expect("inv_dt");
        assert_eq!(
            inv_dt.to_string(),
            "1.0 / select(constants.dt, constants.dtau, constants.dtau > 0.0)"
        );
    }

    #[test]
    fn coeff_expr_handles_product_and_constants() {
        let rho = vol_scalar("rho", si::DENSITY);
        let d_p = vol_scalar("d_p", si::D_P);
        let layout = crate::solver::ir::StateLayout::new(vec![d_p]);
        let slots = slots_from_layout(&layout);
        let coeff = Coefficient::product(
            Coefficient::field(rho).unwrap(),
            Coefficient::field(vol_scalar("d_p", si::D_P)).unwrap(),
        )
        .unwrap();

        let expr = coeff_face_expr(&slots, Some(&coeff), "i", "j", 0.5.into(), 1.0.into());
        assert_eq!(
            expr.to_string(),
            "constants.density * (0.5 * state[i * 1u + 0u] + (1.0 - 0.5) * state[j * 1u + 0u])"
        );
    }

    #[test]
    fn coeff_expr_rejects_vector_coefficients() {
        let u = crate::solver::ir::vol_vector("U", si::VELOCITY);
        let layout = crate::solver::ir::StateLayout::new(vec![vol_scalar("p", si::PRESSURE)]);
        let slots = slots_from_layout(&layout);
        let err = Coefficient::product(
            Coefficient::field(vol_scalar("p", si::PRESSURE)).unwrap(),
            Coefficient::Field(u),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            crate::solver::ir::CodegenError::NonScalarCoefficient { .. }
        ));

        let coeff = Coefficient::field(vol_scalar("p", si::PRESSURE)).unwrap();
        let expr = coeff_cell_expr(&slots, Some(&coeff), "idx", 1.0.into());
        assert_eq!(expr.to_string(), "state[idx * 1u + 0u]");
    }
}
