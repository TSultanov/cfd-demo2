use crate::solver::codegen::dsl::{DynExpr, DslType};
use crate::solver::codegen::state_access::{state_scalar_typed, state_vec2_typed, state_vec3_typed};
use crate::solver::codegen::wgsl_ast::Expr;
use crate::solver::ir::ports::{PortFieldKind, ResolvedStateSlotsSpec};
use crate::solver::ir::Coefficient;
use crate::solver::units::si;

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

/// Named coefficient expression with unit tracking (dynamic version).
/// Returns the same WGSL expressions as `coeff_named_expr`, but with correct `UnitDim` metadata.
pub fn coeff_named_expr_dyn(name: &str) -> Option<DynExpr> {
    match name {
        "rho" => Some(DynExpr::new(
            Expr::ident("constants").field("density"),
            DslType::f32(),
            si::DENSITY,
        )),
        "inv_dt" => {
            let dt = Expr::ident("constants").field("dt");
            let dtau = Expr::ident("constants").field("dtau");
            let dt_eff = Expr::call_named("select", vec![dt, dtau.clone(), dtau.clone().gt(0.0)]);
            let expr = Expr::from(1.0) / dt_eff;
            Some(DynExpr::new(expr, DslType::f32(), si::INV_TIME))
        }
        // Dynamic viscosity (SI): Pa·s = kg/(m·s). Historically this was called `nu`,
        // but `nu` is conventionally kinematic viscosity; accept both for now.
        "mu" | "nu" => Some(DynExpr::new(
            Expr::ident("constants").field("viscosity"),
            DslType::f32(),
            si::DYNAMIC_VISCOSITY,
        )),

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
            let expr = mu * cp / pr;
            // kappa = POWER / (LENGTH * TEMPERATURE)
            let kappa_unit = si::POWER / (si::LENGTH * si::TEMPERATURE);
            Some(DynExpr::new(expr, DslType::f32(), kappa_unit))
        }
        "eos_gamma" => Some(DynExpr::new(
            Expr::ident("constants").field("eos_gamma"),
            DslType::f32(),
            si::DIMENSIONLESS,
        )),
        "eos_gm1" => Some(DynExpr::new(
            Expr::ident("constants").field("eos_gm1"),
            DslType::f32(),
            si::DIMENSIONLESS,
        )),
        // eos_r: P = rho * R * T => R = P / (rho * T)
        "eos_r" => {
            let eos_r_unit = si::PRESSURE / (si::DENSITY * si::TEMPERATURE);
            Some(DynExpr::new(
                Expr::ident("constants").field("eos_r"),
                DslType::f32(),
                eos_r_unit,
            ))
        }
        // eos_dp_drho: derivative of pressure w.r.t. density
        "eos_dp_drho" => {
            let eos_dp_drho_unit = si::PRESSURE / si::DENSITY;
            Some(DynExpr::new(
                Expr::ident("constants").field("eos_dp_drho"),
                DslType::f32(),
                eos_dp_drho_unit,
            ))
        }
        "eos_p_offset" => Some(DynExpr::new(
            Expr::ident("constants").field("eos_p_offset"),
            DslType::f32(),
            si::PRESSURE,
        )),
        "eos_theta_ref" => Some(DynExpr::new(
            Expr::ident("constants").field("eos_theta_ref"),
            DslType::f32(),
            si::TEMPERATURE,
        )),
        _ => None,
    }
}

/// Legacy named coefficient expression (without unit tracking).
/// Delegates to `coeff_named_expr_dyn` and extracts the expression.
fn coeff_named_expr(name: &str) -> Option<Expr> {
    coeff_named_expr_dyn(name).map(|d| d.expr)
}

/// Find a slot by name in the resolved state slots spec.
fn find_slot<'a>(slots: &'a ResolvedStateSlotsSpec, field: &str) -> Option<&'a crate::solver::ir::ports::ResolvedStateSlotSpec> {
    slots.slots.iter().find(|s| s.name == field)
}

/// Coefficient expression with unit tracking (dynamic version).
/// Mirrors `coeff_expr` but uses `DynExpr` operations for unit checking.
fn coeff_expr_dyn(slots: &ResolvedStateSlotsSpec, coeff: &Coefficient, sample: CoeffSample<'_>) -> DynExpr {
    match coeff {
        Coefficient::Constant { value, unit } => {
            DynExpr::f32(*value as f32, *unit)
        }
        Coefficient::Field(field) => {
            if let Some(slot) = find_slot(slots, field.name()) {
                if slot.kind != PortFieldKind::Scalar {
                    panic!("coefficient '{}' must be scalar", field.name());
                }
                match sample {
                    CoeffSample::Cell { idx } => {
                        state_scalar_typed(slots, "state", idx, field.name())
                    }
                    CoeffSample::Face {
                        owner_idx,
                        neighbor_idx,
                        interp,
                    } => {
                        let own = state_scalar_typed(slots, "state", owner_idx, field.name());
                        let neigh = state_scalar_typed(slots, "state", neighbor_idx, field.name());
                        // interp is dimensionless scalar
                        let interp_dyn = DynExpr::new(interp, DslType::f32(), si::DIMENSIONLESS);
                        let one = DynExpr::f32(1.0, si::DIMENSIONLESS);
                        let one_minus_interp = one - interp_dyn.clone();
                        interp_dyn * own + one_minus_interp * neigh
                    }
                }
            } else {
                coeff_named_expr_dyn(field.name()).unwrap_or_else(|| {
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
                    let v = state_scalar_typed(slots, "state", idx, field.name());
                    // v * v (square the unit)
                    v.clone() * v
                }
                PortFieldKind::Vector2 => {
                    let v = state_vec2_typed(slots, "state", idx, field.name());
                    // dot product: v · v (produces squared unit)
                    v.clone().dot(&v).expect("dot product for vec2")
                }
                PortFieldKind::Vector3 => {
                    let v = state_vec3_typed(slots, "state", idx, field.name());
                    // dot product: v · v (produces squared unit)
                    v.clone().dot(&v).expect("dot product for vec3")
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
                    let interp_dyn = DynExpr::new(interp, DslType::f32(), si::DIMENSIONLESS);
                    let one = DynExpr::f32(1.0, si::DIMENSIONLESS);
                    let one_minus_interp = one - interp_dyn.clone();
                    interp_dyn * own + one_minus_interp * neigh
                }
            }
        }
        Coefficient::Product(lhs, rhs) => {
            let lhs_expr = coeff_expr_dyn(slots, lhs, sample.clone());
            let rhs_expr = coeff_expr_dyn(slots, rhs, sample);
            lhs_expr * rhs_expr
        }
    }
}

/// Legacy coefficient expression (without unit tracking).
/// Delegates to `coeff_expr_dyn` and extracts the expression.
fn coeff_expr(slots: &ResolvedStateSlotsSpec, coeff: &Coefficient, sample: CoeffSample<'_>) -> Expr {
    coeff_expr_dyn(slots, coeff, sample).expr
}

/// Cell coefficient expression with unit tracking (dynamic version).
pub fn coeff_cell_expr_dyn(
    slots: &ResolvedStateSlotsSpec,
    coeff: Option<&Coefficient>,
    idx: &str,
    fallback: DynExpr,
) -> DynExpr {
    match coeff {
        None => fallback,
        Some(value) => coeff_expr_dyn(slots, value, CoeffSample::Cell { idx }),
    }
}

/// Face coefficient expression with unit tracking (dynamic version).
pub fn coeff_face_expr_dyn(
    slots: &ResolvedStateSlotsSpec,
    coeff: Option<&Coefficient>,
    owner_idx: &str,
    neighbor_idx: &str,
    interp: Expr,
    fallback: DynExpr,
) -> DynExpr {
    match coeff {
        None => fallback,
        Some(value) => coeff_expr_dyn(
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

/// Legacy cell coefficient expression.
/// Delegates to `coeff_cell_expr_dyn` and extracts the expression.
pub fn coeff_cell_expr(
    slots: &ResolvedStateSlotsSpec,
    coeff: Option<&Coefficient>,
    idx: &str,
    fallback: Expr,
) -> Expr {
    let fallback_dyn = DynExpr::new(fallback, DslType::f32(), si::DIMENSIONLESS);
    coeff_cell_expr_dyn(slots, coeff, idx, fallback_dyn).expr
}

/// Legacy face coefficient expression.
/// Delegates to `coeff_face_expr_dyn` and extracts the expression.
pub fn coeff_face_expr(
    slots: &ResolvedStateSlotsSpec,
    coeff: Option<&Coefficient>,
    owner_idx: &str,
    neighbor_idx: &str,
    interp: Expr,
    fallback: Expr,
) -> Expr {
    let fallback_dyn = DynExpr::new(fallback, DslType::f32(), si::DIMENSIONLESS);
    coeff_face_expr_dyn(slots, coeff, owner_idx, neighbor_idx, interp, fallback_dyn).expr
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

    // ============================================================================
    // Unit tracking tests for the dyn APIs
    // ============================================================================

    #[test]
    fn coeff_named_expr_dyn_inv_dt_has_inv_time_unit() {
        let inv_dt = coeff_named_expr_dyn("inv_dt").expect("inv_dt should exist");
        assert_eq!(inv_dt.unit, si::INV_TIME);
        assert_eq!(inv_dt.ty, DslType::f32());
    }

    #[test]
    fn coeff_named_expr_dyn_kappa_has_correct_unit() {
        let kappa = coeff_named_expr_dyn("kappa").expect("kappa should exist");
        let expected_kappa_unit = si::POWER / (si::LENGTH * si::TEMPERATURE);
        assert_eq!(kappa.unit, expected_kappa_unit);
        assert_eq!(kappa.ty, DslType::f32());
    }

    #[test]
    fn coeff_named_expr_dyn_eos_r_has_correct_unit() {
        let eos_r = coeff_named_expr_dyn("eos_r").expect("eos_r should exist");
        let expected_eos_r_unit = si::PRESSURE / (si::DENSITY * si::TEMPERATURE);
        assert_eq!(eos_r.unit, expected_eos_r_unit);
        assert_eq!(eos_r.ty, DslType::f32());
    }

    #[test]
    fn coeff_named_expr_dyn_eos_dp_drho_has_correct_unit() {
        let eos_dp_drho = coeff_named_expr_dyn("eos_dp_drho").expect("eos_dp_drho should exist");
        let expected_unit = si::PRESSURE / si::DENSITY;
        assert_eq!(eos_dp_drho.unit, expected_unit);
        assert_eq!(eos_dp_drho.ty, DslType::f32());
    }

    #[test]
    fn coeff_named_expr_dyn_rho_has_density_unit() {
        let rho = coeff_named_expr_dyn("rho").expect("rho should exist");
        assert_eq!(rho.unit, si::DENSITY);
        assert_eq!(rho.ty, DslType::f32());
    }

    #[test]
    fn coeff_named_expr_dyn_mu_has_dynamic_viscosity_unit() {
        let mu = coeff_named_expr_dyn("mu").expect("mu should exist");
        assert_eq!(mu.unit, si::DYNAMIC_VISCOSITY);
        assert_eq!(mu.ty, DslType::f32());
    }

    #[test]
    fn coeff_named_expr_dyn_nu_has_dynamic_viscosity_unit() {
        let nu = coeff_named_expr_dyn("nu").expect("nu should exist");
        assert_eq!(nu.unit, si::DYNAMIC_VISCOSITY);
        assert_eq!(nu.ty, DslType::f32());
    }

    #[test]
    fn coeff_expr_dyn_product_combines_units() {
        let rho = vol_scalar("rho", si::DENSITY);
        let d_p = vol_scalar("d_p", si::D_P);
        let layout = crate::solver::ir::StateLayout::new(vec![rho.clone(), d_p.clone()]);
        let slots = slots_from_layout(&layout);
        
        // Create a product: rho * d_p
        let coeff = Coefficient::product(
            Coefficient::field(rho).unwrap(),
            Coefficient::field(d_p).unwrap(),
        ).unwrap();

        let dyn_expr = coeff_expr_dyn(&slots, &coeff, CoeffSample::Cell { idx: "i" });
        
        // Expected unit: DENSITY * D_P
        let expected_unit = si::DENSITY * si::D_P;
        assert_eq!(dyn_expr.unit, expected_unit);
        assert_eq!(dyn_expr.ty, DslType::f32());
    }

    #[test]
    fn coeff_expr_dyn_constant_preserves_unit() {
        let coeff = Coefficient::Constant { value: 3.14, unit: si::PRESSURE };
        let slots = ResolvedStateSlotsSpec {
            stride: 0,
            slots: vec![],
        };
        
        let dyn_expr = coeff_expr_dyn(&slots, &coeff, CoeffSample::Cell { idx: "i" });
        
        assert_eq!(dyn_expr.unit, si::PRESSURE);
        assert_eq!(dyn_expr.ty, DslType::f32());
        assert_eq!(dyn_expr.expr.to_string(), "3.14");
    }

    #[test]
    fn coeff_expr_dyn_mag_sqr_scalar_squares_unit() {
        // Scalar field: MagSqr should square the unit
        let p = vol_scalar("p", si::PRESSURE);
        let layout = crate::solver::ir::StateLayout::new(vec![p.clone()]);
        let slots = slots_from_layout(&layout);
        
        let coeff = Coefficient::MagSqr(p);
        let dyn_expr = coeff_expr_dyn(&slots, &coeff, CoeffSample::Cell { idx: "i" });
        
        // Expected unit: PRESSURE^2
        let expected_unit = si::PRESSURE * si::PRESSURE;
        assert_eq!(dyn_expr.unit, expected_unit);
        assert_eq!(dyn_expr.ty, DslType::f32());
    }

    #[test]
    fn coeff_expr_dyn_mag_sqr_vector2_squares_unit() {
        // Vector2 field: MagSqr (dot product) should square the unit
        let u = crate::solver::ir::vol_vector("U", si::VELOCITY);
        let layout = crate::solver::ir::StateLayout::new(vec![u.clone()]);
        let slots = slots_from_layout(&layout);
        
        let coeff = Coefficient::MagSqr(u);
        let dyn_expr = coeff_expr_dyn(&slots, &coeff, CoeffSample::Cell { idx: "i" });
        
        // Expected unit: VELOCITY^2
        let expected_unit = si::VELOCITY * si::VELOCITY;
        assert_eq!(dyn_expr.unit, expected_unit);
        assert_eq!(dyn_expr.ty, DslType::f32());
    }

    #[test]
    fn coeff_expr_dyn_mag_sqr_vector3_squares_unit() {
        // Vector3 field: MagSqr (dot product) should square the unit
        let u = crate::solver::ir::vol_vector3("U", si::VELOCITY);
        let layout = crate::solver::ir::StateLayout::new(vec![u.clone()]);
        let slots = slots_from_layout(&layout);
        
        let coeff = Coefficient::MagSqr(u);
        let dyn_expr = coeff_expr_dyn(&slots, &coeff, CoeffSample::Cell { idx: "i" });
        
        // Expected unit: VELOCITY^2
        let expected_unit = si::VELOCITY * si::VELOCITY;
        assert_eq!(dyn_expr.unit, expected_unit);
        assert_eq!(dyn_expr.ty, DslType::f32());
    }

    #[test]
    fn coeff_cell_expr_dyn_returns_fallback_when_coeff_none() {
        let layout = crate::solver::ir::StateLayout::new(vec![]);
        let slots = slots_from_layout(&layout);
        let fallback = DynExpr::f32(42.0, si::PRESSURE);
        
        let result = coeff_cell_expr_dyn(&slots, None, "i", fallback.clone());
        
        assert_eq!(result.unit, fallback.unit);
        assert_eq!(result.expr.to_string(), "42.0");
    }

    #[test]
    fn coeff_face_expr_dyn_interpolates_with_correct_units() {
        let rho = vol_scalar("rho", si::DENSITY);
        let layout = crate::solver::ir::StateLayout::new(vec![rho.clone()]);
        let slots = slots_from_layout(&layout);
        
        let coeff = Coefficient::field(rho).unwrap();
        let interp = Expr::from(0.75);
        let fallback = DynExpr::f32(0.0, si::DIMENSIONLESS);
        
        let dyn_expr = coeff_face_expr_dyn(&slots, Some(&coeff), "owner", "neigh", interp, fallback);
        
        // Result should have DENSITY unit
        assert_eq!(dyn_expr.unit, si::DENSITY);
        assert_eq!(dyn_expr.ty, DslType::f32());
        // WGSL should contain interpolation
        let wgsl = dyn_expr.expr.to_string();
        assert!(wgsl.contains("0.75"));
        assert!(wgsl.contains("owner"));
        assert!(wgsl.contains("neigh"));
    }
}
