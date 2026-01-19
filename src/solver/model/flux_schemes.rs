use crate::solver::ir::{
    FaceScalarExpr as S, FaceSide, FaceVec2Expr as V, FluxLayout, FluxModuleKernelSpec,
    LimiterSpec,
};
use crate::solver::model::backend::ast::EquationSystem;
use crate::solver::model::flux_module::FluxSchemeSpec;
use crate::solver::scheme::Scheme;

use crate::solver::ir::reconstruction::{
    limited_linear_face_value, quick_face_value, FaceExprBuilder,
};

pub fn lower_flux_scheme(
    flux_scheme: &FluxSchemeSpec,
    system: &EquationSystem,
    reconstruction: Scheme,
) -> Result<FluxModuleKernelSpec, String> {
    match *flux_scheme {
        FluxSchemeSpec::EulerCentralUpwind => euler_central_upwind(system, reconstruction),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_state_named(expr: &S, name: &str) -> bool {
        matches!(
            expr,
            S::State {
                name: n,
                ..
            } if n == name
        )
    }

    fn is_lit(expr: &S, value: f32) -> bool {
        matches!(expr, S::Literal(v) if *v == value)
    }

    fn is_mul_named_states(expr: &S, a: &str, b: &str) -> bool {
        match expr {
            S::Mul(x, y) => {
                (is_state_named(x, a) && is_state_named(y, b))
                    || (is_state_named(x, b) && is_state_named(y, a))
            }
            _ => false,
        }
    }

    fn contains_sign_guard_for_states(expr: &S, a: &str, b: &str) -> bool {
        fn walk(expr: &S, f: &mut impl FnMut(&S)) {
            f(expr);
            match expr {
                S::Add(x, y)
                | S::Sub(x, y)
                | S::Mul(x, y)
                | S::Div(x, y)
                | S::Max(x, y)
                | S::Min(x, y)
                | S::Lerp(x, y) => {
                    walk(x, f);
                    walk(y, f);
                }
                S::Neg(x) | S::Abs(x) | S::Sqrt(x) => walk(x, f),
                S::Dot(_, _) => {}
                S::Literal(_)
                | S::Builtin(_)
                | S::Constant { .. }
                | S::LowMachParam(_)
                | S::State { .. }
                | S::Primitive { .. } => {}
            }
        }

        let mut found = false;
        walk(expr, &mut |node| {
            let S::Div(num, denom) = node else {
                return;
            };

            fn is_max_of(expr: &S, mut a: impl FnMut(&S) -> bool, mut b: impl FnMut(&S) -> bool) -> bool {
                match expr {
                    S::Max(x, y) => (a(x) && b(y)) || (a(y) && b(x)),
                    _ => false,
                }
            }

            let p = |e: &S| is_mul_named_states(e, a, b);
            let abs_p = |e: &S| matches!(e, S::Abs(inner) if p(inner));
            let num_ok = is_max_of(num, p, |e| is_lit(e, 0.0));
            let denom_ok = is_max_of(denom, abs_p, |e| is_lit(e, crate::solver::ir::VANLEER_EPS));

            if num_ok && denom_ok {
                found = true;
            }
        });

        found
    }

    #[test]
    fn contract_flux_module_vanleer_has_opposite_slope_sign_guard() {
        // This is a structural/IR contract (not a brittle WGSL string match):
        // VanLeer MUSCL must include a guard that zeros the correction when diff*delta <= 0.
        let diff = S::state(FaceSide::Owner, "diff");
        let delta = S::state(FaceSide::Owner, "delta");
        let delta_limited =
            crate::solver::ir::reconstruction::vanleer_delta_limited::<FaceExprBuilder>(diff, delta);

        assert!(
            contains_sign_guard_for_states(&delta_limited, "diff", "delta"),
            "expected VanLeer sign-guard structure (max(p,0)/max(abs(p),eps))"
        );
    }
}

fn euler_central_upwind(
    system: &EquationSystem,
    reconstruction: Scheme,
) -> Result<FluxModuleKernelSpec, String> {
    let flux_layout = FluxLayout::from_system(system);
    let components: Vec<String> = flux_layout
        .components
        .iter()
        .map(|c| c.name.clone())
        .collect();

    let ex = V::vec2(S::lit(1.0), S::lit(0.0));
    let ey = V::vec2(S::lit(0.0), S::lit(1.0));

    // Conventional compressible field names (model-side convention).
    let rho_name = "rho";
    let rho_u_name = "rho_u";
    let rho_e_name = "rho_e";

    let other_side = |side: FaceSide| {
        if side == FaceSide::Owner {
            FaceSide::Neighbor
        } else {
            FaceSide::Owner
        }
    };

    let limiter_for_scheme = |scheme: Scheme| -> LimiterSpec {
        match scheme {
            Scheme::SecondOrderUpwindMinMod | Scheme::QUICKMinMod => LimiterSpec::MinMod,
            Scheme::SecondOrderUpwindVanLeer | Scheme::QUICKVanLeer => LimiterSpec::VanLeer,
            _ => LimiterSpec::None,
        }
    };

    let approximate_gradient = |side: FaceSide, phi_cell: S, phi_other: S| -> V {
        // Simple two-point gradient estimate based on neighbor differences:
        //   grad ≈ (phi_other - phi_cell) * d / max(dot(d,d), eps)
        // where d is the vector from the cell center to the opposite cell center.
        let d = V::Sub(
            Box::new(V::cell_to_face(side)),
            Box::new(V::cell_to_face(other_side(side))),
        );
        let diff = S::Sub(Box::new(phi_other), Box::new(phi_cell.clone()));
        let denom = S::Max(
            Box::new(S::Dot(Box::new(d.clone()), Box::new(d.clone()))),
            Box::new(S::lit(1e-12)),
        );
        let scale = S::Div(Box::new(diff), Box::new(denom));
        V::MulScalar(Box::new(d), Box::new(scale))
    };

    let reconstruct_scalar =
        |side: FaceSide, phi_cell: S, phi_other: S| -> S {
            let grad = approximate_gradient(side, phi_cell.clone(), phi_other.clone());

            match reconstruction {
                Scheme::Upwind => phi_cell,

                Scheme::SecondOrderUpwind
                | Scheme::SecondOrderUpwindMinMod
                | Scheme::SecondOrderUpwindVanLeer => limited_linear_face_value::<FaceExprBuilder>(
                    phi_cell,
                    phi_other,
                    grad,
                    V::cell_to_face(side),
                    limiter_for_scheme(reconstruction),
                ),

                Scheme::QUICK | Scheme::QUICKMinMod | Scheme::QUICKVanLeer => {
                    let d = V::Sub(
                        Box::new(V::cell_to_face(side)),
                        Box::new(V::cell_to_face(other_side(side))),
                    );
                    quick_face_value::<FaceExprBuilder>(
                        phi_cell,
                        phi_other,
                        grad,
                        d,
                        limiter_for_scheme(reconstruction),
                    )
                }
            }
        };

    let rho_raw = |side: FaceSide| S::state(side, rho_name);
    let rho_e_raw = |side: FaceSide| S::state(side, rho_e_name);
    let rho_u_raw = |side: FaceSide| V::state_vec2(side, rho_u_name);
    let rho_u_x_raw = |side: FaceSide| S::Dot(Box::new(rho_u_raw(side)), Box::new(ex.clone()));
    let rho_u_y_raw = |side: FaceSide| S::Dot(Box::new(rho_u_raw(side)), Box::new(ey.clone()));
    let p_raw = |side: FaceSide| S::state(side, "p");

    let rho = |side: FaceSide| {
        reconstruct_scalar(side, rho_raw(side), rho_raw(other_side(side)))
    };

    let rho_e = |side: FaceSide| {
        reconstruct_scalar(side, rho_e_raw(side), rho_e_raw(other_side(side)))
    };

    let rho_u_x = |side: FaceSide| {
        reconstruct_scalar(side, rho_u_x_raw(side), rho_u_x_raw(other_side(side)))
    };

    let rho_u_y = |side: FaceSide| {
        reconstruct_scalar(side, rho_u_y_raw(side), rho_u_y_raw(other_side(side)))
    };

    let rho_u = |side: FaceSide| V::vec2(rho_u_x(side), rho_u_y(side));

    let p = |side: FaceSide| {
        reconstruct_scalar(side, p_raw(side), p_raw(other_side(side)))
    };

    let inv_rho = |side: FaceSide| S::Div(Box::new(S::lit(1.0)), Box::new(rho(side)));
    let u_vec = |side: FaceSide| V::MulScalar(Box::new(rho_u(side)), Box::new(inv_rho(side)));
    let u_n = |side: FaceSide| S::Dot(Box::new(u_vec(side)), Box::new(V::normal()));

    let c2 = |side: FaceSide| {
        // Generalized wave speed:
        //   c^2 = gamma * p / rho + dp_drho
        // where dp_drho is nonzero for barotropic closures (e.g. linear compressibility).
        S::Add(
            Box::new(S::Div(
                Box::new(S::Mul(
                    Box::new(S::constant("eos_gamma")),
                    Box::new(p(side)),
                )),
                Box::new(rho(side)),
            )),
            Box::new(S::constant("eos_dp_drho")),
        )
    };

    let u_n2 = |side: FaceSide| {
        let un = u_n(side);
        S::Mul(Box::new(un.clone()), Box::new(un))
    };

    let low_mach_model = S::low_mach_model();
    let low_mach_theta_floor = S::low_mach_theta_floor();

    let weight_for = |value: f32| {
        S::Max(
            Box::new(S::lit(0.0)),
            Box::new(S::Sub(
                Box::new(S::lit(1.0)),
                Box::new(S::Abs(Box::new(S::Sub(
                    Box::new(low_mach_model.clone()),
                    Box::new(S::lit(value)),
                )))),
            )),
        )
    };

    let w_legacy = weight_for(0.0);
    let w_weiss_smith = weight_for(1.0);
    let w_off = weight_for(2.0);

    let c_eff2_legacy = |side: FaceSide| {
        let c2_side = c2(side);
        S::Min(Box::new(u_n2(side)), Box::new(c2_side))
    };

    let c_eff2_weiss_smith = |side: FaceSide| {
        let c2_side = c2(side);
        let floor = S::Mul(Box::new(low_mach_theta_floor.clone()), Box::new(c2_side.clone()));
        S::Min(
            Box::new(S::Max(Box::new(u_n2(side)), Box::new(floor))),
            Box::new(c2_side),
        )
    };

    let c_eff2 = |side: FaceSide| {
        let c2_side = c2(side);
        S::Add(
            Box::new(S::Add(
                Box::new(S::Mul(Box::new(w_off.clone()), Box::new(c2_side))),
                Box::new(S::Mul(
                    Box::new(w_legacy.clone()),
                    Box::new(c_eff2_legacy(side)),
                )),
            )),
            Box::new(S::Mul(
                Box::new(w_weiss_smith.clone()),
                Box::new(c_eff2_weiss_smith(side)),
            )),
        )
    };

    let c_eff = |side: FaceSide| S::Sqrt(Box::new(c_eff2(side)));

    let n_x = S::Dot(Box::new(V::normal()), Box::new(ex.clone()));
    let n_y = S::Dot(Box::new(V::normal()), Box::new(ey.clone()));

    let flux_mass = |side: FaceSide| S::Mul(Box::new(rho(side)), Box::new(u_n(side)));
    let flux_mom_x = |side: FaceSide| {
        S::Add(
            Box::new(S::Mul(Box::new(rho_u_x(side)), Box::new(u_n(side)))),
            Box::new(S::Mul(Box::new(p(side)), Box::new(n_x.clone()))),
        )
    };
    let flux_mom_y = |side: FaceSide| {
        S::Add(
            Box::new(S::Mul(Box::new(rho_u_y(side)), Box::new(u_n(side)))),
            Box::new(S::Mul(Box::new(p(side)), Box::new(n_y.clone()))),
        )
    };
    let flux_energy = |side: FaceSide| {
        S::Mul(
            Box::new(S::Add(Box::new(rho_e(side)), Box::new(p(side)))),
            Box::new(u_n(side)),
        )
    };

    let mut u_left = Vec::new();
    let mut u_right = Vec::new();
    let mut flux_left = Vec::new();
    let mut flux_right = Vec::new();

    for name in &components {
        if name == rho_name {
            u_left.push(rho(FaceSide::Owner));
            u_right.push(rho(FaceSide::Neighbor));
            flux_left.push(flux_mass(FaceSide::Owner));
            flux_right.push(flux_mass(FaceSide::Neighbor));
        } else if name == &format!("{rho_u_name}_x") {
            u_left.push(rho_u_x(FaceSide::Owner));
            u_right.push(rho_u_x(FaceSide::Neighbor));
            flux_left.push(flux_mom_x(FaceSide::Owner));
            flux_right.push(flux_mom_x(FaceSide::Neighbor));
        } else if name == &format!("{rho_u_name}_y") {
            u_left.push(rho_u_y(FaceSide::Owner));
            u_right.push(rho_u_y(FaceSide::Neighbor));
            flux_left.push(flux_mom_y(FaceSide::Owner));
            flux_right.push(flux_mom_y(FaceSide::Neighbor));
        } else if name == rho_e_name {
            u_left.push(rho_e(FaceSide::Owner));
            u_right.push(rho_e(FaceSide::Neighbor));
            flux_left.push(flux_energy(FaceSide::Owner));
            flux_right.push(flux_energy(FaceSide::Neighbor));
        } else {
            // Auxiliary coupled unknowns (e.g. primitive fields coupled via diffusion/constraints)
            // get zero face flux by default.
            u_left.push(S::lit(0.0));
            u_right.push(S::lit(0.0));
            flux_left.push(S::lit(0.0));
            flux_right.push(S::lit(0.0));
        }
    }

    let a_plus = S::Max(
        Box::new(S::lit(0.0)),
        Box::new(S::Max(
            Box::new(S::Add(
                Box::new(u_n(FaceSide::Owner)),
                Box::new(c_eff(FaceSide::Owner)),
            )),
            Box::new(S::Add(
                Box::new(u_n(FaceSide::Neighbor)),
                Box::new(c_eff(FaceSide::Neighbor)),
            )),
        )),
    );
    let a_minus = S::Min(
        Box::new(S::lit(0.0)),
        Box::new(S::Min(
            Box::new(S::Sub(
                Box::new(u_n(FaceSide::Owner)),
                Box::new(c_eff(FaceSide::Owner)),
            )),
            Box::new(S::Sub(
                Box::new(u_n(FaceSide::Neighbor)),
                Box::new(c_eff(FaceSide::Neighbor)),
            )),
        )),
    );

    Ok(FluxModuleKernelSpec::CentralUpwind {
        reconstruction,
        components,
        u_left,
        u_right,
        flux_left,
        flux_right,
        a_plus,
        a_minus,
    })
}
