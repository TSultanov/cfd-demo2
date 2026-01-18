use crate::solver::ir::{
    FaceScalarExpr as S, FaceSide, FaceVec2Expr as V, FluxLayout, FluxModuleKernelSpec,
    FluxReconstructionSpec, LimiterSpec,
};
use crate::solver::model::backend::ast::EquationSystem;
use crate::solver::model::flux_module::FluxSchemeSpec;

pub fn lower_flux_scheme(
    scheme: &FluxSchemeSpec,
    system: &EquationSystem,
    reconstruction: FluxReconstructionSpec,
) -> Result<FluxModuleKernelSpec, String> {
    match *scheme {
        FluxSchemeSpec::EulerCentralUpwind => euler_central_upwind(system, reconstruction),
    }
}

fn euler_central_upwind(
    system: &EquationSystem,
    reconstruction: FluxReconstructionSpec,
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

    let limited_linear = |side: FaceSide,
                          phi_cell: S,
                          phi_other: S,
                          grad_field: &'static str,
                          limiter: LimiterSpec|
     -> S {
        let grad = V::state_vec2(side, grad_field);
        let r = V::cell_to_face(side);
        let delta = S::Dot(Box::new(grad), Box::new(r));

        match limiter {
            LimiterSpec::None => S::Add(Box::new(phi_cell), Box::new(delta)),

            // Clamp the projected gradient so the reconstructed face value stays within
            // the local two-point bounds.
            LimiterSpec::MinMod => {
                let diff = S::Sub(Box::new(phi_other), Box::new(phi_cell.clone()));
                let min_diff = S::Min(Box::new(diff.clone()), Box::new(S::lit(0.0)));
                let max_diff = S::Max(Box::new(diff), Box::new(S::lit(0.0)));

                let delta_limited = S::Min(
                    Box::new(S::Max(Box::new(delta), Box::new(min_diff))),
                    Box::new(max_diff),
                );
                S::Add(Box::new(phi_cell), Box::new(delta_limited))
            }

            // Smoothly scale the projected gradient based on the neighbor difference.
            // This is not a full VanLeer TVD limiter, but it ensures the knob is honored
            // and behaves distinctly from MinMod.
            LimiterSpec::VanLeer => {
                let diff = S::Sub(Box::new(phi_other), Box::new(phi_cell.clone()));
                let abs_diff = S::Abs(Box::new(diff));
                let abs_delta = S::Abs(Box::new(delta.clone()));
                let denom = S::Max(Box::new(abs_diff.clone()), Box::new(S::Add(Box::new(abs_delta), Box::new(S::lit(1e-8)))));
                let scale = S::Div(Box::new(abs_diff), Box::new(denom));
                let delta_limited = S::Mul(Box::new(delta), Box::new(scale));
                S::Add(Box::new(phi_cell), Box::new(delta_limited))
            }
        }
    };

    let rho_raw = |side: FaceSide| S::state(side, rho_name);
    let rho_e_raw = |side: FaceSide| S::state(side, rho_e_name);
    let rho_u_raw = |side: FaceSide| V::state_vec2(side, rho_u_name);
    let rho_u_x_raw = |side: FaceSide| S::Dot(Box::new(rho_u_raw(side)), Box::new(ex.clone()));
    let rho_u_y_raw = |side: FaceSide| S::Dot(Box::new(rho_u_raw(side)), Box::new(ey.clone()));
    let p_raw = |side: FaceSide| S::state(side, "p");

    let rho = |side: FaceSide| match reconstruction {
        FluxReconstructionSpec::FirstOrder => rho_raw(side),
        FluxReconstructionSpec::Muscl { limiter } => limited_linear(
            side,
            rho_raw(side),
            rho_raw(other_side(side)),
            "grad_rho",
            limiter,
        ),
    };

    let rho_e = |side: FaceSide| match reconstruction {
        FluxReconstructionSpec::FirstOrder => rho_e_raw(side),
        FluxReconstructionSpec::Muscl { limiter } => limited_linear(
            side,
            rho_e_raw(side),
            rho_e_raw(other_side(side)),
            "grad_rho_e",
            limiter,
        ),
    };

    let rho_u_x = |side: FaceSide| match reconstruction {
        FluxReconstructionSpec::FirstOrder => rho_u_x_raw(side),
        FluxReconstructionSpec::Muscl { limiter } => limited_linear(
            side,
            rho_u_x_raw(side),
            rho_u_x_raw(other_side(side)),
            "grad_rho_u_x",
            limiter,
        ),
    };

    let rho_u_y = |side: FaceSide| match reconstruction {
        FluxReconstructionSpec::FirstOrder => rho_u_y_raw(side),
        FluxReconstructionSpec::Muscl { limiter } => limited_linear(
            side,
            rho_u_y_raw(side),
            rho_u_y_raw(other_side(side)),
            "grad_rho_u_y",
            limiter,
        ),
    };

    let rho_u = |side: FaceSide| V::vec2(rho_u_x(side), rho_u_y(side));

    let p = |side: FaceSide| match reconstruction {
        FluxReconstructionSpec::FirstOrder => p_raw(side),
        FluxReconstructionSpec::Muscl { limiter } => limited_linear(
            side,
            p_raw(side),
            p_raw(other_side(side)),
            "grad_p",
            limiter,
        ),
    };

    let inv_rho = |side: FaceSide| S::Div(Box::new(S::lit(1.0)), Box::new(rho(side)));
    let u_vec = |side: FaceSide| V::MulScalar(Box::new(rho_u(side)), Box::new(inv_rho(side)));
    let u_n = |side: FaceSide| S::Dot(Box::new(u_vec(side)), Box::new(V::normal()));

    let c = |side: FaceSide| {
        // Generalized wave speed:
        //   c^2 = gamma * p / rho + dp_drho
        // where dp_drho is nonzero for barotropic closures (e.g. linear compressibility).
        S::Sqrt(Box::new(S::Add(
            Box::new(S::Div(
                Box::new(S::Mul(
                    Box::new(S::constant("eos_gamma")),
                    Box::new(p(side)),
                )),
                Box::new(rho(side)),
            )),
            Box::new(S::constant("eos_dp_drho")),
        )))
    };

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
                Box::new(c(FaceSide::Owner)),
            )),
            Box::new(S::Add(
                Box::new(u_n(FaceSide::Neighbor)),
                Box::new(c(FaceSide::Neighbor)),
            )),
        )),
    );
    let a_minus = S::Min(
        Box::new(S::lit(0.0)),
        Box::new(S::Min(
            Box::new(S::Sub(
                Box::new(u_n(FaceSide::Owner)),
                Box::new(c(FaceSide::Owner)),
            )),
            Box::new(S::Sub(
                Box::new(u_n(FaceSide::Neighbor)),
                Box::new(c(FaceSide::Neighbor)),
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
