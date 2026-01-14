use crate::solver::ir::{
    FaceScalarExpr as S, FaceSide, FaceVec2Expr as V, FluxLayout, FluxModuleKernelSpec,
};
use crate::solver::model::flux_module::FluxSchemeSpec;
use crate::solver::model::backend::ast::EquationSystem;

pub fn lower_flux_scheme(
    scheme: &FluxSchemeSpec,
    system: &EquationSystem,
) -> Result<FluxModuleKernelSpec, String> {
    match *scheme {
        FluxSchemeSpec::EulerIdealGasCentralUpwind { gamma } => {
            euler_ideal_gas_central_upwind(system, gamma)
        }
    }
}

fn euler_ideal_gas_central_upwind(
    system: &EquationSystem,
    gamma: f32,
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

    let rho = |side: FaceSide| S::state(side, rho_name);
    let rho_e = |side: FaceSide| S::state(side, rho_e_name);
    let rho_u = |side: FaceSide| V::state_vec2(side, rho_u_name);
    let rho_u_x = |side: FaceSide| S::Dot(Box::new(rho_u(side)), Box::new(ex.clone()));
    let rho_u_y = |side: FaceSide| S::Dot(Box::new(rho_u(side)), Box::new(ey.clone()));

    let inv_rho = |side: FaceSide| S::Div(Box::new(S::lit(1.0)), Box::new(rho(side)));
    let u_vec = |side: FaceSide| V::MulScalar(Box::new(rho_u(side)), Box::new(inv_rho(side)));
    let u_n = |side: FaceSide| S::Dot(Box::new(u_vec(side)), Box::new(V::normal()));

    let p = |side: FaceSide| S::state(side, "p");
    let c = |side: FaceSide| {
        S::Sqrt(Box::new(S::Div(
            Box::new(S::Mul(Box::new(S::lit(gamma)), Box::new(p(side)))),
            Box::new(rho(side)),
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
        components,
        u_left,
        u_right,
        flux_left,
        flux_right,
        a_plus,
        a_minus,
    })
}
