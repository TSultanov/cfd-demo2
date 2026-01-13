use crate::solver::gpu::enums::GpuBoundaryType;
use crate::solver::model::backend::ast::{
    fvm, surface_scalar, surface_vector, vol_scalar, vol_vector, EquationSystem,
    FieldRef, FluxRef,
};
use crate::solver::model::backend::state_layout::StateLayout;
use crate::solver::model::gpu_spec::ModelGpuSpec;
use crate::solver::units::si;

use super::{BoundaryCondition, BoundarySpec, FieldBoundarySpec, ModelSpec};

#[derive(Debug, Clone)]
pub struct CompressibleFields {
    pub rho: FieldRef,
    pub rho_u: FieldRef,
    pub rho_e: FieldRef,
    pub p: FieldRef,
    pub u: FieldRef,
    pub mu: FieldRef,
    pub phi_rho: FluxRef,
    pub phi_rho_u: FluxRef,
    pub phi_rho_e: FluxRef,
}

impl CompressibleFields {
    pub fn new() -> Self {
        Self {
            rho: vol_scalar("rho", si::DENSITY),
            rho_u: vol_vector("rho_u", si::MOMENTUM_DENSITY),
            rho_e: vol_scalar("rho_e", si::ENERGY_DENSITY),
            p: vol_scalar("p", si::PRESSURE),
            u: vol_vector("u", si::VELOCITY),
            mu: vol_scalar("mu", si::DYNAMIC_VISCOSITY),
            phi_rho: surface_scalar("phi_rho", si::MASS_FLUX),
            phi_rho_u: surface_vector("phi_rho_u", si::FORCE),
            phi_rho_e: surface_scalar("phi_rho_e", si::POWER),
        }
    }
}

fn build_compressible_system(fields: &CompressibleFields) -> EquationSystem {
    let rho_eqn =
        (fvm::ddt(fields.rho) + fvm::div_flux(fields.phi_rho, fields.rho)).eqn(fields.rho);
    let rho_u_eqn =
        (fvm::ddt(fields.rho_u) + fvm::div_flux(fields.phi_rho_u, fields.rho_u)).eqn(fields.rho_u);
    let rho_e_eqn =
        (fvm::ddt(fields.rho_e) + fvm::div_flux(fields.phi_rho_e, fields.rho_e)).eqn(fields.rho_e);

    let mut system = EquationSystem::new();
    system.add_equation(rho_eqn);
    system.add_equation(rho_u_eqn);
    system.add_equation(rho_e_eqn);
    system
}

pub fn compressible_system() -> EquationSystem {
    let fields = CompressibleFields::new();
    build_compressible_system(&fields)
}

pub fn compressible_model() -> ModelSpec {
    let fields = CompressibleFields::new();
    let system = build_compressible_system(&fields);
    let layout = StateLayout::new(vec![
        fields.rho,
        fields.rho_u,
        fields.rho_e,
        fields.p,
        fields.u,
    ]);

    let gamma = 1.4;
    let flux_kernel = compressible_euler_central_upwind_flux_module_kernel(&system, &fields, gamma)
        .expect("failed to build compressible flux kernel spec");

    let mut boundaries = BoundarySpec::default();
    boundaries.set_field(
        "rho",
        FieldBoundarySpec::new()
            // Inlet density is driven via the runtime `Density` plan param, which updates the
            // GPU `bc_value` table for `GpuBoundaryType::Inlet` (see `param_density`).
            .set_uniform(
                GpuBoundaryType::Inlet,
                1,
                BoundaryCondition::dirichlet(1.0, si::DENSITY),
            )
            .set_uniform(
                GpuBoundaryType::Outlet,
                1,
                BoundaryCondition::zero_gradient(si::DENSITY / si::LENGTH),
            )
            .set_uniform(
                GpuBoundaryType::Wall,
                1,
                BoundaryCondition::zero_gradient(si::DENSITY / si::LENGTH),
            )
            .set_uniform(
                GpuBoundaryType::SlipWall,
                1,
                BoundaryCondition::zero_gradient(si::DENSITY / si::LENGTH),
            ),
    );
    boundaries.set_field(
        "rho_u",
        FieldBoundarySpec::new()
            // Inlet momentum density is driven via the runtime `InletVelocity` plan param
            // (see `param_inlet_velocity`); initial value is a placeholder.
            .set_uniform(
                GpuBoundaryType::Inlet,
                2,
                BoundaryCondition::dirichlet(0.0, si::MOMENTUM_DENSITY),
            )
            .set_uniform(
                GpuBoundaryType::Outlet,
                2,
                BoundaryCondition::zero_gradient(si::MOMENTUM_DENSITY / si::LENGTH),
            )
            .set_uniform(
                GpuBoundaryType::Wall,
                2,
                BoundaryCondition::dirichlet(0.0, si::MOMENTUM_DENSITY),
            )
            .set_uniform(
                GpuBoundaryType::SlipWall,
                2,
                BoundaryCondition::zero_gradient(si::MOMENTUM_DENSITY / si::LENGTH),
            ),
    );
    boundaries.set_field(
        "rho_e",
        FieldBoundarySpec::new()
            // Inlet energy is updated alongside rho/rho_u when inlet parameters change.
            .set_uniform(
                GpuBoundaryType::Inlet,
                1,
                BoundaryCondition::dirichlet(0.0, si::ENERGY_DENSITY),
            )
            .set_uniform(
                GpuBoundaryType::Outlet,
                1,
                BoundaryCondition::zero_gradient(si::ENERGY_DENSITY / si::LENGTH),
            )
            .set_uniform(
                GpuBoundaryType::Wall,
                1,
                BoundaryCondition::zero_gradient(si::ENERGY_DENSITY / si::LENGTH),
            )
            .set_uniform(
                GpuBoundaryType::SlipWall,
                1,
                BoundaryCondition::zero_gradient(si::ENERGY_DENSITY / si::LENGTH),
            ),
    );

    ModelSpec {
        id: "compressible",
        // Route compressible through the generic coupled pipeline; KT flux + primitive recovery
        // remain transitional kernels behind model-driven IDs.
        method: crate::solver::model::method::MethodSpec::GenericCoupledImplicit { outer_iters: 1 },
        eos: crate::solver::model::eos::EosSpec::IdealGas { gamma },
        system,
        state_layout: layout,
        boundaries,

        extra_kernels: Vec::new(),
        linear_solver: None,
        flux_module: Some(crate::solver::model::flux_module::FluxModuleSpec::Kernel {
            gradients: None,
            kernel: flux_kernel,
        }),
        primitives: crate::solver::model::primitives::PrimitiveDerivations::euler_ideal_gas(gamma),
        generated_kernels: Vec::new(),
        gpu: ModelGpuSpec::default(),
    }
    .with_derived_gpu()
}

fn compressible_euler_central_upwind_flux_module_kernel(
    system: &EquationSystem,
    fields: &CompressibleFields,
    gamma: f32,
) -> Result<crate::solver::ir::FluxModuleKernelSpec, String> {
    use crate::solver::ir::{
        FaceScalarExpr as S, FaceSide, FaceVec2Expr as V, FluxLayout, FluxModuleKernelSpec,
    };

    let flux_layout = FluxLayout::from_system(system);
    let components: Vec<String> = flux_layout
        .components
        .iter()
        .map(|c| c.name.clone())
        .collect();

    let ex = V::vec2(S::lit(1.0), S::lit(0.0));
    let ey = V::vec2(S::lit(0.0), S::lit(1.0));

    let rho_name = fields.rho.name();
    let rho_u_name = fields.rho_u.name();
    let rho_e_name = fields.rho_e.name();

    let rho = |side: FaceSide| S::state(side, rho_name);
    let rho_e = |side: FaceSide| S::state(side, rho_e_name);
    let rho_u = |side: FaceSide| V::state_vec2(side, rho_u_name);
    let rho_u_x = |side: FaceSide| S::Dot(Box::new(rho_u(side)), Box::new(ex.clone()));
    let rho_u_y = |side: FaceSide| S::Dot(Box::new(rho_u(side)), Box::new(ey.clone()));

    let inv_rho = |side: FaceSide| S::Div(Box::new(S::lit(1.0)), Box::new(rho(side)));
    let u_vec = |side: FaceSide| V::MulScalar(Box::new(rho_u(side)), Box::new(inv_rho(side)));
    let u_n = |side: FaceSide| S::Dot(Box::new(u_vec(side)), Box::new(V::normal()));

    let p = |side: FaceSide| S::primitive(side, "p");
    let c = |side: FaceSide| {
        S::Sqrt(Box::new(S::Div(
            Box::new(S::Mul(Box::new(S::lit(gamma)), Box::new(p(side)))),
            Box::new(rho(side)),
        )))
    };

    let n_x = S::Dot(Box::new(V::normal()), Box::new(ex.clone()));
    let n_y = S::Dot(Box::new(V::normal()), Box::new(ey.clone()));

    // Viscous terms (simple Laplacian-style model):
    // - Momentum: add `-mu * (u_N - u_O) / dist` to the face-normal momentum flux.
    // - Energy: add `-(tau·n)·u_avg - k * (T_N - T_O) / dist`,
    //   with `tau·n ≈ mu * (u_N - u_O) / dist`, `T = p / rho` (R=1), and `k = mu * Cp / Pr`.
    //
    // This is intentionally a low-Mach-friendly stabilizer; it is not a full Newtonian stress
    // tensor discretization on skewed meshes.
    let mu = S::constant("viscosity");
    let dist = S::dist();

    let u_o = u_vec(FaceSide::Owner);
    let u_nbr = u_vec(FaceSide::Neighbor);
    let du = V::Sub(Box::new(u_nbr.clone()), Box::new(u_o.clone()));
    let inv_dist = S::Div(Box::new(S::lit(1.0)), Box::new(dist.clone()));

    let du_x = S::Dot(Box::new(du.clone()), Box::new(ex.clone()));
    let du_y = S::Dot(Box::new(du.clone()), Box::new(ey.clone()));
    let grad_n_u_x = S::Mul(Box::new(du_x), Box::new(inv_dist.clone()));
    let grad_n_u_y = S::Mul(Box::new(du_y), Box::new(inv_dist.clone()));
    let visc_mom_x = S::Neg(Box::new(S::Mul(Box::new(mu.clone()), Box::new(grad_n_u_x))));
    let visc_mom_y = S::Neg(Box::new(S::Mul(Box::new(mu.clone()), Box::new(grad_n_u_y))));

    let u_avg = V::MulScalar(
        Box::new(V::Add(Box::new(u_o), Box::new(u_nbr))),
        Box::new(S::lit(0.5)),
    );
    let grad_n_u_vec = V::MulScalar(Box::new(du), Box::new(inv_dist.clone()));
    let tau_dot_n_dot_u = S::Mul(
        Box::new(mu.clone()),
        Box::new(S::Dot(Box::new(grad_n_u_vec), Box::new(u_avg))),
    );
    let visc_work = S::Neg(Box::new(tau_dot_n_dot_u));

    let t = |side: FaceSide| S::Div(Box::new(p(side)), Box::new(rho(side)));
    let d_t = S::Sub(
        Box::new(t(FaceSide::Neighbor)),
        Box::new(t(FaceSide::Owner)),
    );
    let cp_over_pr = gamma / (gamma - 1.0); // Cp for R=1, with Pr=1 (matches OpenFOAM reference cases)
    let kappa = S::Mul(Box::new(mu), Box::new(S::lit(cp_over_pr)));
    let heat_flux = S::Neg(Box::new(S::Mul(
        Box::new(S::Mul(Box::new(kappa), Box::new(d_t))),
        Box::new(inv_dist),
    )));
    let visc_energy = S::Add(Box::new(visc_work), Box::new(heat_flux));

    let flux_mass = |side: FaceSide| S::Mul(Box::new(rho(side)), Box::new(u_n(side)));
    let flux_mom_x = |side: FaceSide| {
        S::Add(
            Box::new(S::Mul(Box::new(rho_u_x(side)), Box::new(u_n(side)))),
            Box::new(S::Add(
                Box::new(S::Mul(Box::new(p(side)), Box::new(n_x.clone()))),
                Box::new(visc_mom_x.clone()),
            )),
        )
    };
    let flux_mom_y = |side: FaceSide| {
        S::Add(
            Box::new(S::Mul(Box::new(rho_u_y(side)), Box::new(u_n(side)))),
            Box::new(S::Add(
                Box::new(S::Mul(Box::new(p(side)), Box::new(n_y.clone()))),
                Box::new(visc_mom_y.clone()),
            )),
        )
    };
    let flux_energy = |side: FaceSide| {
        let inviscid = S::Mul(
            Box::new(S::Add(Box::new(rho_e(side)), Box::new(p(side)))),
            Box::new(u_n(side)),
        );
        S::Add(Box::new(inviscid), Box::new(visc_energy.clone()))
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
            return Err(format!(
                "compressible central-upwind kernel builder does not know how to flux component '{name}'"
            ));
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
