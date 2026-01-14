use crate::solver::gpu::enums::GpuBoundaryType;
use crate::solver::model::backend::ast::{
    fvm, surface_scalar, surface_vector, vol_scalar, vol_vector, Coefficient, EquationSystem,
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
    pub t: FieldRef,
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
            t: vol_scalar("T", si::PRESSURE / si::DENSITY),
            u: vol_vector("u", si::VELOCITY),
            mu: vol_scalar("mu", si::DYNAMIC_VISCOSITY),
            phi_rho: surface_scalar("phi_rho", si::MASS_FLUX),
            phi_rho_u: surface_vector("phi_rho_u", si::FORCE),
            phi_rho_e: surface_scalar("phi_rho_e", si::POWER),
        }
    }
}

fn build_compressible_system(fields: &CompressibleFields, gamma: f32) -> EquationSystem {
    let rho_eqn =
        (fvm::ddt(fields.rho) + fvm::div_flux(fields.phi_rho, fields.rho)).eqn(fields.rho);
    let rho_u_eqn = (fvm::ddt(fields.rho_u)
        + fvm::div_flux(fields.phi_rho_u, fields.rho_u)
        // Viscous momentum term (Newtonian stress, simplified):
        //   ∇·τ ≈ μ ∇²u  (i.e., -∇·(μ∇u) moved to the LHS as `laplacian(mu, u)`).
        + fvm::laplacian(
            Coefficient::field(fields.mu).expect("mu must be scalar"),
            fields.u,
        ))
    .eqn(fields.rho_u);
    let rho_e_eqn =
        (fvm::ddt(fields.rho_e) + fvm::div_flux(fields.phi_rho_e, fields.rho_e)).eqn(fields.rho_e);

    // Primitive velocity recovery as an algebraic constraint:
    //   rho_u = rho * u
    //
    // This makes `u` part of the coupled unknown set so viscous terms can be treated implicitly.
    let u_eqn = {
        let rho = Coefficient::field(fields.rho).expect("rho must be scalar");
        // Scale the algebraic constraint by `1/dt` to improve conditioning in implicit time
        // schemes (BDF2 in particular).
        let inv_dt = Coefficient::field(vol_scalar("inv_dt", si::INV_TIME))
            .expect("inv_dt must be scalar");

        let rho_over_dt =
            Coefficient::product(rho, inv_dt.clone()).expect("rho/dt coefficient must be scalar");
        let minus_rho_over_dt = Coefficient::product(Coefficient::constant(-1.0), rho_over_dt)
            .expect("rho/dt coefficient must be scalar");

        (fvm::source_coeff(minus_rho_over_dt, fields.u)
            + fvm::source_coeff(inv_dt, fields.rho_u))
        .eqn(fields.u)
    };

    // Primitive pressure recovery as an algebraic constraint (ideal gas EOS with total energy):
    //
    //   p = (gamma - 1) * (rho_e - 0.5 * rho * |u|^2)
    //
    // We treat |u|^2 as a coefficient evaluated from the current state (outer-iteration
    // linearization) and scale the constraint by `1/dt` for conditioning.
    let p_eqn = {
        let gm1 = gamma - 1.0;
        let inv_dt = Coefficient::field(vol_scalar("inv_dt", si::INV_TIME))
            .expect("inv_dt must be scalar");
        let u2 = Coefficient::mag_sqr(fields.u);

        let minus_gm1_over_dt = Coefficient::product(Coefficient::constant(-(gm1 as f64)), inv_dt.clone())
            .expect("(gamma-1)/dt coefficient must be scalar");
        let rho_coeff = Coefficient::product(
            Coefficient::product(Coefficient::constant(0.5 * gm1 as f64), inv_dt.clone())
                .expect("0.5*(gamma-1)/dt coefficient must be scalar"),
            u2,
        )
        .expect("rho*u^2/dt coefficient must be scalar");

        (fvm::source_coeff(inv_dt, fields.p)
            + fvm::source_coeff(minus_gm1_over_dt, fields.rho_e)
            + fvm::source_coeff(rho_coeff, fields.rho))
        .eqn(fields.p)
    };

    // Temperature recovery as an algebraic constraint (ideal gas, R=1):
    //
    //   p = rho * T  <=>  rho*T - p = 0
    //
    // Scale by `1/dt` for conditioning.
    let t_eqn = {
        let rho = Coefficient::field(fields.rho).expect("rho must be scalar");
        let inv_dt = Coefficient::field(vol_scalar("inv_dt", si::INV_TIME))
            .expect("inv_dt must be scalar");
        let rho_over_dt =
            Coefficient::product(rho, inv_dt.clone()).expect("rho/dt coefficient must be scalar");
        let minus_inv_dt =
            Coefficient::product(Coefficient::constant(-1.0), inv_dt).expect("inv_dt must be scalar");

        (fvm::source_coeff(rho_over_dt, fields.t) + fvm::source_coeff(minus_inv_dt, fields.p))
            .eqn(fields.t)
    };

    let mut system = EquationSystem::new();
    system.add_equation(rho_eqn);
    system.add_equation(rho_u_eqn);
    system.add_equation(rho_e_eqn);
    system.add_equation(u_eqn);
    system.add_equation(p_eqn);
    system.add_equation(t_eqn);
    system
}

pub fn compressible_system() -> EquationSystem {
    let fields = CompressibleFields::new();
    let gamma = 1.4;
    build_compressible_system(&fields, gamma)
}

pub fn compressible_model() -> ModelSpec {
    let fields = CompressibleFields::new();
    let gamma = 1.4;
    let system = build_compressible_system(&fields, gamma);
    let layout = StateLayout::new(vec![
        fields.rho,
        fields.rho_u,
        fields.rho_e,
        fields.p,
        fields.t,
        fields.u,
    ]);

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
        "u",
        FieldBoundarySpec::new()
            // Inlet velocity is driven via the runtime `InletVelocity` plan param
            // (see `param_inlet_velocity`); initial value is a placeholder.
            .set_uniform(
                GpuBoundaryType::Inlet,
                2,
                BoundaryCondition::dirichlet(0.0, si::VELOCITY),
            )
            .set_uniform(
                GpuBoundaryType::Outlet,
                2,
                BoundaryCondition::zero_gradient(si::INV_TIME),
            )
            // Walls: no-slip (matches `rho_u` Dirichlet=0).
            .set_uniform(
                GpuBoundaryType::Wall,
                2,
                BoundaryCondition::dirichlet(0.0, si::VELOCITY),
            )
            .set_uniform(
                GpuBoundaryType::SlipWall,
                2,
                BoundaryCondition::zero_gradient(si::INV_TIME),
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
        // Route compressible through the generic coupled pipeline.
        method: crate::solver::model::method::MethodSpec::GenericCoupledImplicit { outer_iters: 1 },
        eos: crate::solver::model::eos::EosSpec::IdealGas { gamma },
        system,
        state_layout: layout,
        boundaries,

        extra_kernels: Vec::new(),
        linear_solver: None,
        flux_module: Some(crate::solver::model::flux_module::FluxModuleSpec::Scheme {
            gradients: None,
            scheme: crate::solver::model::flux_module::FluxSchemeSpec::EulerIdealGasCentralUpwind {
                gamma,
            },
        }),
        primitives: crate::solver::model::primitives::PrimitiveDerivations::identity(),
        generated_kernels: Vec::new(),
        gpu: ModelGpuSpec::default(),
    }
    .with_derived_gpu()
}
