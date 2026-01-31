use crate::solver::gpu::enums::GpuBoundaryType;
use crate::solver::model::backend::ast::{
    fvc, fvm, surface_scalar, surface_vector, vol_scalar, vol_vector, Coefficient, EquationSystem,
    FieldRef, FluxRef,
};
use crate::solver::model::backend::state_layout::StateLayout;
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
            t: vol_scalar("T", si::TEMPERATURE),
            u: vol_vector("u", si::VELOCITY),
            mu: vol_scalar("mu", si::DYNAMIC_VISCOSITY),
            phi_rho: surface_scalar("phi_rho", si::MASS_FLUX),
            phi_rho_u: surface_vector("phi_rho_u", si::FORCE),
            phi_rho_e: surface_scalar("phi_rho_e", si::POWER),
        }
    }
}

fn build_compressible_system(fields: &CompressibleFields) -> EquationSystem {
    // NOTE: The compressible model uses complex algebraic constraints with derived fields
    // that don't have straightforward dimension mappings in the typed builder.
    // For now, we use the untyped builder with runtime validation.
    // TODO: Migrate to typed builder once dimension algebra for derived coefficients is clearer.

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

    // Total energy (rho_e) equation.
    let kappa = vol_scalar("kappa", si::POWER / (si::LENGTH * si::TEMPERATURE));
    let rho_e_eqn = (fvm::ddt(fields.rho_e)
        + fvm::div_flux(fields.phi_rho_e, fields.rho_e)
        + fvm::laplacian(
            Coefficient::field(kappa).expect("kappa must be scalar"),
            fields.t,
        ))
    .eqn(fields.rho_e);

    // Primitive velocity recovery as an algebraic constraint:
    let u_eqn = {
        let rho = Coefficient::field(fields.rho).expect("rho must be scalar");
        let inv_dt =
            Coefficient::field(vol_scalar("inv_dt", si::INV_TIME)).expect("inv_dt must be scalar");

        let rho_over_dt =
            Coefficient::product(rho, inv_dt.clone()).expect("rho/dt coefficient must be scalar");
        let minus_rho_over_dt = Coefficient::product(Coefficient::constant(-1.0), rho_over_dt)
            .expect("rho/dt coefficient must be scalar");

        (fvm::source_coeff(minus_rho_over_dt, fields.u) + fvm::source_coeff(inv_dt, fields.rho_u))
            .eqn(fields.u)
    };

    // Primitive pressure recovery as an algebraic constraint:
    let p_eqn = {
        let inv_dt =
            Coefficient::field(vol_scalar("inv_dt", si::INV_TIME)).expect("inv_dt must be scalar");
        let gm1 = Coefficient::field(vol_scalar("eos_gm1", si::DIMENSIONLESS))
            .expect("eos_gm1 must be scalar");
        let dp_drho = Coefficient::field(vol_scalar("eos_dp_drho", si::PRESSURE / si::DENSITY))
            .expect("eos_dp_drho must be scalar");
        let p_offset = Coefficient::field(vol_scalar("eos_p_offset", si::PRESSURE))
            .expect("eos_p_offset must be scalar");
        let u2 = Coefficient::mag_sqr(fields.u);

        let minus_gm1_over_dt = Coefficient::product(
            Coefficient::product(Coefficient::constant(-1.0), gm1.clone())
                .expect("gm1 coefficient must be scalar"),
            inv_dt.clone(),
        )
        .expect("(gamma-1)/dt coefficient must be scalar");
        let rho_coeff = Coefficient::product(
            Coefficient::product(
                Coefficient::product(Coefficient::constant(0.5), gm1)
                    .expect("0.5*(gamma-1) coefficient must be scalar"),
                inv_dt.clone(),
            )
            .expect("0.5*(gamma-1)/dt coefficient must be scalar"),
            u2,
        )
        .expect("rho*u^2/dt coefficient must be scalar");

        let minus_dp_drho_over_dt = Coefficient::product(
            Coefficient::product(Coefficient::constant(-1.0), dp_drho)
                .expect("-dp/drho coefficient must be scalar"),
            inv_dt.clone(),
        )
        .expect("dp/drho/dt coefficient must be scalar");

        let minus_p_offset_over_dt = Coefficient::product(
            Coefficient::product(Coefficient::constant(-1.0), p_offset)
                .expect("-p_offset coefficient must be scalar"),
            inv_dt.clone(),
        )
        .expect("p_offset/dt coefficient must be scalar");

        (fvm::source_coeff(inv_dt, fields.p)
            + fvm::source_coeff(minus_gm1_over_dt, fields.rho_e)
            + fvm::source_coeff(rho_coeff, fields.rho)
            + fvm::source_coeff(minus_dp_drho_over_dt, fields.rho)
            + fvc::source_coeff(minus_p_offset_over_dt, fields.p))
        .eqn(fields.p)
    };

    // Temperature recovery as an algebraic constraint (ideal gas):
    let t_eqn = {
        let rho = Coefficient::field(fields.rho).expect("rho must be scalar");
        let r = Coefficient::field(vol_scalar(
            "eos_r",
            si::PRESSURE / (si::DENSITY * si::TEMPERATURE),
        ))
        .expect("eos_r must be scalar");
        let inv_dt =
            Coefficient::field(vol_scalar("inv_dt", si::INV_TIME)).expect("inv_dt must be scalar");
        let rho_r_over_dt = Coefficient::product(
            Coefficient::product(rho, r).expect("rho*R coefficient must be scalar"),
            inv_dt.clone(),
        )
        .expect("rho*R/dt coefficient must be scalar");
        let minus_inv_dt = Coefficient::product(Coefficient::constant(-1.0), inv_dt)
            .expect("inv_dt must be scalar");

        (fvm::source_coeff(rho_r_over_dt, fields.t) + fvm::source_coeff(minus_inv_dt, fields.p))
            .eqn(fields.t)
    };

    let mut system = EquationSystem::new();
    system.add_equation(rho_eqn);
    system.add_equation(rho_u_eqn);
    system.add_equation(rho_e_eqn);
    system.add_equation(u_eqn);
    system.add_equation(p_eqn);
    system.add_equation(t_eqn);

    // Validate units to ensure the system is consistent
    system
        .validate_units()
        .expect("compressible system failed unit validation");

    system
}

pub fn compressible_system() -> EquationSystem {
    let fields = CompressibleFields::new();
    build_compressible_system(&fields)
}

pub fn compressible_model() -> ModelSpec {
    compressible_model_with_eos(crate::solver::model::eos::EosSpec::IdealGas {
        gamma: 1.4,
        // Default to nondimensional theta_ref=1 (p=rho*R*T with rho=1 -> p=1).
        gas_constant: 1.0,
        temperature: 1.0,
    })
}

pub fn compressible_model_with_eos(eos: crate::solver::model::eos::EosSpec) -> ModelSpec {
    let fields = CompressibleFields::new();
    let system = build_compressible_system(&fields);
    // Flux module reconstruction uses gradient fields in the state layout when enabled.
    // These are computed by the optional `flux_module_gradients` stage (Gauss gradients).
    let grad_rho = vol_vector("grad_rho", si::DENSITY / si::LENGTH);
    let grad_rho_u_x = vol_vector("grad_rho_u_x", si::MOMENTUM_DENSITY / si::LENGTH);
    let grad_rho_u_y = vol_vector("grad_rho_u_y", si::MOMENTUM_DENSITY / si::LENGTH);
    let grad_rho_e = vol_vector("grad_rho_e", si::ENERGY_DENSITY / si::LENGTH);
    let grad_t = vol_vector("grad_T", si::TEMPERATURE / si::LENGTH);
    let grad_u_x = vol_vector("grad_u_x", si::VELOCITY / si::LENGTH);
    let grad_u_y = vol_vector("grad_u_y", si::VELOCITY / si::LENGTH);
    let layout = StateLayout::new(vec![
        fields.rho,
        fields.rho_u,
        grad_rho_u_x,
        grad_rho_u_y,
        fields.rho_e,
        fields.p,
        fields.t,
        fields.u,
        grad_rho,
        grad_rho_e,
        grad_t,
        grad_u_x,
        grad_u_y,
    ]);

    let mut boundaries = BoundarySpec::default();
    boundaries.set_field(
        "rho",
        FieldBoundarySpec::new()
            // Inlet density is Dirichlet (placeholder value); update via the solver's boundary
            // table API (and keep `rho_u`/`rho_e` consistent with the chosen inlet state).
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
            )
            .set_uniform(
                GpuBoundaryType::MovingWall,
                1,
                BoundaryCondition::zero_gradient(si::DENSITY / si::LENGTH),
            ),
    );
    boundaries.set_field(
        "rho_u",
        FieldBoundarySpec::new()
            // Inlet momentum density is Dirichlet (placeholder value); update via the solver's
            // boundary table API (typically derived from `rho` and inlet `u`).
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
            )
            .set_uniform(
                GpuBoundaryType::MovingWall,
                2,
                BoundaryCondition::dirichlet(0.0, si::MOMENTUM_DENSITY),
            ),
    );
    boundaries.set_field(
        "u",
        FieldBoundarySpec::new()
            // Inlet velocity is Dirichlet (placeholder value); update via the solver's boundary
            // table API.
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
            )
            .set_uniform(
                GpuBoundaryType::MovingWall,
                2,
                BoundaryCondition::dirichlet(0.0, si::VELOCITY),
            ),
    );
    boundaries.set_field(
        "rho_e",
        FieldBoundarySpec::new()
            // Inlet energy is Dirichlet (placeholder value); update via the solver's boundary
            // table API, typically derived from inlet `rho`, `u`, and EOS params.
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
            )
            .set_uniform(
                GpuBoundaryType::MovingWall,
                1,
                BoundaryCondition::zero_gradient(si::ENERGY_DENSITY / si::LENGTH),
            ),
    );
    boundaries.set_field(
        "p",
        FieldBoundarySpec::new()
            // Inlet pressure is Dirichlet (placeholder value); update via the solver's boundary
            // table API to match the chosen inlet thermodynamic state.
            .set_uniform(
                GpuBoundaryType::Inlet,
                1,
                BoundaryCondition::dirichlet(0.0, si::PRESSURE),
            )
            .set_uniform(
                GpuBoundaryType::Outlet,
                1,
                BoundaryCondition::zero_gradient(si::PRESSURE / si::LENGTH),
            )
            .set_uniform(
                GpuBoundaryType::Wall,
                1,
                BoundaryCondition::zero_gradient(si::PRESSURE / si::LENGTH),
            )
            .set_uniform(
                GpuBoundaryType::SlipWall,
                1,
                BoundaryCondition::zero_gradient(si::PRESSURE / si::LENGTH),
            )
            .set_uniform(
                GpuBoundaryType::MovingWall,
                1,
                BoundaryCondition::zero_gradient(si::PRESSURE / si::LENGTH),
            ),
    );
    boundaries.set_field(
        "T",
        FieldBoundarySpec::new()
            // Isothermal inlet temperature (placeholder value); update via boundary table API.
            .set_uniform(
                GpuBoundaryType::Inlet,
                1,
                BoundaryCondition::dirichlet(0.0, si::TEMPERATURE),
            )
            .set_uniform(
                GpuBoundaryType::Outlet,
                1,
                BoundaryCondition::zero_gradient(si::TEMPERATURE / si::LENGTH),
            )
            .set_uniform(
                GpuBoundaryType::Wall,
                1,
                BoundaryCondition::zero_gradient(si::TEMPERATURE / si::LENGTH),
            )
            .set_uniform(
                GpuBoundaryType::SlipWall,
                1,
                BoundaryCondition::zero_gradient(si::TEMPERATURE / si::LENGTH),
            )
            .set_uniform(
                GpuBoundaryType::MovingWall,
                1,
                BoundaryCondition::zero_gradient(si::TEMPERATURE / si::LENGTH),
            ),
    );
    let method = crate::solver::model::method::MethodSpec::Coupled(
        crate::solver::model::method::CoupledCapabilities {
            // Dual-time stepping can require under-relaxation to stabilize pseudo-time iterations
            // at high acoustic CFL. Keep standard implicit stepping (dtau=0) unchanged.
            apply_relaxation_in_update: true,
            relaxation_requires_dtau: true,
            requires_flux_module: true,
            gradient_storage: crate::solver::model::gpu_spec::GradientStorage::PackedState,
        },
    );
    let flux = crate::solver::model::flux_module::FluxModuleSpec::Scheme {
        gradients: Some(
            crate::solver::model::flux_module::FluxModuleGradientsSpec::FromStateLayout,
        ),
        scheme: crate::solver::model::flux_module::FluxSchemeSpec::EulerCentralUpwind,
    };
    let primitives = crate::solver::model::primitives::PrimitiveDerivations::identity();

    // Clone system and layout for flux_module_module since we need to move them into ModelSpec
    let system_for_flux = system.clone();
    let layout_for_flux = layout.clone();
    let flux_module_module = crate::solver::model::modules::flux_module::flux_module_module(
        flux,
        &system_for_flux,
        &layout_for_flux,
        &primitives,
    )
    .expect("failed to build flux_module module");

    let model = ModelSpec {
        id: "compressible",
        // Route compressible through the generic coupled pipeline.
        system,
        state_layout: layout,
        boundaries,

        modules: vec![
            crate::solver::model::modules::eos::eos_module(eos),
            flux_module_module,
            crate::solver::model::modules::generic_coupled::generic_coupled_module(method),
        ],
        // The compressible model couples conserved and primitive fields in a single solve.
        // Use tighter defaults than the generic solver so small-magnitude velocity updates are
        // not lost when residual norms are dominated by large pressure/energy components.
        linear_solver: Some(crate::solver::model::linear_solver::ModelLinearSolverSpec {
            preconditioner: crate::solver::model::linear_solver::ModelPreconditionerSpec::Default,
            solver: crate::solver::model::linear_solver::ModelLinearSolverSettings {
                solver_type: crate::solver::model::linear_solver::ModelLinearSolverType::Fgmres {
                    max_restart: 60,
                },
                max_iters: 200,
                tolerance: 1e-10,
                tolerance_abs: 1e-12,
            },
        }),
        primitives,
    };

    model
}
