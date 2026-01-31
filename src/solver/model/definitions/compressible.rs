use crate::solver::gpu::enums::GpuBoundaryType;
use crate::solver::model::backend::ast::{
    surface_scalar, surface_vector, vol_scalar, vol_vector, EquationSystem, FieldRef, FluxRef,
};
use crate::solver::model::backend::state_layout::StateLayout;
use crate::solver::model::backend::typed_ast::{
    typed_fvc, typed_fvm, Scalar, TypedCoeff, TypedFieldRef, TypedFluxRef, Vector2,
};
use crate::solver::units::si;
use cfd2_ir::solver::dimensions::{
    Density, Dimensionless, DynamicViscosity, EnergyDensity, Force, InvTime, Length, MassFlux,
    MomentumDensity, MulDim, Power, Pressure, Temperature, Velocity,
};

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

fn build_compressible_system(_fields: &CompressibleFields) -> EquationSystem {
    // NOTE: This model uses typed builder APIs with explicit cast_to() calls to align
    // terms to canonical dimension types. Type-level dimension expressions are not normalized,
    // so semantically equivalent dimensions are different types; cast_to() unifies them.

    // Define typed field references for conservative variables
    let rho_typed = TypedFieldRef::<Density, Scalar>::new("rho");
    let rho_u_typed = TypedFieldRef::<MomentumDensity, Vector2>::new("rho_u");
    let rho_e_typed = TypedFieldRef::<EnergyDensity, Scalar>::new("rho_e");

    // Define typed field references for primitive variables
    let u_typed = TypedFieldRef::<Velocity, Vector2>::new("u");
    let p_typed = TypedFieldRef::<Pressure, Scalar>::new("p");
    let t_typed = TypedFieldRef::<Temperature, Scalar>::new("T");
    let mu_typed = TypedFieldRef::<DynamicViscosity, Scalar>::new("mu");

    // Define typed flux references
    let phi_rho_typed = TypedFluxRef::<MassFlux, Scalar>::new("phi_rho");
    let phi_rho_u_typed = TypedFluxRef::<Force, Vector2>::new("phi_rho_u");
    let phi_rho_e_typed = TypedFluxRef::<Power, Scalar>::new("phi_rho_e");

    // Build coefficients
    let mu_coeff = TypedCoeff::from_field(mu_typed);

    // ========================================
    // Continuity equation: ddt(rho) + div(phi_rho, rho) = 0
    // ========================================
    let rho_ddt = typed_fvm::ddt(rho_typed);
    let rho_div = typed_fvm::div_flux(phi_rho_typed, rho_typed);

    let rho_eqn = (rho_ddt.cast_to::<MassFlux>() + rho_div.cast_to::<MassFlux>()).eqn(rho_typed);

    // ========================================
    // Momentum equation: ddt(rho_u) + div(phi_rho_u, rho_u) - laplacian(mu, u) = 0
    // ========================================
    let rho_u_ddt = typed_fvm::ddt(rho_u_typed);
    let rho_u_div = typed_fvm::div_flux(phi_rho_u_typed, rho_u_typed);
    let viscous_term = typed_fvm::laplacian(mu_coeff, u_typed);

    let rho_u_eqn = (rho_u_ddt.cast_to::<Force>()
        + rho_u_div.cast_to::<Force>()
        + viscous_term.cast_to::<Force>())
    .eqn(rho_u_typed);

    // ========================================
    // Energy equation: ddt(rho_e) + div(phi_rho_e, rho_e) - laplacian(kappa, T) = 0
    // ========================================
    // Thermal conductivity field coefficient: kappa has unit Power/(Length*Temperature)
    let kappa_typed = TypedCoeff::from_field(
        TypedFieldRef::<cfd2_ir::solver::dimensions::DivDim<Power, MulDim<Length, Temperature>>, Scalar>::new("kappa")
    );

    let rho_e_ddt = typed_fvm::ddt(rho_e_typed);
    let rho_e_div = typed_fvm::div_flux(phi_rho_e_typed, rho_e_typed);
    let heat_flux = typed_fvm::laplacian(kappa_typed, t_typed);

    let rho_e_eqn = (rho_e_ddt.cast_to::<Power>()
        + rho_e_div.cast_to::<Power>()
        + heat_flux.cast_to::<Power>())
    .eqn(rho_e_typed);

    // ========================================
    // Primitive velocity recovery: u = rho_u / rho
    // Implemented as: (rho/dt) * u = (1/dt) * rho_u
    // ========================================
    // Field-based coefficients (preserving original semantics)
    let inv_dt_typed = TypedFieldRef::<InvTime, Scalar>::new("inv_dt");
    let inv_dt_coeff = TypedCoeff::from_field(inv_dt_typed);
    let rho_coeff = TypedCoeff::from_field(rho_typed);
    let minus_one_coeff: TypedCoeff<Dimensionless> = TypedCoeff::constant(-1.0);

    // Coefficients for u recovery
    let rho_over_dt = rho_coeff.clone().mul(inv_dt_coeff.clone());
    let minus_rho_over_dt = minus_one_coeff.clone().mul(rho_over_dt);

    let u_source_1 = typed_fvm::source_coeff(minus_rho_over_dt, u_typed);
    let u_source_2 = typed_fvm::source_coeff(inv_dt_coeff.clone(), rho_u_typed);

    let u_eqn = (u_source_1.cast_to::<Force>() + u_source_2.cast_to::<Force>()).eqn(u_typed);

    // ========================================
    // Primitive pressure recovery (algebraic constraint)
    // ========================================
    // EOS field-based coefficients (preserving original semantics)
    let gm1_typed = TypedCoeff::from_field(TypedFieldRef::<Dimensionless, Scalar>::new("eos_gm1"));
    let dp_drho_typed = TypedCoeff::from_field(
        TypedFieldRef::<cfd2_ir::solver::dimensions::DivDim<Pressure, Density>, Scalar>::new("eos_dp_drho")
    );
    let p_offset_typed = TypedCoeff::from_field(TypedFieldRef::<Pressure, Scalar>::new("eos_p_offset"));
    let half_coeff: TypedCoeff<Dimensionless> = TypedCoeff::constant(0.5);

    // minus_gm1/dt
    let minus_gm1 = minus_one_coeff.clone().mul(gm1_typed.clone());
    let minus_gm1_over_dt = minus_gm1.mul(inv_dt_coeff.clone());

    // 0.5 * gm1 / dt * |u|^2
    let u2 = TypedCoeff::mag_sqr(u_typed);
    let half_gm1_over_dt = half_coeff.mul(gm1_typed).mul(inv_dt_coeff.clone());
    let rho_coeff_term = half_gm1_over_dt.mul(u2);

    // minus_dp_drho/dt
    let minus_dp_drho = minus_one_coeff.clone().mul(dp_drho_typed);
    let minus_dp_drho_over_dt = minus_dp_drho.mul(inv_dt_coeff.clone());

    // minus_p_offset/dt
    let minus_p_offset = minus_one_coeff.clone().mul(p_offset_typed);
    let minus_p_offset_over_dt = minus_p_offset.mul(inv_dt_coeff.clone());

    let p_source_1 = typed_fvm::source_coeff(inv_dt_coeff.clone(), p_typed);
    let p_source_2 = typed_fvm::source_coeff(minus_gm1_over_dt, rho_e_typed);
    let p_source_3 = typed_fvm::source_coeff(rho_coeff_term, rho_typed);
    let p_source_4 = typed_fvm::source_coeff(minus_dp_drho_over_dt, rho_typed);
    let p_source_5 = typed_fvc::source_coeff(minus_p_offset_over_dt, p_typed);

    let p_eqn = (p_source_1.cast_to::<Power>()
        + p_source_2.cast_to::<Power>()
        + p_source_3.cast_to::<Power>()
        + p_source_4.cast_to::<Power>()
        + p_source_5.cast_to::<Power>())
    .eqn(p_typed);

    // ========================================
    // Temperature recovery: T = p / (rho * R)
    // Implemented as: (rho*R/dt) * T = (1/dt) * p
    // ========================================
    // EOS gas constant field coefficient (preserving original semantics)
    let r_typed = TypedCoeff::from_field(
        TypedFieldRef::<cfd2_ir::solver::dimensions::DivDim<Pressure, MulDim<Density, Temperature>>, Scalar>::new("eos_r")
    );

    let rho_r_over_dt = rho_coeff.mul(r_typed).mul(inv_dt_coeff.clone());
    let minus_inv_dt = minus_one_coeff.mul(inv_dt_coeff);

    let t_source_1 = typed_fvm::source_coeff(rho_r_over_dt, t_typed);
    let t_source_2 = typed_fvm::source_coeff(minus_inv_dt, p_typed);

    let t_eqn = (t_source_1.cast_to::<Power>() + t_source_2.cast_to::<Power>()).eqn(t_typed);

    // ========================================
    // Assemble equation system
    // ========================================
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
