// STABILITY ENVELOPE (model contract, measured June 2026 — probe matrix in
// tests/mms_compressible_order_test.rs::probe_inviscid_margin_matrix):
// this discretization (explicit KT/vanLeer flux + implicit inv_dt-scaled
// EOS-recovery rows) develops a slow secular instability in the inviscid
// limit at moderate Mach: a smooth interior thermo-field mode whose growth
// rate rises with mesh resolution and is damped ONLY by physical viscosity
// (mu k^2 must beat it; mu = 5e-3 holds through n = 32 at the MMS box's
// scales, mu = 0.05 is robust everywhere measured). Time scheme, outer
// iterations, pseudo-time damping, and low-Mach preconditioning were all
// probed and refuted as cures — preconditioning makes it WORSE at moderate
// Mach (it removes acoustic-scale dissipation). Time-accurate compressible
// marching therefore REQUIRES nonzero physical viscosity; do not run this
// model inviscid.

use crate::solver::gpu::enums::GpuBoundaryType;
use crate::solver::model::backend::algebraic::{
    add_algebraic_equation, typed_alg, TypedAlgExpr, TypedParamRef,
};
use crate::solver::model::backend::ast::{
    surface_scalar_dim, surface_vector_dim, vol_scalar_dim, vol_vector_dim, EquationSystem,
    FieldRef, FluxRef,
};
use crate::solver::model::backend::typed_ast::{
    typed_fvc, typed_fvm, Scalar, TypedCoeff, TypedFieldRef, TypedFluxRef, Vector2,
};
use crate::solver::model::ports::PortRegistry;
// si module no longer needed for boundary conditions - using type-level dimensions
use cfd2_ir::dimensions::{
    Density, Dimensionless, DivDim, DynamicViscosity, EnergyDensity, Force, InvTime, Length,
    MassFlux, MomentumDensity, MulDim, Power, Pressure, Temperature, Velocity, Volume,
};
// Type-level dimensions for boundary conditions (re-exported for convenience)
type DensityGradient = DivDim<Density, Length>;
type MomentumDensityGradient = DivDim<MomentumDensity, Length>;
type EnergyDensityGradient = DivDim<EnergyDensity, Length>;
type PressureGradient = DivDim<Pressure, Length>;
type TemperatureGradient = DivDim<Temperature, Length>;

use super::{BoundaryCondition, BoundarySpec, FieldBoundarySpec, ModelSpec};

/// Manufactured continuity source field on the `compressible_mms` variant (scalar).
pub const COMPRESSIBLE_MMS_SOURCE_RHO_FIELD: &str = "mms_src_rho";
/// Manufactured momentum source field on the `compressible_mms` variant (Vector2).
pub const COMPRESSIBLE_MMS_SOURCE_RHO_U_FIELD: &str = "mms_src_rho_u";
/// Manufactured energy source field on the `compressible_mms` variant (scalar).
pub const COMPRESSIBLE_MMS_SOURCE_RHO_E_FIELD: &str = "mms_src_rho_e";

// Per-volume source units of each conservation equation (equation unit / Volume).
type RhoSourceUnit = DivDim<MassFlux, Volume>;
type RhoUSourceUnit = DivDim<Force, Volume>;
type RhoESourceUnit = DivDim<Power, Volume>;

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
            rho: vol_scalar_dim::<Density>("rho"),
            rho_u: vol_vector_dim::<MomentumDensity>("rho_u"),
            rho_e: vol_scalar_dim::<EnergyDensity>("rho_e"),
            p: vol_scalar_dim::<Pressure>("p"),
            t: vol_scalar_dim::<Temperature>("T"),
            u: vol_vector_dim::<Velocity>("u"),
            mu: vol_scalar_dim::<DynamicViscosity>("mu"),
            phi_rho: surface_scalar_dim::<MassFlux>("phi_rho"),
            phi_rho_u: surface_vector_dim::<Force>("phi_rho_u"),
            phi_rho_e: surface_scalar_dim::<Power>("phi_rho_e"),
        }
    }
}

impl Default for CompressibleFields {
    fn default() -> Self {
        Self::new()
    }
}

// EOS uniform parameters. The names and units must match the port manifest in
// modules/eos_ports.rs (asserted by `eos_params_match_uniform_port_manifest`);
// host-side values come from `EosSpec`.
const EOS_GAMMA: TypedParamRef<Dimensionless> = TypedParamRef::new("eos_gamma");
const EOS_GM1: TypedParamRef<Dimensionless> = TypedParamRef::new("eos_gm1");
const EOS_R: TypedParamRef<DivDim<Pressure, MulDim<Density, Temperature>>> =
    TypedParamRef::new("eos_r");
const EOS_DP_DRHO: TypedParamRef<DivDim<Pressure, Density>> = TypedParamRef::new("eos_dp_drho");
const EOS_P_OFFSET: TypedParamRef<Pressure> = TypedParamRef::new("eos_p_offset");

/// Declared squared sound speed of the linearized EOS: `c^2 = gamma * R * T`.
///
/// The central-upwind flux derivation lowers this to the acoustic speed used
/// in the Kurganov wave bounds (`sqrt(gamma * R * T)` over cell temperatures).
pub fn compressible_wave_speed_sq() -> TypedAlgExpr<MulDim<Velocity, Velocity>, Scalar> {
    let t_typed = TypedFieldRef::<Temperature, Scalar>::new("T");
    (typed_alg::param(EOS_GAMMA) * typed_alg::param(EOS_R) * typed_alg::field(t_typed))
        .cast_to::<MulDim<Velocity, Velocity>>()
}

/// Declared generalized squared wave speed: `c^2 = gamma * p / rho + dp_drho`.
///
/// The `dp_drho` term covers barotropic closures (linear compressibility);
/// it is zero for an ideal gas. The flux derivation evaluates this over
/// reconstructed face states for low-Mach dissipation scaling.
pub fn compressible_generalized_wave_speed_sq(
) -> TypedAlgExpr<DivDim<Pressure, Density>, Scalar> {
    let p_typed = TypedFieldRef::<Pressure, Scalar>::new("p");
    let rho_typed = TypedFieldRef::<Density, Scalar>::new("rho");
    ((typed_alg::param(EOS_GAMMA) * typed_alg::field(p_typed)) / typed_alg::field(rho_typed))
        .cast_to::<DivDim<Pressure, Density>>()
        + typed_alg::param(EOS_DP_DRHO)
}

/// Central-upwind flux declaration for this model: which fields play which
/// conserved/primitive role, plus the EOS relations as math. The face
/// pressure relation `rho * R * T` is the same algebra as the temperature
/// recovery row (`rho * R * T = p`), declared here in the direction the flux
/// needs it.
pub fn compressible_central_upwind_decl() -> crate::solver::model::flux_schemes::CentralUpwindDecl
{
    let rho_typed = TypedFieldRef::<Density, Scalar>::new("rho");
    let t_typed = TypedFieldRef::<Temperature, Scalar>::new("T");
    crate::solver::model::flux_schemes::CentralUpwindDecl {
        density: "rho",
        momentum: "rho_u",
        energy: "rho_e",
        temperature: "T",
        velocity: "u",
        pressure_field: "p",
        pressure: (typed_alg::field(rho_typed) * typed_alg::param(EOS_R)
            * typed_alg::field(t_typed))
        .cast_to::<Pressure>()
        .to_untyped(),
        wave_speed_sq: compressible_wave_speed_sq().to_untyped(),
        generalized_wave_speed_sq: compressible_generalized_wave_speed_sq().to_untyped(),
    }
}

fn build_compressible_system(fields: &CompressibleFields) -> EquationSystem {
    build_compressible_system_impl(fields, false)
}

fn build_compressible_system_impl(
    _fields: &CompressibleFields,
    with_mms_sources: bool,
) -> EquationSystem {
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

    let mut rho_sum = rho_ddt.cast_to::<MassFlux>() + rho_div.cast_to::<MassFlux>();
    if with_mms_sources {
        let mms_rho = TypedCoeff::from_field(TypedFieldRef::<RhoSourceUnit, Scalar>::new(
            COMPRESSIBLE_MMS_SOURCE_RHO_FIELD,
        ));
        rho_sum = rho_sum + typed_fvc::source_coeff(mms_rho, rho_typed).cast_to::<MassFlux>();
    }
    let rho_eqn = rho_sum.eqn(rho_typed);

    // ========================================
    // Momentum equation: ddt(rho_u) + div(phi_rho_u, rho_u) - laplacian(mu, u) = 0
    // ========================================
    let rho_u_ddt = typed_fvm::ddt(rho_u_typed);
    let rho_u_div = typed_fvm::div_flux(phi_rho_u_typed, rho_u_typed);
    let viscous_term = typed_fvm::laplacian(mu_coeff, u_typed);

    let mut rho_u_sum = rho_u_ddt.cast_to::<Force>()
        + rho_u_div.cast_to::<Force>()
        + viscous_term.cast_to::<Force>();
    if with_mms_sources {
        let mms_rho_u = TypedFieldRef::<RhoUSourceUnit, Vector2>::new(
            COMPRESSIBLE_MMS_SOURCE_RHO_U_FIELD,
        );
        rho_u_sum =
            rho_u_sum + typed_fvc::source_vector(mms_rho_u, rho_u_typed).cast_to::<Force>();
    }
    let rho_u_eqn = rho_u_sum.eqn(rho_u_typed);

    // ========================================
    // Energy equation: ddt(rho_e) + div(phi_rho_e, rho_e) - laplacian(kappa, T) = 0
    // ========================================
    // Thermal conductivity field coefficient: kappa has unit Power/(Length*Temperature)
    let kappa_typed = TypedCoeff::from_field(TypedFieldRef::<
        cfd2_ir::dimensions::DivDim<Power, MulDim<Length, Temperature>>,
        Scalar,
    >::new("kappa"));

    let rho_e_ddt = typed_fvm::ddt(rho_e_typed);
    let rho_e_div = typed_fvm::div_flux(phi_rho_e_typed, rho_e_typed);
    let heat_flux = typed_fvm::laplacian(kappa_typed, t_typed);

    let mut rho_e_sum = rho_e_ddt.cast_to::<Power>()
        + rho_e_div.cast_to::<Power>()
        + heat_flux.cast_to::<Power>();
    if with_mms_sources {
        let mms_rho_e = TypedCoeff::from_field(TypedFieldRef::<RhoESourceUnit, Scalar>::new(
            COMPRESSIBLE_MMS_SOURCE_RHO_E_FIELD,
        ));
        rho_e_sum =
            rho_e_sum + typed_fvc::source_coeff(mms_rho_e, rho_e_typed).cast_to::<Power>();
    }
    let rho_e_eqn = rho_e_sum.eqn(rho_e_typed);

    // ========================================
    // Primitive recovery, declared as algebraic relations.
    //
    // These lower mechanically to the inv_dt-scaled coupled source rows
    // (see cfd2_ir::equation::algebraic). Fields multiplying the linear
    // unknown of each product (e.g. rho in `rho * u`) are frozen at the
    // current state (Picard linearization).
    // ========================================

    // Velocity recovery: rho * u = rho_u.
    let u_recovery = typed_alg::equation(
        u_typed,
        typed_alg::field(rho_u_typed),
        (typed_alg::field(rho_typed) * typed_alg::field(u_typed)).cast_to::<MomentumDensity>(),
    );

    // Linearized EOS: p = (gamma-1)*rho_e - (gamma-1)/2*|u|^2*rho
    //                     + dp_drho*rho + p_offset.
    // Ideal gas sets dp_drho = p_offset = 0; linear compressibility sets
    // gamma-1 = 0 (the uniform params absorb the EOS variant).
    let pressure_eos = typed_alg::equation(
        p_typed,
        typed_alg::field(p_typed),
        (typed_alg::param(EOS_GM1) * typed_alg::field(rho_e_typed)).cast_to::<Pressure>()
            - (typed_alg::constant(0.5)
                * typed_alg::param(EOS_GM1)
                * typed_alg::mag_sqr(u_typed)
                * typed_alg::field(rho_typed))
            .cast_to::<Pressure>()
            + (typed_alg::param(EOS_DP_DRHO) * typed_alg::field(rho_typed)).cast_to::<Pressure>()
            + typed_alg::param(EOS_P_OFFSET),
    );

    // Temperature recovery: rho * R * T = p.
    let temperature_recovery = typed_alg::equation(
        t_typed,
        (typed_alg::field(rho_typed) * typed_alg::param(EOS_R) * typed_alg::field(t_typed))
            .cast_to::<Pressure>(),
        typed_alg::field(p_typed),
    );

    // ========================================
    // Assemble equation system
    // ========================================
    let mut system = EquationSystem::new();
    system.add_equation(rho_eqn);
    system.add_equation(rho_u_eqn);
    system.add_equation(rho_e_eqn);
    add_algebraic_equation(&mut system, &u_recovery)
        .expect("compressible velocity recovery failed algebraic lowering");
    add_algebraic_equation(&mut system, &pressure_eos)
        .expect("compressible EOS pressure equation failed algebraic lowering");
    add_algebraic_equation(&mut system, &temperature_recovery)
        .expect("compressible temperature recovery failed algebraic lowering");

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

pub fn compressible_model() -> Result<ModelSpec, String> {
    compressible_model_with_eos(crate::solver::model::eos::EosSpec::IdealGas {
        gamma: 1.4,
        // Default to nondimensional theta_ref=1 (p=rho*R*T with rho=1 -> p=1).
        gas_constant: 1.0,
        temperature: 1.0,
    })
}

pub fn compressible_model_with_eos(eos: crate::solver::model::eos::EosSpec) -> Result<ModelSpec, String> {
    compressible_model_impl(eos, false)
}

/// `compressible` plus manufactured source fields on the three conservation
/// equations, for MMS convergence studies (the EOS/recovery rows are exact
/// algebraic constraints and need no sources). Uses the nondimensional
/// default EOS (gamma = 1.4, R = 1, theta_ref = 1).
pub fn compressible_mms_model() -> Result<ModelSpec, String> {
    compressible_model_impl(
        crate::solver::model::eos::EosSpec::IdealGas {
            gamma: 1.4,
            gas_constant: 1.0,
            temperature: 1.0,
        },
        true,
    )
}

fn compressible_model_impl(
    eos: crate::solver::model::eos::EosSpec,
    with_mms_sources: bool,
) -> Result<ModelSpec, String> {
    let fields = CompressibleFields::new();
    let system = build_compressible_system_impl(&fields, with_mms_sources);
    // Flux module reconstruction uses gradient fields in the state layout when enabled.
    // These are computed by the optional `flux_module_gradients` stage (Gauss gradients).
    let grad_rho = vol_vector_dim::<DivDim<Density, Length>>("grad_rho");
    let grad_rho_u_x = vol_vector_dim::<DivDim<MomentumDensity, Length>>("grad_rho_u_x");
    let grad_rho_u_y = vol_vector_dim::<DivDim<MomentumDensity, Length>>("grad_rho_u_y");
    let grad_rho_e = vol_vector_dim::<DivDim<EnergyDensity, Length>>("grad_rho_e");
    let grad_t = vol_vector_dim::<DivDim<Temperature, Length>>("grad_T");
    let grad_u_x = vol_vector_dim::<DivDim<Velocity, Length>>("grad_u_x");
    let grad_u_y = vol_vector_dim::<DivDim<Velocity, Length>>("grad_u_y");
    let mut layout_fields = vec![
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
    ];
    if with_mms_sources {
        layout_fields.push(vol_scalar_dim::<RhoSourceUnit>(
            COMPRESSIBLE_MMS_SOURCE_RHO_FIELD,
        ));
        layout_fields.push(vol_vector_dim::<RhoUSourceUnit>(
            COMPRESSIBLE_MMS_SOURCE_RHO_U_FIELD,
        ));
        layout_fields.push(vol_scalar_dim::<RhoESourceUnit>(
            COMPRESSIBLE_MMS_SOURCE_RHO_E_FIELD,
        ));
    }
    let layout = PortRegistry::from_fields(layout_fields).into_state_layout();

    // ========================================
    // Boundary conditions.
    //
    // Inlet: rho and u are prescribed (placeholder Dirichlet values; set via
    // the solver's boundary table API). The dependent inlet entries
    // (p/T/rho_e/rho_u) are declared expressions that keep the prescribed
    // state thermodynamically consistent with the interior pressure every
    // outer iteration. Outlet: p is prescribed; everything else
    // extrapolates from the interior. The expressions reproduce the retired
    // hand-written compressible_runtime_bc kernel (including its safety
    // floors); the generic bc_expr module lowers them to one Faces kernel.
    // ========================================
    use crate::solver::gpu::enums::GpuBcKind;
    use crate::solver::model::backend::boundary::BoundaryExpr as B;

    let p_owner = || B::interior(fields.p).max(B::lit(1.0e-6));
    let gm1 = || B::param(EOS_GM1.to_untyped());
    let gm1_safe = || gm1().max(B::lit(1.0e-6));
    let r_safe = || B::param(EOS_R.to_untyped()).max(B::lit(1.0e-12));

    // Inlet: kinetic energy of the prescribed state; total energy follows
    // the interior pressure (ideal gas) or is purely kinetic (barotropic).
    let inlet_ke = || {
        B::lit(0.5)
            * B::bc(fields.rho)
            * (B::bc_comp(fields.u, 0) * B::bc_comp(fields.u, 0)
                + B::bc_comp(fields.u, 1) * B::bc_comp(fields.u, 1))
    };
    let inlet_p = p_owner();
    let inlet_t = p_owner() / (B::bc(fields.rho).max(B::lit(1.0e-6)) * r_safe());
    let inlet_rho_e = gm1().select_gt(
        B::lit(0.0),
        p_owner() / gm1_safe() + inlet_ke(),
        inlet_ke(),
    );
    let inlet_rho_u = |component: u32| B::bc(fields.rho) * B::bc_comp(fields.u, component);

    // Outlet: extrapolate the non-pressure state from the interior.
    let outlet_rho = || B::interior(fields.rho).max(B::lit(1.0e-6));
    let outlet_u = |component: u32| B::interior_comp(fields.u, component);
    let outlet_ke =
        || B::lit(0.5) * outlet_rho() * (outlet_u(0) * outlet_u(0) + outlet_u(1) * outlet_u(1));
    let outlet_p = || B::bc(fields.p);
    let outlet_t = outlet_p() / (outlet_rho() * r_safe());
    let outlet_rho_e = gm1().select_gt(
        B::lit(0.0),
        outlet_p() / gm1_safe() + outlet_ke(),
        outlet_ke(),
    );
    let outlet_rho_u = |component: u32| outlet_rho() * outlet_u(component);

    let mut boundaries = BoundarySpec::default();
    boundaries.set_field(
        "rho",
        FieldBoundarySpec::new()
            // Inlet density is Dirichlet (placeholder value); update via the solver's boundary
            // table API (and keep `rho_u`/`rho_e` consistent with the chosen inlet state).
            .set_uniform(
                GpuBoundaryType::Inlet,
                1,
                BoundaryCondition::dirichlet_dim::<Density>(1.0),
            )
            .set_uniform(
                GpuBoundaryType::Outlet,
                1,
                BoundaryCondition::with_expr_value_dim::<Density>(
                    GpuBcKind::ZeroGradient,
                    outlet_rho(),
                )?,
            )
            .set_uniform(
                GpuBoundaryType::Wall,
                1,
                BoundaryCondition::zero_gradient_dim::<DensityGradient>(),
            )
            .set_uniform(
                GpuBoundaryType::SlipWall,
                1,
                BoundaryCondition::zero_gradient_dim::<DensityGradient>(),
            )
            .set_uniform(
                GpuBoundaryType::MovingWall,
                1,
                BoundaryCondition::zero_gradient_dim::<DensityGradient>(),
            ),
    );
    boundaries.set_field(
        "rho_u",
        FieldBoundarySpec::new()
            // Inlet momentum density follows the prescribed rho and u.
            .set_components(
                GpuBoundaryType::Inlet,
                vec![
                    BoundaryCondition::with_expr_value_dim::<MomentumDensity>(
                        GpuBcKind::Dirichlet,
                        inlet_rho_u(0),
                    )?,
                    BoundaryCondition::with_expr_value_dim::<MomentumDensity>(
                        GpuBcKind::Dirichlet,
                        inlet_rho_u(1),
                    )?,
                ],
            )
            .set_components(
                GpuBoundaryType::Outlet,
                vec![
                    BoundaryCondition::with_expr_value_dim::<MomentumDensity>(
                        GpuBcKind::ZeroGradient,
                        outlet_rho_u(0),
                    )?,
                    BoundaryCondition::with_expr_value_dim::<MomentumDensity>(
                        GpuBcKind::ZeroGradient,
                        outlet_rho_u(1),
                    )?,
                ],
            )
            .set_uniform(
                GpuBoundaryType::Wall,
                2,
                BoundaryCondition::dirichlet_dim::<MomentumDensity>(0.0),
            )
            .set_uniform(
                GpuBoundaryType::SlipWall,
                2,
                BoundaryCondition::zero_gradient_dim::<MomentumDensityGradient>(),
            )
            .set_uniform(
                GpuBoundaryType::MovingWall,
                2,
                BoundaryCondition::dirichlet_dim::<MomentumDensity>(0.0),
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
                BoundaryCondition::dirichlet_dim::<Velocity>(0.0),
            )
            .set_components(
                GpuBoundaryType::Outlet,
                vec![
                    BoundaryCondition::with_expr_value_dim::<Velocity>(
                        GpuBcKind::ZeroGradient,
                        outlet_u(0),
                    )?,
                    BoundaryCondition::with_expr_value_dim::<Velocity>(
                        GpuBcKind::ZeroGradient,
                        outlet_u(1),
                    )?,
                ],
            )
            // Walls: no-slip (matches `rho_u` Dirichlet=0).
            .set_uniform(
                GpuBoundaryType::Wall,
                2,
                BoundaryCondition::dirichlet_dim::<Velocity>(0.0),
            )
            .set_uniform(
                GpuBoundaryType::SlipWall,
                2,
                BoundaryCondition::zero_gradient_dim::<InvTime>(),
            )
            .set_uniform(
                GpuBoundaryType::MovingWall,
                2,
                BoundaryCondition::dirichlet_dim::<Velocity>(0.0),
            ),
    );
    boundaries.set_field(
        "rho_e",
        FieldBoundarySpec::new()
            // Inlet energy follows the prescribed rho/u and the interior pressure.
            .set_uniform(
                GpuBoundaryType::Inlet,
                1,
                BoundaryCondition::with_expr_value_dim::<EnergyDensity>(
                    GpuBcKind::Dirichlet,
                    inlet_rho_e,
                )?,
            )
            .set_uniform(
                GpuBoundaryType::Outlet,
                1,
                BoundaryCondition::with_expr_value_dim::<EnergyDensity>(
                    GpuBcKind::ZeroGradient,
                    outlet_rho_e,
                )?,
            )
            .set_uniform(
                GpuBoundaryType::Wall,
                1,
                BoundaryCondition::zero_gradient_dim::<EnergyDensityGradient>(),
            )
            .set_uniform(
                GpuBoundaryType::SlipWall,
                1,
                BoundaryCondition::zero_gradient_dim::<EnergyDensityGradient>(),
            )
            .set_uniform(
                GpuBoundaryType::MovingWall,
                1,
                BoundaryCondition::zero_gradient_dim::<EnergyDensityGradient>(),
            ),
    );
    boundaries.set_field(
        "p",
        FieldBoundarySpec::new()
            // Inlet pressure floats with the interior (subsonic inflow).
            .set_uniform(
                GpuBoundaryType::Inlet,
                1,
                BoundaryCondition::with_expr_value_dim::<Pressure>(
                    GpuBcKind::Dirichlet,
                    inlet_p,
                )?,
            )
            .set_uniform(
                GpuBoundaryType::Outlet,
                1,
                BoundaryCondition::dirichlet_dim::<Pressure>(0.0),
            )
            .set_uniform(
                GpuBoundaryType::Wall,
                1,
                BoundaryCondition::zero_gradient_dim::<PressureGradient>(),
            )
            .set_uniform(
                GpuBoundaryType::SlipWall,
                1,
                BoundaryCondition::zero_gradient_dim::<PressureGradient>(),
            )
            .set_uniform(
                GpuBoundaryType::MovingWall,
                1,
                BoundaryCondition::zero_gradient_dim::<PressureGradient>(),
            ),
    );
    boundaries.set_field(
        "T",
        FieldBoundarySpec::new()
            // Inlet temperature follows the prescribed rho and interior pressure.
            .set_uniform(
                GpuBoundaryType::Inlet,
                1,
                BoundaryCondition::with_expr_value_dim::<Temperature>(
                    GpuBcKind::Dirichlet,
                    inlet_t,
                )?,
            )
            .set_uniform(
                GpuBoundaryType::Outlet,
                1,
                BoundaryCondition::with_expr_value_dim::<Temperature>(
                    GpuBcKind::ZeroGradient,
                    outlet_t,
                )?,
            )
            .set_uniform(
                GpuBoundaryType::Wall,
                1,
                BoundaryCondition::zero_gradient_dim::<TemperatureGradient>(),
            )
            .set_uniform(
                GpuBoundaryType::SlipWall,
                1,
                BoundaryCondition::zero_gradient_dim::<TemperatureGradient>(),
            )
            .set_uniform(
                GpuBoundaryType::MovingWall,
                1,
                BoundaryCondition::zero_gradient_dim::<TemperatureGradient>(),
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
        scheme: crate::solver::model::flux_module::FluxSchemeSpec::CentralUpwind(
            compressible_central_upwind_decl(),
        ),
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
    .map_err(|e| format!("failed to build flux_module module: {e}"))?;

    Ok(ModelSpec {
        id: if with_mms_sources {
            "compressible_mms"
        } else {
            "compressible"
        },
        // Route compressible through the generic coupled pipeline.
        system,
        state_layout: layout,
        boundaries,

        modules: vec![
            crate::solver::model::modules::eos::eos_module(eos),
            flux_module_module,
            crate::solver::model::modules::bc_expr::bc_expr_module(),
            {
                let mut m =
                    crate::solver::model::modules::generic_coupled::generic_coupled_module(method);
                // Use full updates by default for compressible dual-time stepping.
                // Runtime fallback damping can still be applied when pseudo-time convergence
                // is explicitly classified as nonconverged.
                m.relaxation_defaults =
                    Some(crate::solver::model::module::RelaxationDefaults {
                        alpha_u: 1.0,
                        alpha_p: 1.0,
                    });
                m
            },
        ],
        // Use global defaults.
        linear_solver: None,
        primitives,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::model::backend::algebraic::{AlgExpr, ParamRef};
    use crate::solver::model::backend::ast::Equation;
    use crate::solver::model::backend::typed_ast::typed_fvc;
    use cfd2_ir::dimensions::UnitDimension;

    /// The hand-written pseudo-source recovery rows exactly as they were
    /// declared before algebraic-equation lowering replaced them. This is the
    /// golden reference: the lowering must reproduce these terms bit-for-bit
    /// (same ops, same fields, same coefficient trees, same order), which is
    /// what guarantees byte-identical generated WGSL.
    fn handwritten_recovery_equations() -> Vec<Equation> {
        let rho_typed = TypedFieldRef::<Density, Scalar>::new("rho");
        let rho_u_typed = TypedFieldRef::<MomentumDensity, Vector2>::new("rho_u");
        let rho_e_typed = TypedFieldRef::<EnergyDensity, Scalar>::new("rho_e");
        let u_typed = TypedFieldRef::<Velocity, Vector2>::new("u");
        let p_typed = TypedFieldRef::<Pressure, Scalar>::new("p");
        let t_typed = TypedFieldRef::<Temperature, Scalar>::new("T");

        // Primitive velocity recovery: u = rho_u / rho
        // Implemented as: (rho/dt) * u = (1/dt) * rho_u
        let inv_dt_typed = TypedFieldRef::<InvTime, Scalar>::new("inv_dt");
        let inv_dt_coeff = TypedCoeff::from_field(inv_dt_typed);
        let rho_coeff = TypedCoeff::from_field(rho_typed);
        let minus_one_coeff: TypedCoeff<Dimensionless> = TypedCoeff::constant(-1.0);

        let rho_over_dt = rho_coeff.clone().multiply(inv_dt_coeff.clone());
        let minus_rho_over_dt = minus_one_coeff.clone().multiply(rho_over_dt);

        let u_source_1 = typed_fvm::source_coeff(minus_rho_over_dt, u_typed);
        let u_source_2 = typed_fvm::source_coeff(inv_dt_coeff.clone(), rho_u_typed);

        let u_eqn = (u_source_1.cast_to::<Force>() + u_source_2.cast_to::<Force>()).eqn(u_typed);

        // Primitive pressure recovery (algebraic constraint)
        let gm1_typed =
            TypedCoeff::from_field(TypedFieldRef::<Dimensionless, Scalar>::new("eos_gm1"));
        let dp_drho_typed = TypedCoeff::from_field(TypedFieldRef::<
            cfd2_ir::dimensions::DivDim<Pressure, Density>,
            Scalar,
        >::new("eos_dp_drho"));
        let p_offset_typed =
            TypedCoeff::from_field(TypedFieldRef::<Pressure, Scalar>::new("eos_p_offset"));
        let half_coeff: TypedCoeff<Dimensionless> = TypedCoeff::constant(0.5);

        let minus_gm1 = minus_one_coeff.clone().multiply(gm1_typed.clone());
        let minus_gm1_over_dt = minus_gm1.multiply(inv_dt_coeff.clone());

        let u2 = TypedCoeff::mag_sqr(u_typed);
        let half_gm1_over_dt = half_coeff
            .multiply(gm1_typed)
            .multiply(inv_dt_coeff.clone());
        let rho_coeff_term = half_gm1_over_dt.multiply(u2);

        let minus_dp_drho = minus_one_coeff.clone().multiply(dp_drho_typed);
        let minus_dp_drho_over_dt = minus_dp_drho.multiply(inv_dt_coeff.clone());

        let minus_p_offset = minus_one_coeff.clone().multiply(p_offset_typed);
        let minus_p_offset_over_dt = minus_p_offset.multiply(inv_dt_coeff.clone());

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

        // Temperature recovery: T = p / (rho * R)
        // Implemented as: (rho*R/dt) * T = (1/dt) * p
        let r_typed = TypedCoeff::from_field(TypedFieldRef::<
            cfd2_ir::dimensions::DivDim<Pressure, MulDim<Density, Temperature>>,
            Scalar,
        >::new("eos_r"));

        let rho_r_over_dt = rho_coeff.multiply(r_typed).multiply(inv_dt_coeff.clone());
        let minus_inv_dt = minus_one_coeff.multiply(inv_dt_coeff);

        let t_source_1 = typed_fvm::source_coeff(rho_r_over_dt, t_typed);
        let t_source_2 = typed_fvm::source_coeff(minus_inv_dt, p_typed);

        let t_eqn = (t_source_1.cast_to::<Power>() + t_source_2.cast_to::<Power>()).eqn(t_typed);

        vec![u_eqn, p_eqn, t_eqn]
    }

    #[test]
    fn algebraic_recovery_rows_match_handwritten_golden() {
        let system = compressible_system();
        let eqs = system.equations();
        assert_eq!(eqs.len(), 6, "3 conservation + 3 recovery rows");

        let golden = handwritten_recovery_equations();
        assert_eq!(eqs[3], golden[0], "u recovery row");
        assert_eq!(eqs[4], golden[1], "p EOS row");
        assert_eq!(eqs[5], golden[2], "T recovery row");
    }

    #[test]
    fn eos_params_match_uniform_port_manifest() {
        let manifest = crate::solver::model::modules::eos_ports::eos_uniform_port_manifest();
        let declared = [
            (EOS_GAMMA.name(), Dimensionless::UNIT),
            (EOS_GM1.name(), Dimensionless::UNIT),
            (
                EOS_R.name(),
                DivDim::<Pressure, MulDim<Density, Temperature>>::UNIT,
            ),
            (EOS_DP_DRHO.name(), DivDim::<Pressure, Density>::UNIT),
            (EOS_P_OFFSET.name(), Pressure::UNIT),
        ];
        for (name, unit) in declared {
            let spec = manifest
                .params
                .iter()
                .find(|p| p.wgsl_field == name)
                .unwrap_or_else(|| panic!("param '{name}' missing from EOS port manifest"));
            assert_eq!(spec.unit, unit, "unit mismatch for param '{name}'");
        }
    }

    #[test]
    fn wave_speed_sq_declares_gamma_r_t() {
        // cast_to inside the constructor already asserts the runtime unit is
        // Velocity^2; here we pin the declared structure.
        let expr = compressible_wave_speed_sq().to_untyped();
        let expected = AlgExpr::Mul(
            Box::new(AlgExpr::Mul(
                Box::new(AlgExpr::Param(ParamRef::new(
                    "eos_gamma",
                    Dimensionless::UNIT,
                ))),
                Box::new(AlgExpr::Param(ParamRef::new(
                    "eos_r",
                    DivDim::<Pressure, MulDim<Density, Temperature>>::UNIT,
                ))),
            )),
            Box::new(AlgExpr::Field(vol_scalar_dim::<Temperature>("T"))),
        );
        assert_eq!(expr, expected);
    }
}
