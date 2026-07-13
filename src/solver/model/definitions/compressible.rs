// Model contract: this discretization (explicit KT/vanLeer flux + implicit
// inv_dt-scaled EOS-recovery rows) develops a slow secular thermo-field
// instability in the inviscid limit at moderate Mach, whose growth rate rises
// with mesh resolution and is damped ONLY by physical viscosity (mu k^2 must
// beat it; mu = 5e-3 holds through n = 32, mu = 0.05 is robust). Low-Mach
// preconditioning makes it WORSE (it removes acoustic-scale dissipation). Do
// not run this model inviscid.

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
use cfd2_ir::dimensions::{
    Density, Dimensionless, DivDim, DynamicViscosity, EnergyDensity, Force, InvTime, Length,
    MassFlux, MomentumDensity, MulDim, Power, Pressure, Temperature, Velocity, Volume,
};
type DensityGradient = DivDim<Density, Length>;
type MomentumDensityGradient = DivDim<MomentumDensity, Length>;
type EnergyDensityGradient = DivDim<EnergyDensity, Length>;
type PressureGradient = DivDim<Pressure, Length>;
type TemperatureGradient = DivDim<Temperature, Length>;

// Implicit biharmonic. The auxiliary undivided-Laplacian unknowns
// `lap_X = laplacian(X)` carry unit [X * Length]: the surface-integral
// Laplacian's integrated unit is `X * Area/Length = X * Length`, so the static
// identity row `sp(-1, lap_X) + laplacian(1, X) = 0` is unit-consistent and
// `laplacian(bih_eps4, lap_X)` (bih_eps4 a velocity) lands on the conserved
// equation's own unit.
type LapDensity = MulDim<Density, Length>;
type LapMomentumDensity = MulDim<MomentumDensity, Length>;
type LapEnergyDensity = MulDim<EnergyDensity, Length>;

use super::incompressible_momentum::IBM_MOMENTUM_PENALTY_FIELD;
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
const EOS_P_REF: TypedParamRef<Pressure> = TypedParamRef::new("eos_p_ref");
const EOS_RHO_REF: TypedParamRef<Density> = TypedParamRef::new("eos_rho_ref");
// Gauge-storage references: the conserved state fields store deviations from a
// constant reference state (`rho_state = rho_abs - gauge_rho_ref`,
// `rho_e_state = rho_e_abs - gauge_e_ref`, `p_state = p_abs - gauge_p_ref`);
// all-zero references recover the historical absolute storage bit-for-bit.
// `gauge_p_bias = gm1*gauge_e_ref + p_ref - gauge_p_ref` is precomputed on the
// host in f64 so the affine constants of the STATE-form pressure closure
// cancel exactly (zero for a self-consistent gauge, `p_ref` when gauge is off).
const EOS_GAUGE_RHO_REF: TypedParamRef<Density> = TypedParamRef::new("eos_gauge_rho_ref");
const EOS_GAUGE_P_REF: TypedParamRef<Pressure> = TypedParamRef::new("eos_gauge_p_ref");
const EOS_GAUGE_P_BIAS: TypedParamRef<Pressure> = TypedParamRef::new("eos_gauge_p_bias");
// Inlet driving mode (0 = velocity inlet, 1 = pressure inlet + floating
// outlet). A runtime constant, so ONE committed bc_expr kernel serves both
// modes; the closures below branch with `select_gt(mode, 0.5, ..)` and the
// velocity-mode branch reproduces the historical expressions bit-for-bit.
const BC_PRESSURE_INLET: TypedParamRef<Dimensionless> = TypedParamRef::new("bc_pressure_inlet");

/// Declared squared sound speed of the runtime EOS:
/// `c^2 = gamma * R * T + dp_drho`.
///
/// The central-upwind flux derivation lowers this to the acoustic speed used
/// in the Kurganov wave bounds. The affine barotropic contribution is
/// constant, so the existing analytic `grad(c)` remains exact for both the
/// ideal-gas (`dp_drho=0`) and linear-compressibility (`gamma=0`) families.
pub fn compressible_wave_speed_sq() -> TypedAlgExpr<MulDim<Velocity, Velocity>, Scalar> {
    let t_typed = TypedFieldRef::<Temperature, Scalar>::new("T");
    (typed_alg::param(EOS_GAMMA) * typed_alg::param(EOS_R) * typed_alg::field(t_typed))
        .cast_to::<MulDim<Velocity, Velocity>>()
        + typed_alg::param(EOS_DP_DRHO).cast_to::<MulDim<Velocity, Velocity>>()
}

/// Declared generalized squared wave speed: `c^2 = gamma * p / rho + dp_drho`.
///
/// The `dp_drho` term covers barotropic closures (linear compressibility);
/// it is zero for an ideal gas. The flux derivation evaluates this over
/// reconstructed face states for low-Mach dissipation scaling.
pub fn compressible_generalized_wave_speed_sq() -> TypedAlgExpr<DivDim<Pressure, Density>, Scalar> {
    let p_typed = TypedFieldRef::<Pressure, Scalar>::new("p");
    let rho_typed = TypedFieldRef::<Density, Scalar>::new("rho");
    // Gauge storage: the acoustic speed is a property of the ABSOLUTE state.
    ((typed_alg::param(EOS_GAMMA)
        * (typed_alg::field(p_typed) + typed_alg::param(EOS_GAUGE_P_REF)))
        / (typed_alg::field(rho_typed) + typed_alg::param(EOS_GAUGE_RHO_REF)))
    .cast_to::<DivDim<Pressure, Density>>()
        + typed_alg::param(EOS_DP_DRHO)
}

/// Central-upwind flux declaration for this model: which fields play which
/// conserved/primitive role, plus the EOS relations as math. Face pressure is
/// closed from one reconstructed conserved state:
///
/// `p_f = gm1 * (rho_e_f - |rho_u_f|^2/(2*rho_f))`
/// `      + dp_drho*(rho_f-rho_ref) + p_ref`.
///
/// This is exact for both runtime EOS families: ideal gas sets the affine
/// terms to zero, while linear compressibility sets `gm1=0`. Temperature
/// remains available independently for heat conduction and acoustic-speed
/// reconstruction, but it is not a second, potentially inconsistent pressure
/// oracle at a high-order face.
pub fn compressible_central_upwind_decl() -> crate::solver::model::flux_schemes::CentralUpwindDecl {
    let rho_typed = TypedFieldRef::<Density, Scalar>::new("rho");
    let rho_u_typed = TypedFieldRef::<MomentumDensity, Vector2>::new("rho_u");
    let rho_e_typed = TypedFieldRef::<EnergyDensity, Scalar>::new("rho_e");
    // STATE-form closure: yields the STORED (gauge) pressure from the STORED
    // conserved state, in both conventions. The kinetic-energy division uses
    // the ABSOLUTE density; the barotropic term groups the two reference
    // constants first (exact constant-constant arithmetic) so a gauge-stored
    // liquid density never round-trips through its large absolute value; and
    // `gauge_p_bias` carries the host-cancelled affine tail (`p_ref` when the
    // gauge is off, exactly zero when it is on).
    let kinetic_energy = (typed_alg::constant(0.5) * typed_alg::mag_sqr(rho_u_typed)
        / (typed_alg::field(rho_typed) + typed_alg::param(EOS_GAUGE_RHO_REF)))
    .cast_to::<EnergyDensity>();
    let ideal_pressure = (typed_alg::param(EOS_GM1)
        * (typed_alg::field(rho_e_typed) - kinetic_energy))
        .cast_to::<Pressure>();
    let barotropic_pressure = (typed_alg::param(EOS_DP_DRHO)
        * (typed_alg::field(rho_typed)
            + (typed_alg::param(EOS_GAUGE_RHO_REF) - typed_alg::param(EOS_RHO_REF))))
    .cast_to::<Pressure>();
    crate::solver::model::flux_schemes::CentralUpwindDecl {
        density: "rho",
        momentum: "rho_u",
        energy: "rho_e",
        temperature: "T",
        velocity: "u",
        pressure_field: "p",
        pressure: (ideal_pressure + barotropic_pressure + typed_alg::param(EOS_GAUGE_P_BIAS))
            .to_untyped(),
        wave_speed_sq: compressible_wave_speed_sq().to_untyped(),
        generalized_wave_speed_sq: compressible_generalized_wave_speed_sq().to_untyped(),
    }
}

fn build_compressible_system(fields: &CompressibleFields) -> EquationSystem {
    build_compressible_system_impl(fields, false, false, false)
}

fn build_compressible_system_impl(
    _fields: &CompressibleFields,
    with_mms_sources: bool,
    biharmonic: bool,
    ibm: bool,
) -> EquationSystem {
    // cast_to() aligns terms to canonical dimension types: type-level dimension
    // expressions are not normalized, so semantically equivalent dimensions are
    // distinct types that cast_to() unifies.
    let rho_typed = TypedFieldRef::<Density, Scalar>::new("rho");
    let rho_u_typed = TypedFieldRef::<MomentumDensity, Vector2>::new("rho_u");
    let rho_e_typed = TypedFieldRef::<EnergyDensity, Scalar>::new("rho_e");

    let u_typed = TypedFieldRef::<Velocity, Vector2>::new("u");
    let p_typed = TypedFieldRef::<Pressure, Scalar>::new("p");
    let t_typed = TypedFieldRef::<Temperature, Scalar>::new("T");
    let mu_typed = TypedFieldRef::<DynamicViscosity, Scalar>::new("mu");

    let phi_rho_typed = TypedFluxRef::<MassFlux, Scalar>::new("phi_rho");
    let phi_rho_u_typed = TypedFluxRef::<Force, Vector2>::new("phi_rho_u");
    let phi_rho_e_typed = TypedFluxRef::<Power, Scalar>::new("phi_rho_e");

    let mu_coeff = TypedCoeff::from_field(mu_typed);

    // Auxiliary undivided-Laplacian unknowns `lap_X` (one per conserved field)
    // and the velocity-scaled coefficient field `bih_eps4` (= eps4 * acoustic
    // speed), set uniformly at runtime like `mu`. `laplacian(-bih_eps4, lap_X)`
    // on a conserved row assembles `+bih_eps4 * lap2_undiv(lap_X)` (the implicit
    // diffusion operator is `-coeff * lap2`), i.e. the dissipative
    // `-bih_eps4 * grad^4 X`.
    let lap_rho_typed = TypedFieldRef::<LapDensity, Scalar>::new("lap_rho");
    let lap_rho_u_typed = TypedFieldRef::<LapMomentumDensity, Vector2>::new("lap_rho_u");
    let lap_rho_e_typed = TypedFieldRef::<LapEnergyDensity, Scalar>::new("lap_rho_e");
    let neg_bih_eps4 = TypedCoeff::<Dimensionless>::constant(-1.0).multiply(
        TypedCoeff::from_field(TypedFieldRef::<Velocity, Scalar>::new("bih_eps4")),
    );

    // Continuity: ddt(rho) + div(phi_rho, rho) = 0
    let rho_ddt = typed_fvm::ddt(rho_typed);
    let rho_div = typed_fvm::div_flux(phi_rho_typed, rho_typed);

    let mut rho_sum = rho_ddt.cast_to::<MassFlux>() + rho_div.cast_to::<MassFlux>();
    if with_mms_sources {
        let mms_rho = TypedCoeff::from_field(TypedFieldRef::<RhoSourceUnit, Scalar>::new(
            COMPRESSIBLE_MMS_SOURCE_RHO_FIELD,
        ));
        rho_sum = rho_sum + typed_fvc::source_coeff(mms_rho, rho_typed).cast_to::<MassFlux>();
    }
    if biharmonic {
        rho_sum = rho_sum
            + typed_fvm::laplacian(neg_bih_eps4.clone(), lap_rho_typed).cast_to::<MassFlux>();
    }
    let rho_eqn = rho_sum.eqn(rho_typed);

    // Momentum: ddt(rho_u) + div(phi_rho_u, rho_u) - laplacian(mu, u) = 0
    let rho_u_ddt = typed_fvm::ddt(rho_u_typed);
    let rho_u_div = typed_fvm::div_flux(phi_rho_u_typed, rho_u_typed);
    let viscous_term = typed_fvm::laplacian(mu_coeff, u_typed);

    let mut rho_u_sum = rho_u_ddt.cast_to::<Force>()
        + rho_u_div.cast_to::<Force>()
        + viscous_term.cast_to::<Force>();
    if with_mms_sources {
        let mms_rho_u =
            TypedFieldRef::<RhoUSourceUnit, Vector2>::new(COMPRESSIBLE_MMS_SOURCE_RHO_U_FIELD);
        rho_u_sum = rho_u_sum + typed_fvc::source_vector(mms_rho_u, rho_u_typed).cast_to::<Force>();
    }
    if biharmonic {
        rho_u_sum = rho_u_sum
            + typed_fvm::laplacian(neg_bih_eps4.clone(), lap_rho_u_typed).cast_to::<Force>();
    }
    if ibm {
        // Immersed-boundary Brinkman penalisation (structured only). The momentum
        // unknown here is the CONSERVED momentum density rho_u (not primitive U),
        // so the implicit sink `source_coeff(Sp, rho_u)` drives rho_u -> 0 (hence
        // u = rho_u/rho -> 0) where the mask is large. Sp has FREQUENCY (1/Time)
        // units so `Sp * rho_u * V` integrates to Force. Sp=0 in the fluid is the
        // identity recovery of the base operator. NOTE: this pins the VELOCITY but
        // the mass/energy KT fluxes (phi_rho, phi_rho_e) still cross solid faces,
        // so the obstacle is porous to mass/energy — a true no-penetration wall
        // additionally needs those fluxes masked (documented limitation).
        let penalty_typed = TypedFieldRef::<InvTime, Scalar>::new(IBM_MOMENTUM_PENALTY_FIELD);
        let penalty_coeff = TypedCoeff::from_field(penalty_typed);
        rho_u_sum =
            rho_u_sum + typed_fvm::source_coeff(penalty_coeff, rho_u_typed).cast_to::<Force>();
    }
    let rho_u_eqn = rho_u_sum.eqn(rho_u_typed);

    // Energy: ddt(rho_e) + div(phi_rho_e, rho_e) - laplacian(kappa, T) = 0
    // kappa has unit Power/(Length*Temperature).
    let kappa_typed = TypedCoeff::from_field(TypedFieldRef::<
        cfd2_ir::dimensions::DivDim<Power, MulDim<Length, Temperature>>,
        Scalar,
    >::new("kappa"));

    let rho_e_ddt = typed_fvm::ddt(rho_e_typed);
    let rho_e_div = typed_fvm::div_flux(phi_rho_e_typed, rho_e_typed);
    let heat_flux = typed_fvm::laplacian(kappa_typed, t_typed);

    let mut rho_e_sum =
        rho_e_ddt.cast_to::<Power>() + rho_e_div.cast_to::<Power>() + heat_flux.cast_to::<Power>();
    if with_mms_sources {
        let mms_rho_e = TypedCoeff::from_field(TypedFieldRef::<RhoESourceUnit, Scalar>::new(
            COMPRESSIBLE_MMS_SOURCE_RHO_E_FIELD,
        ));
        rho_e_sum = rho_e_sum + typed_fvc::source_coeff(mms_rho_e, rho_e_typed).cast_to::<Power>();
    }
    if biharmonic {
        rho_e_sum =
            rho_e_sum + typed_fvm::laplacian(neg_bih_eps4, lap_rho_e_typed).cast_to::<Power>();
    }
    let rho_e_eqn = rho_e_sum.eqn(rho_e_typed);

    // Primitive recovery, declared as algebraic relations. These lower to
    // inv_dt-scaled coupled source rows; fields multiplying the linear unknown
    // of each product (e.g. rho in `rho * u`) are frozen at the current state
    // (Picard linearization).

    // Velocity recovery: rho_abs * u = rho_u, with the ABSOLUTE density
    // reconstructed from the gauge-stored state (`rho + gauge_rho_ref`; the
    // reference is zero for absolute storage). The lowering distributes the
    // sum and merges both target-linear products into one summed coefficient.
    let u_recovery = typed_alg::equation(
        u_typed,
        typed_alg::field(rho_u_typed),
        ((typed_alg::field(rho_typed) + typed_alg::param(EOS_GAUGE_RHO_REF))
            * typed_alg::field(u_typed))
        .cast_to::<MomentumDensity>(),
    );

    // Linearized EOS: p = (gamma-1)*rho_e - (gamma-1)/2*|u|^2*rho
    //                     + dp_drho*(rho-rho_ref) + p_ref.
    // The reference-centered form avoids subtracting O(K) affine terms to
    // recover an O(p_ref) liquid pressure in f32. Ideal gas sets
    // dp_drho = rho_ref = p_ref = 0; linear compressibility sets gamma-1 = 0.
    let pressure_eos = typed_alg::equation(
        p_typed,
        typed_alg::field(p_typed),
        (typed_alg::param(EOS_GM1) * typed_alg::field(rho_e_typed)).cast_to::<Pressure>()
            - (typed_alg::constant(0.5)
                * typed_alg::param(EOS_GM1)
                * typed_alg::mag_sqr(u_typed)
                * (typed_alg::field(rho_typed) + typed_alg::param(EOS_GAUGE_RHO_REF)))
            .cast_to::<Pressure>()
            + (typed_alg::param(EOS_DP_DRHO)
                * (typed_alg::field(rho_typed)
                    + (typed_alg::param(EOS_GAUGE_RHO_REF) - typed_alg::param(EOS_RHO_REF))))
            .cast_to::<Pressure>()
            + typed_alg::param(EOS_GAUGE_P_BIAS),
    );

    // Temperature recovery: rho_abs * R * T = p_abs (temperature stays an
    // ABSOLUTE field; both sides reconstruct absolutes from the gauge state).
    let temperature_recovery = typed_alg::equation(
        t_typed,
        ((typed_alg::field(rho_typed) + typed_alg::param(EOS_GAUGE_RHO_REF))
            * typed_alg::param(EOS_R)
            * typed_alg::field(t_typed))
        .cast_to::<Pressure>(),
        typed_alg::field(p_typed) + typed_alg::param(EOS_GAUGE_P_REF),
    );

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

    if biharmonic {
        // Auxiliary undivided-Laplacian constraint rows, appended last so the
        // lap unknowns occupy coupled slots 8..12 (matching
        // `CompressibleBiharmonicAxis2D`). Each is the static identity
        // `sp(-1, lap_X) + laplacian(1, X) = 0` ⟹ `lap_X = laplacian(X)`; RHS is
        // zero, so the steady state is unchanged by the auxiliary block. rho_u
        // is a Vector2, so `lap_rho_u` is per-component.
        let one = TypedCoeff::<Dimensionless>::constant(1.0);
        let minus_one = TypedCoeff::<Dimensionless>::constant(-1.0);
        let lap_rho_eqn = (typed_fvm::sp(minus_one.clone(), lap_rho_typed).cast_to::<LapDensity>()
            + typed_fvm::laplacian(one.clone(), rho_typed).cast_to::<LapDensity>())
        .eqn(lap_rho_typed);
        let lap_rho_u_eqn = (typed_fvm::sp(minus_one.clone(), lap_rho_u_typed)
            .cast_to::<LapMomentumDensity>()
            + typed_fvm::laplacian(one.clone(), rho_u_typed).cast_to::<LapMomentumDensity>())
        .eqn(lap_rho_u_typed);
        let lap_rho_e_eqn = (typed_fvm::sp(minus_one, lap_rho_e_typed)
            .cast_to::<LapEnergyDensity>()
            + typed_fvm::laplacian(one, rho_e_typed).cast_to::<LapEnergyDensity>())
        .eqn(lap_rho_e_typed);
        system.add_equation(lap_rho_eqn);
        system.add_equation(lap_rho_u_eqn);
        system.add_equation(lap_rho_e_eqn);
    }

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

pub fn compressible_model_with_eos(
    eos: crate::solver::model::eos::EosSpec,
) -> Result<ModelSpec, String> {
    compressible_model_impl(eos, false, false)
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
        false,
    )
}

/// `compressible_mms` plus IMPLICIT k-selective biharmonic dissipation.
/// Promotes the auxiliary undivided-Laplacian unknowns `lap_X` to the coupled
/// block (stride 8 -> 12) with the constraint rows `lap_X = laplacian(X)` and
/// adds `laplacian(-bih_eps4, lap_X)` to each conserved equation, so the whole
/// `-eps4*grad^4 X` dissipation lives in the matrix (no explicit flux term, no
/// `dt <~ C*h^2` limit). The coefficient is the per-cell `bih_eps4` field
/// (= eps4 * acoustic speed), set uniformly at runtime like `mu`, so one
/// registered model serves any eps4. Used by the inviscid-mode probes; NOT a
/// shipped default. At eps4>0 the bounded-domain MMS order is conditioning-
/// limited (the implicit `grad^4` operator, condition ~h^-4, outruns
/// Jacobi+FGMRES at fine mesh).
pub fn compressible_mms_biharmonic_model() -> Result<ModelSpec, String> {
    compressible_model_impl(
        crate::solver::model::eos::EosSpec::IdealGas {
            gamma: 1.4,
            gas_constant: 1.0,
            temperature: 1.0,
        },
        true,
        true,
    )
}

/// STRUCTURED (`TopologyMode::Structured2D`) density-based compressible model:
/// the same conserved (rho, rho_u, rho_e) central-upwind solver on the dense
/// Cartesian grid.
pub fn compressible_structured_model() -> Result<ModelSpec, String> {
    compressible_model_impl_topo(
        crate::solver::model::eos::EosSpec::IdealGas {
            gamma: 1.4,
            gas_constant: 1.0,
            temperature: 1.0,
        },
        false,
        false,
        cfd2_ir::equation::TopologyMode::Structured2D,
    )
}

fn compressible_model_impl(
    eos: crate::solver::model::eos::EosSpec,
    with_mms_sources: bool,
    biharmonic: bool,
) -> Result<ModelSpec, String> {
    compressible_model_impl_topo(
        eos,
        with_mms_sources,
        biharmonic,
        cfd2_ir::equation::TopologyMode::Unstructured,
    )
}

fn compressible_model_impl_topo(
    eos: crate::solver::model::eos::EosSpec,
    with_mms_sources: bool,
    biharmonic: bool,
    topology: cfd2_ir::equation::TopologyMode,
) -> Result<ModelSpec, String> {
    let fields = CompressibleFields::new();
    // The structured (dense-Cartesian) variant is immersed-boundary-capable:
    // obstacles ride a per-cell Brinkman momentum mask, never cut from the grid.
    let ibm = topology == cfd2_ir::equation::TopologyMode::Structured2D;
    let mut system = build_compressible_system_impl(&fields, with_mms_sources, biharmonic, ibm);
    system.set_topology(topology);
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
    if biharmonic {
        // Auxiliary undivided-Laplacian unknowns `lap_X` (solved via the
        // `lap_X = laplacian(X)` constraint rows) plus the runtime coefficient
        // field `bih_eps4`. `lap_rho_u` is a single Vector2 (matching `rho_u`,
        // so the biharmonic diffusion is component-wise). Present ONLY in this
        // variant; the distinct model id keeps default models on their committed
        // (stride-22/26) kernels.
        layout_fields.push(vol_scalar_dim::<LapDensity>("lap_rho"));
        layout_fields.push(vol_vector_dim::<LapMomentumDensity>("lap_rho_u"));
        layout_fields.push(vol_scalar_dim::<LapEnergyDensity>("lap_rho_e"));
        // `bih_eps4` = eps4 * acoustic speed (velocity units), the coefficient of
        // `laplacian(-bih_eps4, lap_X)`. A uniform-valued storage field (like
        // `mu`) set at runtime, so one registered model serves any eps4 without
        // recompiling the shader.
        layout_fields.push(vol_scalar_dim::<Velocity>("bih_eps4"));
    }
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
    if ibm {
        // Per-cell Brinkman momentum-penalty mask (structured immersed obstacles).
        // Frequency (1/Time) coefficient of source_coeff(Sp, rho_u); see the
        // momentum equation in build_compressible_system_impl.
        layout_fields.push(vol_scalar_dim::<InvTime>(IBM_MOMENTUM_PENALTY_FIELD));
    }
    let layout = PortRegistry::from_fields(layout_fields).into_state_layout();

    // Boundary conditions. Inlet: rho and u are prescribed (placeholder
    // Dirichlet values; set via the solver's boundary table API). The dependent
    // inlet entries (p/T/rho_e/rho_u) are declared expressions that keep the
    // prescribed state thermodynamically consistent with the interior pressure
    // every outer iteration. Outlet: p is prescribed; everything else
    // extrapolates from the interior. bc_expr lowers them to one Faces kernel.
    use crate::solver::gpu::enums::GpuBcKind;
    use crate::solver::model::backend::boundary::BoundaryExpr as B;

    let gm1 = || B::param(EOS_GM1.to_untyped());
    let gm1_safe = || gm1().max(B::lit(1.0e-6));
    let r_safe = || B::param(EOS_R.to_untyped()).max(B::lit(1.0e-12));
    // Gauge storage: the interior pressure is STORED (gauge); the positivity
    // floor is a property of the ABSOLUTE pressure. Flooring the stored value
    // directly would rectify legitimate negative gauge pressures (acoustic
    // rarefactions below the reference) at every inlet face. Zero references
    // reduce this to the historical `interior(p).max(1e-6)`.
    let p_owner = || {
        (B::interior(fields.p) + B::param(EOS_GAUGE_P_REF.to_untyped())).max(B::lit(1.0e-6))
            - B::param(EOS_GAUGE_P_REF.to_untyped())
    };

    // Inlet: kinetic energy of the prescribed state; total energy follows
    // the interior pressure for an ideal gas, while a barotropic EOS preserves
    // the thermodynamically compatible conserved-energy value seeded by the
    // host EOS oracle.
    // Gauge storage: BC tables hold STORED (gauge) values; thermodynamic
    // reconstructions inside these expressions use the ABSOLUTE state
    // (`+ eos_gauge_rho_ref` / `+ eos_gauge_p_ref`, zero when gauge is off).
    let gauge_rho = || B::param(EOS_GAUGE_RHO_REF.to_untyped());
    let gauge_p = || B::param(EOS_GAUGE_P_REF.to_untyped());
    // Inlet driving mode: `sel(pressure, velocity)` picks the pressure-inlet
    // expression when the runtime constant is 1 and the historical
    // velocity-inlet expression (bit-identical arithmetic) when it is 0.
    let mode = || B::param(BC_PRESSURE_INLET.to_untyped());
    let sel = |on_pressure: B, on_velocity: B| {
        mode().select_gt(B::lit(0.5), on_pressure, on_velocity)
    };
    // Pressure mode: the table's p channel holds the prescribed gauge inlet
    // pressure and the T channel the reservoir temperature; the axial velocity
    // extrapolates from the interior while the transverse component is pinned.
    let inlet_p_used = || sel(B::bc(fields.p), p_owner());
    let inlet_u_used = |component: u32| {
        if component == 0 {
            sel(B::interior_comp(fields.u, 0), B::bc_comp(fields.u, 0))
        } else {
            sel(B::lit(0.0), B::bc_comp(fields.u, 1))
        }
    };
    // Pressure-mode density from the prescribed state: ideal gas inverts
    // p = rho R T at the reservoir temperature; a barotropic EOS inverts its
    // own linear closure `p = p_ref + dp_drho (rho - rho_ref)`.
    let inlet_rho_abs_pressure = || {
        gm1().select_gt(
            B::lit(0.0),
            (inlet_p_used() + gauge_p())
                / (r_safe() * B::bc(fields.t).max(B::lit(1.0e-6))),
            B::param(EOS_RHO_REF.to_untyped())
                + (inlet_p_used() + gauge_p() - B::param(EOS_P_REF.to_untyped()))
                    / B::param(EOS_DP_DRHO.to_untyped()).max(B::lit(1.0e-12)),
        )
    };
    let inlet_rho_abs = || {
        sel(
            inlet_rho_abs_pressure().max(B::lit(1.0e-6)),
            B::bc(fields.rho) + gauge_rho(),
        )
    };
    let inlet_rho = || inlet_rho_abs() - gauge_rho();
    let inlet_ke = || {
        B::lit(0.5)
            * inlet_rho_abs()
            * (inlet_u_used(0) * inlet_u_used(0) + inlet_u_used(1) * inlet_u_used(1))
    };
    let inlet_p = inlet_p_used();
    let inlet_t =
        (inlet_p_used() + gauge_p()) / (inlet_rho_abs().max(B::lit(1.0e-6)) * r_safe());
    let inlet_rho_e = gm1().select_gt(
        B::lit(0.0),
        inlet_p_used() / gm1_safe() + inlet_ke(),
        // For a barotropic EOS, the host seeds the thermodynamic conserved
        // energy from EosSpec. Preserve that prescribed table entry under the
        // bc_expr kernel's snapshot semantics instead of overwriting it with KE.
        B::bc(fields.rho_e),
    );
    let inlet_rho_u = |component: u32| inlet_rho_abs() * inlet_u_used(component);

    // Outlet: extrapolate the non-pressure state from the interior. The
    // positivity floor applies to the ABSOLUTE density; the stored ghost value
    // subtracts the gauge reference back out.
    let outlet_rho_abs = || (B::interior(fields.rho) + gauge_rho()).max(B::lit(1.0e-6));
    let outlet_rho = || outlet_rho_abs() - gauge_rho();
    let outlet_u = |component: u32| B::interior_comp(fields.u, component);
    let outlet_ke = || {
        B::lit(0.5) * outlet_rho_abs() * (outlet_u(0) * outlet_u(0) + outlet_u(1) * outlet_u(1))
    };
    // Pressure-inlet mode floats the outlet (ghost energy from the interior
    // pressure — the supersonic-outlet driving); velocity mode anchors it at
    // the table's back-pressure (the subsonic channel's gauge anchor).
    let outlet_p = || sel(p_owner(), B::bc(fields.p));
    let outlet_t = (outlet_p() + gauge_p()) / (outlet_rho_abs() * r_safe());
    let outlet_rho_e = gm1().select_gt(
        B::lit(0.0),
        outlet_p() / gm1_safe() + outlet_ke(),
        outlet_ke(),
    );
    let outlet_rho_u = |component: u32| outlet_rho_abs() * outlet_u(component);

    let mut boundaries = BoundarySpec::default();
    boundaries.set_field(
        "rho",
        FieldBoundarySpec::new()
            // Inlet density: prescribed via the boundary table in velocity
            // mode; recomputed from the prescribed pressure/reservoir
            // temperature in pressure-inlet mode.
            .set_uniform(
                GpuBoundaryType::Inlet,
                1,
                BoundaryCondition::with_expr_value_dim::<Density>(
                    GpuBcKind::Dirichlet,
                    inlet_rho(),
                )?,
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
            // Inlet velocity: prescribed via the boundary table in velocity
            // mode; extrapolated axially (transverse pinned) in pressure mode.
            .set_components(
                GpuBoundaryType::Inlet,
                vec![
                    BoundaryCondition::with_expr_value_dim::<Velocity>(
                        GpuBcKind::Dirichlet,
                        inlet_u_used(0),
                    )?,
                    BoundaryCondition::with_expr_value_dim::<Velocity>(
                        GpuBcKind::Dirichlet,
                        inlet_u_used(1),
                    )?,
                ],
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
                BoundaryCondition::with_expr_value_dim::<Pressure>(GpuBcKind::Dirichlet, inlet_p)?,
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
    if biharmonic {
        // The auxiliary lap unknowns get zero-gradient on every boundary. Two
        // reasons: (1) every coupled unknown must appear in the boundary table or
        // the WHOLE coupled system's boundary closure is ill-posed (an undeclared
        // unknown corrupts the bounded-domain solve even at eps4=0); (2)
        // zero-gradient makes the biharmonic flux through domain-boundary faces
        // zero, keeping the dissipation interior. The lap values at boundary cells
        // are still pinned by the constraint `lap_X = laplacian(X)` via X's own BC.
        let all_types = [
            GpuBoundaryType::Inlet,
            GpuBoundaryType::Outlet,
            GpuBoundaryType::Wall,
            GpuBoundaryType::SlipWall,
            GpuBoundaryType::MovingWall,
        ];
        let mut lap_rho_bc = FieldBoundarySpec::new();
        let mut lap_rho_u_bc = FieldBoundarySpec::new();
        let mut lap_rho_e_bc = FieldBoundarySpec::new();
        for t in all_types {
            lap_rho_bc =
                lap_rho_bc.set_uniform(t, 1, BoundaryCondition::zero_gradient_dim::<Density>());
            lap_rho_u_bc = lap_rho_u_bc.set_uniform(
                t,
                2,
                BoundaryCondition::zero_gradient_dim::<MomentumDensity>(),
            );
            lap_rho_e_bc = lap_rho_e_bc.set_uniform(
                t,
                1,
                BoundaryCondition::zero_gradient_dim::<EnergyDensity>(),
            );
        }
        boundaries.set_field("lap_rho", lap_rho_bc);
        boundaries.set_field("lap_rho_u", lap_rho_u_bc);
        boundaries.set_field("lap_rho_e", lap_rho_e_bc);
    }
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
    let central_upwind_decl = compressible_central_upwind_decl();
    let flux = crate::solver::model::flux_module::FluxModuleSpec::Scheme {
        gradients: Some(
            crate::solver::model::flux_module::FluxModuleGradientsSpec::FromStateLayout,
        ),
        scheme: crate::solver::model::flux_module::FluxSchemeSpec::CentralUpwind(
            central_upwind_decl,
        ),
    };
    let primitives = crate::solver::model::primitives::PrimitiveDerivations::identity();
    let explicit_primitives =
        crate::solver::model::primitives::PrimitiveDerivations::compressible_runtime_eos();

    let system_for_flux = system.clone();
    let layout_for_flux = layout.clone();
    let flux_module_module = crate::solver::model::modules::flux_module::flux_module_module(
        flux,
        &system_for_flux,
        &layout_for_flux,
        &primitives,
        (!biharmonic).then_some(&explicit_primitives),
    )
    .map_err(|e| format!("failed to build flux_module module: {e}"))?;

    Ok(ModelSpec {
        // The id keys committed-WGSL / pipeline lookup, so the biharmonic variants
        // MUST get a distinct id — otherwise they reuse the stride-22/26 kernels of
        // the plain model while their buffers are the larger (lap-extended) stride,
        // misaligning every cell (gradients never land, the conserved state collapses).
        id: match (with_mms_sources, biharmonic, topology) {
            (false, false, cfd2_ir::equation::TopologyMode::Structured2D) => {
                "compressible_structured"
            }
            (true, true, _) => "compressible_mms_biharmonic",
            (true, false, _) => "compressible_mms",
            (false, true, _) => "compressible_biharmonic",
            (false, false, _) => "compressible",
        },
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
                m.relaxation_defaults = Some(crate::solver::model::module::RelaxationDefaults {
                    alpha_u: 1.0,
                    alpha_p: 1.0,
                });
                m
            },
        ],
        // The implicit biharmonic couples a 4th-order (condition ~ h^-4)
        // operator into the block. The stiffness is dominated by the INTRA-CELL
        // coupling (inv_dt-scaled recovery rows + the -I/+4 auxiliary-Laplacian
        // block); the per-cell block-Jacobi preconditioner (the 12-unknown stride
        // fits MAX_BLOCK_JACOBI=16) resolves it and converges in ~80 FGMRES
        // iters/step at n=48. Point-Jacobi stalls and needs thousands of iters,
        // so bump the budget over the inexact-Picard default of 200. Callers MUST
        // select BlockJacobi for this model; point-Jacobi is conditioning-limited
        // at fine mesh.
        linear_solver: if biharmonic {
            Some(crate::solver::model::linear_solver::ModelLinearSolverSpec {
                solver: crate::solver::model::linear_solver::ModelLinearSolverSettings {
                    max_iters: 1000,
                    ..Default::default()
                },
                ..Default::default()
            })
        } else {
            None
        },
        primitives,
        explicit_primitives: (!biharmonic).then_some(explicit_primitives),
        explicit_mass_closure_proof: super::ExplicitMassClosureProof::ExactSymbolic,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::model::backend::algebraic::{AlgExpr, ParamRef};
    use crate::solver::model::backend::ast::Equation;
    use crate::solver::model::backend::typed_ast::typed_fvc;
    use cfd2_ir::dimensions::UnitDimension;

    /// Golden reference: the hand-written pseudo-source recovery rows that the
    /// algebraic-equation lowering must reproduce bit-for-bit (same ops, fields,
    /// coefficient trees, order), which guarantees byte-identical generated WGSL.
    fn handwritten_recovery_equations() -> Vec<Equation> {
        let rho_typed = TypedFieldRef::<Density, Scalar>::new("rho");
        let rho_u_typed = TypedFieldRef::<MomentumDensity, Vector2>::new("rho_u");
        let rho_e_typed = TypedFieldRef::<EnergyDensity, Scalar>::new("rho_e");
        let u_typed = TypedFieldRef::<Velocity, Vector2>::new("u");
        let p_typed = TypedFieldRef::<Pressure, Scalar>::new("p");
        let t_typed = TypedFieldRef::<Temperature, Scalar>::new("T");

        // Raw coefficient atoms mirroring the lowering rule (sign, then
        // non-mag_sqr frozen factors in declaration order, then inv_dt, then
        // mag_sqr; the target term wraps its sign around the whole product;
        // multiple target-linear products merge into one summed coefficient).
        use crate::solver::model::backend::ast::{Coefficient, Discretization, Term, TermOp};
        use cfd2_ir::dimensions::InvTime as InvTimeDim;
        let minus_one = || Coefficient::constant(-1.0);
        let field = |name: &'static str, unit: cfd2_ir::units::UnitDim| {
            Coefficient::Field(crate::solver::model::backend::ast::FieldRef::new(
                name,
                cfd2_ir::equation::FieldKind::Scalar,
                unit,
            ))
        };
        let inv_dt = || field("inv_dt", InvTimeDim::UNIT);
        let rho_c = || field("rho", Density::UNIT);
        let gauge_rho = || field("eos_gauge_rho_ref", Density::UNIT);
        let gauge_p = || field("eos_gauge_p_ref", Pressure::UNIT);
        let gauge_p_bias = || field("eos_gauge_p_bias", Pressure::UNIT);
        let gm1 = || field("eos_gm1", Dimensionless::UNIT);
        let dp_drho = || {
            field(
                "eos_dp_drho",
                <cfd2_ir::dimensions::DivDim<Pressure, Density> as UnitDimension>::UNIT,
            )
        };
        let rho_ref = || field("eos_rho_ref", Density::UNIT);
        let r_gas = || {
            field(
                "eos_r",
                <cfd2_ir::dimensions::DivDim<Pressure, MulDim<Density, Temperature>> as UnitDimension>::UNIT,
            )
        };
        let half = || Coefficient::constant(0.5);
        let u2 = || Coefficient::MagSqr(u_typed.to_untyped());
        let prod = |a: Coefficient, b: Coefficient| Coefficient::Product(Box::new(a), Box::new(b));
        let sum = |a: Coefficient, b: Coefficient| Coefficient::Sum(Box::new(a), Box::new(b));
        // Velocity recovery: (rho + gauge_rho_ref) * u = rho_u.
        // Both target products carry the RHS sign (-1) wrapped whole.
        let mut u_eqn = Equation::new(u_typed.to_untyped());
        u_eqn.add_term(Term::new(
            TermOp::Source,
            Discretization::Implicit,
            u_typed.to_untyped(),
            None,
            Some(sum(
                prod(minus_one(), prod(rho_c(), inv_dt())),
                prod(minus_one(), prod(gauge_rho(), inv_dt())),
            )),
        ));
        u_eqn.add_term(Term::new(
            TermOp::Source,
            Discretization::Implicit,
            rho_u_typed.to_untyped(),
            None,
            Some(inv_dt()),
        ));

        // Pressure EOS (state form): p = gm1*rho_e - 0.5*gm1*|u|^2*(rho + G)
        //   + dp_drho*(rho + (G - rho_ref)) + gauge_p_bias.
        let mut p_eqn = Equation::new(p_typed.to_untyped());
        // Target term first: +p.
        p_eqn.add_term(Term::new(
            TermOp::Source,
            Discretization::Implicit,
            p_typed.to_untyped(),
            None,
            Some(inv_dt()),
        ));
        // -gm1*rho_e (implicit in rho_e).
        p_eqn.add_term(Term::new(
            TermOp::Source,
            Discretization::Implicit,
            rho_e_typed.to_untyped(),
            None,
            Some(prod(prod(minus_one(), gm1()), inv_dt())),
        ));
        // +0.5*gm1*|u|^2*rho (implicit in rho; u frozen through mag_sqr).
        p_eqn.add_term(Term::new(
            TermOp::Source,
            Discretization::Implicit,
            rho_typed.to_untyped(),
            None,
            Some(prod(prod(prod(half(), gm1()), inv_dt()), u2())),
        ));
        // -dp_drho*rho (implicit in rho).
        p_eqn.add_term(Term::new(
            TermOp::Source,
            Discretization::Implicit,
            rho_typed.to_untyped(),
            None,
            Some(prod(prod(minus_one(), dp_drho()), inv_dt())),
        ));
        // Explicit (unknown-free) tail, declaration order:
        // +0.5*gm1*|u|^2*gauge_rho_ref.
        p_eqn.add_term(Term::new(
            TermOp::Source,
            Discretization::Explicit,
            p_typed.to_untyped(),
            None,
            Some(prod(prod(prod(prod(half(), gm1()), gauge_rho()), inv_dt()), u2())),
        ));
        // -dp_drho*gauge_rho_ref.
        p_eqn.add_term(Term::new(
            TermOp::Source,
            Discretization::Explicit,
            p_typed.to_untyped(),
            None,
            Some(prod(prod(prod(minus_one(), dp_drho()), gauge_rho()), inv_dt())),
        ));
        // +dp_drho*rho_ref.
        p_eqn.add_term(Term::new(
            TermOp::Source,
            Discretization::Explicit,
            p_typed.to_untyped(),
            None,
            Some(prod(prod(dp_drho(), rho_ref()), inv_dt())),
        ));
        // -gauge_p_bias.
        p_eqn.add_term(Term::new(
            TermOp::Source,
            Discretization::Explicit,
            p_typed.to_untyped(),
            None,
            Some(prod(prod(minus_one(), gauge_p_bias()), inv_dt())),
        ));

        // Temperature recovery: (rho + G) * R * T = p + gauge_p_ref.
        let mut t_eqn = Equation::new(t_typed.to_untyped());
        t_eqn.add_term(Term::new(
            TermOp::Source,
            Discretization::Implicit,
            t_typed.to_untyped(),
            None,
            Some(sum(
                prod(prod(rho_c(), r_gas()), inv_dt()),
                prod(prod(gauge_rho(), r_gas()), inv_dt()),
            )),
        ));
        t_eqn.add_term(Term::new(
            TermOp::Source,
            Discretization::Implicit,
            p_typed.to_untyped(),
            None,
            Some(prod(minus_one(), inv_dt())),
        ));
        // -gauge_p_ref (explicit).
        t_eqn.add_term(Term::new(
            TermOp::Source,
            Discretization::Explicit,
            t_typed.to_untyped(),
            None,
            Some(prod(prod(minus_one(), gauge_p()), inv_dt())),
        ));

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
            (EOS_P_REF.name(), Pressure::UNIT),
            (EOS_RHO_REF.name(), Density::UNIT),
            (EOS_GAUGE_RHO_REF.name(), Density::UNIT),
            (EOS_GAUGE_P_REF.name(), Pressure::UNIT),
            ("eos_gauge_e_ref", Pressure::UNIT),
            (EOS_GAUGE_P_BIAS.name(), Pressure::UNIT),
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
    fn wave_speed_sq_declares_ideal_and_barotropic_contributions() {
        // cast_to inside the constructor already asserts the runtime unit is
        // Velocity^2; here we pin the declared structure.
        let expr = compressible_wave_speed_sq().to_untyped();
        let expected = AlgExpr::Add(
            Box::new(AlgExpr::Mul(
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
            )),
            Box::new(AlgExpr::Param(ParamRef::new(
                "eos_dp_drho",
                DivDim::<Pressure, Density>::UNIT,
            ))),
        );
        assert_eq!(expr, expected);
    }
}
