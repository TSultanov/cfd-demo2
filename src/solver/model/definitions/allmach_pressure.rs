//! All-Mach, f32, pressure-based solver model.
//!
//! This is the pressure-based counterpart to the density-based `compressible`
//! model. It is built as an *additive extension* of `incompressible_momentum`:
//! the velocity/pressure unknowns, the Rhie–Chow collocated coupling, the Schur
//! preconditioner and the coupled assembly are all reused verbatim. The only
//! physics added is **compressibility**, expressed as a single time term in the
//! continuity/pressure equation:
//!
//! ```text
//!   momentum :  ddt(rho,U) + div(phi,U) - lap(mu,U) + grad(p) + dev2(mu,(grad U)^T) = 0
//!   pressure :  ddt(psi,p) + lap(rho*d_p,p) + div_flux(phi,p)                        = 0
//! ```
//!
//! `psi = d(rho)/d(p) = 1/c^2` is the compressibility (units Density/Pressure).
//! The `ddt(psi,p)` term is the discrete `d(rho)/dt = (1/c^2) dp/dt` of the
//! continuity equation under a barotropic EOS `rho = rho_ref + psi*(p - p_ref)`.
//!
//! **All-Mach by construction:** as Mach -> 0 (`c -> inf`, `psi -> 0`,
//! `rho -> rho_ref`) every added term vanishes and the system reduces *exactly*
//! to `incompressible_momentum`, so the validated incompressible behaviour is the
//! `psi -> 0` limit. As `psi` grows, acoustic/compressible effects appear.
//!
//! **f32 by construction:** `p` is a gauge/perturbation pressure (O(rho*U^2),
//! centred at 0, pinned to 0 at the outlet — inherited from the incompressible
//! model), so `grad(p)` is resolved at full f32 precision. This is the property
//! the density-based solver lacks (it carries absolute thermodynamic pressure
//! ~1e5, burying the O(1e-4) wake signal below f32 epsilon).
//!
//! `psi` is a runtime state field (set uniformly to `1/c^2`); the density `rho`
//! is likewise a state field. For the first slice `rho` is held at `rho_ref`
//! (correct to O(Mach^2)); a `rho = rho_ref + psi*p` refresh kernel (true
//! barotropic density coupling) is layered on in a later slice.

use crate::solver::gpu::enums::GpuBoundaryType;
use crate::solver::model::backend::ast::{
    surface_scalar_dim, vol_scalar_dim, vol_vector_dim, EquationSystem, FieldRef, FluxRef,
};
use crate::solver::model::backend::typed_ast::{
    typed_fvc, typed_fvm, Scalar, TypedCoeff, TypedFieldRef, TypedFluxRef, Vector2,
};
use crate::solver::model::ports::PortRegistry;
use crate::solver::model::primitives::PrimitiveDerivations;
use cfd2_codegen::solver::codegen::dsl::XY;
use cfd2_ir::ast::Expr;
use cfd2_ir::dimensions::{
    Density, DivDim, DynamicViscosity, Force, InvTime, Length, MassFlux, MulDim, Pressure,
    Temperature, Time, Velocity, Volume,
};
use std::collections::HashMap;

use super::{BoundaryCondition, BoundarySpec, FieldBoundarySpec, ModelSpec};

/// Compressibility `psi = d(rho)/d(p) = 1/c^2`, units Density / Pressure.
type Compressibility = DivDim<Density, Pressure>;

/// Manufactured per-component momentum source (Vector2, Force/Volume).
pub const ALLMACH_MMS_SOURCE_U_FIELD: &str = "mms_src_U";
/// Manufactured continuity/pressure source (Scalar, MassFlux/Volume).
pub const ALLMACH_MMS_SOURCE_P_FIELD: &str = "mms_src_p";
/// Manufactured temperature source on the thermal `_mms` variant (Scalar).
pub const ALLMACH_MMS_SOURCE_T_FIELD: &str = "mms_src_T";

/// Solved temperature field of the thermal all-Mach variant.
pub const ALLMACH_TEMPERATURE_FIELD: &str = "T";
/// Constant `rho_ref * T_ref` field (unit Density*Temperature) of the thermal
/// variant. The EOS density is recovered on-device as
/// `rho = rho_t_ref / T + psi*p` (ideal-gas `rho = p_ref/(R*T)` written
/// f32-safely: the dominant thermal part `rho_t_ref/T` is full precision, the
/// acoustic `psi*p` is the small perturbation). At `T = T_ref` this reduces
/// exactly to the barotropic `rho = rho_ref + psi*p`. MUST be seeded to
/// `density * T_ref` (an unseeded 0 field makes `rho` blow up).
pub const ALLMACH_RHO_T_REF_FIELD: &str = "rho_t_ref";

/// Thermal-expansion coefficient `rho_dT = d(rho)/dT = -rho_t_ref/T^2` (unit
/// Density/Temperature, always negative: density falls as temperature rises).
/// Recovered on-device from `T` each outer iteration (alongside `rho`), it is the
/// coefficient of the `rho_dT * dT/dt` thermal-expansion term in the continuity/
/// pressure equation (the `dT/dt` half of `d(rho)/dt = psi*dp/dt + rho_dT*dT/dt`).
/// The term vanishes at steady state (`dT/dt -> 0`), so every steady validation is
/// unchanged; it is the acoustic-thermal coupling that matters for transient and
/// high-Mach compressible heating. Thermal variant only.
pub const ALLMACH_RHO_DT_FIELD: &str = "rho_dT";

/// EOS density floor (unit Density), seeded by the driver to `psi * ABS_PRESSURE_FLOOR`
/// — the density at the absolute-pressure floor, since `rho = psi * P_abs` for the
/// barotropic model. The on-device thermal recovery clamps `rho = max(rho_t_ref/T +
/// psi*p, rho_floor)` so a transient gauge-pressure undershoot below `-P_REF` (which
/// would drive `P_abs < 0` and hence `rho <= 0`, breaking the momentum/flux terms that
/// divide by density) is held at a small positive density instead of blowing up. A
/// constant per-cell field (never an equation target), seeded like `rho_t_ref`; the
/// non-thermal model applies the same floor on the host. Production thermal variant only
/// (the steady `_mms` variant never approaches vacuum, so it is omitted to stay
/// byte-identical). The clamp vanishes wherever the EOS is well-posed.
pub const ALLMACH_RHO_FLOOR_FIELD: &str = "rho_floor";

/// On-device pressure-advection field `u_dot_grad_p = U . grad_p` (Pressure/Time),
/// recovered from the velocity and the stored Rhie–Chow pressure gradient. It is
/// the `U.grad(p)` half of the compression-heating source `-(1/cp)*Dp/Dt`.
pub const ALLMACH_U_DOT_GRAD_P_FIELD: &str = "u_dot_grad_p";

/// Reference-temperature field (= [`ALLMACH_T_REF`], Temperature), seeded by the
/// driver. Present only to keep the `psi_real = gamma*psi*t_ref/T` and `rho_dT`
/// recoveries unit-clean (a Temperature factor the resolver can track).
pub const ALLMACH_T_REF_FIELD: &str = "t_ref";

/// Reference temperature of the thermal EOS linearization. The canonical value
/// the MMS manufactured solution and the GUI default are derived from; seeders
/// set `rho_t_ref = density * ALLMACH_T_REF`.
pub const ALLMACH_T_REF: f64 = 1.0;
/// Thermal conduction coefficient divided by specific heat (`k / cp`, i.e.
/// `rho * thermal_diffusivity`). Baked as a typed constant for now; promote to
/// a runtime uniform param when GUI tuning is needed.
pub const ALLMACH_K_OVER_CP: f64 = 1.0e-2;

/// Ratio of specific heats (diatomic / air). Sets the compression-heating
/// coefficient and the isentropic exponent (gamma-1)/gamma.
pub const ALLMACH_GAMMA: f64 = 1.4;

/// Unit of the temperature equation (declared divided by cp): Density*Temperature*Volume/Time.
type TEquationUnit = DivDim<MulDim<MulDim<Density, Temperature>, Volume>, Time>;
/// Unit of the manufactured temperature source: Density*Temperature/Time.
type TSourceUnit = DivDim<MulDim<Density, Temperature>, Time>;
/// Unit of the `k/cp` conduction coefficient (matches the buoyant model).
type KOverCpUnit = DivDim<MulDim<Density, Volume>, MulDim<Length, Time>>;
/// Unit of the constant `rho_ref * T_ref` recovery field: Density*Temperature.
type RhoTRefUnit = MulDim<Density, Temperature>;
/// Unit of the thermal-expansion coefficient `rho_dT = d(rho)/dT`: Density/Temperature.
type RhoDtUnit = DivDim<Density, Temperature>;
/// Unit of the on-device `u_dot_grad_p = U . grad_p` field: Velocity *
/// PressureGradient = (Length/Time)*(Pressure/Length) = Pressure/Time.
type PressureRateUnit = DivDim<Pressure, Time>;

#[derive(Debug, Clone)]
pub struct AllMachPressureFields {
    pub u: FieldRef,
    pub p: FieldRef,
    pub phi: FluxRef,
    pub mu: FieldRef,
    pub rho: FieldRef,
    pub d_p: FieldRef,
    pub grad_p: FieldRef,
    pub grad_p_old: FieldRef,
    /// Compressibility psi = 1/c^2 (runtime field; 0 => incompressible).
    pub psi: FieldRef,
    /// Preconditioned pseudo-compressibility used ONLY by the pressure-row `ddt`
    /// time term (the acoustic/time coupling). Decoupled from the physical `psi`
    /// (=1/c^2), which still drives every density recovery and the thermal
    /// compression-heating coefficient. The driver seeds it per-cell as
    /// `max(psi, 1/beta^2)` with `beta = max(|U|, k*U_inlet)` — a low-Mach
    /// preconditioner (Turkel) that rescales the pseudo sound speed toward the
    /// local velocity so a convective timestep is acoustically stable, while the
    /// real `psi` keeps the density near-incompressible. At steady state `dp/dt -> 0`
    /// so this term vanishes and the converged solution is the real-`psi` physics.
    /// Defaults equal to `psi` until the driver seeds it (see `allmach_psi_precond`).
    pub psi_precond: FieldRef,
    /// Per-cell local pseudo-time step for steady-state acceleration (Local Time
    /// Stepping). When 0 (default), the assembly falls back to the global
    /// `constants.dt` (time-accurate, byte-identical to a model without this
    /// field); when filled with per-cell stable steps (`cfl*h_c/lambda_c`) each
    /// cell advances toward steady state at its own maximal rate.
    pub dt_local: FieldRef,
}

impl AllMachPressureFields {
    pub fn new() -> Self {
        Self {
            u: vol_vector_dim::<Velocity>("U"),
            p: vol_scalar_dim::<Pressure>("p"),
            phi: surface_scalar_dim::<MassFlux>("phi"),
            mu: vol_scalar_dim::<DynamicViscosity>("mu"),
            rho: vol_scalar_dim::<Density>("rho"),
            d_p: vol_scalar_dim::<cfd2_ir::dimensions::D_P>("d_p"),
            grad_p: vol_vector_dim::<DivDim<Pressure, Length>>("grad_p"),
            grad_p_old: vol_vector_dim::<DivDim<Pressure, Length>>("grad_p_old"),
            psi: vol_scalar_dim::<Compressibility>("psi"),
            psi_precond: vol_scalar_dim::<Compressibility>("psi_precond"),
            dt_local: vol_scalar_dim::<Time>("dt_local"),
        }
    }
}

impl Default for AllMachPressureFields {
    fn default() -> Self {
        Self::new()
    }
}

/// Mirror of the incompressible momentum model's shipped viscous stress form
/// (`FullDev2`): laplacian(mu,U) plus the explicit transpose/deviatoric
/// correction. Kept identical so the incompressible limit (psi -> 0) is
/// byte-comparable to `incompressible_momentum`.
const USE_FULL_DEV2: bool = true;

fn build_allmach_system(
    _fields: &AllMachPressureFields,
    with_mms_source: bool,
    thermal: bool,
) -> EquationSystem {
    let u_typed = TypedFieldRef::<Velocity, Vector2>::new("U");
    let p_typed = TypedFieldRef::<Pressure, Scalar>::new("p");
    let phi_typed = TypedFluxRef::<MassFlux, Scalar>::new("phi");
    let rho_typed = TypedFieldRef::<Density, Scalar>::new("rho");
    let mu_typed = TypedFieldRef::<DynamicViscosity, Scalar>::new("mu");
    let d_p_typed = TypedFieldRef::<cfd2_ir::dimensions::D_P, Scalar>::new("d_p");
    let psi_precond_typed = TypedFieldRef::<Compressibility, Scalar>::new("psi_precond");

    let rho_coeff = TypedCoeff::from_field(rho_typed);
    let mu_coeff = TypedCoeff::from_field(mu_typed);
    // Pressure-Laplacian coefficient `rho_dp = rho * d_p` (the elliptic pressure
    // coupling). The transonic/supersonic robustness comes from the pressure-flux
    // Newton linearization attached to the `div_flux` term below (an implicit,
    // upwinded hyperbolic coupling that stays diagonally dominant as the gas
    // expands), not from the d_p formulation.
    let rho_dp_coeff =
        TypedCoeff::from_field(rho_typed).multiply(TypedCoeff::from_field(d_p_typed));
    // The pressure-row ddt uses the LOW-MACH PRECONDITIONED compressibility (not the
    // physical psi). The thermal compression-heating terms below build their own
    // physical-`psi` refs inline (real thermodynamics, not the pseudo time scale).
    let psi_precond_coeff = TypedCoeff::from_field(psi_precond_typed);

    // ----- Momentum equation (identical to incompressible_momentum) -----
    let ddt_term = typed_fvm::ddt_coeff(rho_coeff, u_typed);
    let div_term = typed_fvm::div(phi_typed, u_typed).bounded();
    let laplacian_term = typed_fvm::laplacian(mu_coeff, u_typed);
    let grad_term = typed_fvc::grad(p_typed);

    let mut momentum_sum = ddt_term.cast_to::<Force>()
        + div_term.cast_to::<Force>()
        + laplacian_term.cast_to::<Force>()
        + grad_term.cast_to::<Force>();
    if USE_FULL_DEV2 {
        let mu_coeff2 = TypedCoeff::from_field(mu_typed);
        momentum_sum = momentum_sum
            + typed_fvc::div_dev2_grad_transpose(mu_coeff2, u_typed).cast_to::<Force>();
    }
    if with_mms_source {
        let mms_src_typed = TypedFieldRef::<DivDim<Force, Volume>, Vector2>::new(
            ALLMACH_MMS_SOURCE_U_FIELD,
        );
        momentum_sum =
            momentum_sum + typed_fvc::source_vector(mms_src_typed, u_typed).cast_to::<Force>();
    }
    let momentum_eqn = momentum_sum.eqn(u_typed);

    // ----- Continuity/pressure equation: incompressible terms + ddt(psi_precond,p) -----
    // ddt(psi_precond,p) integrates to: psi_precond * p * Vol / Time =
    //   (Density/Pressure) * Pressure * Vol/Time = Density * Vol / Time = MassFlux. ✓
    // The coefficient is the LOW-MACH PRECONDITIONED compressibility (not the physical
    // psi): it sets only the pseudo-acoustic time scale and vanishes at steady state
    // (dp/dt -> 0), so the converged solution is the real-psi physics. The physical psi
    // still drives the density recovery (host) and the thermal compression heating below.
    let compressibility_term = typed_fvm::ddt_coeff(psi_precond_coeff, p_typed);
    let p_laplacian_term = typed_fvm::laplacian(rho_dp_coeff, p_typed);
    // The predicted mass-flux divergence `div(phi_pred)` is the explicit source
    // of the pressure equation. On its own it is elliptic-only (the implicit
    // p-coupling lives entirely in the Laplacian above), so at a SUPERSONIC
    // outlet — where the pressure is extrapolated and the Laplacian contributes
    // no constraint — the lagged `phi = rho_f * U_f` feedback runs the exit
    // density to vacuum. Attaching the deferred-correction Newton linearization
    // (Jacobian `d(div phi)/dp = psi * U.n * A`, upwinded) makes the pressure
    // row well-posed there without touching the converged solution: the implicit
    // damping and its frozen-state RHS correction cancel at convergence, so every
    // low-Mach and steady result is unchanged, while the transonic/supersonic
    // iteration stops diverging. Omitted on the `_mms` variant (byte-identical
    // steady order test; its pinned box never approaches the runaway) and when
    // `psi = 0` the term is identically zero (incompressible limit is unaffected).
    let p_div_flux_term = {
        let term = typed_fvm::div_flux(phi_typed, p_typed);
        if with_mms_source {
            term
        } else {
            let psi_lin = TypedCoeff::from_field(TypedFieldRef::<Compressibility, Scalar>::new(
                "psi",
            ));
            term.with_pressure_flux_linearization(psi_lin)
        }
    };

    let mut pressure_sum = compressibility_term.cast_to::<MassFlux>()
        + p_laplacian_term.cast_to::<MassFlux>()
        + p_div_flux_term.cast_to::<MassFlux>();
    // Thermal expansion in continuity: d(rho)/dt = psi*dp/dt + rho_dT*dT/dt.
    // `compressibility_term` is the psi*dp/dt half; this adds the rho_dT*dT/dt half
    // (a p<-T cross-coupling). rho_dT = d(rho)/dT = -rho_t_ref/T^2 < 0 is recovered
    // on-device. Units: ddt_coeff(rho_dT,T) = rho_dT*T*Vol/Time =
    // (Density/Temp)*Temp*Vol/Time = Density*Vol/Time = MassFlux. ✓
    // It is a TRANSIENT term (zero at steady state, dT/dt -> 0), so it is OMITTED
    // from the `_mms` variant: the steady manufactured-solution order test measures
    // spatial-operator accuracy, to which this term contributes nothing, and
    // including it only stalls the fully-pinned closed-box march. It is validated
    // for real (open) flows by the functional + transient tests. Production
    // (`allmach_thermal`) always carries it.
    if thermal && !with_mms_source {
        let t_typed_p = TypedFieldRef::<Temperature, Scalar>::new(ALLMACH_TEMPERATURE_FIELD);
        let rho_dt_coeff =
            TypedCoeff::from_field(TypedFieldRef::<RhoDtUnit, Scalar>::new(ALLMACH_RHO_DT_FIELD));
        let thermal_expansion_term = typed_fvm::ddt_coeff(rho_dt_coeff, t_typed_p);
        pressure_sum = pressure_sum + thermal_expansion_term.cast_to::<MassFlux>();
    }
    if with_mms_source {
        // Scalar continuity source: mms_src_p (MassFlux/Volume) * V = MassFlux.
        let mms_src_p_typed =
            TypedFieldRef::<DivDim<MassFlux, Volume>, Scalar>::new(ALLMACH_MMS_SOURCE_P_FIELD);
        let mms_src_p_coeff = TypedCoeff::from_field(mms_src_p_typed);
        pressure_sum =
            pressure_sum + typed_fvc::source_coeff(mms_src_p_coeff, p_typed).cast_to::<MassFlux>();
    }
    let pressure_eqn = pressure_sum.eqn(p_typed);

    let mut system = EquationSystem::new();
    system.add_equation(momentum_eqn);
    system.add_equation(pressure_eqn);

    // ----- Temperature transport (thermal variant only) -----
    // Low-Mach energy, declared divided by cp (so all terms share unit
    // Density*Temperature*Volume/Time): ddt(rho,T) + div(phi,T) - lap(k/cp,T).
    // T is advected by the SOLVED Rhie–Chow mass flux phi (same as buoyant),
    // and rho is the EOS density recovered from (p,T) each outer iteration.
    if thermal {
        let t_typed = TypedFieldRef::<Temperature, Scalar>::new(ALLMACH_TEMPERATURE_FIELD);
        let rho_coeff_t = TypedCoeff::from_field(rho_typed);
        let t_ddt = typed_fvm::ddt_coeff(rho_coeff_t, t_typed);
        // Production (compressible) uses BOUNDED convection so the energy is a clean
        // rho*DT/Dt: the .bounded() form subtracts T*div(phi) = +T*d(rho)/dt, which
        // cancels the conservative-form defect (ddt(rho,T)+div(phi,T) = rho*DT/Dt -
        // T*d(rho)/dt). The steady _mms variant keeps the validated conservative form
        // (compression terms are zero at steady state, so it stays byte-identical).
        let t_div = if with_mms_source {
            typed_fvm::div(phi_typed, t_typed)
        } else {
            typed_fvm::div(phi_typed, t_typed).bounded()
        };
        let k_over_cp: TypedCoeff<KOverCpUnit> = TypedCoeff::constant(ALLMACH_K_OVER_CP);
        let t_lap = typed_fvm::laplacian(k_over_cp, t_typed);

        let mut t_sum = t_ddt.cast_to::<TEquationUnit>()
            + t_div.cast_to::<TEquationUnit>()
            + t_lap.cast_to::<TEquationUnit>();

        // Compression heating (production only): the energy gains -(1/cp)*Dp/Dt so the
        // gas heats under compression (stagnation / weak-shock T-rise) — the enabling
        // physics for transonic flow. This first increment adds the dp/dt half as an
        // implicit T<-p cross-ddt (rides the cross-variable ddt path in
        // time_integration.rs). inv_cp = (gamma-1)*T_ref*psi keeps it consistent with
        // the EOS (psi = 1/c^2) so the isentropic relation T/T0 = (p/p0)^((g-1)/g)
        // emerges. SIGN (subtract) certified empirically by the uniform-fill test:
        // a positive dp/dt must drive dT/dt > 0.
        if !with_mms_source {
            // (T1) dp/dt half: implicit T<-p cross-ddt, coefficient -inv_cp.
            let inv_cp_const: TypedCoeff<Temperature> =
                TypedCoeff::constant(-(ALLMACH_GAMMA - 1.0) * ALLMACH_T_REF);
            let inv_cp = inv_cp_const.multiply(TypedCoeff::from_field(
                TypedFieldRef::<Compressibility, Scalar>::new("psi"),
            ));
            let comp_ddt = typed_fvm::ddt_coeff(inv_cp, p_typed);
            t_sum = t_sum + comp_ddt.cast_to::<TEquationUnit>();

            // (T2) U.grad(p) half: explicit source for the pressure-advection part of
            // -(1/cp)*Dp/Dt. u_dot_grad_p (=U.grad_p) is recovered on-device; using
            // the stored grad_p makes it Picard-lagged by one outer iteration
            // (vanishes at convergence). Coeff unit: Temperature * (Density/Pressure)
            // * (Pressure/Time) = Density*Temp/Time = TSourceUnit, so source_coeff(.,T)
            // integrates to TEquationUnit.
            //
            // SIGN: +inv_cp here, the OPPOSITE source-code sign to the T1 dp/dt half
            // (-inv_cp), even though both represent the same -(1/cp)Dp/Dt. The reason
            // is the codegen convention: an implicit cross-ddt nets a sign flip via its
            // matrix term (so ddt_coeff(-inv_cp,p) forces T by +inv_cp*dp/dt, heating),
            // whereas an explicit source goes straight to the RHS unflipped (so
            // source_coeff(c,T) forces T by +c). To get the SAME +inv_cp*U.grad(p)
            // heating we therefore need c=+inv_cp. Certified empirically (a wrong sign
            // cools under compression — verified to reduce, not add, the box T-rise).
            let inv_cp_src: TypedCoeff<Temperature> =
                TypedCoeff::constant((ALLMACH_GAMMA - 1.0) * ALLMACH_T_REF);
            let comp_adv_coeff = inv_cp_src
                .multiply(TypedCoeff::from_field(
                    TypedFieldRef::<Compressibility, Scalar>::new("psi"),
                ))
                .multiply(TypedCoeff::from_field(
                    TypedFieldRef::<PressureRateUnit, Scalar>::new(ALLMACH_U_DOT_GRAD_P_FIELD),
                ));
            t_sum =
                t_sum + typed_fvc::source_coeff(comp_adv_coeff, t_typed).cast_to::<TEquationUnit>();
        }

        if with_mms_source {
            let mms_src_t =
                TypedCoeff::from_field(TypedFieldRef::<TSourceUnit, Scalar>::new(
                    ALLMACH_MMS_SOURCE_T_FIELD,
                ));
            t_sum = t_sum + typed_fvc::source_coeff(mms_src_t, t_typed).cast_to::<TEquationUnit>();
        }
        system.add_equation(t_sum.eqn(t_typed));
    }

    system
        .validate_units()
        .expect("allmach_pressure system failed unit validation");

    system
}

pub fn allmach_pressure_system() -> EquationSystem {
    let fields = AllMachPressureFields::new();
    build_allmach_system(&fields, false, false)
}

pub fn allmach_pressure_model() -> Result<ModelSpec, String> {
    allmach_pressure_model_impl(false, false)
}

/// `allmach_pressure` plus manufactured momentum + continuity source fields for
/// MMS order tests.
pub fn allmach_pressure_mms_model() -> Result<ModelSpec, String> {
    allmach_pressure_model_impl(true, false)
}

/// Thermal all-Mach: `allmach_pressure` + a temperature transport equation and
/// an on-device ideal-gas density recovery `rho = rho_t_ref/T + psi*p` (declared
/// as a `PrimitiveDerivations` math expression, lowered to the coupled Update
/// kernel — no hand-written kernel). The barotropic model is the `T = T_ref`
/// limit.
pub fn allmach_thermal_model() -> Result<ModelSpec, String> {
    allmach_pressure_model_impl(false, true)
}

/// `allmach_thermal` plus manufactured momentum + continuity + temperature
/// source fields for MMS order tests.
pub fn allmach_thermal_mms_model() -> Result<ModelSpec, String> {
    allmach_pressure_model_impl(true, true)
}

/// In-place flip of an all-Mach model's Inlet/Outlet boundary KINDS to the
/// physically-correct CD-nozzle driving: a **pressure inlet** + a **supersonic
/// (fully-extrapolated) outlet**.
///
/// The shipping model pins the gauge at the OUTLET (`p` Dirichlet there) and drives
/// the flow with an inlet VELOCITY. This moves the single gauge anchor UPSTREAM and
/// lets the outlet float:
/// - `U` Inlet: Dirichlet(velocity) -> [ZeroGradient (axial speed develops with the
///   pressure drop), Dirichlet(0) (transverse pinned — blocks corner backflow)];
/// - `p` Inlet: ZeroGradient -> **Dirichlet** (the NEW gauge anchor; value set at
///   runtime via `set_boundary_scalar(Inlet, "p", inlet_pressure)`);
/// - `p` Outlet: Dirichlet(0) -> **ZeroGradient** (no back-pressure; extrapolate);
/// - `T` Outlet (thermal only): Dirichlet(T_ref) -> ZeroGradient (a supersonic outlet
///   must let the gas cool, not pin the reservoir temperature).
///
/// The pressure-Dirichlet face count is unchanged (one boundary's worth moves from
/// Outlet to Inlet), so the discrete pressure operator stays non-singular. BC kind is
/// a runtime `bc_table` (not baked into kernels), so the model id / committed kernels
/// are untouched — only the table differs. Validated stable + vacuum-free in
/// `tests/nozzle_pressure_inlet_probe.rs`; NB this driving is over-expanded in the
/// artificial-compressibility model (throat over-chokes, diverging section diffuses).
pub fn apply_pressure_inlet_nozzle_bcs(model: &mut ModelSpec) {
    use cfd2_ir::dimensions::{DivDim, InvTime, Length, Pressure, Temperature, Velocity};

    if let Some(u) = model.boundaries.fields.get_mut("U") {
        u.by_boundary.insert(
            GpuBoundaryType::Inlet,
            vec![
                BoundaryCondition::zero_gradient_dim::<InvTime>(), // U_x develops with the drop
                BoundaryCondition::dirichlet_dim::<Velocity>(0.0), // U_y pinned axial
            ],
        );
    }
    if let Some(p) = model.boundaries.fields.get_mut("p") {
        p.by_boundary.insert(
            GpuBoundaryType::Inlet,
            vec![BoundaryCondition::dirichlet_dim::<Pressure>(0.0)],
        );
        p.by_boundary.insert(
            GpuBoundaryType::Outlet,
            vec![BoundaryCondition::zero_gradient_dim::<DivDim<Pressure, Length>>()],
        );
    }
    if let Some(t) = model.boundaries.fields.get_mut(ALLMACH_TEMPERATURE_FIELD) {
        t.by_boundary.insert(
            GpuBoundaryType::Outlet,
            vec![BoundaryCondition::zero_gradient_dim::<DivDim<Temperature, Length>>()],
        );
    }
}

fn allmach_pressure_model_impl(with_mms_source: bool, thermal: bool) -> Result<ModelSpec, String> {
    let fields = AllMachPressureFields::new();
    let system = build_allmach_system(&fields, with_mms_source, thermal);

    // Keep U,p,d_p,grad_p,grad_p_old at the same offsets as incompressible
    // (0,2,3,4,6); append psi (and any MMS sources) after.
    let mut layout_fields = vec![
        fields.u,
        fields.p,
        fields.d_p,
        fields.grad_p,
        fields.grad_p_old,
        fields.psi,
        fields.psi_precond,
        fields.dt_local,
        // Variable density: when `rho` is a state-layout field, the Rhie–Chow flux
        // deriver and the ddt(rho,U) coefficient read it per-cell (instead of the
        // uniform `constants.density`), giving a genuinely compressible continuity
        // `div(rho*U)=0`. Must be initialised to rho_ref and refreshed from the EOS
        // (`rho = rho_ref + psi*p`) each outer iteration.
        fields.rho,
    ];
    if thermal {
        // Solved temperature, plus the constant `rho_ref * T_ref` recovery
        // field (seeded to `density * T_ref`; never an equation target).
        layout_fields.push(vol_scalar_dim::<Temperature>(ALLMACH_TEMPERATURE_FIELD));
        layout_fields.push(vol_scalar_dim::<RhoTRefUnit>(ALLMACH_RHO_T_REF_FIELD));
        // Thermal-expansion coefficient d(rho)/dT, recovered on-device from T.
        // Only present where the term is (production, not the steady `_mms` variant).
        if !with_mms_source {
            layout_fields.push(vol_scalar_dim::<RhoDtUnit>(ALLMACH_RHO_DT_FIELD));
            // U.grad(p), recovered on-device for the compression-heating source.
            layout_fields.push(vol_scalar_dim::<PressureRateUnit>(ALLMACH_U_DOT_GRAD_P_FIELD));
            // EOS density floor (= psi * absolute-pressure floor), seeded by the driver.
            // Clamps the on-device density recovery positive against transient pressure
            // undershoot through vacuum (see ALLMACH_RHO_FLOOR_FIELD).
            layout_fields.push(vol_scalar_dim::<Density>(ALLMACH_RHO_FLOOR_FIELD));
            // Reference-temperature field (= T_REF), seeded by the driver. The density
            // recovery uses the REAL T-VARYING ideal-gas compressibility
            // `d(rho)/d(p)|_T = gamma*psi*T_ref/T` (= 1/(R*T) at the effective scale): as
            // the gas cools (accelerates), the compressibility rises (c falls), so the
            // continuity stiffens correctly — the coupling the constant-psi model lacked.
            // `t_ref` is the Temperature factor that keeps that recovery unit-clean.
            layout_fields.push(vol_scalar_dim::<Temperature>(ALLMACH_T_REF_FIELD));
        }
    }
    if with_mms_source {
        layout_fields.push(vol_vector_dim::<DivDim<Force, Volume>>(
            ALLMACH_MMS_SOURCE_U_FIELD,
        ));
        layout_fields.push(vol_scalar_dim::<DivDim<MassFlux, Volume>>(
            ALLMACH_MMS_SOURCE_P_FIELD,
        ));
        if thermal {
            layout_fields.push(vol_scalar_dim::<TSourceUnit>(ALLMACH_MMS_SOURCE_T_FIELD));
        }
    }
    let layout = PortRegistry::from_fields(layout_fields).into_state_layout();

    // Historical ClosedForm d_p (= α_u·dt/ρ_ref): preserves the manufactured-solution
    // order (the SIMPLEC `FromAssembledRowSum` row-sum d_p was tried for the supersonic
    // runaway but degraded the coupled MMS u-order to ~1.6 < 1.65 — the runaway is now
    // cured structurally by the pressure-flux Newton linearization instead, which does
    // not touch the spatial operator / MMS order).
    let derived_rhie_chow =
        crate::solver::model::flux_derivation::derive_rhie_chow(&system, &layout)
            .map_err(|e| format!("failed to derive Rhie–Chow flux: {e}"))?;

    let (u0, u1, p) = {
        use crate::solver::model::ports::{PortRegistry, Pressure as PortPressure, Velocity as PortVelocity};

        let mut registry = PortRegistry::new(layout.clone());
        registry
            .validate_vector2_field::<PortVelocity>("allmach_pressure_model", "U")
            .map_err(|e| format!("state layout validation failed: {e}"))?;
        registry
            .validate_scalar_field::<PortPressure>("allmach_pressure_model", "p")
            .map_err(|e| format!("state layout validation failed: {e}"))?;

        let u_port = registry
            .register_vector2_field::<PortVelocity>("U")
            .map_err(|e| format!("U field registration failed: {e}"))?;
        let p_port = registry
            .register_scalar_field::<PortPressure>("p")
            .map_err(|e| format!("p field registration failed: {e}"))?;

        let u0 = u_port
            .component(XY::X.to_usize() as u32)
            .ok_or_else(|| "U component x not found".to_string())?
            .full_offset();
        let u1 = u_port
            .component(XY::Y.to_usize() as u32)
            .ok_or_else(|| "U component y not found".to_string())?
            .full_offset();
        let p = p_port.offset();
        (u0, u1, p)
    };

    // Schur velocity-block indices are COUPLED-SYSTEM RANKS, not state offsets.
    // For U_x,U_y,p the rank equals the state offset (first fields), so the
    // barotropic case reuses (u0,u1). The thermal variant adds T to the u-block
    // (like buoyant_incompressible) at T's coupled rank (offset_for = rank).
    let schur_u: Vec<u32> = if thermal {
        let fl = crate::solver::model::FluxLayout::from_system(&system);
        let t_rank = fl
            .offset_for(ALLMACH_TEMPERATURE_FIELD)
            .ok_or_else(|| "T not found in coupled layout".to_string())?;
        vec![u0, u1, t_rank]
    } else {
        vec![u0, u1]
    };

    let mut boundaries = BoundarySpec::default();
    boundaries.set_field(
        "U",
        FieldBoundarySpec::new()
            .set_uniform(
                GpuBoundaryType::Inlet,
                2,
                BoundaryCondition::dirichlet_dim::<Velocity>(0.0),
            )
            .set_uniform(
                GpuBoundaryType::Outlet,
                2,
                BoundaryCondition::zero_gradient_dim::<InvTime>(),
            )
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
        "p",
        FieldBoundarySpec::new()
            .set_uniform(
                GpuBoundaryType::Inlet,
                1,
                BoundaryCondition::zero_gradient_dim::<DivDim<Pressure, Length>>(),
            )
            .set_uniform(
                GpuBoundaryType::Wall,
                1,
                BoundaryCondition::zero_gradient_dim::<DivDim<Pressure, Length>>(),
            )
            .set_uniform(
                GpuBoundaryType::SlipWall,
                1,
                BoundaryCondition::zero_gradient_dim::<DivDim<Pressure, Length>>(),
            )
            .set_uniform(
                GpuBoundaryType::MovingWall,
                1,
                BoundaryCondition::zero_gradient_dim::<DivDim<Pressure, Length>>(),
            )
            // Outlet: fix gauge by pinning pressure to 0.
            .set_uniform(
                GpuBoundaryType::Outlet,
                1,
                BoundaryCondition::dirichlet_dim::<Pressure>(0.0),
            ),
    );
    if thermal {
        // T defaults: Inlet/Outlet/MovingWall isothermal (Dirichlet, values set
        // per-face by the seeder/MMS), Wall/SlipWall adiabatic (zero-gradient).
        let t_ref = ALLMACH_T_REF;
        boundaries.set_field(
            ALLMACH_TEMPERATURE_FIELD,
            FieldBoundarySpec::new()
                .set_uniform(
                    GpuBoundaryType::Inlet,
                    1,
                    BoundaryCondition::dirichlet_dim::<Temperature>(t_ref),
                )
                .set_uniform(
                    GpuBoundaryType::Outlet,
                    1,
                    BoundaryCondition::dirichlet_dim::<Temperature>(t_ref),
                )
                .set_uniform(
                    GpuBoundaryType::Wall,
                    1,
                    BoundaryCondition::zero_gradient_dim::<DivDim<Temperature, Length>>(),
                )
                .set_uniform(
                    GpuBoundaryType::SlipWall,
                    1,
                    BoundaryCondition::zero_gradient_dim::<DivDim<Temperature, Length>>(),
                )
                .set_uniform(
                    GpuBoundaryType::MovingWall,
                    1,
                    BoundaryCondition::dirichlet_dim::<Temperature>(t_ref),
                ),
        );
    }

    let method = crate::solver::model::method::MethodSpec::Coupled(
        crate::solver::model::method::CoupledCapabilities {
            apply_relaxation_in_update: true,
            relaxation_requires_dtau: false,
            requires_flux_module: true,
            gradient_storage: crate::solver::model::gpu_spec::GradientStorage::PackedState,
        },
    );
    let flux_module = crate::solver::model::flux_module::FluxModuleSpec::Kernel {
        gradients: Some(crate::solver::model::flux_module::FluxModuleGradientsSpec::FromStateLayout),
        kernel: derived_rhie_chow.flux_kernel,
    };
    // Thermal variant: recover the EOS density on-device from the solved (p,T)
    // every outer iteration, declared as a math expression (lowered to the
    // coupled Update kernel by the codegen — see PrimitiveDerivations). Units
    // stay consistent for the unit-checked resolver: rho_t_ref (Density*Temp) / T
    // (Temp) = Density, psi (Density/Pressure) * p (Pressure) = Density. The
    // barotropic model keeps identity primitives (host-side refresh).
    let primitives = if thermal {
        let mut derivations = HashMap::new();
        // EOS density recovery rho = rho_t_ref/T + (compressibility)*p. The pressure-
        // compressibility term differs by variant:
        //  - MMS (`with_mms_source`): the CONSTANT `psi*p` — the steady order test
        //    isolates the 1/T density and never activates the real-EOS path, so this
        //    keeps the validated barotropic-in-p form.
        //  - PRODUCTION: the REAL T-VARYING ideal-gas compressibility
        //    d(rho)/d(p)|_T = gamma*psi*t_ref/T, i.e. the term is gamma*psi*t_ref*p/T
        //    (= p/(R*T) at the effective scale). This is what makes the solver a genuine
        //    pressure-based COMPRESSIBLE solver: density now responds to pressure through
        //    the local temperature, so as the gas accelerates and cools the compressibility
        //    rises (sound speed falls) and the supersonic area-Mach coupling can form.
        // Units stay clean: gamma (dimensionless) * psi (Density/Pressure) * t_ref (Temp)
        // * p (Pressure) / T (Temp) = Density, matching rho_t_ref/T (Density).
        // In production, clamp at `rho_floor` (= psi * absolute-pressure floor) so a
        // transient gauge undershoot below -P_REF cannot drive rho non-positive.
        let p_compr = if with_mms_source {
            Expr::ident("psi") * Expr::ident("p")
        } else {
            Expr::lit_f32(ALLMACH_GAMMA as f32)
                * Expr::ident("psi")
                * Expr::ident(ALLMACH_T_REF_FIELD)
                * Expr::ident("p")
                / Expr::ident(ALLMACH_TEMPERATURE_FIELD)
        };
        let rho_recovery = Expr::ident(ALLMACH_RHO_T_REF_FIELD)
            / Expr::ident(ALLMACH_TEMPERATURE_FIELD)
            + p_compr;
        let rho_recovery = if with_mms_source {
            rho_recovery
        } else {
            Expr::call_named("max", vec![rho_recovery, Expr::ident(ALLMACH_RHO_FLOOR_FIELD)])
        };
        derivations.insert("rho".to_string(), rho_recovery);
        // Thermal-expansion coefficient rho_dT = d(rho)/dT, recovered on-device. With the
        // real T-varying compressibility, rho = (rho_t_ref + gamma*psi*t_ref*p)/T, so
        // rho_dT = -(rho_t_ref + gamma*psi*t_ref*p)/T^2 (= -rho/T). Units: (Density*Temp)/
        // T^2 = Density/Temperature; the Add is Density*Temp + Density*Temp (no panic).
        if !with_mms_source {
            let rho_numer = Expr::ident(ALLMACH_RHO_T_REF_FIELD)
                + Expr::lit_f32(ALLMACH_GAMMA as f32)
                    * Expr::ident("psi")
                    * Expr::ident(ALLMACH_T_REF_FIELD)
                    * Expr::ident("p");
            derivations.insert(
                ALLMACH_RHO_DT_FIELD.to_string(),
                -(rho_numer
                    / (Expr::ident(ALLMACH_TEMPERATURE_FIELD)
                        * Expr::ident(ALLMACH_TEMPERATURE_FIELD))),
            );
            // u_dot_grad_p = U . grad_p (vector-component access via the `_x`/`_y`
            // suffixes the recovery resolver understands). Resolver units:
            // Velocity * (Pressure/Length) = Pressure/Time per term, Add of equal
            // units -> no panic. This is the U.grad(p) half of -(1/cp)*Dp/Dt.
            derivations.insert(
                ALLMACH_U_DOT_GRAD_P_FIELD.to_string(),
                Expr::ident("U_x") * Expr::ident("grad_p_x")
                    + Expr::ident("U_y") * Expr::ident("grad_p_y"),
            );
        }
        PrimitiveDerivations { derivations }
    } else {
        PrimitiveDerivations::identity()
    };

    let layout_for_flux = layout.clone();
    let flux_module_module = crate::solver::model::modules::flux_module::flux_module_module(
        flux_module,
        &system,
        &layout_for_flux,
        &primitives,
    )
    .map_err(|e| format!("failed to build flux_module module: {e}"))?;

    Ok(ModelSpec {
        id: match (thermal, with_mms_source) {
            (true, true) => "allmach_thermal_mms",
            (true, false) => "allmach_thermal",
            (false, true) => "allmach_pressure_mms",
            (false, false) => "allmach_pressure",
        },
        system,
        state_layout: layout,
        boundaries,
        modules: vec![
            crate::solver::model::modules::eos::eos_module(
                crate::solver::model::eos::EosSpec::Constant,
            ),
            flux_module_module,
            crate::solver::model::modules::generic_coupled::generic_coupled_module(method),
            derived_rhie_chow.aux_module,
        ],
        linear_solver: Some(crate::solver::model::linear_solver::ModelLinearSolverSpec {
            preconditioner: crate::solver::model::linear_solver::ModelPreconditionerSpec::Schur {
                omega: 1.0,
                layout: crate::solver::model::linear_solver::SchurBlockLayout::from_u_p(
                    &schur_u,
                    p,
                )
                .map_err(|e| format!("invalid SchurBlockLayout: {e}"))?,
            },
            ..Default::default()
        }),
        primitives,
    })
}
