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
//! centred at 0, pinned to 0 at the outlet), so `grad(p)` is resolved at full f32
//! precision — a density-based solver carrying absolute pressure ~1e5 buries the
//! O(1e-4) wake signal below f32 epsilon.
//!
//! `psi` and the density `rho` are runtime state fields. `rho` is initialised to
//! `rho_ref` and refreshed from the EOS (`rho = rho_ref + psi*p`) each outer
//! iteration.

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

/// Reference compressibility `psi_ref = 1/(gamma*R*T_ref) = 1/c_ref^2` (unit
/// Compressibility), seeded by the driver to `params.compressibility_psi`. This is
/// the CONSTANT reference value the ideal-gas EOS coefficients read: the density
/// recovery `rho = rho_t_ref/T + gamma*psi_ref*t_ref*p/T` (= `p/(R*T)`), the
/// thermal-expansion `rho_dT`, and the compression-heating `1/cp =
/// (gamma-1)*T_ref*psi_ref`. It is DECOUPLED from the `psi` field, which carries the
/// LOCAL `1/c^2(T)` consumed by the low-Mach preconditioner and the Mach diagnostics
/// (see the `psi = psi_ref*t_ref/T` recovery). At the reference temperature
/// `T = T_ref` the two coincide (`psi == psi_ref`), so a model seeded with a uniform
/// `psi = psi_ref` is byte-unchanged. Thermal production variant only.
pub const ALLMACH_PSI_REF_FIELD: &str = "psi_ref";

/// Reference inlet-scale velocity `u_ref = k * U_inlet` (unit Velocity), seeded by
/// the driver (`k` = the low-Mach Turkel floor multiplier). It sets the
/// pseudo-acoustic floor `beta^2 = max(|U|^2, u_ref^2)` in the on-device
/// `psi_precond` recovery so the pseudo-Mach stays <= 1 in quiescent regions.
/// Thermal production variant only.
pub const ALLMACH_U_REF_FIELD: &str = "u_ref";

/// Low-Mach preconditioner enable mask (dimensionless 0/1), seeded by the driver to
/// 1 when compressibility is on (`psi_ref > 0`) and 0 otherwise. It multiplies the
/// on-device `psi_precond`, so the incompressible limit (`psi_ref = 0`) forces
/// `psi_precond = 0` exactly (the `ddt(psi_precond,p)` acoustic term vanishes),
/// reproducing the host `allmach_psi_precond`'s `real_psi <= 0 => 0` special case.
/// Thermal production variant only.
pub const ALLMACH_PRECOND_MASK_FIELD: &str = "precond_mask";

/// Reference temperature of the thermal EOS linearization. The canonical value
/// the MMS manufactured solution and the GUI default are derived from; seeders
/// set `rho_t_ref = density * ALLMACH_T_REF`.
pub const ALLMACH_T_REF: f64 = 1.0;
/// Thermal conduction coefficient divided by specific heat (`k / cp`, i.e.
/// `rho * thermal_diffusivity`).
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

/// Full viscous stress (`FullDev2`): laplacian(mu,U) plus the explicit
/// transpose/deviatoric correction. Kept identical to `incompressible_momentum`
/// so the incompressible limit (psi -> 0) is byte-comparable.
const USE_FULL_DEV2: bool = true;

fn build_allmach_system(
    _fields: &AllMachPressureFields,
    with_mms_source: bool,
    compressible_mms: bool,
    thermal: bool,
    ale: bool,
) -> EquationSystem {
    // Two manufactured-solution modes:
    //  - BAROTROPIC mms (`with_mms_source && !compressible_mms`): strips the
    //    compressible physics to the barotropic `psi*p` density and isolates the 1/T
    //    variation (`strip_all == true`).
    //  - COMPRESSIBLE mms (`with_mms_source && compressible_mms`): keeps the FULL
    //    production physics (real-EOS `rho(p,T)`, U.grad(p) heating, viscous
    //    dissipation) and ADDS manufactured sources on top, so the compressible
    //    spatial operator is order-verified. The genuinely TRANSIENT terms it carries
    //    (dp/dt heating, thermal expansion, acoustic ddt) are identically zero at the
    //    steady manufactured solution, so they don't affect the manufactured source or
    //    the observed order — they only shape the march to steady state.
    let strip_all = with_mms_source && !compressible_mms;
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
    // coupling). Transonic/supersonic robustness comes from the pressure-flux
    // Newton linearization on the `div_flux` term below, not from the d_p form.
    let rho_dp_coeff =
        TypedCoeff::from_field(rho_typed).multiply(TypedCoeff::from_field(d_p_typed));
    // The pressure-row ddt uses the LOW-MACH PRECONDITIONED compressibility (not the
    // physical psi). The thermal compression-heating terms below build their own
    // physical-`psi` refs inline (real thermodynamics, not the pseudo time scale).
    let psi_precond_coeff = TypedCoeff::from_field(psi_precond_typed);

    // Momentum equation (identical to incompressible_momentum).
    let ddt_term = typed_fvm::ddt_coeff(rho_coeff, u_typed);
    // On a moving mesh the momentum convection consumes the mesh-relative mass
    // flux `phi - rho_f * mesh_fluxes[face]` (variable-density `rho_f`; see
    // `Term::relative_to_mesh` and unified_assembly `ale_relative_flux_expr`).
    let div_term = {
        let d = typed_fvm::div(phi_typed, u_typed).bounded();
        if ale {
            d.with_mesh_relative()
        } else {
            d
        }
    };
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

    // Continuity/pressure equation: incompressible terms + ddt(psi_precond,p).
    // ddt(psi_precond,p) integrates to: psi_precond * p * Vol / Time =
    //   (Density/Pressure) * Pressure * Vol/Time = Density * Vol / Time = MassFlux.
    // The coefficient is the LOW-MACH PRECONDITIONED compressibility (not the physical
    // psi): it sets only the pseudo-acoustic time scale and vanishes at steady state
    // (dp/dt -> 0), so the converged solution is the real-psi physics. The physical psi
    // still drives the density recovery (host) and the thermal compression heating below.
    let compressibility_term = typed_fvm::ddt_coeff(psi_precond_coeff, p_typed);
    let p_laplacian_term = typed_fvm::laplacian(rho_dp_coeff, p_typed);
    // The predicted mass-flux divergence `div(phi_pred)` is the explicit source of
    // the pressure equation, elliptic-only on its own. At a SUPERSONIC outlet (p
    // extrapolated, Laplacian gives no constraint) the lagged `phi = rho_f * U_f`
    // feedback would run the exit density to vacuum. The deferred-correction Newton
    // linearization (Jacobian `d(div phi)/dp = psi * U.n * A`, upwinded) makes the
    // pressure row well-posed there: its implicit damping and frozen-state RHS
    // correction cancel at convergence, so every low-Mach/steady result is
    // unchanged. Omitted on the `_mms` variant; identically zero when `psi = 0`.
    let p_div_flux_term = {
        let term = typed_fvm::div_flux(phi_typed, p_typed);
        // The deferred-Newton pressure-flux linearization is a CONVERGENCE aid for the
        // supersonic outlet; it cancels at convergence and is NOT needed for accuracy.
        // Omit it on EVERY manufactured-solution path (barotropic AND compressible mms):
        // gated on `with_mms_source`, not `strip_all`, so the compressible mms verifies
        // the real spatial operator without the linearization perturbing the steady
        // pressure row (which otherwise strands the manufactured velocity).
        let term = if with_mms_source {
            term
        } else {
            let psi_lin = TypedCoeff::from_field(TypedFieldRef::<Compressibility, Scalar>::new(
                "psi",
            ));
            term.with_pressure_flux_linearization(psi_lin)
        };
        // Continuity on the moving mesh is mesh-relative too: the compensating
        // volume-change source `+rho_ref*(V^{n+1}-V^n)/dt` (barotropic split;
        // exact by SCL construction) lands with the moving-volume ddt.
        if ale {
            term.with_mesh_relative()
        } else {
            term
        }
    };

    let mut pressure_sum = compressibility_term.cast_to::<MassFlux>()
        + p_laplacian_term.cast_to::<MassFlux>()
        + p_div_flux_term.cast_to::<MassFlux>();
    // Thermal expansion in continuity: d(rho)/dt = psi*dp/dt + rho_dT*dT/dt.
    // `compressibility_term` is the psi*dp/dt half; this adds the rho_dT*dT/dt half
    // (a p<-T cross-coupling). rho_dT = d(rho)/dT = -rho_t_ref/T^2 < 0 is recovered
    // on-device. Units: ddt_coeff(rho_dT,T) = rho_dT*T*Vol/Time = MassFlux.
    // TRANSIENT term (zero at steady state), so OMITTED from the `_mms` steady
    // order test (contributes nothing to spatial-operator accuracy, only stalls the
    // pinned closed-box march); production (`allmach_thermal`) always carries it.
    if thermal && !strip_all {
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

    // Temperature transport (thermal variant only). Low-Mach energy, declared
    // divided by cp (all terms share unit Density*Temperature*Volume/Time):
    // ddt(rho,T) + div(phi,T) - lap(k/cp,T). T is advected by the SOLVED Rhie–Chow
    // mass flux phi; rho is the EOS density recovered from (p,T) each outer iteration.
    if thermal {
        let t_typed = TypedFieldRef::<Temperature, Scalar>::new(ALLMACH_TEMPERATURE_FIELD);
        let rho_coeff_t = TypedCoeff::from_field(rho_typed);
        let t_ddt = typed_fvm::ddt_coeff(rho_coeff_t, t_typed);
        // Production uses BOUNDED convection for a clean rho*DT/Dt: .bounded()
        // subtracts T*div(phi) = +T*d(rho)/dt, cancelling the conservative-form
        // defect (ddt(rho,T)+div(phi,T) = rho*DT/Dt - T*d(rho)/dt). The `_mms`
        // variant keeps the conservative form (compression terms zero at steady state).
        let t_div = {
            // Conservative div(phi,T) on EVERY manufactured-solution path (gated on
            // with_mms_source, not strip_all). The production BOUNDED form subtracts
            // T*div(phi); for a manufactured flow with a forced div(phi)=div(m*)!=0 that
            // correction is O(T*·div(m*)), which is significant because T is O(1) (unlike
            // the O(0.1) velocity, where the analogous bounded momentum correction is
            // negligible) — it would leave a constant T error. Conservative div(m*T*) is
            // what the FD manufactured source expresses and matches the barotropic mms.
            let d = if with_mms_source {
                typed_fvm::div(phi_typed, t_typed)
            } else {
                typed_fvm::div(phi_typed, t_typed).bounded()
            };
            // Temperature is advected by the SOLVED mass flux, so on a moving
            // mesh it consumes the mesh-relative flux like the momentum/pressure
            // convection.
            if ale {
                d.with_mesh_relative()
            } else {
                d
            }
        };
        let k_over_cp: TypedCoeff<KOverCpUnit> = TypedCoeff::constant(ALLMACH_K_OVER_CP);
        let t_lap = typed_fvm::laplacian(k_over_cp, t_typed);

        let mut t_sum = t_ddt.cast_to::<TEquationUnit>()
            + t_div.cast_to::<TEquationUnit>()
            + t_lap.cast_to::<TEquationUnit>();

        // Compression heating (production only): the energy gains -(1/cp)*Dp/Dt so
        // the gas heats under compression. Two halves: (T1) the implicit dp/dt
        // cross-ddt, (T2) the explicit U.grad(p) source. inv_cp = (gamma-1)*T_ref*psi
        // keeps it consistent with the EOS (psi = 1/c^2) so the isentropic relation
        // T/T0 = (p/p0)^((g-1)/g) emerges. SIGN certified empirically: a positive
        // dp/dt must drive dT/dt > 0.
        if !strip_all {
            // (T1) dp/dt half: implicit T<-p cross-ddt, coefficient -inv_cp.
            let inv_cp_const: TypedCoeff<Temperature> =
                TypedCoeff::constant(-(ALLMACH_GAMMA - 1.0) * ALLMACH_T_REF);
            // 1/cp = (gamma-1)*T_ref*psi_ref uses the REFERENCE compressibility so cp
            // stays constant (an ideal gas) as `psi` becomes the local 1/c^2.
            let inv_cp = inv_cp_const.multiply(TypedCoeff::from_field(
                TypedFieldRef::<Compressibility, Scalar>::new(ALLMACH_PSI_REF_FIELD),
            ));
            let comp_ddt = typed_fvm::ddt_coeff(inv_cp, p_typed);
            t_sum = t_sum + comp_ddt.cast_to::<TEquationUnit>();

            // (T2) U.grad(p) half: explicit source for the pressure-advection part of
            // -(1/cp)*Dp/Dt. u_dot_grad_p (=U.grad_p) is recovered on-device from the
            // stored grad_p, so it is Picard-lagged one outer iteration (vanishes at
            // convergence). Coeff unit: Temperature * (Density/Pressure) *
            // (Pressure/Time) = TSourceUnit.
            //
            // SIGN: +inv_cp here, OPPOSITE to the T1 half (-inv_cp), though both are
            // the same -(1/cp)Dp/Dt. Codegen convention: an implicit cross-ddt nets a
            // sign flip via its matrix term (ddt_coeff(-inv_cp,p) forces T by
            // +inv_cp*dp/dt), whereas an explicit source hits the RHS unflipped
            // (source_coeff(c,T) forces T by +c); so the same +inv_cp*U.grad(p) heating
            // needs c=+inv_cp. Certified empirically (wrong sign cools under compression).
            let inv_cp_src: TypedCoeff<Temperature> =
                TypedCoeff::constant((ALLMACH_GAMMA - 1.0) * ALLMACH_T_REF);
            let comp_adv_coeff = inv_cp_src
                .multiply(TypedCoeff::from_field(
                    TypedFieldRef::<Compressibility, Scalar>::new(ALLMACH_PSI_REF_FIELD),
                ))
                .multiply(TypedCoeff::from_field(
                    TypedFieldRef::<PressureRateUnit, Scalar>::new(ALLMACH_U_DOT_GRAD_P_FIELD),
                ));
            t_sum =
                t_sum + typed_fvc::source_coeff(comp_adv_coeff, t_typed).cast_to::<TEquationUnit>();

            // Viscous dissipation Phi = tau:grad(U): the deviatoric strain-rate
            // contraction that heats the gas under shear (always >= 0, so it can
            // only raise T). Declared on the energy row reading U's gradient tensor
            // from grad_state (the same buffer the momentum dev2 term consumes).
            // Coefficient = 1/cp * mu = (gamma-1)*T_ref*psi_ref * mu folds in the
            // constant 1/cp (reference psi, matching the compression-heating terms);
            // the codegen supplies Phi_grad = 2[(du/dx)^2 + (dv/dy)^2 +
            // 0.5(du/dy+dv/dx)^2 - (1/3)(div U)^2] (unit Velocity^2/Length^2), so the
            // term integrates to TEquationUnit. STEADY-active (nonzero at steady
            // state), unlike the transient compression/thermal-expansion terms.
            let visc_diss_coeff =
                TypedCoeff::<Temperature>::constant((ALLMACH_GAMMA - 1.0) * ALLMACH_T_REF)
                    .multiply(TypedCoeff::from_field(
                        TypedFieldRef::<Compressibility, Scalar>::new(ALLMACH_PSI_REF_FIELD),
                    ))
                    .multiply(TypedCoeff::from_field(
                        TypedFieldRef::<DynamicViscosity, Scalar>::new("mu"),
                    ));
            t_sum = t_sum
                + typed_fvc::viscous_dissipation(visc_diss_coeff, u_typed)
                    .cast_to::<TEquationUnit>();
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
    build_allmach_system(&fields, false, false, false, false)
}

pub fn allmach_pressure_model() -> Result<ModelSpec, String> {
    allmach_pressure_model_impl(false, false, false, false)
}

/// `allmach_pressure` plus manufactured momentum + continuity source fields for
/// MMS order tests.
pub fn allmach_pressure_mms_model() -> Result<ModelSpec, String> {
    allmach_pressure_model_impl(true, false, false, false)
}

/// Thermal all-Mach: `allmach_pressure` + a temperature transport equation and
/// an on-device ideal-gas density recovery `rho = rho_t_ref/T + psi*p` (declared
/// as a `PrimitiveDerivations` math expression, lowered to the coupled Update
/// kernel — no hand-written kernel). The barotropic model is the `T = T_ref`
/// limit.
pub fn allmach_thermal_model() -> Result<ModelSpec, String> {
    allmach_pressure_model_impl(false, false, true, false)
}

/// `allmach_thermal` plus manufactured momentum + continuity + temperature
/// source fields for MMS order tests. BAROTROPIC mms: strips the compressible
/// physics to isolate the 1/T density variation (PSI = 0 in the test).
pub fn allmach_thermal_mms_model() -> Result<ModelSpec, String> {
    allmach_pressure_model_impl(true, false, true, false)
}

/// COMPRESSIBLE thermal MMS: `allmach_thermal` with the FULL production physics
/// (real-EOS `rho(p,T)`, U.grad(p) compression heating, viscous dissipation Φ) plus
/// manufactured momentum/continuity/temperature sources. Unlike the barotropic
/// `allmach_thermal_mms`, it does NOT strip the compressible terms, so a steady
/// order test at PSI > 0 verifies the compressible spatial operator. Its genuinely
/// transient terms (dp/dt heating, thermal expansion, acoustic ddt) vanish at the
/// steady manufactured solution and so leave the observed order unaffected.
pub fn allmach_thermal_compressible_mms_model() -> Result<ModelSpec, String> {
    allmach_pressure_model_impl(true, true, true, false)
}

/// ALE (moving-mesh) variant of the compressible thermal MMS: same kept-physics +
/// manufactured sources with mesh-relative convection. With `mesh_fluxes` zero-filled
/// and equal volume history it reproduces the static `allmach_thermal_compressible_mms`
/// bitwise (the ALE zero-flux invariant), which is what the ALE MMS gate checks.
pub fn allmach_thermal_compressible_mms_ale_model() -> Result<ModelSpec, String> {
    allmach_pressure_model_impl(true, true, true, true)
}

/// ALE (moving-mesh) variant of `allmach_pressure`: the barotropic all-Mach
/// compressible solver with mesh-relative convection. `div(phi,U).bounded()` and
/// `div_flux(phi,p)` are declared `.with_mesh_relative()`, so assembly consumes
/// `phi_rel = phi - rho_f * mesh_fluxes[face]` with the variable-density face
/// density `rho_f`; the moving-volume ddt and the barotropic continuity volume
/// source (`+rho_ref*(V^{n+1}-V^n)/dt`) are emitted by the shared ALE codegen.
/// Its own model id ⇒ own generated kernels; with `mesh_fluxes` zero-filled and
/// equal volume history it reproduces the static `allmach_pressure` bitwise.
pub fn allmach_pressure_ale_model() -> Result<ModelSpec, String> {
    allmach_pressure_model_impl(false, false, false, true)
}

/// `allmach_pressure_ale` + manufactured sources (prescribed-motion compressible
/// MMS order test).
pub fn allmach_pressure_ale_mms_model() -> Result<ModelSpec, String> {
    allmach_pressure_model_impl(true, false, false, true)
}

/// ALE (moving-mesh) variant of `allmach_thermal`: the thermal all-Mach solver
/// with mesh-relative convection on momentum, continuity AND temperature. The
/// cross-variable thermal-expansion / compression-heating ddt terms get the same
/// moving-volume weighting as the primary ddts.
pub fn allmach_thermal_ale_model() -> Result<ModelSpec, String> {
    allmach_pressure_model_impl(false, false, true, true)
}

/// `allmach_thermal_ale` + manufactured sources (prescribed-motion thermal
/// compressible MMS order test). BAROTROPIC mms.
pub fn allmach_thermal_ale_mms_model() -> Result<ModelSpec, String> {
    allmach_pressure_model_impl(true, false, true, true)
}

/// In-place flip of an all-Mach model's Inlet/Outlet boundary KINDS to CD-nozzle
/// driving: a pressure inlet + a supersonic (fully-extrapolated) outlet. Moves the
/// single gauge anchor from the outlet upstream to the inlet and lets the outlet float:
/// - `U` Inlet -> [ZeroGradient (axial speed develops with the drop), Dirichlet(0)
///   (transverse pinned, blocks corner backflow)];
/// - `p` Inlet -> Dirichlet (the new gauge anchor; value set at runtime via
///   `set_boundary_scalar(Inlet, "p", inlet_pressure)`);
/// - `p` Outlet -> ZeroGradient (no back-pressure; extrapolate);
/// - `T` Outlet (thermal only) -> ZeroGradient (supersonic outlet must let the gas cool).
///
/// The pressure-Dirichlet face count is unchanged (one boundary's worth moves from
/// Outlet to Inlet), so the discrete pressure operator stays non-singular. BC kind is
/// a runtime `bc_table` (not baked into kernels), so the model id / committed kernels
/// are untouched. NB this driving is over-expanded in the artificial-compressibility
/// model (throat over-chokes, diverging section diffuses).
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

fn allmach_pressure_model_impl(
    with_mms_source: bool,
    compressible_mms: bool,
    thermal: bool,
    ale: bool,
) -> Result<ModelSpec, String> {
    // See `build_allmach_system`: `strip_all` strips the compressible physics for the
    // BAROTROPIC mms only; the COMPRESSIBLE mms keeps the full production physics
    // (real-EOS recovery, layout fields, Rhie–Chow) and just adds manufactured sources.
    let strip_all = with_mms_source && !compressible_mms;
    let fields = AllMachPressureFields::new();
    let system = build_allmach_system(&fields, with_mms_source, compressible_mms, thermal, ale);

    // Keep U,p,d_p,grad_p,grad_p_old at the same offsets as incompressible
    // (0,2,3,4,6); append psi (and any MMS sources) after. Offsets are load-bearing.
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
        // Present for production AND the compressible mms (kept physics), absent for
        // the barotropic mms (`strip_all`).
        if !strip_all {
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
            // Reference compressibility (= 1/c_ref^2), seeded by the driver. The
            // ideal-gas EOS coefficients (rho recovery, rho_dT, 1/cp compression
            // heating) read THIS constant, decoupling them from the `psi` field that
            // carries the LOCAL 1/c^2(T). Appended last so every prior field's offset
            // is unchanged (offsets are load-bearing).
            layout_fields.push(vol_scalar_dim::<Compressibility>(ALLMACH_PSI_REF_FIELD));
            // Preconditioner inputs for the on-device psi_precond recovery: the
            // inlet-scale velocity floor `u_ref` and the 0/1 enable `precond_mask`.
            layout_fields.push(vol_scalar_dim::<Velocity>(ALLMACH_U_REF_FIELD));
            layout_fields.push(vol_scalar_dim::<cfd2_ir::dimensions::Dimensionless>(
                ALLMACH_PRECOND_MASK_FIELD,
            ));
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

    // ClosedForm d_p (= α_u·dt/ρ_ref) preserves the manufactured-solution order;
    // the SIMPLEC `FromAssembledRowSum` row-sum d_p degrades coupled MMS u-order to
    // ~1.6 < 1.65. The supersonic runaway is cured by the pressure-flux Newton
    // linearization instead, which does not touch the spatial operator / MMS order.
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
        let p_compr = if strip_all {
            Expr::ident("psi") * Expr::ident("p")
        } else {
            // Uses the REFERENCE compressibility (constant), not the local `psi`:
            // gamma*psi_ref*t_ref/T = 1/(R*T) is the isothermal d(rho)/d(p)|_T, so the
            // recovery stays a consistent ideal gas rho = p/(R*T) as `psi` goes local.
            // Kept for the compressible mms so the real-EOS operator is order-verified.
            Expr::lit_f32(ALLMACH_GAMMA as f32)
                * Expr::ident(ALLMACH_PSI_REF_FIELD)
                * Expr::ident(ALLMACH_T_REF_FIELD)
                * Expr::ident("p")
                / Expr::ident(ALLMACH_TEMPERATURE_FIELD)
        };
        let rho_recovery = Expr::ident(ALLMACH_RHO_T_REF_FIELD)
            / Expr::ident(ALLMACH_TEMPERATURE_FIELD)
            + p_compr;
        let rho_recovery = if strip_all {
            rho_recovery
        } else {
            Expr::call_named("max", vec![rho_recovery, Expr::ident(ALLMACH_RHO_FLOOR_FIELD)])
        };
        derivations.insert("rho".to_string(), rho_recovery);
        // Thermal-expansion coefficient rho_dT = d(rho)/dT, recovered on-device. With the
        // real T-varying compressibility, rho = (rho_t_ref + gamma*psi*t_ref*p)/T, so
        // rho_dT = -(rho_t_ref + gamma*psi*t_ref*p)/T^2 (= -rho/T). Units: (Density*Temp)/
        // T^2 = Density/Temperature; the Add is Density*Temp + Density*Temp (no panic).
        if !strip_all {
            let rho_numer = Expr::ident(ALLMACH_RHO_T_REF_FIELD)
                + Expr::lit_f32(ALLMACH_GAMMA as f32)
                    * Expr::ident(ALLMACH_PSI_REF_FIELD)
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
            // LOCAL sound speed (the "real sound speed"): psi = 1/c^2(T) =
            // psi_ref * t_ref / T = 1/(gamma*R*T) (isentropic ideal-gas 1/c^2).
            // Recovered on-device each outer iteration from T alone — it reads only
            // the constant psi_ref/t_ref and the solved T, so there is NO psi<->rho
            // cycle (rho uses the constant psi_ref), and it is division-only for
            // cross-backend byte parity. As the gas accelerates and cools, psi rises
            // (c falls) LOCALLY. The div_flux Newton Jacobian and the preconditioner
            // read this state `psi` (one-outer-iteration Picard lag); the EOS
            // coefficients keep the constant psi_ref (see the rho recovery above).
            // SIGN SAFETY ONLY: clamp psi at a tiny positive floor. As T -> 0+ the
            // local psi grows unboundedly, which is CORRECT and stabilising — a low
            // local sound speed makes the gas "feel" more compressible and self-limits
            // the acceleration at a cold exit. The only pathology is a transient T
            // undershoot THROUGH zero, where psi_ref*t_ref/T flips sign and would invert
            // the div_flux Jacobian; `max(..., psi_ref)` keeps psi >= the reference value
            // (never negative) without capping the physical stiffening at small positive
            // T. Inert wherever T >= t_ref.
            derivations.insert(
                "psi".to_string(),
                Expr::call_named(
                    "max",
                    vec![
                        Expr::ident(ALLMACH_PSI_REF_FIELD) * Expr::ident(ALLMACH_T_REF_FIELD)
                            / Expr::ident(ALLMACH_TEMPERATURE_FIELD),
                        Expr::ident(ALLMACH_PSI_REF_FIELD),
                    ],
                ),
            );
            // Low-Mach (Turkel) preconditioned pseudo-compressibility, moved on-device
            // so the raw-solver nozzle tests exercise it (it was host-seeded before).
            // psi_precond = precond_mask * max(psi, 1/beta^2), beta^2 =
            // max(|U|^2, u_ref^2), using the LOCAL psi so the pseudo-acoustic scale
            // tracks the true local sound speed (the plan's real-sound-speed
            // preconditioner). NOTE the design trade-off: at a deeply-cooled exit the
            // local psi grows, so the pressure ddt mass term over-damps the pseudo-time
            // there — a benign, graceful slowdown (the exit develops slowly) rather
            // than the catastrophic blow-up the bounded-reference variant produced on
            // the over-driven GUI-default nozzle. The 0/1 precond_mask reproduces the
            // incompressible special case exactly: psi_ref=0 => psi=0 but 1/beta^2 != 0,
            // so mask=0 forces psi_precond=0 (the acoustic ddt term vanishes). Reads
            // psi (ordered after it), U and the constants u_ref/precond_mask => acyclic.
            // Unit: Compressibility reduces to 1/Velocity^2, so max(psi, 1/beta^2) is
            // unit-consistent.
            let beta2 = Expr::call_named(
                "max",
                vec![
                    Expr::ident("U_x") * Expr::ident("U_x")
                        + Expr::ident("U_y") * Expr::ident("U_y"),
                    Expr::ident(ALLMACH_U_REF_FIELD) * Expr::ident(ALLMACH_U_REF_FIELD),
                ],
            );
            derivations.insert(
                "psi_precond".to_string(),
                Expr::ident(ALLMACH_PRECOND_MASK_FIELD)
                    * Expr::call_named(
                        "max",
                        vec![Expr::ident("psi"), Expr::lit_f32(1.0) / beta2],
                    ),
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
        id: match (thermal, with_mms_source, compressible_mms, ale) {
            // Compressible thermal MMS (kept-physics + manufactured sources).
            (true, true, true, false) => "allmach_thermal_compressible_mms",
            (true, true, true, true) => "allmach_thermal_compressible_mms_ale",
            // Barotropic mms + production (compressible_mms = false).
            (true, true, false, false) => "allmach_thermal_mms",
            (true, false, false, false) => "allmach_thermal",
            (false, true, false, false) => "allmach_pressure_mms",
            (false, false, false, false) => "allmach_pressure",
            (true, true, false, true) => "allmach_thermal_ale_mms",
            (true, false, false, true) => "allmach_thermal_ale",
            (false, true, false, true) => "allmach_pressure_ale_mms",
            (false, false, false, true) => "allmach_pressure_ale",
            // compressible_mms only valid on the thermal mms path.
            (_, _, true, _) => {
                return Err(
                    "compressible_mms requires thermal=true and with_mms_source=true".into(),
                )
            }
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
                // Explicit (not the 1.0 = auto heavy-ball 1.95): the all-Mach
                // pressure row is upwinded (deferred-Newton flux Jacobian), so
                // its Jacobi-scaled spectrum has imaginary parts. The heavy-ball
                // stability ellipse collapses onto the real axis as omega -> 2
                // (imag tolerance ~0.025 at 1.95 vs ~0.28 at 1.6); at 1.95 the
                // rocket-scale nozzle demo blows into the near-vacuum degeneracy
                // (gate nozzle_interior_vacuum_probe), while 1.6 reproduces the
                // plain-Jacobi trajectory and is still ~2x faster end-to-end.
                omega: 1.6,
                // The psi/dtau mass term boosts the pressure diagonal, so the
                // inner relaxation saturates by ~24-32 sweeps on the fine
                // nozzle (0.80s at 32 vs 0.93s at the shared 64); 32 keeps
                // margin above the measured saturation point.
                sweeps_cap: 32,
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
