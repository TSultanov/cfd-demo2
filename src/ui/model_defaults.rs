//! Per-model default GUI solver parameters.
//!
//! Single source of truth so GUI startup (`App::new`), model switching, and the
//! headless convergence gate all apply identical values. The incompressible and
//! compressible models need very different defaults: incompressible wants early
//! outer-loop exit; compressible wants low-Mach preconditioning, conservative CFL,
//! and TVD reconstruction or the near-incompressible Air default blows up on the
//! acoustic wave speed (c ~347 m/s at Mach ~0.003).

use crate::sim::RuntimeParams;
use crate::solver::model::eos::EosSpec;
use crate::solver::scheme::Scheme;
use crate::solver::{GpuLowMachPrecondModel, PreconditionerType, TimeScheme as GpuTimeScheme};

/// Default GUI solver parameters for one model.
///
/// Only the runtime/solver knobs live here; geometry, mesh, and fluid selection
/// are independent of the model and are left to the existing GUI state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelGuiDefaults {
    pub advection_scheme: Scheme,
    pub time_scheme: GpuTimeScheme,
    /// Preconditioner the GUI selects. Ignored for models that own their
    /// preconditioner (the incompressible coupled solver forces Schur); kept so
    /// the GUI selection box has a defined starting value per model.
    pub preconditioner: PreconditionerType,
    pub alpha_u: f64,
    pub alpha_p: f64,
    /// Upper bound on outer (SIMPLE / pseudo-time) iterations per timestep.
    pub outer_iters: u32,
    /// When true, the per-step outer loop may exit early once the coupled residual
    /// falls below the monitor tolerance (decoupled from the console logging flag).
    pub outer_auto_converge: bool,
    pub target_cfl: f64,
    /// Physical timestep. With `adaptive_dt` it is the seed before adaptive-dt
    /// takes over; otherwise it is the fixed timestep.
    pub timestep: f64,
    /// Use the acoustic-aware adaptive timestep. Off for the compressible default:
    /// a fixed small dt + low-Mach preconditioning is the checkerboard-free recipe;
    /// the adaptive low-Mach dt inflation is unstable at a single outer iteration.
    pub adaptive_dt: bool,
    /// Enable pseudo-transient continuation (dual time, `dtau > 0`). On for the
    /// compressible default: at Air's near-zero Mach the time-accurate density-based
    /// solver is marginally unstable on the collocated cut-cell mesh (an inlet-seeded
    /// odd/even pressure mode grows unbounded); the pseudo-transient term damps it and
    /// relaxes toward the quasi-steady solution. Must be paired with the
    /// uniform-freestream IC (see `App` worker init), else inlet momentum piles up at
    /// the inlet with no convective transport from rest.
    pub dual_time: bool,
    /// Pseudo-time step for `dual_time`. Stable band: stable for `dtau <~ 1e-2`,
    /// diverges by `~1e-1`; 1e-3 leaves ~10x margin.
    pub dtau: f64,
    pub low_mach_model: GpuLowMachPrecondModel,
    /// `f32` to match the GUI state and the `set_precond_*` solver setters.
    pub low_mach_theta_floor: f32,
    pub low_mach_pressure_coupling_alpha: f32,
    /// Default inlet velocity (m/s). Per-model because Re scales with it: the
    /// incompressible default uses a low speed so the real Air viscosity gives a
    /// laminar, stable Re on the coarse cut-cell geometries (Reynolds-honest, unlike
    /// flooring the viscosity).
    pub inlet_velocity: f32,
    /// Gauge back-pressure pinned at the outlet. `0.0` for every standard case; the
    /// supersonic-nozzle demo uses a negative value to pull the diverging section
    /// past Mach 1. Applied to the gauge-pressure (`allmach_*`) models only.
    pub outlet_back_pressure: f32,
    /// Minimum preconditioner reference velocity (all-Mach only) — the pseudo-sound
    /// floor that both stops the compressible-ALE outlet divergence and (raised
    /// toward ~1.0) cleans the residual standing pressure mode toward the
    /// incompressible field. See [`RuntimeParams::allmach_precond_uref_min`]. `0.0`
    /// for non-all-Mach models (ignored); the all-Mach presets set it explicitly.
    pub allmach_precond_uref_min: f32,
    /// Drive the CD nozzle with a pressure inlet + supersonic (extrapolated) outlet
    /// instead of the velocity-inlet / pressure-outlet default. See
    /// [`RuntimeParams::pressure_inlet`]. `false` for every standard case.
    pub pressure_inlet: bool,
    /// Inlet gauge pressure pinned when [`Self::pressure_inlet`] is set. (Ignored
    /// otherwise.)
    pub inlet_pressure: f32,
}

impl ModelGuiDefaults {
    /// Canonical `ModelGuiDefaults` → [`RuntimeParams`] mapping: the composition of
    /// the GUI's `apply_model_defaults` and `current_runtime_params`, in one place so
    /// GUI startup and the headless gate build the identical bag the shared
    /// [`crate::sim::SolverDriver`] consumes.
    ///
    /// Fluid properties are passed as primitives — the driver is `ui`-independent and
    /// never sees the `ui`-gated `Fluid`. Headless sweeps that vary the inlet speed
    /// mutate the `ModelGuiDefaults` before calling this.
    pub fn to_runtime_params(&self, density: f32, viscosity: f32, eos: EosSpec) -> RuntimeParams {
        RuntimeParams {
            adaptive_dt: self.adaptive_dt,
            target_cfl: self.target_cfl,
            requested_dt: self.timestep as f32,
            // Pseudo-transient continuation only when the model enables dual time.
            dtau: if self.dual_time {
                self.dtau.max(0.0) as f32
            } else {
                0.0
            },
            log_convergence: false,
            log_every_steps: 50,
            advection_scheme: self.advection_scheme,
            time_scheme: self.time_scheme,
            preconditioner: self.preconditioner,
            outer_iters: self.outer_iters.max(1),
            outer_auto_converge: self.outer_auto_converge,
            low_mach_model: self.low_mach_model,
            low_mach_theta_floor: self.low_mach_theta_floor,
            low_mach_pressure_coupling_alpha: self.low_mach_pressure_coupling_alpha,
            alpha_u: self.alpha_u as f32,
            alpha_p: self.alpha_p as f32,
            inlet_velocity: self.inlet_velocity,
            density,
            viscosity,
            eos,
            // psi = 1/c^2 from the material EOS: 0 for an incompressible `Constant`
            // EOS (all-Mach recovers incompressible), real 1/c^2 for a gas.
            compressibility_psi: eos.compressibility(density as f64) as f32,
            outlet_back_pressure: self.outlet_back_pressure,
            allmach_precond_uref_min: self.allmach_precond_uref_min,
            pressure_inlet: self.pressure_inlet,
            inlet_pressure: self.inlet_pressure,
        }
    }
}

/// Incompressible momentum (coupled SIMPLE) defaults.
///
/// Three levers chosen so the default flow is both stable and sheds a Kármán vortex
/// street (the channel-obstacle headline demo):
///
/// * **Van Leer (TVD) advection**: on the coarse cut-cell mesh (~8 cells across the
///   cylinder) Upwind's numerical diffusion is ~10x the real Air viscosity, which
///   collapses the effective Re below the shedding threshold. Van Leer is 2nd-order
///   in the smooth wake so effective Re tracks physical; TVD (bounded), verified
///   stable on the cut-cell slivers (no-slip immersed-wall BC supplies the damping,
///   see `generate_cut_cell_mesh`).
/// * A **laminar-but-shedding inlet speed**: Re = U·D/ν, D = 0.2, ν = μ/ρ ≈ 1.48e-5
///   ⇒ U = 0.011 gives Re ≈ 150, inside the 2D-laminar shedding band (onset ≈ 47,
///   3D transition ≈ 190). Also keeps the backward-step laminar (channel Re ≈ 750,
///   step Re ≈ 375, below the ≈ 1200 transition).
/// * A **low fixed outer cap** (`outer_iters`) for cost — Ghia shows ~5 under-relaxed
///   SIMPLE sweeps per step already march correctly.
const INCOMPRESSIBLE: ModelGuiDefaults = ModelGuiDefaults {
    advection_scheme: Scheme::SecondOrderUpwindVanLeer,
    time_scheme: GpuTimeScheme::BDF2,
    preconditioner: PreconditionerType::Jacobi,
    alpha_u: 0.7,
    alpha_p: 0.3,
    // Few under-relaxed SIMPLE sweeps per step (Ghia validates 5; 8 keeps headroom
    // for the harder near-inviscid Air default). `outer_auto_converge` keeps the
    // monitor on (residual readout + opportunistic early exit) but is not relied
    // upon: the coupled break does not reliably fire for this model.
    outer_iters: 8,
    outer_auto_converge: true,
    target_cfl: 0.9,
    timestep: 0.02,
    adaptive_dt: true,
    dual_time: false,
    dtau: 1e-5,
    low_mach_model: GpuLowMachPrecondModel::Off,
    low_mach_theta_floor: 1e-6,
    low_mach_pressure_coupling_alpha: 1.0,
    // Real Air viscosity + this inlet speed -> Re ≈ 150 on the cylinder (D = 0.2):
    // the 2D-laminar vortex-shedding band, no viscosity floor.
    inlet_velocity: 0.011,
    outlet_back_pressure: 0.0,
    allmach_precond_uref_min: 0.0, // non-all-Mach: ignored
    pressure_inlet: false,
    inlet_pressure: 0.0,
};

/// Compressible (density-based, implicit) defaults.
///
/// Van Leer reconstruction, a single outer iteration, Weiss-Smith low-Mach
/// preconditioning, a low (laminar-Re) inlet speed, and — the stabilizer —
/// pseudo-transient continuation (`dual_time`) started from a uniform-freestream IC.
///
/// The low-Mach inlet instability motivating those last two: at Air's near-zero Mach
/// the time-accurate density-based solver started from REST on the collocated cut-cell
/// mesh is unstable — the inlet injects momentum that, with no convective transport
/// from rest, piles up and grows an odd/even pressure mode without bound. Not fixed by
/// the advection scheme, `low_mach_theta_floor`, or `low_mach_pressure_coupling_alpha`
/// (all inert here). The fix touches no solver code: (1) uniform-freestream IC so
/// convection is established everywhere, and (2) `dual_time` (`dtau > 0`) to damp the
/// marginal mode and relax toward the quasi-steady solution. Validated MMS / lid /
/// OpenFOAM-reference cases set their own `dtau`/IC and are unaffected.
const COMPRESSIBLE: ModelGuiDefaults = ModelGuiDefaults {
    advection_scheme: Scheme::SecondOrderUpwindVanLeer,
    time_scheme: GpuTimeScheme::BDF2,
    preconditioner: PreconditionerType::Jacobi,
    alpha_u: 1.0,
    alpha_p: 1.0,
    outer_iters: 1,
    outer_auto_converge: true,
    target_cfl: 0.3,
    timestep: 1e-5,
    adaptive_dt: false,
    // Pseudo-transient low-Mach stabilizer (see the field doc on `dual_time`).
    dual_time: true,
    dtau: 1e-3,
    low_mach_model: GpuLowMachPrecondModel::WeissSmith,
    low_mach_theta_floor: 1e-8,
    low_mach_pressure_coupling_alpha: 0.01,
    // Same obstacle regime as the incompressible and pressure-based (all-Mach)
    // demos: Re = U*D/nu ~ 150 with D = 0.2 (the 2D-laminar shedding band), so
    // every model family runs the SAME physical default case. The previous
    // 0.002 sat at Re ~ 27 — below the ~47 shedding threshold, so the default
    // obstacle could never develop a street under this model at all.
    inlet_velocity: 0.011,
    outlet_back_pressure: 0.0,
    allmach_precond_uref_min: 0.0, // density-based compressible: floor unused
    pressure_inlet: false,
    inlet_pressure: 0.0,
};

/// Density-based compressible on the NOZZLE geometry: pressure-driven inlet.
/// A velocity inlet is useless for a CD nozzle (nothing chokes); the committed
/// bc_expr closures instead prescribe the gauge inlet pressure + reservoir
/// temperature and let the outlet float (see `BC_PRESSURE_INLET` in the
/// compressible model). `inlet_pressure` is in gauge Pa against the absolute
/// ~101 kPa reference.
pub const COMPRESSIBLE_NOZZLE: ModelGuiDefaults = ModelGuiDefaults {
    pressure_inlet: true,
    inlet_pressure: 5.0e4,
    adaptive_dt: true,
    ..COMPRESSIBLE
};

/// All-Mach pressure-based (`allmach_pressure`) defaults.
///
/// The incompressible coupled solver (gauge pressure, Rhie–Chow, Schur) plus a
/// `ddt(psi,p)` compressibility term and per-cell variable density `rho = rho_ref +
/// psi*p`. Runs in the incompressible branch of the driver (no `eos.gamma` → Coupled
/// stepping), so the validated incompressible knobs apply verbatim. The one addition
/// is a positive compressibility `psi = 1/c^2 = Fluid::compressibility()` (real, no
/// exaggeration). For Air (c ≈ 347) `psi ≈ 8.3e-6`, so the default obstacle flow is
/// near-incompressible (inlet Mach ≈ 3e-5) — the incompressible Kármán street the
/// all-Mach model reduces to at small `psi`.
///
/// The real `psi` is acoustically stiff (`c = 1/√psi ≈ 347`, acoustic CFL ≈ 667 at the
/// convective dt → step-0 blow-up). The driver cures this with low-Mach (Turkel)
/// preconditioning: a decoupled pseudo-compressibility `psi_precond = max(psi,
/// 1/(k·U_inlet)²)` that only the pressure-row time term reads (density keeps the real
/// `psi`), rescaling the pseudo sound speed toward the local velocity. It vanishes at
/// steady state, so the converged solution is the real-`psi` physics.
const ALLMACH: ModelGuiDefaults = ModelGuiDefaults {
    advection_scheme: Scheme::SecondOrderUpwindVanLeer,
    time_scheme: GpuTimeScheme::BDF2,
    preconditioner: PreconditionerType::Jacobi,
    alpha_u: 0.7,
    alpha_p: 0.3,
    outer_iters: 8,
    outer_auto_converge: true,
    target_cfl: 0.9,
    timestep: 0.02,
    adaptive_dt: true,
    // No pseudo-transient: the gauge pressure has no low-Mach inlet instability
    // (that pathology is specific to the density-based solver's absolute-pressure state).
    dual_time: false,
    dtau: 1e-5,
    low_mach_model: GpuLowMachPrecondModel::Off,
    low_mach_theta_floor: 1e-6,
    low_mach_pressure_coupling_alpha: 1.0,
    inlet_velocity: 0.011,
    // Standard outlet for the channel/backstep cases (the nozzle demo overrides this).
    outlet_back_pressure: 0.0,
    // Preconditioner floor 1.0 (not the bare-stability 0.2): under the adaptive dt the
    // larger developed timestep + this lower psi_precond make the pressure nearly
    // elliptic, collapsing the residual standing pseudo-acoustic mode onto the
    // incompressible field (the "pressure looks wrong" cure). Step-0 safe via the
    // moving driver's startup dt growth-cap. GUI-tunable via the all-Mach floor slider.
    allmach_precond_uref_min: 1.0,
    pressure_inlet: false,
    inlet_pressure: 0.0,
};

/// All-Mach thermal supersonic-nozzle demo defaults (real Air `c ≈ 347 m/s`,
/// `psi = 1/c² ≈ 8.3e-6`, no exaggeration).
///
/// Same gauge-pressure compressible knobs as [`ALLMACH`], driven for the
/// converging–diverging nozzle geometry (`generate_structured_nozzle_mesh`, area ratio
/// exit/throat = 2). A pressure inlet pins the inlet gauge pressure and the supersonic
/// outlet floats, so the throughflow (O(100s of m/s)) accelerates through the throat.
/// `inlet_velocity` is the preconditioner / CFL throughflow scale; `inlet_pressure`
/// drives it; `outlet_back_pressure` is the velocity-inlet fallback. Paired with
/// `allmach_thermal` so the demo also shows expansion cooling.
pub const ALLMACH_THERMAL_NOZZLE: ModelGuiDefaults = ModelGuiDefaults {
    advection_scheme: Scheme::SecondOrderUpwindVanLeer,
    time_scheme: GpuTimeScheme::BDF2,
    preconditioner: PreconditionerType::Jacobi,
    alpha_u: 0.7,
    alpha_p: 0.3,
    outer_iters: 10,
    outer_auto_converge: true,
    // Conservative acoustic CFL and a tiny seed dt: at c≈347 m/s the throughflow is
    // O(300 m/s), so the acoustic-aware adaptive dt settles to O(h/c); the small seed
    // keeps step 0 in-bounds before it adapts (a 0.02 seed would blow up step 0). A
    // gentle CFL keeps the from-rest acoustic transient from over-expanding the throat.
    target_cfl: 0.25,
    timestep: 1e-5,
    adaptive_dt: true,
    // Pseudo-transient continuation ON: the from-rest start is a violent acoustic
    // transient (the pressure drop slams the throat) and under the LOCAL sound speed a
    // momentary over-expansion drives a cold spot whose `psi = psi_ref*t_ref/T` blows up
    // and freezes the flow. A dual-time (pseudo-time) derivative damps that overshoot so
    // the nozzle marches to the steady supersonic CD state instead of crashing.
    dual_time: true,
    dtau: 1e-3,
    low_mach_model: GpuLowMachPrecondModel::Off,
    low_mach_theta_floor: 1e-6,
    low_mach_pressure_coupling_alpha: 1.0,
    // Low-Mach preconditioner / startup-CFL scale — NOT a prescribed inlet velocity.
    // The flow develops entirely from the pressure drop (inlet U is zero-gradient, see
    // `apply_pressure_inlet_nozzle_bcs`); this value only sets the acoustic-damping
    // preconditioner floor and the step-0 dt while the flow is still at rest. It
    // deliberately UNDER-estimates the developed throughflow (~300–580 m/s at 0.3 MPa,
    // throat to supersonic exit): a lower scale means a stronger pseudo-compressibility
    // floor, which damps the from-rest acoustic transient.
    inlet_velocity: 313.0,
    // Velocity-inlet fallback back-pressure (gauge); the default driving is the
    // pressure inlet below.
    outlet_back_pressure: -0.045,
    // Inert here: the throughflow scale (inlet_velocity 313) dominates this floor, so
    // the preconditioner never binds to it and the ramp is a no-op for the nozzle.
    allmach_precond_uref_min: 0.2,
    // Pressure inlet + supersonic (extrapolated) outlet at the real sound speed: the
    // gauge anchors at the inlet, the outlet floats. The elliptic pressure row is made
    // well-posed at the supersonic exit by the pressure-flux Newton linearization;
    // without it this real-c drive ran the exit to vacuum.
    //
    // `inlet_pressure` = 0.3 MPa gauge (~4 bar absolute), Pascals. The pressure ratio
    // still comfortably exceeds the ~1.9 choking threshold (so the throat chokes and the
    // diverging section runs supersonic), but it is FAR gentler than the 1 MPa rocket
    // slam that over-expanded the throat to a near-vacuum cold spot from rest. The exit
    // Mach is set by the AREA RATIO (isentropic area–Mach), not the pressure, so a
    // gentler ratio reaches the SAME supersonic branch while developing cleanly from rest.
    pressure_inlet: true,
    inlet_pressure: 3.0e5,
};

/// GUI solver defaults for `model_id`.
///
/// Unknown ids fall back to the incompressible defaults, which is the GUI's
/// startup model.
pub fn gui_defaults_for(model_id: &str) -> ModelGuiDefaults {
    // Structured model ids are `*_structured` siblings of the same physics —
    // map them to the base family's knobs so outer_auto_converge / schemes /
    // relaxation match the unstructured defaults.
    let base = model_id.strip_suffix("_structured").unwrap_or(model_id);
    match base {
        "compressible" => COMPRESSIBLE,
        // The thermal variant shares the gauge-pressure knobs; the nozzle case
        // overrides via `ALLMACH_THERMAL_NOZZLE` when the nozzle geometry is selected.
        "allmach_pressure" | "allmach_thermal" => ALLMACH,
        _ => INCOMPRESSIBLE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incompressible_enables_early_exit_with_low_cap() {
        let d = gui_defaults_for("incompressible_momentum");
        assert!(d.outer_auto_converge, "early-exit must be on by default");
        assert!(d.outer_iters <= 10, "outer cap should be low");
        assert!(d.outer_iters > 1, "cap must allow the break to fire");
        assert_eq!(d.low_mach_model, GpuLowMachPrecondModel::Off);
    }

    #[test]
    fn incompressible_defaults_to_vortex_shedding_regime() {
        // The headline obstacle demo must SHED a vortex street, which needs both a
        // low-diffusion (TVD) advection scheme — Upwind's numerical diffusion on the
        // coarse cut-cell mesh smears the street into a steady blob — and an inlet
        // speed that puts the cylinder (D = 0.2, ν ≈ 1.48e-5) in the 2D-laminar
        // shedding band (Re ≈ 47..190). Re = U·D/ν.
        let d = gui_defaults_for("incompressible_momentum");
        assert_eq!(
            d.advection_scheme,
            Scheme::SecondOrderUpwindVanLeer,
            "obstacle vortex street needs a TVD (Van Leer) scheme, not first-order Upwind"
        );
        let nu = 1.81e-5_f64 / 1.225; // Air μ/ρ
        let re = d.inlet_velocity as f64 * 0.2 / nu;
        assert!(
            (47.0..190.0).contains(&re),
            "inlet speed {} gives cylinder Re {:.0}, outside the 2D-laminar shedding band [47,190]",
            d.inlet_velocity,
            re
        );
    }

    #[test]
    fn compressible_uses_low_mach_and_conservative_cfl() {
        let d = gui_defaults_for("compressible");
        assert_eq!(d.low_mach_model, GpuLowMachPrecondModel::WeissSmith);
        assert!(d.target_cfl <= 0.5, "acoustic CFL must be conservative");
        assert_eq!(d.advection_scheme, Scheme::SecondOrderUpwindVanLeer);
        assert_eq!(d.outer_iters, 1);
    }

    #[test]
    fn compressible_enables_pseudo_transient_stabilizer() {
        // The low-Mach inlet instability is cured by pseudo-transient continuation
        // (paired with the uniform-freestream IC the app applies), not by the
        // scheme or low-Mach knobs. A `dtau` in the stable band is the contract.
        let d = gui_defaults_for("compressible");
        assert!(d.dual_time, "compressible default must run pseudo-transient");
        assert!(
            d.dtau > 0.0 && d.dtau <= 1e-2,
            "dtau {} must be in the stable band (<= ~1e-2)",
            d.dtau
        );
        // Incompressible stays time-accurate (no pseudo-transient).
        assert!(!gui_defaults_for("incompressible_momentum").dual_time);
    }

    #[test]
    fn both_models_default_to_a_laminar_inlet_speed() {
        // Air at 1 m/s is turbulent Re on the default geometries; the honest fix
        // is a low inlet speed (real viscosity), not a hidden viscosity floor.
        for id in ["incompressible_momentum", "compressible"] {
            let d = gui_defaults_for(id);
            assert!(
                d.inlet_velocity > 0.0 && d.inlet_velocity < 0.1,
                "{id}: inlet speed {} should be low (laminar Re)",
                d.inlet_velocity
            );
        }
    }

    #[test]
    fn unknown_model_falls_back_to_incompressible() {
        assert_eq!(
            gui_defaults_for("nonexistent"),
            gui_defaults_for("incompressible_momentum")
        );
    }

    #[test]
    fn allmach_default_compressibility_is_eos_derived_not_artificial() {
        // The all-Mach psi is ALWAYS the real EOS `1/c^2` from the fluid's sound speed
        // — no exaggeration factor anywhere. For Air (c≈347) that is ~8.3e-6 (honestly
        // near-incompressible).
        let d = gui_defaults_for("allmach_pressure");
        // The derived runtime psi matches 1/c^2 of the supplied fluid (Air).
        let air = EosSpec::IdealGas {
            gamma: 1.4,
            gas_constant: 287.0,
            temperature: 300.0,
        };
        let psi = d.to_runtime_params(1.225, 1.81e-5, air).compressibility_psi as f64;
        let psi_expected = air.compressibility(1.225); // 1/(gamma*R*T) ≈ 8.3e-6
        assert!(
            (psi - psi_expected).abs() / psi_expected < 1e-4,
            "derived psi {psi:.3e} must equal EOS 1/c^2 {psi_expected:.3e}"
        );
        assert!((8.0e-6..8.6e-6).contains(&psi), "Air 1/c^2 ≈ 8.3e-6, got {psi:.3e}");
        // A Constant (incompressible) EOS yields psi = 0 regardless of the factor.
        assert_eq!(
            d.to_runtime_params(1.0, 1e-3, EosSpec::Constant).compressibility_psi,
            0.0,
            "Constant EOS must give psi = 0 (incompressible limit)"
        );
        // Incompressible-branch knobs (no eos => Coupled; gauge pressure => no
        // pseudo-transient low-Mach stabilizer needed).
        assert!(!d.dual_time, "all-Mach uses gauge pressure: no pseudo-transient");
        assert_eq!(d.low_mach_model, GpuLowMachPrecondModel::Off);
        assert_eq!(d.advection_scheme, Scheme::SecondOrderUpwindVanLeer);
    }

    #[test]
    fn nozzle_demo_uses_real_compressibility_no_exaggeration() {
        // The supersonic nozzle uses the real Air compressibility psi = 1/c^2 ≈
        // 8.3e-6 (c ≈ 347 m/s).
        let air = EosSpec::IdealGas {
            gamma: 1.4,
            gas_constant: 287.0,
            temperature: 300.0,
        };
        let psi = ALLMACH_THERMAL_NOZZLE
            .to_runtime_params(1.225, 1.81e-5, air)
            .compressibility_psi as f64;
        assert!(
            (8.0e-6..8.6e-6).contains(&psi),
            "nozzle psi {psi:.3e} must be real Air 1/c^2 ≈ 8.3e-6 (no exaggeration)"
        );
    }
}
