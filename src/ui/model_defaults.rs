//! Per-model default GUI solver parameters.
//!
//! The desktop GUI exposes a single set of solver knobs, but the incompressible
//! and compressible models need very different defaults to behave well out of the
//! box:
//!
//! * **Incompressible** wants the per-step outer (SIMPLE) loop to *exit early*
//!   once the coupled residual drops, instead of always grinding through the full
//!   iteration cap. The break already exists in the coupled solver; it just has
//!   to be enabled by default and given a sane cap.
//! * **Compressible** wants low-Mach acoustic preconditioning plus a conservative
//!   CFL and a TVD (Van Leer) reconstruction, otherwise the near-incompressible
//!   Air default blows up on the acoustic wave speed (sound speed ~347 m/s at a
//!   Mach number ~0.003), and the near-inviscid Air viscosity sits below the
//!   coarse-mesh inviscid-stability floor.
//!
//! This module is the single source of truth for those defaults so that startup
//! (`App::new`) and model switching apply exactly the same values, and so the
//! headless convergence gate (`tests/gui_default_convergence_test.rs`) tunes the
//! same numbers the GUI ships.

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
    /// preconditioner (the incompressible coupled solver forces Schur), but kept
    /// here so the GUI's selection box has a defined starting value per model.
    pub preconditioner: PreconditionerType,
    pub alpha_u: f64,
    pub alpha_p: f64,
    /// Upper bound on outer (SIMPLE / pseudo-time) iterations per timestep.
    pub outer_iters: u32,
    /// When true, the per-step outer loop is allowed to exit early once the
    /// coupled residual falls below the monitor tolerance. This is decoupled
    /// from the console logging flag so the early-exit ships on by default.
    pub outer_auto_converge: bool,
    pub target_cfl: f64,
    /// Physical timestep. With `adaptive_dt` it is the seed before adaptive-dt
    /// takes over; otherwise it is the fixed timestep.
    pub timestep: f64,
    /// Use the acoustic-aware adaptive timestep. Off for the compressible default
    /// (a fixed small dt + genuine low-Mach preconditioning is the validated,
    /// checkerboard-free recipe; the adaptive low-Mach dt inflation is unstable
    /// at a single outer iteration).
    pub adaptive_dt: bool,
    /// Enable pseudo-transient continuation (dual time, `dtau > 0`). On for the
    /// compressible default: at Air's near-zero Mach the time-accurate
    /// density-based solver is marginally unstable on the collocated cut-cell
    /// mesh — an inlet-seeded odd/even pressure mode grows without bound. The
    /// pseudo-transient term damps that marginal mode and relaxes the flow toward
    /// the (quasi-steady) solution, which is the physically meaningful state for
    /// such a slow flow. Paired with the uniform-freestream initial condition
    /// (see `App` worker init): from rest the inlet-injected momentum has no
    /// convective transport and piles up at the inlet, so the freestream IC is
    /// what makes the low-Mach default both stable and non-trivial.
    pub dual_time: bool,
    /// Pseudo-time step for `dual_time`. Comfortably inside the stable band
    /// (stable for `dtau <~ 1e-2`, diverges by `~1e-1`); 1e-3 leaves ~10x margin.
    pub dtau: f64,
    pub low_mach_model: GpuLowMachPrecondModel,
    /// `f32` to match the GUI state and the `set_precond_*` solver setters.
    pub low_mach_theta_floor: f32,
    pub low_mach_pressure_coupling_alpha: f32,
    /// Default inlet velocity (m/s). Per-model because the Reynolds number scales
    /// with it: the incompressible default uses a low speed so the *real* Air
    /// viscosity gives a laminar, stable Re on the coarse cut-cell geometries.
    /// Lowering the speed is Reynolds-honest — unlike flooring the viscosity — and
    /// leaves the step count to develop unchanged (the adaptive dt grows as
    /// 1/speed, so it is CFL-limited either way); only the magnitudes shrink.
    pub inlet_velocity: f32,
}

impl ModelGuiDefaults {
    /// Canonical `ModelGuiDefaults` → [`RuntimeParams`] mapping: the composition of
    /// the GUI's `apply_model_defaults` (defaults → app state) and
    /// `current_runtime_params` (app state → runtime bag). Having it in one place
    /// means the GUI startup and the headless gate build the *identical* bag the
    /// shared [`crate::sim::SolverDriver`] consumes.
    ///
    /// Fluid properties are passed as primitives — the driver is `ui`-independent
    /// and never sees the `ui`-gated `Fluid`. `inlet_velocity` is taken from the
    /// defaults (`self.inlet_velocity`); headless sweeps that vary the inlet speed
    /// mutate the `ModelGuiDefaults` before calling this. `log_convergence` /
    /// `log_every_steps` default to the non-logging GUI worker values.
    pub fn to_runtime_params(&self, density: f32, viscosity: f32, eos: EosSpec) -> RuntimeParams {
        RuntimeParams {
            adaptive_dt: self.adaptive_dt,
            target_cfl: self.target_cfl,
            requested_dt: self.timestep as f32,
            // Pseudo-transient continuation only when the model enables dual time
            // (mirrors `current_runtime_params`).
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
        }
    }
}

/// Incompressible momentum (coupled SIMPLE) defaults.
///
/// Two levers: a low fixed outer cap (`outer_iters`) for cost — Ghia shows ~5
/// under-relaxed sweeps per step already give correct results, so a low cap
/// marches correctly at a fraction of the old fixed-50 cost — and a low
/// `inlet_velocity` so the *real* Air viscosity yields a laminar Reynolds number
/// (Air at 1 m/s is Re ~ 10^4-10^5 = turbulent, which a 2D laminar coarse-mesh
/// solver cannot represent and genuinely diverges on the obstacle wake).
const INCOMPRESSIBLE: ModelGuiDefaults = ModelGuiDefaults {
    advection_scheme: Scheme::Upwind,
    time_scheme: GpuTimeScheme::BDF2,
    preconditioner: PreconditionerType::Jacobi,
    alpha_u: 0.7,
    alpha_p: 0.3,
    // Few under-relaxed SIMPLE sweeps per step (Ghia validates 5; 8 keeps a little
    // headroom for the harder near-inviscid Air default) instead of a fixed 50 —
    // the user's "too slow" complaint was the outer count per step, and this low
    // cap is what delivers the speedup. `outer_auto_converge` keeps the monitor on
    // (GUI residual readout + opportunistic early exit) but is not relied upon: the
    // coupled break does not reliably fire for this model (see the gate's notes).
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
    // Real Air viscosity + a low inlet speed -> laminar Re (~25 on the obstacle),
    // honest and stable. The small velocity magnitudes are physically correct for
    // slow Air; the step count to develop is unchanged (CFL-limited).
    inlet_velocity: 0.002,
};

/// Compressible (density-based, implicit) defaults.
///
/// Van Leer reconstruction, a single outer iteration, Weiss-Smith low-Mach
/// preconditioning, a low inlet speed, and — the stabilizer — **pseudo-transient
/// continuation** (`dual_time`) started from a **uniform-freestream** initial
/// condition. Two failure modes have to be avoided:
///
/// * **Turbulence** — Air at 1 m/s is Re ~ 10^4 here too, so the low inlet speed
///   (laminar Re) is needed for the compressible solver just as for the
///   incompressible one.
/// * **Low-Mach inlet instability** — at Air's near-zero Mach, the time-accurate
///   density-based solver started from REST on the collocated cut-cell mesh is
///   unstable: the inlet injects momentum that, with no convective transport from
///   rest, piles up on the inlet-carrying cells and grows an odd/even pressure
///   mode without bound (it blows up slowly on the GPU and freezes the
///   deterministic f64 CPU solve). It is *not* fixed by the advection scheme,
///   `low_mach_theta_floor`, or `low_mach_pressure_coupling_alpha` (all verified
///   inert here). The fix is two ingredients that are each physically standard
///   and touch no solver code: (1) initialize at the uniform freestream so
///   convection is established everywhere (the through-flow IC the OpenFOAM
///   reference uses), and (2) `dual_time` (`dtau > 0`) to damp the residual
///   marginal mode and relax toward the quasi-steady solution.
///
/// The flow is genuinely slow (Mach ~ 1e-5), so it develops gradually toward the
/// steady recirculation; it stays bounded and smooth, which is what a default
/// must do. The validated MMS / lid / OpenFOAM-reference cases set their own
/// `dtau`/IC and are unaffected by these GUI defaults.
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
    // Pseudo-transient continuation: the low-Mach stabilizer for the cut-cell
    // through-flow default (see the field doc on `dual_time`).
    dual_time: true,
    dtau: 1e-3,
    low_mach_model: GpuLowMachPrecondModel::WeissSmith,
    low_mach_theta_floor: 1e-8,
    low_mach_pressure_coupling_alpha: 0.01,
    inlet_velocity: 0.002,
};

/// GUI solver defaults for `model_id`.
///
/// Unknown ids fall back to the incompressible defaults, which is the GUI's
/// startup model.
pub fn gui_defaults_for(model_id: &str) -> ModelGuiDefaults {
    match model_id {
        "compressible" => COMPRESSIBLE,
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
        assert_eq!(d.advection_scheme, Scheme::Upwind);
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
}
