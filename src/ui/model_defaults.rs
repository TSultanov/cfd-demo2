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
    /// Initial physical timestep, used as the seed before adaptive-dt takes over
    /// (adaptive growth is capped per step, so a small seed gives a gentle ramp).
    pub timestep: f64,
    pub low_mach_model: GpuLowMachPrecondModel,
    /// `f32` to match the GUI state and the `set_precond_*` solver setters.
    pub low_mach_theta_floor: f32,
    pub low_mach_pressure_coupling_alpha: f32,
    /// Lower bound applied to the effective fluid viscosity for this model
    /// (`None` keeps the fluid's own viscosity). The coarse-mesh compressible
    /// demonstrator is inviscidly unstable at the Air default (nu = 1.81e-5), so
    /// it floors the viscosity to keep the default case bounded without mutating
    /// the user-visible fluid preset.
    pub viscosity_floor: Option<f64>,
}

impl ModelGuiDefaults {
    /// Effective viscosity for `fluid_viscosity` under this model's floor.
    pub fn effective_viscosity(&self, fluid_viscosity: f64) -> f64 {
        match self.viscosity_floor {
            Some(floor) => fluid_viscosity.max(floor),
            None => fluid_viscosity,
        }
    }
}

/// Incompressible momentum (coupled SIMPLE) defaults.
///
/// Upwind keeps the high cell-Reynolds Air default well-damped. The fixed outer
/// cap (`outer_iters`) is the cost-control lever: Ghia shows ~5 under-relaxed
/// sweeps per step already give correct results, so a low cap marches correctly
/// at a fraction of the old fixed-50 cost.
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
    low_mach_model: GpuLowMachPrecondModel::Off,
    low_mach_theta_floor: 1e-6,
    low_mach_pressure_coupling_alpha: 1.0,
    viscosity_floor: None,
};

/// Compressible (density-based, implicit) defaults.
///
/// Van Leer reconstruction, a single outer iteration, and Weiss-Smith low-Mach
/// preconditioning with weak pressure coupling — as in the validated OpenFOAM
/// lid-cavity recipe — but tuned by the headless gate for unconditional stability
/// as a GUI default:
///
/// * `target_cfl = 0.3` is the **acoustic** CFL. The low-Mach machinery normally
///   relaxes the acoustic timestep so a near-incompressible flow can take a large
///   dt, but with a single outer iteration and a Jacobi preconditioner that
///   inflated dt diverges (the gate's sweep confirms it diverges at outer = 1, 4
///   *and* 8). So `low_mach_theta_floor = 1.0` keeps the adaptive timestep tied to
///   the *true* sound speed: the resulting acoustic CFL is then exactly
///   `target_cfl` regardless of the flow speed (with theta < 1 the dt tracks the
///   advective speed and the true acoustic CFL blows past 1 at low velocities).
///   The Weiss-Smith pressure coupling is retained for its checkerboard damping.
/// * `viscosity_floor = 0.05` keeps the coarse-mesh flow comfortably laminar
///   (domain Re ~ 12, cell Re ~ 0.6), below the inviscid-stability floor that the
///   near-inviscid Air default (nu = 1.81e-5) sits beneath.
///
/// Consequence: the physical timestep stays acoustically small (~2e-5 s), so a
/// near-incompressible flow develops slowly — but it does not diverge, which is
/// the property a default must guarantee.
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
    low_mach_model: GpuLowMachPrecondModel::WeissSmith,
    low_mach_theta_floor: 1.0,
    low_mach_pressure_coupling_alpha: 0.01,
    viscosity_floor: Some(0.05),
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
        assert!(d.viscosity_floor.is_none());
    }

    #[test]
    fn compressible_uses_low_mach_and_conservative_cfl() {
        let d = gui_defaults_for("compressible");
        assert_eq!(d.low_mach_model, GpuLowMachPrecondModel::WeissSmith);
        assert!(d.target_cfl <= 0.5, "acoustic CFL must be conservative");
        assert_eq!(d.advection_scheme, Scheme::SecondOrderUpwindVanLeer);
        assert_eq!(d.outer_iters, 1);
        assert!(d.low_mach_pressure_coupling_alpha < 1.0);
    }

    #[test]
    fn unknown_model_falls_back_to_incompressible() {
        assert_eq!(
            gui_defaults_for("nonexistent"),
            gui_defaults_for("incompressible_momentum")
        );
    }

    #[test]
    fn effective_viscosity_applies_floor() {
        let mut d = gui_defaults_for("compressible");
        d.viscosity_floor = Some(0.01);
        assert_eq!(d.effective_viscosity(1.81e-5), 0.01);
        assert_eq!(d.effective_viscosity(0.5), 0.5);
        d.viscosity_floor = None;
        assert_eq!(d.effective_viscosity(1.81e-5), 1.81e-5);
    }
}
