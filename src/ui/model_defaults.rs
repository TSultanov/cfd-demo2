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
    /// Physical timestep. With `adaptive_dt` it is the seed before adaptive-dt
    /// takes over; otherwise it is the fixed timestep.
    pub timestep: f64,
    /// Use the acoustic-aware adaptive timestep. Off for the compressible default
    /// (a fixed small dt + genuine low-Mach preconditioning is the validated,
    /// checkerboard-free recipe; the adaptive low-Mach dt inflation is unstable
    /// at a single outer iteration).
    pub adaptive_dt: bool,
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
/// preconditioning, and a low inlet speed. Two failure modes have to be avoided:
///
/// * **Turbulence** — Air at 1 m/s is Re ~ 10^4 here too, so the low inlet speed
///   (laminar Re) is needed for the compressible solver just as for the
///   incompressible one.
/// * **Low-Mach checkerboard** — collocated density-based solvers decouple
///   odd/even pressure at low Mach; `low_mach_theta_floor` / `target_cfl` /
///   `low_mach_pressure_coupling_alpha` are the levers (tuned by the gate). The
///   adaptive timestep stays acoustic-CFL-limited (the low-Mach dt inflation is
///   unstable at a single outer iteration).
///
/// The physical timestep is acoustically small, so a low-Mach flow develops
/// slowly — but it stays bounded and smooth, which is what a default must do.
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
