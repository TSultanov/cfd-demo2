//! Runtime solver knobs the driver applies to a live solver.
//!
//! Moved out of `src/ui/app.rs` so both the GUI worker and the headless tests
//! build the identical parameter bag. The canonical
//! `ModelGuiDefaults` → `RuntimeParams` mapping lives on
//! `ModelGuiDefaults::to_runtime_params` (`src/ui/model_defaults.rs`).

use crate::solver::model::eos::EosSpec;
use crate::solver::scheme::Scheme;
use crate::solver::{GpuLowMachPrecondModel, PreconditionerType, TimeScheme as GpuTimeScheme};

/// The full set of runtime-tunable solver knobs.
///
/// Every field is a primitive (or a `Copy` solver enum) so the bag is `Copy`
/// and carries no `ui` dependency — fluid properties arrive as raw
/// `density`/`viscosity`/`eos`, never as the `ui`-gated `Fluid`.
#[derive(Clone, Copy)]
pub struct RuntimeParams {
    pub adaptive_dt: bool,
    pub target_cfl: f64,
    pub requested_dt: f32,
    pub dtau: f32,
    pub log_convergence: bool,
    pub log_every_steps: u32,
    pub advection_scheme: Scheme,
    pub time_scheme: GpuTimeScheme,
    pub preconditioner: PreconditionerType,
    pub outer_iters: u32,
    pub outer_auto_converge: bool,
    pub low_mach_model: GpuLowMachPrecondModel,
    pub low_mach_theta_floor: f32,
    pub low_mach_pressure_coupling_alpha: f32,
    pub alpha_u: f32,
    pub alpha_p: f32,
    pub inlet_velocity: f32,
    pub density: f32,
    pub viscosity: f32,
    pub eos: EosSpec,
    /// All-Mach compressibility `psi = d(rho)/d(p) = 1/c^2` (units Density/Pressure).
    /// Only the `allmach_pressure` model reads it (seeded into its per-cell `psi`
    /// state field); `0.0` is the incompressible limit. Other models ignore it.
    pub compressibility_psi: f32,
    /// Gauge back-pressure pinned at the outlet (units Pressure). `0.0` is the
    /// default (outlet at reference pressure — the validated incompressible/all-Mach
    /// behaviour). A NEGATIVE value drives a converging–diverging nozzle SUPERSONIC:
    /// lowering the outlet pressure below critical pulls the diverging-section flow
    /// past Mach 1 (see the supersonic-nozzle demo). Applied by the driver to the
    /// gauge-pressure (`allmach_*`) models only.
    pub outlet_back_pressure: f32,
}
