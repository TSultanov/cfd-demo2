//! Runtime solver knobs the driver applies to a live solver.
//!
//! Shared by the GUI worker and the headless tests so both build the identical
//! parameter bag. The canonical `ModelGuiDefaults` → `RuntimeParams` mapping lives
//! on `ModelGuiDefaults::to_runtime_params` (`src/ui/model_defaults.rs`).

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
    /// All-Mach compressibility `psi = d(rho)/d(p) = 1/c^2` (units Density/Pressure),
    /// seeded into the `allmach_*` models' per-cell `psi` field. EOS-derived: the
    /// GUI fills it as `(1/c^2) * exaggeration`, where `1/c^2`
    /// comes from the fluid's real sound speed (`Fluid::compressibility`) and
    /// `exaggeration` is the dimensionless GUI factor (×1 = real physics). `0.0` is the
    /// incompressible limit (an incompressible `Constant` EOS gives `1/c^2 = 0`). Other
    /// models ignore it.
    pub compressibility_psi: f32,
    /// Gauge back-pressure pinned at the outlet (units Pressure). `0.0` is the
    /// default (outlet at reference pressure — the validated incompressible/all-Mach
    /// behaviour). A NEGATIVE value drives a converging–diverging nozzle SUPERSONIC:
    /// lowering the outlet pressure below critical pulls the diverging-section flow
    /// past Mach 1 (see the supersonic-nozzle demo). Applied by the driver to the
    /// gauge-pressure (`allmach_*`) models only.
    pub outlet_back_pressure: f32,
    /// Minimum preconditioner reference velocity for the all-Mach low-Mach
    /// preconditioner (units Velocity). Floors the pseudo-sound `beta = k*U_ref`
    /// so a pathologically-slow near-incompressible inlet's low-Mach pressure
    /// mode convects out instead of standing (see `allmach_psi_precond` in
    /// `driver.rs`). `0.2` is the conservative default that stops the
    /// compressible-ALE outlet divergence on every mesh; RAISING it (toward ~1.0)
    /// shrinks the residual standing pseudo-acoustic pressure mode toward the
    /// incompressible field (the "pressure looks wrong" cure), at the cost of
    /// step-0 startup damping — which the moving driver's startup dt growth-cap
    /// (`flow_adaptive_dt`) restores, so a higher value is safe. Ignored by
    /// non-all-Mach models. Env override: `ALLMACH_PRECOND_UREF_MIN`.
    pub allmach_precond_uref_min: f32,
    /// Drive the all-Mach CD nozzle with a PRESSURE INLET + SUPERSONIC (extrapolated)
    /// OUTLET instead of the default velocity-inlet / pressure-outlet. When true, the
    /// driver flips the Inlet/Outlet boundary KINDS (via
    /// `apply_pressure_inlet_nozzle_bcs`): pins the inlet gauge pressure
    /// ([`Self::inlet_pressure`], the new gauge anchor) and lets the outlet float
    /// (no back-pressure). The throughflow speed is then a RESULT of the pressure drop;
    /// [`Self::inlet_velocity`] is kept only as the preconditioner / CFL scale. Applied
    /// to the `allmach_*` models only; ignored otherwise.
    pub pressure_inlet: bool,
    /// Inlet gauge pressure pinned when [`Self::pressure_inlet`] is set (units Pressure).
    /// Higher → stronger drop → faster throughflow. (Ignored unless `pressure_inlet`.)
    pub inlet_pressure: f32,
}
