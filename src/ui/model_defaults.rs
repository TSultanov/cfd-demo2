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
    /// Dimensionless exaggeration of the **EOS-derived** all-Mach compressibility.
    /// The wire value is `psi = (1/c^2) * compressibility_exaggeration`, where
    /// `1/c^2 = Fluid::compressibility()` comes from the material's real EOS (sound
    /// speed). `1.0` = real physics; `0.0` forces the incompressible limit
    /// (`psi = 0`) regardless of the fluid. Values `>> 1` lower the *effective* sound
    /// speed so compressibility/supersonic become visible at the solver's stable
    /// laminar (low-velocity) operating point — at the real sound speed (~347 m/s for
    /// Air) these flows would need turbulent velocities, and the gauge-pressure field
    /// is numerically unstable (pressure undershoot through vacuum) at that scale.
    /// The multiply happens in `to_runtime_params` (which has the fluid's EOS).
    pub compressibility_exaggeration: f32,
    /// Gauge back-pressure pinned at the outlet. `0.0` for every standard case; the
    /// supersonic-nozzle demo uses a negative value to pull the diverging section
    /// past Mach 1 (the standard way a CD nozzle is driven). Applied to the
    /// gauge-pressure (`allmach_*`) models only.
    pub outlet_back_pressure: f32,
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
            // EOS-derived compressibility: psi = (1/c^2) * exaggeration. `1/c^2` is
            // the material's real isentropic compressibility (0 for an incompressible
            // `Constant` EOS, so those models stay byte-identical at psi=0).
            compressibility_psi: (eos.compressibility(density as f64)
                * self.compressibility_exaggeration as f64) as f32,
            outlet_back_pressure: self.outlet_back_pressure,
        }
    }
}

/// Incompressible momentum (coupled SIMPLE) defaults.
///
/// Three levers, all chosen so the **default** flow is both stable AND physically
/// interesting (the channel-obstacle case sheds a Kármán vortex street, the
/// headline demo):
///
/// * **Van Leer (TVD) advection** instead of first-order Upwind. On the coarse
///   cut-cell mesh (~8 cells across the cylinder) Upwind's numerical diffusion is
///   ~10x the real Air viscosity, which collapses the *effective* Reynolds number
///   below the shedding threshold and freezes the wake into a steady blob. Van Leer
///   is 2nd-order in the smooth wake, so the effective Re tracks the physical one
///   and the vortex street actually forms. Van Leer is TVD (bounded), and it is
///   verified stable on the cut-cell slivers (the no-slip immersed-wall BC supplies
///   the damping; see `generate_cut_cell_mesh`).
/// * A **laminar-but-shedding inlet speed**. Re = U·D/ν with D = 0.2, ν = μ/ρ ≈
///   1.48e-5 ⇒ U = 0.011 gives Re ≈ 150 — comfortably in the 2D-laminar
///   vortex-shedding band (onset ≈ 47, 3D transition ≈ 190). The coarse-mesh /
///   Van Leer numerical diffusion pulls the *effective* Re down to ≈ 110, the
///   textbook clean-street regime. (Air at 1 m/s would be Re ~ 10^4-10^5 =
///   turbulent, which a 2D laminar coarse-mesh solver cannot represent.) The same
///   speed keeps the backward-step laminar (channel Re ≈ 750, step Re ≈ 375; both
///   well below the ≈ 1200 step-flow transition) with a longer, more visible
///   recirculation bubble.
/// * A **low fixed outer cap** (`outer_iters`) for cost — Ghia shows ~5
///   under-relaxed SIMPLE sweeps per step already march correctly, so a low cap is
///   a fraction of the old fixed-50 cost.
const INCOMPRESSIBLE: ModelGuiDefaults = ModelGuiDefaults {
    // TVD Van Leer: low enough numerical diffusion that the coarse-mesh effective
    // Reynolds number stays in the shedding regime (Upwind smears the street away).
    advection_scheme: Scheme::SecondOrderUpwindVanLeer,
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
    // Real Air viscosity + this inlet speed -> Re ≈ 150 on the cylinder (D = 0.2):
    // the 2D-laminar vortex-shedding band. Honest (no viscosity floor), bounded
    // (verified on the cut-cell slivers), and physically interesting by default.
    inlet_velocity: 0.011,
    // Incompressible: force psi = 0 (the all-Mach pressure eqn reduces exactly to
    // incompressible; irrelevant for this model anyway). Factor 0 ⇒ psi 0 for any fluid.
    compressibility_exaggeration: 0.0,
    // Standard outlet (reference gauge pressure).
    outlet_back_pressure: 0.0,
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
    // Density-based compressible carries its own EOS; the all-Mach `psi` knob is
    // not used here (factor 0 ⇒ psi 0).
    compressibility_exaggeration: 0.0,
    outlet_back_pressure: 0.0,
};

/// All-Mach pressure-based (`allmach_pressure`) defaults.
///
/// This model is the incompressible coupled solver (gauge pressure, Rhie–Chow,
/// Schur) plus a `ddt(psi,p)` compressibility term and a per-cell variable density
/// `rho = rho_ref + psi*p`. It runs in the **incompressible** branch of the driver
/// (no `eos.gamma` → Coupled stepping), so the stable, validated incompressible
/// knobs apply verbatim — Van Leer TVD advection, the laminar inlet speed, the
/// convective adaptive dt, a low outer cap. The single addition is a **positive
/// compressibility** `psi`, now **derived from the fluid's real EOS**
/// (`psi = 1/c^2 = Fluid::compressibility()`) rather than a hardcoded constant. The
/// default `compressibility_exaggeration = 1.0` is honest physics — for Air
/// (c ≈ 347 m/s) `psi ≈ 8.3e-6`, so the default obstacle flow is near-incompressible
/// (inlet Mach ≈ 3e-5), exactly the validated incompressible Kármán street the
/// all-Mach model reduces to at small `psi`.
///
/// The real `psi` is acoustically **stiff** (the explicit `ddt(psi,p)` coupling has
/// sound speed `c = 1/√psi ≈ 347`, an acoustic CFL ≈ 667 at the convective dt → step-0
/// blow-up). The driver cures this with **low-Mach (Turkel) preconditioning**: a
/// decoupled pseudo-compressibility `psi_precond = max(psi, 1/(k·U_inlet)²)` that ONLY
/// the pressure-row time term reads (the density keeps the real `psi`), rescaling the
/// pseudo sound speed toward the local velocity so a convective dt is stable. It
/// vanishes at steady state, so the converged solution is the real-`psi` physics. This
/// is what makes `×1` a genuine, stable default rather than an exaggeration.
///
/// The GUI still exposes a **compressibility exaggeration** slider (×1 = real) so the
/// user can RAISE the real `psi` to visualize density variation / acoustics — the
/// regime that, at the real sound speed, would require turbulent velocities.
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
    // Coupled (incompressible-branch) stepping; no pseudo-transient — the gauge
    // pressure has no low-Mach inlet instability (that pathology is specific to the
    // density-based compressible solver's absolute-pressure state).
    dual_time: false,
    dtau: 1e-5,
    low_mach_model: GpuLowMachPrecondModel::Off,
    low_mach_theta_floor: 1e-6,
    low_mach_pressure_coupling_alpha: 1.0,
    inlet_velocity: 0.011,
    // REAL EOS physics (×1): psi = 1/c^2 from the fluid's sound speed. For Air
    // (c≈347 m/s) this is psi≈8.3e-6 ⇒ inlet Mach≈3e-5: the obstacle/backstep default
    // is honestly near-incompressible (the all-Mach model reduces to the validated
    // incompressible Kármán-street case). The GUI "compressibility exaggeration"
    // slider walks this up to visualize density variation — no artificial default.
    compressibility_exaggeration: 1.0,
    // Standard outlet for the channel/backstep cases (the nozzle demo overrides this).
    outlet_back_pressure: 0.0,
};

/// Exaggeration factor for the supersonic-nozzle demo, tuned for **Air**. Real Air
/// compressibility is `1/c^2 ≈ 8.3e-6`; this factor restores the validated effective
/// `psi ≈ 50` (effective `c ≈ 0.14 m/s`) at which the nozzle reaches a *stable*,
/// visible supersonic exit at laminar inlet speeds.
///
/// Unlike the obstacle (where low-Mach preconditioning makes the real `psi` stable),
/// the nozzle's exaggeration is load-bearing for a SEPARATE, physical reason that
/// preconditioning does NOT cure: at real `psi` the gauge-pressure EOS floor sits at
/// `p = -rho_ref/psi = -P_REF ≈ -1.5e5`, and driving a CHOKED nozzle supersonic pushes
/// the gauge pressure through that vacuum floor into NEGATIVE density (probed in
/// `allmach_real_units_nozzle_probe`) — an EOS-positivity limit, not an acoustic one.
/// The real-units alternative is also turbulent (Re≈4.5e7). The exaggeration lowers the
/// effective sound speed so `P_REF_eff = rho/psi ≈ 0.025` keeps the vacuum floor far
/// from the operating pressures while staying laminar.
const NOZZLE_EXAGGERATION: f32 = 6.027e6; // 50 / (1/347.2^2)

/// All-Mach thermal **supersonic nozzle** demo defaults.
///
/// Same gauge-pressure (incompressible-branch) knobs as [`ALLMACH`], but tuned for
/// the converging–diverging nozzle geometry to demonstrate SUPERSONIC throughflow:
///
/// * `inlet_velocity = 0.09` ≈ the natural choking inlet speed for the bundled
///   nozzle (`generate_structured_nozzle_mesh`, area ratio exit/throat = 2): the
///   inflow enters subsonic (inlet Mach ≈ 0.64·c… actually ≈ 0.09/0.1414 ≈ 0.64),
///   accelerates through the sonic throat (M ≈ 1), and continues into the diverging
///   section.
/// * `outlet_back_pressure = -0.045` (gauge) is the supersonic driver: lowering the
///   outlet pressure below critical pulls the diverging-section flow to a
///   SUPERSONIC exit (M_exit ≈ 1.07, validated in
///   `tests/allmach_thermal_supersonic_test.rs`). This is the stable edge of the
///   envelope — the gauge-pressure EOS hits its near-vacuum floor below ≈ −0.06.
///   NB: −0.045 is a *soft* Dirichlet target — the solved field relaxes well above it,
///   so the realized outlet absolute pressure stays POSITIVE (P_abs ≈ +0.007, never
///   vacuum) even though P_REF + p_back = −0.0205 is sub-vacuum on paper. The clean
///   accelerating-supersonic exit needs this pull; choking harder to raise the spec
///   above vacuum over-expands into a shock (throat M > exit M). Probed in
///   `tests/nozzle_interior_vacuum_probe.rs`.
/// * `compressibility_exaggeration = NOZZLE_EXAGGERATION` ⇒ effective `psi ≈ 50`,
///   effective sound speed `c ≈ 0.1414 m/s` — a stable, laminar, visible-supersonic
///   regime (real Air `c ≈ 347 m/s` would be turbulent AND numerically unstable; see
///   `NOZZLE_EXAGGERATION`).
///
/// Paired with the `allmach_thermal` model so the demo also shows the EXPANSION
/// COOLING (the `T` field drops to ≈ 0.75·T_ref as the gas accelerates).
pub const ALLMACH_THERMAL_NOZZLE: ModelGuiDefaults = ModelGuiDefaults {
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
    dual_time: false,
    dtau: 1e-5,
    low_mach_model: GpuLowMachPrecondModel::Off,
    low_mach_theta_floor: 1e-6,
    low_mach_pressure_coupling_alpha: 1.0,
    // Near the natural choking inlet speed for the bundled nozzle (at the EXAGGERATED
    // effective sound speed c≈0.14 m/s, so a laminar Re).
    inlet_velocity: 0.09,
    // EOS-derived 1/c^2 (Air) times the teaching exaggeration ⇒ effective psi≈50.
    compressibility_exaggeration: NOZZLE_EXAGGERATION,
    // The supersonic driver: a sub-critical (negative gauge) back-pressure.
    outlet_back_pressure: -0.045,
};

/// GUI solver defaults for `model_id`.
///
/// Unknown ids fall back to the incompressible defaults, which is the GUI's
/// startup model.
pub fn gui_defaults_for(model_id: &str) -> ModelGuiDefaults {
    match model_id {
        "compressible" => COMPRESSIBLE,
        // The thermal variant shares the gauge-pressure (incompressible-branch)
        // knobs; the supersonic-nozzle case overrides inlet speed + back-pressure
        // via `ALLMACH_THERMAL_NOZZLE` when the nozzle geometry is selected.
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
        // The all-Mach default no longer hardcodes psi: it defaults to REAL EOS
        // physics (exaggeration ×1), so the wire `psi` is `1/c^2` from the fluid's
        // sound speed. For Air (c≈347) that is ~8.3e-6 (honestly near-incompressible);
        // the visible-compressibility "cartoon" is the GUI exaggeration slider, not a
        // baked default.
        let d = gui_defaults_for("allmach_pressure");
        assert_eq!(
            d.compressibility_exaggeration, 1.0,
            "all-Mach default must be real EOS physics (×1), not an artificial psi"
        );
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
    fn nozzle_demo_exaggeration_restores_effective_psi_50_for_air() {
        // The supersonic nozzle keeps a *stable, visible* supersonic at laminar speeds
        // by exaggerating the (EOS-derived) compressibility back to the validated
        // effective psi≈50 — real units are turbulent + numerically unstable there.
        let air = EosSpec::IdealGas {
            gamma: 1.4,
            gas_constant: 287.0,
            temperature: 300.0,
        };
        let psi = ALLMACH_THERMAL_NOZZLE
            .to_runtime_params(1.225, 1.81e-5, air)
            .compressibility_psi as f64;
        assert!(
            (48.0..52.0).contains(&psi),
            "nozzle effective psi {psi:.2} should be ≈50 (validated supersonic regime)"
        );
        assert!(ALLMACH_THERMAL_NOZZLE.outlet_back_pressure < 0.0);
    }
}
