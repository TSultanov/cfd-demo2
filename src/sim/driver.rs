//! The shared solver driver.
//!
//! `SolverDriver` owns a [`UnifiedSolver`] plus the runtime knobs and the small
//! amount of scalar context the driver layer needs (min cell size, sound-speed
//! support, the compressible/incompressible split, and the last max velocity that
//! feeds the adaptive timestep). It is the single home for the logic the GUI
//! worker and the headless tests used to each hand-copy:
//!
//! * [`build`](SolverDriver::build) — derive the [`SolverConfig`] (stepping mode +
//!   preconditioner), construct the solver, apply the phase-1 setters, and set the
//!   initial / boundary conditions (the compressible-vs-incompressible branch).
//! * [`apply_params`](SolverDriver::apply_params) — the phase-2 live parameter
//!   application (was `solver_worker_apply_params`).
//! * [`step`](SolverDriver::step) — the acoustic-aware adaptive timestep, one
//!   `step_with_stats`, and divergence / steady-state detection, returned as a
//!   [`StepOutcome`]. GUI-only concerns (viz upload, channel publishing, trace) stay
//!   in the worker, which calls `step` and reacts to the outcome.
//! * [`run_steps`](SolverDriver::run_steps) — a thin headless driving loop for tests.

use std::ops::ControlFlow;

use super::{DivergeReason, FieldStats, Readback, RunResult, RuntimeParams, StepOutcome};
use crate::solver::mesh::Mesh;
use crate::solver::model::helpers::{
    SolverCompressibleIdealGasExt, SolverCompressibleInletExt, SolverFieldAliasesExt,
    SolverIncompressibleStatsExt, SolverInletVelocityExt, SolverRuntimeParamsExt,
};
use crate::solver::model::ModelSpec;
use crate::solver::{
    GpuLowMachPrecondModel, PreconditionerType, SolverConfig, SteppingMode, UnifiedSolver,
};

/// Mirrors the GUI's `UI_MAX_BLOCK_JACOBI`: a model with more unknowns/cell than
/// this falls back to a Jacobi preconditioner even if BlockJacobi was requested.
const UI_MAX_BLOCK_JACOBI: u32 = 16;

/// A backend-agnostic, GUI-independent driver around [`UnifiedSolver`].
pub struct SolverDriver {
    solver: UnifiedSolver,
    params: RuntimeParams,
    /// `model().named_param_keys()` cached at build (drives the `has_param` gating
    /// in [`apply_params`](SolverDriver::apply_params)).
    named_params: Vec<&'static str>,
    /// `sqrt(min cell volume)` — the adaptive-dt length scale.
    min_cell_size: f64,
    /// The model exposes `eos.gamma` (so it has a meaningful sound speed).
    supports_sound_speed: bool,
    /// The model carries `rho`/`rho_u`/`rho_e`/`u` (density-based compressible).
    compressible: bool,
    /// The model is `allmach_pressure` (pressure-based all-Mach): it carries the
    /// extra `psi`/`rho`/`dt_local` state fields this driver seeds at build, keeps
    /// `psi` live in [`apply_params`](SolverDriver::apply_params), and refreshes
    /// `rho = rho_ref + psi*p` from the gauge pressure on each *readback* — i.e. at the
    /// caller's readback cadence (the GUI's ~100ms snapshot interval, not every step),
    /// with history-preserving current-buffer writes. Between refreshes `rho` lags `p`;
    /// that is a bounded low-Mach approximation (constant `rho` was independently
    /// validated stable — the stabilization is the implicit `ddt(psi,p)` diagonal, not
    /// the density coupling). The proper fix is an on-device `rho` refresh kernel.
    allmach: bool,
    /// Last observed max velocity (from a readback); feeds the adaptive timestep.
    prev_max_vel: f64,
}

/// The product of [`SolverDriver::build`]: the driver plus the initial field
/// snapshot it applied (so the caller can seed its first render without
/// re-deriving the initial condition).
pub struct DriverBuild {
    pub driver: SolverDriver,
    pub cached_u: Vec<(f64, f64)>,
    pub cached_p: Vec<f64>,
}

/// Pseudo sound-speed floor multiplier on the convective scale (`k` in
/// `beta_min = k * U_inlet`): holds the step-0 / quiescent-region acoustic CFL near
/// `k * target_cfl ~ O(1)` instead of the unpreconditioned `c * dt / h ~ 667`.
const ALLMACH_PRECOND_MACH_K: f64 = 2.0;

/// Absolute-pressure floor for the all-Mach EOS (Pascals, gauge-referenced as
/// `P_abs = P_REF + p`). The barotropic density is `rho = psi * P_abs`, so a transient
/// gauge-pressure undershoot below `-P_REF` would drive `P_abs < 0` and `rho <= 0`,
/// breaking every term that divides by density. Clamping `P_abs >= ABS_PRESSURE_FLOOR`
/// (equivalently `rho >= psi * ABS_PRESSURE_FLOOR`) holds the EOS at a tiny positive
/// density so a temporary negative-pressure numerical artifact stays well-posed and the
/// solve can recover, rather than blowing up. A near-vacuum floor (1e-5 Pa): inert
/// wherever the pressure is physical.
const ALLMACH_ABS_PRESSURE_FLOOR: f64 = 1.0e-5;

/// Low-Mach preconditioned pseudo-compressibility for the all-Mach pressure model.
///
/// Returns the per-cell `psi_precond` the pressure-row `ddt` term consumes, decoupled
/// from the physical `psi` (=1/c^2) that drives the density recovery. The pseudo sound
/// speed is rescaled toward the local velocity (Turkel-style low-Mach preconditioning):
///
/// ```text
///   beta^2      = max(|U|^2, (k*U_ref)^2)     // pseudo-sound-speed^2 (a velocity^2)
///   psi_precond = max(real_psi, 1/beta^2)     // >= physical psi (never less compressible)
/// ```
///
/// Because `beta >= |U|` the pseudo-Mach is <= 1, so the acoustic CFL tracks the
/// (bounded) convective CFL and a convective timestep is acoustically stable — curing
/// the real-`psi` step-0 blow-up (acoustic CFL ~667). The `max` self-disables it where
/// the REAL Mach >= 1 (`1/|U|^2 <= real_psi` => `psi_precond = real_psi` => full physical
/// acoustics). When `real_psi == 0` (incompressible limit) it returns 0, keeping the
/// model byte-identical to the incompressible solver (the acoustic term vanishes).
fn allmach_psi_precond(u: &[(f64, f64)], real_psi: f64, u_ref: f64) -> Vec<f64> {
    if real_psi <= 0.0 {
        return vec![0.0; u.len()];
    }
    let conv_floor2 = (ALLMACH_PRECOND_MACH_K * u_ref.abs()).powi(2);
    u.iter()
        .map(|&(vx, vy)| {
            // 1e-12 only guards beta->0 in the degenerate closed-box (U_ref=0, |U|=0) case;
            // for every inlet-driven demo the convective floor dominates.
            let beta2 = (vx * vx + vy * vy).max(conv_floor2).max(1e-12);
            real_psi.max(1.0 / beta2)
        })
        .collect()
}

impl SolverDriver {
    /// Build a configured solver (phase 1).
    ///
    /// Mirrors the solver portion of the GUI's `build_init_outcome`: it derives the
    /// `SolverConfig` (the `eos.gamma ? Implicit{1} : Coupled` stepping rule and the
    /// BlockJacobi→Jacobi clamp), constructs the [`UnifiedSolver`], clears state,
    /// applies the phase-1 setters, and sets the initial + boundary conditions. The
    /// **compressible** branch uses a uniform-freestream initial condition derived
    /// from `params` and ignores `initial_u`/`initial_p`; the **incompressible**
    /// branch seeds `initial_u`/`initial_p` (the GUI passes its geometry IC — which
    /// is currently rest — and the tests pass rest).
    ///
    /// Phase-2 knobs (dtau, outer_iters, low-Mach, under-relaxation,
    /// convergence-stats collection) are **not** applied here — call
    /// [`apply_params`](SolverDriver::apply_params) after building, exactly as the
    /// GUI applies them via `sync_worker_params` after `SetSolver`.
    #[allow(clippy::too_many_arguments)]
    pub async fn build(
        mesh: &Mesh,
        mut model: ModelSpec,
        params: &RuntimeParams,
        initial_u: &[(f64, f64)],
        initial_p: &[f64],
        device: Option<wgpu::Device>,
        queue: Option<wgpu::Queue>,
    ) -> Result<DriverBuild, String> {
        let named_params = model.named_param_keys();
        let supports_preconditioner = named_params.iter().any(|&k| k == "preconditioner");
        let supports_sound_speed = named_params.iter().any(|&k| k == "eos.gamma");
        let unknowns_per_cell = model.system.unknowns_per_cell();

        // Preconditioner: clamp BlockJacobi to Jacobi for wide blocks, and force
        // Jacobi when the model doesn't expose a preconditioner knob (matches the
        // GUI's `effective_preconditioner`).
        let mut selected_preconditioner = params.preconditioner;
        if matches!(selected_preconditioner, PreconditionerType::BlockJacobi)
            && unknowns_per_cell > UI_MAX_BLOCK_JACOBI
        {
            selected_preconditioner = PreconditionerType::Jacobi;
        }
        let effective_preconditioner = if supports_preconditioner {
            selected_preconditioner
        } else {
            PreconditionerType::Jacobi
        };

        let config = SolverConfig {
            advection_scheme: params.advection_scheme,
            time_scheme: params.time_scheme,
            preconditioner: effective_preconditioner,
            // EOS-aware (compressible) models run a single implicit outer iteration;
            // the saddle-point (incompressible) models run the coupled solver.
            stepping: if supports_sound_speed {
                SteppingMode::Implicit { outer_iters: 1 }
            } else {
                SteppingMode::Coupled
            },
        };

        // Pressure-inlet CD nozzle: flip the all-Mach Inlet/Outlet boundary KINDS to a
        // pressure inlet + supersonic (extrapolated) outlet BEFORE the solver bakes the
        // bc_table. The model id / committed kernels are untouched (BC kind is a runtime
        // table, not a kernel). Only the all-Mach nozzle preset sets this.
        if params.pressure_inlet {
            crate::solver::model::apply_pressure_inlet_nozzle_bcs(&mut model);
        }

        let mut solver = UnifiedSolver::new(mesh, model, config, device, queue).await?;

        let n_cells = mesh.num_cells();
        let stride = solver.model().state_layout.stride() as usize;
        let _ = solver.write_state_f32(&vec![0.0f32; n_cells * stride]);
        solver.set_dt(params.requested_dt);
        let _ = solver.set_viscosity(params.viscosity);
        solver.set_advection_scheme(params.advection_scheme);
        solver.set_time_scheme(params.time_scheme);
        let _ = solver.set_eos(&params.eos);
        if supports_preconditioner {
            solver.set_preconditioner(effective_preconditioner);
        }

        // Compressible iff the model carries the full conservative state.
        let mut has_rho = false;
        let mut has_rho_u = false;
        let mut has_rho_e = false;
        let mut has_u = false;
        for eqn in solver.model().system.equations() {
            match eqn.target().name() {
                "rho" => has_rho = true,
                "rho_u" => has_rho_u = true,
                "rho_e" => has_rho_e = true,
                "u" => has_u = true,
                _ => {}
            }
        }
        let compressible = has_rho && has_rho_u && has_rho_e && has_u;
        // All-Mach pressure-based models: run in the incompressible (Coupled) branch
        // but carry extra `psi`/`rho`/`dt_local` state fields to seed. The `thermal`
        // variant additionally carries a temperature `T` and its EOS reference
        // `rho_t_ref` (the on-device density recovery `rho = rho_t_ref/T + psi*p`).
        let model_id = solver.model().id;
        let allmach = model_id == "allmach_pressure" || model_id == "allmach_thermal";
        let thermal = model_id == "allmach_thermal";

        let (cached_u, cached_p) = if compressible {
            let p_ref = params.eos.pressure_for_density(params.density as f64);
            let _ = solver.set_density(params.density);
            let _ = solver.set_compressible_inlet_isothermal_x(
                params.density,
                params.inlet_velocity,
                &params.eos,
            );
            // Uniform-freestream IC (matching the inlet), not rest: from rest the
            // inlet-injected momentum has no convective transport on the collocated
            // cut-cell mesh and seeds the low-Mach inlet instability.
            let u0 = params.inlet_velocity;
            solver.set_uniform_state(params.density, [u0, 0.0], p_ref as f32);
            (vec![(u0 as f64, 0.0); n_cells], vec![p_ref; n_cells])
        } else {
            let _ = solver.set_density(params.density);
            let _ = solver.set_alpha_u(params.alpha_u);
            let _ = solver.set_alpha_p(params.alpha_p);
            let _ = solver.set_inlet_velocity(params.inlet_velocity);
            solver.set_u(initial_u);
            solver.set_p(initial_p);
            if params.pressure_inlet {
                // Pressure-inlet nozzle: seed a linear gauge-pressure ramp
                // inlet_pressure -> 0 (inlet -> outlet) so step 0 already carries the
                // driving gradient, instead of launching an acoustic pulse from a flat
                // field. Uniform-axial U (set_u above, from the caller's IC) completes
                // the near-steady start. Cuts the start-up transient (validated in the
                // pressure-inlet probe).
                let x_min = mesh.cell_cx.iter().cloned().fold(f64::INFINITY, f64::min);
                let x_max = mesh
                    .cell_cx
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                let span = (x_max - x_min).max(1e-12);
                let p0 = params.inlet_pressure as f64;
                let ramp: Vec<f64> = (0..n_cells)
                    .map(|c| p0 * (1.0 - (mesh.cell_cx[c] - x_min) / span))
                    .collect();
                let _ = solver.set_field_scalar("p", &ramp);
            }
            // All-Mach: seed the extra state fields the bare incompressible path has
            // no concept of. `psi` (compressibility = 1/c^2) activates the
            // `ddt(psi,p)` term and sets the Mach regime; `rho` MUST start at the
            // reference density (0 would break the Rhie–Chow mass flux and the
            // `ddt(rho,U)` coefficient — it is then refreshed to `rho_ref + psi*p`
            // each readback); `dt_local = 0` selects the global (time-accurate) dt.
            // `set_field_scalar` (initial-condition semantics, writes all history
            // buffers) is intentional HERE — this is the IC, and `initialize_history`
            // below re-propagates it; the mid-run refreshes use `_current` instead.
            if allmach {
                let psi = params.compressibility_psi.max(0.0) as f64;
                let _ = solver.set_field_scalar("psi", &vec![psi; n_cells]);
                // Low-Mach preconditioned pseudo-compressibility (ddt-only), seeded from
                // the IC velocity floored at k*U_inlet so step 0 is acoustic-CFL ~O(1),
                // not ~667 — the cure for the real-psi step-0 blow-up. `set_field_scalar`
                // (IC semantics) matches the other seeds here; readback refreshes per-cell.
                let psi_precond =
                    allmach_psi_precond(initial_u, psi, params.inlet_velocity.abs() as f64);
                let _ = solver.set_field_scalar("psi_precond", &psi_precond);
                let _ = solver.set_field_scalar("rho", &vec![params.density as f64; n_cells]);
                let _ = solver.set_field_scalar("dt_local", &vec![0.0; n_cells]);
                // Thermal variant: seed the temperature at the reference and the
                // constant EOS reference `rho_t_ref = rho_ref * T_ref`. The density
                // is recovered on-device as `rho = rho_t_ref/T + psi*p`; an unseeded
                // (0) `rho_t_ref` would make `rho` blow up. Matches the manual seeding
                // in the thermal validation tests.
                if thermal {
                    let t_ref = crate::solver::model::ALLMACH_T_REF;
                    let _ = solver
                        .set_field_scalar("rho_t_ref", &vec![params.density as f64 * t_ref; n_cells]);
                    let _ = solver.set_field_scalar("T", &vec![t_ref; n_cells]);
                    // EOS density floor = psi * absolute-pressure floor (rho = psi*P_abs),
                    // so the on-device recovery clamps rho positive against a transient
                    // gauge-pressure undershoot through vacuum. Constant field; refreshed
                    // on a psi (slider) change in apply_params.
                    let _ = solver.set_field_scalar(
                        "rho_floor",
                        &vec![psi * ALLMACH_ABS_PRESSURE_FLOOR; n_cells],
                    );
                }
            }
            (initial_u.to_vec(), initial_p.to_vec())
        };
        solver.initialize_history();

        let min_cell_size = mesh
            .cell_vol
            .iter()
            .map(|&v| v.sqrt())
            .fold(f64::INFINITY, f64::min);

        Ok(DriverBuild {
            driver: SolverDriver {
                solver,
                params: *params,
                named_params,
                min_cell_size,
                supports_sound_speed,
                compressible,
                allmach,
                prev_max_vel: 0.0,
            },
            cached_u,
            cached_p,
        })
    }

    /// Apply the runtime parameters to the live solver (phase 2).
    ///
    /// Byte-for-byte the GUI's `solver_worker_apply_params`: enable the convergence
    /// monitor, push `dt`, then the `has_param`-gated setters, then re-apply the
    /// inlet boundary condition (compressible vs incompressible).
    pub fn apply_params(&mut self, params: &RuntimeParams) {
        self.params = *params;
        let solver = &mut self.solver;
        // Outer-convergence monitoring drives the GUI residual readout and the
        // opportunistic per-step break.
        solver.set_collect_convergence_stats(params.outer_auto_converge || params.log_convergence);
        solver.set_dt(params.requested_dt);

        let named = &self.named_params;
        let has_param = |key: &str| named.iter().any(|&k| k == key);

        if has_param("dtau") {
            let _ = solver.set_dtau(params.dtau);
        }
        if has_param("density") {
            let _ = solver.set_density(params.density);
        }
        if has_param("viscosity") {
            let _ = solver.set_viscosity(params.viscosity);
        }
        if has_param("eos.gamma") {
            let _ = solver.set_eos(&params.eos);
        }
        if has_param("outer_iters") {
            let _ = solver.set_outer_iters(params.outer_iters as usize);
        }
        if has_param("low_mach.model") {
            let _ = solver.set_precond_model(params.low_mach_model);
        }
        if has_param("low_mach.theta_floor") {
            let _ = solver.set_precond_theta_floor(params.low_mach_theta_floor);
        }
        if has_param("low_mach.pressure_coupling_alpha") {
            let _ = solver.set_precond_pressure_coupling_alpha(params.low_mach_pressure_coupling_alpha);
        }
        if has_param("advection_scheme") {
            solver.set_advection_scheme(params.advection_scheme);
        }
        if has_param("time_scheme") {
            solver.set_time_scheme(params.time_scheme);
        }
        if has_param("preconditioner") {
            solver.set_preconditioner(params.preconditioner);
        }
        if has_param("alpha_u") {
            let _ = solver.set_alpha_u(params.alpha_u);
        }
        if has_param("alpha_p") {
            let _ = solver.set_alpha_p(params.alpha_p);
        }

        if self.compressible {
            let _ = solver.set_compressible_inlet_isothermal_x(
                params.density,
                params.inlet_velocity,
                &params.eos,
            );
        } else {
            let _ = solver.set_inlet_velocity(params.inlet_velocity);
        }

        // All-Mach: keep the compressibility `psi` live so the GUI slider takes
        // effect without a rebuild. `_current` (not set_field_scalar) updates only the
        // current buffer, preserving the BDF2 history — `psi` is the ddt(psi,p)
        // coefficient at the current time, never read from history. `rho` is refreshed
        // from the new `psi` on the next readback (≤ one snapshot interval); `dt_local`
        // stays 0 (global time-accurate dt).
        if self.allmach {
            let n = solver.num_cells() as usize;
            let psi = params.compressibility_psi.max(0.0) as f64;
            let _ = solver.set_field_scalar_current("psi", &vec![psi; n]);
            // Keep psi_precond consistent with the new psi using the last-known velocity
            // scale (uniform); the next readback refreshes it per-cell. Ensures the ddt
            // coefficient stays >= the physical psi after a slider change.
            let psi_precond =
                allmach_psi_precond(&vec![(self.prev_max_vel, 0.0); n], psi, params.inlet_velocity.abs() as f64);
            let _ = solver.set_field_scalar_current("psi_precond", &psi_precond);
            // Keep the EOS density floor (= psi * absolute-pressure floor) consistent with
            // the new psi. A no-op for non-thermal / non-allmach (no `rho_floor` field).
            let _ = solver.set_field_scalar_current(
                "rho_floor",
                &vec![psi * ALLMACH_ABS_PRESSURE_FLOOR; n],
            );
            if params.pressure_inlet {
                // Pressure-inlet nozzle: pin the INLET gauge pressure (the gauge anchor
                // moved upstream by `apply_pressure_inlet_nozzle_bcs`). The outlet `p` is
                // now ZeroGradient, so it floats (supersonic outlet, no back-pressure).
                let _ = solver.set_boundary_scalar(
                    crate::solver::gpu::enums::GpuBoundaryType::Inlet,
                    "p",
                    params.inlet_pressure,
                );
            } else {
                // Outlet gauge back-pressure: pins the outlet `p` Dirichlet value. `0.0`
                // is the standard outlet (reference pressure); a negative value drives a
                // converging–diverging nozzle supersonic. Live so the GUI slider / a
                // per-case default takes effect without a rebuild.
                let _ = solver.set_boundary_scalar(
                    crate::solver::gpu::enums::GpuBoundaryType::Outlet,
                    "p",
                    params.outlet_back_pressure,
                );
            }
        }
    }

    /// Advance one timestep and report the outcome.
    ///
    /// Computes the timestep (acoustic-aware adaptive — true sound speed reduced by
    /// the low-Mach preconditioning floor — or the fixed `requested_dt`), runs one
    /// `step_with_stats`, and classifies divergence / steady-state. When `readback`
    /// is true it reads the fields, updates the adaptive-dt velocity scale, and (if
    /// the fields went non-finite) reports it as divergence. The numbers are
    /// identical to the GUI worker's inline loop; only the GUI-only side effects
    /// (viz upload, publishing, trace) are left to the caller.
    pub fn step(&mut self, readback: bool) -> StepOutcome {
        if self.params.adaptive_dt {
            let sound_speed = if self.supports_sound_speed {
                self.params.eos.sound_speed(self.params.density as f64)
            } else {
                0.0
            };
            let adv_speed = self.prev_max_vel.max(self.params.inlet_velocity.abs() as f64);
            let effective_sound_speed = match self.params.low_mach_model {
                GpuLowMachPrecondModel::Off => sound_speed,
                GpuLowMachPrecondModel::Legacy => sound_speed.min(adv_speed),
                GpuLowMachPrecondModel::WeissSmith => {
                    let theta = (self.params.low_mach_theta_floor as f64).max(0.0);
                    let c_floor = sound_speed * theta.sqrt();
                    sound_speed.min(adv_speed.max(c_floor))
                }
            };
            let wave_speed = adv_speed + effective_sound_speed;
            if self.min_cell_size > 1e-12 && wave_speed.is_finite() && wave_speed > 1e-12 {
                let current_dt = self.solver.dt() as f64;
                let mut next_dt = self.params.target_cfl * self.min_cell_size / wave_speed;
                if next_dt > current_dt * 1.2 {
                    next_dt = current_dt * 1.2;
                }
                next_dt = next_dt.clamp(1e-9, 100.0);
                self.solver.set_dt(next_dt as f32);
            }
        } else {
            self.solver.set_dt(self.params.requested_dt);
        }

        let step_start = std::time::Instant::now();
        let step_result = self.solver.step_with_stats();
        let step_time_ms = step_start.elapsed().as_secs_f32() * 1000.0;
        let linear_stats = match step_result {
            Ok(stats) => stats,
            Err(err) => {
                let dt = self.solver.dt();
                return StepOutcome {
                    dt,
                    step_time_ms,
                    linear_stats: Vec::new(),
                    outer_iters: None,
                    diverged: Some(DivergeReason::StepError(err)),
                    should_stop: false,
                    readback: None,
                };
            }
        };
        let dt = self.solver.dt();
        let should_stop = self.solver.incompressible_should_stop();
        let outer_iters = self.solver.step_stats().outer_iterations;

        let linear_diverged = linear_stats
            .iter()
            .any(|s| s.diverged || !s.residual.is_finite() || s.residual > 1e12);
        let mut diverged = if linear_diverged {
            Some(DivergeReason::LinearSolver)
        } else {
            None
        };

        let readback = if readback {
            let rb = self.read_back();
            if diverged.is_none() && (rb.stats.nonfinite_u > 0 || rb.stats.nonfinite_p > 0) {
                diverged = Some(DivergeReason::NonFinite {
                    u: rb.stats.nonfinite_u,
                    p: rb.stats.nonfinite_p,
                });
            }
            Some(rb)
        } else {
            None
        };

        StepOutcome {
            dt,
            step_time_ms,
            linear_stats,
            outer_iters,
            diverged,
            should_stop,
            readback,
        }
    }

    /// Read the velocity/pressure (and, for compressible models, density) fields,
    /// returning the raw fields plus their summary and updating the adaptive-dt
    /// velocity scale. Matches the GUI worker's readback block.
    fn read_back(&mut self) -> Readback {
        let u = pollster::block_on(self.solver.get_u());
        let p = pollster::block_on(self.solver.get_p());

        let mut max_vel = 0.0_f64;
        let mut nonfinite_u = 0usize;
        for (vx, vy) in &u {
            if !(vx.is_finite() && vy.is_finite()) {
                nonfinite_u += 1;
                continue;
            }
            let v = (vx * vx + vy * vy).sqrt();
            if v > max_vel {
                max_vel = v;
            }
        }
        self.prev_max_vel = max_vel;

        let mut p_min = f64::INFINITY;
        let mut p_max = f64::NEG_INFINITY;
        let mut nonfinite_p = 0usize;
        for &pv in &p {
            if !pv.is_finite() {
                nonfinite_p += 1;
                continue;
            }
            p_min = p_min.min(pv);
            p_max = p_max.max(pv);
        }

        let rho = if self.allmach {
            // Barotropic density from the gauge pressure: `rho = rho_ref + psi*p`,
            // floored positive (a non-positive density would break the viscous /
            // flux terms). Refresh the per-cell `rho` the Rhie–Chow flux and the
            // `ddt(rho,U)` coefficient read, and report its spread — the
            // compressibility signature the GUI density view and the gate observe.
            let psi = self.params.compressibility_psi.max(0.0) as f64;
            let rho_ref = self.params.density as f64;
            // Absolute-pressure floor: rho = psi*P_abs, so clamping rho >= psi*P_FLOOR
            // holds P_abs >= P_FLOOR (a tiny positive floor) against a transient gauge
            // pressure undershoot below -P_REF (which would give rho <= 0 and blow up).
            // Mirrors the on-device `rho_floor` clamp used by the thermal recovery.
            let floor = psi * ALLMACH_ABS_PRESSURE_FLOOR;
            let rho_vals: Vec<f64> = p.iter().map(|&pv| (rho_ref + psi * pv).max(floor)).collect();
            let lo = rho_vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = rho_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            // `_current` (NOT set_field_scalar): write rho into the CURRENT buffer only,
            // preserving the BDF2 time history. set_field_scalar writes all three
            // ping-pong buffers (initial-condition semantics) — mid-run that would reset
            // U/p history (old = current) on every readback, zeroing the velocity
            // time-derivative, making the GUI transient non-deterministic (readback is
            // wall-clock throttled) and divergent between the GPU (history reset) and CPU
            // (history preserved) backends. rho is a coefficient sampled at the current
            // time (ddt(rho,U) uses U_old, never rho_old), so a current-only update is
            // exact and backend-consistent.
            let _ = self.solver.set_field_scalar_current("rho", &rho_vals);
            // Refresh the preconditioned pseudo-compressibility from the live velocity
            // (one-snapshot lag, same discipline + correctness argument as `rho`: a
            // current-time coefficient never read from BDF history). Tracks the wake so
            // the pseudo-Mach stays ~<=1 as the flow develops.
            let psi_precond =
                allmach_psi_precond(&u, psi, self.params.inlet_velocity.abs() as f64);
            let _ = self.solver.set_field_scalar_current("psi_precond", &psi_precond);
            Some((lo, hi))
        } else if self.compressible {
            let rho = pollster::block_on(self.solver.get_rho());
            Some((
                rho.iter().cloned().fold(f64::INFINITY, f64::min),
                rho.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            ))
        } else {
            None
        };

        Readback {
            u,
            p,
            stats: FieldStats {
                max_vel,
                nonfinite_u,
                p_min,
                p_max,
                p_finite: nonfinite_p == 0,
                nonfinite_p,
                rho,
            },
        }
    }

    /// Drive `n_steps` for headless callers, reading back every `readback_every`
    /// steps (and on the final step). `on_sample` is invoked every step with the
    /// step index and the [`StepOutcome`]; returning [`ControlFlow::Break`] stops the
    /// loop. The loop also stops on hard divergence. It does **not** auto-stop on
    /// `should_stop` (see [`RunResult`]).
    pub fn run_steps(
        &mut self,
        n_steps: usize,
        readback_every: usize,
        mut on_sample: impl FnMut(usize, &StepOutcome) -> ControlFlow<()>,
    ) -> RunResult {
        let mut result = RunResult::default();
        let every = readback_every.max(1);
        for step in 0..n_steps {
            let readback = step % every == 0 || step == n_steps.saturating_sub(1);
            let outcome = self.step(readback);
            result.executed_steps = step + 1;
            let flow = on_sample(step, &outcome);
            if let Some(reason) = outcome.diverged {
                result.diverged = Some(reason);
                result.stop_step = Some(step);
                return result;
            }
            if flow.is_break() {
                result.stopped_by_caller = true;
                result.stop_step = Some(step);
                return result;
            }
        }
        result
    }

    /// Borrow the underlying solver (viz upload, field reads, stats, logging).
    pub fn solver(&self) -> &UnifiedSolver {
        &self.solver
    }

    /// Mutably borrow the underlying solver (e.g. trace / profiling setup).
    pub fn solver_mut(&mut self) -> &mut UnifiedSolver {
        &mut self.solver
    }

    /// Consume the driver and return the configured solver. Useful for headless
    /// callers (e.g. diagnostic tests) that want to drive the raw solver directly
    /// while still building it through the shared driver path.
    pub fn into_solver(self) -> UnifiedSolver {
        self.solver
    }

    /// The adaptive-dt length scale (`sqrt(min cell volume)`).
    pub fn min_cell(&self) -> f64 {
        self.min_cell_size
    }

    /// Whether the model is density-based compressible (carries the full
    /// conservative state).
    pub fn compressible(&self) -> bool {
        self.compressible
    }

    /// The parameters last applied to the driver.
    pub fn params(&self) -> &RuntimeParams {
        &self.params
    }
}
