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
        model: ModelSpec,
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

        let rho = if self.compressible {
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
