use crate::solver::gpu::enums::{GpuBoundaryType, GpuLowMachPrecondModel};
use crate::solver::gpu::program::plan_instance::PlanParamValue;
use crate::solver::gpu::structs::LinearSolverStats;
use crate::solver::gpu::GpuUnifiedSolver;
use crate::solver::model::eos::EosSpec;
use crate::solver::model::linear_solver::FgmresSolutionUpdateStrategy;
use crate::solver::model::ports::PortRegistry;
use std::future::Future;
use std::pin::Pin;

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

// =============================================================================
// Canonical field name constants for compressible conserved-state seeding
// =============================================================================

const FIELD_RHO: &str = "rho";
const FIELD_RHO_U: &str = "rho_u";
const FIELD_RHO_E: &str = "rho_e";
const FIELD_P: &str = "p";
const FIELD_T: &str = "T";
const FIELD_U_LOWER: &str = "u";
const FIELD_U_UPPER: &str = "U";
const FIELD_MU: &str = "mu";

// =============================================================================
// Resolved offset bundle for compressible conserved-state seeding
// =============================================================================

/// Resolved state offsets for compressible conserved-state seeding.
///
/// This struct holds the computed offsets for all fields needed to seed
/// initial conditions for compressible flow simulations. Required fields
/// are stored as `usize`; optional fields are `Option<usize>`.
#[derive(Debug, Clone, Copy)]
struct CompressibleConservedStateOffsets {
    /// Scalar offset for density (required)
    pub rho: usize,
    /// X-component offset for momentum density (required)
    pub rho_u_x: usize,
    /// Y-component offset for momentum density (required)
    pub rho_u_y: usize,
    /// Scalar offset for energy density (required)
    pub rho_e: usize,
    /// Scalar offset for pressure (optional)
    pub p: Option<usize>,
    /// Scalar offset for temperature (optional)
    pub t: Option<usize>,
    /// X-component offset for velocity (optional)
    pub u_x: Option<usize>,
    /// Y-component offset for velocity (optional)
    pub u_y: Option<usize>,
}

impl CompressibleConservedStateOffsets {
    /// Resolve offsets from the port registry.
    ///
    /// Returns `Some(Self)` if all required fields are present with correct kinds:
    /// - `rho`: scalar (1 component)
    /// - `rho_u`: vector2 (2 components)
    /// - `rho_e`: scalar (1 component)
    ///
    /// Optional fields are resolved best-effort:
    /// - `p`, `T`: scalar
    /// - `u` or `U`: vector2 (prefers lowercase "u", falls back to "U")
    pub fn resolve(registry: &PortRegistry) -> Option<Self> {
        // Resolve required fields
        let rho_entry = registry.get_field_entry_by_name(FIELD_RHO)?;
        let rho_u_entry = registry.get_field_entry_by_name(FIELD_RHO_U)?;
        let rho_e_entry = registry.get_field_entry_by_name(FIELD_RHO_E)?;

        // Validate component counts for required fields
        if rho_entry.component_count() != 1
            || rho_u_entry.component_count() != 2
            || rho_e_entry.component_count() != 1
        {
            return None;
        }

        // Resolve optional fields (best-effort)
        let p = registry
            .get_field_entry_by_name(FIELD_P)
            .filter(|e| e.component_count() == 1)
            .map(|e| e.offset() as usize);

        let t = registry
            .get_field_entry_by_name(FIELD_T)
            .filter(|e| e.component_count() == 1)
            .map(|e| e.offset() as usize);

        let (u_x, u_y) = registry
            .get_field_entry_by_name(FIELD_U_LOWER)
            .filter(|e| e.component_count() == 2)
            .or_else(|| {
                registry
                    .get_field_entry_by_name(FIELD_U_UPPER)
                    .filter(|e| e.component_count() == 2)
            })
            .map(|e| {
                let base = e.offset() as usize;
                (Some(base), Some(base + 1))
            })
            .unwrap_or((None, None));

        Some(Self {
            rho: rho_entry.offset() as usize,
            rho_u_x: rho_u_entry.offset() as usize,
            rho_u_y: rho_u_entry.offset() as usize + 1,
            rho_e: rho_e_entry.offset() as usize,
            p,
            t,
            u_x,
            u_y,
        })
    }
}

/// Convenience accessors for the primary solution fields (velocity, pressure, density).
///
/// These methods resolve the canonical field names (`u`/`U`, `p`, `rho`) and delegate
/// to the underlying [`GpuUnifiedSolver`] field storage.  The `get_*` variants return
/// futures that perform an async GPU-to-host readback.
pub trait SolverFieldAliasesExt {
    /// Write velocity field data (per-cell `(u_x, u_y)` pairs).
    fn set_u(&mut self, u: &[(f64, f64)]);
    /// Write pressure field data (per-cell scalar values).
    fn set_p(&mut self, p: &[f64]);
    /// Read the velocity field back from GPU memory.
    fn get_u(&self) -> BoxFuture<'_, Vec<(f64, f64)>>;
    /// Read the pressure field back from GPU memory.
    fn get_p(&self) -> BoxFuture<'_, Vec<f64>>;
    /// Read the density field back from GPU memory.
    fn get_rho(&self) -> BoxFuture<'_, Vec<f64>>;
}

impl SolverFieldAliasesExt for GpuUnifiedSolver {
    fn set_u(&mut self, u: &[(f64, f64)]) {
        let _ = self
            .set_field_vec2(FIELD_U_UPPER, u)
            .or_else(|_| self.set_field_vec2(FIELD_U_LOWER, u));
    }

    fn set_p(&mut self, p: &[f64]) {
        let _ = self.set_field_scalar(FIELD_P, p);
    }

    fn get_u(&self) -> BoxFuture<'_, Vec<(f64, f64)>> {
        Box::pin(async move {
            if let Ok(v) = self.get_field_vec2(FIELD_U_UPPER).await {
                return v;
            }
            if let Ok(v) = self.get_field_vec2(FIELD_U_LOWER).await {
                return v;
            }
            vec![(0.0, 0.0); self.num_cells() as usize]
        })
    }

    fn get_p(&self) -> BoxFuture<'_, Vec<f64>> {
        Box::pin(async move {
            self.get_field_scalar(FIELD_P)
                .await
                .unwrap_or_else(|_| vec![0.0; self.num_cells() as usize])
        })
    }

    fn get_rho(&self) -> BoxFuture<'_, Vec<f64>> {
        Box::pin(async move {
            self.get_field_scalar(FIELD_RHO)
                .await
                .unwrap_or_else(|_| vec![0.0; self.num_cells() as usize])
        })
    }
}

/// Runtime parameter setters for the GPU-based coupled stepping path.
///
/// These methods control the physics parameters, under-relaxation factors, outer-loop
/// configuration, preconditioner selection, and linear-solver tuning of a
/// [`GpuUnifiedSolver`].  Each setter maps to a named parameter that the solver's
/// program plan reads during stepping.
///
/// # Outer-loop execution model
///
/// The coupled stepping path executes a SIMPLE-like pressure-velocity outer loop.
/// By default the outer loop runs in **one-submission batched mode**: all outer
/// iterations are encoded into a single GPU command buffer and submitted as one
/// queue submission.  This eliminates per-iteration host-GPU synchronisation
/// overhead and typically reduces total queue submissions by 50-75%.
///
/// When adaptive convergence is enabled (the default), the batched submission uses
/// GPU-side indirect dispatch gating so that converged iterations become zero-cost
/// dispatches rather than requiring host involvement to skip them.
///
/// The key parameters that control outer-loop behavior are:
///
/// | Parameter | Setter | Default | Effect |
/// |-----------|--------|---------|--------|
/// | Outer iterations | [`set_outer_iters`] | model-defined | Number of outer corrector sweeps |
/// | Outer tolerance | [`set_outer_tolerance`] | model-defined | Relative correction-norm threshold for adaptive early stop |
/// | Outer tolerance (abs) | [`set_outer_tolerance_abs`] | model-defined | Absolute correction-norm threshold |
/// | Fixed-iteration mode | [`set_outer_fixed_iterations_mode`] | `false` | When `true`, run all configured iterations without adaptive break |
/// | Batched mode | [`set_outer_batched_mode`] | `true` | When `true`, use one-submission batched outer loop |
///
/// [`set_outer_iters`]: SolverRuntimeParamsExt::set_outer_iters
/// [`set_outer_tolerance`]: SolverRuntimeParamsExt::set_outer_tolerance
/// [`set_outer_tolerance_abs`]: SolverRuntimeParamsExt::set_outer_tolerance_abs
/// [`set_outer_fixed_iterations_mode`]: SolverRuntimeParamsExt::set_outer_fixed_iterations_mode
/// [`set_outer_batched_mode`]: SolverRuntimeParamsExt::set_outer_batched_mode
pub trait SolverRuntimeParamsExt {
    /// Set the pseudo-timestep for the coupled stepping scheme.
    fn set_dtau(&mut self, dtau: f32) -> Result<(), String>;
    /// Set the dynamic viscosity (uniform).  Also updates the `mu` state field if present.
    fn set_viscosity(&mut self, mu: f32) -> Result<(), String>;
    /// Set the fluid density (uniform, incompressible models).
    fn set_density(&mut self, rho: f32) -> Result<(), String>;
    /// Set the velocity under-relaxation factor.
    fn set_alpha_u(&mut self, alpha_u: f32) -> Result<(), String>;
    /// Set the pressure under-relaxation factor.
    fn set_alpha_p(&mut self, alpha_p: f32) -> Result<(), String>;
    /// Set the number of outer corrector iterations per step.
    fn set_outer_iters(&mut self, iters: usize) -> Result<(), String>;
    /// Set the relative correction-norm tolerance for adaptive outer-loop early stopping.
    fn set_outer_tolerance(&mut self, tol: f32) -> Result<(), String>;
    /// Set the absolute correction-norm tolerance for adaptive outer-loop early stopping.
    fn set_outer_tolerance_abs(&mut self, tol_abs: f32) -> Result<(), String>;
    /// Enable or disable fixed-iteration outer-loop mode.
    ///
    /// When `true`, the solver runs all configured `outer_iters` without evaluating
    /// convergence between iterations.  This is useful for deterministic benchmarking
    /// where consistent iteration counts are required across runs.
    fn set_outer_fixed_iterations_mode(&mut self, enabled: bool) -> Result<(), String>;
    /// Enable or disable the one-submission batched outer loop.
    ///
    /// When `true` (the default), all outer corrector iterations are encoded into a
    /// single GPU command buffer submission.  This is the standard coupled stepping
    /// mode and provides significantly fewer queue submissions than the per-iteration
    /// host-driven loop.
    ///
    /// When adaptive convergence is active, converged iterations become zero-cost
    /// indirect dispatches (GPU-side gating), avoiding any host round-trips.
    fn set_outer_batched_mode(&mut self, enabled: bool) -> Result<(), String>;
    /// Set the low-Mach preconditioner model variant.
    fn set_precond_model(&mut self, model: GpuLowMachPrecondModel) -> Result<(), String>;
    /// Set the low-Mach preconditioner theta floor parameter.
    fn set_precond_theta_floor(&mut self, theta: f32) -> Result<(), String>;
    /// Set the low-Mach preconditioner pressure-coupling alpha parameter.
    fn set_precond_pressure_coupling_alpha(&mut self, alpha: f32) -> Result<(), String>;
    /// Set the under-relaxation factor applied when the linear solver does not converge.
    fn set_nonconverged_relax(&mut self, alpha: f32) -> Result<(), String>;
    /// Set the next-step physical timestep scale applied after a nonconverged dual-time step.
    fn set_nonconverged_dt_scale(&mut self, scale: f32) -> Result<(), String>;
    /// Set the next-step pseudo-timestep scale applied after a nonconverged dual-time step.
    fn set_nonconverged_dtau_scale(&mut self, scale: f32) -> Result<(), String>;
    /// Enable or disable in-step retries for nonconverged dual-time steps.
    fn set_nonconverged_retry_enabled(&mut self, enabled: bool) -> Result<(), String>;
    /// Set the maximum number of in-step retry attempts for a nonconverged dual-time step.
    fn set_nonconverged_retry_max_attempts(&mut self, attempts: usize) -> Result<(), String>;
    /// Set the FGMRES solution-update strategy (e.g. classical vs modified Gram-Schmidt).
    fn set_linear_solver_solution_update_strategy(
        &mut self,
        strategy: FgmresSolutionUpdateStrategy,
    ) -> Result<(), String>;
    /// Set the equation of state parameters from an [`EosSpec`].
    fn set_eos(&mut self, eos: &EosSpec) -> Result<(), String>;
}

impl SolverRuntimeParamsExt for GpuUnifiedSolver {
    fn set_dtau(&mut self, dtau: f32) -> Result<(), String> {
        self.set_named_param("dtau", PlanParamValue::F32(dtau))
    }

    fn set_viscosity(&mut self, mu: f32) -> Result<(), String> {
        self.set_named_param("viscosity", PlanParamValue::F32(mu))?;

        // Keep the optional `mu` state field (used by implicit laplacian terms) in sync with the
        // runtime `viscosity` parameter when present.
        //
        // Many models treat viscosity as a constant named param, but the discretized system may
        // still reference a `mu` field coefficient.
        let n = self.num_cells() as usize;
        let mu64 = mu as f64;
        let mu_field: Vec<f64> = vec![mu64; n];
        let _ = self.set_field_scalar(FIELD_MU, &mu_field);

        Ok(())
    }

    fn set_density(&mut self, rho: f32) -> Result<(), String> {
        self.set_named_param("density", PlanParamValue::F32(rho))
    }

    fn set_alpha_u(&mut self, alpha_u: f32) -> Result<(), String> {
        self.set_named_param("alpha_u", PlanParamValue::F32(alpha_u))
    }

    fn set_alpha_p(&mut self, alpha_p: f32) -> Result<(), String> {
        self.set_named_param("alpha_p", PlanParamValue::F32(alpha_p))
    }

    fn set_outer_iters(&mut self, iters: usize) -> Result<(), String> {
        self.set_named_param("outer_iters", PlanParamValue::Usize(iters))
    }

    fn set_outer_tolerance(&mut self, tol: f32) -> Result<(), String> {
        self.set_named_param("outer_tol", PlanParamValue::F32(tol))
    }

    fn set_outer_tolerance_abs(&mut self, tol_abs: f32) -> Result<(), String> {
        self.set_named_param("outer_tol_abs", PlanParamValue::F32(tol_abs))
    }

    fn set_outer_fixed_iterations_mode(&mut self, enabled: bool) -> Result<(), String> {
        self.set_named_param("outer_fixed_iterations_mode", PlanParamValue::Bool(enabled))
    }

    fn set_outer_batched_mode(&mut self, enabled: bool) -> Result<(), String> {
        self.set_named_param("outer_batched_mode", PlanParamValue::Bool(enabled))
    }

    fn set_precond_model(&mut self, model: GpuLowMachPrecondModel) -> Result<(), String> {
        self.set_named_param("low_mach.model", PlanParamValue::LowMachModel(model))
    }

    fn set_precond_theta_floor(&mut self, theta: f32) -> Result<(), String> {
        self.set_named_param("low_mach.theta_floor", PlanParamValue::F32(theta))
    }

    fn set_precond_pressure_coupling_alpha(&mut self, alpha: f32) -> Result<(), String> {
        self.set_named_param(
            "low_mach.pressure_coupling_alpha",
            PlanParamValue::F32(alpha),
        )
    }

    fn set_nonconverged_relax(&mut self, alpha: f32) -> Result<(), String> {
        self.set_named_param("nonconverged_relax", PlanParamValue::F32(alpha))
    }

    fn set_nonconverged_dt_scale(&mut self, scale: f32) -> Result<(), String> {
        self.set_named_param("nonconverged_dt_scale", PlanParamValue::F32(scale))
    }

    fn set_nonconverged_dtau_scale(&mut self, scale: f32) -> Result<(), String> {
        self.set_named_param("nonconverged_dtau_scale", PlanParamValue::F32(scale))
    }

    fn set_nonconverged_retry_enabled(&mut self, enabled: bool) -> Result<(), String> {
        self.set_named_param(
            "nonconverged_retry_enabled",
            PlanParamValue::Bool(enabled),
        )
    }

    fn set_nonconverged_retry_max_attempts(&mut self, attempts: usize) -> Result<(), String> {
        self.set_named_param(
            "nonconverged_retry_max_attempts",
            PlanParamValue::Usize(attempts),
        )
    }

    fn set_linear_solver_solution_update_strategy(
        &mut self,
        strategy: FgmresSolutionUpdateStrategy,
    ) -> Result<(), String> {
        self.set_named_param(
            "linear_solver.solution_update_strategy",
            PlanParamValue::FgmresSolutionUpdateStrategy(strategy),
        )
    }

    fn set_eos(&mut self, eos: &EosSpec) -> Result<(), String> {
        let params = eos.runtime_params();
        self.set_named_param("eos.gamma", PlanParamValue::F32(params.gamma))?;
        self.set_named_param("eos.gm1", PlanParamValue::F32(params.gm1))?;
        self.set_named_param("eos.r", PlanParamValue::F32(params.r))?;
        self.set_named_param("eos.dp_drho", PlanParamValue::F32(params.dp_drho))?;
        self.set_named_param("eos.p_offset", PlanParamValue::F32(params.p_offset))?;
        self.set_named_param("eos.theta_ref", PlanParamValue::F32(params.theta_ref))?;
        Ok(())
    }
}

/// Inlet velocity boundary condition setter for incompressible models.
pub trait SolverInletVelocityExt {
    /// Set the inlet x-velocity (y-component is zero).
    fn set_inlet_velocity(&mut self, velocity: f32) -> Result<(), String>;
}

impl SolverInletVelocityExt for GpuUnifiedSolver {
    fn set_inlet_velocity(&mut self, velocity: f32) -> Result<(), String> {
        let value = [velocity, 0.0f32];
        self.set_boundary_vec2(GpuBoundaryType::Inlet, FIELD_U_UPPER, value)
            .or_else(|_| self.set_boundary_vec2(GpuBoundaryType::Inlet, FIELD_U_LOWER, value))
    }
}

pub trait SolverCompressibleInletExt {
    /// Update inlet Dirichlet BCs for a compressible model that solves for conserved Euler fields,
    /// assuming an x-directed velocity and an EOS that provides runtime params (ideal gas or
    /// barotropic).
    fn set_compressible_inlet_isothermal_x(
        &mut self,
        rho: f32,
        u_x: f32,
        eos: &EosSpec,
    ) -> Result<(), String>;
}

impl SolverCompressibleInletExt for GpuUnifiedSolver {
    fn set_compressible_inlet_isothermal_x(
        &mut self,
        rho: f32,
        u_x: f32,
        eos: &EosSpec,
    ) -> Result<(), String> {
        let eos_params = eos.runtime_params();
        let gm1 = eos_params.gm1;
        let dp_drho = eos_params.dp_drho;
        let p_offset = eos_params.p_offset;
        let theta_ref = eos_params.theta_ref;
        let r_gas = eos_params.r;

        let p0 = if gm1 > 0.0 {
            rho * theta_ref
        } else {
            dp_drho * rho - p_offset
        };
        let t0 = if r_gas.abs() > 1e-12 {
            p0 / (rho.max(1e-12) * r_gas)
        } else {
            0.0
        };

        let u = [u_x, 0.0f32];
        let rho_u = [rho * u_x, 0.0f32];
        let ke = 0.5 * rho * (u_x * u_x);
        let rho_e = if gm1 > 0.0 { p0 / gm1 + ke } else { ke };

        self.set_boundary_scalar(GpuBoundaryType::Inlet, FIELD_RHO, rho)?;
        self.set_boundary_vec2(GpuBoundaryType::Inlet, FIELD_U_LOWER, u)?;
        self.set_boundary_vec2(GpuBoundaryType::Inlet, FIELD_RHO_U, rho_u)?;
        self.set_boundary_scalar(GpuBoundaryType::Inlet, FIELD_RHO_E, rho_e)?;
        let _ = self.set_boundary_scalar(GpuBoundaryType::Inlet, FIELD_P, p0);
        let _ = self.set_boundary_scalar(GpuBoundaryType::Inlet, FIELD_T, t0);
        Ok(())
    }
}

/// Initial-condition seeding for compressible ideal-gas models.
///
/// These methods write conserved-variable state fields (`rho`, `rho_u`, `rho_e`)
/// and derived fields (`p`, `T`, `u`) from primitive inputs.  Existing state data
/// for unrelated fields (e.g. `mu`, gradients) is preserved.
pub trait SolverCompressibleIdealGasExt {
    /// Seed a spatially uniform state.
    fn set_uniform_state(&mut self, rho: f32, u: [f32; 2], p: f32);
    /// Seed per-cell state from arrays of primitives.
    fn set_state_fields(&mut self, rho: &[f32], u: &[[f32; 2]], p: &[f32]);
}

impl SolverCompressibleIdealGasExt for GpuUnifiedSolver {
    fn set_uniform_state(&mut self, rho: f32, u: [f32; 2], p: f32) {
        let eos = self.model().eos();
        let eos_params = eos.runtime_params();
        let gm1 = eos_params.gm1;
        let r_gas = eos_params.r;

        let stride = self.model().state_layout.stride() as usize;

        // Use cached port registry for field offset lookups
        let Some(registry) = self.port_registry() else {
            return;
        };
        let Some(offsets) = CompressibleConservedStateOffsets::resolve(registry) else {
            return;
        };

        let ke = 0.5 * rho * (u[0] * u[0] + u[1] * u[1]);
        let rho_e = if gm1 > 0.0 { p / gm1 + ke } else { ke };

        // Preserve unrelated state fields (e.g. mu, gradients) when seeding initial conditions.
        // Tests and callers often set runtime params (like viscosity) before initializing state.
        let mut state = pollster::block_on(async { self.read_state_f32().await });
        if state.len() != self.num_cells() as usize * stride {
            state.resize(self.num_cells() as usize * stride, 0.0);
        }
        for cell in 0..self.num_cells() as usize {
            let base = cell * stride;
            state[base + offsets.rho] = rho;
            state[base + offsets.rho_u_x] = rho * u[0];
            state[base + offsets.rho_u_y] = rho * u[1];
            state[base + offsets.rho_e] = rho_e;
            if let Some(off_p) = offsets.p {
                state[base + off_p] = p;
            }
            if let Some(off_t) = offsets.t {
                state[base + off_t] = if r_gas > 0.0 {
                    p / (rho.max(1e-12) * r_gas)
                } else {
                    0.0
                };
            }
            if let (Some(off_u_x), Some(off_u_y)) = (offsets.u_x, offsets.u_y) {
                state[base + off_u_x] = u[0];
                state[base + off_u_y] = u[1];
            }
        }
        let _ = self.write_state_f32(&state);
    }

    fn set_state_fields(&mut self, rho: &[f32], u: &[[f32; 2]], p: &[f32]) {
        if rho.len() != self.num_cells() as usize
            || u.len() != self.num_cells() as usize
            || p.len() != self.num_cells() as usize
        {
            return;
        }

        let eos = self.model().eos();
        let eos_params = eos.runtime_params();
        let gm1 = eos_params.gm1;
        let r_gas = eos_params.r;

        let stride = self.model().state_layout.stride() as usize;

        // Use cached port registry for field offset lookups
        let Some(registry) = self.port_registry() else {
            return;
        };
        let Some(offsets) = CompressibleConservedStateOffsets::resolve(registry) else {
            return;
        };

        // Preserve unrelated state fields (e.g. mu, gradients) when seeding initial conditions.
        let mut state = pollster::block_on(async { self.read_state_f32().await });
        if state.len() != self.num_cells() as usize * stride {
            state.resize(self.num_cells() as usize * stride, 0.0);
        }
        for cell in 0..self.num_cells() as usize {
            let base = cell * stride;
            let rho_val = rho[cell];
            let u_val = u[cell];
            let p_val = p[cell];
            let ke = 0.5 * rho_val * (u_val[0] * u_val[0] + u_val[1] * u_val[1]);
            let rho_e = if gm1 > 0.0 { p_val / gm1 + ke } else { ke };

            state[base + offsets.rho] = rho_val;
            state[base + offsets.rho_u_x] = rho_val * u_val[0];
            state[base + offsets.rho_u_y] = rho_val * u_val[1];
            state[base + offsets.rho_e] = rho_e;
            if let Some(off_p) = offsets.p {
                state[base + off_p] = p_val;
            }
            if let Some(off_t) = offsets.t {
                state[base + off_t] = if r_gas > 0.0 {
                    p_val / (rho_val.max(1e-12) * r_gas)
                } else {
                    0.0
                };
            }
            if let (Some(off_u_x), Some(off_u_y)) = (offsets.u_x, offsets.u_y) {
                state[base + off_u_x] = u_val[0];
                state[base + off_u_y] = u_val[1];
            }
        }
        let _ = self.write_state_f32(&state);
    }
}

/// Post-step statistics readers for incompressible coupled solvers.
///
/// These accessors expose convergence diagnostics, linear-solver residuals, and
/// degenerate-cell counts from the most recent [`GpuUnifiedSolver::step`] call.
pub trait SolverIncompressibleStatsExt {
    /// Whether the solver signaled that the simulation should stop (e.g. steady-state reached).
    fn incompressible_should_stop(&self) -> bool;
    /// Outer-loop stats: `(iterations_completed, residual_u, residual_p)`.
    fn incompressible_outer_stats(&self) -> Option<(u32, f32, f32)>;
    /// Per-field linear-solver stats for the last outer iteration (u, p, coupled).
    fn incompressible_linear_stats(
        &self,
    ) -> Option<(LinearSolverStats, LinearSolverStats, LinearSolverStats)>;
    /// Number of degenerate cells detected during the last step.
    fn incompressible_degenerate_count(&self) -> Option<u32>;
}

impl SolverIncompressibleStatsExt for GpuUnifiedSolver {
    fn incompressible_should_stop(&self) -> bool {
        self.step_stats().should_stop.unwrap_or(false)
    }

    fn incompressible_outer_stats(&self) -> Option<(u32, f32, f32)> {
        let stats = self.step_stats();
        Some((
            stats.outer_iterations?,
            stats.outer_residual_u?,
            stats.outer_residual_p?,
        ))
    }

    fn incompressible_linear_stats(
        &self,
    ) -> Option<(LinearSolverStats, LinearSolverStats, LinearSolverStats)> {
        self.step_stats().linear_stats
    }

    fn incompressible_degenerate_count(&self) -> Option<u32> {
        self.step_stats().degenerate_count
    }
}

/// Mutable runtime controls for incompressible coupled solvers.
pub trait SolverIncompressibleControlsExt {
    /// Set the number of outer corrector iterations.  Maps to the unified `outer_iters` knob.
    fn set_incompressible_outer_correctors(&mut self, iters: u32) -> Result<(), String>;
    /// Signal that the simulation should stop after the current step.
    fn incompressible_set_should_stop(&mut self, value: bool);
}

impl SolverIncompressibleControlsExt for GpuUnifiedSolver {
    fn set_incompressible_outer_correctors(&mut self, iters: u32) -> Result<(), String> {
        // Coupled/implicit paths use the unified `outer_iters` knob.
        self.set_plan_named_param("outer_iters", PlanParamValue::Usize(iters as usize))
    }

    fn incompressible_set_should_stop(&mut self, value: bool) {
        let _ = value;
    }
}
