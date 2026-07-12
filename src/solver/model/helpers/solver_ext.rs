use crate::solver::gpu::enums::{GpuBoundaryType, GpuLowMachPrecondModel};
use crate::solver::gpu::program::plan_instance::PlanParamValue;
use crate::solver::gpu::structs::LinearSolverStats;
use crate::solver::gpu::GpuUnifiedSolver;
use crate::solver::model::eos::{EosRuntimeParams, EosSpec};
use crate::solver::model::linear_solver::FgmresSolutionUpdateStrategy;
use crate::solver::model::ports::PortRegistry;
use std::future::Future;
use std::pin::Pin;

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

const FIELD_RHO: &str = "rho";
const FIELD_RHO_U: &str = "rho_u";
const FIELD_RHO_E: &str = "rho_e";
const FIELD_P: &str = "p";
const FIELD_T: &str = "T";
const FIELD_U_LOWER: &str = "u";
const FIELD_U_UPPER: &str = "U";
const FIELD_MU: &str = "mu";

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
/// By default the outer loop runs in **GPU-batched mode**: the complete outer
/// schedule is encoded without per-iteration host decisions.  FGMRES is split
/// into bounded restart-chunk command buffers to stay within Metal/backend
/// command-buffer limits, so this mode removes host convergence round-trips but
/// does not promise one literal queue submission or fewer submissions than the
/// independently fused host-driven path.
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
/// | Batched mode | [`set_outer_batched_mode`] | `true` | When `true`, encode the outer schedule without per-iteration host decisions |
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
    /// Enable or disable the GPU-batched outer loop.
    ///
    /// When `true` (the default), all outer corrector iterations are scheduled
    /// without per-iteration host decisions.  Linear solves are submitted in
    /// bounded restart chunks to avoid backend command-buffer limits.
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
    ///
    /// Compressible dual-time runs enable this by default; callers can opt out
    /// to accept a nonconverged step without an in-step retry.
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
    /// Set the equation of state from precomputed runtime parameters (e.g.
    /// [`EosSpec::runtime_params_gauged`] for gauge-storage runs).
    fn set_eos_runtime(&mut self, params: &EosRuntimeParams) -> Result<(), String>;
}

impl SolverRuntimeParamsExt for GpuUnifiedSolver {
    fn set_dtau(&mut self, dtau: f32) -> Result<(), String> {
        if self.config().stepping == crate::solver::SteppingMode::Explicit && dtau != 0.0 {
            return Err("explicit RK4 does not support pseudo-time stepping (dtau must be zero)"
                .to_string());
        }
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
        self.set_eos_runtime(&eos.runtime_params())
    }

    fn set_eos_runtime(&mut self, params: &EosRuntimeParams) -> Result<(), String> {
        self.cache_eos_gauge_refs([
            params.gauge_rho_ref,
            params.gauge_p_ref,
            params.gauge_e_ref,
        ]);
        self.set_named_param("eos.gamma", PlanParamValue::F32(params.gamma))?;
        self.set_named_param("eos.gm1", PlanParamValue::F32(params.gm1))?;
        self.set_named_param("eos.r", PlanParamValue::F32(params.r))?;
        self.set_named_param("eos.dp_drho", PlanParamValue::F32(params.dp_drho))?;
        self.set_named_param("eos.p_ref", PlanParamValue::F32(params.p_ref))?;
        self.set_named_param("eos.theta_ref", PlanParamValue::F32(params.theta_ref))?;
        self.set_named_param("eos.rho_ref", PlanParamValue::F32(params.rho_ref))?;
        self.set_named_param(
            "eos.gauge_rho_ref",
            PlanParamValue::F32(params.gauge_rho_ref),
        )?;
        self.set_named_param("eos.gauge_p_ref", PlanParamValue::F32(params.gauge_p_ref))?;
        self.set_named_param("eos.gauge_e_ref", PlanParamValue::F32(params.gauge_e_ref))?;
        self.set_named_param(
            "eos.gauge_p_bias",
            PlanParamValue::F32(params.gauge_p_bias),
        )?;
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
            .or_else(|_| self.set_boundary_vec2(GpuBoundaryType::Inlet, FIELD_U_LOWER, value))?;
        // Expression-valued stage-time inlet closures read an immutable target
        // from the uniform rather than recursively multiplying the BC table
        // value they refresh. Models without that closure simply ignore it.
        let _ = self.set_named_param("inlet_velocity", PlanParamValue::F32(velocity));
        Ok(())
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
        // Gauge storage: BC tables hold STORED (gauge) values. The caller
        // still passes the ABSOLUTE inlet state; the solver's active gauge
        // references convert (zero when the gauge is off).
        let mut eos_params = eos.runtime_params();
        let [gauge_rho, gauge_p, gauge_e] = self.eos_gauge_refs();
        eos_params.gauge_rho_ref = gauge_rho;
        eos_params.gauge_p_ref = gauge_p;
        eos_params.gauge_e_ref = gauge_e;
        eos_params.gauge_p_bias = (f64::from(eos_params.gm1) * f64::from(gauge_e)
            + f64::from(eos_params.p_ref)
            - f64::from(gauge_p)) as f32;

        // Inlet pressure policy: the thermodynamically consistent reference
        // pressure for the prescribed (rho, theta_ref) state. This stands in
        // for the interior pressure when seeding the dependent entries (the
        // runtime bc_expr kernel then keeps them synchronized with the real
        // interior).
        let p0 = if eos_params.gm1 > 0.0 {
            rho * eos_params.theta_ref
        } else {
            eos_params.dp_drho * (rho - eos_params.rho_ref) + eos_params.p_ref
        };

        let u = [u_x, 0.0f32];

        // Seed the dependent entries (rho_u, rho_e, T) by evaluating the
        // model's DECLARED inlet expressions, so this helper cannot drift
        // from the GPU-side boundary math. The declared expressions consume
        // and produce STORED (gauge) values.
        let seeded = evaluate_inlet_declarations(self.model(), rho, u, p0, &eos_params)?;

        self.set_boundary_scalar(GpuBoundaryType::Inlet, FIELD_RHO, rho - gauge_rho)?;
        self.set_boundary_vec2(GpuBoundaryType::Inlet, FIELD_U_LOWER, u)?;
        self.set_boundary_vec2(GpuBoundaryType::Inlet, FIELD_RHO_U, seeded.rho_u)?;
        self.set_boundary_scalar(GpuBoundaryType::Inlet, FIELD_RHO_E, seeded.rho_e)?;
        let _ = self.set_boundary_scalar(GpuBoundaryType::Inlet, FIELD_P, p0 - gauge_p);
        if let Some(t0) = seeded.t {
            let _ = self.set_boundary_scalar(GpuBoundaryType::Inlet, FIELD_T, t0);
        }
        let _ = self.set_boundary_scalar(GpuBoundaryType::Outlet, FIELD_P, p0 - gauge_p);
        Ok(())
    }
}

/// Inlet entries seeded from the model's declared boundary expressions.
struct SeededInletValues {
    rho_u: [f32; 2],
    rho_e: f32,
    t: Option<f32>,
}

/// Density-linear energy gauge used by compressible host seeding.
///
/// A `C*rho` addition is exactly transported by continuity and only shifts
/// specific energy by the constant `C`. Zero pins the linear-EOS internal
/// energy to zero at its reference density without changing the ideal-gas
/// branch's existing f32 arithmetic.
const BAROTROPIC_ENERGY_GAUGE: f64 = 0.0;

fn seeded_total_energy_density(
    eos: &EosSpec,
    rho: f32,
    u: [f32; 2],
    p: f32,
) -> f32 {
    let ke = 0.5 * rho * (u[0] * u[0] + u[1] * u[1]);
    if let Some(internal) = eos.barotropic_internal_energy_density(
        f64::from(rho),
        BAROTROPIC_ENERGY_GAUGE,
    ) {
        return internal as f32 + ke;
    }

    let gm1 = eos.runtime_params().gm1;
    if gm1 > 0.0 {
        p / gm1 + ke
    } else {
        ke
    }
}

/// Evaluate the model's declared inlet expressions for `rho_u`, `rho_e`,
/// and `T` against a prescribed `(rho, u)` state and a reference pressure
/// standing in for the interior pressure.
///
/// Falls back to closed forms for entries a model does not declare (so the
/// helper still works on hand-rolled model variants).
fn evaluate_inlet_declarations(
    model: &crate::solver::model::ModelSpec,
    rho: f32,
    u: [f32; 2],
    p0: f32,
    eos_params: &crate::solver::model::eos::EosRuntimeParams,
) -> Result<SeededInletValues, String> {
    use crate::solver::model::backend::ast::FieldRef;
    use crate::solver::model::backend::boundary::eval_boundary_expr_f32;

    // Gauge storage: the declared boundary expressions consume STORED (gauge)
    // table values and the interior's stored pressure, exactly like the
    // generated bc_expr kernel. Callers pass ABSOLUTE (rho, p0).
    let gauge_rho_ref = eos_params.gauge_rho_ref;
    let gauge_p_ref = eos_params.gauge_p_ref;
    let interior = move |f: &FieldRef, _c: u32| -> Result<f32, String> {
        match f.name() {
            // The reference pressure stands in for the interior pressure.
            "p" => Ok(p0 - gauge_p_ref),
            other => Err(format!(
                "inlet seeding: interior({other}) is not available host-side"
            )),
        }
    };
    let bc = move |f: &FieldRef, c: u32| -> Result<f32, String> {
        match (f.name(), c) {
            (FIELD_RHO, 0) => Ok(rho - gauge_rho_ref),
            (FIELD_U_LOWER, 0 | 1) => Ok(u[c as usize]),
            other => Err(format!("inlet seeding: bc({other:?}) is not prescribed")),
        }
    };
    let eos = *eos_params;
    let param = move |p: &crate::solver::model::backend::algebraic::ParamRef| -> Result<f32, String> {
        Ok(match p.name() {
            "eos_gamma" => eos.gamma,
            "eos_gm1" => eos.gm1,
            "eos_r" => eos.r,
            "eos_dp_drho" => eos.dp_drho,
            "eos_p_ref" => eos.p_ref,
            "eos_theta_ref" => eos.theta_ref,
            "eos_rho_ref" => eos.rho_ref,
            "eos_gauge_rho_ref" => eos.gauge_rho_ref,
            "eos_gauge_p_ref" => eos.gauge_p_ref,
            "eos_gauge_e_ref" => eos.gauge_e_ref,
            "eos_gauge_p_bias" => eos.gauge_p_bias,
            other => return Err(format!("inlet seeding: unknown param '{other}'")),
        })
    };

    let declared = |field: &str, component: usize| -> Result<Option<f32>, String> {
        let Some(spec) = model.boundaries.field(field) else {
            return Ok(None);
        };
        let Some(conditions) = spec.by_boundary.get(&GpuBoundaryType::Inlet) else {
            return Ok(None);
        };
        let Some(condition) = conditions.get(component) else {
            return Ok(None);
        };
        let Some(expr) = condition.expr_value() else {
            return Ok(None);
        };
        eval_boundary_expr_f32(expr, &interior, &bc, &param).map(Some)
    };

    // Closed-form fallback for undeclared entries (ABSOLUTE thermodynamics,
    // stored-gauge conversion where the entry is a gauge-stored field).
    let eos_spec = model.eos();
    let legacy_rho_e = seeded_total_energy_density(&eos_spec, rho, u, p0) - eos_params.gauge_e_ref;
    let legacy_t = if eos_params.r.abs() > 1e-12 {
        Some(p0 / (rho.max(1e-12) * eos_params.r))
    } else {
        None
    };

    let rho_u = [
        declared(FIELD_RHO_U, 0)?.unwrap_or(rho * u[0]),
        declared(FIELD_RHO_U, 1)?.unwrap_or(rho * u[1]),
    ];
    let declared_rho_e = if matches!(eos_spec, EosSpec::LinearCompressibility { .. }) {
        // The barotropic declaration deliberately preserves bc(rho_e); that
        // snapshot value is the host oracle result being computed here.
        None
    } else {
        declared(FIELD_RHO_E, 0)?
    };
    // The generic boundary declaration historically used kinetic energy for
    // gm1=0. Override that host seed for a linear EOS with the EOS-owned
    // barotropic energy; retain the declared expression bit-for-bit for ideal
    // gas and hand-rolled non-barotropic models.
    let rho_e = if matches!(eos_spec, EosSpec::LinearCompressibility { .. }) {
        legacy_rho_e
    } else {
        declared_rho_e.unwrap_or(legacy_rho_e)
    };
    let t = match declared(FIELD_T, 0)? {
        Some(v) => Some(v),
        None => legacy_t,
    };

    Ok(SeededInletValues { rho_u, rho_e, t })
}

/// Initial-condition seeding for density-based compressible models.
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
        let r_gas = eos_params.r;
        // Gauge storage: callers pass ABSOLUTE primitives; the stored state
        // holds deviations from the solver's active references (zero when the
        // gauge is off, so the historical seeding is bit-identical).
        let [gauge_rho, gauge_p, gauge_e] = self.eos_gauge_refs();

        let stride = self.model().state_layout.stride() as usize;

        // CPU backend: the port registry is GPU-only, so seed via the routed
        // field-name API (which writes the CPU state buffer directly). Touches
        // only the named fields, preserving mu/gradients like the GPU path.
        if self.is_cpu() {
            let n = self.num_cells() as usize;
            let rho_e = seeded_total_energy_density(&eos, rho, u, p);
            let t = if r_gas > 0.0 { p / (rho.max(1e-12) * r_gas) } else { 0.0 };
            let _ = self.set_field_scalar(FIELD_RHO, &vec![(rho - gauge_rho) as f64; n]);
            let _ = self
                .set_field_vec2("rho_u", &vec![((rho * u[0]) as f64, (rho * u[1]) as f64); n]);
            let _ = self.set_field_scalar("rho_e", &vec![(rho_e - gauge_e) as f64; n]);
            let _ = self.set_field_scalar(FIELD_P, &vec![(p - gauge_p) as f64; n]);
            let _ = self.set_field_scalar("T", &vec![t as f64; n]);
            let _ = self
                .set_field_vec2(FIELD_U_UPPER, &vec![(u[0] as f64, u[1] as f64); n])
                .or_else(|_| self.set_field_vec2(FIELD_U_LOWER, &vec![(u[0] as f64, u[1] as f64); n]));
            return;
        }

        // Use cached port registry for field offset lookups
        let Some(registry) = self.port_registry() else {
            return;
        };
        let Some(offsets) = CompressibleConservedStateOffsets::resolve(registry) else {
            return;
        };

        let rho_e = seeded_total_energy_density(&eos, rho, u, p);

        // Preserve unrelated state fields (e.g. mu, gradients) when seeding initial conditions.
        // Tests and callers often set runtime params (like viscosity) before initializing state.
        let mut state = pollster::block_on(async { self.read_state_f32().await });
        if state.len() != self.num_cells() as usize * stride {
            state.resize(self.num_cells() as usize * stride, 0.0);
        }
        for cell in 0..self.num_cells() as usize {
            let base = cell * stride;
            state[base + offsets.rho] = rho - gauge_rho;
            state[base + offsets.rho_u_x] = rho * u[0];
            state[base + offsets.rho_u_y] = rho * u[1];
            state[base + offsets.rho_e] = rho_e - gauge_e;
            if let Some(off_p) = offsets.p {
                state[base + off_p] = p - gauge_p;
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
        let r_gas = eos_params.r;
        // Gauge storage: callers pass ABSOLUTE primitives (see set_uniform_state).
        let [gauge_rho, gauge_p, gauge_e] = self.eos_gauge_refs();

        let stride = self.model().state_layout.stride() as usize;

        // CPU backend: seed via the routed field-name API (port registry is
        // GPU-only).
        if self.is_cpu() {
            let n = self.num_cells() as usize;
            let rho_f: Vec<f64> = rho.iter().map(|&v| (v - gauge_rho) as f64).collect();
            let rho_u: Vec<(f64, f64)> = (0..n)
                .map(|i| ((rho[i] * u[i][0]) as f64, (rho[i] * u[i][1]) as f64))
                .collect();
            let rho_e: Vec<f64> = (0..n)
                .map(|i| {
                    (seeded_total_energy_density(&eos, rho[i], u[i], p[i]) - gauge_e) as f64
                })
                .collect();
            let p_f: Vec<f64> = p.iter().map(|&v| (v - gauge_p) as f64).collect();
            let t_f: Vec<f64> = (0..n)
                .map(|i| if r_gas > 0.0 { (p[i] / (rho[i].max(1e-12) * r_gas)) as f64 } else { 0.0 })
                .collect();
            let u_f: Vec<(f64, f64)> = (0..n).map(|i| (u[i][0] as f64, u[i][1] as f64)).collect();
            let _ = self.set_field_scalar(FIELD_RHO, &rho_f);
            let _ = self.set_field_vec2("rho_u", &rho_u);
            let _ = self.set_field_scalar("rho_e", &rho_e);
            let _ = self.set_field_scalar(FIELD_P, &p_f);
            let _ = self.set_field_scalar("T", &t_f);
            let _ = self
                .set_field_vec2(FIELD_U_UPPER, &u_f)
                .or_else(|_| self.set_field_vec2(FIELD_U_LOWER, &u_f));
            return;
        }

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
            let rho_e = seeded_total_energy_density(&eos, rho_val, u_val, p_val);

            state[base + offsets.rho] = rho_val - gauge_rho;
            state[base + offsets.rho_u_x] = rho_val * u_val[0];
            state[base + offsets.rho_u_y] = rho_val * u_val[1];
            state[base + offsets.rho_e] = rho_e - gauge_e;
            if let Some(off_p) = offsets.p {
                state[base + off_p] = p_val - gauge_p;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Ideal-gas inlet energy remains bit-for-bit identical to the declared
    /// expression, while linear-EOS host seeding uses the thermodynamic
    /// barotropic energy rather than the historical kinetic-only fallback.
    #[test]
    fn inlet_seeding_matches_declared_expressions() {
        // Physically sensible prescribed states per EOS (the declared
        // expressions apply the GPU kernel's pressure floor, so a state
        // with negative reference pressure would intentionally diverge
        // from the unfloored closed form).
        let cases = [
            (
                EosSpec::IdealGas {
                    gamma: 1.4,
                    gas_constant: 287.0,
                    temperature: 300.0,
                },
                1.2f32,
                [30.0f32, 0.0],
            ),
            (
                EosSpec::LinearCompressibility {
                    bulk_modulus: 2.2e9,
                    rho_ref: 1000.0,
                    p_ref: 1.0e5,
                },
                1000.5f32,
                [2.0f32, 0.0],
            ),
        ];
        for (eos, rho, u) in cases {
            let model = crate::solver::model::compressible_model_with_eos(eos).expect("model");
            let params = eos.runtime_params();
            let p0 = if params.gm1 > 0.0 {
                rho * params.theta_ref
            } else {
                params.dp_drho * (rho - params.rho_ref) + params.p_ref
            };

            let seeded =
                evaluate_inlet_declarations(&model, rho, u, p0, &params).expect("seeding");

            let ke = 0.5 * rho * (u[0] * u[0] + u[1] * u[1]);
            let expected_rho_e = match eos {
                EosSpec::IdealGas { .. } => p0 / params.gm1 + ke,
                EosSpec::LinearCompressibility { .. } => {
                    eos.barotropic_internal_energy_density(
                        f64::from(rho),
                        BAROTROPIC_ENERGY_GAUGE,
                    )
                    .expect("linear EOS energy") as f32
                        + ke
                }
                EosSpec::Constant => ke,
            };
            let expected_t = p0 / (rho.max(1e-12) * params.r);

            assert_eq!(seeded.rho_u, [rho * u[0], rho * u[1]], "{eos:?}");
            assert_eq!(seeded.rho_e, expected_rho_e, "{eos:?}");
            if matches!(eos, EosSpec::LinearCompressibility { .. }) {
                assert_ne!(
                    seeded.rho_e.to_bits(),
                    ke.to_bits(),
                    "off-reference linear-EOS energy regressed to kinetic-only seeding"
                );
            }
            assert_eq!(seeded.t, Some(expected_t), "{eos:?}");
        }
    }

    #[test]
    fn ideal_seeded_energy_keeps_legacy_f32_operation_order() {
        let eos = EosSpec::IdealGas {
            gamma: 1.4,
            gas_constant: 287.0,
            temperature: 300.0,
        };
        let params = eos.runtime_params();
        let rho = 1.2345_f32;
        let u = [31.25_f32, -2.5];
        let p = rho * params.theta_ref;
        let ke = 0.5 * rho * (u[0] * u[0] + u[1] * u[1]);
        let legacy = p / params.gm1 + ke;
        assert_eq!(
            seeded_total_energy_density(&eos, rho, u, p).to_bits(),
            legacy.to_bits()
        );
    }

    #[test]
    fn inlet_energy_boundary_preserves_barotropic_seed_and_ideal_numeric_branch() {
        use crate::solver::model::backend::boundary::{
            eval_boundary_expr_f32, BoundaryExpr,
        };

        let evaluate = |eos: EosSpec, prescribed_rho_e: f32| {
            let model = crate::solver::model::compressible_model_with_eos(eos).expect("model");
            let expression = model
                .boundaries
                .field(FIELD_RHO_E)
                .and_then(|spec| spec.by_boundary.get(&GpuBoundaryType::Inlet))
                .and_then(|conditions| conditions.first())
                .and_then(|condition| condition.expr_value())
                .expect("declared inlet rho_e expression");
            let mut preserves_rho_e = false;
            expression.visit(&mut |node| {
                if matches!(
                    node,
                    BoundaryExpr::BcValue { field, component: 0 }
                        if field.name() == FIELD_RHO_E
                ) {
                    preserves_rho_e = true;
                }
            });
            assert!(
                preserves_rho_e,
                "inlet expression lost the prescribed barotropic energy branch"
            );

            let rho = 1000.5_f32;
            let velocity = [2.0_f32, -0.25];
            let interior_pressure = 125_000.0_f32;
            let params = eos.runtime_params();
            let interior = |field: &crate::solver::model::backend::ast::FieldRef,
                            _component: u32| {
                if field.name() == FIELD_P {
                    Ok(interior_pressure)
                } else {
                    Err(format!("unexpected interior field '{}'", field.name()))
                }
            };
            let bc = |field: &crate::solver::model::backend::ast::FieldRef,
                      component: u32| {
                match (field.name(), component) {
                    (FIELD_RHO, 0) => Ok(rho),
                    (FIELD_U_LOWER, 0 | 1) => Ok(velocity[component as usize]),
                    (FIELD_RHO_E, 0) => Ok(prescribed_rho_e),
                    other => Err(format!("unexpected boundary field {other:?}")),
                }
            };
            let param = |parameter: &crate::solver::model::backend::algebraic::ParamRef| {
                Ok(match parameter.name() {
                    "eos_gamma" => params.gamma,
                    "eos_gm1" => params.gm1,
                    "eos_r" => params.r,
                    "eos_dp_drho" => params.dp_drho,
                    "eos_p_ref" => params.p_ref,
                    "eos_theta_ref" => params.theta_ref,
                    "eos_rho_ref" => params.rho_ref,
                    "eos_gauge_rho_ref" => params.gauge_rho_ref,
                    "eos_gauge_p_ref" => params.gauge_p_ref,
                    "eos_gauge_e_ref" => params.gauge_e_ref,
                    "eos_gauge_p_bias" => params.gauge_p_bias,
                    other => return Err(format!("unexpected EOS param '{other}'")),
                })
            };
            (
                eval_boundary_expr_f32(expression, &interior, &bc, &param)
                    .expect("evaluate inlet energy"),
                rho,
                velocity,
                interior_pressure,
                params,
            )
        };

        let ideal = EosSpec::IdealGas {
            gamma: 1.4,
            gas_constant: 287.0,
            temperature: 300.0,
        };
        let (ideal_value, rho, velocity, pressure, ideal_params) = evaluate(ideal, -9_999.0);
        let ideal_ke = 0.5
            * rho
            * (velocity[0] * velocity[0] + velocity[1] * velocity[1]);
        let ideal_expected = pressure / ideal_params.gm1.max(1.0e-6) + ideal_ke;
        assert_eq!(ideal_value.to_bits(), ideal_expected.to_bits());

        let linear = EosSpec::LinearCompressibility {
            bulk_modulus: 2.2e9,
            rho_ref: 1000.0,
            p_ref: 1.0e5,
        };
        let prescribed = 12_345.25_f32;
        let (linear_value, ..) = evaluate(linear, prescribed);
        assert_eq!(linear_value.to_bits(), prescribed.to_bits());
    }
}
