use crate::solver::gpu::program::plan_instance::PlanParamValue;
use crate::solver::gpu::structs::LinearSolverStats;
use crate::solver::gpu::GpuUnifiedSolver;
use crate::solver::gpu::enums::{GpuBoundaryType, GpuLowMachPrecondModel};
use crate::solver::model::eos::EosSpec;
use std::future::Future;
use std::pin::Pin;

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

pub trait SolverFieldAliasesExt {
    fn set_u(&mut self, u: &[(f64, f64)]);
    fn set_p(&mut self, p: &[f64]);
    fn get_u(&self) -> BoxFuture<'_, Vec<(f64, f64)>>;
    fn get_p(&self) -> BoxFuture<'_, Vec<f64>>;
    fn get_rho(&self) -> BoxFuture<'_, Vec<f64>>;
}

impl SolverFieldAliasesExt for GpuUnifiedSolver {
    fn set_u(&mut self, u: &[(f64, f64)]) {
        let _ = self
            .set_field_vec2("U", u)
            .or_else(|_| self.set_field_vec2("u", u));
    }

    fn set_p(&mut self, p: &[f64]) {
        let _ = self.set_field_scalar("p", p);
    }

    fn get_u(&self) -> BoxFuture<'_, Vec<(f64, f64)>> {
        Box::pin(async move {
            if let Ok(v) = self.get_field_vec2("U").await {
                return v;
            }
            if let Ok(v) = self.get_field_vec2("u").await {
                return v;
            }
            vec![(0.0, 0.0); self.num_cells() as usize]
        })
    }

    fn get_p(&self) -> BoxFuture<'_, Vec<f64>> {
        Box::pin(async move {
            self.get_field_scalar("p")
                .await
                .unwrap_or_else(|_| vec![0.0; self.num_cells() as usize])
        })
    }

    fn get_rho(&self) -> BoxFuture<'_, Vec<f64>> {
        Box::pin(async move {
            self.get_field_scalar("rho")
                .await
                .unwrap_or_else(|_| vec![0.0; self.num_cells() as usize])
        })
    }
}

pub trait SolverRuntimeParamsExt {
    fn set_dtau(&mut self, dtau: f32) -> Result<(), String>;
    fn set_viscosity(&mut self, mu: f32) -> Result<(), String>;
    fn set_density(&mut self, rho: f32) -> Result<(), String>;
    fn set_alpha_u(&mut self, alpha_u: f32) -> Result<(), String>;
    fn set_alpha_p(&mut self, alpha_p: f32) -> Result<(), String>;
    fn set_outer_iters(&mut self, iters: usize) -> Result<(), String>;
    fn set_precond_model(&mut self, model: GpuLowMachPrecondModel) -> Result<(), String>;
    fn set_precond_theta_floor(&mut self, theta: f32) -> Result<(), String>;
    fn set_precond_pressure_coupling_alpha(&mut self, alpha: f32) -> Result<(), String>;
    fn set_nonconverged_relax(&mut self, alpha: f32) -> Result<(), String>;
    fn set_eos(&mut self, eos: &EosSpec) -> Result<(), String>;
}

impl SolverRuntimeParamsExt for GpuUnifiedSolver {
    fn set_dtau(&mut self, dtau: f32) -> Result<(), String> {
        self.set_named_param("dtau", PlanParamValue::F32(dtau))
    }

    fn set_viscosity(&mut self, mu: f32) -> Result<(), String> {
        self.set_named_param("viscosity", PlanParamValue::F32(mu))
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

pub trait SolverInletVelocityExt {
    fn set_inlet_velocity(&mut self, velocity: f32) -> Result<(), String>;
}

impl SolverInletVelocityExt for GpuUnifiedSolver {
    fn set_inlet_velocity(&mut self, velocity: f32) -> Result<(), String> {
        let value = [velocity, 0.0f32];
        self.set_boundary_vec2(GpuBoundaryType::Inlet, "U", value)
            .or_else(|_| self.set_boundary_vec2(GpuBoundaryType::Inlet, "u", value))
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

        self.set_boundary_scalar(GpuBoundaryType::Inlet, "rho", rho)?;
        self.set_boundary_vec2(GpuBoundaryType::Inlet, "u", u)?;
        self.set_boundary_vec2(GpuBoundaryType::Inlet, "rho_u", rho_u)?;
        self.set_boundary_scalar(GpuBoundaryType::Inlet, "rho_e", rho_e)?;
        let _ = self.set_boundary_scalar(GpuBoundaryType::Inlet, "p", p0);
        let _ = self.set_boundary_scalar(GpuBoundaryType::Inlet, "T", t0);
        Ok(())
    }
}

pub trait SolverCompressibleIdealGasExt {
    fn set_uniform_state(&mut self, rho: f32, u: [f32; 2], p: f32);
    fn set_state_fields(&mut self, rho: &[f32], u: &[[f32; 2]], p: &[f32]);
}

impl SolverCompressibleIdealGasExt for GpuUnifiedSolver {
    fn set_uniform_state(&mut self, rho: f32, u: [f32; 2], p: f32) {
        let eos = self.model().eos();
        let eos_params = eos.runtime_params();
        let gm1 = eos_params.gm1;
        let r_gas = eos_params.r;

        let stride = self.model().state_layout.stride() as usize;
        let (Some(off_rho), Some(off_rho_u), Some(off_rho_e)) = (
            self.model().state_layout.offset_for("rho"),
            self.model().state_layout.offset_for("rho_u"),
            self.model().state_layout.offset_for("rho_e"),
        ) else {
            return;
        };

        let off_rho = off_rho as usize;
        let off_rho_u = off_rho_u as usize;
        let off_rho_e = off_rho_e as usize;
        let off_p = self
            .model()
            .state_layout
            .offset_for("p")
            .map(|v| v as usize);
        let off_t = self
            .model()
            .state_layout
            .offset_for("T")
            .map(|v| v as usize);
        let off_u = self
            .model()
            .state_layout
            .offset_for("u")
            .or_else(|| self.model().state_layout.offset_for("U"))
            .map(|v| v as usize);

        let ke = 0.5 * rho * (u[0] * u[0] + u[1] * u[1]);
        let rho_e = if gm1 > 0.0 { p / gm1 + ke } else { ke };

        let mut state = vec![0.0f32; self.num_cells() as usize * stride];
        for cell in 0..self.num_cells() as usize {
            let base = cell * stride;
            state[base + off_rho] = rho;
            state[base + off_rho_u] = rho * u[0];
            state[base + off_rho_u + 1] = rho * u[1];
            state[base + off_rho_e] = rho_e;
            if let Some(off_p) = off_p {
                state[base + off_p] = p;
            }
            if let Some(off_t) = off_t {
                state[base + off_t] = if r_gas > 0.0 {
                    p / (rho.max(1e-12) * r_gas)
                } else {
                    0.0
                };
            }
            if let Some(off_u) = off_u {
                state[base + off_u] = u[0];
                state[base + off_u + 1] = u[1];
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
        let (Some(off_rho), Some(off_rho_u), Some(off_rho_e)) = (
            self.model().state_layout.offset_for("rho"),
            self.model().state_layout.offset_for("rho_u"),
            self.model().state_layout.offset_for("rho_e"),
        ) else {
            return;
        };

        let off_rho = off_rho as usize;
        let off_rho_u = off_rho_u as usize;
        let off_rho_e = off_rho_e as usize;
        let off_p = self
            .model()
            .state_layout
            .offset_for("p")
            .map(|v| v as usize);
        let off_t = self
            .model()
            .state_layout
            .offset_for("T")
            .map(|v| v as usize);
        let off_u = self
            .model()
            .state_layout
            .offset_for("u")
            .or_else(|| self.model().state_layout.offset_for("U"))
            .map(|v| v as usize);

        let mut state = vec![0.0f32; self.num_cells() as usize * stride];
        for cell in 0..self.num_cells() as usize {
            let base = cell * stride;
            let rho_val = rho[cell];
            let u_val = u[cell];
            let p_val = p[cell];
            let ke = 0.5 * rho_val * (u_val[0] * u_val[0] + u_val[1] * u_val[1]);
            let rho_e = if gm1 > 0.0 { p_val / gm1 + ke } else { ke };

            state[base + off_rho] = rho_val;
            state[base + off_rho_u] = rho_val * u_val[0];
            state[base + off_rho_u + 1] = rho_val * u_val[1];
            state[base + off_rho_e] = rho_e;
            if let Some(off_p) = off_p {
                state[base + off_p] = p_val;
            }
            if let Some(off_t) = off_t {
                state[base + off_t] = if r_gas > 0.0 {
                    p_val / (rho_val.max(1e-12) * r_gas)
                } else {
                    0.0
                };
            }
            if let Some(off_u) = off_u {
                state[base + off_u] = u_val[0];
                state[base + off_u + 1] = u_val[1];
            }
        }
        let _ = self.write_state_f32(&state);
    }
}

pub trait SolverIncompressibleStatsExt {
    fn incompressible_should_stop(&self) -> bool;
    fn incompressible_outer_stats(&self) -> Option<(u32, f32, f32)>;
    fn incompressible_linear_stats(
        &self,
    ) -> Option<(LinearSolverStats, LinearSolverStats, LinearSolverStats)>;
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

pub trait SolverIncompressibleControlsExt {
    fn set_incompressible_outer_correctors(&mut self, iters: u32) -> Result<(), String>;
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
