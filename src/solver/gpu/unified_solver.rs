use crate::solver::gpu::plans::plan_instance::{
    build_plan_instance, GpuPlanInstance, PlanAction, PlanParam, PlanParamValue,
};
use crate::solver::gpu::enums::{GpuLowMachPrecondModel, TimeScheme};
use crate::solver::gpu::structs::{LinearSolverStats, PreconditionerType};
use crate::solver::gpu::profiling::ProfilingStats;
use crate::solver::mesh::Mesh;
use crate::solver::model::ModelSpec;
use crate::solver::scheme::Scheme;
use std::sync::Arc;

pub use crate::solver::gpu::plans::plan_instance::FgmresSizing;

#[derive(Debug, Clone, Copy)]
pub struct SolverConfig {
    pub advection_scheme: Scheme,
    pub time_scheme: TimeScheme,
    pub preconditioner: PreconditionerType,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            advection_scheme: Scheme::Upwind,
            time_scheme: TimeScheme::Euler,
            preconditioner: PreconditionerType::Jacobi,
        }
    }
}

pub struct GpuUnifiedSolver {
    model: ModelSpec,
    plan: Box<dyn GpuPlanInstance>,
    config: SolverConfig,
}

impl GpuUnifiedSolver {
    fn offset_for_any(&self, fields: &[&str]) -> Option<usize> {
        fields
            .iter()
            .find_map(|name| self.model.state_layout.offset_for(name))
            .map(|v| v as usize)
    }

    pub async fn new(
        mesh: &Mesh,
        model: ModelSpec,
        config: SolverConfig,
        device: Option<wgpu::Device>,
        queue: Option<wgpu::Queue>,
    ) -> Result<Self, String> {
        let mut plan: Box<dyn GpuPlanInstance> = build_plan_instance(mesh, &model, device, queue).await?;
        plan.set_advection_scheme(config.advection_scheme);
        plan.set_time_scheme(config.time_scheme);
        plan.set_preconditioner(config.preconditioner);

        Ok(Self {
            model,
            plan,
            config,
        })
    }

    pub fn model(&self) -> &ModelSpec {
        &self.model
    }

    pub fn config(&self) -> SolverConfig {
        self.config
    }

    pub fn num_cells(&self) -> u32 {
        self.plan.num_cells()
    }

    pub fn time(&self) -> f32 {
        self.plan.time()
    }

    pub fn dt(&self) -> f32 {
        self.plan.dt()
    }

    pub fn state_buffer(&self) -> &wgpu::Buffer {
        self.plan.state_buffer()
    }

    pub fn set_dt(&mut self, dt: f32) {
        self.plan.set_dt(dt);
    }

    pub fn set_dtau(&mut self, dtau: f32) {
        let _ = self
            .plan
            .set_param(PlanParam::Dtau, PlanParamValue::F32(dtau));
    }

    pub fn set_viscosity(&mut self, mu: f32) {
        let _ = self
            .plan
            .set_param(PlanParam::Viscosity, PlanParamValue::F32(mu));
    }

    pub fn set_density(&mut self, rho: f32) {
        let _ = self
            .plan
            .set_param(PlanParam::Density, PlanParamValue::F32(rho));
    }

    pub fn set_alpha_u(&mut self, alpha_u: f32) {
        let _ = self
            .plan
            .set_param(PlanParam::AlphaU, PlanParamValue::F32(alpha_u));
    }

    pub fn set_alpha_p(&mut self, alpha_p: f32) {
        let _ = self
            .plan
            .set_param(PlanParam::AlphaP, PlanParamValue::F32(alpha_p));
    }

    pub fn set_inlet_velocity(&mut self, velocity: f32) {
        let _ = self.plan.set_param(
            PlanParam::InletVelocity,
            PlanParamValue::F32(velocity),
        );
    }

    pub fn set_ramp_time(&mut self, time: f32) {
        let _ = self
            .plan
            .set_param(PlanParam::RampTime, PlanParamValue::F32(time));
    }

    pub fn set_advection_scheme(&mut self, scheme: Scheme) {
        self.config.advection_scheme = scheme;
        self.plan.set_advection_scheme(scheme);
    }

    pub fn set_time_scheme(&mut self, scheme: TimeScheme) {
        self.config.time_scheme = scheme;
        self.plan.set_time_scheme(scheme);
    }

    pub fn set_preconditioner(&mut self, preconditioner: PreconditionerType) {
        self.config.preconditioner = preconditioner;
        self.plan.set_preconditioner(preconditioner);
    }

    pub fn set_outer_iters(&mut self, iters: usize) {
        let _ = self
            .plan
            .set_param(PlanParam::OuterIters, PlanParamValue::Usize(iters));
    }

    pub fn set_incompressible_outer_correctors(&mut self, iters: u32) -> Result<(), String> {
        self.plan.set_param(
            PlanParam::IncompressibleOuterCorrectors,
            PlanParamValue::U32(iters),
        )
    }

    pub fn incompressible_set_should_stop(&mut self, value: bool) {
        let _ = self.plan.set_param(
            PlanParam::IncompressibleShouldStop,
            PlanParamValue::Bool(value),
        );
    }

    pub fn incompressible_should_stop(&self) -> bool {
        self.plan.step_stats().should_stop.unwrap_or(false)
    }

    pub fn incompressible_outer_stats(&self) -> Option<(u32, f32, f32)> {
        let stats = self.plan.step_stats();
        Some((
            stats.outer_iterations?,
            stats.outer_residual_u?,
            stats.outer_residual_p?,
        ))
    }

    pub fn incompressible_linear_stats(&self) -> Option<(LinearSolverStats, LinearSolverStats, LinearSolverStats)> {
        self.plan.step_stats().linear_stats
    }

    pub fn incompressible_degenerate_count(&self) -> Option<u32> {
        self.plan.step_stats().degenerate_count
    }

    pub fn set_u(&mut self, u: &[(f64, f64)]) {
        let _ = self.set_field_vec2_any(&["U", "u"], u);
    }

    pub fn set_p(&mut self, p: &[f64]) {
        let _ = self.set_field_scalar("p", p);
    }

    pub fn set_uniform_state(&mut self, rho: f32, u: [f32; 2], p: f32) {
        let stride = self.model.state_layout.stride() as usize;
        let Some(offset_rho) = self.offset_for_any(&["rho"]) else {
            return;
        };
        let Some(offset_rho_u) = self.offset_for_any(&["rho_u"]) else {
            return;
        };
        let Some(offset_rho_e) = self.offset_for_any(&["rho_e"]) else {
            return;
        };
        let Some(offset_p) = self.offset_for_any(&["p"]) else {
            return;
        };
        let Some(offset_u) = self.offset_for_any(&["u", "U"]) else {
            return;
        };

        let gamma = 1.4f32;
        let ke = 0.5 * rho * (u[0] * u[0] + u[1] * u[1]);
        let rho_e = p / (gamma - 1.0) + ke;

        let mut state = vec![0.0f32; self.num_cells() as usize * stride];
        for cell in 0..self.num_cells() as usize {
            let base = cell * stride;
            state[base + offset_rho] = rho;
            state[base + offset_rho_u] = rho * u[0];
            state[base + offset_rho_u + 1] = rho * u[1];
            state[base + offset_rho_e] = rho_e;
            state[base + offset_p] = p;
            state[base + offset_u] = u[0];
            state[base + offset_u + 1] = u[1];
        }
        let _ = self.plan.write_state_bytes(bytemuck::cast_slice(&state));
    }

    pub fn set_state_fields(&mut self, rho: &[f32], u: &[[f32; 2]], p: &[f32]) {
        if rho.len() != self.num_cells() as usize || u.len() != self.num_cells() as usize || p.len() != self.num_cells() as usize {
            return;
        }

        let stride = self.model.state_layout.stride() as usize;
        let Some(offset_rho) = self.offset_for_any(&["rho"]) else {
            return;
        };
        let Some(offset_rho_u) = self.offset_for_any(&["rho_u"]) else {
            return;
        };
        let Some(offset_rho_e) = self.offset_for_any(&["rho_e"]) else {
            return;
        };
        let Some(offset_p) = self.offset_for_any(&["p"]) else {
            return;
        };
        let Some(offset_u) = self.offset_for_any(&["u", "U"]) else {
            return;
        };

        let gamma = 1.4f32;
        let mut state = vec![0.0f32; self.num_cells() as usize * stride];
        for cell in 0..self.num_cells() as usize {
            let base = cell * stride;
            let rho_val = rho[cell];
            let u_val = u[cell];
            let p_val = p[cell];
            let ke = 0.5 * rho_val * (u_val[0] * u_val[0] + u_val[1] * u_val[1]);
            let rho_e = p_val / (gamma - 1.0) + ke;

            state[base + offset_rho] = rho_val;
            state[base + offset_rho_u] = rho_val * u_val[0];
            state[base + offset_rho_u + 1] = rho_val * u_val[1];
            state[base + offset_rho_e] = rho_e;
            state[base + offset_p] = p_val;
            state[base + offset_u] = u_val[0];
            state[base + offset_u + 1] = u_val[1];
        }
        let _ = self.plan.write_state_bytes(bytemuck::cast_slice(&state));
    }

    pub fn step(&mut self) {
        self.plan.step();
    }

    pub fn step_with_stats(&mut self) -> Result<Vec<LinearSolverStats>, String> {
        self.plan.step_with_stats()
    }

    pub fn initialize_history(&self) {
        self.plan.initialize_history();
    }

    pub fn set_precond_model(&mut self, model: GpuLowMachPrecondModel) -> Result<(), String> {
        self.plan.set_param(
            PlanParam::LowMachModel,
            PlanParamValue::LowMachModel(model),
        )
    }

    pub fn set_precond_theta_floor(&mut self, theta: f32) -> Result<(), String> {
        self.plan.set_param(
            PlanParam::LowMachThetaFloor,
            PlanParamValue::F32(theta),
        )
    }

    pub fn set_nonconverged_relax(&mut self, alpha: f32) -> Result<(), String> {
        self.plan.set_param(
            PlanParam::NonconvergedRelax,
            PlanParamValue::F32(alpha),
        )
    }

    pub fn enable_detailed_profiling(&mut self, enable: bool) -> Result<(), String> {
        self.plan.set_param(
            PlanParam::DetailedProfilingEnabled,
            PlanParamValue::Bool(enable),
        )
    }

    pub fn start_profiling_session(&self) -> Result<(), String> {
        self.plan.perform(PlanAction::StartProfilingSession)
    }

    pub fn end_profiling_session(&self) -> Result<(), String> {
        self.plan.perform(PlanAction::EndProfilingSession)
    }

    pub fn get_profiling_stats(&self) -> Result<Arc<ProfilingStats>, String> {
        self.plan
            .profiling_stats()
            .ok_or_else(|| "profiling stats are not supported by this plan".into())
    }

    pub fn print_profiling_report(&self) -> Result<(), String> {
        self.plan.perform(PlanAction::PrintProfilingReport)
    }

    pub async fn read_state_f32(&self) -> Vec<f32> {
        let stride = self.model.state_layout.stride() as u64;
        let bytes = self.num_cells() as u64 * stride * 4;
        let raw = self.plan.read_state_bytes(bytes).await;
        bytemuck::cast_slice(&raw).to_vec()
    }

    pub fn set_field_scalar(&mut self, field: &str, values: &[f64]) -> Result<(), String> {
        let stride = self.model.state_layout.stride() as usize;
        let offset = self
            .model
            .state_layout
            .offset_for(field)
            .ok_or_else(|| format!("field '{field}' not found in layout"))? as usize;
        if values.len() != self.num_cells() as usize {
            return Err(format!(
                "value length {} does not match num_cells {}",
                values.len(),
                self.num_cells()
            ));
        }

        let mut state = pollster::block_on(async { self.read_state_f32().await });
        if state.len() != self.num_cells() as usize * stride {
            state.resize(self.num_cells() as usize * stride, 0.0);
        }
        for (i, &v) in values.iter().enumerate() {
            state[i * stride + offset] = v as f32;
        }
        self.plan.write_state_bytes(bytemuck::cast_slice(&state))
    }

    pub async fn get_field_scalar(&self, field: &str) -> Result<Vec<f64>, String> {
        let data = self.read_state_f32().await;
        let stride = self.model.state_layout.stride() as usize;
        let offset = self
            .model
            .state_layout
            .offset_for(field)
            .ok_or_else(|| format!("field '{field}' not found in layout"))? as usize;
        Ok((0..self.num_cells() as usize)
            .map(|i| data[i * stride + offset] as f64)
            .collect())
    }

    pub async fn get_p(&self) -> Vec<f64> {
        let data = self.read_state_f32().await;
        let stride = self.model.state_layout.stride() as usize;
        let offset = self.model.state_layout.offset_for("p").unwrap_or(0) as usize;
        (0..self.num_cells() as usize)
            .map(|i| data[i * stride + offset] as f64)
            .collect()
    }

    pub async fn get_u(&self) -> Vec<(f64, f64)> {
        let data = self.read_state_f32().await;
        let stride = self.model.state_layout.stride() as usize;
        let offset = self
            .model
            .state_layout
            .offset_for("U")
            .or_else(|| self.model.state_layout.offset_for("u"))
            .unwrap_or(0) as usize;
        (0..self.num_cells() as usize)
            .map(|i| {
                let base = i * stride + offset;
                (data[base] as f64, data[base + 1] as f64)
            })
            .collect()
    }

    pub async fn get_rho(&self) -> Vec<f64> {
        let data = self.read_state_f32().await;
        let stride = self.model.state_layout.stride() as usize;
        let offset = self.model.state_layout.offset_for("rho").unwrap_or(0) as usize;
        (0..self.num_cells() as usize)
            .map(|i| data[i * stride + offset] as f64)
            .collect()
    }

    pub fn set_linear_system(&self, matrix_values: &[f32], rhs: &[f32]) -> Result<(), String> {
        self.plan.set_linear_system(matrix_values, rhs)
    }

    pub fn solve_linear_system_cg_with_size(
        &mut self,
        n: u32,
        max_iters: u32,
        tol: f32,
    ) -> Result<LinearSolverStats, String> {
        self.plan.solve_linear_system_cg_with_size(n, max_iters, tol)
    }

    pub async fn get_linear_solution(&self) -> Result<Vec<f32>, String> {
        self.plan.get_linear_solution().await
    }

    pub fn coupled_unknowns(&self) -> Result<u32, String> {
        self.plan.coupled_unknowns()
    }

    pub fn fgmres_sizing(&mut self, max_restart: usize) -> Result<FgmresSizing, String> {
        self.plan.fgmres_sizing(max_restart)
    }

    fn set_field_vec2_any(&mut self, fields: &[&str], values: &[(f64, f64)]) -> Result<(), String> {
        let stride = self.model.state_layout.stride() as usize;
        let offset = self
            .offset_for_any(fields)
            .ok_or_else(|| format!("field '{}' not found in layout", fields.join("'/'")))?;
        if values.len() != self.num_cells() as usize {
            return Err(format!(
                "value length {} does not match num_cells {}",
                values.len(),
                self.num_cells()
            ));
        }
        let mut state = pollster::block_on(async { self.read_state_f32().await });
        for (i, &(x, y)) in values.iter().enumerate() {
            let base = i * stride + offset;
            state[base] = x as f32;
            state[base + 1] = y as f32;
        }
        self.plan.write_state_bytes(bytemuck::cast_slice(&state))
    }
}
