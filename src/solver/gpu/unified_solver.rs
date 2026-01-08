use crate::solver::gpu::plans::compressible::CompressiblePlanResources;
use crate::solver::gpu::plans::generic_coupled::GpuGenericCoupledSolver;
use crate::solver::gpu::plans::plan_instance::{GpuPlanInstance, PlanAction, PlanParam, PlanParamValue};
use crate::solver::gpu::enums::{GpuLowMachPrecondModel, TimeScheme};
use crate::solver::gpu::linear_solver::fgmres::workgroups_for_size;
use crate::solver::gpu::structs::{GpuSolver, LinearSolverStats, PreconditionerType};
use crate::solver::gpu::profiling::ProfilingStats;
use crate::solver::mesh::Mesh;
use crate::solver::model::{ModelFields, ModelSpec};
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
    fn plan_ref<T: 'static>(&self) -> Option<&T> {
        self.plan.as_any().downcast_ref::<T>()
    }

    fn plan_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.plan.as_any_mut().downcast_mut::<T>()
    }

    pub async fn new(
        mesh: &Mesh,
        model: ModelSpec,
        config: SolverConfig,
        device: Option<wgpu::Device>,
        queue: Option<wgpu::Queue>,
    ) -> Result<Self, String> {
        let mut plan: Box<dyn GpuPlanInstance> = match &model.fields {
            ModelFields::Incompressible(_) => {
                Box::new(GpuSolver::new(mesh, device, queue).await)
            }
            ModelFields::Compressible(_) => {
                Box::new(CompressiblePlanResources::new(mesh, device, queue).await)
            }
            ModelFields::GenericCoupled(_) => {
                let solver = GpuGenericCoupledSolver::new(mesh, model.clone(), device, queue).await?;
                Box::new(solver)
            }
        };
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
        if let Some(plan) = self.plan_ref::<GpuSolver>() {
            plan.set_u(u);
        }
    }

    pub fn set_p(&mut self, p: &[f64]) {
        if let Some(plan) = self.plan_ref::<GpuSolver>() {
            plan.set_p(p);
        }
    }

    pub fn set_uniform_state(&mut self, rho: f32, u: [f32; 2], p: f32) {
        if let Some(plan) = self.plan_ref::<CompressiblePlanResources>() {
            plan.set_uniform_state(rho, u, p);
        }
    }

    pub fn set_state_fields(&mut self, rho: &[f32], u: &[[f32; 2]], p: &[f32]) {
        if let Some(plan) = self.plan_ref::<CompressiblePlanResources>() {
            plan.set_state_fields(rho, u, p);
        }
    }

    pub fn step(&mut self) {
        self.plan.step();
    }

    pub fn step_with_stats(&mut self) -> Result<Vec<LinearSolverStats>, String> {
        if let Some(plan) = self.plan_mut::<CompressiblePlanResources>() {
            Ok(plan.step_with_stats())
        } else {
            Err("step_with_stats is only supported for compressible plans".into())
        }
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
        self.plan
            .write_state_bytes(bytemuck::cast_slice(&state))
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
        if let Some(plan) = self.plan_ref::<GpuSolver>() {
            plan.set_linear_system(matrix_values, rhs);
            Ok(())
        } else {
            Err("set_linear_system is only supported for incompressible plans".into())
        }
    }

    pub fn solve_linear_system_cg_with_size(
        &self,
        n: u32,
        max_iters: u32,
        tol: f32,
    ) -> Result<LinearSolverStats, String> {
        if let Some(plan) = self.plan_ref::<GpuSolver>() {
            Ok(plan.solve_linear_system_cg_with_size(n, max_iters, tol))
        } else {
            Err("CG solve is only supported for incompressible plans".into())
        }
    }

    pub async fn get_linear_solution(&self) -> Result<Vec<f32>, String> {
        if let Some(plan) = self.plan_ref::<GpuSolver>() {
            Ok(plan.get_linear_solution().await)
        } else {
            Err("get_linear_solution is only supported for incompressible plans".into())
        }
    }

    pub fn coupled_unknowns(&self) -> Result<u32, String> {
        if let Some(plan) = self.plan_ref::<GpuSolver>() {
            Ok(plan.coupled_unknowns())
        } else {
            Err("coupled_unknowns is only supported for incompressible plans".into())
        }
    }

    pub fn fgmres_sizing(&mut self, max_restart: usize) -> Result<FgmresSizing, String> {
        let _ = max_restart;
        if let Some(plan) = self.plan_ref::<GpuSolver>() {
            let n = plan.coupled_unknowns();
            Ok(FgmresSizing {
                num_unknowns: n,
                num_dot_groups: workgroups_for_size(n),
            })
        } else {
            Err("fgmres_sizing is only supported for incompressible plans".into())
        }
    }
}
