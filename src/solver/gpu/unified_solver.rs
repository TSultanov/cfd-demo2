use crate::solver::gpu::plans::compressible::CompressiblePlanResources;
use crate::solver::gpu::plans::generic_coupled::GpuGenericCoupledSolver;
use crate::solver::gpu::plans::plan_instance::GpuPlanInstance;
use crate::solver::gpu::enums::{GpuLowMachPrecondModel, TimeScheme};
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
        self.plan.set_dtau(dtau);
    }

    pub fn set_viscosity(&mut self, mu: f32) {
        self.plan.set_viscosity(mu);
    }

    pub fn set_density(&mut self, rho: f32) {
        self.plan.set_density(rho);
    }

    pub fn set_alpha_u(&mut self, alpha_u: f32) {
        self.plan.set_alpha_u(alpha_u);
    }

    pub fn set_alpha_p(&mut self, alpha_p: f32) {
        self.plan.set_alpha_p(alpha_p);
    }

    pub fn set_inlet_velocity(&mut self, velocity: f32) {
        self.plan.set_inlet_velocity(velocity);
    }

    pub fn set_ramp_time(&mut self, time: f32) {
        self.plan.set_ramp_time(time);
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
        self.plan.set_outer_iters(iters);
    }

    pub fn set_incompressible_outer_correctors(&mut self, iters: u32) -> Result<(), String> {
        self.plan.set_incompressible_outer_correctors(iters)
    }

    pub fn incompressible_set_should_stop(&mut self, value: bool) {
        self.plan.incompressible_set_should_stop(value);
    }

    pub fn incompressible_should_stop(&self) -> bool {
        self.plan.incompressible_should_stop()
    }

    pub fn incompressible_outer_stats(&self) -> Option<(u32, f32, f32)> {
        self.plan.incompressible_outer_stats()
    }

    pub fn incompressible_linear_stats(&self) -> Option<(LinearSolverStats, LinearSolverStats, LinearSolverStats)> {
        self.plan.incompressible_linear_stats()
    }

    pub fn incompressible_degenerate_count(&self) -> Option<u32> {
        self.plan.incompressible_degenerate_count()
    }

    pub fn set_u(&mut self, u: &[(f64, f64)]) {
        self.plan.set_u(u);
    }

    pub fn set_p(&mut self, p: &[f64]) {
        self.plan.set_p(p);
    }

    pub fn set_uniform_state(&mut self, rho: f32, u: [f32; 2], p: f32) {
        self.plan.set_uniform_state(rho, u, p);
    }

    pub fn set_state_fields(&mut self, rho: &[f32], u: &[[f32; 2]], p: &[f32]) {
        self.plan.set_state_fields(rho, u, p);
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
        self.plan.set_precond_model(model)
    }

    pub fn set_precond_theta_floor(&mut self, theta: f32) -> Result<(), String> {
        self.plan.set_precond_theta_floor(theta)
    }

    pub fn set_nonconverged_relax(&mut self, alpha: f32) -> Result<(), String> {
        self.plan.set_nonconverged_relax(alpha)
    }

    pub fn enable_detailed_profiling(&self, enable: bool) -> Result<(), String> {
        self.plan.enable_detailed_profiling(enable)
    }

    pub fn start_profiling_session(&self) -> Result<(), String> {
        self.plan.start_profiling_session()
    }

    pub fn end_profiling_session(&self) -> Result<(), String> {
        self.plan.end_profiling_session()
    }

    pub fn get_profiling_stats(&self) -> Result<Arc<ProfilingStats>, String> {
        self.plan.get_profiling_stats()
    }

    pub fn print_profiling_report(&self) -> Result<(), String> {
        self.plan.print_profiling_report()
    }

    pub async fn read_state_f32(&self) -> Vec<f32> {
        let stride = self.model.state_layout.stride() as u64;
        let bytes = self.num_cells() as u64 * stride * 4;
        let raw = self.plan.read_state_bytes(bytes).await;
        bytemuck::cast_slice(&raw).to_vec()
    }

    pub fn set_field_scalar(&mut self, field: &str, values: &[f64]) -> Result<(), String> {
        self.plan.set_field_scalar(field, values)
    }

    pub async fn get_field_scalar(&self, field: &str) -> Result<Vec<f64>, String> {
        self.plan.get_field_scalar(field.to_string()).await
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
        &self,
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
}
