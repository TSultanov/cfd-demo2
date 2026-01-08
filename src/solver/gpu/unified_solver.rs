use crate::solver::gpu::plans::compressible::CompressiblePlanResources;
use crate::solver::gpu::plans::generic_coupled::GpuGenericCoupledSolver;
use crate::solver::gpu::enums::{GpuLowMachPrecondModel, TimeScheme};
use crate::solver::gpu::structs::{GpuSolver, LinearSolverStats, PreconditionerType};
use crate::solver::gpu::profiling::ProfilingStats;
use crate::solver::mesh::Mesh;
use crate::solver::model::{ModelFields, ModelSpec};
use crate::solver::scheme::Scheme;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FgmresSizing {
    pub num_unknowns: u32,
    pub num_dot_groups: u32,
}

trait GpuPlanInstance: Send {
    fn num_cells(&self) -> u32;
    fn time(&self) -> f32;
    fn dt(&self) -> f32;
    fn state_buffer(&self) -> &wgpu::Buffer;

    fn set_dt(&mut self, dt: f32);
    fn set_dtau(&mut self, _dtau: f32) {}
    fn set_viscosity(&mut self, mu: f32);
    fn set_density(&mut self, _rho: f32) {}
    fn set_alpha_u(&mut self, alpha_u: f32);
    fn set_alpha_p(&mut self, _alpha_p: f32) {}
    fn set_inlet_velocity(&mut self, velocity: f32);
    fn set_ramp_time(&mut self, _time: f32) {}

    fn set_advection_scheme(&mut self, scheme: Scheme);
    fn set_time_scheme(&mut self, scheme: TimeScheme);
    fn set_preconditioner(&mut self, preconditioner: PreconditionerType);
    fn set_outer_iters(&mut self, _iters: usize) {}

    fn set_incompressible_outer_correctors(&mut self, _iters: u32) -> Result<(), String> {
        Err("set_incompressible_outer_correctors is supported for Incompressible models only".to_string())
    }
    fn incompressible_set_should_stop(&mut self, _value: bool) {}
    fn incompressible_should_stop(&self) -> bool {
        false
    }
    fn incompressible_outer_stats(&self) -> Option<(u32, f32, f32)> {
        None
    }
    fn incompressible_linear_stats(
        &self,
    ) -> Option<(LinearSolverStats, LinearSolverStats, LinearSolverStats)> {
        None
    }
    fn incompressible_degenerate_count(&self) -> Option<u32> {
        None
    }

    fn set_u(&mut self, _u: &[(f64, f64)]) {}
    fn set_p(&mut self, _p: &[f64]) {}

    fn set_uniform_state(&mut self, _rho: f32, _u: [f32; 2], _p: f32) {}
    fn set_state_fields(&mut self, _rho: &[f32], _u: &[[f32; 2]], _p: &[f32]) {}

    fn step(&mut self);
    fn step_with_stats(&mut self) -> Result<Vec<LinearSolverStats>, String> {
        Err("step_with_stats is supported for Compressible models only".to_string())
    }

    fn initialize_history(&self);

    fn set_precond_model(&mut self, _model: GpuLowMachPrecondModel) -> Result<(), String> {
        Err("set_precond_model is supported for Compressible models only".to_string())
    }
    fn set_precond_theta_floor(&mut self, _theta: f32) -> Result<(), String> {
        Err("set_precond_theta_floor is supported for Compressible models only".to_string())
    }
    fn set_nonconverged_relax(&mut self, _alpha: f32) -> Result<(), String> {
        Err("set_nonconverged_relax is supported for Compressible models only".to_string())
    }

    fn enable_detailed_profiling(&self, _enable: bool) -> Result<(), String> {
        Err("enable_detailed_profiling is supported for Incompressible models only".to_string())
    }
    fn start_profiling_session(&self) -> Result<(), String> {
        Err("start_profiling_session is supported for Incompressible models only".to_string())
    }
    fn end_profiling_session(&self) -> Result<(), String> {
        Err("end_profiling_session is supported for Incompressible models only".to_string())
    }
    fn get_profiling_stats(&self) -> Result<Arc<ProfilingStats>, String> {
        Err("get_profiling_stats is supported for Incompressible models only".to_string())
    }
    fn print_profiling_report(&self) -> Result<(), String> {
        Err("print_profiling_report is supported for Incompressible models only".to_string())
    }

    fn read_state_bytes(&self, bytes: u64) -> Pin<Box<dyn Future<Output = Vec<u8>> + '_>>;

    fn set_field_scalar(&mut self, _field: &str, _values: &[f64]) -> Result<(), String> {
        Err("set_field_scalar is supported for GenericCoupled models only".to_string())
    }
    fn get_field_scalar(
        &self,
        _field: String,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<f64>, String>> + '_>> {
        Box::pin(async move {
            Err("get_field_scalar is supported for GenericCoupled models only".to_string())
        })
    }

    fn set_linear_system(&self, _matrix_values: &[f32], _rhs: &[f32]) -> Result<(), String> {
        Err(
            "set_linear_system is supported for Incompressible and GenericCoupled backends only"
                .to_string(),
        )
    }
    fn solve_linear_system_cg_with_size(
        &self,
        _n: u32,
        _max_iters: u32,
        _tol: f32,
    ) -> Result<LinearSolverStats, String> {
        Err(
            "solve_linear_system_cg_with_size is supported for Incompressible and GenericCoupled backends only"
                .to_string(),
        )
    }
    fn get_linear_solution(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<f32>, String>> + '_>> {
        Box::pin(async move {
            Err("get_linear_solution is supported for Incompressible and GenericCoupled backends only".to_string())
        })
    }
    fn coupled_unknowns(&self) -> Result<u32, String> {
        Err(
            "coupled_unknowns is supported for Incompressible and GenericCoupled backends only"
                .to_string(),
        )
    }
    fn fgmres_sizing(&mut self, _max_restart: usize) -> Result<FgmresSizing, String> {
        Err(
            "fgmres_sizing is supported for Incompressible and GenericCoupled backends only"
                .to_string(),
        )
    }
}

impl GpuPlanInstance for GpuSolver {
    fn num_cells(&self) -> u32 {
        self.num_cells
    }

    fn time(&self) -> f32 {
        self.constants.time
    }

    fn dt(&self) -> f32 {
        self.constants.dt
    }

    fn state_buffer(&self) -> &wgpu::Buffer {
        &self.b_state
    }

    fn set_dt(&mut self, dt: f32) {
        self.set_dt(dt);
    }

    fn set_viscosity(&mut self, mu: f32) {
        self.set_viscosity(mu);
    }

    fn set_density(&mut self, rho: f32) {
        self.set_density(rho);
    }

    fn set_alpha_u(&mut self, alpha_u: f32) {
        self.set_alpha_u(alpha_u);
    }

    fn set_alpha_p(&mut self, alpha_p: f32) {
        self.set_alpha_p(alpha_p);
    }

    fn set_inlet_velocity(&mut self, velocity: f32) {
        self.set_inlet_velocity(velocity);
    }

    fn set_ramp_time(&mut self, time: f32) {
        self.set_ramp_time(time);
    }

    fn set_advection_scheme(&mut self, scheme: Scheme) {
        self.set_scheme(scheme.gpu_id());
    }

    fn set_time_scheme(&mut self, scheme: TimeScheme) {
        self.set_time_scheme(scheme as u32);
    }

    fn set_preconditioner(&mut self, preconditioner: PreconditionerType) {
        self.set_precond_type(preconditioner);
    }

    fn set_incompressible_outer_correctors(&mut self, iters: u32) -> Result<(), String> {
        self.n_outer_correctors = iters;
        Ok(())
    }

    fn incompressible_set_should_stop(&mut self, value: bool) {
        self.should_stop = value;
    }

    fn incompressible_should_stop(&self) -> bool {
        self.should_stop
    }

    fn incompressible_outer_stats(&self) -> Option<(u32, f32, f32)> {
        Some((
            *self.outer_iterations.lock().unwrap(),
            *self.outer_residual_u.lock().unwrap(),
            *self.outer_residual_p.lock().unwrap(),
        ))
    }

    fn incompressible_linear_stats(
        &self,
    ) -> Option<(LinearSolverStats, LinearSolverStats, LinearSolverStats)> {
        Some((
            *self.stats_ux.lock().unwrap(),
            *self.stats_uy.lock().unwrap(),
            *self.stats_p.lock().unwrap(),
        ))
    }

    fn incompressible_degenerate_count(&self) -> Option<u32> {
        Some(self.degenerate_count)
    }

    fn set_u(&mut self, u: &[(f64, f64)]) {
        GpuSolver::set_u(self, u);
    }

    fn set_p(&mut self, p: &[f64]) {
        GpuSolver::set_p(self, p);
    }

    fn step(&mut self) {
        crate::solver::gpu::plans::coupled::plan::step_coupled(self);
    }

    fn initialize_history(&self) {
        self.initialize_history();
    }

    fn enable_detailed_profiling(&self, enable: bool) -> Result<(), String> {
        self.enable_detailed_profiling(enable);
        Ok(())
    }

    fn start_profiling_session(&self) -> Result<(), String> {
        self.start_profiling_session();
        Ok(())
    }

    fn end_profiling_session(&self) -> Result<(), String> {
        self.end_profiling_session();
        Ok(())
    }

    fn get_profiling_stats(&self) -> Result<Arc<ProfilingStats>, String> {
        Ok(self.get_profiling_stats())
    }

    fn print_profiling_report(&self) -> Result<(), String> {
        self.print_profiling_report();
        Ok(())
    }

    fn read_state_bytes(&self, bytes: u64) -> Pin<Box<dyn Future<Output = Vec<u8>> + '_>> {
        Box::pin(async move { self.read_buffer(self.state_buffer(), bytes).await })
    }

    fn set_linear_system(&self, matrix_values: &[f32], rhs: &[f32]) -> Result<(), String> {
        self.set_linear_system(matrix_values, rhs);
        Ok(())
    }

    fn solve_linear_system_cg_with_size(
        &self,
        n: u32,
        max_iters: u32,
        tol: f32,
    ) -> Result<LinearSolverStats, String> {
        Ok(self.solve_linear_system_cg_with_size(n, max_iters, tol))
    }

    fn get_linear_solution(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<f32>, String>> + '_>> {
        Box::pin(async move { Ok(self.get_linear_solution().await) })
    }

    fn coupled_unknowns(&self) -> Result<u32, String> {
        Ok(self.coupled_unknowns())
    }

    fn fgmres_sizing(&mut self, max_restart: usize) -> Result<FgmresSizing, String> {
        let resources = self.init_fgmres_resources(max_restart);
        Ok(FgmresSizing {
            num_unknowns: resources.fgmres.n(),
            num_dot_groups: resources.fgmres.num_dot_groups(),
        })
    }
}

impl GpuPlanInstance for CompressiblePlanResources {
    fn num_cells(&self) -> u32 {
        self.num_cells
    }

    fn time(&self) -> f32 {
        self.constants.time
    }

    fn dt(&self) -> f32 {
        self.constants.dt
    }

    fn state_buffer(&self) -> &wgpu::Buffer {
        &self.b_state
    }

    fn set_dt(&mut self, dt: f32) {
        self.set_dt(dt);
    }

    fn set_dtau(&mut self, dtau: f32) {
        self.set_dtau(dtau);
    }

    fn set_viscosity(&mut self, mu: f32) {
        self.set_viscosity(mu);
    }

    fn set_alpha_u(&mut self, alpha_u: f32) {
        self.set_alpha_u(alpha_u);
    }

    fn set_inlet_velocity(&mut self, velocity: f32) {
        self.set_inlet_velocity(velocity);
    }

    fn set_advection_scheme(&mut self, scheme: Scheme) {
        self.set_scheme(scheme.gpu_id());
    }

    fn set_time_scheme(&mut self, scheme: TimeScheme) {
        self.set_time_scheme(scheme as u32);
    }

    fn set_preconditioner(&mut self, preconditioner: PreconditionerType) {
        self.set_precond_type(preconditioner);
    }

    fn set_outer_iters(&mut self, iters: usize) {
        self.set_outer_iters(iters);
    }

    fn set_uniform_state(&mut self, rho: f32, u: [f32; 2], p: f32) {
        CompressiblePlanResources::set_uniform_state(self, rho, u, p);
    }

    fn set_state_fields(&mut self, rho: &[f32], u: &[[f32; 2]], p: &[f32]) {
        CompressiblePlanResources::set_state_fields(self, rho, u, p);
    }

    fn step(&mut self) {
        crate::solver::gpu::plans::compressible::plan::step(self);
    }

    fn step_with_stats(&mut self) -> Result<Vec<LinearSolverStats>, String> {
        Ok(crate::solver::gpu::plans::compressible::plan::step_with_stats(self))
    }

    fn initialize_history(&self) {
        self.initialize_history();
    }

    fn set_precond_model(&mut self, model: GpuLowMachPrecondModel) -> Result<(), String> {
        self.set_precond_model(model as u32);
        Ok(())
    }

    fn set_precond_theta_floor(&mut self, theta: f32) -> Result<(), String> {
        self.set_precond_theta_floor(theta);
        Ok(())
    }

    fn set_nonconverged_relax(&mut self, alpha: f32) -> Result<(), String> {
        self.set_nonconverged_relax(alpha);
        Ok(())
    }

    fn read_state_bytes(&self, bytes: u64) -> Pin<Box<dyn Future<Output = Vec<u8>> + '_>> {
        Box::pin(async move { self.read_buffer(self.state_buffer(), bytes).await })
    }
}

impl GpuPlanInstance for GpuGenericCoupledSolver {
    fn num_cells(&self) -> u32 {
        self.linear.num_cells
    }

    fn time(&self) -> f32 {
        self.linear.constants.time
    }

    fn dt(&self) -> f32 {
        self.linear.constants.dt
    }

    fn state_buffer(&self) -> &wgpu::Buffer {
        self.state_buffer()
    }

    fn set_dt(&mut self, dt: f32) {
        self.linear.set_dt(dt);
    }

    fn set_viscosity(&mut self, mu: f32) {
        self.linear.set_viscosity(mu);
    }

    fn set_density(&mut self, rho: f32) {
        self.linear.set_density(rho);
    }

    fn set_alpha_u(&mut self, alpha_u: f32) {
        self.linear.set_alpha_u(alpha_u);
    }

    fn set_alpha_p(&mut self, alpha_p: f32) {
        self.linear.set_alpha_p(alpha_p);
    }

    fn set_inlet_velocity(&mut self, velocity: f32) {
        self.linear.set_inlet_velocity(velocity);
    }

    fn set_ramp_time(&mut self, time: f32) {
        self.linear.set_ramp_time(time);
    }

    fn set_advection_scheme(&mut self, scheme: Scheme) {
        self.linear.set_scheme(scheme.gpu_id());
    }

    fn set_time_scheme(&mut self, scheme: TimeScheme) {
        self.linear.set_time_scheme(scheme as u32);
    }

    fn set_preconditioner(&mut self, preconditioner: PreconditionerType) {
        self.linear.set_precond_type(preconditioner);
    }

    fn step(&mut self) {
        let _ = self.step();
    }

    fn initialize_history(&self) {}

    fn enable_detailed_profiling(&self, enable: bool) -> Result<(), String> {
        self.linear.enable_detailed_profiling(enable);
        Ok(())
    }

    fn start_profiling_session(&self) -> Result<(), String> {
        self.linear.start_profiling_session();
        Ok(())
    }

    fn end_profiling_session(&self) -> Result<(), String> {
        self.linear.end_profiling_session();
        Ok(())
    }

    fn get_profiling_stats(&self) -> Result<Arc<ProfilingStats>, String> {
        Ok(self.linear.get_profiling_stats())
    }

    fn print_profiling_report(&self) -> Result<(), String> {
        self.linear.print_profiling_report();
        Ok(())
    }

    fn read_state_bytes(&self, bytes: u64) -> Pin<Box<dyn Future<Output = Vec<u8>> + '_>> {
        Box::pin(async move { self.linear.read_buffer(self.state_buffer(), bytes).await })
    }

    fn set_field_scalar(&mut self, field: &str, values: &[f64]) -> Result<(), String> {
        GpuGenericCoupledSolver::set_field_scalar(self, field, values)
    }

    fn get_field_scalar(
        &self,
        field: String,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<f64>, String>> + '_>> {
        Box::pin(async move { GpuGenericCoupledSolver::get_field_scalar(self, &field).await })
    }

    fn set_linear_system(&self, matrix_values: &[f32], rhs: &[f32]) -> Result<(), String> {
        self.linear.set_linear_system(matrix_values, rhs);
        Ok(())
    }

    fn solve_linear_system_cg_with_size(
        &self,
        n: u32,
        max_iters: u32,
        tol: f32,
    ) -> Result<LinearSolverStats, String> {
        Ok(self.linear.solve_linear_system_cg_with_size(n, max_iters, tol))
    }

    fn get_linear_solution(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<f32>, String>> + '_>> {
        Box::pin(async move { Ok(self.linear.get_linear_solution().await) })
    }

    fn coupled_unknowns(&self) -> Result<u32, String> {
        Ok(self.linear.coupled_unknowns())
    }

    fn fgmres_sizing(&mut self, max_restart: usize) -> Result<FgmresSizing, String> {
        let resources = self.linear.init_fgmres_resources(max_restart);
        Ok(FgmresSizing {
            num_unknowns: resources.fgmres.n(),
            num_dot_groups: resources.fgmres.num_dot_groups(),
        })
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
        let plan: Box<dyn GpuPlanInstance> = match &model.fields {
            ModelFields::Incompressible(_) => {
                let mut solver = GpuSolver::new(mesh, device, queue).await;
                apply_config_incompressible(&mut solver, config);
                Box::new(solver)
            }
            ModelFields::Compressible(_) => {
                let mut solver = CompressiblePlanResources::new(mesh, device, queue).await;
                apply_config_compressible(&mut solver, config);
                Box::new(solver)
            }
            ModelFields::GenericCoupled(_) => {
                let mut solver = GpuGenericCoupledSolver::new(mesh, model.clone(), device, queue).await?;
                apply_config_incompressible(&mut solver.linear, config);
                Box::new(solver)
            }
        };

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

fn apply_config_incompressible(solver: &mut GpuSolver, config: SolverConfig) {
    solver.set_scheme(config.advection_scheme.gpu_id());
    solver.set_time_scheme(config.time_scheme as u32);
    solver.set_precond_type(config.preconditioner);
}

fn apply_config_compressible(solver: &mut CompressiblePlanResources, config: SolverConfig) {
    solver.set_scheme(config.advection_scheme.gpu_id());
    solver.set_time_scheme(config.time_scheme as u32);
    solver.set_precond_type(config.preconditioner);
}
