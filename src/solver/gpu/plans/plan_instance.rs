use crate::solver::gpu::enums::{GpuLowMachPrecondModel, TimeScheme};
use crate::solver::gpu::plans::compressible::CompressiblePlanResources;
use crate::solver::gpu::plans::generic_coupled::GpuGenericCoupledSolver;
use crate::solver::gpu::profiling::ProfilingStats;
use crate::solver::gpu::structs::{GpuSolver, LinearSolverStats, PreconditionerType};
use crate::solver::scheme::Scheme;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FgmresSizing {
    pub num_unknowns: u32,
    pub num_dot_groups: u32,
}

pub(crate) type PlanFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

pub(crate) trait GpuPlanInstance: Send {
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
        Err(
            "set_incompressible_outer_correctors is supported for Incompressible models only"
                .to_string(),
        )
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

    fn read_state_bytes(&self, bytes: u64) -> PlanFuture<'_, Vec<u8>>;

    fn set_field_scalar(&mut self, _field: &str, _values: &[f64]) -> Result<(), String> {
        Err("set_field_scalar is supported for GenericCoupled models only".to_string())
    }
    fn get_field_scalar(&self, _field: String) -> PlanFuture<'_, Result<Vec<f64>, String>> {
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
    fn get_linear_solution(&self) -> PlanFuture<'_, Result<Vec<f32>, String>> {
        Box::pin(async move {
            Err(
                "get_linear_solution is supported for Incompressible and GenericCoupled backends only"
                    .to_string(),
            )
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
        GpuSolver::set_dt(self, dt);
    }

    fn set_viscosity(&mut self, mu: f32) {
        GpuSolver::set_viscosity(self, mu);
    }

    fn set_density(&mut self, rho: f32) {
        GpuSolver::set_density(self, rho);
    }

    fn set_alpha_u(&mut self, alpha_u: f32) {
        GpuSolver::set_alpha_u(self, alpha_u);
    }

    fn set_alpha_p(&mut self, alpha_p: f32) {
        GpuSolver::set_alpha_p(self, alpha_p);
    }

    fn set_inlet_velocity(&mut self, velocity: f32) {
        GpuSolver::set_inlet_velocity(self, velocity);
    }

    fn set_ramp_time(&mut self, time: f32) {
        GpuSolver::set_ramp_time(self, time);
    }

    fn set_advection_scheme(&mut self, scheme: Scheme) {
        GpuSolver::set_scheme(self, scheme.gpu_id());
    }

    fn set_time_scheme(&mut self, scheme: TimeScheme) {
        GpuSolver::set_time_scheme(self, scheme as u32);
    }

    fn set_preconditioner(&mut self, preconditioner: PreconditionerType) {
        GpuSolver::set_precond_type(self, preconditioner);
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
        GpuSolver::initialize_history(self);
    }

    fn enable_detailed_profiling(&self, enable: bool) -> Result<(), String> {
        GpuSolver::enable_detailed_profiling(self, enable);
        Ok(())
    }

    fn start_profiling_session(&self) -> Result<(), String> {
        GpuSolver::start_profiling_session(self);
        Ok(())
    }

    fn end_profiling_session(&self) -> Result<(), String> {
        GpuSolver::end_profiling_session(self);
        Ok(())
    }

    fn get_profiling_stats(&self) -> Result<Arc<ProfilingStats>, String> {
        Ok(GpuSolver::get_profiling_stats(self))
    }

    fn print_profiling_report(&self) -> Result<(), String> {
        GpuSolver::print_profiling_report(self);
        Ok(())
    }

    fn read_state_bytes(&self, bytes: u64) -> PlanFuture<'_, Vec<u8>> {
        Box::pin(async move { self.read_buffer(self.state_buffer(), bytes).await })
    }

    fn set_linear_system(&self, matrix_values: &[f32], rhs: &[f32]) -> Result<(), String> {
        GpuSolver::set_linear_system(self, matrix_values, rhs);
        Ok(())
    }

    fn solve_linear_system_cg_with_size(
        &self,
        n: u32,
        max_iters: u32,
        tol: f32,
    ) -> Result<LinearSolverStats, String> {
        Ok(GpuSolver::solve_linear_system_cg_with_size(
            self, n, max_iters, tol,
        ))
    }

    fn get_linear_solution(&self) -> PlanFuture<'_, Result<Vec<f32>, String>> {
        Box::pin(async move { Ok(GpuSolver::get_linear_solution(self).await) })
    }

    fn coupled_unknowns(&self) -> Result<u32, String> {
        Ok(GpuSolver::coupled_unknowns(self))
    }

    fn fgmres_sizing(
        &mut self,
        max_restart: usize,
    ) -> Result<FgmresSizing, String> {
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
        CompressiblePlanResources::set_dt(self, dt);
    }

    fn set_dtau(&mut self, dtau: f32) {
        CompressiblePlanResources::set_dtau(self, dtau);
    }

    fn set_viscosity(&mut self, mu: f32) {
        CompressiblePlanResources::set_viscosity(self, mu);
    }

    fn set_alpha_u(&mut self, alpha_u: f32) {
        CompressiblePlanResources::set_alpha_u(self, alpha_u);
    }

    fn set_inlet_velocity(&mut self, velocity: f32) {
        CompressiblePlanResources::set_inlet_velocity(self, velocity);
    }

    fn set_advection_scheme(&mut self, scheme: Scheme) {
        CompressiblePlanResources::set_scheme(self, scheme.gpu_id());
    }

    fn set_time_scheme(&mut self, scheme: TimeScheme) {
        CompressiblePlanResources::set_time_scheme(self, scheme as u32);
    }

    fn set_preconditioner(&mut self, preconditioner: PreconditionerType) {
        CompressiblePlanResources::set_precond_type(self, preconditioner);
    }

    fn set_outer_iters(&mut self, iters: usize) {
        CompressiblePlanResources::set_outer_iters(self, iters);
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
        CompressiblePlanResources::initialize_history(self);
    }

    fn set_precond_model(&mut self, model: GpuLowMachPrecondModel) -> Result<(), String> {
        CompressiblePlanResources::set_precond_model(self, model as u32);
        Ok(())
    }

    fn set_precond_theta_floor(&mut self, theta: f32) -> Result<(), String> {
        CompressiblePlanResources::set_precond_theta_floor(self, theta);
        Ok(())
    }

    fn set_nonconverged_relax(&mut self, alpha: f32) -> Result<(), String> {
        CompressiblePlanResources::set_nonconverged_relax(self, alpha);
        Ok(())
    }

    fn read_state_bytes(&self, bytes: u64) -> PlanFuture<'_, Vec<u8>> {
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
        let _ = GpuGenericCoupledSolver::step(self);
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

    fn read_state_bytes(&self, bytes: u64) -> PlanFuture<'_, Vec<u8>> {
        Box::pin(async move { self.linear.read_buffer(self.state_buffer(), bytes).await })
    }

    fn set_field_scalar(&mut self, field: &str, values: &[f64]) -> Result<(), String> {
        GpuGenericCoupledSolver::set_field_scalar(self, field, values)
    }

    fn get_field_scalar(&self, field: String) -> PlanFuture<'_, Result<Vec<f64>, String>> {
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

    fn get_linear_solution(&self) -> PlanFuture<'_, Result<Vec<f32>, String>> {
        Box::pin(async move { Ok(self.linear.get_linear_solution().await) })
    }

    fn coupled_unknowns(&self) -> Result<u32, String> {
        Ok(self.linear.coupled_unknowns())
    }

    fn fgmres_sizing(
        &mut self,
        max_restart: usize,
    ) -> Result<FgmresSizing, String> {
        let resources = self.linear.init_fgmres_resources(max_restart);
        Ok(FgmresSizing {
            num_unknowns: resources.fgmres.n(),
            num_dot_groups: resources.fgmres.num_dot_groups(),
        })
    }
}
