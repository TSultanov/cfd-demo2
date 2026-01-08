use crate::solver::gpu::enums::{GpuLowMachPrecondModel, TimeScheme};
use crate::solver::gpu::plans::compressible::CompressiblePlanResources;
use crate::solver::gpu::plans::generic_coupled::GpuGenericCoupledSolver;
use crate::solver::gpu::profiling::ProfilingStats;
use crate::solver::gpu::structs::{GpuSolver, LinearSolverStats, PreconditionerType};
use crate::solver::mesh::Mesh;
use crate::solver::model::{ModelFields, ModelSpec};
use crate::solver::scheme::Scheme;
use std::any::Any;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

pub(crate) type PlanFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FgmresSizing {
    pub num_unknowns: u32,
    pub num_dot_groups: u32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PlanStepStats {
    pub should_stop: Option<bool>,
    pub degenerate_count: Option<u32>,
    pub outer_iterations: Option<u32>,
    pub outer_residual_u: Option<f32>,
    pub outer_residual_p: Option<f32>,
    pub linear_stats: Option<(LinearSolverStats, LinearSolverStats, LinearSolverStats)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanParam {
    Viscosity,
    Density,
    AlphaU,
    AlphaP,
    InletVelocity,
    RampTime,
    Dtau,
    OuterIters,
    IncompressibleOuterCorrectors,
    IncompressibleShouldStop,
    LowMachModel,
    LowMachThetaFloor,
    NonconvergedRelax,
    DetailedProfilingEnabled,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlanParamValue {
    F32(f32),
    U32(u32),
    Usize(usize),
    Bool(bool),
    LowMachModel(GpuLowMachPrecondModel),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanAction {
    StartProfilingSession,
    EndProfilingSession,
    PrintProfilingReport,
}

pub(crate) trait GpuPlanInstance: Any + Send {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn num_cells(&self) -> u32;
    fn time(&self) -> f32;
    fn dt(&self) -> f32;
    fn state_buffer(&self) -> &wgpu::Buffer;

    fn set_dt(&mut self, dt: f32);

    fn set_advection_scheme(&mut self, scheme: Scheme);
    fn set_time_scheme(&mut self, scheme: TimeScheme);
    fn set_preconditioner(&mut self, preconditioner: PreconditionerType);

    fn set_param(&mut self, _param: PlanParam, _value: PlanParamValue) -> Result<(), String> {
        Err("parameter is not supported by this plan".into())
    }

    fn write_state_bytes(&self, bytes: &[u8]) -> Result<(), String>;

    fn step_stats(&self) -> PlanStepStats {
        PlanStepStats::default()
    }

    fn perform(&self, _action: PlanAction) -> Result<(), String> {
        Err("action is not supported by this plan".into())
    }

    fn profiling_stats(&self) -> Option<Arc<ProfilingStats>> {
        None
    }

    fn step_with_stats(&mut self) -> Result<Vec<LinearSolverStats>, String> {
        Err("step_with_stats is not supported by this plan".into())
    }

    fn set_linear_system(&self, _matrix_values: &[f32], _rhs: &[f32]) -> Result<(), String> {
        Err("set_linear_system is not supported by this plan".into())
    }

    fn solve_linear_system_cg_with_size(
        &self,
        _n: u32,
        _max_iters: u32,
        _tol: f32,
    ) -> Result<LinearSolverStats, String> {
        Err("solve_linear_system_cg_with_size is not supported by this plan".into())
    }

    fn get_linear_solution(&self) -> PlanFuture<'_, Result<Vec<f32>, String>> {
        Box::pin(async { Err("get_linear_solution is not supported by this plan".into()) })
    }

    fn coupled_unknowns(&self) -> Result<u32, String> {
        Err("coupled_unknowns is not supported by this plan".into())
    }

    fn fgmres_sizing(&mut self, _max_restart: usize) -> Result<FgmresSizing, String> {
        Err("fgmres_sizing is not supported by this plan".into())
    }

    fn step(&mut self);

    fn initialize_history(&self);

    fn read_state_bytes(&self, bytes: u64) -> PlanFuture<'_, Vec<u8>>;
}

impl GpuPlanInstance for GpuSolver {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

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

    fn set_advection_scheme(&mut self, scheme: Scheme) {
        GpuSolver::set_scheme(self, scheme.gpu_id());
    }

    fn set_time_scheme(&mut self, scheme: TimeScheme) {
        GpuSolver::set_time_scheme(self, scheme as u32);
    }

    fn set_preconditioner(&mut self, preconditioner: PreconditionerType) {
        GpuSolver::set_precond_type(self, preconditioner);
    }

    fn set_param(&mut self, param: PlanParam, value: PlanParamValue) -> Result<(), String> {
        match (param, value) {
            (PlanParam::Viscosity, PlanParamValue::F32(mu)) => {
                self.set_viscosity(mu);
                Ok(())
            }
            (PlanParam::Density, PlanParamValue::F32(rho)) => {
                self.set_density(rho);
                Ok(())
            }
            (PlanParam::AlphaU, PlanParamValue::F32(alpha)) => {
                self.set_alpha_u(alpha);
                Ok(())
            }
            (PlanParam::AlphaP, PlanParamValue::F32(alpha)) => {
                self.set_alpha_p(alpha);
                Ok(())
            }
            (PlanParam::InletVelocity, PlanParamValue::F32(velocity)) => {
                self.set_inlet_velocity(velocity);
                Ok(())
            }
            (PlanParam::RampTime, PlanParamValue::F32(time)) => {
                self.set_ramp_time(time);
                Ok(())
            }
            (PlanParam::IncompressibleOuterCorrectors, PlanParamValue::U32(iters)) => {
                self.n_outer_correctors = iters.max(1);
                Ok(())
            }
            (PlanParam::IncompressibleShouldStop, PlanParamValue::Bool(value)) => {
                self.should_stop = value;
                Ok(())
            }
            (PlanParam::DetailedProfilingEnabled, PlanParamValue::Bool(enable)) => {
                self.enable_detailed_profiling(enable);
                Ok(())
            }
            _ => Err("parameter is not supported by this plan".into()),
        }
    }

    fn write_state_bytes(&self, bytes: &[u8]) -> Result<(), String> {
        self.context.queue.write_buffer(&self.b_state, 0, bytes);
        Ok(())
    }

    fn step_stats(&self) -> PlanStepStats {
        PlanStepStats {
            should_stop: Some(self.should_stop),
            degenerate_count: Some(self.degenerate_count),
            outer_iterations: Some(*self.outer_iterations.lock().unwrap()),
            outer_residual_u: Some(*self.outer_residual_u.lock().unwrap()),
            outer_residual_p: Some(*self.outer_residual_p.lock().unwrap()),
            linear_stats: Some((
                *self.stats_ux.lock().unwrap(),
                *self.stats_uy.lock().unwrap(),
                *self.stats_p.lock().unwrap(),
            )),
        }
    }

    fn perform(&self, action: PlanAction) -> Result<(), String> {
        match action {
            PlanAction::StartProfilingSession => {
                self.start_profiling_session();
                Ok(())
            }
            PlanAction::EndProfilingSession => {
                self.end_profiling_session();
                Ok(())
            }
            PlanAction::PrintProfilingReport => {
                self.print_profiling_report();
                Ok(())
            }
        }
    }

    fn profiling_stats(&self) -> Option<Arc<ProfilingStats>> {
        Some(self.get_profiling_stats())
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
        Ok(GpuSolver::solve_linear_system_cg_with_size(self, n, max_iters, tol))
    }

    fn get_linear_solution(&self) -> PlanFuture<'_, Result<Vec<f32>, String>> {
        Box::pin(async move { Ok(GpuSolver::get_linear_solution(self).await) })
    }

    fn coupled_unknowns(&self) -> Result<u32, String> {
        Ok(GpuSolver::coupled_unknowns(self))
    }

    fn fgmres_sizing(&mut self, _max_restart: usize) -> Result<FgmresSizing, String> {
        let n = GpuSolver::coupled_unknowns(self);
        Ok(FgmresSizing {
            num_unknowns: n,
            num_dot_groups: crate::solver::gpu::linear_solver::fgmres::workgroups_for_size(n),
        })
    }

    fn step(&mut self) {
        crate::solver::gpu::plans::coupled::plan::step_coupled(self);
    }

    fn initialize_history(&self) {
        GpuSolver::initialize_history(self);
    }

    fn read_state_bytes(&self, bytes: u64) -> PlanFuture<'_, Vec<u8>> {
        Box::pin(async move { self.read_buffer(self.state_buffer(), bytes).await })
    }
}

impl GpuPlanInstance for CompressiblePlanResources {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

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

    fn set_advection_scheme(&mut self, scheme: Scheme) {
        CompressiblePlanResources::set_scheme(self, scheme.gpu_id());
    }

    fn set_time_scheme(&mut self, scheme: TimeScheme) {
        CompressiblePlanResources::set_time_scheme(self, scheme as u32);
    }

    fn set_preconditioner(&mut self, preconditioner: PreconditionerType) {
        CompressiblePlanResources::set_precond_type(self, preconditioner);
    }

    fn set_param(&mut self, param: PlanParam, value: PlanParamValue) -> Result<(), String> {
        match (param, value) {
            (PlanParam::Viscosity, PlanParamValue::F32(mu)) => {
                self.set_viscosity(mu);
                Ok(())
            }
            (PlanParam::AlphaU, PlanParamValue::F32(alpha)) => {
                self.set_alpha_u(alpha);
                Ok(())
            }
            (PlanParam::InletVelocity, PlanParamValue::F32(velocity)) => {
                self.set_inlet_velocity(velocity);
                Ok(())
            }
            (PlanParam::Dtau, PlanParamValue::F32(dtau)) => {
                self.set_dtau(dtau);
                Ok(())
            }
            (PlanParam::OuterIters, PlanParamValue::Usize(iters)) => {
                self.set_outer_iters(iters);
                Ok(())
            }
            (PlanParam::LowMachModel, PlanParamValue::LowMachModel(model)) => {
                self.set_precond_model(model as u32);
                Ok(())
            }
            (PlanParam::LowMachThetaFloor, PlanParamValue::F32(theta)) => {
                self.set_precond_theta_floor(theta);
                Ok(())
            }
            (PlanParam::NonconvergedRelax, PlanParamValue::F32(relax)) => {
                self.set_nonconverged_relax(relax);
                Ok(())
            }
            _ => Err("parameter is not supported by this plan".into()),
        }
    }

    fn write_state_bytes(&self, bytes: &[u8]) -> Result<(), String> {
        CompressiblePlanResources::write_state_bytes(self, bytes);
        Ok(())
    }

    fn step_with_stats(&mut self) -> Result<Vec<LinearSolverStats>, String> {
        Ok(self.step_with_stats())
    }

    fn step(&mut self) {
        crate::solver::gpu::plans::compressible::plan::step(self);
    }

    fn initialize_history(&self) {
        CompressiblePlanResources::initialize_history(self);
    }

    fn read_state_bytes(&self, bytes: u64) -> PlanFuture<'_, Vec<u8>> {
        Box::pin(async move { self.read_buffer(self.state_buffer(), bytes).await })
    }
}

impl GpuPlanInstance for GpuGenericCoupledSolver {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

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

    fn set_advection_scheme(&mut self, scheme: Scheme) {
        self.linear.set_scheme(scheme.gpu_id());
    }

    fn set_time_scheme(&mut self, scheme: TimeScheme) {
        self.linear.set_time_scheme(scheme as u32);
    }

    fn set_preconditioner(&mut self, preconditioner: PreconditionerType) {
        self.linear.set_precond_type(preconditioner);
    }

    fn set_param(&mut self, _param: PlanParam, _value: PlanParamValue) -> Result<(), String> {
        Err("parameter is not supported by this plan".into())
    }

    fn write_state_bytes(&self, bytes: &[u8]) -> Result<(), String> {
        GpuGenericCoupledSolver::write_state_bytes(self, bytes);
        Ok(())
    }

    fn step(&mut self) {
        let _ = GpuGenericCoupledSolver::step(self);
    }

    fn initialize_history(&self) {}

    fn read_state_bytes(&self, bytes: u64) -> PlanFuture<'_, Vec<u8>> {
        Box::pin(async move { self.linear.read_buffer(self.state_buffer(), bytes).await })
    }
}

pub(crate) async fn build_plan_instance(
    mesh: &Mesh,
    model: &ModelSpec,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
) -> Result<Box<dyn GpuPlanInstance>, String> {
    let plan: Box<dyn GpuPlanInstance> = match &model.fields {
        ModelFields::Incompressible(_) => Box::new(GpuSolver::new(mesh, device, queue).await),
        ModelFields::Compressible(_) => {
            Box::new(CompressiblePlanResources::new(mesh, device, queue).await)
        }
        ModelFields::GenericCoupled(_) => {
            let solver = GpuGenericCoupledSolver::new(mesh, model.clone(), device, queue).await?;
            Box::new(solver)
        }
    };
    Ok(plan)
}
