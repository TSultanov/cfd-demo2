use crate::solver::gpu::enums::{GpuLowMachPrecondModel, TimeScheme};
use crate::solver::gpu::plans::compressible::CompressiblePlanResources;
use crate::solver::gpu::plans::generic_coupled::GpuGenericCoupledSolver;
use crate::solver::gpu::structs::{GpuSolver, PreconditionerType};
use crate::solver::scheme::Scheme;
use std::any::Any;
use std::future::Future;
use std::pin::Pin;

pub(crate) type PlanFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FgmresSizing {
    pub num_unknowns: u32,
    pub num_dot_groups: u32,
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

    fn step(&mut self) {
        let _ = GpuGenericCoupledSolver::step(self);
    }

    fn initialize_history(&self) {}

    fn read_state_bytes(&self, bytes: u64) -> PlanFuture<'_, Vec<u8>> {
        Box::pin(async move { self.linear.read_buffer(self.state_buffer(), bytes).await })
    }
}
