use crate::solver::gpu::enums::TimeScheme;
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

    fn step(&mut self) {
        let _ = GpuGenericCoupledSolver::step(self);
    }

    fn initialize_history(&self) {}

    fn read_state_bytes(&self, bytes: u64) -> PlanFuture<'_, Vec<u8>> {
        Box::pin(async move { self.linear.read_buffer(self.state_buffer(), bytes).await })
    }
}
