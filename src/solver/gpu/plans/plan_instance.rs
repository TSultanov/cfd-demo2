use crate::solver::gpu::enums::{GpuLowMachPrecondModel, TimeScheme};
use crate::solver::gpu::structs::{LinearSolverStats, PreconditionerType};
use crate::solver::scheme::Scheme;
use std::future::Future;
use std::pin::Pin;

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

#[derive(Debug, Clone, Copy)]
pub struct PlanInitConfig {
    pub advection_scheme: Scheme,
    pub time_scheme: TimeScheme,
    pub preconditioner: PreconditionerType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlanParam {
    Dt,
    AdvectionScheme,
    TimeScheme,
    Preconditioner,
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
    Scheme(Scheme),
    TimeScheme(TimeScheme),
    Preconditioner(PreconditionerType),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanAction {
    StartProfilingSession,
    EndProfilingSession,
    PrintProfilingReport,
}

pub(crate) trait PlanLinearSystemDebug: Send {
    fn set_linear_system(&self, matrix_values: &[f32], rhs: &[f32]) -> Result<(), String>;

    fn solve_linear_system_with_size(
        &mut self,
        n: u32,
        max_iters: u32,
        tol: f32,
    ) -> Result<LinearSolverStats, String>;

    fn get_linear_solution(&self) -> PlanFuture<'_, Result<Vec<f32>, String>>;
}
