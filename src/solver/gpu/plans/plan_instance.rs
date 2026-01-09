use crate::solver::gpu::enums::{GpuLowMachPrecondModel, TimeScheme};
use crate::solver::gpu::profiling::ProfilingStats;
use crate::solver::gpu::structs::{LinearSolverStats, PreconditionerType};
use crate::solver::scheme::Scheme;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlanCapability {
    /// Plan supports setting/solving/reading a linear system via the debug hooks.
    LinearSystemDebug,
    /// Plan exposes the notion of "coupled unknowns" for its linear system.
    CoupledUnknowns,
    /// Plan can report FGMRES sizing information.
    FgmresSizing,
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

pub(crate) trait PlanCoupledUnknowns: Send {
    fn coupled_unknowns(&self) -> Result<u32, String>;
}

pub(crate) trait PlanFgmresSizing: Send {
    fn fgmres_sizing(&mut self, max_restart: usize) -> Result<FgmresSizing, String>;
}

pub(crate) trait GpuPlanInstance: Send {
    fn num_cells(&self) -> u32;
    fn time(&self) -> f32;
    fn dt(&self) -> f32;
    fn state_buffer(&self) -> &wgpu::Buffer;

    fn supports(&self, _capability: PlanCapability) -> bool {
        false
    }

    fn set_param(&mut self, param: PlanParam, value: PlanParamValue) -> Result<(), String>;

    fn write_state_bytes(&self, bytes: &[u8]) -> Result<(), String>;

    fn step_stats(&self) -> PlanStepStats {
        PlanStepStats::default()
    }

    fn perform(&self, action: PlanAction) -> Result<(), String> {
        let stats = self.profiling_stats();
        match action {
            PlanAction::StartProfilingSession => {
                stats.start_session();
                Ok(())
            }
            PlanAction::EndProfilingSession => {
                stats.end_session();
                Ok(())
            }
            PlanAction::PrintProfilingReport => {
                stats.print_report();
                Ok(())
            }
        }
    }

    fn profiling_stats(&self) -> Arc<ProfilingStats>;

    fn step_with_stats(&mut self) -> Result<Vec<LinearSolverStats>, String> {
        self.step();
        let stats = self.step_stats();
        if let Some((a, b, c)) = stats.linear_stats {
            Ok(vec![a, b, c])
        } else {
            Ok(Vec::new())
        }
    }

    fn linear_system_debug(&mut self) -> Option<&mut dyn PlanLinearSystemDebug> {
        None
    }

    fn coupled_unknowns_debug(&mut self) -> Option<&mut dyn PlanCoupledUnknowns> {
        None
    }

    fn fgmres_sizing_debug(&mut self) -> Option<&mut dyn PlanFgmresSizing> {
        None
    }

    fn step(&mut self);

    fn initialize_history(&self);

    fn read_state_bytes(&self, bytes: u64) -> PlanFuture<'_, Vec<u8>>;
}
