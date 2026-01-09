use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::plans::plan_instance::PlanParam;
use crate::solver::gpu::plans::program::{
    ModelGpuProgramSpec, ProgramF32Fn, ProgramInitRun, ProgramLinearDebugProvider,
    ProgramOpDispatcher, ProgramParamHandler, ProgramResources, ProgramSetParamFallback,
    ProgramSpec, ProgramStateBufferFn, ProgramStepStatsFn, ProgramStepWithStatsFn, ProgramU32Fn,
    ProgramWriteStateFn,
};
use crate::solver::gpu::profiling::ProfilingStats;
use crate::solver::model::ModelSpec;
use std::collections::HashMap;
use std::sync::Arc;

pub(crate) struct ModelGpuProgramSpecParts {
    pub ops: Arc<dyn ProgramOpDispatcher + Send + Sync>,
    pub num_cells: ProgramU32Fn,
    pub time: ProgramF32Fn,
    pub dt: ProgramF32Fn,
    pub state_buffer: ProgramStateBufferFn,
    pub write_state_bytes: ProgramWriteStateFn,
    pub initialize_history: Option<ProgramInitRun>,
    pub params: HashMap<PlanParam, ProgramParamHandler>,
    pub set_param_fallback: Option<ProgramSetParamFallback>,
    pub step_stats: Option<ProgramStepStatsFn>,
    pub step_with_stats: Option<ProgramStepWithStatsFn>,
    pub linear_debug: Option<ProgramLinearDebugProvider>,
}

impl ModelGpuProgramSpecParts {
    pub fn into_spec(self, program: ProgramSpec) -> ModelGpuProgramSpec {
        ModelGpuProgramSpec {
            ops: self.ops,
            num_cells: self.num_cells,
            time: self.time,
            dt: self.dt,
            state_buffer: self.state_buffer,
            write_state_bytes: self.write_state_bytes,
            program,
            initialize_history: self.initialize_history,
            params: self.params,
            set_param_fallback: self.set_param_fallback,
            step_stats: self.step_stats,
            step_with_stats: self.step_with_stats,
            linear_debug: self.linear_debug,
        }
    }
}

pub(crate) struct LoweredProgramParts {
    pub model: ModelSpec,
    pub context: GpuContext,
    pub profiling_stats: Arc<ProfilingStats>,
    pub resources: ProgramResources,
    pub spec: ModelGpuProgramSpecParts,
}
