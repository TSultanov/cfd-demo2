use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::profiling::ProfilingStats;
use crate::solver::gpu::program::plan::{
    ModelGpuProgramSpec, ProgramF32Fn, ProgramInitRun, ProgramLinearDebugProvider,
    ProgramOpDispatcher, ProgramOpRegistry, ProgramParamHandler, ProgramResources,
    ProgramSetBcValueFn, ProgramSetNamedParamFallback, ProgramSpec, ProgramStateBufferFn,
    ProgramStepStatsFn, ProgramStepWithStatsFn, ProgramU32Fn, ProgramWriteStateFn,
};
use crate::solver::model::ModelSpec;
use std::collections::HashMap;
use std::sync::Arc;

pub(crate) struct ModelGpuProgramSpecParts {
    pub ops: ProgramOpRegistry,
    pub num_cells: ProgramU32Fn,
    pub time: ProgramF32Fn,
    pub dt: ProgramF32Fn,
    pub state_buffer: ProgramStateBufferFn,
    pub write_state_bytes: ProgramWriteStateFn,
    pub set_bc_value: Option<ProgramSetBcValueFn>,
    pub initialize_history: Option<ProgramInitRun>,
    pub named_params: HashMap<&'static str, ProgramParamHandler>,
    pub set_named_param_fallback: Option<ProgramSetNamedParamFallback>,
    pub step_stats: Option<ProgramStepStatsFn>,
    pub step_with_stats: Option<ProgramStepWithStatsFn>,
    pub linear_debug: Option<ProgramLinearDebugProvider>,
}

impl ModelGpuProgramSpecParts {
    pub fn into_spec(self, program: ProgramSpec) -> Result<ModelGpuProgramSpec, String> {
        self.ops.validate_program_spec(&program)?;
        let ops: Arc<dyn ProgramOpDispatcher + Send + Sync> = Arc::new(self.ops);
        Ok(ModelGpuProgramSpec {
            ops,
            num_cells: self.num_cells,
            time: self.time,
            dt: self.dt,
            state_buffer: self.state_buffer,
            write_state_bytes: self.write_state_bytes,
            set_bc_value: self.set_bc_value,
            program,
            initialize_history: self.initialize_history,
            named_params: self.named_params,
            set_named_param_fallback: self.set_named_param_fallback,
            step_stats: self.step_stats,
            step_with_stats: self.step_with_stats,
            linear_debug: self.linear_debug,
        })
    }
}

pub(crate) struct LoweredProgramParts {
    pub model: ModelSpec,
    pub context: GpuContext,
    pub profiling_stats: Arc<ProfilingStats>,
    pub resources: ProgramResources,
    pub spec: ModelGpuProgramSpecParts,
}
