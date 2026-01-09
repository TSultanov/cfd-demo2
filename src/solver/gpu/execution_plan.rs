use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::modules::graph::{GpuComputeModule, ModuleGraph, RuntimeDims};
use crate::solver::gpu::modules::graph::ModuleGraphTimings;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphExecMode {
    SingleSubmit,
    SplitTimed,
}

pub fn run_module_graph<M: GpuComputeModule>(
    graph: &ModuleGraph<M>,
    context: &GpuContext,
    module: &M,
    runtime: RuntimeDims,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    match mode {
        GraphExecMode::SingleSubmit => {
            let start = std::time::Instant::now();
            graph.execute(context, module, runtime);
            (start.elapsed().as_secs_f64(), None)
        }
        GraphExecMode::SplitTimed => {
            let detail = graph.execute_split_timed(context, module, runtime);
            (detail.total_seconds, Some(GraphDetail::Module(detail)))
        }
    }
}

#[derive(Debug, Clone)]
pub enum GraphDetail {
    Module(ModuleGraphTimings),
}
