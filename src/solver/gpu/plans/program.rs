use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::execution_plan::{GraphDetail, GraphExecMode};
use crate::solver::gpu::plans::plan_instance::{
    PlanAction, PlanFuture, PlanLinearSystemDebug, PlanParam, PlanParamValue, PlanStepStats,
};
use crate::solver::gpu::profiling::ProfilingStats;
use crate::solver::gpu::readback::{read_buffer_cached, StagingBufferCache};
use crate::solver::gpu::structs::LinearSolverStats;
use crate::solver::model::ModelSpec;
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;

pub(crate) type ProgramGraphRun = fn(&GpuProgramPlan, &GpuContext, GraphExecMode) -> (f64, Option<GraphDetail>);
pub(crate) type ProgramHostRun = fn(&mut GpuProgramPlan);
pub(crate) type ProgramInitRun = fn(&GpuProgramPlan);
pub(crate) type ProgramParamHandler =
    fn(&mut GpuProgramPlan, PlanParamValue) -> Result<(), String>;
pub(crate) type ProgramSetParamFallback =
    fn(&mut GpuProgramPlan, PlanParam, PlanParamValue) -> Result<(), String>;
pub(crate) type ProgramU32Fn = fn(&GpuProgramPlan) -> u32;
pub(crate) type ProgramF32Fn = fn(&GpuProgramPlan) -> f32;
pub(crate) type ProgramStateBufferFn =
    for<'a> fn(&'a GpuProgramPlan) -> &'a wgpu::Buffer;
pub(crate) type ProgramWriteStateFn =
    fn(&GpuProgramPlan, bytes: &[u8]) -> Result<(), String>;
pub(crate) type ProgramStepStatsFn = fn(&GpuProgramPlan) -> PlanStepStats;
pub(crate) type ProgramStepWithStatsFn =
    fn(&mut GpuProgramPlan) -> Result<Vec<LinearSolverStats>, String>;
pub(crate) type ProgramLinearDebugProvider =
    fn(&mut GpuProgramPlan) -> Option<&mut dyn PlanLinearSystemDebug>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ProgramGraphId(pub u16);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ProgramHostId(pub u16);

pub(crate) struct ProgramResources {
    by_type: HashMap<TypeId, Box<dyn Any + Send>>,
}

impl ProgramResources {
    pub fn new() -> Self {
        Self {
            by_type: HashMap::new(),
        }
    }

    pub fn insert<T: Any + Send>(&mut self, value: T) {
        self.by_type.insert(TypeId::of::<T>(), Box::new(value));
    }

    pub fn has<T: Any + Send>(&self) -> bool {
        self.by_type.contains_key(&TypeId::of::<T>())
    }

    pub fn get<T: Any + Send>(&self) -> Option<&T> {
        self.by_type
            .get(&TypeId::of::<T>())
            .and_then(|v| v.downcast_ref::<T>())
    }

    pub fn get_mut<T: Any + Send>(&mut self) -> Option<&mut T> {
        self.by_type
            .get_mut(&TypeId::of::<T>())
            .and_then(|v| v.downcast_mut::<T>())
    }
}

pub(crate) enum ProgramNode {
    Graph {
        label: &'static str,
        id: ProgramGraphId,
        mode: GraphExecMode,
    },
    Host {
        label: &'static str,
        id: ProgramHostId,
    },
    If {
        label: &'static str,
        cond: fn(&GpuProgramPlan) -> bool,
        then_plan: Arc<ProgramExecutionPlan>,
        else_plan: Option<Arc<ProgramExecutionPlan>>,
    },
    Repeat {
        label: &'static str,
        times: fn(&GpuProgramPlan) -> usize,
        body: Arc<ProgramExecutionPlan>,
    },
    While {
        label: &'static str,
        max_iters: fn(&GpuProgramPlan) -> usize,
        cond: fn(&GpuProgramPlan) -> bool,
        body: Arc<ProgramExecutionPlan>,
    },
}

pub(crate) struct ProgramExecutionPlan {
    nodes: Vec<ProgramNode>,
}

impl ProgramExecutionPlan {
    pub fn new(nodes: Vec<ProgramNode>) -> Self {
        Self { nodes }
    }

    pub fn execute(&self, plan: &mut GpuProgramPlan) {
        for node in &self.nodes {
            match node {
                ProgramNode::Graph { id, mode, .. } => {
                    let run = plan
                        .spec
                        .graph_ops
                        .get(id)
                        .copied()
                        .unwrap_or_else(|| panic!("missing graph op for id={id:?}"));
                    let _ = run(&*plan, &plan.context, *mode);
                }
                ProgramNode::Host { id, .. } => {
                    let run = plan
                        .spec
                        .host_ops
                        .get(id)
                        .copied()
                        .unwrap_or_else(|| panic!("missing host op for id={id:?}"));
                    run(plan)
                }
                ProgramNode::If {
                    cond,
                    then_plan,
                    else_plan,
                    ..
                } => {
                    if cond(&*plan) {
                        then_plan.execute(plan);
                    } else if let Some(else_plan) = else_plan {
                        else_plan.execute(plan);
                    }
                }
                ProgramNode::Repeat { times, body, .. } => {
                    for _ in 0..times(&*plan) {
                        body.execute(plan);
                    }
                }
                ProgramNode::While {
                    max_iters,
                    cond,
                    body,
                    ..
                } => {
                    for _ in 0..max_iters(&*plan) {
                        if !cond(&*plan) {
                            break;
                        }
                        body.execute(plan);
                    }
                }
            }
        }
    }
}

pub(crate) struct ModelGpuProgramSpec {
    pub graph_ops: HashMap<ProgramGraphId, ProgramGraphRun>,
    pub host_ops: HashMap<ProgramHostId, ProgramHostRun>,
    pub num_cells: ProgramU32Fn,
    pub time: ProgramF32Fn,
    pub dt: ProgramF32Fn,
    pub state_buffer: ProgramStateBufferFn,
    pub write_state_bytes: ProgramWriteStateFn,
    pub step: Arc<ProgramExecutionPlan>,
    pub initialize_history: Option<ProgramInitRun>,
    pub params: HashMap<PlanParam, ProgramParamHandler>,
    pub set_param_fallback: Option<ProgramSetParamFallback>,
    pub step_stats: Option<ProgramStepStatsFn>,
    pub step_with_stats: Option<ProgramStepWithStatsFn>,
    pub linear_debug: Option<ProgramLinearDebugProvider>,
}

pub(crate) struct GpuProgramPlan {
    pub model: ModelSpec,
    pub context: GpuContext,
    pub profiling_stats: Arc<ProfilingStats>,
    pub staging_cache: StagingBufferCache,
    pub resources: ProgramResources,
    pub spec: ModelGpuProgramSpec,
    pub last_linear_stats: LinearSolverStats,
}

impl GpuProgramPlan {
    pub fn new(
        model: ModelSpec,
        context: GpuContext,
        profiling_stats: Arc<ProfilingStats>,
        resources: ProgramResources,
        spec: ModelGpuProgramSpec,
    ) -> Self {
        Self {
            model,
            context,
            profiling_stats,
            staging_cache: StagingBufferCache::default(),
            resources,
            spec,
            last_linear_stats: LinearSolverStats::default(),
        }
    }

    pub fn set_supported_param(
        &mut self,
        param: PlanParam,
        value: PlanParamValue,
    ) -> Result<(), String> {
        if let Some(handler) = self.spec.params.get(&param).copied() {
            return handler(self, value);
        }
        if let Some(fallback) = self.spec.set_param_fallback {
            return fallback(self, param, value);
        }
        Err("parameter is not supported by this plan".into())
    }

    pub fn num_cells(&self) -> u32 {
        (self.spec.num_cells)(self)
    }

    pub fn time(&self) -> f32 {
        (self.spec.time)(self)
    }

    pub fn dt(&self) -> f32 {
        (self.spec.dt)(self)
    }

    pub fn state_buffer(&self) -> &wgpu::Buffer {
        (self.spec.state_buffer)(self)
    }

    pub fn profiling_stats(&self) -> Arc<ProfilingStats> {
        Arc::clone(&self.profiling_stats)
    }

    pub fn set_param(&mut self, param: PlanParam, value: PlanParamValue) -> Result<(), String> {
        self.set_supported_param(param, value)
    }

    pub fn write_state_bytes(&self, bytes: &[u8]) -> Result<(), String> {
        (self.spec.write_state_bytes)(self, bytes)
    }

    pub fn step_stats(&self) -> PlanStepStats {
        if let Some(stats) = self.spec.step_stats {
            stats(self)
        } else {
            PlanStepStats::default()
        }
    }

    pub fn perform(&self, action: PlanAction) -> Result<(), String> {
        match action {
            PlanAction::StartProfilingSession => {
                self.profiling_stats.start_session();
                Ok(())
            }
            PlanAction::EndProfilingSession => {
                self.profiling_stats.end_session();
                Ok(())
            }
            PlanAction::PrintProfilingReport => {
                self.profiling_stats.print_report();
                Ok(())
            }
        }
    }

    pub fn step_with_stats(&mut self) -> Result<Vec<LinearSolverStats>, String> {
        if let Some(step) = self.spec.step_with_stats {
            return step(self);
        }
        self.last_linear_stats = LinearSolverStats::default();
        self.step();
        let stats = self.step_stats();
        if let Some((a, b, c)) = stats.linear_stats {
            return Ok(vec![a, b, c]);
        }
        Ok(if self.last_linear_stats.iterations > 0 {
            vec![self.last_linear_stats]
        } else {
            Vec::new()
        })
    }

    pub fn linear_system_debug(&mut self) -> Option<&mut dyn PlanLinearSystemDebug> {
        let provider = self.spec.linear_debug?;
        provider(self)
    }

    pub fn step(&mut self) {
        let step = Arc::clone(&self.spec.step);
        step.execute(self);
    }

    pub fn initialize_history(&self) {
        if let Some(init) = self.spec.initialize_history {
            init(self);
        }
    }

    pub fn read_state_bytes(&self, bytes: u64) -> PlanFuture<'_, Vec<u8>> {
        Box::pin(async move {
            read_buffer_cached(
                &self.context,
                &self.staging_cache,
                &self.profiling_stats,
                self.state_buffer(),
                bytes,
                "GpuProgramPlan Staging Buffer (cached)",
            )
            .await
        })
    }
}
