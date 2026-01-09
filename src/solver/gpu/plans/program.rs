use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::execution_plan::{GraphDetail, GraphExecMode};
use crate::solver::gpu::plans::plan_instance::{
    GpuPlanInstance, PlanFuture, PlanLinearSystemDebug, PlanParam, PlanParamValue,
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
pub(crate) type ProgramU32Fn = fn(&GpuProgramPlan) -> u32;
pub(crate) type ProgramF32Fn = fn(&GpuProgramPlan) -> f32;
pub(crate) type ProgramStateBufferFn =
    for<'a> fn(&'a GpuProgramPlan) -> &'a wgpu::Buffer;
pub(crate) type ProgramWriteStateFn =
    fn(&GpuProgramPlan, bytes: &[u8]) -> Result<(), String>;
pub(crate) type ProgramSetLinearSystemFn =
    fn(&GpuProgramPlan, matrix_values: &[f32], rhs: &[f32]) -> Result<(), String>;
pub(crate) type ProgramSolveLinearSystemFn =
    fn(&mut GpuProgramPlan, n: u32, max_iters: u32, tol: f32) -> Result<LinearSolverStats, String>;
pub(crate) type ProgramGetLinearSolutionFn =
    for<'a> fn(&'a GpuProgramPlan) -> PlanFuture<'a, Result<Vec<f32>, String>>;

pub(crate) struct ProgramLinearDebug {
    pub set_linear_system: ProgramSetLinearSystemFn,
    pub solve_linear_system_with_size: ProgramSolveLinearSystemFn,
    pub get_linear_solution: ProgramGetLinearSolutionFn,
}

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
        run: ProgramGraphRun,
        mode: GraphExecMode,
    },
    Host {
        label: &'static str,
        run: ProgramHostRun,
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
                ProgramNode::Graph { run, mode, .. } => {
                    let _ = run(&*plan, &plan.context, *mode);
                }
                ProgramNode::Host { run, .. } => run(plan),
            }
        }
    }
}

pub(crate) struct ModelGpuProgramSpec {
    pub num_cells: ProgramU32Fn,
    pub time: ProgramF32Fn,
    pub dt: ProgramF32Fn,
    pub state_buffer: ProgramStateBufferFn,
    pub write_state_bytes: ProgramWriteStateFn,
    pub step: Arc<ProgramExecutionPlan>,
    pub initialize_history: Option<ProgramInitRun>,
    pub params: HashMap<PlanParam, ProgramParamHandler>,
    pub linear_debug: Option<ProgramLinearDebug>,
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
        let handler = *self
            .spec
            .params
            .get(&param)
            .ok_or_else(|| "parameter is not supported by this plan".to_string())?;
        handler(self, value)
    }

}

impl GpuPlanInstance for GpuProgramPlan {
    fn num_cells(&self) -> u32 {
        (self.spec.num_cells)(self)
    }

    fn time(&self) -> f32 {
        (self.spec.time)(self)
    }

    fn dt(&self) -> f32 {
        (self.spec.dt)(self)
    }

    fn state_buffer(&self) -> &wgpu::Buffer {
        (self.spec.state_buffer)(self)
    }

    fn profiling_stats(&self) -> Arc<ProfilingStats> {
        Arc::clone(&self.profiling_stats)
    }

    fn set_param(&mut self, param: PlanParam, value: PlanParamValue) -> Result<(), String> {
        self.set_supported_param(param, value)
    }

    fn write_state_bytes(&self, bytes: &[u8]) -> Result<(), String> {
        (self.spec.write_state_bytes)(self, bytes)
    }

    fn step_with_stats(&mut self) -> Result<Vec<LinearSolverStats>, String> {
        self.last_linear_stats = LinearSolverStats::default();
        self.step();
        Ok(if self.last_linear_stats.iterations > 0 {
            vec![self.last_linear_stats]
        } else {
            Vec::new()
        })
    }

    fn linear_system_debug(&mut self) -> Option<&mut dyn PlanLinearSystemDebug> {
        self.spec.linear_debug.as_ref()?;
        Some(self)
    }

    fn step(&mut self) {
        let step = Arc::clone(&self.spec.step);
        step.execute(self);
    }

    fn initialize_history(&self) {
        if let Some(init) = self.spec.initialize_history {
            init(self);
        }
    }

    fn read_state_bytes(&self, bytes: u64) -> PlanFuture<'_, Vec<u8>> {
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

impl PlanLinearSystemDebug for GpuProgramPlan {
    fn set_linear_system(&self, matrix_values: &[f32], rhs: &[f32]) -> Result<(), String> {
        let debug = self
            .spec
            .linear_debug
            .as_ref()
            .ok_or_else(|| "linear system debug not supported by this plan".to_string())?;
        (debug.set_linear_system)(self, matrix_values, rhs)
    }

    fn solve_linear_system_with_size(
        &mut self,
        n: u32,
        max_iters: u32,
        tol: f32,
    ) -> Result<LinearSolverStats, String> {
        let debug = self
            .spec
            .linear_debug
            .as_ref()
            .ok_or_else(|| "linear system debug not supported by this plan".to_string())?;
        (debug.solve_linear_system_with_size)(self, n, max_iters, tol)
    }

    fn get_linear_solution(&self) -> PlanFuture<'_, Result<Vec<f32>, String>> {
        let debug = match self.spec.linear_debug.as_ref() {
            Some(debug) => debug,
            None => {
                return Box::pin(async {
                    Err("linear system debug not supported by this plan".into())
                })
            }
        };
        (debug.get_linear_solution)(self)
    }
}
