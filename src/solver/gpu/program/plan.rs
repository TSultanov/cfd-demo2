use super::plan_instance::{
    PlanAction, PlanFuture, PlanLinearSystemDebug, PlanParamValue, PlanStepStats,
};
use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::execution_plan::{GraphDetail, GraphExecMode};
use crate::solver::gpu::profiling::ProfilingStats;
use crate::solver::gpu::readback::{read_buffer_cached, StagingBufferCache};
use crate::solver::gpu::structs::LinearSolverStats;
use crate::solver::model::ModelSpec;
use std::any::{Any, TypeId};
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

pub(crate) type ProgramInitRun = fn(&GpuProgramPlan);
pub(crate) type ProgramParamHandler = fn(&mut GpuProgramPlan, PlanParamValue) -> Result<(), String>;
pub(crate) type ProgramSetNamedParamFallback =
    fn(&mut GpuProgramPlan, &str, PlanParamValue) -> Result<(), String>;
pub(crate) type ProgramU32Fn = fn(&GpuProgramPlan) -> u32;
pub(crate) type ProgramF32Fn = fn(&GpuProgramPlan) -> f32;
pub(crate) type ProgramStateBufferFn = for<'a> fn(&'a GpuProgramPlan) -> &'a wgpu::Buffer;
pub(crate) type ProgramWriteStateFn = fn(&GpuProgramPlan, bytes: &[u8]) -> Result<(), String>;
pub(crate) type ProgramSetBcValueFn =
    fn(&GpuProgramPlan, crate::solver::gpu::enums::GpuBoundaryType, u32, f32) -> Result<(), String>;
pub(crate) type ProgramStepStatsFn = fn(&GpuProgramPlan) -> PlanStepStats;
pub(crate) type ProgramStepWithStatsFn =
    fn(&mut GpuProgramPlan) -> Result<Vec<LinearSolverStats>, String>;
pub(crate) type ProgramLinearDebugProvider =
    fn(&mut GpuProgramPlan) -> Option<&mut dyn PlanLinearSystemDebug>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GraphOpKind(pub &'static str);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct HostOpKind(pub &'static str);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct CountOpKind(pub &'static str);

pub(crate) trait ProgramOpDispatcher {
    fn run_graph(
        &self,
        kind: GraphOpKind,
        plan: &GpuProgramPlan,
        context: &GpuContext,
        mode: GraphExecMode,
    ) -> (f64, Option<GraphDetail>);

    fn run_host(&self, kind: HostOpKind, plan: &mut GpuProgramPlan);

    fn eval_count(&self, kind: CountOpKind, plan: &GpuProgramPlan) -> usize;
}

pub(crate) type GraphOpHandler =
    fn(&GpuProgramPlan, &GpuContext, GraphExecMode) -> (f64, Option<GraphDetail>);
pub(crate) type HostOpHandler = fn(&mut GpuProgramPlan);
pub(crate) type CountOpHandler = fn(&GpuProgramPlan) -> usize;

#[derive(Default)]
pub(crate) struct ProgramOpRegistry {
    graph: HashMap<GraphOpKind, GraphOpHandler>,
    host: HashMap<HostOpKind, HostOpHandler>,
    count: HashMap<CountOpKind, CountOpHandler>,
}

impl ProgramOpRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn validate_program_spec(&self, program: &ProgramSpec) -> Result<(), String> {
        let mut missing = BTreeSet::<String>::new();

        if program.blocks.get(program.root.0 as usize).is_none() {
            missing.insert(format!("missing root program block: {:?}", program.root));
        }

        for (block_idx, block) in program.blocks.iter().enumerate() {
            for node in &block.nodes {
                match *node {
                    ProgramSpecNode::Graph { label, kind, .. } => {
                        if !self.graph.contains_key(&kind) {
                            missing.insert(format!(
                                "missing graph op handler: {kind:?} (node '{label}', block {block_idx})"
                            ));
                        }
                    }
                    ProgramSpecNode::Host { label, kind } => {
                        if !self.host.contains_key(&kind) {
                            missing.insert(format!(
                                "missing host op handler: {kind:?} (node '{label}', block {block_idx})"
                            ));
                        }
                    }
                    ProgramSpecNode::Repeat { label, times, body } => {
                        if !self.count.contains_key(&times) {
                            missing.insert(format!(
                                "missing count op handler: {times:?} (node '{label}', block {block_idx})"
                            ));
                        }
                        if program.blocks.get(body.0 as usize).is_none() {
                            missing.insert(format!(
                                "missing ProgramSpec block referenced by node '{label}': body={body:?}"
                            ));
                        }
                    }
                }
            }
        }

        if missing.is_empty() {
            return Ok(());
        }
        Err(format!(
            "ProgramSpec validation failed; missing op handlers:\n- {}",
            missing.into_iter().collect::<Vec<_>>().join("\n- ")
        ))
    }

    pub fn register_graph(
        &mut self,
        kind: GraphOpKind,
        handler: GraphOpHandler,
    ) -> Result<(), String> {
        if self.graph.insert(kind, handler).is_some() {
            return Err(format!("graph op already registered: {kind:?}"));
        }
        Ok(())
    }

    pub fn register_host(
        &mut self,
        kind: HostOpKind,
        handler: HostOpHandler,
    ) -> Result<(), String> {
        if self.host.insert(kind, handler).is_some() {
            return Err(format!("host op already registered: {kind:?}"));
        }
        Ok(())
    }

    pub fn register_count(
        &mut self,
        kind: CountOpKind,
        handler: CountOpHandler,
    ) -> Result<(), String> {
        if self.count.insert(kind, handler).is_some() {
            return Err(format!("count op already registered: {kind:?}"));
        }
        Ok(())
    }

    /// Check if a host op is registered.
    #[cfg(test)]
    pub fn has_host(&self, kind: &HostOpKind) -> bool {
        self.host.contains_key(kind)
    }

    /// Check if a graph op is registered.
    #[cfg(test)]
    pub fn has_graph(&self, kind: &GraphOpKind) -> bool {
        self.graph.contains_key(kind)
    }

    /// Check if a count op is registered.
    #[cfg(test)]
    pub fn has_count(&self, kind: &CountOpKind) -> bool {
        self.count.contains_key(kind)
    }

    /// Merge another registry into this one.
    /// Returns an error if there are duplicate registrations.
    pub fn merge(&mut self, other: Self) -> Result<(), String> {
        for (kind, handler) in other.graph {
            self.register_graph(kind, handler)?;
        }
        for (kind, handler) in other.host {
            self.register_host(kind, handler)?;
        }
        for (kind, handler) in other.count {
            self.register_count(kind, handler)?;
        }
        Ok(())
    }
}

impl ProgramOpDispatcher for ProgramOpRegistry {
    fn run_graph(
        &self,
        kind: GraphOpKind,
        plan: &GpuProgramPlan,
        context: &GpuContext,
        mode: GraphExecMode,
    ) -> (f64, Option<GraphDetail>) {
        let handler = self
            .graph
            .get(&kind)
            .copied()
            .unwrap_or_else(|| panic!("missing graph op handler registration: {kind:?}"));
        handler(plan, context, mode)
    }

    fn run_host(&self, kind: HostOpKind, plan: &mut GpuProgramPlan) {
        let handler = self
            .host
            .get(&kind)
            .copied()
            .unwrap_or_else(|| panic!("missing host op handler registration: {kind:?}"));
        handler(plan);
    }

    fn eval_count(&self, kind: CountOpKind, plan: &GpuProgramPlan) -> usize {
        let handler = self
            .count
            .get(&kind)
            .copied()
            .unwrap_or_else(|| panic!("missing count op handler registration: {kind:?}"));
        handler(plan)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ProgramBlockId(pub u16);

#[derive(Debug, Clone, Copy)]
pub(crate) enum ProgramSpecNode {
    Graph {
        label: &'static str,
        kind: GraphOpKind,
        mode: GraphExecMode,
    },
    Host {
        label: &'static str,
        kind: HostOpKind,
    },
    Repeat {
        label: &'static str,
        times: CountOpKind,
        body: ProgramBlockId,
    },
}

#[derive(Debug, Clone)]
pub struct StepGraphTiming {
    pub label: &'static str,
    pub seconds: f64,
    pub detail: Option<GraphDetail>,
}

#[derive(Debug, Default, Clone)]
pub(crate) struct ProgramBlock {
    pub nodes: Vec<ProgramSpecNode>,
}

#[derive(Debug, Clone)]
pub(crate) struct ProgramSpec {
    pub root: ProgramBlockId,
    pub blocks: Vec<ProgramBlock>,
}

impl ProgramSpec {
    pub fn root() -> Self {
        Self {
            root: ProgramBlockId(0),
            blocks: vec![ProgramBlock::default()],
        }
    }

    pub fn block(&self, id: ProgramBlockId) -> &ProgramBlock {
        self.blocks
            .get(id.0 as usize)
            .unwrap_or_else(|| panic!("missing program block for id={id:?}"))
    }
}

pub(crate) struct ProgramSpecBuilder {
    spec: ProgramSpec,
}

impl ProgramSpecBuilder {
    pub fn new() -> Self {
        Self {
            spec: ProgramSpec::root(),
        }
    }

    pub fn root(&self) -> ProgramBlockId {
        self.spec.root
    }

    pub fn new_block(&mut self) -> ProgramBlockId {
        let id = ProgramBlockId(self.spec.blocks.len() as u16);
        self.spec.blocks.push(ProgramBlock::default());
        id
    }

    pub fn push(&mut self, block: ProgramBlockId, node: ProgramSpecNode) {
        self.spec
            .blocks
            .get_mut(block.0 as usize)
            .unwrap_or_else(|| panic!("missing program block for id={block:?}"))
            .nodes
            .push(node);
    }

    pub fn build(self) -> ProgramSpec {
        self.spec
    }
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

pub(crate) struct ModelGpuProgramSpec {
    pub ops: Arc<dyn ProgramOpDispatcher + Send + Sync>,
    pub num_cells: ProgramU32Fn,
    pub time: ProgramF32Fn,
    pub dt: ProgramF32Fn,
    pub state_buffer: ProgramStateBufferFn,
    pub write_state_bytes: ProgramWriteStateFn,
    pub set_bc_value: Option<ProgramSetBcValueFn>,
    pub program: ProgramSpec,
    pub initialize_history: Option<ProgramInitRun>,
    pub named_params: HashMap<&'static str, ProgramParamHandler>,
    pub set_named_param_fallback: Option<ProgramSetNamedParamFallback>,
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
    pub collect_convergence_stats: bool,
    pub collect_trace: bool,
    pub step_linear_stats: Vec<LinearSolverStats>,
    pub step_graph_timings: Vec<StepGraphTiming>,
    pub outer_iterations: u32,
    pub outer_residual_u: Option<f32>,
    pub outer_residual_p: Option<f32>,
    pub outer_field_residuals: Vec<(String, f32)>,
    pub outer_field_residuals_scaled: Vec<(String, f32)>,
    pub repeat_break: bool,
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
            collect_convergence_stats: false,
            collect_trace: false,
            step_linear_stats: Vec::new(),
            step_graph_timings: Vec::new(),
            outer_iterations: 0,
            outer_residual_u: None,
            outer_residual_p: None,
            outer_field_residuals: Vec::new(),
            outer_field_residuals_scaled: Vec::new(),
            repeat_break: false,
        }
    }

    pub fn set_supported_named_param(
        &mut self,
        name: &str,
        value: PlanParamValue,
    ) -> Result<(), String> {
        if let Some(handler) = self.spec.named_params.get(name).copied() {
            return handler(self, value);
        }
        if let Some(fallback) = self.spec.set_named_param_fallback {
            return fallback(self, name, value);
        }
        let mut known: Vec<&str> = self.spec.named_params.keys().copied().collect();
        known.sort_unstable();
        Err(if known.is_empty() {
            format!("unknown named parameter '{name}'")
        } else {
            format!(
                "unknown named parameter '{name}'; known: {}",
                known.join(", ")
            )
        })
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

    pub fn set_named_param(&mut self, name: &str, value: PlanParamValue) -> Result<(), String> {
        self.set_supported_named_param(name, value)
    }

    pub fn write_state_bytes(&self, bytes: &[u8]) -> Result<(), String> {
        (self.spec.write_state_bytes)(self, bytes)
    }

    pub fn set_bc_value(
        &self,
        boundary: crate::solver::gpu::enums::GpuBoundaryType,
        unknown_component: u32,
        value: f32,
    ) -> Result<(), String> {
        let Some(set_bc_value) = self.spec.set_bc_value else {
            return Err("plan does not support bc_value updates".into());
        };
        set_bc_value(self, boundary, unknown_component, value)
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
        self.step_linear_stats.clear();
        self.step();
        let stats = self.step_stats();
        if let Some((a, b, c)) = stats.linear_stats {
            return Ok(vec![a, b, c]);
        }
        if !self.step_linear_stats.is_empty() {
            return Ok(std::mem::take(&mut self.step_linear_stats));
        }
        let last = self.last_linear_stats;
        Ok(if last.iterations > 0 || last.converged || last.diverged {
            vec![last]
        } else {
            Vec::new()
        })
    }

    pub fn linear_system_debug(&mut self) -> Option<&mut dyn PlanLinearSystemDebug> {
        let provider = self.spec.linear_debug?;
        provider(self)
    }

    pub fn step(&mut self) {
        self.execute_block(self.spec.program.root);
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

    fn execute_block(&mut self, block: ProgramBlockId) {
        let nodes = self.spec.program.block(block).nodes.clone();
        for node in nodes {
            match node {
                ProgramSpecNode::Graph { label, kind, mode } => {
                    let (seconds, detail) =
                        self.spec.ops.run_graph(kind, &*self, &self.context, mode);
                    if self.collect_trace {
                        self.step_graph_timings.push(StepGraphTiming {
                            label,
                            seconds,
                            detail,
                        });
                    }
                }
                ProgramSpecNode::Host { kind, .. } => {
                    let ops = Arc::clone(&self.spec.ops);
                    ops.run_host(kind, self);
                }
                ProgramSpecNode::Repeat { times, body, .. } => {
                    let times = self.spec.ops.eval_count(times, &*self);
                    for _ in 0..times {
                        self.execute_block(body);
                        if self.repeat_break {
                            self.repeat_break = false;
                            break;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_graph(
        _plan: &GpuProgramPlan,
        _context: &GpuContext,
        _mode: GraphExecMode,
    ) -> (f64, Option<GraphDetail>) {
        (0.0, None)
    }

    fn dummy_host(_plan: &mut GpuProgramPlan) {}

    fn dummy_count(_plan: &GpuProgramPlan) -> usize {
        0
    }

    #[test]
    fn validate_program_spec_reports_missing_handlers() {
        let mut builder = ProgramSpecBuilder::new();
        let root = builder.root();
        let repeat_block = builder.new_block();

        builder.push(
            root,
            ProgramSpecNode::Graph {
                label: "t:g",
                kind: GraphOpKind("CompressibleExplicitGraph"),
                mode: GraphExecMode::SingleSubmit,
            },
        );
        builder.push(
            root,
            ProgramSpecNode::Host {
                label: "t:h",
                kind: HostOpKind("CompressibleExplicitPrepare"),
            },
        );
        builder.push(
            root,
            ProgramSpecNode::Repeat {
                label: "t:repeat",
                times: CountOpKind("CompressibleImplicitOuterIters"),
                body: repeat_block,
            },
        );

        builder.push(
            repeat_block,
            ProgramSpecNode::Host {
                label: "t:repeat:h",
                kind: HostOpKind("IncompressibleCoupledBeginStep"),
            },
        );

        let program = builder.build();

        let registry = ProgramOpRegistry::new();
        let err = registry
            .validate_program_spec(&program)
            .expect_err("expected missing handler validation to fail");
        assert!(err.contains("CompressibleExplicitGraph"));
        assert!(err.contains("CompressibleExplicitPrepare"));
        assert!(err.contains("CompressibleImplicitOuterIters"));
        assert!(err.contains("IncompressibleCoupledBeginStep"));
    }

    #[test]
    fn validate_program_spec_passes_with_complete_registry() -> Result<(), String> {
        let mut builder = ProgramSpecBuilder::new();
        let root = builder.root();
        let repeat_block = builder.new_block();

        builder.push(
            root,
            ProgramSpecNode::Graph {
                label: "t:g",
                kind: GraphOpKind("CompressibleExplicitGraph"),
                mode: GraphExecMode::SingleSubmit,
            },
        );
        builder.push(
            root,
            ProgramSpecNode::Host {
                label: "t:h",
                kind: HostOpKind("CompressibleExplicitPrepare"),
            },
        );
        builder.push(
            root,
            ProgramSpecNode::Repeat {
                label: "t:repeat",
                times: CountOpKind("CompressibleImplicitOuterIters"),
                body: repeat_block,
            },
        );

        builder.push(
            repeat_block,
            ProgramSpecNode::Host {
                label: "t:repeat:h",
                kind: HostOpKind("IncompressibleCoupledBeginStep"),
            },
        );

        let program = builder.build();

        let mut registry = ProgramOpRegistry::new();
        registry.register_graph(GraphOpKind("CompressibleExplicitGraph"), dummy_graph)?;
        registry.register_host(HostOpKind("CompressibleExplicitPrepare"), dummy_host)?;
        registry.register_host(HostOpKind("IncompressibleCoupledBeginStep"), dummy_host)?;
        registry.register_count(CountOpKind("CompressibleImplicitOuterIters"), dummy_count)?;

        registry.validate_program_spec(&program)
    }
}
