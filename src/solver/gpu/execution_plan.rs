use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::kernel_graph::KernelGraphTimings;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphExecMode {
    SingleSubmit,
    SplitTimed,
}

pub struct GraphNode<S> {
    pub label: &'static str,
    /// Executes a graph-like node and returns `(seconds, optional detail)`.
    ///
    /// `detail` is currently only used for `KernelGraph` split timings; other graph
    /// executors can return `None`.
    pub run: fn(&S, &GpuContext, GraphExecMode) -> (f64, Option<KernelGraphTimings>),
    pub mode: GraphExecMode,
}

pub struct HostNode<S> {
    pub label: &'static str,
    pub run: fn(&mut S),
}

pub enum PlanNode<S> {
    Graph(GraphNode<S>),
    Host(HostNode<S>),
}

pub struct ExecutionPlan<S> {
    context: fn(&S) -> &GpuContext,
    nodes: Vec<PlanNode<S>>,
}

impl<S> ExecutionPlan<S> {
    pub fn new(context: fn(&S) -> &GpuContext, nodes: Vec<PlanNode<S>>) -> Self {
        Self { context, nodes }
    }

    pub fn execute(&self, solver: &mut S) -> PlanTimings {
        let mut timings = PlanTimings::default();
        for node in &self.nodes {
            match node {
                PlanNode::Graph(graph_node) => {
                    let context = (self.context)(&*solver);
                    let (seconds, detail) = (graph_node.run)(&*solver, context, graph_node.mode);
                    timings.push(PlanNodeTiming::Graph(GraphTiming {
                        label: graph_node.label,
                        seconds,
                        detail,
                    }));
                }
                PlanNode::Host(host_node) => {
                    let start = std::time::Instant::now();
                    (host_node.run)(solver);
                    let secs = start.elapsed().as_secs_f64();
                    timings.push(PlanNodeTiming::Host(HostTiming {
                        label: host_node.label,
                        seconds: secs,
                    }));
                }
            }
        }
        timings
    }
}

#[derive(Default, Debug, Clone)]
pub struct PlanTimings {
    pub total_seconds: f64,
    pub nodes: Vec<PlanNodeTiming>,
}

impl PlanTimings {
    fn push(&mut self, timing: PlanNodeTiming) {
        self.total_seconds += timing.seconds();
        self.nodes.push(timing);
    }

    pub fn graph_detail(&self, label: &'static str) -> Option<&KernelGraphTimings> {
        self.nodes.iter().find_map(|node| match node {
            PlanNodeTiming::Graph(graph) if graph.label == label => graph.detail.as_ref(),
            _ => None,
        })
    }

    pub fn seconds_for(&self, label: &'static str) -> f64 {
        self.nodes
            .iter()
            .find(|node| node.label() == label)
            .map(|node| node.seconds())
            .unwrap_or(0.0)
    }
}

#[derive(Debug, Clone)]
pub enum PlanNodeTiming {
    Graph(GraphTiming),
    Host(HostTiming),
}

impl PlanNodeTiming {
    pub fn label(&self) -> &'static str {
        match self {
            PlanNodeTiming::Graph(node) => node.label,
            PlanNodeTiming::Host(node) => node.label,
        }
    }

    pub fn seconds(&self) -> f64 {
        match self {
            PlanNodeTiming::Graph(node) => node.seconds,
            PlanNodeTiming::Host(node) => node.seconds,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GraphTiming {
    pub label: &'static str,
    pub seconds: f64,
    pub detail: Option<KernelGraphTimings>,
}

#[derive(Debug, Clone)]
pub struct HostTiming {
    pub label: &'static str,
    pub seconds: f64,
}
