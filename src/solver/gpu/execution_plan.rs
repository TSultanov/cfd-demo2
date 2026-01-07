use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::kernel_graph::{KernelGraph, KernelGraphTimings};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphExecMode {
    SingleSubmit,
    SplitTimed,
}

pub struct GraphNode<S> {
    pub label: &'static str,
    pub graph: fn(&S) -> &KernelGraph<S>,
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
                    match graph_node.mode {
                        GraphExecMode::SingleSubmit => {
                            let start = std::time::Instant::now();
                            let graph = (graph_node.graph)(&*solver);
                            graph.execute(context, &*solver);
                            let secs = start.elapsed().as_secs_f64();
                            timings.push(PlanNodeTiming::Graph(GraphTiming {
                                label: graph_node.label,
                                seconds: secs,
                                detail: None,
                            }));
                        }
                        GraphExecMode::SplitTimed => {
                            let graph = (graph_node.graph)(&*solver);
                            let detail = graph.execute_split_timed(context, &*solver);
                            timings.push(PlanNodeTiming::Graph(GraphTiming {
                                label: graph_node.label,
                                seconds: detail.total_seconds,
                                detail: Some(detail),
                            }));
                        }
                    }
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

