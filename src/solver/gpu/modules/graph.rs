use crate::solver::gpu::context::GpuContext;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub struct RuntimeDims {
    pub num_cells: u32,
    pub num_faces: u32,
}

#[derive(Clone, Debug)]
pub enum DispatchKind {
    Cells,
    Faces,
    Custom {
        x: u32,
        y: u32,
        z: u32,
    },
    /// Indirect dispatch: workgroup counts are read from a GPU buffer at the given offset.
    /// The buffer must contain 3 × u32 (x, y, z) at the specified byte offset.
    Indirect {
        buffer: Arc<wgpu::Buffer>,
        offset: u64,
    },
}

pub trait GpuComputeModule {
    type PipelineKey: Copy;
    type BindKey: Copy;

    fn pipeline(&self, key: Self::PipelineKey) -> &wgpu::ComputePipeline;
    fn bind(&self, key: Self::BindKey, pass: &mut wgpu::ComputePass);
    fn dispatch(&self, kind: DispatchKind, runtime: RuntimeDims) -> (u32, u32, u32);
}

#[derive(Clone, Debug)]
pub struct ComputeSpec<P: Copy, B: Copy> {
    pub label: &'static str,
    pub pipeline: P,
    pub bind: B,
    pub dispatch: DispatchKind,
}

#[derive(Clone, Debug)]
pub enum ModuleNode<P: Copy, B: Copy> {
    Compute(ComputeSpec<P, B>),
}

pub struct ModuleGraph<M: GpuComputeModule> {
    nodes: Vec<ModuleNode<M::PipelineKey, M::BindKey>>,
}

impl<M: GpuComputeModule> ModuleGraph<M> {
    pub fn new(nodes: Vec<ModuleNode<M::PipelineKey, M::BindKey>>) -> Self {
        Self { nodes }
    }

    /// Create a clone of this graph containing only the nodes whose label
    /// passes `keep`. Used for schedule variants that drop kernels which are
    /// redundant in context (e.g. the pressure-gradient recompute on outer
    /// iterations after the first, where `rhie_chow/grad_p_update` already
    /// left an identical gradient).
    pub fn clone_filtered(&self, keep: impl Fn(&str) -> bool) -> Self {
        let nodes = self
            .nodes
            .iter()
            .filter(|node| match node {
                ModuleNode::Compute(spec) => keep(spec.label),
            })
            .cloned()
            .collect();
        Self { nodes }
    }

    /// Create a clone of this graph where every dispatch is replaced with indirect dispatch.
    /// The `map_fn` receives the original `DispatchKind` and returns the indirect buffer + offset
    /// to use for that dispatch. This allows different indirect buffers for Cells vs Faces.
    pub fn clone_with_indirect_dispatch(
        &self,
        map_fn: impl Fn(&DispatchKind) -> (Arc<wgpu::Buffer>, u64),
    ) -> Self {
        let nodes = self
            .nodes
            .iter()
            .map(|node| match node {
                ModuleNode::Compute(spec) => {
                    let (buffer, offset) = map_fn(&spec.dispatch);
                    ModuleNode::Compute(ComputeSpec {
                        label: spec.label,
                        pipeline: spec.pipeline,
                        bind: spec.bind,
                        dispatch: DispatchKind::Indirect { buffer, offset },
                    })
                }
            })
            .collect();
        Self { nodes }
    }

    pub fn encode_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        module: &M,
        runtime: RuntimeDims,
    ) {
        for node in &self.nodes {
            node.encode(encoder, module, runtime);
        }
    }

    pub fn execute(&self, context: &GpuContext, module: &M, runtime: RuntimeDims) {
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ModuleGraph Encoder"),
            });
        self.encode_into(&mut encoder, module, runtime);
        context.queue.submit(Some(encoder.finish()));
        crate::count_submission!("Module Graph", "execute");
    }

    pub fn execute_split_timed(
        &self,
        context: &GpuContext,
        module: &M,
        runtime: RuntimeDims,
    ) -> ModuleGraphTimings {
        let mut timings = ModuleGraphTimings::default();
        for node in &self.nodes {
            let start = std::time::Instant::now();
            let mut encoder =
                context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some(node.label()),
                    });
            node.encode(&mut encoder, module, runtime);
            context.queue.submit(Some(encoder.finish()));
            crate::count_submission!("Module Graph", node.label());
            let secs = start.elapsed().as_secs_f64();
            timings.nodes.push(ModuleNodeTiming {
                label: node.label(),
                seconds: secs,
            });
            timings.total_seconds += secs;
        }
        timings
    }
}

impl<P: Copy, B: Copy> ModuleNode<P, B> {
    pub fn label(&self) -> &'static str {
        match self {
            ModuleNode::Compute(spec) => spec.label,
        }
    }

    pub fn encode<M: GpuComputeModule<PipelineKey = P, BindKey = B>>(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        module: &M,
        runtime: RuntimeDims,
    ) {
        match self {
            ModuleNode::Compute(spec) => {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(spec.label),
                    timestamp_writes: None,
                });
                pass.set_pipeline(module.pipeline(spec.pipeline));
                module.bind(spec.bind, &mut pass);
                match &spec.dispatch {
                    DispatchKind::Indirect { buffer, offset } => {
                        pass.dispatch_workgroups_indirect(buffer, *offset);
                    }
                    other => {
                        let (x, y, z) = module.dispatch(other.clone(), runtime);
                        pass.dispatch_workgroups(x, y, z);
                    }
                }

                // Count this dispatch for profiling
                crate::count_dispatch!("Kernel Graph", spec.label);
            }
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct ModuleGraphTimings {
    pub total_seconds: f64,
    pub nodes: Vec<ModuleNodeTiming>,
}

impl ModuleGraphTimings {
    pub fn seconds_for(&self, label: &'static str) -> f64 {
        self.nodes
            .iter()
            .find(|node| node.label == label)
            .map(|node| node.seconds)
            .unwrap_or(0.0)
    }
}

#[derive(Debug, Clone)]
pub struct ModuleNodeTiming {
    pub label: &'static str,
    pub seconds: f64,
}
