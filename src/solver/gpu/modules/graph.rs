use crate::solver::gpu::context::GpuContext;

#[derive(Clone, Copy, Debug)]
pub struct RuntimeDims {
    pub num_cells: u32,
    pub num_faces: u32,
}

#[derive(Clone, Copy, Debug)]
pub enum DispatchKind {
    Cells,
    Faces,
    Custom { x: u32, y: u32, z: u32 },
}

pub trait GpuComputeModule {
    type PipelineKey: Copy;
    type BindKey: Copy;

    fn pipeline(&self, key: Self::PipelineKey) -> &wgpu::ComputePipeline;
    fn bind(&self, key: Self::BindKey, pass: &mut wgpu::ComputePass);
    fn dispatch(&self, kind: DispatchKind, runtime: RuntimeDims) -> (u32, u32, u32);
}

#[derive(Clone, Copy, Debug)]
pub struct ComputeSpec<P: Copy, B: Copy> {
    pub label: &'static str,
    pub pipeline: P,
    pub bind: B,
    pub dispatch: DispatchKind,
}

#[derive(Clone, Copy, Debug)]
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

    pub fn execute(&self, context: &GpuContext, module: &M, runtime: RuntimeDims) {
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ModuleGraph Encoder"),
            });
        for node in &self.nodes {
            node.encode(&mut encoder, module, runtime);
        }
        context.queue.submit(Some(encoder.finish()));
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
                let (x, y, z) = module.dispatch(spec.dispatch, runtime);
                pass.dispatch_workgroups(x, y, z);
                
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
