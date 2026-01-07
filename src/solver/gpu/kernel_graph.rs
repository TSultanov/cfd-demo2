use crate::solver::gpu::context::GpuContext;

pub struct KernelGraph<S> {
    nodes: Vec<KernelNode<S>>,
}

impl<S> KernelGraph<S> {
    pub fn new(nodes: Vec<KernelNode<S>>) -> Self {
        Self { nodes }
    }

    pub fn execute(&self, context: &GpuContext, solver: &S) {
        let mut encoder =
            context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("KernelGraph Encoder"),
                });

        for node in &self.nodes {
            node.encode(&mut encoder, solver);
        }

        context.queue.submit(Some(encoder.finish()));
    }

    pub fn execute_split_timed(&self, context: &GpuContext, solver: &S) -> KernelGraphTimings {
        let mut timings = KernelGraphTimings::default();
        for node in &self.nodes {
            let start = std::time::Instant::now();
            let mut encoder =
                context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some(node.label()),
                    });
            node.encode(&mut encoder, solver);
            context.queue.submit(Some(encoder.finish()));
            let secs = start.elapsed().as_secs_f64();
            timings.nodes.push(KernelNodeTiming {
                label: node.label(),
                seconds: secs,
            });
            timings.total_seconds += secs;
        }
        timings
    }
}

pub enum KernelNode<S> {
    CopyBuffer(CopyBufferNode<S>),
    Compute(ComputeNode<S>),
}

impl<S> KernelNode<S> {
    pub fn label(&self) -> &'static str {
        match self {
            KernelNode::CopyBuffer(node) => node.label,
            KernelNode::Compute(node) => node.label,
        }
    }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder, solver: &S) {
        match self {
            KernelNode::CopyBuffer(node) => node.encode(encoder, solver),
            KernelNode::Compute(node) => node.encode(encoder, solver),
        }
    }
}

pub struct CopyBufferNode<S> {
    pub label: &'static str,
    pub src: fn(&S) -> &wgpu::Buffer,
    pub dst: fn(&S) -> &wgpu::Buffer,
    pub size_bytes: fn(&S) -> u64,
}

impl<S> CopyBufferNode<S> {
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, solver: &S) {
        let src = (self.src)(solver);
        let dst = (self.dst)(solver);
        let size = (self.size_bytes)(solver);
        encoder.copy_buffer_to_buffer(src, 0, dst, 0, size);
    }
}

pub struct ComputeNode<S> {
    pub label: &'static str,
    pub pipeline: fn(&S) -> &wgpu::ComputePipeline,
    pub bind_groups: fn(&S, &mut wgpu::ComputePass),
    pub workgroups: fn(&S) -> (u32, u32, u32),
}

impl<S> ComputeNode<S> {
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, solver: &S) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(self.label),
            timestamp_writes: None,
        });
        cpass.set_pipeline((self.pipeline)(solver));
        (self.bind_groups)(solver, &mut cpass);
        let (x, y, z) = (self.workgroups)(solver);
        cpass.dispatch_workgroups(x, y, z);
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelGraphTimings {
    pub total_seconds: f64,
    pub nodes: Vec<KernelNodeTiming>,
}

impl KernelGraphTimings {
    pub fn seconds_for(&self, label: &'static str) -> f64 {
        self.nodes
            .iter()
            .find(|node| node.label == label)
            .map(|node| node.seconds)
            .unwrap_or(0.0)
    }
}

#[derive(Debug, Clone)]
pub struct KernelNodeTiming {
    pub label: &'static str,
    pub seconds: f64,
}

