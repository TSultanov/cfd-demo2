pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    pub async fn new(device: Option<wgpu::Device>, queue: Option<wgpu::Queue>) -> Result<Self, String> {
        if let (Some(device), Some(queue)) = (device, queue) {
            return Ok(Self { device, queue });
        }

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("no compatible GPU adapter found: {e}"))?;

        // Get the adapter's supported limits to allow larger buffers for fine meshes
        let adapter_limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 31,
                    // Use adapter's max buffer size to support large meshes
                    max_buffer_size: adapter_limits.max_buffer_size,
                    max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
                    ..wgpu::Limits::downlevel_defaults()
                },
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| format!("failed to create GPU device: {e}"))?;

        Ok(Self { device, queue })
    }
}
