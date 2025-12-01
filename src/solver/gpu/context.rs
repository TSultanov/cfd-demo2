pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
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
            .unwrap();

        // Get the adapter's supported limits to allow larger buffers for fine meshes
        let adapter_limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        max_storage_buffers_per_shader_stage: 31,
                        // Use adapter's max buffer size to support large meshes
                        max_buffer_size: adapter_limits.max_buffer_size,
                        max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
                        ..wgpu::Limits::downlevel_defaults()
                    },
                },
                None,
            )
            .await
            .unwrap();

        Self { device, queue }
    }
}
