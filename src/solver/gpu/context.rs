pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// True when the device was created with `TIMESTAMP_QUERY` and the profiler
    /// can resolve GPU-side timestamps from a `QuerySet`. `false` for
    /// externally-supplied devices (whose feature set we don't control) and on
    /// adapters without timestamp support.
    pub timestamp_query: bool,
    /// True when `TIMESTAMP_QUERY_INSIDE_ENCODERS` is also available, so the
    /// profiler can write a timestamp on a bare command encoder (between passes /
    /// between graph submissions) rather than only at compute-pass boundaries.
    /// This is what lets us bracket a whole graph's GPU work without touching each
    /// pass descriptor.
    pub timestamps_inside_encoders: bool,
    /// Nanoseconds per timestamp tick (`queue.get_timestamp_period()`), used to
    /// convert raw query deltas to durations. `0.0` when timestamps are disabled.
    pub timestamp_period_ns: f32,
}

impl GpuContext {
    pub async fn new(device: Option<wgpu::Device>, queue: Option<wgpu::Queue>) -> Result<Self, String> {
        if let (Some(device), Some(queue)) = (device, queue) {
            // Externally-supplied device: we didn't request timestamp features, so
            // report the profiler's timestamp path as unavailable and fall back to
            // the poll-based per-graph timer.
            return Ok(Self {
                device,
                queue,
                timestamp_query: false,
                timestamps_inside_encoders: false,
                timestamp_period_ns: 0.0,
            });
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

        // Opt into timestamp queries when the adapter supports them, so the
        // profiler can measure true per-graph GPU time (a poll-based barrier has a
        // ~1ms round-trip floor on integrated GPUs that swamps sub-ms graphs).
        // These are inert unless a profiling session is running.
        let adapter_features = adapter.features();
        let timestamp_query = adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY);
        let timestamps_inside_encoders =
            adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
        let mut required_features = wgpu::Features::empty();
        if timestamp_query {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        if timestamps_inside_encoders {
            required_features |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features,
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

        let timestamp_period_ns = if timestamp_query {
            queue.get_timestamp_period()
        } else {
            0.0
        };

        if std::env::var("CFD2_GPU_INFO").is_ok() {
            eprintln!(
                "[gpu] adapter={:?} timestamp_query={timestamp_query} inside_encoders={timestamps_inside_encoders} period_ns={timestamp_period_ns}",
                adapter.get_info().name,
            );
        }

        Ok(Self {
            device,
            queue,
            timestamp_query,
            timestamps_inside_encoders,
            timestamp_period_ns,
        })
    }
}
