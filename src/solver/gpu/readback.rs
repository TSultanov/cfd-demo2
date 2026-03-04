use std::collections::HashMap;
use std::sync::Mutex;

use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::profiling::ProfilingStats;

#[cfg(feature = "profiling")]
use crate::solver::gpu::profiling::ProfileCategory;
#[cfg(feature = "profiling")]
use std::time::Instant;

#[derive(Default)]
pub struct StagingBufferCache {
    buffers: Mutex<HashMap<u64, wgpu::Buffer>>,
}

impl StagingBufferCache {
    pub fn take_or_create(
        &self,
        device: &wgpu::Device,
        size: u64,
        label: &'static str,
    ) -> wgpu::Buffer {
        if let Some(buffer) = self.buffers.lock().unwrap().remove(&size) {
            return buffer;
        }
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub fn put(&self, size: u64, buffer: wgpu::Buffer) {
        self.buffers.lock().unwrap().insert(size, buffer);
    }
}

/// Record a profiling event if the `profiling` feature is enabled and active.
///
/// This macro eliminates the need to duplicate the entire function body for each
/// cfg(feature = "profiling") / cfg(not(feature = "profiling")) branch.
macro_rules! profile_location {
    ($profiling:expr, $location:expr, $category:expr, $body:expr) => {{
        #[cfg(feature = "profiling")]
        {
            if $profiling.is_enabled() {
                let _t = Instant::now();
                let result = $body;
                $profiling.record_location($location, $category, _t.elapsed(), 0);
                result
            } else {
                $body
            }
        }
        #[cfg(not(feature = "profiling"))]
        {
            $body
        }
    }};
}

pub async fn read_buffer_cached(
    context: &GpuContext,
    cache: &StagingBufferCache,
    profiling: &ProfilingStats,
    buffer: &wgpu::Buffer,
    size: u64,
    label: &'static str,
) -> Vec<u8> {
    #[cfg(not(feature = "profiling"))]
    let _ = profiling;

    let staging_buffer = profile_location!(
        profiling,
        "read_buffer:create_staging",
        ProfileCategory::GpuResourceCreation,
        {
            let buf = cache.take_or_create(&context.device, size, label);
            #[cfg(feature = "profiling")]
            if profiling.is_enabled() {
                profiling.record_gpu_alloc(label, size);
            }
            buf
        }
    );

    let submission_index = profile_location!(
        profiling,
        "read_buffer:submit_copy",
        ProfileCategory::GpuDispatch,
        {
            let mut encoder = context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
            let idx = context.queue.submit(Some(encoder.finish()));
            crate::count_submission!("Readback", "read_buffer_copy");
            idx
        }
    );

    let slice = staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();

    profile_location!(
        profiling,
        "read_buffer:map_async_request",
        ProfileCategory::Other,
        {
            slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        }
    );

    profile_location!(
        profiling,
        "read_buffer:device_poll_wait",
        ProfileCategory::GpuSync,
        {
            let _ = context.device.poll(wgpu::PollType::Wait {
                submission_index: Some(submission_index),
                timeout: None,
            });
        }
    );

    profile_location!(
        profiling,
        "read_buffer:channel_recv",
        ProfileCategory::Other,
        {
            rx.recv().unwrap().unwrap();
        }
    );

    let result = profile_location!(
        profiling,
        "read_buffer:memcpy",
        ProfileCategory::Other,
        {
            let data = slice.get_mapped_range();
            let result = data.to_vec();
            drop(data);
            staging_buffer.unmap();
            #[cfg(feature = "profiling")]
            if profiling.is_enabled() {
                profiling.record_cpu_alloc("read_buffer:cpu_copy", size);
            }
            result
        }
    );

    cache.put(size, staging_buffer);
    result
}
