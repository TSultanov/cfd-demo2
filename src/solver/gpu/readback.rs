use std::collections::HashMap;
use std::sync::Mutex;

use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::profiling::ProfilingStats;

#[cfg(feature = "profiling")]
use crate::solver::gpu::profiling::ProfileCategory;

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

    let staging_buffer = {
        #[cfg(feature = "profiling")]
        {
            if profiling.is_enabled() {
                use std::time::Instant;

                let t0 = Instant::now();
                let staging_buffer = cache.take_or_create(&context.device, size, label);
                profiling.record_gpu_alloc(label, size);
                profiling.record_location(
                    "read_buffer:create_staging",
                    ProfileCategory::GpuResourceCreation,
                    t0.elapsed(),
                    0,
                );
                staging_buffer
            } else {
                cache.take_or_create(&context.device, size, label)
            }
        }

        #[cfg(not(feature = "profiling"))]
        {
            cache.take_or_create(&context.device, size, label)
        }
    };

    let submission_index = {
        #[cfg(feature = "profiling")]
        {
            if profiling.is_enabled() {
                use std::time::Instant;

                let t1 = Instant::now();
                let mut encoder = context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
                let submission_index = context.queue.submit(Some(encoder.finish()));
                profiling.record_location(
                    "read_buffer:submit_copy",
                    ProfileCategory::GpuDispatch,
                    t1.elapsed(),
                    0,
                );
                submission_index
            } else {
                let mut encoder = context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
                context.queue.submit(Some(encoder.finish()))
            }
        }

        #[cfg(not(feature = "profiling"))]
        {
            let mut encoder = context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
            context.queue.submit(Some(encoder.finish()))
        }
    };

    let slice = staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();

    {
        #[cfg(feature = "profiling")]
        {
            if profiling.is_enabled() {
                use std::time::Instant;

                let t2 = Instant::now();
                slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
                profiling.record_location(
                    "read_buffer:map_async_request",
                    ProfileCategory::Other,
                    t2.elapsed(),
                    0,
                );
            } else {
                slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
            }
        }

        #[cfg(not(feature = "profiling"))]
        {
            slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        }
    }

    {
        #[cfg(feature = "profiling")]
        {
            if profiling.is_enabled() {
                use std::time::Instant;

                let t3 = Instant::now();
                let _ = context.device.poll(wgpu::PollType::Wait {
                    submission_index: Some(submission_index),
                    timeout: None,
                });
                profiling.record_location(
                    "read_buffer:device_poll_wait",
                    ProfileCategory::GpuSync,
                    t3.elapsed(),
                    0,
                );
            } else {
                let _ = context.device.poll(wgpu::PollType::Wait {
                    submission_index: Some(submission_index),
                    timeout: None,
                });
            }
        }

        #[cfg(not(feature = "profiling"))]
        {
            let _ = context.device.poll(wgpu::PollType::Wait {
                submission_index: Some(submission_index),
                timeout: None,
            });
        }
    }

    {
        #[cfg(feature = "profiling")]
        {
            if profiling.is_enabled() {
                use std::time::Instant;

                let t4 = Instant::now();
                rx.recv().unwrap().unwrap();
                profiling.record_location("read_buffer:channel_recv", ProfileCategory::Other, t4.elapsed(), 0);
            } else {
                rx.recv().unwrap().unwrap();
            }
        }

        #[cfg(not(feature = "profiling"))]
        {
            rx.recv().unwrap().unwrap();
        }
    }

    let result = {
        #[cfg(feature = "profiling")]
        {
            if profiling.is_enabled() {
                use std::time::Instant;

                let t5 = Instant::now();
                let data = slice.get_mapped_range();
                let result = data.to_vec();
                drop(data);
                staging_buffer.unmap();
                profiling.record_cpu_alloc("read_buffer:cpu_copy", size);
                profiling.record_location("read_buffer:memcpy", ProfileCategory::Other, t5.elapsed(), size);
                result
            } else {
                let data = slice.get_mapped_range();
                let result = data.to_vec();
                drop(data);
                staging_buffer.unmap();
                result
            }
        }

        #[cfg(not(feature = "profiling"))]
        {
            let data = slice.get_mapped_range();
            let result = data.to_vec();
            drop(data);
            staging_buffer.unmap();
            result
        }
    };

    cache.put(size, staging_buffer);
    result
}
