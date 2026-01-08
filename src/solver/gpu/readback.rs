use std::collections::HashMap;
use std::sync::Mutex;

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

