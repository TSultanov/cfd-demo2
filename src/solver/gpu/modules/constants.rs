use crate::solver::gpu::structs::GpuConstants;
use wgpu::util::DeviceExt;

pub struct ConstantsModule {
    values: GpuConstants,
    buffer: wgpu::Buffer,
}

impl ConstantsModule {
    pub fn new(device: &wgpu::Device, values: GpuConstants, label: &'static str) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(&values),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        Self { values, buffer }
    }

    pub fn values(&self) -> &GpuConstants {
        &self.values
    }

    pub fn values_mut(&mut self) -> &mut GpuConstants {
        &mut self.values
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn write(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&self.values));
    }
}
