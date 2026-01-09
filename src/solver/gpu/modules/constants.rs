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

    pub fn set_dt(&mut self, queue: &wgpu::Queue, dt: f32) {
        if self.values.time <= 0.0 {
            self.values.dt_old = dt;
        } else {
            self.values.dt_old = self.values.dt;
        }
        self.values.dt = dt;
        self.write(queue);
    }

    pub fn finalize_dt_old(&mut self, queue: &wgpu::Queue) {
        self.values.dt_old = self.values.dt;
        self.write(queue);
    }

    pub fn advance_time(&mut self, queue: &wgpu::Queue) {
        self.values.time += self.values.dt;
        self.write(queue);
    }
}
