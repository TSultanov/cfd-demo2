use crate::solver::gpu::structs::GpuConstants;
use wgpu::util::DeviceExt;

#[derive(Clone, Copy, Debug)]
pub struct PackedStateConfig {
    pub state_stride: u32,
    pub flux_stride: u32,
}

/// Buffers-only resources for compressible packed-state fields.
pub struct CompressibleFieldBuffers {
    pub b_state: wgpu::Buffer,
    pub b_state_old: wgpu::Buffer,
    pub b_state_old_old: wgpu::Buffer,
    pub state_buffers: Vec<wgpu::Buffer>,
    pub b_fluxes: wgpu::Buffer,
    pub b_constants: wgpu::Buffer,
    pub constants: GpuConstants,
}

/// Full field resources including bind groups (created after pipelines).
pub struct CompressibleFieldResources {
    pub b_state: wgpu::Buffer,
    pub b_state_old: wgpu::Buffer,
    pub b_state_old_old: wgpu::Buffer,
    pub state_buffers: Vec<wgpu::Buffer>,
    pub b_fluxes: wgpu::Buffer,
    pub b_constants: wgpu::Buffer,
    pub bg_fields: wgpu::BindGroup,
    pub bg_fields_ping_pong: Vec<wgpu::BindGroup>,
    pub constants: GpuConstants,
}

pub fn init_compressible_field_buffers(
    device: &wgpu::Device,
    num_cells: u32,
    num_faces: u32,
    config: PackedStateConfig,
) -> CompressibleFieldBuffers {
    let state_len = num_cells as usize * config.state_stride as usize;
    let zero_state = vec![0.0f32; state_len];

    let b_state = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Compressible State Buffer"),
        contents: bytemuck::cast_slice(&zero_state),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let b_state_old = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Compressible State Old Buffer"),
        contents: bytemuck::cast_slice(&zero_state),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let b_state_old_old = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Compressible State Old Old Buffer"),
        contents: bytemuck::cast_slice(&zero_state),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let flux_len = num_faces as usize * config.flux_stride as usize;
    let zero_fluxes = vec![0.0f32; flux_len];
    let b_fluxes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Compressible Fluxes Buffer"),
        contents: bytemuck::cast_slice(&zero_fluxes),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let constants = GpuConstants {
        dt: 0.0001,
        dt_old: 0.0001,
        time: 0.0,
        viscosity: 0.0,
        density: 1.0,
        component: 0,
        alpha_p: 1.0,
        scheme: 0,
        alpha_u: 1.0,
        stride_x: 65535 * 64,
        time_scheme: 0,
        inlet_velocity: 0.0,
        ramp_time: 0.0,
        precond_type: 0,
    };
    let b_constants = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Compressible Constants Buffer"),
        contents: bytemuck::bytes_of(&constants),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let state_buffers = vec![
        b_state.clone(),
        b_state_old.clone(),
        b_state_old_old.clone(),
    ];

    CompressibleFieldBuffers {
        b_state,
        b_state_old,
        b_state_old_old,
        state_buffers,
        b_fluxes,
        b_constants,
        constants,
    }
}

pub fn create_compressible_field_bind_groups(
    device: &wgpu::Device,
    buffers: CompressibleFieldBuffers,
    bgl_fields: &wgpu::BindGroupLayout,
) -> CompressibleFieldResources {
    let mut bg_fields_ping_pong = Vec::new();

    for i in 0..3 {
        let (idx_state, idx_old, idx_old_old) = match i {
            0 => (0, 1, 2),
            1 => (2, 0, 1),
            2 => (1, 2, 0),
            _ => (0, 1, 2),
        };

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Compressible Fields Bind Group {}", i)),
            layout: bgl_fields,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.state_buffers[idx_state].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.state_buffers[idx_old].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.state_buffers[idx_old_old].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.b_fluxes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.b_constants.as_entire_binding(),
                },
            ],
        });
        bg_fields_ping_pong.push(bg);
    }

    let bg_fields = bg_fields_ping_pong[0].clone();

    CompressibleFieldResources {
        b_state: buffers.b_state,
        b_state_old: buffers.b_state_old,
        b_state_old_old: buffers.b_state_old_old,
        state_buffers: buffers.state_buffers,
        b_fluxes: buffers.b_fluxes,
        b_constants: buffers.b_constants,
        bg_fields,
        bg_fields_ping_pong,
        constants: buffers.constants,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_state_config_is_copy() {
        let config = PackedStateConfig {
            state_stride: 7,
            flux_stride: 4,
        };
        let _ = config;
    }
}
