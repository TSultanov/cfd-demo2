use crate::solver::gpu::modules::constants::ConstantsModule;
use crate::solver::gpu::modules::state::PingPongState;
use crate::solver::gpu::structs::{GpuConstants, GpuLowMachParams};
use wgpu::util::DeviceExt;

#[derive(Clone, Copy, Debug)]
pub struct PackedStateConfig {
    pub state_stride: u32,
    pub flux_stride: u32,
}

/// Buffers-only resources for compressible packed-state fields.
pub struct CompressibleFieldBuffers {
    pub state: PingPongState,
    pub b_state_iter: wgpu::Buffer,
    pub b_fluxes: wgpu::Buffer,
    pub b_grad_rho: wgpu::Buffer,
    pub b_grad_rho_u_x: wgpu::Buffer,
    pub b_grad_rho_u_y: wgpu::Buffer,
    pub b_grad_rho_e: wgpu::Buffer,
    pub b_low_mach_params: wgpu::Buffer,
    pub constants: ConstantsModule,
    pub low_mach_params: GpuLowMachParams,
}

/// Full field resources including bind groups (created after pipelines).
pub struct CompressibleFieldResources {
    pub state: PingPongState,
    pub b_state_iter: wgpu::Buffer,
    pub b_fluxes: wgpu::Buffer,
    pub b_grad_rho: wgpu::Buffer,
    pub b_grad_rho_u_x: wgpu::Buffer,
    pub b_grad_rho_u_y: wgpu::Buffer,
    pub b_grad_rho_e: wgpu::Buffer,
    pub b_low_mach_params: wgpu::Buffer,
    pub bg_fields: wgpu::BindGroup,
    pub bg_fields_ping_pong: Vec<wgpu::BindGroup>,
    pub constants: ConstantsModule,
    pub low_mach_params: GpuLowMachParams,
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
    let b_state_iter = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Compressible State Iter Buffer"),
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

    let zero_grad = vec![[0.0f32; 2]; num_cells as usize];
    let b_grad_rho = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Compressible Grad Rho Buffer"),
        contents: bytemuck::cast_slice(&zero_grad),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let b_grad_rho_u_x = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Compressible Grad RhoU X Buffer"),
        contents: bytemuck::cast_slice(&zero_grad),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let b_grad_rho_u_y = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Compressible Grad RhoU Y Buffer"),
        contents: bytemuck::cast_slice(&zero_grad),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let b_grad_rho_e = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Compressible Grad RhoE Buffer"),
        contents: bytemuck::cast_slice(&zero_grad),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let constants = GpuConstants {
        dt: 0.0001,
        dt_old: 0.0001,
        dtau: 0.0,
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
    };
    let constants = ConstantsModule::new(device, constants, "Compressible Constants Buffer");

    let low_mach_params = GpuLowMachParams::default();
    let b_low_mach_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Compressible Low-Mach Params Buffer"),
        contents: bytemuck::bytes_of(&low_mach_params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let state = PingPongState::new([b_state, b_state_old, b_state_old_old]);

    CompressibleFieldBuffers {
        state,
        b_state_iter,
        b_fluxes,
        b_grad_rho,
        b_grad_rho_u_x,
        b_grad_rho_u_y,
        b_grad_rho_e,
        b_low_mach_params,
        constants,
        low_mach_params,
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
                    resource: buffers.state.buffers()[idx_state].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.state.buffers()[idx_old].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.state.buffers()[idx_old_old].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.b_fluxes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.constants.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.b_grad_rho.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buffers.b_grad_rho_u_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buffers.b_grad_rho_u_y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: buffers.b_grad_rho_e.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: buffers.b_state_iter.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: buffers.b_low_mach_params.as_entire_binding(),
                },
            ],
        });
        bg_fields_ping_pong.push(bg);
    }

    let bg_fields = bg_fields_ping_pong[0].clone();

    CompressibleFieldResources {
        state: buffers.state,
        b_state_iter: buffers.b_state_iter,
        b_fluxes: buffers.b_fluxes,
        b_grad_rho: buffers.b_grad_rho,
        b_grad_rho_u_x: buffers.b_grad_rho_u_x,
        b_grad_rho_u_y: buffers.b_grad_rho_u_y,
        b_grad_rho_e: buffers.b_grad_rho_e,
        b_low_mach_params: buffers.b_low_mach_params,
        bg_fields,
        bg_fields_ping_pong,
        constants: buffers.constants,
        low_mach_params: buffers.low_mach_params,
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
