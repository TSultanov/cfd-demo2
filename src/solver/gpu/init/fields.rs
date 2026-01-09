use crate::solver::gpu::modules::constants::ConstantsModule;
use crate::solver::gpu::modules::state::PingPongState;
use crate::solver::gpu::structs::GpuConstants;
use wgpu::util::DeviceExt;

/// Buffers-only resources (created before pipelines)
pub struct FieldBuffers {
    pub state: PingPongState,
    pub b_fluxes: wgpu::Buffer,
    pub constants: ConstantsModule,
}

/// Full field resources including bind groups (created after pipelines)
pub struct FieldResources {
    pub state: PingPongState,
    /// Face-based mass fluxes (per face, not per cell)
    pub b_fluxes: wgpu::Buffer,
    /// Main fields bind group
    pub bg_fields: wgpu::BindGroup,
    /// Ping-pong bind groups for time stepping
    pub bg_fields_ping_pong: Vec<wgpu::BindGroup>,
    pub constants: ConstantsModule,
}

/// Create only the buffers (before pipelines are created)
pub fn init_field_buffers(
    device: &wgpu::Device,
    num_cells: u32,
    num_faces: u32,
    state_stride: u32,
) -> FieldBuffers {
    let zero_state = vec![0.0f32; num_cells as usize * state_stride as usize];

    let b_state = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Incompressible State Buffer"),
        contents: bytemuck::cast_slice(&zero_state),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let b_state_old = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Incompressible State Old Buffer"),
        contents: bytemuck::cast_slice(&zero_state),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let b_state_old_old = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Incompressible State Old Old Buffer"),
        contents: bytemuck::cast_slice(&zero_state),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    // --- Face-based fluxes buffer ---
    let zero_fluxes = vec![0.0f32; num_faces as usize];
    let b_fluxes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Fluxes Buffer"),
        contents: bytemuck::cast_slice(&zero_fluxes),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    // --- Constants uniform buffer ---
    let constants = GpuConstants {
        dt: 0.0001,
        dt_old: 0.0001,
        dtau: 0.0,
        time: 0.0,
        viscosity: 0.01,
        density: 1.0,
        component: 0,
        alpha_p: 1.0,
        scheme: 0,
        alpha_u: 0.7,
        stride_x: 65535 * 64,
        time_scheme: 0,
        inlet_velocity: 1.0,
        ramp_time: 0.1,
    };
    let constants = ConstantsModule::new(device, constants, "Incompressible Constants Buffer");

    FieldBuffers {
        state: PingPongState::new([b_state, b_state_old, b_state_old_old]),
        b_fluxes,
        constants,
    }
}

/// Create bind groups using a layout extracted from the pipeline
pub fn create_field_bind_groups(
    device: &wgpu::Device,
    buffers: FieldBuffers,
    bgl_fields: &wgpu::BindGroupLayout,
) -> FieldResources {
    // Create 3 ping-pong bind groups
    // BG0: state=0, state_old=1, state_old_old=2 (initial)
    // BG1: state=2, state_old=0, state_old_old=1 (after 1 step)
    // BG2: state=1, state_old=2, state_old_old=0 (after 2 steps)
    let mut bg_fields_ping_pong = Vec::new();

    for i in 0..3 {
        let (idx_state, idx_old, idx_old_old) = match i {
            0 => (0, 1, 2),
            1 => (2, 0, 1),
            2 => (1, 2, 0),
            _ => (0, 1, 2),
        };

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Incompressible Fields Bind Group {}", i)),
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
            ],
        });
        bg_fields_ping_pong.push(bg);
    }

    let bg_fields = bg_fields_ping_pong[0].clone();

    FieldResources {
        state: buffers.state,
        b_fluxes: buffers.b_fluxes,
        bg_fields,
        bg_fields_ping_pong,
        constants: buffers.constants,
    }
}
