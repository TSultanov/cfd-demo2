use crate::solver::gpu::structs::GpuConstants;
use crate::solver::gpu::model_defaults::default_incompressible_model;
use wgpu::util::DeviceExt;

/// Buffers-only resources (created before pipelines)
pub struct FieldBuffers {
    pub b_state: wgpu::Buffer,
    pub b_state_old: wgpu::Buffer,
    pub b_state_old_old: wgpu::Buffer,
    pub state_buffers: Vec<wgpu::Buffer>,
    pub b_fluxes: wgpu::Buffer,
    pub b_constants: wgpu::Buffer,
    pub constants: GpuConstants,
}

/// Full field resources including bind groups (created after pipelines)
pub struct FieldResources {
    /// Current fluid state buffer (read/write)
    pub b_state: wgpu::Buffer,
    /// Previous timestep fluid state buffer (read)
    pub b_state_old: wgpu::Buffer,
    /// Two timesteps ago fluid state buffer (read, for BDF2)
    pub b_state_old_old: wgpu::Buffer,
    /// Pool of 3 state buffers for ping-pong time stepping
    pub state_buffers: Vec<wgpu::Buffer>,
    /// Face-based mass fluxes (per face, not per cell)
    pub b_fluxes: wgpu::Buffer,
    /// Simulation constants (uniform buffer)
    pub b_constants: wgpu::Buffer,
    /// Main fields bind group
    pub bg_fields: wgpu::BindGroup,
    /// Ping-pong bind groups for time stepping
    pub bg_fields_ping_pong: Vec<wgpu::BindGroup>,
    /// Simulation constants (CPU-side copy)
    pub constants: GpuConstants,
}

/// Create only the buffers (before pipelines are created)
pub fn init_field_buffers(device: &wgpu::Device, num_cells: u32, num_faces: u32) -> FieldBuffers {
    let state_stride = default_incompressible_model().state_layout.stride();
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
    let b_constants = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Constants Buffer"),
        contents: bytemuck::bytes_of(&constants),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Store buffers for ping-pong (but cloning doesn't actually clone GPU memory)
    let state_buffers = vec![
        b_state.clone(),
        b_state_old.clone(),
        b_state_old_old.clone(),
    ];

    FieldBuffers {
        b_state,
        b_state_old,
        b_state_old_old,
        state_buffers,
        b_fluxes,
        b_constants,
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

    FieldResources {
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
