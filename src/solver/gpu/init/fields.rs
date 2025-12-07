use crate::solver::gpu::bindings::prepare_coupled;
use crate::solver::gpu::structs::GpuConstants;
use wgpu::util::DeviceExt;

pub struct FieldResources {
    pub b_u: wgpu::Buffer,
    pub b_u_old: wgpu::Buffer,
    pub b_u_old_old: wgpu::Buffer,
    pub u_buffers: Vec<wgpu::Buffer>,
    pub bg_fields_ping_pong: Vec<wgpu::BindGroup>,
    pub b_p: wgpu::Buffer,
    pub b_d_p: wgpu::Buffer,
    pub b_fluxes: wgpu::Buffer,
    pub b_grad_p: wgpu::Buffer,
    pub b_grad_component: wgpu::Buffer,
    pub b_temperature: wgpu::Buffer,
    pub b_energy: wgpu::Buffer,
    pub b_density: wgpu::Buffer,
    pub b_grad_e: wgpu::Buffer,
    pub b_constants: wgpu::Buffer,
    pub bg_fields: wgpu::BindGroup,
    pub bgl_fields: wgpu::BindGroupLayout,
    pub constants: GpuConstants,
}

pub fn init_fields(device: &wgpu::Device, num_cells: u32, num_faces: u32) -> FieldResources {
    // --- Field Buffers ---
    let zero_vecs = vec![[0.0f32; 2]; num_cells as usize];
    let b_u = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("U Buffer"),
        contents: bytemuck::cast_slice(&zero_vecs),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let b_u_old = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("U Old Buffer"),
        contents: bytemuck::cast_slice(&zero_vecs),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let b_u_old_old = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("U Old Old Buffer"),
        contents: bytemuck::cast_slice(&zero_vecs),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let zero_scalars = vec![0.0f32; num_cells as usize];
    let b_p = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("P Buffer"),
        contents: bytemuck::cast_slice(&zero_scalars),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let b_d_p = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("D_P Buffer"),
        contents: bytemuck::cast_slice(&zero_scalars),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let zero_fluxes = vec![0.0f32; num_faces as usize];
    let b_fluxes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Fluxes Buffer"),
        contents: bytemuck::cast_slice(&zero_fluxes),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let b_grad_p = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Grad P Buffer"),
        contents: bytemuck::cast_slice(&zero_vecs),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let b_grad_component = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Grad Component Buffer"),
        contents: bytemuck::cast_slice(&zero_vecs),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let b_temperature = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Temperature Buffer"),
        contents: bytemuck::cast_slice(&zero_scalars),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let b_energy = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Energy Buffer"),
        contents: bytemuck::cast_slice(&zero_scalars),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let b_density = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Density Buffer"),
        contents: bytemuck::cast_slice(&zero_scalars),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let b_grad_e = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Grad E Buffer"),
        contents: bytemuck::cast_slice(&zero_vecs),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let constants = GpuConstants {
        dt: 0.0001, // Reduced dt
        dt_old: 0.0001,
        time: 0.0,
        viscosity: 0.01,
        density: 1.0,
        component: 0,
        alpha_p: 1.0, // Default pressure relaxation
        scheme: 0,    // Upwind
        alpha_u: 0.7, // Default velocity under-relaxation
        stride_x: 65535 * 64,
        time_scheme: 0,
        inlet_velocity: 1.0,
        ramp_time: 0.1,
        precond_type: 0,
        gamma: 1.4,
        r_gas: 287.058,
        is_compressible: 0,
        gravity_x: 0.0,
        gravity_y: -9.81,
        pad0: 0.0,
        pad1: 0.0,
    };
    let b_constants = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Constants Buffer"),
        contents: bytemuck::bytes_of(&constants),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Group 1: Fields (Read/Write)
    let bgl_fields =
        device.create_bind_group_layout(&prepare_coupled::WgpuBindGroup1::LAYOUT_DESCRIPTOR);

    // Create 3 bind groups for ping-pong
    // Buffers: [b_u, b_u_old, b_u_old_old] -> [0, 1, 2]
    // BG0: u=0, u_old=1, u_old_old=2
    // BG1: u=2, u_old=0, u_old_old=1
    // BG2: u=1, u_old=2, u_old_old=0

    let u_buffers_vec = vec![b_u.clone(), b_u_old.clone(), b_u_old_old.clone()];
    let mut bg_fields_ping_pong = Vec::new();

    for i in 0..3 {
        let idx_u = match i {
            0 => 0,
            1 => 2,
            2 => 1,
            _ => 0,
        };
        let idx_u_old = match i {
            0 => 1,
            1 => 0,
            2 => 2,
            _ => 0,
        };
        let idx_u_old_old = match i {
            0 => 2,
            1 => 1,
            2 => 0,
            _ => 0,
        };

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Fields Bind Group {}", i)),
            layout: &bgl_fields,
            entries: &prepare_coupled::WgpuBindGroup1Entries::new(
                prepare_coupled::WgpuBindGroup1EntriesParams {
                    u: u_buffers_vec[idx_u].as_entire_buffer_binding(),
                    p: b_p.as_entire_buffer_binding(),
                    fluxes: b_fluxes.as_entire_buffer_binding(),
                    constants: b_constants.as_entire_buffer_binding(),
                    grad_p: b_grad_p.as_entire_buffer_binding(),
                    d_p: b_d_p.as_entire_buffer_binding(),
                    grad_component: b_grad_component.as_entire_buffer_binding(),
                    u_old: u_buffers_vec[idx_u_old].as_entire_buffer_binding(),
                    u_old_old: u_buffers_vec[idx_u_old_old].as_entire_buffer_binding(),
                    temperature: b_temperature.as_entire_buffer_binding(),
                    energy: b_energy.as_entire_buffer_binding(),
                    density: b_density.as_entire_buffer_binding(),
                    grad_e: b_grad_e.as_entire_buffer_binding(),
                },
            )
            .into_array(),
        });
        bg_fields_ping_pong.push(bg);
    }

    let bg_fields = bg_fields_ping_pong[0].clone();

    FieldResources {
        b_u,
        b_u_old,
        b_u_old_old,
        u_buffers: u_buffers_vec,
        bg_fields_ping_pong,
        b_p,
        b_d_p,
        b_fluxes,
        b_grad_p,
        b_grad_component,
        b_temperature,
        b_energy,
        b_density,
        b_grad_e,
        b_constants,
        bg_fields,
        bgl_fields,
        constants,
    }
}
