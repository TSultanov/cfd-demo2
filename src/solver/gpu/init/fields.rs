use crate::solver::gpu::bindings::momentum_assembly_v2;
use crate::solver::gpu::structs::GpuConstants;
use wgpu::util::DeviceExt;

pub struct FieldResources {
    pub b_u: wgpu::Buffer,
    pub b_u_old: wgpu::Buffer,
    pub b_u_old_old: wgpu::Buffer,
    pub b_p: wgpu::Buffer,
    pub b_d_p: wgpu::Buffer,
    pub b_fluxes: wgpu::Buffer,
    pub b_grad_p: wgpu::Buffer,
    pub b_grad_component: wgpu::Buffer,
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
    };
    let b_constants = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Constants Buffer"),
        contents: bytemuck::bytes_of(&constants),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Group 1: Fields (Read/Write)
    let bgl_fields =
        device.create_bind_group_layout(&momentum_assembly_v2::WgpuBindGroup1::LAYOUT_DESCRIPTOR);

    let bg_fields = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Fields Bind Group"),
        layout: &bgl_fields,
        entries: &momentum_assembly_v2::WgpuBindGroup1Entries::new(
            momentum_assembly_v2::WgpuBindGroup1EntriesParams {
                u: b_u.as_entire_buffer_binding(),
                p: b_p.as_entire_buffer_binding(),
                fluxes: b_fluxes.as_entire_buffer_binding(),
                constants: b_constants.as_entire_buffer_binding(),
                grad_p: b_grad_p.as_entire_buffer_binding(),
                d_p: b_d_p.as_entire_buffer_binding(),
                grad_component: b_grad_component.as_entire_buffer_binding(),
                u_old: b_u_old.as_entire_buffer_binding(),
                u_old_old: b_u_old_old.as_entire_buffer_binding(),
            },
        )
        .into_array(),
    });

    FieldResources {
        b_u,
        b_u_old,
        b_u_old_old,
        b_p,
        b_d_p,
        b_fluxes,
        b_grad_p,
        b_grad_component,
        b_constants,
        bg_fields,
        bgl_fields,
        constants,
    }
}
