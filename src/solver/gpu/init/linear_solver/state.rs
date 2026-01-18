use crate::solver::gpu::structs::SolverParams;
use wgpu::util::DeviceExt;

pub struct StateResources {
    pub b_rhs: wgpu::Buffer,
    pub b_x: wgpu::Buffer,
    pub b_r: wgpu::Buffer,
    pub b_r0: wgpu::Buffer,
    pub b_p_solver: wgpu::Buffer,
    pub b_v: wgpu::Buffer,
    pub b_s: wgpu::Buffer,
    pub b_t: wgpu::Buffer,
    pub b_dot_result: wgpu::Buffer,
    pub b_dot_result_2: wgpu::Buffer,
    pub b_scalars: wgpu::Buffer,
    pub b_staging_scalar: wgpu::Buffer,
    pub b_solver_params: wgpu::Buffer,
    pub num_groups: u32,
}

pub fn init_state(device: &wgpu::Device, num_cells: u32) -> StateResources {
    let b_rhs = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("RHS Buffer"),
        size: (num_cells as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let b_x = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("X Buffer"),
        size: (num_cells as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let b_r = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("R Buffer"),
        size: (num_cells as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let b_r0 = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("R0 Buffer"),
        size: (num_cells as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let b_p_solver = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("P Solver Buffer"),
        size: (num_cells as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let b_v = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("V Buffer"),
        size: (num_cells as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let b_s = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("S Buffer"),
        size: (num_cells as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let b_t = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("T Buffer"),
        size: (num_cells as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Dot Product & Params
    let workgroup_size = 64;
    let num_groups = num_cells.div_ceil(workgroup_size);

    let b_dot_result = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Dot Result Buffer"),
        size: (num_groups as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let b_dot_result_2 = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Dot Result Buffer 2"),
        size: (num_groups as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let b_scalars = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Scalars Buffer"),
        size: 64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let b_staging_scalar = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer Scalar"),
        size: 64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let solver_params = SolverParams {
        n: num_cells,
        num_groups,
        padding: [0; 2],
    };
    let b_solver_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Solver Params Buffer"),
        contents: bytemuck::bytes_of(&solver_params),
        usage: wgpu::BufferUsages::UNIFORM
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    StateResources {
        b_rhs,
        b_x,
        b_r,
        b_r0,
        b_p_solver,
        b_v,
        b_s,
        b_t,
        b_dot_result,
        b_dot_result_2,
        b_scalars,
        b_staging_scalar,
        b_solver_params,
        num_groups,
    }
}
