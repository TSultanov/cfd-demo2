use wgpu::util::DeviceExt;

pub struct MatrixResources {
    pub b_row_offsets: wgpu::Buffer,
    pub b_col_indices: wgpu::Buffer,
    pub b_matrix_values: wgpu::Buffer,
    pub num_nonzeros: u32,
}

pub fn init_matrix(
    device: &wgpu::Device,
    row_offsets: &[u32],
    col_indices: &[u32],
) -> MatrixResources {
    let num_nonzeros = row_offsets.last().cloned().unwrap_or(0);

    let b_row_offsets = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Row Offsets Buffer"),
        contents: bytemuck::cast_slice(row_offsets),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let b_col_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Col Indices Buffer"),
        contents: bytemuck::cast_slice(col_indices),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let b_matrix_values = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Matrix Values Buffer"),
        size: (num_nonzeros as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    MatrixResources {
        b_row_offsets,
        b_col_indices,
        b_matrix_values,
        num_nonzeros,
    }
}
