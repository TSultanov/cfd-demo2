use crate::solver::gpu::csr::build_block_csr;
use crate::solver::gpu::init::compressible_fields::{
    init_compressible_field_buffers, CompressibleFieldBuffers, PackedStateConfig,
};
use crate::solver::gpu::init::linear_solver::matrix;
use crate::solver::gpu::init::mesh;
use crate::solver::mesh::Mesh;
use crate::solver::model::backend::StateLayout;
use wgpu::util::DeviceExt;

pub struct CompressibleLowered {
    pub num_cells: u32,
    pub num_faces: u32,
    pub unknowns_per_cell: u32,
    pub num_unknowns: u32,
    pub state_stride: u32,

    pub mesh: mesh::MeshResources,
    pub fields: CompressibleFieldBuffers,

    pub scalar_row_offsets: Vec<u32>,
    pub scalar_col_indices: Vec<u32>,
    pub block_row_offsets: Vec<u32>,
    pub block_col_indices: Vec<u32>,

    pub matrix: matrix::MatrixResources,
    pub b_scalar_row_offsets: wgpu::Buffer,
    pub b_rhs: wgpu::Buffer,
    pub b_x: wgpu::Buffer,
}

impl CompressibleLowered {
    pub fn lower(
        device: &wgpu::Device,
        mesh_input: &Mesh,
        state_layout: &StateLayout,
        unknowns_per_cell: u32,
        flux_stride: u32,
    ) -> Self {
        let num_cells = mesh_input.cell_cx.len() as u32;
        let num_faces = mesh_input.face_owner.len() as u32;
        let num_unknowns = num_cells * unknowns_per_cell;
        let state_stride = state_layout.stride();

        let mesh_res = mesh::init_mesh(device, mesh_input);
        let fields = init_compressible_field_buffers(
            device,
            num_cells,
            num_faces,
            PackedStateConfig {
                state_stride,
                flux_stride,
            },
        );

        let scalar_row_offsets = mesh_res.row_offsets.clone();
        let scalar_col_indices = mesh_res.col_indices.clone();
        let (row_offsets, col_indices) =
            build_block_csr(&mesh_res.row_offsets, &mesh_res.col_indices, unknowns_per_cell);
        let block_row_offsets = row_offsets.clone();
        let block_col_indices = col_indices.clone();

        let matrix = matrix::init_matrix(device, &row_offsets, &col_indices);
        let b_scalar_row_offsets = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Compressible Scalar Row Offsets"),
            contents: bytemuck::cast_slice(&mesh_res.row_offsets),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let b_rhs = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compressible RHS"),
            size: num_unknowns as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let b_x = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compressible Solution"),
            size: num_unknowns as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self {
            num_cells,
            num_faces,
            unknowns_per_cell,
            num_unknowns,
            state_stride,
            mesh: mesh_res,
            fields,
            scalar_row_offsets,
            scalar_col_indices,
            block_row_offsets,
            block_col_indices,
            matrix,
            b_scalar_row_offsets,
            b_rhs,
            b_x,
        }
    }
}

