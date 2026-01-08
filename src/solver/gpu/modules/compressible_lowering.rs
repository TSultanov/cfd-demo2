use crate::solver::gpu::csr::build_block_csr;
use crate::solver::gpu::init::compressible_fields::{
    init_compressible_field_buffers, CompressibleFieldBuffers, PackedStateConfig,
};
use crate::solver::gpu::init::linear_solver::matrix;
use crate::solver::gpu::init::mesh;
use crate::solver::mesh::Mesh;
use crate::solver::model::backend::StateLayout;

use super::ports::{BufF32, BufU32, LoweredBuffers, Lowerer, Port};

#[derive(Clone, Copy, Debug)]
pub struct CompressibleLinearPorts {
    pub rhs: Port<BufF32>,
    pub x: Port<BufF32>,
    pub scalar_row_offsets: Port<BufU32>,
}

pub struct CompressibleLowered {
    pub num_cells: u32,
    pub num_faces: u32,
    pub unknowns_per_cell: u32,
    pub num_unknowns: u32,
    pub state_stride: u32,

    pub ports: CompressibleLinearPorts,
    pub buffers: LoweredBuffers,

    pub mesh: mesh::MeshResources,
    pub fields: CompressibleFieldBuffers,

    pub scalar_row_offsets: Vec<u32>,
    pub scalar_col_indices: Vec<u32>,
    pub block_row_offsets: Vec<u32>,
    pub block_col_indices: Vec<u32>,

    pub matrix: matrix::MatrixResources,
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

        let mut lowerer = Lowerer::new(device);
        let scalar_row_offsets_port = lowerer.buffer_u32_init(
            "compressible:scalar_row_offsets",
            &mesh_res.row_offsets,
            wgpu::BufferUsages::STORAGE,
            "Compressible Scalar Row Offsets",
        );
        let rhs_port = lowerer.buffer_f32(
            "compressible:rhs",
            num_unknowns as u64 * 4,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            "Compressible RHS",
        );
        let x_port = lowerer.buffer_f32(
            "compressible:x",
            num_unknowns as u64 * 4,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            "Compressible Solution",
        );
        let ports = CompressibleLinearPorts {
            rhs: rhs_port,
            x: x_port,
            scalar_row_offsets: scalar_row_offsets_port,
        };
        let buffers: LoweredBuffers = lowerer.finish();

        Self {
            num_cells,
            num_faces,
            unknowns_per_cell,
            num_unknowns,
            state_stride,
            ports,
            buffers,
            mesh: mesh_res,
            fields,
            scalar_row_offsets,
            scalar_col_indices,
            block_row_offsets,
            block_col_indices,
            matrix,
        }
    }
}
