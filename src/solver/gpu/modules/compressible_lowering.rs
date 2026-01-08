use crate::solver::gpu::csr::build_block_csr;
use crate::solver::gpu::init::compressible_fields::{
    init_compressible_field_buffers, CompressibleFieldBuffers, PackedStateConfig,
};
use crate::solver::mesh::Mesh;
use crate::solver::model::backend::StateLayout;

use super::model_lowerer::LoweredCommon;
use super::ports::{BufF32, BufU32, Port};

#[derive(Clone, Copy, Debug)]
pub struct CompressibleLinearPorts {
    pub rhs: Port<BufF32>,
    pub x: Port<BufF32>,
    pub scalar_row_offsets: Port<BufU32>,
}

#[derive(Clone, Copy, Debug)]
pub struct CompressibleMatrixPorts {
    pub row_offsets: Port<BufU32>,
    pub col_indices: Port<BufU32>,
    pub values: Port<BufF32>,
}

pub struct CompressibleLowered {
    pub common: LoweredCommon,
    pub unknowns_per_cell: u32,
    pub num_unknowns: u32,
    pub state_stride: u32,

    pub ports: CompressibleLinearPorts,
    pub matrix_ports: CompressibleMatrixPorts,
    pub fields: CompressibleFieldBuffers,

    pub scalar_row_offsets: Vec<u32>,
    pub scalar_col_indices: Vec<u32>,
    pub block_row_offsets: Vec<u32>,
    pub block_col_indices: Vec<u32>,
}

impl CompressibleLowered {
    pub fn lower(
        device: &wgpu::Device,
        mesh_input: &Mesh,
        state_layout: &StateLayout,
        unknowns_per_cell: u32,
        flux_stride: u32,
    ) -> Self {
        let mut common = LoweredCommon::new(device, mesh_input);
        let num_unknowns = common.num_cells * unknowns_per_cell;
        let state_stride = state_layout.stride();

        let fields = init_compressible_field_buffers(
            device,
            common.num_cells,
            common.num_faces,
            PackedStateConfig {
                state_stride,
                flux_stride,
            },
        );

        let scalar_row_offsets = common.mesh.row_offsets.clone();
        let scalar_col_indices = common.mesh.col_indices.clone();
        let (row_offsets, col_indices) =
            build_block_csr(&common.mesh.row_offsets, &common.mesh.col_indices, unknowns_per_cell);
        let block_row_offsets = row_offsets.clone();
        let block_col_indices = col_indices.clone();

        let num_nonzeros = row_offsets.last().cloned().unwrap_or(0) as u64;

        let (
            scalar_row_offsets_port,
            rhs_port,
            x_port,
            block_row_offsets_port,
            block_col_indices_port,
            block_matrix_values_port,
        ) = {
            let mut lowerer = common.lowerer(device);
            let scalar_row_offsets_port = lowerer.buffer_u32_init(
                "compressible:scalar_row_offsets",
                &scalar_row_offsets,
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
            let block_row_offsets_port = lowerer.buffer_u32_init(
                "compressible:block_row_offsets",
                &block_row_offsets,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                "Compressible Block Row Offsets",
            );
            let block_col_indices_port = lowerer.buffer_u32_init(
                "compressible:block_col_indices",
                &block_col_indices,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                "Compressible Block Col Indices",
            );
            let block_matrix_values_port = lowerer.buffer_f32(
                "compressible:block_matrix_values",
                num_nonzeros * 4,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                "Compressible Block Matrix Values",
            );
            (
                scalar_row_offsets_port,
                rhs_port,
                x_port,
                block_row_offsets_port,
                block_col_indices_port,
                block_matrix_values_port,
            )
        };
        let ports = CompressibleLinearPorts {
            rhs: rhs_port,
            x: x_port,
            scalar_row_offsets: scalar_row_offsets_port,
        };
        let matrix_ports = CompressibleMatrixPorts {
            row_offsets: block_row_offsets_port,
            col_indices: block_col_indices_port,
            values: block_matrix_values_port,
        };

        Self {
            common,
            unknowns_per_cell,
            num_unknowns,
            state_stride,
            ports,
            matrix_ports,
            fields,
            scalar_row_offsets,
            scalar_col_indices,
            block_row_offsets,
            block_col_indices,
        }
    }
}
