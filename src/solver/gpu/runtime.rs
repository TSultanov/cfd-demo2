use crate::solver::gpu::csr::build_block_csr;
use crate::solver::gpu::init::linear_solver;
use crate::solver::gpu::modules::linear_system::LinearSystemPorts;
use crate::solver::gpu::modules::ports::PortSpace;
use crate::solver::gpu::modules::scalar_cg::ScalarCgModule;
use crate::solver::gpu::runtime_common::GpuRuntimeCommon;
use crate::solver::gpu::structs::LinearSolverStats;
use crate::solver::mesh::Mesh;

/// Generic CSR runtime sized by an arbitrary DOF count.
///
/// This keeps mesh resources at the cell level (`common.num_cells`) but allocates
/// the linear system for an expanded CSR over `num_dofs = num_cells * unknowns_per_cell`.
///
/// This is the bridge needed to make `unknowns_per_cell` fully model-driven for the
/// generic coupled path without relying on specialized kernels.
pub(crate) struct GpuCsrRuntime {
    pub common: GpuRuntimeCommon,
    pub num_dofs: u32,
    pub num_nonzeros: u32,

    pub linear_ports: LinearSystemPorts,
    pub linear_port_space: PortSpace,

    pub scalar_cg: ScalarCgModule,
}

impl GpuCsrRuntime {
    pub async fn new(
        mesh: &Mesh,
        unknowns_per_cell: u32,
        device: Option<wgpu::Device>,
        queue: Option<wgpu::Queue>,
    ) -> Self {
        let common = GpuRuntimeCommon::new(mesh, device, queue).await;

        let num_dofs = common
            .num_cells
            .checked_mul(unknowns_per_cell)
            .expect("num_dofs overflow");

        let (row_offsets, col_indices) = build_block_csr(
            &common.mesh.scalar_row_offsets,
            &common.mesh.scalar_col_indices,
            unknowns_per_cell,
        );

        let cg = linear_solver::init_scalar_cg(
            &common.context.device,
            num_dofs,
            &row_offsets,
            &col_indices,
        );

        Self {
            common,
            num_dofs,
            num_nonzeros: cg.num_nonzeros,
            linear_ports: cg.ports,
            linear_port_space: cg.port_space,
            scalar_cg: cg.scalar_cg,
        }
    }

    pub fn solve_linear_system_cg(&self, max_iters: u32, tol: f32) -> LinearSolverStats {
        self.scalar_cg
            .solve(&self.common.context, self.num_dofs, max_iters, tol)
    }

    pub fn set_linear_system(&self, matrix_values: &[f32], rhs: &[f32]) -> Result<(), String> {
        if matrix_values.len() != self.num_nonzeros as usize {
            return Err(format!(
                "matrix_values length {} does not match num_nonzeros {}",
                matrix_values.len(),
                self.num_nonzeros
            ));
        }
        if rhs.len() != self.num_dofs as usize {
            return Err(format!(
                "rhs length {} does not match num_dofs {}",
                rhs.len(),
                self.num_dofs
            ));
        }
        self.common.context.queue.write_buffer(
            self.linear_port_space.buffer(self.linear_ports.values),
            0,
            bytemuck::cast_slice(matrix_values),
        );
        self.common.context.queue.write_buffer(
            self.linear_port_space.buffer(self.linear_ports.rhs),
            0,
            bytemuck::cast_slice(rhs),
        );
        Ok(())
    }
}
