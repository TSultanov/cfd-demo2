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
/// Keeps mesh resources at the cell level (`common.num_cells`) but allocates the
/// linear system for an expanded CSR over `num_dofs = num_cells * unknowns_per_cell`,
/// making `unknowns_per_cell` fully model-driven without specialized kernels.
pub(crate) struct GpuCsrRuntime {
    pub common: GpuRuntimeCommon,
    pub num_dofs: u32,
    pub num_nonzeros: u32,

    pub linear_ports: LinearSystemPorts,
    pub linear_port_space: PortSpace,

    pub scalar_cg: Option<ScalarCgModule>,
}

impl GpuCsrRuntime {
    pub async fn new(
        mesh: &Mesh,
        unknowns_per_cell: u32,
        device: Option<wgpu::Device>,
        queue: Option<wgpu::Queue>,
        capacity: crate::solver::gpu::capacity::CapacityPlan,
    ) -> Result<Self, String> {
        let common = GpuRuntimeCommon::new(mesh, device, queue, capacity).await?;

        let num_dofs = common
            .num_cells
            .checked_mul(unknowns_per_cell)
            .ok_or_else(|| format!("num_dofs overflow: {} * {}", common.num_cells, unknowns_per_cell))?;

        let (row_offsets, col_indices) = build_block_csr(
            &common.mesh.scalar_row_offsets,
            &common.mesh.scalar_col_indices,
            unknowns_per_cell,
        );

        let cg = linear_solver::init_scalar_cg(
            &common.context.device,
            &common.context.pipeline_cache,
            num_dofs,
            &row_offsets,
            &col_indices,
            capacity,
        )?;

        Ok(Self {
            common,
            num_dofs,
            num_nonzeros: cg.num_nonzeros,
            linear_ports: cg.ports,
            linear_port_space: cg.port_space,
            scalar_cg: Some(cg.scalar_cg),
        })
    }

    /// Matrix-free runtime for explicit schemes. Mesh resources are complete,
    /// while the linear port space carries only the stage RHS plus tiny dummy
    /// bindings for the implicit-only names. This avoids allocating block CSR,
    /// scalar-CG, and their O(dof) work vectors for an RK4 plan that never
    /// dispatches a linear kernel.
    pub async fn new_explicit(
        mesh: &Mesh,
        unknowns_per_cell: u32,
        device: Option<wgpu::Device>,
        queue: Option<wgpu::Queue>,
        capacity: crate::solver::gpu::capacity::CapacityPlan,
    ) -> Result<Self, String> {
        let common = GpuRuntimeCommon::new(mesh, device, queue, capacity).await?;
        let num_dofs = common
            .num_cells
            .checked_mul(unknowns_per_cell)
            .ok_or_else(|| {
                format!(
                    "num_dofs overflow: {} * {}",
                    common.num_cells, unknowns_per_cell
                )
            })?;
        let mut linear_port_space = PortSpace::new();
        let linear_ports = {
            let mut lowerer = linear_port_space.lowerer(&common.context.device);
            let storage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST;
            let row_offsets = lowerer.buffer_u32_init(
                "linear:row_offsets",
                &[0],
                storage,
                "Explicit dummy row offsets",
            );
            let col_indices = lowerer.buffer_u32_init(
                "linear:col_indices",
                &[0],
                storage,
                "Explicit dummy column indices",
            );
            let values = lowerer.buffer_f32(
                "linear:matrix_values",
                4,
                storage,
                "Explicit dummy matrix values",
            );
            let rhs = lowerer.buffer_f32(
                "linear:rhs",
                (num_dofs.max(1) as u64) * 4,
                storage,
                "Explicit residual RHS",
            );
            let x = lowerer.buffer_f32(
                "linear:x",
                4,
                storage,
                "Explicit dummy x",
            );
            LinearSystemPorts {
                row_offsets,
                col_indices,
                values,
                rhs,
                x,
            }
        };

        Ok(Self {
            common,
            num_dofs,
            num_nonzeros: 0,
            linear_ports,
            linear_port_space,
            scalar_cg: None,
        })
    }

    /// Rebuild every mesh-topology-derived resource in place for a new mesh with
    /// the SAME cell count (the invariant) but a possibly different face set /
    /// adjacency / nnz.
    ///
    /// Refreshes the mesh buffers + host CSR ([`MeshResources::refresh_topology`]),
    /// updates `num_faces`, and rebuilds the block-expanded CSR + scalar-CG linear
    /// system over it. `num_dofs` is invariant (cells × unknowns). The scalar-CG
    /// module, its bind groups, and the linear port space are replaced with fresh
    /// ones (the block CSR buffers changed size), so callers holding bind groups
    /// over the linear system (FGMRES, Schur, generated kernels) MUST rebuild them
    /// afterward.
    pub fn refresh_topology(
        &mut self,
        mesh: &Mesh,
        unknowns_per_cell: u32,
    ) -> Result<(), String> {
        let device = self.common.context.device.clone();
        self.common.mesh.refresh_topology(&device, mesh)?;
        self.common.num_faces = mesh.face_owner.len() as u32;

        // Explicit runtimes retain their cell-sized RHS/x and dummy linear
        // bindings; only mesh topology/face resources need refreshing.
        if self.scalar_cg.is_none() {
            return Ok(());
        }

        let (row_offsets, col_indices) = build_block_csr(
            &self.common.mesh.scalar_row_offsets,
            &self.common.mesh.scalar_col_indices,
            unknowns_per_cell,
        );

        let cg = linear_solver::init_scalar_cg(
            &device,
            &self.common.context.pipeline_cache,
            self.num_dofs,
            &row_offsets,
            &col_indices,
            self.common.mesh.capacity,
        )?;

        self.num_nonzeros = cg.num_nonzeros;
        self.linear_ports = cg.ports;
        self.linear_port_space = cg.port_space;
        self.scalar_cg = Some(cg.scalar_cg);
        Ok(())
    }

    pub fn solve_linear_system_cg(&self, max_iters: u32, tol: f32) -> LinearSolverStats {
        self.scalar_cg
            .as_ref()
            .expect("linear solve requested from matrix-free explicit runtime")
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
