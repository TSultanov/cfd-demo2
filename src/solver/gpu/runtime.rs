use crate::solver::gpu::init::linear_solver;
use crate::solver::gpu::csr::build_block_csr;
use crate::solver::gpu::modules::constants::ConstantsModule;
use crate::solver::gpu::modules::linear_system::LinearSystemPorts;
use crate::solver::gpu::modules::ports::PortSpace;
use crate::solver::gpu::modules::scalar_cg::ScalarCgModule;
use crate::solver::gpu::modules::time_integration::TimeIntegrationModule;
use crate::solver::gpu::runtime_common::GpuRuntimeCommon;
use crate::solver::gpu::structs::{GpuConstants, LinearSolverStats};
use crate::solver::mesh::Mesh;

pub(crate) struct GpuScalarRuntime {
    pub common: GpuRuntimeCommon,
    pub num_nonzeros: u32,

    pub constants: ConstantsModule,

    pub linear_ports: LinearSystemPorts,
    pub linear_port_space: PortSpace,

    pub scalar_cg: ScalarCgModule,

    pub time_integration: TimeIntegrationModule,
}

/// Generic CSR runtime sized by an arbitrary DOF count.
///
/// This keeps mesh resources at the cell level (`common.num_cells`) but allocates
/// the linear system for an expanded CSR over `num_dofs = num_cells * unknowns_per_cell`.
///
/// This is the bridge needed to make `unknowns_per_cell` fully model-driven for the
/// generic coupled path without relying on specialized kernels.
pub(crate) struct GpuCsrRuntime {
    pub common: GpuRuntimeCommon,
    pub unknowns_per_cell: u32,
    pub num_dofs: u32,
    pub num_nonzeros: u32,

    pub constants: ConstantsModule,

    pub linear_ports: LinearSystemPorts,
    pub linear_port_space: PortSpace,

    pub scalar_cg: ScalarCgModule,

    pub time_integration: TimeIntegrationModule,
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

        let (row_offsets, col_indices) =
            build_block_csr(&common.mesh.row_offsets, &common.mesh.col_indices, unknowns_per_cell);

        let cg = linear_solver::init_scalar_cg(
            &common.context.device,
            num_dofs,
            &row_offsets,
            &col_indices,
        );

        let constants = ConstantsModule::new(
            &common.context.device,
            default_constants(),
            "CSR Runtime Constants Buffer",
        );

        Self {
            common,
            unknowns_per_cell,
            num_dofs,
            num_nonzeros: cg.num_nonzeros,
            constants,
            linear_ports: cg.ports,
            linear_port_space: cg.port_space,
            scalar_cg: cg.scalar_cg,
            time_integration: TimeIntegrationModule::new(),
        }
    }

    pub fn update_constants(&self) {
        self.constants.write(&self.common.context.queue);
    }

    pub fn set_dt(&mut self, dt: f32) {
        self.time_integration
            .set_dt(dt, &mut self.constants, &self.common.context.queue);
    }

    pub fn set_scheme(&mut self, scheme: u32) {
        {
            let values = self.constants.values_mut();
            values.scheme = scheme;
        }
        self.update_constants();
    }

    pub fn set_time_scheme(&mut self, scheme: u32) {
        {
            let values = self.constants.values_mut();
            values.time_scheme = scheme;
        }
        self.update_constants();
    }

    pub fn advance_time(&mut self) {
        self.time_integration
            .prepare_step(&mut self.constants, &self.common.context.queue);
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

impl GpuScalarRuntime {
    pub async fn new(
        mesh: &Mesh,
        device: Option<wgpu::Device>,
        queue: Option<wgpu::Queue>,
    ) -> Self {
        let common = GpuRuntimeCommon::new(mesh, device, queue).await;

        let cg = linear_solver::init_scalar_cg(
            &common.context.device,
            common.num_cells,
            &common.mesh.row_offsets,
            &common.mesh.col_indices,
        );

        let constants = ConstantsModule::new(
            &common.context.device,
            default_constants(),
            "Scalar Runtime Constants Buffer",
        );

        Self {
            common,
            num_nonzeros: cg.num_nonzeros,
            constants,
            linear_ports: cg.ports,
            linear_port_space: cg.port_space,
            scalar_cg: cg.scalar_cg,
            time_integration: TimeIntegrationModule::new(),
        }
    }

    pub fn update_constants(&self) {
        self.constants.write(&self.common.context.queue);
    }

    pub fn set_dt(&mut self, dt: f32) {
        self.time_integration
            .set_dt(dt, &mut self.constants, &self.common.context.queue);
    }

    pub fn set_scheme(&mut self, scheme: u32) {
        {
            let values = self.constants.values_mut();
            values.scheme = scheme;
        }
        self.update_constants();
    }

    pub fn set_time_scheme(&mut self, scheme: u32) {
        {
            let values = self.constants.values_mut();
            values.time_scheme = scheme;
        }
        self.update_constants();
    }

    pub fn advance_time(&mut self) {
        self.time_integration
            .prepare_step(&mut self.constants, &self.common.context.queue);
    }

    pub fn solve_linear_system_cg_with_size(
        &self,
        n: u32,
        max_iters: u32,
        tol: f32,
    ) -> LinearSolverStats {
        self.scalar_cg
            .solve(&self.common.context, n, max_iters, tol)
    }

    pub fn solve_linear_system_cg(&self, max_iters: u32, tol: f32) -> LinearSolverStats {
        self.solve_linear_system_cg_with_size(self.common.num_cells, max_iters, tol)
    }

    pub fn set_linear_system(&self, matrix_values: &[f32], rhs: &[f32]) -> Result<(), String> {
        if matrix_values.len() != self.num_nonzeros as usize {
            return Err(format!(
                "matrix_values length {} does not match num_nonzeros {}",
                matrix_values.len(),
                self.num_nonzeros
            ));
        }
        if rhs.len() != self.common.num_cells as usize {
            return Err(format!(
                "rhs length {} does not match num_cells {}",
                rhs.len(),
                self.common.num_cells
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

    pub async fn get_linear_solution(&self, n: u32) -> Result<Vec<f32>, String> {
        if n != self.common.num_cells {
            return Err(format!(
                "requested solution size {} does not match num_cells {}",
                n, self.common.num_cells
            ));
        }
        let raw = self
            .read_buffer(
                self.linear_port_space.buffer(self.linear_ports.x),
                (n as u64) * 4,
            )
            .await;
        Ok(bytemuck::cast_slice(&raw).to_vec())
    }

    pub async fn read_buffer(&self, buffer: &wgpu::Buffer, size: u64) -> Vec<u8> {
        self.common
            .read_buffer(buffer, size, "Scalar Runtime Staging Buffer (cached)")
            .await
    }
}

fn default_constants() -> GpuConstants {
    GpuConstants {
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
    }
}
