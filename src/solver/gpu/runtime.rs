use crate::solver::gpu::init::linear_solver;
use crate::solver::gpu::modules::linear_system::LinearSystemPorts;
use crate::solver::gpu::modules::ports::PortSpace;
use crate::solver::gpu::modules::scalar_cg::ScalarCgModule;
use crate::solver::gpu::runtime_common::GpuRuntimeCommon;
use crate::solver::gpu::structs::{GpuConstants, LinearSolverStats};
use crate::solver::mesh::Mesh;
use wgpu::util::DeviceExt;

pub(crate) struct GpuScalarRuntime {
    pub common: GpuRuntimeCommon,
    pub num_nonzeros: u32,

    pub b_constants: wgpu::Buffer,
    pub constants: GpuConstants,

    pub linear_ports: LinearSystemPorts,
    pub linear_port_space: PortSpace,

    pub scalar_cg: ScalarCgModule,
}

impl GpuScalarRuntime {
    pub async fn new(mesh: &Mesh, device: Option<wgpu::Device>, queue: Option<wgpu::Queue>) -> Self {
        let common = GpuRuntimeCommon::new(mesh, device, queue).await;

        let cg = linear_solver::init_scalar_cg(
            &common.context.device,
            common.num_cells,
            &common.mesh.row_offsets,
            &common.mesh.col_indices,
        );

        let constants = default_constants();
        let b_constants = common
            .context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Constants Buffer"),
            contents: bytemuck::bytes_of(&constants),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            common,
            num_nonzeros: cg.num_nonzeros,
            b_constants,
            constants,
            linear_ports: cg.ports,
            linear_port_space: cg.port_space,
            scalar_cg: cg.scalar_cg,
        }
    }

    pub fn update_constants(&self) {
        self.common
            .context
            .queue
            .write_buffer(&self.b_constants, 0, bytemuck::bytes_of(&self.constants));
    }

    pub fn set_dt(&mut self, dt: f32) {
        if self.constants.time <= 0.0 {
            self.constants.dt_old = dt;
        } else {
            self.constants.dt_old = self.constants.dt;
        }
        self.constants.dt = dt;
        self.update_constants();
    }

    pub fn set_scheme(&mut self, scheme: u32) {
        self.constants.scheme = scheme;
        self.update_constants();
    }

    pub fn set_time_scheme(&mut self, scheme: u32) {
        self.constants.time_scheme = scheme;
        self.update_constants();
    }

    pub fn advance_time(&mut self) {
        self.constants.time += self.constants.dt;
        self.update_constants();
    }

    pub fn solve_linear_system_cg_with_size(&self, n: u32, max_iters: u32, tol: f32) -> LinearSolverStats {
        self.scalar_cg.solve(&self.common.context, n, max_iters, tol)
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
