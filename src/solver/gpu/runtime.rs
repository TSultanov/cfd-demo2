use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::init::{linear_solver, mesh as mesh_init, scalars};
use crate::solver::gpu::modules::scalar_cg::ScalarCgModule;
use crate::solver::gpu::readback::StagingBufferCache;
use crate::solver::gpu::structs::{GpuConstants, LinearSolverStats};
use crate::solver::mesh::Mesh;
use wgpu::util::DeviceExt;

pub(crate) struct GpuScalarRuntime {
    pub context: GpuContext,
    pub mesh: mesh_init::MeshResources,
    pub num_cells: u32,
    pub num_faces: u32,

    pub b_constants: wgpu::Buffer,
    pub constants: GpuConstants,

    pub b_row_offsets: wgpu::Buffer,

    pub scalar_cg: ScalarCgModule,

    readback_cache: StagingBufferCache,
}

impl GpuScalarRuntime {
    pub async fn new(mesh: &Mesh, device: Option<wgpu::Device>, queue: Option<wgpu::Queue>) -> Self {
        let context = GpuContext::new(device, queue).await;
        let num_cells = mesh.cell_cx.len() as u32;
        let num_faces = mesh.face_owner.len() as u32;

        let mesh_res = mesh_init::init_mesh(&context.device, mesh);

        let linear_res = linear_solver::init_linear_solver(
            &context.device,
            num_cells,
            &mesh_res.row_offsets,
            &mesh_res.col_indices,
        );

        let scalar_res = scalars::init_scalars(
            &context.device,
            &linear_res.b_scalars,
            &linear_res.b_dot_result,
            &linear_res.b_dot_result_2,
            &linear_res.b_solver_params,
        );

        let scalar_cg = ScalarCgModule::new(
            num_cells,
            &linear_res.b_rhs,
            &linear_res.b_x,
            &linear_res.b_matrix_values,
            &linear_res.b_r,
            &linear_res.b_r0,
            &linear_res.b_p_solver,
            &linear_res.b_v,
            &linear_res.b_s,
            &linear_res.b_t,
            &linear_res.b_dot_result,
            &linear_res.b_dot_result_2,
            &linear_res.b_scalars,
            &linear_res.b_solver_params,
            &linear_res.b_staging_scalar,
            &linear_res.bg_linear_matrix,
            &linear_res.bg_linear_state,
            &linear_res.bg_dot_params,
            &linear_res.bg_dot_p_v,
            &linear_res.bg_dot_r_r,
            &scalar_res.bg_scalars,
            &linear_res.bgl_dot_pair_inputs,
            &linear_res.pipeline_spmv_p_v,
            &linear_res.pipeline_dot,
            &linear_res.pipeline_dot_pair,
            &linear_res.pipeline_cg_update_x_r,
            &linear_res.pipeline_cg_update_p,
            &scalar_res.pipeline_init_cg_scalars,
            &scalar_res.pipeline_reduce_r0_v,
            &scalar_res.pipeline_reduce_rho_new_r_r,
            &context.device,
        );

        let constants = default_constants();
        let b_constants = context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Constants Buffer"),
            contents: bytemuck::bytes_of(&constants),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            context,
            mesh: mesh_res,
            num_cells,
            num_faces,
            b_constants,
            constants,
            b_row_offsets: linear_res.b_row_offsets,
            scalar_cg,
            readback_cache: Default::default(),
        }
    }

    pub fn update_constants(&self) {
        self.context
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
        self.scalar_cg.solve(&self.context, n, max_iters, tol)
    }

    pub fn solve_linear_system_cg(&self, max_iters: u32, tol: f32) -> LinearSolverStats {
        self.solve_linear_system_cg_with_size(self.num_cells, max_iters, tol)
    }

    pub async fn read_buffer(&self, buffer: &wgpu::Buffer, size: u64) -> Vec<u8> {
        let staging_buffer = self
            .readback_cache
            .take_or_create(&self.context.device, size, "Staging Buffer (cached)");

        let mut encoder = self
            .context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
        self.context.queue.submit(Some(encoder.finish()));

        let slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        let _ = self
            .context
            .device
            .poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result = data.to_vec();
        drop(data);
        staging_buffer.unmap();
        self.readback_cache.put(size, staging_buffer);
        result
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

