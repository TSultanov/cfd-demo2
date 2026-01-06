use crate::solver::gpu::bindings::generated::{
    compressible_apply as generated_apply, compressible_assembly as generated_assembly,
    compressible_gradients as generated_gradients, compressible_update as generated_update,
};
use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::compressible_fgmres::CompressibleFgmresResources;
use crate::solver::gpu::init::compressible_fields::{
    create_compressible_field_bind_groups, init_compressible_field_buffers, PackedStateConfig,
};
use crate::solver::gpu::init::linear_solver::matrix;
use crate::solver::gpu::init::mesh;
use crate::solver::gpu::structs::{GpuConstants, LinearSolverStats};
use crate::solver::mesh::Mesh;
use crate::solver::model::compressible_model;
use bytemuck::cast_slice;
use std::env;
use wgpu::util::DeviceExt;

#[derive(Clone, Copy, Debug)]
struct CompressibleOffsets {
    stride: u32,
    rho: u32,
    rho_u: u32,
    rho_e: u32,
    p: u32,
    u: u32,
}

#[derive(Clone, Copy, Debug, Default)]
struct CompressibleProfile {
    enabled: bool,
    stride: usize,
    step: usize,
    accum_steps: usize,
    accum_total: f64,
    accum_fgmres: f64,
    accum_grad: f64,
    accum_assembly: f64,
    accum_apply: f64,
    accum_update: f64,
    accum_iters: u64,
}

impl CompressibleProfile {
    fn new() -> Self {
        let enabled = env_flag("CFD2_COMP_PROFILE", false);
        let stride = env_usize("CFD2_COMP_PROFILE_STRIDE", 25).max(1);
        CompressibleProfile {
            enabled,
            stride,
            ..Default::default()
        }
    }

    fn record(
        &mut self,
        total: f64,
        grad: f64,
        assembly: f64,
        fgmres: f64,
        apply: f64,
        update: f64,
        iters: u64,
    ) {
        if !self.enabled {
            return;
        }
        self.step += 1;
        self.accum_steps += 1;
        self.accum_total += total;
        self.accum_grad += grad;
        self.accum_assembly += assembly;
        self.accum_fgmres += fgmres;
        self.accum_apply += apply;
        self.accum_update += update;
        self.accum_iters += iters;

        if self.accum_steps >= self.stride {
            let steps = self.accum_steps as f64;
            let avg_total = self.accum_total / steps;
            let avg_fgmres = self.accum_fgmres / steps;
            let avg_grad = self.accum_grad / steps;
            let avg_assembly = self.accum_assembly / steps;
            let avg_apply = self.accum_apply / steps;
            let avg_update = self.accum_update / steps;
            let avg_iters = self.accum_iters as f64 / steps;
            let fgmres_pct = if avg_total > 0.0 {
                100.0 * avg_fgmres / avg_total
            } else {
                0.0
            };
            println!(
                "compressible_profile step {}..{} avg_total={:.3}s fgmres={:.3}s ({:.1}%) iters={:.1} grad={:.3}s assembly={:.3}s apply={:.3}s update={:.3}s",
                self.step + 1 - self.accum_steps,
                self.step,
                avg_total,
                avg_fgmres,
                fgmres_pct,
                avg_iters,
                avg_grad,
                avg_assembly,
                avg_apply,
                avg_update
            );
            self.accum_steps = 0;
            self.accum_total = 0.0;
            self.accum_fgmres = 0.0;
            self.accum_grad = 0.0;
            self.accum_assembly = 0.0;
            self.accum_apply = 0.0;
            self.accum_update = 0.0;
            self.accum_iters = 0;
        }
    }
}

pub struct GpuCompressibleSolver {
    pub context: GpuContext,
    pub num_cells: u32,
    pub num_faces: u32,
    pub num_unknowns: u32,
    pub state_step_index: usize,
    pub state_buffers: Vec<wgpu::Buffer>,
    pub bg_fields_ping_pong: Vec<wgpu::BindGroup>,
    pub bg_fields: wgpu::BindGroup,
    pub bg_apply_fields_ping_pong: Vec<wgpu::BindGroup>,
    pub bg_apply_fields: wgpu::BindGroup,
    pub b_state: wgpu::Buffer,
    pub b_state_old: wgpu::Buffer,
    pub b_state_old_old: wgpu::Buffer,
    pub b_state_iter: wgpu::Buffer,
    pub b_fluxes: wgpu::Buffer,
    pub b_grad_rho: wgpu::Buffer,
    pub b_grad_rho_u_x: wgpu::Buffer,
    pub b_grad_rho_u_y: wgpu::Buffer,
    pub b_grad_rho_e: wgpu::Buffer,
    pub b_constants: wgpu::Buffer,
    pub constants: GpuConstants,
    pub b_row_offsets: wgpu::Buffer,
    pub b_col_indices: wgpu::Buffer,
    pub b_matrix_values: wgpu::Buffer,
    pub b_rhs: wgpu::Buffer,
    pub b_x: wgpu::Buffer,
    pub b_scalar_row_offsets: wgpu::Buffer,
    pub bg_mesh: wgpu::BindGroup,
    pub bg_solver: wgpu::BindGroup,
    pub bg_apply_solver: wgpu::BindGroup,
    pub pipeline_assembly: wgpu::ComputePipeline,
    pub pipeline_apply: wgpu::ComputePipeline,
    pub pipeline_gradients: wgpu::ComputePipeline,
    pub pipeline_update: wgpu::ComputePipeline,
    pub fgmres_resources: Option<CompressibleFgmresResources>,
    pub outer_iters: usize,
    pub nonconverged_relax: f32,
    pub(crate) scalar_row_offsets: Vec<u32>,
    pub(crate) scalar_col_indices: Vec<u32>,
    pub(crate) block_row_offsets: Vec<u32>,
    pub(crate) block_col_indices: Vec<u32>,
    profile: CompressibleProfile,
    offsets: CompressibleOffsets,
}

impl GpuCompressibleSolver {
    pub async fn new(
        mesh: &Mesh,
        device: Option<wgpu::Device>,
        queue: Option<wgpu::Queue>,
    ) -> Self {
        let context = GpuContext::new(device, queue).await;
        let num_cells = mesh.cell_cx.len() as u32;
        let num_faces = mesh.face_owner.len() as u32;
        let num_unknowns = num_cells * 4;
        let profile = CompressibleProfile::new();

        let model = compressible_model();
        let layout = &model.state_layout;
        let offsets = CompressibleOffsets {
            stride: layout.stride(),
            rho: layout.offset_for("rho").expect("rho offset missing"),
            rho_u: layout.offset_for("rho_u").expect("rho_u offset missing"),
            rho_e: layout.offset_for("rho_e").expect("rho_e offset missing"),
            p: layout.offset_for("p").expect("p offset missing"),
            u: layout.offset_for("u").expect("u offset missing"),
        };

        let mesh_res = mesh::init_mesh(&context.device, mesh);

        let field_buffers = init_compressible_field_buffers(
            &context.device,
            num_cells,
            num_faces,
            PackedStateConfig {
                state_stride: offsets.stride,
                flux_stride: 4,
            },
        );

        let pipeline_assembly =
            generated_assembly::compute::create_main_pipeline_embed_source(&context.device);
        let pipeline_apply =
            generated_apply::compute::create_main_pipeline_embed_source(&context.device);
        let pipeline_gradients =
            generated_gradients::compute::create_main_pipeline_embed_source(&context.device);
        let pipeline_update =
            generated_update::compute::create_main_pipeline_embed_source(&context.device);

        let mesh_layout = context.device.create_bind_group_layout(
            &generated_assembly::WgpuBindGroup0::LAYOUT_DESCRIPTOR,
        );
        let bg_mesh = context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compressible Mesh Bind Group"),
            layout: &mesh_layout,
            entries: &generated_assembly::WgpuBindGroup0Entries::new(
                generated_assembly::WgpuBindGroup0EntriesParams {
                    face_owner: mesh_res.b_face_owner.as_entire_buffer_binding(),
                    face_neighbor: mesh_res.b_face_neighbor.as_entire_buffer_binding(),
                    face_areas: mesh_res.b_face_areas.as_entire_buffer_binding(),
                    face_normals: mesh_res.b_face_normals.as_entire_buffer_binding(),
                    cell_centers: mesh_res.b_cell_centers.as_entire_buffer_binding(),
                    cell_vols: mesh_res.b_cell_vols.as_entire_buffer_binding(),
                    cell_face_offsets: mesh_res.b_cell_face_offsets.as_entire_buffer_binding(),
                    cell_faces: mesh_res.b_cell_faces.as_entire_buffer_binding(),
                    cell_face_matrix_indices: mesh_res
                        .b_cell_face_matrix_indices
                        .as_entire_buffer_binding(),
                    diagonal_indices: mesh_res.b_diagonal_indices.as_entire_buffer_binding(),
                    face_boundary: mesh_res.b_face_boundary.as_entire_buffer_binding(),
                    face_centers: mesh_res.b_face_centers.as_entire_buffer_binding(),
                },
            )
            .into_array(),
        });

        let fields_layout = context.device.create_bind_group_layout(
            &generated_assembly::WgpuBindGroup1::LAYOUT_DESCRIPTOR,
        );
        let fields_res =
            create_compressible_field_bind_groups(&context.device, field_buffers, &fields_layout);

        let b_scalar_row_offsets =
            context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Compressible Scalar Row Offsets"),
                    contents: cast_slice(&mesh_res.row_offsets),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let scalar_row_offsets = mesh_res.row_offsets.clone();
        let scalar_col_indices = mesh_res.col_indices.clone();
        let (row_offsets, col_indices) =
            build_block_csr(&mesh_res.row_offsets, &mesh_res.col_indices, 4);
        let block_row_offsets = row_offsets.clone();
        let block_col_indices = col_indices.clone();
        let matrix_res = matrix::init_matrix(&context.device, &row_offsets, &col_indices);
        let b_rhs = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compressible RHS"),
            size: num_unknowns as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let b_x = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compressible Solution"),
            size: num_unknowns as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let solver_layout = context.device.create_bind_group_layout(
            &generated_assembly::WgpuBindGroup2::LAYOUT_DESCRIPTOR,
        );
        let bg_solver = context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compressible Solver Bind Group"),
            layout: &solver_layout,
            entries: &generated_assembly::WgpuBindGroup2Entries::new(
                generated_assembly::WgpuBindGroup2EntriesParams {
                    matrix_values: matrix_res.b_matrix_values.as_entire_buffer_binding(),
                    rhs: b_rhs.as_entire_buffer_binding(),
                    scalar_row_offsets: b_scalar_row_offsets.as_entire_buffer_binding(),
                },
            )
            .into_array(),
        });

        let apply_fields_layout = context.device.create_bind_group_layout(
            &generated_apply::WgpuBindGroup0::LAYOUT_DESCRIPTOR,
        );
        let mut bg_apply_fields_ping_pong = Vec::new();
        for i in 0..3 {
            let (idx_state, idx_old, idx_old_old) = match i {
                0 => (0, 1, 2),
                1 => (2, 0, 1),
                2 => (1, 2, 0),
                _ => (0, 1, 2),
            };
            let bg = context.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Compressible Apply Fields Bind Group {}", i)),
                layout: &apply_fields_layout,
                entries: &generated_apply::WgpuBindGroup0Entries::new(
                    generated_apply::WgpuBindGroup0EntriesParams {
                        state: fields_res.state_buffers[idx_state].as_entire_buffer_binding(),
                        state_old: fields_res.state_buffers[idx_old].as_entire_buffer_binding(),
                        state_old_old: fields_res.state_buffers[idx_old_old]
                            .as_entire_buffer_binding(),
                        state_iter: fields_res.b_state_iter.as_entire_buffer_binding(),
                        fluxes: fields_res.b_fluxes.as_entire_buffer_binding(),
                        constants: fields_res.b_constants.as_entire_buffer_binding(),
                        grad_rho: fields_res.b_grad_rho.as_entire_buffer_binding(),
                        grad_rho_u_x: fields_res.b_grad_rho_u_x.as_entire_buffer_binding(),
                        grad_rho_u_y: fields_res.b_grad_rho_u_y.as_entire_buffer_binding(),
                        grad_rho_e: fields_res.b_grad_rho_e.as_entire_buffer_binding(),
                    },
                )
                .into_array(),
            });
            bg_apply_fields_ping_pong.push(bg);
        }
        let bg_apply_fields = bg_apply_fields_ping_pong[0].clone();
        let apply_solver_layout = context.device.create_bind_group_layout(
            &generated_apply::WgpuBindGroup1::LAYOUT_DESCRIPTOR,
        );
        let bg_apply_solver = context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compressible Apply Solver Bind Group"),
            layout: &apply_solver_layout,
            entries: &generated_apply::WgpuBindGroup1Entries::new(
                generated_apply::WgpuBindGroup1EntriesParams {
                    solution: b_x.as_entire_buffer_binding(),
                },
            )
            .into_array(),
        });

        Self {
            context,
            num_cells,
            num_faces,
            num_unknowns,
            state_step_index: 0,
            state_buffers: fields_res.state_buffers,
            bg_fields_ping_pong: fields_res.bg_fields_ping_pong,
            bg_fields: fields_res.bg_fields,
            bg_apply_fields_ping_pong,
            bg_apply_fields,
            b_state: fields_res.b_state,
            b_state_old: fields_res.b_state_old,
            b_state_old_old: fields_res.b_state_old_old,
            b_state_iter: fields_res.b_state_iter,
            b_fluxes: fields_res.b_fluxes,
            b_grad_rho: fields_res.b_grad_rho,
            b_grad_rho_u_x: fields_res.b_grad_rho_u_x,
            b_grad_rho_u_y: fields_res.b_grad_rho_u_y,
            b_grad_rho_e: fields_res.b_grad_rho_e,
            b_constants: fields_res.b_constants,
            constants: fields_res.constants,
            b_row_offsets: matrix_res.b_row_offsets,
            b_col_indices: matrix_res.b_col_indices,
            b_matrix_values: matrix_res.b_matrix_values,
            b_rhs,
            b_x,
            b_scalar_row_offsets,
            bg_mesh,
            bg_solver,
            bg_apply_solver,
            pipeline_assembly,
            pipeline_apply,
            pipeline_gradients,
            pipeline_update,
            fgmres_resources: None,
            outer_iters: 1,
            nonconverged_relax: 0.5,
            scalar_row_offsets,
            scalar_col_indices,
            block_row_offsets,
            block_col_indices,
            profile,
            offsets,
        }
    }

    pub fn initialize_history(&self) {
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Compressible Initialize History Encoder"),
                });
        let state_size = self.num_cells as u64 * self.offsets.stride as u64 * 4;
        encoder.copy_buffer_to_buffer(&self.b_state, 0, &self.b_state_old, 0, state_size);
        encoder.copy_buffer_to_buffer(&self.b_state, 0, &self.b_state_old_old, 0, state_size);
        self.context.queue.submit(Some(encoder.finish()));
    }

    pub fn set_dt(&mut self, dt: f32) {
        if self.constants.dt > 0.0 {
            self.constants.dt_old = self.constants.dt;
        } else {
            self.constants.dt_old = dt;
        }
        self.constants.dt = dt;
        self.update_constants();
    }

    pub fn set_dtau(&mut self, dtau: f32) {
        self.constants.dtau = dtau;
        self.update_constants();
    }

    pub fn set_viscosity(&mut self, mu: f32) {
        self.constants.viscosity = mu;
        self.update_constants();
    }

    pub fn set_time_scheme(&mut self, scheme: u32) {
        self.constants.time_scheme = scheme;
        self.update_constants();
    }

    pub fn set_inlet_velocity(&mut self, velocity: f32) {
        self.constants.inlet_velocity = velocity;
        self.update_constants();
    }

    pub fn set_scheme(&mut self, scheme: u32) {
        self.constants.scheme = scheme;
        self.update_constants();
    }

    pub fn set_alpha_u(&mut self, alpha_u: f32) {
        self.constants.alpha_u = alpha_u;
        self.update_constants();
    }

    pub fn set_precond_type(&mut self, precond_type: u32) {
        self.constants.precond_type = precond_type;
        self.update_constants();
    }

    pub fn set_precond_model(&mut self, model: u32) {
        self.constants.precond_model = model;
        self.update_constants();
    }

    pub fn set_precond_theta_floor(&mut self, floor: f32) {
        self.constants.precond_theta_floor = floor;
        self.update_constants();
    }

    pub fn set_outer_iters(&mut self, iters: usize) {
        self.outer_iters = iters.max(1);
    }

    pub fn set_nonconverged_relax(&mut self, relax: f32) {
        self.nonconverged_relax = relax.max(0.0);
    }

    pub fn set_uniform_state(&self, rho: f32, u: [f32; 2], p: f32) {
        let gamma = 1.4f32;
        let ke = 0.5 * rho * (u[0] * u[0] + u[1] * u[1]);
        let rho_e = p / (gamma - 1.0) + ke;

        let mut state = vec![0.0f32; self.num_cells as usize * self.offsets.stride as usize];
        for cell in 0..self.num_cells as usize {
            let base = cell * self.offsets.stride as usize;
            state[base + self.offsets.rho as usize] = rho;
            state[base + self.offsets.rho_u as usize] = rho * u[0];
            state[base + self.offsets.rho_u as usize + 1] = rho * u[1];
            state[base + self.offsets.rho_e as usize] = rho_e;
            state[base + self.offsets.p as usize] = p;
            state[base + self.offsets.u as usize] = u[0];
            state[base + self.offsets.u as usize + 1] = u[1];
        }

        self.write_state_all(&state);
    }

    pub fn set_state_fields(&self, rho: &[f32], u: &[[f32; 2]], p: &[f32]) {
        assert_eq!(rho.len(), self.num_cells as usize);
        assert_eq!(u.len(), self.num_cells as usize);
        assert_eq!(p.len(), self.num_cells as usize);

        let gamma = 1.4f32;
        let mut state = vec![0.0f32; self.num_cells as usize * self.offsets.stride as usize];

        for cell in 0..self.num_cells as usize {
            let base = cell * self.offsets.stride as usize;
            let rho_val = rho[cell];
            let u_val = u[cell];
            let p_val = p[cell];
            let ke = 0.5 * rho_val * (u_val[0] * u_val[0] + u_val[1] * u_val[1]);
            let rho_e = p_val / (gamma - 1.0) + ke;

            state[base + self.offsets.rho as usize] = rho_val;
            state[base + self.offsets.rho_u as usize] = rho_val * u_val[0];
            state[base + self.offsets.rho_u as usize + 1] = rho_val * u_val[1];
            state[base + self.offsets.rho_e as usize] = rho_e;
            state[base + self.offsets.p as usize] = p_val;
            state[base + self.offsets.u as usize] = u_val[0];
            state[base + self.offsets.u as usize + 1] = u_val[1];
        }

        self.write_state_all(&state);
    }

    pub fn step(&mut self) {
        let _ = self.step_with_stats();
    }

    pub fn step_with_stats(&mut self) -> Vec<LinearSolverStats> {
        let step_start = std::time::Instant::now();
        let workgroup_size = 64;
        let num_groups_cells = self.num_cells.div_ceil(workgroup_size);

        self.state_step_index = (self.state_step_index + 1) % 3;
        self.bg_fields = self.bg_fields_ping_pong[self.state_step_index].clone();
        self.bg_apply_fields = self.bg_apply_fields_ping_pong[self.state_step_index].clone();

        let (idx_state, idx_old, idx_old_old) = ping_pong_indices(self.state_step_index);
        self.b_state = self.state_buffers[idx_state].clone();
        self.b_state_old = self.state_buffers[idx_old].clone();
        self.b_state_old_old = self.state_buffers[idx_old_old].clone();

        self.constants.time += self.constants.dt;
        self.update_constants();

        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Compressible Step Encoder"),
                });
        let state_size = self.num_cells as u64 * self.offsets.stride as u64 * 4;
        encoder.copy_buffer_to_buffer(&self.b_state_old, 0, &self.b_state, 0, state_size);
        encoder.copy_buffer_to_buffer(&self.b_state, 0, &self.b_state_iter, 0, state_size);

        self.context.queue.submit(Some(encoder.finish()));

        let base_alpha_u = self.constants.alpha_u;
        let mut stats = Vec::with_capacity(self.outer_iters);
        let tol_base = env_f32("CFD2_COMP_FGMRES_TOL", 1e-8);
        let warm_scale = env_f32("CFD2_COMP_FGMRES_WARM_SCALE", 100.0).max(1.0);
        let warm_iters = env_usize("CFD2_COMP_FGMRES_WARM_ITERS", 4);
        let retry_scale = env_f32("CFD2_COMP_FGMRES_RETRY_SCALE", 0.5)
            .clamp(0.0, 1.0);
        let max_restart = env_usize("CFD2_COMP_FGMRES_MAX_RESTART", 80).max(1);
        let retry_restart = env_usize("CFD2_COMP_FGMRES_RETRY_RESTART", 160).max(1);
        let mut grad_secs = 0.0f64;
        let mut assembly_secs = 0.0f64;
        let mut fgmres_secs = 0.0f64;
        let mut apply_secs = 0.0f64;
        let mut update_secs = 0.0f64;
        let mut fgmres_iters = 0u64;
        for outer_idx in 0..self.outer_iters {
            let stage_start = std::time::Instant::now();
            let mut encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Compressible Gradients Encoder"),
                    });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compressible Gradients Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipeline_gradients);
                cpass.set_bind_group(0, &self.bg_mesh, &[]);
                cpass.set_bind_group(1, &self.bg_fields, &[]);
                cpass.dispatch_workgroups(num_groups_cells, 1, 1);
            }

            self.context.queue.submit(Some(encoder.finish()));
            grad_secs += stage_start.elapsed().as_secs_f64();

            let stage_start = std::time::Instant::now();
            let mut encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Compressible Assembly Encoder"),
                    });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compressible Assembly Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipeline_assembly);
                cpass.set_bind_group(0, &self.bg_mesh, &[]);
                cpass.set_bind_group(1, &self.bg_fields, &[]);
                cpass.set_bind_group(2, &self.bg_solver, &[]);
                cpass.dispatch_workgroups(num_groups_cells, 1, 1);
            }

            self.context.queue.submit(Some(encoder.finish()));
            assembly_secs += stage_start.elapsed().as_secs_f64();

            let tol = if outer_idx < warm_iters {
                tol_base * warm_scale
            } else {
                tol_base
            };
            let retry_tol = (tol * retry_scale).min(tol_base);
            let mut iter_stats = self.solve_compressible_fgmres(max_restart, tol);
            fgmres_secs += iter_stats.time.as_secs_f64();
            fgmres_iters += iter_stats.iterations as u64;
            if !iter_stats.converged {
                let retry_stats = self.solve_compressible_fgmres(retry_restart, retry_tol);
                fgmres_secs += retry_stats.time.as_secs_f64();
                fgmres_iters += retry_stats.iterations as u64;
                if retry_stats.converged || retry_stats.residual < iter_stats.residual {
                    iter_stats = retry_stats;
                }
            }
            stats.push(iter_stats);

            let stage_start = std::time::Instant::now();
            let mut encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Compressible Iter Snapshot Encoder"),
                    });
            encoder.copy_buffer_to_buffer(&self.b_state, 0, &self.b_state_iter, 0, state_size);
            self.context.queue.submit(Some(encoder.finish()));

            let apply_alpha = if iter_stats.converged {
                base_alpha_u
            } else {
                base_alpha_u * self.nonconverged_relax
            };
            if (self.constants.alpha_u - apply_alpha).abs() > 1e-6 {
                self.constants.alpha_u = apply_alpha;
                self.update_constants();
            }

            let mut encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Compressible Apply Encoder"),
                    });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compressible Apply Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipeline_apply);
                cpass.set_bind_group(0, &self.bg_apply_fields, &[]);
                cpass.set_bind_group(1, &self.bg_apply_solver, &[]);
                cpass.dispatch_workgroups(num_groups_cells, 1, 1);
            }

            self.context.queue.submit(Some(encoder.finish()));
            apply_secs += stage_start.elapsed().as_secs_f64();

            if (self.constants.alpha_u - base_alpha_u).abs() > 1e-6 {
                self.constants.alpha_u = base_alpha_u;
                self.update_constants();
            }
        }

        let stage_start = std::time::Instant::now();
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Compressible Update Encoder"),
                });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compressible Update Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline_update);
            cpass.set_bind_group(0, &self.bg_fields, &[]);
            cpass.dispatch_workgroups(num_groups_cells, 1, 1);
        }

        self.context.queue.submit(Some(encoder.finish()));
        update_secs += stage_start.elapsed().as_secs_f64();
        let total_secs = step_start.elapsed().as_secs_f64();
        self.profile.record(
            total_secs,
            grad_secs,
            assembly_secs,
            fgmres_secs,
            apply_secs,
            update_secs,
            fgmres_iters,
        );
        stats
    }

    pub async fn get_rho(&self) -> Vec<f64> {
        let data = self.read_state().await;
        let stride = self.offsets.stride as usize;
        let offset = self.offsets.rho as usize;
        (0..self.num_cells as usize)
            .map(|i| data[i * stride + offset] as f64)
            .collect()
    }

    pub async fn get_u(&self) -> Vec<(f64, f64)> {
        let data = self.read_state().await;
        let stride = self.offsets.stride as usize;
        let offset = self.offsets.u as usize;
        (0..self.num_cells as usize)
            .map(|i| {
                let base = i * stride + offset;
                (data[base] as f64, data[base + 1] as f64)
            })
            .collect()
    }

    pub async fn get_p(&self) -> Vec<f64> {
        let data = self.read_state().await;
        let stride = self.offsets.stride as usize;
        let offset = self.offsets.p as usize;
        (0..self.num_cells as usize)
            .map(|i| data[i * stride + offset] as f64)
            .collect()
    }

    fn update_constants(&self) {
        self.context
            .queue
            .write_buffer(&self.b_constants, 0, bytemuck::bytes_of(&self.constants));
    }

    async fn read_state(&self) -> Vec<f32> {
        let byte_count = self.num_cells as u64 * self.offsets.stride as u64 * 4;
        let raw = self.read_buffer(&self.b_state, byte_count).await;
        cast_slice(&raw).to_vec()
    }

    pub(crate) async fn read_buffer(&self, buffer: &wgpu::Buffer, size: u64) -> Vec<u8> {
        let staging = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compressible Staging Buffer"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Compressible Readback Encoder"),
                });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.context.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = tx.send(v);
        });
        let _ = self
            .context
            .device
            .poll(wgpu::PollType::wait_indefinitely());
        rx.recv().ok().and_then(|v| v.ok()).unwrap();
        let data = slice.get_mapped_range().to_vec();
        staging.unmap();
        data
    }

    pub(crate) async fn read_buffer_f32(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<f32> {
        let raw = self.read_buffer(buffer, count as u64 * 4).await;
        cast_slice(&raw).to_vec()
    }

    fn write_state_all(&self, state: &[f32]) {
        let bytes = cast_slice(state);
        for buffer in &self.state_buffers {
            self.context.queue.write_buffer(buffer, 0, bytes);
        }
    }
}

fn env_flag(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .map(|val| {
            let val = val.to_ascii_lowercase();
            matches!(val.as_str(), "1" | "true" | "yes" | "y" | "on")
        })
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|val| val.parse().ok())
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    env::var(name)
        .ok()
        .and_then(|val| val.parse().ok())
        .unwrap_or(default)
}

fn ping_pong_indices(step_index: usize) -> (usize, usize, usize) {
    match step_index {
        0 => (0, 1, 2),
        1 => (2, 0, 1),
        2 => (1, 2, 0),
        _ => (0, 1, 2),
    }
}

fn build_block_csr(row_offsets: &[u32], col_indices: &[u32], block_size: u32) -> (Vec<u32>, Vec<u32>) {
    let num_cells = row_offsets.len().saturating_sub(1);
    let block_rows = num_cells * block_size as usize;
    let mut block_row_offsets = vec![0u32; block_rows + 1];
    let mut block_col_indices = Vec::new();
    let mut current_offset = 0u32;

    for cell in 0..num_cells {
        let start = row_offsets[cell] as usize;
        let end = row_offsets[cell + 1] as usize;
        let neighbors = &col_indices[start..end];
        for row in 0..block_size {
            block_row_offsets[cell * block_size as usize + row as usize] = current_offset;
            for &neighbor in neighbors {
                for col in 0..block_size {
                    block_col_indices.push(neighbor * block_size + col);
                }
            }
            current_offset += neighbors.len() as u32 * block_size;
        }
    }
    block_row_offsets[block_rows] = current_offset;

    (block_row_offsets, block_col_indices)
}
