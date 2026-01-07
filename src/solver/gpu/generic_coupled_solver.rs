use crate::solver::gpu::bindings::generated::generic_coupled_assembly_generic_diffusion_demo as gen_assembly;
use crate::solver::gpu::bindings::generated::generic_coupled_update_generic_diffusion_demo as gen_update;
use crate::solver::gpu::structs::{GpuSolver, LinearSolverStats};
use crate::solver::model::ModelSpec;
use bytemuck::cast_slice;
use wgpu::util::DeviceExt;

fn ping_pong_indices(i: usize) -> (usize, usize, usize) {
    match i {
        0 => (0, 1, 2),
        1 => (2, 0, 1),
        2 => (1, 2, 0),
        _ => (0, 1, 2),
    }
}

pub struct GpuGenericCoupledSolver {
    pub linear: GpuSolver,
    model: ModelSpec,

    state_step_index: usize,
    state_buffers: Vec<wgpu::Buffer>,

    bg_mesh: wgpu::BindGroup,
    bg_fields_ping_pong: Vec<wgpu::BindGroup>,
    bg_solver: wgpu::BindGroup,
    bg_bc: wgpu::BindGroup,
    bg_update_state_ping_pong: Vec<wgpu::BindGroup>,
    bg_update_solution: wgpu::BindGroup,

    pipeline_assembly: wgpu::ComputePipeline,
    pipeline_update: wgpu::ComputePipeline,

    _b_bc_kind: wgpu::Buffer,
    _b_bc_value: wgpu::Buffer,
}

impl GpuGenericCoupledSolver {
    pub async fn new(
        mesh: &crate::solver::mesh::Mesh,
        model: ModelSpec,
        device: Option<wgpu::Device>,
        queue: Option<wgpu::Queue>,
    ) -> Result<Self, String> {
        let coupled_stride = model.system.unknowns_per_cell();
        if coupled_stride != 1 {
            return Err(format!(
                "GpuGenericCoupledSolver currently supports scalar systems only (unknowns_per_cell=1), got {}",
                coupled_stride
            ));
        }
        if model.id != "generic_diffusion_demo" {
            return Err(format!(
                "GpuGenericCoupledSolver currently supports model id 'generic_diffusion_demo' only, got '{}'",
                model.id
            ));
        }

        let linear = GpuSolver::new(mesh, device, queue).await;
        linear.update_constants();

        let device = &linear.context.device;

        let pipeline_assembly = gen_assembly::compute::create_main_pipeline_embed_source(device);
        let pipeline_update = gen_update::compute::create_main_pipeline_embed_source(device);

        let bg_mesh = {
            let bgl = device.create_bind_group_layout(&gen_assembly::WgpuBindGroup0::LAYOUT_DESCRIPTOR);
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GenericCoupled: mesh bind group"),
                layout: &bgl,
                entries: &gen_assembly::WgpuBindGroup0Entries::new(gen_assembly::WgpuBindGroup0EntriesParams {
                    face_owner: linear.b_face_owner.as_entire_buffer_binding(),
                    face_neighbor: linear.b_face_neighbor.as_entire_buffer_binding(),
                    face_areas: linear.b_face_areas.as_entire_buffer_binding(),
                    face_normals: linear.b_face_normals.as_entire_buffer_binding(),
                    face_centers: linear.b_face_centers.as_entire_buffer_binding(),
                    cell_centers: linear.b_cell_centers.as_entire_buffer_binding(),
                    cell_vols: linear.b_cell_vols.as_entire_buffer_binding(),
                    cell_face_offsets: linear.b_cell_face_offsets.as_entire_buffer_binding(),
                    cell_faces: linear.b_cell_faces.as_entire_buffer_binding(),
                    cell_face_matrix_indices: linear.b_cell_face_matrix_indices.as_entire_buffer_binding(),
                    diagonal_indices: linear.b_diagonal_indices.as_entire_buffer_binding(),
                    face_boundary: linear.b_face_boundary.as_entire_buffer_binding(),
                }).into_array(),
            })
        };

        let stride = model.state_layout.stride() as usize;
        let num_cells = linear.num_cells as usize;
        let zero_state = vec![0.0f32; num_cells * stride];

        let state_buffers = (0..3)
            .map(|i| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("GenericCoupled state buffer {i}")),
                    contents: cast_slice(&zero_state),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                })
            })
            .collect::<Vec<_>>();

        let bg_fields_ping_pong = {
            let bgl = device.create_bind_group_layout(&gen_assembly::WgpuBindGroup1::LAYOUT_DESCRIPTOR);
            let mut out = Vec::new();
            for i in 0..3 {
                let (idx_state, idx_old, idx_old_old) = ping_pong_indices(i);
                out.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("GenericCoupled assembly fields bind group {i}")),
                    layout: &bgl,
                    entries: &gen_assembly::WgpuBindGroup1Entries::new(
                        gen_assembly::WgpuBindGroup1EntriesParams {
                            state: state_buffers[idx_state].as_entire_buffer_binding(),
                            state_old: state_buffers[idx_old].as_entire_buffer_binding(),
                            state_old_old: state_buffers[idx_old_old].as_entire_buffer_binding(),
                            constants: linear.b_constants.as_entire_buffer_binding(),
                        },
                    )
                    .into_array(),
                }));
            }
            out
        };

        let bg_update_state_ping_pong = {
            let bgl = device.create_bind_group_layout(&gen_update::WgpuBindGroup0::LAYOUT_DESCRIPTOR);
            let mut out = Vec::new();
            for i in 0..3 {
                let (idx_state, _, _) = ping_pong_indices(i);
                out.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("GenericCoupled update state bind group {i}")),
                    layout: &bgl,
                    entries: &gen_update::WgpuBindGroup0Entries::new(
                        gen_update::WgpuBindGroup0EntriesParams {
                            state: state_buffers[idx_state].as_entire_buffer_binding(),
                            constants: linear.b_constants.as_entire_buffer_binding(),
                        },
                    )
                    .into_array(),
                }));
            }
            out
        };

        let bg_update_solution = {
            let bgl = device.create_bind_group_layout(&gen_update::WgpuBindGroup1::LAYOUT_DESCRIPTOR);
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GenericCoupled update solution bind group"),
                layout: &bgl,
                entries: &gen_update::WgpuBindGroup1Entries::new(gen_update::WgpuBindGroup1EntriesParams {
                    x: linear.b_x.as_entire_buffer_binding(),
                })
                .into_array(),
            })
        };

        let bg_solver = {
            let bgl = device.create_bind_group_layout(&gen_assembly::WgpuBindGroup2::LAYOUT_DESCRIPTOR);
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GenericCoupled assembly solver bind group"),
                layout: &bgl,
                entries: &gen_assembly::WgpuBindGroup2Entries::new(
                    gen_assembly::WgpuBindGroup2EntriesParams {
                        matrix_values: linear.b_matrix_values.as_entire_buffer_binding(),
                        rhs: linear.b_rhs.as_entire_buffer_binding(),
                        scalar_row_offsets: linear.b_row_offsets.as_entire_buffer_binding(),
                    },
                )
                .into_array(),
            })
        };

        let (bc_kind, bc_value) = model
            .boundaries
            .to_gpu_tables(&model.system)
            .map_err(|e| format!("failed to build BC tables: {e}"))?;

        let b_bc_kind = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GenericCoupled bc_kind"),
            contents: cast_slice(&bc_kind),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let b_bc_value = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GenericCoupled bc_value"),
            contents: cast_slice(&bc_value),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let bg_bc = {
            let bgl = device.create_bind_group_layout(&gen_assembly::WgpuBindGroup3::LAYOUT_DESCRIPTOR);
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GenericCoupled BC bind group"),
                layout: &bgl,
                entries: &gen_assembly::WgpuBindGroup3Entries::new(gen_assembly::WgpuBindGroup3EntriesParams {
                    bc_kind: b_bc_kind.as_entire_buffer_binding(),
                    bc_value: b_bc_value.as_entire_buffer_binding(),
                }).into_array(),
            })
        };

        Ok(Self {
            linear,
            model,
            state_step_index: 0,
            state_buffers,
            bg_mesh,
            bg_fields_ping_pong,
            bg_solver,
            bg_bc,
            bg_update_state_ping_pong,
            bg_update_solution,
            pipeline_assembly,
            pipeline_update,
            _b_bc_kind: b_bc_kind,
            _b_bc_value: b_bc_value,
        })
    }

    pub fn model(&self) -> &ModelSpec {
        &self.model
    }

    pub fn state_buffer(&self) -> &wgpu::Buffer {
        let (idx_state, _, _) = ping_pong_indices(self.state_step_index);
        &self.state_buffers[idx_state]
    }

    pub fn step(&mut self) -> LinearSolverStats {
        let workgroup = 64u32;
        let num_cells = self.linear.num_cells;
        let dispatch = (num_cells + workgroup - 1) / workgroup;

        self.state_step_index = (self.state_step_index + 1) % 3;
        self.linear.constants.time += self.linear.constants.dt;
        self.linear.update_constants();

        let bg_fields = &self.bg_fields_ping_pong[self.state_step_index];
        let bg_update_state = &self.bg_update_state_ping_pong[self.state_step_index];

        let mut encoder = self
            .linear
            .context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GenericCoupled assembly+update"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GenericCoupled assembly"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_assembly);
            pass.set_bind_group(0, &self.bg_mesh, &[]);
            pass.set_bind_group(1, bg_fields, &[]);
            pass.set_bind_group(2, &self.bg_solver, &[]);
            pass.set_bind_group(3, &self.bg_bc, &[]);
            pass.dispatch_workgroups(dispatch, 1, 1);
        }

        self.linear.context.queue.submit(Some(encoder.finish()));

        let stats = self.linear.solve_linear_system_cg(400, 1e-6);

        let mut encoder = self
            .linear
            .context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GenericCoupled update"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GenericCoupled update"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_update);
            pass.set_bind_group(0, bg_update_state, &[]);
            pass.set_bind_group(1, &self.bg_update_solution, &[]);
            pass.dispatch_workgroups(dispatch, 1, 1);
        }

        self.linear.context.queue.submit(Some(encoder.finish()));
        stats
    }

    pub fn set_field_scalar(&self, field: &str, values: &[f64]) -> Result<(), String> {
        let stride = self.model.state_layout.stride() as usize;
        let offset = self
            .model
            .state_layout
            .offset_for(field)
            .ok_or_else(|| format!("field '{field}' not found in layout"))? as usize;
        if values.len() != self.linear.num_cells as usize {
            return Err(format!(
                "value length {} does not match num_cells {}",
                values.len(),
                self.linear.num_cells
            ));
        }
        let mut packed = vec![0.0f32; self.linear.num_cells as usize * stride];
        for (i, &v) in values.iter().enumerate() {
            packed[i * stride + offset] = v as f32;
        }
        let bytes = cast_slice(&packed);
        for buf in &self.state_buffers {
            self.linear.context.queue.write_buffer(buf, 0, bytes);
        }
        Ok(())
    }

    pub async fn get_field_scalar(&self, field: &str) -> Result<Vec<f64>, String> {
        let stride = self.model.state_layout.stride() as usize;
        let offset = self
            .model
            .state_layout
            .offset_for(field)
            .ok_or_else(|| format!("field '{field}' not found in layout"))? as usize;

        let (idx_state, _, _) = ping_pong_indices(self.state_step_index);
        let buf = &self.state_buffers[idx_state];
        let bytes = (self.linear.num_cells as u64) * stride as u64 * 4;
        let raw = self.linear.read_buffer(buf, bytes).await;
        let data: &[f32] = bytemuck::cast_slice(&raw);
        Ok((0..self.linear.num_cells as usize)
            .map(|i| data[i * stride + offset] as f64)
            .collect())
    }
}
