use crate::solver::gpu::runtime::GpuScalarRuntime;
use crate::solver::gpu::plans::plan_instance::{
    FgmresSizing, GpuPlanInstance, PlanCapability, PlanCoupledUnknowns, PlanFgmresSizing,
    PlanFuture, PlanLinearSystemDebug, PlanParam, PlanParamValue,
};
use crate::solver::gpu::profiling::ProfilingStats;
use crate::solver::gpu::structs::LinearSolverStats;
use crate::solver::model::ModelSpec;
use bytemuck::cast_slice;
use crate::solver::gpu::modules::generic_coupled_kernels::{
    GenericCoupledBindGroups, GenericCoupledKernelsModule, GenericCoupledPipeline,
};
use crate::solver::gpu::modules::graph::{ComputeSpec, DispatchKind, ModuleGraph, ModuleNode, RuntimeDims};
use std::sync::Arc;
use wgpu::util::DeviceExt;

macro_rules! with_generic_coupled_kernels {
    ($model_id:expr, |$gen_assembly:ident, $gen_update:ident| $body:block) => {{
        match $model_id {
            "generic_diffusion_demo" => {
                use crate::solver::gpu::bindings::generated::generic_coupled_assembly_generic_diffusion_demo as $gen_assembly;
                use crate::solver::gpu::bindings::generated::generic_coupled_update_generic_diffusion_demo as $gen_update;
                $body
            }
            "generic_diffusion_demo_neumann" => {
                use crate::solver::gpu::bindings::generated::generic_coupled_assembly_generic_diffusion_demo_neumann as $gen_assembly;
                use crate::solver::gpu::bindings::generated::generic_coupled_update_generic_diffusion_demo_neumann as $gen_update;
                $body
            }
            other => Err(format!(
                "GpuGenericCoupledSolver does not have generated kernels for model id '{other}'"
            )),
        }
    }};
}

fn ping_pong_indices(i: usize) -> (usize, usize, usize) {
    match i {
        0 => (0, 1, 2),
        1 => (2, 0, 1),
        2 => (1, 2, 0),
        _ => (0, 1, 2),
    }
}

pub(crate) struct GpuGenericCoupledSolver {
    pub runtime: GpuScalarRuntime,
    model: ModelSpec,

    state_buffers: Vec<wgpu::Buffer>,
    kernels: GenericCoupledKernelsModule,
    assembly_graph: ModuleGraph<GenericCoupledKernelsModule>,
    update_graph: ModuleGraph<GenericCoupledKernelsModule>,

    _b_bc_kind: wgpu::Buffer,
    _b_bc_value: wgpu::Buffer,
}

impl GpuGenericCoupledSolver {
    fn build_assembly_graph() -> ModuleGraph<GenericCoupledKernelsModule> {
        ModuleGraph::new(vec![ModuleNode::Compute(ComputeSpec {
            label: "generic_coupled:assembly",
            pipeline: GenericCoupledPipeline::Assembly,
            bind: GenericCoupledBindGroups::Assembly,
            dispatch: DispatchKind::Cells,
        })])
    }

    fn build_update_graph() -> ModuleGraph<GenericCoupledKernelsModule> {
        ModuleGraph::new(vec![ModuleNode::Compute(ComputeSpec {
            label: "generic_coupled:update",
            pipeline: GenericCoupledPipeline::Update,
            bind: GenericCoupledBindGroups::Update,
            dispatch: DispatchKind::Cells,
        })])
    }

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

        let runtime = GpuScalarRuntime::new(mesh, device, queue).await;
        runtime.update_constants();

        let device = &runtime.common.context.device;

        with_generic_coupled_kernels!(model.id, |gen_assembly, gen_update| {
            let pipeline_assembly = gen_assembly::compute::create_main_pipeline_embed_source(device);
            let pipeline_update = gen_update::compute::create_main_pipeline_embed_source(device);

            let bg_mesh = {
                let bgl = device
                    .create_bind_group_layout(&gen_assembly::WgpuBindGroup0::LAYOUT_DESCRIPTOR);
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("GenericCoupled: mesh bind group"),
                    layout: &bgl,
                    entries: &gen_assembly::WgpuBindGroup0Entries::new(
                        gen_assembly::WgpuBindGroup0EntriesParams {
                            face_owner: runtime.common.mesh.b_face_owner.as_entire_buffer_binding(),
                            face_neighbor: runtime.common.mesh.b_face_neighbor.as_entire_buffer_binding(),
                            face_areas: runtime.common.mesh.b_face_areas.as_entire_buffer_binding(),
                            face_normals: runtime.common.mesh.b_face_normals.as_entire_buffer_binding(),
                            face_centers: runtime.common.mesh.b_face_centers.as_entire_buffer_binding(),
                            cell_centers: runtime.common.mesh.b_cell_centers.as_entire_buffer_binding(),
                            cell_vols: runtime.common.mesh.b_cell_vols.as_entire_buffer_binding(),
                            cell_face_offsets: runtime
                                .common
                                .mesh
                                .b_cell_face_offsets
                                .as_entire_buffer_binding(),
                            cell_faces: runtime.common.mesh.b_cell_faces.as_entire_buffer_binding(),
                            cell_face_matrix_indices: runtime
                                .common
                                .mesh
                                .b_cell_face_matrix_indices
                                .as_entire_buffer_binding(),
                            diagonal_indices: runtime.common.mesh.b_diagonal_indices.as_entire_buffer_binding(),
                            face_boundary: runtime.common.mesh.b_face_boundary.as_entire_buffer_binding(),
                        },
                    )
                    .into_array(),
                })
            };

            let stride = model.state_layout.stride() as usize;
            let num_cells = runtime.common.num_cells as usize;
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
                let bgl = device.create_bind_group_layout(
                    &gen_assembly::WgpuBindGroup1::LAYOUT_DESCRIPTOR,
                );
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
                                constants: runtime.b_constants.as_entire_buffer_binding(),
                            },
                        )
                        .into_array(),
                    }));
                }
                out
            };

            let bg_update_state_ping_pong = {
                let bgl = device
                    .create_bind_group_layout(&gen_update::WgpuBindGroup0::LAYOUT_DESCRIPTOR);
                let mut out = Vec::new();
                for i in 0..3 {
                    let (idx_state, _, _) = ping_pong_indices(i);
                    out.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(&format!("GenericCoupled update state bind group {i}")),
                        layout: &bgl,
                        entries: &gen_update::WgpuBindGroup0Entries::new(
                            gen_update::WgpuBindGroup0EntriesParams {
                                state: state_buffers[idx_state].as_entire_buffer_binding(),
                                constants: runtime.b_constants.as_entire_buffer_binding(),
                            },
                        )
                        .into_array(),
                    }));
                }
                out
            };

            let bg_update_solution = {
                let bgl = device
                    .create_bind_group_layout(&gen_update::WgpuBindGroup1::LAYOUT_DESCRIPTOR);
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("GenericCoupled update solution bind group"),
                    layout: &bgl,
                    entries: &gen_update::WgpuBindGroup1Entries::new(
                        gen_update::WgpuBindGroup1EntriesParams {
                            x: runtime
                                .linear_port_space
                                .buffer(runtime.linear_ports.x)
                                .as_entire_buffer_binding(),
                        },
                    )
                    .into_array(),
                })
            };

            let bg_solver = {
                let bgl = device
                    .create_bind_group_layout(&gen_assembly::WgpuBindGroup2::LAYOUT_DESCRIPTOR);
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("GenericCoupled assembly solver bind group"),
                    layout: &bgl,
                    entries: &gen_assembly::WgpuBindGroup2Entries::new(
                        gen_assembly::WgpuBindGroup2EntriesParams {
                            matrix_values: runtime
                                .linear_port_space
                                .buffer(runtime.linear_ports.values)
                                .as_entire_buffer_binding(),
                            rhs: runtime
                                .linear_port_space
                                .buffer(runtime.linear_ports.rhs)
                                .as_entire_buffer_binding(),
                            scalar_row_offsets: runtime
                                .linear_port_space
                                .buffer(runtime.linear_ports.row_offsets)
                                .as_entire_buffer_binding(),
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
                let bgl = device
                    .create_bind_group_layout(&gen_assembly::WgpuBindGroup3::LAYOUT_DESCRIPTOR);
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("GenericCoupled BC bind group"),
                    layout: &bgl,
                    entries: &gen_assembly::WgpuBindGroup3Entries::new(
                        gen_assembly::WgpuBindGroup3EntriesParams {
                            bc_kind: b_bc_kind.as_entire_buffer_binding(),
                            bc_value: b_bc_value.as_entire_buffer_binding(),
                        },
                    )
                    .into_array(),
                })
            };

            Ok(Self {
                runtime,
                model,
                state_buffers,
                kernels: GenericCoupledKernelsModule::new(
                    bg_mesh,
                    bg_fields_ping_pong,
                    bg_solver,
                    bg_bc,
                    bg_update_state_ping_pong,
                    bg_update_solution,
                    pipeline_assembly,
                    pipeline_update,
                ),
                assembly_graph: GpuGenericCoupledSolver::build_assembly_graph(),
                update_graph: GpuGenericCoupledSolver::build_update_graph(),
                _b_bc_kind: b_bc_kind,
                _b_bc_value: b_bc_value,
            })
        })
    }

    pub fn model(&self) -> &ModelSpec {
        &self.model
    }

    pub fn state_buffer(&self) -> &wgpu::Buffer {
        let (idx_state, _, _) = ping_pong_indices(self.kernels.step_index());
        &self.state_buffers[idx_state]
    }

    pub(crate) fn write_state_bytes(&self, bytes: &[u8]) {
        for buf in &self.state_buffers {
            self.runtime.common.context.queue.write_buffer(buf, 0, bytes);
        }
    }

    pub fn step(&mut self) -> LinearSolverStats {
        let num_cells = self.runtime.common.num_cells;
        let dims = RuntimeDims {
            num_cells,
            num_faces: 0,
        };

        self.kernels.set_step_index(self.kernels.step_index() + 1);
        self.runtime.advance_time();

        self.assembly_graph
            .execute(&self.runtime.common.context, &self.kernels, dims);

        let stats = self.runtime.solve_linear_system_cg(400, 1e-6);

        self.update_graph
            .execute(&self.runtime.common.context, &self.kernels, dims);
        stats
    }

    pub fn set_field_scalar(&self, field: &str, values: &[f64]) -> Result<(), String> {
        let stride = self.model.state_layout.stride() as usize;
        let offset = self
            .model
            .state_layout
            .offset_for(field)
            .ok_or_else(|| format!("field '{field}' not found in layout"))? as usize;
        if values.len() != self.runtime.common.num_cells as usize {
            return Err(format!(
                "value length {} does not match num_cells {}",
                values.len(),
                self.runtime.common.num_cells
            ));
        }
        let mut packed = vec![0.0f32; self.runtime.common.num_cells as usize * stride];
        for (i, &v) in values.iter().enumerate() {
            packed[i * stride + offset] = v as f32;
        }
        let bytes = cast_slice(&packed);
        for buf in &self.state_buffers {
            self.runtime.common.context.queue.write_buffer(buf, 0, bytes);
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

        let (idx_state, _, _) = ping_pong_indices(self.kernels.step_index());
        let buf = &self.state_buffers[idx_state];
        let bytes = (self.runtime.common.num_cells as u64) * stride as u64 * 4;
        let raw = self.runtime.read_buffer(buf, bytes).await;
        let data: &[f32] = bytemuck::cast_slice(&raw);
        Ok((0..self.runtime.common.num_cells as usize)
            .map(|i| data[i * stride + offset] as f64)
            .collect())
    }
}

impl GpuPlanInstance for GpuGenericCoupledSolver {
    fn num_cells(&self) -> u32 {
        self.runtime.common.num_cells
    }

    fn time(&self) -> f32 {
        self.runtime.constants.time
    }

    fn dt(&self) -> f32 {
        self.runtime.constants.dt
    }

    fn state_buffer(&self) -> &wgpu::Buffer {
        GpuGenericCoupledSolver::state_buffer(self)
    }

    fn supports(&self, capability: PlanCapability) -> bool {
        match capability {
            PlanCapability::LinearSystemDebug => true,
            PlanCapability::CoupledUnknowns => true,
            PlanCapability::FgmresSizing => true,
        }
    }

    fn profiling_stats(&self) -> Arc<ProfilingStats> {
        Arc::clone(&self.runtime.common.profiling_stats)
    }

    fn set_param(&mut self, param: PlanParam, value: PlanParamValue) -> Result<(), String> {
        match (param, value) {
            (PlanParam::Dt, PlanParamValue::F32(dt)) => {
                self.runtime.set_dt(dt);
                Ok(())
            }
            (PlanParam::AdvectionScheme, PlanParamValue::Scheme(scheme)) => {
                self.runtime.set_scheme(scheme.gpu_id());
                Ok(())
            }
            (PlanParam::TimeScheme, PlanParamValue::TimeScheme(scheme)) => {
                self.runtime.set_time_scheme(scheme as u32);
                Ok(())
            }
            (PlanParam::Preconditioner, PlanParamValue::Preconditioner(_preconditioner)) => {
                // Generic coupled currently doesn't implement preconditioners.
                Ok(())
            }
            (PlanParam::DetailedProfilingEnabled, PlanParamValue::Bool(enable)) => {
                if enable {
                    self.runtime.common.profiling_stats.enable();
                } else {
                    self.runtime.common.profiling_stats.disable();
                }
                Ok(())
            }
            _ => Err("parameter is not supported by this plan".into()),
        }
    }

    fn write_state_bytes(&self, bytes: &[u8]) -> Result<(), String> {
        GpuGenericCoupledSolver::write_state_bytes(self, bytes);
        Ok(())
    }

    fn step_with_stats(&mut self) -> Result<Vec<LinearSolverStats>, String> {
        Ok(vec![GpuGenericCoupledSolver::step(self)])
    }

    fn step(&mut self) {
        let _ = GpuGenericCoupledSolver::step(self);
    }

    fn initialize_history(&self) {}

    fn read_state_bytes(&self, bytes: u64) -> PlanFuture<'_, Vec<u8>> {
        Box::pin(async move { self.runtime.read_buffer(self.state_buffer(), bytes).await })
    }

    fn linear_system_debug(&mut self) -> Option<&mut dyn PlanLinearSystemDebug> {
        Some(self)
    }

    fn coupled_unknowns_debug(&mut self) -> Option<&mut dyn PlanCoupledUnknowns> {
        Some(self)
    }

    fn fgmres_sizing_debug(&mut self) -> Option<&mut dyn PlanFgmresSizing> {
        Some(self)
    }
}

impl PlanLinearSystemDebug for GpuGenericCoupledSolver {
    fn set_linear_system(&self, matrix_values: &[f32], rhs: &[f32]) -> Result<(), String> {
        self.runtime.set_linear_system(matrix_values, rhs)
    }

    fn solve_linear_system_with_size(
        &mut self,
        n: u32,
        max_iters: u32,
        tol: f32,
    ) -> Result<LinearSolverStats, String> {
        if n != self.runtime.common.num_cells {
            return Err(format!(
                "requested solve size {} does not match num_cells {}",
                n, self.runtime.common.num_cells
            ));
        }
        Ok(self
            .runtime
            .solve_linear_system_cg_with_size(n, max_iters, tol))
    }

    fn get_linear_solution(&self) -> PlanFuture<'_, Result<Vec<f32>, String>> {
        Box::pin(async move {
            self.runtime
                .get_linear_solution(self.runtime.common.num_cells)
                .await
        })
    }
}

impl PlanCoupledUnknowns for GpuGenericCoupledSolver {
    fn coupled_unknowns(&self) -> Result<u32, String> {
        Ok(self.runtime.common.num_cells)
    }
}

impl PlanFgmresSizing for GpuGenericCoupledSolver {
    fn fgmres_sizing(&mut self, _max_restart: usize) -> Result<FgmresSizing, String> {
        let n = self.runtime.common.num_cells;
        Ok(FgmresSizing {
            num_unknowns: n,
            num_dot_groups: (n + 63) / 64,
        })
    }
}
