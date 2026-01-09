use crate::solver::gpu::execution_plan::{run_module_graph, GraphDetail, GraphExecMode};
use crate::solver::gpu::modules::generic_coupled_kernels::{
    GenericCoupledBindGroups, GenericCoupledKernelsModule, GenericCoupledPipeline,
};
use crate::solver::gpu::modules::graph::{
    ComputeSpec, DispatchKind, ModuleGraph, ModuleNode, RuntimeDims,
};
use crate::solver::gpu::plans::plan_instance::{
    PlanFuture, PlanLinearSystemDebug, PlanParam, PlanParamValue,
};
use crate::solver::gpu::plans::program::{
    GpuProgramPlan, ModelGpuProgramSpec, ProgramExecutionPlan, ProgramGraphId, ProgramHostId,
    ProgramNode,
};
use crate::solver::gpu::runtime::GpuScalarRuntime;
use crate::solver::gpu::structs::LinearSolverStats;
use crate::solver::model::ModelSpec;
use bytemuck::cast_slice;
use wgpu::util::DeviceExt;

macro_rules! create_bind_group {
    ($device:expr, $label:expr, $layout:expr, $entries:expr) => {{
        let entries = $entries.into_array();
        $device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some($label),
            layout: $layout,
            entries: &entries,
        })
    }};
}

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
                "GpuProgramPlan does not have generated generic-coupled kernels for model id '{other}'"
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

struct GenericCoupledProgramResources {
    runtime: GpuScalarRuntime,
    state_buffers: Vec<wgpu::Buffer>,
    kernels: GenericCoupledKernelsModule,
    assembly_graph: ModuleGraph<GenericCoupledKernelsModule>,
    update_graph: ModuleGraph<GenericCoupledKernelsModule>,
    _b_bc_kind: wgpu::Buffer,
    _b_bc_value: wgpu::Buffer,
}

impl GenericCoupledProgramResources {
    fn runtime_dims(&self) -> RuntimeDims {
        RuntimeDims {
            num_cells: self.runtime.common.num_cells,
            num_faces: 0,
        }
    }
}

impl PlanLinearSystemDebug for GenericCoupledProgramResources {
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

fn res(plan: &GpuProgramPlan) -> &GenericCoupledProgramResources {
    plan.resources
        .get::<GenericCoupledProgramResources>()
        .expect("missing GenericCoupledProgramResources")
}

fn res_mut(plan: &mut GpuProgramPlan) -> &mut GenericCoupledProgramResources {
    plan.resources
        .get_mut::<GenericCoupledProgramResources>()
        .expect("missing GenericCoupledProgramResources")
}

fn spec_num_cells(plan: &GpuProgramPlan) -> u32 {
    res(plan).runtime.common.num_cells
}

fn spec_time(plan: &GpuProgramPlan) -> f32 {
    res(plan).runtime.constants.time
}

fn spec_dt(plan: &GpuProgramPlan) -> f32 {
    res(plan).runtime.constants.dt
}

fn spec_state_buffer(plan: &GpuProgramPlan) -> &wgpu::Buffer {
    let step = res(plan).kernels.step_index();
    let (idx_state, _, _) = ping_pong_indices(step);
    &res(plan).state_buffers[idx_state]
}

fn spec_write_state_bytes(plan: &GpuProgramPlan, bytes: &[u8]) -> Result<(), String> {
    for buf in &res(plan).state_buffers {
        plan.context.queue.write_buffer(buf, 0, bytes);
    }
    Ok(())
}

fn host_prepare_step(plan: &mut GpuProgramPlan) {
    let r = res_mut(plan);
    let next = r.kernels.step_index() + 1;
    r.kernels.set_step_index(next);
    r.runtime.advance_time();
}

fn host_solve_linear_system(plan: &mut GpuProgramPlan) {
    let r = res(plan);
    let stats = r.runtime.solve_linear_system_cg(400, 1e-6);
    plan.last_linear_stats = stats;
}

fn assembly_graph_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let r = res(plan);
    run_module_graph(
        &r.assembly_graph,
        context,
        &r.kernels,
        r.runtime_dims(),
        mode,
    )
}

fn update_graph_run(
    plan: &GpuProgramPlan,
    context: &crate::solver::gpu::context::GpuContext,
    mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    let r = res(plan);
    run_module_graph(&r.update_graph, context, &r.kernels, r.runtime_dims(), mode)
}

fn param_dt(plan: &mut GpuProgramPlan, value: PlanParamValue) -> Result<(), String> {
    let PlanParamValue::F32(dt) = value else {
        return Err("invalid value type".into());
    };
    res_mut(plan).runtime.set_dt(dt);
    Ok(())
}

fn param_advection_scheme(plan: &mut GpuProgramPlan, value: PlanParamValue) -> Result<(), String> {
    let PlanParamValue::Scheme(scheme) = value else {
        return Err("invalid value type".into());
    };
    res_mut(plan).runtime.set_scheme(scheme.gpu_id());
    Ok(())
}

fn param_time_scheme(plan: &mut GpuProgramPlan, value: PlanParamValue) -> Result<(), String> {
    let PlanParamValue::TimeScheme(scheme) = value else {
        return Err("invalid value type".into());
    };
    res_mut(plan).runtime.set_time_scheme(scheme as u32);
    Ok(())
}

fn param_preconditioner(_plan: &mut GpuProgramPlan, _value: PlanParamValue) -> Result<(), String> {
    Ok(())
}

fn param_detailed_profiling(
    plan: &mut GpuProgramPlan,
    value: PlanParamValue,
) -> Result<(), String> {
    let PlanParamValue::Bool(enable) = value else {
        return Err("invalid value type".into());
    };
    if enable {
        plan.profiling_stats.enable();
    } else {
        plan.profiling_stats.disable();
    }
    Ok(())
}

fn linear_debug_provider(plan: &mut GpuProgramPlan) -> Option<&mut dyn PlanLinearSystemDebug> {
    Some(res_mut(plan) as &mut dyn PlanLinearSystemDebug)
}

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

const G_ASSEMBLY: ProgramGraphId = ProgramGraphId(0);
const G_UPDATE: ProgramGraphId = ProgramGraphId(1);

const H_PREPARE: ProgramHostId = ProgramHostId(0);
const H_SOLVE: ProgramHostId = ProgramHostId(1);

pub(crate) async fn lower_generic_coupled_program(
    mesh: &crate::solver::mesh::Mesh,
    model: ModelSpec,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
) -> Result<GpuProgramPlan, String> {
    let coupled_stride = model.system.unknowns_per_cell();
    if coupled_stride != 1 {
        return Err(format!(
            "generic coupled currently supports scalar systems only (unknowns_per_cell=1), got {}",
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
            let bgl =
                device.create_bind_group_layout(&gen_assembly::WgpuBindGroup0::LAYOUT_DESCRIPTOR);
            create_bind_group!(
                device,
                "GenericCoupled: mesh bind group",
                &bgl,
                gen_assembly::WgpuBindGroup0Entries::new(
                    gen_assembly::WgpuBindGroup0EntriesParams {
                        face_owner: runtime.common.mesh.b_face_owner.as_entire_buffer_binding(),
                        face_neighbor: runtime
                            .common
                            .mesh
                            .b_face_neighbor
                            .as_entire_buffer_binding(),
                        face_areas: runtime.common.mesh.b_face_areas.as_entire_buffer_binding(),
                        face_normals: runtime
                            .common
                            .mesh
                            .b_face_normals
                            .as_entire_buffer_binding(),
                        face_centers: runtime
                            .common
                            .mesh
                            .b_face_centers
                            .as_entire_buffer_binding(),
                        cell_centers: runtime
                            .common
                            .mesh
                            .b_cell_centers
                            .as_entire_buffer_binding(),
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
                        diagonal_indices: runtime
                            .common
                            .mesh
                            .b_diagonal_indices
                            .as_entire_buffer_binding(),
                        face_boundary: runtime
                            .common
                            .mesh
                            .b_face_boundary
                            .as_entire_buffer_binding(),
                    }
                )
            )
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
            let bgl =
                device.create_bind_group_layout(&gen_assembly::WgpuBindGroup1::LAYOUT_DESCRIPTOR);
            let mut out = Vec::new();
            for i in 0..3 {
                let (idx_state, idx_old, idx_old_old) = ping_pong_indices(i);
                out.push(create_bind_group!(
                    device,
                    &format!("GenericCoupled assembly fields bind group {i}"),
                    &bgl,
                    gen_assembly::WgpuBindGroup1Entries::new(
                        gen_assembly::WgpuBindGroup1EntriesParams {
                            state: state_buffers[idx_state].as_entire_buffer_binding(),
                            state_old: state_buffers[idx_old].as_entire_buffer_binding(),
                            state_old_old: state_buffers[idx_old_old].as_entire_buffer_binding(),
                            constants: runtime.b_constants.as_entire_buffer_binding(),
                        }
                    )
                ));
            }
            out
        };

        let bg_update_state_ping_pong = {
            let bgl =
                device.create_bind_group_layout(&gen_update::WgpuBindGroup0::LAYOUT_DESCRIPTOR);
            let mut out = Vec::new();
            for i in 0..3 {
                let (idx_state, _, _) = ping_pong_indices(i);
                out.push(create_bind_group!(
                    device,
                    &format!("GenericCoupled update state bind group {i}"),
                    &bgl,
                    gen_update::WgpuBindGroup0Entries::new(
                        gen_update::WgpuBindGroup0EntriesParams {
                            state: state_buffers[idx_state].as_entire_buffer_binding(),
                            constants: runtime.b_constants.as_entire_buffer_binding(),
                        }
                    )
                ));
            }
            out
        };

        let bg_update_solution = {
            let bgl =
                device.create_bind_group_layout(&gen_update::WgpuBindGroup1::LAYOUT_DESCRIPTOR);
            create_bind_group!(
                device,
                "GenericCoupled update solution bind group",
                &bgl,
                gen_update::WgpuBindGroup1Entries::new(gen_update::WgpuBindGroup1EntriesParams {
                    x: runtime
                        .linear_port_space
                        .buffer(runtime.linear_ports.x)
                        .as_entire_buffer_binding(),
                })
            )
        };

        let bg_solver = {
            let bgl =
                device.create_bind_group_layout(&gen_assembly::WgpuBindGroup2::LAYOUT_DESCRIPTOR);
            create_bind_group!(
                device,
                "GenericCoupled assembly solver bind group",
                &bgl,
                gen_assembly::WgpuBindGroup2Entries::new(
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
                    }
                )
            )
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
            let bgl =
                device.create_bind_group_layout(&gen_assembly::WgpuBindGroup3::LAYOUT_DESCRIPTOR);
            create_bind_group!(
                device,
                "GenericCoupled BC bind group",
                &bgl,
                gen_assembly::WgpuBindGroup3Entries::new(
                    gen_assembly::WgpuBindGroup3EntriesParams {
                        bc_kind: b_bc_kind.as_entire_buffer_binding(),
                        bc_value: b_bc_value.as_entire_buffer_binding(),
                    }
                )
            )
        };

        let kernels = GenericCoupledKernelsModule::new(
            bg_mesh,
            bg_fields_ping_pong,
            bg_solver,
            bg_bc,
            bg_update_state_ping_pong,
            bg_update_solution,
            pipeline_assembly,
            pipeline_update,
        );

        let context = crate::solver::gpu::context::GpuContext {
            device: runtime.common.context.device.clone(),
            queue: runtime.common.context.queue.clone(),
        };
        let profiling_stats = std::sync::Arc::clone(&runtime.common.profiling_stats);

        let resources = GenericCoupledProgramResources {
            runtime,
            state_buffers,
            kernels,
            assembly_graph: build_assembly_graph(),
            update_graph: build_update_graph(),
            _b_bc_kind: b_bc_kind,
            _b_bc_value: b_bc_value,
        };

        let mut program_resources = crate::solver::gpu::plans::program::ProgramResources::new();
        program_resources.insert(resources);

        let mut params = std::collections::HashMap::new();
        params.insert(PlanParam::Dt, param_dt as _);
        params.insert(PlanParam::AdvectionScheme, param_advection_scheme as _);
        params.insert(PlanParam::TimeScheme, param_time_scheme as _);
        params.insert(PlanParam::Preconditioner, param_preconditioner as _);
        params.insert(
            PlanParam::DetailedProfilingEnabled,
            param_detailed_profiling as _,
        );

        let mut graph_ops = std::collections::HashMap::new();
        graph_ops.insert(G_ASSEMBLY, assembly_graph_run as _);
        graph_ops.insert(G_UPDATE, update_graph_run as _);

        let mut host_ops = std::collections::HashMap::new();
        host_ops.insert(H_PREPARE, host_prepare_step as _);
        host_ops.insert(H_SOLVE, host_solve_linear_system as _);

        let cond_ops = std::collections::HashMap::new();
        let count_ops = std::collections::HashMap::new();

        let step = std::sync::Arc::new(ProgramExecutionPlan::new(vec![
            ProgramNode::Host {
                label: "generic_coupled:prepare",
                id: H_PREPARE,
            },
            ProgramNode::Graph {
                label: "generic_coupled:assembly",
                id: G_ASSEMBLY,
                mode: GraphExecMode::SplitTimed,
            },
            ProgramNode::Host {
                label: "generic_coupled:solve",
                id: H_SOLVE,
            },
            ProgramNode::Graph {
                label: "generic_coupled:update",
                id: G_UPDATE,
                mode: GraphExecMode::SingleSubmit,
            },
        ]));

        let spec = ModelGpuProgramSpec {
            graph_ops,
            host_ops,
            cond_ops,
            count_ops,
            num_cells: spec_num_cells,
            time: spec_time,
            dt: spec_dt,
            state_buffer: spec_state_buffer,
            write_state_bytes: spec_write_state_bytes,
            step,
            initialize_history: None,
            params,
            set_param_fallback: None,
            step_stats: None,
            step_with_stats: None,
            linear_debug: Some(linear_debug_provider),
        };

        let plan = GpuProgramPlan::new(model, context, profiling_stats, program_resources, spec);
        Ok(plan)
    })
}
