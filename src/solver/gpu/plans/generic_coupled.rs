use std::collections::HashMap;

use bytemuck::cast_slice;
use wgpu::util::DeviceExt;

use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::lowering::kernel_registry;
use crate::solver::gpu::lowering::models::generic_coupled::{
    linear_debug_provider, param_advection_scheme, param_detailed_profiling, param_dt,
    param_preconditioner, param_time_scheme, register_ops_from_recipe, spec_dt, spec_num_cells,
    spec_state_buffer, spec_time, spec_write_state_bytes, GenericCoupledProgramResources,
};
use crate::solver::gpu::lowering::types::{LoweredProgramParts, ModelGpuProgramSpecParts};
use crate::solver::gpu::modules::unified_field_resources::UnifiedFieldResources;
use crate::solver::gpu::plans::plan_instance::PlanParam;
use crate::solver::gpu::plans::program::{ProgramOpRegistry, ProgramResources};
use crate::solver::gpu::recipe::SolverRecipe;
use crate::solver::gpu::runtime::GpuScalarRuntime;
use crate::solver::gpu::wgsl_reflect;
use crate::solver::mesh::Mesh;
use crate::solver::model::{KernelKind, ModelSpec};

pub struct GenericCoupledPlanResources {
    pub common: GpuScalarRuntime,
}

impl GenericCoupledPlanResources {
    pub async fn new(
        mesh: &Mesh,
        model: ModelSpec,
        recipe: SolverRecipe,
        device: Option<wgpu::Device>,
        queue: Option<wgpu::Queue>,
    ) -> Result<LoweredProgramParts, String> {
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

        let assembly =
            kernel_registry::kernel_source(model.id, KernelKind::GenericCoupledAssembly)?;
        let update = kernel_registry::kernel_source(model.id, KernelKind::GenericCoupledUpdate)?;
        let assembly_bindings = assembly.bindings;
        let update_bindings = update.bindings;

        let pipeline_assembly = (assembly.create_pipeline)(device);
        let pipeline_update = (update.create_pipeline)(device);

        let bg_mesh = {
            let bgl = pipeline_assembly.get_bind_group_layout(0);
            wgsl_reflect::create_bind_group_from_bindings(
                device,
                "GenericCoupled: mesh bind group",
                &bgl,
                assembly_bindings,
                0,
                |name| {
                    runtime
                        .common
                        .mesh
                        .buffer_for_binding_name(name)
                        .map(|buf| wgpu::BindingResource::Buffer(buf.as_entire_buffer_binding()))
                },
            )?
        };

        let stride = model.state_layout.stride();
        let num_cells = runtime.common.num_cells;

        // Create unified field resources from recipe
        // Clone the initial constants from runtime
        let initial_constants = *runtime.constants.values();
        let fields = UnifiedFieldResources::from_recipe(
            device,
            &recipe,
            num_cells,
            runtime.common.num_faces,
            stride,
            initial_constants,
        );

        let bg_fields_ping_pong = {
            let bgl = pipeline_assembly.get_bind_group_layout(1);
            let mut out = Vec::new();
            for i in 0..3 {
                out.push(wgsl_reflect::create_bind_group_from_bindings(
                    device,
                    &format!("GenericCoupled assembly fields bind group {i}"),
                    &bgl,
                    assembly_bindings,
                    1,
                    |name| {
                        // First check unified fields, then fall back to runtime constants
                        if let Some(buf) = fields.buffer_for_binding(name, i) {
                            return Some(wgpu::BindingResource::Buffer(
                                buf.as_entire_buffer_binding(),
                            ));
                        }
                        // Handle constants from runtime (which is updated during time stepping)
                        if name == "constants" {
                            return Some(wgpu::BindingResource::Buffer(
                                runtime.constants.buffer().as_entire_buffer_binding(),
                            ));
                        }
                        None
                    },
                )?);
            }
            out
        };

        let bg_solver = {
            let bgl = pipeline_assembly.get_bind_group_layout(2);
            wgsl_reflect::create_bind_group_from_bindings(
                device,
                "GenericCoupled assembly solver bind group",
                &bgl,
                assembly_bindings,
                2,
                |name| match name {
                    "matrix_values" => Some(wgpu::BindingResource::Buffer(
                        runtime
                            .linear_port_space
                            .buffer(runtime.linear_ports.values)
                            .as_entire_buffer_binding(),
                    )),
                    "rhs" => Some(wgpu::BindingResource::Buffer(
                        runtime
                            .linear_port_space
                            .buffer(runtime.linear_ports.rhs)
                            .as_entire_buffer_binding(),
                    )),
                    "scalar_row_offsets" => Some(wgpu::BindingResource::Buffer(
                        runtime
                            .linear_port_space
                            .buffer(runtime.linear_ports.row_offsets)
                            .as_entire_buffer_binding(),
                    )),
                    _ => None,
                },
            )?
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
            let bgl = pipeline_assembly.get_bind_group_layout(3);
            wgsl_reflect::create_bind_group_from_bindings(
                device,
                "GenericCoupled BC bind group",
                &bgl,
                assembly_bindings,
                3,
                |name| match name {
                    "bc_kind" => Some(wgpu::BindingResource::Buffer(
                        b_bc_kind.as_entire_buffer_binding(),
                    )),
                    "bc_value" => Some(wgpu::BindingResource::Buffer(
                        b_bc_value.as_entire_buffer_binding(),
                    )),
                    _ => None,
                },
            )?
        };

        let bg_update_state_ping_pong = {
            let bgl = pipeline_update.get_bind_group_layout(0);
            let mut out = Vec::new();
            for i in 0..3 {
                out.push(wgsl_reflect::create_bind_group_from_bindings(
                    device,
                    &format!("GenericCoupled update state bind group {i}"),
                    &bgl,
                    update_bindings,
                    0,
                    |name| {
                        // First check unified fields, then fall back to runtime constants
                        if let Some(buf) = fields.buffer_for_binding(name, i) {
                            return Some(wgpu::BindingResource::Buffer(
                                buf.as_entire_buffer_binding(),
                            ));
                        }
                        // Handle constants from runtime
                        if name == "constants" {
                            return Some(wgpu::BindingResource::Buffer(
                                runtime.constants.buffer().as_entire_buffer_binding(),
                            ));
                        }
                        None
                    },
                )?);
            }
            out
        };

        let bg_update_solution = {
            let bgl = pipeline_update.get_bind_group_layout(1);
            wgsl_reflect::create_bind_group_from_bindings(
                device,
                "GenericCoupled update solution bind group",
                &bgl,
                update_bindings,
                1,
                |name| match name {
                    "x" => Some(wgpu::BindingResource::Buffer(
                        runtime
                            .linear_port_space
                            .buffer(runtime.linear_ports.x)
                            .as_entire_buffer_binding(),
                    )),
                    _ => None,
                },
            )?
        };

        let kernels =
            crate::solver::gpu::modules::generic_coupled_kernels::GenericCoupledKernelsModule::new(
                fields.step_handle(),
                bg_mesh,
                bg_fields_ping_pong,
                bg_solver,
                bg_bc,
                bg_update_state_ping_pong,
                bg_update_solution,
                pipeline_assembly,
                pipeline_update,
            );

        let context = GpuContext {
            device: runtime.common.context.device.clone(),
            queue: runtime.common.context.queue.clone(),
        };
        let profiling_stats = std::sync::Arc::clone(&runtime.common.profiling_stats);

        let mut resources = ProgramResources::new();
        resources.insert(GenericCoupledProgramResources::new(
            runtime,
            fields,
            kernels,
            &recipe,
            b_bc_kind,
            b_bc_value,
        ));

        let mut params = HashMap::new();
        params.insert(PlanParam::Dt, param_dt as _);
        params.insert(PlanParam::AdvectionScheme, param_advection_scheme as _);
        params.insert(PlanParam::TimeScheme, param_time_scheme as _);
        params.insert(PlanParam::Preconditioner, param_preconditioner as _);
        params.insert(
            PlanParam::DetailedProfilingEnabled,
            param_detailed_profiling as _,
        );

        let mut ops = ProgramOpRegistry::new();
        register_ops_from_recipe(&recipe, &mut ops)?;

        Ok(LoweredProgramParts {
            model: model.clone(),
            recipe,
            context,
            profiling_stats,
            resources,
            spec: ModelGpuProgramSpecParts {
                ops,
                num_cells: spec_num_cells,
                time: spec_time,
                dt: spec_dt,
                state_buffer: spec_state_buffer,
                write_state_bytes: spec_write_state_bytes,
                initialize_history: None,
                params,
                set_param_fallback: None,
                step_stats: None,
                step_with_stats: None,
                linear_debug: Some(linear_debug_provider),
            },
        })
    }
}
