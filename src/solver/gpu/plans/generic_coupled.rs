use std::collections::HashMap;

use bytemuck::cast_slice;
use wgpu::util::DeviceExt;

use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::lowering::models::generic_coupled::{
    linear_debug_provider, param_advection_scheme, param_alpha_p, param_alpha_u, param_density,
    param_detailed_profiling, param_dt, param_dtau, param_inlet_velocity, param_low_mach_model,
    param_low_mach_theta_floor, param_outer_iters, param_preconditioner, param_ramp_time,
    param_time_scheme, param_viscosity, register_ops_from_recipe, spec_dt, spec_num_cells,
    spec_state_buffer, spec_time, spec_write_state_bytes,
    GenericCoupledProgramResources,
};
use crate::solver::gpu::lowering::types::{LoweredProgramParts, ModelGpuProgramSpecParts};
use crate::solver::gpu::modules::generated_kernels::GeneratedKernelsModule;
use crate::solver::gpu::modules::resource_registry::ResourceRegistry;
use crate::solver::gpu::modules::unified_field_resources::UnifiedFieldResources;
use crate::solver::gpu::plans::plan_instance::PlanParam;
use crate::solver::gpu::plans::program::{ProgramOpRegistry, ProgramResources};
use crate::solver::gpu::recipe::SolverRecipe;
use crate::solver::gpu::runtime::GpuCsrRuntime;
use crate::solver::mesh::Mesh;
use crate::solver::model::ModelSpec;

pub struct GenericCoupledPlanResources {
    pub common: GpuCsrRuntime,
}

impl GenericCoupledPlanResources {
    pub async fn new(
        mesh: &Mesh,
        model: ModelSpec,
        recipe: SolverRecipe,
        device: Option<wgpu::Device>,
        queue: Option<wgpu::Queue>,
    ) -> Result<LoweredProgramParts, String> {
        let unknowns_per_cell = model.system.unknowns_per_cell();
        let runtime = GpuCsrRuntime::new(mesh, unknowns_per_cell, device, queue).await;
        runtime.update_constants();

        let device = &runtime.common.context.device;

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

        // Create buffer for scalar row offsets (mesh-level CSR structure)
        // This is used by assembly kernels to iterate over cell neighbors
        let b_scalar_row_offsets = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GenericCoupled scalar_row_offsets"),
            contents: cast_slice(&runtime.common.mesh.row_offsets),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let b_scalar_col_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GenericCoupled scalar_col_indices"),
            contents: cast_slice(&runtime.common.mesh.col_indices),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

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

        let registry = ResourceRegistry::new()
            .with_mesh(&runtime.common.mesh)
            .with_unified_fields(&fields)
            .with_constants_buffer(runtime.constants.buffer())
            .with_buffer(
                "matrix_values",
                runtime.linear_port_space.buffer(runtime.linear_ports.values),
            )
            .with_buffer(
                "rhs",
                runtime.linear_port_space.buffer(runtime.linear_ports.rhs),
            )
            .with_buffer("x", runtime.linear_port_space.buffer(runtime.linear_ports.x))
            .with_buffer("bc_kind", &b_bc_kind)
            .with_buffer("bc_value", &b_bc_value)
            .with_buffer("scalar_row_offsets", &b_scalar_row_offsets)
            .with_buffer("row_offsets", &b_scalar_row_offsets)
            .with_buffer("col_indices", &b_scalar_col_indices)
            .with_buffer("y", runtime.linear_port_space.buffer(runtime.linear_ports.rhs));

        let kernels = GeneratedKernelsModule::new_from_recipe(
            device,
            model.id,
            &recipe,
            &registry,
            fields.step_handle(),
        )?;

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
            &model,
            &recipe,
            b_scalar_row_offsets,
            b_scalar_col_indices,
            b_bc_kind,
            b_bc_value,
        )?);

        let mut params = HashMap::new();
        params.insert(PlanParam::Dt, param_dt as _);
        params.insert(PlanParam::Dtau, param_dtau as _);
        params.insert(PlanParam::AdvectionScheme, param_advection_scheme as _);
        params.insert(PlanParam::TimeScheme, param_time_scheme as _);
        params.insert(PlanParam::Preconditioner, param_preconditioner as _);
        params.insert(PlanParam::Viscosity, param_viscosity as _);
        params.insert(PlanParam::Density, param_density as _);
        params.insert(PlanParam::AlphaU, param_alpha_u as _);
        params.insert(PlanParam::AlphaP, param_alpha_p as _);
        params.insert(PlanParam::InletVelocity, param_inlet_velocity as _);
        params.insert(PlanParam::RampTime, param_ramp_time as _);
        params.insert(PlanParam::LowMachModel, param_low_mach_model as _);
        params.insert(PlanParam::LowMachThetaFloor, param_low_mach_theta_floor as _);
        params.insert(PlanParam::OuterIters, param_outer_iters as _);
        params.insert(
            PlanParam::DetailedProfilingEnabled,
            param_detailed_profiling as _,
        );

        let mut ops = ProgramOpRegistry::new();
        register_ops_from_recipe(&recipe, &mut ops)?;

        Ok(LoweredProgramParts {
            model: model.clone(),
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
