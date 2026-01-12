use std::collections::HashMap;

use bytemuck::cast_slice;
use wgpu::util::DeviceExt;

use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::lowering::kernel_registry;
use crate::solver::gpu::lowering::models::generic_coupled::{
    linear_debug_provider, param_advection_scheme, param_alpha_p, param_alpha_u, param_density,
    param_detailed_profiling, param_dt, param_dtau, param_inlet_velocity, param_low_mach_model,
    param_low_mach_theta_floor, param_outer_iters, param_preconditioner, param_ramp_time,
    param_time_scheme, param_viscosity, register_ops_from_recipe, spec_dt, spec_num_cells,
    spec_state_buffer, spec_time, spec_write_state_bytes,
    GenericCoupledProgramResources,
};
use crate::solver::gpu::lowering::types::{LoweredProgramParts, ModelGpuProgramSpecParts};
use crate::solver::gpu::modules::resource_registry::ResourceRegistry;
use crate::solver::gpu::modules::unified_field_resources::UnifiedFieldResources;
use crate::solver::gpu::plans::plan_instance::PlanParam;
use crate::solver::gpu::plans::program::{ProgramOpRegistry, ProgramResources};
use crate::solver::gpu::recipe::SolverRecipe;
use crate::solver::gpu::runtime::GpuCsrRuntime;
use crate::solver::gpu::wgsl_reflect;
use crate::solver::mesh::Mesh;
use crate::solver::model::{KernelId, ModelSpec};

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

        let assembly =
            kernel_registry::kernel_source_by_id(model.id, KernelId::GENERIC_COUPLED_ASSEMBLY)?;
        let update =
            kernel_registry::kernel_source_by_id(model.id, KernelId::GENERIC_COUPLED_UPDATE)?;

        let kt_gradients = recipe
            .kernels
            .iter()
            .any(|k| k.id == KernelId::KT_GRADIENTS)
            .then(|| kernel_registry::kernel_source_by_id(model.id, KernelId::KT_GRADIENTS))
            .transpose()?;

        let flux_kt = recipe
            .kernels
            .iter()
            .any(|k| k.id == KernelId::FLUX_KT)
            .then(|| kernel_registry::kernel_source_by_id(model.id, KernelId::FLUX_KT))
            .transpose()?;

        let has_flux = recipe.kernels.iter().any(|k| k.id == KernelId::FLUX_RHIE_CHOW);
        let flux = if has_flux {
            Some(kernel_registry::kernel_source_by_id(
                model.id,
                KernelId::FLUX_RHIE_CHOW,
            )?)
        } else {
            None
        };
        let assembly_bindings = assembly.bindings;
        let update_bindings = update.bindings;

        let pipeline_assembly = (assembly.create_pipeline)(device);
        let pipeline_update = (update.create_pipeline)(device);

        let pipeline_flux = flux.as_ref().map(|src| (src.create_pipeline)(device));

        let pipeline_kt_gradients =
            kt_gradients.as_ref().map(|src| (src.create_pipeline)(device));
        let pipeline_flux_kt = flux_kt.as_ref().map(|src| (src.create_pipeline)(device));

        let bg_mesh = {
            let bgl = pipeline_assembly.get_bind_group_layout(0);
            let registry = ResourceRegistry::new().with_mesh(&runtime.common.mesh);
            wgsl_reflect::create_bind_group_from_bindings(
                device,
                "GenericCoupled: mesh bind group",
                &bgl,
                assembly_bindings,
                0,
                |name| registry.resolve(name),
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

        let (bg_mesh_flux, bg_fields_flux_ping_pong) = if let (Some(flux), Some(pipeline_flux)) =
            (flux.as_ref(), pipeline_flux.as_ref())
        {
            let bg_mesh_flux = {
                let bgl = pipeline_flux.get_bind_group_layout(0);
                let registry = ResourceRegistry::new().with_mesh(&runtime.common.mesh);
                wgsl_reflect::create_bind_group_from_bindings(
                    device,
                    "GenericCoupled: flux mesh bind group",
                    &bgl,
                    flux.bindings,
                    0,
                    |name| registry.resolve(name),
                )?
            };

            let bg_fields_flux_ping_pong = {
                let bgl = pipeline_flux.get_bind_group_layout(1);
                let mut out = Vec::new();
                for i in 0..3 {
                    let registry = ResourceRegistry::new()
                        .with_unified_fields(&fields)
                        .at_ping_pong_phase(i)
                        .with_constants_buffer(runtime.constants.buffer());
                    out.push(wgsl_reflect::create_bind_group_from_bindings(
                        device,
                        &format!("GenericCoupled flux fields bind group {i}"),
                        &bgl,
                        flux.bindings,
                        1,
                        |name| registry.resolve(name),
                    )?);
                }
                out
            };

            (Some(bg_mesh_flux), Some(bg_fields_flux_ping_pong))
        } else {
            (None, None)
        };

        // KT flux module bind groups.
        //
        // The current KT kernels are still backed by the legacy EI generator, but are
        // selected by model structure (`FluxModuleSpec::KurganovTadmor`).
        let (bg_mesh_kt, bg_fields_kt_ping_pong) = if let (
            Some(kt_gradients),
            Some(pipeline_kt_gradients),
            Some(flux_kt),
            Some(pipeline_flux_kt),
        ) = (
            kt_gradients.as_ref(),
            pipeline_kt_gradients.as_ref(),
            flux_kt.as_ref(),
            pipeline_flux_kt.as_ref(),
        ) {
            // Group 0: mesh
            let bg_mesh_kt = {
                let bgl = pipeline_flux_kt.get_bind_group_layout(0);
                let registry = ResourceRegistry::new().with_mesh(&runtime.common.mesh);
                wgsl_reflect::create_bind_group_from_bindings(
                    device,
                    "GenericCoupled: KT mesh bind group",
                    &bgl,
                    flux_kt.bindings,
                    0,
                    |name| registry.resolve(name),
                )?
            };

            // Group 1: fields
            let bg_fields_kt_ping_pong = {
                let bgl = pipeline_kt_gradients.get_bind_group_layout(1);
                let mut out = Vec::new();
                for i in 0..3 {
                    let registry = ResourceRegistry::new()
                        .with_unified_fields(&fields)
                        .at_ping_pong_phase(i)
                        .with_constants_buffer(runtime.constants.buffer());
                    out.push(wgsl_reflect::create_bind_group_from_bindings(
                        device,
                        &format!("GenericCoupled KT fields bind group {i}"),
                        &bgl,
                        kt_gradients.bindings,
                        1,
                        |name| registry.resolve(name),
                    )?);
                }
                out
            };

            (Some(bg_mesh_kt), Some(bg_fields_kt_ping_pong))
        } else {
            // KT expects both kernels present.
            (None, None)
        };

        let bg_fields_ping_pong = {
            let bgl = pipeline_assembly.get_bind_group_layout(1);
            let mut out = Vec::new();
            for i in 0..3 {
                let registry = ResourceRegistry::new()
                    .with_unified_fields(&fields)
                    .at_ping_pong_phase(i)
                    .with_constants_buffer(runtime.constants.buffer());
                out.push(wgsl_reflect::create_bind_group_from_bindings(
                    device,
                    &format!("GenericCoupled assembly fields bind group {i}"),
                    &bgl,
                    assembly_bindings,
                    1,
                    |name| registry.resolve(name),
                )?);
            }
            out
        };

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

        let bg_solver = {
            let bgl = pipeline_assembly.get_bind_group_layout(2);
            let registry = ResourceRegistry::new()
                .with_buffer(
                    "matrix_values",
                    runtime
                        .linear_port_space
                        .buffer(runtime.linear_ports.values),
                )
                .with_buffer(
                    "rhs",
                    runtime.linear_port_space.buffer(runtime.linear_ports.rhs),
                )
                .with_buffer("scalar_row_offsets", &b_scalar_row_offsets);
            wgsl_reflect::create_bind_group_from_bindings(
                device,
                "GenericCoupled assembly solver bind group",
                &bgl,
                assembly_bindings,
                2,
                |name| registry.resolve(name),
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

        let bg_bc_kt = if let (Some(flux_kt), Some(pipeline_flux_kt)) =
            (flux_kt.as_ref(), pipeline_flux_kt.as_ref())
        {
            Some({
                let bgl = pipeline_flux_kt.get_bind_group_layout(2);
                let registry = ResourceRegistry::new()
                    .with_buffer("bc_kind", &b_bc_kind)
                    .with_buffer("bc_value", &b_bc_value);
                wgsl_reflect::create_bind_group_from_bindings(
                    device,
                    "GenericCoupled: KT BC bind group",
                    &bgl,
                    flux_kt.bindings,
                    2,
                    |name| registry.resolve(name),
                )?
            })
        } else {
            None
        };

        let bg_bc = {
            let bgl = pipeline_assembly.get_bind_group_layout(3);
            let registry = ResourceRegistry::new()
                .with_buffer("bc_kind", &b_bc_kind)
                .with_buffer("bc_value", &b_bc_value);
            wgsl_reflect::create_bind_group_from_bindings(
                device,
                "GenericCoupled BC bind group",
                &bgl,
                assembly_bindings,
                3,
                |name| registry.resolve(name),
            )?
        };

        let bg_update_state_ping_pong = {
            let bgl = pipeline_update.get_bind_group_layout(0);
            let mut out = Vec::new();
            for i in 0..3 {
                let registry = ResourceRegistry::new()
                    .with_unified_fields(&fields)
                    .at_ping_pong_phase(i)
                    .with_constants_buffer(runtime.constants.buffer());
                out.push(wgsl_reflect::create_bind_group_from_bindings(
                    device,
                    &format!("GenericCoupled update state bind group {i}"),
                    &bgl,
                    update_bindings,
                    0,
                    |name| registry.resolve(name),
                )?);
            }
            out
        };

        let bg_update_solution = {
            let bgl = pipeline_update.get_bind_group_layout(1);
            let registry = ResourceRegistry::new().with_buffer(
                "x",
                runtime.linear_port_space.buffer(runtime.linear_ports.x),
            );
            wgsl_reflect::create_bind_group_from_bindings(
                device,
                "GenericCoupled update solution bind group",
                &bgl,
                update_bindings,
                1,
                |name| registry.resolve(name),
            )?
        };

        let kernels =
            crate::solver::gpu::modules::generic_coupled_kernels::GenericCoupledKernelsModule::new(
                fields.step_handle(),
                bg_mesh_flux,
                bg_fields_flux_ping_pong,
                bg_mesh_kt,
                bg_fields_kt_ping_pong,
                bg_bc_kt,
                bg_mesh,
                bg_fields_ping_pong,
                bg_solver,
                bg_bc,
                bg_update_state_ping_pong,
                bg_update_solution,
                pipeline_flux,
                pipeline_kt_gradients,
                pipeline_flux_kt,
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
