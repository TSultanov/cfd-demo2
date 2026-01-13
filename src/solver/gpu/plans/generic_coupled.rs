use bytemuck::cast_slice;
use wgpu::util::DeviceExt;

use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::lowering::models::generic_coupled::{
    GenericCoupledProgramResources,
};
use crate::solver::gpu::modules::generated_kernels::GeneratedKernelsModule;
use crate::solver::gpu::modules::resource_registry::ResourceRegistry;
use crate::solver::gpu::modules::unified_field_resources::UnifiedFieldResources;
use crate::solver::gpu::recipe::SolverRecipe;
use crate::solver::gpu::runtime::GpuCsrRuntime;
use crate::solver::gpu::structs::GpuConstants;
use crate::solver::mesh::Mesh;
use crate::solver::model::ModelSpec;

pub(crate) struct GenericCoupledBuilt {
    pub model: ModelSpec,
    pub context: GpuContext,
    pub profiling_stats: std::sync::Arc<crate::solver::gpu::profiling::ProfilingStats>,
    pub backend: GenericCoupledProgramResources,
}

pub(crate) async fn build_generic_coupled_backend(
    mesh: &Mesh,
    model: ModelSpec,
    recipe: SolverRecipe,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
) -> Result<GenericCoupledBuilt, String> {
    let unknowns_per_cell = model.system.unknowns_per_cell();
    let runtime = GpuCsrRuntime::new(mesh, unknowns_per_cell, device, queue).await;

    let device = &runtime.common.context.device;

    let stride = model.state_layout.stride();
    let num_cells = runtime.common.num_cells;

    // Create unified field resources from recipe.
    let fields = UnifiedFieldResources::from_recipe(
        device,
        &recipe,
        num_cells,
        runtime.common.num_faces,
        stride,
        GpuConstants::default(),
    );

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
        .with_buffer(
            "matrix_values",
            runtime.linear_port_space.buffer(runtime.linear_ports.values),
        )
        .with_buffer(
            "rhs",
            runtime.linear_port_space.buffer(runtime.linear_ports.rhs),
        )
        .with_buffer("x", runtime.linear_port_space.buffer(runtime.linear_ports.x))
        .with_buffer(
            "row_offsets",
            runtime
                .linear_port_space
                .buffer(runtime.linear_ports.row_offsets),
        )
        .with_buffer(
            "col_indices",
            runtime
                .linear_port_space
                .buffer(runtime.linear_ports.col_indices),
        )
        .with_buffer("bc_kind", &b_bc_kind)
        .with_buffer("bc_value", &b_bc_value)
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

    let backend = GenericCoupledProgramResources::new(
        runtime,
        fields,
        kernels,
        &model,
        &recipe,
        b_bc_kind,
        b_bc_value,
    )?;

    Ok(GenericCoupledBuilt {
        model,
        context,
        profiling_stats,
        backend,
    })
}
