use bytemuck::cast_slice;
use wgpu::util::DeviceExt;

use cfd2_codegen::solver::codegen::bc_table::{HostBcTable, BOUNDARY_TYPE_COUNT};

use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::lowering::programs::generic_coupled::{
    BoundaryConditionData, GenericCoupledProgramResources,
};
use crate::solver::gpu::modules::generated_kernels::GeneratedKernelsModule;
use crate::solver::gpu::modules::resource_registry::ResourceRegistry;
use crate::solver::gpu::modules::unified_field_resources::UnifiedFieldResources;
use crate::solver::gpu::recipe::SolverRecipe;
use crate::solver::gpu::runtime::GpuCsrRuntime;
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
    let unknowns_per_cell: u32 = recipe
        .unknowns_per_cell
        .try_into()
        .map_err(|_| "recipe.unknowns_per_cell overflows u32".to_string())?;
    // Stage-1 groundwork: build paths reserve EXACT capacity (byte-neutral —
    // sized bindings at full size == whole-buffer bindings). A refresh/churn
    // context can later request headroom here without touching this seam.
    let runtime = GpuCsrRuntime::new(
        mesh,
        unknowns_per_cell,
        device,
        queue,
        crate::solver::gpu::capacity::CapacityPlan::EXACT,
    )
    .await?;

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
        recipe.initial_constants,
    );

    // Boundary-condition buffers are stored per-face x unknown-component so flux modules
    // and assembly use the same indexing semantics.
    //
    // We expand the model-defined boundary-type tables onto boundary faces.
    let (bc_kind_by_type, bc_value_by_type) = model
        .boundaries
        .to_gpu_tables(&model.system)
        .map_err(|e| format!("failed to build BC tables: {e}"))?;

    let coupled_stride = unknowns_per_cell as usize;
    let mut boundary_faces: Vec<Vec<u32>> = vec![Vec::new(); BOUNDARY_TYPE_COUNT];
    for (face_idx, neigh) in mesh.face_neighbor.iter().enumerate() {
        if neigh.is_some() {
            continue; // interior face
        }
        let boundary_idx = match mesh.face_boundary.get(face_idx).copied().flatten() {
            None => 0usize,
            Some(bt) => bt.bc_table_index(),
        };
        boundary_faces[boundary_idx].push(face_idx as u32);
    }

    let table = HostBcTable::new(coupled_stride);
    let num_faces = runtime.common.num_faces as usize;
    let mut bc_kind = vec![0u32; num_faces * coupled_stride];
    let mut bc_value = vec![0.0_f32; num_faces * coupled_stride];
    for face_idx in 0..num_faces {
        if mesh
            .face_neighbor
            .get(face_idx)
            .copied()
            .unwrap_or(None)
            .is_some()
        {
            continue; // interior face
        }

        let boundary_idx = match mesh.face_boundary.get(face_idx).copied().flatten() {
            None => 0usize,
            Some(bt) => bt.bc_table_index(),
        };

        // Only boundary faces use bc_kind/bc_value at runtime; interior entries are ignored.
        if boundary_idx == 0 {
            continue;
        }

        let src_base = table.row_base(boundary_idx);
        let dst_base = table.row_base(face_idx);
        bc_kind[dst_base..dst_base + coupled_stride]
            .copy_from_slice(&bc_kind_by_type[src_base..src_base + coupled_stride]);
        bc_value[dst_base..dst_base + coupled_stride]
            .copy_from_slice(&bc_value_by_type[src_base..src_base + coupled_stride]);
    }

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
            runtime
                .linear_port_space
                .buffer(runtime.linear_ports.values),
        )
        .with_buffer(
            "rhs",
            runtime.linear_port_space.buffer(runtime.linear_ports.rhs),
        )
        .with_buffer(
            "x",
            runtime.linear_port_space.buffer(runtime.linear_ports.x),
        )
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
        .with_buffer(
            "y",
            runtime.linear_port_space.buffer(runtime.linear_ports.rhs),
        );

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
        timestamp_query: runtime.common.context.timestamp_query,
        timestamps_inside_encoders: runtime.common.context.timestamps_inside_encoders,
        timestamp_period_ns: runtime.common.context.timestamp_period_ns,
    };
    let profiling_stats = std::sync::Arc::clone(&runtime.common.profiling_stats);

    let bc_data = BoundaryConditionData {
        b_bc_kind,
        b_bc_value,
        boundary_faces,
    };
    let backend =
        GenericCoupledProgramResources::new(runtime, fields, kernels, &model, &recipe, bc_data)?;

    Ok(GenericCoupledBuilt {
        model,
        context,
        profiling_stats,
        backend,
    })
}
