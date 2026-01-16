use bytemuck::cast_slice;
use wgpu::util::DeviceExt;

use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::lowering::models::generic_coupled::GenericCoupledProgramResources;
use crate::solver::gpu::modules::generated_kernels::GeneratedKernelsModule;
use crate::solver::gpu::modules::resource_registry::ResourceRegistry;
use crate::solver::gpu::modules::unified_field_resources::UnifiedFieldResources;
use crate::solver::gpu::recipe::SolverRecipe;
use crate::solver::gpu::runtime::GpuCsrRuntime;
use crate::solver::gpu::structs::GpuConstants;
use crate::solver::mesh::{BoundaryType, Mesh};
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
    let eos_params = model.eos.runtime_params();
    let mut initial_constants = GpuConstants::default();
    initial_constants.eos_gamma = eos_params.gamma;
    initial_constants.eos_gm1 = eos_params.gm1;
    initial_constants.eos_r = eos_params.r;
    initial_constants.eos_dp_drho = eos_params.dp_drho;
    initial_constants.eos_p_offset = eos_params.p_offset;
    initial_constants.eos_theta_ref = eos_params.theta_ref;
    let fields = UnifiedFieldResources::from_recipe(
        device,
        &recipe,
        num_cells,
        runtime.common.num_faces,
        stride,
        initial_constants,
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
    let boundary_count = 5usize; // None, Inlet, Outlet, Wall, SlipWall

    let mut boundary_faces: Vec<Vec<u32>> = vec![Vec::new(); boundary_count];
    for (face_idx, neigh) in mesh.face_neighbor.iter().enumerate() {
        if neigh.is_some() {
            continue; // interior face
        }
        let boundary_idx = match mesh.face_boundary.get(face_idx).copied().flatten() {
            None => 0usize,
            Some(BoundaryType::Inlet) => 1usize,
            Some(BoundaryType::Outlet) => 2usize,
            Some(BoundaryType::Wall) => 3usize,
            Some(BoundaryType::SlipWall) => 4usize,
        };
        boundary_faces[boundary_idx].push(face_idx as u32);
    }

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
            Some(BoundaryType::Inlet) => 1usize,
            Some(BoundaryType::Outlet) => 2usize,
            Some(BoundaryType::Wall) => 3usize,
            Some(BoundaryType::SlipWall) => 4usize,
        };

        // Only boundary faces use bc_kind/bc_value at runtime; interior entries are ignored.
        if boundary_idx == 0 {
            continue;
        }

        let src_base = boundary_idx * coupled_stride;
        let dst_base = face_idx * coupled_stride;
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
        boundary_faces,
    )?;

    Ok(GenericCoupledBuilt {
        model,
        context,
        profiling_stats,
        backend,
    })
}
