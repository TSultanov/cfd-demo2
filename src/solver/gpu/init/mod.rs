pub mod compressible_fields;
pub mod fields;
pub mod linear_solver;
pub mod mesh;
pub mod scalars;

use crate::solver::mesh::Mesh;
use std::sync::Mutex;

use crate::solver::gpu::bindings::generated::coupled_assembly_merged as generated_coupled_assembly;
use crate::solver::gpu::modules::model_kernels::ModelKernelsModule;
use crate::solver::gpu::modules::time_integration::TimeIntegrationModule;
use crate::solver::gpu::plans::incompressible_linear_solver::IncompressibleLinearSolver;
use crate::solver::gpu::recipe::SolverRecipe;
use crate::solver::model::ModelSpec;

use super::runtime_common::GpuRuntimeCommon;
use super::structs::{GpuSolver, PreconditionerType};

impl GpuSolver {
    pub async fn new(
        mesh: &Mesh,
        model: ModelSpec,
        recipe: SolverRecipe,
        device: Option<wgpu::Device>,
        queue: Option<wgpu::Queue>,
    ) -> Result<Self, String> {
        let common = GpuRuntimeCommon::new(mesh, device, queue).await;

        let num_cells = common.num_cells;
        let num_faces = common.num_faces;
        let state_stride = model.state_layout.stride();
        let unknowns_per_cell = model.system.unknowns_per_cell();

        // Use recipe to determine if gradients are needed
        let scheme_needs_gradients = recipe.needs_gradients();

        // Use recipe stepping mode for outer correctors
        let n_outer_correctors = match recipe.stepping {
            crate::solver::gpu::recipe::SteppingMode::Coupled { outer_correctors } => outer_correctors,
            _ => 20, // default fallback
        };

        // 2. Initialize Field Buffers (phase 1 - before pipelines)
        let field_buffers =
            fields::init_field_buffers(&common.context.device, num_cells, num_faces, state_stride);

        // 3. Initialize Linear Solver
        let linear_res = linear_solver::init_linear_solver(
            &common.context.device,
            num_cells,
            &common.mesh.row_offsets,
            &common.mesh.col_indices,
            unknowns_per_cell,
        );

        // 5. Create fields bind groups (phase 2)
        let bgl_fields = common.context.device.create_bind_group_layout(
            &generated_coupled_assembly::WgpuBindGroup1::LAYOUT_DESCRIPTOR,
        );
        let fields_res =
            fields::create_field_bind_groups(&common.context.device, field_buffers, &bgl_fields);

        let kernels = ModelKernelsModule::new_incompressible(
            &common.context.device,
            &common.mesh,
            &fields_res,
            fields_res.state.step_handle(),
            &linear_res.coupled_resources,
        );

        let mut solver = Self {
            common,
            model,

            fields: fields_res,
            preconditioner: PreconditionerType::Jacobi,
            scheme_needs_gradients,
            kernels,

            num_nonzeros: linear_res.num_nonzeros,

            // Misc
            num_cells,
            num_faces,
            stats_ux: Mutex::new(Default::default()),
            stats_uy: Mutex::new(Default::default()),
            stats_p: Mutex::new(Default::default()),
            outer_residual_u: Mutex::new(0.0),
            outer_residual_p: Mutex::new(0.0),
            outer_iterations: Mutex::new(0),
            linear_solver: IncompressibleLinearSolver::new(),
            n_outer_correctors,
            coupled_resources: Some(linear_res.coupled_resources),
            coupled_should_clear_max_diff: false,
            coupled_last_linear_stats: Default::default(),
            variance_history: Vec::new(),
            degenerate_count: 0,
            prev_u_cpu: Vec::new(),
            steady_state_count: 0,
            should_stop: false,
            coupled_init_prepare_graph: GpuSolver::build_coupled_init_prepare_graph(),
            coupled_prepare_assembly_graph: GpuSolver::build_coupled_prepare_assembly_graph(),
            coupled_assembly_graph: GpuSolver::build_coupled_assembly_graph(),
            coupled_update_graph: GpuSolver::build_coupled_update_graph(),

            linear_ports: linear_res.ports,
            linear_port_space: linear_res.port_space,
            scalar_cg: linear_res.scalar_cg,
            time_integration: TimeIntegrationModule::new(),
        };
        // No longer need to call update_needs_gradients - recipe provides this
        Ok(solver)
    }
}
