pub mod linear_solver;
pub mod mesh;
pub mod scalars;

use crate::solver::mesh::Mesh;
use std::sync::Mutex;

use crate::solver::gpu::coupled_backend::linear_solver::IncompressibleLinearSolver;
use crate::solver::gpu::modules::model_kernels::ModelKernelsModule;
use crate::solver::gpu::modules::time_integration::TimeIntegrationModule;
use crate::solver::gpu::modules::unified_field_resources::UnifiedFieldResources;
use crate::solver::gpu::recipe::SolverRecipe;
use crate::solver::model::ModelSpec;

use super::runtime_common::GpuRuntimeCommon;
use super::structs::{GpuConstants, GpuSolver, PreconditionerType};

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
            crate::solver::gpu::recipe::SteppingMode::Coupled { outer_correctors } => {
                outer_correctors
            }
            _ => 20, // default fallback
        };

        let initial_constants = GpuConstants {
            dt: 0.0001,
            dt_old: 0.0001,
            dtau: 0.0,
            time: 0.0,
            viscosity: 0.01,
            density: 1.0,
            component: 0,
            alpha_p: 1.0,
            scheme: 0,
            alpha_u: 0.7,
            stride_x: 65535 * 64,
            time_scheme: 0,
            inlet_velocity: 1.0,
            ramp_time: 0.1,
        };

        // 3. Initialize Linear Solver
        let linear_res = linear_solver::init_linear_solver(
            &common.context.device,
            num_cells,
            &common.mesh.row_offsets,
            &common.mesh.col_indices,
            unknowns_per_cell,
        );

        // Allocate all required fields directly from the recipe (including face fluxes).
        let fields_res = UnifiedFieldResources::from_recipe(
            &common.context.device,
            &recipe,
            num_cells,
            num_faces,
            state_stride,
            initial_constants,
        );

        let kernels = ModelKernelsModule::new_incompressible(
            &common.context.device,
            &common.mesh,
            &fields_res,
            fields_res.step_handle(),
            &linear_res.coupled_resources,
        );

        let solver = Self {
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
