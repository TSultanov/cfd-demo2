pub mod fields;
pub mod linear_solver;
pub mod mesh;
pub mod scalars;

use crate::solver::mesh::Mesh;
use std::sync::Mutex;

use crate::solver::gpu::bindings::generated::coupled_assembly_merged as generated_coupled_assembly;
use crate::solver::gpu::modules::incompressible_kernels::IncompressibleKernelsModule;

use super::runtime_common::GpuRuntimeCommon;
use super::structs::{GpuSolver, PreconditionerType};

impl GpuSolver {
    pub async fn new(
        mesh: &Mesh,
        device: Option<wgpu::Device>,
        queue: Option<wgpu::Queue>,
    ) -> Self {
        let common = GpuRuntimeCommon::new(mesh, device, queue).await;

        let num_cells = common.num_cells;
        let num_faces = common.num_faces;

        // 2. Initialize Field Buffers (phase 1 - before pipelines)
        let field_buffers = fields::init_field_buffers(&common.context.device, num_cells, num_faces);

        // 3. Initialize Linear Solver
        let linear_res = linear_solver::init_linear_solver(
            &common.context.device,
            num_cells,
            &common.mesh.row_offsets,
            &common.mesh.col_indices,
        );

        // 5. Create fields bind groups (phase 2)
        let bgl_fields = common.context.device.create_bind_group_layout(
            &generated_coupled_assembly::WgpuBindGroup1::LAYOUT_DESCRIPTOR,
        );
        let fields_res =
            fields::create_field_bind_groups(&common.context.device, field_buffers, &bgl_fields);

        let incompressible_kernels = IncompressibleKernelsModule::new(
            &common.context.device,
            &common.mesh,
            &fields_res,
            &linear_res.coupled_resources,
        );

        let mut solver = Self {
            common,

            // Fields (consolidated FluidState buffers)
            b_state: fields_res.b_state,
            b_state_old: fields_res.b_state_old,
            b_state_old_old: fields_res.b_state_old_old,
            state_buffers: fields_res.state_buffers,
            state_step_index: 0,
            b_fluxes: fields_res.b_fluxes,
            b_constants: fields_res.b_constants,
            constants: fields_res.constants,
            preconditioner: PreconditionerType::Jacobi,
            scheme_needs_gradients: false,
            incompressible_kernels,

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
            fgmres_resources: None,
            n_outer_correctors: 20,
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
        };
        solver.update_needs_gradients();
        solver
    }
}
pub mod compressible_fields;
