// Coupled Solver for CFD
//
// This solver solves the momentum and continuity equations simultaneously
// in a block-coupled manner, as opposed to segregated predictor-corrector approaches.
//
// The coupled approach forms a larger block system:
// | A_u  G  | | u |   | b_u |
// | D    0  | | p | = | 0   |
//
// Where:
// - A_u is the momentum matrix (discretized convection + diffusion + time derivative)
// - G is the gradient operator (pressure gradient contribution to momentum)
// - D is the divergence operator (mass flux in continuity)
// - u is the velocity field
// - p is the pressure field
// - b_u is the momentum source term

use crate::solver::gpu::modules::graph::{ComputeSpec, DispatchKind, ModuleGraph, ModuleNode};
use crate::solver::gpu::modules::model_kernels::{
    KernelBindGroups, KernelPipeline, ModelKernelsModule,
};
use crate::solver::gpu::structs::{GpuSolver, LinearSolverStats};
use crate::solver::model::KernelKind;

impl GpuSolver {
    pub(crate) fn build_coupled_init_prepare_graph() -> ModuleGraph<ModelKernelsModule> {
        ModuleGraph::new(vec![ModuleNode::Compute(ComputeSpec {
            label: "coupled:init_prepare",
            pipeline: KernelPipeline::Kernel(KernelKind::PrepareCoupled),
            bind: KernelBindGroups::MeshFieldsSolver,
            dispatch: DispatchKind::Cells,
        })])
    }

    pub(crate) fn build_coupled_prepare_assembly_graph() -> ModuleGraph<ModelKernelsModule> {
        ModuleGraph::new(vec![
            ModuleNode::Compute(ComputeSpec {
                label: "coupled:prepare",
                pipeline: KernelPipeline::Kernel(KernelKind::PrepareCoupled),
                bind: KernelBindGroups::MeshFieldsSolver,
                dispatch: DispatchKind::Cells,
            }),
            ModuleNode::Compute(ComputeSpec {
                label: "coupled:assembly_merged",
                pipeline: KernelPipeline::Kernel(KernelKind::CoupledAssembly),
                bind: KernelBindGroups::MeshFieldsSolver,
                dispatch: DispatchKind::Cells,
            }),
        ])
    }

    pub(crate) fn build_coupled_assembly_graph() -> ModuleGraph<ModelKernelsModule> {
        ModuleGraph::new(vec![ModuleNode::Compute(ComputeSpec {
            label: "coupled:assembly_merged",
            pipeline: KernelPipeline::Kernel(KernelKind::CoupledAssembly),
            bind: KernelBindGroups::MeshFieldsSolver,
            dispatch: DispatchKind::Cells,
        })])
    }

    pub(crate) fn build_coupled_update_graph() -> ModuleGraph<ModelKernelsModule> {
        ModuleGraph::new(vec![ModuleNode::Compute(ComputeSpec {
            label: "coupled:update_fields_max_diff",
            pipeline: KernelPipeline::Kernel(KernelKind::UpdateFieldsFromCoupled),
            bind: KernelBindGroups::UpdateFieldsSolution,
            dispatch: DispatchKind::Cells,
        })])
    }

    pub(crate) fn check_evolution(&mut self) {
        let quiet = std::env::var("CFD2_QUIET").ok().as_deref() == Some("1");

        // Read velocity field to check for evolution and variance
        // This involves a GPU->CPU read, so it has some overhead.
        let u_data =
            pollster::block_on(self.read_buffer_f32(self.fields.state.state(), self.num_cells * 8)); // FluidState is 8 f32s = 32 bytes

        // 1. Calculate Variance
        let mut sum_u = 0.0;
        let mut sum_v = 0.0;
        let mut sum_sq_u = 0.0;
        let mut sum_sq_v = 0.0;
        let n = self.num_cells as f64;

        for i in 0..self.num_cells as usize {
            let u = u_data[i * 2] as f64;
            let v = u_data[i * 2 + 1] as f64;
            sum_u += u;
            sum_v += v;
            sum_sq_u += u * u;
            sum_sq_v += v * v;
        }

        let mean_u = sum_u / n;
        let mean_v = sum_v / n;
        let var_u = (sum_sq_u / n - mean_u * mean_u).max(0.0);
        let var_v = (sum_sq_v / n - mean_v * mean_v).max(0.0);

        self.variance_history.push((var_u, var_v));
        if self.variance_history.len() > 10 {
            self.variance_history.remove(0);
        }

        // 2. Calculate Evolution (Change from previous step)
        let mut evolution_diff = 0.0;
        if !self.prev_u_cpu.is_empty() && self.prev_u_cpu.len() == u_data.len() {
            for i in 0..u_data.len() {
                let diff = u_data[i] - self.prev_u_cpu[i];
                evolution_diff += (diff * diff) as f64;
            }
            evolution_diff = (evolution_diff / n).sqrt(); // RMSE
        } else {
            evolution_diff = f64::MAX;
        }

        // Update prev_u_cpu
        self.prev_u_cpu = u_data;

        // 3. Check Conditions
        let evolution_threshold = 1e-6; // Threshold for "stopped evolving"
        let variance_threshold = 1e-10; // Threshold for "uniform field"

        if evolution_diff < evolution_threshold {
            if var_u < variance_threshold && var_v < variance_threshold {
                // Stopped evolving AND Uniform -> Degenerate
                self.degenerate_count += 1;
                self.steady_state_count = 0;
            } else {
                // Stopped evolving AND Non-Uniform -> Steady State
                self.steady_state_count += 1;
                self.degenerate_count = 0;
            }
        } else {
            // Still evolving
            self.degenerate_count = 0;
            self.steady_state_count = 0;
        }

        // 4. Act
        if self.degenerate_count > 10 {
            if !quiet {
                println!(
                    "Solution is degenerate: Velocity field is uniform and not evolving. Variance U: {:.2e}, V: {:.2e}",
                    var_u, var_v
                );
            }
            self.should_stop = true;
        }

        if self.steady_state_count > 10 {
            if !quiet {
                println!(
                    "Steady state reached. Evolution diff: {:.2e}",
                    evolution_diff
                );
            }
            self.should_stop = true;
        }
    }

    pub(crate) fn solve_coupled_system(&mut self) -> LinearSolverStats {
        if let Some(res) = &self.coupled_resources {
            self.linear_solver.solve(
                &self.common.context,
                &self.common.readback_cache,
                &self.common.profiling_stats,
                res,
                self.linear_ports,
                &self.linear_port_space,
                self.num_cells,
                self.num_nonzeros,
                self.preconditioner,
            )
        } else {
            LinearSolverStats::default()
        }
    }
}