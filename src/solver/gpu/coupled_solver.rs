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

use super::context::GpuContext;
use super::execution_plan::{ExecutionPlan, GraphExecMode, GraphNode, HostNode, PlanNode};
use super::kernel_graph::{ComputeNode, KernelGraph, KernelNode};
use super::profiling::ProfileCategory;
use super::structs::{GpuSolver, LinearSolverStats};
use std::time::Instant;

/// Enable debug reads for diagnosing matrix assembly issues.
/// Set to true to enable debug output on the first iteration.
/// WARNING: This adds significant GPU-CPU synchronization overhead (~65ms per step).
const DEBUG_READS_ENABLED: bool = false;

fn coupled_workgroups_cells(solver: &GpuSolver) -> (u32, u32, u32) {
    let workgroup_size = 64;
    (solver.num_cells.div_ceil(workgroup_size), 1, 1)
}

fn coupled_bind_mesh_fields_solver(solver: &GpuSolver, cpass: &mut wgpu::ComputePass) {
    let res = solver
        .coupled_resources
        .as_ref()
        .expect("Coupled resources not initialized");
    cpass.set_bind_group(0, &solver.bg_mesh, &[]);
    cpass.set_bind_group(1, &solver.bg_fields, &[]);
    cpass.set_bind_group(2, &res.bg_solver, &[]);
}

fn coupled_bind_fields_coupled_solution(solver: &GpuSolver, cpass: &mut wgpu::ComputePass) {
    let res = solver
        .coupled_resources
        .as_ref()
        .expect("Coupled resources not initialized");
    cpass.set_bind_group(0, &solver.bg_fields, &[]);
    cpass.set_bind_group(1, &res.bg_coupled_solution, &[]);
}

fn coupled_context(solver: &GpuSolver) -> &GpuContext {
    &solver.context
}

fn coupled_graph_init_prepare(solver: &GpuSolver) -> &KernelGraph<GpuSolver> {
    &solver.coupled_init_prepare_graph
}

fn coupled_graph_prepare_assembly(solver: &GpuSolver) -> &KernelGraph<GpuSolver> {
    &solver.coupled_prepare_assembly_graph
}

fn coupled_graph_assembly(solver: &GpuSolver) -> &KernelGraph<GpuSolver> {
    &solver.coupled_assembly_graph
}

fn coupled_graph_update(solver: &GpuSolver) -> &KernelGraph<GpuSolver> {
    &solver.coupled_update_graph
}

impl GpuSolver {
    pub(super) fn build_coupled_init_prepare_graph() -> KernelGraph<GpuSolver> {
        KernelGraph::new(vec![KernelNode::Compute(ComputeNode {
            label: "coupled:init_prepare",
            pipeline: |s| &s.pipeline_prepare_coupled,
            bind_groups: coupled_bind_mesh_fields_solver,
            workgroups: coupled_workgroups_cells,
        })])
    }

    pub(super) fn build_coupled_prepare_assembly_graph() -> KernelGraph<GpuSolver> {
        KernelGraph::new(vec![
            KernelNode::Compute(ComputeNode {
                label: "coupled:prepare",
                pipeline: |s| &s.pipeline_prepare_coupled,
                bind_groups: coupled_bind_mesh_fields_solver,
                workgroups: coupled_workgroups_cells,
            }),
            KernelNode::Compute(ComputeNode {
                label: "coupled:assembly_merged",
                pipeline: |s| &s.pipeline_coupled_assembly_merged,
                bind_groups: coupled_bind_mesh_fields_solver,
                workgroups: coupled_workgroups_cells,
            }),
        ])
    }

    pub(super) fn build_coupled_assembly_graph() -> KernelGraph<GpuSolver> {
        KernelGraph::new(vec![KernelNode::Compute(ComputeNode {
            label: "coupled:assembly_merged",
            pipeline: |s| &s.pipeline_coupled_assembly_merged,
            bind_groups: coupled_bind_mesh_fields_solver,
            workgroups: coupled_workgroups_cells,
        })])
    }

    pub(super) fn build_coupled_update_graph() -> KernelGraph<GpuSolver> {
        KernelGraph::new(vec![KernelNode::Compute(ComputeNode {
            label: "coupled:update_fields_max_diff",
            pipeline: |s| &s.pipeline_update_from_coupled,
            bind_groups: coupled_bind_fields_coupled_solution,
            workgroups: coupled_workgroups_cells,
        })])
    }

    fn coupled_host_solve(solver: &mut GpuSolver) {
        solver.coupled_last_linear_stats = solver.solve_coupled_system();
    }

    fn coupled_host_set_component_0(solver: &mut GpuSolver) {
        solver.constants.component = 0;
        solver.update_constants();
    }

    fn coupled_host_clear_max_diff(solver: &mut GpuSolver) {
        if !solver.coupled_should_clear_max_diff {
            return;
        }
        let Some(res) = solver.coupled_resources.as_ref() else {
            return;
        };
        solver
            .context
            .queue
            .write_buffer(&res.b_max_diff_result, 0, &[0u8; 8]);
    }

    fn build_coupled_iter_plan_prepare_assembly_solve() -> ExecutionPlan<GpuSolver> {
        ExecutionPlan::new(
            coupled_context,
            vec![
                PlanNode::Graph(GraphNode {
                    label: "coupled:gradient_assembly_merged",
                    graph: coupled_graph_prepare_assembly,
                    mode: GraphExecMode::SingleSubmit,
                }),
                PlanNode::Host(HostNode {
                    label: "coupled:solve_coupled_system",
                    run: GpuSolver::coupled_host_solve,
                }),
            ],
        )
    }

    fn build_coupled_init_plan() -> ExecutionPlan<GpuSolver> {
        ExecutionPlan::new(
            coupled_context,
            vec![
                PlanNode::Host(HostNode {
                    label: "coupled:set_component_0",
                    run: GpuSolver::coupled_host_set_component_0,
                }),
                PlanNode::Graph(GraphNode {
                    label: "coupled:init_prepare",
                    graph: coupled_graph_init_prepare,
                    mode: GraphExecMode::SingleSubmit,
                }),
            ],
        )
    }

    fn build_coupled_iter_plan_assembly_solve() -> ExecutionPlan<GpuSolver> {
        ExecutionPlan::new(
            coupled_context,
            vec![
                PlanNode::Graph(GraphNode {
                    label: "coupled:gradient_assembly_merged",
                    graph: coupled_graph_assembly,
                    mode: GraphExecMode::SingleSubmit,
                }),
                PlanNode::Host(HostNode {
                    label: "coupled:solve_coupled_system",
                    run: GpuSolver::coupled_host_solve,
                }),
            ],
        )
    }

    fn build_coupled_update_plan() -> ExecutionPlan<GpuSolver> {
        ExecutionPlan::new(
            coupled_context,
            vec![
                PlanNode::Host(HostNode {
                    label: "coupled:clear_max_diff",
                    run: GpuSolver::coupled_host_clear_max_diff,
                }),
                PlanNode::Graph(GraphNode {
                    label: "coupled:update_fields_max_diff",
                    graph: coupled_graph_update,
                    mode: GraphExecMode::SingleSubmit,
                }),
            ],
        )
    }

    fn coupled_iter_plan_prepare_assembly_solve() -> &'static ExecutionPlan<GpuSolver> {
        static PLAN: std::sync::OnceLock<ExecutionPlan<GpuSolver>> = std::sync::OnceLock::new();
        PLAN.get_or_init(GpuSolver::build_coupled_iter_plan_prepare_assembly_solve)
    }

    fn coupled_init_plan() -> &'static ExecutionPlan<GpuSolver> {
        static PLAN: std::sync::OnceLock<ExecutionPlan<GpuSolver>> = std::sync::OnceLock::new();
        PLAN.get_or_init(GpuSolver::build_coupled_init_plan)
    }

    fn coupled_iter_plan_assembly_solve() -> &'static ExecutionPlan<GpuSolver> {
        static PLAN: std::sync::OnceLock<ExecutionPlan<GpuSolver>> = std::sync::OnceLock::new();
        PLAN.get_or_init(GpuSolver::build_coupled_iter_plan_assembly_solve)
    }

    fn coupled_update_plan() -> &'static ExecutionPlan<GpuSolver> {
        static PLAN: std::sync::OnceLock<ExecutionPlan<GpuSolver>> = std::sync::OnceLock::new();
        PLAN.get_or_init(GpuSolver::build_coupled_update_plan)
    }

    /// Performs a single timestep using the coupled solver approach.
    ///
    /// Unlike segregated predictors that iterate between momentum and pressure solves,
    /// the coupled solver assembles and solves the full block system in one go,
    /// with outer iterations for non-linearity.
    pub fn step_coupled(&mut self) {
        let quiet = std::env::var("CFD2_QUIET").ok().as_deref() == Some("1");

        // We need to access coupled resources. If not available, return.
        if self.coupled_resources.is_none() {
            if !quiet {
                println!("Coupled resources not initialized!");
            }
            return;
        }

        // Ping-pong rotation
        self.state_step_index = (self.state_step_index + 1) % 3;

        // Update bind group
        self.bg_fields = self.bg_fields_ping_pong[self.state_step_index].clone();

        // Update buffer references
        let idx_state = match self.state_step_index {
            0 => 0,
            1 => 2,
            2 => 1,
            _ => 0,
        };
        let idx_state_old = match self.state_step_index {
            0 => 1,
            1 => 0,
            2 => 2,
            _ => 0,
        };
        let idx_state_old_old = match self.state_step_index {
            0 => 2,
            1 => 1,
            2 => 0,
            _ => 0,
        };

        self.b_state = self.state_buffers[idx_state].clone();
        self.b_state_old = self.state_buffers[idx_state_old].clone();
        self.b_state_old_old = self.state_buffers[idx_state_old_old].clone();

        // Initialize fluxes and d_p (and gradients)
        {
            let init_dispatch_start = Instant::now();
            GpuSolver::coupled_init_plan().execute(self);
            self.profiling_stats.record_location(
                "coupled:init_prepare",
                ProfileCategory::GpuDispatch,
                init_dispatch_start.elapsed(),
                0,
            );
        }

        // Coupled iteration loop - use more iterations for robustness
        let max_coupled_iters = self.n_outer_correctors.max(10);
        let convergence_tol_u = 1e-5;
        let convergence_tol_p = 1e-4;

        let mut prev_residual_u = f64::MAX;
        let mut prev_residual_p = f64::MAX;

        // Reset async reader to clear old values
        if let Some(res) = self.coupled_resources.as_ref() {
            res.async_scalar_reader.borrow_mut().reset();
        }

        for iter in 0..max_coupled_iters {
            self.profiling_stats.increment_iteration();
            let io_start = Instant::now();
            if !quiet {
                println!("Coupled Iteration: {}", iter + 1);
            }
            self.profiling_stats.record_location(
                "coupled:println_iteration",
                ProfileCategory::CpuCompute,
                io_start.elapsed(),
                0,
            );

            // Debug: Check d_p values on first iteration - DEBUG READ
            if DEBUG_READS_ENABLED && iter == 0 {
                let read_start = Instant::now();
                let d_p_vals = pollster::block_on(self.get_d_p());
                self.profiling_stats.record_location(
                    "coupled:debug_get_d_p",
                    ProfileCategory::GpuRead,
                    read_start.elapsed(),
                    (self.num_cells as u64) * 4,
                );
                let min_dp = d_p_vals.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_dp = d_p_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let avg_dp: f64 = d_p_vals.iter().sum::<f64>() / d_p_vals.len() as f64;
                if !quiet {
                    println!(
                        "d_p stats: min={:.2e}, max={:.2e}, avg={:.2e}",
                        min_dp, max_dp, avg_dp
                    );
                }
            }

            // Compute Gradients AND Assemble Coupled System (Merged Dispatch)
            if self.coupled_resources.is_some() {
                // Merged Prepare Coupled Pass (Flux, D_P, Grad P, Grad U, Grad V)
                let needs_prepare = iter > 0 || self.constants.scheme != 0;
                if needs_prepare {
                    // Ensure component is 0 for d_p calculation (if needed)
                    if iter > 0 {
                        self.constants.component = 0;
                        let update_const_start = Instant::now();
                        self.update_constants();
                        self.profiling_stats.record_location(
                            "coupled:update_constants",
                            ProfileCategory::GpuWrite,
                            update_const_start.elapsed(),
                            std::mem::size_of_val(&self.constants) as u64,
                        );
                    }
                }

                let timings = if needs_prepare {
                    GpuSolver::coupled_iter_plan_prepare_assembly_solve().execute(self)
                } else {
                    GpuSolver::coupled_iter_plan_assembly_solve().execute(self)
                };

                let assembly_secs = timings.seconds_for("coupled:gradient_assembly_merged");
                if assembly_secs > 0.0 {
                    self.profiling_stats.record_location(
                        "coupled:gradient_assembly_merged",
                        ProfileCategory::GpuDispatch,
                        std::time::Duration::from_secs_f64(assembly_secs),
                        0,
                    );
                }
                let solve_secs = timings.seconds_for("coupled:solve_coupled_system");
                if solve_secs > 0.0 {
                    self.profiling_stats.record_location(
                        "coupled:solve_coupled_system",
                        ProfileCategory::CpuCompute,
                        std::time::Duration::from_secs_f64(solve_secs),
                        0,
                    );
                }
            }

            // Debug: Check matrix and RHS after assembly on first iteration - DEBUG READS
            if DEBUG_READS_ENABLED && iter == 0 {
                let sync_start = Instant::now();
                let _ = self
                    .context
                    .device
                    .poll(wgpu::PollType::wait_indefinitely());
                self.profiling_stats.record_location(
                    "coupled:debug_sync",
                    ProfileCategory::GpuSync,
                    sync_start.elapsed(),
                    0,
                );
                let res = self.coupled_resources.as_ref().unwrap();

                // Read diagonal entries (sample first few) - DEBUG READS
                let read_start = Instant::now();
                let matrix_vals = pollster::block_on(
                    self.read_buffer_f32(&res.b_matrix_values, res.num_nonzeros),
                );
                self.profiling_stats.record_location(
                    "coupled:debug_read_matrix",
                    ProfileCategory::GpuRead,
                    read_start.elapsed(),
                    (res.num_nonzeros as u64) * 4,
                );

                let read_start = Instant::now();
                let rhs_vals =
                    pollster::block_on(self.read_buffer_f32(&res.b_rhs, self.num_cells * 3));
                self.profiling_stats.record_location(
                    "coupled:debug_read_rhs",
                    ProfileCategory::GpuRead,
                    read_start.elapsed(),
                    (self.num_cells as u64) * 3 * 4,
                );

                let read_start = Instant::now();
                let col_indices =
                    pollster::block_on(self.read_buffer_u32(&res.b_col_indices, res.num_nonzeros));
                self.profiling_stats.record_location(
                    "coupled:debug_read_col_indices",
                    ProfileCategory::GpuRead,
                    read_start.elapsed(),
                    (res.num_nonzeros as u64) * 4,
                );

                // Find diagonal statistics
                let num_cells = self.num_cells as usize;
                let read_start = Instant::now();
                let row_offsets = pollster::block_on(
                    self.read_buffer_u32(&res.b_row_offsets, self.num_cells * 3 + 1),
                );
                self.profiling_stats.record_location(
                    "coupled:debug_read_row_offsets",
                    ProfileCategory::GpuRead,
                    read_start.elapsed(),
                    ((self.num_cells * 3 + 1) as u64) * 4,
                );

                // Sample diagonal for each equation type - look for actual diagonal
                let mut diag_u_sample = Vec::new();
                let mut diag_v_sample = Vec::new();
                let mut diag_p_sample = Vec::new();

                for cell_idx in 0..5.min(num_cells) {
                    // Find diagonal for u equation (row = cell_idx*3, col = cell_idx*3)
                    let row_u = cell_idx * 3;
                    let target_col_u = cell_idx * 3;
                    let start_u = row_offsets[row_u] as usize;
                    let end_u = row_offsets[row_u + 1] as usize;
                    for k in start_u..end_u {
                        if col_indices[k] as usize == target_col_u {
                            diag_u_sample.push(matrix_vals[k]);
                            break;
                        }
                    }

                    // Find diagonal for v equation (row = cell_idx*3+1, col = cell_idx*3+1)
                    let row_v = cell_idx * 3 + 1;
                    let target_col_v = cell_idx * 3 + 1;
                    let start_v = row_offsets[row_v] as usize;
                    let end_v = row_offsets[row_v + 1] as usize;
                    for k in start_v..end_v {
                        if col_indices[k] as usize == target_col_v {
                            diag_v_sample.push(matrix_vals[k]);
                            break;
                        }
                    }

                    // Find diagonal for p equation (row = cell_idx*3+2, col = cell_idx*3+2)
                    let row_p = cell_idx * 3 + 2;
                    let target_col_p = cell_idx * 3 + 2;
                    let start_p = row_offsets[row_p] as usize;
                    let end_p = row_offsets[row_p + 1] as usize;
                    for k in start_p..end_p {
                        if col_indices[k] as usize == target_col_p {
                            diag_p_sample.push(matrix_vals[k]);
                            break;
                        }
                    }
                }

                if !quiet {
                    println!("Matrix diag_u sample: {:?}", diag_u_sample);
                    println!("Matrix diag_v sample: {:?}", diag_v_sample);
                    println!("Matrix diag_p sample: {:?}", diag_p_sample);
                }

                // Check RHS for NaN
                let rhs_has_nan = rhs_vals.iter().any(|v| v.is_nan());
                let rhs_u: Vec<_> = (0..5.min(num_cells)).map(|i| rhs_vals[i * 3]).collect();
                let rhs_p: Vec<_> = (0..5.min(num_cells)).map(|i| rhs_vals[i * 3 + 2]).collect();
                if !quiet {
                    println!("RHS u sample: {:?}", rhs_u);
                    println!("RHS p sample: {:?}", rhs_p);
                    println!("RHS has NaN: {}", rhs_has_nan);
                }
            }

            // 3. Solve Coupled System using FGMRES-based coupled solver (CPU-side Krylov, GPU SpMV/precond)
            let stats = self.coupled_last_linear_stats;
            *self.stats_p.lock().unwrap() = stats;
            let io_start = Instant::now();
            if !quiet {
                println!(
                    "Coupled linear solve: {} iterations, residual {:.2e}, converged={}",
                    stats.iterations, stats.residual, stats.converged
                );
            }
            self.profiling_stats.record_location(
                "coupled:println_linear_solve",
                ProfileCategory::CpuCompute,
                io_start.elapsed(),
                0,
            );

            if stats.residual.is_nan() {
                panic!("Coupled Linear Solver Diverged: NaN detected in linear residual");
            }

            // 4. Update Fields & Compute Max Diff
            {
                self.coupled_should_clear_max_diff = iter > 0;
                let timings = GpuSolver::coupled_update_plan().execute(self);
                if iter > 0 {
                    let clear_secs = timings.seconds_for("coupled:clear_max_diff");
                    if clear_secs > 0.0 {
                        self.profiling_stats.record_location(
                            "coupled:clear_max_diff",
                            ProfileCategory::GpuWrite,
                            std::time::Duration::from_secs_f64(clear_secs),
                            8,
                        );
                    }
                }

                let update_secs = timings.seconds_for("coupled:update_fields_max_diff");
                if update_secs > 0.0 {
                    self.profiling_stats.record_location(
                        "coupled:update_fields_max_diff",
                        ProfileCategory::GpuDispatch,
                        std::time::Duration::from_secs_f64(update_secs),
                        0,
                    );
                }
            }

            // Check convergence using GPU-computed max-diff (Async)
            if iter > 0 {
                if let Some(res) = self.coupled_resources.as_ref() {
                    // Start async read for CURRENT iteration
                    let async_start = Instant::now();
                    let mut reader = res.async_scalar_reader.borrow_mut();
                    reader.start_read(
                        &self.context.device,
                        &self.context.queue,
                        &res.b_max_diff_result,
                        0,
                    );

                    reader.poll(); // Poll for completion of previous reads
                    self.profiling_stats.record_location(
                        "coupled:async_convergence_check",
                        ProfileCategory::Other,
                        async_start.elapsed(),
                        0,
                    );

                    // Check if we have a result available
                    if let Some(results) = reader.get_last_value_vec(2) {
                        let max_diff_u = results[0] as f64;
                        let max_diff_p = results[1] as f64;

                        if max_diff_u.is_nan() || max_diff_p.is_nan() {
                            panic!(
                                "Coupled Solver Diverged: NaN detected in outer residuals (U: {}, P: {})",
                                max_diff_u, max_diff_p
                            );
                        }

                        // Store outer loop stats
                        *self.outer_residual_u.lock().unwrap() = max_diff_u as f32;
                        *self.outer_residual_p.lock().unwrap() = max_diff_p as f32;
                        *self.outer_iterations.lock().unwrap() = iter + 1;

                        let io_start = Instant::now();
                        if !quiet {
                            println!(
                                "Coupled Residuals - U: {:.2e}, P: {:.2e}",
                                max_diff_u, max_diff_p
                            );
                        }
                        self.profiling_stats.record_location(
                            "coupled:println_residuals",
                            ProfileCategory::CpuCompute,
                            io_start.elapsed(),
                            0,
                        );

                        // Converged if both U and P are below tolerance
                        if max_diff_u < convergence_tol_u && max_diff_p < convergence_tol_p {
                            if !quiet {
                                println!(
                                    "Coupled Solver Converged in {} iterations",
                                    iter + 1
                                );
                            }
                            break;
                        }

                        // Stagnation check
                        let stagnation_factor = 1e-2;
                        let rel_u = if prev_residual_u.is_finite() && prev_residual_u.abs() > 1e-14
                        {
                            ((max_diff_u - prev_residual_u) / prev_residual_u).abs()
                        } else {
                            f64::INFINITY
                        };
                        let rel_p = if prev_residual_p.is_finite() && prev_residual_p.abs() > 1e-14
                        {
                            ((max_diff_p - prev_residual_p) / prev_residual_p).abs()
                        } else {
                            f64::INFINITY
                        };

                        if rel_u < stagnation_factor && rel_p < stagnation_factor && iter > 2 {
                            if !quiet {
                                println!(
                                    "Coupled solver stagnated at iter {}: U={:.2e}, P={:.2e}",
                                    iter + 1,
                                    max_diff_u,
                                    max_diff_p
                                );
                            }
                            break;
                        }

                        prev_residual_u = max_diff_u;
                        prev_residual_p = max_diff_p;
                    }
                }
            } else {
                // First iteration - initialize stats
                *self.outer_residual_u.lock().unwrap() = f32::MAX;
                *self.outer_residual_p.lock().unwrap() = f32::MAX;
                *self.outer_iterations.lock().unwrap() = 1;
            }
        }

        // Update time
        self.constants.time += self.constants.dt;
        self.constants.dt_old = self.constants.dt;
        self.update_constants();

        // Check evolution (Steady State / Degeneracy)
        self.check_evolution();

        let _ = self
            .context
            .device
            .poll(wgpu::PollType::wait_indefinitely());
    }

    fn check_evolution(&mut self) {
        let quiet = std::env::var("CFD2_QUIET").ok().as_deref() == Some("1");

        // Read velocity field to check for evolution and variance
        // This involves a GPU->CPU read, so it has some overhead.
        let u_data = pollster::block_on(self.read_buffer_f32(&self.b_state, self.num_cells * 8)); // FluidState is 8 f32s = 32 bytes

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

    fn solve_coupled_system(&mut self) -> LinearSolverStats {
        // Use the FGMRES-based coupled solver implementation from
        // `coupled_solver_fgmres.rs` to solve the coupled block system.
        self.solve_coupled_fgmres()
    }
}
