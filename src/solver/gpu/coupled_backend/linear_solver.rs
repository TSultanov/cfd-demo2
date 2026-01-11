use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::linear_solver::amg::CsrMatrix;
use crate::solver::gpu::linear_solver::fgmres::{
    write_iter_params, FgmresPrecondBindings, FgmresSolveOnceConfig, FgmresWorkspace, IterParams,
    RawFgmresParams,
};
use crate::solver::gpu::modules::coupled_schur::{CoupledPressureSolveKind, CoupledSchurModule};
use crate::solver::gpu::modules::krylov_precond::{DispatchGrids, KrylovDispatch};
use crate::solver::gpu::modules::krylov_solve::KrylovSolveModule;
use crate::solver::gpu::modules::linear_system::{LinearSystemPorts, LinearSystemView};
use crate::solver::gpu::modules::ports::PortSpace;
use crate::solver::gpu::profiling::{ProfileCategory, ProfilingStats};
use crate::solver::gpu::readback::{read_buffer_cached, StagingBufferCache};
use crate::solver::gpu::structs::{
    CoupledSolverResources, LinearSolverStats, PreconditionerParams, PreconditionerType,
};
use std::time::Instant;

/// Resources for GPU-based FGMRES solver
pub type FgmresResources = KrylovSolveModule<CoupledSchurModule>;

pub struct IncompressibleLinearSolver {
    pub fgmres_resources: Option<FgmresResources>,
}

impl IncompressibleLinearSolver {
    pub fn new() -> Self {
        Self {
            fgmres_resources: None,
        }
    }

    fn ensure_fgmres_resources(
        &mut self,
        context: &GpuContext,
        profiling: &ProfilingStats,
        coupled: &CoupledSolverResources,
        pressure_ports: LinearSystemPorts,
        pressure_port_space: &PortSpace,
        num_cells: u32,
        max_restart: usize,
    ) {
        let n = coupled.num_unknowns;
        let rebuild = match &self.fgmres_resources {
            Some(existing) => {
                existing.fgmres.max_restart() < max_restart || existing.fgmres.n() != n
            }
            None => true,
        };

        if rebuild {
            let resources = self.init_fgmres_resources(
                context,
                profiling,
                coupled,
                pressure_ports,
                pressure_port_space,
                num_cells,
                max_restart,
            );
            self.fgmres_resources = Some(resources);
        }
    }

    /// Initialize FGMRES resources
    fn init_fgmres_resources(
        &self,
        context: &GpuContext,
        profiling: &ProfilingStats,
        coupled: &CoupledSolverResources,
        pressure_ports: LinearSystemPorts,
        pressure_port_space: &PortSpace,
        num_cells: u32,
        max_restart: usize,
    ) -> FgmresResources {
        let device = &context.device;
        let n = coupled.num_unknowns;
        let block_system = LinearSystemView {
            ports: coupled.linear_ports,
            space: &coupled.linear_port_space,
        };

        let fgmres = FgmresWorkspace::new_from_system(
            device,
            n,
            num_cells,
            max_restart,
            block_system,
            FgmresPrecondBindings::DiagWithParams {
                diag_u: &coupled.b_diag_u,
                diag_v: &coupled.b_diag_v,
                diag_p: &coupled.b_diag_p,
                precond_params: &coupled.b_precond_params,
            },
            "Coupled",
        );
        let precond = CoupledSchurModule::new(
            device,
            &fgmres,
            num_cells,
            pressure_port_space.buffer(pressure_ports.row_offsets),
            pressure_port_space.buffer(pressure_ports.col_indices),
            pressure_port_space.buffer(pressure_ports.values),
            CoupledPressureSolveKind::Chebyshev,
            crate::solver::model::KernelId::SCHUR_PRECOND_PREDICT_AND_FORM,
        );

        let resources = KrylovSolveModule::new(fgmres, precond);

        self.record_fgmres_allocations(profiling, &resources);

        resources
    }

    fn record_fgmres_allocations(&self, profiling: &ProfilingStats, fgmres: &FgmresResources) {
        if !profiling.is_enabled() {
            return;
        }

        let record = |label: &str, buf: &wgpu::Buffer| {
            profiling.record_gpu_alloc(label, buf.size());
        };

        record("fgmres:basis", fgmres.fgmres.basis_buffer());
        for (i, buf) in fgmres.fgmres.z_vectors().iter().enumerate() {
            record(&format!("fgmres:z_{}", i), buf);
        }
        record("fgmres:w", fgmres.fgmres.w_buffer());
        record("fgmres:temp", fgmres.fgmres.temp_buffer());
        record("fgmres:dot_partial", fgmres.fgmres.dot_partial_buffer());
        record("fgmres:scalars", fgmres.fgmres.scalars_buffer());
        record("fgmres:temp_p", fgmres.precond.b_temp_p());
        record("fgmres:p_sol", fgmres.precond.b_p_sol());
        record("fgmres:params", fgmres.fgmres.params_buffer());
        record("fgmres:hessenberg", fgmres.fgmres.hessenberg_buffer());
        record("fgmres:givens", fgmres.fgmres.givens_buffer());
        record("fgmres:g", fgmres.fgmres.g_buffer());
        record("fgmres:y", fgmres.fgmres.y_buffer());
        record("fgmres:iter_params", fgmres.fgmres.iter_params_buffer());
        record(
            "fgmres:staging_scalar",
            fgmres.fgmres.staging_scalar_buffer(),
        );
    }

    /// Solve the coupled system using FGMRES with block preconditioning (GPU-accelerated)
    #[allow(clippy::too_many_arguments)]
    pub fn solve(
        &mut self,
        context: &GpuContext,
        staging_cache: &StagingBufferCache,
        profiling: &ProfilingStats,
        coupled_resources: &CoupledSolverResources,
        pressure_ports: LinearSystemPorts,
        pressure_port_space: &PortSpace,
        num_cells: u32,
        num_nonzeros: u32,
        preconditioner: PreconditionerType,
    ) -> LinearSolverStats {
        let start_time = Instant::now();
        // Env overrides are intentionally not supported.
        let quiet = false;

        let n = coupled_resources.num_unknowns;
        let max_restart = 50usize;
        let max_outer = 20usize;
        let tol = 1e-5f32;
        let abstol = 1e-7f32;

        self.ensure_fgmres_resources(
            context,
            profiling,
            coupled_resources,
            pressure_ports,
            pressure_port_space,
            num_cells,
            max_restart,
        );
        let Some(mut fgmres) = self.fgmres_resources.take() else {
            return LinearSolverStats::default();
        };

        let stats = 'stats: {
            let pressure_kind = CoupledPressureSolveKind::from_config(preconditioner);
            fgmres.precond.set_pressure_kind(pressure_kind);
            if pressure_kind == CoupledPressureSolveKind::Amg {
                let row_offsets = pollster::block_on(read_buffer_cached(
                    context,
                    staging_cache,
                    profiling,
                    pressure_port_space.buffer(pressure_ports.row_offsets),
                    (num_cells as u64 + 1) * 4,
                    "AMG Row Offsets",
                ));
                let col_indices = pollster::block_on(read_buffer_cached(
                    context,
                    staging_cache,
                    profiling,
                    pressure_port_space.buffer(pressure_ports.col_indices),
                    num_nonzeros as u64 * 4,
                    "AMG Col Indices",
                ));
                let values = pollster::block_on(read_buffer_cached(
                    context,
                    staging_cache,
                    profiling,
                    pressure_port_space.buffer(pressure_ports.values),
                    num_nonzeros as u64 * 4,
                    "AMG Values",
                ));

                let row_offsets = bytemuck::cast_slice(&row_offsets).to_vec();
                let col_indices = bytemuck::cast_slice(&col_indices).to_vec();
                let values = bytemuck::cast_slice(&values).to_vec();

                let matrix = CsrMatrix {
                    row_offsets,
                    col_indices,
                    values,
                    num_rows: num_cells as usize,
                    num_cols: num_cells as usize,
                };
                fgmres.precond.ensure_amg_resources(&context.device, matrix);
            }

            let KrylovDispatch {
                grids,
                dofs_dispatch_x_threads,
                ..
            } = DispatchGrids::for_sizes(n, num_cells);

            // Initialize IterParams
            let iter_params = IterParams {
                current_idx: 0,
                max_restart: max_restart as u32,
                _pad1: 0,
                _pad2: 0,
            };
            let init_write_start = Instant::now();
            {
                let core = fgmres.fgmres.core(&context.device, &context.queue);
                write_iter_params(&core, &iter_params);
            }
            profiling.record_location(
                "fgmres:write_iter_params_init",
                ProfileCategory::GpuWrite,
                init_write_start.elapsed(),
                std::mem::size_of::<IterParams>() as u64,
            );

            let precond_params = PreconditionerParams {
                n: 0,
                num_cells,
                omega: 1.2,
                unknowns_per_cell: 3,
                u0: 0,
                u1: 1,
                p: 2,
                _pad0: 0,
            };
            let precond_write_start = Instant::now();
            context.queue.write_buffer(
                &coupled_resources.b_precond_params,
                0,
                bytemuck::bytes_of(&precond_params),
            );
            profiling.record_location(
                "fgmres:write_precond_params",
                ProfileCategory::GpuWrite,
                precond_write_start.elapsed(),
                std::mem::size_of::<PreconditionerParams>() as u64,
            );

            let system = LinearSystemView {
                ports: coupled_resources.linear_ports,
                space: &coupled_resources.linear_port_space,
            };
            let rhs_norm = fgmres.rhs_norm(context, system, n);
            if rhs_norm < abstol || !rhs_norm.is_finite() {
                break 'stats LinearSolverStats {
                    iterations: 0,
                    residual: rhs_norm,
                    converged: rhs_norm < abstol,
                    diverged: !rhs_norm.is_finite(),
                    time: start_time.elapsed(),
                };
            }

            // Initial residual r = b - A x stored in V_0
            let mut residual_norm = {
                let start = Instant::now();
                let core = fgmres.fgmres.core(&context.device, &context.queue);
                let norm = fgmres.fgmres.compute_residual_norm_into(
                    &core,
                    system,
                    fgmres.fgmres.basis_binding(0),
                    "FGMRES Residual",
                );
                profiling.record_location(
                    "fgmres:residual_norm_into",
                    ProfileCategory::GpuDispatch,
                    start.elapsed(),
                    0,
                );
                norm
            };

            let target_resid = (tol * rhs_norm).max(abstol);

            if residual_norm < target_resid {
                if !quiet {
                    println!(
                        "FGMRES: Initial guess already converged (||r|| = {:.2e} < {:.2e})",
                        residual_norm, target_resid
                    );
                }
                break 'stats LinearSolverStats {
                    iterations: 0,
                    residual: residual_norm,
                    converged: true,
                    diverged: false,
                    time: start_time.elapsed(),
                };
            }

            // Normalize V_0
            {
                let start = Instant::now();
                let core = fgmres.fgmres.core(&context.device, &context.queue);
                fgmres.fgmres.scale_in_place(
                    &core,
                    fgmres.fgmres.basis_binding(0),
                    1.0 / residual_norm,
                    "FGMRES Normalize V0",
                );
                profiling.record_location(
                    "fgmres:scale_in_place",
                    ProfileCategory::GpuDispatch,
                    start.elapsed(),
                    0,
                );
            }

            // Initialize g on GPU
            let g_init_write_start = Instant::now();
            fgmres.fgmres.write_g0(&context.queue, residual_norm);
            profiling.record_location(
                "fgmres:write_g_init",
                ProfileCategory::GpuWrite,
                g_init_write_start.elapsed(),
                fgmres.fgmres.g_buffer().size(),
            );

            let mut total_iters = 0u32;
            let mut final_resid = residual_norm;
            let mut converged = false;

            let io_start = Instant::now();
            if !quiet {
                println!("FGMRES: Initial residual = {:.2e}", residual_norm);
            }
            profiling.record_location(
                "fgmres:println",
                ProfileCategory::CpuCompute,
                io_start.elapsed(),
                0,
            );

            let mut stagnation_count = 0;
            let mut prev_resid_norm = residual_norm;

            'outer: for outer_iter in 0..max_outer {
                let params = RawFgmresParams {
                    n,
                    num_cells,
                    num_iters: 0,
                    omega: 1.0,
                    dispatch_x: dofs_dispatch_x_threads,
                    max_restart: max_restart as u32,
                    column_offset: 0,
                    _pad3: 0,
                };

                let solve = fgmres.solve_once(
                    context,
                    system,
                    rhs_norm,
                    params,
                    iter_params,
                    FgmresSolveOnceConfig {
                        tol_rel: tol,
                        tol_abs: abstol,
                        reset_x_before_update: false,
                    },
                    DispatchGrids {
                        dofs: grids.dofs,
                        cells: grids.cells,
                    },
                    "FGMRES Preconditioner Step",
                );

                total_iters += solve.basis_size as u32;
                final_resid = solve.residual_est;
                converged = solve.converged;

                if converged {
                    if !quiet {
                        println!(
                            "FGMRES restart {}: estimated residual = {:.2e}",
                            outer_iter + 1,
                            final_resid
                        );
                    }
                    break 'outer;
                }

                // Compute true residual (only when not already converged)
                residual_norm = {
                    let start = Instant::now();
                    let core = fgmres.fgmres.core(&context.device, &context.queue);
                    let norm = fgmres.fgmres.compute_residual_norm_into(
                        &core,
                        system,
                        fgmres.fgmres.basis_binding(0),
                        "FGMRES Residual",
                    );
                    profiling.record_location(
                        "fgmres:residual_norm_into",
                        ProfileCategory::GpuDispatch,
                        start.elapsed(),
                        0,
                    );
                    norm
                };
                final_resid = residual_norm;

                if residual_norm < tol * rhs_norm {
                    converged = true;
                    if !quiet {
                        println!(
                            "FGMRES restart {}: true residual = {:.2e} (converged)",
                            outer_iter + 1,
                            residual_norm
                        );
                    }
                    break 'outer;
                }

                // Prepare for restart
                // Reset g on GPU
                let g_write_start = Instant::now();
                fgmres.fgmres.write_g0(&context.queue, residual_norm);
                profiling.record_location(
                    "fgmres:write_g_restart",
                    ProfileCategory::GpuWrite,
                    g_write_start.elapsed(),
                    fgmres.fgmres.g_buffer().size(),
                );

                if residual_norm <= 0.0 {
                    if !quiet {
                        println!("FGMRES: residual vanished at restart {}", outer_iter + 1);
                    }
                    converged = true;
                    break;
                }

                {
                    let start = Instant::now();
                    let core = fgmres.fgmres.core(&context.device, &context.queue);
                    fgmres.fgmres.scale_in_place(
                        &core,
                        fgmres.fgmres.basis_binding(0),
                        1.0 / residual_norm,
                        "FGMRES Restart Normalize",
                    );
                    profiling.record_location(
                        "fgmres:scale_in_place",
                        ProfileCategory::GpuDispatch,
                        start.elapsed(),
                        0,
                    );
                }

                // Stagnation detection
                let improvement = (prev_resid_norm - residual_norm) / prev_resid_norm;
                if improvement < 1e-3 {
                    stagnation_count += 1;
                    if stagnation_count >= 3 {
                        if !quiet {
                            println!(
                                "FGMRES: Stagnation detected at restart {} (residual {:.2e})",
                                outer_iter + 1,
                                residual_norm
                            );
                        }
                        converged = true;
                        break 'outer;
                    }
                } else {
                    stagnation_count = 0;
                }
                prev_resid_norm = residual_norm;

                if !quiet {
                    println!(
                        "FGMRES restart {}: residual = {:.2e} (target {:.2e})",
                        outer_iter + 1,
                        residual_norm,
                        tol * rhs_norm
                    );
                }
            }

            let io_start = Instant::now();
            if !quiet {
                println!(
                    "FGMRES finished: {} iterations, residual = {:.2e}, converged = {}",
                    total_iters, final_resid, converged
                );
            }
            profiling.record_location(
                "fgmres:println",
                ProfileCategory::CpuCompute,
                io_start.elapsed(),
                0,
            );

            break 'stats LinearSolverStats {
                iterations: total_iters,
                residual: final_resid,
                converged,
                diverged: final_resid.is_nan(),
                time: start_time.elapsed(),
            };
        };

        self.fgmres_resources = Some(fgmres);
        stats
    }
}
