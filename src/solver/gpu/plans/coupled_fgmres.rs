// Coupled Solver using FGMRES with Schur Complement Preconditioning
//
// For the saddle-point system:
// [A   G] [u]   [b_u]
// [D   C] [p] = [b_p]
//
// We use FGMRES (Flexible GMRES) as the outer Krylov solver.
// The preconditioner uses the Schur complement approach:
//
// M^{-1} = [I  -A^{-1}G] [A^{-1}  0  ] [I    0]
//          [0     I    ] [0    S^{-1}] [-DA^{-1} I]
//
// where S = C - D*A^{-1}*G is the Schur complement (pressure Poisson).
//
// This implementation runs FULLY ON THE GPU:
// - All vectors remain on GPU
// - Only scalar values (dot products, norms) are read to CPU
// - Preconditioner sweep
use crate::solver::gpu::linear_solver::amg::CsrMatrix;
use crate::solver::gpu::modules::coupled_schur::{CoupledPressureSolveKind, CoupledSchurModule};
use crate::solver::gpu::modules::krylov_precond::{DispatchGrids, KrylovDispatch};
use crate::solver::gpu::modules::krylov_solve::KrylovSolveModule;
use crate::solver::gpu::modules::linear_system::LinearSystemView;
use crate::solver::gpu::linear_solver::fgmres::{
    write_iter_params, FgmresSolveOnceConfig, FgmresPrecondBindings, FgmresWorkspace, IterParams,
    RawFgmresParams,
};
use crate::solver::gpu::profiling::ProfileCategory;
use crate::solver::gpu::structs::{
    GpuSolver, LinearSolverStats, PreconditionerParams,
};
use std::time::Instant;

/// Resources for GPU-based FGMRES solver
pub type FgmresResources = KrylovSolveModule<CoupledSchurModule>;

impl GpuSolver {
    pub fn coupled_unknowns(&self) -> u32 {
        self.coupled_resources
            .as_ref()
            .map(|res| res.num_unknowns)
            .unwrap_or(self.num_cells * self.model.system.unknowns_per_cell())
    }

    fn ensure_fgmres_resources(&mut self, max_restart: usize) {
        let n = self.coupled_unknowns();
        let rebuild = match &self.fgmres_resources {
            Some(existing) => existing.fgmres.max_restart() < max_restart || existing.fgmres.n() != n,
            None => true,
        };

        if rebuild {
            let resources = self.init_fgmres_resources(max_restart);
            self.fgmres_resources = Some(resources);
        }
    }

    /// Initialize FGMRES resources
    pub fn init_fgmres_resources(&self, max_restart: usize) -> FgmresResources {
        let device = &self.common.context.device;
        let coupled = self
            .coupled_resources
            .as_ref()
            .expect("Coupled resources must be initialized before FGMRES");
        let n = coupled.num_unknowns;
        let block_system = LinearSystemView {
            ports: coupled.linear_ports,
            space: &coupled.linear_port_space,
        };

        let fgmres = FgmresWorkspace::new_from_system(
            device,
            n,
            self.num_cells,
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
        let pressure_system = LinearSystemView {
            ports: self.linear_ports,
            space: &self.linear_port_space,
        };
        let precond = CoupledSchurModule::new(
            device,
            &fgmres,
            self.num_cells,
            pressure_system,
            CoupledPressureSolveKind::Chebyshev,
        );

        let resources = KrylovSolveModule::new(fgmres, precond);

        self.record_fgmres_allocations(&resources);

        resources
    }

    /// Solve the coupled system using FGMRES with block preconditioning (GPU-accelerated)
    pub fn solve_coupled_fgmres(&mut self) -> LinearSolverStats {
        let start_time = Instant::now();
        let quiet = std::env::var("CFD2_QUIET").ok().as_deref() == Some("1");
        if self.coupled_resources.is_none() {
            if !quiet {
                println!("Coupled resources not initialized!");
            }
            return LinearSolverStats::default();
        }

        let num_cells = self.num_cells;
        let n = self.coupled_unknowns();
        let max_restart = 50usize;
        let max_outer = 20usize;
        let tol = 1e-5f32;
        let abstol = 1e-7f32;

        self.ensure_fgmres_resources(max_restart);
        let Some(res) = &self.coupled_resources else {
            return LinearSolverStats::default();
        };
        let Some(mut fgmres) = self.fgmres_resources.take() else {
            return LinearSolverStats::default();
        };

        let stats = 'stats: {
            let pressure_kind = CoupledPressureSolveKind::from_config(self.preconditioner);
            fgmres.precond.set_pressure_kind(pressure_kind);
            if pressure_kind == CoupledPressureSolveKind::Amg {
                let row_offsets =
                    pollster::block_on(self.read_buffer_u32(
                        self.linear_port_space.buffer(self.linear_ports.row_offsets),
                        self.num_cells + 1,
                    ));
                let col_indices =
                    pollster::block_on(self.read_buffer_u32(
                        self.linear_port_space.buffer(self.linear_ports.col_indices),
                        self.num_nonzeros,
                    ));
                let values =
                    pollster::block_on(self.read_buffer_f32(
                        self.linear_port_space.buffer(self.linear_ports.values),
                        self.num_nonzeros,
                    ));
                let matrix = CsrMatrix {
                    row_offsets,
                    col_indices,
                    values,
                    num_rows: self.num_cells as usize,
                    num_cols: self.num_cells as usize,
                };
                fgmres
                    .precond
                    .ensure_amg_resources(&self.common.context.device, matrix);
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
            let core = fgmres
                .fgmres
                .core(&self.common.context.device, &self.common.context.queue);
            write_iter_params(&core, &iter_params);
        }
        self.common.profiling_stats.record_location(
            "fgmres:write_iter_params_init",
            ProfileCategory::GpuWrite,
            init_write_start.elapsed(),
            std::mem::size_of::<IterParams>() as u64,
        );

        let precond_params = PreconditionerParams {
            n: 0,
            num_cells: self.num_cells,
            omega: 1.2,
            _pad0: 0,
        };
        let precond_write_start = Instant::now();
        self.common.context.queue.write_buffer(
            &res.b_precond_params,
            0,
            bytemuck::bytes_of(&precond_params),
        );
        self.common.profiling_stats.record_location(
            "fgmres:write_precond_params",
            ProfileCategory::GpuWrite,
            precond_write_start.elapsed(),
            std::mem::size_of::<PreconditionerParams>() as u64,
        );

        // Refresh block diagonals - REMOVED (Merged into coupled_assembly)

        let system = LinearSystemView {
            ports: res.linear_ports,
            space: &res.linear_port_space,
        };
        let rhs_norm = fgmres.rhs_norm(&self.common.context, system, n);
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
            let core = fgmres
                .fgmres
                .core(&self.common.context.device, &self.common.context.queue);
            let system = LinearSystemView {
                ports: res.linear_ports,
                space: &res.linear_port_space,
            };
            let norm = fgmres.fgmres.compute_residual_norm_into(
                &core,
                system,
                fgmres.fgmres.basis_binding(0),
                "FGMRES Residual",
            );
            self.common.profiling_stats.record_location(
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
            let core = fgmres
                .fgmres
                .core(&self.common.context.device, &self.common.context.queue);
            fgmres.fgmres.scale_in_place(
                &core,
                fgmres.fgmres.basis_binding(0),
                1.0 / residual_norm,
                "FGMRES Normalize V0",
            );
            self.common.profiling_stats.record_location(
                "fgmres:scale_in_place",
                ProfileCategory::GpuDispatch,
                start.elapsed(),
                0,
            );
        }

        // Initialize g on GPU
        let g_init_write_start = Instant::now();
        fgmres.fgmres.write_g0(&self.common.context.queue, residual_norm);
        self.common.profiling_stats.record_location(
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
        self.common.profiling_stats.record_location(
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
                &self.common.context,
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
                let core = fgmres
                    .fgmres
                    .core(&self.common.context.device, &self.common.context.queue);
                let system = LinearSystemView {
                    ports: res.linear_ports,
                    space: &res.linear_port_space,
                };
                let norm = fgmres.fgmres.compute_residual_norm_into(
                    &core,
                    system,
                    fgmres.fgmres.basis_binding(0),
                    "FGMRES Residual",
                );
                self.common.profiling_stats.record_location(
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
            fgmres.fgmres.write_g0(&self.common.context.queue, residual_norm);
            self.common.profiling_stats.record_location(
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
                let core = fgmres
                    .fgmres
                    .core(&self.common.context.device, &self.common.context.queue);
                fgmres.fgmres.scale_in_place(
                    &core,
                    fgmres.fgmres.basis_binding(0),
                    1.0 / residual_norm,
                    "FGMRES Restart Normalize",
                );
                self.common.profiling_stats.record_location(
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
        self.common.profiling_stats.record_location(
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
