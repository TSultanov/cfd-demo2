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
use crate::solver::gpu::modules::krylov_precond::DispatchGrids;
use crate::solver::gpu::modules::krylov_solve::KrylovSolveModule;
use crate::solver::gpu::modules::linear_system::LinearSystemView;
use crate::solver::gpu::linear_solver::fgmres::{
    write_iter_params, FgmresSolveOnceConfig, FgmresPrecondBindings, FgmresWorkspace, IterParams,
    RawFgmresParams,
};
use crate::solver::gpu::model_defaults::default_incompressible_model;
use crate::solver::gpu::profiling::ProfileCategory;
use crate::solver::gpu::structs::{
    CoupledSolverResources, GpuSolver, LinearSolverStats, PreconditionerParams,
};
use bytemuck::cast_slice;
use std::time::Instant;

/// Resources for GPU-based FGMRES solver
pub type FgmresResources = KrylovSolveModule<CoupledSchurModule>;

impl GpuSolver {
    pub fn coupled_unknowns(&self) -> u32 {
        self.coupled_resources
            .as_ref()
            .map(|res| res.num_unknowns)
            .unwrap_or(self.num_cells * default_incompressible_model().system.unknowns_per_cell())
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
        let device = &self.context.device;
        let coupled = self
            .coupled_resources
            .as_ref()
            .expect("Coupled resources must be initialized before FGMRES");
        let n = coupled.num_unknowns;
        let block_system = LinearSystemView {
            ports: coupled.linear_ports,
            space: &coupled.linear_port_space,
        };

        let fgmres = FgmresWorkspace::new(
            device,
            n,
            self.num_cells,
            max_restart,
            block_system.row_offsets(),
            block_system.col_indices(),
            block_system.values(),
            FgmresPrecondBindings::DiagWithParams {
                diag_u: &coupled.b_diag_u,
                diag_v: &coupled.b_diag_v,
                diag_p: &coupled.b_diag_p,
                precond_params: &coupled.b_precond_params,
            },
            "Coupled",
        );
        let scalar_row_offsets = self.linear_port_space.buffer(self.linear_ports.row_offsets);
        let scalar_col_indices = self.linear_port_space.buffer(self.linear_ports.col_indices);
        let scalar_matrix_values = self.linear_port_space.buffer(self.linear_ports.values);
        let precond = CoupledSchurModule::new(
            device,
            &fgmres,
            self.num_cells,
            scalar_row_offsets,
            scalar_col_indices,
            scalar_matrix_values,
            CoupledPressureSolveKind::Chebyshev,
        );

        let resources = KrylovSolveModule::new(fgmres, precond);

        self.record_fgmres_allocations(&resources);

        resources
    }

    const MAX_WORKGROUPS_PER_DIMENSION: u32 = 65535;
    const WORKGROUP_SIZE: u32 = 64;

    fn workgroups_for_size(&self, n: u32) -> u32 {
        n.div_ceil(Self::WORKGROUP_SIZE)
    }

    /// Compute 2D dispatch dimensions for large workgroup counts.
    /// Returns (dispatch_x, dispatch_y) where total workgroups = dispatch_x * dispatch_y
    fn dispatch_2d(&self, workgroups: u32) -> (u32, u32) {
        if workgroups <= Self::MAX_WORKGROUPS_PER_DIMENSION {
            (workgroups, 1)
        } else {
            // Split into 2D grid: use square-ish dimensions
            let dispatch_y = workgroups.div_ceil(Self::MAX_WORKGROUPS_PER_DIMENSION);
            let dispatch_x = workgroups.div_ceil(dispatch_y);
            (dispatch_x, dispatch_y)
        }
    }

    /// Compute the dispatch_x value (in threads, not workgroups) for params
    fn dispatch_x_threads(&self, workgroups: u32) -> u32 {
        let (dispatch_x, _) = self.dispatch_2d(workgroups);
        dispatch_x * Self::WORKGROUP_SIZE
    }

    fn write_scalars(&self, fgmres: &FgmresResources, scalars: &[f32]) {
        let start = Instant::now();
        self.context
            .queue
            .write_buffer(fgmres.fgmres.scalars_buffer(), 0, cast_slice(scalars));
        let bytes = (scalars.len() * 4) as u64;
        self.profiling_stats.record_location(
            "write_scalars",
            ProfileCategory::GpuWrite,
            start.elapsed(),
            bytes,
        );
    }

    fn basis_binding<'a>(
        &self,
        fgmres: &'a FgmresResources,
        idx: usize,
    ) -> wgpu::BindingResource<'a> {
        fgmres.fgmres.basis_binding(idx)
    }

    fn create_vector_bind_group<'a>(
        &self,
        fgmres: &FgmresResources,
        x: wgpu::BindingResource<'a>,
        y: wgpu::BindingResource<'a>,
        z: wgpu::BindingResource<'a>,
        label: &str,
    ) -> wgpu::BindGroup {
        let start = Instant::now();
        let bg = fgmres
            .fgmres
            .create_vector_bind_group(&self.context.device, x, y, z, label);
        self.profiling_stats.record_location(
            "create_vector_bind_group",
            ProfileCategory::CpuCompute,
            start.elapsed(),
            0,
        );
        bg
    }

    fn dispatch_vector_pipeline(
        &self,
        pipeline: &wgpu::ComputePipeline,
        fgmres: &FgmresResources,
        vector_bg: &wgpu::BindGroup,
        group3_bg: &wgpu::BindGroup,
        workgroups: u32,
        label: &str,
    ) {
        let start = Instant::now();
        let mut encoder = self
            .context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });

        self.dispatch_vector_pipeline_with_encoder(
            &mut encoder,
            pipeline,
            fgmres,
            vector_bg,
            group3_bg,
            workgroups,
            label,
        );

        self.context.queue.submit(Some(encoder.finish()));
        self.profiling_stats.record_location(
            label,
            ProfileCategory::GpuDispatch,
            start.elapsed(),
            0,
        );
    }

    fn dispatch_vector_pipeline_with_encoder(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        fgmres: &FgmresResources,
        vector_bg: &wgpu::BindGroup,
        group3_bg: &wgpu::BindGroup,
        workgroups: u32,
        label: &str,
    ) {
        let (dispatch_x, dispatch_y) = self.dispatch_2d(workgroups);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, vector_bg, &[]);
        pass.set_bind_group(1, fgmres.fgmres.matrix_bg(), &[]);
        pass.set_bind_group(2, fgmres.fgmres.precond_bg(), &[]);
        pass.set_bind_group(3, group3_bg, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    /// Compute norm of a vector using GPU reduction and async read
    ///
    /// This uses a batched approach: all GPU work (partial reduction, final reduction,
    /// and buffer copy) is submitted in a single command buffer, then we do an async
    /// map and wait. This avoids multiple sync points.
    fn gpu_norm<'a>(
        &self,
        fgmres: &'a FgmresResources,
        x: wgpu::BindingResource<'a>,
        n: u32,
    ) -> f32 {
        let start = Instant::now();
        let norm = fgmres.fgmres.gpu_norm(&self.context.device, &self.context.queue, x, n);
        self.profiling_stats.record_location(
            "gpu_norm",
            ProfileCategory::GpuDispatch,
            start.elapsed(),
            0,
        );
        norm
    }

    fn compute_residual_into<'a>(
        &self,
        fgmres: &'a FgmresResources,
        res: &'a CoupledSolverResources,
        target: wgpu::BindingResource<'a>,
        workgroups: u32,
        n: u32,
    ) -> f32 {
        let b_x = res.linear_port_space.buffer(res.linear_ports.x);
        let b_rhs = res.linear_port_space.buffer(res.linear_ports.rhs);

        let spmv_bg = self.create_vector_bind_group(
            fgmres,
            b_x.as_entire_binding(),
            fgmres.fgmres.w_buffer().as_entire_binding(),
            fgmres.fgmres.temp_buffer().as_entire_binding(),
            "FGMRES Residual SpMV BG",
        );
        self.dispatch_vector_pipeline(
            fgmres.fgmres.pipeline_spmv(),
            fgmres,
            &spmv_bg,
            fgmres.fgmres.params_bg(),
            workgroups,
            "FGMRES Residual SpMV",
        );

        self.write_scalars(fgmres, &[1.0, -1.0]);
        let residual_bg = self.create_vector_bind_group(
            fgmres,
            b_rhs.as_entire_binding(),
            fgmres.fgmres.w_buffer().as_entire_binding(),
            target.clone(),
            "FGMRES Residual Axpby BG",
        );
        self.dispatch_vector_pipeline(
            fgmres.fgmres.pipeline_axpby(),
            fgmres,
            &residual_bg,
            fgmres.fgmres.params_bg(),
            workgroups,
            "FGMRES Residual Axpby",
        );

        self.gpu_norm(fgmres, target, n)
    }

    fn scale_vector_in_place<'a>(
        &self,
        fgmres: &'a FgmresResources,
        buffer: wgpu::BindingResource<'a>,
        workgroups: u32,
        label: &str,
    ) {
        let vector_bg = self.create_vector_bind_group(
            fgmres,
            fgmres.fgmres.temp_buffer().as_entire_binding(),
            buffer,
            fgmres.fgmres.dot_partial_buffer().as_entire_binding(),
            label,
        );
        self.dispatch_vector_pipeline(
            fgmres.fgmres.pipeline_scale_in_place(),
            fgmres,
            &vector_bg,
            fgmres.fgmres.params_bg(),
            workgroups,
            label,
        );
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
                    pollster::block_on(self.read_buffer_u32(&self.b_row_offsets, self.num_cells + 1));
                let col_indices =
                    pollster::block_on(self.read_buffer_u32(&self.b_col_indices, self.num_nonzeros));
                let values =
                    pollster::block_on(self.read_buffer_f32(&self.b_matrix_values, self.num_nonzeros));
                let matrix = CsrMatrix {
                    row_offsets,
                    col_indices,
                    values,
                    num_rows: self.num_cells as usize,
                    num_cols: self.num_cells as usize,
                };
                fgmres.precond.ensure_amg_resources(&self.context.device, matrix);
            }

            let core = fgmres
                .fgmres
                .core(&self.context.device, &self.context.queue);

            let workgroups_dofs = self.workgroups_for_size(n);
            let workgroups_cells = self.workgroups_for_size(num_cells);
            let (dispatch_x, dispatch_y) = self.dispatch_2d(workgroups_cells);
            let (dofs_dispatch_x, dofs_dispatch_y) = self.dispatch_2d(workgroups_dofs);

        // Initialize IterParams
        let iter_params = IterParams {
            current_idx: 0,
            max_restart: max_restart as u32,
            _pad1: 0,
            _pad2: 0,
        };
        let init_write_start = Instant::now();
        write_iter_params(&core, &iter_params);
        self.profiling_stats.record_location(
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
        self.context.queue.write_buffer(
            &res.b_precond_params,
            0,
            bytemuck::bytes_of(&precond_params),
        );
        self.profiling_stats.record_location(
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
        let rhs_norm = fgmres.rhs_norm(&self.context, system, n);
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
        let mut residual_norm = self.compute_residual_into(
            &fgmres,
            res,
            self.basis_binding(&fgmres, 0),
            workgroups_dofs,
            n,
        );

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
        self.write_scalars(&fgmres, &[1.0 / residual_norm]);
        self.scale_vector_in_place(
            &fgmres,
            self.basis_binding(&fgmres, 0),
            workgroups_dofs,
            "FGMRES Normalize V0",
        ); // Initialize g on GPU
        let mut g_initial = vec![0.0f32; max_restart + 1];
        g_initial[0] = residual_norm;
        let g_init_write_start = Instant::now();
        self.context
            .queue
            .write_buffer(fgmres.fgmres.g_buffer(), 0, cast_slice(&g_initial));
        self.profiling_stats.record_location(
            "fgmres:write_g_init",
            ProfileCategory::GpuWrite,
            g_init_write_start.elapsed(),
            (g_initial.len() * 4) as u64,
        );

        let mut total_iters = 0u32;
        let mut final_resid = residual_norm;
        let mut converged = false;

        let io_start = Instant::now();
        if !quiet {
            println!("FGMRES: Initial residual = {:.2e}", residual_norm);
        }
        self.profiling_stats.record_location(
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
                dispatch_x: self.dispatch_x_threads(workgroups_dofs),
                max_restart: max_restart as u32,
                column_offset: 0,
                _pad3: 0,
            };

            let solve = fgmres.solve_once(
                &self.context,
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
                    dofs: (dofs_dispatch_x, dofs_dispatch_y),
                    cells: (dispatch_x, dispatch_y),
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
            residual_norm = self.compute_residual_into(
                &fgmres,
                res,
                self.basis_binding(&fgmres, 0),
                workgroups_dofs,
                n,
            );
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
            let mut g_initial = vec![0.0f32; max_restart + 1];
            g_initial[0] = residual_norm;
            let g_write_start = Instant::now();
            self.context
                .queue
                .write_buffer(fgmres.fgmres.g_buffer(), 0, cast_slice(&g_initial));
            self.profiling_stats.record_location(
                "fgmres:write_g_restart",
                ProfileCategory::GpuWrite,
                g_write_start.elapsed(),
                (g_initial.len() * 4) as u64,
            );

            if residual_norm <= 0.0 {
                if !quiet {
                    println!("FGMRES: residual vanished at restart {}", outer_iter + 1);
                }
                converged = true;
                break;
            }

            self.write_scalars(&fgmres, &[1.0 / residual_norm]);
            self.scale_vector_in_place(
                &fgmres,
                self.basis_binding(&fgmres, 0),
                workgroups_dofs,
                "FGMRES Restart Normalize",
            );

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
        self.profiling_stats.record_location(
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
