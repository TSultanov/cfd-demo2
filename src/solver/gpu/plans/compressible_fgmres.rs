use super::compressible::CompressiblePlanResources;
use crate::solver::gpu::linear_solver::amg::CsrMatrix;
use crate::solver::gpu::linear_solver::fgmres::{
    dispatch_2d, dispatch_vector_pipeline, dispatch_x_threads,
    workgroups_for_size, write_params, write_scalars, write_zeros, FgmresPrecondBindings,
    FgmresSolveOnceConfig, FgmresWorkspace, IterParams, RawFgmresParams,
};
use crate::solver::gpu::modules::compressible_krylov::{
    CompressibleKrylovModule, CompressibleKrylovPreconditionerKind,
};
use crate::solver::gpu::modules::krylov_precond::DispatchGrids;
use crate::solver::gpu::modules::krylov_solve::KrylovSolveModule;
use crate::solver::gpu::modules::linear_system::LinearSystemView;
use crate::solver::gpu::modules::linear_system::LinearSystemPorts;
use crate::solver::gpu::structs::LinearSolverStats;
use std::collections::HashMap;

pub type CompressibleFgmresResources = KrylovSolveModule<CompressibleKrylovModule>;

impl CompressiblePlanResources {
    pub(crate) fn ensure_fgmres_resources(&mut self, max_restart: usize) {
        let n = self.num_unknowns;
        let rebuild = match &self.fgmres_resources {
            Some(existing) => {
                existing.fgmres.max_restart() < max_restart || existing.fgmres.n() != n
            }
            None => true,
        };
        if rebuild {
            let resources = self.init_fgmres_resources(max_restart);
            self.fgmres_resources = Some(resources);
        }
    }

    pub async fn ensure_amg_resources(&mut self) {
        let needs_init = matches!(
            self.fgmres_resources.as_ref(),
            Some(res) if !res.precond.has_amg_resources()
        );
        if !needs_init {
            return;
        }

        let values = self
            .read_buffer_f32(
                self.port_space.buffer(self.matrix_ports.values),
                self.block_col_indices.len(),
            )
            .await;

        let num_cells = self.num_cells as usize;
        let mut scalar_values = vec![0.0f32; self.scalar_col_indices.len()];

        for cell in 0..num_cells {
            let start = self.scalar_row_offsets[cell] as usize;
            let end = self.scalar_row_offsets[cell + 1] as usize;
            let mut map = HashMap::with_capacity(end - start);
            for idx in start..end {
                map.insert(self.scalar_col_indices[idx] as usize, idx);
            }

            for row in 0..4usize {
                let block_row = cell * 4 + row;
                let bstart = self.block_row_offsets[block_row] as usize;
                let bend = self.block_row_offsets[block_row + 1] as usize;
                for k in bstart..bend {
                    let col_cell = self.block_col_indices[k] as usize / 4;
                    if let Some(&pos) = map.get(&col_cell) {
                        scalar_values[pos] += values[k].abs();
                    }
                }
            }

            if let Some(&diag_pos) = map.get(&cell) {
                if scalar_values[diag_pos].abs() < 1e-12 {
                    scalar_values[diag_pos] = 1.0;
                }
            }
        }

        let matrix = CsrMatrix {
            row_offsets: self.scalar_row_offsets.clone(),
            col_indices: self.scalar_col_indices.clone(),
            values: scalar_values,
            num_rows: num_cells,
            num_cols: num_cells,
        };

        if let Some(fgmres) = &mut self.fgmres_resources {
            fgmres
                .precond
                .ensure_amg_resources(&self.context.device, matrix, 20);
        }
    }

    fn init_fgmres_resources(&self, max_restart: usize) -> CompressibleFgmresResources {
        let device = &self.context.device;
        let n = self.num_unknowns;
        let num_cells = self.num_cells;

        let (b_diag_u, b_diag_v, b_diag_p) = CompressibleKrylovModule::create_diag_buffers(device, n);

        let matrix = LinearSystemView {
            ports: LinearSystemPorts {
                row_offsets: self.matrix_ports.row_offsets,
                col_indices: self.matrix_ports.col_indices,
                values: self.matrix_ports.values,
                rhs: self.linear_ports.rhs,
                x: self.linear_ports.x,
            },
            space: &self.port_space,
        };
        let fgmres = FgmresWorkspace::new(
            device,
            n,
            num_cells,
            max_restart,
            matrix.row_offsets(),
            matrix.col_indices(),
            matrix.values(),
            FgmresPrecondBindings::Diag {
                diag_u: &b_diag_u,
                diag_v: &b_diag_v,
                diag_p: &b_diag_p,
            },
            "Compressible",
        );

        let precond = CompressibleKrylovModule::new(
            device,
            &fgmres,
            num_cells,
            CompressibleKrylovPreconditionerKind::Identity,
            b_diag_u,
            b_diag_v,
            b_diag_p,
        );

        KrylovSolveModule::new(fgmres, precond)
    }

    pub(crate) fn solve_compressible_fgmres(
        &mut self,
        max_restart: usize,
        tol: f32,
    ) -> LinearSolverStats {
        let start = std::time::Instant::now();
        self.ensure_fgmres_resources(max_restart);
        let precond_kind = CompressibleKrylovPreconditionerKind::select(
            self.preconditioner,
            env_flag("CFD2_COMP_AMG", false),
            env_flag("CFD2_COMP_BLOCK_PRECOND", true),
        );
        if precond_kind == CompressibleKrylovPreconditionerKind::Amg {
            pollster::block_on(self.ensure_amg_resources());
        }

        let Some(mut fgmres) = self.fgmres_resources.take() else {
            return LinearSolverStats::default();
        };
        fgmres.precond.set_kind(precond_kind);

        let stats = 'stats: {
            let n = self.num_unknowns;
            let workgroups = workgroups_for_size(n);
            let (dispatch_x, dispatch_y) = dispatch_2d(workgroups);
            let dispatch_x_threads = dispatch_x_threads(workgroups);
            let block_workgroups = workgroups_for_size(self.num_cells);
            let (block_dispatch_x, block_dispatch_y) = dispatch_2d(block_workgroups);
            let dispatch = DispatchGrids {
                dofs: (dispatch_x, dispatch_y),
                cells: (block_dispatch_x, block_dispatch_y),
            };

            let system = LinearSystemView {
                ports: LinearSystemPorts {
                    row_offsets: self.matrix_ports.row_offsets,
                    col_indices: self.matrix_ports.col_indices,
                    values: self.matrix_ports.values,
                    rhs: self.linear_ports.rhs,
                    x: self.linear_ports.x,
                },
                space: &self.port_space,
            };
            self.zero_buffer(system.x(), n);

            let tol_abs = 1e-6f32;
            let rhs_norm = fgmres.rhs_norm(&self.context, system, n);
            if rhs_norm <= tol_abs {
                let stats = LinearSolverStats {
                    iterations: 0,
                    residual: rhs_norm,
                    converged: true,
                    diverged: false,
                    time: start.elapsed(),
                };
                if std::env::var("CFD2_DEBUG_FGMRES").ok().as_deref() == Some("1") {
                    eprintln!(
                        "fgmres: iters=0 rhs_norm={:.3e} residual={:.3e} converged=true",
                        rhs_norm, rhs_norm
                    );
                }
                break 'stats stats;
            }

            let params = RawFgmresParams {
                n,
                num_cells: self.num_cells,
                num_iters: 0,
                omega: 1.0,
                dispatch_x: dispatch_x_threads,
                max_restart: fgmres.fgmres.max_restart() as u32,
                column_offset: 0,
                _pad3: 0,
            };
            let core = fgmres.fgmres.core(&self.context.device, &self.context.queue);
            write_params(&core, &params);
            fgmres.precond.prepare(
                &self.context.device,
                &self.context.queue,
                &fgmres.fgmres,
                system.rhs().as_entire_binding(),
                dispatch.cells,
            );

            let iter_params = IterParams {
                current_idx: 0,
                max_restart: fgmres.fgmres.max_restart() as u32,
                _pad1: 0,
                _pad2: 0,
            };

            write_zeros(&core, fgmres.fgmres.hessenberg_buffer());
            write_zeros(&core, fgmres.fgmres.givens_buffer());
            write_zeros(&core, fgmres.fgmres.y_buffer());
            let mut g_init = vec![0.0f32; fgmres.fgmres.max_restart() + 1];
            g_init[0] = rhs_norm;
            self.context
                .queue
                .write_buffer(fgmres.fgmres.g_buffer(), 0, bytemuck::cast_slice(&g_init));

            let basis0 = fgmres.fgmres.basis_binding(0);
            let copy_bg = fgmres.fgmres.create_vector_bind_group(
                &self.context.device,
                system.rhs().as_entire_binding(),
                basis0,
                fgmres.fgmres.temp_buffer().as_entire_binding(),
                "Compressible FGMRES copy BG",
            );
            dispatch_vector_pipeline(
                &core,
                fgmres.fgmres.pipeline_copy(),
                &copy_bg,
                dispatch_x,
                dispatch_y,
                "Compressible FGMRES copy",
            );

            write_scalars(&core, &[1.0 / rhs_norm]);
            let basis0_y = fgmres.fgmres.basis_binding(0);
            let scale_bg = fgmres.fgmres.create_vector_bind_group(
                &self.context.device,
                fgmres.fgmres.w_buffer().as_entire_binding(),
                basis0_y,
                fgmres.fgmres.temp_buffer().as_entire_binding(),
                "Compressible FGMRES normalize BG",
            );
            dispatch_vector_pipeline(
                &core,
                fgmres.fgmres.pipeline_scale_in_place(),
                &scale_bg,
                dispatch_x,
                dispatch_y,
                "Compressible FGMRES normalize",
            );

            // Solve (single restart) using the shared GPU FGMRES core.
            let solve = fgmres.solve_once(
                &self.context,
                system,
                rhs_norm,
                params,
                iter_params,
                FgmresSolveOnceConfig {
                    tol_rel: tol,
                    tol_abs,
                    reset_x_before_update: true,
                },
                dispatch,
                "Compressible FGMRES Preconditioner",
            );
            let basis_size = solve.basis_size;
            let final_residual = solve.residual_est;
            let converged = solve.converged;

            let stats = LinearSolverStats {
                iterations: basis_size as u32,
                residual: final_residual,
                converged,
                diverged: false,
                time: start.elapsed(),
            };
            if std::env::var("CFD2_DEBUG_FGMRES").ok().as_deref() == Some("1") {
                eprintln!(
                    "fgmres: iters={} rhs_norm={:.3e} residual={:.3e} converged={}",
                    stats.iterations, rhs_norm, stats.residual, stats.converged
                );
            }
            break 'stats stats;
        };

        self.fgmres_resources = Some(fgmres);
        stats
    }

    fn zero_buffer(&self, buffer: &wgpu::Buffer, n: u32) {
        let zeros = vec![0.0f32; n as usize];
        self.context
            .queue
            .write_buffer(buffer, 0, bytemuck::cast_slice(&zeros));
    }
}

fn env_flag(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|val| {
            let val = val.to_ascii_lowercase();
            matches!(val.as_str(), "1" | "true" | "yes" | "y" | "on")
        })
        .unwrap_or(default)
}
