use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::env_utils::env_flag;
use crate::solver::gpu::linear_solver::amg::CsrMatrix;
use crate::solver::gpu::linear_solver::fgmres::{FgmresPrecondBindings, FgmresWorkspace};
use crate::solver::gpu::modules::compressible_krylov::{
    CompressibleKrylovModule, CompressibleKrylovPreconditionerKind,
};
use crate::solver::gpu::modules::krylov_precond::DispatchGrids;
use crate::solver::gpu::modules::krylov_solve::KrylovSolveModule;
use crate::solver::gpu::modules::linear_system::{LinearSystemPorts, LinearSystemView};
use crate::solver::gpu::modules::ports::PortSpace;
use crate::solver::gpu::profiling::ProfilingStats;
use crate::solver::gpu::readback::{read_buffer_cached, StagingBufferCache};
use crate::solver::gpu::structs::{LinearSolverStats, PreconditionerType};
use std::collections::HashMap;

pub type CompressibleFgmresResources = KrylovSolveModule<CompressibleKrylovModule>;

pub struct CompressibleLinearSolver {
    pub resources: Option<CompressibleFgmresResources>,
    pub last_stats: LinearSolverStats,
    pub tol: f32,
    pub max_restart: usize,
    pub retry_tol: f32,
    pub retry_restart: usize,
}

impl CompressibleLinearSolver {
    pub fn new() -> Self {
        Self {
            resources: None,
            last_stats: LinearSolverStats::default(),
            tol: 0.0,
            max_restart: 1,
            retry_tol: 0.0,
            retry_restart: 1,
        }
    }

    pub(crate) fn ensure_resources(
        &mut self,
        context: &GpuContext,
        n: u32,
        num_cells: u32,
        ports: LinearSystemPorts,
        port_space: &PortSpace,
        max_restart: usize,
    ) {
        let rebuild = match &self.resources {
            Some(existing) => {
                existing.fgmres.max_restart() < max_restart || existing.fgmres.n() != n
            }
            None => true,
        };
        if rebuild {
            let resources =
                self.init_resources(context, n, num_cells, ports, port_space, max_restart);
            self.resources = Some(resources);
        }
    }

    fn init_resources(
        &self,
        context: &GpuContext,
        n: u32,
        num_cells: u32,
        ports: LinearSystemPorts,
        port_space: &PortSpace,
        max_restart: usize,
    ) -> CompressibleFgmresResources {
        let device = &context.device;

        let (b_diag_u, b_diag_v, b_diag_p) =
            CompressibleKrylovModule::create_diag_buffers(device, n);

        let matrix = LinearSystemView {
            ports,
            space: port_space,
        };
        let fgmres = FgmresWorkspace::new_from_system(
            device,
            n,
            num_cells,
            max_restart,
            matrix,
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

    pub async fn ensure_amg_resources(
        &mut self,
        context: &GpuContext,
        staging_cache: &StagingBufferCache,
        profiling: &ProfilingStats,
        ports: LinearSystemPorts,
        port_space: &PortSpace,
        topology: &LinearTopology<'_>,
    ) {
        let needs_init = matches!(
            self.resources.as_ref(),
            Some(res) if !res.precond.has_amg_resources()
        );
        if !needs_init {
            return;
        }

        let values = read_buffer_cached(
            context,
            staging_cache,
            profiling,
            port_space.buffer(ports.values),
            topology.block_col_indices.len() as u64 * 4,
            "AMG Matrix Values",
        )
        .await;
        let values: Vec<f32> = bytemuck::cast_slice(&values).to_vec();

        let num_cells = topology.num_cells as usize;
        let mut scalar_values = vec![0.0f32; topology.scalar_col_indices.len()];

        for cell in 0..num_cells {
            let start = topology.scalar_row_offsets[cell] as usize;
            let end = topology.scalar_row_offsets[cell + 1] as usize;
            let mut map = HashMap::with_capacity(end - start);
            for idx in start..end {
                map.insert(topology.scalar_col_indices[idx] as usize, idx);
            }

            for row in 0..4usize {
                let block_row = cell * 4 + row;
                let bstart = topology.block_row_offsets[block_row] as usize;
                let bend = topology.block_row_offsets[block_row + 1] as usize;
                for k in bstart..bend {
                    let col_cell = topology.block_col_indices[k] as usize / 4;
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
            row_offsets: topology.scalar_row_offsets.to_vec(),
            col_indices: topology.scalar_col_indices.to_vec(),
            values: scalar_values,
            num_rows: num_cells,
            num_cols: num_cells,
        };

        if let Some(fgmres) = &mut self.resources {
            fgmres
                .precond
                .ensure_amg_resources(&context.device, matrix, 20);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn solve(
        &mut self,
        context: &GpuContext,
        staging_cache: &StagingBufferCache,
        profiling: &ProfilingStats,
        n: u32,
        num_cells: u32,
        ports: LinearSystemPorts,
        port_space: &PortSpace,
        topology: &LinearTopology<'_>,
        preconditioner: PreconditionerType,
        max_restart: usize,
        tol: f32,
    ) -> LinearSolverStats {
        self.ensure_resources(context, n, num_cells, ports, port_space, max_restart);

        let precond_kind = CompressibleKrylovPreconditionerKind::select(
            preconditioner,
            env_flag("CFD2_COMP_AMG", false),
            env_flag("CFD2_COMP_BLOCK_PRECOND", true),
        );
        if precond_kind == CompressibleKrylovPreconditionerKind::Amg {
            pollster::block_on(self.ensure_amg_resources(
                context,
                staging_cache,
                profiling,
                ports,
                port_space,
                topology,
            ));
        }

        let Some(fgmres) = self.resources.as_mut() else {
            return LinearSolverStats::default();
        };
        fgmres.precond.set_kind(precond_kind);

        self.last_stats = crate::solver::gpu::modules::linear_solver::solve_fgmres(
            context,
            fgmres,
            LinearSystemView {
                ports,
                space: port_space,
            },
            n,
            num_cells,
            DispatchGrids::for_sizes(n, num_cells),
            max_restart,
            tol,
            1e-20, // tol_abs
            "Compressible FGMRES Preconditioner",
        );
        self.last_stats
    }
}

pub struct LinearTopology<'a> {
    pub num_cells: u32,
    pub scalar_row_offsets: &'a [u32],
    pub scalar_col_indices: &'a [u32],
    pub block_row_offsets: &'a [u32],
    pub block_col_indices: &'a [u32],
}
