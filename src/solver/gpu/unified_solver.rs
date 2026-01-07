use crate::solver::gpu::compressible_solver::GpuCompressibleSolver;
use crate::solver::gpu::enums::TimeScheme;
use crate::solver::gpu::structs::{GpuSolver, PreconditionerType};
use crate::solver::mesh::Mesh;
use crate::solver::model::{ModelFields, ModelSpec};
use crate::solver::scheme::Scheme;

#[derive(Debug, Clone, Copy)]
pub struct SolverConfig {
    pub advection_scheme: Scheme,
    pub time_scheme: TimeScheme,
    pub preconditioner: PreconditionerType,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            advection_scheme: Scheme::Upwind,
            time_scheme: TimeScheme::Euler,
            preconditioner: PreconditionerType::Jacobi,
        }
    }
}

pub enum UnifiedSolverBackend {
    Incompressible(GpuSolver),
    Compressible(GpuCompressibleSolver),
}

impl std::fmt::Debug for UnifiedSolverBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnifiedSolverBackend::Incompressible(_) => f.write_str("Incompressible(GpuSolver)"),
            UnifiedSolverBackend::Compressible(_) => f.write_str("Compressible(GpuCompressibleSolver)"),
        }
    }
}

pub struct GpuUnifiedSolver {
    model: ModelSpec,
    backend: UnifiedSolverBackend,
    config: SolverConfig,
}

impl GpuUnifiedSolver {
    pub async fn new(
        mesh: &Mesh,
        model: ModelSpec,
        config: SolverConfig,
        device: Option<wgpu::Device>,
        queue: Option<wgpu::Queue>,
    ) -> Result<Self, String> {
        let backend = match &model.fields {
            ModelFields::Incompressible(_) => {
                let mut solver = GpuSolver::new(mesh, device, queue).await;
                apply_config_incompressible(&mut solver, config);
                UnifiedSolverBackend::Incompressible(solver)
            }
            ModelFields::Compressible(_) => {
                let mut solver = GpuCompressibleSolver::new(mesh, device, queue).await;
                apply_config_compressible(&mut solver, config);
                UnifiedSolverBackend::Compressible(solver)
            }
        };

        Ok(Self {
            model,
            backend,
            config,
        })
    }

    pub fn model(&self) -> &ModelSpec {
        &self.model
    }

    pub fn config(&self) -> SolverConfig {
        self.config
    }

    pub fn num_cells(&self) -> u32 {
        match &self.backend {
            UnifiedSolverBackend::Incompressible(solver) => solver.num_cells,
            UnifiedSolverBackend::Compressible(solver) => solver.num_cells,
        }
    }

    pub fn set_dt(&mut self, dt: f32) {
        match &mut self.backend {
            UnifiedSolverBackend::Incompressible(solver) => solver.set_dt(dt),
            UnifiedSolverBackend::Compressible(solver) => solver.set_dt(dt),
        }
    }

    pub fn set_advection_scheme(&mut self, scheme: Scheme) {
        self.config.advection_scheme = scheme;
        match &mut self.backend {
            UnifiedSolverBackend::Incompressible(solver) => solver.set_scheme(scheme.gpu_id()),
            UnifiedSolverBackend::Compressible(solver) => solver.set_scheme(scheme.gpu_id()),
        }
    }

    pub fn set_time_scheme(&mut self, scheme: TimeScheme) {
        self.config.time_scheme = scheme;
        match &mut self.backend {
            UnifiedSolverBackend::Incompressible(solver) => solver.set_time_scheme(scheme as u32),
            UnifiedSolverBackend::Compressible(solver) => solver.set_time_scheme(scheme as u32),
        }
    }

    pub fn set_preconditioner(&mut self, preconditioner: PreconditionerType) {
        self.config.preconditioner = preconditioner;
        match &mut self.backend {
            UnifiedSolverBackend::Incompressible(solver) => solver.set_precond_type(preconditioner),
            UnifiedSolverBackend::Compressible(solver) => {
                solver.set_precond_type(preconditioner as u32)
            }
        }
    }

    pub fn step(&mut self) {
        match &mut self.backend {
            UnifiedSolverBackend::Incompressible(solver) => solver.step(),
            UnifiedSolverBackend::Compressible(solver) => solver.step(),
        }
    }

    pub async fn read_state_f32(&self) -> Vec<f32> {
        let stride = self.model.state_layout.stride() as u64;
        let bytes = self.num_cells() as u64 * stride * 4;
        match &self.backend {
            UnifiedSolverBackend::Incompressible(solver) => {
                let raw = solver.read_buffer(&solver.b_state, bytes).await;
                bytemuck::cast_slice(&raw).to_vec()
            }
            UnifiedSolverBackend::Compressible(solver) => {
                let raw = solver.read_buffer(&solver.b_state, bytes).await;
                bytemuck::cast_slice(&raw).to_vec()
            }
        }
    }
}

fn apply_config_incompressible(solver: &mut GpuSolver, config: SolverConfig) {
    solver.set_scheme(config.advection_scheme.gpu_id());
    solver.set_time_scheme(config.time_scheme as u32);
    solver.set_precond_type(config.preconditioner);
}

fn apply_config_compressible(solver: &mut GpuCompressibleSolver, config: SolverConfig) {
    solver.set_scheme(config.advection_scheme.gpu_id());
    solver.set_time_scheme(config.time_scheme as u32);
    solver.set_precond_type(config.preconditioner as u32);
}
