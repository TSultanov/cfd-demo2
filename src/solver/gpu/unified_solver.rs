use crate::solver::gpu::compressible_solver::GpuCompressibleSolver;
use crate::solver::gpu::enums::TimeScheme;
use crate::solver::gpu::structs::{GpuSolver, LinearSolverStats, PreconditionerType};
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
            UnifiedSolverBackend::Compressible(_) => {
                f.write_str("Compressible(GpuCompressibleSolver)")
            }
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
            ModelFields::GenericCoupled(_) => {
                return Err("GpuUnifiedSolver does not support GenericCoupled models yet".to_string());
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

    pub fn time(&self) -> f32 {
        match &self.backend {
            UnifiedSolverBackend::Incompressible(solver) => solver.constants.time,
            UnifiedSolverBackend::Compressible(solver) => solver.constants.time,
        }
    }

    pub fn dt(&self) -> f32 {
        match &self.backend {
            UnifiedSolverBackend::Incompressible(solver) => solver.constants.dt,
            UnifiedSolverBackend::Compressible(solver) => solver.constants.dt,
        }
    }

    pub fn state_buffer(&self) -> &wgpu::Buffer {
        match &self.backend {
            UnifiedSolverBackend::Incompressible(solver) => &solver.b_state,
            UnifiedSolverBackend::Compressible(solver) => &solver.b_state,
        }
    }

    pub fn set_dt(&mut self, dt: f32) {
        match &mut self.backend {
            UnifiedSolverBackend::Incompressible(solver) => solver.set_dt(dt),
            UnifiedSolverBackend::Compressible(solver) => solver.set_dt(dt),
        }
    }

    pub fn set_dtau(&mut self, dtau: f32) {
        match &mut self.backend {
            UnifiedSolverBackend::Incompressible(_) => {}
            UnifiedSolverBackend::Compressible(solver) => solver.set_dtau(dtau),
        }
    }

    pub fn set_viscosity(&mut self, mu: f32) {
        match &mut self.backend {
            UnifiedSolverBackend::Incompressible(solver) => solver.set_viscosity(mu),
            UnifiedSolverBackend::Compressible(solver) => solver.set_viscosity(mu),
        }
    }

    pub fn set_density(&mut self, rho: f32) {
        match &mut self.backend {
            UnifiedSolverBackend::Incompressible(solver) => solver.set_density(rho),
            UnifiedSolverBackend::Compressible(_) => {}
        }
    }

    pub fn set_alpha_u(&mut self, alpha_u: f32) {
        match &mut self.backend {
            UnifiedSolverBackend::Incompressible(solver) => solver.set_alpha_u(alpha_u),
            UnifiedSolverBackend::Compressible(solver) => solver.set_alpha_u(alpha_u),
        }
    }

    pub fn set_alpha_p(&mut self, alpha_p: f32) {
        match &mut self.backend {
            UnifiedSolverBackend::Incompressible(solver) => solver.set_alpha_p(alpha_p),
            UnifiedSolverBackend::Compressible(_) => {}
        }
    }

    pub fn set_inlet_velocity(&mut self, velocity: f32) {
        match &mut self.backend {
            UnifiedSolverBackend::Incompressible(solver) => solver.set_inlet_velocity(velocity),
            UnifiedSolverBackend::Compressible(solver) => solver.set_inlet_velocity(velocity),
        }
    }

    pub fn set_ramp_time(&mut self, time: f32) {
        match &mut self.backend {
            UnifiedSolverBackend::Incompressible(solver) => solver.set_ramp_time(time),
            UnifiedSolverBackend::Compressible(_) => {}
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

    pub fn set_outer_iters(&mut self, iters: usize) {
        match &mut self.backend {
            UnifiedSolverBackend::Incompressible(_) => {}
            UnifiedSolverBackend::Compressible(solver) => solver.set_outer_iters(iters),
        }
    }

    pub fn incompressible_set_should_stop(&mut self, value: bool) {
        if let UnifiedSolverBackend::Incompressible(solver) = &mut self.backend {
            solver.should_stop = value;
        }
    }

    pub fn incompressible_should_stop(&self) -> bool {
        match &self.backend {
            UnifiedSolverBackend::Incompressible(solver) => solver.should_stop,
            UnifiedSolverBackend::Compressible(_) => false,
        }
    }

    pub fn incompressible_outer_stats(&self) -> Option<(u32, f32, f32)> {
        let UnifiedSolverBackend::Incompressible(solver) = &self.backend else {
            return None;
        };
        Some((
            *solver.outer_iterations.lock().unwrap(),
            *solver.outer_residual_u.lock().unwrap(),
            *solver.outer_residual_p.lock().unwrap(),
        ))
    }

    pub fn incompressible_linear_stats(&self) -> Option<(LinearSolverStats, LinearSolverStats, LinearSolverStats)> {
        let UnifiedSolverBackend::Incompressible(solver) = &self.backend else {
            return None;
        };
        Some((
            *solver.stats_ux.lock().unwrap(),
            *solver.stats_uy.lock().unwrap(),
            *solver.stats_p.lock().unwrap(),
        ))
    }

    pub fn set_u(&mut self, u: &[(f64, f64)]) {
        match &mut self.backend {
            UnifiedSolverBackend::Incompressible(solver) => solver.set_u(u),
            UnifiedSolverBackend::Compressible(_) => {}
        }
    }

    pub fn set_p(&mut self, p: &[f64]) {
        match &mut self.backend {
            UnifiedSolverBackend::Incompressible(solver) => solver.set_p(p),
            UnifiedSolverBackend::Compressible(_) => {}
        }
    }

    pub fn set_uniform_state(&mut self, rho: f32, u: [f32; 2], p: f32) {
        match &mut self.backend {
            UnifiedSolverBackend::Incompressible(_) => {}
            UnifiedSolverBackend::Compressible(solver) => solver.set_uniform_state(rho, u, p),
        }
    }

    pub fn set_state_fields(&mut self, rho: &[f32], u: &[[f32; 2]], p: &[f32]) {
        match &mut self.backend {
            UnifiedSolverBackend::Incompressible(_) => {}
            UnifiedSolverBackend::Compressible(solver) => solver.set_state_fields(rho, u, p),
        }
    }

    pub fn step(&mut self) {
        match &mut self.backend {
            UnifiedSolverBackend::Incompressible(solver) => solver.step(),
            UnifiedSolverBackend::Compressible(solver) => solver.step(),
        }
    }

    pub fn initialize_history(&self) {
        match &self.backend {
            UnifiedSolverBackend::Incompressible(solver) => solver.initialize_history(),
            UnifiedSolverBackend::Compressible(solver) => solver.initialize_history(),
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

    pub async fn get_p(&self) -> Vec<f64> {
        let data = self.read_state_f32().await;
        let stride = self.model.state_layout.stride() as usize;
        let offset = self.model.state_layout.offset_for("p").unwrap_or(0) as usize;
        (0..self.num_cells() as usize)
            .map(|i| data[i * stride + offset] as f64)
            .collect()
    }

    pub async fn get_u(&self) -> Vec<(f64, f64)> {
        let data = self.read_state_f32().await;
        let stride = self.model.state_layout.stride() as usize;
        let offset = self
            .model
            .state_layout
            .offset_for("U")
            .or_else(|| self.model.state_layout.offset_for("u"))
            .unwrap_or(0) as usize;
        (0..self.num_cells() as usize)
            .map(|i| {
                let base = i * stride + offset;
                (data[base] as f64, data[base + 1] as f64)
            })
            .collect()
    }

    pub async fn get_rho(&self) -> Vec<f64> {
        let data = self.read_state_f32().await;
        let stride = self.model.state_layout.stride() as usize;
        let offset = self.model.state_layout.offset_for("rho").unwrap_or(0) as usize;
        (0..self.num_cells() as usize)
            .map(|i| data[i * stride + offset] as f64)
            .collect()
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
