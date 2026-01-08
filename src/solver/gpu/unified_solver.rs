use crate::solver::gpu::plans::compressible::CompressiblePlanResources;
use crate::solver::gpu::plans::generic_coupled::GpuGenericCoupledSolver;
use crate::solver::gpu::enums::{GpuLowMachPrecondModel, TimeScheme};
use crate::solver::gpu::structs::{GpuSolver, LinearSolverStats, PreconditionerType};
use crate::solver::gpu::profiling::ProfilingStats;
use crate::solver::mesh::Mesh;
use crate::solver::model::{ModelFields, ModelSpec};
use crate::solver::scheme::Scheme;
use std::sync::Arc;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FgmresSizing {
    pub num_unknowns: u32,
    pub num_dot_groups: u32,
}

pub(crate) enum PlanInstance {
    Incompressible(GpuSolver),
    Compressible(CompressiblePlanResources),
    GenericCoupled(GpuGenericCoupledSolver),
}

impl std::fmt::Debug for PlanInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanInstance::Incompressible(_) => f.write_str("Incompressible(GpuSolver)"),
            PlanInstance::Compressible(_) => {
                f.write_str("Compressible(CompressiblePlanResources)")
            }
            PlanInstance::GenericCoupled(_) => {
                f.write_str("GenericCoupled(GpuGenericCoupledSolver)")
            }
        }
    }
}

pub struct GpuUnifiedSolver {
    model: ModelSpec,
    plan: PlanInstance,
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
        let plan = match &model.fields {
            ModelFields::Incompressible(_) => {
                let mut solver = GpuSolver::new(mesh, device, queue).await;
                apply_config_incompressible(&mut solver, config);
                PlanInstance::Incompressible(solver)
            }
            ModelFields::Compressible(_) => {
                let mut solver = CompressiblePlanResources::new(mesh, device, queue).await;
                apply_config_compressible(&mut solver, config);
                PlanInstance::Compressible(solver)
            }
            ModelFields::GenericCoupled(_) => {
                let mut solver = GpuGenericCoupledSolver::new(mesh, model.clone(), device, queue).await?;
                apply_config_incompressible(&mut solver.linear, config);
                PlanInstance::GenericCoupled(solver)
            }
        };

        Ok(Self {
            model,
            plan,
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
        match &self.plan {
            PlanInstance::Incompressible(solver) => solver.num_cells,
            PlanInstance::Compressible(solver) => solver.num_cells,
            PlanInstance::GenericCoupled(solver) => solver.linear.num_cells,
        }
    }

    pub fn time(&self) -> f32 {
        match &self.plan {
            PlanInstance::Incompressible(solver) => solver.constants.time,
            PlanInstance::Compressible(solver) => solver.constants.time,
            PlanInstance::GenericCoupled(solver) => solver.linear.constants.time,
        }
    }

    pub fn dt(&self) -> f32 {
        match &self.plan {
            PlanInstance::Incompressible(solver) => solver.constants.dt,
            PlanInstance::Compressible(solver) => solver.constants.dt,
            PlanInstance::GenericCoupled(solver) => solver.linear.constants.dt,
        }
    }

    pub fn state_buffer(&self) -> &wgpu::Buffer {
        match &self.plan {
            PlanInstance::Incompressible(solver) => &solver.b_state,
            PlanInstance::Compressible(solver) => &solver.b_state,
            PlanInstance::GenericCoupled(solver) => solver.state_buffer(),
        }
    }

    pub fn set_dt(&mut self, dt: f32) {
        match &mut self.plan {
            PlanInstance::Incompressible(solver) => solver.set_dt(dt),
            PlanInstance::Compressible(solver) => solver.set_dt(dt),
            PlanInstance::GenericCoupled(solver) => solver.linear.set_dt(dt),
        }
    }

    pub fn set_dtau(&mut self, dtau: f32) {
        match &mut self.plan {
            PlanInstance::Incompressible(_) => {}
            PlanInstance::Compressible(solver) => solver.set_dtau(dtau),
            PlanInstance::GenericCoupled(_) => {}
        }
    }

    pub fn set_viscosity(&mut self, mu: f32) {
        match &mut self.plan {
            PlanInstance::Incompressible(solver) => solver.set_viscosity(mu),
            PlanInstance::Compressible(solver) => solver.set_viscosity(mu),
            PlanInstance::GenericCoupled(solver) => solver.linear.set_viscosity(mu),
        }
    }

    pub fn set_density(&mut self, rho: f32) {
        match &mut self.plan {
            PlanInstance::Incompressible(solver) => solver.set_density(rho),
            PlanInstance::Compressible(_) => {}
            PlanInstance::GenericCoupled(solver) => solver.linear.set_density(rho),
        }
    }

    pub fn set_alpha_u(&mut self, alpha_u: f32) {
        match &mut self.plan {
            PlanInstance::Incompressible(solver) => solver.set_alpha_u(alpha_u),
            PlanInstance::Compressible(solver) => solver.set_alpha_u(alpha_u),
            PlanInstance::GenericCoupled(solver) => solver.linear.set_alpha_u(alpha_u),
        }
    }

    pub fn set_alpha_p(&mut self, alpha_p: f32) {
        match &mut self.plan {
            PlanInstance::Incompressible(solver) => solver.set_alpha_p(alpha_p),
            PlanInstance::Compressible(_) => {}
            PlanInstance::GenericCoupled(solver) => solver.linear.set_alpha_p(alpha_p),
        }
    }

    pub fn set_inlet_velocity(&mut self, velocity: f32) {
        match &mut self.plan {
            PlanInstance::Incompressible(solver) => solver.set_inlet_velocity(velocity),
            PlanInstance::Compressible(solver) => solver.set_inlet_velocity(velocity),
            PlanInstance::GenericCoupled(solver) => solver.linear.set_inlet_velocity(velocity),
        }
    }

    pub fn set_ramp_time(&mut self, time: f32) {
        match &mut self.plan {
            PlanInstance::Incompressible(solver) => solver.set_ramp_time(time),
            PlanInstance::Compressible(_) => {}
            PlanInstance::GenericCoupled(solver) => solver.linear.set_ramp_time(time),
        }
    }

    pub fn set_advection_scheme(&mut self, scheme: Scheme) {
        self.config.advection_scheme = scheme;
        match &mut self.plan {
            PlanInstance::Incompressible(solver) => solver.set_scheme(scheme.gpu_id()),
            PlanInstance::Compressible(solver) => solver.set_scheme(scheme.gpu_id()),
            PlanInstance::GenericCoupled(solver) => solver.linear.set_scheme(scheme.gpu_id()),
        }
    }

    pub fn set_time_scheme(&mut self, scheme: TimeScheme) {
        self.config.time_scheme = scheme;
        match &mut self.plan {
            PlanInstance::Incompressible(solver) => solver.set_time_scheme(scheme as u32),
            PlanInstance::Compressible(solver) => solver.set_time_scheme(scheme as u32),
            PlanInstance::GenericCoupled(solver) => solver.linear.set_time_scheme(scheme as u32),
        }
    }

    pub fn set_preconditioner(&mut self, preconditioner: PreconditionerType) {
        self.config.preconditioner = preconditioner;
        match &mut self.plan {
            PlanInstance::Incompressible(solver) => solver.set_precond_type(preconditioner),
            PlanInstance::Compressible(solver) => {
                solver.set_precond_type(preconditioner)
            }
            PlanInstance::GenericCoupled(solver) => solver.linear.set_precond_type(preconditioner),
        }
    }

    pub fn set_outer_iters(&mut self, iters: usize) {
        match &mut self.plan {
            PlanInstance::Incompressible(_) => {}
            PlanInstance::Compressible(solver) => solver.set_outer_iters(iters),
            PlanInstance::GenericCoupled(_) => {}
        }
    }

    pub fn set_incompressible_outer_correctors(&mut self, iters: u32) -> Result<(), String> {
        match &mut self.plan {
            PlanInstance::Incompressible(solver) => {
                solver.n_outer_correctors = iters;
                Ok(())
            }
            _ => Err(
                "set_incompressible_outer_correctors is supported for Incompressible models only"
                    .to_string(),
            ),
        }
    }

    pub fn incompressible_set_should_stop(&mut self, value: bool) {
        if let PlanInstance::Incompressible(solver) = &mut self.plan {
            solver.should_stop = value;
        }
    }

    pub fn incompressible_should_stop(&self) -> bool {
        match &self.plan {
            PlanInstance::Incompressible(solver) => solver.should_stop,
            PlanInstance::Compressible(_) => false,
            PlanInstance::GenericCoupled(_) => false,
        }
    }

    pub fn incompressible_outer_stats(&self) -> Option<(u32, f32, f32)> {
        let PlanInstance::Incompressible(solver) = &self.plan else {
            return None;
        };
        Some((
            *solver.outer_iterations.lock().unwrap(),
            *solver.outer_residual_u.lock().unwrap(),
            *solver.outer_residual_p.lock().unwrap(),
        ))
    }

    pub fn incompressible_linear_stats(&self) -> Option<(LinearSolverStats, LinearSolverStats, LinearSolverStats)> {
        let PlanInstance::Incompressible(solver) = &self.plan else {
            return None;
        };
        Some((
            *solver.stats_ux.lock().unwrap(),
            *solver.stats_uy.lock().unwrap(),
            *solver.stats_p.lock().unwrap(),
        ))
    }

    pub fn incompressible_degenerate_count(&self) -> Option<u32> {
        let PlanInstance::Incompressible(solver) = &self.plan else {
            return None;
        };
        Some(solver.degenerate_count)
    }

    pub fn set_u(&mut self, u: &[(f64, f64)]) {
        match &mut self.plan {
            PlanInstance::Incompressible(solver) => solver.set_u(u),
            PlanInstance::Compressible(_) => {}
            PlanInstance::GenericCoupled(_) => {}
        }
    }

    pub fn set_p(&mut self, p: &[f64]) {
        match &mut self.plan {
            PlanInstance::Incompressible(solver) => solver.set_p(p),
            PlanInstance::Compressible(_) => {}
            PlanInstance::GenericCoupled(_) => {}
        }
    }

    pub fn set_uniform_state(&mut self, rho: f32, u: [f32; 2], p: f32) {
        match &mut self.plan {
            PlanInstance::Incompressible(_) => {}
            PlanInstance::Compressible(solver) => solver.set_uniform_state(rho, u, p),
            PlanInstance::GenericCoupled(_) => {}
        }
    }

    pub fn set_state_fields(&mut self, rho: &[f32], u: &[[f32; 2]], p: &[f32]) {
        match &mut self.plan {
            PlanInstance::Incompressible(_) => {}
            PlanInstance::Compressible(solver) => solver.set_state_fields(rho, u, p),
            PlanInstance::GenericCoupled(_) => {}
        }
    }

    pub fn step(&mut self) {
        match &mut self.plan {
            PlanInstance::Incompressible(solver) => super::plans::coupled::plan::step_coupled(solver),
            PlanInstance::Compressible(solver) => super::plans::compressible::plan::step(solver),
            PlanInstance::GenericCoupled(solver) => {
                let _ = solver.step();
            }
        }
    }

    pub fn step_with_stats(&mut self) -> Result<Vec<LinearSolverStats>, String> {
        match &mut self.plan {
            PlanInstance::Compressible(solver) => Ok(super::plans::compressible::plan::step_with_stats(solver)),
            _ => Err("step_with_stats is supported for Compressible models only".to_string()),
        }
    }

    pub fn initialize_history(&self) {
        match &self.plan {
            PlanInstance::Incompressible(solver) => solver.initialize_history(),
            PlanInstance::Compressible(solver) => solver.initialize_history(),
            PlanInstance::GenericCoupled(_) => {}
        }
    }

    pub fn set_precond_model(&mut self, model: GpuLowMachPrecondModel) -> Result<(), String> {
        match &mut self.plan {
            PlanInstance::Compressible(solver) => {
                solver.set_precond_model(model as u32);
                Ok(())
            }
            _ => Err("set_precond_model is supported for Compressible models only".to_string()),
        }
    }

    pub fn set_precond_theta_floor(&mut self, theta: f32) -> Result<(), String> {
        match &mut self.plan {
            PlanInstance::Compressible(solver) => {
                solver.set_precond_theta_floor(theta);
                Ok(())
            }
            _ => Err(
                "set_precond_theta_floor is supported for Compressible models only".to_string(),
            ),
        }
    }

    pub fn set_nonconverged_relax(&mut self, alpha: f32) -> Result<(), String> {
        match &mut self.plan {
            PlanInstance::Compressible(solver) => {
                solver.set_nonconverged_relax(alpha);
                Ok(())
            }
            _ => Err("set_nonconverged_relax is supported for Compressible models only".to_string()),
        }
    }

    pub fn enable_detailed_profiling(&self, enable: bool) -> Result<(), String> {
        match &self.plan {
            PlanInstance::Incompressible(solver) => {
                solver.enable_detailed_profiling(enable);
                Ok(())
            }
            PlanInstance::GenericCoupled(solver) => {
                solver.linear.enable_detailed_profiling(enable);
                Ok(())
            }
            PlanInstance::Compressible(_) => Err(
                "enable_detailed_profiling is supported for Incompressible models only".to_string(),
            ),
        }
    }

    pub fn start_profiling_session(&self) -> Result<(), String> {
        match &self.plan {
            PlanInstance::Incompressible(solver) => {
                solver.start_profiling_session();
                Ok(())
            }
            PlanInstance::GenericCoupled(solver) => {
                solver.linear.start_profiling_session();
                Ok(())
            }
            PlanInstance::Compressible(_) => Err(
                "start_profiling_session is supported for Incompressible models only".to_string(),
            ),
        }
    }

    pub fn end_profiling_session(&self) -> Result<(), String> {
        match &self.plan {
            PlanInstance::Incompressible(solver) => {
                solver.end_profiling_session();
                Ok(())
            }
            PlanInstance::GenericCoupled(solver) => {
                solver.linear.end_profiling_session();
                Ok(())
            }
            PlanInstance::Compressible(_) => Err(
                "end_profiling_session is supported for Incompressible models only".to_string(),
            ),
        }
    }

    pub fn get_profiling_stats(&self) -> Result<Arc<ProfilingStats>, String> {
        match &self.plan {
            PlanInstance::Incompressible(solver) => Ok(solver.get_profiling_stats()),
            PlanInstance::GenericCoupled(solver) => Ok(solver.linear.get_profiling_stats()),
            PlanInstance::Compressible(_) => Err(
                "get_profiling_stats is supported for Incompressible models only".to_string(),
            ),
        }
    }

    pub fn print_profiling_report(&self) -> Result<(), String> {
        match &self.plan {
            PlanInstance::Incompressible(solver) => {
                solver.print_profiling_report();
                Ok(())
            }
            PlanInstance::GenericCoupled(solver) => {
                solver.linear.print_profiling_report();
                Ok(())
            }
            PlanInstance::Compressible(_) => Err(
                "print_profiling_report is supported for Incompressible models only".to_string(),
            ),
        }
    }

    pub async fn read_state_f32(&self) -> Vec<f32> {
        let stride = self.model.state_layout.stride() as u64;
        let bytes = self.num_cells() as u64 * stride * 4;
        match &self.plan {
            PlanInstance::Incompressible(solver) => {
                let raw = solver.read_buffer(&solver.b_state, bytes).await;
                bytemuck::cast_slice(&raw).to_vec()
            }
            PlanInstance::Compressible(solver) => {
                let raw = solver.read_buffer(&solver.b_state, bytes).await;
                bytemuck::cast_slice(&raw).to_vec()
            }
            PlanInstance::GenericCoupled(solver) => {
                let raw = solver.linear.read_buffer(solver.state_buffer(), bytes).await;
                bytemuck::cast_slice(&raw).to_vec()
            }
        }
    }

    pub fn set_field_scalar(&mut self, field: &str, values: &[f64]) -> Result<(), String> {
        match &mut self.plan {
            PlanInstance::GenericCoupled(solver) => solver.set_field_scalar(field, values),
            _ => Err("set_field_scalar is supported for GenericCoupled models only".to_string()),
        }
    }

    pub async fn get_field_scalar(&self, field: &str) -> Result<Vec<f64>, String> {
        match &self.plan {
            PlanInstance::GenericCoupled(solver) => solver.get_field_scalar(field).await,
            _ => Err("get_field_scalar is supported for GenericCoupled models only".to_string()),
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

    pub fn set_linear_system(&self, matrix_values: &[f32], rhs: &[f32]) -> Result<(), String> {
        match &self.plan {
            PlanInstance::Incompressible(solver) => {
                solver.set_linear_system(matrix_values, rhs);
                Ok(())
            }
            PlanInstance::GenericCoupled(solver) => {
                solver.linear.set_linear_system(matrix_values, rhs);
                Ok(())
            }
            PlanInstance::Compressible(_) => Err(
                "set_linear_system is supported for Incompressible and GenericCoupled backends only"
                    .to_string(),
            ),
        }
    }

    pub fn solve_linear_system_cg_with_size(
        &self,
        n: u32,
        max_iters: u32,
        tol: f32,
    ) -> Result<LinearSolverStats, String> {
        match &self.plan {
            PlanInstance::Incompressible(solver) => Ok(solver.solve_linear_system_cg_with_size(n, max_iters, tol)),
            PlanInstance::GenericCoupled(solver) => Ok(solver.linear.solve_linear_system_cg_with_size(n, max_iters, tol)),
            PlanInstance::Compressible(_) => Err(
                "solve_linear_system_cg_with_size is supported for Incompressible and GenericCoupled backends only"
                    .to_string(),
            ),
        }
    }

    pub async fn get_linear_solution(&self) -> Result<Vec<f32>, String> {
        match &self.plan {
            PlanInstance::Incompressible(solver) => Ok(solver.get_linear_solution().await),
            PlanInstance::GenericCoupled(solver) => Ok(solver.linear.get_linear_solution().await),
            PlanInstance::Compressible(_) => Err(
                "get_linear_solution is supported for Incompressible and GenericCoupled backends only"
                    .to_string(),
            ),
        }
    }

    pub fn coupled_unknowns(&self) -> Result<u32, String> {
        match &self.plan {
            PlanInstance::Incompressible(solver) => Ok(solver.coupled_unknowns()),
            PlanInstance::GenericCoupled(solver) => Ok(solver.linear.coupled_unknowns()),
            PlanInstance::Compressible(_) => Err(
                "coupled_unknowns is supported for Incompressible and GenericCoupled backends only"
                    .to_string(),
            ),
        }
    }

    pub fn fgmres_sizing(&mut self, max_restart: usize) -> Result<FgmresSizing, String> {
        let (num_unknowns, num_dot_groups) = match &mut self.plan {
            PlanInstance::Incompressible(solver) => {
                let resources = solver.init_fgmres_resources(max_restart);
                (resources.fgmres.n(), resources.fgmres.num_dot_groups())
            }
            PlanInstance::GenericCoupled(solver) => {
                let resources = solver.linear.init_fgmres_resources(max_restart);
                (resources.fgmres.n(), resources.fgmres.num_dot_groups())
            }
            PlanInstance::Compressible(_) => {
                return Err(
                    "fgmres_sizing is supported for Incompressible and GenericCoupled backends only"
                        .to_string(),
                );
            }
        };
        Ok(FgmresSizing {
            num_unknowns,
            num_dot_groups,
        })
    }
}

fn apply_config_incompressible(solver: &mut GpuSolver, config: SolverConfig) {
    solver.set_scheme(config.advection_scheme.gpu_id());
    solver.set_time_scheme(config.time_scheme as u32);
    solver.set_precond_type(config.preconditioner);
}

fn apply_config_compressible(solver: &mut CompressiblePlanResources, config: SolverConfig) {
    solver.set_scheme(config.advection_scheme.gpu_id());
    solver.set_time_scheme(config.time_scheme as u32);
    solver.set_precond_type(config.preconditioner);
}
