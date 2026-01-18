use crate::solver::gpu::enums::{GpuBoundaryType, TimeScheme};
use crate::solver::gpu::plans::build_plan_instance;
use crate::solver::gpu::plans::plan_instance::{
    PlanAction, PlanInitConfig, PlanParamValue, PlanStepStats,
};
use crate::solver::gpu::plans::program::GpuProgramPlan;
use crate::solver::gpu::profiling::ProfilingStats;
use crate::solver::gpu::structs::{LinearSolverStats, PreconditionerType};
use crate::solver::mesh::Mesh;
use crate::solver::model::ModelSpec;
use crate::solver::model::backend::FieldKind;
use crate::solver::scheme::Scheme;
use crate::solver::gpu::recipe::derive_stepping_mode_from_model;
use std::sync::Arc;

pub use crate::solver::gpu::plans::plan_instance::FgmresSizing;

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

pub struct GpuUnifiedSolver {
    model: ModelSpec,
    plan: GpuProgramPlan,
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
        // Model-owned preconditioners (e.g. GenericCoupled+Schur) must remain authoritative.
        crate::solver::gpu::lowering::validate_model_owned_preconditioner_config(
            &model,
            config.preconditioner,
        )?;

        let stepping = derive_stepping_mode_from_model(&model)?;

        let plan = build_plan_instance(
            mesh,
            &model,
            PlanInitConfig {
                advection_scheme: config.advection_scheme,
                time_scheme: config.time_scheme,
                preconditioner: config.preconditioner,
                stepping,
            },
            device,
            queue,
        )
        .await?;

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
        self.plan.num_cells()
    }

    pub fn time(&self) -> f32 {
        self.plan.time()
    }

    pub fn dt(&self) -> f32 {
        self.plan.dt()
    }

    pub fn step_stats(&self) -> PlanStepStats {
        self.plan.step_stats()
    }

    pub fn state_buffer(&self) -> &wgpu::Buffer {
        self.plan.state_buffer()
    }

    pub(crate) fn set_plan_named_param(
        &mut self,
        name: &str,
        value: PlanParamValue,
    ) -> Result<(), String> {
        self.plan.set_named_param(name, value)
    }

    pub fn set_named_param(&mut self, name: &str, value: PlanParamValue) -> Result<(), String> {
        self.plan.set_named_param(name, value)
    }

    fn coupled_unknown_base_for_field(&self, field: &str) -> Option<(u32, u32)> {
        let mut idx: u32 = 0;
        for eqn in self.model.system.equations() {
            let target = eqn.target();
            let comps = target.kind().component_count() as u32;
            if target.name() == field {
                return Some((idx, comps));
            }
            idx += comps;
        }
        None
    }

    pub fn set_boundary_values(
        &mut self,
        boundary: GpuBoundaryType,
        field: &str,
        values: &[f32],
    ) -> Result<(), String> {
        let coupled_stride = self.model.system.unknowns_per_cell() as u32;
        if coupled_stride == 0 {
            return Err("model has no coupled unknowns".into());
        }
        let Some((base, comps)) = self.coupled_unknown_base_for_field(field) else {
            return Err(format!("field '{field}' is not a coupled unknown"));
        };
        if values.len() != comps as usize {
            return Err(format!(
                "field '{field}' expects {comps} component(s), got {}",
                values.len()
            ));
        }
        for (c, &v) in values.iter().enumerate() {
            self.plan
                .set_bc_value(boundary, base + c as u32, v)?;
        }
        Ok(())
    }

    pub fn set_boundary_scalar(
        &mut self,
        boundary: GpuBoundaryType,
        field: &str,
        value: f32,
    ) -> Result<(), String> {
        self.set_boundary_values(boundary, field, &[value])
    }

    pub fn set_boundary_vec2(
        &mut self,
        boundary: GpuBoundaryType,
        field: &str,
        value: [f32; 2],
    ) -> Result<(), String> {
        self.set_boundary_values(boundary, field, &value)
    }

    pub fn set_dt(&mut self, dt: f32) {
        let _ = self.plan.set_named_param("dt", PlanParamValue::F32(dt));
    }

    pub fn set_advection_scheme(&mut self, scheme: Scheme) {
        self.config.advection_scheme = scheme;
        let _ = self
            .plan
            .set_named_param("advection_scheme", PlanParamValue::Scheme(scheme));
    }

    pub fn set_time_scheme(&mut self, scheme: TimeScheme) {
        self.config.time_scheme = scheme;
        let _ = self
            .plan
            .set_named_param("time_scheme", PlanParamValue::TimeScheme(scheme));
    }

    pub fn set_preconditioner(&mut self, preconditioner: PreconditionerType) {
        // Model-owned preconditioners (e.g. GenericCoupled+Schur) must remain authoritative.
        // If the plan rejects this param, keep the config unchanged.
        if self
            .plan
            .set_named_param("preconditioner", PlanParamValue::Preconditioner(preconditioner))
            .is_ok()
        {
            self.config.preconditioner = preconditioner;
        }
    }

    pub fn step(&mut self) {
        self.plan.step();
    }

    pub fn step_with_stats(&mut self) -> Result<Vec<LinearSolverStats>, String> {
        self.plan.step_with_stats()
    }

    pub fn initialize_history(&self) {
        self.plan.initialize_history();
    }

    pub fn enable_detailed_profiling(&mut self, enable: bool) -> Result<(), String> {
        self.plan
            .set_named_param("detailed_profiling_enabled", PlanParamValue::Bool(enable))
    }

    pub fn start_profiling_session(&self) -> Result<(), String> {
        self.plan.perform(PlanAction::StartProfilingSession)
    }

    pub fn end_profiling_session(&self) -> Result<(), String> {
        self.plan.perform(PlanAction::EndProfilingSession)
    }

    pub fn get_profiling_stats(&self) -> Result<Arc<ProfilingStats>, String> {
        Ok(self.plan.profiling_stats())
    }

    pub fn print_profiling_report(&self) -> Result<(), String> {
        self.plan.perform(PlanAction::PrintProfilingReport)
    }

    pub async fn read_state_f32(&self) -> Vec<f32> {
        let stride = self.model.state_layout.stride() as u64;
        let bytes = self.num_cells() as u64 * stride * 4;
        let raw = self.plan.read_state_bytes(bytes).await;
        bytemuck::cast_slice(&raw).to_vec()
    }

    pub fn write_state_f32(&mut self, state: &[f32]) -> Result<(), String> {
        let stride = self.model.state_layout.stride() as usize;
        let expected = self.num_cells() as usize * stride;
        if state.len() != expected {
            return Err(format!(
                "state length {} does not match expected {} (= num_cells {} * stride {})",
                state.len(),
                expected,
                self.num_cells(),
                stride
            ));
        }
        self.plan.write_state_bytes(bytemuck::cast_slice(state))
    }

    pub fn set_field_scalar(&mut self, field: &str, values: &[f64]) -> Result<(), String> {
        let stride = self.model.state_layout.stride() as usize;
        let state_field = self
            .model
            .state_layout
            .field(field)
            .ok_or_else(|| format!("field '{field}' not found in layout"))?;
        if state_field.kind() != FieldKind::Scalar {
            return Err(format!(
                "field '{field}' is not scalar (kind={})",
                state_field.kind().as_str()
            ));
        }
        let offset = state_field.offset() as usize;
        if values.len() != self.num_cells() as usize {
            return Err(format!(
                "value length {} does not match num_cells {}",
                values.len(),
                self.num_cells()
            ));
        }

        let mut state = pollster::block_on(async { self.read_state_f32().await });
        if state.len() != self.num_cells() as usize * stride {
            state.resize(self.num_cells() as usize * stride, 0.0);
        }
        for (i, &v) in values.iter().enumerate() {
            state[i * stride + offset] = v as f32;
        }
        self.plan.write_state_bytes(bytemuck::cast_slice(&state))
    }

    pub async fn get_field_scalar(&self, field: &str) -> Result<Vec<f64>, String> {
        let data = self.read_state_f32().await;
        let stride = self.model.state_layout.stride() as usize;
        let state_field = self
            .model
            .state_layout
            .field(field)
            .ok_or_else(|| format!("field '{field}' not found in layout"))?;
        if state_field.kind() != FieldKind::Scalar {
            return Err(format!(
                "field '{field}' is not scalar (kind={})",
                state_field.kind().as_str()
            ));
        }
        let offset = state_field.offset() as usize;
        Ok((0..self.num_cells() as usize)
            .map(|i| data[i * stride + offset] as f64)
            .collect())
    }

    pub fn set_field_vec2(&mut self, field: &str, values: &[(f64, f64)]) -> Result<(), String> {
        let stride = self.model.state_layout.stride() as usize;
        let state_field = self
            .model
            .state_layout
            .field(field)
            .ok_or_else(|| format!("field '{field}' not found in layout"))?;
        if state_field.kind() != FieldKind::Vector2 {
            return Err(format!(
                "field '{field}' is not Vector2 (kind={})",
                state_field.kind().as_str()
            ));
        }
        let offset = state_field.offset() as usize;
        if values.len() != self.num_cells() as usize {
            return Err(format!(
                "value length {} does not match num_cells {}",
                values.len(),
                self.num_cells()
            ));
        }

        let mut state = pollster::block_on(async { self.read_state_f32().await });
        if state.len() != self.num_cells() as usize * stride {
            state.resize(self.num_cells() as usize * stride, 0.0);
        }
        for (i, &(x, y)) in values.iter().enumerate() {
            let base = i * stride + offset;
            state[base] = x as f32;
            state[base + 1] = y as f32;
        }
        self.plan.write_state_bytes(bytemuck::cast_slice(&state))
    }

    pub async fn get_field_vec2(&self, field: &str) -> Result<Vec<(f64, f64)>, String> {
        let data = self.read_state_f32().await;
        let stride = self.model.state_layout.stride() as usize;
        let state_field = self
            .model
            .state_layout
            .field(field)
            .ok_or_else(|| format!("field '{field}' not found in layout"))?;
        if state_field.kind() != FieldKind::Vector2 {
            return Err(format!(
                "field '{field}' is not Vector2 (kind={})",
                state_field.kind().as_str()
            ));
        }
        let offset = state_field.offset() as usize;
        Ok((0..self.num_cells() as usize)
            .map(|i| {
                let base = i * stride + offset;
                (data[base] as f64, data[base + 1] as f64)
            })
            .collect())
    }


    pub fn set_linear_system(&mut self, matrix_values: &[f32], rhs: &[f32]) -> Result<(), String> {
        let Some(debug) = self.plan.linear_system_debug() else {
            return Err("plan does not support linear system debug operations".into());
        };
        debug.set_linear_system(matrix_values, rhs)
    }

    pub fn solve_linear_system_with_size(
        &mut self,
        n: u32,
        max_iters: u32,
        tol: f32,
    ) -> Result<LinearSolverStats, String> {
        let Some(debug) = self.plan.linear_system_debug() else {
            return Err("plan does not support linear system debug operations".into());
        };
        debug.solve_linear_system_with_size(n, max_iters, tol)
    }

    pub async fn get_linear_solution(&mut self) -> Result<Vec<f32>, String> {
        let Some(debug) = self.plan.linear_system_debug() else {
            return Err("plan does not support linear system debug operations".into());
        };
        debug.get_linear_solution().await
    }

    pub fn coupled_unknowns(&self) -> Result<u32, String> {
        Ok(self.num_cells() * self.model.system.unknowns_per_cell())
    }

    pub fn fgmres_sizing(&mut self, max_restart: usize) -> Result<FgmresSizing, String> {
        let _ = max_restart;
        let n = self.coupled_unknowns()?;
        Ok(FgmresSizing {
            num_unknowns: n,
            num_dot_groups: (n + 63) / 64,
        })
    }

}
