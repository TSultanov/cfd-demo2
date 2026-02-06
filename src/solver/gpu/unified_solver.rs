use crate::solver::gpu::enums::{GpuBoundaryType, TimeScheme};
use crate::solver::gpu::profiling::ProfilingStats;
use crate::solver::gpu::program::build_program_plan;
use crate::solver::gpu::program::plan::{GpuProgramPlan, StepGraphTiming};
use crate::solver::gpu::program::plan_instance::{
    PlanAction, PlanInitConfig, PlanParamValue, PlanStepStats,
};
use crate::solver::gpu::recipe::SteppingMode;
use crate::solver::gpu::structs::{LinearSolverStats, PreconditionerType};
use crate::solver::mesh::Mesh;
use crate::solver::model::backend::{FieldKind, StateLayout};
use crate::solver::model::ports::PortRegistry;
use crate::solver::model::ModelSpec;
use crate::solver::scheme::Scheme;
use std::sync::Arc;

pub use crate::solver::gpu::program::plan_instance::FgmresSizing;

/// UI-relevant state access metadata.
///
/// This provides a small, stable interface for UI code to access common field offsets
/// without probing StateLayout directly. It supports both PortRegistry (preferred at runtime)
/// and StateLayout (fallback for pre-solver inspection) sources.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UiPortSet {
    /// State stride (number of floats per cell).
    pub stride: u32,
    /// Offset for velocity field (U or u), if present.
    pub u_offset: Option<u32>,
    /// Offset for pressure field (p), if present.
    pub p_offset: Option<u32>,
}

impl UiPortSet {
    /// Create a UiPortSet from a PortRegistry (preferred method at runtime).
    ///
    /// Validates that U/u fields are Vector2 (2 components) and p is Scalar (1 component).
    /// Returns a UiPortSet with optional offsets - each field is independently validated.
    pub fn from_registry(registry: &PortRegistry) -> Self {
        let stride = registry.state_layout().stride();

        // Try "U" first, then "u" for velocity
        let u_offset = registry
            .get_field_entry_by_name("U")
            .or_else(|| registry.get_field_entry_by_name("u"))
            .filter(|entry| entry.component_count() == 2) // must be vec2
            .map(|entry| entry.offset());

        // Get pressure field - must be scalar (1 component)
        let p_offset = registry
            .get_field_entry_by_name("p")
            .filter(|entry| entry.component_count() == 1) // must be scalar
            .map(|entry| entry.offset());

        Self {
            stride,
            u_offset,
            p_offset,
        }
    }

    /// Create a UiPortSet from a StateLayout (fallback for pre-solver model inspection).
    /// Returns a UiPortSet with optional offsets - each field is independently validated.
    pub fn from_layout(layout: &StateLayout) -> Self {
        let stride = layout.stride();

        // Scan layout.fields() once to collect offsets by name/kind
        let mut u_offset: Option<u32> = None;
        let mut p_offset: Option<u32> = None;

        for field in layout.fields() {
            let name = field.name();
            let kind = field.kind();
            let offset = field.offset();

            // Try "U" first, then "u" for velocity - must be Vector2
            if kind == FieldKind::Vector2 && (name == "U" || (name == "u" && u_offset.is_none())) {
                u_offset = Some(offset);
            }
            // Get pressure field - must be Scalar
            else if kind == FieldKind::Scalar && name == "p" {
                p_offset = Some(offset);
            }
        }

        Self {
            stride,
            u_offset,
            p_offset,
        }
    }

    /// Returns true if both velocity (U/u) and pressure (p) fields are present.
    pub fn is_complete(&self) -> bool {
        self.u_offset.is_some() && self.p_offset.is_some()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SolverConfig {
    pub advection_scheme: Scheme,
    pub time_scheme: TimeScheme,
    pub preconditioner: PreconditionerType,
    pub stepping: SteppingMode,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            advection_scheme: Scheme::Upwind,
            time_scheme: TimeScheme::Euler,
            preconditioner: PreconditionerType::Jacobi,
            stepping: SteppingMode::Coupled,
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

        let plan = build_program_plan(
            mesh,
            &model,
            PlanInitConfig {
                advection_scheme: config.advection_scheme,
                time_scheme: config.time_scheme,
                preconditioner: config.preconditioner,
                stepping: config.stepping,
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

    /// Access the cached port registry from the plan resources.
    /// Returns `None` if the registry is not available.
    pub fn port_registry(&self) -> Option<&PortRegistry> {
        self.plan
            .resources
            .get::<Arc<PortRegistry>>()
            .map(|arc| arc.as_ref())
    }

    /// Get the UI port set for accessing common field offsets.
    ///
    /// Prefers the PortRegistry when available (runtime), falling back to
    /// the model's StateLayout for pre-solver inspection.
    pub fn ui_ports(&self) -> UiPortSet {
        if let Some(registry) = self.port_registry() {
            UiPortSet::from_registry(registry)
        } else {
            UiPortSet::from_layout(&self.model.state_layout)
        }
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

    pub fn set_collect_convergence_stats(&mut self, enable: bool) {
        self.plan.collect_convergence_stats = enable;
    }

    pub fn set_collect_trace(&mut self, enable: bool) {
        self.plan.collect_trace = enable;
    }

    pub fn outer_field_residuals(&self) -> Option<&[(String, f32)]> {
        if self.plan.outer_field_residuals.is_empty() {
            None
        } else {
            Some(&self.plan.outer_field_residuals)
        }
    }

    pub fn outer_field_residuals_scaled(&self) -> Option<&[(String, f32)]> {
        if self.plan.outer_field_residuals_scaled.is_empty() {
            None
        } else {
            Some(&self.plan.outer_field_residuals_scaled)
        }
    }

    pub fn step_graph_timings(&self) -> &[StepGraphTiming] {
        &self.plan.step_graph_timings
    }

    pub fn state_buffer(&self) -> &wgpu::Buffer {
        self.plan.state_buffer()
    }

    pub fn state_size_bytes(&self) -> u64 {
        self.num_cells() as u64
            * self.model.state_layout.stride() as u64
            * std::mem::size_of::<f32>() as u64
    }

    pub fn copy_state_to_buffer(&self, dst: &wgpu::Buffer) {
        let size_bytes = self.state_size_bytes();
        if size_bytes == 0 {
            return;
        }

        let device = &self.plan.context.device;
        let queue = &self.plan.context.queue;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GpuUnifiedSolver:copy_state_to_buffer"),
        });
        encoder.copy_buffer_to_buffer(self.state_buffer(), 0, dst, 0, size_bytes);
        queue.submit(Some(encoder.finish()));
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
        let coupled_stride = self.model.system.unknowns_per_cell();
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
            self.plan.set_bc_value(boundary, base + c as u32, v)?;
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
            .set_named_param(
                "preconditioner",
                PlanParamValue::Preconditioner(preconditioner),
            )
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
            num_dot_groups: n.div_ceil(64),
        })
    }
}
