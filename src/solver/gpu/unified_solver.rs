use crate::solver::gpu::enums::{GpuBoundaryType, TimeScheme};
use crate::solver::gpu::profiling::ProfilingStats;
use crate::solver::gpu::program::build_program_plan;
use crate::solver::gpu::program::plan::{GpuProgramPlan, StepGraphTiming};
use crate::solver::gpu::program::plan_instance::{PlanAction, PlanInitConfig, PlanStepStats};
use crate::solver::gpu::recipe::SteppingMode;
use crate::solver::gpu::structs::{LinearSolverStats, PreconditionerType};
use crate::solver::mesh::{Mesh, MeshRefreshLevel};
use crate::solver::model::backend::{FieldKind, StateLayout};
use crate::solver::model::ports::PortRegistry;
use crate::solver::model::ModelSpec;
use crate::solver::scheme::Scheme;
use std::sync::Arc;

pub use crate::solver::gpu::program::plan_instance::FgmresSizing;
// Re-exported: `set_named_param` is public API, so its value type must be
// publicly nameable (the defining module is crate-private).
pub use crate::solver::gpu::program::plan_instance::PlanParamValue;

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

/// Internal backend implementation held by the solver.
enum SolverBackend {
    Gpu(GpuProgramPlan),
    #[cfg(feature = "cpu")]
    Cpu(Box<crate::solver::cpu::CpuSolver>),
}

/// GUI render bridge for the CPU backend: a wgpu mirror of the CPU state buffer,
/// uploaded after each step/write so the existing wgpu visualisation path
/// (`state_buffer`/`copy_state_to_buffer`) works unchanged. Present only when the
/// caller (the GUI) supplies a device+queue.
#[cfg(feature = "cpu")]
struct CpuRender {
    queue: wgpu::Queue,
    buffer: wgpu::Buffer,
    capacity: u64,
}

pub struct GpuUnifiedSolver {
    model: ModelSpec,
    backend: SolverBackend,
    config: SolverConfig,
    /// Post-step State-Redistribution operator for cut-cell small cells. `None`
    /// unless the mesh carries sliver cut cells (see [`crate::solver::gpu::srd`]).
    /// Built but **not applied by default**: the cut-cell small-cell instability
    /// is properly fixed by the immersed no-slip wall BC (see
    /// `generate_cut_cell_mesh`), which also produces the physical boundary
    /// layer. SRD is retained as an opt-in (`set_srd_enabled`) stabilizer.
    srd: Option<crate::solver::gpu::srd::SrdGpu>,
    /// Runtime toggle for the SRD pass (**default off**; only meaningful when
    /// `srd` is `Some`). Opt-in via [`Self::set_srd_enabled`].
    srd_enabled: bool,
    /// ALE sequencing guard: set by [`Self::begin_ale_step`], cleared by
    /// `step`/`step_with_stats`. A second `begin_ale_step` without an
    /// intervening step would double-rotate the volume history (the `V^n`
    /// slot silently becomes `V^{n+1}` — a corrupted moving-volume ddt), so
    /// double-arming is rejected.
    ale_step_armed: bool,
    #[cfg(feature = "cpu")]
    cpu_render: Option<CpuRender>,
}

impl GpuUnifiedSolver {
    pub async fn new(
        mesh: &Mesh,
        model: ModelSpec,
        config: SolverConfig,
        device: Option<wgpu::Device>,
        queue: Option<wgpu::Queue>,
    ) -> Result<Self, String> {
        #[cfg(feature = "cpu")]
        if let Some(cpu_cfg) = cpu_backend_from_env() {
            // Honor the requested stepping mode (config.stepping) exactly like the
            // GPU path below — the GUI selects Implicit for compressible and Coupled
            // for the saddle-point models, so the CPU backend must use the same to
            // match the GPU's outer-loop behavior (was hardcoded Coupled via
            // CpuSolver::new, which mis-stepped compressible in the GUI).
            let cpu = crate::solver::cpu::CpuSolver::with_stepping(
                mesh,
                model.clone(),
                config.advection_scheme,
                config.time_scheme,
                config.stepping,
                cpu_cfg,
            )?;
            // Optional GUI render mirror (the GUI supplies device+queue).
            let cpu_render = match (device.as_ref(), queue) {
                (Some(dev), Some(q)) => {
                    let bytes =
                        cpu.num_cells() as u64 * model.state_layout.stride() as u64 * 4;
                    let buffer = dev.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("CpuUnifiedSolver:state_mirror"),
                        size: bytes.max(4),
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    Some(CpuRender {
                        queue: q,
                        buffer,
                        capacity: bytes,
                    })
                }
                _ => None,
            };
            let solver = Self {
                model,
                backend: SolverBackend::Cpu(Box::new(cpu)),
                config,
                // SRD is a GPU-only cut-cell stabilizer; the CPU backend never
                // builds or applies it.
                srd: None,
                srd_enabled: false,
                ale_step_armed: false,
                cpu_render,
            };
            solver.sync_cpu_render();
            return Ok(solver);
        }

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

        let mut solver = Self {
            model,
            backend: SolverBackend::Gpu(plan),
            config,
            srd: None,
            srd_enabled: false,
            ale_step_armed: false,
            #[cfg(feature = "cpu")]
            cpu_render: None,
        };

        // Build the cut-cell State-Redistribution operator from the mesh (kept
        // available as an opt-in stabilizer; NOT applied by default — the
        // immersed no-slip wall BC is the primary small-cell fix). `None` unless
        // the mesh carries sliver cut cells, so structured/graded meshes are
        // untouched.
        let ports = solver.ui_ports();
        if let Some(u_offset) = ports.u_offset {
            // Borrow the GPU plan to build the operator in an inner scope, then
            // assign (the plan borrow must end before writing `solver.srd`).
            // `plan()` is safe here — the CPU backend returned early above.
            let srd = {
                let plan = solver.plan();
                crate::solver::gpu::srd::build_srd_operator(mesh).map(|csr| {
                    crate::solver::gpu::srd::SrdGpu::new(
                        &plan.context.device,
                        &csr,
                        u_offset,
                        ports.stride,
                        mesh.num_cells() as u32,
                    )
                })
            };
            solver.srd = srd;
        }

        Ok(solver)
    }

    /// Apply the post-step SRD pass to the live velocity field, if built and
    /// enabled. A no-op when the mesh has no slivers (`self.srd` is `None`).
    fn apply_srd(&self) {
        if !self.srd_enabled {
            return;
        }
        if let Some(srd) = self.srd.as_ref() {
            srd.apply(
                &self.plan().context.device,
                &self.plan().context.queue,
                self.plan().state_buffer(),
            );
        }
    }

    /// Whether a cut-cell SRD operator was built for this solver's mesh.
    pub fn srd_active(&self) -> bool {
        self.srd.is_some()
    }

    /// Enable/disable the post-step SRD pass (**default off**). Only meaningful
    /// when [`Self::srd_active`]. Opt-in stabilizer for cut-cell slivers; the
    /// immersed no-slip wall BC is the primary fix, so this is normally left off
    /// (it slightly smooths the near-wall boundary layer). Also used by the
    /// GPU-vs-CPU cross-check and the boundary-layer diagnostic.
    pub fn set_srd_enabled(&mut self, enabled: bool) {
        self.srd_enabled = enabled;
    }

    /// Manually run one SRD pass over the current velocity field, ignoring the
    /// enable toggle. Normally SRD runs automatically after each step; this is
    /// for tests that apply `S` to a known field without stepping.
    pub fn apply_srd_pass(&self) {
        if let Some(srd) = self.srd.as_ref() {
            srd.apply(
                &self.plan().context.device,
                &self.plan().context.queue,
                self.plan().state_buffer(),
            );
        }
    }

    /// Upload the current CPU state into the render-mirror buffer (no-op unless on
    /// the CPU backend with a render bridge).
    #[cfg(feature = "cpu")]
    fn sync_cpu_render(&self) {
        if let (SolverBackend::Cpu(c), Some(r)) = (&self.backend, &self.cpu_render) {
            let bytes = c.read_state_f32();
            let n = (r.capacity as usize / 4).min(bytes.len());
            r.queue
                .write_buffer(&r.buffer, 0, bytemuck::cast_slice(&bytes[..n]));
        }
    }

    /// Access the GPU plan (panics on the CPU backend; only GPU-specific methods
    /// — profiling, render-buffer access, linear-system debug — call this).
    fn plan(&self) -> &GpuProgramPlan {
        match &self.backend {
            SolverBackend::Gpu(p) => p,
            #[cfg(feature = "cpu")]
            SolverBackend::Cpu(_) => panic!("GPU-only operation invoked on the CPU backend"),
        }
    }

    fn plan_mut(&mut self) -> &mut GpuProgramPlan {
        match &mut self.backend {
            SolverBackend::Gpu(p) => p,
            #[cfg(feature = "cpu")]
            SolverBackend::Cpu(_) => panic!("GPU-only operation invoked on the CPU backend"),
        }
    }

    #[cfg(feature = "cpu")]
    fn cpu_ref(&self) -> Option<&crate::solver::cpu::CpuSolver> {
        match &self.backend {
            SolverBackend::Cpu(c) => Some(c),
            _ => None,
        }
    }

    #[cfg(feature = "cpu")]
    fn cpu_mut(&mut self) -> Option<&mut crate::solver::cpu::CpuSolver> {
        match &mut self.backend {
            SolverBackend::Cpu(c) => Some(c),
            _ => None,
        }
    }

    /// True if this solver is running on the CPU backend.
    pub fn is_cpu(&self) -> bool {
        #[cfg(feature = "cpu")]
        {
            matches!(self.backend, SolverBackend::Cpu(_))
        }
        #[cfg(not(feature = "cpu"))]
        {
            false
        }
    }

    pub fn model(&self) -> &ModelSpec {
        &self.model
    }

    pub fn config(&self) -> SolverConfig {
        self.config
    }

    /// Access the cached port registry from the plan resources.
    /// Returns `None` if the registry is not available (e.g. CPU backend).
    pub fn port_registry(&self) -> Option<&PortRegistry> {
        match &self.backend {
            SolverBackend::Gpu(p) => Some(p.resources.port_registry.as_ref()),
            #[cfg(feature = "cpu")]
            SolverBackend::Cpu(_) => None,
        }
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
        #[cfg(feature = "cpu")]
        if let Some(c) = self.cpu_ref() {
            return c.num_cells();
        }
        self.plan().num_cells()
    }

    pub fn time(&self) -> f32 {
        #[cfg(feature = "cpu")]
        if let Some(c) = self.cpu_ref() {
            return c.time();
        }
        self.plan().time()
    }

    pub fn dt(&self) -> f32 {
        #[cfg(feature = "cpu")]
        if let Some(c) = self.cpu_ref() {
            return c.dt();
        }
        self.plan().dt()
    }

    pub fn step_stats(&self) -> PlanStepStats {
        #[cfg(feature = "cpu")]
        if let Some(c) = self.cpu_ref() {
            // CPU has no GPU convergence monitor / per-graph telemetry, but it can
            // report steady-state auto-pause (should_stop) so the GUI behaves like
            // the GPU under pseudo-transient continuation, plus the outer count
            // actually executed (the CPU plateau detector's early exit).
            let mut stats = PlanStepStats::default();
            stats.should_stop = Some(c.should_stop());
            stats.outer_iterations = Some(c.outer_iterations_done());
            return stats;
        }
        self.plan().step_stats()
    }

    #[allow(irrefutable_let_patterns)]
    pub fn set_collect_convergence_stats(&mut self, enable: bool) {
        match &mut self.backend {
            SolverBackend::Gpu(p) => p.collect_convergence_stats = enable,
            #[cfg(feature = "cpu")]
            SolverBackend::Cpu(c) => c.set_collect_convergence_stats(enable),
        }
    }

    #[allow(irrefutable_let_patterns)]
    pub fn set_collect_trace(&mut self, enable: bool) {
        if let SolverBackend::Gpu(p) = &mut self.backend {
            p.collect_trace = enable;
        }
    }

    pub fn outer_field_residuals(&self) -> Option<&[(String, f32)]> {
        match &self.backend {
            SolverBackend::Gpu(p) if !p.outer_field_residuals.is_empty() => {
                Some(&p.outer_field_residuals)
            }
            _ => None,
        }
    }

    pub fn outer_field_residuals_scaled(&self) -> Option<&[(String, f32)]> {
        match &self.backend {
            SolverBackend::Gpu(p) if !p.outer_field_residuals_scaled.is_empty() => {
                Some(&p.outer_field_residuals_scaled)
            }
            _ => None,
        }
    }

    pub fn step_graph_timings(&self) -> &[StepGraphTiming] {
        match &self.backend {
            SolverBackend::Gpu(p) => &p.step_graph_timings,
            #[cfg(feature = "cpu")]
            SolverBackend::Cpu(_) => &[],
        }
    }

    pub fn state_buffer(&self) -> &wgpu::Buffer {
        #[cfg(feature = "cpu")]
        if let Some(r) = &self.cpu_render {
            return &r.buffer;
        }
        self.plan().state_buffer()
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

        #[cfg(feature = "cpu")]
        if let (SolverBackend::Cpu(c), Some(r)) = (&self.backend, &self.cpu_render) {
            let bytes = c.read_state_f32();
            let n = (size_bytes as usize / 4).min(bytes.len());
            r.queue
                .write_buffer(dst, 0, bytemuck::cast_slice(&bytes[..n]));
            // Flush the upload now (an empty submit drains the staging belt), so the
            // GUI render thread sees the new state this frame. The GPU branch below
            // submits its copy explicitly; without this, the CPU write_buffer would
            // only flush at egui's next submit — racy across the worker/render
            // threads, which manifested as "Run does nothing visible" on CPU.
            r.queue.submit(std::iter::empty());
            return;
        }

        let device = &self.plan().context.device;
        let queue = &self.plan().context.queue;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GpuUnifiedSolver:copy_state_to_buffer"),
        });
        encoder.copy_buffer_to_buffer(self.state_buffer(), 0, dst, 0, size_bytes);
        queue.submit(Some(encoder.finish()));
        crate::count_submission!("Unified Solver", "copy_state_to_buffer");
    }

    pub(crate) fn set_plan_named_param(
        &mut self,
        name: &str,
        value: PlanParamValue,
    ) -> Result<(), String> {
        self.set_named_param(name, value)
    }

    pub fn set_named_param(&mut self, name: &str, value: PlanParamValue) -> Result<(), String> {
        match &mut self.backend {
            SolverBackend::Gpu(p) => p.set_named_param(name, value),
            #[cfg(feature = "cpu")]
            SolverBackend::Cpu(c) => {
                cpu_set_param(c, name, value);
                Ok(())
            }
        }
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
            let comp = base + c as u32;
            match &mut self.backend {
                SolverBackend::Gpu(p) => p.set_bc_value(boundary, comp, v)?,
                #[cfg(feature = "cpu")]
                SolverBackend::Cpu(cpu) => {
                    // CPU resolves field+component internally; pass raw component.
                    cpu.set_boundary_values_per_face(boundary, field, c as u32, &|_| v)?
                }
            }
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

    /// Write spatially varying boundary values for one component of `field`:
    /// `value_for_face(face_idx)` is evaluated per boundary face (mesh face index) of the
    /// given boundary type. Geometry lookups (e.g. face centers) are the caller's
    /// responsibility via the mesh used to build this solver.
    pub fn set_boundary_values_per_face(
        &mut self,
        boundary: GpuBoundaryType,
        field: &str,
        component: u32,
        value_for_face: &dyn Fn(u32) -> f32,
    ) -> Result<(), String> {
        let Some((base, comps)) = self.coupled_unknown_base_for_field(field) else {
            return Err(format!("field '{field}' is not a coupled unknown"));
        };
        if component >= comps {
            return Err(format!(
                "component {component} out of range for field '{field}' ({comps} component(s))"
            ));
        }
        match &mut self.backend {
            SolverBackend::Gpu(p) => {
                p.set_bc_values_per_face(boundary, base + component, value_for_face)
            }
            #[cfg(feature = "cpu")]
            SolverBackend::Cpu(c) => {
                // The CPU method resolves field+component to the coupled u_idx
                // itself (mirroring `coupled_unknown_base_for_field`), so pass
                // the raw component, not `base + component`.
                let _ = base;
                c.set_boundary_values_per_face(boundary, field, component, value_for_face)
            }
        }
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
        let _ = self.set_named_param("dt", PlanParamValue::F32(dt));
    }

    pub fn set_advection_scheme(&mut self, scheme: Scheme) {
        self.config.advection_scheme = scheme;
        let _ = self.set_named_param("advection_scheme", PlanParamValue::Scheme(scheme));
    }

    pub fn set_time_scheme(&mut self, scheme: TimeScheme) {
        self.config.time_scheme = scheme;
        let _ = self.set_named_param("time_scheme", PlanParamValue::TimeScheme(scheme));
    }

    pub fn set_preconditioner(&mut self, preconditioner: PreconditionerType) {
        // Model-owned preconditioners (e.g. GenericCoupled+Schur) must remain authoritative.
        // If the plan rejects this param, keep the config unchanged.
        if self
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
        self.ale_step_armed = false;
        match &mut self.backend {
            SolverBackend::Gpu(p) => p.step(),
            #[cfg(feature = "cpu")]
            SolverBackend::Cpu(c) => c.step(),
        }
        // Post-step cut-cell State Redistribution (GPU only; no-op when disabled
        // or no slivers — `srd` is `None` for the CPU backend).
        self.apply_srd();
        // Mirror the CPU state into the GUI render buffer (no-op without a render
        // bridge / on the GPU backend).
        #[cfg(feature = "cpu")]
        self.sync_cpu_render();
    }

    pub fn step_with_stats(&mut self) -> Result<Vec<LinearSolverStats>, String> {
        self.ale_step_armed = false;
        let stats = match &mut self.backend {
            SolverBackend::Gpu(p) => p.step_with_stats()?,
            #[cfg(feature = "cpu")]
            SolverBackend::Cpu(c) => {
                c.step();
                Vec::new()
            }
        };
        self.apply_srd();
        Ok(stats)
    }

    pub fn initialize_history(&self) {
        match &self.backend {
            SolverBackend::Gpu(p) => p.initialize_history(),
            #[cfg(feature = "cpu")]
            SolverBackend::Cpu(c) => c.initialize_history(),
        }
    }

    /// Refresh the solver's mesh-derived state after the caller's `Mesh` changed
    /// (M2 of the meshless/moving-mesh roadmap).
    ///
    /// [`MeshRefreshLevel::Geometry`] (Tier A): the mesh must be
    /// topology-identical to the build-time mesh — same cell/face counts, same
    /// owners/neighbors/boundary tags, same cell→face connectivity (validated;
    /// `Err` on any mismatch) — with only positions moved. The six geometry
    /// arrays (face areas/normals/centers/wrap shifts, cell centers/volumes)
    /// are overwritten in place from the shared f64→f32 cast, on both backends.
    /// Cell-indexed solver state (state ×3 history, warm-start `x`) is never
    /// touched: cell `i` keeps its identity.
    ///
    /// [`MeshRefreshLevel::Topology`] (Tier B) is not yet implemented.
    ///
    /// Note: the opt-in SRD operator (cut-cell sliver stabilizer) bakes mesh
    /// geometry at build and is NOT rebuilt here; refresh is refused while SRD
    /// is enabled (it is default-off, and unsupported with mesh motion in v1).
    pub fn refresh_mesh(
        &mut self,
        mesh: &Mesh,
        level: MeshRefreshLevel,
    ) -> Result<crate::solver::MeshRefreshReport, String> {
        if self.srd_enabled && self.srd.is_some() {
            return Err(
                "mesh refresh is unsupported with the SRD stabilizer enabled (its operator \
                 bakes build-time mesh geometry)"
                    .into(),
            );
        }
        // ALE sequencing guard: on an ALE model, a plain geometry refresh
        // updates `cell_vols` WITHOUT rotating the volume history and leaves
        // the (stale, likely zero) mesh fluxes bound — the moving-volume ddt
        // then sees an inconsistent (V^{n+1}, V^n) pair and the SCL breaks
        // silently. The ALE seam is `begin_ale_step` (rotation + geometry +
        // fluxes in the correct order); use it for any mid-run mesh change.
        // Same hazard applies to a Topology refresh (it reallocates the mesh
        // fluxes zero-filled), so both levels are rejected on ALE models.
        if self.model.system.is_ale() {
            return Err(
                "refresh_mesh on an ALE model is rejected: it would update cell volumes \
                 without rotating the volume history or supplying mesh fluxes (silent SCL \
                 violation). Use begin_ale_step for mesh motion on ALE models."
                    .into(),
            );
        }
        match level {
            MeshRefreshLevel::Geometry => {
                match &mut self.backend {
                    SolverBackend::Gpu(p) => p.refresh_mesh_geometry(mesh)?,
                    #[cfg(feature = "cpu")]
                    SolverBackend::Cpu(c) => c.refresh_mesh_geometry(mesh)?,
                }
                Ok(crate::solver::MeshRefreshReport::default())
            }
            MeshRefreshLevel::Topology => match &mut self.backend {
                SolverBackend::Gpu(p) => p.refresh_mesh_topology(mesh),
                #[cfg(feature = "cpu")]
                SolverBackend::Cpu(c) => c.refresh_mesh_topology(mesh),
            },
        }
    }

    /// Capture the solver's stepping state (M2 Tier B stage 3) — see
    /// [`crate::solver::SolverStateSnapshot`]. The **CPU** backend captures the
    /// full time/warm-start/volume history (`has_history == true`), so a fresh
    /// solver restored from it reproduces the next step byte-identically. The
    /// **GPU** backend captures the current state + scalars only
    /// (`has_history == false`); history/warm-start readback plumbing is a later
    /// stage, so a GPU restore re-seeds the history from the current state.
    pub fn snapshot(&self) -> crate::solver::SolverStateSnapshot {
        #[cfg(feature = "cpu")]
        if let Some(c) = self.cpu_ref() {
            return c.snapshot();
        }
        let state = pollster::block_on(self.read_state_f32());
        crate::solver::SolverStateSnapshot {
            num_cells: self.num_cells() as usize,
            num_faces: 0,
            state_stride: self.model.state_layout.stride(),
            unknowns_per_cell: self.model.system.unknowns_per_cell() as usize,
            state,
            state_old: Vec::new(),
            state_old_old: Vec::new(),
            x: Vec::new(),
            cell_vols_old: Vec::new(),
            cell_vols_old_old: Vec::new(),
            mesh_fluxes: Vec::new(),
            time: self.time(),
            dt: self.dt(),
            dt_old: self.dt(),
            dtau: 0.0,
            step_count: 0,
            last_rel_delta: f64::INFINITY,
            schur_amg_active: false,
            has_history: false,
        }
    }

    /// Restore a snapshot (M2 Tier B stage 3). CPU restores the full history
    /// byte-exactly; GPU writes the current state (initial-condition semantics
    /// propagate it to the time-history buffers) — exact for single-step
    /// schemes, a documented startup fallback for BDF2.
    pub fn restore(&mut self, snap: &crate::solver::SolverStateSnapshot) -> Result<(), String> {
        #[cfg(feature = "cpu")]
        if let Some(c) = self.cpu_mut() {
            return c.restore(snap);
        }
        snap.check_compatible(self.num_cells() as usize, self.model.state_layout.stride())?;
        self.write_state_f32(&snap.state)?;
        if snap.dt > 0.0 {
            self.set_dt(snap.dt);
        }
        Ok(())
    }

    /// ALE step entry (M3.2 of the meshless/moving-mesh roadmap): after the
    /// caller moved the mesh (topology-identical; `recalculate_geometry`
    /// already run), rotate the volume history (`cell_vols_old_old ←
    /// cell_vols_old ← cell_vols`), upload the new geometry, and upload the
    /// per-face volumetric mesh fluxes — in that order (the rotation must
    /// capture the pre-refresh volumes as `V^n`; see
    /// `MeshResources::begin_ale_step`). Call once per step, before `step()`.
    ///
    /// `mesh_fluxes` must be the f32-CLOSED swept rates from
    /// [`crate::solver::mesh::ale::swept_mesh_fluxes_closed`] (owner-signed,
    /// Volume/Time); they are uploaded verbatim so the per-cell SCL closure
    /// `Σ_f flux_f ≈ (V^{n+1}−V^n)/dt` survives byte-exactly. Only `*_ale`
    /// model kernels consume them; calling this on a static model is
    /// harmless but pointless.
    ///
    /// **dt handshake (review-solver-ale F2)**: the closure fixes ONE dt; the
    /// caller must step with exactly that dt (`set_dt` with the same value)
    /// or the SCL silently breaks (Σφ·dt ≠ ΔV). `SolverDriver::begin_ale_step`
    /// enforces this by rejecting `adaptive_dt`; raw-solver callers own the
    /// contract themselves.
    pub fn begin_ale_step(&mut self, mesh: &Mesh, mesh_fluxes: &[f32]) -> Result<(), String> {
        if self.srd_enabled && self.srd.is_some() {
            return Err(
                "ALE stepping is unsupported with the SRD stabilizer enabled (its operator \
                 bakes build-time mesh geometry)"
                    .into(),
            );
        }
        // Sequencing guard: double-arming without an intervening step would
        // rotate the volume history twice (V^n slot silently becomes V^{n+1}),
        // corrupting the moving-volume ddt with no diagnostic downstream.
        if self.ale_step_armed {
            return Err(
                "begin_ale_step called twice without an intervening step(): this would \
                 double-rotate the volume history (V^n <- V^{n+1}) and corrupt the \
                 moving-volume ddt. Call step() (or roll back and rebuild) first."
                    .into(),
            );
        }
        match &mut self.backend {
            SolverBackend::Gpu(p) => p.begin_ale_step(mesh, mesh_fluxes)?,
            #[cfg(feature = "cpu")]
            SolverBackend::Cpu(c) => c.begin_ale_step(mesh, mesh_fluxes)?,
        }
        self.ale_step_armed = true;
        Ok(())
    }

    pub fn enable_detailed_profiling(&mut self, enable: bool) -> Result<(), String> {
        self.set_named_param("detailed_profiling_enabled", PlanParamValue::Bool(enable))
    }

    pub fn start_profiling_session(&self) -> Result<(), String> {
        #[cfg(feature = "cpu")]
        if self.is_cpu() {
            return Ok(());
        }
        self.plan().perform(PlanAction::StartProfilingSession)
    }

    pub fn end_profiling_session(&self) -> Result<(), String> {
        #[cfg(feature = "cpu")]
        if self.is_cpu() {
            return Ok(());
        }
        self.plan().perform(PlanAction::EndProfilingSession)
    }

    pub fn get_profiling_stats(&self) -> Result<Arc<ProfilingStats>, String> {
        #[cfg(feature = "cpu")]
        if self.is_cpu() {
            return Err("profiling stats unavailable on the CPU backend".into());
        }
        Ok(self.plan().profiling_stats())
    }

    pub fn print_profiling_report(&self) -> Result<(), String> {
        #[cfg(feature = "cpu")]
        if self.is_cpu() {
            return Ok(());
        }
        self.plan().perform(PlanAction::PrintProfilingReport)
    }

    pub async fn read_state_f32(&self) -> Vec<f32> {
        #[cfg(feature = "cpu")]
        if let Some(c) = self.cpu_ref() {
            return c.read_state_f32();
        }
        let stride = self.model.state_layout.stride() as u64;
        let bytes = self.num_cells() as u64 * stride * 4;
        let raw = self.plan().read_state_bytes(bytes).await;
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
        #[cfg(feature = "cpu")]
        if let Some(c) = self.cpu_mut() {
            return c.write_state_f32(state);
        }
        self.plan_mut().write_state_bytes(bytemuck::cast_slice(state))
    }

    /// Set a scalar field with initial-condition semantics: the write propagates to all
    /// time-history buffers (`old`, `old_old`), as if the simulation (re)starts from this
    /// state. For mid-run updates that must not disturb multi-step time schemes (BDF2),
    /// use [`Self::set_field_scalar_current`].
    pub fn set_field_scalar(&mut self, field: &str, values: &[f64]) -> Result<(), String> {
        #[cfg(feature = "cpu")]
        if self.is_cpu() {
            if let Some(c) = self.cpu_mut() {
                c.set_field_scalar(field, values)?;
            }
            self.sync_cpu_render();
            return Ok(());
        }
        let state = self.state_with_scalar_field(field, values)?;
        self.plan_mut().write_state_bytes(bytemuck::cast_slice(&state))
    }

    /// Set a scalar field in the current state only, preserving the time history.
    /// Use for mid-run updates of non-solved fields (e.g. time-varying source terms).
    pub fn set_field_scalar_current(&mut self, field: &str, values: &[f64]) -> Result<(), String> {
        #[cfg(feature = "cpu")]
        if self.is_cpu() {
            if let Some(c) = self.cpu_mut() {
                c.set_field_scalar_current(field, values)?;
            }
            self.sync_cpu_render();
            return Ok(());
        }
        let state = self.state_with_scalar_field(field, values)?;
        self.plan_mut()
            .write_state_bytes_current(bytemuck::cast_slice(&state))
    }

    /// Read back the current state and overwrite one scalar field's slots with `values`.
    fn state_with_scalar_field(&self, field: &str, values: &[f64]) -> Result<Vec<f32>, String> {
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
        Ok(state)
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
        #[cfg(feature = "cpu")]
        if self.is_cpu() {
            if let Some(c) = self.cpu_mut() {
                c.set_field_vec2(field, values)?;
            }
            self.sync_cpu_render();
            return Ok(());
        }
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
        self.plan_mut().write_state_bytes(bytemuck::cast_slice(&state))
    }

    /// Set a Vector2 field in the current state only, preserving the time
    /// history (the Vector2 twin of [`Self::set_field_scalar_current`]). Use
    /// for mid-run updates of non-solved fields — e.g. a manufactured MMS
    /// source re-evaluated at moved cell centroids under ALE mesh motion.
    pub fn set_field_vec2_current(&mut self, field: &str, values: &[(f64, f64)]) -> Result<(), String> {
        #[cfg(feature = "cpu")]
        if self.is_cpu() {
            if let Some(c) = self.cpu_mut() {
                c.set_field_vec2_current(field, values)?;
            }
            self.sync_cpu_render();
            return Ok(());
        }
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
        self.plan_mut()
            .write_state_bytes_current(bytemuck::cast_slice(&state))
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
        let Some(debug) = self.plan_mut().linear_system_debug() else {
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
        let Some(debug) = self.plan_mut().linear_system_debug() else {
            return Err("plan does not support linear system debug operations".into());
        };
        debug.solve_linear_system_with_size(n, max_iters, tol)
    }

    pub async fn get_linear_solution(&mut self) -> Result<Vec<f32>, String> {
        let Some(debug) = self.plan_mut().linear_system_debug() else {
            return Err("plan does not support linear system debug operations".into());
        };
        debug.get_linear_solution().await
    }

    /// Read back the assembled block-CSR matrix values (length = num_nonzeros).
    /// Debug-only: CPU↔GPU matrix-level isolation (the assembled system from the
    /// most recent step / `set_linear_system`).
    pub async fn get_linear_matrix(&mut self) -> Result<Vec<f32>, String> {
        let Some(debug) = self.plan_mut().linear_system_debug() else {
            return Err("plan does not support linear system debug operations".into());
        };
        debug.get_linear_matrix().await
    }

    /// Read back the assembled right-hand side (length = num_dofs). Debug-only.
    pub async fn get_linear_rhs(&mut self) -> Result<Vec<f32>, String> {
        let Some(debug) = self.plan_mut().linear_system_debug() else {
            return Err("plan does not support linear system debug operations".into());
        };
        debug.get_linear_rhs().await
    }

    pub fn coupled_unknowns(&self) -> Result<u32, String> {
        Ok(self.num_cells() * self.model.system.unknowns_per_cell())
    }

    pub fn fgmres_sizing(&mut self) -> Result<FgmresSizing, String> {
        let n = self.coupled_unknowns()?;
        Ok(FgmresSizing {
            num_unknowns: n,
            num_dot_groups: n.div_ceil(64),
        })
    }
}

/// Select the CPU backend from the environment, returning its config when
/// `CFD2_BACKEND=cpu`. Engine/threads/SIMD come from `CFD2_CPU_ENGINE`
/// (`interpreter`|`transpiled`), `CFD2_CPU_THREADS` (integer), `CFD2_CPU_SIMD`
/// (`1`/`true`). This keeps backend selection out of `SolverConfig` so existing
/// call sites are unchanged; the GUI sets these before constructing the solver.
#[cfg(feature = "cpu")]
fn cpu_backend_from_env() -> Option<crate::solver::cpu::CpuBackendConfig> {
    use crate::solver::cpu::{CpuBackendConfig, CpuEngine};
    let on = std::env::var("CFD2_BACKEND")
        .map(|v| v.eq_ignore_ascii_case("cpu"))
        .unwrap_or(false);
    if !on {
        return None;
    }
    let engine = match std::env::var("CFD2_CPU_ENGINE").as_deref() {
        Ok(v) if v.eq_ignore_ascii_case("transpiled") || v.eq_ignore_ascii_case("transpile") => {
            CpuEngine::Transpiled
        }
        _ => CpuEngine::Interpreter,
    };
    let threads = std::env::var("CFD2_CPU_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);
    let simd = std::env::var("CFD2_CPU_SIMD")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let precision = match std::env::var("CFD2_CPU_PRECISION").as_deref() {
        Ok("f32") | Ok("F32") => crate::solver::cpu::CpuPrecision::F32,
        _ => crate::solver::cpu::CpuPrecision::F64,
    };
    Some(CpuBackendConfig { engine, threads, simd, precision })
}

/// Route a runtime named-param onto the CPU backend. Params that don't apply to
/// the CPU scalar path are ignored (GPU-specific relaxation/preconditioner knobs).
#[cfg(feature = "cpu")]
fn cpu_set_param(c: &mut crate::solver::cpu::CpuSolver, name: &str, value: PlanParamValue) {
    match (name, value) {
        ("dt", PlanParamValue::F32(v)) => c.set_dt(v),
        ("dtau", PlanParamValue::F32(v)) => c.set_dtau(v),
        ("viscosity", PlanParamValue::F32(v)) => c.set_viscosity(v),
        ("density", PlanParamValue::F32(v)) => c.set_density(v),
        ("alpha_u", PlanParamValue::F32(v)) => c.set_alpha_u(v),
        ("alpha_p", PlanParamValue::F32(v)) => c.set_alpha_p(v),
        ("outer_tol", PlanParamValue::F32(v)) => c.set_outer_tolerance(v as f64),
        ("advection_scheme", PlanParamValue::Scheme(s)) => c.set_advection_scheme(s),
        ("time_scheme", PlanParamValue::TimeScheme(s)) => c.set_time_scheme(s),
        ("outer_iters", PlanParamValue::Usize(n)) => c.set_outer_iters(n),
        ("outer_iters", PlanParamValue::U32(n)) => c.set_outer_iters(n as usize),
        // EOS runtime tuning (compressible): gamma/gm1/r/dp_drho/p_offset/theta_ref.
        // Mirrors the GPU `set_eos` so the GUI's fluid controls affect the CPU too.
        (n, PlanParamValue::F32(v)) if n.starts_with("eos.") => {
            c.set_eos_param(n, v);
        }
        // outer_tol_abs / fixed-iteration / batched modes, low_mach.* preconditioner
        // tuning, nonconverged_* retry policy, and the `preconditioner` selector have
        // no CPU analogue: the CPU driver runs the requested outer_iters with a
        // relative break and uses model-owned preconditioners (Schur / block-Jacobi),
        // so these GPU-solver-internal knobs are intentionally ignored.
        _ => {}
    }
}
