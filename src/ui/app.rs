use crate::solver::mesh::{
    generate_cut_cell_mesh, generate_cvt_mesh, generate_delaunay_mesh,
    generate_structured_nozzle_mesh, generate_voronoi_mesh, BackwardsStep, BoundarySides,
    BoundaryType, ChannelWithObstacle, LloydConfig, Mesh, Nozzle,
};
use crate::solver::model::{
    all_models, compressible_model_with_eos, ModelPreconditionerSpec, ModelSpec,
};
use crate::solver::scheme::Scheme;
use crate::solver::{
    GpuLowMachPrecondModel, LinearSolverStats, OuterStepStatus, PreconditionerType,
    TimeScheme as GpuTimeScheme, UiPortSet,
};
use crate::trace as tracefmt;
use crate::ui::{cfd_renderer, fluid::Fluid};
use eframe::egui;
use egui_plot::{Plot, PlotPoints, Polygon};
use nalgebra::{Point2, Vector2};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use crate::sim::{DivergeReason, DriverBuild, RuntimeParams, SolverDriver};

/// Rendering mode for the mesh visualization
#[derive(PartialEq, Clone, Copy)]
enum RenderMode {
    /// Use egui_plot polygons (slow for large meshes)
    EguiPlot,
    /// Render directly on GPU (zero-copy)
    GpuDirect,
}

#[derive(PartialEq, Clone, Copy)]
enum GeometryType {
    BackwardsStep,
    ChannelObstacle,
    /// Converging–diverging nozzle (structured, no cut cells). Paired with an
    /// all-Mach model + a sub-critical outlet back-pressure it drives the
    /// supersonic demo (subsonic inflow → choked throat → supersonic exit).
    Nozzle,
}

impl Default for GeometryType {
    fn default() -> Self {
        Self::BackwardsStep
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
enum MeshType {
    CutCell,
    Delaunay,
    Voronoi,
    /// Meshless CVT (Lloyd-relaxed) Voronoi mesh (`generate_cvt_mesh`). Seed
    /// positions are optimized instead of running `Mesh::smooth`: vertex
    /// smoothing would move Voronoi vertices off the bisectors, so this mesh
    /// type must never be smoothed after generation.
    VoronoiCvt,
    /// Body-fitted curvilinear structured grid. Only meaningful — and only
    /// offered in the UI — for the converging–diverging nozzle geometry, whose
    /// walls it conforms to exactly (see `generate_structured_nozzle_mesh`).
    Fitted,
}

impl MeshType {
    /// The fitted (structured) mesh is a uniform curvilinear grid: it has no
    /// local refinement or grading, so only a single target cell size applies.
    /// The unstructured meshers honour the full min/max/growth sizing controls.
    fn supports_size_grading(self) -> bool {
        !matches!(self, MeshType::Fitted)
    }
}

/// Build a slider whose range expands to include the current value and which
/// never snaps a hand-typed value back to an end (`SliderClamping::Never`).
///
/// The user can type any value into the drag-value field — finer or coarser than
/// the nominal `base` range — and it stays put; the visible track adapts around it
/// so the handle stays reachable. Dragging the handle is still bounded to the
/// (adapted) track. Consumers add `.text(…)`, `.logarithmic(…)`, etc. as usual.
fn adaptive_slider<Num: egui::emath::Numeric>(
    value: &mut Num,
    base: std::ops::RangeInclusive<Num>,
) -> egui::Slider<'_> {
    let current = value.to_f64();
    let lo = (*base.start()).to_f64().min(current);
    let hi = (*base.end()).to_f64().max(current);
    egui::Slider::new(value, Num::from_f64(lo)..=Num::from_f64(hi))
        .clamping(egui::SliderClamping::Never)
}

impl Default for MeshType {
    fn default() -> Self {
        Self::CutCell
    }
}

#[derive(PartialEq, Clone, Copy)]
enum PlotField {
    Pressure,
    VelocityX,
    VelocityY,
    VelocityMag,
}

struct PlotCache {
    snapshot_seq: u64,
    field: PlotField,
    min: f64,
    max: f64,
    values: Option<Vec<f64>>,
}

#[derive(Default, Clone)]
struct ModelUiCaps {
    plot_stride: u32,
    plot_u_offset: u32,
    plot_p_offset: u32,
    plot_has_u: bool,
    plot_has_p: bool,

    supports_preconditioner: bool,
    model_owns_preconditioner: bool,
    unknowns_per_cell: u32,

    supports_outer_iters: bool,
    supports_dtau: bool,
    supports_low_mach: bool,
    supports_alpha_u: bool,
    supports_alpha_p: bool,
    supports_eos_tuning: bool,
}

struct SolverInitRequest {
    generation: u64,
    model_id: &'static str,
    selected_geometry: GeometryType,
    mesh_type: MeshType,
    min_cell_size: f64,
    max_cell_size: f64,
    growth_rate: f64,
    // Per-knob solver settings now travel as a single `RuntimeParams` snapshot
    // (built via `current_runtime_params`) that the shared `SolverDriver` consumes.
    current_fluid: Fluid,
    params: RuntimeParams,
    wgpu_device: Option<wgpu::Device>,
    wgpu_queue: Option<wgpu::Queue>,
    target_format: wgpu::TextureFormat,
}

struct SolverInitOutcome {
    driver: SolverDriver,
    mesh: Mesh,
    cached_cells: Vec<Vec<[f64; 2]>>,
    actual_min_cell_size: f64,
    cached_u: Vec<(f64, f64)>,
    cached_p: Vec<f64>,
    model_caps: ModelUiCaps,
    renderer: Option<cfd_renderer::CfdRenderResources>,
    viz_field: Option<VizFieldBuffers>,
    trace_init_events: Vec<tracefmt::TraceInitEvent>,
}

#[derive(Clone)]
struct VizFieldBuffers {
    buffers: [wgpu::Buffer; 3],
    size_bytes: u64,
    front_idx: Arc<AtomicUsize>,
    ready_idx: Arc<AtomicUsize>,
}

struct SolverInitResponse {
    generation: u64,
    result: Result<SolverInitOutcome, String>,
}

// Cached GPU solver stats for UI display (avoids lock contention)
#[derive(Default, Clone)]
struct CachedGpuStats {
    dt: f32,
    linear_solves: u32,
    linear_last: LinearSolverStats,
    outer_residual_u: f32,
    outer_residual_p: f32,
    outer_iterations: u32,
    outer_step_status: Option<OuterStepStatus>,
    positivity_min_rho: Option<f32>,
    positivity_min_p: Option<f32>,
    positivity_rho_undershoots: u32,
    positivity_pressure_undershoots: u32,
    step_time_ms: f32,
}

enum SolverWorkerCommand {
    SetSolver {
        driver: SolverDriver,
        viz_field: Option<VizFieldBuffers>,
    },
    ClearSolver,
    SetRunning(bool),
    UpdateParams(RuntimeParams),
    StartTrace {
        path: String,
        header: tracefmt::TraceHeader,
    },
    AppendTraceInit {
        events: Vec<tracefmt::TraceInitEvent>,
    },
    StopTrace,
    Shutdown,
}

enum SolverWorkerEvent {
    Stats {
        stats: CachedGpuStats,
    },
    Snapshot {
        u: Vec<(f64, f64)>,
        p: Vec<f64>,
        stats: CachedGpuStats,
    },
    Message(String),
    Error(String),
    Running(bool),
}

struct SolverTraceSession {
    writer: tracefmt::TraceWriter,
    profiling_enabled: bool,
}

struct SolverWorkerHandle {
    tx: mpsc::Sender<SolverWorkerCommand>,
    rx: mpsc::Receiver<SolverWorkerEvent>,
    thread: Option<thread::JoinHandle<()>>,
}

impl SolverWorkerHandle {
    fn spawn() -> Self {
        let (tx, cmd_rx) = mpsc::channel::<SolverWorkerCommand>();
        let (evt_tx, rx) = mpsc::channel::<SolverWorkerEvent>();
        let thread = thread::spawn(move || solver_worker_main(cmd_rx, evt_tx));
        Self {
            tx,
            rx,
            thread: Some(thread),
        }
    }

    fn send(&self, cmd: SolverWorkerCommand) {
        let _ = self.tx.send(cmd);
    }
}

impl Drop for SolverWorkerHandle {
    fn drop(&mut self) {
        let _ = self.tx.send(SolverWorkerCommand::Shutdown);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

/// Compute-backend dropdown choice (env-driven; applied on Initialize / Reset).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendChoice {
    Gpu,
    CpuInterpreter,
    CpuTranspiled,
    /// Transpiled kernels + the SIMD path for the linear-solve reductions.
    CpuTranspiledSimd,
}

impl BackendChoice {
    const ALL: [BackendChoice; 4] = [
        BackendChoice::Gpu,
        BackendChoice::CpuInterpreter,
        BackendChoice::CpuTranspiled,
        BackendChoice::CpuTranspiledSimd,
    ];

    fn label(self) -> &'static str {
        match self {
            BackendChoice::Gpu => "GPU",
            BackendChoice::CpuInterpreter => "CPU Interpreter",
            BackendChoice::CpuTranspiled => "CPU Transpiled",
            BackendChoice::CpuTranspiledSimd => "CPU Transpiled (SIMD linear)",
        }
    }

    fn is_cpu(self) -> bool {
        !matches!(self, BackendChoice::Gpu)
    }
}

pub struct CFDApp {
    solver_worker: SolverWorkerHandle,
    pending_init_request: Option<SolverInitRequest>,
    init_rx: Option<mpsc::Receiver<SolverInitResponse>>,
    init_in_flight: Option<u64>,
    next_init_generation: u64,
    cached_u: Vec<(f64, f64)>,
    cached_p: Vec<f64>,
    snapshot_seq: u64,
    plot_cache: Option<PlotCache>,
    cached_gpu_stats: CachedGpuStats,
    cached_error: Option<String>,
    cached_message: Option<String>,
    last_init_trace_events: Vec<tracefmt::TraceInitEvent>,
    mesh: Option<Mesh>,
    cached_cells: Vec<Vec<[f64; 2]>>,
    actual_min_cell_size: f64,
    min_cell_size: f64,
    max_cell_size: f64,
    growth_rate: f64,
    timestep: f64,
    selected_geometry: GeometryType,
    mesh_type: MeshType,
    plot_field: PlotField,
    is_running: bool,
    selected_scheme: Scheme,
    current_fluid: Fluid,
    show_mesh_lines: bool,
    // Compute-backend selection (env-driven; applied on Initialize / Reset).
    backend: BackendChoice,
    cpu_threads: usize,
    /// Coupled linear-solve precision for the CPU backends (f64 = reference,
    /// f32 = GPU-like arithmetic, lower bandwidth).
    cpu_precision_f32: bool,
    adaptive_dt: bool,
    target_cfl: f64,
    dual_time: bool,
    dtau: f64,
    log_convergence: bool,
    log_every_steps: u32,
    trace_enabled: bool,
    trace_path: String,
    trace_pending_start: bool,
    render_mode: RenderMode,
    alpha_u: f64,
    alpha_p: f64,
    time_scheme: GpuTimeScheme,
    outer_iters: u32,
    /// Enable the per-step outer-convergence monitor (GUI residual readout +
    /// opportunistic early exit). Decoupled from the console logging flag.
    outer_auto_converge: bool,
    low_mach_model: GpuLowMachPrecondModel,
    low_mach_theta_floor: f32,
    low_mach_pressure_coupling_alpha: f32,
    inlet_velocity: f32,
    /// All-Mach compressibility `psi = 1/c^2` (the wire value). DERIVED, not stored:
    /// the REAL `current_fluid.compressibility()`, recomputed whenever the fluid or
    /// model changes. No exaggeration — always physical.
    compressibility_psi: f32,
    /// Gauge back-pressure pinned at the outlet (all-Mach models). `0.0` is the
    /// standard outlet; the supersonic-nozzle demo sets a negative value to pull the
    /// diverging section past Mach 1. Applied per-case via the GUI defaults.
    outlet_back_pressure: f32,
    /// Drive the CD nozzle with a pressure inlet + supersonic outlet (see
    /// `RuntimeParams::pressure_inlet`). Set per-case via the GUI defaults.
    pressure_inlet: bool,
    /// Inlet gauge pressure pinned when `pressure_inlet` is set.
    inlet_pressure: f32,
    selected_preconditioner: PreconditionerType,
    model_id: &'static str,
    model_caps: ModelUiCaps,
    wgpu_device: Option<wgpu::Device>,
    wgpu_queue: Option<wgpu::Queue>,
    target_format: wgpu::TextureFormat,
    cfd_renderer: Option<Arc<Mutex<cfd_renderer::CfdRenderResources>>>,
    viz_field: Option<VizFieldBuffers>,
    viz_field_front: usize,
}

struct CfdRenderCallback {
    renderer: Arc<Mutex<cfd_renderer::CfdRenderResources>>,
    uniforms: cfd_renderer::CfdUniforms,
    draw_lines: bool,
}

impl eframe::egui_wgpu::CallbackTrait for CfdRenderCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &eframe::egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut eframe::egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let renderer = self.renderer.lock().unwrap();
        renderer.update_uniforms(queue, &self.uniforms);
        resources.insert(renderer.clone());
        Vec::new()
    }

    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &eframe::egui_wgpu::CallbackResources,
    ) {
        if let Some(renderer) = resources.get::<cfd_renderer::CfdRenderResources>() {
            let clip_rect = info.clip_rect;
            render_pass.set_viewport(
                clip_rect.min.x as f32,
                clip_rect.min.y as f32,
                clip_rect.width() as f32,
                clip_rect.height() as f32,
                0.0,
                1.0,
            );

            // SAFETY: The renderer lives in CallbackResources which lives as long as the egui renderer.
            // The render pass is executed synchronously and the resources are valid for the duration.
            // We need to extend the lifetime to 'static to match the RenderPass<'static> signature required by egui_wgpu.
            let renderer: &'static cfd_renderer::CfdRenderResources =
                unsafe { std::mem::transmute(renderer) };

            let pixels_per_point = info.pixels_per_point;
            render_pass.set_viewport(
                clip_rect.min.x * pixels_per_point,
                clip_rect.min.y * pixels_per_point,
                clip_rect.width() * pixels_per_point,
                clip_rect.height() * pixels_per_point,
                0.0,
                1.0,
            );

            renderer.paint(render_pass, self.draw_lines);
        }
    }
}

impl CFDApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let (wgpu_device, wgpu_queue, target_format) =
            if let Some(render_state) = &cc.wgpu_render_state {
                (
                    Some(render_state.device.clone()),
                    Some(render_state.queue.clone()),
                    render_state.target_format,
                )
            } else {
                (None, None, wgpu::TextureFormat::Bgra8Unorm)
            };

        let default_fluid = Fluid::presets()[1].clone();

        let mut app = Self {
            solver_worker: SolverWorkerHandle::spawn(),
            pending_init_request: None,
            init_rx: None,
            init_in_flight: None,
            next_init_generation: 1,
            cached_u: Vec::new(),
            cached_p: Vec::new(),
            snapshot_seq: 0,
            plot_cache: None,
            cached_gpu_stats: CachedGpuStats::default(),
            cached_error: None,
            cached_message: None,
            last_init_trace_events: Vec::new(),
            mesh: None,
            cached_cells: Vec::new(),
            actual_min_cell_size: 0.01,
            min_cell_size: 0.025,
            max_cell_size: 0.025,
            growth_rate: 1.2,
            timestep: 0.02,
            selected_geometry: GeometryType::default(),
            mesh_type: MeshType::default(),
            plot_field: PlotField::VelocityMag,
            is_running: false,
            selected_scheme: Scheme::Upwind,
            current_fluid: default_fluid,
            show_mesh_lines: true,
            backend: BackendChoice::Gpu,
            cpu_threads: 1,
            cpu_precision_f32: false,
            adaptive_dt: true,
            target_cfl: 0.9,
            dual_time: false,
            dtau: 1e-5,
            log_convergence: false,
            log_every_steps: 25,
            trace_enabled: false,
            trace_path: "cfd2_trace.jsonl".to_string(),
            trace_pending_start: false,
            render_mode: RenderMode::GpuDirect,
            alpha_u: 0.7,
            alpha_p: 0.3,
            time_scheme: GpuTimeScheme::BDF2,
            outer_iters: 8,
            outer_auto_converge: true,
            low_mach_model: GpuLowMachPrecondModel::Off,
            low_mach_theta_floor: 1e-6,
            low_mach_pressure_coupling_alpha: 1.0,
            inlet_velocity: 1.0,
            compressibility_psi: 0.0,
            outlet_back_pressure: 0.0,
            pressure_inlet: false,
            inlet_pressure: 0.0,
            selected_preconditioner: PreconditionerType::Jacobi,
            model_id: "incompressible_momentum",
            model_caps: ModelUiCaps::default(),
            wgpu_device,
            wgpu_queue,
            target_format,
            cfd_renderer: None,
            viz_field: None,
            viz_field_front: 0,
        };
        app.apply_model_defaults();
        app.refresh_model_caps();
        app.sync_worker_params();
        app
    }

    fn current_runtime_params(&self) -> RuntimeParams {
        RuntimeParams {
            adaptive_dt: self.adaptive_dt,
            target_cfl: self.target_cfl,
            requested_dt: self.timestep as f32,
            dtau: if self.dual_time {
                self.dtau.max(0.0) as f32
            } else {
                0.0
            },
            log_convergence: self.log_convergence,
            log_every_steps: self.log_every_steps.max(1),
            advection_scheme: self.selected_scheme,
            time_scheme: self.time_scheme,
            preconditioner: self.selected_preconditioner,
            outer_iters: self.outer_iters.max(1),
            outer_auto_converge: self.outer_auto_converge,
            low_mach_model: self.low_mach_model,
            low_mach_theta_floor: self.low_mach_theta_floor,
            low_mach_pressure_coupling_alpha: self.low_mach_pressure_coupling_alpha,
            alpha_u: self.alpha_u as f32,
            alpha_p: self.alpha_p as f32,
            inlet_velocity: self.inlet_velocity,
            density: self.current_fluid.density as f32,
            viscosity: self.current_fluid.viscosity as f32,
            eos: self.current_fluid.eos,
            compressibility_psi: self.compressibility_psi,
            outlet_back_pressure: self.outlet_back_pressure,
            pressure_inlet: self.pressure_inlet,
            inlet_pressure: self.inlet_pressure,
        }
    }

    fn sync_worker_params(&self) {
        self.solver_worker.send(SolverWorkerCommand::UpdateParams(
            self.current_runtime_params(),
        ));
    }

    /// Apply the per-model GUI default solver parameters (single source of truth
    /// in `model_defaults`). Called at startup and whenever the active model
    /// changes, so the incompressible and compressible solvers each get sane,
    /// non-diverging knobs instead of one shared set tuned for neither.
    fn apply_model_defaults(&mut self) {
        // The supersonic-nozzle demo is the composition (all-Mach model + nozzle
        // geometry): override the per-model defaults with the tuned nozzle case
        // (choking inlet speed + sub-critical outlet back-pressure) so selecting the
        // nozzle "just works" as a supersonic case. Every other case keeps a 0
        // back-pressure (standard outlet).
        let is_allmach = self.model_id == "allmach_pressure" || self.model_id == "allmach_thermal";
        let d = if self.selected_geometry == GeometryType::Nozzle && is_allmach {
            crate::ui::model_defaults::ALLMACH_THERMAL_NOZZLE
        } else {
            crate::ui::model_defaults::gui_defaults_for(self.model_id)
        };
        self.selected_scheme = d.advection_scheme;
        self.time_scheme = d.time_scheme;
        self.selected_preconditioner = d.preconditioner;
        self.alpha_u = d.alpha_u;
        self.alpha_p = d.alpha_p;
        self.outer_iters = d.outer_iters;
        self.outer_auto_converge = d.outer_auto_converge;
        self.target_cfl = d.target_cfl;
        self.timestep = d.timestep;
        self.adaptive_dt = d.adaptive_dt;
        // Pseudo-transient continuation: the low-Mach stabilizer for the
        // compressible default (see model_defaults). Off for incompressible.
        self.dual_time = d.dual_time;
        self.dtau = d.dtau;
        self.low_mach_model = d.low_mach_model;
        self.low_mach_theta_floor = d.low_mach_theta_floor;
        self.low_mach_pressure_coupling_alpha = d.low_mach_pressure_coupling_alpha;
        self.inlet_velocity = d.inlet_velocity;
        // All-Mach compressibility: the REAL `psi = 1/c^2` of the current fluid, no
        // exaggeration. Recomputed here (model switch) and on fluid change.
        self.compressibility_psi = self.current_fluid.compressibility() as f32;
        // Outlet back-pressure: negative only for the supersonic-nozzle case (above).
        self.outlet_back_pressure = d.outlet_back_pressure;
        self.pressure_inlet = d.pressure_inlet;
        self.inlet_pressure = d.inlet_pressure;
    }

    fn current_trace_runtime_params(&self) -> tracefmt::TraceRuntimeParams {
        tracefmt::TraceRuntimeParams {
            adaptive_dt: self.adaptive_dt,
            target_cfl: self.target_cfl,
            requested_dt: self.timestep as f32,
            dtau: if self.dual_time {
                self.dtau.max(0.0) as f32
            } else {
                0.0
            },
            advection_scheme: tracefmt::TraceScheme::from(self.selected_scheme),
            time_scheme: tracefmt::TraceTimeScheme::from(self.time_scheme),
            preconditioner: tracefmt::TracePreconditioner::from(self.selected_preconditioner),
            outer_iters: self.outer_iters.max(1),
            low_mach_model: tracefmt::TraceLowMachModel::from(self.low_mach_model),
            low_mach_theta_floor: self.low_mach_theta_floor,
            low_mach_pressure_coupling_alpha: self.low_mach_pressure_coupling_alpha,
            alpha_u: self.alpha_u as f32,
            alpha_p: self.alpha_p as f32,
            inlet_velocity: self.inlet_velocity,
            density: self.current_fluid.density as f32,
            viscosity: self.current_fluid.viscosity as f32,
            eos: tracefmt::TraceEosSpec::from(self.current_fluid.eos),
        }
    }

    fn build_trace_header(&self, mesh: &Mesh) -> Result<tracefmt::TraceHeader, String> {
        let mesh = tracefmt::TraceMesh::from_mesh(mesh)?;

        let geometry = match self.selected_geometry {
            GeometryType::BackwardsStep => tracefmt::TraceGeometry::BackwardsStep,
            GeometryType::ChannelObstacle => tracefmt::TraceGeometry::ChannelObstacle,
            GeometryType::Nozzle => tracefmt::TraceGeometry::Nozzle,
        };
        let mesh_type = match self.mesh_type {
            MeshType::CutCell => tracefmt::TraceMeshType::CutCell,
            MeshType::Delaunay => tracefmt::TraceMeshType::Delaunay,
            MeshType::Voronoi => tracefmt::TraceMeshType::Voronoi,
            MeshType::VoronoiCvt => tracefmt::TraceMeshType::VoronoiCvt,
            MeshType::Fitted => tracefmt::TraceMeshType::Fitted,
        };

        let stepping_mode = if self.model_caps.supports_eos_tuning {
            tracefmt::TraceSteppingMode::Implicit
        } else {
            tracefmt::TraceSteppingMode::Coupled
        };

        let fluid = tracefmt::TraceFluid {
            name: Some(self.current_fluid.name.clone()),
            density: self.current_fluid.density,
            viscosity: self.current_fluid.viscosity,
            eos: tracefmt::TraceEosSpec::from(self.current_fluid.eos),
        };

        let case = tracefmt::TraceCase {
            model_id: self.model_id.to_string(),
            geometry,
            mesh_type,
            min_cell_size: self.min_cell_size,
            max_cell_size: self.max_cell_size,
            growth_rate: self.growth_rate,
            fluid,
            stepping_mode,
            mesh,
        };

        Ok(tracefmt::make_header(
            case,
            self.current_trace_runtime_params(),
            cfg!(feature = "profiling"),
        ))
    }

    fn start_trace(&mut self) {
        let Some(mesh) = &self.mesh else {
            self.trace_pending_start = true;
            return;
        };

        let header = match self.build_trace_header(mesh) {
            Ok(v) => v,
            Err(err) => {
                self.trace_enabled = false;
                self.trace_pending_start = false;
                self.cached_error = Some(format!("failed to start trace: {err}"));
                return;
            }
        };

        self.trace_pending_start = false;
        self.solver_worker.send(SolverWorkerCommand::StartTrace {
            path: self.trace_path.clone(),
            header,
        });
        if !self.last_init_trace_events.is_empty() {
            self.solver_worker
                .send(SolverWorkerCommand::AppendTraceInit {
                    events: self.last_init_trace_events.clone(),
                });
        }
    }

    fn stop_trace(&mut self) {
        self.trace_pending_start = false;
        self.solver_worker.send(SolverWorkerCommand::StopTrace);
    }

    fn model_label(model_id: &'static str) -> &'static str {
        match model_id {
            "incompressible_momentum" => "Incompressible momentum",
            "compressible" => "Compressible",
            "allmach_pressure" => "All-Mach (pressure-based)",
            "allmach_thermal" => "All-Mach (pressure-based, thermal)",
            other => other,
        }
    }

    /// The physical models the GUI Model dropdown offers. `all_models()` also
    /// contains MMS / biharmonic / demo *verification* variants which carry the
    /// velocity+pressure ports (so `UiPortSet::is_complete()` alone would expose
    /// them) but only ever reproduce a manufactured solution — never a physical
    /// flow a user would want to run. The dropdown is therefore restricted to the
    /// genuine flow models; the completeness check is kept as a secondary guard.
    fn supported_ui_models() -> Vec<(&'static str, &'static str)> {
        // Only physical-flow models belong in the GUI (these are exactly the ones
        // `model_label` names). Verification variants (`*_mms*`, `*biharmonic*`,
        // `*demo*`) are excluded.
        const GUI_PHYSICAL_MODELS: &[&str] = &[
            "incompressible_momentum",
            "compressible",
            "allmach_pressure",
            "allmach_thermal",
        ];
        let mut out = Vec::new();
        for model in all_models().expect("failed to build model definitions") {
            if !GUI_PHYSICAL_MODELS.contains(&model.id) {
                continue;
            }
            // Use UiPortSet to check for required fields (validates types too)
            let ui_ports = UiPortSet::from_layout(&model.state_layout);
            if !ui_ports.is_complete() {
                continue;
            }
            out.push((model.id, CFDApp::model_label(model.id)));
        }
        out.sort_by_key(|(id, _)| *id);
        out
    }

    fn build_selected_model(&self) -> Result<ModelSpec, String> {
        if self.model_id == "compressible" {
            return compressible_model_with_eos(self.current_fluid.eos);
        }
        all_models()?
            .into_iter()
            .find(|m| m.id == self.model_id)
            .ok_or_else(|| format!("unknown model id '{}'", self.model_id))
    }

    fn refresh_model_caps(&mut self) {
        let model = match self.build_selected_model() {
            Ok(m) => m,
            Err(_) => {
                self.model_id = "incompressible_momentum";
                self.build_selected_model()
                    .expect("default UI model 'incompressible_momentum' must exist")
            }
        };

        let named_params = model.named_param_keys();
        self.model_caps.supports_preconditioner =
            named_params.iter().any(|&k| k == "preconditioner");
        self.model_caps.model_owns_preconditioner = model
            .linear_solver
            .map(|s| matches!(s.preconditioner, ModelPreconditionerSpec::Schur { .. }))
            .unwrap_or(false);
        self.model_caps.unknowns_per_cell = model.system.unknowns_per_cell();

        self.model_caps.supports_alpha_u = named_params.iter().any(|&k| k == "alpha_u");
        self.model_caps.supports_alpha_p = named_params.iter().any(|&k| k == "alpha_p");
        self.model_caps.supports_eos_tuning = named_params.iter().any(|&k| k == "eos.gamma");
        self.model_caps.supports_outer_iters = named_params.iter().any(|&k| k == "outer_iters");
        self.model_caps.supports_dtau = named_params.iter().any(|&k| k == "dtau");
        self.model_caps.supports_low_mach = named_params.iter().any(|&k| k == "low_mach.model");

        // Use UiPortSet for field offset resolution (validates field types)
        let ui_ports = UiPortSet::from_layout(&model.state_layout);
        self.model_caps.plot_stride = ui_ports.stride;
        self.model_caps.plot_has_u = ui_ports.u_offset.is_some();
        self.model_caps.plot_u_offset = ui_ports.u_offset.unwrap_or(0);
        self.model_caps.plot_has_p = ui_ports.p_offset.is_some();
        self.model_caps.plot_p_offset = ui_ports.p_offset.unwrap_or(0);

        const UI_MAX_BLOCK_JACOBI: u32 = 16;
        if matches!(
            self.selected_preconditioner,
            PreconditionerType::BlockJacobi
        ) && self.model_caps.unknowns_per_cell > UI_MAX_BLOCK_JACOBI
        {
            self.selected_preconditioner = PreconditionerType::Jacobi;
        }
    }

    fn make_init_request(&mut self) -> SolverInitRequest {
        let generation = self.next_init_generation;
        self.next_init_generation = self.next_init_generation.wrapping_add(1);
        SolverInitRequest {
            generation,
            model_id: self.model_id,
            selected_geometry: self.selected_geometry,
            mesh_type: self.mesh_type,
            min_cell_size: self.min_cell_size,
            max_cell_size: self.max_cell_size,
            growth_rate: self.growth_rate,
            current_fluid: self.current_fluid.clone(),
            params: self.current_runtime_params(),
            wgpu_device: self.wgpu_device.clone(),
            wgpu_queue: self.wgpu_queue.clone(),
            target_format: self.target_format,
        }
    }

    fn update_renderer_field(&self) {
        // No-op: the renderer bind group is created at solver init time and points at the
        // visualization buffer. Field selection is driven by uniforms (stride/offset/mode).
    }

    /// Apply the selected compute backend via environment (read by
    /// `UnifiedSolver::new`). CPU options are runtime-switchable; changes take
    /// effect on the next solver (re)build.
    fn apply_backend_env(&self) {
        if self.backend.is_cpu() {
            std::env::set_var("CFD2_BACKEND", "cpu");
            std::env::set_var(
                "CFD2_CPU_ENGINE",
                if self.backend == BackendChoice::CpuInterpreter {
                    "interpreter"
                } else {
                    "transpiled"
                },
            );
            std::env::set_var("CFD2_CPU_THREADS", self.cpu_threads.max(1).to_string());
            std::env::set_var(
                "CFD2_CPU_SIMD",
                if self.backend == BackendChoice::CpuTranspiledSimd { "1" } else { "0" },
            );
            std::env::set_var(
                "CFD2_CPU_PRECISION",
                if self.cpu_precision_f32 { "f32" } else { "f64" },
            );
        } else {
            std::env::remove_var("CFD2_BACKEND");
        }
    }

    fn init_solver(&mut self) {
        self.apply_backend_env();
        self.is_running = false;
        self.solver_worker
            .send(SolverWorkerCommand::SetRunning(false));
        self.solver_worker.send(SolverWorkerCommand::StopTrace);
        self.trace_pending_start = self.trace_enabled;
        self.solver_worker.send(SolverWorkerCommand::ClearSolver);
        self.refresh_model_caps();
        self.pending_init_request = Some(self.make_init_request());
        self.mesh = None;
        self.cached_cells.clear();
        self.cfd_renderer = None;
        self.cached_u.clear();
        self.cached_p.clear();
        self.snapshot_seq = 0;
        self.invalidate_plot_cache();
        self.cached_gpu_stats = CachedGpuStats::default();
        self.cached_error = None;
        self.cached_message = None;
        self.last_init_trace_events.clear();
    }

    fn build_mesh_with(
        selected_geometry: GeometryType,
        mesh_type: MeshType,
        min_cell_size: f64,
        max_cell_size: f64,
        growth_rate: f64,
        trace_init_events: &mut Vec<tracefmt::TraceInitEvent>,
    ) -> Mesh {
        // The sizing sliders accept any hand-typed value (`SliderClamping::Never`),
        // so sanitise before the values reach the mesh generators: sizes must be
        // finite and positive, `min <= max`, and the growth rate at least 1. This
        // keeps the UI free while preventing a NaN/∞/divide-by-zero, and — crucially
        // for the UNSTRUCTURED meshers — bounding the base grid. `generate_cut_cell_mesh`
        // et al. build an uncapped `(domain/max_cell_size)^2` base grid, so a tiny
        // hand-typed size would OOM; the `MIN_CELL_SIZE` floor matches the old
        // Always-clamped slider minimum (base grid ≲ few·1e6 cells on these domains).
        // The fitted structured grid is unaffected: its resolution is bounded by the
        // nx/ny clamp (≈5.9e-3 cell), coarser than this floor, so the floor never binds.
        const MIN_CELL_SIZE: f64 = 1e-3;
        let max_cell_size = if max_cell_size.is_finite() {
            max_cell_size.max(MIN_CELL_SIZE)
        } else {
            0.05
        };
        let min_cell_size = if min_cell_size.is_finite() {
            min_cell_size.clamp(MIN_CELL_SIZE, max_cell_size)
        } else {
            max_cell_size
        };
        let growth_rate = if growth_rate.is_finite() {
            growth_rate.max(1.0)
        } else {
            1.2
        };

        fn mesh_type_id(mesh_type: MeshType) -> &'static str {
            match mesh_type {
                MeshType::CutCell => "cutcell",
                MeshType::Delaunay => "delaunay",
                MeshType::Voronoi => "voronoi",
                MeshType::VoronoiCvt => "voronoi_cvt",
                MeshType::Fitted => "fitted",
            }
        }

        fn geometry_id(geometry: GeometryType) -> &'static str {
            match geometry {
                GeometryType::BackwardsStep => "backwards_step",
                GeometryType::ChannelObstacle => "channel_obstacle",
                GeometryType::Nozzle => "nozzle",
            }
        }

        let geometry = geometry_id(selected_geometry);
        let mesh_kind = mesh_type_id(mesh_type);
        let total_start = std::time::Instant::now();

        let mesh = match selected_geometry {
            GeometryType::BackwardsStep => {
                let length = 3.5;
                let height_outlet = 1.0;
                let height_inlet = 0.5;
                let step_x = 0.5;

                let domain_size = Vector2::new(length, height_outlet);
                let geo = BackwardsStep {
                    length,
                    height_inlet,
                    height_outlet,
                    step_x,
                };

                let gen_start = std::time::Instant::now();
                // `Fitted` is a nozzle-only option; it never reaches this geometry
                // from the UI, so fall back to the cut-cell mesher if it does.
                let mut mesh = match mesh_type {
                    MeshType::CutCell | MeshType::Fitted => generate_cut_cell_mesh(
                        &geo,
                        min_cell_size,
                        max_cell_size,
                        growth_rate,
                        domain_size,
                    ),
                    MeshType::Delaunay => generate_delaunay_mesh(
                        &geo,
                        min_cell_size,
                        max_cell_size,
                        growth_rate,
                        domain_size,
                    ),
                    MeshType::Voronoi => generate_voronoi_mesh(
                        &geo,
                        min_cell_size,
                        max_cell_size,
                        growth_rate,
                        domain_size,
                    ),
                    MeshType::VoronoiCvt => generate_cvt_mesh(
                        &geo,
                        min_cell_size,
                        max_cell_size,
                        growth_rate,
                        domain_size,
                        &LloydConfig::default(),
                    ),
                };

                CFDApp::push_trace_init_event(
                    trace_init_events,
                    format!("mesh.generate.{geometry}.{mesh_kind}"),
                    gen_start.elapsed(),
                    Some(format!(
                        "min={min_cell_size:.4e} max={max_cell_size:.4e} growth={growth_rate:.3} cells={} faces={} vertices={}",
                        mesh.num_cells(),
                        mesh.num_faces(),
                        mesh.num_vertices()
                    )),
                );

                // The CVT mesh optimizes seed positions instead: vertex
                // smoothing would move Voronoi vertices off the bisectors
                // and destroy the mesh's defining property.
                if mesh_type != MeshType::VoronoiCvt {
                    let smooth_start = std::time::Instant::now();
                    mesh.smooth(&geo, 0.3, 50);
                    CFDApp::push_trace_init_event(
                        trace_init_events,
                        format!("mesh.smooth.{geometry}.{mesh_kind}"),
                        smooth_start.elapsed(),
                        Some("factor=0.3 iters=50".to_string()),
                    );
                }

                mesh
            }
            GeometryType::ChannelObstacle => {
                let length = 3.0;
                let domain_size = Vector2::new(length, 1.0);
                let geo = ChannelWithObstacle {
                    length,
                    height: 1.0,
                    obstacle_center: Point2::new(1.0, 0.51), // Offset to trigger vortex shedding
                    obstacle_radius: 0.1,
                };

                let gen_start = std::time::Instant::now();
                // `Fitted` is a nozzle-only option; it never reaches this geometry
                // from the UI, so fall back to the cut-cell mesher if it does.
                let mut mesh = match mesh_type {
                    MeshType::CutCell | MeshType::Fitted => generate_cut_cell_mesh(
                        &geo,
                        min_cell_size,
                        max_cell_size,
                        growth_rate,
                        domain_size,
                    ),
                    MeshType::Delaunay => generate_delaunay_mesh(
                        &geo,
                        min_cell_size,
                        max_cell_size,
                        growth_rate,
                        domain_size,
                    ),
                    MeshType::Voronoi => generate_voronoi_mesh(
                        &geo,
                        min_cell_size,
                        max_cell_size,
                        growth_rate,
                        domain_size,
                    ),
                    MeshType::VoronoiCvt => generate_cvt_mesh(
                        &geo,
                        min_cell_size,
                        max_cell_size,
                        growth_rate,
                        domain_size,
                        &LloydConfig::default(),
                    ),
                };

                CFDApp::push_trace_init_event(
                    trace_init_events,
                    format!("mesh.generate.{geometry}.{mesh_kind}"),
                    gen_start.elapsed(),
                    Some(format!(
                        "min={min_cell_size:.4e} max={max_cell_size:.4e} growth={growth_rate:.3} cells={} faces={} vertices={}",
                        mesh.num_cells(),
                        mesh.num_faces(),
                        mesh.num_vertices()
                    )),
                );

                // The CVT mesh optimizes seed positions instead: vertex
                // smoothing would move Voronoi vertices off the bisectors
                // and destroy the mesh's defining property.
                if mesh_type != MeshType::VoronoiCvt {
                    let smooth_iters = match mesh_type {
                        MeshType::CutCell | MeshType::Fitted => 100,
                        MeshType::Delaunay | MeshType::Voronoi => 50,
                        MeshType::VoronoiCvt => unreachable!(),
                    };

                    let smooth_start = std::time::Instant::now();
                    mesh.smooth(&geo, 0.3, smooth_iters);
                    CFDApp::push_trace_init_event(
                        trace_init_events,
                        format!("mesh.smooth.{geometry}.{mesh_kind}"),
                        smooth_start.elapsed(),
                        Some(format!("factor=0.3 iters={smooth_iters}")),
                    );
                }

                mesh
            }
            GeometryType::Nozzle => {
                // Converging–diverging nozzle, area ratio exit/throat = 2, matching
                // the validated `allmach_thermal_supersonic_test`. `Fitted` builds
                // the body-fitted structured grid (the validated default); the
                // unstructured mesh types conform to the same wall profile via the
                // `Nozzle` SDF geometry and honour the full sizing controls.
                let length = 3.0;
                let height = 1.0;
                let throat_h = 0.40;
                let throat_frac = 0.40;
                let exit_h = 0.80;

                match mesh_type {
                    MeshType::Fitted => {
                        // Uniform curvilinear structured grid. Resolution is derived
                        // from the (target) cell-size slider. The lower clamp keeps the
                        // throat resolved enough to choke; the upper clamp lets the user
                        // drive the mesh much finer than before (down to ~0.006) while
                        // still bounding a hand-typed size so it can't blow up memory.
                        let nx = ((length / max_cell_size).round() as usize).clamp(64, 512);
                        let ny = ((height / max_cell_size).round() as usize).clamp(24, 192);

                        let gen_start = std::time::Instant::now();
                        let mesh = generate_structured_nozzle_mesh(
                            nx,
                            ny,
                            length,
                            height,
                            throat_h,
                            throat_frac,
                            exit_h,
                            BoundarySides {
                                left: BoundaryType::Inlet,
                                right: BoundaryType::Outlet,
                                bottom: BoundaryType::Wall,
                                top: BoundaryType::Wall,
                            },
                        );
                        CFDApp::push_trace_init_event(
                            trace_init_events,
                            format!("mesh.generate.{geometry}.{mesh_kind}"),
                            gen_start.elapsed(),
                            Some(format!(
                                "nx={nx} ny={ny} area_ratio={:.2} cells={} faces={} vertices={}",
                                exit_h / throat_h,
                                mesh.num_cells(),
                                mesh.num_faces(),
                                mesh.num_vertices()
                            )),
                        );

                        mesh
                    }
                    MeshType::CutCell
                    | MeshType::Delaunay
                    | MeshType::Voronoi
                    | MeshType::VoronoiCvt => {
                        // Unstructured mesh conforming to the nozzle SDF. The bounding
                        // box is the inlet-height rectangle; the mesher tags the left
                        // edge Inlet, the right edge Outlet, and the flat bottom Wall.
                        let domain_size = Vector2::new(length, height);
                        let geo = Nozzle {
                            length,
                            height,
                            throat_height: throat_h,
                            throat_frac,
                            exit_height: exit_h,
                        };

                        let gen_start = std::time::Instant::now();
                        let mut mesh = match mesh_type {
                            MeshType::Delaunay => generate_delaunay_mesh(
                                &geo,
                                min_cell_size,
                                max_cell_size,
                                growth_rate,
                                domain_size,
                            ),
                            MeshType::Voronoi => generate_voronoi_mesh(
                                &geo,
                                min_cell_size,
                                max_cell_size,
                                growth_rate,
                                domain_size,
                            ),
                            MeshType::VoronoiCvt => generate_cvt_mesh(
                                &geo,
                                min_cell_size,
                                max_cell_size,
                                growth_rate,
                                domain_size,
                                &LloydConfig::default(),
                            ),
                            // CutCell (and the unreachable Fitted, already handled).
                            _ => generate_cut_cell_mesh(
                                &geo,
                                min_cell_size,
                                max_cell_size,
                                growth_rate,
                                domain_size,
                            ),
                        };
                        CFDApp::push_trace_init_event(
                            trace_init_events,
                            format!("mesh.generate.{geometry}.{mesh_kind}"),
                            gen_start.elapsed(),
                            Some(format!(
                                "min={min_cell_size:.4e} max={max_cell_size:.4e} growth={growth_rate:.3} cells={} faces={} vertices={}",
                                mesh.num_cells(),
                                mesh.num_faces(),
                                mesh.num_vertices()
                            )),
                        );

                        // The CVT mesh optimizes seed positions instead:
                        // vertex smoothing would move Voronoi vertices off
                        // the bisectors and destroy its defining property.
                        if mesh_type != MeshType::VoronoiCvt {
                            let smooth_iters = match mesh_type {
                                MeshType::Delaunay | MeshType::Voronoi => 50,
                                _ => 100,
                            };
                            let smooth_start = std::time::Instant::now();
                            mesh.smooth(&geo, 0.3, smooth_iters);
                            CFDApp::push_trace_init_event(
                                trace_init_events,
                                format!("mesh.smooth.{geometry}.{mesh_kind}"),
                                smooth_start.elapsed(),
                                Some(format!("factor=0.3 iters={smooth_iters}")),
                            );
                        }

                        // The curved top wall (untagged by `classify_boundary`) is
                        // closed as a no-slip wall inside every unstructured mesh
                        // generator (`close_untagged_boundary_faces`).
                        mesh
                    }
                }
            }
        };

        CFDApp::push_trace_init_event(
            trace_init_events,
            format!("mesh.total.{geometry}.{mesh_kind}"),
            total_start.elapsed(),
            Some(format!(
                "cells={} faces={} vertices={}",
                mesh.num_cells(),
                mesh.num_faces(),
                mesh.num_vertices()
            )),
        );

        mesh
    }

    fn push_trace_init_event(
        events: &mut Vec<tracefmt::TraceInitEvent>,
        stage: impl Into<String>,
        elapsed: std::time::Duration,
        detail: Option<String>,
    ) {
        events.push(tracefmt::TraceInitEvent {
            stage: stage.into(),
            wall_time_ms: elapsed.as_secs_f32() * 1000.0,
            detail,
        });
    }

    fn cache_cells(mesh: &Mesh) -> Vec<Vec<[f64; 2]>> {
        let mut cells = Vec::with_capacity(mesh.num_cells());
        for i in 0..mesh.num_cells() {
            let start = mesh.cell_vertex_offsets[i];
            let end = mesh.cell_vertex_offsets[i + 1];
            let polygon_points: Vec<[f64; 2]> = (start..end)
                .map(|k| {
                    let v_idx = mesh.cell_vertices[k];
                    [mesh.vx[v_idx], mesh.vy[v_idx]]
                })
                .collect();
            cells.push(polygon_points);
        }
        cells
    }

    fn build_initial_velocity_with(
        mesh: &Mesh,
        selected_geometry: GeometryType,
        max_cell_size: f64,
        inlet_velocity: f32,
    ) -> Vec<(f64, f64)> {
        // Nozzle: develop FROM REST (zero velocity). No seeded freestream — the flow
        // accelerates purely from the rocket-scale inlet pressure drop. The large
        // pressure ratio (1 MPa gauge, see `ALLMACH_THERMAL_NOZZLE`) forces the throat
        // to choke, so the supersonic branch forms from scratch; the old low-pressure
        // case needed a freestream IC to avoid settling on the subsonic diffuser branch.
        if selected_geometry == GeometryType::Nozzle {
            let _ = inlet_velocity; // no longer used to seed the nozzle IC
            return vec![(0.0, 0.0); mesh.num_cells()];
        }
        let mut u = vec![(0.0, 0.0); mesh.num_cells()];
        for (i, _vel) in u.iter_mut().enumerate() {
            let cx = mesh.cell_cx[i];
            let cy = mesh.cell_cy[i];

            if cx < max_cell_size {
                match selected_geometry {
                    GeometryType::BackwardsStep => {
                        if cy > 0.5 {
                            // Inlet ramp handled by shader
                        }
                    }
                    GeometryType::ChannelObstacle => {
                        // Inlet handled by shader
                    }
                    GeometryType::Nozzle => {
                        // Handled by the uniform-freestream early return above.
                    }
                }
            }
        }
        u
    }

    fn spawn_pending_init(&mut self) {
        if self.pending_init_request.is_none() {
            return;
        }

        // Cancel any in-flight init by dropping the receiver; the background thread will finish
        // but its result will be ignored.
        self.init_rx = None;
        self.init_in_flight = None;

        let request = match self.pending_init_request.take() {
            Some(r) => r,
            None => return, // should be unreachable due to guard above
        };
        self.init_in_flight = Some(request.generation);
        let (tx, rx) = mpsc::channel();
        self.init_rx = Some(rx);

        thread::spawn(move || {
            let generation = request.generation;
            let run = std::panic::AssertUnwindSafe(|| CFDApp::build_init_outcome(request));
            let result = std::panic::catch_unwind(run)
                .map_err(|_| "Solver init panicked".to_string())
                .and_then(|r| r);
            let _ = tx.send(SolverInitResponse { generation, result });
        });
    }

    fn poll_init(&mut self) {
        let Some(rx) = &self.init_rx else {
            return;
        };
        match rx.try_recv() {
            Ok(resp) => {
                self.init_rx = None;
                let in_flight = self.init_in_flight;
                self.init_in_flight = None;
                if in_flight.is_some() && in_flight != Some(resp.generation) {
                    return;
                }
                match resp.result {
                    Ok(outcome) => {
                        self.apply_init_outcome(outcome);
                    }
                    Err(err) => {
                        self.mesh = None;
                        self.cached_cells.clear();
                        self.cfd_renderer = None;
                        self.viz_field = None;
                        self.viz_field_front = 0;
                        self.cached_u.clear();
                        self.cached_p.clear();
                        self.snapshot_seq = 0;
                        self.invalidate_plot_cache();
                        self.cached_gpu_stats = CachedGpuStats::default();
                        self.cached_error = Some(err);
                    }
                }
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                self.init_rx = None;
                self.init_in_flight = None;
                self.cached_error = Some("Solver init thread disconnected".to_string());
            }
        }
    }

    fn poll_solver_worker(&mut self) {
        while let Ok(evt) = self.solver_worker.rx.try_recv() {
            match evt {
                SolverWorkerEvent::Stats { stats } => {
                    self.cached_gpu_stats = stats;
                    self.cached_error = None;
                }
                SolverWorkerEvent::Snapshot { u, p, stats } => {
                    self.cached_u = u;
                    self.cached_p = p;
                    self.snapshot_seq = self.snapshot_seq.wrapping_add(1);
                    self.invalidate_plot_cache();
                    self.cached_gpu_stats = stats;
                    self.cached_error = None;
                }
                SolverWorkerEvent::Message(message) => {
                    self.cached_message = Some(message);
                }
                SolverWorkerEvent::Error(err) => {
                    self.cached_error = Some(err);
                    self.cached_message = None;
                    self.is_running = false;
                }
                SolverWorkerEvent::Running(running) => {
                    self.is_running = running;
                }
            }
        }

        if let (Some(viz), Some(renderer), Some(device)) = (
            self.viz_field.as_ref(),
            self.cfd_renderer.as_ref(),
            self.wgpu_device.as_ref(),
        ) {
            let ready = viz.ready_idx.load(Ordering::Acquire) % viz.buffers.len();
            if ready != self.viz_field_front {
                let mut renderer = renderer.lock().unwrap();
                renderer.update_bind_group(device, &viz.buffers[ready]);
                self.viz_field_front = ready;
                viz.front_idx.store(ready, Ordering::Release);
            }
        }
    }

    fn apply_init_outcome(&mut self, outcome: SolverInitOutcome) {
        let SolverInitOutcome {
            driver,
            mesh,
            cached_cells,
            actual_min_cell_size,
            cached_u,
            cached_p,
            model_caps,
            renderer,
            viz_field,
            trace_init_events,
        } = outcome;

        self.is_running = false;
        self.solver_worker
            .send(SolverWorkerCommand::SetRunning(false));

        self.model_caps = model_caps;
        self.cached_cells = cached_cells;
        self.actual_min_cell_size = actual_min_cell_size;
        self.cached_u = cached_u;
        self.cached_p = cached_p;
        self.snapshot_seq = self.snapshot_seq.wrapping_add(1);
        self.invalidate_plot_cache();
        self.mesh = Some(mesh);

        self.cfd_renderer = renderer.map(|r| Arc::new(Mutex::new(r)));
        self.viz_field = viz_field.clone();
        self.viz_field_front = 0;
        if let Some(viz) = self.viz_field.as_ref() {
            viz.front_idx.store(0, Ordering::Release);
            viz.ready_idx.store(0, Ordering::Release);
        }
        self.last_init_trace_events = trace_init_events;

        self.cached_gpu_stats = CachedGpuStats::default();
        self.cached_error = None;
        self.cached_message = None;

        self.solver_worker.send(SolverWorkerCommand::SetSolver {
            driver,
            viz_field,
        });
        self.sync_worker_params();

        if self.trace_enabled {
            self.start_trace();
        } else {
            self.trace_pending_start = false;
        }
    }

    fn build_init_outcome(request: SolverInitRequest) -> Result<SolverInitOutcome, String> {
        let init_start = std::time::Instant::now();
        let mut trace_init_events: Vec<tracefmt::TraceInitEvent> = Vec::new();

        let mesh = CFDApp::build_mesh_with(
            request.selected_geometry,
            request.mesh_type,
            request.min_cell_size,
            request.max_cell_size,
            request.growth_rate,
            &mut trace_init_events,
        );

        let fields_start = std::time::Instant::now();
        let n_cells = mesh.num_cells();
        let initial_u = CFDApp::build_initial_velocity_with(
            &mesh,
            request.selected_geometry,
            request.max_cell_size,
            request.params.inlet_velocity,
        );
        let initial_p = vec![0.0; n_cells];
        CFDApp::push_trace_init_event(
            &mut trace_init_events,
            "init.fields",
            fields_start.elapsed(),
            Some(format!("cells={n_cells}")),
        );

        let model = if request.model_id == "compressible" {
            compressible_model_with_eos(request.current_fluid.eos)?
        } else {
            all_models()?
                .into_iter()
                .find(|m| m.id == request.model_id)
                .ok_or_else(|| format!("unknown model id '{}'", request.model_id))?
        };

        let named_params = model.named_param_keys();
        let supports_preconditioner = named_params.iter().any(|&k| k == "preconditioner");

        // Build initial model_caps from layout (will be updated after solver creation
        // to use PortRegistry-based ui_ports when available)
        let ui_ports_fallback = UiPortSet::from_layout(&model.state_layout);

        let mut model_caps = ModelUiCaps {
            plot_stride: ui_ports_fallback.stride,
            plot_u_offset: ui_ports_fallback.u_offset.unwrap_or(0),
            plot_p_offset: ui_ports_fallback.p_offset.unwrap_or(0),
            plot_has_u: ui_ports_fallback.u_offset.is_some(),
            plot_has_p: ui_ports_fallback.p_offset.is_some(),
            supports_preconditioner,
            model_owns_preconditioner: model
                .linear_solver
                .map(|s| matches!(s.preconditioner, ModelPreconditionerSpec::Schur { .. }))
                .unwrap_or(false),
            unknowns_per_cell: model.system.unknowns_per_cell(),
            supports_outer_iters: named_params.iter().any(|&k| k == "outer_iters"),
            supports_dtau: named_params.iter().any(|&k| k == "dtau"),
            supports_low_mach: named_params.iter().any(|&k| k == "low_mach.model"),
            supports_alpha_u: named_params.iter().any(|&k| k == "alpha_u"),
            supports_alpha_p: named_params.iter().any(|&k| k == "alpha_p"),
            supports_eos_tuning: named_params.iter().any(|&k| k == "eos.gamma"),
        };

        // The shared driver derives the `SolverConfig` (stepping mode + effective
        // preconditioner), constructs the solver, and applies the phase-1 setters +
        // initial / boundary conditions. Phase-2 knobs arrive via `sync_worker_params`
        // (→ `apply_params`) after `SetSolver`, exactly as before.
        let solver_start = std::time::Instant::now();
        let init_guard = tracefmt::install_init_collector(&mut trace_init_events);
        let DriverBuild {
            driver,
            cached_u,
            cached_p,
        } = pollster::block_on(SolverDriver::build(
            &mesh,
            model,
            &request.params,
            &initial_u,
            &initial_p,
            request.wgpu_device.clone(),
            request.wgpu_queue.clone(),
        ))?;
        drop(init_guard);
        CFDApp::push_trace_init_event(
            &mut trace_init_events,
            "solver.new",
            solver_start.elapsed(),
            Some(format!("model_id={} cells={}", request.model_id, n_cells)),
        );

        // Update model_caps from the solver's ui_ports() (prefers PortRegistry over StateLayout)
        let ui_ports = driver.solver().ui_ports();
        model_caps.plot_stride = ui_ports.stride;
        model_caps.plot_u_offset = ui_ports.u_offset.unwrap_or(0);
        model_caps.plot_p_offset = ui_ports.p_offset.unwrap_or(0);
        model_caps.plot_has_u = ui_ports.u_offset.is_some();
        model_caps.plot_has_p = ui_ports.p_offset.is_some();

        let cache_start = std::time::Instant::now();
        let cached_cells = CFDApp::cache_cells(&mesh);
        CFDApp::push_trace_init_event(
            &mut trace_init_events,
            "mesh.cache_cells",
            cache_start.elapsed(),
            Some(format!("cells={}", mesh.num_cells())),
        );

        let renderer_start = std::time::Instant::now();
        let (renderer, viz_field) = if let (Some(device), Some(queue)) =
            (&request.wgpu_device, &request.wgpu_queue)
        {
            let state_size_bytes = mesh.num_cells() as u64 * model_caps.plot_stride as u64 * 4;
            let viz_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("CFD Viz Field Buffer 0"),
                size: state_size_bytes.max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let viz_buffer_1 = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("CFD Viz Field Buffer 1"),
                size: state_size_bytes.max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let viz_buffer_2 = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("CFD Viz Field Buffer 2"),
                size: state_size_bytes.max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            if state_size_bytes > 0 {
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("cfd_viz:init_copy_state"),
                });
                encoder.copy_buffer_to_buffer(
                    driver.solver().state_buffer(),
                    0,
                    &viz_buffer,
                    0,
                    state_size_bytes,
                );
                encoder.copy_buffer_to_buffer(
                    driver.solver().state_buffer(),
                    0,
                    &viz_buffer_1,
                    0,
                    state_size_bytes,
                );
                encoder.copy_buffer_to_buffer(
                    driver.solver().state_buffer(),
                    0,
                    &viz_buffer_2,
                    0,
                    state_size_bytes,
                );
                queue.submit(Some(encoder.finish()));
            }

            // Size the render buffers from the actual triangulated data: a
            // per-cell heuristic undersizes polygonal (Voronoi) meshes, whose
            // fan triangulation needs 3*(n-2) and wireframe 2*n vertices per
            // n-gon (~12+ for the typical hexagon).
            let vertices = cfd_renderer::build_mesh_vertices(&cached_cells);
            let line_vertices = cfd_renderer::build_line_vertices(&cached_cells);
            let max_vertices = vertices.len().max(line_vertices.len()).max(1);
            let mut renderer = cfd_renderer::CfdRenderResources::new(
                device,
                request.target_format,
                max_vertices,
            );
            renderer.update_mesh(queue, &vertices, &line_vertices);
            renderer.update_bind_group(device, &viz_buffer);
            (
                Some(renderer),
                Some(VizFieldBuffers {
                    buffers: [viz_buffer, viz_buffer_1, viz_buffer_2],
                    size_bytes: state_size_bytes,
                    front_idx: Arc::new(AtomicUsize::new(0)),
                    ready_idx: Arc::new(AtomicUsize::new(0)),
                }),
            )
        } else {
            (None, None)
        };
        CFDApp::push_trace_init_event(
            &mut trace_init_events,
            "renderer.init",
            renderer_start.elapsed(),
            Some(if renderer.is_some() {
                format!("mesh_vertices_cap={}", mesh.num_cells() * 10)
            } else {
                "skipped".to_string()
            }),
        );

        let min_cell_start = std::time::Instant::now();
        let actual_min_cell_size = mesh
            .cell_vol
            .iter()
            .map(|&v| v.sqrt())
            .fold(f64::INFINITY, f64::min);
        CFDApp::push_trace_init_event(
            &mut trace_init_events,
            "mesh.actual_min_cell_size",
            min_cell_start.elapsed(),
            Some(format!("{actual_min_cell_size:.4e}")),
        );

        CFDApp::push_trace_init_event(
            &mut trace_init_events,
            "init.total",
            init_start.elapsed(),
            Some(format!("cells={n_cells}")),
        );

        Ok(SolverInitOutcome {
            driver,
            mesh,
            cached_cells,
            actual_min_cell_size,
            cached_u,
            cached_p,
            model_caps,
            renderer,
            viz_field,
            trace_init_events,
        })
    }

    fn invalidate_plot_cache(&mut self) {
        self.plot_cache = None;
    }

    fn ensure_plot_cache(&mut self, want_values: bool) {
        let snapshot_seq = self.snapshot_seq;
        let field = self.plot_field;

        let cache_ok = self.plot_cache.as_ref().map_or(false, |cache| {
            cache.snapshot_seq == snapshot_seq
                && cache.field == field
                && (!want_values || cache.values.is_some())
        });
        if cache_ok {
            return;
        }

        let (len, has_data) = match field {
            PlotField::Pressure => (self.cached_p.len(), !self.cached_p.is_empty()),
            PlotField::VelocityX | PlotField::VelocityY | PlotField::VelocityMag => {
                (self.cached_u.len(), !self.cached_u.is_empty())
            }
        };

        if !has_data || len == 0 {
            self.plot_cache = Some(PlotCache {
                snapshot_seq,
                field,
                min: 0.0,
                max: 1.0,
                values: None,
            });
            return;
        }

        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        let mut values = want_values.then(|| Vec::with_capacity(len));

        for i in 0..len {
            let val = match field {
                PlotField::Pressure => self.cached_p[i],
                PlotField::VelocityX => self.cached_u[i].0,
                PlotField::VelocityY => self.cached_u[i].1,
                PlotField::VelocityMag => {
                    let (vx, vy) = self.cached_u[i];
                    (vx * vx + vy * vy).sqrt()
                }
            };
            min_val = min_val.min(val);
            max_val = max_val.max(val);
            if let Some(values) = values.as_mut() {
                values.push(val);
            }
        }

        if !(min_val.is_finite() && max_val.is_finite()) || min_val > max_val {
            min_val = 0.0;
            max_val = 1.0;
            values = None;
        } else if (max_val - min_val).abs() < 1e-12 {
            max_val = min_val + 1.0;
        }

        self.plot_cache = Some(PlotCache {
            snapshot_seq,
            field,
            min: min_val,
            max: max_val,
            values,
        });
    }

    fn render_layout_for_field(&self) -> (u32, u32, u32) {
        let stride = self.model_caps.plot_stride;
        let u_offset = self.model_caps.plot_u_offset;
        let p_offset = self.model_caps.plot_p_offset;

        match self.plot_field {
            PlotField::Pressure => (stride, p_offset, 0),
            PlotField::VelocityX => (stride, u_offset, 0),
            PlotField::VelocityY => (stride, u_offset + 1, 0),
            PlotField::VelocityMag => (stride, u_offset, 1),
        }
    }

    fn update_gpu_fluid(&mut self) {
        // The fluid's EOS sets the sound speed, so the REAL compressibility
        // `psi = 1/c^2` must be recomputed when the fluid (preset, density, or
        // viscosity) changes. No exaggeration.
        self.compressibility_psi = self.current_fluid.compressibility() as f32;
        self.sync_worker_params();
    }

    fn update_gpu_dt(&self) {
        self.sync_worker_params();
    }

    fn update_gpu_dtau(&self) {
        self.sync_worker_params();
    }

    fn update_gpu_scheme(&self) {
        self.sync_worker_params();
    }

    fn update_gpu_alpha_u(&self) {
        self.sync_worker_params();
    }

    fn update_gpu_alpha_p(&self) {
        self.sync_worker_params();
    }

    fn update_gpu_time_scheme(&self) {
        self.sync_worker_params();
    }

    fn update_gpu_outer_iters(&self) {
        self.sync_worker_params();
    }

    fn update_gpu_low_mach(&self) {
        self.sync_worker_params();
    }

    fn update_gpu_inlet_velocity(&self) {
        self.sync_worker_params();
    }

    fn update_gpu_preconditioner(&self) {
        self.sync_worker_params();
    }

    /// Render the right panel showing mesh stats and color legend.
    fn render_right_panel(
        &self,
        ctx: &egui::Context,
        has_solver: bool,
        min_val: f32,
        max_val: f32,
    ) {
        egui::SidePanel::right("legend").show(ctx, |ui| {
            if let Some(mesh) = &self.mesh {
                ui.heading("Mesh Stats");
                ui.label(format!("Cells: {}", mesh.num_cells()));
                ui.label(format!("Faces: {}", mesh.num_faces()));
                ui.label(format!("Vertices: {}", mesh.num_vertices()));
                if !mesh.cell_vol.is_empty() {
                    let min_vol = mesh.cell_vol.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max_vol = mesh
                        .cell_vol
                        .iter()
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max);
                    ui.label(format!("Cell vol: {:.2e} - {:.2e}", min_vol, max_vol));
                }
                ui.separator();
            }

            if has_solver {
                ui.heading("Legend");
                ui.label(format!("Max: {:.4}", max_val));

                let (rect, _response) =
                    ui.allocate_at_least(egui::vec2(30.0, 200.0), egui::Sense::hover());
                if ui.is_rect_visible(rect) {
                    let mut mesh = egui::Mesh::default();
                    let n_steps = 20;
                    for i in 0..n_steps {
                        let t0 = i as f32 / n_steps as f32;
                        let y0 = rect.max.y - t0 * rect.height();
                        let y1 = rect.max.y - (i as f32 + 1.0) / n_steps as f32 * rect.height();
                        let c0 = get_color(t0 as f64);

                        mesh.add_colored_rect(
                            egui::Rect::from_min_max(
                                egui::pos2(rect.min.x, y1),
                                egui::pos2(rect.max.x, y0),
                            ),
                            c0,
                        );
                    }
                    ui.painter().add(mesh);
                }

                ui.label(format!("Min: {:.4}", min_val));
            }
        });
    }

    /// Render the bottom panel with plot field selection and render mode.
    fn render_bottom_panel(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("plot_controls").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Plot Field:");
                let old_field = self.plot_field;
                ui.radio_value(&mut self.plot_field, PlotField::Pressure, "Pressure");
                ui.radio_value(&mut self.plot_field, PlotField::VelocityX, "Velocity X");
                ui.radio_value(&mut self.plot_field, PlotField::VelocityY, "Velocity Y");
                ui.radio_value(&mut self.plot_field, PlotField::VelocityMag, "Velocity Mag");
                if old_field != self.plot_field {
                    self.update_renderer_field();
                    self.invalidate_plot_cache();
                }

                ui.separator();
                ui.checkbox(&mut self.show_mesh_lines, "Show Mesh Lines");

                ui.separator();
                ui.label("Render Mode:");
                ui.radio_value(&mut self.render_mode, RenderMode::GpuDirect, "Direct");
                ui.radio_value(&mut self.render_mode, RenderMode::EguiPlot, "Plot (Slow)");
            });
        });
    }

    /// Render the central panel with the CFD visualization.
    fn render_central_panel(
        &self,
        ctx: &egui::Context,
        is_initializing: bool,
        has_solver: bool,
        min_val: f32,
        max_val: f32,
    ) {
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.is_running {
                // GPU solver runs in background thread
            }

            let cells = if has_solver && !self.cached_cells.is_empty() {
                Some(self.cached_cells.as_slice())
            } else {
                None
            };

            if let Some(cells) = cells {
                match self.render_mode {
                    RenderMode::GpuDirect => {
                        let rect = ui.available_rect_before_wrap();
                        let (rect, _response) =
                            ui.allocate_exact_size(rect.size(), egui::Sense::drag());

                        if let Some(renderer) = &self.cfd_renderer {
                            let renderer = renderer.clone();
                            let viewport_size = [rect.width(), rect.height()];

                            // Compute bounds
                            let (min_x, max_x, min_y, max_y) = cfd_renderer::compute_bounds(cells);
                            let mesh_width = max_x - min_x;
                            let mesh_height = max_y - min_y;

                            // Fit to screen preserving aspect ratio
                            let s = (rect.width() / mesh_width as f32)
                                .min(rect.height() / mesh_height as f32);

                            let scale_x = s / rect.width();
                            let scale_y = s / rect.height();

                            let mesh_center_x = (min_x + max_x) / 2.0;
                            let mesh_center_y = (min_y + max_y) / 2.0;

                            let tx = 0.5 - mesh_center_x as f32 * scale_x;
                            let ty = 0.5 - mesh_center_y as f32 * scale_y;

                            let (stride, offset, mode) = self.render_layout_for_field();

                            let cb = eframe::egui_wgpu::Callback::new_paint_callback(
                                rect,
                                CfdRenderCallback {
                                    renderer: renderer.clone(),
                                    uniforms: cfd_renderer::CfdUniforms {
                                        transform: [scale_x, scale_y, tx, ty],
                                        viewport_size,
                                        range: [min_val, max_val],
                                        stride,
                                        offset,
                                        mode,
                                        _padding: 0,
                                    },
                                    draw_lines: self.show_mesh_lines,
                                },
                            );

                            ui.painter().add(cb);
                        }
                    }
                    RenderMode::EguiPlot => {
                        let Some(vals) = self
                            .plot_cache
                            .as_ref()
                            .and_then(|cache| cache.values.as_deref())
                        else {
                            ui.centered_and_justified(|ui| {
                                ui.label("Waiting for field snapshot...");
                            });
                            return;
                        };

                        Plot::new("cfd_plot").data_aspect(1.0).show(ui, |plot_ui| {
                            for (i, polygon_points) in cells.iter().enumerate() {
                                let val = vals.get(i).copied().unwrap_or_default();
                                let t = (val - min_val as f64) / (max_val - min_val) as f64;
                                let color = get_color(t);

                                plot_ui.polygon(
                                    Polygon::new("", PlotPoints::new(polygon_points.clone()))
                                        .fill_color(color)
                                        .stroke(if self.show_mesh_lines {
                                            egui::Stroke::new(1.0, egui::Color32::BLACK)
                                        } else {
                                            egui::Stroke::NONE
                                        }),
                                );
                            }
                        });
                    }
                }
            } else {
                ui.centered_and_justified(|ui| {
                    if is_initializing {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label("Initializing solver...");
                        });
                    } else {
                        ui.label("Press Initialize to start");
                    }
                });
            }
        });
    }
}

impl eframe::App for CFDApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_init();
        self.spawn_pending_init();
        self.poll_solver_worker();
        let init_in_progress = self.init_rx.is_some();
        let init_pending = self.pending_init_request.is_some();
        let is_initializing = init_in_progress || init_pending;

        if self.is_running || is_initializing {
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
        }

        egui::SidePanel::left("controls").show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    ui.heading("CFD Controls");
                    if init_in_progress {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label("Initializing solver...");
                        });
                    } else if init_pending {
                        ui.label("Initializing solver...");
                    }

                    ui.add_enabled_ui(!is_initializing, |ui| {
                    ui.group(|ui| {
                        ui.label("Geometry");
                        let mut geom_changed = false;
                        geom_changed |= ui
                            .radio_value(
                                &mut self.selected_geometry,
                                GeometryType::BackwardsStep,
                                "Backwards Step",
                            )
                            .changed();
                        geom_changed |= ui
                            .radio_value(
                                &mut self.selected_geometry,
                                GeometryType::ChannelObstacle,
                                "Channel w/ Obstacle",
                            )
                            .changed();
                        geom_changed |= ui
                            .radio_value(
                                &mut self.selected_geometry,
                                GeometryType::Nozzle,
                                "Supersonic Nozzle",
                            )
                            .on_hover_text(
                                "Converging–diverging nozzle. Selecting it switches to the \
                                 All-Mach thermal model with a PRESSURE INLET + supersonic \
                                 outlet (toggle in Solver Parameters), driving the flow \
                                 through a choked throat to a SUPERSONIC exit (with \
                                 expansion cooling).",
                            )
                            .changed();
                        if geom_changed {
                            // The nozzle is meaningless without compressibility: when it is
                            // picked, switch to the supersonic-capable thermal model (unless
                            // already on an all-Mach model). Then re-seed the per-case
                            // defaults (the nozzle override sets the choking inlet speed +
                            // sub-critical back-pressure) and rebuild — mirroring the Model
                            // dropdown's apply-defaults-then-reinit behaviour.
                            if self.selected_geometry == GeometryType::Nozzle {
                                if self.model_id != "allmach_pressure"
                                    && self.model_id != "allmach_thermal"
                                {
                                    self.model_id = "allmach_thermal";
                                }
                                // Default the nozzle to the body-fitted structured
                                // grid (the validated configuration). The user can
                                // still switch to an unstructured mesh below.
                                self.mesh_type = MeshType::Fitted;
                            } else if self.mesh_type == MeshType::Fitted {
                                // `Fitted` is nozzle-only; fall back for other shapes.
                                self.mesh_type = MeshType::CutCell;
                            }
                            self.apply_model_defaults();
                            self.init_solver();
                        }
                    });

                    ui.group(|ui| {
                        ui.label("Mesh Parameters");
                        // The fitted (structured) mesh is a uniform grid: only a single
                        // target cell size applies, so hide the min-size / growth-rate
                        // controls it does not honour.
                        let show_grading = self.mesh_type.supports_size_grading();
                        if show_grading {
                            ui.add(
                                adaptive_slider(
                                    &mut self.min_cell_size,
                                    0.001..=self.max_cell_size,
                                )
                                .text("Min Cell Size"),
                            );
                        }
                        // For the fitted grid the sole control is the target cell
                        // size; give it a low floor (and see the raised nx/ny clamp in
                        // `build_mesh_with`) so the user can drive the mesh much finer
                        // than the old 0.025 lower bound — and finer still by typing.
                        let cell_size_base = if show_grading {
                            self.min_cell_size..=0.5
                        } else {
                            0.002..=0.5
                        };
                        ui.add(
                            adaptive_slider(&mut self.max_cell_size, cell_size_base).text(
                                if show_grading {
                                    "Max Cell Size"
                                } else {
                                    "Cell Size"
                                },
                            ),
                        )
                        .on_hover_text(if show_grading {
                            "Upper bound on cell size (coarse regions). Type any value \
                             to go beyond the slider ends."
                        } else {
                            "Target cell size for the structured grid (sets resolution). \
                             Type any value to go finer than the slider end."
                        });
                        if show_grading {
                            ui.add(
                                adaptive_slider(&mut self.growth_rate, 1.0..=2.0)
                                    .text("Growth Rate"),
                            );
                        }
                        ui.separator();
                        ui.label("Mesh Type");
                        // The body-fitted structured grid is only meaningful for the
                        // nozzle (it conforms to the CD-nozzle walls), so offer it only
                        // there. The unstructured meshers work for every geometry.
                        if self.selected_geometry == GeometryType::Nozzle {
                            ui.radio_value(&mut self.mesh_type, MeshType::Fitted, "Fitted")
                                .on_hover_text(
                                    "Body-fitted curvilinear structured grid conforming to \
                                     the nozzle walls. The validated configuration — \
                                     recommended for this case.",
                                );
                        }
                        ui.radio_value(&mut self.mesh_type, MeshType::CutCell, "CutCell");
                        ui.radio_value(&mut self.mesh_type, MeshType::Delaunay, "Delaunay");
                        ui.radio_value(&mut self.mesh_type, MeshType::Voronoi, "Voronoi");
                        ui.radio_value(&mut self.mesh_type, MeshType::VoronoiCvt, "Voronoi (CVT)")
                            .on_hover_text(
                                "Meshless Voronoi mesh with Lloyd/CVT seed relaxation: \
                                 near-hexagonal cells with close-to-zero interior-face \
                                 skewness (no post-generation vertex smoothing).",
                            );
                    });

                        ui.group(|ui| {
                        ui.label("Fluid Properties");
                        egui::ComboBox::from_label("Preset")
                            .selected_text(&self.current_fluid.name)
                            .show_ui(ui, |ui| {
                                for fluid in Fluid::presets() {
                                    if ui
                                        .selectable_value(
                                            &mut self.current_fluid,
                                            fluid.clone(),
                                            &fluid.name,
                                        )
                                        .clicked()
                                    {
                                        self.update_gpu_fluid();
                                    }
                                }
                            });

                        let mut density = self.current_fluid.density;
                        if ui
                            .add(
                                adaptive_slider(&mut density, 0.1..=20000.0)
                                    .text("Density (kg/m³)"),
                            )
                            .changed()
                        {
                            self.current_fluid.density = density;
                            self.current_fluid.name = "Custom".to_string();
                            if let crate::solver::model::eos::EosSpec::LinearCompressibility {
                                rho_ref,
                                ..
                            } = &mut self.current_fluid.eos
                            {
                                *rho_ref = density;
                            }
                            self.update_gpu_fluid();
                        }

                        let mut viscosity = self.current_fluid.viscosity;
                        if ui
                            .add(
                                adaptive_slider(&mut viscosity, 1e-6..=0.1)
                                    .logarithmic(true)
                                    .text("Viscosity (Pa·s)"),
                            )
                            .changed()
                        {
                            self.current_fluid.viscosity = viscosity;
                            self.current_fluid.name = "Custom".to_string();
                            self.update_gpu_fluid();
                        }
                        });

                        ui.group(|ui| {
                        ui.label("Inlet Conditions");

                        if ui
                            .add(
                                adaptive_slider(&mut self.inlet_velocity, 0.0..=10.0)
                                    .text("Inlet Velocity (m/s)"),
                            )
                            .changed()
                        {
                            self.update_gpu_inlet_velocity();
                        }

                        // All-Mach compressibility: ψ = 1/c² is the REAL value from the
                        // fluid's EOS (sound speed) — no exaggeration. Read-only physical
                        // readout of the regime (ψ, sound speed, inlet Mach).
                        if self.model_id == "allmach_pressure"
                            || self.model_id == "allmach_thermal"
                        {
                            let c_phys = self.current_fluid.sound_speed();
                            let mach_real = if c_phys > 0.0 {
                                self.inlet_velocity.abs() as f64 / c_phys
                            } else {
                                0.0
                            };
                            ui.label(format!(
                                "Compressibility ψ = 1/c² = {:.2e} (real EOS) · c = {c_phys:.0} m/s \
                                 · inlet Mach ≈ {mach_real:.2e}",
                                self.current_fluid.compressibility()
                            ));
                        }

                        // Supersonic-nozzle driver: a sub-critical (negative gauge)
                        // outlet back-pressure pulls the diverging section past Mach 1.
                        // Live (no rebuild) via `apply_params`, so the user can drag
                        // from 0 (subsonic) toward the stable floor and watch the exit
                        // go supersonic. Shown only for the all-Mach nozzle case; the
                        // floor (≈ −0.05) stays inside the gauge-EOS envelope (below it
                        // the outlet density crosses zero and the solve diverges).
                        if self.selected_geometry == GeometryType::Nozzle
                            && (self.model_id == "allmach_pressure"
                                || self.model_id == "allmach_thermal")
                        {
                            // Driving-mode toggle. Flipping it changes the Inlet/Outlet
                            // boundary KINDS (baked at build via `apply_pressure_inlet_nozzle_bcs`),
                            // so it must REBUILD the solver — unlike the pressure sliders below,
                            // which are live via `apply_params`. `init_solver` rebuilds from the
                            // current params (which carry the toggled `pressure_inlet`); we do NOT
                            // call `apply_model_defaults` here, which would reset the toggle.
                            let mut pin = self.pressure_inlet;
                            if ui
                                .checkbox(&mut pin, "Pressure inlet + supersonic outlet")
                                .on_hover_text(
                                    "ON: pin the inlet gauge pressure (gauge anchored upstream) \
                                     and let the outlet float — the physically-correct \
                                     supersonic-outlet driving. OFF: velocity inlet + a \
                                     sub-critical outlet back-pressure. Changing this REBUILDS \
                                     the solver (it flips the inlet/outlet boundary kinds).",
                                )
                                .changed()
                            {
                                self.pressure_inlet = pin;
                                self.init_solver();
                            }
                            if self.pressure_inlet {
                                // Pressure-inlet nozzle: tune the pinned inlet gauge
                                // pressure (the gauge anchor). The outlet floats —
                                // supersonic outlet, no back-pressure.
                                let mut p_in = self.inlet_pressure;
                                if ui
                                    .add(
                                        adaptive_slider(&mut p_in, 0.0..=0.12)
                                            .text("Inlet pressure (gauge)"),
                                    )
                                    .on_hover_text(
                                        "Pressure-inlet CD nozzle: pins the inlet gauge \
                                         pressure (the gauge anchor); the outlet floats \
                                         (supersonic, no back-pressure). Higher ⇒ stronger \
                                         drop ⇒ supersonic exit (M_exit ≈ 1.07 at 0.07). \
                                         Over-expanded here (the throat over-chokes).",
                                    )
                                    .changed()
                                {
                                    self.inlet_pressure = p_in;
                                    self.sync_worker_params();
                                }
                            } else {
                                let mut p_back = self.outlet_back_pressure;
                                if ui
                                    .add(
                                        adaptive_slider(&mut p_back, -0.05..=0.0)
                                            .text("Outlet back-pressure (gauge)"),
                                    )
                                    .on_hover_text(
                                        "Drives the converging–diverging nozzle. 0 ⇒ subsonic \
                                         exit; lowering it past the critical value pulls the \
                                         diverging section SUPERSONIC (M_exit up to ≈ 1.07 at \
                                         the stable floor).",
                                    )
                                    .changed()
                                {
                                    self.outlet_back_pressure = p_back;
                                    self.sync_worker_params();
                                }
                            }
                        }

                        // Reynolds Number Estimation
                        let char_length = 1.0; // Characteristic length (channel height)
                        let re = self.current_fluid.density
                            * self.inlet_velocity.abs() as f64
                            * char_length
                            / self.current_fluid.viscosity;
                        ui.label(format!("Est. Reynolds Number: {:.0}", re));
                        });

                        ui.group(|ui| {
                        ui.label("Solver Parameters");

                        // SI units:
                        // - Mesh coordinates/volumes are in meters.
                        // - `timestep` is seconds.
                        // - CFL is dimensionless: (wave_speed * dt / dx).
                        let dt_for_cfl = if self.mesh.is_some()
                            && self.cached_gpu_stats.dt.is_finite()
                            && self.cached_gpu_stats.dt > 0.0
                        {
                            self.cached_gpu_stats.dt as f64
                        } else {
                            self.timestep
                        };
                        let max_vel = self
                            .cached_u
                            .iter()
                            .map(|(vx, vy)| (vx * vx + vy * vy).sqrt())
                            .fold(0.0_f64, f64::max);
                        let adv_speed = max_vel.max(self.inlet_velocity.abs() as f64);
                        let sound_speed = if self.model_caps.supports_eos_tuning {
                            self.current_fluid.sound_speed()
                        } else {
                            0.0
                        };
                        let effective_sound_speed = if self.model_caps.supports_low_mach {
                            match self.low_mach_model {
                                GpuLowMachPrecondModel::Off => sound_speed,
                                GpuLowMachPrecondModel::Legacy => sound_speed.min(adv_speed),
                                GpuLowMachPrecondModel::WeissSmith => {
                                    let theta = (self.low_mach_theta_floor as f64).max(0.0);
                                    let c_floor = sound_speed * theta.sqrt();
                                    sound_speed.min(adv_speed.max(c_floor))
                                }
                            }
                        } else {
                            sound_speed
                        };
                        let wave_speed = adv_speed + effective_sound_speed;
                        let (recommended_dt, cfl) = if self.min_cell_size > 0.0 && wave_speed > 1e-12
                        {
                            (
                                0.5 * self.min_cell_size / wave_speed,
                                wave_speed * dt_for_cfl / self.min_cell_size,
                            )
                        } else {
                            (0.0, 0.0)
                        };

                        if ui
                            .add(
                                adaptive_slider(&mut self.timestep, 0.0001..=0.1)
                                    .text("Timestep (s)"),
                            )
                            .changed()
                        {
                            self.update_gpu_dt();
                        }

                        if ui
                            .checkbox(&mut self.adaptive_dt, "Adaptive Timestep")
                            .changed()
                        {
                            self.sync_worker_params();
                        }
                        if self.adaptive_dt {
                            if ui
                                .add(
                                    adaptive_slider(&mut self.target_cfl, 0.1..=1.0)
                                        .text("Target CFL"),
                                )
                                .changed()
                            {
                                self.sync_worker_params();
                            }
                        }

                        if self.model_caps.supports_outer_iters
                            || self.model_caps.supports_low_mach
                            || self.model_caps.supports_dtau
                        {
                            ui.separator();
                            ui.label("Coupled/Implicit Controls");
                            if self.model_caps.supports_outer_iters
                                && ui
                                    .add(
                                        adaptive_slider(&mut self.outer_iters, 1..=100)
                                            .text("Outer Iterations"),
                                    )
                                    .changed()
                            {
                                self.update_gpu_outer_iters();
                            }
                            if self.model_caps.supports_dtau {
                                ui.separator();
                                ui.label("Dual Time Stepping");
                                if ui
                                    .checkbox(&mut self.dual_time, "Enable pseudo-time (dtau)")
                                    .changed()
                                {
                                    self.update_gpu_dtau();
                                    if self.dual_time && self.model_id == "compressible" {
                                        // Pressure under-relaxation is destabilizing for the
                                        // compressible solver in dual-time mode (unphysical striping
                                        // + runaway velocities). Keep pressure unrelaxed.
                                        if (self.alpha_p - 1.0).abs() > 1e-12 {
                                            self.alpha_p = 1.0;
                                            self.update_gpu_alpha_p();
                                        }
                                        // Provide conservative defaults if the UI is still at the
                                        // typical incompressible starting point.
                                        if (self.alpha_u - 0.7).abs() < 1e-12 {
                                            self.alpha_u = 0.7;
                                            self.update_gpu_alpha_u();
                                        }
                                    }
                                }
                                ui.add_enabled_ui(self.dual_time, |ui| {
                                    if ui
                                        .add(
                                            adaptive_slider(&mut self.dtau, 1e-8..=0.1)
                                                .logarithmic(true)
                                                .text("dtau (s)"),
                                        )
                                        .changed()
                                    {
                                        self.update_gpu_dtau();
                                    }

                                    let pseudo_wave_speed = adv_speed + effective_sound_speed;
                                    if self.min_cell_size > 0.0 && pseudo_wave_speed > 1e-12 {
                                        let pseudo_cfl = pseudo_wave_speed * self.dtau
                                            / self.min_cell_size.max(1e-12);
                                        ui.label(format!("Pseudo CFL≈{:.2}", pseudo_cfl));
                                    }
                                });
                            }
                            if self.model_caps.supports_low_mach {
                                ui.separator();
                                ui.label("Low-Mach Preconditioning");
                                let prev_model = self.low_mach_model;
                                egui::ComboBox::from_label("Model")
                                    .selected_text(match self.low_mach_model {
                                        GpuLowMachPrecondModel::Off => "Off",
                                        GpuLowMachPrecondModel::Legacy => "Legacy",
                                        GpuLowMachPrecondModel::WeissSmith => "Weiss-Smith",
                                    })
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(
                                            &mut self.low_mach_model,
                                            GpuLowMachPrecondModel::Off,
                                            "Off",
                                        );
                                        ui.selectable_value(
                                            &mut self.low_mach_model,
                                            GpuLowMachPrecondModel::Legacy,
                                            "Legacy",
                                        );
                                        ui.selectable_value(
                                            &mut self.low_mach_model,
                                            GpuLowMachPrecondModel::WeissSmith,
                                            "Weiss-Smith",
                                        );
                                    });
                                if self.low_mach_model != prev_model {
                                    self.update_gpu_low_mach();
                                }

                                let theta_enabled = matches!(
                                    self.low_mach_model,
                                    GpuLowMachPrecondModel::WeissSmith
                                );
                                ui.add_enabled_ui(theta_enabled, |ui| {
                                    if ui
                                        .add(
                                            adaptive_slider(
                                                &mut self.low_mach_theta_floor,
                                                1e-8f32..=1e-2f32,
                                            )
                                            .logarithmic(true)
                                            .text("θ floor"),
                                        )
                                        .changed()
                                    {
                                        self.update_gpu_low_mach();
                                    }
                                });
                                if !theta_enabled {
                                    ui.weak("θ floor is used by Weiss-Smith.");
                                }

                                let coupling_enabled = !matches!(
                                    self.low_mach_model,
                                    GpuLowMachPrecondModel::Off
                                );
                                ui.add_enabled_ui(coupling_enabled, |ui| {
                                    if ui
                                        .add(
                                            adaptive_slider(
                                                &mut self.low_mach_pressure_coupling_alpha,
                                                0.0f32..=1.0f32,
                                            )
                                            .text("Pressure coupling α"),
                                        )
                                        .changed()
                                    {
                                        self.update_gpu_low_mach();
                                    }
                                });
                                if !coupling_enabled {
                                    ui.weak("Pressure coupling is active when preconditioning is on.");
                                }

                                if sound_speed > 1e-12 {
                                    let mach = adv_speed / sound_speed;
                                    ui.label(format!("Estimated Mach: {:.3e}", mach));
                                    if mach < 0.3
                                        && matches!(self.low_mach_model, GpuLowMachPrecondModel::Off)
                                    {
                                        ui.weak("Tip: enable preconditioning for low Mach.");
                                    }
                                }
                            }
                        }

                        ui.separator();
                        ui.label("Debug");
                        if ui
                            .checkbox(&mut self.log_convergence, "Log convergence to console")
                            .changed()
                        {
                            self.sync_worker_params();
                        }
                        ui.add_enabled_ui(self.log_convergence, |ui| {
                            if ui
                                .add(
                                    adaptive_slider(&mut self.log_every_steps, 1..=200)
                                        .text("Log every N steps"),
                                )
                                .changed()
                            {
                                self.sync_worker_params();
                            }
                        });

                        ui.separator();
                        ui.label("Tracing / Profiling");
                        let prev_trace = self.trace_enabled;
                        if ui
                            .checkbox(
                                &mut self.trace_enabled,
                                "Save trace to file (for replay + performance)",
                            )
                            .changed()
                        {
                            if self.trace_enabled {
                                self.start_trace();
                            } else {
                                self.stop_trace();
                            }
                        }

                        ui.horizontal(|ui| {
                            ui.label("Trace file");
                            ui.text_edit_singleline(&mut self.trace_path);
                        });
                        if prev_trace && self.trace_path.is_empty() {
                            ui.weak("Tip: set a path before restarting the trace.");
                        } else if self.trace_pending_start {
                            ui.weak("Trace will start after initialization.");
                        } else if self.trace_enabled && !cfg!(feature = "profiling") {
                            ui.weak("Built without `profiling` feature: trace has timings, but no GPU-CPU profiling counters.");
                        }

                        if cfl > 1.0 {
                            if self.model_caps.supports_dtau && self.dual_time {
                                ui.colored_label(
                                    egui::Color32::YELLOW,
                                    format!("Acoustic CFL≈{:.1} (>1) (dual time enabled)", cfl),
                                );
                            } else {
                                ui.colored_label(
                                    egui::Color32::RED,
                                    format!("⚠ CFL≈{:.1} (>1, may be unstable!)", cfl),
                                );
                                ui.colored_label(
                                    egui::Color32::YELLOW,
                                    format!("Recommended dt ≤ {:.4}", recommended_dt),
                                );
                            }
                        } else if cfl > 0.5 {
                            ui.colored_label(
                                egui::Color32::YELLOW,
                                format!("CFL≈{:.2} (moderate)", cfl),
                            );
                        }

                        ui.separator();
                        ui.label("Advection Scheme");
                        if ui
                            .radio(matches!(self.selected_scheme, Scheme::Upwind), "First order (Upwind)")
                            .clicked()
                        {
                            self.selected_scheme = Scheme::Upwind;
                            self.update_gpu_scheme();
                        }
                        if ui
                            .radio(
                                matches!(self.selected_scheme, Scheme::SecondOrderUpwind),
                                "SOU (Second order upwind)",
                            )
                            .clicked()
                        {
                            self.selected_scheme = Scheme::SecondOrderUpwind;
                            self.update_gpu_scheme();
                        }
                        if ui
                            .radio(
                                matches!(self.selected_scheme, Scheme::SecondOrderUpwindMinMod),
                                "SOU (MinMod limiter)",
                            )
                            .clicked()
                        {
                            self.selected_scheme = Scheme::SecondOrderUpwindMinMod;
                            self.update_gpu_scheme();
                        }
                        if ui
                            .radio(
                                matches!(self.selected_scheme, Scheme::SecondOrderUpwindVanLeer),
                                "SOU (VanLeer limiter)",
                            )
                            .clicked()
                        {
                            self.selected_scheme = Scheme::SecondOrderUpwindVanLeer;
                            self.update_gpu_scheme();
                        }
                        if ui
                            .radio(matches!(self.selected_scheme, Scheme::QUICK), "QUICK")
                            .clicked()
                        {
                            self.selected_scheme = Scheme::QUICK;
                            self.update_gpu_scheme();
                        }
                        if ui
                            .radio(
                                matches!(self.selected_scheme, Scheme::QUICKMinMod),
                                "QUICK (MinMod limiter)",
                            )
                            .clicked()
                        {
                            self.selected_scheme = Scheme::QUICKMinMod;
                            self.update_gpu_scheme();
                        }
                        if ui
                            .radio(
                                matches!(self.selected_scheme, Scheme::QUICKVanLeer),
                                "QUICK (VanLeer limiter)",
                            )
                            .clicked()
                        {
                            self.selected_scheme = Scheme::QUICKVanLeer;
                            self.update_gpu_scheme();
                        }

                        if self.model_caps.supports_eos_tuning {
                            ui.label("Compressible solver uses this for KT flux reconstruction + deferred-correction advection.");
                        }

                        ui.separator();
                        ui.label("Preconditioner");
                        const UI_MAX_BLOCK_JACOBI: u32 = 16;
                        let supports_preconditioner = self.model_caps.supports_preconditioner;
                        let model_owns_preconditioner = self.model_caps.model_owns_preconditioner;
                        let block_jacobi_supported =
                            self.model_caps.unknowns_per_cell <= UI_MAX_BLOCK_JACOBI;

                        if model_owns_preconditioner {
                            ui.weak("Model-owned Schur: selector chooses pressure solve (Chebyshev vs AMG).");
                        } else {
                            ui.weak("Krylov preconditioner for the coupled linear solve.");
                        }
                        if supports_preconditioner {
                            if ui
                                .radio(
                                    matches!(self.selected_preconditioner, PreconditionerType::Jacobi),
                                    "Jacobi (diag)",
                                )
                                .clicked()
                            {
                                self.selected_preconditioner = PreconditionerType::Jacobi;
                                self.update_gpu_preconditioner();
                            }
                            ui.add_enabled_ui(block_jacobi_supported, |ui| {
                                if ui
                                    .radio(
                                        matches!(
                                            self.selected_preconditioner,
                                            PreconditionerType::BlockJacobi
                                        ),
                                        "BlockJacobi (cell block)",
                                    )
                                    .clicked()
                                {
                                    self.selected_preconditioner = PreconditionerType::BlockJacobi;
                                    self.update_gpu_preconditioner();
                                }
                            });
                            if !block_jacobi_supported {
                                ui.weak(format!(
                                    "BlockJacobi requires ≤{UI_MAX_BLOCK_JACOBI} unknowns/cell."
                                ));
                            }
                            if ui
                                .radio(
                                    matches!(self.selected_preconditioner, PreconditionerType::Amg),
                                    "AMG (Multigrid)",
                                )
                                .clicked()
                            {
                                self.selected_preconditioner = PreconditionerType::Amg;
                                self.update_gpu_preconditioner();
                            }
                        } else {
                            ui.weak("(not declared by model)");
                        }

                        if self.model_caps.supports_alpha_u || self.model_caps.supports_alpha_p {
                            ui.separator();
                            ui.label("Under-Relaxation Factors");
                            let compressible_dual_time =
                                self.model_id == "compressible" && self.dual_time;
                            ui.weak(if compressible_dual_time {
                                "Tip: compressible dual-time now defaults to α_U=1.0; lower it or set nonconverged fallback damping only if pseudo-time steps stall."
                            } else {
                                "Tip: start with α_U≈0.7 and α_P≈0.3; α=1 can diverge at high Re."
                            });
                            if self.model_caps.supports_alpha_u
                                && ui
                                    .add(
                                        adaptive_slider(&mut self.alpha_u, 0.1..=1.0)
                                            .text("α_U (Velocity)"),
                                    )
                                    .changed()
                            {
                                self.update_gpu_alpha_u();
                            }
                            if self.model_caps.supports_alpha_p
                            {
                                if compressible_dual_time {
                                    if (self.alpha_p - 1.0).abs() > 1e-12 {
                                        self.alpha_p = 1.0;
                                        self.update_gpu_alpha_p();
                                    }
                                    ui.label("α_P (Pressure): 1.0 (locked)");
                                } else if ui
                                    .add(
                                        adaptive_slider(&mut self.alpha_p, 0.1..=1.0)
                                            .text("α_P (Pressure)"),
                                    )
                                    .changed()
                                {
                                    self.update_gpu_alpha_p();
                                }
                            }
                        }
                        });

                        ui.separator();
                        ui.label("Time Stepping Scheme");
                        egui::ComboBox::from_label("Time Scheme")
                            .selected_text(format!("{:?}", self.time_scheme))
                            .show_ui(ui, |ui| {
                                if ui
                                    .selectable_value(
                                        &mut self.time_scheme,
                                        GpuTimeScheme::Euler,
                                        "Euler",
                                    )
                                    .clicked()
                                {
                                    self.update_gpu_time_scheme();
                                }
                                if ui
                                    .selectable_value(
                                        &mut self.time_scheme,
                                        GpuTimeScheme::BDF2,
                                        "BDF2",
                                    )
                                    .clicked()
                                {
                                    self.update_gpu_time_scheme();
                                }
                            });

                        ui.separator();
                        ui.label("Model");
                        let prev_model_id = self.model_id;
                        egui::ComboBox::from_label("Model")
                            .selected_text(CFDApp::model_label(self.model_id))
                            .show_ui(ui, |ui| {
                                for (id, label) in CFDApp::supported_ui_models() {
                                    ui.selectable_value(&mut self.model_id, id, label);
                                }
                            });
                        if prev_model_id != self.model_id {
                            // Re-seed every solver knob from the per-model defaults
                            // (scheme, relaxation, outer cap, adaptive-dt target,
                            // low-Mach preconditioning, viscosity floor) so each
                            // model starts from settings that converge and do not
                            // diverge. `init_solver` refreshes the model caps.
                            self.apply_model_defaults();
                            self.init_solver();
                        }

                        ui.separator();
                        egui::ComboBox::from_label("Compute Backend")
                            .selected_text(self.backend.label())
                            .show_ui(ui, |ui| {
                                for choice in BackendChoice::ALL {
                                    ui.selectable_value(
                                        &mut self.backend,
                                        choice,
                                        choice.label(),
                                    );
                                }
                            })
                            .response
                            .on_hover_text(
                                "CPU backends run the selected model without a GPU \
                                 adapter, at parity with the GPU; GPU-only telemetry \
                                 (profiling, per-graph timings) is unavailable. \
                                 Interpreter = reference tree-walker; Transpiled = \
                                 compiled kernels (fast); SIMD adds vectorized \
                                 block-matvec kernels plus mixed-precision (f32) \
                                 storage for the pressure inner solve and AMG \
                                 levels — a rounding-level result change, fastest \
                                 on solve-heavy runs.",
                            );
                        if self.backend.is_cpu() {
                            ui.add(
                                adaptive_slider(&mut self.cpu_threads, 1..=16).text("Cores"),
                            );
                            egui::ComboBox::from_label("Solver Precision")
                                .selected_text(if self.cpu_precision_f32 {
                                    "f32 (GPU-like)"
                                } else {
                                    "f64 (reference)"
                                })
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut self.cpu_precision_f32,
                                        false,
                                        "f64 (reference)",
                                    );
                                    ui.selectable_value(
                                        &mut self.cpu_precision_f32,
                                        true,
                                        "f32 (GPU-like)",
                                    );
                                })
                                .response
                                .on_hover_text(
                                    "Scalar precision of the coupled linear solve. f64 is \
                                     the reference; f32 mirrors the GPU's arithmetic and \
                                     halves solve bandwidth (results differ at rounding \
                                     level).",
                                );
                            ui.label("Applied on Initialize / Reset.");
                        }

                        ui.separator();
                        ui.label("Pressure-Velocity Coupling");
                        ui.label("Coupled solver (block system)");

                        if ui.button("Initialize / Reset").clicked() {
                            self.init_solver();
                        }
                    });

                    let has_solver = self.mesh.is_some();

                    if ui
                        .add_enabled(
                            has_solver && !is_initializing,
                            egui::Button::new(if self.is_running { "Pause" } else { "Run" }),
                        )
                        .clicked()
                    {
                        self.is_running = !self.is_running;

                        if self.is_running {
                            self.cached_error = None;
                            self.cached_message = None;
                            self.sync_worker_params();
                            self.solver_worker.send(SolverWorkerCommand::SetRunning(true));
                        } else {
                            self.solver_worker.send(SolverWorkerCommand::SetRunning(false));
                        }
                    }

                    ui.separator();

                    if let Some(err) = &self.cached_error {
                        ui.colored_label(egui::Color32::RED, err);
                    }
                    if let Some(message) = &self.cached_message {
                        ui.colored_label(egui::Color32::YELLOW, message);
                    }

                    if has_solver {
                        let stats = &self.cached_gpu_stats;
                        ui.label(format!("dt: {:.2e}", stats.dt));
                        if stats.linear_solves > 0 {
                            let status = if stats.linear_last.diverged
                                || !stats.linear_last.residual.is_finite()
                            {
                                "diverged"
                            } else if stats.linear_last.converged {
                                "converged"
                            } else {
                                "running"
                            };
                            ui.label(format!(
                                "Linear: {} solve(s), iters={} res={:.2e} ({})",
                                stats.linear_solves,
                                stats.linear_last.iterations,
                                stats.linear_last.residual,
                                status
                            ));
                        }
                        if stats.outer_iterations > 0 {
                            let status_suffix = stats
                                .outer_step_status
                                .map(|status| format!(" ({})", status.as_str()))
                                .unwrap_or_default();
                            ui.label(format!(
                                "Coupled: {} iters, U:{:.2e} P:{:.2e}{}",
                                stats.outer_iterations,
                                stats.outer_residual_u,
                                stats.outer_residual_p,
                                status_suffix,
                            ));
                        }
                        if stats.positivity_min_rho.is_some() || stats.positivity_min_p.is_some() {
                            ui.label(format!(
                                "Positivity: rho_min={:.2e} (n={}) p_min={:.2e} (n={})",
                                stats.positivity_min_rho.unwrap_or(f32::NAN),
                                stats.positivity_rho_undershoots,
                                stats.positivity_min_p.unwrap_or(f32::NAN),
                                stats.positivity_pressure_undershoots,
                            ));
                        }
                        ui.label(format!("Step time: {:.1} ms", stats.step_time_ms));
                    }
                });
        });

        let has_solver = self.mesh.is_some();

        if has_solver {
            self.ensure_plot_cache(matches!(self.render_mode, RenderMode::EguiPlot));
        }

        let (min_val, max_val) = self
            .plot_cache
            .as_ref()
            .map(|cache| (cache.min as f32, cache.max as f32))
            .unwrap_or((0.0, 1.0));

        self.render_right_panel(ctx, has_solver, min_val, max_val);
        self.render_bottom_panel(ctx);
        self.render_central_panel(ctx, is_initializing, has_solver, min_val, max_val);
    }
}

fn trace_runtime_params_from_worker(params: RuntimeParams) -> tracefmt::TraceRuntimeParams {
    tracefmt::TraceRuntimeParams {
        adaptive_dt: params.adaptive_dt,
        target_cfl: params.target_cfl,
        requested_dt: params.requested_dt,
        dtau: params.dtau,
        advection_scheme: tracefmt::TraceScheme::from(params.advection_scheme),
        time_scheme: tracefmt::TraceTimeScheme::from(params.time_scheme),
        preconditioner: tracefmt::TracePreconditioner::from(params.preconditioner),
        outer_iters: params.outer_iters,
        low_mach_model: tracefmt::TraceLowMachModel::from(params.low_mach_model),
        low_mach_theta_floor: params.low_mach_theta_floor,
        low_mach_pressure_coupling_alpha: params.low_mach_pressure_coupling_alpha,
        alpha_u: params.alpha_u,
        alpha_p: params.alpha_p,
        inlet_velocity: params.inlet_velocity,
        density: params.density,
        viscosity: params.viscosity,
        eos: tracefmt::TraceEosSpec::from(params.eos),
    }
}

fn solver_worker_stop_trace(
    trace: &mut Option<SolverTraceSession>,
    driver: &mut Option<SolverDriver>,
) {
    let Some(mut session) = trace.take() else {
        return;
    };

    if let Some(s) = driver.as_mut().map(|d| d.solver_mut()) {
        s.set_collect_trace(false);
        let _ = s.enable_detailed_profiling(false);
        if session.profiling_enabled {
            let _ = s.end_profiling_session();
            if let Ok(stats) = s.get_profiling_stats() {
                let categories = stats
                    .get_all_stats()
                    .into_iter()
                    .map(|(category, s)| {
                        (
                            category.name().to_string(),
                            tracefmt::TraceCategoryStats {
                                total_seconds: s.total_time.as_secs_f64(),
                                call_count: s.call_count,
                                min_seconds: s.min_time.as_secs_f64(),
                                max_seconds: s.max_time.as_secs_f64(),
                                total_bytes: s.total_bytes,
                            },
                        )
                    })
                    .collect::<Vec<_>>();

                let mut locations = stats.get_location_stats();
                locations.sort_by(|a, b| b.1.total_time.cmp(&a.1.total_time));

                let locations = locations
                    .into_iter()
                    .take(200)
                    .map(|(name, s)| {
                        (
                            name,
                            tracefmt::TraceCategoryStats {
                                total_seconds: s.total_time.as_secs_f64(),
                                call_count: s.call_count,
                                min_seconds: s.min_time.as_secs_f64(),
                                max_seconds: s.max_time.as_secs_f64(),
                                total_bytes: s.total_bytes,
                            },
                        )
                    })
                    .collect();

                let (mem_cpu, mem_gpu) = stats.get_memory_stats();
                let memory_cpu = tracefmt::TraceMemoryStats {
                    alloc_bytes: mem_cpu.alloc_bytes,
                    alloc_count: mem_cpu.alloc_count,
                    free_bytes: mem_cpu.free_bytes,
                    free_count: mem_cpu.free_count,
                    max_alloc_request: mem_cpu.max_alloc_request,
                };
                let memory_gpu = tracefmt::TraceMemoryStats {
                    alloc_bytes: mem_gpu.alloc_bytes,
                    alloc_count: mem_gpu.alloc_count,
                    free_bytes: mem_gpu.free_bytes,
                    free_count: mem_gpu.free_count,
                    max_alloc_request: mem_gpu.max_alloc_request,
                };

                let memory_locations_cpu = stats
                    .get_memory_location_stats(crate::solver::gpu::profiling::MemoryDomain::Cpu)
                    .into_iter()
                    .map(|(name, s)| {
                        (
                            name,
                            tracefmt::TraceMemoryStats {
                                alloc_bytes: s.alloc_bytes,
                                alloc_count: s.alloc_count,
                                free_bytes: s.free_bytes,
                                free_count: s.free_count,
                                max_alloc_request: s.max_alloc_request,
                            },
                        )
                    })
                    .collect();
                let memory_locations_gpu = stats
                    .get_memory_location_stats(crate::solver::gpu::profiling::MemoryDomain::Gpu)
                    .into_iter()
                    .map(|(name, s)| {
                        (
                            name,
                            tracefmt::TraceMemoryStats {
                                alloc_bytes: s.alloc_bytes,
                                alloc_count: s.alloc_count,
                                free_bytes: s.free_bytes,
                                free_count: s.free_count,
                                max_alloc_request: s.max_alloc_request,
                            },
                        )
                    })
                    .collect();

                let profiling = tracefmt::TraceProfilingEvent {
                    session_total_seconds: stats.get_session_total().as_secs_f64(),
                    iteration_count: stats.get_iteration_count(),
                    categories,
                    locations,
                    memory_cpu,
                    memory_gpu,
                    memory_locations_cpu,
                    memory_locations_gpu,
                };

                let _ = session
                    .writer
                    .write_event(&tracefmt::TraceEvent::Profiling(profiling));
            }
        }
    }

    let _ = session
        .writer
        .write_event(&tracefmt::TraceEvent::Footer(tracefmt::TraceFooter {
            closed_unix_ms: tracefmt::now_unix_ms(),
        }));

    let path = session.writer.path().display().to_string();
    let _ = session.writer.close();
    eprintln!("[cfd2][trace] closed {}", path);
}

fn solver_worker_main(
    cmd_rx: mpsc::Receiver<SolverWorkerCommand>,
    evt_tx: mpsc::Sender<SolverWorkerEvent>,
) {
    let mut driver: Option<SolverDriver> = None;
    let mut trace: Option<SolverTraceSession> = None;
    let mut model_id: &'static str = "<uninitialized>";
    let mut viz_field: Option<VizFieldBuffers> = None;
    let mut params = RuntimeParams {
        adaptive_dt: false,
        target_cfl: 0.9,
        requested_dt: 0.001,
        dtau: 0.0,
        log_convergence: false,
        log_every_steps: 50,
        advection_scheme: Scheme::Upwind,
        time_scheme: GpuTimeScheme::Euler,
        preconditioner: PreconditionerType::Jacobi,
        outer_iters: 1,
        outer_auto_converge: false,
        low_mach_model: GpuLowMachPrecondModel::Off,
        low_mach_theta_floor: 1e-6,
        low_mach_pressure_coupling_alpha: 1.0,
        alpha_u: 1.0,
        alpha_p: 1.0,
        inlet_velocity: 0.0,
        density: 1.0,
        viscosity: 0.0,
        eos: crate::solver::model::eos::EosSpec::Constant,
        compressibility_psi: 0.0,
        outlet_back_pressure: 0.0,
        pressure_inlet: false,
        inlet_pressure: 0.0,
    };

    let mut running = false;
    let mut step_idx: u64 = 0;
    let mut last_stats_publish = std::time::Instant::now();
    let mut last_snapshot_publish = std::time::Instant::now();
    let stats_publish_interval = std::time::Duration::from_millis(33);
    let snapshot_publish_interval = std::time::Duration::from_millis(100);

    loop {
        while let Ok(cmd) = cmd_rx.try_recv() {
            if !solver_worker_handle_cmd(
                cmd,
                &mut driver,
                &mut trace,
                &mut model_id,
                &mut viz_field,
                &mut params,
                &mut running,
                &mut step_idx,
                &mut last_stats_publish,
                &mut last_snapshot_publish,
                &evt_tx,
            ) {
                return;
            }
        }

        if !running {
            match cmd_rx.recv_timeout(std::time::Duration::from_millis(16)) {
                Ok(cmd) => {
                    if !solver_worker_handle_cmd(
                        cmd,
                        &mut driver,
                        &mut trace,
                        &mut model_id,
                        &mut viz_field,
                        &mut params,
                        &mut running,
                        &mut step_idx,
                        &mut last_stats_publish,
                        &mut last_snapshot_publish,
                        &evt_tx,
                    ) {
                        return;
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => return,
            }
            continue;
        }

        let Some(driver) = driver.as_mut() else {
            running = false;
            let _ = evt_tx.send(SolverWorkerEvent::Error(
                "solver worker entered running state without an initialized solver".to_string(),
            ));
            let _ = evt_tx.send(SolverWorkerEvent::Running(false));
            continue;
        };

        // Logging / readback cadence (depends on the current step index + timers).
        let log_every_steps = params.log_every_steps.max(1) as u64;
        let should_log = params.log_convergence && (step_idx % log_every_steps == 0);
        let should_readback = step_idx == 0
            || last_snapshot_publish.elapsed() >= snapshot_publish_interval
            || should_log;

        // Acoustic-aware adaptive timestep + one step + divergence / steady-state
        // detection all live in the shared driver now (was an inline adaptive-dt
        // block + `step_with_stats` + readback). GUI-only concerns — viz upload,
        // publishing, trace — stay here.
        let outcome = driver.step(should_readback);
        if let Some(DivergeReason::StepError(err)) = &outcome.diverged {
            running = false;
            let _ =
                evt_tx.send(SolverWorkerEvent::Error(format!("solver step failed: {err}")));
            let _ = evt_tx.send(SolverWorkerEvent::Running(false));
            continue;
        }
        let solver = driver.solver();
        let step_time_ms = outcome.step_time_ms;

        if let Some(viz_field) = viz_field.as_ref() {
            if viz_field.size_bytes > 0 {
                let n = viz_field.buffers.len().max(1);
                let front = viz_field.front_idx.load(Ordering::Acquire) % n;
                let ready = viz_field.ready_idx.load(Ordering::Acquire) % n;

                let mut write_idx = (ready + 1) % n;
                for _ in 0..n {
                    if write_idx != front && write_idx != ready {
                        break;
                    }
                    write_idx = (write_idx + 1) % n;
                }
                if write_idx == front || write_idx == ready {
                    // Should be unreachable with n>=3, but keep a safe fallback.
                    write_idx = (front + 1) % n;
                }
                solver.copy_state_to_buffer(&viz_field.buffers[write_idx]);
                viz_field.ready_idx.store(write_idx, Ordering::Release);
            }
        }

        if outcome.should_stop {
            running = false;
            let _ = evt_tx.send(SolverWorkerEvent::Running(false));
            continue;
        }

        let linear_solves = outcome.linear_stats.len() as u32;
        let linear_last = outcome.linear_stats.last().copied().unwrap_or_default();
        if matches!(outcome.diverged, Some(DivergeReason::LinearSolver)) {
            running = false;
            let _ = evt_tx.send(SolverWorkerEvent::Error(
                "divergence detected (linear solver)".to_string(),
            ));
            let _ = evt_tx.send(SolverWorkerEvent::Running(false));
            continue;
        }

        let mut stats = CachedGpuStats {
            dt: solver.dt(),
            step_time_ms,
            linear_solves,
            linear_last,
            ..Default::default()
        };

        let step_stats = solver.step_stats();
        if let Some(iters) = step_stats.outer_iterations {
            stats.outer_iterations = iters;
        }
        if let Some(res_u) = step_stats.outer_residual_u {
            stats.outer_residual_u = res_u;
        }
        if let Some(res_p) = step_stats.outer_residual_p {
            stats.outer_residual_p = res_p;
        }
        stats.outer_step_status = step_stats.outer_step_status;
        stats.positivity_min_rho = step_stats.positivity_min_rho;
        stats.positivity_min_p = step_stats.positivity_min_p;
        stats.positivity_rho_undershoots = step_stats.positivity_rho_undershoot_count.unwrap_or(0);
        stats.positivity_pressure_undershoots = step_stats
            .positivity_pressure_undershoot_count
            .unwrap_or(0);

        let mut trace_max_u: Option<f64> = None;
        let mut trace_p_min: Option<f64> = None;
        let mut trace_p_max: Option<f64> = None;

        if let Some(rb) = outcome.readback {
            let now = std::time::Instant::now();
            last_snapshot_publish = now;
            last_stats_publish = now;
            let fs = rb.stats;
            trace_max_u = Some(fs.max_vel);
            trace_p_min = Some(fs.p_min);
            trace_p_max = Some(fs.p_max);

            if should_log || fs.nonfinite_u > 0 || fs.nonfinite_p > 0 {
                let step_stats = solver.step_stats();
                let outer_str = step_stats
                    .outer_iterations
                    .map(|iters| {
                        let status_suffix = step_stats
                            .outer_step_status
                            .map(|status| format!(", status={}", status.as_str()))
                            .unwrap_or_default();
                        let abs_residuals = solver.outer_field_residuals();
                        let scaled_residuals = solver.outer_field_residuals_scaled();

                        if let (Some(abs), Some(scaled)) = (abs_residuals, scaled_residuals) {
                            // Both absolute and scaled residuals available
                            let fields = abs
                                .iter()
                                .zip(scaled.iter())
                                .map(|((name, abs_res), (_, scaled_res))| {
                                    format!("{name}={scaled_res:.3e} (abs={abs_res:.3e})")
                                })
                                .collect::<Vec<_>>()
                                .join(", ");
                            format!(" outer(iters={iters}, res=[{fields}]{status_suffix})")
                        } else if let Some(fields) = abs_residuals {
                            // Only absolute residuals available
                            let fields = fields
                                .iter()
                                .map(|(name, res)| format!("{name}={res:.3e}"))
                                .collect::<Vec<_>>()
                                .join(", ");
                            format!(" outer(iters={iters}, res=[{fields}]{status_suffix})")
                        } else if let (Some(res_u), Some(res_p)) =
                            (step_stats.outer_residual_u, step_stats.outer_residual_p)
                        {
                            format!(
                                " outer(iters={iters}, u={res_u:.3e}, p={res_p:.3e}{status_suffix})"
                            )
                        } else {
                            format!(" outer(iters={iters}{status_suffix})")
                        }
                    })
                    .unwrap_or_default();

                eprintln!(
                    "[cfd2][{}] step={} t={:.4e} dt={:.2e} max|u|={:.3e} p=[{:.3e},{:.3e}] solves={} last(iters={}, res={:.3e}, conv={}, div={}){} nonfinite(u={}, p={})",
                    model_id,
                    step_idx,
                    solver.time(),
                    solver.dt(),
                    fs.max_vel,
                    fs.p_min,
                    fs.p_max,
                    linear_solves,
                    linear_last.iterations,
                    linear_last.residual,
                    linear_last.converged,
                    linear_last.diverged,
                    outer_str,
                    fs.nonfinite_u,
                    fs.nonfinite_p,
                );
            }

            if fs.nonfinite_u > 0 || fs.nonfinite_p > 0 {
                running = false;
                let _ = evt_tx.send(SolverWorkerEvent::Error(format!(
                    "divergence detected (nonfinite u={}, p={})",
                    fs.nonfinite_u, fs.nonfinite_p
                )));
                let _ = evt_tx.send(SolverWorkerEvent::Running(false));
                continue;
            }

            let _ = evt_tx.send(SolverWorkerEvent::Snapshot {
                u: rb.u,
                p: rb.p,
                stats,
            });
        } else if step_idx == 0 || last_stats_publish.elapsed() >= stats_publish_interval {
            last_stats_publish = std::time::Instant::now();
            let _ = evt_tx.send(SolverWorkerEvent::Stats { stats });
        }

        if let Some(trace) = trace.as_mut() {
            let linear_solves = outcome
                .linear_stats
                .iter()
                .copied()
                .map(tracefmt::TraceLinearSolverStats::from)
                .collect::<Vec<_>>();

            let graph = solver
                .step_graph_timings()
                .iter()
                .map(|t| {
                    let nodes = t.detail.as_ref().and_then(|detail| match detail {
                        crate::solver::gpu::execution_plan::GraphDetail::Module(detail) => Some(
                            detail
                                .nodes
                                .iter()
                                .map(|n| tracefmt::TraceGraphNodeTiming {
                                    label: n.label.to_string(),
                                    seconds: n.seconds,
                                })
                                .collect::<Vec<_>>(),
                        ),
                    });
                    tracefmt::TraceGraphTiming {
                        label: t.label.to_string(),
                        seconds: t.seconds,
                        nodes,
                    }
                })
                .collect::<Vec<_>>();

            let event = tracefmt::TraceEvent::Step(tracefmt::TraceStepEvent {
                step: step_idx,
                sim_time: solver.time(),
                dt: solver.dt(),
                wall_time_ms: step_time_ms,
                linear_solves,
                graph,
                max_u: trace_max_u,
                p_min: trace_p_min,
                p_max: trace_p_max,
            });
            let _ = trace.writer.write_event(&event);
        }

        step_idx = step_idx.wrapping_add(1);
        thread::sleep(std::time::Duration::from_millis(1));
    }
}

fn solver_worker_handle_cmd(
    cmd: SolverWorkerCommand,
    driver: &mut Option<SolverDriver>,
    trace: &mut Option<SolverTraceSession>,
    model_id: &mut &'static str,
    viz_field: &mut Option<VizFieldBuffers>,
    params: &mut RuntimeParams,
    running: &mut bool,
    step_idx: &mut u64,
    last_stats_publish: &mut std::time::Instant,
    last_snapshot_publish: &mut std::time::Instant,
    evt_tx: &mpsc::Sender<SolverWorkerEvent>,
) -> bool {
    match cmd {
        SolverWorkerCommand::SetSolver {
            driver: next,
            viz_field: next_viz_field,
        } => {
            if trace.is_some() {
                solver_worker_stop_trace(trace, driver);
            }
            *model_id = next.solver().model().id;
            *driver = Some(next);
            *viz_field = next_viz_field;
            *running = false;
            *step_idx = 0;
            let now = std::time::Instant::now();
            *last_stats_publish = now;
            *last_snapshot_publish = now;
            let _ = evt_tx.send(SolverWorkerEvent::Running(false));

            // Phase-2 parameter application (was `solver_worker_apply_params`), in
            // the same order as before: build sets phase 1, this sets phase 2.
            if let Some(d) = driver.as_mut() {
                d.apply_params(params);
            }
        }
        SolverWorkerCommand::ClearSolver => {
            solver_worker_stop_trace(trace, driver);
            *driver = None;
            *model_id = "<uninitialized>";
            *running = false;
            *step_idx = 0;
            *viz_field = None;
            let now = std::time::Instant::now();
            *last_stats_publish = now;
            *last_snapshot_publish = now;
            let _ = evt_tx.send(SolverWorkerEvent::Running(false));
        }
        SolverWorkerCommand::SetRunning(next_running) => {
            if next_running && driver.is_none() {
                let _ = evt_tx.send(SolverWorkerEvent::Error(
                    "cannot start solver: no solver is initialized".to_string(),
                ));
                let _ = evt_tx.send(SolverWorkerEvent::Running(false));
                *running = false;
                return true;
            }
            if next_running {
                *step_idx = 0;
                let now = std::time::Instant::now();
                *last_stats_publish = now;
                *last_snapshot_publish = now;
            }
            *running = next_running;
            let _ = evt_tx.send(SolverWorkerEvent::Running(*running));
        }
        SolverWorkerCommand::UpdateParams(next_params) => {
            *params = next_params;
            if let Some(d) = driver.as_mut() {
                d.apply_params(params);
            }
            if let (Some(trace), Some(d)) = (trace.as_mut(), driver.as_ref()) {
                let event = tracefmt::TraceEvent::Params(tracefmt::TraceParamsEvent {
                    step: *step_idx,
                    sim_time: d.solver().time(),
                    params: trace_runtime_params_from_worker(*params),
                });
                let _ = trace.writer.write_event(&event);
            }
        }
        SolverWorkerCommand::StartTrace { path, header } => {
            solver_worker_stop_trace(trace, driver);

            match tracefmt::TraceWriter::create(&path) {
                Ok(mut writer) => {
                    let profiling_enabled = header.ui.profiling_enabled;
                    let event = tracefmt::TraceEvent::Header(Box::new(header));
                    let _ = writer.write_event(&event);

                    if let Some(s) = driver.as_mut().map(|d| d.solver_mut()) {
                        s.set_collect_trace(true);
                        let _ = s.enable_detailed_profiling(profiling_enabled);
                        if profiling_enabled {
                            let _ = s.start_profiling_session();
                        }
                    }

                    *trace = Some(SolverTraceSession {
                        writer,
                        profiling_enabled,
                    });
                    eprintln!("[cfd2][trace] recording to {}", path);
                }
                Err(err) => {
                    let _ = evt_tx.send(SolverWorkerEvent::Message(format!(
                        "failed to start trace: {err}"
                    )));
                }
            }
        }
        SolverWorkerCommand::AppendTraceInit { events } => {
            if let Some(trace) = trace.as_mut() {
                for init in events {
                    let _ = trace.writer.write_event(&tracefmt::TraceEvent::Init(init));
                }
            }
        }
        SolverWorkerCommand::StopTrace => {
            solver_worker_stop_trace(trace, driver);
        }
        SolverWorkerCommand::Shutdown => return false,
    }
    true
}

fn get_color(t: f64) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.5 {
        (0.0, t * 2.0, 1.0 - t * 2.0)
    } else {
        ((t - 0.5) * 2.0, 1.0 - (t - 0.5) * 2.0, 0.0)
    };

    egui::Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}
