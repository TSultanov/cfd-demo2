use crate::solver::mesh::{
    generate_cut_cell_mesh, generate_delaunay_mesh, generate_voronoi_mesh, BackwardsStep,
    ChannelWithObstacle, Mesh,
};
use crate::solver::model::helpers::{
    SolverCompressibleIdealGasExt, SolverCompressibleInletExt, SolverFieldAliasesExt,
    SolverIncompressibleStatsExt, SolverInletVelocityExt, SolverRuntimeParamsExt,
};
use crate::solver::model::{
    all_models, compressible_model_with_eos, ModelPreconditionerSpec, ModelSpec,
};
use crate::solver::scheme::Scheme;
use crate::solver::{
    GpuLowMachPrecondModel, LinearSolverStats, PreconditionerType, SolverConfig, SteppingMode,
    TimeScheme as GpuTimeScheme, UnifiedSolver,
};
use crate::trace as tracefmt;
use crate::ui::{cfd_renderer, fluid::Fluid};
use eframe::egui;
use egui_plot::{Plot, PlotPoints, Polygon};
use nalgebra::{Point2, Vector2};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

#[derive(Clone, Copy)]
struct RuntimeParams {
    adaptive_dt: bool,
    target_cfl: f64,
    requested_dt: f32,
    dtau: f32,
    log_convergence: bool,
    log_every_steps: u32,
    advection_scheme: Scheme,
    time_scheme: GpuTimeScheme,
    preconditioner: PreconditionerType,
    outer_iters: u32,
    low_mach_model: GpuLowMachPrecondModel,
    low_mach_theta_floor: f32,
    low_mach_pressure_coupling_alpha: f32,
    alpha_u: f32,
    alpha_p: f32,
    inlet_velocity: f32,
    density: f32,
    viscosity: f32,
    eos: crate::solver::model::eos::EosSpec,
}

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
    timestep: f64,
    selected_scheme: Scheme,
    time_scheme: GpuTimeScheme,
    alpha_u: f64,
    alpha_p: f64,
    inlet_velocity: f32,
    selected_preconditioner: PreconditionerType,
    current_fluid: Fluid,
    wgpu_device: Option<wgpu::Device>,
    wgpu_queue: Option<wgpu::Queue>,
    target_format: wgpu::TextureFormat,
}

struct SolverInitOutcome {
    solver: UnifiedSolver,
    mesh: Mesh,
    cached_cells: Vec<Vec<[f64; 2]>>,
    actual_min_cell_size: f64,
    cached_u: Vec<(f64, f64)>,
    cached_p: Vec<f64>,
    model_caps: ModelUiCaps,
    renderer: Option<cfd_renderer::CfdRenderResources>,
    viz_field: Option<VizFieldBuffer>,
    trace_init_events: Vec<tracefmt::TraceInitEvent>,
}

#[derive(Clone)]
struct VizFieldBuffer {
    buffer: wgpu::Buffer,
    size_bytes: u64,
}

struct SolverInitResponse {
    generation: u64,
    result: Result<SolverInitOutcome, String>,
}

// Cached GPU solver stats for UI display (avoids lock contention)
#[allow(dead_code)]
#[derive(Default, Clone)]
struct CachedGpuStats {
    time: f32,
    dt: f32,
    linear_solves: u32,
    linear_last: LinearSolverStats,
    stats_ux: LinearSolverStats,
    stats_uy: LinearSolverStats,
    stats_p: LinearSolverStats,
    outer_residual_u: f32,
    outer_residual_p: f32,
    outer_iterations: u32,
    step_time_ms: f32,
}

enum SolverWorkerCommand {
    SetSolver {
        solver: UnifiedSolver,
        min_cell_size: f64,
        viz_field: Option<VizFieldBuffer>,
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
    Stats { stats: CachedGpuStats },
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
    low_mach_model: GpuLowMachPrecondModel,
    low_mach_theta_floor: f32,
    low_mach_pressure_coupling_alpha: f32,
    inlet_velocity: f32,
    selected_preconditioner: PreconditionerType,
    model_id: &'static str,
    model_caps: ModelUiCaps,
    wgpu_device: Option<wgpu::Device>,
    wgpu_queue: Option<wgpu::Queue>,
    target_format: wgpu::TextureFormat,
    cfd_renderer: Option<Arc<Mutex<cfd_renderer::CfdRenderResources>>>,
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
            timestep: 0.001,
            selected_geometry: GeometryType::default(),
            mesh_type: MeshType::default(),
            plot_field: PlotField::VelocityMag,
            is_running: false,
            selected_scheme: Scheme::Upwind,
            current_fluid: default_fluid,
            show_mesh_lines: true,
            adaptive_dt: true,
            target_cfl: 0.95,
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
            time_scheme: GpuTimeScheme::Euler,
            outer_iters: 1,
            low_mach_model: GpuLowMachPrecondModel::Off,
            low_mach_theta_floor: 1e-6,
            low_mach_pressure_coupling_alpha: 1.0,
            inlet_velocity: 1.0,
            selected_preconditioner: PreconditionerType::Jacobi,
            model_id: "incompressible_momentum",
            model_caps: ModelUiCaps::default(),
            wgpu_device,
            wgpu_queue,
            target_format,
            cfd_renderer: None,
        };
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
            low_mach_model: self.low_mach_model,
            low_mach_theta_floor: self.low_mach_theta_floor,
            low_mach_pressure_coupling_alpha: self.low_mach_pressure_coupling_alpha,
            alpha_u: self.alpha_u as f32,
            alpha_p: self.alpha_p as f32,
            inlet_velocity: self.inlet_velocity,
            density: self.current_fluid.density as f32,
            viscosity: self.current_fluid.viscosity as f32,
            eos: self.current_fluid.eos,
        }
    }

    fn sync_worker_params(&self) {
        self.solver_worker.send(SolverWorkerCommand::UpdateParams(
            self.current_runtime_params(),
        ));
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
        };
        let mesh_type = match self.mesh_type {
            MeshType::CutCell => tracefmt::TraceMeshType::CutCell,
            MeshType::Delaunay => tracefmt::TraceMeshType::Delaunay,
            MeshType::Voronoi => tracefmt::TraceMeshType::Voronoi,
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
            self.solver_worker.send(SolverWorkerCommand::AppendTraceInit {
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
            other => other,
        }
    }

    fn supported_ui_models() -> Vec<(&'static str, &'static str)> {
        let mut out = Vec::new();
        for model in all_models() {
            let layout = &model.state_layout;
            let has_u = layout
                .offset_for("U")
                .or_else(|| layout.offset_for("u"))
                .is_some();
            let has_p = layout.offset_for("p").is_some();
            if !has_u || !has_p {
                continue;
            }
            out.push((model.id, CFDApp::model_label(model.id)));
        }
        out.sort_by_key(|(id, _)| *id);
        out
    }

    fn build_selected_model(&self) -> Result<ModelSpec, String> {
        if self.model_id == "compressible" {
            return Ok(compressible_model_with_eos(self.current_fluid.eos));
        }
        all_models()
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
                    .expect("default UI model must exist")
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

        let layout = model.state_layout;
        self.model_caps.plot_stride = layout.stride();
        let u_off = layout.offset_for("U").or_else(|| layout.offset_for("u"));
        self.model_caps.plot_has_u = u_off.is_some();
        self.model_caps.plot_u_offset = u_off.unwrap_or(0);
        let p_off = layout.offset_for("p");
        self.model_caps.plot_has_p = p_off.is_some();
        self.model_caps.plot_p_offset = p_off.unwrap_or(0);

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
            timestep: self.timestep,
            selected_scheme: self.selected_scheme,
            time_scheme: self.time_scheme,
            alpha_u: self.alpha_u,
            alpha_p: self.alpha_p,
            inlet_velocity: self.inlet_velocity,
            selected_preconditioner: self.selected_preconditioner,
            current_fluid: self.current_fluid.clone(),
            wgpu_device: self.wgpu_device.clone(),
            wgpu_queue: self.wgpu_queue.clone(),
            target_format: self.target_format,
        }
    }

    fn update_renderer_field(&self) {
        // No-op: the renderer bind group is created at solver init time and points at the
        // visualization buffer. Field selection is driven by uniforms (stride/offset/mode).
    }

    fn init_solver(&mut self) {
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
        fn mesh_type_id(mesh_type: MeshType) -> &'static str {
            match mesh_type {
                MeshType::CutCell => "cutcell",
                MeshType::Delaunay => "delaunay",
                MeshType::Voronoi => "voronoi",
            }
        }

        fn geometry_id(geometry: GeometryType) -> &'static str {
            match geometry {
                GeometryType::BackwardsStep => "backwards_step",
                GeometryType::ChannelObstacle => "channel_obstacle",
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
                let mut mesh = match mesh_type {
                    MeshType::CutCell => generate_cut_cell_mesh(
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

                let smooth_start = std::time::Instant::now();
                mesh.smooth(&geo, 0.3, 50);
                CFDApp::push_trace_init_event(
                    trace_init_events,
                    format!("mesh.smooth.{geometry}.{mesh_kind}"),
                    smooth_start.elapsed(),
                    Some("factor=0.3 iters=50".to_string()),
                );

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
                let mut mesh = match mesh_type {
                    MeshType::CutCell => generate_cut_cell_mesh(
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

                let smooth_iters = match mesh_type {
                    MeshType::CutCell => 100,
                    MeshType::Delaunay | MeshType::Voronoi => 50,
                };

                let smooth_start = std::time::Instant::now();
                mesh.smooth(&geo, 0.3, smooth_iters);
                CFDApp::push_trace_init_event(
                    trace_init_events,
                    format!("mesh.smooth.{geometry}.{mesh_kind}"),
                    smooth_start.elapsed(),
                    Some(format!("factor=0.3 iters={smooth_iters}")),
                );

                mesh
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
    ) -> Vec<(f64, f64)> {
        let mut u = vec![(0.0, 0.0); mesh.num_cells()];
        for (i, _vel) in u.iter_mut().enumerate() {
            let cx = mesh.cell_cx[i];
            let cy = mesh.cell_cy[i];

            // // Add small perturbation to break symmetry
            // let perturbation = (rng.gen::<f64>() - 0.5) * 0.01;
            // vel.1 += perturbation;

            if cx < max_cell_size {
                match selected_geometry {
                    GeometryType::BackwardsStep => {
                        if cy > 0.5 {
                            // *vel = (1.0, 0.0); // Removed to match shader ramp
                        }
                    }
                    GeometryType::ChannelObstacle => {
                        // *vel = (1.0, 0.0);
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

        let request = self.pending_init_request.take().unwrap();
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
    }

    fn apply_init_outcome(&mut self, outcome: SolverInitOutcome) {
        let SolverInitOutcome {
            solver,
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
        self.last_init_trace_events = trace_init_events;

        self.cached_gpu_stats = CachedGpuStats::default();
        self.cached_error = None;
        self.cached_message = None;

        self.solver_worker.send(SolverWorkerCommand::SetSolver {
            solver,
            min_cell_size: self.actual_min_cell_size,
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
        );
        let initial_p = vec![0.0; n_cells];
        CFDApp::push_trace_init_event(
            &mut trace_init_events,
            "init.fields",
            fields_start.elapsed(),
            Some(format!("cells={n_cells}")),
        );

        let model = if request.model_id == "compressible" {
            compressible_model_with_eos(request.current_fluid.eos)
        } else {
            all_models()
                .into_iter()
                .find(|m| m.id == request.model_id)
                .ok_or_else(|| format!("unknown model id '{}'", request.model_id))?
        };

        const UI_MAX_BLOCK_JACOBI: u32 = 16;
        let named_params = model.named_param_keys();
        let supports_preconditioner = named_params.iter().any(|&k| k == "preconditioner");
        let mut selected_preconditioner = request.selected_preconditioner;
        if matches!(selected_preconditioner, PreconditionerType::BlockJacobi)
            && model.system.unknowns_per_cell() > UI_MAX_BLOCK_JACOBI
        {
            selected_preconditioner = PreconditionerType::Jacobi;
        }
        let effective_preconditioner = if supports_preconditioner {
            selected_preconditioner
        } else {
            PreconditionerType::Jacobi
        };

        let layout = &model.state_layout;
        let u_off = layout.offset_for("U").or_else(|| layout.offset_for("u"));
        let p_off = layout.offset_for("p");

        let model_caps = ModelUiCaps {
            plot_stride: layout.stride(),
            plot_u_offset: u_off.unwrap_or(0),
            plot_p_offset: p_off.unwrap_or(0),
            plot_has_u: u_off.is_some(),
            plot_has_p: p_off.is_some(),
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

        let config = SolverConfig {
            advection_scheme: request.selected_scheme,
            time_scheme: request.time_scheme,
            preconditioner: effective_preconditioner,
            stepping: if model_caps.supports_eos_tuning {
                SteppingMode::Implicit { outer_iters: 1 }
            } else {
                SteppingMode::Coupled
            },
        };

        let solver_start = std::time::Instant::now();
        let init_guard = tracefmt::install_init_collector(&mut trace_init_events);
        let mut gpu_solver = pollster::block_on(UnifiedSolver::new(
            &mesh,
            model,
            config,
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

        let solver_setup_start = std::time::Instant::now();
        let stride = gpu_solver.model().state_layout.stride() as usize;
        let _ = gpu_solver.write_state_f32(&vec![0.0f32; n_cells * stride]);
        gpu_solver.set_dt(request.timestep as f32);
        let _ = gpu_solver.set_viscosity(request.current_fluid.viscosity as f32);
        gpu_solver.set_advection_scheme(request.selected_scheme);
        gpu_solver.set_time_scheme(request.time_scheme);
        let _ = gpu_solver.set_eos(&request.current_fluid.eos);
        if supports_preconditioner {
            gpu_solver.set_preconditioner(effective_preconditioner);
        }
        CFDApp::push_trace_init_event(
            &mut trace_init_events,
            "solver.setup",
            solver_setup_start.elapsed(),
            None,
        );

        let model = gpu_solver.model();
        let mut has_rho = false;
        let mut has_rho_u = false;
        let mut has_rho_e = false;
        let mut has_u = false;
        for eqn in model.system.equations() {
            match eqn.target().name() {
                "rho" => has_rho = true,
                "rho_u" => has_rho_u = true,
                "rho_e" => has_rho_e = true,
                "u" => has_u = true,
                _ => {}
            }
        }

        let initial_state_start = std::time::Instant::now();
        let (cached_u, cached_p) = if has_rho && has_rho_u && has_rho_e && has_u {
            let p_ref = request
                .current_fluid
                .pressure_for_density(request.current_fluid.density);
            let _ = gpu_solver.set_density(request.current_fluid.density as f32);
            let _ = gpu_solver.set_compressible_inlet_isothermal_x(
                request.current_fluid.density as f32,
                request.inlet_velocity,
                &request.current_fluid.eos,
            );
            gpu_solver.set_uniform_state(
                request.current_fluid.density as f32,
                [0.0, 0.0],
                p_ref as f32,
            );
            (vec![(0.0, 0.0); n_cells], vec![p_ref; n_cells])
        } else {
            let _ = gpu_solver.set_density(request.current_fluid.density as f32);
            let _ = gpu_solver.set_alpha_u(request.alpha_u as f32);
            let _ = gpu_solver.set_alpha_p(request.alpha_p as f32);
            let _ = gpu_solver.set_inlet_velocity(request.inlet_velocity);
            gpu_solver.set_u(&initial_u);
            gpu_solver.set_p(&initial_p);
            (initial_u, initial_p)
        };
        CFDApp::push_trace_init_event(
            &mut trace_init_events,
            "solver.initial_state",
            initial_state_start.elapsed(),
            Some(if has_rho && has_rho_u && has_rho_e && has_u {
                "compressible".to_string()
            } else {
                "incompressible".to_string()
            }),
        );

        let history_start = std::time::Instant::now();
        gpu_solver.initialize_history();
        CFDApp::push_trace_init_event(
            &mut trace_init_events,
            "solver.initialize_history",
            history_start.elapsed(),
            None,
        );

        let cache_start = std::time::Instant::now();
        let cached_cells = CFDApp::cache_cells(&mesh);
        CFDApp::push_trace_init_event(
            &mut trace_init_events,
            "mesh.cache_cells",
            cache_start.elapsed(),
            Some(format!("cells={}", mesh.num_cells())),
        );

        let renderer_start = std::time::Instant::now();
        let (renderer, viz_field) =
            if let (Some(device), Some(queue)) = (&request.wgpu_device, &request.wgpu_queue) {
                let state_size_bytes =
                    mesh.num_cells() as u64 * model_caps.plot_stride as u64 * 4;
                let viz_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("CFD Viz Field Buffer"),
                    size: state_size_bytes.max(4),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                if state_size_bytes > 0 {
                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("cfd_viz:init_copy_state"),
                        });
                    encoder.copy_buffer_to_buffer(
                        gpu_solver.state_buffer(),
                        0,
                        &viz_buffer,
                        0,
                        state_size_bytes,
                    );
                    queue.submit(Some(encoder.finish()));
                }

                let mut renderer = cfd_renderer::CfdRenderResources::new(
                    device,
                    request.target_format,
                    mesh.num_cells() * 10,
                );
                let vertices = cfd_renderer::build_mesh_vertices(&cached_cells);
                let line_vertices = cfd_renderer::build_line_vertices(&cached_cells);
                renderer.update_mesh(queue, &vertices, &line_vertices);
                renderer.update_bind_group(device, &viz_buffer);
                (
                    Some(renderer),
                    Some(VizFieldBuffer {
                        buffer: viz_buffer,
                        size_bytes: state_size_bytes,
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
            solver: gpu_solver,
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

    fn update_gpu_fluid(&self) {
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
                        ui.radio_value(
                            &mut self.selected_geometry,
                            GeometryType::BackwardsStep,
                            "Backwards Step",
                        );
                        ui.radio_value(
                            &mut self.selected_geometry,
                            GeometryType::ChannelObstacle,
                            "Channel w/ Obstacle",
                        );
                    });

                    ui.group(|ui| {
                        ui.label("Mesh Parameters");
                        ui.add(
                            egui::Slider::new(&mut self.min_cell_size, 0.001..=self.max_cell_size)
                                .text("Min Cell Size"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.max_cell_size, self.min_cell_size..=0.5)
                                .text("Max Cell Size"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.growth_rate, 1.0..=2.0).text("Growth Rate"),
                        );
                        ui.separator();
                        ui.label("Mesh Type");
                        ui.radio_value(&mut self.mesh_type, MeshType::CutCell, "CutCell");
                        ui.radio_value(&mut self.mesh_type, MeshType::Delaunay, "Delaunay");
                        ui.radio_value(&mut self.mesh_type, MeshType::Voronoi, "Voronoi");
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
                                egui::Slider::new(&mut density, 0.1..=20000.0)
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
                                egui::Slider::new(&mut viscosity, 1e-6..=0.1)
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
                                egui::Slider::new(&mut self.inlet_velocity, 0.0..=10.0)
                                    .text("Inlet Velocity (m/s)"),
                            )
                            .changed()
                        {
                            self.update_gpu_inlet_velocity();
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
                                egui::Slider::new(&mut self.timestep, 0.0001..=0.1)
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
                                    egui::Slider::new(&mut self.target_cfl, 0.1..=1.0)
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
                                        egui::Slider::new(&mut self.outer_iters, 1..=100)
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
                                            self.alpha_u = 0.2;
                                            self.update_gpu_alpha_u();
                                        }
                                    }
                                }
                                ui.add_enabled_ui(self.dual_time, |ui| {
                                    if ui
                                        .add(
                                            egui::Slider::new(&mut self.dtau, 1e-8..=0.1)
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
                                            egui::Slider::new(
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
                                            egui::Slider::new(
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
                                    egui::Slider::new(&mut self.log_every_steps, 1..=200)
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
                                "Tip: for compressible dual-time, start with α_U≈0.2; α_P is fixed to 1.0 for stability."
                            } else {
                                "Tip: start with α_U≈0.7 and α_P≈0.3; α=1 can diverge at high Re."
                            });
                            if self.model_caps.supports_alpha_u
                                && ui
                                    .add(
                                        egui::Slider::new(&mut self.alpha_u, 0.1..=1.0)
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
                                        egui::Slider::new(&mut self.alpha_p, 0.1..=1.0)
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
                            self.refresh_model_caps();
                            if self.model_id == "compressible"
                                && (self.alpha_u - 0.7).abs() < 1e-12
                                && (self.alpha_p - 0.3).abs() < 1e-12
                            {
                                // Dual-time stepping for the compressible solver is sensitive
                                // to update damping. Prefer conservative defaults (pressure
                                // stays unrelaxed; all other coupled unknowns use α_U).
                                self.alpha_u = 0.2;
                                self.alpha_p = 1.0;
                            }
                            self.init_solver();
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
                            ui.label(format!(
                                "Coupled: {} iters, U:{:.2e} P:{:.2e}",
                                stats.outer_iterations,
                                stats.outer_residual_u,
                                stats.outer_residual_p,
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
            .map(|cache| (cache.min, cache.max))
            .unwrap_or((0.0, 1.0));

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
                            let min_val = min_val as f32;
                            let max_val = max_val as f32;
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
                                let t = (val - min_val) / (max_val - min_val);
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

fn solver_worker_apply_params(solver: &mut UnifiedSolver, params: RuntimeParams) {
    solver.set_collect_convergence_stats(params.log_convergence);
    solver.set_dt(params.requested_dt);
    let named_params = solver.model().named_param_keys();
    let has_param = |key: &str| named_params.iter().any(|&k| k == key);

    if has_param("dtau") {
        let _ = solver.set_dtau(params.dtau);
    }
    if has_param("density") {
        let _ = solver.set_density(params.density);
    }
    if has_param("viscosity") {
        let _ = solver.set_viscosity(params.viscosity);
    }
    if has_param("eos.gamma") {
        let _ = solver.set_eos(&params.eos);
    }
    if has_param("outer_iters") {
        let _ = solver.set_outer_iters(params.outer_iters as usize);
    }
    if has_param("low_mach.model") {
        let _ = solver.set_precond_model(params.low_mach_model);
    }
    if has_param("low_mach.theta_floor") {
        let _ = solver.set_precond_theta_floor(params.low_mach_theta_floor);
    }
    if has_param("low_mach.pressure_coupling_alpha") {
        let _ = solver.set_precond_pressure_coupling_alpha(
            params.low_mach_pressure_coupling_alpha,
        );
    }

    if has_param("advection_scheme") {
        solver.set_advection_scheme(params.advection_scheme);
    }
    if has_param("time_scheme") {
        solver.set_time_scheme(params.time_scheme);
    }
    if has_param("preconditioner") {
        solver.set_preconditioner(params.preconditioner);
    }

    if has_param("alpha_u") {
        let _ = solver.set_alpha_u(params.alpha_u);
    }
    if has_param("alpha_p") {
        let _ = solver.set_alpha_p(params.alpha_p);
    }

    let model = solver.model();
    let mut has_rho = false;
    let mut has_rho_u = false;
    let mut has_rho_e = false;
    let mut has_u = false;
    for eqn in model.system.equations() {
        match eqn.target().name() {
            "rho" => has_rho = true,
            "rho_u" => has_rho_u = true,
            "rho_e" => has_rho_e = true,
            "u" => has_u = true,
            _ => {}
        }
    }

    if has_rho && has_rho_u && has_rho_e && has_u {
        let _ = solver.set_compressible_inlet_isothermal_x(
            params.density,
            params.inlet_velocity,
            &params.eos,
        );
    } else {
        let _ = solver.set_inlet_velocity(params.inlet_velocity);
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

fn solver_worker_stop_trace(trace: &mut Option<SolverTraceSession>, solver: &mut Option<UnifiedSolver>) {
    let Some(mut session) = trace.take() else {
        return;
    };

    if let Some(s) = solver.as_mut() {
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

    let _ = session.writer.write_event(&tracefmt::TraceEvent::Footer(tracefmt::TraceFooter {
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
    let mut solver: Option<UnifiedSolver> = None;
    let mut trace: Option<SolverTraceSession> = None;
    let mut model_id: &'static str = "<uninitialized>";
    let mut supports_sound_speed = false;
    let mut min_cell_size = 0.0_f64;
    let mut viz_field: Option<VizFieldBuffer> = None;
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
        low_mach_model: GpuLowMachPrecondModel::Off,
        low_mach_theta_floor: 1e-6,
        low_mach_pressure_coupling_alpha: 1.0,
        alpha_u: 1.0,
        alpha_p: 1.0,
        inlet_velocity: 0.0,
        density: 1.0,
        viscosity: 0.0,
        eos: crate::solver::model::eos::EosSpec::Constant,
    };

    let mut running = false;
    let mut step_idx: u64 = 0;
    let mut prev_max_vel = 0.0_f64;
    let mut last_stats_publish = std::time::Instant::now();
    let mut last_snapshot_publish = std::time::Instant::now();
    let stats_publish_interval = std::time::Duration::from_millis(33);
    let snapshot_publish_interval = std::time::Duration::from_millis(100);

    loop {
        while let Ok(cmd) = cmd_rx.try_recv() {
            if !solver_worker_handle_cmd(
                cmd,
                &mut solver,
                &mut trace,
                &mut model_id,
                &mut supports_sound_speed,
                &mut min_cell_size,
                &mut viz_field,
                &mut params,
                &mut running,
                &mut step_idx,
                &mut prev_max_vel,
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
                        &mut solver,
                        &mut trace,
                        &mut model_id,
                        &mut supports_sound_speed,
                        &mut min_cell_size,
                        &mut viz_field,
                        &mut params,
                        &mut running,
                        &mut step_idx,
                        &mut prev_max_vel,
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

        let Some(solver) = solver.as_mut() else {
            running = false;
            let _ = evt_tx.send(SolverWorkerEvent::Error(
                "solver worker entered running state without an initialized solver".to_string(),
            ));
            let _ = evt_tx.send(SolverWorkerEvent::Running(false));
            continue;
        };

        if params.adaptive_dt {
            let sound_speed = if supports_sound_speed {
                params.eos.sound_speed(params.density as f64)
            } else {
                0.0
            };
            let adv_speed = prev_max_vel.max(params.inlet_velocity.abs() as f64);
            let effective_sound_speed = match params.low_mach_model {
                GpuLowMachPrecondModel::Off => sound_speed,
                GpuLowMachPrecondModel::Legacy => sound_speed.min(adv_speed),
                GpuLowMachPrecondModel::WeissSmith => {
                    let theta = (params.low_mach_theta_floor as f64).max(0.0);
                    let c_floor = sound_speed * theta.sqrt();
                    sound_speed.min(adv_speed.max(c_floor))
                }
            };
            // Even in dual-time mode, keep the physical timestep tied to the wave speed of the
            // *preconditioned* system. Low-Mach preconditioning reduces `effective_sound_speed`
            // so low-Mach flows can take acoustic CFL >> 1 without making `dt` so large that the
            // pseudo-time iterations destabilize.
            let wave_speed = adv_speed + effective_sound_speed;
            if min_cell_size > 1e-12 && wave_speed.is_finite() && wave_speed > 1e-12 {
                let current_dt = solver.dt() as f64;
                let mut next_dt = params.target_cfl * min_cell_size / wave_speed;
                if next_dt > current_dt * 1.2 {
                    next_dt = current_dt * 1.2;
                }
                next_dt = next_dt.clamp(1e-9, 100.0);
                solver.set_dt(next_dt as f32);
            }
        } else {
            solver.set_dt(params.requested_dt);
        }

        let start = std::time::Instant::now();
        let step_linear_stats = match solver.step_with_stats() {
            Ok(stats) => stats,
            Err(err) => {
                running = false;
                let _ = evt_tx.send(SolverWorkerEvent::Error(format!(
                    "solver step failed: {err}"
                )));
                let _ = evt_tx.send(SolverWorkerEvent::Running(false));
                continue;
            }
        };
        let step_time_ms = start.elapsed().as_secs_f32() * 1000.0;

        if let Some(viz_field) = viz_field.as_ref() {
            if viz_field.size_bytes > 0 {
                solver.copy_state_to_buffer(&viz_field.buffer);
            }
        }

        if solver.incompressible_should_stop() {
            running = false;
            let _ = evt_tx.send(SolverWorkerEvent::Running(false));
            continue;
        }

        let linear_solves = step_linear_stats.len() as u32;
        let linear_last = step_linear_stats.last().copied().unwrap_or_default();
        let linear_diverged = step_linear_stats
            .iter()
            .any(|s| s.diverged || !s.residual.is_finite() || s.residual > 1e12);

        if linear_diverged {
            running = false;
            let _ = evt_tx.send(SolverWorkerEvent::Error(
                "divergence detected (linear solver)".to_string(),
            ));
            let _ = evt_tx.send(SolverWorkerEvent::Running(false));
            continue;
        }

        let mut stats = CachedGpuStats {
            time: solver.time(),
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
        if step_linear_stats.len() == 3 {
            stats.stats_ux = step_linear_stats[0];
            stats.stats_uy = step_linear_stats[1];
            stats.stats_p = step_linear_stats[2];
        }

        let log_every_steps = params.log_every_steps.max(1) as u64;
        let should_log = params.log_convergence && (step_idx % log_every_steps == 0);

        let should_readback = step_idx == 0
            || last_snapshot_publish.elapsed() >= snapshot_publish_interval
            || should_log;

        let mut trace_max_u: Option<f64> = None;
        let mut trace_p_min: Option<f64> = None;
        let mut trace_p_max: Option<f64> = None;

        if should_readback {
            let now = std::time::Instant::now();
            last_snapshot_publish = now;
            last_stats_publish = now;
            let u = pollster::block_on(solver.get_u());
            let p = pollster::block_on(solver.get_p());

            let mut max_vel = 0.0f64;
            let mut nonfinite_u = 0usize;
            for (vx, vy) in &u {
                if !(vx.is_finite() && vy.is_finite()) {
                    nonfinite_u += 1;
                    continue;
                }
                let v = (vx.powi(2) + vy.powi(2)).sqrt();
                if v > max_vel {
                    max_vel = v;
                }
            }
            prev_max_vel = max_vel;
            trace_max_u = Some(max_vel);

            let mut min_p = f64::INFINITY;
            let mut max_p = f64::NEG_INFINITY;
            let mut nonfinite_p = 0usize;
            for &pv in &p {
                if !pv.is_finite() {
                    nonfinite_p += 1;
                    continue;
                }
                min_p = min_p.min(pv);
                max_p = max_p.max(pv);
            }
            trace_p_min = Some(min_p);
            trace_p_max = Some(max_p);

            if should_log || nonfinite_u > 0 || nonfinite_p > 0 {
                let step_stats = solver.step_stats();
                let outer_str = step_stats
                    .outer_iterations
                    .map(|iters| {
                        if let Some(fields) = solver.outer_field_residuals() {
                            let fields = fields
                                .iter()
                                .map(|(name, res)| format!("{name}={res:.3e}"))
                                .collect::<Vec<_>>()
                                .join(", ");
                            format!(" outer(iters={iters}, res=[{fields}])")
                        } else if let (Some(res_u), Some(res_p)) =
                            (step_stats.outer_residual_u, step_stats.outer_residual_p)
                        {
                            format!(" outer(iters={iters}, u={res_u:.3e}, p={res_p:.3e})")
                        } else {
                            format!(" outer(iters={iters})")
                        }
                    })
                    .unwrap_or_default();

                eprintln!(
                    "[cfd2][{}] step={} t={:.4e} dt={:.2e} max|u|={:.3e} p=[{:.3e},{:.3e}] solves={} last(iters={}, res={:.3e}, conv={}, div={}){} nonfinite(u={}, p={})",
                    model_id,
                    step_idx,
                    solver.time(),
                    solver.dt(),
                    max_vel,
                    min_p,
                    max_p,
                    linear_solves,
                    linear_last.iterations,
                    linear_last.residual,
                    linear_last.converged,
                    linear_last.diverged,
                    outer_str,
                    nonfinite_u,
                    nonfinite_p,
                );
            }

            if nonfinite_u > 0 || nonfinite_p > 0 {
                running = false;
                let _ = evt_tx.send(SolverWorkerEvent::Error(format!(
                    "divergence detected (nonfinite u={nonfinite_u}, p={nonfinite_p})"
                )));
                let _ = evt_tx.send(SolverWorkerEvent::Running(false));
                continue;
            }

            let _ = evt_tx.send(SolverWorkerEvent::Snapshot { u, p, stats });
        } else if step_idx == 0 || last_stats_publish.elapsed() >= stats_publish_interval {
            last_stats_publish = std::time::Instant::now();
            let _ = evt_tx.send(SolverWorkerEvent::Stats { stats });
        }

        if let Some(trace) = trace.as_mut() {
            let linear_solves = step_linear_stats
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
    solver: &mut Option<UnifiedSolver>,
    trace: &mut Option<SolverTraceSession>,
    model_id: &mut &'static str,
    supports_sound_speed: &mut bool,
    min_cell_size: &mut f64,
    viz_field: &mut Option<VizFieldBuffer>,
    params: &mut RuntimeParams,
    running: &mut bool,
    step_idx: &mut u64,
    prev_max_vel: &mut f64,
    last_stats_publish: &mut std::time::Instant,
    last_snapshot_publish: &mut std::time::Instant,
    evt_tx: &mpsc::Sender<SolverWorkerEvent>,
) -> bool {
    match cmd {
        SolverWorkerCommand::SetSolver {
            solver: next,
            min_cell_size: next_min_cell_size,
            viz_field: next_viz_field,
        } => {
            if trace.is_some() {
                solver_worker_stop_trace(trace, solver);
            }
            *model_id = next.model().id;
            *supports_sound_speed = next
                .model()
                .named_param_keys()
                .iter()
                .any(|&k| k == "eos.gamma");
            *solver = Some(next);
            *min_cell_size = next_min_cell_size;
            *viz_field = next_viz_field;
            *running = false;
            *step_idx = 0;
            *prev_max_vel = 0.0;
            let now = std::time::Instant::now();
            *last_stats_publish = now;
            *last_snapshot_publish = now;
            let _ = evt_tx.send(SolverWorkerEvent::Running(false));

            if let Some(s) = solver.as_mut() {
                solver_worker_apply_params(s, *params);
            }
        }
        SolverWorkerCommand::ClearSolver => {
            solver_worker_stop_trace(trace, solver);
            *solver = None;
            *model_id = "<uninitialized>";
            *supports_sound_speed = false;
            *running = false;
            *step_idx = 0;
            *prev_max_vel = 0.0;
            *viz_field = None;
            let now = std::time::Instant::now();
            *last_stats_publish = now;
            *last_snapshot_publish = now;
            let _ = evt_tx.send(SolverWorkerEvent::Running(false));
        }
        SolverWorkerCommand::SetRunning(next_running) => {
            if next_running && solver.is_none() {
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
            if let Some(s) = solver.as_mut() {
                solver_worker_apply_params(s, *params);
            }
            if let (Some(trace), Some(s)) = (trace.as_mut(), solver.as_ref()) {
                let event = tracefmt::TraceEvent::Params(tracefmt::TraceParamsEvent {
                    step: *step_idx,
                    sim_time: s.time(),
                    params: trace_runtime_params_from_worker(*params),
                });
                let _ = trace.writer.write_event(&event);
            }
        }
        SolverWorkerCommand::StartTrace { path, header } => {
            solver_worker_stop_trace(trace, solver);

            match tracefmt::TraceWriter::create(&path) {
                Ok(mut writer) => {
                    let profiling_enabled = header.ui.profiling_enabled;
                    let event = tracefmt::TraceEvent::Header(header);
                    let _ = writer.write_event(&event);

                    if let Some(s) = solver.as_mut() {
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
                    let _ = trace
                        .writer
                        .write_event(&tracefmt::TraceEvent::Init(init));
                }
            }
        }
        SolverWorkerCommand::StopTrace => {
            solver_worker_stop_trace(trace, solver);
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
