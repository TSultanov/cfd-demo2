use crate::solver::mesh::{generate_structured_backwards_step_mesh, Mesh};
#[cfg(feature = "meshgen")]
use crate::solver::mesh::{
    generate_cut_cell_mesh, generate_delaunay_mesh, generate_voronoi_mesh, BackwardsStep,
    ChannelWithObstacle,
};
use crate::solver::model::helpers::{
    SolverCompressibleIdealGasExt, SolverCompressibleInletExt, SolverFieldAliasesExt,
    SolverIncompressibleControlsExt, SolverIncompressibleStatsExt, SolverInletVelocityExt,
    SolverRuntimeParamsExt,
};
use crate::solver::model::{
    compressible_model, compressible_model_with_eos, incompressible_momentum_model,
    ModelPreconditionerSpec,
};
use crate::solver::scheme::Scheme;
use crate::solver::{
    LinearSolverStats, PreconditionerType, SolverConfig, SteppingMode, TimeScheme as GpuTimeScheme,
    UnifiedSolver,
};
use crate::ui::{cfd_renderer, fluid::Fluid};
use eframe::egui;
use egui_plot::{Plot, PlotPoints, Polygon};
#[cfg(feature = "meshgen")]
use nalgebra::{Point2, Vector2};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

#[derive(Clone, Copy)]
struct RuntimeParams {
    adaptive_dt: bool,
    target_cfl: f64,
    requested_dt: f32,
    log_convergence: bool,
    log_every_steps: u32,
    advection_scheme: Scheme,
    time_scheme: GpuTimeScheme,
    preconditioner: PreconditionerType,
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
    #[cfg(feature = "meshgen")]
    ChannelObstacle,
}

impl Default for GeometryType {
    fn default() -> Self {
        Self::BackwardsStep
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
enum MeshType {
    Structured,
    #[cfg(feature = "meshgen")]
    CutCell,
    #[cfg(feature = "meshgen")]
    Delaunay,
    #[cfg(feature = "meshgen")]
    Voronoi,
}

impl Default for MeshType {
    fn default() -> Self {
        #[cfg(feature = "meshgen")]
        {
            Self::CutCell
        }
        #[cfg(not(feature = "meshgen"))]
        {
            Self::Structured
        }
    }
}

#[derive(PartialEq, Clone, Copy)]
enum PlotField {
    Pressure,
    VelocityX,
    VelocityY,
    VelocityMag,
}

#[derive(PartialEq, Clone, Copy, Debug)]
enum SolverKind {
    Incompressible,
    Compressible,
}

#[derive(Default, Clone)]
struct ModelUiCaps {
    supports_preconditioner: bool,
    model_owns_preconditioner: bool,
    unknowns_per_cell: u32,
}

struct SolverInitRequest {
    generation: u64,
    solver_kind: SolverKind,
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
        solver_kind: SolverKind,
        min_cell_size: f64,
    },
    ClearSolver,
    SetRunning(bool),
    UpdateParams(RuntimeParams),
    Shutdown,
}

enum SolverWorkerEvent {
    Snapshot {
        u: Vec<(f64, f64)>,
        p: Vec<f64>,
        stats: CachedGpuStats,
    },
    Error(String),
    Running(bool),
    Cleared,
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
    init_waiting_for_worker_clear: bool,
    cached_u: Vec<(f64, f64)>,
    cached_p: Vec<f64>,
    cached_gpu_stats: CachedGpuStats,
    cached_error: Option<String>,
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
    log_convergence: bool,
    log_every_steps: u32,
    render_mode: RenderMode,
    alpha_u: f64,
    alpha_p: f64,
    time_scheme: GpuTimeScheme,
    inlet_velocity: f32,
    selected_preconditioner: PreconditionerType,
    solver_kind: SolverKind,
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
            init_waiting_for_worker_clear: false,
            cached_u: Vec::new(),
            cached_p: Vec::new(),
            cached_gpu_stats: CachedGpuStats::default(),
            cached_error: None,
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
            log_convergence: false,
            log_every_steps: 25,
            render_mode: RenderMode::GpuDirect,
            alpha_u: 0.7,
            alpha_p: 0.3,
            time_scheme: GpuTimeScheme::Euler,
            inlet_velocity: 1.0,
            selected_preconditioner: PreconditionerType::Jacobi,
            solver_kind: SolverKind::Incompressible,
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
            log_convergence: self.log_convergence,
            log_every_steps: self.log_every_steps.max(1),
            advection_scheme: self.selected_scheme,
            time_scheme: self.time_scheme,
            preconditioner: self.selected_preconditioner,
            alpha_u: self.alpha_u as f32,
            alpha_p: self.alpha_p as f32,
            inlet_velocity: self.inlet_velocity,
            density: self.current_fluid.density as f32,
            viscosity: self.current_fluid.viscosity as f32,
            eos: self.current_fluid.eos,
        }
    }

    fn sync_worker_params(&self) {
        self.solver_worker
            .send(SolverWorkerCommand::UpdateParams(self.current_runtime_params()));
    }

    fn refresh_model_caps(&mut self) {
        let model = match self.solver_kind {
            SolverKind::Incompressible => incompressible_momentum_model(),
            SolverKind::Compressible => compressible_model_with_eos(self.current_fluid.eos),
        };

        self.model_caps.supports_preconditioner = model
            .named_param_keys()
            .into_iter()
            .any(|k| k == "preconditioner");
        self.model_caps.model_owns_preconditioner = model
            .linear_solver
            .map(|s| matches!(s.preconditioner, ModelPreconditionerSpec::Schur { .. }))
            .unwrap_or(false);
        self.model_caps.unknowns_per_cell = model.system.unknowns_per_cell();

        const UI_MAX_BLOCK_JACOBI: u32 = 16;
        if matches!(self.selected_preconditioner, PreconditionerType::BlockJacobi)
            && self.model_caps.unknowns_per_cell > UI_MAX_BLOCK_JACOBI
        {
            self.selected_preconditioner = PreconditionerType::Jacobi;
        }
    }

    fn make_init_request(&mut self) -> SolverInitRequest {
        let generation = self.next_init_generation;
        self.next_init_generation = self.next_init_generation.wrapping_add(1);
        SolverInitRequest {
            generation,
            solver_kind: self.solver_kind,
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
        // No-op: the renderer bind group is created at solver init time and always points at the
        // solver's state buffer; field selection is driven by uniforms (stride/offset/mode).
    }

    fn init_solver(&mut self) {
        self.is_running = false;
        self.solver_worker.send(SolverWorkerCommand::SetRunning(false));
        self.solver_worker.send(SolverWorkerCommand::ClearSolver);
        self.refresh_model_caps();
        self.pending_init_request = Some(self.make_init_request());
        self.init_waiting_for_worker_clear = true;
        self.mesh = None;
        self.cached_cells.clear();
        self.cfd_renderer = None;
        self.cached_u.clear();
        self.cached_p.clear();
        self.cached_gpu_stats = CachedGpuStats::default();
        self.cached_error = None;
    }

    fn build_mesh_with(
        selected_geometry: GeometryType,
        mesh_type: MeshType,
        min_cell_size: f64,
        max_cell_size: f64,
        growth_rate: f64,
    ) -> Mesh {
        let _ = (max_cell_size, growth_rate);
        match selected_geometry {
            GeometryType::BackwardsStep => {
                let length = 3.5;
                let height_outlet = 1.0;
                let height_inlet = 0.5;
                let step_x = 0.5;

                match mesh_type {
                    MeshType::Structured => {
                        let target = min_cell_size.clamp(1e-6, 1e3);
                        let mut nx = ((length / target).round() as i64).max(1) as usize;
                        nx = ((nx + 6) / 7).max(1) * 7;
                        let mut ny =
                            ((height_outlet / target).round() as i64).max(1) as usize;
                        if ny % 2 != 0 {
                            ny += 1;
                        }
                        ny = ny.max(2);
                        generate_structured_backwards_step_mesh(
                            nx,
                            ny,
                            length,
                            height_outlet,
                            height_inlet,
                            step_x,
                        )
                    }
                    #[cfg(feature = "meshgen")]
                    MeshType::CutCell | MeshType::Delaunay | MeshType::Voronoi => {
                        let domain_size = Vector2::new(length, height_outlet);
                        let geo = BackwardsStep {
                            length,
                            height_inlet,
                            height_outlet,
                            step_x,
                        };
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
                            MeshType::Structured => unreachable!("handled above"),
                        };
                        mesh.smooth(&geo, 0.3, 50);
                        mesh
                    }
                }
            }
            #[cfg(feature = "meshgen")]
            GeometryType::ChannelObstacle => {
                let length = 3.0;
                let domain_size = Vector2::new(length, 1.0);
                let geo = ChannelWithObstacle {
                    length,
                    height: 1.0,
                    obstacle_center: Point2::new(1.0, 0.51), // Offset to trigger vortex shedding
                    obstacle_radius: 0.1,
                };
                let mesh_type = match mesh_type {
                    MeshType::Structured => MeshType::CutCell,
                    other => other,
                };
                let mesh = match mesh_type {
                    MeshType::CutCell => {
                        let mut mesh = generate_cut_cell_mesh(
                            &geo,
                            min_cell_size,
                            max_cell_size,
                            growth_rate,
                            domain_size,
                        );
                        mesh.smooth(&geo, 0.3, 100);
                        mesh
                    }
                    MeshType::Delaunay => {
                        let mut mesh = generate_delaunay_mesh(
                            &geo,
                            min_cell_size,
                            max_cell_size,
                            growth_rate,
                            domain_size,
                        );
                        mesh.smooth(&geo, 0.3, 50);
                        mesh
                    }
                    MeshType::Voronoi => {
                        let mut mesh = generate_voronoi_mesh(
                            &geo,
                            min_cell_size,
                            max_cell_size,
                            growth_rate,
                            domain_size,
                        );
                        mesh.smooth(&geo, 0.3, 50);
                        mesh
                    }
                    MeshType::Structured => unreachable!("mapped above"),
                };

                mesh
            }
        }
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
                    #[cfg(feature = "meshgen")]
                    GeometryType::ChannelObstacle => {
                        // *vel = (1.0, 0.0);
                    }
                }
            }
        }
        u
    }

    fn spawn_pending_init(&mut self) {
        if self.init_rx.is_some() {
            return;
        }
        if self.pending_init_request.is_none() {
            return;
        }
        if self.init_waiting_for_worker_clear {
            return;
        }

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
                SolverWorkerEvent::Snapshot { u, p, stats } => {
                    self.cached_u = u;
                    self.cached_p = p;
                    self.cached_gpu_stats = stats;
                    self.cached_error = None;
                }
                SolverWorkerEvent::Error(err) => {
                    self.cached_error = Some(err);
                    self.is_running = false;
                }
                SolverWorkerEvent::Running(running) => {
                    self.is_running = running;
                }
                SolverWorkerEvent::Cleared => {
                    self.init_waiting_for_worker_clear = false;
                }
            }
        }
    }

    fn apply_init_outcome(&mut self, outcome: SolverInitOutcome) {
        self.is_running = false;
        self.solver_worker.send(SolverWorkerCommand::SetRunning(false));

        self.model_caps = outcome.model_caps;
        self.cached_cells = outcome.cached_cells;
        self.actual_min_cell_size = outcome.actual_min_cell_size;
        self.cached_u = outcome.cached_u;
        self.cached_p = outcome.cached_p;
        self.mesh = Some(outcome.mesh);

        self.cfd_renderer = outcome
            .renderer
            .map(|r| Arc::new(Mutex::new(r)));

        self.cached_gpu_stats = CachedGpuStats::default();
        self.cached_error = None;

        self.solver_worker.send(SolverWorkerCommand::SetSolver {
            solver: outcome.solver,
            solver_kind: self.solver_kind,
            min_cell_size: self.actual_min_cell_size,
        });
        self.sync_worker_params();
    }

    fn build_init_outcome(request: SolverInitRequest) -> Result<SolverInitOutcome, String> {
        let mesh = CFDApp::build_mesh_with(
            request.selected_geometry,
            request.mesh_type,
            request.min_cell_size,
            request.max_cell_size,
            request.growth_rate,
        );
        let n_cells = mesh.num_cells();
        let initial_u = CFDApp::build_initial_velocity_with(
            &mesh,
            request.selected_geometry,
            request.max_cell_size,
        );
        let initial_p = vec![0.0; n_cells];

        let (model, initial_u, initial_p, cached_p) = match request.solver_kind {
            SolverKind::Incompressible => (
                incompressible_momentum_model(),
                Some(initial_u),
                Some(initial_p),
                None,
            ),
            SolverKind::Compressible => {
                let p_ref = request
                    .current_fluid
                    .pressure_for_density(request.current_fluid.density);
                (
                    compressible_model_with_eos(request.current_fluid.eos),
                    None,
                    None,
                    Some(p_ref),
                )
            }
        };

        const UI_MAX_BLOCK_JACOBI: u32 = 16;
        let supports_preconditioner = model
            .named_param_keys()
            .into_iter()
            .any(|k| k == "preconditioner");
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

        let model_caps = ModelUiCaps {
            supports_preconditioner,
            model_owns_preconditioner: model
                .linear_solver
                .map(|s| matches!(s.preconditioner, ModelPreconditionerSpec::Schur { .. }))
                .unwrap_or(false),
            unknowns_per_cell: model.system.unknowns_per_cell(),
        };

        let config = SolverConfig {
            advection_scheme: request.selected_scheme,
            time_scheme: request.time_scheme,
            preconditioner: effective_preconditioner,
            stepping: match request.solver_kind {
                SolverKind::Incompressible => SteppingMode::Coupled,
                SolverKind::Compressible => SteppingMode::Implicit { outer_iters: 1 },
            },
        };

        let mut gpu_solver = pollster::block_on(UnifiedSolver::new(
            &mesh,
            model,
            config,
            request.wgpu_device.clone(),
            request.wgpu_queue.clone(),
        ))?;
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

        let (cached_u, cached_p) = match request.solver_kind {
            SolverKind::Incompressible => {
                let _ = gpu_solver.set_density(request.current_fluid.density as f32);
                let _ = gpu_solver.set_alpha_u(request.alpha_u as f32);
                let _ = gpu_solver.set_alpha_p(request.alpha_p as f32);
                let _ = gpu_solver.set_inlet_velocity(request.inlet_velocity);
                gpu_solver.set_u(initial_u.as_ref().unwrap());
                gpu_solver.set_p(initial_p.as_ref().unwrap());
                (initial_u.unwrap(), initial_p.unwrap())
            }
            SolverKind::Compressible => {
                let p_ref = cached_p.unwrap();
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
            }
        };

        gpu_solver.initialize_history();

        let cached_cells = CFDApp::cache_cells(&mesh);

        let renderer = if let (Some(device), Some(queue)) = (&request.wgpu_device, &request.wgpu_queue)
        {
            let mut renderer = cfd_renderer::CfdRenderResources::new(
                device,
                request.target_format,
                mesh.num_cells() * 10,
            );
            let vertices = cfd_renderer::build_mesh_vertices(&cached_cells);
            let line_vertices = cfd_renderer::build_line_vertices(&cached_cells);
            renderer.update_mesh(queue, &vertices, &line_vertices);
            renderer.update_bind_group(device, gpu_solver.state_buffer());
            Some(renderer)
        } else {
            None
        };

        let actual_min_cell_size = mesh
            .cell_vol
            .iter()
            .map(|&v| v.sqrt())
            .fold(f64::INFINITY, f64::min);

        Ok(SolverInitOutcome {
            solver: gpu_solver,
            mesh,
            cached_cells,
            actual_min_cell_size,
            cached_u,
            cached_p,
            model_caps,
            renderer,
        })
    }

    fn render_layout_for_field(&self) -> (u32, u32, u32) {
        let (stride, u_offset, p_offset) = match self.solver_kind {
            SolverKind::Incompressible => {
                let model = incompressible_momentum_model();
                let layout = model.state_layout;
                let stride = layout.stride();
                let u_offset = layout.offset_for("U").unwrap_or(0);
                let p_offset = layout.offset_for("p").unwrap_or(0);
                (stride, u_offset, p_offset)
            }
            SolverKind::Compressible => {
                let model = compressible_model();
                let layout = model.state_layout;
                let stride = layout.stride();
                let u_offset = layout.offset_for("u").unwrap_or(0);
                let p_offset = layout.offset_for("p").unwrap_or(0);
                (stride, u_offset, p_offset)
            }
        };

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
                        #[cfg(feature = "meshgen")]
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
                        ui.radio_value(&mut self.mesh_type, MeshType::Structured, "Structured");
                        #[cfg(feature = "meshgen")]
                        ui.radio_value(&mut self.mesh_type, MeshType::CutCell, "CutCell");
                        #[cfg(feature = "meshgen")]
                        ui.radio_value(&mut self.mesh_type, MeshType::Delaunay, "Delaunay");
                        #[cfg(feature = "meshgen")]
                        ui.radio_value(&mut self.mesh_type, MeshType::Voronoi, "Voronoi");
                        #[cfg(not(feature = "meshgen"))]
                        ui.label("Enable `--features ui_meshgen` for meshgen options.");
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
                        let sound_speed = if matches!(self.solver_kind, SolverKind::Compressible) {
                            self.current_fluid.sound_speed()
                        } else {
                            0.0
                        };
                        let wave_speed = adv_speed + sound_speed;
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

                        if cfl > 1.0 {
                            ui.colored_label(
                                egui::Color32::RED,
                                format!("⚠ CFL≈{:.1} (>1, may be unstable!)", cfl),
                            );
                            ui.colored_label(
                                egui::Color32::YELLOW,
                                format!("Recommended dt ≤ {:.4}", recommended_dt),
                            );
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
                            .radio(matches!(self.selected_scheme, Scheme::QUICK), "QUICK")
                            .clicked()
                        {
                            self.selected_scheme = Scheme::QUICK;
                            self.update_gpu_scheme();
                        }

                        if matches!(self.solver_kind, SolverKind::Compressible) {
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
                        if !supports_preconditioner {
                            ui.weak("(not declared by model)");
                        }

                        ui.add_enabled_ui(supports_preconditioner, |ui| {
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
                        });

                        if matches!(self.solver_kind, SolverKind::Incompressible) {
                            ui.separator();
                            ui.label("Under-Relaxation Factors");
                            ui.weak("Tip: start with α_U≈0.7 and α_P≈0.3; α=1 can diverge at high Re.");
                            if ui
                                .add(
                                    egui::Slider::new(&mut self.alpha_u, 0.1..=1.0)
                                        .text("α_U (Velocity)"),
                                )
                                .changed()
                            {
                                self.update_gpu_alpha_u();
                            }
                            if ui
                                .add(
                                    egui::Slider::new(&mut self.alpha_p, 0.1..=1.0)
                                        .text("α_P (Pressure)"),
                                )
                                .changed()
                            {
                                self.update_gpu_alpha_p();
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
                        ui.label("Solver Type");
                        let prev_solver_kind = self.solver_kind;
                        egui::ComboBox::from_label("Solver")
                            .selected_text(format!("{:?}", self.solver_kind))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut self.solver_kind,
                                    SolverKind::Incompressible,
                                    "Incompressible (Coupled)",
                                );
                                ui.selectable_value(
                                    &mut self.solver_kind,
                                    SolverKind::Compressible,
                                    "Compressible (KT)",
                                );
                            });
                        if prev_solver_kind != self.solver_kind {
                            self.refresh_model_caps();
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
                        if matches!(self.solver_kind, SolverKind::Incompressible) {
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

        let (min_val, max_val, values) = if has_solver {
            let u = &self.cached_u;
            let p = &self.cached_p;

            if !u.is_empty() {
                let mut min_val = f64::MAX;
                let mut max_val = f64::MIN;
                let mut values = Vec::with_capacity(u.len());

                for i in 0..u.len() {
                    let val = match self.plot_field {
                        PlotField::Pressure => p[i],
                        PlotField::VelocityX => u[i].0,
                        PlotField::VelocityY => u[i].1,
                        PlotField::VelocityMag => (u[i].0.powi(2) + u[i].1.powi(2)).sqrt(),
                    };
                    min_val = min_val.min(val);
                    max_val = max_val.max(val);
                    values.push(val);
                }

                if (max_val - min_val).abs() < 1e-6 {
                    max_val = min_val + 1.0;
                }
                (min_val, max_val, Some(values))
            } else {
                (0.0, 1.0, None)
            }
        } else {
            (0.0, 1.0, None)
        };

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

            if let (Some(cells), Some(vals)) = (cells, &values) {
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
                        Plot::new("cfd_plot").data_aspect(1.0).show(ui, |plot_ui| {
                            for (i, polygon_points) in cells.iter().enumerate() {
                                let val = vals[i];
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

fn solver_worker_apply_params(solver: &mut UnifiedSolver, solver_kind: SolverKind, params: RuntimeParams) {
    solver.set_dt(params.requested_dt);
    let _ = solver.set_density(params.density);
    let _ = solver.set_viscosity(params.viscosity);
    let _ = solver.set_eos(&params.eos);
    solver.set_advection_scheme(params.advection_scheme);
    solver.set_time_scheme(params.time_scheme);
    solver.set_preconditioner(params.preconditioner);

    match solver_kind {
        SolverKind::Incompressible => {
            let _ = solver.set_alpha_u(params.alpha_u);
            let _ = solver.set_alpha_p(params.alpha_p);
            let _ = solver.set_inlet_velocity(params.inlet_velocity);
        }
        SolverKind::Compressible => {
            let _ = solver.set_compressible_inlet_isothermal_x(
                params.density,
                params.inlet_velocity,
                &params.eos,
            );
        }
    }
}

fn solver_worker_main(
    cmd_rx: mpsc::Receiver<SolverWorkerCommand>,
    evt_tx: mpsc::Sender<SolverWorkerEvent>,
) {
    let mut solver: Option<UnifiedSolver> = None;
    let mut solver_kind = SolverKind::Incompressible;
    let mut min_cell_size = 0.0_f64;
    let mut params = RuntimeParams {
        adaptive_dt: false,
        target_cfl: 0.9,
        requested_dt: 0.001,
        log_convergence: false,
        log_every_steps: 50,
        advection_scheme: Scheme::Upwind,
        time_scheme: GpuTimeScheme::Euler,
        preconditioner: PreconditionerType::Jacobi,
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
    let mut last_publish = std::time::Instant::now();
    let publish_interval = std::time::Duration::from_millis(16);

    loop {
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                SolverWorkerCommand::SetSolver {
                    solver: next,
                    solver_kind: next_kind,
                    min_cell_size: next_min_cell_size,
                } => {
                    solver = Some(next);
                    solver_kind = next_kind;
                    min_cell_size = next_min_cell_size;
                    running = false;
                    step_idx = 0;
                    prev_max_vel = 0.0;
                    last_publish = std::time::Instant::now();
                    let _ = evt_tx.send(SolverWorkerEvent::Running(false));

                    if let Some(s) = solver.as_mut() {
                        solver_worker_apply_params(s, solver_kind, params);
                    }
                }
                SolverWorkerCommand::ClearSolver => {
                    solver = None;
                    running = false;
                    step_idx = 0;
                    prev_max_vel = 0.0;
                    let _ = evt_tx.send(SolverWorkerEvent::Running(false));
                    let _ = evt_tx.send(SolverWorkerEvent::Cleared);
                }
                SolverWorkerCommand::SetRunning(next_running) => {
                    if next_running && solver.is_none() {
                        let _ = evt_tx.send(SolverWorkerEvent::Error(
                            "cannot start solver: no solver is initialized".to_string(),
                        ));
                        let _ = evt_tx.send(SolverWorkerEvent::Running(false));
                        running = false;
                        continue;
                    }
                    if next_running {
                        step_idx = 0;
                        last_publish = std::time::Instant::now();
                    }
                    running = next_running;
                    let _ = evt_tx.send(SolverWorkerEvent::Running(running));
                }
                SolverWorkerCommand::UpdateParams(next_params) => {
                    params = next_params;
                    if let Some(s) = solver.as_mut() {
                        solver_worker_apply_params(s, solver_kind, params);
                    }
                }
                SolverWorkerCommand::Shutdown => {
                    return;
                }
            }
        }

        if !running {
            match cmd_rx.recv_timeout(std::time::Duration::from_millis(16)) {
                Ok(cmd) => {
                    // Re-process command on next loop so try_recv drains any coalesced updates.
                    match cmd {
                        SolverWorkerCommand::Shutdown => return,
                        other => {
                            // Put it back via local handling.
                            match other {
                                SolverWorkerCommand::SetSolver {
                                    solver: next,
                                    solver_kind: next_kind,
                                    min_cell_size: next_min_cell_size,
                                } => {
                                    solver = Some(next);
                                    solver_kind = next_kind;
                                    min_cell_size = next_min_cell_size;
                                    running = false;
                                    step_idx = 0;
                                    prev_max_vel = 0.0;
                                    last_publish = std::time::Instant::now();
                                    let _ = evt_tx.send(SolverWorkerEvent::Running(false));

                                    if let Some(s) = solver.as_mut() {
                                        solver_worker_apply_params(s, solver_kind, params);
                                    }
                                }
                                SolverWorkerCommand::ClearSolver => {
                                    solver = None;
                                    running = false;
                                    step_idx = 0;
                                    prev_max_vel = 0.0;
                                    let _ = evt_tx.send(SolverWorkerEvent::Running(false));
                                    let _ = evt_tx.send(SolverWorkerEvent::Cleared);
                                }
                                SolverWorkerCommand::SetRunning(next_running) => {
                                    if next_running && solver.is_none() {
                                        let _ = evt_tx.send(SolverWorkerEvent::Error(
                                            "cannot start solver: no solver is initialized"
                                                .to_string(),
                                        ));
                                        let _ = evt_tx.send(SolverWorkerEvent::Running(false));
                                        running = false;
                                    } else {
                                        if next_running {
                                            step_idx = 0;
                                            last_publish = std::time::Instant::now();
                                        }
                                        running = next_running;
                                        let _ = evt_tx.send(SolverWorkerEvent::Running(running));
                                    }
                                }
                                SolverWorkerCommand::UpdateParams(next_params) => {
                                    params = next_params;
                                    if let Some(s) = solver.as_mut() {
                                        solver_worker_apply_params(s, solver_kind, params);
                                    }
                                }
                                SolverWorkerCommand::Shutdown => return,
                            }
                        }
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
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

        if step_idx == 0 && matches!(solver_kind, SolverKind::Incompressible) {
            solver.incompressible_set_should_stop(false);
        }

        if params.adaptive_dt {
            let sound_speed = match solver_kind {
                SolverKind::Compressible => params.eos.sound_speed(params.density as f64),
                SolverKind::Incompressible => 0.0,
            };
            let wave_speed = match solver_kind {
                SolverKind::Compressible => prev_max_vel + sound_speed,
                SolverKind::Incompressible => prev_max_vel,
            };
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

        if matches!(solver_kind, SolverKind::Incompressible) && solver.incompressible_should_stop() {
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

        if matches!(solver_kind, SolverKind::Incompressible) {
            if let Some((iters, res_u, res_p)) = solver.incompressible_outer_stats() {
                stats.outer_iterations = iters;
                stats.outer_residual_u = res_u;
                stats.outer_residual_p = res_p;
            }
            if step_linear_stats.len() == 3 {
                stats.stats_ux = step_linear_stats[0];
                stats.stats_uy = step_linear_stats[1];
                stats.stats_p = step_linear_stats[2];
            }
        }

        let log_every_steps = params.log_every_steps.max(1) as u64;
        let should_log = params.log_convergence && (step_idx % log_every_steps == 0);

        let should_readback =
            step_idx == 0 || last_publish.elapsed() >= publish_interval || should_log;

        if should_readback {
            last_publish = std::time::Instant::now();
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

            if should_log || nonfinite_u > 0 || nonfinite_p > 0 {
                let outer_str = if matches!(solver_kind, SolverKind::Incompressible) {
                    solver
                        .incompressible_outer_stats()
                        .map(|(iters, res_u, res_p)| {
                            format!(
                                " outer(iters={}, u={:.3e}, p={:.3e})",
                                iters, res_u, res_p
                            )
                        })
                        .unwrap_or_default()
                } else {
                    String::new()
                };

                eprintln!(
                    "[cfd2][{:?}] step={} t={:.4e} dt={:.2e} max|u|={:.3e} p=[{:.3e},{:.3e}] solves={} last(iters={}, res={:.3e}, conv={}, div={}){} nonfinite(u={}, p={})",
                    solver_kind,
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
        }

        step_idx = step_idx.wrapping_add(1);
        thread::sleep(std::time::Duration::from_millis(1));
    }
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
