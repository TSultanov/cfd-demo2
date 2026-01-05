use crate::solver::gpu::init::ShaderVariant;
use crate::solver::gpu::structs::{LinearSolverStats, PreconditionerType};
use crate::solver::gpu::GpuSolver;
use crate::solver::mesh::{
    generate_cut_cell_mesh, generate_delaunay_mesh, generate_voronoi_mesh, BackwardsStep,
    ChannelWithObstacle, Mesh,
};
use crate::solver::scheme::Scheme;
use crate::ui::cfd_renderer;
use eframe::egui;
use egui_plot::{Plot, PlotPoints, Polygon};
use nalgebra::{Point2, Vector2};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

struct RuntimeParams {
    target_cfl: f64,
    adaptive_dt: bool,
}

/// Rendering mode for the mesh visualization
#[derive(PartialEq, Clone, Copy)]
enum RenderMode {
    /// Use egui_plot polygons (slow for large meshes)
    EguiPlot,
    /// Render directly on GPU (zero-copy)
    GpuDirect,
}

#[derive(PartialEq)]
enum GeometryType {
    BackwardsStep,
    ChannelObstacle,
}

#[derive(PartialEq, Clone, Copy, Debug)]
enum MeshType {
    CutCell,
    Delaunay,
    Voronoi,
}

#[derive(PartialEq, Clone, Copy)]
enum PlotField {
    Pressure,
    VelocityX,
    VelocityY,
    VelocityMag,
}

#[derive(Clone, PartialEq)]
struct Fluid {
    name: String,
    density: f64,
    viscosity: f64,
}

type SharedResults = Arc<Mutex<Option<(Vec<(f64, f64)>, Vec<f64>)>>>;

impl Fluid {
    fn presets() -> Vec<Fluid> {
        vec![
            Fluid {
                name: "Water".into(),
                density: 1000.0,
                viscosity: 0.001,
            },
            Fluid {
                name: "Air".into(),
                density: 1.225,
                viscosity: 1.81e-5,
            },
            Fluid {
                name: "Alcohol".into(),
                density: 789.0,
                viscosity: 0.0012,
            },
            Fluid {
                name: "Kerosene".into(),
                density: 820.0,
                viscosity: 0.00164,
            },
            Fluid {
                name: "Mercury".into(),
                density: 13546.0,
                viscosity: 0.001526,
            },
            Fluid {
                name: "Custom".into(),
                density: 1.0,
                viscosity: 0.01,
            },
        ]
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
enum TimeScheme {
    Euler,
    BDF2,
}

impl TimeScheme {
    fn gpu_id(self) -> u32 {
        match self {
            TimeScheme::Euler => 0,
            TimeScheme::BDF2 => 1,
        }
    }
}

// Cached GPU solver stats for UI display (avoids lock contention)
#[allow(dead_code)]
#[derive(Default, Clone)]
struct CachedGpuStats {
    time: f32,
    dt: f32,
    stats_ux: LinearSolverStats,
    stats_uy: LinearSolverStats,
    stats_p: LinearSolverStats,
    outer_residual_u: f32,
    outer_residual_p: f32,
    outer_iterations: u32,
    step_time_ms: f32,
}

pub struct CFDApp {
    gpu_solver: Option<Arc<Mutex<GpuSolver>>>,
    gpu_solver_running: Arc<AtomicBool>,
    shared_results: SharedResults,
    shared_gpu_stats: Arc<Mutex<CachedGpuStats>>,
    shared_params: Arc<Mutex<RuntimeParams>>,
    cached_u: Vec<(f64, f64)>,
    cached_p: Vec<f64>,
    cached_gpu_stats: CachedGpuStats,
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
    render_mode: RenderMode,
    alpha_u: f64,
    alpha_p: f64,
    time_scheme: TimeScheme,
    inlet_velocity: f32,
    ramp_time: f32,
    selected_preconditioner: PreconditionerType,
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

        Self {
            gpu_solver: None,
            gpu_solver_running: Arc::new(AtomicBool::new(false)),
            shared_results: Arc::new(Mutex::new(None)),
            shared_gpu_stats: Arc::new(Mutex::new(CachedGpuStats::default())),
            shared_params: Arc::new(Mutex::new(RuntimeParams {
                target_cfl: 0.95,
                adaptive_dt: true,
            })),
            cached_u: Vec::new(),
            cached_p: Vec::new(),
            cached_gpu_stats: CachedGpuStats::default(),
            mesh: None,
            cached_cells: Vec::new(),
            actual_min_cell_size: 0.01,
            min_cell_size: 0.025,
            max_cell_size: 0.025,
            growth_rate: 1.2,
            timestep: 0.001,
            selected_geometry: GeometryType::BackwardsStep,
            mesh_type: MeshType::CutCell,
            plot_field: PlotField::VelocityMag,
            is_running: false,
            selected_scheme: Scheme::Upwind,
            current_fluid: Fluid::presets()[1].clone(),
            show_mesh_lines: true,
            adaptive_dt: true,
            target_cfl: 0.95,
            render_mode: RenderMode::GpuDirect,
            alpha_u: 1.0,
            alpha_p: 1.0,
            time_scheme: TimeScheme::Euler,
            inlet_velocity: 1.0,
            ramp_time: 0.001,
            selected_preconditioner: PreconditionerType::Jacobi,
            wgpu_device,
            wgpu_queue,
            target_format,
            cfd_renderer: None,
        }
    }

    fn with_gpu_solver<F>(&self, f: F)
    where
        F: FnOnce(&mut GpuSolver),
    {
        if let Some(solver) = &self.gpu_solver {
            if let Ok(mut guard) = solver.lock() {
                f(&mut guard);
            }
        }
    }

    fn update_renderer_field(&self) {
        if let (Some(renderer), Some(solver), Some(device)) =
            (&self.cfd_renderer, &self.gpu_solver, &self.wgpu_device)
        {
            if let (Ok(mut renderer), Ok(solver)) = (renderer.lock(), solver.lock()) {
                // Use consolidated FluidState buffer for all fields
                renderer.update_bind_group(device, &solver.b_state);
            }
        }
    }

    fn init_solver(&mut self) {
        self.is_running = false;
        self.gpu_solver_running.store(false, Ordering::Relaxed);

        let mesh = self.build_mesh();
        let n_cells = mesh.num_cells();
        let initial_u = self.build_initial_velocity(&mesh);
        let initial_p = vec![0.0; n_cells];

        let mut gpu_solver = pollster::block_on(GpuSolver::new_with_shader_variant(
            &mesh,
            self.wgpu_device.clone(),
            self.wgpu_queue.clone(),
            ShaderVariant::Generated,
        ));
        gpu_solver.set_dt(self.timestep as f32);
        gpu_solver.set_viscosity(self.current_fluid.viscosity as f32);
        gpu_solver.set_density(self.current_fluid.density as f32);
        gpu_solver.set_scheme(self.selected_scheme.gpu_id());
        gpu_solver.set_alpha_u(self.alpha_u as f32);
        gpu_solver.set_alpha_p(self.alpha_p as f32);
        gpu_solver.set_time_scheme(self.time_scheme.gpu_id());
        gpu_solver.set_u(&initial_u);
        gpu_solver.set_p(&initial_p);
        gpu_solver.set_inlet_velocity(self.inlet_velocity);
        gpu_solver.set_ramp_time(self.ramp_time);
        gpu_solver.set_precond_type(self.selected_preconditioner);
        gpu_solver.initialize_history();

        self.cached_u = initial_u;
        self.cached_p = initial_p;
        self.cached_cells = Self::cache_cells(&mesh);

        // Initialize renderer
        if let Some(device) = &self.wgpu_device {
            let mut renderer = cfd_renderer::CfdRenderResources::new(
                device,
                self.target_format,
                mesh.num_cells() * 10,
            );

            let vertices = cfd_renderer::build_mesh_vertices(&self.cached_cells);
            let line_vertices = cfd_renderer::build_line_vertices(&self.cached_cells);

            // Debug: Check for cells in the step region
            let mut cells_in_step = 0;
            for cell in &self.cached_cells {
                let mut cx = 0.0;
                let mut cy = 0.0;
                for p in cell {
                    cx += p[0];
                    cy += p[1];
                }
                cx /= cell.len() as f64;
                cy /= cell.len() as f64;

                if cx < 0.5 && cy < 0.5 {
                    cells_in_step += 1;
                }
            }
            println!(
                "Init Solver: {} cells, {} vertices. Cells in step (x<0.5, y<0.5): {}",
                self.cached_cells.len(),
                vertices.len(),
                cells_in_step
            );

            if let Some(queue) = &self.wgpu_queue {
                renderer.update_mesh(queue, &vertices, &line_vertices);
                // Set initial bind group based on selected field
                // Use consolidated FluidState buffer for all fields
                renderer.update_bind_group(device, &gpu_solver.b_state);
            }
            self.cfd_renderer = Some(Arc::new(Mutex::new(renderer)));
        }

        // Compute actual min cell size
        self.actual_min_cell_size = mesh
            .cell_vol
            .iter()
            .map(|&v| v.sqrt())
            .fold(f64::INFINITY, f64::min);

        self.mesh = Some(mesh);

        self.gpu_solver = Some(Arc::new(Mutex::new(gpu_solver)));
        self.shared_results = Arc::new(Mutex::new(None));
        self.shared_gpu_stats = Arc::new(Mutex::new(CachedGpuStats::default()));
        self.shared_params = Arc::new(Mutex::new(RuntimeParams {
            target_cfl: self.target_cfl,
            adaptive_dt: self.adaptive_dt,
        }));
        self.cached_gpu_stats = CachedGpuStats::default();
    }

    fn build_mesh(&self) -> Mesh {
        match self.selected_geometry {
            GeometryType::BackwardsStep => {
                let length = 3.5;
                let domain_size = Vector2::new(length, 1.0);
                let geo = BackwardsStep {
                    length,
                    height_inlet: 0.5,
                    height_outlet: 1.0,
                    step_x: 0.5,
                };
                let mut mesh = match self.mesh_type {
                    MeshType::CutCell => generate_cut_cell_mesh(
                        &geo,
                        self.min_cell_size,
                        self.max_cell_size,
                        self.growth_rate,
                        domain_size,
                    ),
                    MeshType::Delaunay => generate_delaunay_mesh(
                        &geo,
                        self.min_cell_size,
                        self.max_cell_size,
                        self.growth_rate,
                        domain_size,
                    ),
                    MeshType::Voronoi => generate_voronoi_mesh(
                        &geo,
                        self.min_cell_size,
                        self.max_cell_size,
                        self.growth_rate,
                        domain_size,
                    ),
                };
                mesh.smooth(&geo, 0.3, 50);
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
                let mesh = match self.mesh_type {
                    MeshType::CutCell => {
                        let mut mesh = generate_cut_cell_mesh(
                            &geo,
                            self.min_cell_size,
                            self.max_cell_size,
                            self.growth_rate,
                            domain_size,
                        );
                        mesh.smooth(&geo, 0.3, 100);
                        mesh
                    }
                    MeshType::Delaunay => {
                        let mut mesh = generate_delaunay_mesh(
                            &geo,
                            self.min_cell_size,
                            self.max_cell_size,
                            self.growth_rate,
                            domain_size,
                        );
                        mesh.smooth(&geo, 0.3, 50);
                        mesh
                    }
                    MeshType::Voronoi => {
                        let mut mesh = generate_voronoi_mesh(
                            &geo,
                            self.min_cell_size,
                            self.max_cell_size,
                            self.growth_rate,
                            domain_size,
                        );
                        // Voronoi meshes are usually good quality, but we can smooth them too if needed.
                        // For now, let's just return it.
                        mesh.smooth(&geo, 0.3, 50);
                        mesh
                    }
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

    fn build_initial_velocity(&self, mesh: &Mesh) -> Vec<(f64, f64)> {
        let mut u = vec![(0.0, 0.0); mesh.num_cells()];
        for (i, _vel) in u.iter_mut().enumerate() {
            let cx = mesh.cell_cx[i];
            let cy = mesh.cell_cy[i];

            // // Add small perturbation to break symmetry
            // let perturbation = (rng.gen::<f64>() - 0.5) * 0.01;
            // vel.1 += perturbation;

            if cx < self.max_cell_size {
                match self.selected_geometry {
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

    fn update_gpu_fluid(&self) {
        self.with_gpu_solver(|solver| {
            solver.set_density(self.current_fluid.density as f32);
            solver.set_viscosity(self.current_fluid.viscosity as f32);
        });
    }

    fn update_gpu_dt(&self) {
        self.with_gpu_solver(|solver| solver.set_dt(self.timestep as f32));
    }

    fn update_gpu_scheme(&self) {
        self.with_gpu_solver(|solver| solver.set_scheme(self.selected_scheme.gpu_id()));
    }

    fn update_gpu_alpha_u(&self) {
        self.with_gpu_solver(|solver| solver.set_alpha_u(self.alpha_u as f32));
    }

    fn update_gpu_alpha_p(&self) {
        self.with_gpu_solver(|solver| solver.set_alpha_p(self.alpha_p as f32));
    }

    fn update_gpu_time_scheme(&self) {
        self.with_gpu_solver(|solver| solver.set_time_scheme(self.time_scheme.gpu_id()));
    }

    fn update_gpu_inlet_velocity(&self) {
        self.with_gpu_solver(|solver| solver.set_inlet_velocity(self.inlet_velocity));
    }

    fn update_gpu_ramp_time(&self) {
        self.with_gpu_solver(|solver| solver.set_ramp_time(self.ramp_time));
    }

    fn update_gpu_preconditioner(&self) {
        self.with_gpu_solver(|solver| solver.set_precond_type(self.selected_preconditioner));
    }
}

impl eframe::App for CFDApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Sync running state from background thread
        if self.is_running && !self.gpu_solver_running.load(Ordering::Relaxed) {
            self.is_running = false;
        }

        egui::SidePanel::left("controls").show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    ui.heading("CFD Controls");

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

                        if ui
                            .add(
                                egui::Slider::new(&mut self.ramp_time, 0.0..=5.0)
                                    .text("Ramp Time (s)"),
                            )
                            .changed()
                        {
                            self.update_gpu_ramp_time();
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

                        let recommended_dt = 0.5 * self.min_cell_size;
                        let cfl = self.timestep / self.min_cell_size;

                        if ui
                            .add(
                                egui::Slider::new(&mut self.timestep, 0.0001..=0.1)
                                    .text("Timestep"),
                            )
                            .changed()
                        {
                            self.update_gpu_dt();
                        }

                        if ui
                            .checkbox(&mut self.adaptive_dt, "Adaptive Timestep")
                            .changed()
                        {
                            if let Ok(mut params) = self.shared_params.lock() {
                                params.adaptive_dt = self.adaptive_dt;
                            }
                        }
                        if self.adaptive_dt {
                            if ui
                                .add(
                                    egui::Slider::new(&mut self.target_cfl, 0.1..=1.0)
                                        .text("Target CFL"),
                                )
                                .changed()
                            {
                                if let Ok(mut params) = self.shared_params.lock() {
                                    params.target_cfl = self.target_cfl;
                                }
                            }
                        }

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
                        ui.label("Discretization Scheme");
                        if ui
                            .radio(matches!(self.selected_scheme, Scheme::Upwind), "Upwind")
                            .clicked()
                        {
                            self.selected_scheme = Scheme::Upwind;
                            self.update_gpu_scheme();
                        }
                        if ui
                            .radio(
                                matches!(self.selected_scheme, Scheme::SecondOrderUpwind),
                                "Second Order Upwind",
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

                        ui.separator();
                        ui.label("Preconditioner");
                        if ui
                            .radio(
                                matches!(self.selected_preconditioner, PreconditionerType::Jacobi),
                                "Jacobi",
                            )
                            .clicked()
                        {
                            self.selected_preconditioner = PreconditionerType::Jacobi;
                            self.update_gpu_preconditioner();
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

                        ui.separator();
                        ui.label("Under-Relaxation Factors");
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
                    });

                    ui.separator();
                    ui.label("Time Stepping Scheme");
                    egui::ComboBox::from_label("Time Scheme")
                        .selected_text(format!("{:?}", self.time_scheme))
                        .show_ui(ui, |ui| {
                            if ui
                                .selectable_value(&mut self.time_scheme, TimeScheme::Euler, "Euler")
                                .clicked()
                            {
                                self.update_gpu_time_scheme();
                            }
                            if ui
                                .selectable_value(&mut self.time_scheme, TimeScheme::BDF2, "BDF2")
                                .clicked()
                            {
                                self.update_gpu_time_scheme();
                            }
                        });

                    ui.separator();
                    ui.label("Pressure-Velocity Coupling");
                    ui.label("Coupled solver (block system)");

                    if ui.button("Initialize / Reset").clicked() {
                        self.init_solver();
                    }

                    if self.gpu_solver.is_some()
                        && ui
                            .button(if self.is_running { "Pause" } else { "Run" })
                            .clicked()
                    {
                        self.is_running = !self.is_running;

                        if let Some(gpu_solver) = &self.gpu_solver {
                            if self.is_running {
                                // Reset stop flag
                                if let Ok(mut solver) = gpu_solver.lock() {
                                    solver.should_stop = false;
                                }

                                self.gpu_solver_running.store(true, Ordering::Relaxed);
                                let solver_arc = gpu_solver.clone();
                                let running_flag = self.gpu_solver_running.clone();
                                let shared_results = self.shared_results.clone();
                                let shared_gpu_stats = self.shared_gpu_stats.clone();
                                let shared_params = self.shared_params.clone();
                                let ctx_clone = ctx.clone();
                                let min_cell_size_clone = self.actual_min_cell_size;
                                thread::spawn(move || {
                                    while running_flag.load(Ordering::Relaxed) {
                                        if let Ok(mut solver) = solver_arc.lock() {
                                            let start = std::time::Instant::now();
                                            solver.step();

                                            if solver.should_stop {
                                                running_flag.store(false, Ordering::Relaxed);
                                            }

                                            let step_time = start.elapsed().as_secs_f32() * 1000.0;
                                            let u = pollster::block_on(solver.get_u());
                                            let p = pollster::block_on(solver.get_p());

                                            let (adaptive_dt, target_cfl) =
                                                if let Ok(params) = shared_params.lock() {
                                                    (params.adaptive_dt, params.target_cfl)
                                                } else {
                                                    (false, 0.9)
                                                };

                                            if adaptive_dt {
                                                let mut max_vel = 0.0f64;
                                                for (vx, vy) in &u {
                                                    let v = (vx.powi(2) + vy.powi(2)).sqrt();
                                                    if v > max_vel {
                                                        max_vel = v;
                                                    }
                                                }

                                                if max_vel > 1e-6 {
                                                    let current_dt = solver.constants.dt as f64;
                                                    let mut next_dt =
                                                        target_cfl * min_cell_size_clone / max_vel;

                                                    // Limit increase to 1.2x to prevent shock
                                                    if next_dt > current_dt * 1.2 {
                                                        next_dt = current_dt * 1.2;
                                                    }

                                                    next_dt = next_dt.clamp(1e-9, 100.0);
                                                    solver.set_dt(next_dt as f32);
                                                }
                                            }

                                            let stats = CachedGpuStats {
                                                time: solver.constants.time,
                                                dt: solver.constants.dt,
                                                stats_ux: *solver.stats_ux.lock().unwrap(),
                                                stats_uy: *solver.stats_uy.lock().unwrap(),
                                                stats_p: *solver.stats_p.lock().unwrap(),
                                                outer_residual_u: *solver
                                                    .outer_residual_u
                                                    .lock()
                                                    .unwrap(),
                                                outer_residual_p: *solver
                                                    .outer_residual_p
                                                    .lock()
                                                    .unwrap(),
                                                outer_iterations: *solver
                                                    .outer_iterations
                                                    .lock()
                                                    .unwrap(),
                                                step_time_ms: step_time,
                                            };

                                            if let Ok(mut results) = shared_results.lock() {
                                                *results = Some((u, p));
                                            }
                                            if let Ok(mut gpu_stats) = shared_gpu_stats.lock() {
                                                *gpu_stats = stats;
                                            }
                                        }
                                        ctx_clone.request_repaint();
                                        thread::sleep(std::time::Duration::from_millis(1));
                                    }
                                });
                            } else {
                                self.gpu_solver_running.store(false, Ordering::Relaxed);
                            }
                        }
                    }

                    ui.separator();

                    if self.gpu_solver.is_some() {
                        if let Ok(stats) = self.shared_gpu_stats.try_lock() {
                            self.cached_gpu_stats = stats.clone();
                        }

                        let stats = &self.cached_gpu_stats;
                        ui.label(format!("dt: {:.2e}", stats.dt));
                        ui.label(format!(
                            "Coupled: {} iters, U:{:.2e} P:{:.2e}",
                            stats.outer_iterations, stats.outer_residual_u, stats.outer_residual_p,
                        ));
                        ui.label(format!("Step time: {:.1} ms", stats.step_time_ms));
                    }
                });
        });

        let (min_val, max_val, values) = if self.gpu_solver.is_some() {
            if let Ok(mut results) = self.shared_results.lock() {
                if let Some((u, p)) = results.take() {
                    self.cached_u = u;
                    self.cached_p = p;
                }
            }

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

            if self.gpu_solver.is_some() {
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

            let cells = if self.gpu_solver.is_some() && !self.cached_cells.is_empty() {
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

                            // FluidState layout (8 floats per cell):
                            // [0]: u.x, [1]: u.y, [2]: p, [3]: d_p,
                            // [4]: grad_p.x, [5]: grad_p.y, [6]: gc.x, [7]: gc.y
                            let (stride, offset, mode) = match self.plot_field {
                                PlotField::Pressure => (8, 2, 0),    // p at offset 2
                                PlotField::VelocityX => (8, 0, 0),   // u.x at offset 0
                                PlotField::VelocityY => (8, 1, 0),   // u.y at offset 1
                                PlotField::VelocityMag => (8, 0, 1), // magnitude mode, u at offset 0,1
                            };

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
                    ui.label("Press Initialize to start");
                });
            }
        });
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
