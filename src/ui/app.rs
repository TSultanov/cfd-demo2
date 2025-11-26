use crate::solver::fvm::Scheme;
use crate::solver::gpu::structs::LinearSolverStats;
use crate::solver::gpu::GpuSolver;
use crate::solver::mesh::{generate_cut_cell_mesh, BackwardsStep, ChannelWithObstacle, Mesh};
use crate::solver::piso::PisoSolver;
use eframe::egui;
use egui_plot::{Plot, PlotPoints, Polygon};
use nalgebra::{Point2, Vector2};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(PartialEq)]
enum GeometryType {
    BackwardsStep,
    ChannelObstacle,
}

#[derive(PartialEq)]
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

use crate::solver::parallel::ParallelPisoSolver;

#[derive(PartialEq, Clone, Copy)]
enum Precision {
    F32,
    F64,
}

enum CpuSolverWrapper {
    SerialF32(PisoSolver<f32>),
    SerialF64(PisoSolver<f64>),
    ParallelF32(ParallelPisoSolver<f32>),
    ParallelF64(ParallelPisoSolver<f64>),
}

impl CpuSolverWrapper {
    fn step(&mut self) {
        match self {
            Self::SerialF32(s) => s.step(),
            Self::SerialF64(s) => s.step(),
            Self::ParallelF32(s) => s.step(),
            Self::ParallelF64(s) => s.step(),
        }
    }

    fn time(&self) -> f64 {
        match self {
            Self::SerialF32(s) => s.time as f64,
            Self::SerialF64(s) => s.time,
            Self::ParallelF32(s) => s.partitions[0].read().unwrap().time as f64,
            Self::ParallelF64(s) => s.partitions[0].read().unwrap().time,
        }
    }

    fn residuals(&self) -> Vec<(String, f64)> {
        match self {
            Self::SerialF32(s) => s
                .residuals
                .iter()
                .map(|(k, v)| (k.clone(), *v as f64))
                .collect(),
            Self::SerialF64(s) => s.residuals.iter().map(|(k, v)| (k.clone(), *v)).collect(),
            Self::ParallelF32(s) => s.partitions[0]
                .read()
                .unwrap()
                .residuals
                .iter()
                .map(|(k, v)| (k.clone(), *v as f64))
                .collect(),
            Self::ParallelF64(s) => s.partitions[0]
                .read()
                .unwrap()
                .residuals
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
        }
    }

    fn get_mesh(&self) -> crate::solver::mesh::Mesh {
        match self {
            Self::SerialF32(s) => s.mesh.clone(),
            Self::SerialF64(s) => s.mesh.clone(),
            Self::ParallelF32(s) => s.partitions[0].read().unwrap().mesh.clone(), // Just for visualization, might be partial? No, parallel solver partitions mesh.
            Self::ParallelF64(s) => s.partitions[0].read().unwrap().mesh.clone(),
        }
    }

    // Helper to get data for plotting.
    // For parallel, we need to gather data. But for now, let's just show rank 0 or handle it properly.
    // Actually, ParallelPisoSolver doesn't have a "gather" method yet.
    // The existing code for parallel plotting was:
    // if let Some(ps) = &self.parallel_solver { ... }
    // It was iterating over partitions.

    fn get_results(&self) -> (Vec<(f64, f64)>, Vec<f64>) {
        match self {
            Self::SerialF32(s) => {
                let u =
                    s.u.vx
                        .iter()
                        .zip(s.u.vy.iter())
                        .map(|(&x, &y)| (x as f64, y as f64))
                        .collect();
                let p = s.p.values.iter().map(|&v| v as f64).collect();
                (u, p)
            }
            Self::SerialF64(s) => {
                let u =
                    s.u.vx
                        .iter()
                        .zip(s.u.vy.iter())
                        .map(|(&x, &y)| (x, y))
                        .collect();
                let p = s.p.values.clone();
                (u, p)
            }
            Self::ParallelF32(s) => {
                // Gather from all partitions
                // This is tricky because we need to map back to global indices or just concat?
                // The partitions are stored in `partitions`.
                // We can just iterate and collect.
                // But for plotting we need positions.
                // The plotting code iterates over cells and uses their positions.
                // So we can just return a flat list of all cells from all partitions.
                let mut u = Vec::new();
                let mut p = Vec::new();
                for part in &s.partitions {
                    let s = part.read().unwrap();
                    for i in 0..s.mesh.num_cells() {
                        u.push((s.u.vx[i] as f64, s.u.vy[i] as f64));
                        p.push(s.p.values[i] as f64);
                    }
                }
                (u, p)
            }
            Self::ParallelF64(s) => {
                let mut u = Vec::new();
                let mut p = Vec::new();
                for part in &s.partitions {
                    let s = part.read().unwrap();
                    for i in 0..s.mesh.num_cells() {
                        u.push((s.u.vx[i], s.u.vy[i]));
                        p.push(s.p.values[i]);
                    }
                }
                (u, p)
            }
        }
    }

    fn get_all_cells_vertices(&self) -> Vec<Vec<[f64; 2]>> {
        match self {
            Self::SerialF32(s) => {
                let mut cells = Vec::with_capacity(s.mesh.num_cells());
                for i in 0..s.mesh.num_cells() {
                    let start = s.mesh.cell_vertex_offsets[i];
                    let end = s.mesh.cell_vertex_offsets[i + 1];
                    let polygon_points: Vec<[f64; 2]> = (start..end)
                        .map(|k| {
                            let v_idx = s.mesh.cell_vertices[k];
                            [s.mesh.vx[v_idx], s.mesh.vy[v_idx]]
                        })
                        .collect();
                    cells.push(polygon_points);
                }
                cells
            }
            Self::SerialF64(s) => {
                let mut cells = Vec::with_capacity(s.mesh.num_cells());
                for i in 0..s.mesh.num_cells() {
                    let start = s.mesh.cell_vertex_offsets[i];
                    let end = s.mesh.cell_vertex_offsets[i + 1];
                    let polygon_points: Vec<[f64; 2]> = (start..end)
                        .map(|k| {
                            let v_idx = s.mesh.cell_vertices[k];
                            [s.mesh.vx[v_idx], s.mesh.vy[v_idx]]
                        })
                        .collect();
                    cells.push(polygon_points);
                }
                cells
            }
            Self::ParallelF32(s) => {
                let mut cells = Vec::new();
                for part in &s.partitions {
                    let s = part.read().unwrap();
                    for i in 0..s.mesh.num_cells() {
                        let start = s.mesh.cell_vertex_offsets[i];
                        let end = s.mesh.cell_vertex_offsets[i + 1];
                        let polygon_points: Vec<[f64; 2]> = (start..end)
                            .map(|k| {
                                let v_idx = s.mesh.cell_vertices[k];
                                [s.mesh.vx[v_idx], s.mesh.vy[v_idx]]
                            })
                            .collect();
                        cells.push(polygon_points);
                    }
                }
                cells
            }
            Self::ParallelF64(s) => {
                let mut cells = Vec::new();
                for part in &s.partitions {
                    let s = part.read().unwrap();
                    for i in 0..s.mesh.num_cells() {
                        let start = s.mesh.cell_vertex_offsets[i];
                        let end = s.mesh.cell_vertex_offsets[i + 1];
                        let polygon_points: Vec<[f64; 2]> = (start..end)
                            .map(|k| {
                                let v_idx = s.mesh.cell_vertices[k];
                                [s.mesh.vx[v_idx], s.mesh.vy[v_idx]]
                            })
                            .collect();
                        cells.push(polygon_points);
                    }
                }
                cells
            }
        }
    }

    fn update_fluid(&mut self, density: f64, viscosity: f64) {
        match self {
            Self::SerialF32(s) => {
                s.density = density as f32;
                s.viscosity = viscosity as f32;
            }
            Self::SerialF64(s) => {
                s.density = density;
                s.viscosity = viscosity;
            }
            Self::ParallelF32(s) => {
                for part in &s.partitions {
                    let mut s = part.write().unwrap();
                    s.density = density as f32;
                    s.viscosity = viscosity as f32;
                }
            }
            Self::ParallelF64(s) => {
                for part in &s.partitions {
                    let mut s = part.write().unwrap();
                    s.density = density;
                    s.viscosity = viscosity;
                }
            }
        }
    }
}

// Cached GPU solver stats for UI display (avoids lock contention)
#[derive(Default, Clone)]
struct CachedGpuStats {
    time: f32,
    stats_ux: LinearSolverStats,
    stats_uy: LinearSolverStats,
    stats_p: LinearSolverStats,
}

pub struct CFDApp {
    cpu_solver: Option<CpuSolverWrapper>,
    gpu_solver: Option<Arc<Mutex<GpuSolver>>>,
    gpu_solver_running: Arc<AtomicBool>,
    shared_results: Arc<Mutex<Option<(Vec<(f64, f64)>, Vec<f64>)>>>,
    shared_gpu_stats: Arc<Mutex<CachedGpuStats>>,
    cached_u: Vec<(f64, f64)>,
    cached_p: Vec<f64>,
    cached_gpu_stats: CachedGpuStats,
    // For GPU mode: store mesh and cell vertices separately since cpu_solver is None
    mesh: Option<Mesh>,
    cached_cells: Vec<Vec<[f64; 2]>>,
    use_parallel: bool,
    use_gpu: bool,
    precision: Precision,
    n_threads: usize,
    min_cell_size: f64,
    max_cell_size: f64,
    timestep: f64,
    selected_geometry: GeometryType,
    plot_field: PlotField,
    is_running: bool,
    selected_scheme: Scheme,
    current_fluid: Fluid,
    show_mesh_lines: bool,
}

impl CFDApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            cpu_solver: None,
            gpu_solver: None,
            gpu_solver_running: Arc::new(AtomicBool::new(false)),
            shared_results: Arc::new(Mutex::new(None)),
            shared_gpu_stats: Arc::new(Mutex::new(CachedGpuStats::default())),
            cached_u: Vec::new(),
            cached_p: Vec::new(),
            cached_gpu_stats: CachedGpuStats::default(),
            mesh: None,
            cached_cells: Vec::new(),
            use_parallel: false,
            use_gpu: false,
            precision: Precision::F64,
            n_threads: 4,
            min_cell_size: 0.025,
            max_cell_size: 0.025,
            timestep: 0.01,
            selected_geometry: GeometryType::BackwardsStep,
            plot_field: PlotField::VelocityMag,
            is_running: false,
            selected_scheme: Scheme::Upwind,
            current_fluid: Fluid::presets()[0].clone(),
            show_mesh_lines: true,
        }
    }

    fn init_solver(&mut self) {
        // use crate::solver::float::Float;

        self.is_running = false;
        self.gpu_solver_running.store(false, Ordering::Relaxed);

        let mesh = match self.selected_geometry {
            GeometryType::BackwardsStep => {
                let length = 3.5;
                let domain_size = Vector2::new(length, 1.0);
                let geo = BackwardsStep {
                    length,
                    height_inlet: 0.5,
                    height_outlet: 1.0,
                    step_x: 0.5,
                };
                let mut mesh = generate_cut_cell_mesh(
                    &geo,
                    self.min_cell_size,
                    self.max_cell_size,
                    domain_size,
                );
                mesh.smooth(&geo, 0.3, 50);
                mesh
            }
            GeometryType::ChannelObstacle => {
                let length = 3.0;
                let domain_size = Vector2::new(length, 1.0);
                let geo = ChannelWithObstacle {
                    length,
                    height: 1.0,
                    obstacle_center: Point2::new(1.0, 0.5),
                    obstacle_radius: 0.2,
                };
                let mut mesh = generate_cut_cell_mesh(
                    &geo,
                    self.min_cell_size,
                    self.max_cell_size,
                    domain_size,
                );
                mesh.smooth(&geo, 0.3, 50);
                mesh
            }
        };

        if self.use_gpu {
            // GPU solver uses f64 for setup but runs in f32 internally (mostly)
            // We'll use f64 PisoSolver for initialization
            let mut solver = PisoSolver::<f64>::new(mesh.clone());
            solver.dt = self.timestep;
            solver.density = self.current_fluid.density;
            solver.viscosity = self.current_fluid.viscosity;
            solver.scheme = self.selected_scheme;

            // Initialize GPU solver
            let mut gpu_solver = pollster::block_on(GpuSolver::new(&solver.mesh));
            gpu_solver.set_dt(self.timestep as f32);
            gpu_solver.set_viscosity(self.current_fluid.viscosity as f32);
            gpu_solver.set_density(self.current_fluid.density as f32);

            let scheme_idx = match self.selected_scheme {
                Scheme::Upwind => 0,
                Scheme::Central => 1,
                Scheme::QUICK => 2,
            };
            gpu_solver.set_scheme(scheme_idx);

            // CPU Init Logic (for initial velocity setup)
            let n_cells = solver.mesh.num_cells();
            for i in 0..n_cells {
                let cx = solver.mesh.cell_cx[i];
                let cy = solver.mesh.cell_cy[i];
                if cx < self.max_cell_size {
                    match self.selected_geometry {
                        GeometryType::BackwardsStep => {
                            if cy > 0.5 {
                                solver.u.vx[i] = 1.0;
                                solver.u.vy[i] = 0.0;
                            }
                        }
                        GeometryType::ChannelObstacle => {
                            solver.u.vx[i] = 1.0;
                            solver.u.vy[i] = 0.0;
                        }
                    }
                }
            }

            // Upload initial U to GPU
            let u_init: Vec<(f64, f64)> = solver
                .u
                .vx
                .iter()
                .zip(solver.u.vy.iter())
                .map(|(&x, &y)| (x, y))
                .collect();
            gpu_solver.set_u(&u_init);

            // Initialize cached values
            self.cached_u = u_init;
            self.cached_p = vec![0.0; n_cells];

            // Cache cell vertices for plotting (since we won't have cpu_solver)
            self.cached_cells = {
                let mut cells = Vec::with_capacity(n_cells);
                for i in 0..n_cells {
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
            };

            // Store mesh for stats display
            self.mesh = Some(mesh);

            // GPU mode: don't keep cpu_solver around
            self.cpu_solver = None;
            self.gpu_solver = Some(Arc::new(Mutex::new(gpu_solver)));
            self.shared_results = Arc::new(Mutex::new(None));
            self.shared_gpu_stats = Arc::new(Mutex::new(CachedGpuStats::default()));
            self.cached_gpu_stats = CachedGpuStats::default();
        } else if self.use_parallel {
            self.gpu_solver = None;
            match self.precision {
                Precision::F32 => {
                    let mut ps = ParallelPisoSolver::<f32>::new(mesh, self.n_threads);
                    for solver in &mut ps.partitions {
                        let mut solver = solver.write().unwrap();
                        solver.dt = self.timestep as f32;
                        solver.density = self.current_fluid.density as f32;
                        solver.viscosity = self.current_fluid.viscosity as f32;
                        solver.scheme = self.selected_scheme;
                        let n_cells = solver.mesh.num_cells();
                        for i in 0..n_cells {
                            let cx = solver.mesh.cell_cx[i];
                            let cy = solver.mesh.cell_cy[i];
                            if cx < self.max_cell_size {
                                match self.selected_geometry {
                                    GeometryType::BackwardsStep => {
                                        if cy > 0.5 {
                                            solver.u.vx[i] = 1.0;
                                            solver.u.vy[i] = 0.0;
                                        }
                                    }
                                    GeometryType::ChannelObstacle => {
                                        solver.u.vx[i] = 1.0;
                                        solver.u.vy[i] = 0.0;
                                    }
                                }
                            }
                        }
                    }
                    self.cpu_solver = Some(CpuSolverWrapper::ParallelF32(ps));
                }
                Precision::F64 => {
                    let mut ps = ParallelPisoSolver::<f64>::new(mesh, self.n_threads);
                    for solver in &mut ps.partitions {
                        let mut solver = solver.write().unwrap();
                        solver.dt = self.timestep;
                        solver.density = self.current_fluid.density;
                        solver.viscosity = self.current_fluid.viscosity;
                        solver.scheme = self.selected_scheme;
                        let n_cells = solver.mesh.num_cells();
                        for i in 0..n_cells {
                            let cx = solver.mesh.cell_cx[i];
                            let cy = solver.mesh.cell_cy[i];
                            if cx < self.max_cell_size {
                                match self.selected_geometry {
                                    GeometryType::BackwardsStep => {
                                        if cy > 0.5 {
                                            solver.u.vx[i] = 1.0;
                                            solver.u.vy[i] = 0.0;
                                        }
                                    }
                                    GeometryType::ChannelObstacle => {
                                        solver.u.vx[i] = 1.0;
                                        solver.u.vy[i] = 0.0;
                                    }
                                }
                            }
                        }
                    }
                    self.cpu_solver = Some(CpuSolverWrapper::ParallelF64(ps));
                }
            }
        } else {
            self.gpu_solver = None;
            match self.precision {
                Precision::F32 => {
                    let mut solver = PisoSolver::<f32>::new(mesh);
                    solver.dt = self.timestep as f32;
                    solver.density = self.current_fluid.density as f32;
                    solver.viscosity = self.current_fluid.viscosity as f32;
                    solver.scheme = self.selected_scheme;

                    for i in 0..solver.mesh.num_cells() {
                        let cx = solver.mesh.cell_cx[i];
                        let cy = solver.mesh.cell_cy[i];
                        if cx < self.max_cell_size {
                            match self.selected_geometry {
                                GeometryType::BackwardsStep => {
                                    if cy > 0.5 {
                                        solver.u.vx[i] = 1.0;
                                        solver.u.vy[i] = 0.0;
                                    }
                                }
                                GeometryType::ChannelObstacle => {
                                    solver.u.vx[i] = 1.0;
                                    solver.u.vy[i] = 0.0;
                                }
                            }
                        }
                    }
                    self.cpu_solver = Some(CpuSolverWrapper::SerialF32(solver));
                }
                Precision::F64 => {
                    let mut solver = PisoSolver::<f64>::new(mesh);
                    solver.dt = self.timestep;
                    solver.density = self.current_fluid.density;
                    solver.viscosity = self.current_fluid.viscosity;
                    solver.scheme = self.selected_scheme;

                    for i in 0..solver.mesh.num_cells() {
                        let cx = solver.mesh.cell_cx[i];
                        let cy = solver.mesh.cell_cy[i];
                        if cx < self.max_cell_size {
                            match self.selected_geometry {
                                GeometryType::BackwardsStep => {
                                    if cy > 0.5 {
                                        solver.u.vx[i] = 1.0;
                                        solver.u.vy[i] = 0.0;
                                    }
                                }
                                GeometryType::ChannelObstacle => {
                                    solver.u.vx[i] = 1.0;
                                    solver.u.vy[i] = 0.0;
                                }
                            }
                        }
                    }
                    self.cpu_solver = Some(CpuSolverWrapper::SerialF64(solver));
                }
            }
        }
    }
}

impl eframe::App for CFDApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("controls").show(ctx, |ui| {
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
                                if let Some(solver) = &mut self.cpu_solver {
                                    solver.update_fluid(
                                        self.current_fluid.density,
                                        self.current_fluid.viscosity,
                                    );
                                }
                            }
                        }
                    });

                let mut density = self.current_fluid.density;
                if ui
                    .add(egui::Slider::new(&mut density, 0.1..=20000.0).text("Density (kg/m³)"))
                    .changed()
                {
                    self.current_fluid.density = density;
                    self.current_fluid.name = "Custom".to_string();
                    if let Some(solver) = &mut self.cpu_solver {
                        solver.update_fluid(density, self.current_fluid.viscosity);
                    }
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
                    if let Some(solver) = &mut self.cpu_solver {
                        solver.update_fluid(self.current_fluid.density, viscosity);
                    }
                }
            });

            ui.group(|ui| {
                ui.label("Solver Parameters");

                // Calculate recommended timestep for stability (CFL < 0.5)
                let recommended_dt = 0.5 * self.min_cell_size; // CFL = u*dt/dx < 0.5 => dt < 0.5*dx (assuming u=1)
                let cfl = self.timestep / self.min_cell_size; // Approximate CFL

                ui.add(egui::Slider::new(&mut self.timestep, 0.0001..=0.1).text("Timestep"));

                // Show CFL warning
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
                    ui.colored_label(egui::Color32::YELLOW, format!("CFL≈{:.2} (moderate)", cfl));
                }

                ui.horizontal(|ui| {
                    ui.label("Precision:");
                    ui.radio_value(&mut self.precision, Precision::F32, "f32");
                    ui.radio_value(&mut self.precision, Precision::F64, "f64");
                });

                ui.checkbox(&mut self.use_parallel, "Use Parallel Solver");
                if self.use_parallel {
                    ui.add(egui::Slider::new(&mut self.n_threads, 1..=16).text("Threads"));
                }

                ui.checkbox(&mut self.use_gpu, "Use GPU Solver");

                ui.label("Discretization Scheme");

                if ui
                    .radio(matches!(self.selected_scheme, Scheme::Upwind), "Upwind")
                    .clicked()
                {
                    self.selected_scheme = Scheme::Upwind;
                    if let Some(solver) = &self.gpu_solver {
                        if let Ok(mut s) = solver.lock() {
                            s.set_scheme(0);
                        }
                    }
                }
                if ui
                    .radio(
                        matches!(self.selected_scheme, Scheme::Central),
                        "Central (2nd Order)",
                    )
                    .clicked()
                {
                    self.selected_scheme = Scheme::Central;
                    if let Some(solver) = &self.gpu_solver {
                        if let Ok(mut s) = solver.lock() {
                            s.set_scheme(1);
                        }
                    }
                }
                if ui
                    .radio(matches!(self.selected_scheme, Scheme::QUICK), "QUICK")
                    .clicked()
                {
                    self.selected_scheme = Scheme::QUICK;
                    if let Some(solver) = &self.gpu_solver {
                        if let Ok(mut s) = solver.lock() {
                            s.set_scheme(2);
                        }
                    }
                }
            });

            if ui.button("Initialize / Reset").clicked() {
                self.init_solver();
            }

            if self.cpu_solver.is_some() || self.gpu_solver.is_some() {
                if ui
                    .button(if self.is_running { "Pause" } else { "Run" })
                    .clicked()
                {
                    self.is_running = !self.is_running;

                    if let Some(gpu_solver) = &self.gpu_solver {
                        if self.is_running {
                            self.gpu_solver_running.store(true, Ordering::Relaxed);
                            let solver_arc = gpu_solver.clone();
                            let running_flag = self.gpu_solver_running.clone();
                            let shared_results = self.shared_results.clone();
                            let shared_gpu_stats = self.shared_gpu_stats.clone();
                            let ctx_clone = ctx.clone();
                            thread::spawn(move || {
                                while running_flag.load(Ordering::Relaxed) {
                                    if let Ok(mut solver) = solver_arc.lock() {
                                        solver.step();
                                        // Fetch results while holding the lock
                                        let u = pollster::block_on(solver.get_u());
                                        let p = pollster::block_on(solver.get_p());

                                        // Capture stats while we have the lock
                                        let stats = CachedGpuStats {
                                            time: solver.constants.time,
                                            stats_ux: *solver.stats_ux.lock().unwrap(),
                                            stats_uy: *solver.stats_uy.lock().unwrap(),
                                            stats_p: *solver.stats_p.lock().unwrap(),
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
            }

            ui.separator();

            ui.label("Plot Field");
            ui.radio_value(&mut self.plot_field, PlotField::Pressure, "Pressure");
            ui.radio_value(&mut self.plot_field, PlotField::VelocityX, "Velocity X");
            ui.radio_value(&mut self.plot_field, PlotField::VelocityY, "Velocity Y");
            ui.radio_value(&mut self.plot_field, PlotField::VelocityMag, "Velocity Mag");

            ui.separator();
            ui.checkbox(&mut self.show_mesh_lines, "Show Mesh Lines");

            ui.separator();

            if let Some(solver) = &self.cpu_solver {
                ui.label(format!("Time: {:.3}", solver.time()));
                ui.label("Residuals:");
                for (name, val) in solver.residuals() {
                    ui.label(format!("{}: {:.6}", name, val));
                }
            } else if self.gpu_solver.is_some() {
                // Update cached stats from shared data
                if let Ok(stats) = self.shared_gpu_stats.try_lock() {
                    self.cached_gpu_stats = stats.clone();
                }

                let stats = &self.cached_gpu_stats;
                ui.label(format!("Time: {:.3}", stats.time));
                ui.label("GPU Solver Stats:");
                ui.label(format!(
                    "Ux: {} iters, res {:.2e}, {:.2?}",
                    stats.stats_ux.iterations, stats.stats_ux.residual, stats.stats_ux.time
                ));
                ui.label(format!(
                    "Uy: {} iters, res {:.2e}, {:.2?}",
                    stats.stats_uy.iterations, stats.stats_uy.residual, stats.stats_uy.time
                ));
                ui.label(format!(
                    "P:  {} iters, res {:.2e}, {:.2?}",
                    stats.stats_p.iterations, stats.stats_p.residual, stats.stats_p.time
                ));
            }
        });

        // Calculate stats for plot and legend
        let (min_val, max_val, values) = if self.gpu_solver.is_some() {
            // GPU mode: Fetch from shared results
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
                    if val < min_val {
                        min_val = val;
                    }
                    if val > max_val {
                        max_val = val;
                    }
                    values.push(val);
                }

                if (max_val - min_val).abs() < 1e-6 {
                    max_val = min_val + 1.0;
                }
                (min_val, max_val, Some(values))
            } else {
                (0.0, 1.0, None)
            }
        } else if let Some(solver) = &self.cpu_solver {
            // CPU mode
            let (u, p) = solver.get_results();
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
                if val < min_val {
                    min_val = val;
                }
                if val > max_val {
                    max_val = val;
                }
                values.push(val);
            }

            if (max_val - min_val).abs() < 1e-6 {
                max_val = min_val + 1.0;
            }
            (min_val, max_val, Some(values))
        } else {
            (0.0, 1.0, None)
        };

        egui::SidePanel::right("legend").show(ctx, |ui| {
            // Show mesh stats from cpu_solver or self.mesh (for GPU mode)
            if let Some(solver) = &self.cpu_solver {
                ui.heading("Mesh Stats");
                let mesh = solver.get_mesh();
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
            } else if let Some(mesh) = &self.mesh {
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

            if self.cpu_solver.is_some() || self.gpu_solver.is_some() {
                ui.heading("Legend");
                ui.label(format!("Max: {:.4}", max_val));

                let (rect, _response) =
                    ui.allocate_at_least(egui::vec2(30.0, 200.0), egui::Sense::hover());
                if ui.is_rect_visible(rect) {
                    let mut mesh = egui::Mesh::default();
                    let n_steps = 20;
                    for i in 0..n_steps {
                        let t0 = i as f32 / n_steps as f32;
                        let t1 = (i + 1) as f32 / n_steps as f32;

                        let y0 = rect.max.y - t0 * rect.height();
                        let y1 = rect.max.y - t1 * rect.height();

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

        egui::CentralPanel::default().show(ctx, |ui| {
            if self.is_running {
                if self.gpu_solver.is_some() {
                    // GPU solver runs in background thread
                } else if let Some(solver) = &mut self.cpu_solver {
                    solver.step();
                    ctx.request_repaint();
                }
            }

            if let Some(solver) = &self.cpu_solver {
                if let Some(vals) = &values {
                    Plot::new("cfd_plot").data_aspect(1.0).show(ui, |plot_ui| {
                        let cells = solver.get_all_cells_vertices();
                        for (i, polygon_points) in cells.iter().enumerate() {
                            let val = vals[i];
                            let t = (val - min_val) / (max_val - min_val);

                            let color = get_color(t);

                            plot_ui.polygon(
                                Polygon::new(PlotPoints::new(polygon_points.clone()))
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
            } else if self.gpu_solver.is_some() && !self.cached_cells.is_empty() {
                // GPU mode: use cached_cells for plotting
                if let Some(vals) = &values {
                    Plot::new("cfd_plot").data_aspect(1.0).show(ui, |plot_ui| {
                        for (i, polygon_points) in self.cached_cells.iter().enumerate() {
                            let val = vals[i];
                            let t = (val - min_val) / (max_val - min_val);

                            let color = get_color(t);

                            plot_ui.polygon(
                                Polygon::new(PlotPoints::new(polygon_points.clone()))
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
    // Simple Rainbow Map: Blue -> Green -> Red
    let (r, g, b) = if t < 0.5 {
        // Blue to Green
        (0.0, (t * 2.0), (1.0 - t * 2.0))
    } else {
        // Green to Red
        ((t - 0.5) * 2.0, (1.0 - (t - 0.5) * 2.0), 0.0)
    };

    egui::Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}
