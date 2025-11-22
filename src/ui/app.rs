use eframe::egui;
use egui_plot::{Plot, Polygon, PlotPoints};
use crate::solver::piso::PisoSolver;
use crate::solver::mesh::{generate_cut_cell_mesh, BackwardsStep, ChannelWithObstacle};
use crate::solver::fvm::Scheme;
use nalgebra::{Vector2, Point2};

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
            Fluid { name: "Water".into(), density: 1000.0, viscosity: 0.001 },
            Fluid { name: "Air".into(), density: 1.225, viscosity: 1.81e-5 },
            Fluid { name: "Alcohol".into(), density: 789.0, viscosity: 0.0012 },
            Fluid { name: "Kerosene".into(), density: 820.0, viscosity: 0.00164 },
            Fluid { name: "Mercury".into(), density: 13546.0, viscosity: 0.001526 },
            Fluid { name: "Custom".into(), density: 1.0, viscosity: 0.01 },
        ]
    }
}

use crate::solver::parallel::ParallelPisoSolver;

pub struct CFDApp {
    solver: Option<PisoSolver>,
    parallel_solver: Option<ParallelPisoSolver>,
    use_parallel: bool,
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
            solver: None,
            parallel_solver: None,
            use_parallel: false,
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
                let mut mesh = generate_cut_cell_mesh(&geo, self.min_cell_size, self.max_cell_size, domain_size);
                mesh.smooth(0.3, 50);
                mesh
            },
            GeometryType::ChannelObstacle => {
                let length = 3.0;
                let domain_size = Vector2::new(length, 1.0);
                let geo = ChannelWithObstacle {
                    length,
                    height: 1.0,
                    obstacle_center: Point2::new(1.0, 0.5),
                    obstacle_radius: 0.2,
                };
                let mut mesh = generate_cut_cell_mesh(&geo, self.min_cell_size, self.max_cell_size, domain_size);
                mesh.smooth(0.3, 50);
                mesh
            },
        };
        
        if self.use_parallel {
            self.parallel_solver = Some(ParallelPisoSolver::new(mesh, self.n_threads));
            self.solver = None;
            // Initialize solver params for all sub-solvers
            if let Some(ps) = &mut self.parallel_solver {
                for solver in &mut ps.partitions {
                    let mut solver = solver.write().unwrap();
                    solver.dt = self.timestep;
                    solver.density = self.current_fluid.density;
                    solver.viscosity = self.current_fluid.viscosity;
                    solver.scheme = self.selected_scheme;
                    // BCs
                    let n_cells = solver.mesh.num_cells();
                    for i in 0..n_cells {
                        let cx = solver.mesh.cell_cx[i];
                        let cy = solver.mesh.cell_cy[i];
                        // Check if cell is in inlet region (global check needed or local?)
                        // Since we partitioned by X, inlet is likely in rank 0.
                        // But we should check coordinates.
                        if cx < self.max_cell_size {
                             match self.selected_geometry {
                                GeometryType::BackwardsStep => {
                                    if cy > 0.5 {
                                        solver.u.vx[i] = 1.0;
                                        solver.u.vy[i] = 0.0;
                                    }
                                },
                                GeometryType::ChannelObstacle => {
                                    solver.u.vx[i] = 1.0;
                                    solver.u.vy[i] = 0.0;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            let mut solver = PisoSolver::new(mesh);
            solver.dt = self.timestep;
            solver.density = self.current_fluid.density;
            solver.viscosity = self.current_fluid.viscosity;
            solver.scheme = match self.selected_scheme {
                Scheme::Upwind => Scheme::Upwind,
                Scheme::Central => Scheme::Central,
                Scheme::QUICK => Scheme::QUICK,
            };
            
            // Set initial BCs (Inlet velocity)
            for i in 0..solver.mesh.num_cells() {
                let cx = solver.mesh.cell_cx[i];
                let cy = solver.mesh.cell_cy[i];
                if cx < self.max_cell_size {
                    // Inlet region
                    match self.selected_geometry {
                        GeometryType::BackwardsStep => {
                            if cy > 0.5 {
                                solver.u.vx[i] = 1.0;
                                solver.u.vy[i] = 0.0;
                            }
                        },
                        GeometryType::ChannelObstacle => {
                            solver.u.vx[i] = 1.0;
                            solver.u.vy[i] = 0.0;
                        }
                    }
                }
            }
            
            self.solver = Some(solver);
            self.parallel_solver = None;
        }
    }
}

impl eframe::App for CFDApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("controls").show(ctx, |ui| {
            ui.heading("CFD Controls");
            
            ui.group(|ui| {
                ui.label("Geometry");
                ui.radio_value(&mut self.selected_geometry, GeometryType::BackwardsStep, "Backwards Step");
                ui.radio_value(&mut self.selected_geometry, GeometryType::ChannelObstacle, "Channel w/ Obstacle");
            });
            
            ui.group(|ui| {
                ui.label("Mesh Parameters");
                ui.add(egui::Slider::new(&mut self.min_cell_size, 0.001..=self.max_cell_size).text("Min Cell Size"));
                ui.add(egui::Slider::new(&mut self.max_cell_size, self.min_cell_size..=0.5).text("Max Cell Size"));
            });
            
            ui.group(|ui| {
                ui.label("Fluid Properties");
                egui::ComboBox::from_label("Preset")
                    .selected_text(&self.current_fluid.name)
                    .show_ui(ui, |ui| {
                        for fluid in Fluid::presets() {
                            if ui.selectable_value(&mut self.current_fluid, fluid.clone(), &fluid.name).clicked() {
                                if let Some(solver) = &mut self.solver {
                                    solver.density = self.current_fluid.density;
                                    solver.viscosity = self.current_fluid.viscosity;
                                }
                            }
                        }
                    });
                
                let mut density = self.current_fluid.density;
                if ui.add(egui::Slider::new(&mut density, 0.1..=20000.0).text("Density (kg/m³)")).changed() {
                    self.current_fluid.density = density;
                    self.current_fluid.name = "Custom".to_string();
                    if let Some(solver) = &mut self.solver {
                        solver.density = density;
                    }
                }
                
                let mut viscosity = self.current_fluid.viscosity;
                if ui.add(egui::Slider::new(&mut viscosity, 1e-6..=0.1).logarithmic(true).text("Viscosity (Pa·s)")).changed() {
                    self.current_fluid.viscosity = viscosity;
                    self.current_fluid.name = "Custom".to_string();
                    if let Some(solver) = &mut self.solver {
                        solver.viscosity = viscosity;
                    }
                }
            });

            ui.group(|ui| {
                ui.label("Solver Parameters");
                ui.add(egui::Slider::new(&mut self.timestep, 0.001..=0.1).text("Timestep"));
                
                ui.checkbox(&mut self.use_parallel, "Use Parallel Solver");
                if self.use_parallel {
                    ui.add(egui::Slider::new(&mut self.n_threads, 1..=16).text("Threads"));
                }

                ui.label("Discretization Scheme");
                
                if ui.radio(matches!(self.selected_scheme, Scheme::Upwind), "Upwind").clicked() {
                    self.selected_scheme = Scheme::Upwind;
                }
                if ui.radio(matches!(self.selected_scheme, Scheme::Central), "Central (2nd Order)").clicked() {
                    self.selected_scheme = Scheme::Central;
                }
                if ui.radio(matches!(self.selected_scheme, Scheme::QUICK), "QUICK").clicked() {
                    self.selected_scheme = Scheme::QUICK;
                }
            });
            
            if ui.button("Initialize / Reset").clicked() {
                self.init_solver();
            }
            
            if self.solver.is_some() || self.parallel_solver.is_some() {
                if ui.button(if self.is_running { "Pause" } else { "Run" }).clicked() {
                    self.is_running = !self.is_running;
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
            
            if let Some(solver) = &self.solver {
                ui.label(format!("Time: {:.3}", solver.time));
                ui.label("Residuals:");
                for (name, val) in &solver.residuals {
                    ui.label(format!("{}: {:.6}", name, val));
                }
            } else if let Some(ps) = &self.parallel_solver {
                // Show residuals from rank 0
                if !ps.partitions.is_empty() {
                    let solver = ps.partitions[0].read().unwrap();
                    ui.label(format!("Time: {:.3}", solver.time));
                    ui.label("Residuals (Rank 0):");
                    for (name, val) in &solver.residuals {
                        ui.label(format!("{}: {:.6}", name, val));
                    }
                }
            }
        });

        // Calculate stats for plot and legend
        let (min_val, max_val, values) = if let Some(solver) = &self.solver {
            let mut min_val = f64::MAX;
            let mut max_val = f64::MIN;
            let mut values = Vec::with_capacity(solver.mesh.num_cells());
            
            for i in 0..solver.mesh.num_cells() {
                    let val = match self.plot_field {
                    PlotField::Pressure => solver.p.values[i],
                    PlotField::VelocityX => solver.u.vx[i],
                    PlotField::VelocityY => solver.u.vy[i],
                    PlotField::VelocityMag => (solver.u.vx[i].powi(2) + solver.u.vy[i].powi(2)).sqrt(),
                };
                if val < min_val { min_val = val; }
                if val > max_val { max_val = val; }
                values.push(val);
            }
            
            if (max_val - min_val).abs() < 1e-6 {
                max_val = min_val + 1.0; // Avoid division by zero
            }
            (min_val, max_val, Some(values))
        } else if let Some(ps) = &self.parallel_solver {
            // Aggregate values from all solvers
            let mut min_val = f64::MAX;
            let mut max_val = f64::MIN;
            
            for solver in &ps.partitions {
                let solver = solver.read().unwrap();
                for i in 0..solver.mesh.num_cells() {
                    let val = match self.plot_field {
                        PlotField::Pressure => solver.p.values[i],
                        PlotField::VelocityX => solver.u.vx[i],
                        PlotField::VelocityY => solver.u.vy[i],
                        PlotField::VelocityMag => (solver.u.vx[i].powi(2) + solver.u.vy[i].powi(2)).sqrt(),
                    };
                    if val < min_val { min_val = val; }
                    if val > max_val { max_val = val; }
                }
            }
             if (max_val - min_val).abs() < 1e-6 {
                max_val = min_val + 1.0;
            }
            (min_val, max_val, None) // None because we handle values in loop
        } else {
            (0.0, 1.0, None)
        };

        egui::SidePanel::right("legend").show(ctx, |ui| {
            if let Some(solver) = &self.solver {
                ui.heading("Mesh Stats");
                ui.label(format!("Cells: {}", solver.mesh.num_cells()));
                ui.label(format!("Max Skewness: {:.4}", solver.mesh.calculate_max_skewness()));
                ui.separator();
            } else if let Some(ps) = &self.parallel_solver {
                ui.heading("Mesh Stats");
                let total_cells: usize = ps.partitions.iter().map(|s| s.read().unwrap().mesh.num_cells()).sum();
                ui.label(format!("Total Cells: {}", total_cells));
                let max_skew = ps.partitions.iter()
                    .map(|s| s.read().unwrap().mesh.calculate_max_skewness())
                    .fold(0.0, f64::max);
                ui.label(format!("Max Skewness: {:.4}", max_skew));
                ui.separator();
            }

            if self.solver.is_some() || self.parallel_solver.is_some() {
                ui.heading("Legend");
                ui.label(format!("Max: {:.4}", max_val));
                
                let (rect, _response) = ui.allocate_at_least(egui::vec2(30.0, 200.0), egui::Sense::hover());
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
                                egui::pos2(rect.max.x, y0)
                            ),
                            c0 // Simplified gradient (flat shading per segment, or use vertex colors if Mesh supported gradients better here)
                        );
                        // Actually Mesh supports vertex colors.
                        // But add_colored_rect uses one color.
                        // Let's just use small rects.
                    }
                    ui.painter().add(mesh);
                }
                
                ui.label(format!("Min: {:.4}", min_val));
            }
        });
        
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.is_running {
                if let Some(solver) = &mut self.solver {
                    solver.step();
                    ctx.request_repaint();
                } else if let Some(ps) = &mut self.parallel_solver {
                    ps.step();
                    ctx.request_repaint();
                }
            }
            
            if let Some(solver) = &self.solver {
                if let Some(vals) = &values {
                    Plot::new("cfd_plot")
                        .data_aspect(1.0)
                        .show(ui, |plot_ui| {
                            for i in 0..solver.mesh.num_cells() {
                                let val = vals[i];
                                let t = (val - min_val) / (max_val - min_val);
                                
                                let color = get_color(t);
                                
                                let start = solver.mesh.cell_vertex_offsets[i];
                                let end = solver.mesh.cell_vertex_offsets[i+1];
                                let polygon_points: Vec<[f64; 2]> = (start..end)
                                    .map(|k| {
                                        let v_idx = solver.mesh.cell_vertices[k];
                                        [solver.mesh.vx[v_idx], solver.mesh.vy[v_idx]]
                                    })
                                    .collect();

                                plot_ui.polygon(
                                    Polygon::new(PlotPoints::new(polygon_points))
                                        .fill_color(color)
                                        .stroke(if self.show_mesh_lines {
                                            egui::Stroke::new(1.0, egui::Color32::BLACK)
                                        } else {
                                            egui::Stroke::NONE
                                        })
                                );
                            }
                            
                            if let Some(pointer) = plot_ui.pointer_coordinate() {
                                let p = Point2::new(pointer.x, pointer.y);
                                if let Some(idx) = solver.mesh.get_cell_at_pos(p) {
                                    let skew = solver.mesh.calculate_cell_skewness(idx);
                                    let val = vals[idx];
                                    let vol = solver.mesh.cell_vol[idx];
                                    plot_ui.text(
                                        egui_plot::Text::new(
                                            pointer, 
                                            format!("Cell {}\nVol: {:.6e}\nSkew: {:.4}\nVal: {:.4}", idx, vol, skew, val)
                                        )
                                        .color(egui::Color32::WHITE)
                                        .anchor(egui::Align2::LEFT_BOTTOM)
                                    );
                                }
                            }
                        });
                }
            } else if let Some(ps) = &self.parallel_solver {
                Plot::new("cfd_plot")
                    .data_aspect(1.0)
                    .show(ui, |plot_ui| {
                        for solver in &ps.partitions {
                            let solver = solver.read().unwrap();
                            for i in 0..solver.mesh.num_cells() {
                                let val = match self.plot_field {
                                    PlotField::Pressure => solver.p.values[i],
                                    PlotField::VelocityX => solver.u.vx[i],
                                    PlotField::VelocityY => solver.u.vy[i],
                                    PlotField::VelocityMag => (solver.u.vx[i].powi(2) + solver.u.vy[i].powi(2)).sqrt(),
                                };
                                let t = (val - min_val) / (max_val - min_val);
                                let color = get_color(t);
                                
                                let start = solver.mesh.cell_vertex_offsets[i];
                                let end = solver.mesh.cell_vertex_offsets[i+1];
                                let polygon_points: Vec<[f64; 2]> = (start..end)
                                    .map(|k| {
                                        let v_idx = solver.mesh.cell_vertices[k];
                                        [solver.mesh.vx[v_idx], solver.mesh.vy[v_idx]]
                                    })
                                    .collect();

                                plot_ui.polygon(
                                    Polygon::new(PlotPoints::new(polygon_points))
                                        .fill_color(color)
                                        .stroke(if self.show_mesh_lines {
                                            egui::Stroke::new(1.0, egui::Color32::BLACK)
                                        } else {
                                            egui::Stroke::NONE
                                        })
                                );
                            }
                        }
                        
                        if let Some(pointer) = plot_ui.pointer_coordinate() {
                            let p = Point2::new(pointer.x, pointer.y);
                            for (s_idx, solver) in ps.partitions.iter().enumerate() {
                                let solver = solver.read().unwrap();
                                if let Some(idx) = solver.mesh.get_cell_at_pos(p) {
                                    let skew = solver.mesh.calculate_cell_skewness(idx);
                                    let val = match self.plot_field {
                                        PlotField::Pressure => solver.p.values[idx],
                                        PlotField::VelocityX => solver.u.vx[idx],
                                        PlotField::VelocityY => solver.u.vy[idx],
                                        PlotField::VelocityMag => (solver.u.vx[idx].powi(2) + solver.u.vy[idx].powi(2)).sqrt(),
                                    };
                                    let vol = solver.mesh.cell_vol[idx];
                                    plot_ui.text(
                                        egui_plot::Text::new(
                                            pointer, 
                                            format!("Part {}\nCell {}\nVol: {:.6e}\nSkew: {:.4}\nVal: {:.4}", s_idx, idx, vol, skew, val)
                                        )
                                        .color(egui::Color32::WHITE)
                                        .anchor(egui::Align2::LEFT_BOTTOM)
                                    );
                                    break;
                                }
                            }
                        }
                    });
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
    
    egui::Color32::from_rgb(
        (r * 255.0) as u8,
        (g * 255.0) as u8,
        (b * 255.0) as u8
    )
}
