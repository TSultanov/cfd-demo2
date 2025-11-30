use cfd2::ui::app::CFDApp;

fn main() -> Result<(), eframe::Error> {
    env_logger::init();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 720.0]),
        ..Default::default()
    };
    eframe::run_native(
        "2D CFD Solver (CutCell)",
        options,
        Box::new(|cc| Ok(Box::new(CFDApp::new(cc)))),
    )
}
