use cfd2::ui::app::CFDApp;
use eframe::egui;
use std::sync::Arc;

fn main() -> Result<(), eframe::Error> {
    env_logger::init();
    // Present WITHOUT vsync by default: with vsync, the render thread blocks
    // inside the surface acquire waiting for the next drawable while holding a
    // wgpu device lock, and every solver-worker `queue.submit` convoys behind
    // it (~8 ms each, one blocked submit per in-flight batch per frame). That
    // quantizes autonomous GPU batches to the display refresh and costs ~8x
    // solver throughput. The app already paces rendering itself via
    // `request_repaint_after(16ms)`, so presentation stays at ~refresh rate
    // while the solver runs free. CFD2_VSYNC=1 restores synced presentation.
    let present_mode = if std::env::var_os("CFD2_VSYNC").is_some() {
        wgpu::PresentMode::AutoVsync
    } else {
        wgpu::PresentMode::AutoNoVsync
    };
    let mut wgpu_options = eframe::egui_wgpu::WgpuConfiguration::default();
    wgpu_options.surface.present_mode = present_mode;
    if let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &mut wgpu_options.wgpu_setup {
        setup.device_descriptor = Arc::new(|adapter| {
            let adapter_limits = adapter.limits();
            let mut limits = adapter_limits.clone();
            limits.max_storage_buffers_per_shader_stage = 31;

            wgpu::DeviceDescriptor {
                label: Some("CFD Device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                trace: wgpu::Trace::Off,
            }
        });
    }
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 720.0]),
        wgpu_options,
        ..Default::default()
    };
    eframe::run_native(
        "2D CFD Solver (CutCell)",
        options,
        Box::new(|cc| Ok(Box::new(CFDApp::new(cc)))),
    )
}
