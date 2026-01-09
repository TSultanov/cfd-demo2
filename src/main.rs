use cfd2::ui::app::CFDApp;
use eframe::egui;
use std::sync::Arc;

fn main() -> Result<(), eframe::Error> {
    env_logger::init();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 720.0]),
        wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
            wgpu_setup: eframe::egui_wgpu::WgpuSetup::CreateNew(
                eframe::egui_wgpu::WgpuSetupCreateNew {
                    device_descriptor: Arc::new(|adapter| {
                        let adapter_limits = adapter.limits();
                        let mut limits = adapter_limits.clone();
                        limits.max_storage_buffers_per_shader_stage = 31;
                        // Ensure we don't exceed adapter limits (though we started with them)
                        // limits.max_buffer_size = adapter_limits.max_buffer_size;
                        // limits.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size;

                        wgpu::DeviceDescriptor {
                            label: Some("CFD Device"),
                            required_features: wgpu::Features::empty(),
                            required_limits: limits,
                            memory_hints: wgpu::MemoryHints::default(),
                            experimental_features: wgpu::ExperimentalFeatures::disabled(),
                            trace: wgpu::Trace::Off,
                        }
                    }),
                    ..Default::default()
                },
            ),
            ..Default::default()
        },
        ..Default::default()
    };
    eframe::run_native(
        "2D CFD Solver (CutCell)",
        options,
        Box::new(|cc| Ok(Box::new(CFDApp::new(cc)))),
    )
}
