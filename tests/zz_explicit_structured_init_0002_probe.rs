//! Probe: why does initializing the explicit structured solver with cell size
//! 0.002 fail on the obstacle case?
//!
//! Run with:
//!   cargo test --release --features "dev-tests cpu meshgen ui" \
//!     --test zz_explicit_structured_init_0002_probe -- --nocapture --exact
#![cfg(all(
    feature = "dev-tests",
    feature = "cpu",
    feature = "meshgen",
    feature = "ui"
))]

use cfd2::ui::app::{
    gui_explicit_rk4_gpu_available, gui_explicit_rk4_smoke, GuiExplicitRk4Case,
};

fn run_presented(
    model_id: &str,
    backend: &str,
    cell_size: f64,
    adaptive: bool,
    steps: usize,
    presentation: &str,
) {
    let case = GuiExplicitRk4Case {
        filter_sigma: None,
        model_id,
        fluid: "Air",
        geometry: "obstacle",
        mesh_kind: "structured",
        backend,
        adaptive,
        presentation,
        moving_mesh: false,
        cell_size,
        steps,
        requested_dt: None,
        advection_scheme: None,
        inlet_velocity: None,
        inlet_pressure: None,
    };
    let started = std::time::Instant::now();
    let result = gui_explicit_rk4_smoke(case);
    let elapsed = started.elapsed();
    match result {
        Ok(smoke) => eprintln!(
            "OK   {model_id} {backend}/{presentation} cell={cell_size} adaptive={adaptive} cells={} dt=[{:.3e},{:.3e}] t={:.3e} ({elapsed:.1?})",
            smoke.cells, smoke.min_dt, smoke.max_dt, smoke.final_time
        ),
        Err(error) => panic!(
            "FAIL {model_id} {backend}/{presentation} cell={cell_size} adaptive={adaptive}: {error} ({elapsed:.1?})"
        ),
    }
}

fn run(model_id: &str, backend: &str, cell_size: f64, adaptive: bool, steps: usize) {
    run_presented(model_id, backend, cell_size, adaptive, steps, "plot");
}

#[test]
fn probe_cell_size_sweep_gpu() {
    if let Err(error) = gui_explicit_rk4_gpu_available() {
        eprintln!("SKIP gpu probe: {error}");
        return;
    }
    // The direct (autonomous) cases at 0.004/0.002 exceed 65,535 workgroups in
    // the control kernels' old flat 1-D dispatch (history/rollback run one
    // thread per state word); the 2D-split + `launch_index` flatten must keep
    // every one of them accepted.
    let cases: &[(&str, &str, f64)] = &[
        ("allmach_thermal_structured", "plot", 0.002),
        ("allmach_thermal_structured", "direct", 0.01),
        ("allmach_thermal_structured", "direct", 0.004),
        ("compressible_structured", "direct", 0.01),
        ("compressible_structured", "direct", 0.004),
        ("allmach_thermal_structured", "direct", 0.002),
        ("compressible_structured", "direct", 0.002),
    ];
    for &(model_id, presentation, cell_size) in cases {
        run_presented(model_id, "gpu", cell_size, false, 2, presentation);
    }
    // Adaptive Direct additionally runs the spectral sampling kernels
    // (pressure_gradient / sample_rhie_chow) each step alongside the split
    // history/rollback dispatches.
    for model_id in ["allmach_thermal_structured", "compressible_structured"] {
        run_presented(model_id, "gpu", 0.002, true, 2, "direct");
    }
}

/// Above the split threshold the autonomous history/rollback kernels run the
/// 2D-flattened dispatch (groups_y > 1); their physics must still match the
/// ordinary Plot route, which advances history via plain buffer copies.
#[test]
fn probe_direct_matches_plot_above_split_threshold() {
    if let Err(error) = gui_explicit_rk4_gpu_available() {
        eprintln!("SKIP gpu probe: {error}");
        return;
    }
    for model_id in ["allmach_thermal_structured", "compressible_structured"] {
        let smoke = |presentation| {
            gui_explicit_rk4_smoke(GuiExplicitRk4Case {
        filter_sigma: None,
                model_id,
                fluid: "Air",
                geometry: "obstacle",
                mesh_kind: "structured",
                backend: "gpu",
                adaptive: false,
                presentation,
                moving_mesh: false,
                cell_size: 0.002,
                steps: 3,
                requested_dt: None,
                advection_scheme: None,
                inlet_velocity: None,
                inlet_pressure: None,
            })
            .unwrap_or_else(|error| panic!("{model_id}/{presentation}: {error}"))
        };
        let plot = smoke("plot");
        let direct = smoke("direct");
        assert_eq!(plot.route, "ordinary");
        assert_eq!(direct.route, "autonomous", "{model_id}: Direct fell back");
        assert_eq!(plot.packed_state.len(), direct.packed_state.len());
        let mut linf = 0.0_f64;
        let mut scale = 0.0_f64;
        for (a, b) in plot.packed_state.iter().zip(&direct.packed_state) {
            linf = linf.max((f64::from(*a) - f64::from(*b)).abs());
            scale = scale.max(f64::from(a.abs())).max(f64::from(b.abs()));
        }
        let normalized = if scale > 0.0 { linf / scale } else { linf };
        eprintln!(
            "OK   {model_id} plot-vs-direct cells={} normalized Linf={normalized:e}",
            plot.cells
        );
        assert!(
            normalized <= 5.0e-4,
            "{model_id}: Direct/Plot state drift above split threshold: normalized Linf={normalized:e}"
        );
    }
}

#[test]
fn probe_cell_size_sweep_cpu() {
    for model_id in ["allmach_thermal_structured", "compressible_structured"] {
        for cell_size in [0.25, 0.01, 0.005, 0.002] {
            run(model_id, "cpu-transpiled", cell_size, false, 2);
        }
    }
}

#[test]
fn probe_cell_size_0002_adaptive_cpu() {
    for model_id in ["allmach_thermal_structured", "compressible_structured"] {
        run(model_id, "cpu-transpiled", 0.002, true, 2);
    }
}
