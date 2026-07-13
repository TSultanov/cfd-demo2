//! Headless gate for the explicit-RK4 combinations exposed by the real GUI.
//!
//! The matrix is intentionally split in two: every geometry/mesher is built and
//! advanced once, while every supported model is advanced with every compute
//! backend and both fixed/adaptive timestep policies on a canonical lean mesh.
//! Unsupported dropdown rows are asserted as rejections, never silently skipped.
#![cfg(all(
    feature = "dev-tests",
    feature = "cpu",
    feature = "meshgen",
    feature = "ui"
))]

use cfd2::ui::app::{
    gui_explicit_rk4_capability_matrix, gui_explicit_rk4_gpu_available,
    gui_explicit_rk4_smoke, GuiExplicitRk4Case, GuiExplicitRk4Smoke,
};

const UNSTRUCTURED_RK4: &[&str] =
    &["allmach_pressure", "allmach_thermal", "compressible"];
const STRUCTURED_RK4: &[&str] =
    &["allmach_thermal_structured", "compressible_structured"];
const CPU_BACKENDS: &[&str] = &[
    "cpu-interpreter",
    "cpu-transpiled",
    "cpu-transpiled-simd",
];

fn case<'a>(
    model_id: &'a str,
    geometry: &'a str,
    mesh_kind: &'a str,
    backend: &'a str,
    adaptive: bool,
    presentation: &'a str,
    steps: usize,
) -> GuiExplicitRk4Case<'a> {
    GuiExplicitRk4Case {
        model_id,
        fluid: "Air",
        geometry,
        mesh_kind,
        backend,
        adaptive,
        presentation,
        moving_mesh: false,
        cell_size: 0.25,
        steps,
        requested_dt: None,
        advection_scheme: None,
        inlet_velocity: None,
        inlet_pressure: None,
    }
}

fn assert_stable(smoke: &GuiExplicitRk4Smoke) {
    assert!(smoke.cells > 0, "empty GUI mesh: {smoke:?}");
    assert!(smoke.steps > 0);
    assert!(smoke.final_time.is_finite() && smoke.final_time > 0.0);
    assert!(smoke.min_dt.is_finite() && smoke.min_dt > 0.0);
    assert!(smoke.max_dt.is_finite() && smoke.max_dt >= smoke.min_dt);
    assert!(smoke.packed_state.iter().all(|value| value.is_finite()));
    assert!(smoke.min_rho.is_none_or(|value| value > 0.0));
    assert!(smoke.min_temperature.is_none_or(|value| value > 0.0));
    assert!(smoke.min_pressure.is_none_or(|value| value > 0.0));
    assert!(
        smoke
            .min_total_energy_density
            .is_none_or(|value| value.is_finite())
    );
    assert!(
        smoke
            .min_internal_energy_density
            .is_none_or(|value| value.is_finite())
    );
}

fn normalized_linf(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "presentation routes changed state shape");
    let mut difference = 0.0_f64;
    let mut scale = 1.0_f64;
    for (&a, &b) in a.iter().zip(b) {
        difference = difference.max(f64::from((a - b).abs()));
        scale = scale.max(f64::from(a.abs().max(b.abs())));
    }
    difference / scale
}

#[test]
fn actual_gui_model_inventory_has_five_rk4_rows_and_two_explicit_rejections() {
    let rows = gui_explicit_rk4_capability_matrix().expect("GUI capability inventory");
    let observed: Vec<_> = rows
        .iter()
        .map(|row| (row.topology, row.model_id, row.supported))
        .collect();
    assert_eq!(
        observed,
        [
            ("unstructured", "allmach_pressure", true),
            ("unstructured", "allmach_thermal", true),
            ("unstructured", "compressible", true),
            ("unstructured", "incompressible_momentum", false),
            ("structured", "incompressible_momentum_structured", false),
            ("structured", "allmach_thermal_structured", true),
            ("structured", "compressible_structured", true),
        ]
    );
    for rejected in rows.iter().filter(|row| !row.supported) {
        let reason = rejected
            .rejection
            .as_deref()
            .expect("a rejected GUI option needs a diagnostic");
        assert!(
            reason.contains("algebraic")
                || reason.contains("differential")
                || reason.contains("Rhie–Chow")
                || reason.contains("`d_p`"),
            "saddle-point RK4 rejection lost its mathematical reason: {rejected:?}"
        );
    }
}

#[test]
fn unsupported_gui_combinations_are_errors_not_skips() {
    for (model_id, mesh_kind) in [
        ("incompressible_momentum", "cutcell"),
        ("incompressible_momentum_structured", "structured"),
    ] {
        let error = gui_explicit_rk4_smoke(case(
            model_id,
            "backstep",
            mesh_kind,
            "cpu-interpreter",
            true,
            "plot",
            1,
        ))
        .expect_err("saddle-point RK4 must be rejected");
        assert!(error.contains("explicitly rejects RK4"), "{error}");
    }

    let mut moving = case(
        "allmach_thermal",
        "backstep",
        "cvt",
        "cpu-transpiled",
        true,
        "plot",
        1,
    );
    moving.moving_mesh = true;
    let error = gui_explicit_rk4_smoke(moving).expect_err("ALE RK4 must be rejected");
    assert!(error.contains("static-only") && error.contains("BDF2"), "{error}");

    let fitted = case(
        "allmach_thermal",
        "backstep",
        "fitted",
        "cpu-transpiled",
        true,
        "plot",
        1,
    );
    assert!(
        gui_explicit_rk4_smoke(fitted)
            .expect_err("fitted backstep must be rejected")
            .contains("nozzle-only")
    );

    let wrong_topology = case(
        "compressible_structured",
        "backstep",
        "cutcell",
        "cpu-transpiled",
        true,
        "plot",
        1,
    );
    assert!(
        gui_explicit_rk4_smoke(wrong_topology)
            .expect_err("structured model with unstructured mesher must fail")
            .contains("requires mesh_kind='structured'")
    );

    let mut unknown_fluid = case(
        "compressible",
        "backstep",
        "cutcell",
        "cpu-transpiled",
        true,
        "plot",
        1,
    );
    unknown_fluid.fluid = "Unobtainium";
    assert!(
        gui_explicit_rk4_smoke(unknown_fluid)
            .expect_err("an unknown fluid must not silently use Air")
            .contains("unknown GUI fluid preset")
    );
}

#[test]
fn all_sixteen_actual_gui_geometry_mesh_cases_construct_seed_and_advance() {
    const UNSTRUCTURED_CASES: &[(&str, &str)] = &[
        ("backstep", "cutcell"),
        ("backstep", "delaunay"),
        ("backstep", "voronoi"),
        ("backstep", "cvt"),
        ("obstacle", "cutcell"),
        ("obstacle", "delaunay"),
        ("obstacle", "voronoi"),
        ("obstacle", "cvt"),
        ("nozzle", "fitted"),
        ("nozzle", "cutcell"),
        ("nozzle", "delaunay"),
        ("nozzle", "voronoi"),
        ("nozzle", "cvt"),
    ];
    for &(geometry, mesh_kind) in UNSTRUCTURED_CASES {
        let smoke = gui_explicit_rk4_smoke(case(
            "allmach_thermal",
            geometry,
            mesh_kind,
            "cpu-transpiled",
            true,
            "plot",
            1,
        ))
        .unwrap_or_else(|error| panic!("{geometry}/{mesh_kind}: {error}"));
        assert_eq!(smoke.topology, "unstructured");
        assert_stable(&smoke);
    }
    for geometry in ["backstep", "obstacle", "nozzle"] {
        let smoke = gui_explicit_rk4_smoke(case(
            "allmach_thermal_structured",
            geometry,
            "structured",
            "cpu-transpiled",
            true,
            "plot",
            1,
        ))
        .unwrap_or_else(|error| panic!("structured/{geometry}: {error}"));
        assert_eq!(smoke.topology, "structured");
        assert_stable(&smoke);
    }
}

#[test]
fn every_cpu_dropdown_backend_runs_every_supported_model_fixed_and_adaptive() {
    for backend in CPU_BACKENDS {
        for model_id in UNSTRUCTURED_RK4
            .iter()
            .chain(STRUCTURED_RK4.iter())
            .copied()
        {
            let mesh_kind = if model_id.ends_with("_structured") {
                "structured"
            } else {
                "cutcell"
            };
            for adaptive in [false, true] {
                let plot = gui_explicit_rk4_smoke(case(
                    model_id,
                    "backstep",
                    mesh_kind,
                    backend,
                    adaptive,
                    "plot",
                    2,
                ))
                .unwrap_or_else(|error| {
                    panic!("{backend}/{model_id}/adaptive={adaptive}/Plot: {error}")
                });
                let direct = gui_explicit_rk4_smoke(case(
                    model_id,
                    "backstep",
                    mesh_kind,
                    backend,
                    adaptive,
                    "direct",
                    2,
                ))
                .unwrap_or_else(|error| {
                    panic!("{backend}/{model_id}/adaptive={adaptive}/Direct: {error}")
                });
                assert_stable(&plot);
                assert_stable(&direct);
                assert_eq!(plot.route, "ordinary");
                assert_eq!(direct.route, "ordinary");
                assert_eq!(
                    plot.packed_state, direct.packed_state,
                    "CPU presentation changed the mathematical trajectory for {backend}/{model_id}/adaptive={adaptive}"
                );
                assert_eq!(plot.final_time.to_bits(), direct.final_time.to_bits());
                assert_eq!(plot.min_dt.to_bits(), direct.min_dt.to_bits());
                assert_eq!(plot.max_dt.to_bits(), direct.max_dt.to_bits());
            }
        }
    }
}

#[test]
fn unstructured_adaptive_plot_and_direct_consume_each_sized_dt_once() {
    if let Err(error) = gui_explicit_rk4_gpu_available() {
        eprintln!("skipping GUI adaptive route parity: {error}");
        return;
    }
    let plot = gui_explicit_rk4_smoke(case(
        "allmach_pressure",
        "backstep",
        "cutcell",
        "gpu",
        true,
        "plot",
        3,
    ))
    .expect("ordinary Plot adaptive sequence");
    let direct = gui_explicit_rk4_smoke(case(
        "allmach_pressure",
        "backstep",
        "cutcell",
        "gpu",
        true,
        "direct",
        3,
    ))
    .expect("autonomous Direct adaptive sequence");

    // requested=5us; build/apply sizes the first accepted step once to 6us,
    // then the accepted-state controller grows 6 -> 7.2 -> 8.64us.
    let expected_time = 5.0e-6_f64 * 1.2 * (1.0 + 1.2 + 1.2 * 1.2);
    assert!(
        (plot.final_time - expected_time).abs() <= expected_time * 2.0e-6,
        "ordinary route double-sized an accepted state: {plot:?}"
    );
    assert!(
        (direct.final_time - expected_time).abs() <= expected_time * 2.0e-6,
        "autonomous route used a different dt sequence: {direct:?}"
    );
    assert_eq!(plot.route, "ordinary");
    assert_eq!(direct.route, "autonomous");
    assert!(
        normalized_linf(&plot.packed_state, &direct.packed_state) <= 2.0e-6,
        "accepted-state ownership changed the three-step RK4 trajectory"
    );
}

#[test]
fn gpu_direct_autonomous_and_plot_ordinary_are_stable_for_every_model_and_dt_policy() {
    if let Err(error) = gui_explicit_rk4_gpu_available() {
        eprintln!("skipping GUI GPU matrix: {error}");
        return;
    }
    for model_id in UNSTRUCTURED_RK4
        .iter()
        .chain(STRUCTURED_RK4.iter())
        .copied()
    {
        let mesh_kind = if model_id.ends_with("_structured") {
            "structured"
        } else {
            "cutcell"
        };
        for adaptive in [false, true] {
            // Three is one complete unstructured accepted-state/history cycle
            // and is also a useful multi-step structured controller probe.
            let plot = gui_explicit_rk4_smoke(case(
                model_id,
                "backstep",
                mesh_kind,
                "gpu",
                adaptive,
                "plot",
                3,
            ))
            .unwrap_or_else(|error| {
                panic!("GPU/{model_id}/adaptive={adaptive}/Plot: {error}")
            });
            let direct = gui_explicit_rk4_smoke(case(
                model_id,
                "backstep",
                mesh_kind,
                "gpu",
                adaptive,
                "direct",
                3,
            ))
            .unwrap_or_else(|error| {
                panic!("GPU/{model_id}/adaptive={adaptive}/Direct: {error}")
            });
            assert_stable(&plot);
            assert_stable(&direct);
            assert_eq!(plot.route, "ordinary");
            assert_eq!(
                direct.route, "autonomous",
                "Direct silently fell back for GPU/{model_id}/adaptive={adaptive}"
            );

            let time_scale = plot.final_time.abs().max(direct.final_time.abs()).max(1.0e-30);
            let relative_time = (plot.final_time - direct.final_time).abs() / time_scale;
            assert!(
                relative_time <= 2.0e-4,
                "Direct/Plot timestep policy drift for {model_id}/adaptive={adaptive}: Plot t={} Direct t={} (rel={relative_time:e})",
                plot.final_time,
                direct.final_time,
            );
            let state_error = normalized_linf(&plot.packed_state, &direct.packed_state);
            assert!(
                state_error <= 5.0e-4,
                "Direct/Plot state drift for {model_id}/adaptive={adaptive}: normalized Linf={state_error:e}"
            );
        }
    }
}

/// The GUI structured obstacle case on the autonomous Direct route — the
/// gauge-storage regression class: a missing gauge reference anywhere in the
/// structured constants plumbing makes the conserved-state health audit
/// reconstruct absolute density as `rho' + 0` and reject EVERY cell ("halted
/// after 0 accepted batch steps"), which this pins out.
#[test]
fn gpu_structured_obstacle_direct_autonomous_accepts_all_steps() {
    if let Err(error) = gui_explicit_rk4_gpu_available() {
        eprintln!("skipping structured obstacle Direct gate: {error}");
        return;
    }
    for adaptive in [false, true] {
        let direct = gui_explicit_rk4_smoke(case(
            "compressible_structured",
            "obstacle",
            "structured",
            "gpu",
            adaptive,
            "direct",
            3,
        ))
        .unwrap_or_else(|error| panic!("structured obstacle Direct (adaptive={adaptive}): {error}"));
        assert_stable(&direct);
        assert_eq!(
            direct.route, "autonomous",
            "structured obstacle Direct fell back to the ordinary route"
        );
    }
}

#[test]
fn gpu_linear_eos_presets_use_one_finite_ordinary_route_for_plot_and_direct() {
    if let Err(error) = gui_explicit_rk4_gpu_available() {
        eprintln!("skipping GUI liquid-preset route matrix: {error}");
        return;
    }
    for fluid in ["Water", "Alcohol", "Kerosene", "Mercury", "Custom"] {
        for (model_id, mesh_kind) in [
            ("compressible", "cutcell"),
            ("compressible_structured", "structured"),
        ] {
            for adaptive in [false, true] {
                let mut plot_case = case(
                    model_id,
                    "backstep",
                    mesh_kind,
                    "gpu",
                    adaptive,
                    "plot",
                    2,
                );
                plot_case.fluid = fluid;
                let mut direct_case = plot_case;
                direct_case.presentation = "direct";

                let plot = gui_explicit_rk4_smoke(plot_case).unwrap_or_else(|error| {
                    panic!("GPU/{fluid}/{model_id}/adaptive={adaptive}/Plot: {error}")
                });
                let direct = gui_explicit_rk4_smoke(direct_case).unwrap_or_else(|error| {
                    panic!("GPU/{fluid}/{model_id}/adaptive={adaptive}/Direct: {error}")
                });
                assert_stable(&plot);
                assert_stable(&direct);
                assert_eq!(
                    plot.route, "ordinary",
                    "Linear EOS unexpectedly entered an autonomous Plot route"
                );
                assert_eq!(
                    direct.route, "ordinary",
                    "Linear EOS must explicitly fall back to the health-audited ordinary route"
                );
                assert_eq!(
                    plot.packed_state, direct.packed_state,
                    "presentation changed the liquid-EOS trajectory for {fluid}/{model_id}/adaptive={adaptive}"
                );
                assert_eq!(plot.final_time.to_bits(), direct.final_time.to_bits());
                assert_eq!(plot.min_dt.to_bits(), direct.min_dt.to_bits());
                assert_eq!(plot.max_dt.to_bits(), direct.max_dt.to_bits());
            }
        }
    }
}
