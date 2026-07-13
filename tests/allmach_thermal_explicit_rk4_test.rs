//! Focused executable gates for pressure-based thermal RK4. These exercise the
//! same static driver used by the GUI, including adaptive dt and readback.
#![cfg(all(
    feature = "dev-tests",
    feature = "cpu",
    feature = "meshgen",
    feature = "ui"
))]

use std::ops::ControlFlow;

use cfd2::sim::{RuntimeParams, SolverDriver};
use cfd2::solver::mesh::{
    generate_structured_rect_mesh, generate_structured_rect_mesh_periodic, BoundarySides,
    BoundaryType,
};
use cfd2::solver::model::{allmach_pressure_model, allmach_thermal_model, eos::EosSpec};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{GpuLowMachPrecondModel, PreconditionerType, TimeScheme};
use cfd2::ui::app::{
    structured_allmach_rk4_live_update_smoke, structured_allmach_rk4_smoke,
    unstructured_allmach_rk4_smoke, unstructured_allmach_rk4_smoke_to_time,
};
use cfd2::ui::fluid::Fluid;

fn params() -> RuntimeParams {
    RuntimeParams {
        filter_sigma: 0.0,
        adaptive_dt: true,
        target_cfl: 0.9,
        requested_dt: 0.02,
        dtau: 0.0,
        log_convergence: false,
        log_every_steps: 100,
        advection_scheme: Scheme::SecondOrderUpwindVanLeer,
        time_scheme: TimeScheme::RK4,
        preconditioner: PreconditionerType::Jacobi,
        outer_iters: 1,
        outer_auto_converge: false,
        low_mach_model: GpuLowMachPrecondModel::Off,
        low_mach_theta_floor: 1.0e-6,
        low_mach_pressure_coupling_alpha: 1.0,
        alpha_u: 0.7,
        alpha_p: 0.3,
        inlet_velocity: 0.1,
        density: 1.225,
        viscosity: 1.81e-5,
        eos: EosSpec::Constant,
        compressibility_psi: (1.0 / (347.0_f64 * 347.0)) as f32,
        outlet_back_pressure: 0.0,
        allmach_precond_uref_min: 1.0,
        pressure_inlet: false,
        inlet_pressure: 0.0,
    }
}

fn mesh() -> cfd2::solver::mesh::Mesh {
    generate_structured_rect_mesh(
        16,
        6,
        2.0,
        0.5,
        BoundarySides {
            left: BoundaryType::Inlet,
            right: BoundaryType::Outlet,
            bottom: BoundaryType::Wall,
            top: BoundaryType::Wall,
        },
    )
}

fn build() -> SolverDriver {
    let mesh = mesh();
    let n = mesh.num_cells();
    let params = params();
    let mut build = pollster::block_on(SolverDriver::build_forced_cpu(
        &mesh,
        allmach_thermal_model().expect("model"),
        &params,
        &vec![(0.0, 0.0); n],
        &vec![0.0; n],
    ))
    .expect("explicit all-Mach driver");
    build.driver.apply_params(&params);
    build.driver
}

#[test]
fn adaptive_rk4_channel_stays_thermodynamically_valid() {
    let mut driver = build();
    let mut first_dt = None;
    let result = driver.run_steps(80, 8, |step, outcome| {
        assert!(
            outcome.diverged.is_none(),
            "step {step}: {:?}",
            outcome.diverged
        );
        assert!(outcome.dt.is_finite() && outcome.dt > 0.0);
        first_dt.get_or_insert(outcome.dt);
        if let Some(rb) = &outcome.readback {
            assert_eq!(rb.stats.nonfinite_u, 0);
            assert_eq!(rb.stats.nonfinite_p, 0);
            let (rho_min, rho_max) = rb.stats.rho.expect("rho stats");
            assert!(rho_min > 0.0 && rho_max.is_finite());
        }
        ControlFlow::Continue(())
    });
    assert!(result.diverged.is_none());
    assert!(first_dt.expect("first dt") < params().requested_dt);

    let layout = driver.solver().model().state_layout.clone();
    let stride = layout.stride() as usize;
    let state = pollster::block_on(driver.solver().read_state_f32());
    for name in ["T", "rho", "psi", "psi_precond", "d_p", "rho_dT"] {
        let off = layout
            .offset_for(name)
            .unwrap_or_else(|| panic!("missing {name}")) as usize;
        assert!(
            state.chunks_exact(stride).all(|row| row[off].is_finite()),
            "non-finite {name}"
        );
    }
    let t = layout.offset_for("T").unwrap() as usize;
    let rho = layout.offset_for("rho").unwrap() as usize;
    assert!(state
        .chunks_exact(stride)
        .all(|row| row[t] > 0.0 && row[rho] > 0.0));
}

#[test]
fn thermal_trajectory_is_independent_of_readback_cadence() {
    let mut frequent = build();
    let mut sparse = build();
    for step in 0..30 {
        let a = frequent.step(true);
        let b = sparse.step(step == 29);
        assert!(a.diverged.is_none() && b.diverged.is_none());
    }
    let a = pollster::block_on(frequent.solver().read_state_f32());
    let b = pollster::block_on(sparse.solver().read_state_f32());
    let max_diff = a
        .iter()
        .zip(&b)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff <= 2.0e-6,
        "readback changed RK4 trajectory by {max_diff:e}"
    );
}

#[test]
fn barotropic_stage_closure_is_independent_of_readback_cadence() {
    fn build_barotropic() -> SolverDriver {
        let mesh = generate_structured_rect_mesh_periodic(8, 4, 1.0, 1.0);
        let mut p = params();
        p.adaptive_dt = false;
        p.requested_dt = 0.01;
        p.advection_scheme = Scheme::Upwind;
        p.inlet_velocity = 0.0;
        p.viscosity = 0.0;
        p.allmach_precond_uref_min = 0.1;
        let velocity: Vec<(f64, f64)> = (0..mesh.num_cells())
            .map(|cell| {
                let x = mesh.cell_cx[cell];
                let y = mesh.cell_cy[cell];
                (
                    0.5 + 0.05 * (2.0 * std::f64::consts::PI * x).sin(),
                    0.03 * (2.0 * std::f64::consts::PI * y).cos(),
                )
            })
            .collect();
        let pressure: Vec<f64> = (0..mesh.num_cells())
            .map(|cell| {
                -999.74
                    + 0.1
                        * (2.0 * std::f64::consts::PI * mesh.cell_cx[cell]).cos()
            })
            .collect();
        let mut build = pollster::block_on(SolverDriver::build_forced_cpu(
            &mesh,
            allmach_pressure_model().expect("barotropic all-Mach"),
            &p,
            &velocity,
            &pressure,
        ))
        .expect("barotropic RK4 driver");
        build.driver.apply_params(&p);
        build.driver
    }

    let mut frequent = build_barotropic();
    let mut never = build_barotropic();
    for step in 0..40 {
        let a = frequent.step(true);
        let b = never.step(false);
        assert!(
            a.diverged.is_none() && b.diverged.is_none(),
            "barotropic cadence step {step}: {:?} / {:?}",
            a.diverged,
            b.diverged
        );
    }
    let a = pollster::block_on(frequent.solver().read_state_f32());
    let b = pollster::block_on(never.solver().read_state_f32());
    assert_eq!(a, b, "readback cadence changed the barotropic RK4 state");

    let layout = frequent.solver().model().state_layout.clone();
    let stride = layout.stride() as usize;
    let offset = |name: &str| layout.offset_for(name).unwrap() as usize;
    let (ux, uy) = (offset("U"), offset("U") + 1);
    let (psi, psi_precond) = (offset("psi"), offset("psi_precond"));
    let (u_ref, mask) = (offset("u_ref"), offset("precond_mask"));
    for row in a.chunks_exact(stride) {
        let beta2 = (row[ux] * row[ux] + row[uy] * row[uy])
            .max(row[u_ref] * row[u_ref])
            .max(1.0e-12);
        let expected = row[psi] + row[mask] * (row[psi].max(1.0 / beta2) - row[psi]);
        assert_eq!(
            row[psi_precond].to_bits(),
            expected.to_bits(),
            "psi_precond was not recovered from the accepted RK stage"
        );
    }
}

#[test]
fn rhie_chow_damps_a_pure_pressure_checkerboard() {
    const NX: usize = 32;
    const NY: usize = 16;
    const AMPLITUDE: f64 = 1.0e-3;

    #[derive(Clone, Copy)]
    enum Mode {
        XNyquist,
        MixedNyquist,
        SixteenCellBand,
    }

    fn shape(mode: Mode, cell: usize) -> f64 {
        let i = cell % NX;
        let j = cell / NX;
        match mode {
            Mode::XNyquist => if i & 1 == 0 { 1.0 } else { -1.0 },
            Mode::MixedNyquist => if (i + j) & 1 == 0 { 1.0 } else { -1.0 },
            Mode::SixteenCellBand => {
                (4.0 * std::f64::consts::PI * (i as f64 + 0.5) / NX as f64).cos()
            }
        }
    }

    fn mode_amplitude(values: &[f64], mode: Mode) -> f64 {
        let (projection, norm) = values
            .iter()
            .enumerate()
            .fold((0.0, 0.0), |(projection, norm), (cell, &value)| {
                let basis = shape(mode, cell);
                (projection + value * basis, norm + basis * basis)
            });
        projection.abs() / norm.max(1.0e-300)
    }

    fn advance(mode: Mode, disable_rhie_chow: bool) -> (Vec<f64>, Vec<f64>) {
        let mesh = generate_structured_rect_mesh_periodic(NX, NY, 1.0, 1.0);
        let mut p = params();
        p.inlet_velocity = 0.0;
        p.target_cfl = 0.9;
        let pressure: Vec<f64> = (0..NX * NY)
            .map(|cell| AMPLITUDE * shape(mode, cell))
            .collect();
        let mut build = pollster::block_on(SolverDriver::build_forced_cpu(
            &mesh,
            allmach_thermal_model().expect("model"),
            &p,
            &vec![(0.0, 0.0); mesh.num_cells()],
            &pressure,
        ))
        .expect("checkerboard driver");
        build.driver.apply_params(&p);
        if disable_rhie_chow {
            build
                .driver
                .solver_mut()
                .set_field_scalar_current("dt_local", &vec![0.0; mesh.num_cells()])
                .expect("disable Rhie-Chow scale");
        }

        let before = pollster::block_on(build.driver.solver().get_field_scalar("p"))
            .expect("initial pressure");
        for step in 0..16 {
            let outcome = build.driver.step(false);
            assert!(
                outcome.diverged.is_none(),
                "checkerboard step {step}: {:?}",
                outcome.diverged
            );
        }
        let after = pollster::block_on(build.driver.solver().get_field_scalar("p"))
            .expect("final pressure");
        (before, after)
    }

    let (x_initial_field, x_final_field) = advance(Mode::XNyquist, false);
    let (mixed_initial_field, mixed_final_field) = advance(Mode::MixedNyquist, false);
    let (_, mixed_control_field) = advance(Mode::MixedNyquist, true);
    let (band_initial_field, band_final_field) = advance(Mode::SixteenCellBand, false);

    let x_initial = mode_amplitude(&x_initial_field, Mode::XNyquist);
    let x_decay = mode_amplitude(&x_final_field, Mode::XNyquist) / x_initial;
    let mixed_initial = mode_amplitude(&mixed_initial_field, Mode::MixedNyquist);
    let mixed_decay = mode_amplitude(&mixed_final_field, Mode::MixedNyquist) / mixed_initial;
    let mixed_control =
        mode_amplitude(&mixed_control_field, Mode::MixedNyquist) / mixed_initial;
    let band_initial = mode_amplitude(&band_initial_field, Mode::SixteenCellBand);
    let band_retained =
        mode_amplitude(&band_final_field, Mode::SixteenCellBand) / band_initial;
    let band_nyquist_leak =
        mode_amplitude(&band_final_field, Mode::MixedNyquist) / band_initial;

    assert!((x_initial - AMPLITUDE).abs() < 1.0e-8, "bad x mode: {x_initial:e}");
    assert!(
        x_decay < 0.15,
        "Rhie-Chow did not damp the x-Nyquist mode: retained={x_decay:e}"
    );
    assert!(
        mixed_decay < 0.05,
        "Rhie-Chow did not damp the mixed Nyquist mode: retained={mixed_decay:e}"
    );
    assert!(
        mixed_control > 0.90,
        "disabled negative control unexpectedly damped: retained={mixed_control:e}"
    );
    assert!(
        band_retained > 0.55,
        "Rhie-Chow over-damped a resolved 16-cell pressure band: retained={band_retained:e}"
    );
    assert!(
        band_nyquist_leak < 1.0e-3,
        "resolved pressure band leaked into the checkerboard mode: {band_nyquist_leak:e}"
    );
}

#[test]
fn live_parameter_update_resamples_cfl_and_eos_before_advancing() {
    let mut driver = build();
    let mut dt_before = 0.0_f32;
    for _ in 0..4 {
        let outcome = driver.step(false);
        assert!(outcome.diverged.is_none());
        dt_before = outcome.dt;
    }

    let mut changed = params();
    changed.inlet_velocity = 50.0;
    changed.viscosity = 5.0e-2;
    changed.density = 4.0;
    driver.apply_params(&changed);

    let rho_t_ref = pollster::block_on(driver.solver().get_field_scalar("rho_t_ref"))
        .expect("rho_t_ref");
    assert!(rho_t_ref.iter().all(|value| (*value - 4.0).abs() < 1.0e-6));
    let outcome = driver.step(true);
    assert!(outcome.diverged.is_none(), "{:?}", outcome.diverged);
    assert!(
        outcome.dt < 0.5 * dt_before,
        "live rate was stale: before={dt_before:e}, after={:e}",
        outcome.dt
    );

    let time_before_invalid = driver.solver().time();
    let mut singular = changed;
    singular.compressibility_psi = 0.0;
    driver.apply_params(&singular);
    let rejected = driver.step(false);
    assert!(rejected.diverged.is_some());
    assert_eq!(driver.solver().time(), time_before_invalid);
}

#[test]
fn structured_gui_geometries_are_stable_on_cpu() {
    for geometry in ["backstep", "obstacle", "nozzle"] {
        let mut p = params();
        if geometry == "nozzle" {
            p.target_cfl = 0.25;
            p.requested_dt = 1.0e-5;
            p.inlet_velocity = 313.0;
            p.pressure_inlet = true;
            p.inlet_pressure = 3.0e5;
        }
        let smoke = structured_allmach_rk4_smoke(geometry, false, 30, 10, 30, p)
            .unwrap_or_else(|error| panic!("structured CPU {geometry}: {error}"));
        assert_eq!(smoke.steps, 30);
        assert!(smoke.min_dt > 0.0 && smoke.max_dt.is_finite());
        assert!(smoke.min_rho > 0.0 && smoke.max_rho.is_finite());
    }
}

#[test]
fn structured_gui_geometries_are_stable_on_gpu() {
    for geometry in ["backstep", "obstacle", "nozzle"] {
        let mut p = params();
        if geometry == "nozzle" {
            p.target_cfl = 0.25;
            p.requested_dt = 1.0e-5;
            p.inlet_velocity = 313.0;
            p.pressure_inlet = true;
            p.inlet_pressure = 3.0e5;
        }
        let smoke = structured_allmach_rk4_smoke(geometry, true, 30, 10, 15, p)
            .unwrap_or_else(|error| panic!("structured GPU {geometry}: {error}"));
        assert_eq!(smoke.steps, 15);
        assert!(smoke.min_dt > 0.0 && smoke.max_dt.is_finite());
        assert!(smoke.min_rho > 0.0 && smoke.max_rho.is_finite());
    }
}

#[test]
fn structured_live_fluid_and_inlet_update_remains_stable() {
    let before = params();
    let mut after = before;
    after.inlet_velocity = 5.0;
    after.viscosity = 1.0e-2;
    after.density = 2.5;
    after.alpha_u = 0.9;
    for gpu in [false, true] {
        let smoke = structured_allmach_rk4_live_update_smoke(
            "obstacle", gpu, 30, 10, 20, before, after,
        )
        .unwrap_or_else(|error| {
            panic!(
                "structured {} live update: {error}",
                if gpu { "GPU" } else { "CPU" }
            )
        });
        assert_eq!(smoke.steps, 20);
        assert!(smoke.min_dt > 0.0 && smoke.min_dt < smoke.max_dt);
        assert!(smoke.min_rho > 0.0 && smoke.max_rho.is_finite());
        assert!(smoke.max_vel.is_finite() && smoke.max_vel > 0.0);
    }
}

#[test]
fn structured_singular_live_update_is_rejected_before_step() {
    let before = params();
    let mut singular = before;
    singular.compressibility_psi = 0.0;
    let error = structured_allmach_rk4_live_update_smoke(
        "obstacle", false, 20, 8, 6, before, singular,
    )
    .expect_err("singular structured mass block must be rejected");
    assert!(error.contains("pre-step"), "unexpected error: {error}");
}

fn gui_mesh_cases() -> Vec<(&'static str, &'static str)> {
    let mut cases = Vec::new();
    for geometry in ["backstep", "obstacle"] {
        for mesh in ["cutcell", "delaunay", "voronoi", "cvt"] {
            cases.push((geometry, mesh));
        }
    }
    for mesh in ["fitted", "cutcell", "delaunay", "voronoi", "cvt"] {
        cases.push(("nozzle", mesh));
    }
    cases
}

fn case_params(geometry: &str) -> RuntimeParams {
    let mut p = params();
    if geometry == "nozzle" {
        p.target_cfl = 0.25;
        p.requested_dt = 1.0e-5;
        p.inlet_velocity = 313.0;
        p.pressure_inlet = true;
        p.inlet_pressure = 3.0e5;
    }
    p
}

#[test]
fn every_unstructured_gui_mesh_case_is_stable_on_cpu() {
    for (geometry, mesh) in gui_mesh_cases() {
        let smoke =
            unstructured_allmach_rk4_smoke(geometry, mesh, false, 0.15, 15, case_params(geometry))
                .unwrap_or_else(|error| panic!("CPU {geometry}/{mesh}: {error}"));
        assert!(smoke.cells > 0 && smoke.steps == 15);
        assert!(smoke.min_dt > 0.0 && smoke.max_dt.is_finite());
        assert!(smoke.min_rho > 0.0 && smoke.max_rho.is_finite());
    }
}

#[test]
fn every_unstructured_gui_mesh_case_constructs_and_steps_on_gpu() {
    for (geometry, mesh) in gui_mesh_cases() {
        let smoke =
            unstructured_allmach_rk4_smoke(geometry, mesh, true, 0.2, 5, case_params(geometry))
                .unwrap_or_else(|error| panic!("GPU {geometry}/{mesh}: {error}"));
        assert!(smoke.cells > 0 && smoke.steps == 5);
        assert!(smoke.min_dt > 0.0 && smoke.max_dt.is_finite());
        assert!(smoke.min_rho > 0.0 && smoke.max_rho.is_finite());
    }
}

#[test]
fn gui_fluid_and_advection_endpoints_remain_stable() {
    for fluid in Fluid::presets() {
        for scheme in [
            Scheme::Upwind,
            Scheme::SecondOrderUpwindVanLeer,
            Scheme::QUICK,
        ] {
            let mut p = params();
            p.density = fluid.density as f32;
            p.viscosity = fluid.viscosity as f32;
            p.eos = fluid.eos;
            p.compressibility_psi = fluid.compressibility() as f32;
            p.advection_scheme = scheme;
            let smoke = unstructured_allmach_rk4_smoke("obstacle", "cutcell", false, 0.2, 12, p)
                .unwrap_or_else(|error| panic!("{} {scheme:?}: {error}", fluid.name));
            assert!(smoke.min_dt > 0.0 && smoke.min_rho > 0.0);
        }
    }
}

#[test]
fn mercury_extreme_and_live_eos_update_cover_both_topologies_and_backends() {
    let mercury = Fluid::presets()
        .into_iter()
        .find(|fluid| fluid.name == "Mercury")
        .expect("Mercury preset");
    let mut mercury_params = params();
    mercury_params.density = mercury.density as f32;
    mercury_params.viscosity = mercury.viscosity as f32;
    mercury_params.eos = mercury.eos;
    mercury_params.compressibility_psi = mercury.compressibility() as f32;

    for gpu in [false, true] {
        let structured = structured_allmach_rk4_smoke(
            "obstacle",
            gpu,
            16,
            6,
            5,
            mercury_params,
        )
        .unwrap_or_else(|error| {
            panic!(
                "structured {} Mercury: {error}",
                if gpu { "GPU" } else { "CPU" }
            )
        });
        assert_eq!(structured.steps, 5);
        assert!(structured.min_dt > 0.0 && structured.min_rho > 0.0);

        let unstructured = unstructured_allmach_rk4_smoke(
            "obstacle",
            "cutcell",
            gpu,
            0.2,
            5,
            mercury_params,
        )
        .unwrap_or_else(|error| {
            panic!(
                "unstructured {} Mercury: {error}",
                if gpu { "GPU" } else { "CPU" }
            )
        });
        assert_eq!(unstructured.steps, 5);
        assert!(unstructured.min_dt > 0.0 && unstructured.min_rho > 0.0);

        let live = structured_allmach_rk4_live_update_smoke(
            "obstacle",
            gpu,
            16,
            6,
            8,
            params(),
            mercury_params,
        )
        .unwrap_or_else(|error| {
            panic!(
                "structured {} Air-to-Mercury update: {error}",
                if gpu { "GPU" } else { "CPU" }
            )
        });
        assert_eq!(live.steps, 8);
        assert!(live.min_dt > 0.0 && live.max_dt.is_finite());
        assert!(live.min_rho > 0.0 && live.max_rho.is_finite());
    }
}

#[test]
fn shipping_resolution_risk_representatives_remain_stable() {
    for gpu in [false, true] {
        for (geometry, mesh, steps) in [
            ("backstep", "cutcell", 40usize),
            ("obstacle", "cutcell", 40usize),
            ("nozzle", "fitted", 80usize),
        ] {
            let smoke = unstructured_allmach_rk4_smoke(
                geometry,
                mesh,
                gpu,
                0.025,
                steps,
                case_params(geometry),
            )
            .unwrap_or_else(|error| {
                panic!(
                    "{} {geometry}/{mesh}: {error}",
                    if gpu { "GPU" } else { "CPU" }
                )
            });
            assert_eq!(smoke.steps, steps);
            assert!(smoke.min_dt > 0.0 && smoke.max_dt.is_finite());
            assert!(smoke.min_rho > 0.0 && smoke.max_rho.is_finite());
            assert!(smoke.max_vel.is_finite());
        }
    }
}

#[test]
fn pressure_inlet_nozzle_survives_multiple_acoustic_transits() {
    for gpu in [false, true] {
        let smoke = unstructured_allmach_rk4_smoke(
            "nozzle",
            "fitted",
            gpu,
            0.05,
            2_000,
            case_params("nozzle"),
        )
        .unwrap_or_else(|error| panic!("{} long nozzle: {error}", if gpu { "GPU" } else { "CPU" }));
        assert_eq!(smoke.steps, 2_000);
        assert!(smoke.max_vel.is_finite() && smoke.max_vel > 1.0 && smoke.max_vel < 5.0e3);
        assert!(smoke.min_rho > 0.0 && smoke.max_rho.is_finite());
        assert!(smoke.min_dt > 0.0 && smoke.max_dt.is_finite());
    }
}

/// Manual parameter probe for the exact GUI backstep shown by the checkerboard
/// regression report. Kept ignored because the shipping-resolution march is
/// intentionally long; the non-ignored regression below uses the shortest
/// time window that still exposes the mode.
#[test]
#[ignore]
fn diagnose_backstep_rhie_chow_mode() {
    let mut p = params();
    p.inlet_velocity = 0.011;
    p.allmach_precond_uref_min = 1.0;
    p.alpha_u = std::env::var("CFD2_RC_ALPHA")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0.7);
    let steps = std::env::var("CFD2_RC_STEPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1_000);
    let gpu = std::env::var("CFD2_RC_GPU").as_deref() == Ok("1");
    let smoke = unstructured_allmach_rk4_smoke("backstep", "cutcell", gpu, 0.025, steps, p)
        .expect("GUI backstep probe");
    eprintln!("[rhie-chow] alpha={} steps={} gpu={} {smoke:#?}", p.alpha_u, steps, gpu);
}

#[test]
fn reported_backstep_has_no_pressure_checkerboard_at_equal_physical_time() {
    let mut p = params();
    p.inlet_velocity = 0.011;
    p.allmach_precond_uref_min = 1.0;
    let target_time = 1.45;
    let smoke = unstructured_allmach_rk4_smoke_to_time(
        "backstep",
        "cutcell",
        false,
        0.025,
        target_time,
        1_600,
        p,
    )
    .expect("reported GUI backstep");
    assert!(smoke.final_time >= target_time);
    assert!(
        smoke.final_time <= target_time + 1.1 * smoke.max_dt,
        "physical-time smoke overshot by more than one adaptive step: {smoke:#?}"
    );
    assert!(
        smoke.rhie_chow_bracket_fraction < 0.03,
        "large Rhie-Chow bracket defect: {smoke:#?}"
    );
    assert!(
        smoke.pressure_gradient_visibility > 0.65 && smoke.pressure_jump_defect < 0.45,
        "pressure jumps became invisible to the reconstructed gradient: {smoke:#?}"
    );
    let max_nyquist = smoke
        .pressure_nyquist_x
        .max(smoke.pressure_nyquist_y)
        .max(smoke.pressure_nyquist_xy);
    assert!(max_nyquist < 0.025, "pressure Nyquist mode is visible: {smoke:#?}");
    assert!(
        smoke.velocity_stripe_fraction < 0.065,
        "startup pseudo-acoustic band remained too strong: {smoke:#?}"
    );
}

#[test]
fn reconstructed_pressure_gradient_remains_visible_on_all_gui_mesh_families() {
    let mut p = params();
    p.inlet_velocity = 0.011;
    p.allmach_precond_uref_min = 1.0;
    for mesh_kind in ["cutcell", "delaunay", "voronoi", "cvt"] {
        let smoke = unstructured_allmach_rk4_smoke(
            "backstep",
            mesh_kind,
            false,
            0.05,
            500,
            p,
        )
        .unwrap_or_else(|error| panic!("backstep/{mesh_kind}: {error}"));
        assert!(
            smoke.pressure_gradient_visibility > 0.65,
            "gradient-invisible pressure on {mesh_kind}: {smoke:#?}"
        );
        assert!(
            smoke.pressure_jump_defect < 0.45,
            "pressure jump/reconstruction mismatch on {mesh_kind}: {smoke:#?}"
        );
        assert!(
            smoke.rhie_chow_bracket_fraction < 0.10,
            "large linearly-exact Rhie-Chow bracket on {mesh_kind}: {smoke:#?}"
        );
    }
}
