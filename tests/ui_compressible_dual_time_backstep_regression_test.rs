#![cfg(all(feature = "meshgen", feature = "dev-tests"))]

use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep};
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::model::helpers::{
    SolverCompressibleIdealGasExt, SolverCompressibleInletExt, SolverFieldAliasesExt,
    SolverRuntimeParamsExt,
};
use cfd2::solver::model::compressible_model_with_eos;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{
    GpuLowMachPrecondModel, PreconditionerType, SolverConfig, SteppingMode, TimeScheme,
    UnifiedSolver,
};
use nalgebra::Vector2;

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_str(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

fn parse_scheme(name: &str) -> Option<Scheme> {
    match name.trim().to_lowercase().as_str() {
        "upwind" => Some(Scheme::Upwind),
        "sou" => Some(Scheme::SecondOrderUpwind),
        "sou_minmod" | "sou-minmod" => Some(Scheme::SecondOrderUpwindMinMod),
        "sou_vanleer" | "sou-vanleer" => Some(Scheme::SecondOrderUpwindVanLeer),
        "quick" => Some(Scheme::QUICK),
        "quick_minmod" | "quick-minmod" => Some(Scheme::QUICKMinMod),
        "quick_vanleer" | "quick-vanleer" => Some(Scheme::QUICKVanLeer),
        _ => None,
    }
}

fn parse_time_scheme(name: &str) -> Option<TimeScheme> {
    match name.trim().to_lowercase().as_str() {
        "euler" => Some(TimeScheme::Euler),
        "bdf2" => Some(TimeScheme::BDF2),
        _ => None,
    }
}

fn min_cell_size(mesh: &cfd2::solver::mesh::Mesh) -> f64 {
    mesh.cell_vol
        .iter()
        .copied()
        .map(|v| v.sqrt())
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn ui_compressible_backstep_dual_time_does_not_blow_up() {
    std::env::set_var("CFD2_QUIET", "1");

    // Match the UI geometry (BackwardsStep) and the mesh stats from the screenshot:
    // - cell size 0.025 aligns exactly with step_x=0.5 and height_inlet=0.5 (no cut cells)
    // - 5200 cells (5600 - 400 in the removed step region)
    let length = 3.5;
    let domain_size = Vector2::new(length, 1.0);
    let geo = BackwardsStep {
        length,
        height_inlet: 0.5,
        height_outlet: 1.0,
        step_x: 0.5,
    };

    let cell = env_f64("CFD2_UI_DUAL_TIME_CELL", 0.025);
    let mut mesh = generate_cut_cell_mesh(&geo, cell, cell, 1.2, domain_size);
    let smooth_iters = env_usize("CFD2_UI_DUAL_TIME_SMOOTH_ITERS", 50);
    mesh.smooth(&geo, 0.3, smooth_iters);
    if (cell - 0.025).abs() < 1e-12 {
        assert_eq!(mesh.num_cells(), 5200, "expected UI-like 5200-cell mesh");
    }
    let (mut inlet_faces, mut outlet_faces, mut wall_faces, mut slip_faces) = (0usize, 0usize, 0usize, 0usize);
    for b in &mesh.face_boundary {
        match b {
            Some(cfd2::solver::mesh::BoundaryType::Inlet) => inlet_faces += 1,
            Some(cfd2::solver::mesh::BoundaryType::Outlet) => outlet_faces += 1,
            Some(cfd2::solver::mesh::BoundaryType::Wall) => wall_faces += 1,
            Some(cfd2::solver::mesh::BoundaryType::SlipWall) => slip_faces += 1,
            None => {}
        }
    }
    eprintln!(
        "[ui_dual_time_backstep] boundary_faces inlet={} outlet={} wall={} slip={}",
        inlet_faces, outlet_faces, wall_faces, slip_faces
    );
    assert!(inlet_faces > 0, "expected at least one inlet face");
    assert!(outlet_faces > 0, "expected at least one outlet face");

    let density = 1.225f32;
    let viscosity = 1.81e-5f32;
    let inlet_u = 1.0f32;
    let eos = EosSpec::IdealGas {
        gamma: 1.4,
        gas_constant: 287.0,
        temperature: 300.0,
    };

    let mut solver = pollster::block_on(UnifiedSolver::new(
        &mesh,
        compressible_model_with_eos(eos),
        SolverConfig {
            advection_scheme: Scheme::SecondOrderUpwind,
            time_scheme: TimeScheme::BDF2,
            preconditioner: PreconditionerType::Jacobi,
            stepping: SteppingMode::Implicit { outer_iters: 1 },
        },
        None,
        None,
    ))
    .expect("solver init");

    // Clear full state to avoid uninitialized auxiliary fields.
    let stride = solver.model().state_layout.stride() as usize;
    solver
        .write_state_f32(&vec![0.0f32; mesh.num_cells() * stride])
        .expect("clear state");

    // UI-like runtime params (override via `CFD2_UI_*` env vars).
    let requested_dt = env_f64("CFD2_UI_DUAL_TIME_DT", 0.001) as f32;
    let target_cfl = env_f64("CFD2_UI_DUAL_TIME_TARGET_CFL", 0.95);
    let outer_iters = env_usize("CFD2_UI_DUAL_TIME_OUTER_ITERS", 30);
    let low_mach_model = match env_str("CFD2_UI_DUAL_TIME_LOW_MACH_MODEL")
        .as_deref()
        .map(str::to_lowercase)
        .as_deref()
    {
        Some("off") => GpuLowMachPrecondModel::Off,
        Some("legacy") => GpuLowMachPrecondModel::Legacy,
        Some("weiss" | "weiss-smith" | "weiss_smith") => GpuLowMachPrecondModel::WeissSmith,
        Some(other) => panic!("unknown CFD2_UI_DUAL_TIME_LOW_MACH_MODEL={other}"),
        None => GpuLowMachPrecondModel::WeissSmith,
    };
    let low_mach_theta_floor = env_f64("CFD2_UI_DUAL_TIME_THETA_FLOOR", 1e-6) as f32;
    let low_mach_pressure_coupling_alpha =
        env_f64("CFD2_UI_DUAL_TIME_PRESSURE_COUPLING_ALPHA", -1.0) as f32;
    let dtau = env_f64("CFD2_UI_DUAL_TIME_DTAU", 1.0e-5) as f32;
    let alpha_u = env_f64("CFD2_UI_DUAL_TIME_ALPHA_U", -1.0);
    let alpha_p = env_f64("CFD2_UI_DUAL_TIME_ALPHA_P", -1.0);
    let scheme = env_str("CFD2_UI_DUAL_TIME_SCHEME")
        .as_deref()
        .and_then(parse_scheme)
        .unwrap_or(Scheme::SecondOrderUpwind);
    let time_scheme = env_str("CFD2_UI_DUAL_TIME_TIME_SCHEME")
        .as_deref()
        .and_then(parse_time_scheme)
        .unwrap_or(TimeScheme::BDF2);

    solver.set_dt(requested_dt);
    solver.set_density(density).unwrap();
    solver.set_viscosity(viscosity).unwrap();
    solver.set_eos(&eos).unwrap();
    solver
        .set_precond_model(low_mach_model)
        .expect("precond model");
    solver
        .set_precond_theta_floor(low_mach_theta_floor)
        .expect("theta floor");
    if low_mach_pressure_coupling_alpha >= 0.0 {
        solver
            .set_precond_pressure_coupling_alpha(low_mach_pressure_coupling_alpha)
            .expect("pressure coupling alpha");
    }
    solver.set_dtau(dtau).expect("dtau");
    solver.set_outer_iters(outer_iters).unwrap();
    solver
        .set_compressible_inlet_isothermal_x(density, inlet_u, &eos)
        .unwrap();

    let p_ref = eos.pressure_for_density(density as f64) as f32;
    solver.set_uniform_state(density, [0.0, 0.0], p_ref);
    solver.initialize_history();

    let h_min = min_cell_size(&mesh);
    let sound_speed = eos.sound_speed(density as f64);

    // Sanity-check initialization: uniform pressure should be present before stepping.
    {
        let p0 = pollster::block_on(solver.get_p());
        let (min_p0, max_p0) = p0.iter().fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(mn, mx), &v| (mn.min(v), mx.max(v)),
        );
        assert!(
            min_p0 > 1e3 && (max_p0 - min_p0).abs() < 1e-6,
            "expected initialized uniform pressure, got p=[{min_p0:.3e},{max_p0:.3e}]"
        );
    }

    // Mirror the UI adaptive-dt behavior:
    // - dt chosen from the preconditioned wave speed (allows acoustic CFL >> 1 at low Mach)
    // - limited growth by 20% per step
    let steps = env_usize("CFD2_UI_DUAL_TIME_STEPS", 2);

    solver.set_advection_scheme(scheme);
    solver.set_time_scheme(time_scheme);
    if alpha_u >= 0.0 {
        solver.set_alpha_u(alpha_u as f32).unwrap();
    }
    if alpha_p >= 0.0 {
        solver.set_alpha_p(alpha_p as f32).unwrap();
    }

    let mut prev_max_vel = 0.0f64;
    for step in 0..steps {
        let adv_speed = prev_max_vel.max(inlet_u as f64);
        let effective_sound_speed = match low_mach_model {
            GpuLowMachPrecondModel::Off => sound_speed,
            GpuLowMachPrecondModel::Legacy => sound_speed.min(adv_speed),
            GpuLowMachPrecondModel::WeissSmith => {
                let theta = (low_mach_theta_floor as f64).max(0.0);
                let c_floor = sound_speed * theta.sqrt();
                sound_speed.min(adv_speed.max(c_floor))
            }
        };
        let wave_speed = adv_speed + effective_sound_speed;
        if h_min > 1e-12 && wave_speed.is_finite() && wave_speed > 1e-12 {
            let current_dt = solver.dt() as f64;
            let mut next_dt = target_cfl * h_min / wave_speed;
            if next_dt > current_dt * 1.2 {
                next_dt = current_dt * 1.2;
            }
            next_dt = next_dt.clamp(1e-9, 100.0);
            solver.set_dt(next_dt as f32);
        }

        let stats = solver.step_with_stats().expect("step with stats");
        let last = stats.last().cloned().unwrap_or_default();
        eprintln!(
            "[ui_dual_time_backstep] step={step} linear_stats_len={} last(iters={}, res={:.3e}, conv={}, div={})",
            stats.len(),
            last.iterations,
            last.residual,
            last.converged,
            last.diverged
        );
        assert!(
            last.residual.is_finite() && last.residual < 1e12,
            "step {step}: linear residual blew up: {last:?}"
        );
        assert!(!last.diverged, "step {step}: linear solver diverged: {last:?}");

        // Debug: if the implicit solve early-exits, `x` may remain zero and clobber state in the update pass.
        if step == 0 {
            let x = pollster::block_on(solver.get_linear_solution()).expect("read linear x");
            let (min_x, max_x) = x.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &v| {
                (mn.min(v), mx.max(v))
            });
            eprintln!(
                "[ui_dual_time_backstep] step={step} x_range=[{min_x:.3e},{max_x:.3e}] x0={:.3e} x4={:.3e} x6={:.3e}",
                x.get(0).copied().unwrap_or_default(),
                x.get(4).copied().unwrap_or_default(),
                x.get(6).copied().unwrap_or_default(),
            );
        }

        let u = pollster::block_on(solver.get_u());
        let p = pollster::block_on(solver.get_p());

        let mut max_vel = 0.0f64;
        for (vx, vy) in &u {
            assert!(vx.is_finite() && vy.is_finite(), "step {step}: non-finite u");
            let v = (vx * vx + vy * vy).sqrt();
            max_vel = max_vel.max(v);
        }
        let mut min_p = f64::INFINITY;
        let mut max_p = f64::NEG_INFINITY;
        for &pv in &p {
            assert!(pv.is_finite(), "step {step}: non-finite p");
            min_p = min_p.min(pv);
            max_p = max_p.max(pv);
        }
        eprintln!(
            "[ui_dual_time_backstep] step={step} dt={:.3e} dtau={dtau:.3e} max|u|={max_vel:.3e} p=[{min_p:.3e},{max_p:.3e}] acoustic_cfl={:.1}",
            solver.dt(),
            sound_speed * solver.dt() as f64 / h_min.max(1e-12)
        );

        assert!(
            min_p > 0.0,
            "step {step}: negative pressure (min_p={min_p:.3e}, max_p={max_p:.3e})"
        );
        assert!(
            max_vel < 50.0,
            "step {step}: unphysical velocity growth (max_vel={max_vel:.3e})"
        );

        prev_max_vel = max_vel;
    }
}
