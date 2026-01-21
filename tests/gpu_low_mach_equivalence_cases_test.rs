#![cfg(all(feature = "meshgen", feature = "dev-tests"))]

use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep, ChannelWithObstacle, Mesh};
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::model::helpers::{
    SolverCompressibleIdealGasExt, SolverCompressibleInletExt, SolverFieldAliasesExt,
    SolverInletVelocityExt, SolverRuntimeParamsExt,
};
use cfd2::solver::model::{compressible_model_with_eos, incompressible_momentum_model};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{
    GpuLowMachPrecondModel, PreconditionerType, SolverConfig, SteppingMode, TimeScheme,
    UnifiedSolver,
};
use nalgebra::{Point2, Vector2};
use std::env;

fn env_f64(name: &str, default: f64) -> f64 {
    env::var(name)
        .ok()
        .and_then(|val| val.parse().ok())
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|val| val.parse().ok())
        .unwrap_or(default)
}

fn min_cell_size(mesh: &Mesh) -> f64 {
    mesh.cell_vol
        .iter()
        .copied()
        .map(|v| v.sqrt())
        .fold(f64::INFINITY, f64::min)
}

fn neighbors(mesh: &Mesh, cell: usize) -> Vec<usize> {
    let start = mesh.cell_face_offsets[cell];
    let end = mesh.cell_face_offsets[cell + 1];
    let mut out = Vec::with_capacity(end - start);
    for idx in start..end {
        let face = mesh.cell_faces[idx];
        let owner = mesh.face_owner[face];
        let neigh = mesh.face_neighbor[face];
        let other = if owner == cell { neigh } else { Some(owner) };
        if let Some(n) = other {
            out.push(n);
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn checkerboard_metric(mesh: &Mesh, values: &[f64], norm: f64) -> f64 {
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for cell in 0..mesh.num_cells() {
        let neighs = neighbors(mesh, cell);
        if neighs.is_empty() {
            continue;
        }
        let mut avg = 0.0f64;
        for &n in &neighs {
            avg += values[n];
        }
        avg /= neighs.len() as f64;
        let diff = values[cell] - avg;
        sum += diff * diff;
        count += 1;
    }
    if count == 0 {
        return 0.0;
    }
    let rms = (sum / count as f64).sqrt();
    rms / norm.max(1e-12)
}

fn sample_line(
    mesh: &Mesh,
    y_mid: f64,
    band: f64,
    u_incomp: &[(f64, f64)],
    p_incomp: &[f64],
    u_comp: &[(f64, f64)],
    p_comp: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut x_line = Vec::new();
    let mut ux_line_incomp = Vec::new();
    let mut ux_line_comp = Vec::new();
    let mut p_line_incomp = Vec::new();
    let mut p_line_comp = Vec::new();

    let min_dist = (0..mesh.num_cells())
        .map(|i| (mesh.cell_cy[i] - y_mid).abs())
        .fold(f64::INFINITY, f64::min);
    let threshold = min_dist + band;

    for i in 0..mesh.num_cells() {
        if (mesh.cell_cy[i] - y_mid).abs() <= threshold {
            x_line.push(mesh.cell_cx[i]);
            ux_line_incomp.push(u_incomp[i].0);
            ux_line_comp.push(u_comp[i].0);
            p_line_incomp.push(p_incomp[i]);
            p_line_comp.push(p_comp[i]);
        }
    }

    let mut order: Vec<usize> = (0..x_line.len()).collect();
    order.sort_by(|&a, &b| x_line[a].partial_cmp(&x_line[b]).unwrap());

    let reorder = |v: &[f64]| order.iter().map(|&i| v[i]).collect::<Vec<f64>>();
    let x_sorted = reorder(&x_line);
    let ux_incomp_sorted = reorder(&ux_line_incomp);
    let ux_comp_sorted = reorder(&ux_line_comp);

    let mut p_incomp_sorted = reorder(&p_line_incomp);
    let mut p_comp_sorted = reorder(&p_line_comp);
    let mean_p_inc = p_incomp_sorted.iter().sum::<f64>() / p_incomp_sorted.len().max(1) as f64;
    let mean_p_cmp = p_comp_sorted.iter().sum::<f64>() / p_comp_sorted.len().max(1) as f64;
    for v in &mut p_incomp_sorted {
        *v -= mean_p_inc;
    }
    for v in &mut p_comp_sorted {
        *v -= mean_p_cmp;
    }

    (
        x_sorted,
        ux_incomp_sorted,
        ux_comp_sorted,
        p_incomp_sorted,
        p_comp_sorted,
    )
}

fn rms_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        sum += d * d;
    }
    (sum / a.len() as f64).sqrt()
}

fn build_backwards_step_mesh(cell: f64, smooth_iters: usize) -> Mesh {
    let length = 3.5;
    let geo = BackwardsStep {
        length,
        height_inlet: 0.5,
        height_outlet: 1.0,
        step_x: 0.5,
    };
    let domain_size = Vector2::new(length, geo.height_outlet);
    let mut mesh = generate_cut_cell_mesh(&geo, cell, cell, 1.2, domain_size);
    if smooth_iters > 0 {
        mesh.smooth(&geo, 0.3, smooth_iters);
    }
    mesh
}

fn build_channel_obstacle_mesh(cell: f64, smooth_iters: usize) -> Mesh {
    let length = 3.0;
    let height = 1.0;
    let geo = ChannelWithObstacle {
        length,
        height,
        obstacle_center: Point2::new(1.0, 0.51),
        obstacle_radius: 0.1,
    };
    let domain_size = Vector2::new(length, height);
    let mut mesh = generate_cut_cell_mesh(&geo, cell, cell, 1.0, domain_size);
    if smooth_iters > 0 {
        mesh.smooth(&geo, 0.3, smooth_iters);
    }
    mesh
}

fn run_low_mach_equivalence_case(case_name: &str, mesh: &Mesh, y_sample: f64) {
    std::env::set_var("CFD2_QUIET", "1");

    let steps = env_usize("CFD2_LOW_MACH_EQ_STEPS", 12);
    let outer_iters = env_usize("CFD2_LOW_MACH_EQ_OUTER_ITERS", 20);
    let target_cfl = env_f64("CFD2_LOW_MACH_EQ_TARGET_CFL", 0.6);
    let target_pseudo_cfl = env_f64("CFD2_LOW_MACH_EQ_TARGET_PSEUDO_CFL", 0.6);
    let theta_floor = env_f64("CFD2_LOW_MACH_EQ_THETA_FLOOR", 1e-4);
    let comp_alpha_u = env_f64("CFD2_LOW_MACH_EQ_ALPHA_U", 0.2);
    let comp_alpha_p = env_f64("CFD2_LOW_MACH_EQ_ALPHA_P", 1.0);

    let u_in = env_f64("CFD2_LOW_MACH_EQ_UIN", 1.0) as f32;
    let density = env_f64("CFD2_LOW_MACH_EQ_RHO", 1.225) as f32;
    let viscosity = env_f64("CFD2_LOW_MACH_EQ_MU", 1.81e-5) as f32;

    // Default to a reduced temperature so pressure/energy magnitudes stay moderate while
    // remaining in a low-Mach regime (acoustic CFL >> 1). Override with `CFD2_LOW_MACH_EQ_T`.
    let temperature = env_f64("CFD2_LOW_MACH_EQ_T", 30.0);
    let eos = EosSpec::IdealGas {
        gamma: 1.4,
        gas_constant: 287.0,
        temperature,
    };
    let p0 = eos.pressure_for_density(density as f64) as f32;
    let sound_speed = eos.sound_speed(density as f64) as f64;

    let h_min = min_cell_size(mesh);
    let dt = target_cfl * h_min / (u_in as f64).max(1e-9);
    let dtau = target_pseudo_cfl * h_min / sound_speed.max(1e-9);

    let mut incomp = pollster::block_on(UnifiedSolver::new(
        mesh,
        incompressible_momentum_model(),
        SolverConfig {
            advection_scheme: Scheme::SecondOrderUpwind,
            time_scheme: TimeScheme::Euler,
            preconditioner: PreconditionerType::Jacobi,
            stepping: SteppingMode::Coupled,
        },
        None,
        None,
    ))
    .expect("incompressible solver init");
    let stride = incomp.model().state_layout.stride() as usize;
    incomp
        .write_state_f32(&vec![0.0f32; mesh.num_cells() * stride])
        .expect("clear state");
    incomp.set_dt(dt as f32);
    incomp.set_dtau(0.0).unwrap();
    incomp.set_density(density).unwrap();
    incomp.set_viscosity(viscosity).unwrap();
    incomp.set_alpha_u(0.7).unwrap();
    incomp.set_alpha_p(0.3).unwrap();
    incomp.set_outer_iters(6).unwrap();
    incomp.set_inlet_velocity(u_in).unwrap();
    incomp.set_u(&vec![(0.0, 0.0); mesh.num_cells()]);
    incomp.set_p(&vec![0.0; mesh.num_cells()]);
    incomp.initialize_history();

    let mut comp = pollster::block_on(UnifiedSolver::new(
        mesh,
        compressible_model_with_eos(eos),
        SolverConfig {
            advection_scheme: Scheme::SecondOrderUpwind,
            time_scheme: TimeScheme::Euler,
            preconditioner: PreconditionerType::Jacobi,
            stepping: SteppingMode::Implicit { outer_iters: 1 },
        },
        None,
        None,
    ))
    .expect("compressible solver init");
    let stride = comp.model().state_layout.stride() as usize;
    comp.write_state_f32(&vec![0.0f32; mesh.num_cells() * stride])
        .expect("clear state");
    comp.set_dt(dt as f32);
    comp.set_dtau(dtau as f32).unwrap();
    comp.set_viscosity(viscosity).unwrap();
    comp.set_eos(&eos).unwrap();
    comp.set_alpha_u(comp_alpha_u as f32).unwrap();
    comp.set_alpha_p(comp_alpha_p as f32).unwrap();
    comp.set_precond_model(GpuLowMachPrecondModel::WeissSmith)
        .expect("precond model");
    comp.set_precond_theta_floor(theta_floor as f32)
        .expect("theta floor");
    comp.set_outer_iters(outer_iters).unwrap();
    comp.set_compressible_inlet_isothermal_x(density, u_in, &eos)
        .unwrap();
    comp.set_uniform_state(density, [0.0, 0.0], p0);
    comp.initialize_history();

    for _ in 0..steps {
        incomp.step();
        comp.step();
    }

    let u_incomp = pollster::block_on(incomp.get_u());
    let p_incomp = pollster::block_on(incomp.get_p());
    let u_comp = pollster::block_on(comp.get_u());
    let p_comp = pollster::block_on(comp.get_p());
    let rho_comp = pollster::block_on(comp.get_rho());
    let theta_ref = eos.runtime_params().theta_ref as f64;
    let p_pert_comp: Vec<f64> = p_comp
        .iter()
        .zip(rho_comp.iter())
        .map(|(p, rho)| p - rho * theta_ref)
        .collect();

    // Compare a 1D cut and pressure oscillation metrics; this is primarily a regression
    // guard against low-Mach compressible divergence/unphysical waves.
    let band = 0.35 * h_min;
    let (_x, ux_incomp, ux_comp, p_incomp_c, p_comp_c) =
        sample_line(mesh, y_sample, band, &u_incomp, &p_incomp, &u_comp, &p_pert_comp);

    let dyn_pressure = 0.5 * density as f64 * (u_in as f64).powi(2);
    let ux_l2_over_uin = rms_diff(&ux_incomp, &ux_comp) / (u_in as f64).max(1e-9);
    let p_l2_over_q = rms_diff(&p_incomp_c, &p_comp_c) / dyn_pressure.max(1e-12);
    let p_comp_zero_mean: Vec<f64> = {
        let mean = p_comp.iter().sum::<f64>() / p_comp.len().max(1) as f64;
        p_comp.iter().map(|v| v - mean).collect()
    };
    let rho_comp_zero_mean: Vec<f64> = {
        let mean = rho_comp.iter().sum::<f64>() / rho_comp.len().max(1) as f64;
        rho_comp.iter().map(|v| v - mean).collect()
    };
    let p_pert_comp_zero_mean: Vec<f64> = {
        let mut vals = p_pert_comp.clone();
        let mean = vals.iter().sum::<f64>() / vals.len().max(1) as f64;
        for v in &mut vals {
            *v -= mean;
        }
        vals
    };
    let checker = checkerboard_metric(mesh, &p_comp_zero_mean, dyn_pressure.max(1e-12));
    let rho_checker = checkerboard_metric(mesh, &rho_comp_zero_mean, density as f64);
    let p_pert_checker = checkerboard_metric(mesh, &p_pert_comp_zero_mean, dyn_pressure.max(1e-12));

    eprintln!(
        "[low_mach_equivalence][{case_name}] cells={} steps={steps} dt={dt:.3e} dtau={dtau:.3e} acoustic_cfl={:.1} ux_l2/u_in={ux_l2_over_uin:.3} p_l2/q={p_l2_over_q:.3} checker={checker:.3} rho_checker={rho_checker:.3} ppert_checker={p_pert_checker:.3}",
        mesh.num_cells(),
        (sound_speed * dt / h_min.max(1e-12)),
    );

    assert!(
        p_pert_checker < 6.0,
        "[{case_name}] compressible dynamic-pressure checkerboarding too high: {p_pert_checker:.3} (dt={dt:.3e} dtau={dtau:.3e})"
    );
    assert!(
        ux_l2_over_uin < 1.2,
        "[{case_name}] u_x mismatch too large (rms/u_in={ux_l2_over_uin:.3})"
    );
    assert!(
        p_l2_over_q < 20.0,
        "[{case_name}] p mismatch too large after mean removal (rms/q={p_l2_over_q:.3})"
    );
}

#[test]
#[ignore]
fn low_mach_backwards_step_incompressible_matches_compressible() {
    let cell = env_f64("CFD2_LOW_MACH_EQ_CELL", 0.08);
    let smooth_iters = env_usize("CFD2_LOW_MACH_EQ_SMOOTH_ITERS", 20);
    let mesh = build_backwards_step_mesh(cell, smooth_iters);
    run_low_mach_equivalence_case("backwards_step", &mesh, 0.75);
}

#[test]
#[ignore]
fn low_mach_channel_obstacle_incompressible_matches_compressible() {
    let cell = env_f64("CFD2_LOW_MACH_EQ_CELL", 0.08);
    let smooth_iters = env_usize("CFD2_LOW_MACH_EQ_SMOOTH_ITERS", 20);
    let mesh = build_channel_obstacle_mesh(cell, smooth_iters);
    run_low_mach_equivalence_case("channel_obstacle", &mesh, 0.5);
}
