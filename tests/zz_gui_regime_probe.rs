//! PROBE (temporary): replicate the GUI channel-with-obstacle regime headlessly
//! for structured vs unstructured, incompressible + all-Mach thermal.
//! GUI regime: Air (rho 1.225, mu 1.81e-5, psi 8.3e-6), inlet 0.011, SOU VanLeer,
//! BDF2, alpha 0.7/0.3, outer cap 8 + auto-converge 1e-3, adaptive dt CFL 0.9
//! (growth cap 1.2x), Schur+AMG preconditioner, 120x40 cells (h = 0.025), Brinkman
//! cylinder at (1.0, 0.51, r=0.1).
//! Run: CFD2_CPU_THREADS=6 cargo test --features "cpu meshgen" --test zz_gui_regime_probe -- --test-threads=1 --nocapture
#![cfg(feature = "cpu")]

use cfd2::meshgen::{generate_cut_cell_mesh, ChannelWithObstacle};
use cfd2::sim::{RuntimeParams, SolverDriver};
use cfd2::solver::banded_schur::CoupledPrecondKind;
use cfd2::solver::cpu::structured::StructuredModelSolver;
use cfd2::solver::gpu::structured::{BcComp, Edge, StructuredGrid};
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::model::{
    allmach_thermal_model, allmach_thermal_structured_model, incompressible_momentum_model,
    incompressible_momentum_structured_model,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{GpuLowMachPrecondModel, PreconditionerType, TimeScheme};
use nalgebra::Vector2;
use std::time::Instant;

const NX: usize = 120;
const NY: usize = 40;
const LX: f64 = 3.0;
const LY: f64 = 1.0;
const H: f64 = LX / NX as f64; // 0.025
const U_IN: f64 = 0.011;
const RHO: f64 = 1.225;
const MU: f64 = 1.81e-5;
const PSI: f64 = 8.3e-6;
const CX: f64 = 1.0;
const CY: f64 = 0.51;
const R: f64 = 0.1;
const STEPS: usize = 150;

fn k1(v: f32) -> BcComp {
    BcComp { kind: 1, value: v }
}
fn k2() -> BcComp {
    BcComp { kind: 2, value: 0.0 }
}

fn threads() -> usize {
    std::env::var("CFD2_CPU_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6)
}

fn seed(s: &mut StructuredModelSolver, name: &str, v: f64) {
    if s.field_offset(name).is_some() {
        s.set_named_field(name, move |_, _| v);
    }
}

/// GUI adaptive dt (structured_pin_dt, no sound speed for these models).
fn pin_dt(s: &mut StructuredModelSolver, prev_max_vel: f64) {
    let wave = prev_max_vel.max(U_IN);
    let cur = s.dt();
    let mut next = 0.9 * H / wave;
    if next > cur * 1.2 {
        next = cur * 1.2;
    }
    s.set_dt(next.clamp(1e-9, 100.0));
}

fn run_structured(name: &str, allmach: bool, precond: CoupledPrecondKind) {
    let model = if allmach {
        allmach_thermal_structured_model().expect("model")
    } else {
        incompressible_momentum_structured_model().expect("model")
    };
    let s_per = model.system.unknowns_per_cell() as usize;
    let mut solver = StructuredModelSolver::with_config(
        StructuredGrid::new(NX, NY, LX, LY),
        &model,
        0.02,
        8,
        Scheme::SecondOrderUpwindVanLeer,
        TimeScheme::BDF2,
    )
    .expect("solver");
    solver.set_engine(cfd2::solver::cpu::CpuEngine::Transpiled, threads());
    solver.set_fluid(RHO, MU);
    solver.set_preconditioner(precond);
    solver.set_outer_iters(8);
    solver.set_outer_auto_converge(true);
    solver.set_alpha_u(0.7);
    solver.set_alpha_p(0.3);

    if allmach {
        // seed_structured_state (app.rs) exactly:
        let t_ref = 1.0;
        seed(&mut solver, "psi", PSI);
        seed(&mut solver, "rho", RHO);
        seed(&mut solver, "rho_t_ref", RHO * t_ref);
        seed(&mut solver, "T", t_ref);
        seed(&mut solver, "t_ref", t_ref);
        seed(&mut solver, "rho_floor", PSI * 1.0e-5);
        seed(&mut solver, "psi_ref", PSI);
        // GUI: uref_min slider = 1.0 for allmach preset.
        let u_ref = 2.0 * U_IN.max(1.0);
        seed(&mut solver, "psi_precond", PSI.max(1.0 / (u_ref * u_ref)));
        seed(&mut solver, "u_ref", u_ref);
        seed(&mut solver, "precond_mask", 1.0);
        seed(&mut solver, "dt_local", 0.0);
    }

    // Brinkman cylinder.
    if let Some(off) = solver.field_offset("ibm_penalty_U") {
        solver.set_state(off, move |x: f64, y: f64| {
            if (x - CX).hypot(y - CY) < R {
                -1.0e5
            } else {
                0.0
            }
        });
    }

    // Channel BCs: inlet left, p-outlet right, no-slip walls (pad extra comps k2).
    let u_in = U_IN as f32;
    solver.set_boundaries(move |e: Edge, _x: f64, _y: f64| {
        let mut comps = match e {
            Edge::Left => vec![k1(u_in), k1(0.0), k2()],
            Edge::Right => vec![k2(), k2(), k1(0.0)],
            _ => vec![k1(0.0), k1(0.0), k2()],
        };
        while comps.len() < s_per {
            comps.push(k2());
        }
        (1u32, comps.into_iter().take(s_per).collect())
    });

    let n = NX * NY;
    let p_off = solver.field_offset("p").unwrap_or(2);
    let mut prev_max_vel = 0.0f64;
    let mut ms_tail = Vec::new();
    for step in 0..STEPS {
        pin_dt(&mut solver, prev_max_vel);
        let t = Instant::now();
        solver.step();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if step >= STEPS - 50 {
            ms_tail.push(ms);
        }
        let ux = solver.state_field(0);
        let uy = solver.state_field(1);
        let mut max_u = 0.0f64;
        for i in 0..n {
            let m = ux[i].hypot(uy[i]);
            if m.is_finite() {
                max_u = max_u.max(m);
            }
        }
        prev_max_vel = max_u;
        if step % 15 == 0 || step == STEPS - 1 {
            let st = solver.last_stats();
            println!(
                "[{name}] step {step:3}: {ms:7.1} ms dt={:.3e} max|U|={:.4e} outer={} lin={} res={:.2e} du={:.2e} dp={:.2e}",
                solver.dt(),
                max_u,
                st.outer_iters,
                st.linear_iters,
                st.linear_res,
                st.outer_du,
                st.outer_dp
            );
        }
    }
    // Obstacle diagnostics: solid-interior max|U| (r<0.6R) and near-field p ptp (R..1.5R).
    let ux = solver.state_field(0);
    let uy = solver.state_field(1);
    let p = solver.state_field(p_off);
    let grid = solver.grid();
    let (mut solid_max_u, mut pmin, mut pmax) = (0.0f64, f64::INFINITY, f64::NEG_INFINITY);
    for i in 0..n {
        let (x, y) = grid.cell_center(i);
        let d = (x - CX).hypot(y - CY);
        if d < 0.6 * R {
            solid_max_u = solid_max_u.max(ux[i].hypot(uy[i]));
        }
        if d > R && d < 1.5 * R {
            pmin = pmin.min(p[i]);
            pmax = pmax.max(p[i]);
        }
    }
    let mean_ms = ms_tail.iter().sum::<f64>() / ms_tail.len() as f64;
    println!(
        "[{name}] SUMMARY: mean {mean_ms:.1} ms/step (last 50) | max|U|={prev_max_vel:.4e} | solid max|U|={solid_max_u:.3e} | near-p ptp={:.3e}",
        pmax - pmin
    );

    // RING metric (interface speckle discriminator): |U| stats over FLUID cells
    // in the 1-2 cell interface ring (R < d < 1.3R) vs the ambient upstream mean
    // (2R < d < 4R, x < CX). Speckle = ring max/mean far above ambient.
    let sp: Vec<f64> = solver
        .field_offset("ibm_penalty_U")
        .map(|off| solver.state_field(off))
        .unwrap_or_else(|| vec![0.0; n]);
    let (mut ring_max, mut ring_sum, mut ring_n) = (0.0f64, 0.0f64, 0usize);
    let (mut amb_sum, mut amb_n) = (0.0f64, 0usize);
    for i in 0..n {
        let (x, y) = grid.cell_center(i);
        let d = (x - CX).hypot(y - CY);
        let m = ux[i].hypot(uy[i]);
        if sp[i] == 0.0 && d > R && d < 1.3 * R {
            ring_max = ring_max.max(m);
            ring_sum += m;
            ring_n += 1;
        }
        if d > 2.0 * R && d < 4.0 * R && x < CX {
            amb_sum += m;
            amb_n += 1;
        }
    }
    let ring_mean = ring_sum / ring_n.max(1) as f64;
    let amb_mean = amb_sum / amb_n.max(1) as f64;
    println!(
        "[{name}] RING: max|U|={ring_max:.4e} mean|U|={ring_mean:.4e} ({ring_n} cells) | ambient mean|U|={amb_mean:.4e} ({amb_n} cells) | ring_max/amb={:.2} ring_mean/amb={:.2}",
        ring_max / amb_mean.max(1e-30),
        ring_mean / amb_mean.max(1e-30)
    );

    // RING ROUGHNESS (speckle discriminator): red/blue speckle = cell-to-cell
    // |U| alternation. For each fluid ring cell, take the max |m_i - m_j| over
    // its fluid 4-neighbours; report the ring mean of that, normalized by the
    // ambient mean. A smooth diverted flow gives O(grad) values; speckle gives
    // values comparable to the field magnitude itself.
    let (mut rough_sum, mut rough_max, mut rough_n) = (0.0f64, 0.0f64, 0usize);
    for j in 0..NY {
        for i in 0..NX {
            let idx = j * NX + i;
            let (x, y) = grid.cell_center(idx);
            let d = (x - CX).hypot(y - CY);
            if sp[idx] != 0.0 || d <= R || d >= 1.3 * R {
                continue;
            }
            let m = ux[idx].hypot(uy[idx]);
            let mut worst = 0.0f64;
            let neighbors = [
                (j > 0, idx.wrapping_sub(NX)),
                (i > 0, idx.wrapping_sub(1)),
                (i + 1 < NX, idx + 1),
                (j + 1 < NY, idx + NX),
            ];
            for (valid, nb) in neighbors {
                if valid && sp[nb] == 0.0 {
                    worst = worst.max((m - ux[nb].hypot(uy[nb])).abs());
                }
            }
            rough_sum += worst;
            rough_max = rough_max.max(worst);
            rough_n += 1;
        }
    }
    let rough_mean = rough_sum / rough_n.max(1) as f64;
    println!(
        "[{name}] RING-ROUGH: mean d|U|={rough_mean:.4e} max d|U|={rough_max:.4e} | mean/amb={:.2} max/amb={:.2}",
        rough_mean / amb_mean.max(1e-30),
        rough_max / amb_mean.max(1e-30)
    );

    // LEAK metric (M1 discriminator): pressure-slot face fluxes across every
    // solid-fluid mixed face of the cylinder, read from the structured `fluxes`
    // buffer `fluxes[(idx*4+k)*stride + c]` (k: 0=S,1=W,2=E,3=N, normal OUTWARD
    // from idx). `net` = signed mass flux out of the solid (cancels for
    // in-one-side-out-the-other leaks); `gross` = sum |phi| (the honest leak
    // magnitude). Normalized by rho*U_in*D (D=2R frontal-area mass-flux scale).
    let fluxes = solver.read_buffer("fluxes");
    let s_per_flux = s_per; // flux stride == coupled unknowns per cell
    let p_comp = 2usize; // coupled unknown order: [U_x, U_y, p, ...]
    let (mut leak_net, mut leak_gross, mut mixed_faces) = (0.0f64, 0.0f64, 0usize);
    for j in 0..NY {
        for i in 0..NX {
            let idx = j * NX + i;
            if sp[idx] == 0.0 {
                continue; // owner side: solid cells only
            }
            let neighbors = [
                (0usize, j > 0, idx.wrapping_sub(NX)),
                (1, i > 0, idx.wrapping_sub(1)),
                (2, i + 1 < NX, idx + 1),
                (3, j + 1 < NY, idx + NX),
            ];
            for (k, valid, nb) in neighbors {
                if !valid || sp[nb] != 0.0 {
                    continue; // interior-solid or domain-boundary face
                }
                let phi = fluxes[(idx * 4 + k) * s_per_flux + p_comp] as f64;
                leak_net += phi; // outward from solid
                leak_gross += phi.abs();
                mixed_faces += 1;
            }
        }
    }
    let scale = RHO * U_IN * 2.0 * R;
    println!(
        "[{name}] LEAK: gross={leak_gross:.4e} net={leak_net:.4e} over {mixed_faces} mixed faces | gross/(rho*Uin*D)={:.4e} net/(rho*Uin*D)={:.4e}",
        leak_gross / scale,
        leak_net / scale
    );
}

fn gui_params(allmach: bool) -> RuntimeParams {
    RuntimeParams {
        filter_sigma: 0.0,
        adaptive_dt: true,
        target_cfl: 0.9,
        requested_dt: 0.02,
        dtau: 0.0,
        log_convergence: false,
        log_every_steps: 100000,
        advection_scheme: Scheme::SecondOrderUpwindVanLeer,
        time_scheme: TimeScheme::BDF2,
        preconditioner: PreconditionerType::Amg,
        outer_iters: 8,
        outer_auto_converge: true,
        low_mach_model: GpuLowMachPrecondModel::Off,
        low_mach_theta_floor: 1e-6,
        low_mach_pressure_coupling_alpha: 1.0,
        alpha_u: 0.7,
        alpha_p: 0.3,
        inlet_velocity: U_IN as f32,
        density: RHO as f32,
        viscosity: MU as f32,
        eos: EosSpec::Constant,
        compressibility_psi: if allmach { PSI as f32 } else { 0.0 },
        outlet_back_pressure: 0.0,
        pressure_inlet: false,
        inlet_pressure: 0.0,
        allmach_precond_uref_min: if allmach { 1.0 } else { 0.2 },
    }
}

fn run_unstructured(name: &str, allmach: bool) {
    let geo = ChannelWithObstacle {
        length: LX,
        height: LY,
        obstacle_center: nalgebra::Point2::new(CX, CY),
        obstacle_radius: R,
    };
    let mesh = generate_cut_cell_mesh(&geo, H * 0.5, H, 1.2, Vector2::new(LX, LY));
    let n = mesh.num_cells();
    let params = gui_params(allmach);
    let model = if allmach {
        allmach_thermal_model().expect("model")
    } else {
        incompressible_momentum_model().expect("model")
    };
    let initial_u = vec![(0.0f64, 0.0f64); n];
    let initial_p = vec![0.0; n];
    let mut build = pollster::block_on(SolverDriver::build_forced_cpu(
        &mesh,
        model,
        &params,
        &initial_u,
        &initial_p,
    ))
    .expect("driver build");
    build.driver.apply_params(&params);
    let mut driver = build.driver;

    let mut ms_tail = Vec::new();
    for step in 0..STEPS {
        let t = Instant::now();
        let _ = driver.step(false);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if step >= STEPS - 50 {
            ms_tail.push(ms);
        }
        if step % 30 == 0 || step == STEPS - 1 {
            println!("[{name}] step {step:3}: {ms:7.1} ms");
        }
    }
    let state = pollster::block_on(driver.solver().read_state_f32());
    let layout = driver.solver().model().state_layout.clone();
    let stride = layout.stride() as usize;
    let mut max_u = 0.0f64;
    for i in 0..n {
        let ux = state[i * stride] as f64;
        let uy = state[i * stride + 1] as f64;
        let m = ux.hypot(uy);
        if m.is_finite() {
            max_u = max_u.max(m);
        }
    }
    let mean_ms = ms_tail.iter().sum::<f64>() / ms_tail.len() as f64;
    println!("[{name}] SUMMARY: {n} cells | mean {mean_ms:.1} ms/step (last 50) | max|U|={max_u:.4e}");
}

#[test]
fn a_structured_allmach_schur_amg() {
    run_structured("S-allmach-SchurAmg", true, CoupledPrecondKind::SchurAmg);
}
#[test]
fn b_structured_allmach_block_jacobi() {
    run_structured("S-allmach-BlockJacobi", true, CoupledPrecondKind::BlockJacobi);
}
#[test]
fn c_structured_incomp_schur_amg() {
    run_structured("S-incomp-SchurAmg", false, CoupledPrecondKind::SchurAmg);
}
#[test]
fn d_structured_incomp_block_jacobi() {
    run_structured("S-incomp-BlockJacobi", false, CoupledPrecondKind::BlockJacobi);
}
#[test]
fn e_unstructured_allmach() {
    run_unstructured("U-allmach", true);
}
#[test]
fn f_unstructured_incomp() {
    run_unstructured("U-incomp", false);
}
