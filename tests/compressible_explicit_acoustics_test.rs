//! Acoustics-level gates for the explicit RK4 compressible solver.
//!
//! An isentropic 200 Pa Gaussian pulse (~2.6e4 f32 quanta on the 105 kPa Air
//! base state, i.e. comfortably representable) is released on each topology
//! and must propagate as a clean cylindrical wave: transit speed between two
//! monitors within a few percent of c = sqrt(gamma R T), physical 2D decay,
//! bounded neighbor-mean roughness, and positive rho/p throughout.
//!
//! GPU gates skip without an adapter. Set CFD2_PROBE_OUT=<dir> to also dump
//! PPM snapshots of (p - P0) for visual inspection; CFD2_PROBE_REFINE=<n>
//! refines the structured case for dispersion studies.
#![cfg(all(feature = "dev-tests", feature = "cpu", feature = "meshgen", feature = "ui"))]

use cfd2::meshgen::{generate_cut_cell_mesh, ChannelWithObstacle};
use cfd2::sim::SolverDriver;
use cfd2::solver::gpu::structured::{BcComp, StructuredGpuSolver, StructuredGrid};
use cfd2::solver::mesh::Mesh;
use cfd2::solver::model::compressible_model_with_eos;
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::TimeScheme;
use cfd2::ui::model_defaults::gui_defaults_for;
use nalgebra::{Point2, Vector2};

const GAMMA: f64 = 1.4;
const R_GAS: f64 = 287.0;
const T0: f64 = 300.0;
const RHO0: f64 = 1.225;
const P0: f64 = RHO0 * R_GAS * T0; // 105472.5
const PULSE_AMP: f64 = 200.0;
const PULSE_SIGMA: f64 = 0.08;
const DT: f64 = 5.0e-6;

fn air_eos() -> EosSpec {
    EosSpec::IdealGas {
        gamma: GAMMA,
        gas_constant: R_GAS,
        temperature: T0,
    }
}

fn sound_speed() -> f64 {
    (GAMMA * R_GAS * T0).sqrt()
}

/// Isentropic pulse: p = P0 + A exp(-r^2 / 2 sigma^2), rho = RHO0 (p/P0)^(1/gamma).
fn pulse_p(x: f64, y: f64, cx: f64, cy: f64) -> f64 {
    let r2 = (x - cx).powi(2) + (y - cy).powi(2);
    P0 + PULSE_AMP * (-r2 / (2.0 * PULSE_SIGMA * PULSE_SIGMA)).exp()
}

struct Snapshot {
    time: f64,
    ring_radius: f64,
    amplitude: f64,
    roughness_rms: f64,
    roughness_max: f64,
}

/// Ring-peak radius of |p - P0| (radial-bin mean) and roughness metrics.
fn analyze(
    centers: &[(f64, f64)],
    neighbors_mean: &dyn Fn(&[f64]) -> Vec<Option<f64>>,
    p: &[f64],
    pulse_center: (f64, f64),
    time: f64,
) -> Snapshot {
    let dr = 0.015;
    let mut bins = vec![(0.0f64, 0usize); 200];
    let mut amplitude = 0.0f64;
    for (i, &(x, y)) in centers.iter().enumerate() {
        let dp = (p[i] - P0).abs();
        amplitude = amplitude.max(dp);
        let r = ((x - pulse_center.0).powi(2) + (y - pulse_center.1).powi(2)).sqrt();
        let b = (r / dr) as usize;
        if b < bins.len() {
            bins[b].0 += dp;
            bins[b].1 += 1;
        }
    }
    let mut ring_radius = 0.0;
    let mut best = 0.0;
    for (b, &(sum, cnt)) in bins.iter().enumerate() {
        if cnt == 0 {
            continue;
        }
        let mean = sum / cnt as f64;
        if mean > best {
            best = mean;
            ring_radius = (b as f64 + 0.5) * dr;
        }
    }
    let means = neighbors_mean(p);
    let mut acc = 0.0;
    let mut m = 0usize;
    let mut rmax = 0.0f64;
    for (i, mean) in means.iter().enumerate() {
        if let Some(mean) = mean {
            let r = (p[i] - mean).abs();
            acc += r * r;
            m += 1;
            rmax = rmax.max(r);
        }
    }
    Snapshot {
        time,
        ring_radius,
        amplitude,
        roughness_rms: (acc / m.max(1) as f64).sqrt(),
        roughness_max: rmax,
    }
}

fn report(kind: &str, snaps: &[Snapshot]) {
    let c = sound_speed();
    for s in snaps {
        eprintln!(
            "[pulse:{kind}] t={:.4e} ring_r={:.3} amp={:.3} rough_rms={:.4e} ({:.3}% of amp) max={:.4e} ({:.2}% of amp)",
            s.time, s.ring_radius, s.amplitude,
            s.roughness_rms, s.roughness_rms / s.amplitude * 100.0,
            s.roughness_max, s.roughness_max / s.amplitude * 100.0,
        );
    }
    if snaps.len() >= 2 {
        let first = &snaps[0];
        let last = &snaps[snaps.len() - 1];
        let speed = (last.ring_radius - first.ring_radius) / (last.time - first.time);
        eprintln!(
            "[pulse:{kind}] ring-peak speed {speed:.1} m/s vs physical c={c:.1} ({:+.1}%)",
            (speed / c - 1.0) * 100.0
        );
    }
}

/// Differential arrival-time wave speed between two monitors: offset-free
/// (the outgoing ring detaches at r0 ~ sigma, so distance/t at one monitor is
/// biased; the peak-to-peak transit between two radii is not).
fn report_arrival(
    kind: &str,
    d1: f64,
    h1: &[(f64, f64)],
    d2: f64,
    h2: &[(f64, f64)],
) -> (f64, f64, f64) {
    let c = sound_speed();
    let peak = |h: &[(f64, f64)]| {
        let (mut t_peak, mut best) = (0.0, f64::NEG_INFINITY);
        for &(t, p) in h {
            if p > best {
                best = p;
                t_peak = t;
            }
        }
        (t_peak, best - P0)
    };
    let (t1, a1) = peak(h1);
    let (t2, a2) = peak(h2);
    let speed = (d2 - d1) / (t2 - t1);
    eprintln!(
        "[pulse:{kind}] two-monitor transit: peaks {a1:.2} Pa @ t={t1:.4e} (r={d1:.3}) and {a2:.2} Pa @ t={t2:.4e} (r={d2:.3}) -> speed {speed:.1} m/s vs c={c:.1} ({:+.1}%)",
        (speed / c - 1.0) * 100.0
    );
    (speed, a1, a2)
}

/// Gate thresholds for one pulse run. Margins are wide enough for backend
/// and adapter variation yet tight enough that a broken KT dissipation
/// (checkerboard), a preconditioning leak into the time-accurate flux, or a
/// stage/dt bookkeeping error fails immediately.
fn assert_clean_pulse(kind: &str, speed: f64, a1: f64, a2: f64, snaps: &[Snapshot]) {
    let c = sound_speed();
    let rel = speed / c - 1.0;
    assert!(
        rel.abs() < 0.06,
        "[{kind}] transit speed {speed:.1} m/s deviates {:+.1}% from c={c:.1}",
        rel * 100.0
    );
    assert!(
        a1 > 0.1 * PULSE_AMP,
        "[{kind}] first-monitor peak {a1:.2} Pa lost against the {PULSE_AMP} Pa pulse"
    );
    assert!(
        a2 > 0.3 * a1 && a2 < a1,
        "[{kind}] second-monitor peak {a2:.2} Pa vs {a1:.2} Pa is outside physical 2D decay"
    );
    let last = snaps.last().expect("snapshots");
    assert!(
        last.amplitude.is_finite() && last.amplitude > 0.0,
        "[{kind}] final amplitude invalid: {}",
        last.amplitude
    );
    let rms_frac = last.roughness_rms / last.amplitude;
    let max_frac = last.roughness_max / last.amplitude;
    assert!(
        rms_frac < 0.025,
        "[{kind}] final roughness rms {:.3}% of amplitude (checkerboard?)",
        rms_frac * 100.0
    );
    assert!(
        max_frac < 0.12,
        "[{kind}] final max roughness {:.2}% of amplitude (local instability?)",
        max_frac * 100.0
    );
}

/// Nearest-cell rasterization of (p - P0) -> binary PPM, symmetric range.
fn write_dp_ppm(centers: &[(f64, f64)], p: &[f64], path: &str) {
    let (w, h) = (600usize, 200usize);
    let mut amp = 0.0f64;
    for (i, _) in centers.iter().enumerate() {
        amp = amp.max((p[i] - P0).abs());
    }
    let amp = amp.max(1e-30);
    let nx = 300usize;
    let ny = 100usize;
    let mut grid: Vec<Vec<usize>> = vec![Vec::new(); nx * ny];
    for (c, &(x, y)) in centers.iter().enumerate() {
        let gx = ((x / 3.0 * nx as f64) as usize).min(nx - 1);
        let gy = ((y / 1.0 * ny as f64) as usize).min(ny - 1);
        grid[gy * nx + gx].push(c);
    }
    let mut img = vec![0u8; w * h * 3];
    for py in 0..h {
        for px in 0..w {
            let x = (px as f64 + 0.5) / w as f64 * 3.0;
            let y = 1.0 - (py as f64 + 0.5) / h as f64 * 1.0;
            let gx = ((x / 3.0 * nx as f64) as usize).min(nx - 1);
            let gy = ((y / 1.0 * ny as f64) as usize).min(ny - 1);
            let mut best = usize::MAX;
            let mut best_d = f64::INFINITY;
            for dgy in gy.saturating_sub(1)..(gy + 2).min(ny) {
                for dgx in gx.saturating_sub(1)..(gx + 2).min(nx) {
                    for &c in &grid[dgy * nx + dgx] {
                        let d = (centers[c].0 - x).powi(2) + (centers[c].1 - y).powi(2);
                        if d < best_d {
                            best_d = d;
                            best = c;
                        }
                    }
                }
            }
            let idx = (py * w + px) * 3;
            if best == usize::MAX {
                continue;
            }
            let t = (((p[best] - P0) / amp) * 0.5 + 0.5).clamp(0.0, 1.0);
            let (r, g, b) = if t < 0.5 {
                let s = t * 2.0;
                (0.0, s, 1.0 - s)
            } else {
                let s = (t - 0.5) * 2.0;
                (s, 1.0 - s, 0.0)
            };
            img[idx] = (r * 255.0) as u8;
            img[idx + 1] = (g * 255.0) as u8;
            img[idx + 2] = (b * 255.0) as u8;
        }
    }
    let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
    out.extend_from_slice(&img);
    std::fs::write(path, out).unwrap();
}

fn unstructured_neighbor_means(mesh: &Mesh) -> impl Fn(&[f64]) -> Vec<Option<f64>> + '_ {
    move |p: &[f64]| {
        let n = mesh.num_cells();
        let mut sum = vec![0.0f64; n];
        let mut cnt = vec![0u32; n];
        for f in 0..mesh.num_faces() {
            let Some(nb) = mesh.face_neighbor[f] else {
                continue;
            };
            let o = mesh.face_owner[f];
            sum[o] += p[nb];
            cnt[o] += 1;
            sum[nb] += p[o];
            cnt[nb] += 1;
        }
        (0..n)
            .map(|i| (cnt[i] >= 3).then(|| sum[i] / cnt[i] as f64))
            .collect()
    }
}

fn unstructured_pulse(backend: &str) {
    let geo = ChannelWithObstacle {
        length: 3.0,
        height: 1.0,
        obstacle_center: Point2::new(1.0, 0.51),
        obstacle_radius: 0.1,
    };
    let mesh = generate_cut_cell_mesh(&geo, 0.03, 0.03, 1.2, Vector2::new(3.0, 1.0));
    let eos = air_eos();
    let model = compressible_model_with_eos(eos).expect("compressible model");
    let mut params = gui_defaults_for("compressible").to_runtime_params(RHO0 as f32, 1.81e-5, eos);
    params.time_scheme = TimeScheme::RK4;
    params.adaptive_dt = false;
    params.requested_dt = DT as f32;
    params.inlet_velocity = 0.0;

    let initial_u = vec![(0.0, 0.0); mesh.num_cells()];
    let initial_p = vec![0.0; mesh.num_cells()];
    let mut build = match backend {
        "gpu" => {
            let ctx = match pollster::block_on(cfd2::solver::gpu::context::GpuContext::new(
                None, None,
            )) {
                Ok(ctx) => ctx,
                Err(error) => {
                    eprintln!("[pulse] no GPU adapter ({error}); skipping");
                    return;
                }
            };
            pollster::block_on(SolverDriver::build(
                &mesh,
                model,
                &params,
                &initial_u,
                &initial_p,
                Some(ctx.device),
                Some(ctx.queue),
            ))
            .expect("gpu build")
        }
        "cpu" => pollster::block_on(SolverDriver::build_forced_cpu_transpiled(
            &mesh,
            model,
            &params,
            &initial_u,
            &initial_p,
        ))
        .expect("cpu build"),
        other => panic!("unknown pulse backend '{other}'"),
    };
    build.driver.apply_params(&params);

    // Overwrite the uniform IC with the isentropic pulse (away from the obstacle).
    let center = (2.2, 0.5);
    let n = mesh.num_cells();
    let mut rho = vec![0.0; n];
    let mut rho_e = vec![0.0; n];
    let mut p_ic = vec![0.0; n];
    let mut t_ic = vec![0.0; n];
    for i in 0..n {
        let p = pulse_p(mesh.cell_cx[i], mesh.cell_cy[i], center.0, center.1);
        let r = RHO0 * (p / P0).powf(1.0 / GAMMA);
        rho[i] = r;
        rho_e[i] = p / (GAMMA - 1.0);
        p_ic[i] = p;
        t_ic[i] = p / (r * R_GAS);
    }
    let solver = build.driver.solver_mut();
    solver.set_field_scalar("rho", &rho).expect("rho IC");
    solver.set_field_scalar("rho_e", &rho_e).expect("rho_e IC");
    solver.set_field_scalar("p", &p_ic).expect("p IC");
    solver.set_field_scalar("T", &t_ic).expect("T IC");

    let layout = build.driver.solver().model().state_layout.clone();
    let stride = layout.stride() as usize;
    let p_off = layout.offset_for("p").expect("p offset") as usize;
    let centers: Vec<(f64, f64)> = (0..n).map(|i| (mesh.cell_cx[i], mesh.cell_cy[i])).collect();
    let means = unstructured_neighbor_means(&mesh);
    let out_dir = std::env::var("CFD2_PROBE_OUT").ok();

    // Two monitors at different radii for the differential transit speed.
    let nearest = |target: (f64, f64)| {
        (0..n)
            .min_by(|&a, &b| {
                let da = (centers[a].0 - target.0).powi(2) + (centers[a].1 - target.1).powi(2);
                let db = (centers[b].0 - target.0).powi(2) + (centers[b].1 - target.1).powi(2);
                da.partial_cmp(&db).unwrap()
            })
            .unwrap()
    };
    let m1 = nearest((center.0 + 0.2, center.1));
    let m2 = nearest((center.0 + 0.45, center.1));
    let radius =
        |i: usize| ((centers[i].0 - center.0).powi(2) + (centers[i].1 - center.1).powi(2)).sqrt();

    let mut snaps = Vec::new();
    let mut h1 = Vec::new();
    let mut h2 = Vec::new();
    let mut done = 0usize;
    for (k, &target) in [40usize, 120, 200, 280].iter().enumerate() {
        while done < target {
            let outcome = build.driver.step(false);
            if let Some(reason) = outcome.diverged {
                panic!("diverged at step {done}: {reason:?}");
            }
            done += 1;
            if done % 2 == 0 {
                let state = pollster::block_on(build.driver.solver().read_state_f32());
                let time = build.driver.solver().time() as f64;
                h1.push((time, state[m1 * stride + p_off] as f64));
                h2.push((time, state[m2 * stride + p_off] as f64));
            }
        }
        let state = pollster::block_on(build.driver.solver().read_state_f32());
        let p: Vec<f64> = state
            .chunks_exact(stride)
            .map(|row| row[p_off] as f64)
            .collect();
        snaps.push(analyze(
            &centers,
            &means,
            &p,
            center,
            build.driver.solver().time() as f64,
        ));
        if let Some(out_dir) = &out_dir {
            write_dp_ppm(
                &centers,
                &p,
                &format!("{out_dir}/pulse_unstructured_{backend}_{k}.ppm"),
            );
        }
    }
    report(&format!("unstructured-{backend}"), &snaps);
    let (speed, a1, a2) = report_arrival(
        &format!("unstructured-{backend}"),
        radius(m1),
        &h1,
        radius(m2),
        &h2,
    );
    assert_clean_pulse(&format!("unstructured-{backend}"), speed, a1, a2, &snaps);
}

#[test]
fn unstructured_gpu_pulse_is_clean_acoustics() {
    unstructured_pulse("gpu");
}

#[test]
fn unstructured_cpu_pulse_is_clean_acoustics() {
    unstructured_pulse("cpu");
}

fn structured_pulse(backend: &str) {
    let model =
        cfd2::solver::model::compressible_structured_model().expect("structured compressible");
    let refine: usize = std::env::var("CFD2_PROBE_REFINE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let (nx, ny, lx, ly) = (100usize * refine, 34usize * refine, 3.0f64, 1.0f64);
    let grid = StructuredGrid::new(nx, ny, lx, ly);
    let center = (1.5, 0.5);

    let seed = |set_named: &mut dyn FnMut(&str, &dyn Fn(f64, f64) -> f64)| {
        set_named("rho", &|x, y| {
            RHO0 * (pulse_p(x, y, center.0, center.1) / P0).powf(1.0 / GAMMA)
        });
        set_named("rho_e", &|x, y| {
            pulse_p(x, y, center.0, center.1) / (GAMMA - 1.0)
        });
        set_named("p", &|x, y| pulse_p(x, y, center.0, center.1));
        set_named("T", &|x, y| {
            let p = pulse_p(x, y, center.0, center.1);
            p / (RHO0 * (p / P0).powf(1.0 / GAMMA) * R_GAS)
        });
    };

    // Zero-gradient walls on every edge (reflective for the pulse).
    let bc = |unknowns: usize| {
        move |_edge: cfd2::solver::gpu::structured::Edge, _x: f64, _y: f64| {
            (
                3u32,
                vec![
                    BcComp {
                        kind: 0,
                        value: 0.0,
                    };
                    unknowns
                ],
            )
        }
    };

    let centers: Vec<(f64, f64)> = (0..nx * ny)
        .map(|i| {
            let gi = i % nx;
            let gj = i / nx;
            (
                (gi as f64 + 0.5) * lx / nx as f64,
                (gj as f64 + 0.5) * ly / ny as f64,
            )
        })
        .collect();
    let means = |p: &[f64]| -> Vec<Option<f64>> {
        (0..nx * ny)
            .map(|i| {
                let gi = i % nx;
                let gj = i / nx;
                if gi == 0 || gj == 0 || gi == nx - 1 || gj == ny - 1 {
                    return None;
                }
                Some((p[i - 1] + p[i + 1] + p[i - nx] + p[i + nx]) / 4.0)
            })
            .collect()
    };

    let out_dir = std::env::var("CFD2_PROBE_OUT").ok();
    // Two monitors at different radii for the differential transit speed.
    let nearest = |target: (f64, f64)| {
        (0..nx * ny)
            .min_by(|&a, &b| {
                let da = (centers[a].0 - target.0).powi(2) + (centers[a].1 - target.1).powi(2);
                let db = (centers[b].0 - target.0).powi(2) + (centers[b].1 - target.1).powi(2);
                da.partial_cmp(&db).unwrap()
            })
            .unwrap()
    };
    let m1 = nearest((center.0 + 0.25, center.1));
    let m2 = nearest((center.0 + 0.55, center.1));
    let radius =
        |i: usize| ((centers[i].0 - center.0).powi(2) + (centers[i].1 - center.1).powi(2)).sqrt();

    let backend_label = backend.to_string();
    let run = |packed: Vec<Vec<f32>>, times: Vec<f64>, layout: &cfd2::solver::model::backend::state_layout::StateLayout| {
        let stride = layout.stride() as usize;
        let p_off = layout.offset_for("p").expect("p offset") as usize;
        let mut snaps = Vec::new();
        for (k, (state, time)) in packed.iter().zip(&times).enumerate() {
            let p: Vec<f64> = state
                .chunks_exact(stride)
                .map(|row| row[p_off] as f64)
                .collect();
            snaps.push(analyze(&centers, &means, &p, center, *time));
            if let Some(out_dir) = &out_dir {
                write_dp_ppm(
                    &centers,
                    &p,
                    &format!("{out_dir}/pulse_structured_{backend_label}_{k}.ppm"),
                );
            }
        }
        snaps
    };

    let snapshot_steps = [40 * refine, 120 * refine, 200 * refine, 320 * refine];
    let dt = DT / refine as f64;
    if backend == "gpu" {
        let ctx = match pollster::block_on(cfd2::solver::gpu::context::GpuContext::new(None, None))
        {
            Ok(ctx) => ctx,
            Err(error) => {
                eprintln!("[pulse] no GPU adapter ({error}); skipping");
                return;
            }
        };
        let mut solver = StructuredGpuSolver::with_config(
            ctx,
            grid,
            &model,
            dt,
            1,
            cfd2::solver::scheme::Scheme::SecondOrderUpwindVanLeer,
            TimeScheme::RK4,
        )
        .expect("structured gpu solver");
        solver.set_fluid(RHO0, 1.81e-5);
        solver.set_eos(air_eos().runtime_params());
        let unknowns = solver.unknowns();
        solver.set_boundaries(bc(unknowns));
        let mut set_named = |name: &str, f: &dyn Fn(f64, f64) -> f64| {
            solver.set_named_field(name, f);
        };
        seed(&mut set_named);

        let layout = solver.state_layout().clone();
        let stride = layout.stride() as usize;
        let p_off = layout.offset_for("p").expect("p offset") as usize;
        let mut packed = Vec::new();
        let mut times = Vec::new();
        let mut h1 = Vec::new();
        let mut h2 = Vec::new();
        let mut done = 0usize;
        for &target in &snapshot_steps {
            while done < target {
                solver.step();
                done += 1;
                if done % 2 == 0 {
                    let state = solver.packed_state_f32();
                    h1.push((solver.time(), state[m1 * stride + p_off] as f64));
                    h2.push((solver.time(), state[m2 * stride + p_off] as f64));
                }
            }
            packed.push(solver.packed_state_f32());
            times.push(solver.time());
        }
        let snaps = run(packed, times, &layout);
        report("structured-gpu", &snaps);
        let (speed, a1, a2) =
            report_arrival("structured-gpu", radius(m1), &h1, radius(m2), &h2);
        assert_clean_pulse("structured-gpu", speed, a1, a2, &snaps);
    } else {
        let mut solver = cfd2::solver::cpu::structured::StructuredModelSolver::with_config(
            grid,
            &model,
            dt,
            1,
            cfd2::solver::scheme::Scheme::SecondOrderUpwindVanLeer,
            TimeScheme::RK4,
        )
        .expect("structured cpu solver");
        solver.set_engine(cfd2::solver::cpu::CpuEngine::Transpiled, 4);
        solver.set_fluid(RHO0, 1.81e-5);
        solver.set_eos(air_eos().runtime_params());
        let unknowns = solver.unknowns();
        solver.set_boundaries(bc(unknowns));
        let mut set_named = |name: &str, f: &dyn Fn(f64, f64) -> f64| {
            solver.set_named_field(name, f);
        };
        seed(&mut set_named);

        let layout = solver.state_layout().clone();
        let stride = layout.stride() as usize;
        let p_off = layout.offset_for("p").expect("p offset") as usize;
        let mut packed = Vec::new();
        let mut times = Vec::new();
        let mut h1 = Vec::new();
        let mut h2 = Vec::new();
        let mut done = 0usize;
        for &target in &snapshot_steps {
            while done < target {
                solver.step();
                done += 1;
                if done % 2 == 0 {
                    let state = solver.packed_state_f32();
                    h1.push((solver.time(), state[m1 * stride + p_off] as f64));
                    h2.push((solver.time(), state[m2 * stride + p_off] as f64));
                }
            }
            packed.push(solver.packed_state_f32());
            times.push(solver.time());
        }
        let snaps = run(packed, times, &layout);
        report("structured-cpu", &snaps);
        let (speed, a1, a2) =
            report_arrival("structured-cpu", radius(m1), &h1, radius(m2), &h2);
        assert_clean_pulse("structured-cpu", speed, a1, a2, &snaps);
    }
}

#[test]
fn structured_gpu_pulse_is_clean_acoustics() {
    structured_pulse("gpu");
}

#[test]
fn structured_cpu_pulse_is_clean_acoustics() {
    structured_pulse("cpu");
}
