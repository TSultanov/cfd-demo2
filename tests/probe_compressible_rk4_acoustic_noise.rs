//! PROBE (diagnostic, not a gate): reproduce the GUI compressible + explicit
//! RK4 obstacle case headlessly and quantify the reported pressure noise and
//! the startup wave.
//!
//! Runs the exact GUI defaults (Air, channel+cylinder cut-cell mesh, RK4) on
//! the requested backend and reports:
//!   - the pressure range (wave amplitude) over time,
//!   - a high-frequency roughness metric (RMS of the deviation of each cell's
//!     pressure from its face-neighbor mean), normalized by the wave range,
//!   - PPM snapshots of the pressure field for visual comparison with the GUI.
//!
//! Backend is chosen with CFD2_PROBE_BACKEND = gpu | cpu-f32 | cpu-f64
//! (default gpu). Steps with CFD2_PROBE_STEPS (default 320).
#![cfg(all(feature = "dev-tests", feature = "cpu", feature = "meshgen", feature = "ui"))]

use cfd2::meshgen::{generate_cut_cell_mesh, ChannelWithObstacle};
use cfd2::sim::SolverDriver;
use cfd2::solver::mesh::Mesh;
use cfd2::solver::model::compressible_model_with_eos;
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::TimeScheme;
use cfd2::ui::model_defaults::gui_defaults_for;
use nalgebra::{Point2, Vector2};

fn air_eos() -> EosSpec {
    EosSpec::IdealGas {
        gamma: 1.4,
        gas_constant: 287.0,
        temperature: 300.0,
    }
}

fn gui_obstacle_mesh(cell: f64) -> Mesh {
    let geo = ChannelWithObstacle {
        length: 3.0,
        height: 1.0,
        obstacle_center: Point2::new(1.0, 0.51),
        obstacle_radius: 0.1,
    };
    generate_cut_cell_mesh(&geo, cell, cell, 1.2, Vector2::new(3.0, 1.0))
}

struct NoiseReport {
    p_min: f64,
    p_max: f64,
    roughness_rms: f64,
    roughness_max: f64,
}

/// Deviation of each cell's p from the mean of its face neighbors: ~0 for
/// smooth acoustic fields, ~noise amplitude for cell-to-cell speckle.
fn noise_metrics(mesh: &Mesh, p: &[f64]) -> NoiseReport {
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
    let mut p_min = f64::INFINITY;
    let mut p_max = f64::NEG_INFINITY;
    let mut acc = 0.0f64;
    let mut m = 0usize;
    let mut rmax = 0.0f64;
    for i in 0..n {
        p_min = p_min.min(p[i]);
        p_max = p_max.max(p[i]);
        if cnt[i] >= 2 {
            let r = (p[i] - sum[i] / cnt[i] as f64).abs();
            acc += r * r;
            m += 1;
            rmax = rmax.max(r);
        }
    }
    NoiseReport {
        p_min,
        p_max,
        roughness_rms: (acc / m.max(1) as f64).sqrt(),
        roughness_max: rmax,
    }
}

/// Nearest-cell rasterization of p over the bounding box -> binary PPM.
fn write_ppm(mesh: &Mesh, p: &[f64], path: &str, lo: f64, hi: f64) {
    let (w, h) = (600usize, 200usize);
    // For every pixel find the containing cell by scanning cells' bounding
    // boxes via a coarse grid of cell indices (cheap enough at this size).
    let nx = 300usize;
    let ny = 100usize;
    let mut grid: Vec<Vec<usize>> = vec![Vec::new(); nx * ny];
    for c in 0..mesh.num_cells() {
        let gx = ((mesh.cell_cx[c] / 3.0 * nx as f64) as usize).min(nx - 1);
        let gy = ((mesh.cell_cy[c] / 1.0 * ny as f64) as usize).min(ny - 1);
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
                        let d = (mesh.cell_cx[c] - x).powi(2) + (mesh.cell_cy[c] - y).powi(2);
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
            let t = ((p[best] - lo) / (hi - lo).max(1e-30)).clamp(0.0, 1.0);
            // blue -> green -> red
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

#[test]
fn probe_gui_compressible_rk4_noise() {
    let backend = std::env::var("CFD2_PROBE_BACKEND").unwrap_or_else(|_| "gpu".into());
    let steps: usize = std::env::var("CFD2_PROBE_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(320);
    let out_dir = std::env::var("CFD2_PROBE_OUT").unwrap_or_else(|_| ".".into());

    let mesh = gui_obstacle_mesh(0.03);
    eprintln!(
        "[probe] mesh: {} cells, {} faces",
        mesh.num_cells(),
        mesh.num_faces()
    );

    let eos = air_eos();
    let model = compressible_model_with_eos(eos).expect("compressible model");
    let mut params = gui_defaults_for("compressible").to_runtime_params(1.225, 1.81e-5, eos);
    params.time_scheme = TimeScheme::RK4;
    params.adaptive_dt = false;
    params.requested_dt = 5.0e-6;

    let initial_u = vec![(0.0, 0.0); mesh.num_cells()];
    let initial_p = vec![0.0; mesh.num_cells()];

    let mut build = match backend.as_str() {
        "gpu" => {
            let ctx = match pollster::block_on(cfd2::solver::gpu::context::GpuContext::new(
                None, None,
            )) {
                Ok(ctx) => ctx,
                Err(error) => {
                    eprintln!("[probe] no GPU adapter ({error}); skipping");
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
        "cpu-f32" | "cpu-f64" => {
            std::env::set_var(
                "CFD2_CPU_PRECISION",
                if backend == "cpu-f32" { "f32" } else { "f64" },
            );
            std::env::set_var("CFD2_CPU_THREADS", "4");
            pollster::block_on(SolverDriver::build_forced_cpu_transpiled(
                &mesh,
                model,
                &params,
                &initial_u,
                &initial_p,
            ))
            .expect("cpu build")
        }
        other => panic!("unknown CFD2_PROBE_BACKEND '{other}'"),
    };
    build.driver.apply_params(&params);

    let layout = build.driver.solver().model().state_layout.clone();
    let stride = layout.stride() as usize;
    let p_off = layout.offset_for("p").expect("p offset") as usize;

    let snapshot_steps = [steps / 5, (steps * 2) / 5, steps];
    let mut done = 0usize;
    for (k, &target) in snapshot_steps.iter().enumerate() {
        while done < target {
            let outcome = build.driver.step(false);
            if let Some(reason) = outcome.diverged {
                panic!("diverged at step {done}: {reason:?}");
            }
            done += 1;
        }
        let state = pollster::block_on(build.driver.solver().read_state_f32());
        let p: Vec<f64> = state
            .chunks_exact(stride)
            .map(|row| row[p_off] as f64)
            .collect();
        let rep = noise_metrics(&mesh, &p);
        let range = (rep.p_max - rep.p_min).max(1e-30);
        eprintln!(
            "[probe:{backend}] step {done} t={:.4e}: p=[{:.4}, {:.4}] range={:.4e} \
             roughness rms={:.4e} ({:.2}% of range) max={:.4e} ({:.2}% of range)",
            build.driver.solver().time(),
            rep.p_min,
            rep.p_max,
            range,
            rep.roughness_rms,
            rep.roughness_rms / range * 100.0,
            rep.roughness_max,
            rep.roughness_max / range * 100.0,
        );
        write_ppm(
            &mesh,
            &p,
            &format!("{out_dir}/probe_p_{backend}_{k}.ppm"),
            rep.p_min,
            rep.p_max,
        );
        // Same field under the production representable-precision display
        // floor (64 f32 ULPs of the field magnitude), for visual comparison.
        let floor_span =
            64.0 * f64::from(f32::EPSILON) * rep.p_min.abs().max(rep.p_max.abs());
        if rep.p_max - rep.p_min < floor_span {
            let mid = 0.5 * (rep.p_min + rep.p_max);
            write_ppm(
                &mesh,
                &p,
                &format!("{out_dir}/probe_p_{backend}_{k}_floored.ppm"),
                mid - 0.5 * floor_span,
                mid + 0.5 * floor_span,
            );
        }
    }
}
