#![cfg(all(feature = "meshgen", feature = "dev-tests"))]

use cfd2::solver::mesh::geometry::ChannelWithObstacle;
use cfd2::solver::mesh::{generate_cut_cell_mesh, Mesh};
use cfd2::solver::model::helpers::{
    SolverCompressibleIdealGasExt, SolverCompressibleInletExt, SolverFieldAliasesExt,
    SolverInletVelocityExt, SolverRuntimeParamsExt,
};
use cfd2::solver::model::{compressible_model, incompressible_momentum_model};
use cfd2::solver::options::{
    GpuLowMachPrecondModel, PreconditionerType, SteppingMode, TimeScheme,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{SolverConfig, UnifiedSolver};
use image::{Rgb, RgbImage};
use nalgebra::{Point2, Vector2};
use std::env;
use std::fs;
use std::path::PathBuf;

fn build_mesh(cell: f64, smooth_alpha: f64, smooth_iters: usize) -> Mesh {
    let length = 3.0;
    let height = 1.0;
    let domain_size = Vector2::new(length, height);
    let geo = ChannelWithObstacle {
        length,
        height,
        obstacle_center: Point2::new(1.0, 0.51),
        obstacle_radius: 0.1,
    };
    let mut mesh = generate_cut_cell_mesh(&geo, cell, cell, 1.0, domain_size);
    if smooth_iters > 0 {
        mesh.smooth(&geo, smooth_alpha, smooth_iters);
    }
    mesh
}

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

fn env_bool(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .map(|val| {
            let val = val.to_ascii_lowercase();
            matches!(val.as_str(), "1" | "true" | "yes" | "y" | "on")
        })
        .unwrap_or(default)
}

fn find_probe_index(mesh: &Mesh, x: f64, y: f64) -> usize {
    let mut best = 0usize;
    let mut best_dist = f64::INFINITY;
    for i in 0..mesh.num_cells() {
        let dx = mesh.cell_cx[i] - x;
        let dy = mesh.cell_cy[i] - y;
        let dist = dx * dx + dy * dy;
        if dist < best_dist {
            best_dist = dist;
            best = i;
        }
    }
    best
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

fn gradients(mesh: &Mesh, values: &[f64]) -> Vec<(f64, f64)> {
    let mut grads = vec![(0.0, 0.0); mesh.num_cells()];
    for i in 0..mesh.num_cells() {
        let cx = mesh.cell_cx[i];
        let cy = mesh.cell_cy[i];
        let mut a11 = 0.0;
        let mut a12 = 0.0;
        let mut a22 = 0.0;
        let mut b1 = 0.0;
        let mut b2 = 0.0;
        for n in neighbors(mesh, i) {
            let dx = mesh.cell_cx[n] - cx;
            let dy = mesh.cell_cy[n] - cy;
            let dv = values[n] - values[i];
            a11 += dx * dx;
            a12 += dx * dy;
            a22 += dy * dy;
            b1 += dx * dv;
            b2 += dy * dv;
        }
        let det = a11 * a22 - a12 * a12;
        if det.abs() > 1e-12 {
            let gx = (a22 * b1 - a12 * b2) / det;
            let gy = (-a12 * b1 + a11 * b2) / det;
            grads[i] = (gx, gy);
        }
    }
    grads
}

fn vorticity(mesh: &Mesh, u: &[(f64, f64)]) -> Vec<f64> {
    let u_x: Vec<f64> = u.iter().map(|val| val.0).collect();
    let u_y: Vec<f64> = u.iter().map(|val| val.1).collect();
    let grad_x = gradients(mesh, &u_x);
    let grad_y = gradients(mesh, &u_y);
    grad_x
        .iter()
        .zip(grad_y.iter())
        .map(|(gx, gy)| gy.0 - gx.1)
        .collect()
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

fn mesh_bounds(mesh: &Mesh) -> (f64, f64, f64, f64) {
    let min_x = mesh.vx.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_x = mesh.vx.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_y = mesh.vy.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_y = mesh.vy.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (min_x, max_x, min_y, max_y)
}

fn cell_polygon(mesh: &Mesh, cell: usize) -> Vec<[f64; 2]> {
    let start = mesh.cell_vertex_offsets[cell];
    let end = mesh.cell_vertex_offsets[cell + 1];
    let mut points = Vec::with_capacity(end - start);
    for idx in start..end {
        let v = mesh.cell_vertices[idx];
        points.push([mesh.vx[v], mesh.vy[v]]);
    }
    points
}

fn point_in_poly(px: f64, py: f64, poly: &[[f64; 2]]) -> bool {
    if poly.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let xi = poly[i][0];
        let yi = poly[i][1];
        let xj = poly[j][0];
        let yj = poly[j][1];
        let crosses = (yi > py) != (yj > py) && px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi;
        if crosses {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn colormap_sequential(t: f64) -> Rgb<u8> {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.5 {
        let s = t * 2.0;
        (0.0, s, 1.0 - s)
    } else {
        let s = (t - 0.5) * 2.0;
        (s, 1.0 - s, 0.0)
    };
    Rgb([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8])
}

fn save_scalar_image(path: &PathBuf, mesh: &Mesh, values: &[f64], width: usize, height: usize) {
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    for &val in values {
        min_val = min_val.min(val);
        max_val = max_val.max(val);
    }
    if (max_val - min_val).abs() < 1e-6 {
        max_val = min_val + 1.0;
    }

    let (min_x, max_x, min_y, max_y) = mesh_bounds(mesh);
    let mesh_w = (max_x - min_x).max(1e-12);
    let mesh_h = (max_y - min_y).max(1e-12);
    let scale_x = (width as f64 - 1.0) / mesh_w;
    let scale_y = (height as f64 - 1.0) / mesh_h;
    let scale = scale_x.min(scale_y);
    let center_x = 0.5 * (min_x + max_x);
    let center_y = 0.5 * (min_y + max_y);
    let offset_x = (width as f64 - 1.0) * 0.5;
    let offset_y = (height as f64 - 1.0) * 0.5;

    let mut sum = vec![0.0; width * height];
    let mut count = vec![0u32; width * height];

    for cell in 0..mesh.num_cells() {
        let poly = cell_polygon(mesh, cell);
        if poly.len() < 3 {
            continue;
        }
        let mut poly_px = Vec::with_capacity(poly.len());
        let mut min_px = f64::INFINITY;
        let mut max_px = f64::NEG_INFINITY;
        let mut min_py = f64::INFINITY;
        let mut max_py = f64::NEG_INFINITY;
        for p in &poly {
            let x = (p[0] - center_x) * scale + offset_x;
            let y = (center_y - p[1]) * scale + offset_y;
            min_px = min_px.min(x);
            max_px = max_px.max(x);
            min_py = min_py.min(y);
            max_py = max_py.max(y);
            poly_px.push([x, y]);
        }
        let x0 = min_px.floor().max(0.0) as i32;
        let x1 = max_px.ceil().min(width as f64 - 1.0) as i32;
        let y0 = min_py.floor().max(0.0) as i32;
        let y1 = max_py.ceil().min(height as f64 - 1.0) as i32;
        for yi in y0..=y1 {
            for xi in x0..=x1 {
                let px = xi as f64 + 0.5;
                let py = yi as f64 + 0.5;
                if point_in_poly(px, py, &poly_px) {
                    let idx = yi as usize * width + xi as usize;
                    sum[idx] += values[cell];
                    count[idx] += 1;
                }
            }
        }
    }

    let mut img = RgbImage::new(width as u32, height as u32);
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let color = if count[idx] == 0 {
                Rgb([10, 10, 10])
            } else {
                let val = sum[idx] / count[idx] as f64;
                let t = (val - min_val) / (max_val - min_val).max(1e-12);
                colormap_sequential(t)
            };
            img.put_pixel(x as u32, y as u32, color);
        }
    }
    let _ = img.save(path);
}

fn std_dev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f64>()
        / values.len() as f64;
    var.sqrt()
}

#[test]
#[ignore]
fn low_mach_equivalence_vortex_street() {
    let save_plots = env_bool("CFD2_SAVE_PLOTS", false);
    let steps = env_usize("CFD2_LOW_MACH_STEPS", 1200);
    let dt = env_f64("CFD2_LOW_MACH_DT", 0.001);
    let dtau = env_f64("CFD2_LOW_MACH_DTAU", 0.0);
    let alpha_u = env_f64("CFD2_LOW_MACH_ALPHA_U", 1.0);
    let precond_model = match env_usize("CFD2_LOW_MACH_PRECOND_MODEL", 1) as u32 {
        0 => GpuLowMachPrecondModel::Legacy,
        1 => GpuLowMachPrecondModel::WeissSmith,
        _ => GpuLowMachPrecondModel::Off,
    };
    let precond_theta_floor = env_f64("CFD2_LOW_MACH_PRECOND_THETA_FLOOR", 1e-6);
    let nonconv_relax = env_f64("CFD2_LOW_MACH_NONCONV_RELAX", 0.5);
    let checker_max = env_f64("CFD2_LOW_MACH_CB_MAX", 2.0);
    let cell = env_f64("CFD2_LOW_MACH_CELL", 0.05);
    let smooth_alpha = env_f64("CFD2_LOW_MACH_SMOOTH_ALPHA", 0.3);
    let smooth_iters = env_usize("CFD2_LOW_MACH_SMOOTH_ITERS", 40);
    let base_pressure = env_f64("CFD2_LOW_MACH_BASE_P", 25.0);
    let perturb_amp = env_f64("CFD2_LOW_MACH_PERTURB", 5e-3);
    let comp_iters = env_usize("CFD2_LOW_MACH_COMP_ITERS", 40);
    let probe_stride = env_usize("CFD2_LOW_MACH_PROBE_STRIDE", 10);
    let progress = env_bool("CFD2_LOW_MACH_PROGRESS", true);
    let progress_stride = env_usize("CFD2_LOW_MACH_PROGRESS_STRIDE", 25);
    let skip_comp = env_bool("CFD2_LOW_MACH_SKIP_COMP", false);
    let plot_width = env_usize("CFD2_PLOT_WIDTH", 480);
    let plot_height = env_usize("CFD2_PLOT_HEIGHT", 160);

    let mesh = build_mesh(cell, smooth_alpha, smooth_iters);
    let probe_idx = find_probe_index(&mesh, 1.6, 0.5);

    let u_in = env_f64("CFD2_LOW_MACH_UIN", 1.0) as f32;
    let density = 1.0f32;
    let nu = 0.001f32;

    let mut incomp = pollster::block_on(UnifiedSolver::new(
        &mesh,
        incompressible_momentum_model(),
        SolverConfig {
            advection_scheme: Scheme::QUICK,
            time_scheme: TimeScheme::Euler,
            preconditioner: PreconditionerType::Jacobi,
            stepping: SteppingMode::Coupled,
        },
        None,
        None,
    ))
    .expect("solver init");
    incomp.set_dt(dt as f32);
    incomp.set_viscosity(nu).unwrap();
    incomp.set_density(density).unwrap();
    incomp.set_advection_scheme(Scheme::QUICK);
    incomp.set_inlet_velocity(u_in).unwrap();
    let mut u_seed = vec![(0.0f64, 0.0f64); mesh.num_cells()];
    let length = 3.0;
    for i in 0..mesh.num_cells() {
        let phase = mesh.cell_cx[i] / length * std::f64::consts::TAU;
        u_seed[i].1 = perturb_amp * phase.sin();
    }
    incomp.set_u(&u_seed);
    incomp.set_p(&vec![0.0f64; mesh.num_cells()]);
    incomp.initialize_history();

    let start = std::time::Instant::now();

    if skip_comp {
        for step in 0..steps {
            incomp.step();
            if progress && (step % progress_stride == 0 || step + 1 == steps) {
                let elapsed = start.elapsed().as_secs_f64();
                let done = (step + 1) as f64;
                let avg = elapsed / done.max(1.0);
                let remaining = avg * ((steps - step - 1) as f64);
                println!(
                    "low_mach_equivalence(incomp-only) step {}/{} ({:.1}%) elapsed {:.1}s est_remain {:.1}s",
                    step + 1,
                    steps,
                    100.0 * done / steps.max(1) as f64,
                    elapsed,
                    remaining
                );
            }
        }
        let elapsed = start.elapsed().as_secs_f64();
        println!(
            "low_mach_equivalence(incomp-only) finished in {:.1}s ({:.3}s/step)",
            elapsed,
            elapsed / steps.max(1) as f64
        );
        return;
    }

    let mut comp = pollster::block_on(UnifiedSolver::new(
        &mesh,
        compressible_model(),
        SolverConfig {
            advection_scheme: Scheme::QUICK,
            time_scheme: TimeScheme::BDF2,
            preconditioner: PreconditionerType::Jacobi,
            stepping: SteppingMode::Implicit { outer_iters: 1 },
        },
        None,
        None,
    ))
    .expect("solver init");
    comp.set_dt(dt as f32);
    comp.set_dtau(dtau as f32).unwrap();
    comp.set_viscosity(nu).unwrap();
    let eos = comp.model().eos();
    comp.set_compressible_inlet_isothermal_x(density, u_in, &eos)
        .unwrap();
    comp.set_precond_model(precond_model)
        .expect("precond model");
    comp.set_precond_theta_floor(precond_theta_floor as f32)
        .expect("theta floor");
    let _ = nonconv_relax;
    comp.set_outer_iters(comp_iters).unwrap();
    let rho_init = vec![density; mesh.num_cells()];
    let p_init = vec![base_pressure as f32; mesh.num_cells()];
    let mut u_init = vec![[0.0f32, 0.0f32]; mesh.num_cells()];
    for i in 0..mesh.num_cells() {
        let phase = mesh.cell_cx[i] / length * std::f64::consts::TAU;
        u_init[i][1] = (perturb_amp * phase.sin()) as f32;
    }
    comp.set_state_fields(&rho_init, &u_init, &p_init);
    comp.initialize_history();

    let mut probe_incomp = Vec::new();
    let mut probe_comp = Vec::new();
    let mut best_probe = 0.0f64;
    let mut best_u_incomp: Option<Vec<(f64, f64)>> = None;
    let mut best_u_comp: Option<Vec<(f64, f64)>> = None;
    for step in 0..steps {
        incomp.step();
        comp.step();
        if progress && (step % progress_stride == 0 || step + 1 == steps) {
            let elapsed = start.elapsed().as_secs_f64();
            let done = (step + 1) as f64;
            let avg = elapsed / done.max(1.0);
            let remaining = avg * ((steps - step - 1) as f64);
            println!(
                "low_mach_equivalence step {}/{} ({:.1}%) elapsed {:.1}s est_remain {:.1}s",
                step + 1,
                steps,
                100.0 * done / steps.max(1) as f64,
                elapsed,
                remaining
            );
        }
        if step % probe_stride == 0 {
            let u_incomp = pollster::block_on(incomp.get_u());
            let u_comp = pollster::block_on(comp.get_u());
            probe_incomp.push(u_incomp[probe_idx].1);
            probe_comp.push(u_comp[probe_idx].1);
            let probe_abs = u_incomp[probe_idx].1.abs();
            if probe_abs > best_probe {
                best_probe = probe_abs;
                best_u_incomp = Some(u_incomp.clone());
                best_u_comp = Some(u_comp.clone());
            }
        }
    }

    let u_incomp = best_u_incomp.unwrap_or_else(|| pollster::block_on(incomp.get_u()));
    let p_incomp = pollster::block_on(incomp.get_p());
    let u_comp = best_u_comp.unwrap_or_else(|| pollster::block_on(comp.get_u()));
    let p_comp = pollster::block_on(comp.get_p());
    let dyn_pressure = 0.5 * density as f64 * (u_in as f64).powi(2);
    let norm = dyn_pressure.max(1e-8);
    let checker_incomp = checkerboard_metric(&mesh, &p_incomp, norm);
    let checker_comp = checkerboard_metric(&mesh, &p_comp, norm);

    let rms_num = u_incomp
        .iter()
        .zip(u_comp.iter())
        .map(|(a, b)| {
            let dx = a.0 - b.0;
            let dy = a.1 - b.1;
            dx * dx + dy * dy
        })
        .sum::<f64>()
        / u_incomp.len().max(1) as f64;
    let rms_den = u_incomp
        .iter()
        .map(|val| val.0 * val.0 + val.1 * val.1)
        .sum::<f64>()
        / u_incomp.len().max(1) as f64;
    let rel_rms = (rms_num.sqrt()) / rms_den.sqrt().max(1e-6);

    let std_incomp = std_dev(&probe_incomp);
    let std_comp = std_dev(&probe_comp);

    if save_plots {
        let out_dir = PathBuf::from("target/test_plots/low_mach");
        let _ = fs::create_dir_all(&out_dir);
        let vort_incomp = vorticity(&mesh, &u_incomp);
        let vort_comp = vorticity(&mesh, &u_comp);
        let speed_diff: Vec<f64> = u_incomp
            .iter()
            .zip(u_comp.iter())
            .map(|(a, b)| {
                let s_in = (a.0 * a.0 + a.1 * a.1).sqrt();
                let s_comp = (b.0 * b.0 + b.1 * b.1).sqrt();
                s_comp - s_in
            })
            .collect();
        let speed_incomp: Vec<f64> = u_incomp
            .iter()
            .map(|val| (val.0 * val.0 + val.1 * val.1).sqrt())
            .collect();
        let speed_comp: Vec<f64> = u_comp
            .iter()
            .map(|val| (val.0 * val.0 + val.1 * val.1).sqrt())
            .collect();
        save_scalar_image(
            &out_dir.join("low_mach_incomp_vort.png"),
            &mesh,
            &vort_incomp,
            plot_width,
            plot_height,
        );
        save_scalar_image(
            &out_dir.join("low_mach_comp_vort.png"),
            &mesh,
            &vort_comp,
            plot_width,
            plot_height,
        );
        save_scalar_image(
            &out_dir.join("low_mach_incomp_speed.png"),
            &mesh,
            &speed_incomp,
            plot_width,
            plot_height,
        );
        save_scalar_image(
            &out_dir.join("low_mach_comp_speed.png"),
            &mesh,
            &speed_comp,
            plot_width,
            plot_height,
        );
        save_scalar_image(
            &out_dir.join("low_mach_speed_diff.png"),
            &mesh,
            &speed_diff,
            plot_width,
            plot_height,
        );

        let gamma = 1.4;
        let mut nu_num_sum = 0.0f64;
        let mut nu_num_count = 0usize;
        let mut nu_num_max = 0.0f64;
        for (i, (u_val, p_val)) in u_comp.iter().zip(p_comp.iter()).enumerate() {
            if !p_val.is_finite() {
                continue;
            }
            let speed = (u_val.0 * u_val.0 + u_val.1 * u_val.1).sqrt();
            let c = (gamma * p_val / density as f64).sqrt();
            if !c.is_finite() || c <= 0.0 {
                continue;
            }
            let mach = speed / c;
            let mach2 = mach * mach;
            let theta = if matches!(precond_model, GpuLowMachPrecondModel::WeissSmith) {
                mach2.max(precond_theta_floor).min(1.0)
            } else {
                mach2
            };
            let c_eff = (theta * c * c + (1.0 - theta) * speed * speed).sqrt();
            let h = mesh.cell_vol[i].sqrt();
            let nu_num = 0.5 * (speed + c_eff) * h;
            if nu_num.is_finite() {
                nu_num_sum += nu_num;
                nu_num_count += 1;
                nu_num_max = nu_num_max.max(nu_num);
            }
        }
        let nu_num_mean = if nu_num_count > 0 {
            nu_num_sum / nu_num_count as f64
        } else {
            f64::NAN
        };

        let mut summary = String::new();
        summary.push_str(&format!(
            "probe_std_incomp={:.3e}\nprobe_std_comp={:.3e}\nrel_rms={:.3}\n",
            std_incomp, std_comp, rel_rms
        ));
        summary.push_str(&format!(
            "nu_num_mean={:.3e}\nnu_num_max={:.3e}\n",
            nu_num_mean, nu_num_max
        ));
        summary.push_str(&format!(
            "checker_incomp={:.3e}\nchecker_comp={:.3e}\n",
            checker_incomp, checker_comp
        ));
        let _ = fs::write(out_dir.join("low_mach_summary.txt"), summary);
    }

    assert!(std_incomp > 1e-4, "incompressible probe std too low");
    assert!(std_comp > 1e-5, "compressible probe std too low");
    assert!(
        rel_rms < 0.6,
        "relative RMS difference {:.3} too large",
        rel_rms
    );
    assert!(
        checker_comp < checker_max,
        "checkerboarding metric {:.3} exceeds {:.3}",
        checker_comp,
        checker_max
    );
}
